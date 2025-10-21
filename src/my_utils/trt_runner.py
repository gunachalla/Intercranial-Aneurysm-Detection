#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight runner to execute a TensorRT engine via the Python API.

Features:
- Assumes explicit batch (EXPLICIT_BATCH in ONNX export)
- Single input and single output tensor (input, logits)
- Supports dynamic shapes (N and spatial dims) by setting binding shapes per run
- Executes asynchronously on the same stream as the input PyTorch CUDA tensor
"""

from __future__ import annotations

import os
from typing import Optional

import torch


class TRTRunner:
    """TensorRT runner for single-input, single-output models."""

    def __init__(self, engine_path: str, device: torch.device | int | None = None):
        # Args:
        #     engine_path: Path to the TensorRT engine file (.engine)
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        import tensorrt as trt  # type: ignore

        # Explicitly fix CUDA device (avoid context mismatch in multi-GPU)
        if isinstance(device, int):
            self.device = torch.device(f"cuda:{device}")
        elif isinstance(device, torch.device):
            self.device = device
        else:
            # Bind to current device when unspecified
            cur = torch.cuda.current_device() if torch.cuda.is_available() else 0
            self.device = torch.device(f"cuda:{cur}")

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)

        # Create engine and execution context on the target device
        with torch.cuda.device(self.device):
            with open(engine_path, "rb") as f:
                engine_bytes = f.read()
            self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")
        # Create execution context on the same device
        with torch.cuda.device(self.device):
            self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        # Initialize assuming TensorRT 10.7 I/O tensor API
        self.io_tensor_num = self.engine.num_io_tensors
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None

        for i in range(self.io_tensor_num):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            if tensor_mode == self.trt.TensorIOMode.INPUT and self.input_name is None:
                self.input_name = tensor_name
            elif tensor_mode == self.trt.TensorIOMode.OUTPUT and self.output_name is None:
                self.output_name = tensor_name

        if self.input_name is None:
            raise RuntimeError("Failed to locate input tensor binding")
        if self.output_name is None:
            raise RuntimeError("Failed to locate output tensor binding")

        # Map output dtype to torch dtype
        trt_dtype_out = self.engine.get_tensor_dtype(self.output_name)
        trt_dtype_in = self.engine.get_tensor_dtype(self.input_name)

        self.torch_dtype_out = self._to_torch_dtype(trt_dtype_out)
        self.torch_dtype_in = self._to_torch_dtype(trt_dtype_in)

    # ===== Internal utilities =====
    def _to_torch_dtype(self, trt_dtype) -> torch.dtype:
        # Convert tensorrt.DataType to torch.dtype
        if trt_dtype == self.trt.DataType.FLOAT:
            return torch.float32
        if trt_dtype == self.trt.DataType.HALF:
            return torch.float16
        if trt_dtype == self.trt.DataType.INT8:
            # Not expected for this use case
            return torch.int8
        if trt_dtype == self.trt.DataType.INT32:
            return torch.int32
        # Default to fp32
        return torch.float32

    # ===== Inference =====
    def run(self, x: torch.Tensor, *, enforce_half_output: bool = True) -> torch.Tensor:
        """Run inference with the engine.

        Args:
            x: Input CUDA tensor (N, C, H, W) or (N, C, D, H, W)
            enforce_half_output: Convert outputs to float16 (align with nnUNet aggregation dtype)
        Returns:
            Output logits tensor (N, C_out, ...)
        """
        if x.device.type != "cuda":
            raise RuntimeError("TRTRunner supports CUDA tensors only")
        # Move input to execution device if on a different GPU
        if x.device != self.device:
            x = x.to(self.device)
        if x.dtype != self.torch_dtype_in:
            x = x.to(dtype=self.torch_dtype_in)
        if not x.is_contiguous():
            x = x.contiguous()

        # Execute inside the target device context
        with torch.cuda.device(self.device):
            # Set input shape (for dynamic shapes)
            in_shape = tuple(int(d) for d in x.shape)
            # Use TensorRT 10.7 explicit I/O API
            self.context.set_input_shape(self.input_name, in_shape)

            # Query output shape
            out_shape = tuple(int(d) for d in self.context.get_tensor_shape(self.output_name))
            # Allocate output tensor
            y = torch.empty(out_shape, dtype=self.torch_dtype_out, device=x.device)

            # Use current PyTorch stream for async execution
            stream = torch.cuda.current_stream(x.device)

            try:
                self.context.set_tensor_address(self.input_name, int(x.data_ptr()))
                self.context.set_tensor_address(self.output_name, int(y.data_ptr()))
            except Exception as e:
                raise RuntimeError(f"Failed to set_tensor_address: {e}")

            ok = self.context.execute_async_v3(stream_handle=int(stream.cuda_stream))
        if not ok:
            raise RuntimeError("TensorRT execution failed")

        # Convert to half precision if requested
        if enforce_half_output and y.dtype != torch.float16:
            return y.to(dtype=torch.float16)
        return y
