#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert trained nnU-Net v2 PyTorch models to ONNX and TensorRT engines.

Features:
- Reconstruct the network from an nnU-Net checkpoint
- Export to ONNX using a dummy input (patch_size), with optional dynamic axes
- Build TensorRT engines via the Python API (or subprocess using trtexec if available)

Assumptions:
- Relies on the nnU-Net v2 package structure
- Reuses logic from `nnUNet/nnunetv2/inference/adaptive_sparse_predictor.py`
- TensorRT must be installed locally (otherwise only ONNX is produced)

Example (structured outputs recommended):
  # Saves model.onnx and model_{fp16|fp32}.engine under {out_dir}/{trainer}__{plans}__{config}/fold_{fold}/
     python scripts/convert_nnunet_to_onnx_trt.py \
       --model-dir /workspace/logs/nnUNet_results/YourTask \
       --fold 0 \
       --checkpoint-name checkpoint_final.pth \
       --out-dir /workspace/outputs/nnunet_exports \
       --fp16 --dynamic --opt-shape 1,1,96,192,192 --min-shape 1,1,64,128,128 --max-shape 1,1,160,256,256

Notes:
- Supports both 2D and 3D (rank=4/5)
- INT8 is not supported (no calibration). Could be extended if needed.
- By default, patch_size is used as the optimal shape (opt-shape).
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import subprocess
from typing import List, Optional, Sequence, Tuple

import torch

# Resolve nnUNet v2 from repository path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
_NNUNET_ROOT = os.path.join(_REPO_ROOT, "nnUNet")
if os.path.isdir(_NNUNET_ROOT) and _NNUNET_ROOT not in sys.path:
    # Add local clone nnUNet package root to sys.path
    sys.path.insert(0, _NNUNET_ROOT)

# nnUNet v2 依存モジュール
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2


def _parse_shape(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    """Parse shape string (e.g., "1,1,96,192,192") into a tuple.
    Returns None for None/empty strings.
    """
    # Flexible parser for shape specifications
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None
    try:
        parts = [int(x) for x in s.split(",")]
        if any(p <= 0 for p in parts):
            raise ValueError("all dims must be > 0")
        return tuple(parts)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid shape '{text}': {e}")


def _infer_rank_from_patch_size(patch_size: Sequence[int]) -> int:
    """Infer network rank (2D=4, 3D=5) from patch_size length.
    - Expects (D,H,W) or (H,W) as input spatial dims
    - Adds batch/channel dims to yield 2D->4, 3D->5
    """
    # nnU-Net patch_size has spatial dims only (2 for 2D, 3 for 3D)
    if len(patch_size) == 2:
        return 4
    if len(patch_size) == 3:
        return 5
    raise ValueError(f"Unsupported patch_size dims: {patch_size}")


def _make_dummy_input(
    in_channels: int,
    spatial: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create a dummy input tensor. nnU-Net uses (N,C,*) with N=1."""
    # Create a dummy tensor for ONNX export
    shape = (1, int(in_channels), *[int(x) for x in spatial])
    return torch.zeros(shape, dtype=dtype, device=device)


def _dynamic_axes(rank: int) -> dict:
    """Create dynamic_axes for ONNX export.
    - Batch (N) is always dynamic
    - Spatial dims are dynamic as well
    """
    # Name input/output axes for dynamic specification
    if rank == 5:
        # (N,C,D,H,W)
        return {
            "input": {0: "N", 2: "D", 3: "H", 4: "W"},
            "logits": {0: "N", 2: "D", 3: "H", 4: "W"},
        }
    elif rank == 4:
        # (N,C,H,W)
        return {
            "input": {0: "N", 2: "H", 3: "W"},
            "logits": {0: "N", 2: "H", 3: "W"},
        }
    else:
        raise ValueError(f"Unsupported rank: {rank}")


def _resolve_fold_dir_name(fold: str) -> str:
    """Normalize fold spec into actual folder name (fold_*).
    - Numeric string -> fold_0, etc.
    - If already prefixed with fold_, use as-is.
    """
    # Also accept "all" or "fold_all"
    fold_norm = str(fold).strip()
    if not fold_norm:
        raise ValueError("fold must not be empty")
    if fold_norm.lower() == "all":
        fold_norm = "all"
    if fold_norm.startswith("fold_"):
        return fold_norm
    return f"fold_{fold_norm}"


def _load_nnunet_network(
    model_dir: str,
    fold: str,
    checkpoint_name: str,
    device: torch.device,
    use_half: bool,
):
    """Restore nnU-Net network from a checkpoint and return it.

    Returns:
        network (torch.nn.Module), plans_manager, configuration_manager, dataset_json (dict)
    """
    # Restore directly without depending on AdaptiveSparsePredictor (equivalent steps)
    dataset_json = load_json(join(model_dir, "dataset.json"))
    plans = load_json(join(model_dir, "plans.json"))
    plans_manager = PlansManager(plans)

    # チェックポイントのロード（fold_* ディレクトリ配下）
    fold_dir = _resolve_fold_dir_name(str(fold))
    ckpt_path = join(model_dir, fold_dir, checkpoint_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)

    trainer_name = checkpoint["trainer_name"]
    configuration_name = checkpoint["init_args"]["configuration"]
    network_weights = checkpoint["network_weights"]

    configuration_manager = plans_manager.get_configuration(configuration_name)

    # Number of input channels and segmentation heads
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_seg_heads = label_manager.num_segmentation_heads

    trainer_class = recursive_find_python_class(
        os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name,
        "nnunetv2.training.nnUNetTrainer",
    )
    if trainer_class is None:
        raise RuntimeError(f"Trainer class not found: {trainer_name}")

    network = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        num_seg_heads,
        enable_deep_supervision=False,
    )
    network.load_state_dict(network_weights)
    network.eval()
    network.to(device)
    if use_half and device.type == "cuda":
        # Cast to half precision (useful if ONNX export in fp16 is desired)
        network.half()
    # Return additional identifiers for naming
    return network, plans_manager, configuration_manager, dataset_json, trainer_name, configuration_name


def _export_to_onnx(
    network: torch.nn.Module,
    onnx_out: str,
    dummy_input: torch.Tensor,
    rank: int,
    opset: int,
    dynamic: bool,
    verbose: bool,
) -> None:
    """Export a PyTorch model to ONNX.
    - Add dynamic axes when dynamic=True
    - Use 'logits' as the output name
    """
    # Standard ONNX export procedure
    input_names = ["input"]
    output_names = ["logits"]
    dynamic_axes = _dynamic_axes(rank) if dynamic else None

    torch.onnx.export(
        network,
        dummy_input,
        onnx_out,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=verbose,
    )

    # Simplify with onnx-simplifier when available
    try:
        import onnx  # type: ignore
        from onnxsim import simplify  # type: ignore

        model = onnx.load(onnx_out)
        model_simp, check = simplify(model)
        if check:
            onnx.save(model_simp, onnx_out)
    except Exception:
        # Non-fatal if simplification fails
        pass


def _trt_build_engine_via_python(
    onnx_path: str,
    engine_out: str,
    input_name_hint: str,
    min_shape: Tuple[int, ...],
    opt_shape: Tuple[int, ...],
    max_shape: Tuple[int, ...],
    use_fp16: bool,
) -> None:
    """Build and save a TensorRT engine from ONNX using the Python API.
    - TRT 10.x: build_serialized_network() returns IHostMemory
    - TRT 8/9: build_engine() returns ICudaEngine and serialize() yields IHostMemory
    - Obtain input name from ONNX and set optimization profile accordingly
    """
    import tensorrt as trt  # type: ignore

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = []
            for i in range(parser.num_errors):
                errors.append(str(parser.get_error(i)))
            raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

    # Determine actual input name (fallback to hint)
    try:
        net_input_name = network.get_input(0).name if network.num_inputs > 0 else input_name_hint
    except Exception:
        net_input_name = input_name_hint

    config = builder.create_builder_config()
    # Workspace setting (handle TRT version differences)
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    except Exception:
        config.max_workspace_size = 2 << 30

    if use_fp16 and getattr(builder, "platform_has_fast_fp16", True):
        try:
            config.set_flag(trt.BuilderFlag.FP16)
        except Exception:
            pass

    profile = builder.create_optimization_profile()
    profile.set_shape(net_input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # Handle build API differences across TRT versions
    serialized_obj = None
    if hasattr(builder, "build_serialized_network"):
        # TRT10系
        serialized_obj = builder.build_serialized_network(network, config)
        if serialized_obj is None:
            raise RuntimeError("TensorRT build_serialized_network failed")
    else:
        # TRT8/9系
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("TensorRT build_engine failed")
        serialized_obj = engine.serialize()

    # Convert IHostMemory to bytes and save
    try:
        data = bytearray(serialized_obj)
    except Exception:
        try:
            data = memoryview(serialized_obj).tobytes()
        except Exception:
            data = serialized_obj  # 最終fallback

    with open(engine_out, "wb") as f:
        f.write(data)


def _trt_build_engine_via_trtexec(
    onnx_path: str,
    engine_out: str,
    min_shape: Tuple[int, ...],
    opt_shape: Tuple[int, ...],
    max_shape: Tuple[int, ...],
    use_fp16: bool,
) -> None:
    """trtexec コマンドでエンジンをビルド（Python APIが無い環境向け）。
    - --shapes は input名=shape 形式。ここでは input を固定名とする
    """
    # 日本語コメント: trtexec が無ければ呼び出しに失敗
    input_name = "input"
    shape_str = f"{input_name}:{'x'.join(str(x) for x in opt_shape)}"  # opt を --shapes に指定
    min_str = f"{input_name}:{'x'.join(str(x) for x in min_shape)}"
    max_str = f"{input_name}:{'x'.join(str(x) for x in max_shape)}"

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_out}",
        f"--minShapes={min_str}",
        f"--optShapes={shape_str}",
        f"--maxShapes={max_str}",
        "--explicitBatch",
    ]
    if use_fp16:
        cmd.append("--fp16")

    # 実行
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"trtexec failed (code={proc.returncode})\n{proc.stdout}")


def main() -> None:
    # 日本語コメント: CLI引数の定義
    parser = argparse.ArgumentParser(description="Convert nnUNet model to ONNX and TensorRT")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="nnUNet出力ディレクトリ（plans.json, dataset.json がある）",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="0",
        help="使用するfold指定（例: 0, 1, all, fold_all）",
    )
    parser.add_argument(
        "--checkpoint-name", type=str, default="checkpoint_final.pth", help="チェックポイント名"
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Base output dir. Saves under trainer__plans__config/fold_*",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic shapes (N and spatial dims)")
    parser.add_argument("--fp16", action="store_true", help="Build TensorRT engine in FP16 when possible")
    parser.add_argument(
        "--onnx-fp16", action="store_true", help="Export ONNX in FP16 (usually not recommended)"
    )
    parser.add_argument(
        "--trt-out", type=str, default=None, help="Output path for TensorRT engine (build only if specified)"
    )

    parser.add_argument(
        "--min-shape", type=_parse_shape, default=None, help="Dynamic min shape (e.g., 1,1,64,128,128)"
    )
    parser.add_argument(
        "--opt-shape", type=_parse_shape, default=None, help="Dynamic opt shape (e.g., 1,1,96,192,192)"
    )
    parser.add_argument(
        "--max-shape", type=_parse_shape, default=None, help="Dynamic max shape (e.g., 1,1,160,256,256)"
    )

    parser.add_argument("--device", type=str, default="cuda", help="Device for export (cuda/cpu)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Load network
    device = torch.device(args.device)
    network, plans_manager, configuration_manager, dataset_json, trainer_name, configuration_name = _load_nnunet_network(
        args.model_dir, args.fold, args.checkpoint_name, device, use_half=args.onnx_fp16
    )

    # Input channels and patch size
    in_ch = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    patch_size = list(configuration_manager.patch_size)
    # Determine rank (2D:4, 3D:5)
    rank = _infer_rank_from_patch_size(patch_size)

    # Dummy input for ONNX export
    dummy = _make_dummy_input(
        in_channels=in_ch,
        spatial=patch_size,
        dtype=torch.float16 if args.onnx_fp16 and device.type == "cuda" else torch.float32,
        device=device,
    )

    # Build output path. Identifier: trainer__plans__configuration
    setting_identifier = f"{trainer_name}__{plans_manager.plans_name}__{configuration_name}"
    fold_dir = _resolve_fold_dir_name(str(args.fold))
    setting_dir = os.path.join(args.out_dir, setting_identifier, fold_dir)
    os.makedirs(setting_dir, exist_ok=True)
    onnx_out = os.path.join(setting_dir, "model.onnx")
    trt_out = os.path.join(setting_dir, f"model_{'fp16' if args.fp16 else 'fp32'}.engine")

    # Export ONNX
    os.makedirs(os.path.dirname(onnx_out), exist_ok=True)
    _export_to_onnx(
        network,
        onnx_out=onnx_out,
        dummy_input=dummy,
        rank=rank,
        opset=args.opset,
        dynamic=args.dynamic,
        verbose=args.verbose,
    )
    print(f"[OK] ONNX exported: {onnx_out}")

    # If TensorRT engine is not needed, stop here.
    # Otherwise, save TensorRT engine in the same directory (skip if TRT unavailable)

    # Decide min/opt/max shapes for optimization profile.
    # If unspecified: opt=(1,C,*patch), min/max=opt (static)
    expected_rank = 2 + len(patch_size)  # N,C + spatial

    def ensure_shape(name: str, shp: Optional[Tuple[int, ...]]) -> Optional[Tuple[int, ...]]:
        if shp is None:
            return None
        if len(shp) != expected_rank:
            raise ValueError(f"{name} rank mismatch: expected {expected_rank}, got {len(shp)}")
        return shp

    opt_shape = ensure_shape("opt-shape", args.opt_shape)
    if opt_shape is None:
        opt_shape = (1, in_ch, *patch_size)
    min_shape = ensure_shape("min-shape", args.min_shape) or opt_shape
    max_shape = ensure_shape("max-shape", args.max_shape) or opt_shape

    # Build with TensorRT
    os.makedirs(os.path.dirname(trt_out), exist_ok=True)
    try:
        import tensorrt  # type: ignore # noqa: F401

        _trt_build_engine_via_python(
            onnx_path=onnx_out,
            engine_out=trt_out,
            input_name_hint="input",
            min_shape=min_shape,
            opt_shape=opt_shape,
            max_shape=max_shape,
            use_fp16=bool(args.fp16),
        )
        print(f"[OK] TensorRT engine built via Python API: {trt_out}")
        return
    except Exception as e:
        print(f"[WARN] TensorRT Python API not available or failed: {e}")

    # trtexec フォールバック
    try:
        _trt_build_engine_via_trtexec(
            onnx_path=onnx_out,
            engine_out=trt_out,
            min_shape=min_shape,
            opt_shape=opt_shape,
            max_shape=max_shape,
            use_fp16=bool(args.fp16),
        )
        print(f"[OK] TensorRT engine built via trtexec: {trt_out}")
        return
    except Exception as e:
        print(f"[ERROR] trtexec failed: {e}")
        print("Could not build TensorRT engine. ONNX has been exported.")


if __name__ == "__main__":
    main()
