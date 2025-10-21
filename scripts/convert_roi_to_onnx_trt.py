#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert an ROI classification model (AneurysmVesselSegROILitModule family)
restored from a Hydra experiment to ONNX and a TensorRT engine.

Features:
- Restore model from experiment YAML, fold, and checkpoint type (last/best)
- Use `cfg.data.input_size` as the input size
- Define ONNX IO according to presence of extra segmentations (num_extra_mask_branches>0)
- Build a TensorRT engine via Python API when available; otherwise try `trtexec` (multi-input support)

Output layout (example):
  {out_dir}/{experiment_name}/fold_{fold}/
    - roi_model.onnx
    - roi_model_fp16.engine or roi_model_fp32.engine

Example:
  python scripts/convert_roi_to_onnx_trt.py \
    --experiment configs/experiment/251011-seg_tf-v4-nnunet_truncate1-pretrained_1e-3_e30-ex_dav6w3-m32g64-w1_1_01.yaml \
    --fold 0 --ckpt last --out-dir /workspace/logs/trt_roi --fp16 --dynamic \
    --opt-shape 1,1,96,192,192 --min-shape 1,1,64,128,128 --max-shape 1,1,160,256,256

Notes:
- Instantiates the trained LightningModule from Hydra config
- Metadata branches are omitted (pass None even if enabled in cfg)
- When num_extra_mask_branches>0, include extra_vessel in inputs
"""

from __future__ import annotations

import argparse
import os
import sys
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import hydra

# Root setup (resolve local package imports)
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Use existing utilities
from src.my_utils.kaggle_utils import load_experiment_config


def _parse_shape(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    """Parse shape string (e.g., "1,1,96,192,192") into a tuple (None for empty)."""
    # Flexible for CLI specs
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None
    try:
        parts = [int(x) for x in s.replace("x", ",").split(",")]
        if any(p <= 0 for p in parts):
            raise ValueError("all dims must be > 0")
        return tuple(parts)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid shape '{text}': {e}")


def _stem_experiment_name(path_or_name: str) -> str:
    """Derive Hydra experiment name (filename without suffix) from argument."""
    # Support both path/filename and bare name
    p = Path(path_or_name)
    if p.suffix == ".yaml" and p.exists():
        return p.stem
    return path_or_name


def _find_ckpt_path(log_dir: Path, ckpt_type: str) -> Optional[Path]:
    """Find checkpoint (last/best)."""
    # Same search logic as rsna_submission_roi.py
    if ckpt_type == "best":
        pattern = re.compile(r"epoch_(\d+)\.ckpt")
        epoch_ckpts: List[Path] = []
        ckpt_dir = log_dir
        if ckpt_dir.exists():
            for p in ckpt_dir.glob("epoch_*.ckpt"):
                if pattern.match(p.name):
                    epoch_ckpts.append(p)
        if not epoch_ckpts:
            return None

        def _epoch_num(p: Path) -> int:
            m = pattern.match(p.name)
            if not m:
                return -1
            try:
                return int(m.group(1))
            except Exception:
                return -1

        return max(epoch_ckpts, key=_epoch_num)
    elif ckpt_type == "last":
        p = log_dir / "last.ckpt"
        return p if p.exists() else None
    else:
        raise ValueError(f"unknown ckpt_type: {ckpt_type}")


class RoiForExport(torch.nn.Module):
    """推論用ラッパーモジュール（ONNX/TRT化のためにI/Oを簡素化）。"""

    def __init__(self, module: torch.nn.Module, use_ema: bool = True) -> None:
        super().__init__()
        self.module = module
        self.use_ema = bool(use_ema)

    def forward(
        self,
        roi: torch.Tensor,
        vessel_seg: torch.Tensor,
        vessel_union: torch.Tensor,
        extra_vessel: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 日本語コメント: LitModuleのforwardへ委譲し、14クラスlogitsのみを返す
        out = self.module(
            roi,
            vessel_seg=vessel_seg,
            vessel_union=vessel_union,
            extra_vessel_seg=extra_vessel,
            use_ema=self.use_ema,
        )
        logits_loc = out["logits_loc"]  # (B,13)
        logit_ap = out["logit_ap"].unsqueeze(1)  # (B,1)
        logits = torch.cat([logits_loc, logit_ap], dim=1).contiguous()  # (B,14)
        return logits


def _export_onnx(
    model: torch.nn.Module,
    onnx_out: str,
    dummy_inputs: Dict[str, torch.Tensor],
    dynamic: bool,
    opset: int,
    verbose: bool,
) -> None:
    """ONNXへエクスポート（必要に応じて動的軸を付与）。"""
    # 日本語コメント: 入力名/出力名を固定して推論側との整合性を保つ
    input_names = list(dummy_inputs.keys())
    output_names = ["logits"]

    if dynamic:
        # 日本語コメント: N,D,H,W を動的化（Cは固定）。
        dyn = {}
        for k, t in dummy_inputs.items():
            if t.dim() == 5:
                dyn[k] = {0: "N", 2: "D", 3: "H", 4: "W"}
            elif t.dim() == 4:
                dyn[k] = {0: "N", 2: "H", 3: "W"}
        dyn["logits"] = {0: "N"}
    else:
        dyn = None

    # 日本語コメント: 推論モード/勾配無効で追跡（ONNXエクスポート時のメモリ削減）
    torch.onnx.export(
        model,
        tuple(dummy_inputs[k] for k in input_names),
        onnx_out,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dyn,
        verbose=verbose,
    )

    # 日本語コメント: onnx-simplifier で簡約（任意）
    try:
        import onnx  # type: ignore
        from onnxsim import simplify  # type: ignore

        m = onnx.load(onnx_out)
        ms, ok = simplify(m)
        if ok:
            onnx.save(ms, onnx_out)
    except Exception:
        pass


def _trt_build_engine_multi_input_python(
    onnx_path: str,
    engine_out: str,
    profiles: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]],
    use_fp16: bool,
    *,
    workspace_mb: int = 4096,
    tactic_mem_mb: Optional[int] = 1024,
    opt_level: int = 4,
    max_aux_streams: int = 4,
    restrict_tactics: str = "all",
) -> None:
    """TensorRT Python APIで多入力ONNXからエンジンをビルド（省メモリ対応）。

    引数:
      - workspace_mb: WORKSPACE上限（MB）。小さくするとアルゴリズム候補が減り省メモリ
      - tactic_mem_mb: TACTIC_SHARED_MEMORY上限（MB、TRT>=8.6等で有効）
      - opt_level: Builderの最適化レベル（0..5）。0は最小メモリ
      - max_aux_streams: 補助ストリーム数。小さいほどメモリ節約
      - restrict_tactics: タクティック制限（"all"/"cublas"/"cublas_cudnn"）
    """
    # 日本語コメント: 入力ごとに最適化プロファイルを設定し、BuilderConfigを省メモリ設定
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

    config = builder.create_builder_config()

    # 日本語コメント: WORKSPACEメモリ上限（TRT10API優先）
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mb) << 20)
    except Exception:
        config.max_workspace_size = int(workspace_mb) << 20

    # 日本語コメント: TACTIC共有メモリ上限（利用可能な場合のみ）
    if tactic_mem_mb is not None:
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.TACTIC_SHARED_MEMORY, int(tactic_mem_mb) << 20)
        except Exception:
            pass

    # 日本語コメント: 最適化レベル/補助ストリーム抑制
    try:
        config.set_builder_optimization_level(int(opt_level))
    except Exception:
        pass
    try:
        config.set_max_aux_streams(int(max_aux_streams))
    except Exception:
        try:
            config.max_aux_streams = int(max_aux_streams)  # 古いAPI
        except Exception:
            pass

    # 日本語コメント: タクティック探索を制限（cuDNN省略でメモリ節約）
    try:
        src = 0
        has = getattr(trt, "TacticSource", None)
        if has is not None:
            if restrict_tactics == "all":
                src = (
                    trt.TacticSource.CUBLAS
                    | getattr(trt.TacticSource, "CUBLAS_LT", 0)
                    | getattr(trt.TacticSource, "CUDNN", 0)
                )
            elif restrict_tactics == "cublas_cudnn":
                src = trt.TacticSource.CUBLAS | getattr(trt.TacticSource, "CUDNN", 0)
            else:  # "cublas"
                src = trt.TacticSource.CUBLAS | getattr(trt.TacticSource, "CUBLAS_LT", 0)
            config.set_tactic_sources(src)
    except Exception:
        pass

    # 日本語コメント: FP16を有効（対応GPUのみ）
    if use_fp16 and getattr(builder, "platform_has_fast_fp16", True):
        try:
            config.set_flag(trt.BuilderFlag.FP16)
        except Exception:
            pass
    # 日本語コメント: シリアライズ高速化（ビルド時間/一部メモリ抑制の期待）
    try:
        config.set_flag(trt.BuilderFlag.FAST_SERIALIZE)
    except Exception:
        pass

    # 日本語コメント: 最適化プロファイル設定
    profile = builder.create_optimization_profile()
    net_input_names = [network.get_input(i).name for i in range(network.num_inputs)]
    for name in net_input_names:
        if name not in profiles:
            raise KeyError(f"opt profile missing for input: {name}")
        mn, op, mx = profiles[name]
        profile.set_shape(name, mn, op, mx)
    config.add_optimization_profile(profile)

    # 日本語コメント: エンジンをビルド
    serialized_obj = None
    if hasattr(builder, "build_serialized_network"):
        serialized_obj = builder.build_serialized_network(network, config)
        if serialized_obj is None:
            raise RuntimeError("TensorRT build_serialized_network failed")
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("TensorRT build_engine failed")
        serialized_obj = engine.serialize()

    data = bytearray(serialized_obj) if not isinstance(serialized_obj, (bytes, bytearray)) else serialized_obj
    with open(engine_out, "wb") as f:
        f.write(data)


def _trt_build_engine_multi_input_trtexec(
    onnx_path: str,
    engine_out: str,
    profiles: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]],
    use_fp16: bool,
) -> None:
    """trtexec で多入力最適化プロファイルを設定してエンジンをビルド。"""
    # 日本語コメント: --minShapes/--optShapes/--maxShapes を入力ごとに複数回指定
    cmd: List[str] = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_out}",
        "--explicitBatch",
    ]

    def fmt(shp: Tuple[int, ...]) -> str:
        return "x".join(str(x) for x in shp)

    for name, (mn, op, mx) in profiles.items():
        cmd += [
            f"--minShapes={name}:{fmt(mn)}",
            f"--optShapes={name}:{fmt(op)}",
            f"--maxShapes={name}:{fmt(mx)}",
        ]
    if use_fp16:
        cmd.append("--fp16")

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"trtexec failed (code={proc.returncode})\n{proc.stdout}")


def main() -> None:
    # 日本語コメント: CLI引数
    ap = argparse.ArgumentParser(
        description="Convert ROI classification model to ONNX/TensorRT from Hydra experiment"
    )
    ap.add_argument("--experiment", type=str, required=True, help="Hydra experiment (path or name)")
    ap.add_argument("--fold", type=int, default=0, help="Fold index")
    ap.add_argument("--ckpt", type=str, default="last", choices=["last", "best"], help="Checkpoint type")
    ap.add_argument("--out-dir", type=str, required=True, help="Output root directory")
    ap.add_argument("--device", type=str, default="cpu", help="Export device")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    ap.add_argument("--dynamic", action="store_true", help="Make N,D,H,W dynamic")
    ap.add_argument("--fp16", action="store_true", help="Build TensorRT in FP16 when possible")
    ap.add_argument("--onnx-fp16", action="store_true", help="Export ONNX in FP16 (usually not recommended)")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs")

    # 日本語コメント: TensorRTビルド関連のオプション（16GB前提で高性能寄りの既定値）
    ap.add_argument("--workspace-mb", type=int, default=4096, help="TRT WORKSPACE limit (MB). 4096-8192 recommended")
    ap.add_argument("--tactic-mem-mb", type=int, default=1024, help="TRT TACTIC_SHARED_MEMORY limit (MB). Ignored if unsupported")
    ap.add_argument("--opt-level", type=int, default=4, help="Builder optimization level (0..5). Higher is faster/more memory")
    ap.add_argument("--max-aux-streams", type=int, default=4, help="Max auxiliary streams (higher may increase throughput)")
    ap.add_argument(
        "--restrict-tactics",
        type=str,
        default="all",
        choices=["all", "cublas", "cublas_cudnn"],
        help="Restrict tactic search (use 'cublas' for memory saving)",
    )
    ap.add_argument("--profile-static", action="store_true", help="Force opt-only profiles (min=max=opt)")
    ap.add_argument("--no-trtexec-fallback", action="store_true", help="Disable trtexec fallback")
    ap.add_argument("--mem-friendly", action="store_true", help="Apply memory-friendly defaults")
    ap.add_argument("--free-gpu-before-trt", action="store_true", help="Free GPU tensors before TRT build")
    ap.set_defaults(free_gpu_before_trt=True)

    # Shape options (5D for roi input). Other inputs replace only the C dim.
    ap.add_argument("--min-shape", type=_parse_shape, default=None, help="Min shape (e.g., 1,1,64,128,128)")
    ap.add_argument("--opt-shape", type=_parse_shape, default=None, help="Opt shape (e.g., 1,1,96,192,192)")
    ap.add_argument("--max-shape", type=_parse_shape, default=None, help="Max shape (e.g., 1,1,160,256,256)")

    args = ap.parse_args()

    # Normalize experiment name
    experiment_name = _stem_experiment_name(args.experiment)

    # Load config (disable compile for stable export)
    cfg = load_experiment_config(experiment_name, config_dir=os.getenv("CONFIG_DIR", "/workspace/configs"))
    if cfg.model.get("compile"):
        cfg.model.compile = False

    # fold反映
    cfg.data.fold = int(args.fold)

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model)
    model.eval()

    # Find/load checkpoint
    ckpt_dir = Path(cfg.log_dir) / "checkpoints" / f"fold{args.fold}"
    ckpt_path = _find_ckpt_path(ckpt_dir, args.ckpt)
    if ckpt_path is None:
        raise FileNotFoundError(f"checkpoint not found: {ckpt_dir} (mode={args.ckpt})")
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    # Remove '_orig_mod.' prefix potentially added by torch.compile
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if args.verbose:
        if missing:
            print(f"[WARN] missing keys: {len(missing)}")
        if unexpected:
            print(f"[WARN] unexpected keys: {len(unexpected)}")

    # 日本語コメント: 推論ラッパを作成
    wrapper = RoiForExport(model, use_ema=True)

    # 日本語コメント: 入力サイズ/追加セグの有無を取得
    sz = getattr(cfg.data, "input_size", None)
    if sz is None or len(sz) != 3:
        raise RuntimeError("cfg.data.input_size が不正です（長さ3の配列を想定）")
    D, H, W = int(sz[0]), int(sz[1]), int(sz[2])
    num_extra = int(getattr(model.hparams, "num_extra_mask_branches", 0))
    use_extra = num_extra > 0

    device = torch.device(args.device)
    if args.onnx_fp16 and device.type == "cuda":
        wrapper = wrapper.half()
    wrapper = wrapper.to(device)

    # Dummy inputs (fixed C: roi=1, seg=13, union=1, extra=13*num_extra)
    def make(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=device)

    in_roi = make(
        (1, 1, D, H, W), torch.float16 if args.onnx_fp16 and device.type == "cuda" else torch.float32
    )
    in_seg = make((1, 13, D, H, W), in_roi.dtype)
    in_union = make((1, 1, D, H, W), in_roi.dtype)
    dummy_inputs: Dict[str, torch.Tensor] = {
        "roi": in_roi,
        "vessel_seg": in_seg,
        "vessel_union": in_union,
    }
    if use_extra:
        in_extra = make((1, 13 * num_extra, D, H, W), in_roi.dtype)
        dummy_inputs["extra_vessel"] = in_extra

    # Output paths
    fold_dir = f"fold_{args.fold}"
    save_dir = os.path.join(args.out_dir, experiment_name, fold_dir)
    os.makedirs(save_dir, exist_ok=True)
    onnx_out = os.path.join(save_dir, "roi_model.onnx")
    engine_out = os.path.join(save_dir, f"roi_model_{'fp16' if args.fp16 else 'fp32'}.engine")

    # ONNX export
    _export_onnx(
        wrapper, onnx_out, dummy_inputs, dynamic=bool(args.dynamic), opset=args.opset, verbose=args.verbose
    )
    print(f"[OK] ONNX exported: {onnx_out}")

    # Build TensorRT optimization profiles (static if unspecified: min=max=opt)
    def ensure_5d(name: str, shp: Optional[Tuple[int, ...]]) -> Optional[Tuple[int, ...]]:
        if shp is None:
            return None
        if len(shp) != 5:
            raise ValueError(f"{name} must be 5D (N,C,D,H,W), got {shp}")
        return shp

    opt_roi = ensure_5d("opt-shape", args.opt_shape) or (1, 1, D, H, W)
    min_roi = ensure_5d("min-shape", args.min_shape) or opt_roi
    max_roi = ensure_5d("max-shape", args.max_shape) or opt_roi

    # Memory-friendly: fix profiles to narrow dynamic range
    if args.mem_friendly or args.profile_static:
        min_roi = opt_roi
        max_roi = opt_roi

    # Replace only C for each input
    def repl_C(shp: Tuple[int, ...], C: int) -> Tuple[int, ...]:
        return (shp[0], C, shp[2], shp[3], shp[4])

    profiles: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = {
        "roi": (min_roi, opt_roi, max_roi),
        "vessel_seg": (repl_C(min_roi, 13), repl_C(opt_roi, 13), repl_C(max_roi, 13)),
        "vessel_union": (repl_C(min_roi, 1), repl_C(opt_roi, 1), repl_C(max_roi, 1)),
    }
    if use_extra:
        Cx = 13 * num_extra
        profiles["extra_vessel"] = (repl_C(min_roi, Cx), repl_C(opt_roi, Cx), repl_C(max_roi, Cx))

    # Free as many GPU tensors/models as possible before TRT build
    if args.free_gpu_before_trt and torch.cuda.is_available():
        try:
            del in_roi, in_seg, in_union
            if use_extra:
                del in_extra
            del dummy_inputs
            del wrapper
            del model
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # TensorRT build (Python API -> trtexec fallback)
    try:
        import tensorrt  # type: ignore # noqa: F401

        _trt_build_engine_multi_input_python(
            onnx_path=onnx_out,
            engine_out=engine_out,
            profiles=profiles,
            use_fp16=bool(args.fp16),
            workspace_mb=int(args.workspace_mb),
            tactic_mem_mb=int(args.tactic_mem_mb) if args.tactic_mem_mb is not None else None,
            opt_level=int(args.opt_level if not args.mem_friendly else 0),
            max_aux_streams=int(args.max_aux_streams),
            restrict_tactics=("cublas" if args.mem_friendly else args.restrict_tactics),
        )
        print(f"[OK] TensorRT engine built via Python API: {engine_out}")
        return
    except Exception as e:
        print(f"[WARN] TensorRT Python API not available or failed: {e}")

    if not args.no_trtexec_fallback:
        try:
            _trt_build_engine_multi_input_trtexec(
                onnx_path=onnx_out,
                engine_out=engine_out,
                profiles=profiles,
                use_fp16=bool(args.fp16),
            )
            print(f"[OK] TensorRT engine built via trtexec: {engine_out}")
            return
        except Exception as e:
            print(f"[ERROR] trtexec failed: {e}")
            print("Could not build TensorRT engine. ONNX has been exported.")


if __name__ == "__main__":
    main()
