# -*- coding: utf-8 -*-
import numpy as np
import torch
# scikit-learn は DBSCAN ベース疎探索（DBSCANAdaptiveSparsePredictor）でのみ必要。
# 未使用時（= VESSEL_NNUNET_SPARSE_MODEL_DIR 未指定など）に ImportError とならないよう、
# ここでのインポートを任意化する。
try:
    from sklearn.cluster import DBSCAN as SklearnDBSCAN  # type: ignore
except Exception:
    SklearnDBSCAN = None  # type: ignore[assignment]
from typing import Tuple, List, Optional, Dict, Any, Union
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
)

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2
from monai.inferers import SlidingWindowInferer
from src.my_utils.trt_runner import TRTRunner  # TensorRT実行ラッパー


class AdaptiveSparsePredictor(nnUNetPredictor):
    # 適応型疎探索を使用したnnUNet推論クラス
    # 大規模ボリュームでは疎探索→詳細推論の2段階処理を行い、
    # 小規模ボリュームでは通常の推論を実行する

    # 定数定義
    MORPHOLOGY_KERNEL_SIZE = 2
    BINARY_MASK_THRESHOLD = 0.7

    def __init__(
        self,
        # 疎探索判定パラメータ
        window_count_threshold: int = 100,
        # 疎探索パラメータ
        sparse_downscale_factor: float = 2.0,
        sparse_overlap: float = 0.2,
        detection_threshold: float = 0.3,
        # 疎探索で使うROIマージン
        sparse_bbox_margin_voxels: int = 20,
        # 密探索（ROI内スライディング）のオーバーラップ率
        dense_overlap: float = 0.3,
        # ROIの上下（SI）方向制御
        limit_si_extent: bool = True,
        max_si_extent_mm: float = 150.0,
        si_axis: Optional[int] = 0,
        # ROIの最小長（mm）保証
        min_si_extent_mm: float = 100.0,
        min_xy_extent_mm: float = 120.0,
        # 通常のパラメータ
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        use_mirroring: bool = False,
        perform_everything_on_device: bool = True,
        device: torch.device = torch.device("cuda"),
        verbose: bool = False,
        verbose_preprocessing: bool = False,
        allow_tqdm: bool = False,
    ):
        # Args:
        #     window_count_threshold: 疎探索を開始するスライディングウィンドウ分割数の閾値
        #     sparse_downscale_factor: 疎探索時のダウンサンプリング率
        #     sparse_overlap: 疎探索時のオーバーラップ率
        #     detection_threshold: 血管領域検出の閾値
        #     sparse_bbox_margin_voxels: 疎探索で得た検出領域(BBox)に付与するマージン（ボクセル数）
        super().__init__(
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_device=perform_everything_on_device,
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose_preprocessing,
            allow_tqdm=allow_tqdm,
        )

        self.window_count_threshold = window_count_threshold
        self.sparse_downscale_factor = sparse_downscale_factor
        self.sparse_overlap = sparse_overlap
        self.detection_threshold = detection_threshold
        # 疎探索のBBoxマージン
        self.sparse_bbox_margin_voxels = int(sparse_bbox_margin_voxels)
        # 密探索のオーバーラップ率（tile_step_size = 1 - dense_overlap に変換して内部使用）
        self.dense_overlap = float(dense_overlap)
        # ROIの上下（SI）方向制御
        self.limit_si_extent = limit_si_extent
        self.max_si_extent_mm = max_si_extent_mm
        self.si_axis = si_axis
        # ROI最小長（mm）設定（SI軸とXY軸で別指定）
        self.min_si_extent_mm = float(min_si_extent_mm)
        self.min_xy_extent_mm = float(min_xy_extent_mm)
        # 単一foldのネットワーク重み（state_dict）を保持する
        # アンサンブルは呼び出し側で実施する想定
        self.network_weights = None
        # TensorRT統合: ランナーと制御フラグ
        self.trt_runner: Optional[TRTRunner] = None
        self.trt_enforce_half_output: bool = True
        # 識別用の学習設定名を保持（エンジン探索等で使用）
        self.trainer_name: Optional[str] = None
        self.configuration_name: Optional[str] = None

    # ===== ヘルパー群（重複処理の集約） =====
    @staticmethod
    def _slices_to_tuples(slices: Tuple[slice, ...]) -> List[Tuple[int, int]]:
        # sliceのタプルを(start, stop)の配列に変換（JSON保存用）
        tuples: List[Tuple[int, int]] = []
        for s in slices:
            if isinstance(s, slice):
                tuples.append((int(s.start), int(s.stop)))
            elif isinstance(s, (list, tuple)) and len(s) == 2:
                tuples.append((int(s[0]), int(s[1])))
            else:
                raise TypeError(f"Unsupported bbox element type: {type(s)}")
        return tuples

    @staticmethod
    def _add_slices(a: Tuple[slice, ...], b: Tuple[slice, ...]) -> Tuple[slice, ...]:
        # 2つのスライスを加算してグローバル座標のスライスを返す
        out: List[slice] = []
        for s1, s2 in zip(a, b):
            out.append(slice(int(s1.start) + int(s2.start), int(s1.start) + int(s2.stop)))
        return tuple(out)

    @staticmethod
    def _bbox_from_mask(
        mask: torch.Tensor, margin: int | Tuple[int, ...], full_size: torch.Tensor
    ) -> Optional[Tuple[slice, ...]]:
        # バイナリマスクからBBoxを抽出し、マージンと境界クリップを適用
        # margin は単一値または各軸ごとのタプルを受け付ける
        if not torch.any(mask):
            return None
        coords = torch.nonzero(mask, as_tuple=False)
        mins = coords.min(dim=0).values
        maxs = coords.max(dim=0).values
        # マージンの正規化
        if isinstance(margin, (tuple, list)):
            if len(margin) != len(mins):
                # 要素数が一致しない場合は不足分を最後の値で埋める
                mm = list(margin) + [int(margin[-1])] * (len(mins) - len(margin))
            else:
                mm = list(margin)
            margin_t = torch.tensor(mm, device=mins.device, dtype=mins.dtype)
        else:
            margin_t = torch.tensor([int(margin)] * len(mins), device=mins.device, dtype=mins.dtype)
        if torch.any(margin_t > 0):
            mins = torch.clamp(mins - margin_t, min=0)
            maxs = torch.clamp(maxs + margin_t, max=full_size - 1)
        return tuple(slice(int(mins[i].item()), int(maxs[i].item()) + 1) for i in range(len(mins)))

    def _build_transform_info(
        self,
        preprocessed: dict,
        data: torch.Tensor,
        roi_bbox: Tuple[slice, ...],
        refined_local_bbox: Optional[Tuple[slice, ...]],
    ) -> dict:
        # 座標変換情報の辞書を生成（元空間→ネットワーク空間）
        props = preprocessed["data_properties"]
        transpose_forward = list(self.plans_manager.transpose_forward)
        transpose_backward = list(self.plans_manager.transpose_backward)
        spacing_transposed = [props["spacing"][i] for i in transpose_forward]
        dim = len(roi_bbox)
        target_spacing = list(self.configuration_manager.spacing)
        if len(target_spacing) < dim:
            target_spacing = [spacing_transposed[0]] + target_spacing
        scale_factors = [spacing_transposed[i] / target_spacing[i] for i in range(dim)]

        roi_bbox_network = roi_bbox
        if refined_local_bbox is not None:
            roi_bbox_network_refined = self._add_slices(roi_bbox_network, refined_local_bbox)
        else:
            roi_bbox_network_refined = roi_bbox_network

        return {
            "transpose_forward": transpose_forward,
            "transpose_backward": transpose_backward,
            "spacing_original": list(props["spacing"]),
            "spacing_after_resampling": target_spacing,
            "scale_factors_orig2net": scale_factors,
            "shape_before_cropping": list(props.get("shape_before_cropping", [])),
            "shape_after_cropping_and_before_resampling": list(
                props.get("shape_after_cropping_and_before_resampling", [])
            ),
            "bbox_used_for_cropping": self._slices_to_tuples(props.get("bbox_used_for_cropping", [])),
            "network_shape": list(data.shape[1:]),
            "roi_bbox_network": self._slices_to_tuples(roi_bbox_network),
            "roi_offset_network": [int(s.start) for s in roi_bbox_network],
            "roi_bbox_network_refined": self._slices_to_tuples(roi_bbox_network_refined),
            "roi_offset_network_refined": [int(s.start) for s in roi_bbox_network_refined],
        }

    def _refine_local_bbox_from_logits(
        self,
        prediction_logits: torch.Tensor,
        threshold: float,
        refine_margin_voxels: int | Tuple[int, ...],
    ) -> Optional[Tuple[slice, ...]]:
        # 密探索logitsから背景以外クラス確率を用いてROIを再抽出
        probs = torch.softmax(prediction_logits, dim=0)
        if probs.shape[0] > 1:
            vessel_prob = torch.max(probs[1:], dim=0)[0]
        else:
            vessel_prob = probs[0]
        mask = vessel_prob > float(threshold)
        full_size = torch.tensor(prediction_logits.shape[1:], device=prediction_logits.device)
        return self._bbox_from_mask(mask, refine_margin_voxels, full_size)

    def should_use_sparse_search(self, image_size: Tuple[int, ...]) -> bool:
        # 画像サイズとパッチサイズから疎探索を使用するか判定
        # Args:
        #     image_size: 入力画像のサイズ (C, H, W, D)形式を想定
        # Returns:
        #     疎探索を使用する場合True
        # 設定がロードされていない場合はFalse
        if self.configuration_manager is None:
            return False

        # チャンネル次元を除いた空間次元のサイズ
        spatial_size = image_size[1:] if len(image_size) == 4 else image_size

        # スライディングウィンドウの分割数チェック
        patch_size = self.configuration_manager.patch_size
        steps = compute_steps_for_sliding_window(spatial_size, patch_size, self.tile_step_size)
        total_windows = np.prod([len(s) for s in steps])

        if total_windows > self.window_count_threshold:
            if self.verbose:
                print(f"ウィンドウ分割数 {total_windows} > {self.window_count_threshold}、疎探索を使用します")
            return True

        if self.verbose:
            print(f"ウィンドウ分割数 {total_windows} <= {self.window_count_threshold}、通常推論を実行します")
        return False

    @torch.inference_mode()
    def sparse_search(
        self, input_image: torch.Tensor, return_context: bool = False
    ) -> Union[Tuple[slice, ...], Tuple[Tuple[slice, ...], Dict[str, Any]]]:
        # 疎探索フェーズ: ダウンサンプリングした画像で高速スキャンし、
        # GPU上で前処理を完結させて単一BBoxを返す
        # Args:
        #     input_image: 入力画像テンソル (C, H, W, D)
        # Returns:
        #     roi_bbox: 単一のROIを示すスライスのタプル
        if self.verbose:
            print("疎探索フェーズを開始...")

        # ダウンサンプリング
        downscale_factors = [1] + [1 / self.sparse_downscale_factor] * (input_image.ndim - 1)
        downsampled = torch.nn.functional.interpolate(
            input_image.unsqueeze(0),
            scale_factor=downscale_factors[1:],
            mode="trilinear" if input_image.ndim == 4 else "bilinear",
            align_corners=False,
        ).squeeze(0)

        if self.verbose:
            print(f"ダウンサンプリング: {input_image.shape} -> {downsampled.shape}")

        # 原解像度サイズを事前に取得
        target_size = tuple(int(x) for x in input_image.shape[1:])

        # 疎探索用の一時的なパラメータ設定
        original_tile_step_size = self.tile_step_size
        try:
            self.tile_step_size = 1.0 - self.sparse_overlap  # オーバーラップ率をステップサイズに変換

            # スライディングウィンドウ推論（疎）
            # Prefer fp16 compute on CUDA with autocast; logits will be cast to fp32 for softmax later
            if self.device.type == "cuda":
                downsampled = downsampled.to(device=self.device, dtype=torch.half)
                with torch.autocast(device_type="cuda"):
                    sparse_logits = self.predict_sliding_window_return_logits(downsampled)
            else:
                sparse_logits = self.predict_sliding_window_return_logits(downsampled)
        finally:
            # パラメータを必ず元に戻す
            self.tile_step_size = original_tile_step_size

        # ソフトマックスを適用して確率に変換
        sparse_probs = torch.softmax(sparse_logits, dim=0)

        # 必要に応じて疎探索のコンテキスト情報を作成
        sparse_context: Optional[Dict[str, Any]] = None
        if return_context:
            seg_lowres = torch.argmax(sparse_probs, dim=0)
            seg_lowres_cpu = seg_lowres.detach().to(device="cpu", dtype=torch.int16)
            seg_highres = torch.nn.functional.interpolate(
                seg_lowres[None, None].to(dtype=torch.float32),
                size=target_size,
                mode="nearest",
            )[0, 0]
            seg_highres_cpu = seg_highres.detach().to(device="cpu", dtype=torch.int16)
            sparse_context = {
                "segmentation_lowres": seg_lowres_cpu,
                "segmentation_highres": seg_highres_cpu,
                "num_classes": int(sparse_logits.shape[0]),
                "input_shape": tuple(int(x) for x in input_image.shape),
                "downsampled_shape": tuple(int(x) for x in downsampled.shape),
                "network_spacing": tuple(
                    float(x) for x in getattr(self.configuration_manager, "spacing", [])
                ),
            }
            del seg_lowres, seg_highres

        # 背景クラス以外の最大確率を取得（血管領域の検出）
        if sparse_probs.shape[0] > 1:
            vessel_probs = torch.max(sparse_probs[1:], dim=0)[0]  # 背景以外のクラスの最大値
        else:
            vessel_probs = sparse_probs[0]

        # 閾値処理でバイナリマスクを作成（GPU上で実施）
        # メモリ削減のため、まず低解像度（ダウンサンプリング後）のマスクでノイズ除去を行い、
        # その後に原解像度へアップサンプリングする順序に変更
        binary_mask = (vessel_probs > self.detection_threshold).to(dtype=torch.half)

        # ノイズ除去（モルフォロジカルオープニング）: erosion → dilation（GPU近似）
        # 低解像度のまま実施してメモリ使用量を抑える
        k_open = int(self.MORPHOLOGY_KERNEL_SIZE)
        if k_open > 1:
            pad_open = k_open // 2
            inv = 1.0 - binary_mask
            if binary_mask.ndim == 3:
                inv_dil = torch.nn.functional.max_pool3d(
                    inv[None, None], kernel_size=k_open, stride=1, padding=pad_open
                )[0, 0]
                eroded = 1.0 - inv_dil
                opened = torch.nn.functional.max_pool3d(
                    eroded[None, None], kernel_size=k_open, stride=1, padding=pad_open
                )[0, 0]
            elif binary_mask.ndim == 2:
                inv_dil = torch.nn.functional.max_pool2d(
                    inv[None, None], kernel_size=k_open, stride=1, padding=pad_open
                )[0, 0]
                eroded = 1.0 - inv_dil
                opened = torch.nn.functional.max_pool2d(
                    eroded[None, None], kernel_size=k_open, stride=1, padding=pad_open
                )[0, 0]
            else:
                raise RuntimeError(f"Unsupported mask ndim: {binary_mask.ndim}")
        else:
            opened = binary_mask

        cleaned_lowres = opened if torch.any(opened > 0.5) else binary_mask

        # 原解像度にアップサンプリング（最近傍）。オープニング後のマスクを使用
        if cleaned_lowres.ndim == 3:
            cleaned = torch.nn.functional.interpolate(
                cleaned_lowres[None, None], size=target_size, mode="nearest"
            )[0, 0]
        elif cleaned_lowres.ndim == 2:
            cleaned = torch.nn.functional.interpolate(
                cleaned_lowres[None, None], size=target_size, mode="nearest"
            )[0, 0]
        else:
            raise RuntimeError(f"Unsupported mask ndim: {cleaned_lowres.ndim}")

        # 念のため、0/1へ閾値で再バイナリ化
        cleaned = (cleaned > self.BINARY_MASK_THRESHOLD).to(dtype=torch.half)

        if self.verbose:
            vessel_ratio = ((cleaned > 0.5).sum() / cleaned.numel()).item()
            print(f"疎探索完了: 血管領域 {vessel_ratio*100:.1f}%")

        # 単一BBoxの抽出（空なら全体）
        full_size = torch.tensor(target_size, device=cleaned.device)
        # 疎探索で得たバイナリマスクからBBoxを作成し、疎探索用マージンを付与
        bbox = self._bbox_from_mask(cleaned > 0.5, int(self.sparse_bbox_margin_voxels), full_size)
        if bbox is None:
            return tuple(slice(0, s) for s in target_size)
        mins = torch.tensor([s.start for s in bbox], device=cleaned.device)
        maxs = torch.tensor([s.stop - 1 for s in bbox], device=cleaned.device)

        # ここから: ROIの上下方向（SI）過大広がりを抑制
        # - スキャン範囲が肩まで含まれると、誤検出によりROIが縦に伸びすぎ密探索が重くなる
        # - 上限（mm）を超える場合は、マスクの重心（COM）付近に切り詰める
        if self.limit_si_extent and cleaned.ndim == 3:
            # spacingの取得（ネットワーク空間）。2D構成の互換は不要（3Dのみ実施）
            try:
                target_spacing = list(self.configuration_manager.spacing)
            except Exception:
                target_spacing = [1.0, 1.0, 1.0]
            # 次元数調整（万一長さが合わない場合に先頭へ複製して合わせる）
            while len(target_spacing) < cleaned.ndim:
                target_spacing = [target_spacing[0]] + target_spacing

            # SI軸の決定
            if self.si_axis is not None and 0 <= int(self.si_axis) < cleaned.ndim:
                si_ax = int(self.si_axis)
            else:
                # nnU-Net前処理の転置後は、空間軸の先頭（0番目）が基本的にSI軸になる想定
                # （高解像度＝小spacingの軸は後段に来るため）。spacingの大小には依存しない。
                si_ax = 0

            # 現ROIのSI長（mm）
            extent_vox = (maxs - mins + 1).to(dtype=torch.int64)
            extent_mm = float(extent_vox[si_ax].item()) * float(target_spacing[si_ax])

            if extent_mm > float(self.max_si_extent_mm):
                # スライスごとの占有率プロファイル（A_z）: SI軸に沿った各スライスのマスク割合（GPU上で計算）
                mask_bool = cleaned > 0.5
                inplane_axes = [ax for ax in range(3) if ax != si_ax]
                A = mask_bool.float().mean(dim=inplane_axes)  # shape=(Z,)

                # 質量中心（COM）に基づくトリム範囲（mm上限）をGPUで計算
                idxs = torch.arange(A.shape[0], device=A.device, dtype=A.dtype)
                mass = A.sum()
                if mass > 0:
                    com_t = (A * idxs).sum() / mass
                else:
                    com_t = (mins[si_ax].to(A.dtype) + maxs[si_ax].to(A.dtype)) * 0.5

                half_extent_vox = int(
                    np.ceil(0.5 * float(self.max_si_extent_mm) / float(target_spacing[si_ax]))
                )
                z_center = int(torch.round(com_t).item())
                z0 = max(0, z_center - half_extent_vox)
                z1 = min(int(full_size[si_ax].item()) - 1, z_center + half_extent_vox)

                # 元ROIとの共通部分（過度な切り過ぎ防止のため、まずは交差を取る）
                new_min_z = max(int(mins[si_ax].item()), z0)
                new_max_z = min(int(maxs[si_ax].item()), z1)
                if new_max_z <= new_min_z:
                    # 交差が極端に小さい場合はCOM中心の範囲を採用
                    new_min_z, new_max_z = z0, z1

                mins = mins.clone()
                maxs = maxs.clone()
                mins[si_ax] = int(new_min_z)
                maxs[si_ax] = int(new_max_z)

                if self.verbose:
                    after_extent_mm = (maxs[si_ax] - mins[si_ax] + 1).item() * float(target_spacing[si_ax])
                    print(
                        f"SIガード適用: 軸{si_ax}, {extent_mm:.1f}mm -> {after_extent_mm:.1f}mm (上限 {self.max_si_extent_mm}mm)"
                    )

        # ここから: ROIの最小長（mm）を保証（BBox確定直前に拡張）
        # - SI軸は min_si_extent_mm、その他の空間軸は min_xy_extent_mm を最低限確保する
        # - 可能な限り中心を保つように左右対称に拡張し、画像境界で制限された余りは反対側に配分する
        try:
            target_spacing_min = list(self.configuration_manager.spacing)
        except Exception:
            # spacingが取得できない場合は1mm仮定
            target_spacing_min = [1.0] * cleaned.ndim
        while len(target_spacing_min) < cleaned.ndim:
            target_spacing_min = [target_spacing_min[0]] + target_spacing_min

        # SI軸を再決定（3Dのみ）。2Dの場合は全軸XY扱い
        if cleaned.ndim == 3:
            if self.si_axis is not None and 0 <= int(self.si_axis) < cleaned.ndim:
                si_ax_for_min = int(self.si_axis)
            else:
                si_ax_for_min = 0
        else:
            si_ax_for_min = None

        for ax in range(cleaned.ndim):
            # 軸ごとの目標最小長（mm）
            min_len_mm = (
                self.min_si_extent_mm
                if (si_ax_for_min is not None and ax == si_ax_for_min)
                else self.min_xy_extent_mm
            )
            # 必要ボクセル数に変換
            desired_vox = int(np.ceil(float(min_len_mm) / float(target_spacing_min[ax])))
            current_vox = int((maxs[ax] - mins[ax] + 1).item())
            if desired_vox <= 1 or current_vox >= desired_vox:
                continue

            # 中心を基準に左右へ拡張
            need = desired_vox - current_vox
            left = need // 2
            right = need - left
            new_min = int(max(0, int(mins[ax].item()) - left))
            new_max = int(min(int(full_size[ax].item()) - 1, int(maxs[ax].item()) + right))

            # 境界制限により不足した分を反対側に再配分
            final_extent = new_max - new_min + 1
            if final_extent < desired_vox:
                remaining = desired_vox - final_extent
                # 左側に寄せられるだけ寄せる
                shift_left = min(remaining, new_min)
                new_min = new_min - shift_left
                remaining -= shift_left
                if remaining > 0:
                    # 右側に寄せる
                    max_allow = int(full_size[ax].item()) - 1
                    available_right = max_allow - new_max
                    shift_right = min(remaining, available_right)
                    new_max = new_max + shift_right

            # 更新
            mins[ax] = int(new_min)
            maxs[ax] = int(new_max)

        roi_bbox = tuple(slice(int(mins[i].item()), int(maxs[i].item()) + 1) for i in range(len(mins)))

        if return_context:
            return roi_bbox, sparse_context if sparse_context is not None else {}
        return roi_bbox

    def load_model_for_inference(
        self,
        model_training_output_dir: str,
        fold: Optional[int] = 0,
        checkpoint_name: str = "checkpoint_final.pth",
        torch_compile: bool = False,
    ) -> None:
        # Kaggle用: 単一foldのモデルのみを読み込む
        # Args:
        #     model_training_output_dir: モデルが保存されているディレクトリ
        #     fold: 使用するfold番号
        #     checkpoint_name: チェックポイントファイル名
        if fold is None:
            fold = 0
        print(f"モデルを読み込み中 (fold: {fold})...")

        # データセットとプランをロード
        self.dataset_json = load_json(join(model_training_output_dir, "dataset.json"))
        plans = load_json(join(model_training_output_dir, "plans.json"))
        self.plans_manager = PlansManager(plans)

        # パラメータをロード（単一fold）
        checkpoint = torch.load(
            join(model_training_output_dir, f"fold_{fold}", checkpoint_name),
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        trainer_name = checkpoint["trainer_name"]
        configuration_name = checkpoint["init_args"]["configuration"]
        # 識別名をインスタンスにも保存
        self.trainer_name = str(trainer_name)
        self.configuration_name = str(configuration_name)
        self.allowed_mirroring_axes = checkpoint.get("inference_allowed_mirroring_axes", None)
        self.network_weights = checkpoint["network_weights"]

        self.configuration_manager = self.plans_manager.get_configuration(configuration_name)

        # リサンプリングをtorch gpu実行する
        self.configuration_manager.configuration["resampling_fn_data"] = "resample_torch_fornnunet"
        self.configuration_manager.configuration["resampling_fn_seg"] = "resample_torch_fornnunet"
        self.configuration_manager.configuration["resampling_fn_probabilities"] = "resample_torch_fornnunet"
        self.configuration_manager.configuration["resampling_fn_data_kwargs"] = {
            "is_seg": False,
            "device": "cuda",
            "force_separate_z": None,
        }
        self.configuration_manager.configuration["resampling_fn_seg_kwargs"] = {
            "is_seg": True,
            "device": "cuda",
            "force_separate_z": None,
        }
        self.configuration_manager.configuration["resampling_fn_probabilities_kwargs"] = {
            "is_seg": False,
            "device": "cuda",
            "force_separate_z": None,
        }

        # ラベルマネージャーを初期化
        self.label_manager = self.plans_manager.get_label_manager(self.dataset_json)

        # ネットワークを復元
        num_input_channels = determine_num_input_channels(
            self.plans_manager, self.configuration_manager, self.dataset_json
        )
        trainer_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
            trainer_name,
            "nnunetv2.training.nnUNetTrainer",
        )

        if trainer_class is None:
            raise RuntimeError(f"トレーナークラス '{trainer_name}' が見つかりません")

        self.network = trainer_class.build_network_architecture(
            self.configuration_manager.network_arch_class_name,
            self.configuration_manager.network_arch_init_kwargs,
            self.configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            self.label_manager.num_segmentation_heads,
            enable_deep_supervision=False,
        )

        self.network = self.network.to(self.device)
        # Enable fp16 network weights for CUDA inference to reduce memory/IO
        if self.device.type == "cuda":
            self.network = self.network.half()

        # ネットワーク重みをロード（半精度GPU最適化済みネットワークに）
        self.network.load_state_dict(self.network_weights)

        if torch_compile:
            # nnUNetでは効果小さいが、ローカルでは有効化
            # Kaggle環境ではエラーになるので無効化
            self.network = torch.compile(self.network)

        print("モデル読み込み完了: 単一fold")

    def enable_tensorrt(self, engine_path: str, *, enforce_half_output: bool = True) -> None:
        # TensorRTエンジンをロードし、推論をTRTに切り替える
        # Args:
        #     engine_path: .engineファイルのパス
        #     enforce_half_output: 出力を半精度へ変換して返す（nnUNet内部の集約dtypeと整合）
        # 実行デバイスに合わせてエンジンをバインド
        dev = self.device
        try:
            # torch.device("cuda") の場合 index が None → current_device を取得
            if dev.type == "cuda" and dev.index is None:
                dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        except Exception:
            pass
        self.trt_runner = TRTRunner(engine_path, device=dev)
        self.trt_enforce_half_output = bool(enforce_half_output)
        if self.verbose:
            print(f"TensorRTエンジンを有効化: {engine_path}")

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        # ミラーリング（TTA）を考慮したネットワーク呼び出しを、TRTが有効なら置き換える
        if self.trt_runner is None:
            # 従来のPyTorchパス
            return super()._internal_maybe_mirror_and_predict(x)

        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        # まず素の推論
        prediction = self.trt_runner.run(x, enforce_half_output=self.trt_enforce_half_output)

        if mirror_axes is not None:
            # xは4D(2D画像)または5D(3D画像)。ミラー軸は空間軸（0始まり）なので、
            # テンソル次元に合わせて +2 してN,Cをスキップ
            assert len(x.shape) in (4, 5), "入力テンソルは4Dまたは5Dを想定しています"
            assert max(mirror_axes) <= x.ndim - 3, "mirror_axes does not match the dimension of the input!"
            axes_adj = [m + 2 for m in mirror_axes]

            import itertools

            axes_combinations = [
                c for i in range(len(axes_adj)) for c in itertools.combinations(axes_adj, i + 1)
            ]
            for axes in axes_combinations:
                x_flipped = torch.flip(x, axes)
                y = self.trt_runner.run(x_flipped, enforce_half_output=self.trt_enforce_half_output)
                prediction += torch.flip(y, axes)
            prediction /= len(axes_combinations) + 1
        return prediction

    def _preprocess_data(
        self,
        image: np.ndarray,
        properties: dict,
        seg_from_prev_stage: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, dict]:
        # データの前処理を実行
        # Args:
        #     image: 入力画像
        #     properties: 画像のプロパティ
        #     seg_from_prev_stage: 前段階のセグメンテーション
        # Returns:
        #     data: 前処理済みデータ
        #     preprocessed: 前処理結果
        preprocessor = PreprocessAdapterFromNpy(
            [image],
            [seg_from_prev_stage] if seg_from_prev_stage is not None else None,
            [properties],
            [None],  # output_file_truncated
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_threads_in_multithreaded=1,
            verbose=False,
        )

        preprocessed = next(preprocessor)

        return preprocessed

    def _determine_roi_from_sparse_search(self, data: torch.Tensor) -> Optional[Tuple[slice, ...]]:
        # 疎探索を使用してROIを決定
        # Args:
        #     data: 入力データ
        # Returns:
        #     roi_bbox: ROIのバウンディングボックス（存在しない場合はNone）
        if not self.should_use_sparse_search(data.shape):
            return None

        # 疎探索を実行し単一のROIを取得
        roi_bbox = self.sparse_search(data)

        if self.verbose and roi_bbox is not None:
            roi_size = tuple(s.stop - s.start for s in roi_bbox)
            print(f"ROI決定: サイズ {roi_size}")

        return roi_bbox

    def _predict_logits(
        self, data: torch.Tensor, roi_bbox: Optional[Tuple[slice, ...]] = None
    ) -> torch.Tensor:
        # 単一foldでの推論（ROI指定時はその領域のみ詳細推論）
        if roi_bbox is not None:
            results_device = self.device if self.perform_everything_on_device else torch.device("cpu")
            pred_full = torch.zeros(
                (self.label_manager.num_segmentation_heads, *data.shape[1:]),
                dtype=(torch.half if self.device.type == "cuda" else torch.float32),
                device=results_device,
            )
            roi_slices = (slice(None),) + roi_bbox
            roi_data = data[roi_slices]
            # 密探索時もオーバーラップ率からステップサイズへ変換
            original_tile_step_size = self.tile_step_size
            try:
                self.tile_step_size = 1.0 - float(self.dense_overlap)
                if self.device.type == "cuda":
                    roi_data = roi_data.half()
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        roi_logits = self.predict_sliding_window_return_logits(roi_data)
                else:
                    roi_logits = self.predict_sliding_window_return_logits(roi_data)
            finally:
                self.tile_step_size = original_tile_step_size
            pred_full[(slice(None),) + roi_bbox] = roi_logits
            return pred_full
        else:
            original_tile_step_size = self.tile_step_size
            try:
                self.tile_step_size = 1.0 - float(self.dense_overlap)
                if self.device.type == "cuda":
                    data_cast = data.half()
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = self.predict_sliding_window_return_logits(data_cast)
                else:
                    logits = self.predict_sliding_window_return_logits(data)
            finally:
                self.tile_step_size = original_tile_step_size
        return logits


class DBSCANAdaptiveSparsePredictor(AdaptiveSparsePredictor):
    """DBSCANによるROI抽出を導入した疎探索推論クラス"""

    def __init__(
        self,
        *args,
        dbscan_eps_voxels: float = 20.0,
        dbscan_min_samples: int = 100,
        dbscan_max_points: int = 60000,
        roi_extent_mm: Union[float, Tuple[float, ...], List[float]] = (130.0, 130.0, 130.0),
        **kwargs,
    ) -> None:
        # Args:
        #     dbscan_eps_voxels: DBSCANのε（ボクセル単位）
        #     dbscan_min_samples: コアポイント判定の最小近傍数
        #     dbscan_max_points: クラスタリング対象とする最大ポイント数
        #     roi_extent_mm: ROIの物理サイズ(mm)。単一値または軸ごとのシーケンスで指定
        super().__init__(*args, **kwargs)
        self.dbscan_eps_voxels = float(dbscan_eps_voxels)
        self.dbscan_min_samples = int(dbscan_min_samples)
        self.dbscan_max_points = int(dbscan_max_points)
        if isinstance(roi_extent_mm, (tuple, list)):
            self.dbscan_roi_extent_mm = tuple(float(x) for x in roi_extent_mm)
        else:
            self.dbscan_roi_extent_mm = (float(roi_extent_mm),)

    @staticmethod
    def _dbscan_labels(coords: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
        """DBSCANでクラスタラベルを取得"""
        # scikit-learn 未インストール環境では DBSCAN は使用不可。
        # DBSCAN パスを利用するときにのみ明示的にエラーを投げる。
        if SklearnDBSCAN is None:  # type: ignore[truthy-function]
            raise ImportError(
                "scikit-learn が見つかりません。DBSCAN ベースの疎探索を使用しない場合はこのままで問題ありません。"
                "DBSCAN を使用する場合は scikit-learn をインストールしてください。"
            )
        if coords.size == 0:
            return np.empty((0,), dtype=np.int32)

        clustering = SklearnDBSCAN(eps=float(eps), min_samples=int(min_samples))
        labels = clustering.fit_predict(coords.astype(np.float32, copy=False))
        return labels.astype(np.int32, copy=False)

    @torch.inference_mode()
    def sparse_search(
        self, input_image: torch.Tensor, return_context: bool = False
    ) -> Union[Tuple[slice, ...], Tuple[Tuple[slice, ...], Dict[str, Any]]]:
        # roi_bbox = tuple(slice(0, s) for s in tuple(int(x) for x in input_image.shape[1:]))
        # if return_context:
        #     return roi_bbox, {}
        # return roi_bbox

        # DBSCANクラスタ重心を用いた固定サイズROI疎探索
        if self.verbose:
            print("疎探索(DBSCAN)フェーズを開始...")

        downscale_factors = [1] + [1 / self.sparse_downscale_factor] * (input_image.ndim - 1)
        downsampled = torch.nn.functional.interpolate(
            input_image.unsqueeze(0),
            scale_factor=downscale_factors[1:],
            mode="trilinear" if input_image.ndim == 4 else "bilinear",
            align_corners=False,
        ).squeeze(0)

        if self.verbose:
            print(f"ダウンサンプリング: {input_image.shape} -> {downsampled.shape}")

        target_size = tuple(int(x) for x in input_image.shape[1:])
        original_tile_step_size = self.tile_step_size

        try:
            self.tile_step_size = 1.0 - self.sparse_overlap
            if self.device.type == "cuda":
                downsampled = downsampled.to(device=self.device, dtype=torch.half)
                with torch.autocast(device_type="cuda"):
                    sparse_logits = self.predict_sliding_window_return_logits(downsampled)
            else:
                sparse_logits = self.predict_sliding_window_return_logits(downsampled)
        finally:
            self.tile_step_size = original_tile_step_size

        sparse_probs = torch.softmax(sparse_logits, dim=0)

        # SI方向ガード（DBSCAN疎探索内のみ適用）
        # - SI軸の物理長が他軸より最大かつ max_si_extent_mm を超える場合、
        #   SI軸の上端側 vox_allow 分のみを保持し、それより下（胸部側）を背景として扱う
        if self.si_axis is not None and sparse_probs.dim() >= 2:
            try:
                si_ax = int(self.si_axis)
            except Exception:
                si_ax = -1
            spatial_nd = sparse_probs.dim() - 1
            if 0 <= si_ax < spatial_nd:
                try:
                    spacing = list(self.configuration_manager.spacing)  # type: ignore[attr-defined]
                except Exception:
                    spacing = [1.0] * spatial_nd
                while len(spacing) < spatial_nd:
                    spacing = [spacing[0]] + spacing
                spacing = spacing[:spatial_nd]

                extents_mm = [float(target_size[i]) * float(spacing[i]) for i in range(spatial_nd)]
                si_extent_mm = float(extents_mm[si_ax])
                max_extent_mm = float(max(extents_mm)) if extents_mm else 0.0

                if si_extent_mm >= max_extent_mm - 1e-6 and si_extent_mm > float(self.max_si_extent_mm):
                    eff_spacing_lowres = float(spacing[si_ax]) * float(self.sparse_downscale_factor)
                    vox_allow = int(np.ceil(float(self.max_si_extent_mm) / max(1e-6, eff_spacing_lowres)))
                    full_len = int(sparse_probs.shape[1 + si_ax])
                    if 0 < vox_allow < full_len:
                        # 復元用に元の確率を保持（低解像度）。COM中心の許容窓内は最終的に元へ戻す
                        sparse_probs_backup = sparse_probs.clone()
                        keep_start = full_len - vox_allow
                        sl = [slice(None)] * sparse_probs.dim()
                        sl[1 + si_ax] = slice(0, keep_start)
                        sparse_probs[(slice(1, None),) + tuple(sl[1:])] = 0.0
                        sparse_probs[(0,) + tuple(sl[1:])] = 1.0
                        if self.verbose:
                            print(
                                f"DBSCAN-sparse SIガード: 軸{si_ax}, 物理長 {si_extent_mm:.1f}mm (最大 {max_extent_mm:.1f}mm) > 上限 {float(self.max_si_extent_mm):.1f}mm\n"
                                f" -> 低解像度で下側 {keep_start} voxel を背景化"
                            )

                        # 追加の安全策: SI方向の重心を用いた中央ウィンドウでのトリム
                        # - 重心（COM）を中心に max_si_extent_mm の範囲のみ保持し、それ以外を背景化
                        try:
                            if sparse_probs.shape[0] > 1:
                                vp = torch.max(sparse_probs[1:], dim=0)[0]
                            else:
                                vp = sparse_probs[0]
                            # SI軸以外を総和して1D分布へ
                            reduce_axes = tuple(i for i in range(vp.dim()) if i != si_ax)
                            prof = vp.sum(dim=reduce_axes)
                            mass = prof.sum()
                            if mass > 0:
                                idxs = torch.arange(full_len, device=vp.device, dtype=vp.dtype)
                                com_t = (prof * idxs).sum() / mass
                            else:
                                com_t = torch.tensor((full_len - 1) * 0.5, device=vp.device, dtype=vp.dtype)

                            half_extent = int(
                                np.ceil(0.5 * float(self.max_si_extent_mm) / max(1e-6, eff_spacing_lowres))
                            )
                            c_idx = int(torch.round(com_t).item())
                            z0 = max(0, c_idx - half_extent)
                            z1 = min(full_len - 1, c_idx + half_extent)
                            if z1 <= z0:
                                z0 = max(0, (full_len // 2) - half_extent)
                                z1 = min(full_len - 1, z0 + 2 * half_extent)

                            # COM中心ウィンドウ外を背景化（[0:z0) と (z1+1:full_len)）
                            if z0 > 0:
                                sl_pre = [slice(None)] * sparse_probs.dim()
                                sl_pre[1 + si_ax] = slice(0, z0)
                                sparse_probs[(slice(1, None),) + tuple(sl_pre[1:])] = 0.0
                                sparse_probs[(0,) + tuple(sl_pre[1:])] = 1.0
                            if z1 + 1 < full_len:
                                sl_post = [slice(None)] * sparse_probs.dim()
                                sl_post[1 + si_ax] = slice(z1 + 1, full_len)
                                sparse_probs[(slice(1, None),) + tuple(sl_post[1:])] = 0.0
                                sparse_probs[(0,) + tuple(sl_post[1:])] = 1.0

                            # COM許容窓 [z0:z1] 内は元の疎探索確率へ復元（先に下側で切られていても復元する）
                            restore_sl = [slice(None)] * sparse_probs.dim()
                            restore_sl[1 + si_ax] = slice(z0, z1 + 1)
                            sparse_probs[(slice(None),) + tuple(restore_sl[1:])] = sparse_probs_backup[
                                (slice(None),) + tuple(restore_sl[1:])
                            ]

                            if self.verbose:
                                print(
                                    f"DBSCAN-sparse SIガード(重心調整): COM={c_idx}, 範囲[{z0}:{z1}] vox, 半幅 {half_extent}vox — 許容窓を元予測に復元"
                                )
                        except Exception:
                            # COM計算に失敗しても推論は継続
                            pass

        sparse_context: Optional[Dict[str, Any]] = None
        if return_context:
            seg_lowres = torch.argmax(sparse_probs, dim=0)
            seg_lowres_cpu = seg_lowres.detach().to(device="cpu", dtype=torch.int16)
            seg_highres = torch.nn.functional.interpolate(
                seg_lowres[None, None].to(dtype=torch.float32),
                size=target_size,
                mode="nearest",
            )[0, 0]
            seg_highres_cpu = seg_highres.detach().to(device="cpu", dtype=torch.int16)
            sparse_context = {
                "segmentation_lowres": seg_lowres_cpu,
                "segmentation_highres": seg_highres_cpu,
                "num_classes": int(sparse_logits.shape[0]),
                "input_shape": tuple(int(x) for x in input_image.shape),
                "downsampled_shape": tuple(int(x) for x in downsampled.shape),
                "network_spacing": tuple(
                    float(x) for x in getattr(self.configuration_manager, "spacing", [])
                ),
            }
            del seg_lowres, seg_highres

        if sparse_probs.shape[0] > 1:
            vessel_probs = torch.max(sparse_probs[1:], dim=0)[0]
        else:
            vessel_probs = sparse_probs[0]

        mask_bool_lowres = vessel_probs > 0.8

        if mask_bool_lowres.sum() == 0:
            roi_bbox = tuple(slice(0, s) for s in target_size)
            if return_context:
                return roi_bbox, sparse_context if sparse_context is not None else {}
            return roi_bbox

        coords = torch.nonzero(mask_bool_lowres, as_tuple=False)
        vessel_probs_cpu = vessel_probs.detach().to(device="cpu")
        coords_cpu = coords.detach().to(device="cpu")
        point_values = vessel_probs_cpu[tuple(coords_cpu.t())]

        total_points = coords_cpu.shape[0]
        if total_points > self.dbscan_max_points:
            topk = torch.topk(point_values, k=self.dbscan_max_points)
            coords_cpu = coords_cpu[topk.indices]
            point_values = topk.values

        coords_np = coords_cpu.numpy().astype(np.float32, copy=False)
        values_np = point_values.numpy().astype(np.float32, copy=False)

        labels_np = self._dbscan_labels(coords_np, self.dbscan_eps_voxels, self.dbscan_min_samples)
        unique_labels = [lab for lab in np.unique(labels_np) if lab >= 0]

        cluster_summaries: List[Dict[str, Any]] = []
        for lab in unique_labels:
            member_idx = np.where(labels_np == lab)[0]
            if member_idx.size == 0:
                continue
            m_coords = coords_np[member_idx]
            m_vals = values_np[member_idx]
            wsum = float(np.sum(m_vals))
            if wsum > 0.0:
                centroid_lowres = (m_coords * m_vals[:, None]).sum(axis=0) / wsum
            else:
                centroid_lowres = m_coords.mean(axis=0)
            score = float(np.sum(m_vals))
            cluster_summaries.append(
                {
                    "label": int(lab),
                    "score": score,
                    "count": int(member_idx.size),
                    "centroid_lowres": centroid_lowres,
                }
            )

        if not cluster_summaries:
            selected_highres = torch.nn.functional.interpolate(
                mask_bool_lowres.float()[None, None], size=target_size, mode="nearest"
            )[0, 0]
            selected_mask_highres = (selected_highres > 0.5).to(dtype=torch.half)
            full_size = torch.tensor(target_size, device=selected_mask_highres.device)
            bbox = self._bbox_from_mask(
                selected_mask_highres > 0.5, int(self.sparse_bbox_margin_voxels), full_size
            )
            roi_bbox = bbox if bbox is not None else tuple(slice(0, s) for s in target_size)
            if return_context:
                return roi_bbox, sparse_context if sparse_context is not None else {}
            return roi_bbox

        # 向き非依存クラスタ選択: 密度・確信度・サイズの順に優先
        def _cluster_metrics(item: Dict[str, Any]) -> Tuple[float, float, float]:
            lab = int(item["label"])  # クラスタラベル
            member_idx = np.where(labels_np == lab)[0]
            if member_idx.size == 0:
                return (0.0, 0.0, 0.0)
            pts = coords_np[member_idx]
            vals = values_np[member_idx]
            # 上位Kの平均確信度（過分散な大クラスタより局所的に濃いクラスタを優先）
            k = int(min(256, vals.shape[0]))
            conf = float(np.mean(np.partition(vals, -k)[-k:])) if k > 0 else 0.0
            size = float(pts.shape[0])
            return (conf, size)

        best_cluster = max(cluster_summaries, key=_cluster_metrics)
        centroid_lowres = best_cluster["centroid_lowres"]

        lowres_shape = np.array(mask_bool_lowres.shape, dtype=np.float32)
        scale = np.divide(
            np.array(target_size, dtype=np.float32),
            lowres_shape,
            out=np.ones_like(lowres_shape),
            where=lowres_shape > 0,
        )
        center_highres = centroid_lowres * scale
        center_highres = np.clip(center_highres, 0.0, np.array(target_size, dtype=np.float32) - 1.0)

        try:
            spacing = list(self.configuration_manager.spacing)
        except Exception:
            spacing = [1.0] * len(target_size)
        while len(spacing) < len(target_size):
            spacing = [spacing[0]] + spacing
        spacing = spacing[: len(target_size)]

        if len(self.dbscan_roi_extent_mm) == 1:
            extents_mm: List[float] = [self.dbscan_roi_extent_mm[0]] * len(target_size)
        else:
            extents_mm = list(self.dbscan_roi_extent_mm[: len(target_size)])
            if len(extents_mm) < len(target_size):
                extents_mm.extend([extents_mm[-1]] * (len(target_size) - len(extents_mm)))

        roi_slices: List[slice] = []
        for axis, size_axis in enumerate(target_size):
            spacing_axis = float(spacing[axis]) if axis < len(spacing) else 1.0
            desired_mm = float(extents_mm[axis]) if axis < len(extents_mm) else float(extents_mm[-1])
            if desired_mm <= 0.0 or spacing_axis <= 0.0:
                desired_vox = size_axis
            else:
                desired_vox = int(np.ceil(desired_mm / spacing_axis))
            desired_vox = max(1, min(desired_vox, size_axis))

            center_vox = float(center_highres[axis])
            start = int(round(center_vox - desired_vox / 2.0))
            max_start = max(0, size_axis - desired_vox)
            start = max(0, min(start, max_start))
            end = start + desired_vox
            if end > size_axis:
                end = size_axis
                start = max(0, end - desired_vox)
            roi_slices.append(slice(start, end))

        roi_bbox = tuple(roi_slices)

        if return_context:
            cluster_meta = {
                "score": best_cluster["score"],
                "count": best_cluster["count"],
                "center_vox": [float(center_highres[i]) for i in range(len(center_highres))],
            }
            if sparse_context is None:
                sparse_context = {}
            sparse_context["dbscan_cluster"] = cluster_meta
            # 選択クラスタ以外の予測をゼロ化（向き非依存）
            try:
                best_lab = int(best_cluster.get("label", -1))
                # 低解像度のクラスタマスクを作成
                selected_idx = np.where(labels_np == best_lab)[0]
                if selected_idx.size > 0:
                    # マスクはCPUテンソルで保持
                    sel_mask_low = torch.zeros(mask_bool_lowres.shape, dtype=torch.bool, device="cpu")
                    sel_coords = coords_cpu[selected_idx].long()
                    if sel_coords.numel() > 0:
                        sel_mask_low[tuple(sel_coords.t())] = True

                    # segmentation_lowres が存在すればゼロ化を適用
                    if isinstance(sparse_context.get("segmentation_lowres"), torch.Tensor):
                        seg_lr = sparse_context["segmentation_lowres"].clone()
                        seg_lr[~sel_mask_low] = 0
                        sparse_context["segmentation_lowres"] = seg_lr

                    # 高解像度側のマスクへ最近傍アップサンプリングしてゼロ化
                    if isinstance(sparse_context.get("segmentation_highres"), torch.Tensor):
                        high_shape = tuple(int(x) for x in target_size)
                        sel_mask_high = torch.nn.functional.interpolate(
                            sel_mask_low.float()[None, None], size=high_shape, mode="nearest"
                        )[0, 0].to(dtype=torch.bool)
                        seg_hr = sparse_context["segmentation_highres"].clone()
                        seg_hr[~sel_mask_high] = 0
                        sparse_context["segmentation_highres"] = seg_hr
            except Exception:
                # 失敗しても推論自体は継続
                pass
            return roi_bbox, sparse_context
        return roi_bbox

    def predict_single_npy_array(self) -> np.ndarray:
        raise NotImplementedError("predict_single_npy_array is not supported")

    def predict_from_list_of_npy_arrays(self) -> List[np.ndarray]:
        raise NotImplementedError("predict_from_list_of_npy_arrays is not supported")
