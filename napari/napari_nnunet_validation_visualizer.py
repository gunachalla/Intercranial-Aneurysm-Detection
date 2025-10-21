"""
nnUNetのvalidation予測結果をnapariで可視化するツール
"""

import numpy as np
import napari
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import nibabel as nib
import json
import argparse
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.my_utils.rsna_utils import load_nifti_and_convert_to_ras


class NapariNnunetValidationVisualizer:
    """nnUNetのvalidation結果をnapariで可視化するクラス"""

    def __init__(
        self,
        result_dir: str,
        fold: Union[int, str] = 0,
        images_dir: str = "/workspace/data/nnUNet/nnUNet_raw/Dataset001_VesselSegmentation/imagesTr",
        labels_dir: str = "/workspace/data/nnUNet/nnUNet_raw/Dataset001_VesselSegmentation/labelsTr",
    ):
        """
        初期化

        Args:
            result_dir (str): nnUNet結果ディレクトリのパス
            fold (Union[int, str]): 表示するfold番号または"all"
            images_dir (str): 元画像のディレクトリパス
            labels_dir (str): Ground Truthのディレクトリパス
        """
        self.result_dir = Path(result_dir).expanduser()
        if not self.result_dir.exists():
            raise ValueError(f"Result directory not found: {self.result_dir}")
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)

        # 利用可能なfoldリストを取得
        self.available_folds = self._get_available_folds()
        if len(self.available_folds) == 0:
            raise ValueError(f"No validation directories found under: {self.result_dir}")

        # fold引数を正規化
        self.fold = self._resolve_fold_selection(fold)
        if self.fold != "all" and self.fold not in self.available_folds:
            raise ValueError(f"Fold {self.fold} not found in result directory")

        # fold切り替え候補を準備
        self.fold_options: List[Union[int, str]] = list(self.available_folds)
        if (len(self.available_folds) > 1 or self.fold == "all") and "all" not in self.fold_options:
            self.fold_options.append("all")

        # メトリクスとfold情報を保持するキャッシュ
        self._metrics_cache: Dict[Union[int, str], Optional[Dict]] = {}
        self.path_to_fold: Dict[Path, Union[int, str]] = {}

        # ビューワー関連
        self.viewer = None
        self.prediction_list = self._get_prediction_list()
        self.current_index = 0

        # 表示モード（0: 予測のみ, 1: GTのみ, 2: 差分表示）
        self.display_mode = 0
        self.display_mode_names = ["予測のみ", "GTのみ", "差分表示"]

        self.show_metrics = False

    def _fold_sort_key(self, fold: Union[int, str]) -> Tuple[int, Union[int, str]]:
        """fold一覧整列用のキーを生成"""
        if isinstance(fold, int):
            return (0, fold)
        return (1, fold)

    def _get_available_folds(self) -> List[Union[int, str]]:
        """利用可能なfoldのリストを取得"""
        folds: List[Union[int, str]] = []
        for fold_dir in sorted(self.result_dir.glob("fold_*")):
            if not fold_dir.is_dir():
                continue

            suffix = fold_dir.name.split("_", 1)[1]
            if not (fold_dir / "validation").exists():
                continue

            if suffix.isdigit():
                folds.append(int(suffix))
            else:
                folds.append(suffix)

        folds.sort(key=self._fold_sort_key)
        return folds

    def _resolve_fold_selection(self, fold: Union[int, str]) -> Union[int, str]:
        """fold引数を正規化"""
        if isinstance(fold, str):
            fold_lower = fold.lower()
            if fold_lower == "all":
                return "all"
            if fold_lower.startswith("fold_"):
                remainder = fold_lower[len("fold_") :]
                if remainder.isdigit():
                    return int(remainder)
                if remainder:
                    return remainder
            try:
                return int(fold)
            except ValueError as exc:
                raise ValueError(f"Invalid fold value: {fold}") from exc
        return fold

    def _get_validation_dir(self, fold: Union[int, str]) -> Path:
        """foldに対応するvalidationディレクトリを取得"""
        validation_dir = self.result_dir / f"fold_{fold}" / "validation"
        if not validation_dir.exists():
            raise ValueError(f"Validation directory not found: {validation_dir}")
        return validation_dir

    def _format_selection_label(self, selection: Union[int, str]) -> str:
        """fold選択の表示用文字列を生成"""
        if selection == "all":
            return "fold_all"
        return f"fold_{selection}"

    def _get_prediction_list(self) -> List[Path]:
        """予測ファイルのリストを取得"""
        self.path_to_fold.clear()

        if self.fold == "all":
            target_folds = self.available_folds
        else:
            target_folds = [self.fold]

        predictions: List[Path] = []
        for fold in target_folds:
            validation_dir = self._get_validation_dir(fold)
            fold_predictions = [
                p for p in validation_dir.glob("*.nii.gz") if p.name != "summary.json"
            ]
            fold_predictions.sort()
            for pred_path in fold_predictions:
                self.path_to_fold[pred_path] = fold
            predictions.extend(fold_predictions)

        predictions.sort(
            key=lambda p: self._fold_sort_key(self.path_to_fold[p]) + (p.name,)
        )

        if self.fold == "all":
            print(
                f"Found {len(predictions)} prediction files across {len(target_folds)} folds"
            )
        else:
            print(f"Found {len(predictions)} prediction files in fold_{self.fold}")

        return predictions

    def _get_metrics_for_fold(self, fold: Union[int, str]) -> Optional[Dict]:
        """validation/summary.jsonから評価メトリクスを読み込む"""
        if fold in self._metrics_cache:
            return self._metrics_cache[fold]

        summary_file = self._get_validation_dir(fold) / "summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, "r") as f:
                    data = json.load(f)
                print(f"Loaded evaluation metrics from {summary_file}")
                self._metrics_cache[fold] = data
                return data
            except Exception as e:
                print(f"Failed to load metrics for fold_{fold}: {e}")

        self._metrics_cache[fold] = None
        return None

    def _get_uid_from_prediction_path(self, prediction_path: Path) -> str:
        """予測ファイル名からUIDを取得"""
        # ファイル名から.nii.gzを除去してUIDを取得
        return prediction_path.name.replace(".nii.gz", "")

    def _get_image_path(self, uid: str) -> Path:
        """UIDから元画像のパスを取得"""
        return self.images_dir / f"{uid}_0000.nii.gz"

    def _get_label_path(self, uid: str) -> Path:
        """UIDからGround Truthのパスを取得"""
        return self.labels_dir / f"{uid}.nii.gz"

    def _setup_keyboard_bindings(self):
        """キーボードイベントをセットアップ"""

        @self.viewer.bind_key("n")
        def next_case(viewer):
            """次のケースに移動 (n key)"""
            if len(self.prediction_list) == 0:
                print("No prediction files available")
                return

            self.current_index = (self.current_index + 1) % len(self.prediction_list)
            current_file = self.prediction_list[self.current_index]
            print(f"\n=== Next Case ({self.current_index + 1}/{len(self.prediction_list)}) ===")
            print(f"File: {current_file.name}")
            self._load_and_display_case(current_file)

        @self.viewer.bind_key("p")
        def previous_case(viewer):
            """前のケースに移動 (p key)"""
            if len(self.prediction_list) == 0:
                print("No prediction files available")
                return

            self.current_index = (self.current_index - 1) % len(self.prediction_list)
            current_file = self.prediction_list[self.current_index]
            print(f"\n=== Previous Case ({self.current_index + 1}/{len(self.prediction_list)}) ===")
            print(f"File: {current_file.name}")
            self._load_and_display_case(current_file)

        @self.viewer.bind_key("v")
        def toggle_display_mode(viewer):
            """表示モードを切り替え (v key)"""
            self.display_mode = (self.display_mode + 1) % 3
            mode_name = self.display_mode_names[self.display_mode]
            print(f"表示モード: {mode_name}")
            self._update_display_mode()

            # 差分表示の場合は凡例を表示
            if self.display_mode == 2:  # 差分表示
                print("  凡例: ■ 白=True Positive, ■ 赤=False Positive, ■ 黄=False Negative")

        @self.viewer.bind_key("c")
        def reset_camera(viewer):
            """カメラをリセット (c key)"""
            print("Camera reset")
            viewer.reset_view()

        @self.viewer.bind_key("f")
        def switch_fold(viewer):
            """foldを切り替え (f key)"""
            if len(self.fold_options) <= 1:
                print("Only one fold option available")
                return

            current_fold_index = self.fold_options.index(self.fold)
            next_fold_index = (current_fold_index + 1) % len(self.fold_options)
            self.fold = self.fold_options[next_fold_index]

            self.prediction_list = self._get_prediction_list()
            self.current_index = 0

            selection_label = self._format_selection_label(self.fold)
            print(f"\nSwitched to {selection_label}")
            if len(self.prediction_list) > 0:
                self._load_and_display_case(self.prediction_list[0])

        @self.viewer.bind_key("e")
        def toggle_metrics(viewer):
            """評価メトリクスの表示を切り替え (e key)"""
            self.show_metrics = not self.show_metrics
            if self.show_metrics:
                self._display_metrics()
            else:
                print("Metrics display: OFF")

        print("\nKeyboard bindings:")
        print("  'n' - Next case")
        print("  'p' - Previous case")
        print("  'v' - Toggle display mode (Prediction/GT/Diff)")
        print("  'c' - Reset camera")
        print(
            "  'f' - Switch fold/all"
            if len(self.fold_options) > 1
            else "  'f' - Switch fold/all (disabled - only one option)"
        )
        print("  'e' - Toggle metrics display")

    def _update_display_mode(self):
        """現在の表示モードに基づいてレイヤーの表示を更新"""
        pred_layer = self._get_layer_by_name("予測")
        gt_layer = self._get_layer_by_name("Ground Truth")
        diff_layer = self._get_layer_by_name("差分")

        if self.display_mode == 0:  # 予測のみ
            if pred_layer:
                pred_layer.visible = True
            if gt_layer:
                gt_layer.visible = False
            if diff_layer:
                diff_layer.visible = False
        elif self.display_mode == 1:  # GTのみ
            if pred_layer:
                pred_layer.visible = False
            if gt_layer:
                gt_layer.visible = True
            if diff_layer:
                diff_layer.visible = False
        elif self.display_mode == 2:  # 差分表示
            if pred_layer:
                pred_layer.visible = False
            if gt_layer:
                gt_layer.visible = False
            if diff_layer:
                diff_layer.visible = True

        # タイトルを更新
        self._update_title()

    def _update_title(self):
        """ビューワーのタイトルを更新"""
        if self.viewer and len(self.prediction_list) > 0:
            current_file = self.prediction_list[self.current_index]
            case_fold = self.path_to_fold.get(current_file)
            mode_name = self.display_mode_names[self.display_mode]
            if self.fold == "all":
                if case_fold is not None and case_fold != "all":
                    fold_label = f"All (case fold {case_fold})"
                else:
                    fold_label = "All"
            else:
                fold_label = str(self.fold)
            title = (
                f"nnUNet Validation - Fold {fold_label} - Case "
                f"{self.current_index + 1}/{len(self.prediction_list)} - Mode: {mode_name}"
            )
            self.viewer.title = title

    def _display_metrics(self):
        """評価メトリクスを表示"""
        if len(self.prediction_list) == 0:
            print("No prediction files available")
            return

        current_file = self.prediction_list[self.current_index]
        case_fold = self.path_to_fold.get(current_file)
        if case_fold is None:
            print("Fold information not available for current case")
            return

        metrics = self._get_metrics_for_fold(case_fold)
        if not metrics:
            print(f"No metrics available for fold_{case_fold}")
            return

        uid = self._get_uid_from_prediction_path(current_file)

        print("\n=== Evaluation Metrics ===")
        print(f"Fold: fold_{case_fold}")

        mean_metrics = metrics.get("mean") if isinstance(metrics, dict) else None
        if isinstance(mean_metrics, dict):
            dice_value = mean_metrics.get("Dice")
            if isinstance(dice_value, (int, float)):
                print(f"Overall Mean Dice: {dice_value:.4f}")

        metric_per_case = metrics.get("metric_per_case") if isinstance(metrics, dict) else None
        if isinstance(metric_per_case, list):
            for case_metrics in metric_per_case:
                reference_file = case_metrics.get("reference_file", "")
                if uid in reference_file:
                    print(f"\nCurrent Case ({uid}):")
                    for key, value in case_metrics.items():
                        if key not in ["reference_file", "prediction_file"] and isinstance(
                            value, (int, float)
                        ):
                            print(f"  {key}: {value:.4f}")
                    break

    def _get_layer_by_name(self, name_pattern: str):
        """名前パターンでレイヤーを検索"""
        if not self.viewer:
            return None
        for layer in self.viewer.layers:
            if name_pattern in layer.name:
                return layer
        return None

    def _update_or_create_layer(self, layer_name: str, data=None, layer_type="image", **kwargs):
        """レイヤーが存在する場合はデータを更新、存在しない場合は新規作成"""
        if not self.viewer:
            return

        existing_layer = self._get_layer_by_name(layer_name)

        if data is None:
            if existing_layer is not None:
                existing_layer.visible = False
            return

        if existing_layer is not None:
            existing_layer.data = data
            existing_layer.visible = True

            if "scale" in kwargs:
                existing_layer.scale = kwargs["scale"]
            if "opacity" in kwargs:
                existing_layer.opacity = kwargs["opacity"]
            if "colormap" in kwargs and hasattr(existing_layer, "colormap"):
                existing_layer.colormap = kwargs["colormap"]
            if "color" in kwargs and hasattr(existing_layer, "color"):
                existing_layer.color = kwargs["color"]
        else:
            if layer_type == "image":
                self.viewer.add_image(data, name=layer_name, **kwargs)
            elif layer_type == "labels":
                self.viewer.add_labels(data, name=layer_name, **kwargs)

    def _compute_diff_mask(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """予測とGTの差分マスクを計算

        Returns:
            差分マスク（0: 背景, 1: True Positive, 2: False Positive, 3: False Negative）
        """
        diff = np.zeros_like(pred, dtype=np.uint8)

        # True Positive (両方が1)
        diff[(pred > 0) & (gt > 0)] = 1

        # False Positive (予測のみ1)
        diff[(pred > 0) & (gt == 0)] = 2

        # False Negative (GTのみ1)
        diff[(pred == 0) & (gt > 0)] = 3

        return diff

    def _load_and_display_case(self, prediction_path: Path):
        """指定された予測ファイルを読み込んで表示"""
        try:
            # UIDを取得
            uid = self._get_uid_from_prediction_path(prediction_path)
            print(f"Loading UID: {uid}")
            case_fold = self.path_to_fold.get(prediction_path)
            if case_fold is not None:
                print(f"Fold: fold_{case_fold}")

            # 予測を読み込む
            pred_data, _, _ = load_nifti_and_convert_to_ras(prediction_path)
            pred_zyx = np.transpose(pred_data, (2, 1, 0)).astype(np.uint32)

            # 元画像を読み込む
            image_path = self._get_image_path(uid)
            if not image_path.exists():
                print(f"Warning: Image file not found: {image_path}")
                return

            image_data, _, _ = load_nifti_and_convert_to_ras(image_path)
            image_zyx = np.transpose(image_data, (2, 1, 0))

            # ボクセルスペーシングを取得
            nii_img = nib.load(str(image_path))
            zooms_xyz = nii_img.header.get_zooms()[:3]
            voxel_spacing_zyx = (zooms_xyz[2], zooms_xyz[1], zooms_xyz[0])

            # 画像を正規化
            image_normalized = self._normalize_volume(image_zyx)

            # Ground Truthを読み込む
            gt_data = None
            label_path = self._get_label_path(uid)
            if label_path.exists():
                gt_data_raw, _, _ = load_nifti_and_convert_to_ras(label_path)
                gt_data = np.transpose(gt_data_raw, (2, 1, 0)).astype(np.uint32)
            else:
                print(f"Warning: Ground Truth file not found: {label_path}")

            # 差分マスクを計算
            diff_data = None
            if gt_data is not None:
                diff_data = self._compute_diff_mask(pred_zyx, gt_data)

            # レイヤーを更新
            self._update_or_create_layer(
                "元画像",
                image_normalized,
                layer_type="image",
                colormap="gray",
                scale=voxel_spacing_zyx,
                opacity=0.8,
            )

            # 予測レイヤー（青系）
            self._update_or_create_layer(
                "予測",
                pred_zyx,
                layer_type="labels",
                opacity=0.5,
                scale=voxel_spacing_zyx,
            )

            # Ground Truthレイヤー（緑系）
            if gt_data is not None:
                self._update_or_create_layer(
                    "Ground Truth",
                    gt_data,
                    layer_type="labels",
                    opacity=0.5,
                    scale=voxel_spacing_zyx,
                )

            # 差分レイヤー
            if diff_data is not None:
                # カスタムカラーマップを作成
                # napariのラベルレイヤーの色指定
                diff_colors = {
                    0: [0, 0, 0, 0],  # 背景（透明）
                    1: [0, 0, 1, 1],  # TP（青）
                    2: [1, 0, 0, 1],  # FP（赤）
                    3: [1, 1, 0, 1],  # FN（黄）
                }

                # 差分レイヤーを作成または更新
                self._update_or_create_layer(
                    "差分",
                    diff_data,
                    layer_type="labels",
                    opacity=0.8,
                    scale=voxel_spacing_zyx,
                    colormap=diff_colors,
                )

                # レイヤー名を更新して凡例を含める
                diff_layer = self._get_layer_by_name("差分")
                if diff_layer:
                    diff_layer.name = "差分 (青:TP, 赤:FP, 黄:FN)"

            # 表示モードを適用
            self._update_display_mode()

            # カメラをリセット
            if self.viewer:
                self.viewer.reset_view()

            # データ情報を表示
            print(f"\n=== データ情報 ===")
            print(f"予測形状: {pred_zyx.shape}")
            print(f"画像形状: {image_zyx.shape}")
            if gt_data is not None:
                print(f"GT形状: {gt_data.shape}")

                # 簡易メトリクスを計算
                pred_binary = pred_zyx > 0
                gt_binary = gt_data > 0

                tp = np.sum(pred_binary & gt_binary)  # True Positive
                fp = np.sum(pred_binary & ~gt_binary)  # False Positive
                fn = np.sum(~pred_binary & gt_binary)  # False Negative

                dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
                iou = tp / (tp + fp + fn + 1e-8)

                print(f"Quick Dice: {dice:.4f}")
                print(f"Quick IoU: {iou:.4f}")
                print(f"TP: {tp:,}, FP: {fp:,}, FN: {fn:,}")

            print(f"Voxel spacing: {voxel_spacing_zyx}")

            # メトリクスを表示（有効な場合）
            if self.show_metrics:
                self._display_metrics()

        except Exception as e:
            print(f"Error loading case {prediction_path}: {e}")
            import traceback

            traceback.print_exc()

    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """ボリュームデータを正規化"""
        volume_float = volume.astype(np.float32)

        # パーセンタイルベースの正規化
        p1 = np.percentile(volume_float, 1)
        p99 = np.percentile(volume_float, 99)

        if p99 > p1:
            volume_normalized = np.clip(volume_float, p1, p99)
            volume_normalized = (volume_normalized - p1) / (p99 - p1)
        else:
            if volume_float.max() > volume_float.min():
                volume_normalized = (volume_float - volume_float.min()) / (
                    volume_float.max() - volume_float.min()
                )
            else:
                volume_normalized = volume_float

        return volume_normalized

    def visualize(self, start_index: int = 0, uid: Optional[str] = None) -> napari.Viewer:
        """
        Napariビューワーを起動

        Args:
            start_index (int): 開始インデックス
            uid (str, optional): 特定のUIDを指定

        Returns:
            napari.Viewer: napariビューワー
        """
        if len(self.prediction_list) == 0:
            print("No prediction files found")
            return None

        # 特定のUIDが指定された場合、そのインデックスを探す
        if uid:
            uid_found = False
            for i, pred_path in enumerate(self.prediction_list):
                path_uid = self._get_uid_from_prediction_path(pred_path)
                if path_uid == uid:
                    start_index = i
                    uid_found = True
                    print(f"Found specified UID at index {i + 1}: {uid}")
                    break

            if not uid_found:
                print(f"Warning: Specified UID '{uid}' not found in validation results")

        # 開始インデックスを調整
        start_index = max(0, min(start_index, len(self.prediction_list) - 1))
        self.current_index = start_index

        # 最初のファイル
        first_file = self.prediction_list[start_index]
        print(f"Starting with file {start_index + 1}/{len(self.prediction_list)}: {first_file.name}")
        first_fold = self.path_to_fold.get(first_file)
        if first_fold is not None:
            print(f"Case fold: fold_{first_fold}")

        # napariビューワーを作成
        self.viewer = napari.Viewer(title=f"nnUNet Validation Viewer - {first_file.name}")

        # キーボードバインディングをセットアップ
        self._setup_keyboard_bindings()

        # 最初のケースを表示
        self._load_and_display_case(first_file)

        return self.viewer


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="nnUNet Validation Visualizer")
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Path to nnUNet result directory (e.g., /workspace/logs/nnUNet_results/Dataset001.../RSNA2025Trainer...)",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="0",
        help="Fold number to visualize or 'all' (default: 0)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="/workspace/data/nnUNet/nnUNet_raw/Dataset001_VesselSegmentation/imagesTr",
        help="Directory containing original images",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="/workspace/data/nnUNet/nnUNet_raw/Dataset001_VesselSegmentation/labelsTr",
        help="Directory containing ground truth labels",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index",
    )
    parser.add_argument(
        "--uid",
        type=str,
        default=None,
        help="Specific UID to start with",
    )

    args = parser.parse_args()

    print("=== nnUNet Validation Visualizer ===")
    print(f"Result directory: {args.result_dir}")
    fold_arg: Union[int, str]
    if args.fold.lower() == "all":
        fold_arg = "all"
    else:
        try:
            fold_arg = int(args.fold)
        except ValueError:
            print(f"Warning: Invalid fold argument '{args.fold}', falling back to raw string")
            fold_arg = args.fold

    print(f"Fold: {fold_arg}")
    print(f"Images directory: {args.images_dir}")
    print(f"Labels directory: {args.labels_dir}")

    try:
        # ビジュアライザーを作成
        visualizer = NapariNnunetValidationVisualizer(
            result_dir=args.result_dir,
            fold=fold_arg,
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
        )

        # 利用可能なfoldを表示
        if len(visualizer.available_folds) > 0:
            print(f"Available folds: {visualizer.available_folds}")

        # ビューワーを起動
        viewer = visualizer.visualize(start_index=args.start_index, uid=args.uid)

        if viewer:
            print("\n=== Napariビューワーが開きました ===")
            print("基本操作:")
            print("- マウスホイール: ズーム")
            print("- 右クリック+ドラッグ: 回転")
            print("- 左クリック+ドラッグ: 平行移動")
            print("- スライダー: 各軸のスライス位置調整")

            # napariを実行
            napari.run()
        else:
            print("可視化に失敗しました")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
