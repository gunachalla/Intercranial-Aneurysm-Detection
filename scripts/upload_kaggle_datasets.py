# %%
import kagglehub
import subprocess
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.my_utils.kaggle_utils import my_zip

competition_name = "rsna2025"

# %%
kagglehub.dataset_upload(f"tomoon33/{competition_name}-src", "/workspace/src")

# %%
kagglehub.dataset_upload(f"tomoon33/{competition_name}-nnunet", "/workspace/nnUNet")

# %%
kagglehub.dataset_upload(f"tomoon33/{competition_name}-configs", "/workspace/configs")

# zip_path = my_zip(f"{competition_name}-configs", "/workspace/configs")

# kagglehub.dataset_upload(f"tomoon33/{competition_name}-configs", zip_path)

# %%
# kagglehub.dataset_upload(f"tomoon33/{competition_name}-scripts", "/workspace/scripts")

zip_path = my_zip(f"{competition_name}-scripts", "/workspace/scripts")
kagglehub.dataset_upload(f"tomoon33/{competition_name}-scripts", zip_path)

# %%
kagglehub.dataset_upload(f"tomoon33/{competition_name}-pip-packages", "/workspace/pip_packages")

# %%
# nnUNet models
run_name = "nnUNetTrainerSkeletonRecall_more_DAv3_ep800__nnUNetResEncUNetMPlans__3d_fullres"
dataset_name = "nnUNet-da3-sklr-ep800"

zip_path = my_zip(
    dataset_name,
    f"/workspace/logs/nnUNet_results/Dataset001_VesselSegmentation/{run_name}",
    exclude_patterns=["./fold_*/validation/*"],
)

kagglehub.dataset_upload(f"tomoon33/{dataset_name}", zip_path)

# %%
# nnUNet models for sparse search
run_name = "RSNA2025Trainer_moreDAv7__nnUNetResEncUNetMPlans__3d_fullres"
dataset_name = "nnUNet-vessel-grouping-da7"

zip_path = my_zip(
    dataset_name,
    f"/workspace/logs/nnUNet_results/Dataset003_VesselGrouping/{run_name}",
    exclude_patterns=["./fold_*/validation/*"],
)

kagglehub.dataset_upload(f"tomoon33/{dataset_name}", zip_path)

# %%
# ROI models
run_name = "251011-seg_tf-v4-nnunet_truncate1-pretrained_1e-3_e30-ex_dav6w3-m32g64-w1_1_01"
dataset_name = "251011-pretrained-1e-3-e30-ex-dav6w3"

zip_path = my_zip(
    dataset_name,
    base_dir="/workspace/logs/train/runs",
    target_dir=f"./{run_name}",
)

kagglehub.dataset_upload(f"tomoon33/{dataset_name}", zip_path)

# %%
# ROI models
run_name = "251012-seg_tf-v4-nnunet_truncate1_preV6_1-ex_dav6w3-m32g64-e30-w01_005_1"
dataset_name = "251012-preV6_1-ex-dav6w3-e30-w01-005-1"

zip_path = my_zip(
    dataset_name,
    base_dir="/workspace/logs/train/runs",
    target_dir=f"./{run_name}",
)

kagglehub.dataset_upload(f"tomoon33/{dataset_name}", zip_path)

# %%
# ROI models
run_name = "251013-seg_tf-v4-nnunet_truncate1_preV6_1-ex_dav6w3-m32g64-e25-w01_005_1-s128_256_256"
dataset_name = "251013-prev6-1-ex-dav6w3-e25-w01-005-1-s128-256"

zip_path = my_zip(
    dataset_name,
    base_dir="/workspace/logs/train/runs",
    target_dir=f"./{run_name}",
)

kagglehub.dataset_upload(f"tomoon33/{dataset_name}", zip_path)


# %%
