import subprocess
import os
from hydra import compose, initialize_config_dir


def get_ckpt_name(ckpt_type):
    if ckpt_type == "last":
        ckpt_name = "last.ckpt"
    elif ckpt_type == "best":
        ckpt_name = f"epoch_*.ckpt"
    else:
        assert False, f"unknown ckpt_type: {ckpt_type}"
    return ckpt_name


def my_zip(zip_name, base_dir, target_dir=".", exclude_patterns=[], save_dir="/tmp"):
    if exclude_patterns is None:
        exclude_patterns = []

    zip_path = os.path.join(save_dir, f"{zip_name}.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    # zip -r <zip_path> <target_dir> [-x <pat1> <pat2> ...]
    cmd = ["zip", "-r", zip_path, target_dir]
    if exclude_patterns:
        cmd += ["-x", *exclude_patterns]

    print("argv:", cmd)  # Show actual argv as a list (not joined)
    subprocess.run(cmd, check=True, cwd=base_dir)  # Use cwd instead of chdir
    return zip_path


class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def load_experiment_config(experiment: str, config_dir: str = "/workspace/configs"):
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}"],
            return_hydra_config=True,
        )
        cfg.paths.output_dir = "${hydra.runtime.output_dir}"
        cfg.paths.work_dir = "${hydra.runtime.cwd}"
        cfg.hydra.run.dir = cfg.log_dir
        cfg.hydra.runtime.output_dir = cfg.hydra.run.dir

    if cfg.model.get("pretrained_ckpt_path"):
        cfg.model.pretrained_ckpt_path = None
    if cfg.model.get("my_pretrained_path"):
        cfg.model.my_pretrained_path = None
    if cfg.model.net.get("pretrained"):
        cfg.model.net.pretrained = False
    if cfg.model.net.get("backbone"):
        if cfg.model.net.backbone.get("pretrained"):
            cfg.model.net.backbone.pretrained = False
    if cfg.model.net.get("grad_checkpointing"):
        cfg.model.net.grad_checkpointing = False

    if cfg.model.get("pretrained_net_ckpt"):
        cfg.model.pretrained_net_ckpt = None

    return cfg
