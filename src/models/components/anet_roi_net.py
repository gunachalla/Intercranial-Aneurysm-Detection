from __future__ import annotations

import math
import torch
import torch.nn as nn

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Pretrained backbone using nnUNet (dynamic_network_architectures)
import json  # noqa: E402

try:
    # dynamic_network_architectures provides nnUNet v2 implementations
    from dynamic_network_architectures.architectures.unet import UNetDecoder
    from dynamic_network_architectures.building_blocks.residual import (
        BasicBlockD,
        BottleneckD,
    )
    from dynamic_network_architectures.building_blocks.regularization import DropPath
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
except Exception:  # pragma: no cover
    UNetDecoder = None  # type-check bypass
    get_network_from_plans = None
    BasicBlockD = None
    BottleneckD = None
    DropPath = None


class AneurysmRoiBackboneNnUNet(nn.Module):
    """
    Reuse a pretrained nnUNet (3D) encoder/decoder and output the last decoder features
    as ROI-classification features.

    - Load `plans.json` from the specified nnUNet results directory to reconstruct the network
    - Load the specified fold checkpoint (e.g., `checkpoint_final.pth`)
    - Project last decoder features (high-res) via 1x1 conv to desired channels
    - Output 1-channel logits for a sphere-mask auxiliary head

    Outputs:
      - feat: (B, C, D, H, W)  high-res features for classification (last decoder -> 1x1 projection)
      - logits_sphere: (B, 1, D, H, W)  logits for sphere mask
    """

    def __init__(
        self,
        # nnUNet-related
        nnunet_model_dir: str,
        fold: int = 0,
        pretrained: bool = True,
        checkpoint_name: str = "checkpoint_final.pth",
        configuration: str = "3d_fullres",
        nnunet_in_channels: int = 1,
        # Output feature channels (to match existing heads)
        out_channels: int = None,
        # Whether to freeze the nnUNet part
        freeze_nnunet: bool = False,
        # Sphere head
        sphere_mid_channels: int = 32,
    ) -> None:
        super().__init__()
        self._init_nnunet(
            nnunet_model_dir,
            fold,
            pretrained,
            checkpoint_name,
            configuration,
            nnunet_in_channels,
            freeze_nnunet,
        )

        self.encoder_feature_channels = self.nnunet.encoder.output_channels[-1]

        # Last decoder output channels = encoder's first output channels
        # In ResidualEncoderUNet, the last decoder output equals encoder.output_channels[0]
        enc_out_ch0 = int(self.nnunet.encoder.output_channels[0])

        if out_channels is not None:
            # 1x1 projection: last decoder features -> out_channels
            self.proj = nn.Conv3d(enc_out_ch0, out_channels, kernel_size=1, bias=False)
            # self.proj_norm = nn.InstanceNorm3d(out_channels, affine=True)
            # self.proj_act = nn.SiLU(inplace=True)
        else:
            self.proj = nn.Identity()
            out_channels = enc_out_ch0

        self.out_channels = int(out_channels)

        # Sphere head (1ch logits)
        self._init_sphere_head(sphere_mid_channels)

    def _init_nnunet(
        self,
        nnunet_model_dir: str,
        fold: int,
        pretrained: bool,
        checkpoint_name: str,
        configuration: str,
        input_channels: int,
        freeze_nnunet: bool,
    ):
        assert get_network_from_plans is not None, "Failed to import nnUNetv2. Please check dependencies."

        # Load plans.json and obtain network configuration for the given configuration
        plans_path = f"{nnunet_model_dir.rstrip('/')}/plans.json"
        with open(plans_path, "r") as f:
            plans = json.load(f)
        cfg = plans["configurations"][configuration]
        arch = cfg["architecture"]

        self._nnunet_input_channels = int(input_channels)

        # Build nnUNet network (weights loaded later). Input/output channels typically depend
        # on dataset; here we use a single input and ignore final class output. We do not use
        # the final seg head, only decoder features, so set num_output_channels=1 (state_dict
        # loads only matching keys).
        self.nnunet = get_network_from_plans(
            arch_class_name=arch["network_class_name"],
            arch_kwargs=arch["arch_kwargs"],
            arch_kwargs_req_import=arch["_kw_requires_import"],
            input_channels=int(input_channels),
            output_channels=1,
            allow_init=True,
            deep_supervision=False,
        )

        if pretrained:
            # Load checkpoint (load matching encoder/decoder keys; ignore others)
            ckpt_path = f"{nnunet_model_dir.rstrip('/')}/fold_{fold}/{checkpoint_name}"
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            nn_state: dict = state.get("network_weights", state)
            # Exclude segmentation output layers (decoder.seg_layers.*) due to class-count mismatch
            filtered_state = {k: v for k, v in nn_state.items() if not k.startswith("decoder.seg_layers.")}
            model_state = self.nnunet.state_dict()
            filtered_state = self._adjust_input_conv_weights(filtered_state, model_state)
            missing, unexpected = self.nnunet.load_state_dict(filtered_state, strict=False)
            # Optionally freeze nnUNet
            if freeze_nnunet:
                for p in self.nnunet.parameters():
                    p.requires_grad_(False)

    def _adjust_input_conv_weights(
        self,
        pretrained_state: dict[str, torch.Tensor],
        model_state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Adjust first conv-like weights to match input channel count"""
        if self._nnunet_input_channels == 1:
            return pretrained_state

        for key, weight in list(pretrained_state.items()):
            if key not in model_state:
                continue
            model_weight = model_state[key]
            if weight.shape == model_weight.shape:
                continue
            if (
                weight.ndim == 5
                and model_weight.ndim == 5
                and weight.shape[0] == model_weight.shape[0]
                and weight.shape[2:] == model_weight.shape[2:]
            ):
                pretrained_state[key] = self._resize_conv_weight(weight, model_weight.shape[1])
        return pretrained_state

    @staticmethod
    def _resize_conv_weight(weight: torch.Tensor, target_in_channels: int) -> torch.Tensor:
        """Match input channels following timm-style approach"""
        out_c, in_c_pre, k_t, k_h, k_w = weight.shape
        if target_in_channels == in_c_pre:
            return weight
        if target_in_channels == 1 and in_c_pre > 1:
            # Average multi-channel pretrained weights into 1 channel
            return weight.mean(dim=1, keepdim=True)
        if target_in_channels < in_c_pre:
            # Truncate extra channels
            return weight[:, :target_in_channels, :, :, :].contiguous()
        repeat_times = math.ceil(target_in_channels / in_c_pre)
        weight_rep = weight.repeat(1, repeat_times, 1, 1, 1)[:, :target_in_channels, :, :, :]
        # Adjust factor to preserve output scale
        return weight_rep * (in_c_pre / float(target_in_channels))

    def _init_sphere_head(self, sphere_mid_channels: int):
        self.head_sphere = nn.Sequential(
            nn.Conv3d(self.out_channels, sphere_mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(sphere_mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(sphere_mid_channels, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.head_sphere[-1].weight)
        nn.init.constant_(self.head_sphere[-1].bias, -4.0)

    @torch.no_grad()
    def feature_channels(self) -> int:
        """Number of channels of classification feature maps"""
        return int(self.out_channels)

    def _forward_decoder_to_last_feature(self, skips: list[torch.Tensor]) -> torch.Tensor:
        """Run nnUNet decoder and return last-stage feature map (before seg head)."""
        dec: UNetDecoder = self.nnunet.decoder  # type: UNetDecoder
        lres_input = skips[-1]
        for s in range(len(dec.stages)):
            x = dec.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = dec.stages[s](x)
            lres_input = x
        # x is the final-stage (high-res) feature
        return x

    def forward(
        self, x: torch.Tensor, vessel_seg: torch.Tensor, vessel_union: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # Get encoder skip connections
        skips = self.nnunet.encoder(x)
        # Run decoder to final stage and get features
        dec_feat = self._forward_decoder_to_last_feature(skips)
        # Projection (optional) + normalization/activation (if enabled)
        # last_feat = self.proj_act(self.proj_norm(self.proj(dec_feat)))
        last_feat = self.proj(dec_feat)
        # Aux sphere-mask logits
        logits_sphere = self.head_sphere(last_feat)
        return {"enc_feat": skips[-1], "dec_feat": last_feat, "logits_sphere": logits_sphere}


class AneurysmRoiBackboneNnUNetTruncatedDecoder(AneurysmRoiBackboneNnUNet):
    """
    Early-stop decoder upon reaching the last k stages and return an
    intermediate-resolution feature map (no PixelShuffle used).

    - For normal stages, use original transposed conv (decoder.transpconvs) + skip + stage
    - Stop before entering the last k stages (s == N - k) and output the feature there
    - Reinit sphere head to match channels at the stop point
    - If out_channels is provided, rebuild a 1x1 projection (proj) to that dim

    Outputs:
      - dec_feat: (B, C, D, H, W)  intermediate feature (after 1x1 projection; identity if None)
      - logits_sphere: (B, 1, D, H, W)
    """

    def __init__(
        self,
        # nnUNet-related
        nnunet_model_dir: str,
        fold: int = 0,
        pretrained: bool = True,
        checkpoint_name: str = "checkpoint_final.pth",
        configuration: str = "3d_fullres",
        # Output feature channels (use 1x1 projection when specified)
        out_channels: int | None = None,
        # Whether to freeze nnUNet
        freeze_nnunet: bool = False,
        # Early-stop at last k stages
        num_truncate_stages: int = 1,
        # Sphere head
        sphere_mid_channels: int = 32,
        nnunet_in_channels: int = 1,
    ) -> None:
        # Build nnUNet, load weights, and init (temporary) head in parent class
        super().__init__(
            nnunet_model_dir=nnunet_model_dir,
            fold=fold,
            pretrained=pretrained,
            checkpoint_name=checkpoint_name,
            configuration=configuration,
            out_channels=out_channels,
            freeze_nnunet=freeze_nnunet,
            nnunet_in_channels=nnunet_in_channels,
            sphere_mid_channels=sphere_mid_channels,
        )

        # Validate inputs and compute cutoff index for early stop
        dec: UNetDecoder = self.nnunet.decoder  # type: UNetDecoder
        num_stages_total = len(dec.stages)
        k = int(num_truncate_stages)
        if k < 0 or k > num_stages_total:
            raise ValueError(f"num_truncate_stages must be in 0..{num_stages_total}: {k}")
        self._cutoff_stage_index: int = num_stages_total - k  # s == cutoff -> early return

        # Estimate feature channels at cutoff:
        # cutoff == 0 -> matches channels of skips[-1]
        # cutoff in [1, N] -> matches in_channels of transpconvs[cutoff]
        if self._cutoff_stage_index == num_stages_total:
            # No early stop (k==0): same as parent last-stage output channels
            truncated_out_ch = self.out_channels
        else:
            # Ensure cutoff in range and read from in_channels
            transp_next: nn.Module = dec.transpconvs[self._cutoff_stage_index]
            if not hasattr(transp_next, "in_channels"):
                raise RuntimeError("decoder.transpconvs structure unexpected (missing in_channels).")
            truncated_out_ch = int(getattr(transp_next, "in_channels"))

        # Rebuild heads/projection to match channels at cutoff
        # - out_channels None: keep Identity projection; reinit sphere head only
        # - out_channels set: rebuild 1x1 projection and attach sphere head to its output channels
        if out_channels is None:
            # Keep Identity projection; output channels = truncated_out_ch
            self.out_channels = truncated_out_ch
            self._init_sphere_head(sphere_mid_channels)
        else:
            # Rebuild projection (in: truncated_out_ch -> out: out_channels)
            self.proj = nn.Conv3d(truncated_out_ch, int(out_channels), kernel_size=1, bias=False)
            self.out_channels = int(out_channels)
            self._init_sphere_head(sphere_mid_channels)

    def _forward_decoder_to_last_feature(self, skips: list[torch.Tensor]) -> torch.Tensor:
        """Run decoder and return feature right before the last k stages (early stop)."""
        dec: UNetDecoder = self.nnunet.decoder
        lres_input = skips[-1]
        for s in range(len(dec.stages)):
            # Do not enter the last k stages (>= cutoff); return previous feature
            if s == self._cutoff_stage_index:
                return lres_input
            x = dec.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = dec.stages[s](x)
            lres_input = x
        # For k==0 (cutoff==N), process through to the final stage
        return lres_input

    def forward(
        self, x: torch.Tensor, vessel_seg: torch.Tensor, vessel_union: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # Get encoder skip connections
        skips = self.nnunet.encoder(x)
        # Run decoder up to the cutoff and get features
        dec_feat_trunc = self._forward_decoder_to_last_feature(skips)
        # 1x1 projection (identity if not specified)
        last_feat = self.proj(dec_feat_trunc)
        # Aux sphere-mask logits
        logits_sphere = self.head_sphere(last_feat)
        return {"enc_feat": skips[-1], "dec_feat": last_feat, "logits_sphere": logits_sphere}


class AneurysmRoiBackboneNnUNetTruncatedDecoderStochasticDepth(AneurysmRoiBackboneNnUNetTruncatedDecoder):
    """TruncatedDecoder variant with Stochastic Depth applied to nnUNet encoder"""

    def __init__(
        self,
        nnunet_model_dir: str,
        fold: int = 0,
        pretrained: bool = True,
        checkpoint_name: str = "checkpoint_final.pth",
        configuration: str = "3d_fullres",
        out_channels: int | None = None,
        freeze_nnunet: bool = False,
        num_truncate_stages: int = 1,
        sphere_mid_channels: int = 32,
        nnunet_in_channels: int = 1,
        stochastic_depth_max_rate: float = 0.1,
        stochastic_depth_mode: str = "linear",
    ) -> None:
        # Explicitly fail if DropPath is unavailable
        if DropPath is None or BasicBlockD is None or BottleneckD is None:
            raise RuntimeError("dynamic_network_architectures is required to enable Stochastic Depth")

        self._requested_stochastic_depth_rate = float(stochastic_depth_max_rate)
        self._stochastic_depth_mode = stochastic_depth_mode

        super().__init__(
            nnunet_model_dir=nnunet_model_dir,
            fold=fold,
            pretrained=pretrained,
            checkpoint_name=checkpoint_name,
            configuration=configuration,
            out_channels=out_channels,
            freeze_nnunet=freeze_nnunet,
            num_truncate_stages=num_truncate_stages,
            sphere_mid_channels=sphere_mid_channels,
            nnunet_in_channels=nnunet_in_channels,
        )

        if self._requested_stochastic_depth_rate <= 0.0:
            return

        if not (0.0 <= self._requested_stochastic_depth_rate < 1.0):
            raise ValueError("stochastic_depth_max_rate must be in [0.0, 1.0)")

        self._enable_stochastic_depth(
            max_rate=self._requested_stochastic_depth_rate,
            mode=self._stochastic_depth_mode,
        )

    def _enable_stochastic_depth(self, max_rate: float, mode: str) -> None:
        """Assign DropPath to residual blocks in the encoder"""
        encoder = getattr(self.nnunet, "encoder", None)
        if encoder is None or not hasattr(encoder, "stages"):
            raise RuntimeError("Failed to access nnUNet encoder")

        residual_blocks = self._collect_residual_blocks(encoder.stages)
        if not residual_blocks:
            raise RuntimeError("No residual blocks found to apply Stochastic Depth")

        drop_rates = self._build_drop_rates(len(residual_blocks), max_rate, mode)
        for block, drop_prob in zip(residual_blocks, drop_rates):
            self._assign_drop_path(block, drop_prob)

    def _collect_residual_blocks(self, stages: nn.Sequential) -> list[nn.Module]:
        """Collect residual blocks in depth order"""
        ordered_blocks: list[nn.Module] = []
        seen: set[int] = set()
        for stage in stages:
            for module in stage.modules():
                if isinstance(module, (BasicBlockD, BottleneckD)) and id(module) not in seen:
                    seen.add(id(module))
                    ordered_blocks.append(module)
        return ordered_blocks

    def _build_drop_rates(self, num_blocks: int, max_rate: float, mode: str) -> list[float]:
        """Generate DropPath rates based on the number of residual blocks"""
        if num_blocks <= 0:
            return []
        if mode not in {"linear", "uniform"}:
            raise ValueError("stochastic_depth_mode must be 'linear' or 'uniform'")

        if mode == "uniform":
            return [max_rate] * num_blocks

        linspace = torch.linspace(0.0, max_rate, steps=num_blocks)
        return [float(x) for x in linspace]

    def _assign_drop_path(self, block: nn.Module, drop_prob: float) -> None:
        """Attach DropPath module to a residual block"""
        apply = drop_prob > 0.0
        block.apply_stochastic_depth = apply
        block.drop_path = DropPath(drop_prob=drop_prob)
