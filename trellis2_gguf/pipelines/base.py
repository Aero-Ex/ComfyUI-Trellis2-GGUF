from typing import *
import torch
import torch.nn as nn
from .. import models


def _find_sdnq_model_dir(model_path: str, svd_rank: int = 32):
    """
    Given a standard model path (e.g. .../ckpts/ss_flow_img_dit_1_3B_64_bf16),
    locate the equivalent SDNQ directory (e.g. .../Trellis2/sdnq/ss_flow_img_dit_1_3B_64_uint4_svd32/).
    Returns the SDNQ directory path if found, else None.
    """
    import os
    try:
        import folder_paths
        models_base = folder_paths.models_dir
    except ImportError:
        models_base = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models")
    models_base = os.path.normpath(models_base)

    model_basename = os.path.basename(model_path)
    # e.g. ss_flow_img_dit_1_3B_64_bf16 → ss_flow_img_dit_1_3B_64_uint4_svd32
    sdnq_name = model_basename.replace("_bf16", f"_uint4_svd{svd_rank}")

    for case in ["Trellis2", "trellis2", "TRELLIS2"]:
        sdnq_dir = os.path.join(models_base, case, "sdnq")
        candidate = os.path.join(sdnq_dir, sdnq_name)
        # Flat layout: sdnq/{name}_quantization_config.json
        if os.path.isfile(candidate + "_quantization_config.json"):
            return candidate
        # Legacy subdir layout: sdnq/{name}/quantization_config.json
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "quantization_config.json")):
            return candidate
    return None


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        # for model in self.models.values():
            # model.eval()

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/{config_file}")

        if is_local:
            config_file = f"{path}/{config_file}"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, config_file)

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            _models[k] = None            
            # if hasattr(cls, 'model_names_to_load') and k not in cls.model_names_to_load:
                # continue
            # try:
                # _models[k] = models.from_pretrained(f"{path}/{v}")
            # except Exception as e:
                # _models[k] = models.from_pretrained(v)

        _models['shape_slat_encoder'] = None

        new_pipeline = cls(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        if hasattr(self, '_device'):
            return self._device
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            if model is not None:
                model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))