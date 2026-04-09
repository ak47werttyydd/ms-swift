"""ms-swift external plugin: register Qwen3.5-LatentMoE-MLA for GKD training.

Usage:
    swift rlhf --external_plugins gkd_plugin.py --model_type qwen3_5_latentmoe_mla ...
"""
from swift.model import ModelMeta, ModelGroup, Model, register_model
from swift.template import TemplateType

register_model(
    ModelMeta(
        # This must match config.json "model_type"
        'qwen3_5_latentmoe_mla',
        [ModelGroup(
            [Model()],  # path filled at runtime via --model
            TemplateType.qwen3_5,
        )],
        architectures=['Qwen3_5LatentMoeMLAForConditionalGeneration'],
        requires=['transformers>=4.57'],
        tags=['vision', 'video'],
    )
)
