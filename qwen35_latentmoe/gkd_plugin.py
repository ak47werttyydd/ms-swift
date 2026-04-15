"""ms-swift external plugin: register Qwen3.5-LatentMoE (non-MLA) for GKD training.

Usage:
    swift rlhf --external_plugins qwen35_latentmoe/gkd_plugin.py \
               --model_type qwen3_5_latentmoe \
               --model qwen35_latentmoe ...

This registers a custom swift model_type so that ms-swift's ModelLoader
uses AutoModelForCausalLM.from_pretrained(trust_remote_code=True), which
picks up the auto_map in config.json and loads the custom
Qwen3_5LatentMoeForConditionalGeneration class (with LatentMoE replacement
and the _repack_legacy_state_dict pre-hook).

Without this plugin, ms-swift would match config.json model_type "qwen3_5_moe"
to its built-in Qwen3_5MoeLoader, which hard-codes the vanilla
Qwen3_5MoeForConditionalGeneration — bypassing the LatentMoE architecture
entirely.

The custom Loader also sets leaf_modules to Qwen3_5LatentMoeSparseMoeBlock
so that DeepSpeed Zero3 recognises the replaced MoE blocks instead of
looking for the vanilla Qwen3_5MoeSparseMoeBlock (which no longer exists).
"""
from swift.model import ModelMeta, ModelGroup, Model, ModelLoader, register_model
from swift.template import TemplateType


class Qwen3_5LatentMoeLoader(ModelLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use string matching instead of class reference, because
        # trust_remote_code=True loads the class under transformers_modules.*
        # which is a different Python class object from a sys.path import.
        # DeepSpeed's set_z3_leaf_modules supports string matching (by class name).
        self.leaf_modules = ['Qwen3_5LatentMoeSparseMoeBlock']


register_model(
    ModelMeta(
        'qwen3_5_latentmoe',
        [ModelGroup(
            [Model()],
            TemplateType.qwen3_5,
        )],
        Qwen3_5LatentMoeLoader,
        architectures=['Qwen3_5LatentMoeForConditionalGeneration'],
        requires=['transformers>=5.2.0'],
    )
)
