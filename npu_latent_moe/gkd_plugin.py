"""ms-swift external plugin: register Qwen3.5-LatentMoE for NPU GKD training.

Usage:
    swift rlhf --external_plugins npu_latent_moe/gkd_plugin.py \
               --model_type qwen3_5_latentmoe \
               --model npu_latent_moe/ckpt_latentmoe_40l ...

Why this plugin is needed
-------------------------
ms-swift's built-in loader would match the checkpoint's model_type
"qwen3_5_moe_text" to `Qwen3_5MoeLoader`, which hard-codes the vanilla
`Qwen3_5MoeForCausalLM` and bypasses the LatentMoE architecture entirely.

Registering a custom `qwen3_5_latentmoe` model_type forces the loader path
through `AutoModelForCausalLM.from_pretrained(trust_remote_code=True)`,
which picks up `auto_map` in config.json and loads the NPU-aware custom
class `v5molae_qwen35_latentmoe.Qwen3_5LatentMoeForCausalLM`.

NPU-specific notes
------------------
* The custom model class swaps `Qwen3_5MoeRMSNorm` for `Qwen3_5NpuRMSNorm`
  at __init__ when `config.use_npu_rmsnorm=True`. That key is set in the
  accompanying config.json.
* `use_grouped_expert_matmul=True` enables `torch_npu.npu_grouped_matmul`
  for the LatentMoE experts path.
* DeepSpeed Zero3 must be told that `Qwen3_5LatentMoeSparseMoeBlock` is a
  leaf module (the vanilla `Qwen3_5MoeSparseMoeBlock` does not exist after
  the in-place replacement). We match by *string* name because
  trust_remote_code loads the class under `transformers_modules.*`, which
  is a different Python object from a sys.path import.
"""
from swift.llm import ModelMeta, ModelGroup, Model, register_model
from swift.llm.model.register import ModelLoader
from swift.llm.template import TemplateType


class Qwen3_5LatentMoeLoader(ModelLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.leaf_modules = ['Qwen3_5LatentMoeSparseMoeBlock']


register_model(
    ModelMeta(
        'qwen3_5_latentmoe',
        [ModelGroup(
            [Model()],
            TemplateType.qwen3_5,
        )],
        Qwen3_5LatentMoeLoader,
        architectures=['Qwen3_5LatentMoeForCausalLM'],
        requires=['transformers>=5.2.0'],
    )
)
