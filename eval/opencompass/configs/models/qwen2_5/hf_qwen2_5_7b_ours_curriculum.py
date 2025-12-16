# 在 configs/models/vllm/ 下创建配置文件
from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='vllm_qwen2_5_7b_ours_curriculum',
        path='/root/bayes-tmp/bzx_data/code/ours_low_en_kl_v2_curriculum/hf_model/Qwen_ours_low_en_kl_v2_curriculum_stage2',
        model_kwargs=dict(
            tensor_parallel_size=1,
            max_model_len=None
        ),
        max_out_len=8000,
        max_seq_len=None,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        generation_kwargs=dict(
            temperature=1.0,
            top_p=0.9,
            top_k=50
        )
    )
]