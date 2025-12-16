# examples/eval_aime2024_local.py

#from opencompass.models import HuggingFace
#from opencompass.datasets import AIME2024Dataset
#from opencompass.partitioners import NaivePartitioner
#from opencompass.runners.local import LocalRunner
#from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
#from opencompass.openicl.icl_evaluator import AccuracyEvaluator


from opencompass.configs.datasets.aime2024.aime2024_llmverify_repeat8_gen_e8fcee import aime2024_datasets

datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')),
    [],
)
# ===== 模型配置 =====
models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='deepseek-r1-qwen-1.5b',
        path='/data1/wym_data/llm/DeepSeek-R1-Distill-Qwen-1.5B',
        tokenizer_path='/data1/wym_data/llm/DeepSeek-R1-Distill-Qwen-1.5B',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_out_len=512,
        max_seq_len=2048,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        gen_kwargs=dict(
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            max_new_tokens=512
        )
    )
]

# ===== 数据集配置 =====
datasets = [
    dict(
        type=AIME2024Dataset,
        abbr='aime2024',
        path='aime_2024_problems.parquet',  # 使用你自己的路径
        reader_cfg=dict(
            input_columns=['question'],
            output_column='answer',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='plain',  # 默认不使用 few-shot
                template='Question: {{ question }}\nAnswer:',
                dict_mode='format'
            ),
            retriever=dict(type='ZeroRetriever'),
            inferencer=dict(type='GenInferencer'),
        ),
        eval_cfg=dict(
            evaluator=dict(type=AccuracyEvaluator),
            pred_role='BOT'
        )
    )
]

# ===== 推理配置 =====
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLInferTask)),
)

# ===== 评估配置 =====
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLEvalTask)),
)

# 多运行结果平均配置
summary_groups = [
    {
        'name': 'AIME2024-Aveage8',
        'subsets':[[f'aime2024-run{idx}', 'accuracy'] for idx in range(8)]
    },
    # 其他数据集平均配置...
]

summarizer = dict(
    dataset_abbrs=[
        ['AIME2024-Aveage8', 'naive_average'],
        # 其他数据集指标...
    ],
    summary_groups=summary_groups
)


# ===== 输出目录 =====
work_dir = 'outputs/aime2024_local'
