from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CustomDataset
from opencompass.evaluator import MATHVerifyEvaluator
from opencompass.datasets import MATHEvaluator, math_postprocess_v2

math_reader_cfg = dict(input_columns=['problem'], output_column='answer')

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.',
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=8000),
)

'''
math_eval_cfg = dict(
    evaluator=dict(type=MATHVerifyEvaluator),
    #evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2)
)
'''

math_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'),
    pred_postprocessor=dict(type=math_postprocess_v2))

math_datasets = [
    dict(
        type=CustomDataset,
        abbr='math-500',
        path='opencompass/math',
        k=[16],
        n=16,
        file_name='test_prm800k_500.jsonl',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
