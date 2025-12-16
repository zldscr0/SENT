from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CustomDataset
from opencompass.datasets import generic_llmjudge_postprocess
from opencompass.evaluator import (
    CascadeEvaluator,
    GenericLLMEvaluator,
    MATHVerifyEvaluator
)
from opencompass.datasets import MATHEvaluator, math_postprocess_v2

olympiadbench_reader_cfg = dict(input_columns=['question'], output_column='answer')

olympiadbench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                #dict(role='HUMAN', prompt='{question}\nRemember to put your final answer within \\boxed{}.'),
                dict(role='HUMAN', prompt='{question}\nPlease reason step by step, and put your final answer within \\boxed{}.'),
                
            ],
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=8000)
)


olympiadbench_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2)
)

olympiadbench_datasets = [
    dict(
        type=CustomDataset,
        abbr='olympiadbench',
        path='opencompass/olympiadbench',
        k=[16],
        n=16,
        reader_cfg=olympiadbench_reader_cfg,
        infer_cfg=olympiadbench_infer_cfg,
        eval_cfg=olympiadbench_eval_cfg,
    )
]
