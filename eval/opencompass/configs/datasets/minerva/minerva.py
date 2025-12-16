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

minerva_reader_cfg = dict(input_columns=['question'], output_column='answer')

minerva_infer_cfg = dict(
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


minerva_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2)
)

minerva_datasets = [
    dict(
        type=CustomDataset,
        abbr='minerva',
        path='opencompass/minerva',
        k=[16],
        n=16,
        reader_cfg=minerva_reader_cfg,
        infer_cfg=minerva_infer_cfg,
        eval_cfg=minerva_eval_cfg,
    )
]
