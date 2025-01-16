from deepeval import assert_test, models, evaluate
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    ToxicityMetric,
    HallucinationMetric,
    JsonCorrectnessMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.vulnerability import Misinformation
from deepeval.vulnerability.misinformation import MisinformationType
import pandas as pd

dataset = EvaluationDataset()
dataset.add_test_cases_from_csv_file(
    "test_rag_kpis.csv",
    input_col_name="input",
    actual_output_col_name="actual_output",
    retrieval_context_col_name="retrieval_context",
    retrieval_context_col_delimiter=";",
)


def test_answer_relevancy():
    answer_relevancy_metric = AnswerRelevancyMetric(model="gpt-4o-mini")
    bias_metric = BiasMetric(model="gpt-4o-mini")
    toxicity_metric = ToxicityMetric(model="gpt-4o-mini")
    hallucination_metric = HallucinationMetric(model="gpt-4o-mini")
    contextual_relevancy_metric = ContextualRelevancyMetric(model="gpt-4o-mini")
    faithfulness_metric = FaithfulnessMetric(model="gpt-4o-mini")
    evaluate(
        dataset,
        [
            answer_relevancy_metric,
            bias_metric,
            toxicity_metric,
            hallucination_metric,
            contextual_relevancy_metric,
            faithfulness_metric,
        ],
    )
