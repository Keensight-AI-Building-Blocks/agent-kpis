import pandas as pd
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()


def create_test_dataset(data: pd.DataFrame):

    for i in range(len(data)):
        test_case = LLMTestCase(
            input=data["comments"][i], actual_output=data["response"][i]
        )
        dataset.add_test_case(test_case)

    return dataset


data = pd.read_csv("init_test_data.csv")
dataset = create_test_dataset(data)
