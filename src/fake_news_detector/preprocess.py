"""
Preprocessing
"""

import torch
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


class preprocess:
    def __init__(self, df: pd.DataFrame,
                 max_length: int,
                 padding: str,
                 test_size: float,
                 text_column: str,
                 pretrained_model: str = "prajjwal1/bert-tiny") -> None:
        self.max_length = max_length
        self.padding = padding
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower_case=True)
        self.text = df[text_column]
        self.test_size = test_size

    @staticmethod
    def get_mask(x: int) -> list:
        masked = list(map(lambda x: int(x > 0), x))
        return masked

    def tokenize(self) -> list:
        input_ids = self.text.map(lambda x: self.tokenizer.encode(x, truncation=True, add_special_tokens=True, padding=self.padding, max_length=self.max_length)).tolist()
        return input_ids

    def attention_masks(self, input_ids: list) -> list:
        attention_masks = list(map(lambda x: self.get_mask(x), input_ids))
        return attention_masks

    def data_split(self, input_ids: list, encoded_label: np.array) -> Tuple[list, list, np.array, np.array]:
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids,
                                                                                            encoded_label,
                                                                                            random_state=1,
                                                                                            test_size=self.test_size)
        return train_inputs, validation_inputs, train_labels, validation_labels

    def mask_split(self, attention_masks: list, encoded_label: np.array) -> Tuple[list, list]:
        train_masks, validation_masks, _, _ = train_test_split(attention_masks,
                                                               encoded_label,
                                                               random_state=1,
                                                               test_size=self.test_size)
        return train_masks, validation_masks

    def convert_to_tensor(self,
                          train_inputs: list,
                          validation_inputs: list,
                          train_labels: np.array,
                          validation_labels: np.array,
                          train_masks: list,
                          validation_masks: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)

        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)

        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)

        return train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks
