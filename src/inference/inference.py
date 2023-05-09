"""
Inference class
"""

import yaml
import torch
import numpy as np
from typing import List
from transformers import AutoTokenizer
from mlserver import MLModel
from mlserver.codecs import decode_args
from mlserver.utils import get_model_uri

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)


class FakeNewsModel(MLModel):
    async def load(self) -> bool:
        model_uri = await get_model_uri(self._settings)
        self.model = torch.load(model_uri, map_location=torch.device('cpu'))
        self.model.eval()
        self.model.to('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(config['params']['PRETRAINED_MODEL'], do_lower_case=True)

        return True

    @decode_args
    async def predict(self,
                      text: List[str]) -> List[str]:

        input_ids = self.tokenizer(text, truncation=True, add_special_tokens=True, padding='max_length', max_length=256)
        input_id = torch.tensor(input_ids['input_ids'])
        attention_mask = torch.tensor(input_ids['attention_mask'])

        outputs = self.model(input_id, attention_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()

        result = ["FAKE" if pred_flat[0] == 1 else "REAL"]

        return result
