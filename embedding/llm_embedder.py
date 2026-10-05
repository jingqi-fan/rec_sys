from typing import List

from transformers import AutoModel, AutoTokenizer

import torch
import numpy as np


class LLMEmbedder:
    """
    This is a wrapper class to generate text embeddings using LLM. We
    leverage the huggingface's transformers library to load the model and
    to the inference. The transformer library provide rich support for the
    open source LLM, with the model class implementation and other utilities
    """

    def __init__(self, model_name: str):
        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def embed(self, texts: List[List[str]]) -> List[np.ndarray]:
        """
        Generate text embeddings using the LLM.

        Args:
            texts: A list of list of text strings (sentences) to embed.

        Returns:
            A list of numpy arrays, which is reduced from the last hidden state of the model.
        """
        # since we are using the batch inference, the inputs would be padded to be the same length,
        # which would be the max length of the sentences in the batch
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)  # B x max_len in batch

        # in the input, besides `input_ids` which is the tokenized input, there would be `attention_mask`
        # which is the mask for the padding tokens
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)  # B x max_len in batch x hidden_size
        # we use the last hidden state as the semantic embedding for the sentence
        last_hidden_state = outputs.hidden_states[-1]  # B x max_len in batch x hidden_size
        # we use the attention mask to mask the padding tokens
        masked_hidden_state = last_hidden_state * inputs["attention_mask"].unsqueeze(-1)
        # due to masking, we could not directly use the mean to reduce
        embeddings = masked_hidden_state.sum(dim=1) / inputs["attention_mask"].sum(dim=1).unsqueeze(-1)  # B x hidden_size
        return embeddings.cpu().numpy()
