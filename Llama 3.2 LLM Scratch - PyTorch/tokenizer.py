from typing import Union, List, Optional
import sentencepiece as spm
import torch

class NanoLlamaTokenizer:
    def __init__(self, model_path: str = "nanollama_tokenizer.model",
                 bos_id: int = 2, eos_id: int = 3, pad_id: int = 4):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(model_path)
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

    def encode(self, text: Union[str, List[str]], add_bos: bool = True, add_eos: bool = True,
               max_length: Optional[int] = None, pad: bool = True, return_tensor: bool = False
               ) -> Union[List[int], List[List[int]], torch.Tensor]:
        def _encode_single(s: str) -> List[int]:
            ids = self.tokenizer.encode(s)
            if add_bos:
                ids = [self.bos_id] + ids
            if add_eos:
                ids = ids + [self.eos_id]
            if max_length:
                ids = ids[:max_length]
            if pad and max_length:
                ids += [self.pad_id] * (max_length - len(ids))
            return ids

        if isinstance(text, str):
            encoded = _encode_single(text)
        elif isinstance(text, list):
            encoded = [_encode_single(s) for s in text]
        else:
            raise ValueError("Input must be a string or a list of strings.")

        if return_tensor:
            if isinstance(encoded[0], list):
                return torch.tensor(encoded)
            else:
                return torch.tensor([encoded])
        return encoded

    def decode(self, ids: Union[List[int], List[List[int]], torch.Tensor]) -> Union[str, List[str]]:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids[0], list):
            return [self.tokenizer.decode(id_seq) for id_seq in ids]
        else:
            return self.tokenizer.decode(ids)

    def __call__(self, text: Union[str, List[str]], **kwargs):
        return self.encode(text, **kwargs)
