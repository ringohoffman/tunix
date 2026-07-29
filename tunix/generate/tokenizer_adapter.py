# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapt tokenizers to a common interface."""

from __future__ import annotations

from collections.abc import Mapping
import functools
from typing import Any, Generic, Literal, TypeVar, Unpack, overload

from etils import epath
import numpy as np
import sentencepiece as spm
import transformers
from tunix.utils import token_sanitization

TokenizerT = TypeVar(
    'TokenizerT',
    spm.SentencePieceProcessor,
    transformers.PreTrainedTokenizerBase,
)


class TokenizerAdapter(Generic[TokenizerT]):
  """Wrapper for different tokenizers used in sampler."""

  def __init__(self, tokenizer: TokenizerT):
    self._tokenizer = tokenizer

  def encode(self, text: str, **kwargs) -> list[int]:
    if isinstance(self._tokenizer, spm.SentencePieceProcessor):
      return self._tokenizer.EncodeAsIds(text, **kwargs)
    return self._tokenizer.encode(text, **kwargs)

  def decode(self, ids: list[int], **kwargs) -> str:
    if isinstance(self._tokenizer, spm.SentencePieceProcessor):
      return self._tokenizer.DecodeIds(ids, **kwargs)
    return self._tokenizer.decode(ids, **kwargs)

  def bos_id(self) -> int:
    if isinstance(self._tokenizer, transformers.PreTrainedTokenizerBase):
      return self._tokenizer.bos_token_id
    return self._tokenizer.bos_id()

  def eos_id(self) -> int:
    if isinstance(self._tokenizer, transformers.PreTrainedTokenizerBase):
      return self._tokenizer.eos_token_id
    return self._tokenizer.eos_id()

  def pad_id(self) -> int:
    """Returns the pad token id."""
    if isinstance(self._tokenizer, spm.SentencePieceProcessor):
      ret_id = self._tokenizer.pad_id()
      if ret_id == -1:
        raise ValueError('SentencePiece tokenizer has an undefined pad_id.')
      return ret_id
    if isinstance(self._tokenizer, transformers.PreTrainedTokenizerBase):
      if self._tokenizer.pad_token_id is None:
        self._tokenizer.pad_token = self._tokenizer.eos_token
      return self._tokenizer.pad_token_id
    return self._tokenizer.pad_id()

  def dedup_bos_ids(self, ids: list[int]) -> list[int]:
    """Deduplicates the bos_id at the beginning of the list."""
    i = 0
    while i < len(ids) - 1 and ids[i] == ids[i + 1] == self.bos_id():
      i += 1
    return ids[i:]

  def _missing_methods(self) -> list[str]:
    """Checks if the tokenizer has any missing methods."""
    required_methods = ['encode', 'decode', 'bos_id', 'eos_id', 'pad_id']
    missing_methods = []
    for method in required_methods:
      if not hasattr(self._tokenizer, method):
        missing_methods.append(method)
    return missing_methods

  @property
  def tokenizer(self) -> Any:
    return self._tokenizer

  @property
  def vocab_size(self) -> int:
    if isinstance(self._tokenizer, spm.SentencePieceProcessor):
      return self._tokenizer.GetPieceSize()
    if isinstance(self._tokenizer, transformers.PreTrainedTokenizerBase):
      vs = self._tokenizer.vocab_size
      return vs() if callable(vs) else vs
    if type(self._tokenizer).__name__ == 'MockVocab':
      return self._tokenizer.GetPieceSize()
    return len(self._tokenizer)

  @functools.cached_property
  def token_id_to_str(self) -> Mapping[int, str]:
    """Returns a dictionary mapping token ID -> decoded string efficiently."""
    if isinstance(self._tokenizer, transformers.PreTrainedTokenizerBase):
      vocab = self._tokenizer.get_vocab()
      res: dict[int, str] = {}
      for piece, tid in vocab.items():
        if piece.startswith('<0x') and piece.endswith('>') and len(piece) == 6:
          try:
            res[tid] = chr(int(piece[3:5], 16))
          except ValueError:
            res[tid] = piece
        else:
          res[tid] = piece.replace('\u2581', ' ').replace('Ġ', ' ')
      return res

    if (
        isinstance(self._tokenizer, spm.SentencePieceProcessor)
        or type(self._tokenizer).__name__ == 'MockVocab'
    ):
      res: dict[int, str] = {}
      for tid in range(self.vocab_size):
        try:
          piece = self._tokenizer.IdToPiece(tid)
        except Exception:
          piece = self.decode([tid])
        res[tid] = piece.replace('\u2581', ' ')
      return res

    token_map: dict[int, str] = {}
    for tid in range(self.vocab_size):
      try:
        token_map[tid] = self.decode([tid])
      except Exception:
        token_map[tid] = ''
    return token_map

  def __getattr__(self, name: str) -> Any:
    """Delegate unknown attributes to the wrapped tokenizer.

    This keeps the adapter compatible with callers that expect Hugging Face
    tokenizer attributes such as bos_token/eos_token while still using the
    normalized adapter interface for encode/decode/id helpers.
    """
    return getattr(self._tokenizer, name)

  @overload
  def apply_chat_template(
      self,
      conversation: transformers.Conversation,
      *,
      tokenize: Literal[False],
      return_dict: bool,
      **kwargs: Unpack[transformers.ChatTemplateKwargs],
  ) -> str:
    ...

  @overload
  def apply_chat_template(
      self,
      conversation: transformers.Conversation,
      *,
      tokenize: Literal[True] = ...,
      return_dict: Literal[True],
      **kwargs: Unpack[transformers.ChatTemplateKwargs],
  ) -> transformers.BatchEncoding[list[int]]:
    ...

  @overload
  def apply_chat_template(
      self,
      conversation: transformers.Conversation,
      *,
      tokenize: Literal[True] = ...,
      return_dict: Literal[False] = ...,
      **kwargs: Unpack[transformers.ChatTemplateKwargs],
  ) -> list[int]:
    ...

  @overload
  def apply_chat_template(
      self,
      conversation: transformers.Conversation,
      *,
      tokenize: bool = ...,
      **kwargs: Unpack[transformers.ChatTemplateKwargs],
  ) -> (
      str
      | list[int]
      | list[str]
      | list[list[int]]
      | transformers.BatchEncoding[list[int]]
  ):
    ...

  def apply_chat_template(
      self,
      conversation: transformers.Conversation,
      *,
      return_dict: bool = False,
      tokenize: bool = True,
      **kwargs: Unpack[transformers.ChatTemplateKwargs],
  ) -> (
      str
      | list[int]
      | list[str]
      | list[list[int]]
      | transformers.BatchEncoding[list[int]]
  ):
    """Applies a chat template to format a list of messages.

    Primarily for HuggingFace tokenizers, this formats conversation history
    into a single string or token sequence.

    Args:
      messages: Conversation turns, each with 'role' and 'content'.
      add_generation_prompt: Whether to append a generation prompt.
      tokenize: If True, returns token IDs; otherwise, returns a string.
      **kwargs: Additional args for the underlying `apply_chat_template`.

    Returns:
      The formatted chat as a string or list of token IDs.

    Raises:
      NotImplementedError: If chat templating is not supported by the tokenizer.
    """
    conversation = [
        {
            **m,
            'content': token_sanitization.sanitize_control_tokens(m['content']),
        }
        for m in conversation
    ]
    if isinstance(self._tokenizer, transformers.PreTrainedTokenizerBase):
      return self._tokenizer.apply_chat_template(
          conversation,
          return_dict=return_dict,
          tokenize=tokenize,
          **kwargs,
      )
    # Implements the Gemma chat template format as a fallback for SentencePiece / basic tokenizers.
    return self._apply_gemma_chat_template(
        conversation, kwargs['add_generation_prompt'], tokenize
    )

  def _apply_gemma_chat_template(
      self,
      messages: list[dict[str, str]],
      add_generation_prompt: bool,
      tokenize: bool,
  ) -> str | list[int]:
    """Applies the Gemma chat template format."""
    chat_str = ''
    for message in messages:
      role = message.get('role')
      content = message.get('content')
      if role in ('user', 'model'):
        chat_str += f'<start_of_turn>{role}\n{content}<end_of_turn>\n'

    if add_generation_prompt:
      chat_str += '<start_of_turn>model\n'

    if tokenize:
      return self.encode(chat_str)
    return chat_str


class Tokenizer(TokenizerAdapter):
  """Tokenizing and encoding/decoding text using TokenizerAdapter."""

  def __init__(
      self,
      tokenizer_type: str = 'sentencepiece',
      tokenizer_path: str = 'gs://gemma-data/tokenizers/tokenizer_gemma2.model',
      add_bos: bool | None = True,
      add_eos: bool | None = True,
      hf_access_token: str | None = None,
  ):

    self.tokenizer_type = tokenizer_type
    if tokenizer_type == 'huggingface':
      import transformers  # pylint: disable=g-import-not-at-top

      tokenizer = transformers.AutoTokenizer.from_pretrained(
          pretrained_model_name_or_path=tokenizer_path,
          add_bos_token=add_bos,
          add_eos_token=add_eos,
          token=hf_access_token,
          extra_special_tokens={},
      )
    elif tokenizer_type == 'sentencepiece':
      model_proto = epath.Path(tokenizer_path).read_bytes()
      tokenizer = spm.SentencePieceProcessor()
      tokenizer.LoadFromSerializedProto(model_proto)
      options = []
      if add_bos:
        options.append('bos')
      if add_eos:
        options.append('eos')

      extra_options_str = ':'.join(options)
      if extra_options_str:
        tokenizer.SetEncodeExtraOptions(extra_options_str)
    else:
      raise ValueError(f'Unsupported tokenizer_type: {tokenizer_type}')
    super().__init__(tokenizer)

  def tokenize(
      self,
      example: str,
      prefix: str = '',
      suffix: str = '',
      add_eos: bool = True,
  ) -> np.ndarray:
    """The tokenization function.

    Args:
      example: Input string to tokenize.
      prefix:  Prefix to add to the input string.
      suffix:  Suffix to add to the input string.
      add_eos: If True, add an "end of sentence" token at the end of the output
        sequence.

    Returns:
      Tokens corresponding to the input string.
    """
    example = token_sanitization.sanitize_control_tokens(example)
    int_list = []
    if self.bos_id():
      int_list.append(self.bos_id())
    if self.tokenizer_type == 'huggingface':
      int_list.extend(
          self.encode(prefix + example + suffix, add_special_tokens=False)
      )
    else:
      # sentencepiece
      int_list.extend(self.tokenizer.EncodeAsIds(prefix + example + suffix))
    if add_eos:
      int_list.append(self.eos_id())
    return np.array(int_list, dtype=np.int32)
