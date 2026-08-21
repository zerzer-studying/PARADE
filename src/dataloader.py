
import torch
from torch.utils.data import Dataset
from typing import List
import numpy as np
import random


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ======================================================================
# Dataset / DataLoader
# ======================================================================

class QADataset(Dataset):
    IGNORED = -100

    def __init__(self, qa_pairs: list, tokenizer, max_length: int = 512,
                 use_augmented: bool = True, use_tool_call: bool = False,
                 answer_value_only: bool = False,
                 dynamic_padding: bool = False):
        """
        Args:
            qa_pairs: List of QA pairs
            tokenizer: Tokenizer
            max_length: Max sequence length
            use_augmented: If True, expects 'answer' (reasoning) and 'answer_value' (option number).
                          If False, expects only 'answer' (option number directly).
            use_tool_call: If True, uses Qwen native tool-call format.
                          Expects 'augmented_data' field in qa_pairs.
            answer_value_only: If True, trains the assistant target as only
                          the option number from 'answer_value'.
            dynamic_padding: Keep variable-length tensors and pad per batch.
        """
        input_ids_items = []
        labels_items = []
        attention_mask_items = []
        pad_id = (tokenizer.pad_token_id
                  if tokenizer.pad_token_id is not None
                  else tokenizer.eos_token_id)

        has_chat_template = (hasattr(tokenizer, 'apply_chat_template')
                             and tokenizer.chat_template)

        for qa in qa_pairs:
            if answer_value_only:
                user_content = (f"Question: {qa['question']}\n\n"
                               f"Write ONLY your chosen option number.\n\n"
                               f"Answer:")
                answer_value = qa.get('answer_value', qa.get('answer', ""))
                assistant_content = str(answer_value)
            elif use_tool_call and 'augmented_data' in qa:
                user_content = (f"Question: {qa['question']}\n\n"
                               f"First, provide your reasoning and explanation. "
                               f"Then, on a new line, write ONLY your chosen option number inside <answer></answer> tags.\n\n"
                               f"Required format:\n"
                               f"[Your reasoning and explanation]\n"
                               f"<answer>[option number]</answer>\n\n"
                               f"Your response:")
                answer_value = qa.get('answer_value', qa['answer'])
                assistant_content = f"{qa['augmented_data']}\n<answer>{answer_value}</answer>"
            elif use_augmented:
                user_content = (f"Question: {qa['question']}\n\n"
                               f"First, provide your reasoning and explanation. "
                               f"Then, on a new line, write ONLY your chosen option number inside <answer></answer> tags.\n\n"
                               f"Required format:\n"
                               f"[Your reasoning and explanation]\n"
                               f"<answer>[option number]</answer>\n\n"
                               f"Your response:")
                answer_value = qa.get('answer_value', qa['answer'])
                assistant_content = f"{qa['answer']}\n<answer>{answer_value}</answer>"
            else:
                user_content = f"Question: {qa['question']}"
                answer_value = qa.get('answer_value')
                if answer_value is not None:
                    assistant_content = f"{answer_value}. {qa['answer']}"
                else:
                    assistant_content = str(qa['answer'])

            if has_chat_template:
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                full = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                    enable_thinking=False)
                # Build prefix (everything up to assistant's content) for label masking
                prefix_messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": user_content},
                ]
                prefix = tokenizer.apply_chat_template(
                    prefix_messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
            else:
                if answer_value_only or use_tool_call or use_augmented:
                    prefix = user_content + " "
                    full = prefix + assistant_content
                else:
                    prefix = f"Question: {qa['question']}\n\nAnswer: "
                    full = f"{prefix}{qa['answer']}"

            prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
            enc = tokenizer(full, truncation=True, max_length=max_length,
                            add_special_tokens=False)

            ids = enc['input_ids']
            labels = list(ids)
            for i in range(min(len(prefix_ids), len(labels))):
                labels[i] = self.IGNORED

            if dynamic_padding:
                attn = [1] * len(ids)
            else:
                pad_n = max_length - len(ids)
                attn = [1] * len(ids) + [0] * pad_n
                ids = ids + [pad_id] * pad_n
                labels = labels + [self.IGNORED] * pad_n

            input_ids_items.append(ids)
            labels_items.append(labels)
            attention_mask_items.append(attn)

        if dynamic_padding:
            self.input_ids = [
                torch.tensor(item, dtype=torch.long) for item in input_ids_items
            ]
            self.labels = [
                torch.tensor(item, dtype=torch.long) for item in labels_items
            ]
            self.attention_mask = [
                torch.tensor(item, dtype=torch.long)
                for item in attention_mask_items
            ]
        else:
            self.input_ids = torch.tensor(input_ids_items, dtype=torch.long)
            self.labels = torch.tensor(labels_items, dtype=torch.long)
            self.attention_mask = torch.tensor(
                attention_mask_items, dtype=torch.long
            )

    def __len__(self):
        if isinstance(self.input_ids, list):
            return len(self.input_ids)
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx],
            'attention_mask': self.attention_mask[idx],
        }
