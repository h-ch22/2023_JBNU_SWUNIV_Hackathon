import torch

from torch.utils.data import Dataset
from transformers import BertTokenizer, BertTokenizerFast, RobertaTokenizer


class TestDataset(Dataset):

    def __init__(self, df):
        self.df_data = df
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    def __getitem__(self, index):
        sentence1 = self.df_data.loc[index, 'Title']

        encoded_dict = self.tokenizer.encode_plus(
            sentence1,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt',
        )

        padded_token_list = encoded_dict['input_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]

        sample = (padded_token_list, att_mask)

        return sample

    def __len__(self):
        return len(self.df_data)
