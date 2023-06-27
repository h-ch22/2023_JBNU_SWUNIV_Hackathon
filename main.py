import gc
import os

import numpy as np
import pandas as pd
import torch
import warnings
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from timm.optim import AdamP

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm

from models.BertDataset import BertDataset
from models.TestDataset import TestDataset


def clean_text(texts):
    corpus = []
    for i in range(0, len(texts)):
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"\n\]\[\>]', '',
                        texts[i])
        review = re.sub(r'\s+', ' ', review)
        review = re.sub(r'<[^>]+>', '', review)
        review = re.sub(r'\s+', ' ', review)
        review = re.sub(r"^\s+", '', review)
        review = re.sub(r'\s+$', '', review)
        review = re.sub(r'_', ' ', review)
        corpus.append(review)

    return corpus


def train(model, NUM_EPOCHS, train_dataloader, val_dataloader):
    loss_fn = torch.nn.CrossEntropyLoss()

    loss_values = []

    for epoch in range(NUM_EPOCHS):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, NUM_EPOCHS))

        stacked_val_labels = []
        targets_list = []

        # Training

        print('Training...')

        model.train()
        torch.set_grad_enabled(True)

        total_train_loss = 0

        for i, batch in enumerate(tqdm(train_dataloader)):
            train_status = 'Batch ' + str(i) + ' of ' + str(len(train_dataloader))

            print(train_status, end='\r')

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            attention_mask=b_input_mask)

            loss = loss_fn(outputs[0], b_labels)

            total_train_loss = total_train_loss + loss.item()

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        print('\nTrain loss:', total_train_loss)

        # Validation

        print('\n==========Validation==========')

        model.eval()

        torch.set_grad_enabled(False)

        total_val_loss = 0

        for j, batch in enumerate(tqdm(val_dataloader)):

            val_status = 'Batch ' + str(j) + ' of ' + str(len(val_dataloader))

            print(val_status, end='\r')

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            outputs = model(b_input_ids,
                            attention_mask=b_input_mask)

            preds = outputs[0]

            loss = loss_fn(preds, b_labels)
            total_val_loss = total_val_loss + loss.item()

            val_preds = preds.detach().cpu().numpy()

            targets_np = b_labels.to('cpu').numpy()

            targets_list.extend(targets_np)

            if j == 0:
                stacked_val_preds = val_preds

            else:
                stacked_val_preds = np.vstack((stacked_val_preds, val_preds))

        y_true = targets_list
        y_pred = np.argmax(stacked_val_preds, axis=1)

        val_acc = accuracy_score(y_true, y_pred)

        print('Val loss:', total_val_loss)
        print('Val acc: ', val_acc)

        torch.save(model, f'./output_roBerta/roBerta_epoch_{epoch}_acc{val_acc}_model.pt')

        gc.collect()


def convert_category_to_index(label):
    indexes = []

    CATEGORIES = {
        'Arts, Photography': 0, 'Biographies, Memoirs': 1, 'Calendars': 2,
        'Childrens Books': 3, 'Computers, Technology': 4, 'Cookbooks, Food, Wine': 5,
        'Crafts, Hobbies, Home': 6, 'Education, Teaching': 7, 'Engineering, Transportation': 8,
        'Health, Fitness, Dieting': 9, 'Humor, Entertainment': 10, 'Law': 11,
        'Literature, Fiction': 12, 'Medical Books': 13, 'Mystery, Thriller, Suspense': 14,
        'Parenting, Relationships': 15, 'Reference': 16, 'Religion, Spirituality': 17,
        'Science Fiction, Fantasy': 18, 'Science, Math': 19, 'Self Help': 20,
        'Sports, Outdoors': 21, 'Test Preparation': 22, 'Travel': 23
    }

    for i in range(0, len(label)):
        indexes.append(CATEGORIES.get(label[i]))

    return indexes


def convert_index_to_category(index):
    CATEGORIES = {
        'Arts, Photography': 0, 'Biographies, Memoirs': 1, 'Calendars': 2,
        'Childrens Books': 3, 'Computers, Technology': 4, 'Cookbooks, Food, Wine': 5,
        'Crafts, Hobbies, Home': 6, 'Education, Teaching': 7, 'Engineering, Transportation': 8,
        'Health, Fitness, Dieting': 9, 'Humor, Entertainment': 10, 'Law': 11,
        'Literature, Fiction': 12, 'Medical Books': 13, 'Mystery, Thriller, Suspense': 14,
        'Parenting, Relationships': 15, 'Reference': 16, 'Religion, Spirituality': 17,
        'Science Fiction, Fantasy': 18, 'Science, Math': 19, 'Self Help': 20,
        'Sports, Outdoors': 21, 'Test Preparation': 22, 'Travel': 23
    }

    return list(CATEGORIES)[index]


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    torch.manual_seed(1016)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_df = pd.read_csv('./data/train_data.csv', skiprows=0)
    test_df = pd.read_csv('./data/test_data.csv', skiprows=0)
    submit_df = pd.read_csv('./data/sample_submission.csv')

    train_df['Title'] = clean_text(train_df['Title'])
    test_df['Title'] = clean_text(test_df['Title'])

    train_df['label'] = convert_category_to_index(train_df['label'])

    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    train_dataset, val_dataset = train_test_split(train_df, test_size=0.1)
    train_dataset = train_dataset.reset_index(drop=True)
    val_dataset = val_dataset.reset_index(drop=True)

    NUM_EPOCHS = 10
    L_RATE = 1e-5
    MAX_LEN = 512

    TRAIN_BATCH_SIZE = 4
    TEST_BATCH_SIZE = 1

    NUM_CORES = os.cpu_count()
    print("NUM_CORES of CPU : ", NUM_CORES)

    train_data = BertDataset(train_dataset)
    val_data = BertDataset(val_dataset)
    test_data = TestDataset(test_df)

    train_dataloader = DataLoader(train_data,
                                  batch_size=TRAIN_BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_CORES)

    val_dataloader = DataLoader(val_data,
                                batch_size=TRAIN_BATCH_SIZE,
                                shuffle=True,
                                num_workers=NUM_CORES)

    test_dataloader = DataLoader(test_data,
                                 batch_size=TEST_BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=NUM_CORES)

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-large",
        num_labels=24,
    )
    model.to(device)

    optimizer = AdamP(model.parameters(), lr=2e-5)

    train(model, NUM_EPOCHS, train_dataloader, val_dataloader)

    # model = torch.load(r'./output/epoch_3_acc0.7053428446644344_model.pt')
    model = torch.load(r'./output/epoch_9_acc0.6966079487552773_model.pt')
    model.to(device)

    results = []

    for j, batch in enumerate(tqdm(test_dataloader)):

        inference_status = 'Batch ' + str(j + 1) + ' of ' + str(len(test_dataloader))

        print(inference_status, end='\r')

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)

        outputs = model(b_input_ids,
                        attention_mask=b_input_mask)

        preds = outputs[0]
        preds = preds.detach().cpu().numpy()

        if j == 0:
            stacked_preds = preds

        else:
            stacked_preds = np.vstack((stacked_preds, preds))

    preds = np.argmax(stacked_preds, axis=1)
    preds_categorical = []

    for pred in preds:
        preds_categorical.append(convert_index_to_category(pred))

    submit_df['label'] = preds_categorical
    submit_df.to_csv(r'C:\Users\USER\Desktop\2023\JBNU_SWUNIV_HACKATHON\Classification\output\submit_E10l.csv', index=False)
