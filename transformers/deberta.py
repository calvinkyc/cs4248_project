import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from sklearn import preprocessing, model_selection
from sklearn.metrics import f1_score, confusion_matrix

from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer
import transformers

NUM_EPOCHS = 20
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 256
MODEL_PATH = "model.bin"
TEST_SIZE = 0.2


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = v.to(device)
        return data
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


class SentenceTagDataset(Dataset):
    def __init__(self, tokenizer, sentences, aspect_tags, max_length=128):
        self.sentences = sentences
        self.aspect_tags = aspect_tags
        self.max_length = max_length
        self.items_to_replace = {101, 102}

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]  # Get a sentence
        aspect_tags = self.aspect_tags[idx]  # Get the corresponding aspect tags

        sentence_encoding = self.tokenizer(
            sentence,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # print(len(sentence), sentence)
        # print(len(aspect_tags), aspect_tags)
        # print(self.tokenizer.convert_ids_to_tokens(sentence_encoding["input_ids"][0]))
        # print(torch.LongTensor(
        #     [[t not in self.tokenizer.all_special_ids for t in sentence_encoding["input_ids"][0]]])
        #     .nonzero(as_tuple=True))
        # aspect_tags_encoding = torch.zeros_like(sentence_encoding["input_ids"])
        # aspect_tags_encoding[torch.LongTensor(
        #     [[t not in self.tokenizer.all_special_ids for t in sentence_encoding["input_ids"][0]]])
        #     .nonzero(as_tuple=True)] = torch.LongTensor([aspect_tags])
        word_ids = sentence_encoding.word_ids(batch_index=0)
        aspect_tags_encoding = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                aspect_tags_encoding.append(-100)
            else:
                aspect_tags_encoding.append(aspect_tags[word_idx])
        aspect_tags_encoding = torch.LongTensor(aspect_tags_encoding)
        # print(aspect_tags_encoding)
        # exit()

        return {
            "input_ids": sentence_encoding["input_ids"][0],
            "attention_mask": sentence_encoding["attention_mask"][0],
            "token_type_ids": sentence_encoding["token_type_ids"][0],
            "aspect_tags": aspect_tags_encoding
        }


def test_dataset(train_sentences, train_aspect_tags, encoder, device):
    train_dataset = SentenceTagDataset(sentences=train_sentences,
                                       aspect_tags=train_aspect_tags)

    train_data_loader = DeviceDataLoader(torch.utils.data.DataLoader(
        train_dataset, batch_size=32), device)

    print(train_dataset[0])

    data = train_dataset[0]
    input_ids = data['input_ids']
    attention_mask = np.logical_not(data['attention_mask'])
    aspect_tags = data['aspect_tags']
    input_ids = np.ma.compressed(np.ma.masked_where(attention_mask, input_ids))
    aspect_tags = np.ma.compressed(np.ma.masked_where(attention_mask, aspect_tags))

    print(len(input_ids))
    # print(len(aspect_tags))
    # print(input_ids)
    # print(aspect_tags)

    # items_to_replace = set([101, 102])
    # aspect_tags = [1 if x in items_to_replace else x for x in aspect_tags]
    # print(aspect_tags)

    print(train_dataset.tokenizer.convert_ids_to_tokens(input_ids))
    print(encoder.inverse_transform(aspect_tags))

    for batch in train_data_loader:
        print(batch)
        input_ids_list = batch['input_ids']
        print(input_ids_list.shape)
        break


def loss_fn(pred, mask, label):
    label.masked_fill_(~mask, -100)
    pred = pred.view(-1, 2)
    label = label.view(-1)
    loss = torch.nn.functional.cross_entropy(pred, label)
    return loss


class AspectExtractionModel(nn.Module):
    def __init__(self, num_aspect_tags):
        super(AspectExtractionModel, self).__init__()
        self.num_aspect_tags = num_aspect_tags
        self.model = AutoModel.from_pretrained("microsoft/deberta-base")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, self.num_aspect_tags)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask=attention_mask, return_dict=False)
        tag_out = self.dropout(out[0])
        tag_out = self.fc(tag_out)

        return tag_out


def cal_acc(pred_tags, true_tags):
    if isinstance(pred_tags, list):
        pred_tags = torch.cat(pred_tags, 0)
        true_tags = torch.cat(true_tags, 0)
    pred_tags = pred_tags[true_tags!=-100]
    true_tags = true_tags[true_tags!=-100]
    acc = (pred_tags == true_tags).sum() / pred_tags.numel()
    f1 = f1_score(true_tags.cpu().numpy(), pred_tags.cpu().numpy(), labels=[0, 1], average='weighted')
    cm = confusion_matrix(true_tags.cpu().numpy(), pred_tags.cpu().numpy())

    return acc, f1, cm


def random_test(test_dataset, test_data_loader, model, encoder, device, num=5, model_path=None):
    if model_path is not None:  # load the saved model
        print('Loading saved model from: {}'.format(model_path))
        model.load_state_dict(torch.load(model_path))
    model = to_device(model, device)

    with torch.no_grad():
        for i in range(num):
            data = next(iter(test_data_loader))

            pred_tags, _ = model(**data)

            input_ids = data['input_ids']
            pred_tags = torch.argmax(pred_tags, dim=2)
            true_tags = data['aspect_tags']
            mask = data['attention_mask']

            # Randomly pick a test data from this batch
            #
            idx = np.random.randint(0, pred_tags.shape[0], size=1)[0]

            ids_array = input_ids[idx].cpu().numpy()
            pred_array = pred_tags[idx].cpu().numpy()
            true_array = true_tags[idx].cpu().numpy()
            mask_array = mask[idx].cpu().numpy()

            # Remove the padding as we do not want to print them
            #
            mask_array = np.logical_not(mask_array)

            # Only print the unpadded portion
            ids_unpadded = np.ma.compressed(np.ma.masked_where(mask_array, ids_array))
            pred_unpadded = np.ma.compressed(np.ma.masked_where(mask_array, pred_array))
            true_unpadded = np.ma.compressed(np.ma.masked_where(mask_array, true_array))

            acc = np.sum(pred_unpadded == true_unpadded) / len(pred_unpadded)

            print("Acc: {:.2f}%".format(acc * 100))
            print("Predicted:")
            print(encoder.inverse_transform(pred_unpadded))
            print("True:")
            print(encoder.inverse_transform(true_unpadded))
            print("Sentence:")
            print(test_dataset.tokenizer.convert_ids_to_tokens(ids_unpadded))
            print()


def main():
    path = "../data/restaurants_laptop_train_with_pos.csv"

    df = pd.read_csv(path)

    encoder = preprocessing.LabelEncoder()
    df.loc[:, "aspect_tag"] = encoder.fit_transform(df["aspect_tag"])
    print('num of aspect tags: {}'.format(len(encoder.classes_)))

    sentences = df.groupby("num")["text"].apply(list).values
    aspect_tags = df.groupby("num")["aspect_tag"].apply(list).values

    device = get_default_device()
    print(device)

    NUM_ASPECT_TAGS = len(encoder.classes_)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base", add_prefix_space=True)

    (train_sentences, test_sentences,
     train_aspect_tags, test_aspect_tags) = model_selection.train_test_split(
        sentences, aspect_tags, random_state=42, test_size=TEST_SIZE)

    train_dataset = SentenceTagDataset(tokenizer=tokenizer, sentences=train_sentences,
                                       aspect_tags=train_aspect_tags)
    print(f"train samples:{len(train_dataset)}")
    train_data_loader = DeviceDataLoader(torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True), device)

    test_dataset = SentenceTagDataset(tokenizer=tokenizer, sentences=test_sentences,
                                      aspect_tags=test_aspect_tags)
    print(f"valid samples:{len(test_dataset)}")
    test_data_loader = DeviceDataLoader(torch.utils.data.DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE), device)

    model = to_device(AspectExtractionModel(num_aspect_tags=NUM_ASPECT_TAGS), device)
    print(model)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    best_loss = np.inf

    for epoch in range(NUM_EPOCHS):
        train_losses = []
        train_acc = []
        test_loss = []
        test_acc = []

        model.train()
        preds = []
        labels = []
        for data in tqdm(train_data_loader, total=len(train_data_loader)):
            feature, mask, label = data["input_ids"], data["attention_mask"], data["aspect_tags"]
            feature, mask, label = feature.long(), mask.bool(), label.long()
            optimizer.zero_grad()
            pred_logits = model(feature, mask)
            loss = loss_fn(pred_logits, mask, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            pred_tags = pred_logits.max(-1)[1]
            preds.append(pred_tags)
            labels.append(label)
        avg_train_acc, avg_train_f1, train_cm = cal_acc(preds, labels)
        avg_train_loss = sum(train_losses) / len(train_losses)

        model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for data in tqdm(test_data_loader, total=len(test_data_loader)):
                feature, mask, label = data["input_ids"], data["attention_mask"], data["aspect_tags"]
                feature, mask, label = feature.long(), mask.bool(), label.long()
                pred_logits = model(feature, mask)
                loss = loss_fn(pred_logits, mask, label)

                test_loss.append(loss.item())

                pred_tags = pred_logits.max(-1)[1]
                preds.append(pred_tags)
                labels.append(label)

        avg_test_acc, avg_test_f1, test_cm = cal_acc(preds, labels)
        avg_test_loss = sum(test_loss) / len(test_loss)

        print(f"\nepoch {epoch}")
        print("\ttrain_loss:{:.3f} valid_loss:{:.3f}".format(avg_train_loss, avg_test_loss))
        print("\ttrain_acc:{:.2%} valid_acc:{:.2%}".format(avg_train_acc, avg_test_acc))
        print("\ttrain_f1:{:.3f} valid_f1:{:.3f}".format(avg_train_f1, avg_test_f1))
        print(f"\ttrain_confusion_matrix:\n{np.flip(train_cm)}")
        print(f"\tvalid_confusion_matrix:\n{np.flip(test_cm)}")

        if avg_test_loss < best_loss:
            torch.save(model.state_dict(), MODEL_PATH)
            best_loss = avg_test_loss


if __name__ == '__main__':
    main()
