import pandas as pd
import numpy as np
import scipy
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch.optim import AdamW
from fasttext import load_model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

DATA_DIR = "./data/"
VALID_SIZE = .2


# generate word_index list
def build_vocab(data_dir, plain=[]):
    """plain is a empty str file which will record all text from official dataset"""
    for fn in os.listdir(data_dir):
        if fn.endswith('.xml'):
            with open(data_dir + fn) as f:
                dom = ET.parse(f)
                root = dom.getroot()
                for sent in root.iter("sentence"):
                    text = sent.find('text').text.lower()
                    token = word_tokenize(text)
                    plain = plain + token
    vocab = sorted(set(plain))
    with open(os.path.join(data_dir, "plain.txt"), "w+", encoding="utf8") as f:
        for v in vocab:
            f.write(f"{v}\n")
    word_idx = {}
    for idx, word in enumerate(vocab):
        word_idx[word] = idx + 1
    return word_idx


def gen_np_embedding(fn, word_idx, dim=100, emb=False):
    if emb:
        model = load_model(fn + ".bin")
    embedding = np.zeros((len(word_idx) + 2, dim))

    with open(fn, encoding="utf8") as f:
        for l in f:
            # for each line, get the word and its vector
            rec = l.rstrip().split(' ')
            if len(rec) == 2:  # skip the first line.
                continue
                # if the word in word_idx, fill the embedding
            if rec[0] in word_idx:
                embedding[word_idx[rec[0]]] = np.array([float(r) for r in rec[1:]])
    for w in word_idx:
        if embedding[word_idx[w]].sum() == 0.:
            if emb:
                embedding[word_idx[w]] = model.get_word_vector(w)
    return embedding


def create_train_data_restaurant(fn, word_idx, sent_len=83):
    dom = ET.parse(fn)
    root = dom.getroot()
    train_X = np.zeros((len(root), sent_len), np.int16)
    mask = np.zeros_like(train_X)

    train_y = np.zeros((len(root), sent_len), np.int16)
    take = np.ones(len(root), dtype=bool)

    dom = ET.parse(fn)
    root = dom.getroot()
    # iterate the sentence
    for sx, sent in enumerate(root.iter("sentence")):
        # TODO temporary to compare this and transformers
        if not [_ for _ in sent.iter("aspectTerm")]:
            take[sx] = False
            continue
        text = sent.find('text').text.lower()
        # tokenize the current sentence
        token = word_tokenize(text)

        # write word index and tag in train_X
        try:
            for wx, word in enumerate(token):
                train_X[sx, wx] = word_idx[word]
                mask[sx, wx] = 1
        except KeyError:
            continue

        for ox, apin in enumerate(sent.iter('aspectTerms')):
            for ax, opin in enumerate(apin.iter('aspectTerm')):
                target, polarity, start, end = opin.attrib['term'], opin.attrib['polarity'], int(
                    opin.attrib['from']), int(opin.attrib['to'])
                # find word index (instead of str index) if start,end is not (0,0)
                if end != 0:
                    if start != 0:
                        start = len(word_tokenize(text[:start]))
                    end = len(word_tokenize(text[:end])) - 1
                    # for training only identify aspect word, but not polarity
                    train_y[sx, start] = 1
                    if end > start:
                        # train_y[sx, start + 1:end] = 2
                        train_y[sx, start + 1:end] = 1
    return (train_X[take], mask[take]), train_y[take]


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


def loss_fn(pred, mask, label):
    label.masked_fill_(~mask, -100)
    pred = pred.view(-1, 2)
    label = label.view(-1)
    loss = torch.nn.functional.cross_entropy(pred, label)
    return loss


def cal_acc(pred_tags, mask, true_tags):
    if isinstance(pred_tags, list):
        pred_tags = torch.cat(pred_tags, 0)
        mask = torch.cat(mask, 0)
        true_tags = torch.cat(true_tags, 0)
    pred_tags = pred_tags[mask]
    true_tags = true_tags[mask]
    acc = (pred_tags == true_tags).sum() / pred_tags.numel()
    f1 = f1_score(true_tags.cpu().numpy(), pred_tags.cpu().numpy(), labels=[0, 1], average='weighted')
    cm = confusion_matrix(true_tags.cpu().numpy(), pred_tags.cpu().numpy())

    return acc, f1, cm


class Model(torch.nn.Module):
    def __init__(self, gen_emb, domain_emb, num_classes=3, dropout=0.5):
        super(Model, self).__init__()
        self.gen_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.gen_embedding.weight = torch.nn.Parameter(torch.from_numpy(gen_emb), requires_grad=False)
        self.domain_embedding = torch.nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight = torch.nn.Parameter(torch.from_numpy(domain_emb), requires_grad=False)
        self.conv1 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], 128, 5, padding=2)
        self.conv2 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], 128, 3, padding=1)
        self.dropout = torch.nn.Dropout(dropout)

        self.conv3 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv4 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv5 = torch.nn.Conv1d(256, 256, 5, padding=2)

        self.lstm = nn.LSTM(256, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)

        self.linear_ae = torch.nn.Linear(256, num_classes)

    def forward(self, x_train):
        x_emb = torch.cat((self.gen_embedding(x_train), self.domain_embedding(x_train)), dim=2)
        x_emb = self.dropout(x_emb).transpose(1, 2)

        x_conv = torch.nn.functional.relu(torch.cat((self.conv1(x_emb.float()), self.conv2(x_emb.float())), dim=1))
        x_conv = self.dropout(x_conv)

        x_conv = torch.nn.functional.relu(self.conv3(x_conv))
        x_conv = self.dropout(x_conv)

        x_conv = torch.nn.functional.relu(self.conv4(x_conv))
        x_conv = self.dropout(x_conv)

        x_conv = torch.nn.functional.relu(self.conv5(x_conv))
        x_conv = x_conv.transpose(1, 2)

        x_lstm, (hidden, cell) = self.lstm(x_conv)

        x_logit = self.linear_ae(x_lstm)

        return x_logit


def main():
    word_indx = build_vocab(DATA_DIR)
    fn = DATA_DIR + 'restaurant_emb.vec'
    res_domain_embedding = gen_np_embedding(fn, word_indx, dim=100, emb=True)

    # fn = DATA_DIR + 'laptop_emb.vec'
    # lap_domain_embedding = gen_np_embedding(fn, word_indx, dim=100, emb=True)

    # res_domain_embedding = np.concatenate([res_domain_embedding, lap_domain_embedding], axis=0)

    fn = DATA_DIR + 'glove.840B.300d.txt'
    general_embedding = gen_np_embedding(fn, word_indx, dim=300, emb=False)

    fn = DATA_DIR + 'Restaurants_Train_v2.xml'
    (X_train_res, mask_res), y_train_res = create_train_data_restaurant(fn, word_indx, sent_len=100)
    X, mask, y = X_train_res, mask_res, y_train_res
    # fn = DATA_DIR + 'Laptop_Train_v2.xml'
    # (X_train_lap, mask_lap), y_train_lap = create_train_data_restaurant(fn, word_indx, sent_len=100)
    # X = np.concatenate([X_train_res, X_train_lap], axis=0)
    # mask = np.concatenate([mask_res, mask_lap], axis=0)
    # y = np.concatenate([y_train_res, y_train_lap], axis=0)
    X_train, X_valid, mask_train, mask_valid, y_train, y_valid = train_test_split(X, mask, y, test_size=VALID_SIZE)

    # print(X_train[:3])
    # print(mask_train[:3])
    # print(y_train[:3])

    device = get_default_device()

    NUM_EPOCHS = 20
    TRAIN_BATCH_SIZE = 128
    VALID_BATCH_SIZE = 1024

    NUM_ASPECT_TAGS = 3

    dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(mask_train), torch.Tensor(y_train))
    print(f"train samples:{len(dataset)}")
    train_loader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)

    dataset_valid = TensorDataset(torch.Tensor(X_valid), torch.Tensor(mask_valid), torch.Tensor(y_valid))
    print(f"valid samples:{len(dataset_valid)}")
    test_loader = DataLoader(dataset_valid, batch_size=VALID_BATCH_SIZE)

    model = to_device(Model(general_embedding, res_domain_embedding, num_classes=2), device)

    torch.cuda.empty_cache()

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(parameters, lr=1e-4)


    for epoch in range(NUM_EPOCHS):
        train_losses = []
        train_acc = []
        test_loss = []
        test_acc = []
        train_f1 = []
        test_f1 = []

        model.train()
        preds = []
        masks = []
        labels = []
        for data in tqdm(train_loader, total=len(train_loader)):
            for i in range(len(data)):
                data[i] = data[i].to(device)
            feature, mask, label = data
            feature, mask, label = feature.long(), mask.bool(), label.long()
            optimizer.zero_grad()

            pred_logits = model(feature)
            loss = loss_fn(pred_logits, mask, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            pred_tags = pred_logits.max(-1)[1]
            preds.append(pred_tags)
            masks.append(mask)
            labels.append(label)

        avg_train_acc, avg_train_f1, train_cm = cal_acc(preds, masks, labels)
        avg_train_loss = sum(train_losses) / len(train_losses)

        preds = []
        masks = []
        labels = []
        with torch.no_grad():
            for data in tqdm(test_loader, total=len(test_loader)):
                for i in range(len(data)):
                    data[i] = data[i].to(device)
                feature, mask, label = data
                feature, mask, label = feature.long(), mask.bool(), label.long()
                pred_logits = model(feature)
                loss = loss_fn(pred_logits, mask, label)

                test_loss.append(loss.item())

                pred_tags = pred_logits.max(-1)[1]

                preds.append(pred_tags)
                masks.append(mask)
                labels.append(label)

        avg_test_acc, avg_test_f1, test_cm = cal_acc(preds, masks, labels)
        avg_test_loss = sum(test_loss) / len(test_loss)

        print(f"\nepoch {epoch}")
        print("\ttrain_loss:{:.3f} valid_loss:{:.3f}".format(avg_train_loss, avg_test_loss))
        print("\ttrain_acc:{:.2%} valid_acc:{:.2%}".format(avg_train_acc, avg_test_acc))
        print("\ttrain_f1:{:.3f} valid_f1:{:.3f}".format(avg_train_f1, avg_test_f1))
        print(f"\ttrain_confusion_matrix:\n{train_cm}")
        print(f"\tvalid_confusion_matrix:\n{test_cm}")


if __name__ == '__main__':
    main()
