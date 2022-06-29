import os
import random
from turtle import pos
import torch.utils.data as data
from transformers import AutoTokenizer


class Unsupervised(data.Dataset):
    def __init__(self, root="./data/") -> None:
        super(Unsupervised, self).__init__()
        self.root = root
        self.all = os.path.join(self.root, "corpus.tsv")
        self.all_data = []
        self._create_train_data()
        self.tokenizer = AutoTokenizer.from_pretrained(
            'hfl/chinese-roberta-wwm-ext')

    def _create_train_data(self):
        with open(self.all, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                self.all_data.append((line[1], line[1]))

    def __getitem__(self, index):
        sample = self._post_process(self.all_data[index])
        return sample

    def _post_process(self, text):
        anchor = self.tokenizer([text, text],
                                truncation=True,
                                add_special_tokens=True,
                                max_length=48,
                                padding='max_length',
                                return_tensors='pt').to("cuda:0")
        return anchor

    def __len__(self):
        return len(self.all_data)


class Supervised(data.Dataset):
    def __init__(self, root="./data/") -> None:
        super(Supervised, self).__init__()
        self.root = root
        self.all = os.path.join(self.root, "corpus.tsv")
        self.train = os.path.join(self.root, "train.query.txt")
        self.corr = os.path.join(self.root, "qrels.train.tsv")
        self.test_labeled = os.path.join(self.root, "test_query.tsv")
        self.test_query = os.path.join(self.root, "dev.query.txt")

        self.test2train = {}
        self.test_query2num = {}

        self.split_char = "|||"
        self.all_data = {}
        self.train_data = {}
        self._create_train_data()
        self.tokenizer = AutoTokenizer.from_pretrained(
            'hfl/chinese-roberta-wwm-ext')



    def _create_train_data(self):
        with open(self.train, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                self.train_data[line[0]] = line[1] + self.split_char

        line_bias = 100001
        with open(self.test_query, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                self.train_data[str(line_bias)] = line[1] + self.split_char
                self.test2train[line[0]] = str(line_bias)
                self.test_query2num[line[1]] = str(line_bias)
                line_bias += 1

        with open(self.all, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                self.all_data[line[0]] = line[1]

        aug_item = 0
        with open('data/jd_item_0.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                aug_item += 1
                line = line.strip().split("\t")
                # print(line)
                k = line[0]
                v = eval(line[1])
                for i in range(len(v)):
                    self.train_data[self.test_query2num[k]] += v[i]
        with open('data/jd_item_1.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                aug_item += 1
                line = line.strip().split("\t")
                # print(line)
                k = line[0]
                v = eval(line[1])
                for i in range(len(v)):
                    self.train_data[self.test_query2num[k]] += v[i]
        with open('data/jd_item_2.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                aug_item += 1
                line = line.strip().split("\t")
                # print(line)
                k = line[0]
                v = eval(line[1])
                for i in range(1):
                    self.train_data[self.test_query2num[k]] += v[i]
        print(aug_item)

        with open(self.test_labeled, 'r') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                k = line[1]
                v = eval(line[2])
                for i in range(len(v)):
                    self.train_data[self.test2train[k]] += self.all_data[str(v[i])]

        with open(self.corr, 'r') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                k = line[0]
                v = line[1]
                self.train_data[k] += self.all_data[v]

    def __getitem__(self, index):
        index = str(index + 1)
        anchor_text, pos_text = self.train_data[index].split(self.split_char)
        # 考虑增加负采样
        tmp = random.randint(1, 1001492)
        neg_text = self.all_data[str(tmp)]

        sample = self._post_process(anchor_text, pos_text, neg_text)
        return sample

    def _post_process(self, anchor_text, pos_text, neg_text):
        sample = self.tokenizer([anchor_text, pos_text, neg_text],
                                truncation=True,
                                add_special_tokens=True,
                                max_length=48,
                                padding='max_length',
                                return_tensors='pt').to("cuda:0")

        return sample

    def __len__(self):
        return len(self.train_data)


class TESTDATA(data.Dataset):
    def __init__(self, root="./data/", certain="corpus.tsv") -> None:
        super(TESTDATA, self).__init__()
        self.root = root
        self.all = os.path.join(self.root, certain)
        self.all_data = {}
        self.query_extend = {}
        self.certain = certain
        self._create_eval_data()
        self.start = 0
        self.length = 48


        if certain != "corpus.tsv":
            self.start = 200000
        self.tokenizer = AutoTokenizer.from_pretrained(
            'hfl/chinese-roberta-wwm-ext')

    def _create_eval_data(self):

        # with open('data/jd_item_0.txt', 'r', encoding='utf-8') as f:
        #     for line in f.readlines():
        #         line = line.strip().split("\t")
        #         k = line[0]
        #         v = eval(line[1])
        #         self.query_extend[k] = v[0] + v[1]
        #
        # with open('data/jd_item_1.txt', 'r', encoding='utf-8') as f:
        #     for line in f.readlines():
        #
        #         line = line.strip().split("\t")
        #         # print(line)
        #         k = line[0]
        #         v = eval(line[1])
        #         self.query_extend[k] = v[0] + v[1]
        #
        # with open('data/jd_item_2.txt', 'r', encoding='utf-8') as f:
        #     for line in f.readlines():
        #         line = line.strip().split("\t")
        #         # print(line)
        #         k = line[0]
        #         v = eval(line[1])
        #         self.query_extend[k] = v[0] + v[1]

        with open(self.all, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                # if self.certain != "corpus.tsv":
                #     if line[1] in self.query_extend:
                #         self.all_data[line[0]] = line[1] + self.query_extend[line[1]]
                #     else:
                #         self.all_data[line[0]] = line[1]
                # else:
                self.all_data[line[0]] = line[1]

    def __getitem__(self, index):
        id_, text = str(index + self.start +
                        1), self.all_data[str(index + self.start + 1)]
        data = self.tokenizer(text,
                              truncation=True,
                              add_special_tokens=True,
                              max_length=self.length,
                              padding='max_length',
                              return_tensors='pt').to("cuda:0")

        return id_, data

    def __len__(self):
        return len(self.all_data)
