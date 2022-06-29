import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data import Unsupervised, Supervised, TESTDATA
from model import TextBackbone
from datetime import datetime
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

def unsup_loss(y_pred, lamda=0.05):
    idxs = torch.arange(0, y_pred.shape[0], device="cuda")
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)

    similarities = similarities - torch.eye(y_pred.shape[0], device="cuda") * 1e12

    similarities = similarities / lamda

    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def sup_loss(y_pred, lamda=0.05):
    # lambda 考虑变化
    row = torch.arange(0, y_pred.shape[0], 3, device="cuda")
    col = torch.arange(y_pred.shape[0], device="cuda")
    col = torch.where(col % 3 != 0)[0].cuda()
    y_true = torch.arange(0, len(col), 2, device="cuda")
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)

    similarities = torch.index_select(similarities, 0, row)
    similarities = torch.index_select(similarities, 1, col)

    similarities = similarities / lamda

    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def train(dataloader, model, fgm, optimizer, schedular, criterion, log_file, mode="unsup"):
    num = 2
    if mode == "sup":
        num = 3
    model.train()
    all_loss = []
    for idx, data in enumerate(tqdm(dataloader)):
        input_ids = data["input_ids"].view(len(data["input_ids"]) * num, -1).cuda()
        attention_mask = (
            data["attention_mask"].view(len(data["attention_mask"]) * num, -1).cuda()
        )
        token_type_ids = (
            data["token_type_ids"].view(len(data["token_type_ids"]) * num, -1).cuda()
        )
        pred = model(input_ids, attention_mask, token_type_ids)
        # optimizer.zero_grad()
        loss = criterion(pred)
        all_loss.append(loss.item())
        loss.backward() # 反向传播，得到正常的grad

        # 对抗训练
        fgm.attack()  # 在embedding上添加对抗扰动
        pred_adv = model(input_ids, attention_mask, token_type_ids)
        loss_adv = criterion(pred_adv)
        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore()  # 恢复embedding参数

        optimizer.step()
        schedular.step()
        optimizer.zero_grad()

        if idx % 500 == 499:
            with open(log_file, "a+") as f:
                t = sum(all_loss) / len(all_loss)
                info = str(idx) + " == {} == ".format(mode) + str(t) + "\n"
                f.write(info)
                all_loss = []


def prepare():
    os.makedirs("./output", exist_ok=True)
    now = datetime.now()
    log_file = now.strftime("%Y_%m_%d_%H_%M_%S") + "_log.txt"
    return "./output/" + log_file


def unsupervised_train():
    dataset = Unsupervised()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    model = TextBackbone().cuda()
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    # unspervise train
    epochs = 1
    num_train_steps = int(len(dataloader) * epochs)
    schedular = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.05 * num_train_steps,
        num_training_steps=num_train_steps,
    )

    criterion = unsup_loss

    log_file = prepare()

    for epoch in range(1, epochs + 1):
        train(dataloader, model, optimizer, schedular, criterion, log_file)
        torch.save(model.state_dict(), "./output/unsup_epoch_{}.pt".format(epoch))

    return log_file, model

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='extractor.embeddings.word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                # print('fgm attack')
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='extractor.embeddings.word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}

if __name__ == "__main__":

    # log_file, model = unsupervised_train()
    log_file = prepare()
    model = TextBackbone().cuda()
    # FGM
    fgm = FGM(model)
    dataset = Supervised()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5) # 3e-5
    epochs = 5
    num_train_steps = int(len(dataloader) * epochs)
    schedular = get_linear_schedule_with_warmup(
        optimizer,
        # num_warmup_steps=0.05 * num_train_steps,
        num_warmup_steps=0,
        num_training_steps=num_train_steps,
    )

    criterion = sup_loss

    for epoch in range(1, epochs + 1):
        train(dataloader, model, fgm, optimizer, schedular, criterion, log_file, mode="sup")
        torch.save(model.state_dict(), "./output/sup_epoch_{}.pt".format(epoch))

    if os.path.exists("doc_embedding"):
        os.remove("doc_embedding")
    if os.path.exists("query_embedding"):
        os.remove("query_embedding")

    # model.load_state_dict(torch.load("./output/sup_epoch_5.pt"))
    model.eval()

    testdata = TESTDATA(certain="dev.query.txt")
    testloader = DataLoader(testdata, batch_size=1, shuffle=False)
    for idx, x in tqdm(testloader):
        with torch.no_grad():
            y = model.predict(x)[0].detach().cpu().numpy().tolist()
            y = [str(i) for i in y]
            info = idx[0] + "\t"
            info = info + ",".join(y)
            with open("query_embedding", "a+") as f:
                f.write(info + "\n")

    testdata = TESTDATA(certain="corpus.tsv")
    testloader = DataLoader(testdata, batch_size=60, shuffle=False)
    for idx, x in tqdm(testloader):
        with torch.no_grad():
            y = model.predict(x).detach().cpu().numpy().tolist()
            for x1, y1 in zip(idx, y):
                y1 = [str(round(i, 8)) for i in y1]
                info = x1 + "\t"
                info = info + ",".join(y1)
                with open("doc_embedding", "a+") as f:
                    f.write(info + "\n")
