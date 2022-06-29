import torch
from transformers import AutoModel, AutoConfig
import torch.nn.functional as F


class TextBackbone(torch.nn.Module):
    def __init__(self,
                 pretrained='hfl/chinese-roberta-wwm-ext',
                 output_dim=128) -> None:
        super(TextBackbone, self).__init__()

        config = AutoConfig.from_pretrained(pretrained)
        config.update(
            {
                "hidden_dropout_prob": 0.1,
                "layer_norm_eps": 1e-7,
            }
        )

        self.extractor = AutoModel.from_pretrained(pretrained, config=config).cuda()
        self.drop = torch.nn.Dropout(p=0.2)
        self.dropout = torch.nn.Dropout(0.1)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.dropout3 = torch.nn.Dropout(0.3)
        self.dropout4 = torch.nn.Dropout(0.4)
        self.dropout5 = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(768, output_dim)

        self.use_drop = False


    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.extractor(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             output_hidden_states=True)

        first = out.hidden_states[1].transpose(1, 2)
        last = out.hidden_states[-1].transpose(1, 2)
        first_avg = torch.avg_pool1d(
            first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(
            -1)  # [batch, 768]
        avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)),
                        dim=1)  # [batch, 2, 768]
        out = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)
        if self.use_drop:
            # dropout
            # pooled_output = self.dropout(out)
            logits0 = self.fc(out)
            logits1 = self.fc(self.dropout1(out))
            logits2 = self.fc(self.dropout2(out))
            logits3 = self.fc(self.dropout3(out))
            logits4 = self.fc(self.dropout4(out))
            logits5 = self.fc(self.dropout5(out))
            x = logits0 + logits1 + logits2 + logits3 + logits4 + logits5
        else:
            x = self.fc(out)
        x = F.normalize(x, p=2, dim=-1)
        return x

    # def forward(self, input_ids, attention_mask, token_type_ids):
    #     x = self.extractor(input_ids,
    #                        attention_mask=attention_mask,
    #                        token_type_ids=token_type_ids).pooler_output
    #
    #     x = self.drop(x)
    #     x = self.fc(x)
    #     x = F.normalize(x, p=2, dim=-1)
    #     return x

    def predict(self, x):
        x["input_ids"] = x["input_ids"].squeeze(1)
        x["attention_mask"] = x["attention_mask"].squeeze(1)
        x["token_type_ids"] = x["token_type_ids"].squeeze(1)

        out = self.extractor(**x,
                           output_hidden_states=True)

        first = out.hidden_states[1].transpose(1, 2)
        last = out.hidden_states[-1].transpose(1, 2)
        first_avg = torch.avg_pool1d(
            first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(
            -1)  # [batch, 768]
        avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)),
                        dim=1)  # [batch, 2, 768]
        out = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)
        if self.use_drop:
            # dropout
            # pooled_output = self.dropout(out)
            logits0 = self.fc(out)
            logits1 = self.fc(self.dropout1(out))
            logits2 = self.fc(self.dropout2(out))
            logits3 = self.fc(self.dropout3(out))
            logits4 = self.fc(self.dropout4(out))
            logits5 = self.fc(self.dropout5(out))
            x = logits0 + logits1 + logits2 + logits3 + logits4 + logits5
        else:
            x = self.fc(out)
        x = F.normalize(x, p=2, dim=-1)
        return x


    # def predict(self, x):
    #     x["input_ids"] = x["input_ids"].squeeze(1)
    #     x["attention_mask"] = x["attention_mask"].squeeze(1)
    #     x["token_type_ids"] = x["token_type_ids"].squeeze(1)
    #
    #     x = self.extractor(**x).pooler_output
    #     x = self.fc(x)
    #     x = F.normalize(x, p=2, dim=-1)
    #
    #     return x
