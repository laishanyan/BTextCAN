# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.dropout)

        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

        self.attention = Attention(config.pad_size)
        self.cans = nn.ModuleList(
            [CAN(config.pad_size, k) for k in config.filter_sizes]
        )
        self.pad_size = config.pad_size
        self.emb = config.hidden_size


    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = encoder_out.unsqueeze(1)
        out = torch.cat([can(out) for can in self.cans], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out

class CAN(nn.Module):
    def __init__(self, pad_size, k):
        super(CAN, self).__init__()
        self.dropout = 0.0
        self.emb = 768
        self.pad_size = pad_size - k + 1
        self.conv = nn.Conv2d(1, 1, (k, self.emb))
        self.attention = Attention(self.pad_size)
        self.filter_num = 8

    def attention_cnn_layer(self, x):
        out = F.relu(self.conv(x)).squeeze(3)
        out = self.attention(out)
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        return out


    def forward(self, x):
        out = self.attention_cnn_layer(x)
        for i in range(self.filter_num-1):
            out = torch.cat((out, self.attention_cnn_layer(x)), dim=1)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Attention(nn.Module):
    def __init__(self, pad_size, dropout=0.0):
        super(Attention, self).__init__()
        self.fc_Q = nn.Linear(pad_size, pad_size)
        self.fc_K = nn.Linear(pad_size, pad_size)
        self.fc_V = nn.Linear(pad_size, pad_size)
        self.attention = Scaled_Dot_Product_Attention()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(pad_size)
        self.pad_size = 40


    def forward(self, x):
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)
        out = self.dropout(context)
        out = self.layer_norm(out)
        return out
