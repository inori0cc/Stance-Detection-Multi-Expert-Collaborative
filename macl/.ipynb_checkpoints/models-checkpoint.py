import torch.nn as nn
from torch.autograd import Variable
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertTokenizer, AutoModel, BertForMaskedLM,AutoTokenizer, BertForSequenceClassification
import macl.unsupervised_simCSE as unsupervised_simCSE
from torch.utils.data import DataLoader
from sklearn import metrics
from utils.Losses import simCSELoss
from transformers import logging
from .moe import MoE
from .vae import loss_vae, VAE


class MyVAE(nn.Module):

    def __init__(self, opt):
        super(MyVAE, self).__init__()
        self.opt = opt

        self.en11 = nn.Sequential(nn.Linear(opt.hidden_dim, opt.hidden_dim), nn.Dropout(), nn.ReLU())
        self.en12 = nn.Sequential(nn.Linear(opt.hidden_dim, opt.hidden_dim), nn.Dropout(), nn.ReLU())

        self.de = nn.Sequential(nn.Linear(opt.hidden_dim, opt.hidden_dim), nn.Dropout(), nn.ReLU())

    def encode(self, x):
        return F.relu(self.en11(x)), F.relu(self.en12(x)) #, self.en21(x), self.en22(x)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.de(z)


    def forward(self, x):
        mu1, logvar1 = self.encode(x.view(-1, self.opt.hidden_dim))
        z1 = self.reparametrize(mu1, logvar1)
        # z2 = self.reparametrize(mu2, logvar2)
        # z = torch.concat([z1, z2], dim=1)
        z = z1

        return z, self.decode(z), mu1, logvar1#, mu2, logvar2)

class MyModel(nn.Module):
    def __init__(self, opt):
        super(MyModel, self).__init__()
        self.opt = opt
        self.dropout = nn.Dropout(0.3)        
        self.vae = VAE(input_dim=opt.hidden_dim, h_dim=opt.hidden_dim, z_dim=opt.mdim)
        self.bert = AutoModel.from_pretrained(opt.plm, config=opt.bertConfig)
        self.bert1 = AutoModel.from_pretrained(opt.plm, config=opt.bertConfig)
        self.bert2 = AutoModel.from_pretrained(opt.plm, config=opt.bertConfig)
        self.bert3 = AutoModel.from_pretrained(opt.plm, config=opt.bertConfig)

        # fusion gate
        self.gate = nn.Sequential(
            nn.Linear(opt.experts * opt.hidden_dim, opt.experts * opt.hidden_dim), # concat
            nn.ReLU(),
            nn.Dropout()
        )

        self.fc = nn.Sequential(nn.Linear(opt.hidden_dim, 283), 
                                nn.ReLU(),
                                nn.Dropout(), 
                                nn.Linear(283, 3)
        )

    def forward(self, inputs):
        input_ids = inputs['input_ids'].view(len(inputs['input_ids']) , -1).to(self.opt.device)
        attention_mask = inputs['attention_mask'].view(len(inputs['attention_mask']), -1).to(self.opt.device)
        token_type_ids = inputs['token_type_ids'].view(len(inputs['token_type_ids']), -1).to(self.opt.device)


        x_hat, mu, log_var, z = None, None, None, None #self.vae(feats)

        '''不同bert均适用最后一层隐藏层作为特征输出层'''
#         feats = self.bert(input_ids, attention_mask=attention_mask,
#                     token_type_ids=token_type_ids,
#                     output_hidden_states=True, return_dict=False)[1]
        
#         feats1 = self.bert1(input_ids, attention_mask=attention_mask,
#                     token_type_ids=token_type_ids,
#                     output_hidden_states=True, return_dict=False)[1]
        
#         feats2 = self.bert2(input_ids, attention_mask=attention_mask,
#                     token_type_ids=token_type_ids,
#                     output_hidden_states=True, return_dict=False)[1]
        
#         feats3 = self.bert3(input_ids, attention_mask=attention_mask,
#                             token_type_ids=token_type_ids,
#                             output_hidden_states=True, return_dict=False)[1]


        ''''不同的bert使用不同的层次的隐藏层的[CLS]句子特征向量作为输出'''
        # 提取隐藏状态
        hidden_states = self.bert(input_ids, attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  output_hidden_states=True, return_dict=True)['hidden_states']
        hidden_states1 = self.bert1(input_ids, attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    output_hidden_states=True, return_dict=True)['hidden_states']
        hidden_states2 = self.bert2(input_ids, attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    output_hidden_states=True, return_dict=True)['hidden_states']
        hidden_states3 = self.bert3(input_ids, attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    output_hidden_states=True, return_dict=True)['hidden_states']

        # 分别从不同层提取特征并进行池化操作
        feats = hidden_states[-1][:, 0]  # 使用最后一层的 [CLS] token
        feats1 = hidden_states1[-5][:, 0]  # 使用倒数第5层的 [CLS] token
        feats2 = hidden_states2[-6][:, 0]  # 使用倒数第6层的 [CLS] token
        feats3 = hidden_states3[-7][:, 0]  # 使用倒数第7层的 [CLS] token


        feats_lst = [feats, feats1, feats2, feats3]

        # 使用ReLU单元进行融合
        feats = torch.cat(feats_lst, dim=-1)
        feats = self.dropout(feats)

        masks = self.gate(feats)
        new_feats = feats * masks
        new_feats = new_feats.reshape(input_ids.shape[0], self.opt.experts, -1)
        new_feats = torch.mean(new_feats, dim=1, dtype=torch.float32)

        '''消融实验'''
        # 使用Max-Pool进行融合
        # feats = torch.stack(feats_lst, dim=-1)  # 在新的维度上堆叠特征
        # feats = self.dropout(feats)
        #
        # max_pooled_feats = torch.max(feats, dim=-1)[0]  # 在最后一个维度（专家维度）上应用最大池化
        # max_pooled_feats = self.dropout(max_pooled_feats)
        # new_feats = max_pooled_feats

        # 使用Avg-pool进行融合
        # feats = torch.cat(feats_lst, dim=-1)
        # feats = self.dropout(feats)

        # 使用Avg-pool进行融合
        # feats = torch.cat(feats_lst, dim=-1)
        # feats = self.dropout(feats)
        # feats = feats.reshape(input_ids.shape[0], self.opt.experts, -1)

        # 进线性层分类
        predict = self.fc(new_feats)

        return predict, feats, x_hat, mu, log_var, None
