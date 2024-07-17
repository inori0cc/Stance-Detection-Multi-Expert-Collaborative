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
from transformers import logging


class MyModel(nn.Module):
    def __init__(self, opt):
        super(MyModel, self).__init__()
        self.opt = opt
        self.dropout = nn.Dropout(0.3)        
        self.bert = AutoModel.from_pretrained(opt.plm, config=opt.bertConfig)
        self.bert1 = AutoModel.from_pretrained(opt.plm, config=opt.bertConfig)
        self.bert2 = AutoModel.from_pretrained(opt.plm, config=opt.bertConfig)
        self.bert3 = AutoModel.from_pretrained(opt.plm, config=opt.bertConfig)

        # fusion gate
        self.gate = nn.Sequential(
            nn.Linear(opt.experts * opt.hidden_dim, opt.experts * opt.hidden_dim),
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

        
        feats = hidden_states[-1][:, 0]
        feats1 = hidden_states1[-5][:, 0]  
        feats2 = hidden_states2[-6][:, 0]  
        feats3 = hidden_states3[-7][:, 0]  


        feats_lst = [feats, feats1, feats2, feats3]

        feats = torch.cat(feats_lst, dim=-1)
        feats = self.dropout(feats)

        masks = self.gate(feats)
        new_feats = feats * masks
        new_feats = new_feats.reshape(input_ids.shape[0], self.opt.experts, -1)
        new_feats = torch.mean(new_feats, dim=1, dtype=torch.float32)

        predict = self.fc(new_feats)

        return predict, feats
