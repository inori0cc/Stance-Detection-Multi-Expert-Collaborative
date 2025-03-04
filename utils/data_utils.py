import os
import pickle
import numpy as np
from transformers import BertTokenizer
import random
from sklearn.model_selection import train_test_split


stop_words = {'of', 'is', 'a', 'an', }


def rand_mask(txt, p=0.15):
    ts = txt.split(' ')
    for i, e in enumerate(ts):
        if random.random() < p:
            ts[i] = '[MASK]'
    return ' '.join(ts)


class Dataset(object):
    def __init__(self, targets, texts, stances, input_idss, attention_masks,
                 token_type_idss, cl_input_idss, cl_attention_masks, cl_token_type_idss):
        self.targets = targets
        self.texts = texts
        self.stances = stances
        self.input_idss = input_idss
        self.attention_masks = attention_masks
        self.token_type_idss = token_type_idss
        self.cl_input_idss = cl_input_idss
        self.cl_attention_masks = cl_attention_masks
        self.cl_token_type_idss = cl_token_type_idss

    def __getitem__(self, index):
        return self.targets[index], self.texts[index], self.stances[index], \
            self.input_idss[index], self.attention_masks[index], self.token_type_idss[index], \
            self.cl_input_idss[index], self.cl_attention_masks[index], self.cl_token_type_idss[index]

    def __len__(self):
        return len(self.targets)


class DatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_raw = lines[i].lower().strip()
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(fnames, tokenizer, maxlen):
        lines = []
        lines_count = [] 
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = lines + fin.readlines()
            fin.close()
            if len(lines_count) > 0:
                lines_count.append(len(lines) - lines_count[len(lines_count) - 1])
            else:
                lines_count.append(len(lines))

        targets = []
        texts = []
        stances = []
        input_idss = []
        attention_masks = []
        token_type_idss = []
        cl_input_idss = []
        cl_attention_masks = []
        cl_token_type_idss = []

        for i in range(0, len(lines), 4): 
            target = lines[i + 2].lower().strip()
            text = lines[i].lower().strip()
            maskd_text = lines[i + 1].lower().strip()

            stance = lines[i + 3].strip()

            ts = [s for s in target.split(' ') if s not in stop_words]
            for t in ts:
                maskd_text = maskd_text.replace(t, '[MASK]')

            if i < lines_count[0]:

#                 s = 0
#                 if int(stance) == 0:
#                     continue
#                 elif int(stance) == 1:
#                     s = 2
#                 elif int(stance) == 2:
#                     s = 1
#                 else:
#                     s = 0
#                 stance = s

                stance = int(stance) + 1  
            else:
                stance = -9

            org_token = tokenizer(target,
                                  text,
                                  add_special_tokens=True,
                                  max_length=maxlen,
                                  return_tensors='pt',
                                  padding='max_length',
                                  truncation=True)
            cl_token = tokenizer([target, target],
                                 [text, text], 
                                 add_special_tokens=True,
                                 max_length=maxlen,
                                 truncation=True,
                                 padding='max_length',
                                 return_tensors='pt')

            targets.append(target)
            texts.append(text)
            stances.append(stance)
            input_idss.append(org_token['input_ids'])
            attention_masks.append(org_token['attention_mask'])
            token_type_idss.append(org_token['token_type_ids'])
            cl_input_idss.append(cl_token['input_ids'])
            cl_attention_masks.append(cl_token['attention_mask'])
            cl_token_type_idss.append(cl_token['token_type_ids'])

        return targets, texts, stances, input_idss, attention_masks, token_type_idss, cl_input_idss, cl_attention_masks, cl_token_type_idss

    def __init__(self, opt, tokenizer, dataset='dt_hc'):
        print("preparing {0} dataset ...".format(dataset))

        # region db2db
        fname = {
            'vast': {
                'train': './proceed_dataset/vast/train.tp-5wd-6.masked',
                'test': './proceed_dataset/vast/vast_test.tp-5wd-6.masked'
            },
            'nac_ac': {
                'train': './proceed_dataset/wtwt_mask/nac.masked',
                'test': './proceed_dataset/wtwt_mask/ANTM_CI.tp-5wd-6.masked'
            },
            'nah_ah': {
                'train': './proceed_dataset/wtwt_mask/nah.masked',
                'test': './proceed_dataset/wtwt_mask/AET_HUM.tp-5wd-6.masked'
            },
            'nca_ca': {
                'train': './proceed_dataset/wtwt_mask/nca.masked',
                'test': './proceed_dataset/wtwt_mask/CVS_AET.tp-5wd-6.masked'
            },
            'nce_ce': {
                'train': './proceed_dataset/wtwt_mask/nce.masked',
                'test': './proceed_dataset/wtwt_mask/CI_ESRX.tp-5wd-6.masked'
            },
            'ac_ah': {
                'train': './proceed_dataset/wtwt_mask/ANTM_CI.tp-5wd-6.masked',
                'test': './proceed_dataset/wtwt_mask/AET_HUM.tp-5wd-6.masked'
            },
            'ac_ca': {
                'train': './proceed_dataset/wtwt_mask/ANTM_CI.tp-5wd-6.masked',
                'test': './proceed_dataset/wtwt_mask/CVS_AET.tp-5wd-6.masked'
            },
            'ac_ce': {
                'train': './proceed_dataset/wtwt_mask/ANTM_CI.tp-5wd-6.masked',
                'test': './proceed_dataset/wtwt_mask/CI_ESRX.tp-5wd-6.masked'
            },
            'ca_ce': {
                'train': './proceed_dataset/wtwt_mask/CVS_AET.tp-5wd-6.masked',
                'test': './proceed_dataset/wtwt_mask/CI_ESRX.tp-5wd-6.masked'
            },
            'ah_ca': {
                'train': './proceed_dataset/wtwt_mask/AET_HUM.tp-5wd-6.masked',
                'test': './proceed_dataset/wtwt_mask/CVS_AET.tp-5wd-6.masked'
            },
            'ah_ce': {
                'train': './proceed_dataset/wtwt_mask/AET_HUM.tp-5wd-6.masked',
                'test': './proceed_dataset/wtwt_mask/CI_ESRX.tp-5wd-6.masked'
            }
        }
        # endregion



        self.tokenizer = tokenizer
        targets, texts, stances, input_idss, attention_masks, token_type_idss, cl_input_idss, cl_attention_masks, cl_token_type_idss = \
            DatesetReader.__read_data__([fname[dataset]['train']], tokenizer, opt.maxlen)
        
        train_targets, val_targets, train_texts, val_texts, train_stances, val_stances, \
            train_input_idss, val_input_idss, train_attention_masks, val_attention_masks, \
            train_token_type_idss, val_token_type_idss, train_cl_input_idss, val_cl_input_idss, \
            train_cl_attention_masks, val_cl_attention_masks, train_cl_token_type_idss, val_cl_token_type_idss = \
            train_test_split(targets, texts, stances, input_idss, attention_masks, token_type_idss,
                             cl_input_idss, cl_attention_masks, cl_token_type_idss, test_size=0.2, random_state=42)

        self.train_data = Dataset(train_targets, train_texts, train_stances, train_input_idss,
                                  train_attention_masks, train_token_type_idss,
                                  train_cl_input_idss, train_cl_attention_masks,
                                  train_cl_token_type_idss)

        self.val_data = Dataset(val_targets, val_texts, val_stances, val_input_idss,
                                val_attention_masks, val_token_type_idss,
                                val_cl_input_idss, val_cl_attention_masks,
                                val_cl_token_type_idss)

        targets, texts, stances, input_idss, attention_masks, token_type_idss, cl_input_idss, cl_attention_masks, cl_token_type_idss = \
            DatesetReader.__read_data__([fname[dataset]['test']], tokenizer, opt.maxlen)
        self.test_data = Dataset(targets, texts, stances, input_idss,
                                 attention_masks, token_type_idss,
                                 cl_input_idss, cl_attention_masks,
                                 cl_token_type_idss)
