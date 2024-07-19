import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn import metrics
from utils.data_utils import DatesetReader
from .models import MyModel


class Instructor:
    def __init__(self, opt) -> None:
        self.opt = opt
        self.max_acc = 0
        self.max_f1 = 0
        self.max_f1_m = 0
        self.max_f1_all = 0
        self.max_f1_0 = 0
        self.max_f1_1 = 0
        self.max_f1_2 = 0
        self.max_f1_val = 0
        self.max_f1_m_val = 0
        self.max_f1_all_val = 0
        self.max_f1_0_val = 0
        self.max_f1_1_val = 0
        self.max_f1_2_val = 0
        self.best_val_prescision = None
        self.best_val_recall = None
        
        # region
        self.tokenizer = AutoTokenizer.from_pretrained(opt.plm)  # bert-base-uncased
        self.stance_dataset = DatesetReader(opt, self.tokenizer, dataset=opt.db)
        self.train_data_loader = DataLoader(self.stance_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(self.stance_dataset.test_data, batch_size=opt.batch_size, shuffle=False)
        self.val_data_loader = DataLoader(self.stance_dataset.val_data, batch_size=opt.batch_size, shuffle=True)


        self.model = MyModel(self.opt).to(self.opt.device)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=opt.lr,
                                          weight_decay=opt.wd)

        self.bert_criterion = nn.CrossEntropyLoss()
        # endregion

    def train(self):
        global_step = 0

        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print(' ' * 100)
            print('epoch: ', epoch)

            n_correct, n_total = 0, 0
            for _, sample_batched in enumerate(self.train_data_loader):
                global_step += 1
                self.model.train()
                self.optimizer.zero_grad()

                targets, texts, stances, input_idss, attention_masks, token_type_idss, cl_input_idss, cl_attention_masks, cl_token_type_idss, \
                    = sample_batched
                if len(stances) == 1: 
                    continue
                labels = stances.to(self.opt.device)
                inputs = {
                    'input_ids': input_idss.squeeze().to(self.opt.device),
                    'attention_mask': attention_masks.squeeze().to(self.opt.device),
                    'token_type_ids': token_type_idss.squeeze().to(self.opt.device),
                    'cl_input_ids': cl_input_idss.squeeze().to(self.opt.device),
                    'cl_attention_mask': cl_attention_masks.squeeze().to(self.opt.device),
                    'cl_token_type_ids': cl_token_type_idss.squeeze().to(self.opt.device)
                }

                predict, _ = self.model(inputs)
                bert_loss = self.bert_criterion(predict, labels)
                loss = bert_loss

                loss.backward()
                self.optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(predict, -1) == labels).sum().item()
                    n_total += len(predict)
                    train_acc = n_correct / n_total

                    val_acc, val_f1, val_f1_m, f1_0_val, f1_1_val, f1_2_val, f1_all_val, val_precision, val_recall = self.validate(epoch)
                    test_acc, test_f1, test_f1_m, f1_0, f1_1, f1_2, f1_all, test_precision, test_recall = self.test(epoch)
                    if self.opt.db == 'fvast' or self.opt.db == 'vast':
                        print('-' * 100)
                        print(f'bert_loss: {bert_loss.item():.4f}, f1_0_val:{f1_0_val:.4f}, f1_1_val:{f1_1_val:.4f}, f1_2_val:{f1_2_val:.4f}, f1_all_val:{f1_all_val:.4f},val_precision:{val_precision:.4f},val_recall:{val_recall:.4f}')
                        print(f'bert_loss: {bert_loss.item():.4f}, f1_0:{f1_0:.4f}, f1_1:{f1_1:.4f}, f1_2:{f1_2:.4f}, f1_all:{f1_all:.4f},precision:{precision:.4f},recall:{recall:.4f}')
                    else:
                        print('-' * 100)
                        print('bert_loss: {:.4f}, val_acc: {:.4f}, val_f1: {:.4f}, val_f1_m: {:.4f}, val_precision:{:.4f}, val_recall:{:.4f}'.
                              format(bert_loss.item(), val_acc, val_f1, val_f1_m, val_precision, val_recall))
                        print('bert_loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}, test_f1_m: {:.4f}, test_precision:{:.4f}, test_recall:{:.4f}'.
                              format(bert_loss.item(), train_acc, test_acc, test_f1, test_f1_m, test_precision, test_recall))


            if self.opt.db == 'fvast' or self.opt.db == 'vast':
                result = f'{self.opt.db}, lr: {self.opt.lr}, max_epoch: {self.max_epoch}, max_f1_0: {self.max_f1_0:.4f}, max_f1_1: {self.max_f1_1:.4f}, max_f1_2: {self.max_f1_2:.4f}, max_f1_all:{self.max_f1_all:.4f}, max_f1_all_val:{self.max_f1_all_val:.4f}'
                print(result)
            else:
                print(
                    f'{self.opt.db}, max_epoch: {self.max_epoch}, max_f1: {self.max_f1:.4f}, max_f1_m: {self.max_f1_m:.4f}, max_f1_m_val:{self.max_f1_m_val:.4f}')

            if epoch - self.max_epoch > self.opt.endurance:
                print('early stop.')
                break

        try:
            if self.opt.db == 'fvast' or self.opt.db == 'vast':
                result = f'{self.opt.db}, lr: {self.opt.lr}, max_epoch: {self.max_epoch}, max_f1_0: {self.max_f1_0:.4f}, max_f1_1: {self.max_f1_1:.4f}, max_f1_2: {self.max_f1_2:.4f}, max_f1_all:{self.max_f1_all:.4f}'
            else:
                result = f'{self.opt.db}, lr: {self.opt.lr}, max_epoch: {self.max_epoch}, max_f1: {self.max_f1:.4f}, max_f1_m: {self.max_f1_m:.4f}, max_f1_m_val:{self.max_f1_m_val:.4f}'
                

            print('final_output')
            self.model = torch.load('best_model.pt')
            self.model.eval()
            test_acc, test_f1, test_f1_m, f1_0, f1_1, f1_2, f1_all, test_precision, test_recall = self.test(epoch)
            result = f'test_f1_m: {test_f1_m}, f1_all:{f1_all}, test_precision: {test_precision}, test_recall: {test_recall}'
            print('best_val——>best_test', result)

        except:
            pass
        return self.max_f1, self.max_f1_m

    def validate(self, epoch):
        n_val_correct, n_val_total = 0, 0
        v_labels_all, v_outputs_all = None, None

        self.model.eval()
        with torch.no_grad():
            for _, v_sample_batched in enumerate(self.val_data_loader):
                targets, texts, stances, input_idss, attention_masks, token_type_idss, cl_input_idss, cl_attention_masks, cl_token_type_idss = v_sample_batched
                if len(stances) == 1:
                    continue
                labels = stances.to(self.opt.device)
                inputs = {
                    'input_ids': input_idss.squeeze().to(self.opt.device),
                    'attention_mask': attention_masks.squeeze().to(self.opt.device),
                    'token_type_ids': token_type_idss.squeeze().to(self.opt.device),
                    'cl_input_ids': cl_input_idss.squeeze().to(self.opt.device),
                    'cl_attention_mask': cl_attention_masks.squeeze().to(self.opt.device),
                    'cl_token_type_ids': cl_token_type_idss.squeeze().to(self.opt.device)
                }
                predict, _ = self.model(inputs)

                n_val_correct += (torch.argmax(predict, -1) == labels).sum().item()
                n_val_total += len(predict)

                if v_labels_all is None:
                    v_labels_all = labels
                    v_outputs_all = predict
                else:
                    v_labels_all = torch.cat((v_labels_all, labels), dim=0)
                    v_outputs_all = torch.cat((v_outputs_all, predict), dim=0)

        val_acc = n_val_correct / n_val_total
        f1_all = metrics.f1_score(v_labels_all.cpu(), torch.argmax(v_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        f1 = metrics.f1_score(v_labels_all.cpu(), torch.argmax(v_outputs_all, -1).cpu(), labels=[0, 2], average='macro')
        f1_mi = metrics.f1_score(v_labels_all.cpu(), torch.argmax(v_outputs_all, -1).cpu(), labels=[0, 2], average='micro')
        f1_m = 0.5 * (f1 + f1_mi)
        precision = metrics.precision_score(v_labels_all.cpu(), torch.argmax(v_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        recall = metrics.recall_score(v_labels_all.cpu(), torch.argmax(v_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')

        f1_0 = metrics.f1_score(v_labels_all.cpu(), torch.argmax(v_outputs_all, -1).cpu(), labels=[0], average='macro')
        f1_1 = metrics.f1_score(v_labels_all.cpu(), torch.argmax(v_outputs_all, -1).cpu(), labels=[1], average='macro')
        f1_2 = metrics.f1_score(v_labels_all.cpu(), torch.argmax(v_outputs_all, -1).cpu(), labels=[2], average='macro')

        if f1 > self.max_f1_val:
            self.max_f1_val = f1
            self.max_epoch_val = epoch

        if f1_m > self.max_f1_m_val:
            self.max_f1_m_val = f1_m
            self.max_epoch_m_val = epoch
            torch.save(self.model, 'best_model.pt')

        if f1_all > self.max_f1_all_val:
            self.max_f1_all_val = f1_all
            self.max_epoch_val = epoch
            self.max_f1_0_val = f1_0
            self.max_f1_1_val = f1_1
            self.max_f1_2_val = f1_2


        return val_acc, f1, f1_m, f1_0, f1_1, f1_2, f1_all, precision, recall


    def test(self, epoch):
        n_test_correct, n_test_total = 0, 0
        t_labels_all, t_outputs_all = None, None

        self.model.eval()
        with torch.no_grad():
            for _, t_sample_batched in enumerate(self.test_data_loader):
                targets, texts, stances, input_idss, attention_masks, token_type_idss, cl_input_idss, cl_attention_masks, cl_token_type_idss = t_sample_batched
                if len(stances) == 1:
                    continue
                labels = stances.to(self.opt.device)
                inputs = {
                    'input_ids': input_idss.squeeze().to(self.opt.device),
                    'attention_mask': attention_masks.squeeze().to(self.opt.device),
                    'token_type_ids': token_type_idss.squeeze().to(self.opt.device),
                    'cl_input_ids': cl_input_idss.squeeze().to(self.opt.device),
                    'cl_attention_mask': cl_attention_masks.squeeze().to(self.opt.device),
                    'cl_token_type_ids': cl_token_type_idss.squeeze().to(self.opt.device)
                }
                predict, _ = self.model(inputs)

                n_test_correct += (torch.argmax(predict, -1) == labels).sum().item()
                n_test_total += len(predict)

                if t_labels_all is None:
                    t_labels_all = labels
                    t_outputs_all = predict
                else:
                    t_labels_all = torch.cat((t_labels_all, labels), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, predict), dim=0)

        test_acc = n_test_correct / n_test_total
        f1_all = metrics.f1_score(t_labels_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                                  average='macro')
        f1 = metrics.f1_score(t_labels_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 2], average='macro')
        f1_mi = metrics.f1_score(t_labels_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 2],
                                 average='micro')
        f1_m = 0.5 * (f1 + f1_mi)
        precision = metrics.precision_score(t_labels_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                                            average='macro')
        recall = metrics.recall_score(t_labels_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                                      average='macro')

        f1_0 = metrics.f1_score(t_labels_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0], average='macro')
        f1_1 = metrics.f1_score(t_labels_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[1], average='macro')
        f1_2 = metrics.f1_score(t_labels_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[2], average='macro')

        if f1 > self.max_f1:
            self.max_f1 = f1
            self.max_epoch = epoch

        if f1_m > self.max_f1_m:
            self.max_f1_m = f1_m
            self.max_epoch = epoch

        if f1_all > self.max_f1_all:
            self.max_f1_all = f1_all
            self.max_epoch = epoch
            self.max_f1_0 = f1_0
            self.max_f1_1 = f1_1
            self.max_f1_2 = f1_2

        return test_acc, f1, f1_m, f1_0, f1_1, f1_2, f1_all, precision, recall
