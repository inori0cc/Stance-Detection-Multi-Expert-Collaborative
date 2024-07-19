from transformers import BertConfig, AutoConfig
from transformers import logging
import argparse
from macl.main import Instructor
logging.set_verbosity_error()


#bert-base-uncased
class opt: maxlen = 128; device = 'cuda'; seed = None; log_step = 10; repeats = 5; plm = 'bert-base-uncased'; \
        db = None; endurance = 4; batch_size = 64; num_epoch = 5; wd = 1e-5; lr = 2e-5; \
        hidden_dim = 768; mdim = 283; temperature = 0.2;\
        experts = 4; \
        bertConfig = AutoConfig.from_pretrained(plm)


parser = argparse.ArgumentParser()
parser.add_argument('--db_name', default='', type=str)
ps = parser.parse_args()
if ps.db_name:
    opt.db = ps.db_name


dbs = ['nac_ac', 'nca_ca','nce_ce','nah_ah'] # ['vast'] 


for db in dbs:
    for _ in range(opt.repeats):
        opt.db = db
        ins = Instructor(opt)
        ins.train()
