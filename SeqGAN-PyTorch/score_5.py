import os
import os.path as osp
import argparse
import numpy as np
import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models

#from coco_loader import coco_loader
#from convcap import convcap
#from vggfeats import Vgg16Feats
from tqdm import tqdm
#from test import test
import discriminator as dis1
#import dis2
import pickle

SEED = 88
BATCH_SIZE = 15
TOTAL_BATCH = 200
GENERATED_NUM = 10000
POSITIVE_FILE = 'real.txt'
NEGATIVE_FILE = 'gene.data'
EVAL_FILE = 'eval.data'
VOCAB_SIZE = 9221
PRE_EPOCH_NUM = 5


# Genrator Parameters
g_emb_dim = 32
g_hidden_dim = 32
g_sequence_len = 15

# Discriminator Parameters
d_emb_dim = 64
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100]

d_dropout = 0.75
d_num_class = 2


worddict_tmp = pickle.load(open('data/wordlist.p', 'rb'))
wordlist = [l for l in iter(worddict_tmp.keys()) if l != '</S>']
wordlist = ['EOS'] + sorted(wordlist)
numwords = len(wordlist)
print('[DEBUG] #words in wordlist: %d' % (numwords))
#model_convcap = convcap(9221, 3, is_attention=True)
#model_convcap.cuda()
#model_convcap.train(False)
model_dis_gan = dis1.Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout).cuda()
model_dis_gan.train(False)
if True:
      print('Loading Model......')
      modelfn = osp.join('./', 'dis.pth')
      checkpoint = torch.load(modelfn, map_location='cuda:0')
    #  model_convcap.load_state_dict(checkpoint['state_dict'])
    #  model_imgcnn.load_state_dict(checkpoint['img_state_dict'])
      model_dis_gan.load_state_dict(checkpoint)
      #model_dis2.load_state_dict(checkpoint['dis2_dict'])
        #   'state_dict': model_convcap.state_dict(),
        # 'img_state_dict': model_imgcnn.state_dict(),
        # 'gan_state_dict': model_dis_gan.state_dict(),
        # 'optimizer' : optimizer.state_dict(),
        # 'img_optimizer' : img_optimizer_dict,
        # 'gan_optimize' : disc_optimizer.state_dict(),
caption = input('ur world:\n')
words = str(caption).lower().strip().split()
words[len(words)-1] = words[len(words)-1].strip('.')
words = ['<S>'] + words
num_words = min(len(words), 14)
#sentence_mask[i, :(num_words+1)] = 1
i = 0
wordclass = torch.LongTensor(1, 15).zero_()
for word_i, word in enumerate(words):
    if(word_i >= num_words):
          break
    if(word not in wordlist):
          #print('unk',word)
          word = 'UNK'
    wordclass[i, word_i] = wordlist.index(word)
wordclass = wordclass.cuda()
print(wordclass)
yp = model_dis_gan(wordclass)
print(yp)
