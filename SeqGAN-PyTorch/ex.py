import numpy as np
from coco_loader import coco_loader
from torch.utils.data import DataLoader
train_data = coco_loader('/coco2014', split='train', ncap_per_img=5)
train_data_loader = DataLoader(dataset=train_data, num_workers=16,batch_size=1)
for batch_idx, (imgs, captions, wordclass, mask, _) in enumerate(train_data_loader):
   ls = wordclass.numpy().tolist()
   for i in range(5):
     for j in range(15):
       open('out.txt','a').write('%d ' % ls[0][i][j])
     open('out.txt','a').write('\n')
   if batch_idx == 320:
     break
