from coco_loader import coco_loader
data = coco_loader('/mnt/data1/wch/data/cucu/coco/coco2014', split='train', ncap_per_img=1)
sen = ''
f = open('gene.data','r')
lines = f.readlines()
for line in lines:
  l = line.strip().split(' ')
  for s in l:
    word = data.wordlist[int(s)]
    sen = sen + word + ' '
  print(sen)
  sen = ''
