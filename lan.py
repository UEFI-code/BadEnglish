data = ?`
sen = ''
f = open('eval.data','r')
lines = f.readlines()
for line in lines:
  l = line.strip().split(' ')
  for s in l:
    word = data.wordlist[int(s)]
    sen = sen + word + ' '
  print(sen)
  sen = ''
