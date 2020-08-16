import os
import numpy as np # linear algebra

labels=[0,1,2,3,4]

for i in labels:
    os.makedirs('./train/'+str(i))


for i in labels:
    os.makedirs('./test/'+str(i))


originPath='aptos2019-blindness-detection/train_images/'
for row in csv_reader:
	os.system('cp '+originPath+str(row[0])+'.png ./train/'+str(row[1])+'/'+str(row[0])+'.png')


import csv
train = {}
for i in range(5):
    train[i] = []
    
prueba = []
with open('/kaggle/input/aptos2019-blindness-detection/train.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
        #print(row['diagnosis'],row['id_code'])
        #prueba.append(int(row['diagnosis']) )
        train[ int(row['diagnosis']) ].append(row['id_code'])
        line_count += 1
    
    print(f'Processed {line_count} lines.')

#print(train)

print(len(train[0]))
print(len(train[1]), len(train[0])/len(train[1]))
print(len(train[2]), len(train[0])/len(train[2]))
print(len(train[3]), len(train[0])/len(train[3]))
print(len(train[4]), len(train[0])/len(train[4]))

#balanceo
import itertools
values = [1,5,2,9,6]
for i in range(len(values)):
    print(values[i])
    temp = list(itertools.repeat(train[i], values[i]))
    resul = []
    for j in temp:
        resul = resul + j
    train[i] = resul

## Funcion de arreglos
test = []
train = []
testLabels = []
trainLabels = []
for i in train:
    a = train[i]
    v = (len(a)* 70) // 100
    train += len(a[:v])
    test += len(a[v:])
    testLabels += np.ones(len(a) - v).tolist() * i
    trainLabels += np.ones(v).tolist() * i
#print(test)