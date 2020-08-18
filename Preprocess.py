import os
import numpy as np # linear algebra

labels=[0,1,2,3,4]
originPath='..\\aptos2019-blindness-detection\\'

def createDirectories():
    for i in labels:
        os.makedirs('..\\train\\'+str(i))


    for i in labels:
        os.makedirs('..\\test\\'+str(i))

def deleteDirectories():
    for i in labels:
        os.rmdir('..\\train\\'+str(i))


    for i in labels:
        os.rmdir('..\\test\\'+str(i))    

'''
Se crean los directorios
Si ya existen, se borran
y se vuelven a crear
'''
try:
    createDirectories()
except Exception as e:
    deleteDirectories()
    createDirectories()



print(os.listdir(originPath))


'''
for row in csv_reader:
	os.system('cp '+originPath+str(row[0])+'.png ./train/'+str(row[1])+'/'+str(row[0])+'.png')
'''

import csv
train = {}
for i in range(5):
    train[i] = []
    
prueba = []
with open(originPath+'train.csv', mode='r') as csv_file:
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
'''
print(len(train[0]))
print(len(train[1]), len(train[0])/len(train[1]))
print(len(train[2]), len(train[0])/len(train[2]))
print(len(train[3]), len(train[0])/len(train[3]))
print(len(train[4]), len(train[0])/len(train[4]))
'''

#balanceo
import itertools
values = [1,5,2,9,6]
for i in range(len(values)):
    #print(values[i])
    temp = list(itertools.repeat(train[i], values[i]))
    resul = []
    for j in temp:
        resul = resul + j

    train[i] = resul

## Funcion de arreglos
test = []
trainArray = []
testLabels = []
trainLabels = []
for i in train:
    #print(i)
    a = train[i]
    v = (len(a)* 70) // 100
    #print(len(a) - v)
    trainArray += a[:v]
    test += a[v:]
    #print(len(test))
    testLabels += (np.ones(len(a) - v) * i).tolist()
    trainLabels += (np.ones(v) * i).tolist() 


print("train")
print(len(trainArray))
print(len(trainLabels))
print()
print("test")
print(len(test))
print(len(testLabels))
print()

# Copiamos todas las imagenes de train
trainOriginPath=originPath+'train_images\\'

print('Copiando imagenes train...')
for imgName,label in zip(trainArray,trainLabels):
    print(imgName)
    print(label)
    comando='copy "'+trainOriginPath+imgName+'.png" "..\\train\\'+str(int(label))+'\\'+imgName+'.png"'
    print(comando)
    os.system(comando)
    
print('Copiando imagenes test...')
for imgName,label in zip(test,testLabels):
    print(imgName)
    print(label)
    comando='copy "'+trainOriginPath+imgName+'.png" "..\\test\\'+str(int(label))+'\\'+imgName+'.png"'
    print(comando)
    os.system(comando)
    
