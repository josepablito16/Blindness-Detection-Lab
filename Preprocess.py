import os

labels=[0,1,2,3,4]

for i in labels:
    os.makedirs('./train/'+str(i))


for i in labels:
    os.makedirs('./test/'+str(i))


originPath='aptos2019-blindness-detection/train_images/'
for row in csv_reader:
	os.system('cp '+originPath+str(row[0])+'.png ./train/'+str(row[1])+'/'+str(row[0])+'.png')