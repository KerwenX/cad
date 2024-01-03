import os
import math

# filename = os.listdir('roca_pc')
# print(len(filename))

filename = os.listdir('./rendered_images')
print(len(filename))
k_fold = 5

catlist = [
    '02808440',
    '02818832',
    '02747177',
    '02871439',
    '02933112',
    '03001627',
    '03211117',
    '04256520',
    '04379243'
]
filename = [name for name in filename if name.split('-')[0] in catlist]
print(len(filename))
split_len = math.ceil(len(filename)/5)
train_name = filename[:4*split_len]
test_name = filename[4*split_len+1:]
train_name = [x+'.npy\n' for x in train_name]
test_name = [x+'.npy\n' for x in test_name]

with open('train.txt','w') as f:
    f.writelines(train_name)

with open('test.txt','w') as f:
    f.writelines(test_name)
