import os
import pandas as pd

# all = ['./dataset/val_1/imgs/'+img for img in os.listdir('./val_1/imgs')]
# pd.DataFrame({'img_path':all}).to_csv('./val.csv', index=False)

os.system('find ./ -name ".DS_Store" -depth -exec rm {} \;')
# combine different dataset
datasets = os.listdir('./train')

print('-----')
print('total {} datasets'.format(len(datasets)))
print('-----')

all_images = []

for dataset in datasets:
    
    cur_images = os.listdir('./train/'+dataset+'/imgs')

    cur_list = ['./dataset/train/'+dataset+'/imgs/'+img for img in cur_images]

    all_images += list(filter(lambda x:x.endswith('png') or x.endswith('jpg'), cur_list))

pd.DataFrame({'img_path':all_images}).to_csv('./train.csv', index=False)



