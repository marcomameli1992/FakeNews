import pandas as pd
import os
import glob

dataset = pd.read_csv('./data/validation.csv', delimiter=',')
dataset.set_index('image_id')

new_Dataset = pd.DataFrame(columns=['text', 'image_url', 'image_id', 'label'])

list_file = glob.glob('./data/validation/*.jpg')

for value in list_file:
    id = value.split('/')[-1].split('.')[0]
    new_Dataset = new_Dataset.append(dataset.loc[dataset['image_id'] == id])


new_Dataset.to_csv('./data/validation1.csv', index=False)