import os.path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.io import imread
from skimage.transform import resize
# For the dimension warning to deactivate it
import PIL

PIL.Image.MAX_IMAGE_PIXELS = 933120000

class MyDataset(Dataset):
    def __init__(self, filename, maxlen, tokenizer, imageFolder,
                 dataColumnName='sentence', dataColumnLabel='label', delimiter='\t',
                 dataColumnImage='image_id'):
        '''
        Init method for the dataset class
        :param filename: the path to the file
        :param maxlen: the maximum length of a input data
        :param tokenizer: the tokenizer algorithm
        :param dataColumnName: the column name where is the data
        :param dataColumnLabel: the column name where is the label of the data
        '''
        # Opening dataset
        self.df = pd.read_csv(filename, delimiter=delimiter)
        # Inizialize tokenizer
        self.tokenizer = tokenizer
        # Maximum length of the token list to keep all the sequence of fixed size
        self.maxlen = maxlen
        # define column:
        self.columnName = dataColumnName
        self.columnLabel = dataColumnLabel
        self.columnImage = dataColumnImage
        self.imageFolder = imageFolder
        self.transform = transforms.ToTensor()


    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        # Select the sentence and the label at the specified index in the data frame opened in the init
        sentence = self.df.loc[item, self.columnName]
        label = self.df.loc[item, self.columnLabel]
        imageID = self.df.loc[item, self.columnImage]

        # preprocess the text input to be usable in the transformer
        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[:self.maxlen - 1] + ['[SEP]']

        # Obtain the indices of the tokens in the BERT Vocabulary
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids)
        # Obtain the attention mask i.e. a tensor containing 1s for no padded tokens and 0s for padded ones
        attention_mask = (input_ids != 0).long()

        # Image data
        #print(imageID)
        image = imread(os.path.join(self.imageFolder, imageID + '.jpg'))#PIL.Image.open(os.path.join(self.imageFolder, imageID + '.jpg'))#imread(os.path.join(self.imageFolder, imageID + '.jpg'))
        #image = torch.from_numpy(image)
        image = resize(image, (224, 224, 3), anti_aliasing=True)#image.resize((224,224))#resize(image, (224, 224, 3), anti_aliasing=True)
        image = image.transpose((2, 0, 1))
        image = self.transform(image)#torch.from_numpy(image.transpose((2, 0, 1)))

        # label data
        base_label = [0, 0]
        if label == 0:
            base_label = [1, 0]
        elif label == 1:
            base_label = [0, 1]

        label = torch.tensor(base_label)

        return input_ids, attention_mask, image, label