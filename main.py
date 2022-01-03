import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import AutoConfig, AutoTokenizer
from dataset.dataset import MyDataset
from evaluate import evaluate
from arguments import args
from train import train
from test import test

# For Logging
import neptune.new as neptune

from models.mixed_model import MixModel

if __name__ == "__main__":

    ## Start neptune
    run = neptune.init(project='FakeNews-Analysis', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZWJkNDEyYS01NjI0LTRjMDAtODI5Yi0wMzI4NWU5NDc0ZmMifQ==', source_files=['*.py'])

    # Get arguments:
    bertModelNameOrPath = args.text_model_name_or_path if args.text_model_name_or_path is not None else 'bert-base-uncased'
    vggModelPath = args.image_model_path
    lr = args.lr
    epochs = args.num_eps
    batchSize = args.batch_size
    threads = args.num_threads
    saveDir = args.save_dir
    maxLenTrain = args.maxlen_train
    maxLenVal = args.maxlen_val
    trainFile = args.train_file
    testFile = args.test_file
    valFile = args.val_file
    textColumnName = args.text_column_name
    labelColumnName = args.label_column_name
    imageColumnName = args.image_column_name
    mode = args.mode
    trainImageFolder = args.train_image_folder
    valImageFolder = args.val_image_folder
    testImageFolder = args.test_image_folder

    # Config to neptune
    run['config/model/Text Model'] = bertModelNameOrPath
    run['config/model/Visual Model'] = 'VGG16'
    run['config/hyperparameter/learning rate'] = lr
    run['config/hyperparameter/number of epochs'] = epochs
    run['config/hyperparameter/Batch'] = batchSize
    run['config/hyperparameter/Text Max Lean for Training'] = maxLenTrain
    run['config/hyperparameter/Text Max Lean for Validation'] = maxLenVal
    run['config/dataset'] = 'Fake Eddit'


    # Configurationfor the desired transformer model
    config = AutoConfig.from_pretrained(bertModelNameOrPath)

    # Tokenizer for the desired transformer model
    tokenizer = AutoTokenizer.from_pretrained(bertModelNameOrPath)

    # Model creation
    model = MixModel(config)

    # Takes as the input the logits of the positive class and computes the binary cross-entropy
    criterion = nn.BCEWithLogitsLoss()

    run['config/criterion'] = type(criterion).__name__

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    run['config/optimizers'] = type(optimizer).__name__

    trainSet = MyDataset(filename=trainFile, maxlen=maxLenTrain, tokenizer=tokenizer,
                         dataColumnName=textColumnName, dataColumnLabel=labelColumnName, dataColumnImage=imageColumnName, delimiter=',', imageFolder=trainImageFolder)
    valSet = MyDataset(filename=valFile, maxlen=maxLenVal, tokenizer=tokenizer,
                       dataColumnName=textColumnName, dataColumnLabel=labelColumnName, dataColumnImage=imageColumnName, delimiter=',', imageFolder=valImageFolder)

    train_loader = DataLoader(dataset=valSet, batch_size=batchSize, num_workers=threads)
    val_loader = DataLoader(dataset=valSet, batch_size=batchSize, num_workers=threads)


    os.makedirs(saveDir, exist_ok=True)

    if mode == 'train':
        train(epochs=epochs, model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader,
              val_loader=val_loader, config=config, tokenizer=tokenizer, savePath=saveDir, run=run)
    if mode == 'test':
        testSet = MyDataset(filename=testFile, maxlen=maxLenTrain, tokenizer=tokenizer,
                         dataColumnName=textColumnName, dataColumnLabel=labelColumnName, dataColumnImage=imageColumnName, delimiter=',', imageFolder=trainImageFolder)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #model.classification.load_state_dict(torch.load(os.path.join(saveDir, 'classificationLayer', 'classificationLayer.pth')))
        model = MixModel(config, classification_path=os.path.join(saveDir, 'classificationLayer', 'classificationLayer.pth'))
        test(model, criterion, val_loader, device, run)

    run.stop()