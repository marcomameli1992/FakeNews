import torch
from tqdm import tqdm, trange
from evaluate import evaluate
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt

def test(model, criterion, dataloader, device, run):
    run['test-config/device'] = device
    model.to(device)
    model.eval()
    val_acc, val_loss = evaluate(model=model, criterion=criterion, dataloader=dataloader, device=device)
    run['test/loss'] = val_loss
    run['test/accuracy'] = val_acc
    print("Test complete! \n Test results: \n\tAccuracy: {} \n\t  Loss: {}".format(val_acc, val_loss))
    print("Creating confusion matrix...")
    y_pred = []
    y_true = []

    for i, (input_ids, attention_mask, image, labels) in enumerate(tqdm(iterable=dataloader, desc='Testing')):
        input_ids, attention_mask, image, labels = input_ids.to(device), attention_mask.to(device), image.to(device, dtype=torch.float), labels.to(device)
        predicted_class = model(text=input_ids, text_input_mask=attention_mask, image=image)
        predicted_output = predicted_class.data.to('cpu').numpy()
        desired_output = (torch.max(torch.exp(labels), 1)[1]).data.cpu().numpy()#labels.data.to('cpu').numpy()

        #output = (predicted_output > 0.5).astype(int)
        output = (torch.max(torch.exp(predicted_class), 1)[1]).data.cpu().numpy()

        y_pred.extend(output)

        y_true.extend(desired_output)

    classes = ('News', 'Fake News')
    cf_matrix = confusion_matrix(y_true, y_pred, normalize='true')

    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index=[i for i in classes], columns=[i for i in classes])

    plt.figure()
    sn.heatmap(df_cm, annot=True)
    plt.savefig('./confusion_matrix_normalized.png')