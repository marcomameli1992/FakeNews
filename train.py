import os

import torch
from tqdm import tqdm, trange
from evaluate import evaluate

import neptune.new as neptune

# TODO write training function

def train(epochs, model, criterion, optimizer, train_loader, val_loader, config, tokenizer, savePath, run):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run['config/device'] = device
    best_acc = 0
    model.to(device)
    for epoch in trange(epochs, desc="Epoch"):
        model.train()
        for i, (input_ids, attention_mask, image, labels) in enumerate(tqdm(iterable=train_loader, desc='Training')):
            optimizer.zero_grad()
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            img_list = []
            for i in range(image.shape[0]):
                img_list.append(image[i, :, :, :])

            clas = model(text=input_ids, text_input_mask=attention_mask, image=img_list)
            loss = criterion(input=clas.squeeze(-1), target=labels.float())
            run['training/batch/loss'].log(loss)
            loss.backward()
            optimizer.step()
        val_acc, val_loss = evaluate(model=model, criterion=criterion, dataloader=val_loader, device=device)
        run['validation/loss'].log(val_loss)
        run['validation/accuracy'].log(val_acc)
        print("Epoch {} complete! \n Validation: \n\tAccuracy: {} \n\t  Loss: {}".format(epoch, val_acc, val_loss))
        if val_acc > best_acc:
            print("Best validation accuracy improved from {} to {}, saving model...". format(best_acc, val_acc))
            best_acc = val_acc
            model.bert.save_pretrained(save_directory=os.path.join(savePath, 'textmodel'))
            os.makedirs(os.path.join(savePath, 'visualmodel'), exist_ok=True)
            torch.save(model.vgg.state_dict(), os.path.join(savePath, 'visualmodel', 'vgg.pth'))
            os.makedirs(os.path.join(savePath, 'classificationLayer'), exist_ok=True)
            torch.save(model.classification.state_dict(), os.path.join(savePath, 'classificationLayer', 'classificationLayer.pt'))
            #model.vgg.save_pretrained(save_directory=os.path.join(savePath, 'visualmodel'))
            config.save_pretrained(save_directory=os.path.join(savePath, 'textmodel'))
            tokenizer.save_pretrained(save_directory=os.path.join(savePath, 'textmodel'))
