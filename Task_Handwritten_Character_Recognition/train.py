import argparse, json, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from src.data_utils import ROOT, load_datasets
from src.model import CharacterCNN

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def evaluate(model, loader, criterion, device):
    model.eval(); total_loss=0.0; ys=[]; preds=[]
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            logits=model(x); loss=criterion(logits,y)
            total_loss += loss.item()*x.size(0)
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            ys.extend(y.cpu().numpy().tolist())
    return total_loss/len(loader.dataset), accuracy_score(ys,preds), np.asarray(ys), np.asarray(preds)

def main():
    parser=argparse.ArgumentParser(description="Train a CNN on EMNIST ByClass")
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--workers',type=int,default=0,help='Use 0 for Windows compatibility')
    args=parser.parse_args()
    random.seed(42); seed_everything()
    (ROOT/'models').mkdir(exist_ok=True); (ROOT/'results').mkdir(exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print('Loading EMNIST ByClass; the first run downloads the dataset...')
    train_set,val_set,test_set,classes=load_datasets()
    pin=device.type=='cuda'
    train_loader=DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=pin)
    val_loader=DataLoader(val_set,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=pin)
    test_loader=DataLoader(test_set,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=pin)
    model=CharacterCNN(len(classes)).to(device); criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=1,factor=0.5)
    best_val=float('inf'); patience=3; stale=0
    history={'train_loss':[],'train_accuracy':[],'val_loss':[],'val_accuracy':[]}
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.; correct=0; seen=0
        for batch_idx,(x,y) in enumerate(train_loader,1):
            x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
            logits=model(x); loss=criterion(logits,y); loss.backward(); optimizer.step()
            total += loss.item()*x.size(0); correct += (logits.argmax(1)==y).sum().item(); seen += y.size(0)
            if batch_idx%100==0: print(f'  batch {batch_idx}/{len(train_loader)}')
        tr_loss=total/seen; tr_acc=correct/seen
        val_loss,val_acc,_,_=evaluate(model,val_loader,criterion,device); scheduler.step(val_loss)
        for k,v in [('train_loss',tr_loss),('train_accuracy',tr_acc),('val_loss',val_loss),('val_accuracy',val_acc)]: history[k].append(float(v))
        print(f'Epoch {epoch}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}')
        if val_loss<best_val:
            best_val=val_loss; stale=0
            torch.save({'model_state_dict':model.state_dict(),'classes':classes,'num_classes':len(classes)},ROOT/'models'/'emnist_cnn.pth')
        else:
            stale+=1
            if stale>=patience: print('Early stopping: validation loss did not improve.'); break
    (ROOT/'results'/'training_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    checkpoint=torch.load(ROOT/'models'/'emnist_cnn.pth',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss,test_acc,y_true,y_pred=evaluate(model,test_loader,criterion,device)
    metrics={'test_loss':float(test_loss),'test_accuracy':float(test_acc),'test_samples':int(len(y_true)),'classes':classes}
    (ROOT/'results'/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    report=classification_report(y_true,y_pred,labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0)
    (ROOT/'results'/'classification_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    cm=confusion_matrix(y_true,y_pred,labels=list(range(len(classes))))
    pd.DataFrame(cm,index=classes,columns=classes).to_csv(ROOT/'results'/'confusion_matrix.csv')
    fig,ax=plt.subplots(figsize=(12,10)); im=ax.imshow(cm,interpolation='nearest',cmap='Blues')
    ax.set_title('EMNIST ByClass — Test Confusion Matrix'); ticks=np.arange(0,len(classes),4)
    ax.set_xticks(ticks,[classes[i] for i in ticks],rotation=45,ha='right'); ax.set_yticks(ticks,[classes[i] for i in ticks])
    ax.set_xlabel('Predicted label'); ax.set_ylabel('True label'); fig.colorbar(im,ax=ax); fig.tight_layout()
    fig.savefig(ROOT/'results'/'confusion_matrix.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(history['train_loss'],label='Train'); axes[0].plot(history['val_loss'],label='Validation'); axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend()
    axes[1].plot(history['train_accuracy'],label='Train'); axes[1].plot(history['val_accuracy'],label='Validation'); axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend()
    fig.tight_layout(); fig.savefig(ROOT/'results'/'training_curves.png',dpi=150); plt.close(fig)
    print('\nTEST RESULTS'); print(json.dumps(metrics,indent=2)); print('Training complete. Model: models/emnist_cnn.pth'); print('Reports and plots saved under results/')

if __name__=='__main__': main()
