import json,numpy as np,pandas as pd,torch
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,classification_report,confusion_matrix
from src.common import PROCESSED,MODELS,TABLES,FIGURES,EMOTIONS
from src._model import CNNLSTM
class DS(Dataset):
 def __init__(self,x,y): self.x=torch.tensor(x,dtype=torch.float32); self.y=torch.tensor(y,dtype=torch.long)
 def __len__(self): return len(self.y)
 def __getitem__(self,i): return self.x[i],self.y[i]
def predict_eval(model,loader,device,criterion=None):
 model.eval(); losses=[]; yt=[]; yp=[]
 with torch.no_grad():
  for x,y in loader:
   x,y=x.to(device),y.to(device); z=model(x)
   if criterion: losses.append(criterion(z,y).item())
   yt.extend(y.cpu().numpy()); yp.extend(z.argmax(1).cpu().numpy())
 return (float(np.mean(losses)) if losses else None),accuracy_score(yt,yp),np.array(yt),np.array(yp)
def main():
 np.random.seed(42);torch.manual_seed(42); f=PROCESSED/'ravdess_mfcc.npz'; m=PROCESSED/'metadata.csv'
 if not f.exists() or not m.exists(): raise FileNotFoundError('Run python -m src.02_prepare_data first.')
 X=np.load(f)['X'];meta=pd.read_csv(m); enc=LabelEncoder().fit(EMOTIONS); y=enc.transform(meta.emotion)
 ids={s:np.where(meta.split.to_numpy()==s)[0] for s in ['train','validation','test']}
 for s,v in ids.items():
  if not len(v): raise ValueError(f'Empty {s} split; check actor folders.')
 loaders={s:DataLoader(DS(X[v],y[v]),batch_size=32,shuffle=(s=='train')) for s,v in ids.items()}
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=CNNLSTM(len(enc.classes_)).to(device);criterion=torch.nn.CrossEntropyLoss();opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
 MODELS.mkdir(exist_ok=True);TABLES.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True);hist=[];best=-1;patience=0
 for epoch in range(1,41):
  model.train(); losses=[]
  for x,yb in loaders['train']:
   x,yb=x.to(device),yb.to(device);opt.zero_grad();loss=criterion(model(x),yb);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();losses.append(loss.item())
  vl,va,_,_=predict_eval(model,loaders['validation'],device,criterion);tl=float(np.mean(losses));hist.append({'epoch':epoch,'train_loss':tl,'validation_loss':vl,'validation_accuracy':va});print(f'Epoch {epoch:02d}/40 train_loss={tl:.4f} val_loss={vl:.4f} val_accuracy={va:.4f}')
  if va>best:
   best=va;patience=0;torch.save({'model_state_dict':model.state_dict(),'classes':enc.classes_.tolist()},MODELS/'emotion_cnn_lstm.pt')
  else:
   patience+=1
   if patience>=8: print('Early stopping.');break
 pd.DataFrame(hist).to_csv(TABLES/'training_history.csv',index=False)
 ck=torch.load(MODELS/'emotion_cnn_lstm.pt',map_location=device,weights_only=False);model.load_state_dict(ck['model_state_dict']);_,acc,yt,yp=predict_eval(model,loaders['test'],device)
 p,r,f1,_=precision_recall_fscore_support(yt,yp,average='weighted',zero_division=0);metrics={'test_accuracy':float(acc),'weighted_precision':float(p),'weighted_recall':float(r),'weighted_f1':float(f1),'test_samples':int(len(yt)),'device':str(device),'classes':enc.classes_.tolist(),'split':'actors 01-18 train, 19-21 validation, 22-24 test'}
 (TABLES/'test_metrics.json').write_text(json.dumps(metrics,indent=2));pd.DataFrame(classification_report(yt,yp,labels=np.arange(len(enc.classes_)),target_names=enc.classes_,output_dict=True,zero_division=0)).transpose().to_csv(TABLES/'classification_report.csv')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt;import seaborn as sns
 cm=confusion_matrix(yt,yp,labels=np.arange(len(enc.classes_)));plt.figure(figsize=(9,7));sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=enc.classes_,yticklabels=enc.classes_);plt.xlabel('Predicted');plt.ylabel('Actual');plt.title('Actor-Independent Test Confusion Matrix');plt.tight_layout();plt.savefig(FIGURES/'confusion_matrix.png',dpi=160);plt.close()
 h=pd.DataFrame(hist);plt.figure(figsize=(8,5));plt.plot(h.epoch,h.train_loss,label='Training loss');plt.plot(h.epoch,h.validation_loss,label='Validation loss');plt.xlabel('Epoch');plt.ylabel('Loss');plt.legend();plt.tight_layout();plt.savefig(FIGURES/'loss_curve.png',dpi=160);plt.close()
 print('\nHeld-out test metrics (real dataset run):');print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
