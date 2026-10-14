import argparse,torch,numpy as np
from pathlib import Path
from src.common import MODELS,mfcc
from src._model import CNNLSTM
def main():
 p=argparse.ArgumentParser(description='Predict an emotion from a WAV file');p.add_argument('audio');a=p.parse_args();path=Path(a.audio)
 if not path.is_file(): raise FileNotFoundError(path)
 f=MODELS/'emotion_cnn_lstm.pt'
 if not f.exists(): raise FileNotFoundError('Train first with python -m src.03_train_model')
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');ck=torch.load(f,map_location=dev,weights_only=False);classes=ck['classes'];model=CNNLSTM(len(classes)).to(dev);model.load_state_dict(ck['model_state_dict']);model.eval()
 with torch.no_grad(): probs=torch.softmax(model(torch.tensor(mfcc(path)[None,...],dtype=torch.float32).to(dev)),1)[0].cpu().numpy()
 order=np.argsort(probs)[::-1];print(f'Audio: {path.name}\nPredicted emotion: {classes[order[0]].title()}\nConfidence: {probs[order[0]]:.1%}\n');[print(f'{classes[i]:10s} {probs[i]:.1%}') for i in order]
if __name__=='__main__': main()
