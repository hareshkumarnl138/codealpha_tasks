import argparse
from pathlib import Path
import torch
from PIL import Image
from src.data_utils import ROOT, image_to_tensor
from src.model import CharacterCNN

def main():
    parser=argparse.ArgumentParser(description='Predict one handwritten character image')
    parser.add_argument('image',help='Path to PNG/JPG containing one character')
    args=parser.parse_args(); image_path=Path(args.image)
    if not image_path.exists(): raise FileNotFoundError(f'Image not found: {image_path}')
    path=ROOT/'models'/'emnist_cnn.pth'
    if not path.exists(): raise FileNotFoundError('Model missing. Run python train.py first.')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt=torch.load(path,map_location=device,weights_only=False); classes=ckpt['classes']
    model=CharacterCNN(len(classes)).to(device); model.load_state_dict(ckpt['model_state_dict']); model.eval()
    x=image_to_tensor(Image.open(image_path)).to(device)
    with torch.no_grad(): probs=torch.softmax(model(x),dim=1)[0]; values,indices=torch.topk(probs,k=5)
    print(f'Image: {image_path}\nPrediction: {classes[indices[0].item()]}\nModel score: {values[0].item()*100:.2f}%\nTop 5:')
    for p,i in zip(values,indices): print(f'  {classes[i.item()]}: {p.item()*100:.2f}%')
if __name__=='__main__': main()
