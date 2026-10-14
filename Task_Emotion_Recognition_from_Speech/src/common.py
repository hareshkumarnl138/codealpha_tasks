from pathlib import Path
import numpy as np, librosa
ROOT=Path(__file__).resolve().parents[1]
DATA_ROOT=ROOT/'data'/'raw'; PROCESSED=ROOT/'data'/'processed'; MODELS=ROOT/'models'; RESULTS=ROOT/'results'; FIGURES=RESULTS/'figures'; TABLES=RESULTS/'tables'
EMOTION_MAP={'01':'neutral','02':'calm','03':'happy','04':'sad','05':'angry','06':'fearful','07':'disgust','08':'surprised'}
EMOTIONS=[EMOTION_MAP[k] for k in sorted(EMOTION_MAP)]
def parse_file(path):
 p=path.stem.split('-')
 if len(p)!=7 or p[2] not in EMOTION_MAP or not path.parent.name.startswith('Actor_'): return None
 try: actor=int(path.parent.name.split('_')[1])
 except Exception: return None
 return EMOTION_MAP[p[2]],actor
def discover(): return sorted(DATA_ROOT.rglob('*.wav'))
def mfcc(path,sr=22050,duration=3.0,n_mfcc=40,frames=130):
 y,_=librosa.load(str(path),sr=sr,mono=True,duration=duration); n=int(sr*duration)
 if len(y)<n: y=np.pad(y,(0,n-len(y)))
 else: y=y[:n]
 x=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=n_mfcc,n_fft=2048,hop_length=512)
 if x.shape[1]<frames: x=np.pad(x,((0,0),(0,frames-x.shape[1])))
 else: x=x[:,:frames]
 x=(x-x.mean(axis=1,keepdims=True))/(x.std(axis=1,keepdims=True)+1e-6)
 return x.T.astype(np.float32)
