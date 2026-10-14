import numpy as np,pandas as pd
from src.common import PROCESSED,discover,parse_file,mfcc,DATA_ROOT
def main():
 PROCESSED.mkdir(parents=True,exist_ok=True); X=[]; rows=[]; files=discover()
 if not files: raise FileNotFoundError(f'No WAV files in {DATA_ROOT}; see README.md for dataset setup.')
 for i,p in enumerate(files,1):
  parsed=parse_file(p)
  if not parsed: continue
  emotion,actor=parsed
  try: feat=mfcc(p)
  except Exception as e: print(f'Skip {p.name}: {e}'); continue
  split='train' if actor<=18 else 'validation' if actor<=21 else 'test' if actor<=24 else 'unused'
  X.append(feat); rows.append({'file':str(p.relative_to(DATA_ROOT)),'emotion':emotion,'actor':actor,'split':split})
  if i%100==0: print(f'Processed {i}/{len(files)} files')
 if not rows: raise RuntimeError('No valid RAVDESS files found; check folder structure.')
 np.savez_compressed(PROCESSED/'ravdess_mfcc.npz',X=np.stack(X)); meta=pd.DataFrame(rows); meta.to_csv(PROCESSED/'metadata.csv',index=False)
 print(f'Saved {len(rows)} samples. Actor split: train 01-18, validation 19-21, test 22-24.')
 print(meta.groupby(['split','emotion']).size().unstack(fill_value=0).to_string())
if __name__=='__main__': main()
