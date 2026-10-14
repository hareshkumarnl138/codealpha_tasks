from collections import Counter
from src.common import DATA_ROOT,discover,parse_file,EMOTIONS
def main():
 files=discover()
 if not files: print(f'No WAV files found under {DATA_ROOT}. Extract RAVDESS speech Actor_01 ... Actor_24 folders there.'); return
 counts=Counter(); actors=set(); valid=0
 for p in files:
  x=parse_file(p)
  if x: counts[x[0]]+=1; actors.add(x[1]); valid+=1
 print(f'Audio files: {len(files)} | Valid RAVDESS files: {valid} | Actors: {len(actors)}')
 for e in EMOTIONS: print(f'{e:10s} {counts[e]}')
if __name__=='__main__': main()
