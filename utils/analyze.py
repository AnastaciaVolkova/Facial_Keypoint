import sys
import numpy as np


def main():
  with open(sys.argv[1]) as fid:
    lines = fid.read().splitlines()
  
  epochs = []
  for line in lines:
    tokens = line.split(",")
    d = dict()
    for tok in tokens:
      key, val = tok.split(":")
      d[key.strip()] = float(val)
    epochs.append(d)
  prev = -1
  ar = []

  for e in epochs:
    if e['Epoch'] != prev:
      if prev != -1:
        ar.append(r)
      r = []
      prev = e['Epoch']

    r.append(e['Avg. Loss'])
  ar.append(r)
  ar = np.array(ar)
  m = ar.mean(axis=1)
  print(m)
    

if __name__ == '__main__':
  main()

