import os

def get_dir(dir_to_search):
  dirs = [0]
  for root, ds, files in os.walk(dir_to_search):
    for d in ds:
      if d.isdigit():
        dirs.append(int(d))
  
  dirs.sort()
  d = os.path.join(dir_to_search, "{:02}".format(dirs[-1]+1))
  os.mkdir(d)
  return d

