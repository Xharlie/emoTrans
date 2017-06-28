import shutil
import re
import os
from PIL import Image

HOME = os.path.expanduser("~")
source_directory = HOME + "/datasets/BosphorusDB"
out_dirctory = HOME + "/datasets/BosphorusDB_extracted"
SIZE = 182

def main():
  if not os.path.exists(out_dirctory):
    os.mkdir(out_dirctory)
  for dir_name in os.listdir(source_directory):
    if (not os.path.isdir(os.path.join(source_directory,dir_name))) or dir_name.find("bs") == -1:
      continue
    person_dest_dir = os.path.join(out_dirctory, dir_name)
    person_src_dir = os.path.join(source_directory, dir_name)
    if not os.path.exists(person_dest_dir):
      os.mkdir(person_dest_dir)
    for fileName in os.listdir(person_src_dir):
      pat = re.compile('.*(\_LFAU\_|\_CAU\_|\_UFAU\_|\_N\_N\_|\_E\_).*')
      if (os.path.isdir(os.path.join(person_src_dir,fileName))) \
          or re.match(pat,fileName) is None or fileName.find("png") == -1:
        continue
      # simpleCopy(person_src_dir, person_dest_dir, fileName)
      padding_squarize(person_src_dir, person_dest_dir, fileName)

def simpleCopy(person_src_dir, person_dest_dir, fileName):
  shutil.copy(os.path.join(person_src_dir, fileName), os.path.join(person_dest_dir, fileName))

def padding_squarize(person_src_dir, person_dest_dir, fileName):
  img = Image.open(os.path.join(person_src_dir, fileName))
  size = (max(img.size),) * 2
  layer = Image.new('RGB', size, img.convert('RGB').getpixel((0,max(img.size)/5*4)))
  layer.paste(img, tuple(map(lambda x: (x[0] - x[1]) / 2, zip(size, img.size))))
  layer = layer.resize((SIZE, SIZE), resample=Image.ANTIALIAS)
  layer.save(os.path.join(person_dest_dir, fileName))
if __name__ == "__main__":
  main();