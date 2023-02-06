import os
from PIL import Image

cwd = os.getcwd()
img1 = Image.open(os.path.join(cwd, '255_mdb252_01.bmp'))
img2 = Image.open(os.path.join(cwd, '273_mdb270_01.bmp'))

grid = Image.new('RGB', size=(128 + 128 + 200, 128 + 100), color=(255, 255, 255))
grid.paste(img1, box=[50, 50])
grid.paste(img2, box=[128 + 150, 50])

grid.save(os.path.join(cwd, 'grid.png'))