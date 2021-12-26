import os

translate = {
	"cane": "dog",
	"cavallo": "horse", 
	"elefante": "elephant", 
	"farfalla": "butterfly", 
	"gallina": "chicken", 
	"gatto": "cat", 
	"mucca": "cow", 
	"pecora": "sheep", 
	"scoiattolo": "squirrel",
	"ragno" : "spider"
}

dirnames = os.listdir('raw-img')

for dirname in dirnames:
	os.rename(os.path.join('raw-img', dirname), os.path.join('raw-img', translate[dirname]))