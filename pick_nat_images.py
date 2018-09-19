# author : Ashwin de Silva
# pick base natural images

# pick 5 categories and take 10 images out of each category and create a base image set of 50 images

# import libraries
from coco import COCO
from shutil import copyfile

# define useful params
IMGS_PER_CAT = 10
DATA_PATH = '/home/563/ls1729/gdata/phase_unwrapping/dataset/coco/val2017/'
SAVE_PATH = '/home/563/ls1729/gdata/phase_unwrapping/dataset/coco/orig/images_2'

# get the annotation file path
annFile='/home/563/ls1729/gdata/phase_unwrapping/dataset/coco/annotations/instances_val2017.json'

# initialize COCO api for instance annotations
coco = COCO(annFile)

# load the categories
cats = coco.loadCats(coco.getCatIds())
cat_names = ['cat','dog','cake','apple','phone']

# pick images from each category
for i in range(len(cat_names)):
    catIds = coco.getCatIds(catNms=[cat_names[i]])
    imgIds = coco.getImgIds(catIds=catIds)
    print('loading...',cat_names[i])
    for j in range(IMGS_PER_CAT):
        img = coco.loadImgs(imgIds[j])[0]
        src = DATA_PATH + img['file_name']
        dst = SAVE_PATH
        copyfile(src, dst)

print('Complete!')





