import os


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


class ImageFeature:
    def __init__(self, name, features=None):
        self.name = name
        self.features = features

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


from exif import Image
from os import listdir
from os.path import isfile, join
#Select a path
mypath = "D:\Python\Es_Neo4j\database\\0.jpg"
#get each file from the path
files = [mypath]
image_info = {}
#iterate each image
for image_path in files:
    #read image
    with open(image_path, 'rb') as image_file:
        print("\n"+image_path+"\n")
        #transform into exif image format
        current_image = Image(image_file)
        #create my temp db
        image_info[current_image] = {}
        #check if it has a exif
        if(current_image.has_exif):
            #check each stat that the exif contains
            for stat in current_image.list_all():
                # try to open each stat and save it into my temp db
                try:
                    func = eval("current_image."+stat)
                    image_info[current_image][stat] = func
                except Exception as e:
                    print("error of " + stat + " cause: " + str(e) + "\n")
        else:
            continue
        # print results
        print(image_info[current_image])