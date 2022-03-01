from PIL import Image
import os, sys

path = ""
output_path = ""

dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path + item):
            im = Image.open(path + item)
            imResize = im.resize((112,112), Image.ANTIALIAS)
            print(f'saving image: {output_path + item}')
            imResize.save(output_path + item, 'JPEG', quality=90)


def write_paths():
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [join(output_path, f) for f in listdir(output_path) if isfile(join(output_path, f))]

    textfile = open("fusion_swagan_e4e.txt", "w")
    for element in onlyfiles:
        textfile.write(element + "\n")
    textfile.close()


# resize()
# write_paths()