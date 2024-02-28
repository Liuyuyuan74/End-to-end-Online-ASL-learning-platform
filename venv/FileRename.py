'''
FILERENAME.PY

USED THE CHANGE THE DIR NAMES OF A DIRECTORY TO UPPER CASE, BEFORE USING THE DIRCOMBINE.PY FILE TO MERGE ALL DIRS


'''

import os


def main():
    folder = r'C:\Users\corey\PycharmProjects\ASL1\venv\images\Ayush_set'

    for filename in os.listdir(folder):
        dst = f"{filename.upper()}"
        src = f"{folder}/{filename}"
        dst = f"{folder}/{dst}"
        os.rename(src, dst)

if __name__ == '__main__':
    main()