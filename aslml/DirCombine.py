'''
DIRCOMBINE.PY

USED THE COMBINE TESTING AND TRAINING SETS INTO ONE SET OF A TO Y DIRECTORIES

'''

# Import os & shutil module
import os
import shutil

# TRAINING SETS
# directory1 = r'C:\Users\corey\PycharmProjects\ASL1\venv\images\ASL_Train'
# directory2 = r'C:\Users\corey\PycharmProjects\ASL1\venv\images\ASL_Sign'
#
# new_directory_name = r"C:\Users\corey\PycharmProjects\ASL1\venv\images\Full_Training_Dataset"

# TESTING SETS
directory1 = r'C:\Users\corey\PycharmProjects\ASL1\venv\images\ASL_Alphabet_Test'
directory2 = r'C:\Users\corey\PycharmProjects\ASL1\venv\images\Ayush_set'
directory3 = r'C:\Users\corey\PycharmProjects\ASL1\venv\images\DIY_signs'

new_directory_name = r"C:\Users\corey\PycharmProjects\ASL1\venv\images\Full_Testing_Dataset"

#define a function & pass dst. directory and src. directories
def merge_directories(new_directory_name, *directories_to_merge):
    if not os.path.exists(new_directory_name):
        os.makedirs(new_directory_name) #create a dst. directory if not exist

    for directory in directories_to_merge:
        print(f"working on directory {directory}\n")
        for item in os.listdir(directory):  #iterate sub-directory from source folders
            #join path of folder and sub-folder
            s = os.path.join(directory, item)
            d = os.path.join(new_directory_name, item)
            if os.path.isdir(s):
                if item in os.listdir(new_directory_name):
                    files = os.listdir(s)
                    for file in files:  #iterate file from sub-folder
                        j = os.path.join(s, file)
                        k = os.path.join(d, file)
                        shutil.copy2(j,k)  #paste file in already existed sub-directory
                else:
                    shutil.copytree(s, d)  #create a sub-directory in dst directory then paste file
            else:
                shutil.copy2(s, d)  #paste file in already existed sub-directory

def main():
    # merge_directories(new_directory_name,directory1,directory2)
    merge_directories(new_directory_name,directory3)

    totalImage = 0
    for i in os.listdir(new_directory_name):
        dirImgCount = len(os.listdir(os.path.join(new_directory_name,i)))
        totalImage += dirImgCount
    print(f"Total number of images in the combined set is: {totalImage}")



if __name__=="__main__":
    main()
