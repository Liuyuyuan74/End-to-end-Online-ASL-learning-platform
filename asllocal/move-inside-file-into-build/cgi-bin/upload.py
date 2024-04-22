#!/usr/bin/env python
import cgi
import os
from http import cookies
import json
from asl_main import asl_main_launch
# The directory to save uploaded files

UPLOAD_DIR = r'S:\Program\GitHub\ASL-Recognition\aslfront\temp_store_image'

def process(fname):
     # Check if the file has contents
    
    # COPIED FILE. NOW RUN AGAINST MODEL
    # THIS WILL OUTPUT or PRINT the result
    return asl_main_launch(UPLOAD_DIR, fname)
        

def main():
    
    # Create instance of FieldStorage
    form = cgi.FieldStorage()

    # Check if the file was uploaded
    if 'photo' in form:
        fileitem = form['photo']
        if fileitem.filename:
            fname = os.path.basename(fileitem.filename)
            open(os.path.join(UPLOAD_DIR, fname), 'wb').write(fileitem.file.read())
   
            js = process(fname)
            return js
        else:
            return {"SuccessCode": 3, "InferResult": "None"}
    else:
        return {"SuccessCode": 3, "InferResult": "None"}


if __name__ == "__main__":
    print("Content-Type: application/json\n") 
    result = main()
    print(json.dumps(result)) 

    