#!/usr/bin/env python3
import cgi
import os
from http import cookies
import json

# The directory to save uploaded files
UPLOAD_DIR = r'S:\Program\GitHub\ASL-Recognition\aslfront\temp_store_image'

def main():
    print("Content-Type: application/json")  # JSON response
    print()  # End of headers

    # Create instance of FieldStorage
    form = cgi.FieldStorage()

    # Check if the file was uploaded
    if 'photo' in form:
        fileitem = form['photo']
        
        # Check if the file has contents
        if fileitem.filename:
            fname = os.path.basename(fileitem.filename)
            open(os.path.join(UPLOAD_DIR, fname), 'wb').write(fileitem.file.read())
            
            response = {
                'result': 'success',
                'message': f'The file "{fname}" was uploaded successfully.'
            }
        else:
            response = {
                'result': 'error',
                'message': 'No file was uploaded.'
            }
    else:
        response = {
            'result': 'error',
            'message': 'Upload field not found.'
        }

    print(json.dumps(response))

if __name__ == "__main__":
    main()
