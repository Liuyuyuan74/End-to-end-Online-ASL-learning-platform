# ASL-Recognition
  1. asllocal folder - linked project
  2. aslml folder - the seperate ml code
  3. aslretrainml folder - the seperate ml code for retraining
  4. awsrelated - files related to hosting the code in AWS
FROM SCRATCH - BUILD MODEL AND RUN APPLICATION TO PERFORM INFERENCE ON IMAGES


SET UP 
	Using a virtual environment is recommended so as not to conflict with other existing and possibly not compatible versions of Python(between 3.8 and 3.11). Python versions are limited because of Scikit-Learn library requirements. This Scikit-Learn version (1.4.0) install is handled by the requirements.txt. If you decide not to use a virtual environment, be sure to uninstall other non-compatible versions of Python.  Also check for existing non-compatible versions of dependencies listed in the requirements.txt file. This can be a bit of a chore, so it’s best to instead use a virtual environment and let Python figure out all the acceptable versioning for you. 

1 - Download/Install usable 64bit Python version (anything between 3.8 and 3.11)

2 - From Windows command prompt install Virtualenv
–example : pip install virtualenv

3 - Create a virtualenv in your project directory
	--example : python -m virtualenv --python python310 ASL  [If using Python 3.10]
		– to see which versions of python are installed on your system: py -0
-!warning : if using windows/powershell, you may need to change your execution policy to allow scripts to run in order to activate your virtualenv
		--example : Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted -Force

4 - Activate your new virtualenv from inside your project directory
	--example : .\ASL\Scripts\activate

5 - Should see an updated command prompt showing an activated virtualenv
	--example : (ASL) PS C:\Projects\Capstone\ASL

6 - Check version of python used by virtualenv
	--example : python --version
	--output : Python 3.10.0 [or whatever your version is]

7 - CD into the virtualenv directory
	--example : CD ASL

8 - Clone github repo to local machine
	--example : git clone https://github.com/cpetrella-sketch/ASL-Recognition.git
	--output : Cloning into 'ASL-Recognition'...
			remote: Enumerating objects: 518, done.
			remote: Counting objects: 100% (88/88), done.
			remote: Compressing objects: 100% (54/54), done.
			remote: Total 518 (delta 35), reused 72 (delta 27), pack-reused 430
			Receiving objects: 100% (518/518), 40.60 MiB | 3.62 MiB/s, done.
			Resolving deltas: 100% (270/270), done.

9 - Install required python dependencies
	--Change directory: CD .\ASL-Recognition\aslml\
	--Install dependencies
	--example : pip install -r requirements.txt
	--output : ...Installing collected packages: 
		
10 - Download both Training and Testing Datasets from the below links
	–Full_Training_Dataset.zip (2.51 GB)
https://drive.google.com/file/d/1Ups86xkwbjnrWF7qNheXk4iNfLLgjvtK/view?usp=sharing
–Extract and save to ~./ASL-Recognition/aslml/data/
	– path to dir should be: ~./ASL-Recognition/aslml/images/Full_Training_Dataset/
– directory should have one sub directory for each letter in Alphabet(excluding J,Z)

–Full_Testing_Dataset.zip (38.8 MB)
https://drive.google.com/file/d/1UrN66JNtXcS-S_1kvrsH11pE3vbP3Vd-/view?usp=sharing
–Extract and save to ~./ASL-Recognition/aslml/data/
	– path to dir should be: ~./ASL-Recognition/aslml/images/Full_Testing_Dataset/
– directory should have one sub directory for each letter in Alphabet(excluding J,Z)

11 - Create landmark dataset from Full_Training_Dataset images
– example : from inside ./ASL-Recognition/aslml/
–  inside the create_dataset.py, change the “sampleSizePercentage” to your desired sample rate.  Default is set to 100% of all images.
		– python .\create_dataset.py
– output : 
Currently working on directory A…
	Currently working on directory B…
	…
	Currently working on directory V…
Currently working on directory Y...

Dataset sample size selected: 10%
Total number of images processed (10% of Full Dataset): 8033
Successful detections (79.73359890451886%): 6405
Failed detections: 1628
Landmark Detection Complete...Exporting x/y coords and labels to 'data.pickle'
Execution Time: 2184 Seconds

12 - Find best Random Forest Classifier Params and Train a model on dataset
	– example : from ~./ASL-Recognition/aslml
		– python .\train_classifier.py
	– output : 		
Splitting data into testing and training with 20.0% reserved for testing.

Starting Grid Search...
Fitting 5 folds for each of 16 candidates, totalling 80 fits
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   5.1s
[CV] END bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   5.4s
…
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   9.6s
[CV] END bootstrap=False, max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=200; total time=   9.2s
Here are the best params found:

{'bootstrap': False, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
CLASSIFICATION REPORT:

              precision    recall  f1-score   support

           S       0.74      0.95      0.83        58
           T       0.94      0.96      0.95        53
           U       0.67      0.73      0.70        56
           V       0.81      0.75      0.78        59
           W       1.00      0.96      0.98        56
           X       0.98      0.94      0.96        52
           Y       0.97      0.97      0.97        58

    accuracy                           0.91      1281
   macro avg       0.92      0.91      0.91      1281
weighted avg       0.92      0.91      0.92      1281

91.49102263856362% of samples were classified correctly

Execution Time: 103.08926582336426 Seconds


13 - Test the accuracy of newly created model on new testing data
	– example : from ~./ASL-Recognition/aslml
		– python .\InferenceTester.py
	– output : 
		Image file: hand2_a_dif_seg_2_cropped.jpeg
Inside failed inference classifier
Failed to detect landmarks in user image: hand2_a_dif_seg_2_cropped.jpeg


Image file: A0001_test.jpg
Successfully detected landmarks in user image: A0001_test.jpg

The model predicted an A
dirName is: A
CORRECT!!


Image file: A0024_test.jpg
Successfully detected landmarks in user image: A0024_test.jpg

The model predicted an A
dirName is: A
CORRECT!!
…

Image file: hand3_y_dif_seg_5_cropped.jpeg
Successfully detected landmarks in user image: hand3_y_dif_seg_5_cropped.jpeg

The model predicted an Y
dirName is: Y
CORRECT!!

Using RandomForestClassifer trained model:
Percentage Successful Landmark Detection: 69%
Percentage Successful Letter Predictions Detection: 76%

Total number of Testing Images Available: 2510
26% random sampling.
Total number of Images Processed: 622
Total number of Correct predictions: 332
Total number of Incorrect predictions: 103
Total number of Successful Landmark detections: 435
Total number of Unsuccessful Landmark detections: 187



USE APPLICATION

14 - Copy newly created model to cgi-bin 
– example : copy ‘aslModel.job’ from ‘~.\ASL-Recognition\aslml\models’ to ‘~.\ASL-Recognition\asllocal\build\models’
15 - From inside the '~.\ASL-Recognition\asllocal\build' directory, start the web server
	-- example : python -m http.server --cgi 8990
	– output : Serving HTTP on :: port 8990 (http://[::]:8990/) ...



USE APPLICATION

1 - Open a web browser and access the web page
	-- example : http://localhost:8990
2 - Upload a .jpg ASL gesture image for inference
	-- click the "Upload File" button
	-- select an image from your local storage
	-- wait for status pop up
		-- example : localhost:8990 says Upload successful
	-- click "ok"
	-- screen updates with image uploaded and inference result
		-- example : 
