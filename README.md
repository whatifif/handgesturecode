[Home](/README.md) | [Edit Guide](/editguide.md) | <button class="nav" ><a href="https://github.com/whatifif/handgesture/">Project Home on Github</a></button>  |  <button class="nav" ><a href="https://whatifif.github.io/handgesture/">Project Home on Web</a></button>


project name: Controlling a Computer by Hand Gesture  
project homepage: [https://github.com/whatifif/handgesture](https://github.com/whatifif/handgesture)  
project code page: [https://github.com/whatifif/handgesturecode](https://github.com/whatifif/handgesturecode)  
project slack: [https://sml109.slack.com](https://sml109.slack.com)  
project team name: Team Echo  [https://www.meetup.com/Sydney-Machine-Learning/](https://www.meetup.com/Sydney-Machine-Learning/)


# Code for the project "Controlling a Computer by Hand Gesture"


## Brief Introduction of the project

Almost people use a desktop or laptop these days. One of serious problem is that we are stuck to the keyboard and mouse, which will cause a serious health problem on a long run. Moreover in VR/AR age, we cannot use keyboard/mouse. Our purpose is to replace a keyboard and mouse with hand gestures. We have devised a virtual keyboard and virtual mouse with subtle hand gestures and made ML recognise our gestures so that we can control our computer remotely. Amazon echo has ear now. It will have eye in future. We need to make a standard gestures for people to adopt easily like a standard keyboard and mouse. ML will address this for us, human. We used Deep Learning as ML in this project.

## Dependencies
- MxNet 0.11.0
- Numpy 1.13.1
- Pandas 0.20.3
- Opencv 3.2.0
- Python2.7
- Jupyter Notebook

Best to install Anaconda2 :) 

## Main program

#### How to run 
```
jupyter notebook
```
and run the main.ipynb

#### How to use

- Press g to enter GUI mode to capture a hand image
- Press d to enter demo mode
- Press d and Press c to enter calculator mode
- Press m to add mouse mode
- Press esc to quit program

## Training the MxNet Model

#### Model is trained by 200x200 pixel images and 64x64 pixel images

200x200 pixel data caused the out of memory problem on NVdia GTX 960 ( 2GB Graphic memory).  
So 64x64 version of data and program is prepared.

Since model file is somewhat large (~200MB), it cannot be uploaded to github.
When the proj-train.ipynb is run, models folder automaticall created and can be used as model.

#### How to preprocess

run preprocessing-v2.ipynb

This will create hand_pic, mask_pic, and picv ( if mouseMode is True ) in each data folder and be used for training.

#### How to train

run proj-train.ipynb.  

It will read the preprocessed hand_pic, mask_pic and train the model. 400 epochs are repeated.  

The trained model is saved in models folder automatically.

#### How to connect the trained model to application

First export your test-streamline-64x64.ipynb as test64x64.py.  

test64x64.py will read the trained model from models folder.    

So import 'test64x64.py' to your application and use 'predict' function in your application.

test-streamline-64x64.ipynb is 64x64 version of test-streamline-v1.ipynb  









