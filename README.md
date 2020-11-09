# Facial-Emotion-Recognition

This is a Facial Emotion Recognition program that works real time . I decided to create this program to further expand my knowledge on CNN and face detection . I didn't code the face detection part from scratch but the CNN model architecture was completely my own using Keras Python Library . This model achieves 61% accuracy which is not very good but also not bad . I think there is room for improvement by modifiying the architecture a bit or by using Transfer Learning.

The data I used to train this model is the FER-2013 you can downalod it here <a href='https://drive.google.com/file/d/1X60B-uR3NtqPd4oosdotpbDgy8KOfUdr/view'> fer-2013 </a>. And you can also use your own data to train the model by specifyin the folder containing the images.

To launch this program you have to options :
either you want to Train the model and try it in case you want to tweak it a bit or change :
just use this cmd command:
```console
python webcam.py haarcascade_frontalface_default.xml "path to the folder containing the data" train
```
if you just want to use the model fruits.h5 and try the program just use the command:
```console
python webcam.py haarcascade_frontalface_default.xml "path to the folder containing the data" test
```
