# Facial-Emotion-Recognition

This is a Facial Emotion Recognition program that works real time . I decided to create this program to further expand my knowledge on CNN and face detection . I didn't code the face detection part from scratch but the CNN model architecture was completely my own using Keras Python Library . This model achieves 61% accuracy which is not very good but also not bad . I think there is room for improvement by modifiying the architecture a bit or by using Transfer Learning.

To launch this program you have to options :
either you want to Train the model and try it in case you want to tweak it a bit or change :
just use this cmd command:
```console
python webcam.py haarcascade_frontalface_default.xml data train
```
if you just want to use the model fruits.h5 and try the program just use the command:
```console
python webcam.py haarcascade_frontalface_default.xml data test
```
