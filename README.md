# Mango-grade-classification-demo
## Environment
```
python :     3.7.6
tensorflow : 1.14.0
keras :      2.3.1
IDE :        Spyder
```
## Install
```
pip install TensorFlow-gpu==1.14.0
pip install keras==2.3.1
pip install mrcnn==0.2
pip install PyQt5==5.9.2
pip install scikit-image==0.18.1
pip install pandas
pip uninstall protobuf
pip install protobuf==3.20.*
pip install opencv-python==4.5.1.48
```
### Model
`Please use Right-click and save link as a file`
| Model Name           | Download Link     |
|:--------------------:| :---------------: |
| SPP_AlexNet          | [Link](http://ncyusclab.synology.me/mango/model/SPP_AlexNet.zip)|
| shapes20200813T1158  | [Link](http://ncyusclab.synology.me/mango/model/shapes20200813T1158.zip)|

## Excute
1. Download the "SPP_AlexNet" and "shapes20200813T1158" models and place them to "logs" folder
2. Open Spyder
3. Excute "Mango_sys.py"
4. Click "choose file" and choose test image folder
5. Click "check", detect the mango grade and generate a txt to record result

### Demo
[![IMAGE ALT TEXT](http://img.youtube.com/vi/g52fmQ8ifak/0.jpg)](https://www.youtube.com/watch?v=g52fmQ8ifak "Mango grade classification demo")
