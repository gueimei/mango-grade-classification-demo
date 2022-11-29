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
| Model Name           | Download Link    |
| -------------------- | ---------------  |
| SPP_AlexNet          | [GoogleDrive](https://drive.google.com/drive/folders/1AoRSJfHSf889OiMLc-HR2lSibynYODIG?usp=sharing)|
| shapes20200813T1158  | [GoogleDrive](https://drive.google.com/drive/folders/172CdJBEgfRXXKFbxkFHxNMQHs85ijNQR?usp=sharing)|

## Excute
1. Download the "SPP_AlexNet" and "shapes20200813T1158" models and place them to "logs" folder
2. Open Spyder
3. Excute "Mango_sys.py"
4. Click "choose file" and choose test image folder
5. Click "check", detect the mango grade and generate a txt to record result

### Demo
[![IMAGE ALT TEXT](https://www.youtube.com/watch?v=g52fmQ8ifak&ab_channel=%E6%A2%85)
