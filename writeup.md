# RobodND Deep Learning Project

---

This project to is to build a Fully Convolutional Network for images segmentation. The goal of such segmentation is coordinating the hero's position (`Fig. 1`) in the images taken by a quadrotor operating in a simulation environment so that it can follow the hero. The implementation of this FCN is consituted by 3 building blocks which are the encoder, the 1x1 convolution, and the decoder. The function of each block, how it is formed, and the its role in the whole network is described in following section. 

## 1. Network Architect
Outputs of an well-trained FCN are images containing all the pixels of the original images at the same position along with the prediction for these pixels. Therefore, the obtained information is not only objects' presence but also their position. Such outputs are generated thanks to the integration of the 3 building blocks mentioned above into a unified architect shown in `Fig.2`.

In this architect, the **encoder** carries out the extraction of features from the input images. Recall that, a single convolutional layer is capable of recognizing patterns (e.g. shape or color) and the deeper the convolutions stack is, the more complex the recognized patterns are. As a result, a convolutional layer is a feasible choice for an encoder. 

At the top of the encoder, a **1x1 convolution** is placed instead of a fully connected layer in the CNN architect. This 1x1 convolution is for ...   
