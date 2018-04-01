# RobodND Deep Learning Project

---

[//]: # (Image References)
[img2]: ./misc/FCN_architect.PNG

This project to is to build a Fully Convolutional Network for images segmentation. The goal of such segmentation is coordinating the hero's position (`Fig. 1`) in the images taken by a quadrotor operating in a simulation environment so that it can follow the hero. The implementation of this FCN is consituted by 3 building blocks which are the encoder, the 1x1 convolution, and the decoder. The function of each block, how it is formed, and the its role in the whole network is described in following sections. 

## 1. Network Architect
Outputs of an well-trained FCN are images containing all the pixels of the original images at the same position along with the prediction for these pixels. Therefore, the obtained information is not only objects' presence but also their position. Such outputs are generated thanks to the integration of the 3 building blocks mentioned above into a unified architect shown in `Fig.2`.

![alt text][img2]
*Fig.2 FCN architect*

In this architect, the **encoder** carries out the features extraction from the input images. Recall that, a single convolutional layer is capable of recognizing patterns (e.g. shape or color) and the deeper the convolutions stack is, the more complex the recognized patterns are. As a result, a convolutional layer is a feasible choice for an encoder. Each encoder (i.e. a convolutional layer) squeezes the input such that it reduces height and width, but increases in depth.  

At the end of the encoders queue, a **1x1 convolution** is placed instead of a fully connected layer, compared to the CNN architect. The reason for this replacement is the 4-D tensor outputed by the 1x1 convolution. Such high-dimention output preserves the spatial information (the location of pixels), which is lost the flattened out output of a fully connected layer.    

The output of the 1x1 convolution containing all the features extracted by the encoders (convolutional layer) queue is up scaled by the **decoder** to the size of the original input image. This is made possible by the spatial information preserved in the 1x1 convolution. The techniques utilized to implement the decoder are the Bilinear Upsampling and Skip Connection. While the Bilinear Upsampling is for scaling up the input, the Skip Connection helps improve the scale accuracy by enabling the decoder to use inputs at different resolution by directly feeding it with the encoders output. 
