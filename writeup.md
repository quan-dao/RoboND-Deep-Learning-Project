# RobodND Deep Learning Project

---

[//]: # (Image References)
[img1]: ./misc/fcn_target.PNG
[img2]: ./misc/FCN_architect.PNG
[img3]: ./misc/regular_conv.PNG
[img4]: ./misc/seperable_Conv.PNG
[img5]: ./misc/decoder_block.PNG

This project to is to build a Fully Convolutional Network for images segmentation. The goal of such segmentation is coordinating the hero's position (Fig. 1) in the images taken by a quadrotor operating in a simulation environment so that it can follow the hero. The implementation of this FCN is consituted by 3 building blocks which are the encoder, the 1x1 convolution, and the decoder. The function of each block, how it is formed, and the its role in the whole network is described in following sections. 

![alt text][img1]
*Fig.1 The Hero*

## 1. Network Architect
Outputs of an well-trained FCN are images containing all the pixels of the original images at the same position along with the prediction for these pixels. Therefore, the obtained information is not only objects' presence but also their position. Such outputs are generated thanks to the integration of the 3 building blocks mentioned above into a unified architect shown in Fig.2.

![alt text][img2]
*Fig.2 FCN architect*

In this architect, the **encoder** carries out the features extraction from the input images. Recall that, a single convolutional layer is capable of recognizing patterns (e.g. shape or color) and the deeper the convolutions stack is, the more complex the recognized patterns are. As a result, a convolutional layer is a feasible choice for an encoder. Each encoder (i.e. a convolutional layer) squeezes the input such that it reduces height and width, but increases in depth.  

At the end of the encoders queue, a **1x1 convolution** is placed instead of a fully connected layer, compared to the CNN architect. The reason for this replacement is the 4-D tensor outputed by the 1x1 convolution. Such high-dimention output preserves the spatial information (the location of pixels), which is lost the flattened out output of a fully connected layer.    

The output of the 1x1 convolution containing all the features extracted by the encoders (convolutional layer) queue is up scaled by the **decoder** to the size of the original input image. This is made possible by the spatial information preserved in the 1x1 convolution. The techniques utilized to implement the decoder are the Bilinear Upsampling and Skip Connection. While the Bilinear Upsampling is for scaling up the input, the Skip Connection helps improve the scale accuracy by enabling the decoder to use inputs at different resolution by directly feeding it with the encoders output. The skip connections are represented by the green arrow in Fig.2.

## 2. The Encoder

The encoder is made out of **Seperable Convolutional Layer** equipped with **Batch Normalization** technique.

### 2.1 Seperable Convolutional Layer

The Seperable Convolutional Layer is a variation of the regular convolutional layer. Perform the same functionality of extracting feature from inputs, the Seperable Convolution Layer has the advantage of containing less parameters than the regular one thanks to the twist in its operation. This twist is explained in the following example.

Consider an 32x32x3 image as the input for a regular convolutional layer. This layer operates with a filter of the size 3x3x3 to produce 9 output channels. This operation is illustrated by Fig.3.      

![alt text][img3]

*Fig.3 The operation of a regular convolutional layer*

The 3x3x3 filter seperates the input to smaller patches of the same size. Each patch forms a input tensor `x` (containing the value of all pixels in this patch) that is then fed to a hidden neuron through a number of connections equaling to the total pixels in the patch (patch's volume). This hidden neuron carries out the matrix multiplication between `x` and the matrix of connections weight `W` to produce a **scalar** output, this scalar is a pixel of one output channel. As the filter sweepts through the input, the scalars outputed by the hidden neuron pads next to each other to produce a complete output channel. Given this way of operation, 27 (=3x3x3) parameters are required for an output channel. Therefore, 243 (= 9x27) parameters are needed to produce 9 channels of output.

The Seperable Convolution shown in Fig.4, on the other hand, traverses each input channel individually with a distinct `kernel` (a filter and a hidden neuron) to produce a feature map. This feature map then get to be traversed again by a **1x1 Convolution** to produce an output channel. This secondary process is similar to the way a regular convolutional layer operates on its inputs, except the filter size is set to `1x1`. 

![alt text][img4]
*Fig.4 The operation of a seperable convolutional layer*   

Applying the opertion of Seperable Convolutional Layer in the example of an 32x32x3 input image, the number of kernels involving is 3 and there are 3 resulted feature maps because the input image has 3 channels. As each kernel utilizes a 3x3 filter, the total parameters of 3 kernels is 27 (=3x(3x3)). Furthermore, to create 9 output channels, each feature map needs to be traversed by 9 1x1 Convolution; so, 9 more parameters are required for each feature map. The total parameters of a seperable convolutional layer for producing 9 output channels are 27 + 3x9 = 54. This is much smaller than the equivalent of a regular convolutional layer (243 parameters).      

Because of the advantage of requiring less parameters to produce the same result, the Seperable Confolutional Layer makes the whole network more efficient with improved runtime performance and reduces overfitting.

### 2.2 Batch Normalization

Batch Normalization is an additional way to optimize the network training. Its main idea is that not only the input to the whole network is normalized but also the input to every layer within the network. The normalization of layers input is carried out using the mean and variance of values in the current mini-batch, hence the name Batch Normalization.

## 3. The 1x1 Convolution

As mentioned in the previous sections, the 1x1 Convolution is used at the end of the encoder queue to preserve the spatial information or by the Seperable Convolution to produce output channels. 

The implementation of a 1x1 Convolution is the same as a regular convolution which traverses the input stride by stride using a filter. The only difference is that the size of filters is fixed to 1x1 in the case of 1x1 Convolution, while the regular convolutional layer makes use of filters of any size.

The 1x1 Convolution plays the role of a minimal neuron that takes 1 input, multiplies it by a weight, add a bias, then pass to an activation function. This role is similar to a fully connected layer. However, by moving across the input pixel by pixel, the 1x1 Convolution passes on the information of pixels position relative to each other to the output. Such spatial information is completely loss in the flatten-out output of a fully connected layer. Therefore, the 1x1 Convolution is put a the end of the encoder queue to prepare information for the decoding process, while the fully connected layer is placed at the top of the CNN architect to classify the features extracted by the previous convolutional layers.      

## 4. The Decoder

A decoder block is constructed by 3 sub building blocks that are the Bilinear Upsampling Layer, the Skip Connection, and the Seperable Convolutional Layer. The architect of one decoder block used in this project is shown in Fig.5.

![alt text][img5]
*Fig.5 The architect of a decoder block*

### 4.1 The Bilinear Upsampling Layer

A Bilinear Upsampling layer takes the output of the previous layer as input and upscales it according 2 predefined scaling factors (1 for x-axis, 1 for y-axis) that define the output size. Because the padding of the convolutional layers in the encoders queue is chosen to be `SAME` padding, the height and width of output of these layers are kept equal to the equivalent of the input. Therefore, the 2 scaling factors of every upsampling layer are kept at **1** to enforece the size compatibility as well as ensure the output image of the whole network has the same size as the input. 

### 4.2 The Skip Connection

The Skip Connection is the means to pass output between non-adjacent layers. In this project, this connection is implemented as the concatenation of the output of the upsampling layer and the output the encoder having the same distance to the 1x1 Convolution (at the middle of the FCN). Such concatenation is only possible when the size of the encoder output and the upsampling output are equal. This size equality is what the "size compatibility" mentioned in the previous subsection about.

## 5. Hyper parameters tuning

The hyper parameters involving in training the FCN used in this project are the `learning rate`, the `batch size`, the `number of Epoches`, and the `steps per epoch`. These prameters are chosen stochastically by the initial value of 0.01, 50, 1, 200 respectively.

The ***number of epoch*** is set to 1 to test whether the training process proceeds error free. The large value of ***batch size*** (50) make the memory needed for storing the images that get propagted through the network in a single pass exceed the totatl memory of the machine used to train the network. As a result, the Batch Size is decreased to 20. 

Once the training process smoothly proceeds, the ***number of epoch*** is increased to 3 and the ***steps per epoch*** is doubled to 400. The resulted model is scored nearly 30% below the acceptable threshold (40%).

Another training cycle is carried out with the same ***learning rate*** (0.01), the ***batch size*** increased to 30, the ***number of epoches*** inceased to 5, and the ***the steps per epoch*** decreased to 200. The model score is decreased to around 23%. This informs the adjustment of hyper parameters is in the wrong direction.

The 30% score of the previous training cycle seems promising. So, ***number of epoches*** is increased to 10, while others paramters is kept the same. That leaves 0.01 ***learning rate***, 20 ***batch size***, and 400 ***step per epoch***. This time the model is scored 43% above the threshold. Therefore, the hyper parameters are finallized.
