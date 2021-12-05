# Application-of-Autoencoders and GANs

Abstract: With the ever-going advancements in field of deep learning has changed the lives forever. A century back nobody imagined we will have self driving cars, unmanned aerial vehicles work. But with the state-of-the-art machine learning algorithms had made it possible and one of those technologies is computer vision. It has made significant advancements in image processing and analysis. Its applications are used over all domains from health sector to military ops. Researchers are continuously thriving for new accomplishments which are building stones for human race. Autoencoders and GANs are typical CNNs which has many real-life applications. For instance, AI powered cameras use autoencoders for image colorization and GANs for image translation. In this paper we elaborate these applications.

1. Introduction
        Today’s deep learning models can reach human-level accuracy in analysing and segmenting an Image. convolutional neural network (CNN, or ConvNet) is a class of deep neural networks, most commonly applied to analysing visual imagery. They have applications in image and video recognition, recommender systems, image classification, medical image analysis, natural language processing, and financial time series. They are also known as shift invariant or space invariant artificial neural networks (SIANN).1
2. Working of CNN
In neural networks, Convolutional neural network (ConvNets or CNNs) is one of the main categories to do images recognition, images classifications. Objects detections, recognition faces etc., are some of the areas where CNNs are widely used. CNN image classifications take an input image, process it and classify it under certain categories (Eg., Dog, Cat, Tiger, Lion). Computers sees an input image as array of pixels and it depends on the image resolution. Based on the image resolution, it will see h x w x d( h = Height, w = Width, d = Dimension). Technically, deep learning CNN models to train and test, each input image will pass it through a series of convolution layers with filters (Kernals), Pooling, fully connected layers (FC) and apply Softmax function to classify an object with probabilistic values between 0 and 1. The below figure is a complete flow of CNN to process an input image and classifies the objects based on values.
 
Convolution Layer 
Convolution is the first layer to extract features from an input image. Convolution preserves the relationship between pixels by learning image features using small squares of input data. It is a mathematical operation that takes two inputs such as image matrix and a filter or kernel.
 
Consider a 5 x 5 whose image pixel values are 0, 1 and filter matrix 3 x 3 as shown in below
 
Then the convolution of 5 x 5 image matrix multiplies with 3 x 3 filter matrix which is called “Feature Map” as output shown in below
 
Convolution of an image with different filters can perform operations such as edge detection, blur and sharpen by applying filters. The below example shows various convolution image after applying different types of filters (Kernels).
 
Strides
Stride is the number of pixels shifts over the input matrix. When the stride is 1 then we move the filters to 1 pixel at a time. When the stride is 2 then we move the filters to 2 pixels at a time and so on. The below figure shows convolution would work with a stride of 2.
 
Padding
Sometimes filter does not fit perfectly fit the input image. We have two options:
•	Pad the picture with zeros (zero-padding) so that it fits
•	Drop the part of the image where the filter did not fit. This is called valid padding which keeps only valid part of the image.
Non Linearity (ReLU)
ReLU stands for Rectified Linear Unit for a non-linear operation. The output is ƒ(x) = max(0,x). ReLU’s purpose is to introduce non-linearity in our ConvNet. Since, the real world data would want our ConvNet to learn would be non-negative linear values.
 
There are other non linear functions such as tanh or sigmoid that can also be used instead of ReLU. Most of the data scientists use ReLU since performance wise ReLU is better than the other two.
Pooling Layer
Pooling layers section would reduce the number of parameters when the images are too large. Spatial pooling also called subsampling or down sampling which reduces the dimensionality of each map but retains important information. Spatial pooling can be of different types:
•	Max Pooling
•	Average Pooling
•	Sum Pooling
Max pooling takes the largest element from the rectified feature map. Taking the largest element could also take the average pooling. Sum of all elements in the feature map call as sum pooling.
 
Fully Connected Layer
The layer we call as FC layer, we flattened our matrix into vector and feed it into a fully connected layer like a neural network.
 
In the above diagram, the feature map matrix will be converted as vector (x1, x2, x3, …). With the fully connected layers, we combined these features together to create a model. Finally, we have an activation function such as softmax or sigmoid to classify the outputs as cat, dog, car, truck etc.,
 

3. Image Colorization using Autoencoders
An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore signal “noise” 3.
Autoencoder Components:
Autoencoders consists of 4 main parts:
1- Encoder: In which the model learns how to reduce the input dimensions and compress the input data into an encoded representation.
2- Bottleneck: which is the layer that contains the compressed representation of the input data. This is the lowest possible dimensions of the input data.
3- Decoder: In which the model learns how to reconstruct the data from the encoded representation to be as close to the original input as possible.
4- Reconstruction Loss: This is the method that measures measure how well the decoder is performing and how close the output is to the original input.
The training then involves using back propagation in order to minimize the network’s reconstruction loss.
The network architecture for autoencoders can vary between a simple Feedforward network, LSTM network or Convolutional Neural Network depending on the use case
 
Other applications of autoencoders include Anomaly detection and image denoising
Process of converting grayscale image to rgb image using autoencoders
Here we used the Lego minifigures classification dataset which consist of 172 rgb images using cv2 library in python we have converted it into gray scale images and used it as input data  to our model and original rgb images as labels.
After that process we define the model which consist of encoder and decoder. Here encoder has 7 layers and input shape is (256,256,1). For decoder we use  8 layers. We used Relu activation and tanh activation function and Conv2D and upsampling layer in our model.
Summary of the model is as follows
Model: "sequential" _________________________________________________________________ Layer (type) Output Shape Param # ================================================================= conv2d (Conv2D) (None, 128, 128, 64) 640 _________________________________________________________________ conv2d_1 (Conv2D) (None, 128, 128, 128) 73856 _________________________________________________________________ conv2d_2 (Conv2D) (None, 64, 64, 128) 147584 _________________________________________________________________ conv2d_3 (Conv2D) (None, 64, 64, 256) 295168 _________________________________________________________________ conv2d_4 (Conv2D) (None, 32, 32, 256) 590080 _________________________________________________________________ conv2d_5 (Conv2D) (None, 32, 32, 512) 1180160 _________________________________________________________________ conv2d_6 (Conv2D) (None, 32, 32, 512) 2359808 _________________________________________________________________ conv2d_7 (Conv2D) (None, 32, 32, 256) 1179904 _________________________________________________________________ conv2d_8 (Conv2D) (None, 32, 32, 128) 295040 _________________________________________________________________ up_sampling2d (UpSampling2D) (None, 64, 64, 128) 0 _________________________________________________________________ conv2d_9 (Conv2D) (None, 64, 64, 64) 73792 _________________________________________________________________ up_sampling2d_1 (UpSampling2 (None, 128, 128, 64) 0 _________________________________________________________________ conv2d_10 (Conv2D) (None, 128, 128, 32) 18464 _________________________________________________________________ conv2d_11 (Conv2D) (None, 128, 128, 16) 4624 _________________________________________________________________ conv2d_12 (Conv2D) (None, 128, 128, 2) 290 _________________________________________________________________ up_sampling2d_2 (UpSampling2 (None, 256, 256, 2) 0 ================================================================= Total params: 6,219,410 Trainable params: 6,219,410  Non-trainable params: 0 _________________________________
After compile and fitting the model we save the model as colorize_autoencoder.model
Now we load the model and test with some gray scale  image and the results are as follows
                  input                                                                                            output
 
 
4. Image translation using GANs
A generative adversarial network (GAN) is a class of machine learning frameworks designed by Ian Goodfellow and his colleagues in 2014. Two neural networks contest with each other in a game (in the form of a zero-sum game, where one agent's gain is another agent's loss).Given a training set, this technique learns to generate new data with the same statistics as the training set. For example, a GAN trained on photographs can generate new photographs that look at least superficially authentic to human observers, having many realistic characteristics. Though originally proposed as a form of generative model for unsupervised learning, GANs have also proven useful for semi-supervised learning, fully supervised learning, and reinforcement learning. The core idea of a GANs is based on the "indirect" training through the discriminator, which itself is also being updated dynamically. This basically means that the generator is not trained to minimize the distance to a specific image, but rather to fool the discriminator. This enables the model to learn in an unsupervised manner. 5     
GANs consist of three components:
Generative:  Creates a fake data
Adversarial: it is generator and discriminator each competing to win against each other. Generator trying to fake and discriminator, trying not to be fooled.
Network:  It is a CNN   
Architecture of GANs:
 
                                                                                        
Implementing GAN
Here we use Lego minifigures classification dataset and convert into edge such that only edges are visible and take it as input to our model. Furthermore, normal images as label for this model.
1. Define GAN architecture based on the application 
2.Train discriminator to distinguish real vs fake data 
3. Train the generator to fake data that can fool discriminator
4. Continue discriminator and generator training for multiple epochs
5. Save generator model to create new, realistic fake data
6. We train this model for 100 epochs and save model
7.we load the model and test with new data and results are as follows:
                         Input                                                                            Output
          
References:
1.  https://en.wikipedia.org/wiki/Convolutional_neural_network
2.  https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148  
3.https://en.wikipedia.org/wiki/Autoencoder#:~:text=An%20autoencoder%20is%20a%20type,to%20ignore%20signal%20%E2%80%9Cnoise%E2%80%9 
4. https://towardsdatascience.com/auto-encoder-what-is-it-and-what-is-it-used-for-part-1-3e5c6f017726 
5.  https://en.wikipedia.org/wiki/Generative_adversarial_network
6.http://www.wisdom.weizmann.ac.il/~vision/courses/2018_2/Advanced_Topics_in_Computer_Vision/files/DomainTransfer.pdf
NOTE:
Since, our project is application of Autoencoders and GANs there is no need of plotting and accuracy as we are not doing classification.
