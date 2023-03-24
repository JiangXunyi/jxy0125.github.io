

# Lecture 2



- Range: [0,255] a byte = 8 bits， **brightness**

- 0 代表黑色， 255 代表白色

- **Binary Image：** 只有0，1

- **position** of pixel and **value** of pixel.   

![image-20220915104146765](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220915104146765.png)



## Pixel Processing: Basics

Black/White change: 255-pixel value  = image negate/ invert

![image-20220915104654237](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220915104654237.png)

in Matlab: I = 255-I



### 2. Contrast Scaling

![image-20220915105024956](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220915105024956.png)

增强对比度： 把[L,H]上的拉伸， e.g. select [23,230], 则当在这个之外的就直接变成0，255， 在其中的值均匀的分布在[0,255]上



### 3. Hazard of Overflow

![image-20220915105932741](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220915105932741.png)

Scale back into [0,255]

byte 



## Intensity Histograms

![image-20220915111041904](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220915111041904.png)

### entropy！

If there are n bars, then there are maximum value of entropy----flatness, 

Larger -> lager uncertainty

Cross entropy: similarity of two distribution



## Thresholding

separate pixels into two categories: Binary image, often called binarization

$B(i,j) = \lbrace $

Different Threshold -> diferent results? authomatically -> analysis histogram

![image-20220915113544881](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220915113544881.png)

如果是一个白色的物体在dark background. Bimodal (双分布) histogram

如果有overlap？ 如何解决？ cateria?



### Optimal Thresholding

Seperation: how to decide?

- interclass distance/ between class distance: the larger, the better seperation
- Intraclass distance/ within class variance: the smaller , the compact



### ==Otus's Method==

![image-20220915114731839](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220915114731839.png)

Min: intra; Max: inter

![image-20220915115037841](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220915115037841.png)

![image-20220915115156118](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220915115156118.png)

![image-20220915115616275](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220915115616275.png)

```matlab
level = graythresh(I)

BW = Im2bw(I, level) 


```

#### ==Auto Thresholding - Approximate==

![image-20220915115959347](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220915115959347.png)

$(\mu_1 + \mu_2)/2$ 直到不改变 1-D K-Means Method



## Change Detection and Background Extraction

### Motivation: Situation Awareness

一些行为的探查

### Image Sequences

- a typical frame-rate: 25Hz
- $I(x,y,t), t = 0,1,...$



### Change Detection: Temporal Difference

$D(x,y,t) = |I(x,y,t-1) - I(x,y,t)|$

![image-20220922103523172](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220922103523172.png)



### Environmental situation

fix / dynastic



## Background Subtraction

$F(t) = |I(t)- B(t)|$ , where I is image, B is a background

- Average of the first n pics

- $B(t) = w_B  B(t-1) + w_I I(t)$    This is a linear combination. 

  $$
  B(t) = w_B(w_BB(t-2) + w_II(t-1)) + w_II(t) \\
  = w_B^2 B(t-2) + w_I(w_BI(t-1) + I(t))  
  = ... =w_B^n B(0) + w_I\sum w_B^{i} I(t-i) 
  $$
  Adavy, moving average

- Gussian: for each point in background , supppose ~ $N(\mu,\sigma^2)$ , using $[-3\sigma, 3\sigma]$ to detect the change in the picture for each point. If the point is beyond the confidence interval, then it will be consider as the forward figure.
- **GMM**: mixture of different Gussian.




# Lab

[这个是计算各种距离的Matlab](https://www.cnblogs.com/wkwzn2019/p/13167318.html)























# Lecture3-Color

## Color Space

### interaction of light and surfaces

- illumination 
- Reflectance
- Color signal



### Color space and conversion

There are 3 color space: RGB, HSV, LAB

**Conversion:** 

![image-20220922112926461](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220922112926461.png)

RGB: each pixel has 3 byte (how strong that. channel is? )

Grey: each pixel has 1 byte.



### Linear color sapces: RGB

Every color is a linear combination of the primaries.



### Nonlinear color spaces: HSV

more intutive than: RGB

angle means different color, the distance from the center means saturation

![image-20220922114631618](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220922114631618.png)





## Color histogram

seperate the 256 into different bins(e.g. 8),  then there are 8^3 bins . For every pixel, 

Oct $\rightarrow$ decimal . we consider the index of bins a oct number , then we can function this a

[10,2,2] into decimal: 



### Commaring Two Distributions 

![image-20220922121230160](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220922121230160.png)





## Lab3

reshape()

Jet()



# Lecture4

histogram is too **finer**: all of them are different

## Color-based skin detection

Pixels -> HS space: 2D Gaussian model to estimate.

$N(\mu,\Sigma)$, calculate the mean and covariance of two variable.



## Sampling

Def: obtain values at regular intervals

sampling rate:  the number of samples per second/period



- the smaller, the accuracy, more memory. trade off
- smallest number so that we can accurately reconstruct ...

### The best choice of sampling rate: 

- **2 samples per period**

Q: if the smaple point do not on the peak and bottom , then the 

- if the frequencey is not the same: 2 times of the highest frequency.



## Fourier Transform

![image-20220929105517158](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220929105517158.png)

#### Assumption: Peoridic



## 2D sinusoidal Basis

![image-20220929105829888](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220929105829888.png)



![image-20220929110332510](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220929110332510.png)

It can be written as a linear combination of orthonormal basis.

$x$ is the sample point of the curve, then if can be 



### 2D Fourier Transform

![image-20220929112217859](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220929112217859.png)



### 2D DFT(Discrete FT)

![image-20220929113333903](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220929113333903.png)

FFT shifted: / FFTshift2

![image-20220929114427286](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220929114427286.png)



在变换回去之前要再做一个fftshift



![image-20220929115634520](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20220929115634520.png)

pixels are between [0,1], which means after summarization, it will left a constant . 

## Summerize

- 将整个pixel视作sampling， 则我可以探测到的最大频率为1/2* sample frequency



## Filtering in Frequency Domain

Lower filter/ High filter



# Lecture5-Convolution and Spatial Filter

## Convolution operation

#### Basic insigths and definition

- Linear combination
- Statistics

2. Convolution/correlation?

   - filpped to signal to 

   - Size: $N-M+1$, where N is the length of the original image, M is the length of Mask

   - How to determine the mask? In next part

   - Get the value:
     > Steps:
     >
     > 1. From input sum get the convolution , 
     > 2. then normalize,
     > 3. usually devide sum?

   - Padding : ==full== output?
   
     Mirror operations, flipped the values, do a peoridic assumption
   
2. Gaussian (assumption) mask

   Isotropic: kernel is separate, which means independently. Fast calculation



#### What can convolution do?

1. remove noise
2. Looking for patterns(like Guassion or step function)
3. template matching, e.g. Match the dog, create dog mask
4. Feature extraction
   - Local Fourier / Werelet / ...
5. Scale Space representation



#### Step Size

Different step means you have different sample size



## Image Noise

- Salt and pepper noise
- Guassian Noise

#### Creating Salt and Pepper Noise

1. random select the (x,y) , then sign $Y = 255X \sim Bernoulli(0.5) $

#### Creating Guassion Noise

1. geberate from $X\sim N(0,1)$,( then do linear transformation $\sigma X + \mu $)
2. large $\sigma$ generate more severe corrupted

#### Remove Noise

1. Mean Filtering
2. Median Filtering : can be used to remove salt and pepper noise

##### Comparation

Median Filter overweigh the mean filter when the noise is salt and pepper noise



# Lecture5-Scale Space and Edge

## Scale Space and Guassian Pyramid

$\sigma$ is the scale parameter

![image-20221013114651526](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20221013114651526.png)

Fourier transformation of Guassian  is still Guassian. Since the convolution of Fourier transformation is the Fourier tranformation of convoluiton. Then we can say that 

$\mathcal{F}(f(x) * g(x)) = \mathcal{F}(f(x)) \cdot \mathcal{F}(g(x)) $

==低通滤波器 $\sim$ Guassian==

[高斯低通滤波器](https://blog.csdn.net/weixin_44580210/article/details/105091620)



![image-20221013120231904](/Users/xunyijiang/Library/Application Support/typora-user-images/image-20221013120231904.png)

## Origin of Edges

### Characterizing edges

Definition: rapid change in the image intensity function

### Differentiation and Convolution



# Lecture6-Edge Detect&Optical Flow

## Edge Detection
### Image Gradient

- How to find out the gradient of a image?

- Differential and convolution

  - Kernel: [-1,1] then you can get the **horizontal gradient** differential of the 

  - Kernel: $[-1,1]^T$ can be use to calculate **verticle gradient**

  - magnitude:$\sqrt{(dx^2+dy^2)} \geq \text{threshold}$

  - There are tremendous different kernel to estimate.

    ![image-20221020104210031](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20221020104210031.png)

    - $[[-1,-1,-1]^T,[1,1,1]^T] = [1,1,1]^T [-1,1]$can be considered as fisrt do a smoothing ,then do a differential
    - $[-1,0,1]$:  centered gradient, this is more symmetric way 
    - **Sobel**: like Guassian smoothing, given different weights

  - How to judge which is the horizontal/vertical?

    Detect the difference horizontally / vertically



### Effect of noise

![image-20221020104750911](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20221020104750911.png)

How to solve this problem?

- Using kernel to remove the noise
- Then get the gradient

How to save time?  **Combine Them!!!**

![image-20221020105125331](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20221020105125331.png)

- This way can save $O(m^2n^2-2mn^2)$ 



### Canny's Edge Detector

![image-20221020110120824](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20221020110120824.png)

- Where is the edge?

​       Use non-maximum suppression

#### Non-maximum suppression

Direction of gradient, we can get the pixel with maximun magnitude in the neiborhood 

This is a kind of shrink.

#### Hysteresis Thresholding

![image-20221020113007452](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20221020113007452.png)

#### Step Summary

- Filter image with derivative of Gaussian
- Find magnitude and orientation of grandien
- Non-maximun/Hysteresis



## Optical Flow

motion field and optical flow is just like the projection and its inverse projection

### Motion Field

#### Motion Field and Parallax

- in the real world

​      $X(t) = [x(t),y(t),z(t)]$, where $X(t)$ is the location of the dots

​      $V(t) = [v_x(t),v_y(t),v_z(t)]$, This is the velocity of dots in 3D world

- In the picture, only 2D information

​      $X(t) = [x(t),y(t)]$

​      $V(t) = [v_x(t),v_y(t)]$,

- How to get 2D from 3D? **pinhole model**

  $x_{\text{2D}}(t) = \dfrac{f}{z_{\text{3D}}(t)} x_{\text{3D}}(t)$

  $v_{2D}(t) = f\dfrac{v_{x_{3D}}(t)z_{3D}(t) - }{}$





### Optical Flow

- Definition: optical flow is the *apparent* motion of **brightness** patterns in the image

#### Estimating optical flow

> Key Assuptions
>
> - Brigthness constancy: projection of the same point looks same in every frame
>
>   即一个点的appearance是不会变化的
>
>   $$I(x_0,y_0,t_0) \doteq I(x_0+u, y_0+v, t_0+1) \doteq  I(x_0,y_0,t_0) + I_x(x_0) u + I_y(y_0) v + I_t(t_0)$$ 
>   $$
>   I_x(x_0) u + I_y(y_0) v=- I_t       \text(or) (I(x,y,t+1)- I(x,y,t))
>   $$
>   
>
>   按时间间隔为1， where （u,v) is the velocity of the point.
>
> - Small motion: point do not move very far
>
>   这个点不会移出去
>
> - Spatial coherence： 这个点跟他的邻居们的移动是相似的






#### Aperture Problem 
- motion along the Edge dirction can not be observed. **ILLUSION**
  $$
  [I_x, I_y]^T[u,v]= 0
  $$



##### Solve the aperture problem

![image-20221027110341516](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20221027110341516.png)
$$
Ax = b
\Leftrightarrow
x = (A^TA)^{-1}Ab
$$

##### Conditions for solvability

- Bad cases: single straight edge
- Good: **Corner**



# Objective Detect

> Outline
>
> - SVM
> - V-J Intergral hot
> - HoG



### People detection

- Search exhaustively the scale-space image

  由于无法确定人的大小，因此会进行从大到小的框框

- Classify a window whether is a human

#### Summary of methods

![image-20221027113423493](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20221027113423493.png)



![image-20221027113537740](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20221027113537740.png)

#### Vectorize

![image-20221027114444002](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20221027114444002.png)

- can first use some filter to extract features, then using features to classifier
- Then deliever these vector to SVM
- An example
- ![image-20221027114828076](/Users/xunyijiang/Library/Application Support/typora-user-images/image-20221027114828076.png)





#### How to extract features?

![image-20221027115113209](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20221027115113209.png)

[VJ](https://www.cnblogs.com/hrlnw/p/3374707.html)

白色部分的和（sum）-  灰色部分的和（sum）

![image-20221027120030859](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20221027120030859.png)







#### Integral Image

通过这个方法可以很大程度减小计算复杂度

![image-20221027120257640](/Users/xunyijiang/Library/Application Support/typora-user-images/image-20221027120257640.png)

#### Adaboost

Stronger from weaker (threshold/linear)

##### Weak classifier

threshold which is just better than chance level.

Using one threshold to 



##### Attention Cascade

注意力级联

![image-20221027194111373](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20221027194111373.png)



![image-20221027194928262](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20221027194928262.png)



#### V-J detector

![image-20221027200226151](/Users/xunyijiang/Documents/StudyMaterials/2022Fall/AI_B/notes/images/:Users:xunyijiang:Library:Application Support:typora-user-images:image-20221027200226151.png)

> 1. Rec features using integral
> 2. Feture Select Adaboost
> 3. Cascade -> fast rejection

https://blog.csdn.net/qq_16829085/article/details/108680639

#### HoG

Descriptor: 3-D HIstogram of Oriented Gradient.

Pics -> preperation -> compute gradient -> histogram(weighted, spatial, voting) -> contrast norm(blocks)

##### Gamma/Colour Normalisation

- Test RGB and LAB, Grayscale
- ![image-20221103104450031](/Users/xunyijiang/Library/Application Support/typora-user-images/image-20221103104450031.png)



##### Gardient Computation

一般我们求gradient的时候要先smooth， 但是发现会降低performance， 是由于remove finer details。 texture removed？

##### Orientation and Spatial Voting

首先分成小的cells, 4*4

###### orientation histogram

每个像素点都对应一个梯度$[g_x, g_y]$以及 梯度的大小，orientation histogram最通过梯度的方向，根据方向将16个梯度分到8个bins中，而值则通过计算这些梯度大小的和





# Week8BoW_Tracking

### Bag pf features

Using Harris / SIFT

首先得到很多的features， 然后clustering？

k-means

visual vocabulary, dictionary, codeword

Lose spatial information-orderless

ii



### Tracking

#### mean shift

- find local peaks 
- Track



# Week10 NN and SVM

## Classification

definition：Assign one or many data vectors into one or man classes.

#### KNN

When is a linear classifier?

- decision boundary is 

- hinge loss： 
  $$
  argmin \sum max(0,-y(w^Tx + b))
  $$

- MSE
- Cross Entropy: 



#### SVM

https://blog.csdn.net/weixin_39653948/article/details/105859281

about kernel

##### Summary of SVM

![image-20221124202204748](/Users/xunyijiang/Library/Application Support/typora-user-images/image-20221124202204748.png)

##### 优缺点

![image-20221124202308631](/Users/xunyijiang/Library/Application Support/typora-user-images/image-20221124202308631.png)
