# Temporal-Relations-of-Informative-Frames-in-Action-Recognition

In this [paper](https://www.researchgate.net/publication/379714148_PDF_Temporal_Relations_of_Informative_Frames_in_Action_Recognition), to detect actions with transfer learning + RNNs we have 3 steps:

1. In the first step, we use a frame selection algorithm to avoid the redundancy of videos which is explained in this paper [Adaptive Frame Selection In Two-Dimensional Convolutional Neural Network Action Recognition](https://www.researchgate.net/publication/368726751_Adaptive_Frame_Selection_In_Two_Dimensional_Convolutional_Neural_Network_Action_Recognition) and code can be found here([Code](https://github.com/Alirezarahnamaa/Adaptive-Frame-Selection-Algorithm))

2. In the next stage, we use this repository([Feature extraction](https://github.com/Alirezarahnamaa/Feature_Extraction)) to extract the spatial features from each selected frame by pre-trained ResNet-50 to have one spatial feature vector for each selected frame.

3. In the end, we use a temporal pooling method to divide each video into 4 parts and have strong spatial-temporal feature vectors for each video; after feature extraction, the RNN models are trained to classify actions. Moreover, using LOOCV helps to have reasonable results because we evaluate and train all videos of UCF11.

### Architecture

![](Readme_images/Architecture.png)
