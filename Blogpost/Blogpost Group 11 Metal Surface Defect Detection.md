# 1. Introduction

Quality inspection and control in the steel-manufacturing industry have been a critical issue for assuring the product quality and increasing its productivity. As a steel defect is deemed to be one of the main causes of the production cost increase, monitoring the quality of steel products is inevitable during the manufacturing process [1]. The defects can be attributed to various factors, e.g., operational conditions and facilities [2]. For an immediate response and control about the flaws, detecting steel defects should be preceded to analyze the failure causes. To this end, a sophisticated diagnostic model is required to detect the failures properly and to enhance the capability of quality control [3].

In particular, a vision-based diagnostics system for detecting the steel surface defects has received considerable attention. The traditional human inspection system has several disadvantages such as a less-automatic and time-consuming procedure [4]. An image-based system, on the other hand, is developed to enable more elaborate, rapid and automatic inspection than the existing methods [5]. Furthermore, it is widely known that the surface defect accounts for more than 90% of entire defects in steel products, e.g., plate and strip [6]. Defects on the steel surface, e.g., scratches, patches, and inclusions exert maleficent influence on material properties, i.e., fatigue strength and corrosion resistance, as well as the appearance [7]. Likewise, the development of a visual inspection system for identifying steel surface defects should be conducted to secure the reliability of the process and the product.

Over recent years, a variety of research-based on machine learning and deep learning techniques have been conducted to establish defect diagnostics model of the steel surface with machine-vision, showing feasible performance for an automatic inspection system. For example, Jia, et al. [8] suggest a real-time surface defect detection method using a support vector machine (SVM) classifier, demonstrating the prediction accuracy of 85%. Especially, convolutional neural network (CNN) based detection methods are widely utilized as a way of the end-to-end framework for image processing, feature learning, and classification, achieving remarkable improvement in diagnosis performance [9]. More recently, Gao, et al. [10] propose a semi-supervised approach for steel surface defect recognition based on CNN. The method has better performances with 17.53% improvement compared to the baselines. Also, it has been applied in a real-world detection scenario with a limited labeled dataset. 

Although several studies have been conducted to enhance the defect detection performance in the steel surface, there are still challenging issues for practical use, which motivates this study. Firstly, the local binary patterns (LBP) method with SVM has merits of low computational complexity, meticulous descriptive quality, and illumination variation robustness [11]. Secondly, the optimization of a deep neural network-based model should be conducted. Tuning the hyper-parameters and building the optimized architectural structure should be carried out to maximize the classification performance for detecting the steel surface defects. Besides, training the network and an over fitting problem could be a practical issue while operating for both observed and unobserved data [12]. Although the deep learning-based models promoted the defects detection heavily, there is a room for improvement of detection speed and accuracy. In this work, an improved computational model named ResNet-50 & VGG16 convolutional neural network is proposed for surface defect images of hot rolled strip [13]. Furthermore, Generative adversarial networks (GANs) are algorithmic architectures that use two neural networks, pitting one against the other (thus the “adversarial”) in order to generate new, synthetic instances of data that can pass for real data. A new GAN-based classification method for detecting surface defects of strip steel is proposed. The GANs are used to generate more data sets in order to bloat the training images to make the model more robust [14].

## 1.1 Dataset

In the Northeastern University (NEU) surface defect database, six kinds of typical surface defects of the hot-rolled steel strip are collected, i.e., rolled-in scale (RS), patches (Pa), crazing (Cr), pitted surface (PS), inclusion (In) and scratches (Sc). The database includes 1,800 grayscale images: 300 samples each of six different kinds of typical surface defects.

Fig. 1 shows the sample images of six kinds of typical surface defects, the original resolution of each image is 200×200 pixels. From Fig. 1, we can clearly observe that the intra-class defects existing large differences in appearance, for instance, the scratches (the last column) may be horizontal scratch, vertical scratch, and slanting scratch, etc. Meanwhile the inter-class defects have similar aspects, e.g., rolled-in scale, crazing, and pitted surface. In addition, due to the influence of the illumination and material changes, the grayscale of the intra-class defect images is varied. In short, the NEU surface defect database includes two difficult challenges, i.e., the intra-class defects existing large differences in appearance while the inter-class defects have similar aspects, the defect images suffer from the influence of illumination and material changes.

![FzeYrqRveYRn.jpg](attachment:FzeYrqRveYRn.jpg)
Fig1: Defects on hot-rolled steel

For defect detection task, we provided annotations which indicate the class and location of a defect in each image. We have carefully clicked annotations of each target in these images. Fig. 2 shows some examples of detection results on NEU-DET. For each defect, the yellow box is the bounding box indicating its location and the green label is the class score [15].

![fig2.jpg](attachment:fig2.jpg)
Fig2: Detected defects on defect images

## 1.2 Dataset Preparation

The given data is categorised randomly into three subfolders namely Training, Testing, and Validation respectively. The training dataset consists of 276 images of each defect and 12 images are taken for testing and validation dataset each. 

# 2. Methodology

## 2.1. Support Vector Machines (SVM)

Support Vector Machines(SVM) is considered to be a classification approach but it can be employed in both types of classification and regression problems. It can easily handle multiple continuous and categorical variables. SVM constructs a hyperplane in multidimensional space to separate different classes. SVM generates optimal hyperplane in an iterative manner, which is used to minimize an error. The core idea of SVM is to find a maximum marginal hyperplane(MMH) that best divides the dataset into classes.

SVM works by mapping data to a high-dimensional feature space so that data points can be categorized, even though they are not linearly separable with existing dimension. As the dimensions of feature plane are increased for better categorization, the separators are no more lines but hyperplane. The mathematical function used for transformation of data to get hyperplane is Kernel functions [16].

![72598SVM%20%281%29.png](attachment:72598SVM%20%281%29.png)
Fig3: SVM Architecture

### 2.1.1 SVM on Trainig Data

Firstly, SVM is directly employed on training dataset available. The Image matrix is flattened and provided to SVM for training. The working of algorithm can be deduced by fig 4.

![SVM%20Working.png](attachment:SVM%20Working.png)
Fig 4. Working of SVM


### 2.1.2 SVM with Feature extraction (Local binary pattern)

There are two important issues to be addressed. First, effective defect features must be explored, extracted and optimized. Defect features characterize surface defects and determine the complexity of the classification system. The second important issue is the classifier design, which determines the performance of the entire vision system in classifying defects. We have developed a machine learning system based on the Support Vector Machine. During the design of the machine vision system, we address three main requirements: 

(1) it should be robust and capable of providing good discrimination even in the case of noisy input data. 

(2) It must be fast enough in order to meet inline speed requirements. 

(3) It should be incrementally scalable to incorporate new known objects, particularly new known defect classes, without retraining with the whole data set.

Local Binary Pattern (LBP) is a Texture pattern descriptor introduced by Ojala et.al[17] to describe the local texture patterns of an image. LBP is computed as a binary encoding of difference in pixel intensities with the local neighborhood. The process is illustated in following figure (Fig 5.)

![SVM%20LBP.png](attachment:SVM%20LBP.png)

### 2.1.3 SVM with Feature extraction (Local binary pattern) and Image Augmentation.

The SVM with LBP Feature Extraction method produced improved result compared to standard deployment. To further improve the model Image Augmentation Techniques such as rotation and flipping were employed on dataset and then augmeted images fed to SVM with LBP Feature Extractor. The reference process flow chart depicts the process of SVM with LBP Feature Extractor for augmented image dataset (Fig 6).

![SVM%20LBP%20Aug.png](attachment:SVM%20LBP%20Aug.png)

## 2.2 Convolutional Neural Networks (CNN - Hyper Paramter Tuning)

A convolutional neural network (CNN) is a type of deep neural network using successive operations across a variety of layers, which is specified to deal with a two-dimensional image. CNN, firstly introduced in [18], is known to be a successful neural network algorithm for image processing and recognition. The CNN architecture is typically made up of two main stages, i.e., feature extraction and classification, while it is learned to describe spatial information of the images across the layers. Extracted feature representations are fed into the latter part of the architecture, where the model draws a probability for belonging to a certain class. Likewise, weights and biases of the model are optimized by training the neural network via the back propagation algorithm.

There are conventionally three different types of layers in the CNN architecture, i.e., convolutional, pooling and fully-connected layer. The convolutional layer utilizes convolution operation to extract spatial features of the image, herein feature maps are computed by utilizing element-wise multiplication between the input image and the operator called kernel or filter. The pooling layer is carried out as a sub-sampling technique, followed by the convolution layer. It is aimed at downsizing the convoluted feature maps to reduce the number of trainable parameters, as well as to improve invariance for shift and distortion. A typical pooling method, i.e., max pooling, is used by taking the highest-value tensors from each certain region in the feature maps. Lastly, the fully-connected layer utilizes intensive features created through two types of layers, i.e., convolutional and pooling layers, for categorizing the input images into classes.

![Typical%20Arch%20of%20CNN.png](attachment:Typical%20Arch%20of%20CNN.png)

Keras Tuner is an easy-to-use, distributable hyperparameter optimization framework that solves the pain points of performing a hyperparameter search. Keras Tuner makes it easy to define a search space and leverage included algorithms to find the best hyperparameter values. Keras Tuner comes with Bayesian Optimization, Hyperband, and Random Search algorithms built-in, and is also designed to be easy for researchers to extend in order to experiment with new search algorithms.

In this project, we employed Bayesian Optimization technique for hyper-parameter tuning with labelled images. The model is trained several times for different activation function, input units and dropouts until best model is selected. The best model architecture is as follows-

![my_model.h5.png](attachment:my_model.h5.png)

## 2.3 Convolutional Neural Networks (CNN - Transfer Learning)

Transfer learning can be understood as learning knowledge or experience from previous tasks and applying them to new tasks. Based on the theory of transfer learning, we can achieve small-scale defect image data sets classification by decomposing and changing the large-scale CNN networks pre-trained on ImageNet [19].

In general, existing strip surface defect classification networks tend to use off-the-shelf deep learning network structures and their various variants, including AlexNet, VGGNet, GoogleNet, ResNet, DenseNet, SENet, ShuffleNet and MobileNet, etc. Compared with traditional algorithms such as machine learning, deep learning algorithms have higher accuracy; however, deep learning often requires a larger amount of data.


### 2.3.1 Data Augmentation

Data Augmentation is a way to ensure that the model doesn’t treat the object asdifferent in case it is subjected to small variations like rotation, variations in lightning conditions, etc. By applying the data augmentation, the diversityincreases and learning process is boosted as pointed in countless literature and it makes the model robust. The following data augmentation techniques were used:

    1. RandomContrast

    2. RandomFlip(mode="horizontal_and_vertical"),

    3. RandomZoom

    4. RandomRotation

    5. RandomTranslation


The models were subjected to 4 cases respectively:

1. VGG16/Resnet50 during training were not trained but the rest of the layers were
trained with no data augmentation.

2. VGG16/Resnet50 during training were not trained but the rest of the layers were
trained with data augmentation.

3. All layers of model including VGG16/Resnet50 were trained with no data
augmentation.

4. All layers of model including VGG16/Resnet50 were trained with data
augmentation.

This project proposes study of VGG16 and ResNet50 CNN Architectures on the training dataset. The best model architectures is as follows-
![VGG16_training_false_aug_false%20%282%29.h5.png](attachment:VGG16_training_false_aug_false%20%282%29.h5.png)

![Resnet50_training_false_aug_false%20%281%29.h5.png](attachment:Resnet50_training_false_aug_false%20%281%29.h5.png)

The models were iterated for 50 epochs with Adam optimizer. The learning rate of 0.0001 was set for modified Resnet50 model and 0.0005 for modified VGG16 model using the idea of learning rate finder from fast.ai. The model was run for 1 epoch with varied learning rates learning rate. The point where the slope changes maximum beforethe loss rises again is selected as the learning rate for faster learning.


## 2.4 Generative Adviseral Networks (GANs)

The lack of high-quality steel strip defect datasets makes the effectiveness of deep learning in steel strip defect classification somewhat limited. Thus, A new GAN-based classification method for detecting surface defects of strip steel is proposed. GANs work by training a generator network that outputs synthetic data. The generative model captures the data distribution of the training data to generate the synthetic data. This synthetic data is then run through the discriminator network to get a gradient. The discriminative model here estimates the probability if the synthetic data is from the training sample or from the generator [20]. The gradient indicates how much the synthetic data needs to change in order for it to look/represent a data that is more realistic than representing a data or an image that is synthetically generated.

### 2.4.1 Conditional GANs

The inspiration of the application of a conditional GANs network was solely due to the lack of diversity of data in the available data set. This resulted in the application of conditional GANs over the conventional GANs as it would have been advantageous for the project to generate synthetic data relevant for each class. Hence, targeted image generation was important which gave reason for the application of conditional GANs.
M. Mirza proposed the Conditional-GANs model which slightly modifies the conventional GAN architecture by adding a label parameter to the input of the generator on which the generator can condition on [21]. There is also an additional label parameter for the discriminator to distinguish the real data better.

There are various tutorials that help developers implement the conditional GANs using python, one of which was used in this project [22].  In this project, we merely have 300 images of size (200, 200) for each defect of hot-rolled steel. After defining the number of classes as six, the generator model function definition is created to accommodate the new size image. The model itself is a CNN network 3 convolution layers, with a couple of LeakyReLU activations. The general architecture can be described as the generation of an image for a particular label based on a combination of latent vector and its label (conditional GANs).

![GAN1.png](attachment:GAN1.png)

The discriminator architecture is responsible for classifying if the image is real or generated based on a probabilistic scale. This again is a combination of CNN consisting of convolution layers, LeakyReLU activation and the last layer being a sigmoid activation for classification after which training funcion with different iterations and batch sizes is defined.
![GAN2.png](attachment:GAN2.png)

# 3. Result and Discussion 

## 3.1 Support Vector Machines (SVM)

After training of each method proposed. The model is evaluated and deployed based on two performace metrics i.e. Classification Matrix and Classification Report

### 2.1.1 SVM on Trainig Data

The Classification Report and Confusion matrix are as follows:

### 2.1.2 SVM with Feature extraction (Local binary pattern)

The Classification Report and Confusion matrix are as follows:

### 2.1.3 SVM with Feature extraction (Local binary pattern) and Image Augmentation.

The Classification Report and Confusion matrix are as follows:

## 3.2 Convolutional Neural Networks (CNN - Hyper Paramter Tuning)

The prediction accuracy of model is illustrated in classification report for trainig and validation dataset, it can be seen that best model achieves the best classification performance with the accuracy of 95.83% with a loss of 0.099, and only 3 images were misclassified for test images and no misclassification for validation dataset. In order to show the classification effect of each kind of defects more carefully, the confusion matrix is shown. It can be seen that model model can achieve 100% of classification accuracy for all six defects for validation dataset.

![Classification%20Report.png](attachment:Classification%20Report.png)

![Confusion%20Matrix%20CNN%20HPT.png](attachment:Confusion%20Matrix%20CNN%20HPT.png)

## 3.3 Convolutional Neural Networks (CNN - Transfer Learning)

### 3.3.1 ResNet50

The model was explored for following 4 cases:

1. When Resnet50 was not trained but the rest of the layers were trained with no
data augmentation. (Training- False and data augmentation-False)

2. When Resnet50 was not trained but the rest of the layers were trained with data
augmentation. (Training- False and data augmentation-True)

3. When all layers including Resnet50 were trained with no data augmentation
(Training- True and data augmentation-False)

4. When all layers including Resnet50 were trained with data augmentation
(Training- True and data augmentation-True)

From the training and accuracy curves, the best performance was observed was when all the layers of the model were trained with and without augmentation. They are robust and the losses and accuracy on training and validation set converges.


![ResNet50.png](attachment:ResNet50.png)

### 3.3.2 VGG16

The model was explored for following 4 cases:

1. When VGG16 was not trained but the rest of the layers were trained with no
data augmentation (Training- False and data augmentation-False)

2. When VGG16 was not trained but the rest of the layers were trained with data
augmentation (Training- False and data augmentation-True)

3. When all layers including VGG16 were trained with no data augmentation
(Training- True and data augmentation-False)

4. When all layers including VGG16were trained with data augmentation (TrainingTrue and data augmentation-True)

From the training and accuracy curves, the best performance was observed when the layers of the model except VGG16 were trained with and without augmentation. They are robust and the losses and accuracy on training and validation set converges. The model got stuck at local minima during training when VGG16 was included in the training.


![VGG16.png](attachment:VGG16.png)

## 3.4 Generative Adviseral Networks (GANs)

Unfortunately the results of the generated images did not serve the required purpose of bloating the training image set. There were different methods that were attempted:

1) Just feed one class training data and see if the generated image were close the real ones

2) Change the optimizer to see if the results get better

3) Use of data augmentation methods while training the GAN model

None of the above attempts succeeded in generating relevant images for the defects. Below is an example of different images generated for the defect Crazing.

![GAN.png](attachment:GAN.png)

# 4. Conclusion

SVM alone had poor prediction accuracy when directly employed on training dataset. But, the prediction accuracy of SVM increased drastically in couple with feature extractors LBP. Moreover, the accuracy is further increased by 5% when data augmentation is performed on dataset resulting 94% accuracy. Since the results are not satisfacotry deep learning methods like CNN with Hyperparameter Tuning and Transfer learning are employed.

The Hypetuned CNN model performed 95.83% accurately on test dataset with augmented training dataset. On the other hand, it has performed exceptionally for validation dataset. In order to reduce the model size and computational power the transfer learning approach is selected for same results with lowcost model.

Two transfer learning models were employed on data viz. VGG16 and ResNet50 for different conditions. VGG16 when not trained with the rest of the layers with no data augmentation turned out to be the best model achieving 100% accuracy on test dataset. 

The usage of cGANs, although understandable, did not result in the best of results. This can be primarily attributed to a few reasons:

1. GANs in itself require a lot of data. Considering just 1800 images for 6 classes resulted in an underperforming model
2. The GAN model could have been constructed from scratch drawing inspirations from latest models that have worked well with less data
3. Other generative models could have been explored

To conclude, Machine Learning and Deep Learning approaches were discussed in this project for the classification of surface defects of hot-rolled steel. Support Vector Machines performed better with augmented images and LBP feature extractor. Moreover, CNNs outperformed SVM in terms of accuracy and loss rate. The hypertuned model was computationally expensive hence transfer learning approach is selected, which showed same result. The GANs, on the other hand, did not performed as per expectation and could be improved with other generative models. 

# References

1. Song, G.W.; Tama, B.A.; Park, J.; Hwang, J.Y.; Bang, J.; Park, S.J.; Lee, S. Temperature Control Optimization in a Steel-Making Continuous Casting Process Using Multimodal Deep Learning Approach. Steel Res. Int. 2019, 90, 1900321.
2. Luo, Q.; He, Y. A cost-effective and automatic surface defect inspection system for hot-rolled flat steel. Robot. Comput.-Integr. Manuf. 2016, 38, 16–30.
3. Ghorai, S.; Mukherjee, A.; Gangadaran, M.; Dutta, P.K. Automatic defect detection on hot-rolled flat steel products. IEEE Trans. Instrum. Meas. 2012, 62, 612–621.
4. He, Y.; Song, K.; Meng, Q.; Yan, Y. An End-to-end Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features. IEEE Trans. Instrum. Meas. 2019.
5. Liu, K.; Wang, H.; Chen, H.; Qu, E.; Tian, Y.; Sun, H. Steel surface defect detection using a new Haar–Weibull-variance model in unsupervised manner. IEEE Trans. Instrum. Meas. 2017, 66, 2585–2596.
6. Chen, W.; Gao, Y.; Gao, L.; Li, X. A New Ensemble Approach based on Deep Convolutional Neural Networks for Steel Surface Defect classification. Procedia CIRP 2018, 72, 1069–1072. 
7. Song, K.; Yan, Y. A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects. Appl. Surf. Sci. 2013, 285, 858–864.
8. Jia, H.; Murphey, Y.L.; Shi, J.; Chang, T.S. An intelligent real-time vision system for surface defect detection. In Proceedings of the 17th International Conference on Pattern Recognition, ICPR 2004, Cambridge, UK, 26 August 2004; Volume 3, pp. 239–242.
9. Park, J.K.; Kwon, B.K.; Park, J.H.; Kang, D.J. Machine learning-based imaging system for surface defect inspection. Int. J. Precis. Eng. Manuf.-Green Technol. 2016, 3, 303–310.
10. Gao, Y.; Gao, L.; Li, X.; Yan, X. A semi-supervised convolutional neural network-based method for steel surface defect recognition. Robot. Comput.-Integr. Manuf. 2020, 61, 101825.
11. Luo, Qiwu; Fang, Xiaoxin; Sun, Yichuang; Liu, Li; Ai, Jiaqiu; Yang, Chunhua; Simpson, Oluyomi (2019): Surface Defect Classification for Hot-Rolled Steel Strips by Selectively Dominant Local Binary Patterns. In IEEE Access 7, pp. 23488–23499.
12. Lee, Soo Young; Tama, Bayu Adhi; Moon, Seok Jun; Lee, Seungchul (2019): Steel Surface Defect Diagnostics Using Deep Convolutional Neural Network and Class Activation Map. In Applied Sciences 9 (24), p. 5449.
13. Wang, Wenyan; Lu, Kun; Wu, Ziheng; Long, Hongming; Zhang, Jun; Chen, Peng; Wang, Bing (2021): Surface Defects Classification of Hot Rolled Strip Based on Improved Convolutional Neural Network. In ISIJ Int. 61 (5), pp. 1579–1583.
14. K. Liu; A. Li; X. Wen; H. Chen; P. Yang (2019): Steel Surface Defect Detection Using GAN and One-Class Classifier. In : 2019 25th International Conference on Automation and Computing (ICAC). 2019 25th International Conference on Automation and Computing (ICAC), pp. 1–6.
15. Kechen Song and Yunhui Yan (2022): NEU-surface-defect-database. Available online at https://www.kaggle.com/datasets/rdsunday/neu-urface-defect-database, updated on 4/12/2022, checked on 9/28/2022.
16. Taha, Bahauddin (2021): Build an Image Classifier With SVM! In Analytics Vidhya, 6/18/2021. Available online at https://www.analyticsvidhya.com/blog/2021/06/build-an-image-classifier-with-svm/, checked on 9/28/2022.
17. T. Ojala, M. Pietikainen and T. Maenpaa, "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 24, no. 7, pp. 971-987, July 2002.
18. Haralick, R.M.; Shanmugam, K.; Dinstein, I.H. Textural features for image classification. IEEE Trans. Syst. Man Cybern. 1973, 3, 610–621.
19. J Deng, W Dong, R Socher et al., "Imagenet: A large-scale hierarchical image database[C]", 2009 IEEE conference on computer vision and pattern recognition, pp. 248-255, 2009.
20. reddit (2022): r/MachineLearning - Generative Adversarial Networks for Text. Available online at https://www.reddit.com/r/MachineLearning/comments/40ldq6/generative_adversarial_networks_for_text/, updated on 9/28/2022, checked on 9/28/2022.
21. Mirza, Mehdi; Osindero, Simon (2014): Conditional Generative Adversarial Nets. Available online at https://arxiv.org/pdf/1411.1784.
22. Saxena, Pawan (2021): Synthetic Data Generation Using Conditional-GAN - Towards Data Science. In Towards Data Science, 8/12/2021. Available online at https://towardsdatascience.com/synthetic-data-generation-using-conditional-gan-45f91542ec6b, checked on 9/28/2022.
