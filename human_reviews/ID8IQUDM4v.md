# Distributed Linear Dimensionality Reduction Assisted by Centralized NN for Classification

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 1, 3, 3, 3

## Abstract
Linear dimensionality reduction is a widely used technique in data compression, especially under computationally-constrained platforms. This paper presents a linear dimensionality reduction technique tailored for distributed edge devices, balancing resource constraints like data-rate and computing power at the device side, while ensuring high classification accuracy at the server side. The core concept of our approach is the simultaneous training of a unique single-layer for each distributed device, determined by its compression needs, coupled with a centralized deep neural network on the server for all-device classification. A standout feature of our approach is its adaptability: when integrating a new device aiming to compress data in an untrained dimension, only minimal training for the device's initial two layers is needed, leaving the server's  centralized deep neural network and the compression layers for all existing devices untouched. Additionally, our findings indicate that the peak accuracy attainable through our method approaches that of the optimal accuracy achievable by the ideal Maximum Likelihood classifier, outperforming traditional matrix decomposition-based techniques like Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA). Compared to distance-metric-based strategies like Neighborhood Component Analysis (NCA), our technique offers a marked reduction in training complexity for large datasets. Experimental studies show that our approaches result in significant improvements in classification accuracy under the same data-rate requirements compared to existing linear dimensionality reduction approaches on real data sets.

## Human Reviews

## Human Reviewer 1

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper focuses on a classification scenario that the central server will take as input compressed data from distributed devices and then execute classification on those data. The paper claimed that the traditional linear dimension reduction methods like PCA and LDA cannot achieve good results especially when the distributed devices have different compression rates. Therefore they proposed to use a trainable linear transformation to accomplish the compression.

### Strengths
- The paper is easy to understand and the proposed method is quite simple.

### Weaknesses
- The method is too simple to be published in a top-tier machine learning conference. 
- I personally have some concerns about the experimental results.   
Please refer to the questions part. Thanks.

### Questions
Major Problems:
- Theoretical Section (3.1): The theoretical content in Section 3.1 appears to be somewhat redundant and perhaps unnecessary. In the current landscape of neural network research, discussions on the global optimality of neural networks have gained prominence [1,2]. Theorems 1 and Proposition 1 do not seem to introduce any particularly unique or insightful content compared to these existing discussions.
- Experimental Settings: The choice of experimental settings raises some concerns. Since the paper deals with classification tasks, it is imperative to conduct experiments using well-established neural network models such as ResNet and Transformer, rather than Random Forest and K-Nearest Neighbors. Additionally, using more standard and widely recognized classification datasets like CIFAR10, CIFAR100, and ImageNet would be also necessary for evaluation. The choice of MNIST and a simple face classification dataset might be considered less suitable (it's weird to choose a face classification dataset as well). 
- Experimental Results: The experimental results presented in the paper are not entirely convincing. The lack of details on how PCA or LDA was employed in the experiments is a significant concern. To ensure fairness and clarity in the experiments, it is essential to compare the proposed method with PCA by only switching the $W_1$ to $W_N$ in Figure 3 to PCA, keeping both the DNN part and the transformations in the server side unchanged, and then training the neural network with the PCA-based data. Moreover, PCA, while not trainable like the proposed method, is still a linear transformation, and the paper should justify the observed test error gap between the two methods.

Minor Problems:
- Reference Format: The format of the references appears to be problematic, with the reference text mixed with the main text. I suggest reviewing and correcting the reference format to ensure it aligns with standard citation conventions.
- Paper Title: The paper title may require reconsideration. The current title suggests that the proposed reduction method is assisted by the centralized neural network. However, it might be more appropriate to frame it as the centralized neural network being assisted by the proposed reduction method in adapting to data with varying compression rates.

[1] Haeffele B D, Vidal R. Global optimality in neural network training, CVPR2017.    
[2] Sun R. Optimization for deep learning: theory and algorithms[J]. arXiv preprint arXiv:1912.08957, 2019.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a linear dimensionality reduction method for distributed edge devices, balancing resource constraints like data-rate and computing power at the device side, while ensuring high classification accuracy at the server side. The proposed method conducts the simultaneous training of a unique single-layer for each distributed device, determined by its compression needs, coupled with a centralized deep neural network on the server for all-device classification. When integrating a new device aiming to compress data in an untrained dimension, only minimal training for the device’s initial two layers is needed, leaving the server’s centralized deep neural network and the
compression layers for all existing devices untouched.

### Strengths
1. The paper is well-written and easy to follow.

2. The proposed method has correct derivations.

### Weaknesses
1. It is unclear this method is useful in distributed learning. Actually there is no practical application for this proposed method. There is no need for linear dimensionality reduction. The deep neural networks conduct the nonlinear way and can achieve better performance.

2. The experiments were conducted on extremely small dataset with a small number of devices.

### Questions
The experiment section is very weak. In edge computing, we expect the system has large data and many devices.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This submission proposed to conduct data compression in client device and the compressed data are transfered to server device for leraning. Compression is performed by linear projection into various dimension in different clients. The server unifies the dimension by using a fully-connected layer for each client, then performan training for all data.

### Strengths
N.A.

### Weaknesses
It seems the proposed method contains few novelty: data compression by linear projection for transmission and re-projection for training is a very straight forward idea.

### Questions
I don't have question currently. Please clarify my concern on novelty.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper introduces a linear dimensionality reduction technique specifically designed for distributed edge devices. The primary goal is to balance the constraints of data-rate and computing power on the device side while ensuring high classification accuracy on the server side. The approach involves training a unique single-layer for each distributed device based on its compression needs. The paper claims that the accuracy achieved through this method is close to the optimal accuracy of the Maximum Likelihood classifier, outperforming traditional techniques like PCA and LDA. Additionally, the method offers reduced training complexity for large datasets compared to distance-metric-based strategies.

### Strengths
1. The method allows for the easy integration of new devices without the need to retrain the entire system.

### Weaknesses
1. The evaluation is only performed on very small scale dataset, which is a toy dataset for modern NN system. It's not persuasive for the effectiveness of proposed methodology, especially under such a practical application scenario. I would suggest use larger dataset like images for autonomous driving, multi-dimensional time series data, etc.
2. For the problem setting in section 2, why is this topic important? why is this problem challenging?
3.  I did not see much technical merits of the proposal methodology. I would suggest the author highlight the technical contribution, conclude it with an illustrative figure and explain with plain words.
4. There is no testing performed on real devices. We cannot see the improvement of efficiency.

### Questions
1. What are the popular datasets for this domain and the popular testbeds/devices for the problem?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor
