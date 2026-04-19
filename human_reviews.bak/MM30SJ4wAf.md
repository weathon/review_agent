# Point Neighborhood Embeddings

- Decision: Reject
- Scores: 6, 8, 3, 5

## Abstract
Point convolution operations rely on different embedding mechanisms to encode the neighborhood information of each point in order to detect patterns in 3D space. However, as convolutions are usually evaluated as a whole, not much work has been done to investigate which is the ideal mechanism to encode such neighborhood information. In this paper, we provide the first extensive study that analyzes such Point Neighborhood Embeddings (PNE) alone in a controlled experimental setup. From our experiments, we derive a set of recommendations for PNE that can help to improve future designs of neural network architectures for point clouds. Our most surprising finding shows that the most commonly used embedding based on a Multi-Layer Perceptron (MLP) with ReLU activation functions provides the lowest performance among all embeddings, even being surpassed on some tasks by a simple linear combination of the point coordinates. Additionally, we show that a neural network architecture using simple convolutions based on such embeddings is able to achieve state-of-the-art results on several tasks, outperforming recent and more complex operations. Lastly, we show that these findings extrapolate to other more complex convolution operations, where we show how following our recommendations we are able to improve recent state-of-the-art architectures.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper gives a comprehensive study on a variety of existing point neighborhood embedding in a controlled setting. Accordingly, it concludes several practical suggestions regarding designing the neighborhood embedding. The paper validates it’s suggestions in the recent point transformer and show the improvement.

### Strengths
The paper provides a comprehensive and extremely insightful investigation about the point aggregation module -- the core in point cloud architecture where, given a target point, how we aggregate information from other source points. 

This paper fills a blank of this point convolution area -- i.e.Which type of point aggregation is the best? From my viewpoint, most existing works on point cloud architecture like PointNet++ PointConv and KPConv are trying to improve this convolution module in terms of different subtle details like neighborhood querying, kernel function etc. Even the recent point transformer falls in this scope by replacing conv-based aggregation with attention-based way under a mild assumption that the feature of target point is known whereas the point conv doesn’t require it.  Although those works gradually improve the performance on the common benchmark, comprehensive study is still missing -- in other words, different modules are not compared in a controlled setting -- and leads to the best design choice unknown. Because of this, I’ve been bothered so many times by having no idea about which modules are better. So from my perspective, this paper fixes this problem in this area and I believe it can inspire future works like mine. 

The derived suggestions are very useful. It includes: no ReLU in MLP, ball-query better KNN and kernel point is better than MLP. For the latter two, I think it’s not surprising as I also observe this. But it’s still useful to have a fair comparison to show it. And for the first one, it’s something new to me. 

Also, from my viewpoint,  the formula presented by the author is easy to follow. I would consider using the same formula -- it summarizes the most of existing convolution in a unified way-- when I teach a lecture about point convolution next time.

### Weaknesses
The paper has several weaknesses which I’ll detail below. 

More modules could be considered -- for example like the very powerful PAConv that defines the convex kernel function. Also the recent PointMLP [1] defines an interesting local aggregation module that relies on normalization. I think the author can further improve the paper by including more options. 

The paper utilizes the KPConv with fixed kernel point location which is suboptimal in original KPConv paper. So I would recommend discussing how the deformable KPConv fits in the formula. This would make the paper more comprehensive. 

I’m not very sure if I can agree with the argument made in Eq. 1 where the Point transformer is not a convolution. From my perspective,  it also has a similar formula if we consider the query point’s features. But this is my personal viewpoint. I wouldn’t convince authors to convey point transformers in a convolution way. However, given that the author doesn’t consider the point transformer as a kind of convolution, it’s a bit weird to validate the proposed tips derived from convolution into the point transformer. So I think the author might want to clarify it a bit in the paper. 

The paper didn’t use the ModelNet40. Although it’s a synthetic dataset that is highly saturated by lots of existing methods, I believe that it’s still a proper dataset to fairly investigate the different subtle modules in point convolution. Or the paper needs to justify why the paper doesn’t use the popular ModelNet40 in my opinion. 

[1]Ma, Xu, et al. "Rethinking network design and local geometry in point cloud: A simple residual MLP framework."  ICLR 22.

### Questions
Please address the questions above

### Soundness
3 good

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a comprehensive study in the performances of various Point Neighborhood Embeddings (PNE) mechanisms in point convolutional neural network architectures, and further offers recommendations for improving model designs based on the findings. They validated that their recommendations can outperform most existing methods in several tasks with simple design and can improve existing complex convolution operations.

### Strengths
1. Their findings and recommendations can benefit future arch design of point clouds
2. They did comprehensive experiments to explore the different design choices and validate their recommendations.
3. The writing is good with detailed introduction and analysis.

### Weaknesses
-

### Questions
-

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents extensive study that analyzes point cloud embeddings based on activation functions used, correlation functions used, MLP vs Kernel points embeddings, etc. The paper also talks about different convolution operation and the neighbourhood election based on ball query vs k-NN. The authors performed two downstream tasks classification and segmentation on two benchmark datasets.

### Strengths
1. The paper is well-written and provides a good analysis of point cloud embeddings. 
2. This can help build new algorithms/architectures to improve embeddings. 
3. The study provides interesting results.
4. The architecture includes simple modification and not expensive operations like transformers.

### Weaknesses
1. This paper can be seen as a good experimental study paper which is not up to the level of ICLR. This is like a review paper although in a different direction where it provides extensive study on multiple points. 
2. The paper is very weak in novelty and makes some claims without evidence or explanations except results. 
3. The work mentions a lot of comparisons between activation functions, MLP vs. KP, etc. However, the whole paper lacks in explaining “why” something is better or worse. For example, kNN vs BQ talks about having high variance with kNN. It does not explain why. 
4. Support of the embedding is not clearly defined. I believe the receptive field is neighbourhood. 
5. There is no fixed proposed architecture in the paper. The results are not generalized based on a particular setting. The architecture used is the existing architecture mentioned in 5.1 with some changes like activation function, using KP/MLP, and different correlation function based on KP embeddings.

### Questions
Why validation set is used for PNE and the validation+test set is used for other methods?

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper discusses different types of embeddings and aggregation methods for neighboring 3D point clouds. By experimenting with different combinations on ScanObjNN and ScanNet, the author summarizes a set of best practices for designing new PNE.

### Strengths
1. The review of previous 3D point cloud embeddings is comprehensive and logical.

2. The experiments and analysis on two tasks (classification and segmentation) are careful and in-depth.

### Weaknesses
The improvement is not obvious compared to existing methods in Table 2 compared to 3. Does this mean the effect of PNE is less important when training data is less? I recommend to experiment on larger-scale 3D datasets for classification, such as Objverse, or other 3D tasks, such as indoor/outdoor 3D object detection. The two experiments on the paper is not sufficient enough to demonstrate the conclusion.

### Questions
One related work investigating Sin for 3D classification is expected to be included and discussed: *Starting from Non-Parametric Networks for 3D Point Cloud Analysis* accepted by CVPR 2023.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
