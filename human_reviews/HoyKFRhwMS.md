# Towards flexible perception with visual memory

- Decision: Reject
- Scores: 6, 3, 6, 8

## Abstract
Training a neural network is a monolithic endeavor, akin to carving knowledge into stone: once the process is completed, editing the knowledge in a network is nearly impossible, since all information is distributed across the network's weights. We here explore a simple, compelling alternative by marrying the representational power of deep neural networks with the flexibility of a database. Decomposing the task of image classification into image similarity (from a pre-trained embedding) and search (via fast nearest neighbor retrieval from a knowledge database), we build a simple and flexible visual memory that has the following key capabilities:
(1.) The ability to flexibly add data across scales: from individual samples all the way to entire classes and billion-scale data;
(2.) The ability to remove data through unlearning and memory pruning;
(3.) An interpretable decision-mechanism on which we can intervene to control its behavior.
Taken together, these capabilities comprehensively demonstrate the benefits of an explicit visual memory. We hope that it might contribute to a conversation on how knowledge should be represented in deep vision models---beyond carving it in "stone" weights.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a flexible and interpretable visual memory system that integrates deep neural networks with a database-like structure to improve image classification. By separating knowledge representation from the model’s weights, the system enables easy addition, removal, and modification of data without retraining, addressing the limitations of static knowledge in traditional models. Key features include scalable data management, machine unlearning, and transparent decision-making processes. The system uses a k-nearest neighbour retrieval mechanism and RankVoting to enhance sample aggregation for improved performance, highlighting the potential of using visual memory in deep learning models to adapt to dynamic data and evolving knowledge.

### Strengths
1. The paper is well-written and easy to follow.
2. Investigation into different voting strategies looks interesting.
3. The idea of using visual memory database + retrieval is a promising strategy for visual recognition, given its flexibility, OOD robustness, and adaptation capacity to ever-growing concepts.

### Weaknesses
1. One limitation of retrieval and search engines is that the cost of storage and inference increases and the accuracy of recognition decreases as the memory database size grows. Although computing budget and accuracy can be traded off, this remains a limitation. Additionally, regarding the computation cost, I wonder if there is a rough estimate of the inference time when the database size increases to 1 billion images.
2. The adaptation capacity of the proposed visual memory to accommodate ever-growing concepts assumes that the image encoder can produce meaningful embeddings for new concepts. While for geometrically distinctive concepts, I believe this is less of a concern for DINO representations, as they are observed to contain rich geometric information, for concepts where class label correlates more with semantics rather than geometry, I wonder if the adaptation capacity still holds.

### Questions
The result of using Gemini re-ranking is quite strong and interesting. I wonder if this is due to VLM's in-context learning capacity. A discussion on this can be helpful.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper makes the argument that performing classification through nearest-neighbour search (in an embedding space) has many more desirable properties than more common approaches to classification found today - namely: that its much easier to add new data (and new classes) as this doesn't require retraining, that things (both images and classes) can easily be unlearned/removed, and the decisions might be more interpretable. The authors demonstrate that embeddings from recent pre-trained (vision-language) models are effective for use in a knn context and with appropriate weighting functions can outperform linear probing.

### Strengths
Overall, the paper is well written and easy to read if somewhat whimsical in style. The paper does highlight well some of the problems with current approaches to image classification, particularly in the context of deployment of models where the training data might change (for various reasons) regularly. The proposed approach of using weighted KNN seems to work reasonably, and the use of a scalable NN search engine helps with scaling to large numbers of instances in the memory. 

A wide range of different aggregation methods are tested, and there are comparisons against standard linear probing baselines. I like the experiments that look at OOD performance and this is particularly interesting.

### Weaknesses
I think the biggest problem I have is that the contribution of the paper is not really clear - almost everything presented is rather obvious and based on well studied approaches; use of NN methods for classifying images based on features dates at least back to the days of Turk & Pentland (https://sites.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf) in 1991 (if not before), and research on fast nearest neighbour approaches for solving KNN problems based on image features also has a long history (e.g. Sivic's VideoGoogle from the early 2000s was using inverted indexes for object matching/detection - https://www.robots.ox.ac.uk/~vgg/publications/2003/Sivic03/sivic03.pdf). Similarly, there is much research on KNN aggregation methods, including many that utilise rank (e.g. http://www.jcomputers.us/vol6/jcp0605-01.pdf & there are even articles suggesting reciprocal rank: https://visualstudiomagazine.com/articles/2019/04/01/weighted-k-nn-classification.aspx), so the contribution of the proposed RankVoting scheme is not clear. That new "memories" can be added/removed seems "obvious".

Experimentally, I feel there are quite a few problems. Firstly, it's not clear how the extreme class-imbalance in datasets like imagenet are taken into account; its non-obvious to me that it even makes sense to use a fixed value of $k$ in such cases. Secondly, Many of the presented results also do not show a clear win for any particular method (e.g. table 1 - difficult to claim RankVoting is better than softmax; table 2 - quite a mixed bag).

In the cases where a linear probe has been trained its not clear what procedure was adopted and how optimal the learned decision boundary is; was this a linear SVM or just a linear layer trained with gradient descent? does that not make a difference? 

The pruning experiments are interesting, but are they not rather highlighting problems with the data (labels)? Presumably, more generally, this could be an approach for improving data quality rather than specifically improving the approach presented (although of course if does reduce computational complexity).

The paper doesn't seem to give any thought to the complexity of the proposed approaches - how big is the database of embeddings? how long does retrieval take? How does this compare to state of the art classification models (e.g. inference time, and perhaps comparing number of model weights to feature-extractor model weights + database size)?

### Questions
As per my comment above, have the authors considered the effect of class-imbalance in the training data and what this means with respect to $k$ and potential accuracy? When you get improvements in performance where do these come from?

What is the set-up for the linear probes? How was this validated?

Can you comment on the computational efficiency aspects? how does it scale with instances? How long does a query take? How much storage is required?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces a "visual memory" framework for image classification, aiming to make perception more adaptable by decoupling representation from memory. By combining image similarity from pretrained embeddings with fast nearest-neighbor retrieval, this framework allows flexible handling of knowledge updates, enabling data addition, memory pruning, and unlearning. The proposed RankVoting aggregation method surpasses previous methods and achieves high accuracy across ImageNet and billion-scale datasets. The work illustrates the potential of visual memory in achieving continual learning without re-training while supporting interpretability and control in deep learning models.

### Strengths
1. Combines the strengths of deep neural networks with database principles, enabling flexible memory operations such as adding, removing, and unlearning data at scale.
2. Achieves high accuracy and flexibility, outperforming baseline voting methods, especially with RankVoting and Gemini-based reranking.
3. Supports continual learning, efficient memory management, and interpretability, addressing practical needs like model drift and data privacy through machine unlearning.

### Weaknesses
1. Limited application focus on other visual tasks (e.g., object detection, segmentation) beyond classification. How might the visual memory approach be adapted for tasks like object detection or segmentation? What unique challenges would arise in those domains?
2. Reliance on a fixed pretrained embedding model may limit adaptability under significant distribution shifts. Have you considered approaches to update the embedding model incrementally as new data becomes available? This could help address potential limitations in adaptability under distribution shifts.
3. Sensitivity of SoftmaxVoting to hyperparameters could impact performance stability.
4. Limited ablation studies on the impact of visual memory size across various neural network architectures.
5. No detailed analysis on memory retrieval latency and computational overhead for large-scale memory databases. Could you provide more information on the memory retrieval latency and computational overhead, especially for the billion-scale experiments? This would help readers understand the practical implications of scaling up the visual memory approach.
6. Lack of performance benchmarking across diverse dataset distributions.
7. Absence of discussions on handling adversarial attacks or noise within the visual memory framework. How robust is the visual memory approach to adversarial examples or noisy data? It would be valuable to see an analysis of how the system performs when the memory contains corrupted or adversarially crafted samples.
8. Limited interpretability assessment for users to control specific decision boundaries. Could you provide more details on how users might interact with or control the decision-making process in your visual memory framework? Specifically, how might one adjust decision boundaries or influence the model's predictions in practice?
9. Potential limitations in scalability for different embedding models with high memory footprints.
10. Insufficient exploration of non-image applications (e.g., text-based memory) within the flexible memory paradigm.

### Questions
1. Could the framework extend to tasks like segmentation or object detection, and if so, how would visual memory be adapted?
2. How does RankVoting handle adversarial or noisy samples in memory retrieval? Would further methods mitigate this?
3. Would memory retrieval latency impact real-time performance, especially in billion-scale databases?
4. What would be the computational cost for re-ranking with Gemini, particularly on constrained devices?
5. Could the proposed system be combined with incremental learning techniques to improve distribution shift resilience?
6. How might RankVoting's performance vary with different embedding models?
7. What measures could prevent sensitive or incorrect data from being memorized, impacting model reliability?
8. Are there optimizations to manage storage costs of billion-scale datasets without affecting retrieval accuracy?
9. Could memory pruning methods be customized for specific tasks, improving model performance further?
10. How does the model handle cases where similar but incorrect neighbors dominate the nearest neighbors?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors address the memory-based image classification task by utilizing large, unsupervised pretrained visual models to extract image descriptors. They perform classification based on nearest neighbor retrieval in the embedding space of these models. A key contribution is their novel aggregation approach called _RankVoting_, which outperforms state-of-the-art methods. Unlike previous methods, _RankVoting_ improves performance as the number of considered neighbors increases. Through various experiments, the authors demonstrate the advantages of visual memory-based classification over traditional model-based classification, highlighting benefits such as flexibility, scalability, and interpretability.

### Strengths
* __Presentation.__ The paper is written in a highly engaging and accessible manner, capturing the reader's interest from the outset. It is well-positioned within the context of related works, providing a clear understanding of how it contributes to the field. The novelty of the research is evident, and the authors effectively communicate the motivation behind each experiment.

* __Experimental design.__ The authors have conducted a comprehensive set of experiments that excellently demonstrate the advantages of memory-based image classification over traditional model-based approaches. They show how their method easily scales to accommodate new data, illustrating its effectiveness in scenarios requiring continual learning. Furthermore, they highlight the flexibility of their approach in machine unlearning and memory pruning, showcasing how it can efficiently remove or adjust the influence of specific data points. The interpretability of their method is also emphasized, providing clear insights into the decision-making process of the model.

* __Novelty.__ The main highlight of the paper is the introduction of the novel neighbor aggregation method called RankVoting. This method is clearly and thoroughly described, with an intuitive rationale that makes its underlying principles easy to grasp. The authors present solid empirical evidence of its advantages over state-of-the-art distance-based softmax voting methods, demonstrating superior performance, especially as the number of considered neighbors increases. This contribution significantly advances the field by providing a more effective way to aggregate information from nearest neighbors in neighbor-based classification.

### Weaknesses
While some may argue that the methodological novelty of the paper is modest—introducing only a single, simple formula to slightly improve kNN classification—I believe that this advancement is substantial in its own right. The elegance and simplicity of the _RankVoting_ method support its effectiveness, providing a meaningful enhancement to nearest neighbor classification. Moreover, the authors contribute significantly to the broader discussion of neighbor-based versus model-based image classification. They present strong, well-articulated arguments that underscore the advantages of visual memory-based approaches. The paper not only provides a practical improvement but also stimulates important discourse that could shape future research directions.

### Questions
In Table 2, for the ImageNet-A dataset, the performance achieved through linear probing surpasses even that of the JFT memory combined with Gemini reranking. This is not the case for the other datasets. Initially, I considered that this anomaly might indicate some form of overfitting. However, later in the paper you mention potential issues with the ground-truth labels in the ImageNet-A dataset, which could affect the trend.

Did you observe similar issues with any of the other datasets used in your experiments? Moreover, do you think it is possible to quantify this problem—for instance, by assessing how reassigning or correcting problematic labels influences the performance?

### Soundness
4

### Presentation
4

### Contribution
3
