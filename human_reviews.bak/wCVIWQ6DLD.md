# A Simple and Efficient Baseline for Data Attribution on Images

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 3, 5

## Abstract
Data attribution methods play a crucial role in understanding machine learning models, providing insight into which training data points are most responsible for model outputs during deployment. However, current state-of-the-art approaches require a large ensemble of as many as 300,000 models to accurately attribute model predictions. These approaches therefore come at a high computational cost, are memory intensive, and are hard to scale to large models or datasets. 
In this work, we focus on a minimalist baseline, utilizing the feature space of a backbone pretrained via self-supervised learning to perform data attribution. Our method is model-agnostic and scales easily to large datasets. We show results on CIFAR-10 and ImageNet, achieving strong performance that rivals or outperforms state-of-the-art approaches at a fraction of the compute or memory cost. Contrary to prior work, our results reinforce the intuition that a model's prediction on one image is most impacted by visually similar training samples. Our approach serves as a simple and efficient baseline for data attribution on images.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper develops a data attribution method, for attributing classification performance of a neural network on a given sample to training samples, that provides a better efficiency-accuracy trade-off compared to existing attribution methods. The proposed method computes the affinity of a given evaluation image with all images in the training set, in the latent space of a pretrained neural network and under a particular metric. The paper provides experiments on CIFAR-10 and ImageNet showing on-par performance with state-of-the-art data attribution methods using only a fraction of their memory and computational budget. The main finding of the paper is that relying only on visual similarity is effective in discovering the smallest set that affects the classification performance of a given image.

### Strengths
The paper tackles an interesting problem, namely making data attribution practical, and provides an insightful comparison with state-of-the-art data models. The writing is also clear and easy to follow.

### Weaknesses
1- The main scientific finding of this paper is an expected result: that training images with high visual similarity to an evaluation image are important for correctly classifying that image. If this is a surprising finding (the abstract seems to claim it is), the paper should try to emphasize the arguments against it in prior works, and discuss the fault in those arguments. As it stands, I find its main scientific contribution not very significant.

2- While the main practical contribution of the paper, its method, cannot outperform state-of-the-art in the studied datasets, the paper correctly explains that it is more practical. However, the paper does not provide any real-world experiments to show the usefulness of its method in an application. This makes it hard to judge the significance of the proposed method in practice.

3- The paper does not explain its related works in sufficient detail, and as a result, a lot of the methodologies it borrows from prior works is unclear. In particular, a detailed discussion of metrics and their relation to the considered metrics is missing. The Appendix provides some more detail, specially on LDS, which I think must be part of the main paper.

### Questions
My three concerns mentioned in the weaknesses section contain my suggestions.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper studies (and revisits) the effectiveness of similarity-based baselines for the problem of data attribution---understanding
how training data influence model predictions. The paper introduces two new metrics based on brittleness, and evaluate their baseline relative to recent SOTA methods (datamodels, TRAK) in terms of performance and other cost measures (time, memory).

### Strengths
- The paper is very written and placed well within the context of prior work (though there are some misleading claims throughout, which I highlight below).
- The experiments and evaluations are thorough and well documented

### Weaknesses
At a high level, I have serious concerns about the claims made in the paper and the way the overall message is presented:

- The paper claims that the model agnosticity of their approach (and similarity based methods) is a feature, but I strongly disagree. Data attribution at a most basic level is about understanding why the given specific model/algorithm behaves, so it *has* to be model dependent. Otherwise, just by definition, the method cannot capture any biases unique to the model/algorithm (and as many prior works show, different models have very different biases, for example CNNs vs ViTs, etc.). The paper acknowledges this point in passing, but I think this needs to be much more prominent.

Now, as the authors acknowledge, it's possible that there is a largely "model-independent" component that can account for model behavior (it's not crazy to think that different DNNs leverage data in similar ways), but I think it's a long jump to conclude this just from the brittleness metrics (I expand on this below).

- The paper also continues to propagate the misunderstanding in some prior works in this area that similarity is same as influence, which is just not true! This is not even true when you consider the simplest case of a linear model: there, the similarity is given by the natural euclidean product, whereas the influence looks and behaves very differently (it involves the inverse of a gram matrix).

- Also, a similarity-based approach cannot readily surface negative influencers, or even assign relative weights to the positive influencers. It's almost misleading to call even call the approach "attribution" when you cannot assign quantitative weights to examples (that reflect their counterfactual importance). Similarity (used directly) can only capture the relative ordering among positive influencers. 

- The issue of metric: while I agree that brittleness-based metrics are also informative, it seems misleading to base most of the paper's claims on two new metrics, which do not capture the points I mentioned above (negative influencers and calibration among positive influencers), while delaying discussion of metric (linear datamodeling score) considered in prior works to the Appendix.

In in fact, their evaluation shows that the similarity baseline only achieves an LDS of 0.05 on CIFAR-10, which is hardly significant. 
TRAK ([2], App. E.3) using just 5 independently models can achieve an LDS of 0.329. It might be true that at the same level of minimal compute (a single checkpoint), the baseline method outperforms prior methods. But importantly, the proposed approach method cannot improve even with more compute! (prior work[1] indeed shows that more "ensembling" has marginal effect on similarity-based approaches).
So one could claim that at the certain level of budget, the proposed approach is the best performing, but it is very misleading to say throughout the paper that this simple approach also beats SOTA, which is clearly not the case (and only shown in the Appendix).

All of these concerns considered, I think a more reasonable (and less misleading) conclusion of the paper would have been more along:
similarity-based approaches can be an effective baseline; rather than "similarity is all you need" message that seems more prominent and is misleading given the various reasons above.

Other concerns:
- Not sure why space efficiency is a big consideration at all. Storing both embeddings (for computing similarities) and storing projected gradients (for a single checkpoint) both require same order of memory. 

[1] Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Madry. "Datamodels: Predicting Predictions from Training Data."
[2] Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, Aleksander Madry. "TRAK: Attributing Model Behavior at Scale
"

### Questions
Concerns were raised above.

### Soundness
2 fair

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper focuses on the problem of data attribution, i.e., estimating how individual training data points influence model outputs. The main contribution of this paper is a compute-efficient and storage-efficient data attribution baseline for images.  Specifically, the baseline identifies “important” training data points via visual similarity using self-supervised feature extractors (i.e., distance in embedding space).  The experiments suggest that this baseline can outperform existing data attribution methods in identifying small data-removal and data-mislabel support sizes.

### Strengths
- The paper is well-written in that the problem setting is clear and the experiments (figures, plots, etc) are easy to follow. The design choices in the proposed baseline are described in detail as well.
- The experiments show that for standard image classification tasks, visually similarity (measured via distance in some embedding space) can surface training data points with high positive influence on model outputs.

### Weaknesses
- “Our work shows that strong data attribution can be achieved solely based on knowledge of the training set.” In general, the effect or influence of a training data point on a model output is a function of the learning algorithm used to train the model. The data attribution scores of a standard ERM classifier trained on CIFAR would be very different from a random CIFAR classifier. The proposed method, however, would output the same attribution scores for the random classifier and an ERM classifier trained on CIFAR. This is an issue for at least two downstream applications.
    - As noted in the paper, data attribution can be used for debugging model biases (https://arxiv.org/abs/2211.12491) where you want to compare data attributions of two learning/training algorithms (e.g., with and without data augmentation) and see how data changes in the learning algorithm change data attributions. This method, however, would output the same data attribution scores for both algorithms, so it cannot be used to compare data attribution scores in general.
    - Another application of data attribution is to identify backdoor attacks in training data (https://arxiv.org/abs/2307.10163), as data attributions of backdoored test examples would rely on backdoored training examples. By relying solely on visual similarity (not a good proxy in this case) and not the learning algorithm, the proposed baseline is unlikely to succeed in identifying visually dissimilar + backdoored training points that have high influence.
- This method only focuses on examples with high positive influence. In general, data attribution methods identify data points with high influence, but also data points with ~zero influence and negative influence. Identifying points with almost no influence can be used to prune the dataset, whereas identifying points with negative influence can be used to understand what in the training dataset causes a model to misclassify a test example. The appendix suggests a heuristic to identify negative influence, but it is unclear if this works as well as the data brittleness experiments because the LDS score is quite low (0.05) and Figure 7 only visualizes the positive influencers. One concrete way to check this would be to identify how many negative influencers need to be removed to make an incorrectly classified test point correct.
- The method makes a strong implicit assumption: The data attribution score of training example j and test example i does not depend on other training examples in the dataset. However, if there are multiple copies of training example j in the dataset, then the influence of each copy is down-weighted. Intuitively, this is because the effect of removing a single copy on the model output is small if there are other copies in the training dataset that aren’t removed. The proposed method does not account for this, so it will not estimate the influence of individual training data points in scenarios like the one above.
- The actual method identifies high-influence datapoints by comparing visual similarity of a test example to other examples in the same class. This implicitly assumes that training data points from other related classes cannot be positive influencers. However, even for standard image classification tasks, it is possible to have training data points with positive influence that do not belong to the same class. Furthermore, given that this heuristic is “critical” (S3.1) it is unclear how one would extend this method to vision tasks (e.g. data attribution for CLIP) that does not have a fixed class set.

### Questions
Writing is vague at times; a few examples below. It would be great to get some clarity about these statements:
    - “It is important to highlight that while Datamodels and TRAK outperform our baseline in terms of LDS with extensive model ensembles, this metric provides limited insights into understanding machine learning models.”
    - “Thus, the latter metric [data removal and data mislabel support size] serves as a better proxy [than LDS] for the data attribution method’s usefulness as a debugging tool.”

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper uses self-supervised models as a feature extractor for data attribution. It matches or outperforms previous baselines (TRAK and data models) on CIFAR and ImageNet while being much more efficient in terms of compute and storage.

### Strengths
- Results are comparable to the baselines, while reducing cost of data attribution
- Shows that a straight forward approach that was previously abandoned can be effective by using self supervised models
- Architecture transfer ablation is important
- Presentation is clear

### Weaknesses
- Unsure of some of the assumptions made in the paper - see questions
- Lacks quantitative comparison with baselines of samples chosen - could be good to compute some statistics across this approach and the baselines. For example, are the subsets chosen by this approach more similar to or different than TRAK and data models

### Questions
1. Why is smallest subset the right thing to do? Isn't it possible for there to be two disjoint subsets of different size that both can cause a misclassification? In this case, shouldn't there be some attribution to samples in both groups?

2. Is there an explanation for why self-supervised extractors work better for attribution, and why DINO is an exception?

3. Subsetting on images of the same class is justified by empirical results, but in ImageNet there are classes that are very similar. How can you be sure that there is no attribution in this case?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good
