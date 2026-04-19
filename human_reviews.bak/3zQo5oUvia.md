# REBAR: Retrieval-Based Reconstruction for Time-series Contrastive Learning

- Decision: Accept (poster)
- Scores: 8, 5, 6, 5

## Abstract
The success of self-supervised contrastive learning hinges on identifying positive data pairs, such that when they are pushed together in embedding space, the space encodes useful information for subsequent downstream tasks. Constructing positive pairs is non-trivial as the pairing must be similar enough to reflect a shared semantic meaning, but different enough to capture within-class variation. Classical approaches in vision use augmentations to exploit well-established invariances to construct positive pairs, but invariances in the time-series domain are much less obvious. In our work, we propose a novel method of using a learned measure for identifying positive pairs. Our Retrieval-Based Reconstruction (REBAR) measure measures the similarity between two sequences as the reconstruction error that results from reconstructing one sequence with retrieved information from the other. Then, if the two sequences have high REBAR similarity, we label them as a positive pair. Through validation experiments, we show that the REBAR error is a predictor of mutual class membership. Once integrated into a contrastive learning framework, our REBAR method learns an embedding that achieves state-of-the-art performance on downstream tasks across various modalities.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a time-series contrastive learning framework that replaces the data augmentation module with a retrieval-based pair construction strategy. The idea sounds interesting and is proved to be effective on three time-series datasets.

### Strengths
1. This work proposes a retrieval-based mask reconstruction strategy to help the model identify similar time series, which I think is a smart design.
2. The authors show that using contiguous and intermittent masks during the training and evaluation respectively leads to the best performance. Such a result could bring some new insights to the time-series learning community.
3. By constructing contrastive pairs retrieval, the proposed method does not rely on data augmentations, which could harm the pattern of signals, to perform contrastive learning. Experiments on three datasets demonstrate the effectiveness of the proposed method.

### Weaknesses
1. Figure 4 shows that the diagonal pattern is worse on the PPG and ECG data compared with that on the HAR data. Some explanations need to be provided here to help readers understand the potential limitations of the method.
2. During the contrastive learning stage, the positive counterpart is selected as the one most similar to the anchor. However, it is possible that there is more than one candidate that shares the same class label with the anchor. Would such false negative pairs influence the performance of contrastive learning? Have you tried other positive selecting strategies such as hard threshold?
3. Typo: we uniformly at random sample -> we uniformly random sample

### Questions
Please refer to the weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This article proposes a new perspective for determining positive and negative samples in time series contrastive learning. If one subsequence can be successfully reconstructed by retrieving information from another, it should form a positive pair. Based on this, the author trains a cross-attention module to reconstruct the masked query input subsequence from the key input subsequence. The subsequence with the lowest reconstruction error is labelled as positive, and the others are labelled as negative. Experiments show that the REBAR method of this article achieves state-of-the-art results in learning a class-discriminative embedding space.

### Strengths
The method proposed in this article is intuitive and easy to understand.

### Weaknesses
1. Format issue: All formulas in this article are not numbered. The notations in the first formula on page 5 have no corresponding definition.

2. The experimental volume of this paper is insufficient. As a new perspective in the field of time series contrastive learning, the author should validate the method on a wider dataset to demonstrate its universality for time series.

3. The article lacks discussion and analysis of key parameters. Specifically, when training the cross-attention module, what impact will the length of subsequences and the proportion of random masking have on the reconstruction effect of key subsequences? Is the reconstruction effect of the cross-attention module directly related to downstream task performance? How to set the number of candidate subsequences (proportion of positive and negative samples) when obtaining Pos/Neg labels?

### Questions
I noticed that when applying the REBAR metric in contrastive learning, an anchor sequence and n candidate sequences are sampled randomly. Only the candidate sequence with the smallest reconstruction loss will be determined as a positive sample of the anchor sequence. Is there a situation where, for example, when sequence A is used as an anchor sequence, the candidate sequence with the smallest reconstruction loss is sequence B, and then A and B are mutually positive samples? However, when B is used as an anchor, the candidate sequence with the smallest reconstruction loss may be another sequence C. Therefore, B and C are positive samples. But if A is also in the candidate sequences, this method will divide A into negative samples of B. This leads to conflicting conclusions when the anchor sequence is different.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a novel method for constructing positive pairs for contrastive learning in time-series data. It presents experiments across three datasets to validate the approach.

--- post rebuttal ---

I appreciate the efforts made by the authors in addressing the concerns raised in my initial review. The manuscript has undergone significant changes, resulting in notable improvements in its quality. Considering these enhancements, I have revised my score from 3 to 6.

### Strengths
1. The paper is motivated. The proposed method is grounded on a cogent hypothesis *"if one time-series is useful for reconstructing another, then they likely share reconstruction information and thus should be drawn together within the embedding space as positive examples."*. Essentially, the author posits that time-series with similar semantics are capable of aiding in each other's reconstruction.

2. An intriguing observation made in the paper is the difference in sparsity between cross-attention mechanisms when trained with *contiguous masks* versus *intermittent masks.* 

3. The author provides a comprehensive comparison, including many relevant baselines.

### Weaknesses
1. The rationale for preferring a contiguous mask over an intermittent mask is presented but could be articulated with greater clarity to enhance its persuasiveness. Additionally, there seems to be some confusion regarding Figure 1. Clarification is needed as to whether the author implies that a) the contiguous mask is utilized during the training of REBAR, and b) the intermittent mask is employed when applying REBAR in contrastive learning. If this is the case, the reasons for using different masks in these contexts should be explicitly stated.

2. The experimental scale appears somewhat limited. The paper does not specify the exact number of samples within the datasets, which seem to be on the smaller side. This limitation is accentuated when compared to previous works, such as TS2VEC [1], which utilized a much larger array of datasets, including 125 from the UCR archive and 29 from the UEA archive.

3. The explanation of results requires expansion. For instance, the acronyms ARI and NMI in Table 2 are not defined within the context of the paper, leaving their significance unclear. Moreover, there is a notable difference in the results reported for the TNC on the HAR dataset between the original TNC paper [2] and this manuscript. In the origianl paper, it was reported AUPRC 0.94, Accuracy 88 while in this manuscript, it is reported AUPRC 0.98 and Accuracy 94. More information about the potential factors leading to these discrepancies would be beneficial for the reader's comprehension.


* [1] Yue, Zhihan, et al. "Ts2vec: Towards universal representation of time series." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 8. 2022.

* [2] Tonekaboni, Sana, Danny Eytan, and Anna Goldenberg. "Unsupervised representation learning for time series with temporal neighborhood coding." arXiv preprint arXiv:2106.00750 (2021).

### Questions
In Table 3, the author evaluates the influence of different mask types on performance. It would be beneficial to clarify why the training stage favors a contiguous mask, while the evaluation stage shows a preference for an intermittent mask.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a novel approach called Retrieval-Based Reconstruction (REBAR) for self-supervised contrastive learning in time-series data. The REBAR method utilizes retrieval-based reconstruction to identify positive data pairs in time-series, leading to state-of-the-art performance on downstream tasks.

### Strengths
1. Novel approach: The paper introduces a novel approach called Retrieval-Based Reconstruction (REBAR) for self-supervised contrastive learning in time-series data. This approach utilizes retrieval-based reconstruction to identify positive data pairs in time-series, which is a unique and effective way to address the challenges of creating positive pairs via augmentations in time-series data.

2. State-of-the-art performance: The paper demonstrates that the REBAR method achieves state-of-the-art performance on downstream tasks across diverse modalities, including speech, motion, and physiological data.

3. Comprehensive evaluation on two tasks including classification and cluster agreement.

### Weaknesses
1. Lack of ablation studies: The paper does not include ablation studies to analyze the contribution of each component of the REBAR method. This makes it difficult to understand the relative importance of each component and how they interact with each other.

2. Detailed explanation of the results. There is no detailed studies for table 1 and 2, e.g., visualizations of the learned embedding or positive/negative pairs. 

3. Limited discussion of hyperparameters: While the paper provides some details about the hyperparameters used in the experiments, it does not provide a comprehensive analysis of the sensitivity of the method to different hyperparameters. 

4. Without comparison with baselines, Figure 4 doesn't show any advantages of the proposed model since the diagonal pattern would be obvious for most of the baselines.

5. Section 3.1, notations are used without clear definition

### Questions
1. Ablation study and hyperparameters selection.
2. Include more visualizations or examples of the positive and negative pairs identified by the REBAR method
3. "During evaluation, we use an an intermittent mask", explain the intuition why different masks are used in the training and evaluation
4. How does the REBAR method perform on time-series data with different characteristics, such as varying lengths or noise levels?
5. Provide more detailed explanations of the convolutional cross-attention architecture used in the REBAR method.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
