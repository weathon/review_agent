# Learning A Disentangling Representation For PU Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 5

## Abstract
In this paper, we address the problem of learning a binary (positive vs. negative) classifier given Positive and Unlabeled data commonly referred to as PU learning. Although rudimentary techniques like clustering, out-of-distribution detection, or positive density estimation can be used to solve the problem in low-dimensional settings, their efficacy progressively deteriorates with higher dimension due to the increasing complexities in the data distribution. In this paper we propose to learn a neural network-based data representation using a loss function that can be used to project the unlabeled data into two (positive and negative) clusters that can be easily identified using simple clustering techniques, effectively emulating the phenomenon observed in low-dimensional settings. We adopt a vector quantization technique to the learned representations to amplify the separation between the learned unlabeled data clusters. We conduct experiments on simulated PU data  that demonstrate the improved performance of our proposed method compared to the current state-of-the-art approaches. We also provide some theoretical justification for our two cluster-based approach and some of our algorithmic choices.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new PU learning method. Specifically, the authors develop a loss function that can be used to project the unlabeled data into two (positive and negative) clusters that can be easily identified. They adopt a vector quantization technique for the learned representations to amplify the separation between the learned unlabeled data clusters.

### Strengths
1.	The studied problem in this paper is very important.
2.	The experiments are sufficient.

### Weaknesses
1.	The authors claim that the existing PU learning methods will suffer a gradual decline in performance as the dimensionality of the data increases. It would be better if the authors can visualize this effect. This is very important as this is the research motivation of this paper.
2.	Since the authors claim that the high dimensionality is harmful for the PU methods, have the authors tried to firstly implement dimension reduction via some existing approaches and then deploy traditional PU classifiers?
3.	In problem setup, the authors should clarify whether their method belongs to case-control PU learning or censoring PU learning, as their generation ways of P data and U data are quite different. 
4.	The proposed algorithm contains Kmeans operation. Note that if there are many examples with high dimension, Kmeans will be very inefficient.
5.	The authors should compare their algorithm with SOTA methods and typical methods on these benchmark datasets.
6.	The figures in this paper are in low quality. Besides, the writing of this paper is also far from perfect.

### Questions
see the weakness part.

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper works on PU learning and proposes a new representation learning method for it. The paper uses a codebook to store representations and forces P and U data to be similar to different codebook vectors, respectively. Then, they use a K-means algorithm to cluster feature representations and derive the classifier. Experiments validate the effectiveness of the proposed approach.

### Strengths
- The paper is well written.
- The idea of introducing codebook representations into PU learning is novel.

### Weaknesses
- It is still unclear to me why the proposed method works for PU learning. Although the authors provided some theoretical explanations, I am still not clear why the proposed method can separate feature representations of P and N data.
- The proposed method is influenced by the center representations of P and U data ($\mu_P$ and $\mu_U$). If the two representations are too close, there seems to be no guarantee that the method will work well. 
- In Eq.(6), the authors claim that they do not need $\alpha$. However, they still need to know the labels of the unlabeled data. But if we know the labels of the unlabeled data, we can calculate $\alpha$. So I do not think the analysis is useful here.
- The experiment design is too simple. The authors should include more experiments, such as more compared approaches, and more experimental settings (such as different $\alpha$). The current experiments are too simple to validate the effectiveness of the proposed approach.

### Questions
- Why does the proposed method work well? 
- Is the method affected by the feature separability of the training data?
- Can the authors add more experiments to verify the proposal?

### Soundness
1 poor

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
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors in this paper focus on positive-unlabeled (PU) learning and attempt to encode positive and unlabeled instances into a more discriminative representation space followed by a simple cluster method, such as K-means. They directly apply the existing vector quantization technique to project the unlabeled data into two distinct clusters. The experimental results show the effectiveness of the vector quantization method.

Though the idea of learning a disentangling representation for PU learning may be interesting, applying the existing vector quantization technique directly limits the contribution of this paper.

### Strengths
-	This paper is well-written and quite easy to follow.
-	The experimental results and ablation study show the effectiveness of the proposed method.

### Weaknesses
-	The innovation of this paper seems to be limited. In this paper, the authors directly employ the exited vector quantization technique [1] to learn a disentangling representation for PU learning with little modification. Though the idea of learning a disentangling representation for PU learning may be interesting, the contribution of this paper is very limited. Otherwise, there lacks of reference to the original paper “Neural discrete representation learning” [1] of the vector quantization technique.
-	There lack of some current PU approaches as baselines in experiments, such as Robust-PU [2], Dist-PU [3], P3Mix [4].
-	Equation (1) misses a “)”, and should be $sg(\mathbf{v}_j(\mathbf{x}_{i_p};\theta))$.

[1] Aaron Van Den Oord, and Oriol Vinyals. "Neural discrete representation learning." Advances in neural information processing systems 30 (2017).

[2] Zhangchi Zhu, Lu Wang, Pu Zhao, Chao Du, Wei Zhang, Hang Dong, Bo Qiao, Qingwei Lin, Saravan Rajmohan, and Dongmei Zhang. "Robust Positive-Unlabeled Learning via Noise Negative Sample Self-correction." In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 3663-3673. 2023.

[3] Yunrui Zhao, Qianqian Xu, Yangbangyan Jiang, Peisong Wen, and Qingming Huang. 2022. Dist-PU: Positive-Unlabeled Learning From a Label Distribution Perspective. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 14461–14470.

[4] Changchun Li, Ximing Li, Lei Feng, and Jihong Ouyang. 2022. Who is your right mixup partner in positive and unlabeled learning. In International Conference on Learning Representations.

### Questions
Please see the weakness for details.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a simple method to solve the problem of learning a binary classifier with positive and unlabelled data. The proposed method is based on vector quantiza- tion technique to perform dimension reduction first, and then apply standard k-means algorithm to cluster the unlabelled data into positive and negative 2 clusters. In addition to the experimental evaluation, the paper provides some math intuition and ablation study to support and explain how the proposed method works. In the experiment section, the paper shows that the proposed method can produce comparable results w.r.t state-of-the-art GAN based methods.

### Strengths
The paper is well written and easy to understand. The results sound good and perhaps easy to re-produce if the authors can publish their code.

### Weaknesses
1. The idea is simple and the novelty may not be strong enough to publish in such a high standard conference.
2. The k-means algorithms need to keep running in each iteration. Although the idea is simple, it will be very slow if the data size is huge.
3. The proposed method is not convinced to handle the case when the labels are imbalanced.

### Questions
1. According to figure 4, the proposed method seems to fall into an interesting situation where the validation is good but the center of two clusters are closer after more epochs. Can author explain the reason? 
2. Can the proposed method handle imbalanced labelled data? This happens in many real situations, such as CTR prediction. Typically clicks are much less than impressions. However, there will be lots of inventory that may not become impressions and therefore there is no label associate to it.
3. Would the proposed algorithm sensitive to the initialization of the cluster center?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
