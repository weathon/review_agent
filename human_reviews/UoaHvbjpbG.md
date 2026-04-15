# Semantic-Enhanced Prototypical Network for Universal Novel Category Discovery

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5

## Abstract
We address the challenging task of Universal Novel Category Discovery (UniNCD) in image classification, where models must distinguish between common and novel categories while avoiding the misclassification of novel categories as private-known ones. Previous prototype-based approaches face two major challenges: first, they significantly increase the negative transfer risk by often misaligning novel categories with private-known categories; second, they lead to sub-optimal prototypes because traditional prototype learning ignores diverse object characteristics of images, resulting in insufficient semantic guidance when optimizing instance representations using only instance-level prototypical distributions. To tackle these challenges, we present a Semantic-Enhanced Prototypical Network, dubbed SEPNet. This prototypical network is enhanced by refined prototypes and enriched semantics to learn better representations and avoid negative transfer, including three key ideas: (1) we design a Prototype Refinement (PR) strategy that can decouple common, private-known, and novel categories from unlabeled data, which can exclude misaligned prototypes to avoid negative transfer; (2) we attach prototypical distribution to each patch of images, which embed enhanced semantic information to prototypes and guide prototypical contrastive learning and, (3) we design a patch-entropy balance (PEB) method to encourage sparser patch-level prototypical distributions while maintaining the uniformity of dense distributions, sparsity emphasizes dominant category characteristics, and uniformity avoids the misguidance of irrelevant disturbance, thereby enhancing the distinctiveness of instances to the prototypes. Our method demonstrates superior performance on the UniNCD task across three benchmark datasets, outperforming existing state-of-the-art approaches by approximately 3.4% in terms of accuracy. We will release our code for reproduction.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper aims to address Universal Novel Category Discovery (UniNCD) in image classification. In the UniNCD setting, the dataset categories are practically divided into three groups: "private known," "common," and "novel." Specifically, the authors have introduced a Prototype Refinement (PR) strategy, which is used during initial training to prevent negative transfer. Recognizing that finer-grained patches may contain more semantic information, the authors have associated prototypical distributions with each patch of an image. To enhance distinctiveness, the authors have introduced the patch-entropy balance loss to optimize the model. Extensive experiments demonstrate the effectiveness of this approach.

### Strengths
1.	In comparison to previous settings, the UniNCD setting considered in this paper is more realistic. To tackle this challenge, the authors have introduced a novel method called SEPNet. The experimental results demonstrate the effectiveness of their method.

2.	The writing and supplementary materials of this paper are well-constructed, enhancing the readers' understanding.

3.	The utilization of patch-level information to enhance semantic knowledge in contrastive learning is both interesting and effective. I believe this approach can serve as inspiration for future research.

### Weaknesses
1.	To balance the trade-off between smaller and larger patches, this paper employs a 2D pooling operation on the patches. However, it's worth noting that such a pooling operation may result in the loss of finer-grained information. To further investigate this trade-off, I think it is necessary to conduct corresponding experiments. 

2.	Treating an image as a composition of patches is a common operation in computer vision tasks [1-2]. To enhance the novelty of this paper, further modification strategies should be applied. I think that certain filtering strategies could be effective in such task, particularly for addressing various meaningless patches (e.g., the background). 

3.	The basic idea of PEB is that a concept word can be utilized to describe one image. However, what I find interesting is whether PEB can be meaningful in a multi-classification task. Further experiments could be conducted in such a scenario to explore the effectiveness of PEB.

4.	The pre-trained parameters could potentially impact the experimental results. To provide a more comprehensive illustration of the effectiveness of the PR strategy, I recommend comparing PR with other methods in the experiments.

5.	Based on the Z-score, PR distinguishes the misaligned prototypes from others. However, it may make erroneous decisions when dealing with intra-class diversity and inter-class similarity. Regarding these difficult samples, experiments should be conducted to further illustrate the effectiveness of PR.

6.	The overview in Figure 3 should be revised to include additional details about your pipeline. I found it confusing to understand the descriptions of PCL and PEB in this figure. Furthermore, it is advisable to carefully review the symbols (e.g., C_n) used in this paper to minimize unnecessary comprehension difficulties for the reader.

[1] Liu, Ze, et al. "Swin transformer: Hierarchical vision transformer using shifted windows." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
[2] Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).

### Questions
See the above.

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a prototypical network (SEPNet) for the task of universal novel category discovery, which has a more practical setting compared to the standard NCD problem. The proposed SEPNet finds the optimal match between labeled and unlabeled datasets, and distinguishes the private and common known classes by utilizing a threshold based on matched prototypes' distances. Furthermore, the paper also develops a patch-entropy balance loss to promote sparsity in patch-level prototype distributions and maintains the uniformity of dense ones. The proposed method is evaluated on three benchmarks with comparison to existing approaches.

### Strengths
1. This paper introduces a new setting of Novel Category Discovery, which aims to address a more practical and challenging problem.
2. The proposed semantics enrichment strategy seems to be novel for the novel class representation learning.
3. Extensive experiments demonstrate the method's effectiveness, showcasing a significant performance gap compared to existing approaches.

### Weaknesses
1. The novelty of the proposed method is limited. The proposed method can be decoupled into two parts: 1. Prototypes Alignment. Apart from filtering private-known classes, the rest of this part is very similar to [1]. It is unclear how the proposed method differs from [1] in prototype alignment. 2. Patch-Entropy Balance for Semantic Enrichment. As explained below, $\mathcal{L}_{PEB}$ is not specifically designed for UniNCD, and using unsupervised learning loss in NCD is not new.  

2. The motivation for introducing two parts is unclear. First, the Patch-Entropy Balance for the Semantic Enrichment module appears less relevant to the UniNCD task. In particular, the proposed loss 
can be applied to any representation learning problem.  Second, the ablation shows that without $\mathcal{L}_{PEB}$, the SEPNet is worse than many other methods like PromptCAL and DPN w/PR.  It seems to indicate that the Prototype Refinement is less necessary for the overall framework. 

3. The robustness of the threshold hyperparameter is unclear. What if the split ratio between private-known classes and common-known classes is not 3:1? It is more convincing to provide some experiments when the split ratios changes for validating the effectiveness of the model.

4. In traditional NCD & GCD tasks, there are many experiments conducted on fine-grained datasets such as Standford Cars, CUB-200-2011 and FGVC-Aircraft. Can the authors also provide similar experiment results on these fine-grained datasets?

5. The paper lacks clarity in multiple places:
 - What is the cluster label $c^u_{i,j}$ in Eq.6b?
 - What does $U_c$ represent in Eq.7?
 - The meaning of $x^j_i$ is unclear in Eq.8.
 - There are several typos in Eq.12, Eq.14, and Eq.16. 
 - What is the definition of known category ratio in Appendix E? Does the ratio between private-known classes and common-known classes keep constant while the known category ratio changes? Moreover, what does "PARSE" represent in Fig 7?


[1] Wenbin An, Feng Tian, Qinghua Zheng, Wei Ding, QianYing Wang, and Ping Chen. Generalized category discovery with decoupled prototypical network. In Proceedings of the AAAI Conference on Artificial Intelligence, number 11, pp. 12527–12535, 2023.

### Questions
See detailed comments in Weaknesses section.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a new task, namely Universal Novel Category Discovery (UniNCD), which involves the concept of private-known classes that is ignored by existing NCD methods. To address the negative transfer issue caused by these private-known classes, the authors proposed a prototypical contrastive learning based method with refined prototypes and enriched semantics. Extensive experiments have demonstrated the effectiveness of the proposed method.

### Strengths
- The proposed method is novel and evaluated to be highly effective.
- The experiments are comprehensive.

### Weaknesses
- The motivation of the new task seems unclear to me. For example, why holding out some private known classes is significant in the real world? Also, what does this "universal" mean?
- The computational requirement of the proposed method seems to be much more demanding than current SOTAs. For example, as another ViT-based approach, GCD only needs to train the last block of the ViT, but training the whole ViT model seems a necessity for the proposed method, if I understand correctly. 
- The writing is somewhat hard to follow. There are many standalone modules and their own acronyms, making the paper not easy to understand. Also, the proposed method seems complicated, and the narration of the methodology is in lack of high-level intuitions and justification of the design choices.

### Questions
The proposed PEB seems to be the largest contribution. What is its intuition? Why balanced patch distribution is good? Also, in Eq 7 there is already an entropy loss term that helps rectify P towards the uniform distribution, then why another KLDU is necessary, and vice versa? If they have different impacts, apart from the theoretical analysis, is there any empirical proof?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
