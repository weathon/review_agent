# Semi-Supervised Contrastive Learning with Orthonormal Prototypes

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Contrastive learning has emerged as a powerful method in deep learning, excelling at learning effective representations through contrasting samples from different distributions. However, dimensional collapse, where embeddings converge into a lower-dimensional space, poses a significant challenge, especially in semi-supervised and self-supervised setups. In this paper, we propose CLOP, a novel semi-supervised loss function designed to prevent dimensional collapse by promoting the formation of orthogonal linear subspaces among class embeddings. Through extensive experiments on real and synthetic datasets, we demonstrate that CLOP improves performance in image classification and object detection tasks while also exhibiting greater stability across different learning rates and batch sizes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work has two main contributions. First, it identifies a mechanism of representation collape from repulsive force, instead of only gravitational force. Second, it identifies a solution simple solution which mitigates the problem: randomly generating fixed orthonormal protypes assingned to classes. This proves to be quite effective on numerous classfication and object detection tasks, including semi-supervised learning and learning with unbalanced labels.

### Strengths
* The analysis is interesting, in particular showing that repulisive forces cause collapse and correspond to stationary points. 
* The results are fairly strong on semi-supervised tasks and imbalanced label tasks. 
* Findings about needing smaller batch sizes are also interesting.

### Weaknesses
* Transfer learning and object detection results are much more marginal than semi-supervised and imbalanced label results. 
* CLOP doesn't seem to be useful self-supervised learning. 
* (Minor) introduces another hyper-parameter, increasing tuning cost, though it appears to be stable.

### Questions
* Would it make sense to do some kind of smart assingment of prototypes at the beginning of training?
* Is there some kind of extension that can be applied for self-supervised learning?
* On a related note, this work seems related to the protytypes in SwAV?  [1] It could be good to discuss the similarities and differences of these works. 

[1] Caron, Mathilde, et al. "Unsupervised learning of visual features by contrasting cluster assignments." Advances in neural information processing systems 33 (2020): 9912-9924.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The work presents a semi-supervised learning loss function called CLOP that initializes orthonormal vectors as many as the number of classes in order to draw the similarity of class specific embedding functions towards these vectors to enable better separation and mitigate collapse. This is added as a regularization term to the InfoNCE loss function. Empirical and visual analyses of the method against known methods are provided.

### Strengths
1. The method is well performant.

2. The regularizer is modular and could serve as an addendum to other methods.

3. Tackles a known issue of dimensional collapse.

### Weaknesses
1. The proposed contribution is an incremental one that combines existing notions of orthonormalization with standard contrastive learning without introducing any new theoretical insight.

2. Comparisons against known approaches [1] for this problem aren't conducted.

3. Initialization of prototypes in an orthonormal manner may be misguided since several concepts or classes in datasets may be semantically very related.

4. The theoretical contribution may be a restatement from known work [1].


[1] Jing, Li, et al. "Understanding dimensional collapse in contrastive self-supervised learning." arXiv preprint arXiv:2110.09348 (2021).

### Questions
1. Is the Lemma 1 a known result that all-equal or co-linear embeddings are stationary points? [1, 2]

2. Could this work be a special case of the work in SWAV wherein the the unsupervised clustering and assignment mechanism is replaced with fixed, label-anchored prototypes? [3]


[1] Jing, Li, et al. "Understanding dimensional collapse in contrastive self-supervised learning." arXiv preprint arXiv:2110.09348 (2021).

[2] Wang, Tongzhou, and Phillip Isola. "Understanding contrastive representation learning through alignment and uniformity on the hypersphere." International conference on machine learning. PMLR, 2020.

[3] Caron, Mathilde, et al. "Unsupervised learning of visual features by contrasting cluster assignments." Advances in neural information processing systems 33 (2020): 9912-9924.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces CLOP, a semi-supervised contrastive learning method that mitigates dimensional collapse by aligning embeddings with orthonormal class prototypes. The approach theoretically and empirically shows that conventional contrastive losses (e.g., InfoNCE) suffer from degenerate optima and that enforcing orthogonal subspaces preserves representational diversity. CLOP consistently outperforms prior methods like SupCon and SimMatch across CIFAR and ImageNet benchmarks, showing strong robustness to small batch sizes and high learning rates. The paper is clearly written, well-motivated, and experimentally solid, making it a strong accept candidate despite assuming fixed class structures.

### Strengths
1.	Solid Theory – Offers a clear theoretical analysis explaining why InfoNCE leads to dimensional collapse and how orthogonal prototypes can prevent it.
2.	Novel Loss Design – Proposes CLOP, a simple yet effective loss that enforces orthogonality among class embeddings to maintain diversity.
3.	Strong Empirical Results – Demonstrates consistent gains over baselines like SupCon and SimMatch on CIFAR and ImageNet.
4.	Robustness – Performs stably under large learning rates and small batch sizes, avoiding collapse seen in prior methods.
5.	Practical Implementation – Easy to integrate into existing contrastive frameworks with minimal computational overhead.
6.	Clear Presentation – Well-written, logically structured, and easy to follow with theory and experiments well aligned.

### Weaknesses
1.	Fixed Prototype Assumption – CLOP assumes a fixed number of well-separated classes and static orthonormal prototypes. How would the method adapt to open-set or hierarchical label scenarios where class structures evolve over time?
2.	Limited Scope of Evaluation – All experiments are in vision-based benchmarks (CIFAR, ImageNet). Can CLOP generalize to non-visual domains such as text, graphs, or multimodal tasks, if you don't have time, please disccuss its possibility?
3.	Lack of Computational Analysis – The paper does not report the training overhead introduced by prototype orthogonalization or additional loss terms. How significant is the extra cost compared to standard contrastive methods?
4.	Dependence on Label Quality – CLOP relies on a subset of labeled data to guide prototype alignment. How robust is it to noisy or inaccurate labels, and would incorrect prototype supervision lead to representation drift?

### Questions
Please check weaknesses, and try to argue them, I will definitely read your response, good luck!

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Contrastive Learning With Orthonormal Prototypes (CLOP), which forms orthonormal prototypes to prevent dimensional collapse of the embeddings learned by semi-supervised loss functions.

### Strengths
* This paper focuses on an important research area of semi-supervised contrastive learning

### Weaknesses
* The authors did not follow the standard ICLR style

* No theoretical results supporting the success of CLOP

* It is unclear when/why CLOP works well

### Questions
None

### Soundness
2

### Presentation
2

### Contribution
2
