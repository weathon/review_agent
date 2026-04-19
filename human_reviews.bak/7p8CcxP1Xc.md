# Proximal Mapping Loss: Understanding Loss Functions in Crowd Counting & Localization

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Crowd counting and localization involve extracting the number and distribution of crowds from images or videos using computer vision techniques. Most counting methods are based on density regression and are based on an ``intersection'' hypothesis, *i.e.*, one pixel is influenced by multiple points in the ground truth, which is inconsistent with reality since one pixel would not contain two objects. This paper proposes Proximal Mapping Loss (PML), a density regression method that eliminates this hypothesis. {PML} divides the predicted density map into multiple point-neighbor cases through the nearest neighbor, and then dynamically constructs a learning target for each sub-case via proximal mapping, leading to more robust and accurate training. {Furthermore}, PML is theoretically linked to various existing loss functions, such as Gaussian-blurred L2 loss, Bayesian loss, and the training schemes in P2PNet and DMC, demonstrating its versatility and adaptability. Experimentally, PML significantly improves the performance of crowd counting and localization, and illustrates the robustness against annotation noise. The code is available at [https://github.com/Elin24/pml](https://github.com/Elin24/pml).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper argues that the intersection hypothesis in existing density-map based counting methods is consistent with reality, where one pixel is influenced by multiple ground-truth points. To eliminate such a hypothesis, the authors propose a proximal mapping loss, which follows the philosophy of divide-and-conquer. Specifically, the predicted density map is first divided into non-overlapping irregular patches. Then, proximal mapping loss (PML) is computed within each point-neighbor case. Experimental results on several crowd counting datasets demonstrate the effectiveness of the proposed method. Additionally, this paper also shows that the proposed PML compasses many existing loss functions.

### Strengths
1. This paper presents an interesting argument about the intersection hypothesis in existing density-regression counting methods.
2. The paper proposes a divide-and-conquer strategy to eliminate the intersection hypothesis, featuring a proximal-mapping loss.
3. Experiments show that the proposed method achieves state-of-the-art counting and localization results.

### Weaknesses
1. The main concern is the training efficiency. The proposed method not only needs to compute the nearest neighbor, but also constructs learning targets for each sub-problem presented in Eq. 1. This could be computationally expensive when dealing with congested scenes.
2. The robustness to noisy annotations needs further clarifications, e.g., performance comparisons with existing methods would be sufficient.

### Questions
1. Does the proposed method suffer from slow convergence speed? As shown in Fig. 4, the training epoch reaches 2000. 
2. Did the authors use different epsilon for different datasets? Fig. 8 shows that the choice of epsilon does affect the performance.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a proximal loss function for crowd counting and localization. The key idea is to divide the predicted density map into multiple non-overlapped patches and compute loss for each patch via proximal mapping. Such an operation eliminates the intersection hypothesis in density regression-based methods, which improves the robustness and accuracy of the counting models. Evaluations on several crowd counting datasets demonstrate the superiority of the proposed loss over previous loss functions.

### Strengths
* This paper presents a new proximal mapping loss, eliminating the intersection hypothesis in existing density-map based methods.
* The authors comprehensively discuss the relationship between the proposed loss and existing loss functions.
* Extensive experiments validate the effectiveness of the proposed loss.

### Weaknesses
* It would be better to give a quantitative comparison of training efficiency between the proposed loss and existing losses. While the authors mention that the loss computation of PML is more efficient than GL, it is unclear whether the convergence speed of PML is faster than existing loss functions.
* It appears that retaining the spatial resolution is important to make PML work as occluded objects could degenerate into a single point when downsampling. In this case, it is beneficial to analyze the characteristics of the proposed loss under different crowd densities.
* Following the previous comments, it is suggested to discuss in what scenarios PML is better than previous methods.

### Questions
* Did the authors use the same output resolution to train different models in Table 4? As shown in the paper, spatial resolution could affect the performance.
* Does PML converge faster than existing methods? This is important when applying the proposed method to large-scale datasets.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes the Proximal Mapping Loss (PML), a novel density regression loss for crowd counting and localization. PML studies the so-called intersection hypothesis of classic density map and addresses the intersection issue by dividing the predicted density map into multiple patches through nearest neighbor analysis. It then constructs a dynamic learning target for each case, leading to more tractable and robust training. In addition, the paper also shows the theoretical connection between the PML loss and other closely related loss functions. Experiments were conducted to demonstrate the efficacy of PML in both crowd counting and localization.

### Strengths
+ **Good motivation**. The paper targets an appropriate problem in crowd localization, featured by the intersection hypothesis. The problem sounds.
+ **A novel loss**. To evade the sum property of density map, a new loss function is proposed to constrain the regression target by dividing the training sample into many point-neighbor cases and then computing sub-loss by comparing the difference between the prediction and a dynamic learning target defined via proximal mapping. According to Fig. 2, it does advance the prior Bayesian Loss and Generalized Loss in discriminating close individuals.
+ **Theoretically sound**. The derivations in Sec. 3.2 show how the loss function works in a relatively intuitive way during training, and its connections with several loss functions and training schemes have been proven, which is theoretically sound.
+ **SOTA performance**. Through evaluations on several benchmark datasets, PML is shown to outperform current state-of-the-art methods in general.

### Weaknesses
- **Some closely related work is missing**. For example, the same problem of intersection hypothesis has been discussed in [a]. While the solution differs, the way for addressing such an issue in [a] could be discussed and compared in Fig.1 as well.

[a] Decoupled two-stage crowd counting and beyond, IEEE Transactions on Image Processing, 2021.

- **Inappropriate baselines**. Since the main focus of the work is the loss function, it should demonstrate its effectiveness on SOTA models instead of constructing a new one (the HRNet based model). It would be difficult to interpret where the real improvement comes from.
- Some important experiments are missing such that some results are not directly comparable. For example, the proposed approach should report the results of vgg-16bn as well.

- **A proof-of-concept experiment is missing**. The initial claim of the paper is to solve the intersection hypothesis problem, but it turns out to focus on how to better predict points. There lacks some comparative experiments to prove the soundness of the initial claim.
- The paper claims that PML is robust to label noise. Though visualization results are given to demonstrate its robustness, there lacks evidence to support this. 

Minor issues：
- Some cited results may be wrong. E.g., CSRNet in Table 2 uses vgg-16 rather than vgg-16bn.

### Questions
- Please consider rephasing the intersection hypothesis. It does not make sense to me that in reality one pixel would not contain two objects. If two objects occlude, one pixel can certainly contain part of the two objects.
- Further ablation experiments should be conducted to justify the effect of noise on the model and the effectiveness of PML in coping with it.
- The paper has demonstrated its work on removing the intersection hypothesis. Experiments, however, has shown nothing related to the hypothesis and mainly focus on the efficacy of PML. Could you explain through experiments or visualizations whether PML really addresses such an issue?

- Table 4 only shows plug-and-play results for MCNN and CSRNet; they are somewhat outdated. It is better to report the use of PML on recent SOTA models. 

- It would be good to showcase how much PML can actually eliminate the intersection hypothesis? Can it be shown in a quantitative way?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Proximal Mapping Loss (PML) to improve crowd counting and localization and challenges the basis of prevailing loss functions used in density regression. PML divides the predicted density map into smaller, non-overlapping point neighbors and is processed independently to construct a target for learning. The paper also links various existing loss functions used in crowd counting to PML, showcasing the generalizability across existing methods.

### Strengths
The paper demonstrates limited originality through its approach, which addresses a key limitation, the intersection hypothesis, in existing methods. The method offers a fresh perspective on how crowd density can be modeled. The paper provides good theoretical foundations for the proposed method and empirical results compared to existing methods. The paper is well-structured and progresses coherently to deliver the proposed method.

### Weaknesses
While the paper demonstrates that PML outperforms several existing loss functions, it does not explain why PML works better for density-based crowd-counting methods. The method’s reliance on nearest-neighbor assignment for dividing the predicted density map into point-neighbor cases could be problematic, particularly in dense crowd structures. NN algorithms may not capture the correct relationships between nearby points, leading to inaccurate density estimations in such edge cases. The paper does not address how this issue is handled or provide evidence that PML performs well in extreme cases.

### Questions
Major Comments

1. There are existing density-based crowd-counting methods that already don’t rely on the intersection hypothesis, such as STEERER and CrowdDiff. STEERER achieves non-overlapping density through scaling, and CrowdDiff uses narrow kernels to prevent overlapping. Both methods use Gaussian kernels and avoid overlapping in density kernels. In that case, why would the non-overlapping nature of PML work better than non-overlapping Gaussian kernels?
2. The paper lacks a comparison against CrowdDiff, the state-of-the-art NWPU crowd-counting benchmark.
3. How does eliminating the intersection hypothesis improve crowd counting and localization performance?
4. Fig. 3 shows the density maps obtained with or without applying PML. If it is the former, how is localization achieved from the density maps? Also, in some cases, more than one density kernel is predicted on a single face. How are these false positives filtered for localization and counting?
5. How does the nearest neighbor approach in the divide and conquer stage impact performance in dense versus sparse crowd scenes?
6. The counting loss was tested using the nearest neighbors in CLTR. How does the PML’s nearest neighbor assignment compare to CLTR?

Minor comments

7. How does PML handle edge cases where crowd members are extremely close together or partially occluded? Especially with partial occlusions, even though the pixel values are not influenced by neighboring objects, the density kernels could have some overlapping as this is a distribution predicted over image coordinates conditioned on pixel values.
8. The detailed results from the NWPU evaluation could help address these edge cases for the crowd-counting community.
9. Missing citation for RAZNet in line 135.

### Soundness
3

### Presentation
3

### Contribution
2
