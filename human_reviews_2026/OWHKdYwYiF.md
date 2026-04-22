# Towards Principled Dataset Distillation: A Spectral Distribution Perspective

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Dataset distillation (DD) compresses large datasets into compact synthetic sets for efficient model training. Among various DD approaches, distribution matching (DM) has emerged as a promising direction due to its ability to bypass the computational complexity of bi-level optimization while maintaining strong performance. However, most current DM approaches adopt metrics that are not theoretically well-founded and thus fail to accurately capture distributional discrepancies. This stems from a lack of in-depth theoretical analysis of the metrics themselves. Therefore, we revisit existing metrics from the spectral domain and provide theoretical insights to guide future metric design. Based on this analysis, we propose Spectral Distribution Matching (SDM), a Fourier-based approach that introduces theoretically motivated, discriminative metrics and achieves linear computational complexity through a Fourier-based algorithm that enables fast and scalable computation. Our method not only proves effective on standard datasets, but also demonstrates superior performance on more challenging long-tailed datasets. To address the issue of class imbalance caused by long-tailed data distributions, we leverage the unified metric formulation of SDM to further propose Class-Aware Spectral Distribution Matching (CSDM), which adaptively balances amplitude and phase information based on class imbalance, while enhancing the realism of head classes and preserving the diversity of tail classes. Overall, our proposed SDM and CSDM not only provide a principled rethinking of distribution matching from the spectral perspective, but also introduce a novel class-aware mechanism that addresses the often-overlooked challenge of long-tailed distributions in dataset distillation. By bridging theoretical insights with algorithmic efficiency, our methods consistently deliver excellent performance across both standard and long-tailed benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new dataset distillation method that is leveraged to use the MMD with diverse kernel functions and to adaptively match the long-tail classification. The authors derives the kernel matching formula from the MMD distance definition of two distributions, which result in the introduction of SDD of two kernel functions from two distributions. Then, the authors add a customized derivation further to decompose the kernel functions into class-aware settings. There are typical benchmark experiments showing the advance from the work.

### Strengths
1. 
This is a typical derivation of using MMD to measure and reduce the distribution of two datasets. Well principled from the integral probability metric (IPM) perspective.

2.
Authors adds a long-tail class aware loss function to the MMD distance, which could be new development in the field.

3.
It is true that MMD can measure more diverse aspect of distribution distance if it is being used more complicated kernel functions.

### Weaknesses
1.
The major weakness of using MMD is its runtime. Kernel functions are calculated across the instance pairs of two distributions, which can be O(N^2). Surely, authors may choose to sample some of them, but it will impair the distribution distance estimation. Anyhow, authors should have been presenting such the time complexity issue and its counter mechanism on calculating kernel functions in time efficient manner.

2.
The kernel functions are listed in Table 1, which is very traditional and conventional. First, these kernel functions can be multiplied or added, which again becomes a kernel function. Therefore, such nice composition of kernel functions have been explored in many works using MMD. Moreover, you may train a neural network to be a kernel function by enforcing the learned function to follow the limitation of kernel functions. These aspects have not been explored enough.

3.
Kernel matching and using the spectral density is not a new idea in dataset distillation. For example, the below paper already utilizes the frequency domain for the dataset distillation. Using MMD can be regarded as an incremental development along this line of research.

DongHyeok Shin, Seungjae Shin, and Il-Chul Moon, Frequency Domain-based Dataset Distillation, Neural Information Processing Systems (NeurIPS 2023), New Orleans, USA, Dec 10-Dec 16, 2023

At least, some comparison could be beneficial.

4.
Authors may present what would have been chose data instances or features from the distillation. Currently, all of tables and figures are derived from the performance metric, which nuance that better performance indicates better results. However, it is necessary to check what has been distilled and what have been remembered by this method, particularly in comparison to other distillation methods.

### Questions
Please see the above four weakness to answer the questions

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a a new method for dataset distillation based on distribution matching. The proposed method, termed CSDM, is based on the Maximum Mean Discrepancy (MMD) metric and classical methods for distinguishing two distributions via kernel embeddings. The authors first argue that existing distribution matching methods for data distillation suffer from two main issues: (1) lack of expressivity in the distribution matching objective (related to the use of linear kernels for MMD) and (2) uniform treatment of classes in long-tailed datasets. The main contributions of CSDM are the use of nonlinear, universal kernels when computing MMD and the use of class-specific weighting  in the characteristic function matching objective. The paper evaluates CSDM on long-tailed versions of CIFAR-10, CIFAR-100, and ImageNet subsets, showing that it performs better across various imbalance factors and image-per-class.

### Strengths
The paper identifies two important limitations of existing distribution matching approaches for dataset distillation: the use of linear kernels for MMD and treating all classes the same in imbalanced datasets. The first of these problems is not a novel observation (see [1]), but the proposed solution of using universal nonlinear kernels is a very natural and sensible one, and I am actually a bit surprised I was not able to find this explicitly in the existing literature. The proposed method is able to achieve good performance on common long-tailed datasets like CIFAR-10-LT and CIFAR-100-LT.

### Weaknesses
While I believe the general idea proposed is valuable, I have a few concerns related to the comparison with NCFM  [1] and some vague/imprecise claims made in the paper. Given clarifications on these points and writing improvements (see Questions below), I would be willing to raise my score.

- Comparison to NCFM: As stated in the paper, the main difference between these methods is that CSDM performs characteristic function matching with respect to the spectral measure of a universal kernel, rather than a learned weight function. The authors note that NCFM performs well on a balanced dataset, but degrades under class imbalance. While this may be true, it does not immediately follow that the "neural weighting" used by NCFM is inferior to CSDM's approach. Is the benefit of CSDM coming from the fact that it uses a universal kernel or the fact that is uses class-dependent weights for the amplitude and phase? In particular, how would NCFM compare if it used the "neural weighting" combined with class-based amplitude/phase coefficients? Are there any computational advantages to CSDM? 

- The connection between amplitude/phase and "diversity"/"realism" seems quite vague. This seems somewhat important to clarify, since much of the methodology is based on the assumption that weighting the amplitude and phase differently based on class size is a good idea. Some statements here require more justification, e.g., "Phase Difference, on the other hand, captures misalignment in the phase component, which is indicative of the realism of the data". Why? The provided reference also doesn't seem to address this. (Not sure if I am just unaware of something in the literature for this, though)

- Similarly, the following claim in Section 4.2.4 is also not justified: "In long-tailed settings, tail classes demand more realism to prevent mode collapse while head classes benefit from greater diversity."

- Insufficient discussion of how $\alpha(c)$ is chosen. This is a key part of the methodology that distinguishes it from prior works, but only receives a very small amount of attention in Section 5.3. It is unclear if this is a purely heuristic choice and how it chosen for the provided experimental results.

References:
[1] Wang et al. "Dataset distillation with neural characteristic function: A minmax perspective". CVPR 2025.

### Questions
- See weaknesses section for the main questions I have that impact the score
- How is $\alpha(c)$ chosen in experiments? Or more generally, can this part be done in any principled way?
- How does distribution matching with universal kernels perform on balanced datasets?  

More minor things (not affecting the score):
- Figure 1's captions are a bit vague "Minimize previous metric". What is the previous metric?
- Figure 2 is a bit hard to follow due to the small size and amount of text. Perhaps it can be made larger
- The notation for MMD is inconsistent, sometimes taking three arguments (function class, P, Q) and sometimes only two (P,Q). In the latter case, the function class seems to implicitly be the unit ball in an RKHS?
- Section 1 typo: "we attribute the drawbacks of previous DD methods *to*..."

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem of dataset distillation on long-tailed (imbalanced) datasets and identifies two key limitations of prior methods:

(1) they rely on simplistic distribution matching metrics (e.g. MSE or linear-kernel MMD) that only align first-order statistics (means), failing to truly match distributions.

(2) they treat all classes uniformly, which is ineffective when some classes have far fewer samples.

To overcome these issues, the authors propose Class-Aware Spectral Distribution Matching (CSDM). The core idea is to use a universal kernel and Bochner’s theorem to map data into a frequency domain and define a Spectral Distribution Distance (SDD) that can distinguish full distributions rather than just their means.

They further decompose this spectral difference into amplitude (related to diversity of samples) and phase (related to realistic detail) components. These disentanglement could be especially effective, when we combine these into dataset distillation, whose budget is limited and still require diversity and detailedness.

By assigning class-specific weights to these components, CSDM emphasizes realistic fidelity for under-represented tail classes while maintaining diversity for head classes, thus dynamically handling class imbalance. This principled metric design allows the distilled synthetic set to better capture the overall data distribution, especially the rare classes that previous approaches synthesized poorly.

### Strengths
By assigning class-specific weights to these components, CSDM emphasizes realistic fidelity for under-represented tail classes while maintaining diversity for head classes, thus dynamically handling class imbalance. This principled metric design allows the distilled synthetic set to better capture the overall data distribution, especially the rare classes that previous approaches synthesized poorly.

Empirically, CSDM shows substantial gains: on the severely imbalanced CIFAR-10-LT benchmark (imbalance factor 200), using just 10 synthetic images per class, it improves accuracy by 14.0% over the best existing methods, and its performance only drops by 5.7% even when tail classes’ images are reduced from 500 to 25 – indicating strong robustness to data scarcity.

Also, The approach is theoretically grounded (since MMD with a characteristic kernel ensures aligning entire distributions) and addresses a significant practical challenge in dataset distillation.

### Weaknesses
One potential concern is the added complexity of designing and computing the spectral kernel embedding, but the authors argue that the Fourier-based implementation is efficient. It would be informative if author provide pseudo-code or big-O notation about fourier-related computation.

Although this paper nicely derive spectral distribution-based dataset distillation, this paper lacks extensive search and discussions with existing methods. For example, FreD (https://arxiv.org/abs/2311.08819) primarily provides spectral distribution-based matching, but there is not any discussion about it. So i am leaning toward rejecting this paper, with lack of discussions and comparisons with current methods.

Also, author claims that explicit modeling of 1) diversity and 2) detailedness, but they do not demonstrate any distilled image compared to other methods, so it is hard for reviewer to conclude that performance improvements are mainly from these modelings. It would be really helpful. 

Although i am leaning toward rejecting this manuscript, i could change my score if weaknesses above are treated well from rebuttal period.

### Questions
Discussed in weaknesses section.

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
5

### Summary
This research claims that the previous dataset distillation methods rely on heuristic metrics, which only align first moment; and treat all classes uniformly, which might be fail on long-tailed datasets. To mitigate this problem, this research suggests Class-aware spectral distribution matching (CSDM), which formulates distribution matching in the spectral domain and amplitude-phase decomposition with class-specific weighting. CSDM employs Spectral distribution distance (SDD), which is derived from kernel theory with universal kernel. This approach enables better distribution matching and balanced learning for class imbalanced dataset.

### Strengths
1. The point raised that many distribution matching methods employ linear kernel, which fail to satisfy universality, is a theoretical limitation of considerable merit.

2. This manuscript is well written and easy to reading. Furthermore, the core idea of CSDM is simple and intuitive, making it easy to understand and apply.

3. In high imbalanced factor experiments, CSDM achieved significant performance gap compared to the baseline, which constitutes highly appropriate results for long-tailed dataset distillation, the primary target of this research.

### Weaknesses
1. Theoretical and experimental comparisons with several previous studies[1,2,3,4,5] that performed distribution matching considering higher moments are lacking. In particular, there is insufficient discussion regarding M3D[1] and IID[2] despite being mentioned in this paper. Furthermore, the only experimental comparison with the core baseline, NCFM[4], is on the long-tailed CIFAR dataset. Therefore, there is insufficient justification to develop the argument solely based on the shortcomings of first-moment distribution matching.

2. SDD lacks novelty and contribution as it is not newly proposed in this paper. Firstly, Theorem 3, stating that Squared MMD can be expressed as a Characteristic function, is already established theory (see Corollary 4 in [6]). Furthermore, SDD is also widely used in various fields under the name Characteristic function distance (CFD)[4,7,8]. However, as this paper does not reference relevant previous works, it could give the impression that these concepts were first proposed herein. Therefore, since the content related to SDD is already a widely known and used concept, I believe this paper lacks sufficient novelty and contribution.

3. The second idea, the class-specific coefficient, is likewise merely a naïve extension of prior research. As mentioned in this paper, the idea of decomposing the discrepancy in the characteristic function into amplitude and phase discrepancies is well-established. Furthermore, the concept of introducing a weight parameter $\alpha$ to adjust these two components also already exists [4,8]. Therefore, in this paper's CSDM, $\alpha$ is simply changed to a class-specific weight $\alpha (c)$. Furthermore, since this value is not determined systematically but is treated as a hyperparameter, I consider it to lack novelty and contribution. Moreover, there is no adequate analysis as to whether the introduction of the class-specific weight $\alpha (c)$ does not hinder the original objective of distribution matching and can still satisfy optimality.

4. There is no complexity analysis based on theoretical and experimental grounds. Calculating the SDD requires the computation of a characteristic function involving the expectation of complex numbers and Monte-Carlo sampling, resulting in a nested summation form. This can incur significant computational cost, necessitating a complexity analysis.

5. Key previous literature is missing. FreD[9] and NSD[10] are prior studies addressing the frequency domain in dataset distillation and should therefore be mentioned in the Related Works section.


[1] M3D: Dataset Condensation by Minimizing Maximum Mean Discrepancy

[2] Exploiting Inter-sample and Inter-feature Relations in Dataset Distillation

[3] Diversified Semantic Distribution Matching for Dataset Distillation

[4] Dataset Distillation with Neural Characteristic Function: A Minmax Perspective

[5] Dataset Distillation via the Wasserstein Metric

[6] Hilbert Space Embeddings and Metrics on Probability Measures

[7] A Characteristic Function Approach to Deep Implicit Generative Modeling

[8] Reciprocal Adversarial Learning via Characteristic Functions

[9] Frequency Domain-based Dataset Distillation

[10] Neural Spectral Decomposition for Dataset Distillation

### Questions
1. SDD (or CFD) is not an idea limited for long-tailed dataset distillation. Therefore, to argue that existing distribution matching methods suffer from inadequate alignment, I believe performance comparisons in class-balanced dataset distillation are also necessary.

2. I am curious as to why the baseline for Long-tailed ImageNet is less than that for Long-tailed CIFAR.

3. I am also curious whether minimizing the convex combination of amplitude-phase components still satisfies distribution matching under optimal conditions.

4. I suggest dividing Section 5.4 into separate sections. A comparison with NCFM would be more appropriately placed in Section 4.1 or 4.2. Table 5 is redundant as it duplicates the experimental results presented in Table 2.

### Soundness
3

### Presentation
3

### Contribution
2
