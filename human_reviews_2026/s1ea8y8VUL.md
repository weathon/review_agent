# Delving into Spectral Clustering with Vision-Language Representations

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Spectral clustering is known as a powerful technique in unsupervised data analysis. 
The vast majority of approaches to spectral clustering are driven by a single modality, leaving the rich information in multi-modal representations untapped. 
Inspired by the recent success of vision-language pre-training, this paper enriches the landscape of spectral clustering from a single-modal to a multi-modal regime. 
Particularly, we propose Neural Tangent Kernel Spectral Clustering that leverages cross-modal alignment in pre-trained vision-language models. 
By anchoring the neural tangent kernel with positive nouns, i.e., those semantically close to the images of interest, we arrive at formulating the affinity between images as a coupling of their visual proximity and semantic overlap. 
We show that this formulation amplifies within-cluster connections while suppressing spurious ones across clusters, hence encouraging block-diagonal structures. 
In addition, we present a regularized affinity diffusion mechanism that adaptively ensembles affinity matrices induced by different prompts.
Extensive experiments on \textbf{16} benchmarks---including classical, large-scale, fine-grained and domain-shifted datasets---manifest that our method consistently outperforms the state-of-the-art by a large margin.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a spectral clustering method for features extracted from vision language models. The main contributions lie in integrating the affinities from the vision and text modalities, as well as multiple prompts. Experimental results on 13 datasets demonstrate the effectiveness of the proposed method.

### Strengths
1. The proposed method is mathematically grounded, with detailed derivations provided in the appendix.
2. The paper is clearly organized and written in general.
3. Experiments show that the proposed method outperforms existing baselines on both classic and more challenging, fine-grained datasets.
4. Parameter analyses are conducted to investigate the robustness of the method.

### Weaknesses
1. To improve the readability of the paper, I encourage the authors to provide a figure to illustrate the proposed method, instead of purely presenting mathematical notations and equations.
2. What is the clustering performance of the NTK-induced affinity matrix $A_{NTK}$?
3. The mentioned baseline TAC has a variation that trains a clustering head to further improve the performance. Currently, it seems that the authors only compare its training-free version. A more thorough comparison is expected.
4. Ablation studies on each component of the method could be attached.
5. What is the time and space complexity of the proposed method? Considering the computational efficiency of spectral clustering methods, can the proposed method scale to large datasets? How much time does it take to perform clustering on ImageNet-1K?

### Questions
I expect the authors to address my concerns on the weakness section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper is about a new clustering approach named Neural Tangent Kernel Spectral Clustering. This approach that extends traditional spectral clustering to leverage multi modal repreentations from pre trained vision language models. The core contribution is formulating image affinity as a coupling of visual proximity and semantic overlap through anchoring neural tangent kernels with semantically relevant positive nouns. The paper includes a regularized affinity diffusion mechanism to ensemble affinity matrices from different prompts and demonstrate substantial improvements across 16 datasets.

### Strengths
+ The integration of neural tangent kernel theory with vision language representations for spectral clustering is novel. The idea of anchoring NTK with positive nouns to create semantically aware affinity matrices represents a way to incorporate linguistic priors into clustering.
+ The paper includes a comprehensive experimental validation, where testing is performed on 16 benchmarks. The consistent improvements suggest that the method is generalizable.
+ The presentation is excellent, the paper is easy to read and well written.

### Weaknesses
- The method uses some positive nouns that are semantically close to images of interest. This could limit practical applicability and introduce some bias. The paper should provide some analysis about it.
- A minor weakness, is that the main paper is condensed and the related works in that are limited.

### Questions
- How does performance degrade with noisy or out of distribution images?

### Soundness
4

### Presentation
4

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
The paper proposes Neural Tangent Kernel Spectral Clustering (NTK-SC) for image clustering with vision–language models. 
The core idea is to anchor a proxy network at text features of positive nouns, compute an empirical NTK over CLIP image embeddings, and use it as an affinity for spectral clustering. 

Concretely, with CLIP's unit-normalized visual features \\(z=f_X(x)\in\mathbb{R}^d\\) and noun features \\(W=[w_1,\dots,w_N]\\), the proxy is chosen as a log-sum-exp \\(g_{\theta_0}(z)=\log\sum_{k=1}^N e^{w_k^\top z/\tau}\\) (Eq. (7)). The empirical NTK between two images has the closed-form (Eq. (8)).

Affinities are sparsified via mutual \\(q\\)-NN and fed to SC (Eq. (2)). RAD alternates a diffusion update

$$
\hat A \leftarrow \sum_{b=1}^B \frac{\beta[b]}{\mu+1}\,S^{(b)}\hat A S^{(b)\top}+\frac{\mu}{\mu+1}\,E
$$

(Eqs. (11)-(13)) and a Lasso-like weight update for \\(\beta\\) (Eq. (14)), with \\(E=I\\). Experiments on 16 datasets claim consistent gains over TAC/SIC/CLIP baselines.

### Strengths
- The paper is easy to follow.
- Eq. (8) decomposes the affinity into visual proximity and a text-induced overlap. It is easy to compute once noun logits are available. 
- The RAD update and its fixed-point interpretation are derived and come with a convergence argument for the linearized step.   
- Results cover classic, fine-grained, and domain-shift benchmarks. Ablations on \\(\tau,q,\mu,\lambda\\) are provided.

### Weaknesses
> 1. I wonder if negative weights violate standard SC assumptions. 

Since CLIP features lie on the unit sphere, \\(\langle z_i,z_j\rangle\in[-1,1]\\) (Eq. (1)). With Eq. (8), the NTK inherits negative values whenever the cosine is negative:

$$
K_{\theta_0}(z_i,z_j)=\frac{1}{\tau^2}\langle z_i,z_j\rangle \sum_k s_i[k]s_j[k].
$$

Yet the method sets \\(A_{ij}=K_{\theta_0}\\) on mutual \\(q\\)-NN (Eq. (6)) without rectification or signed-graph treatment. Normalized-cut SC (Eq. (2)) typically presumes nonnegative affinities, negative edges break stochastic interpretations and many guarantees. No analysis or ablation is given for rectifying \\(A\ge 0\\) (e.g., \\(A\leftarrow\max(0,K)\\) or shifting).

> 2. Graph quality and connectivity under fixed q are unexamined.

The paper fixes \\(\tau=0.04,\,q=30,\,\mu=0.1,\,\lambda=10\\) for all datasets (sizes and densities vary widely). There is no report of connected components or sensitivity on large graphs (e.g., ImageNet-1K). Disconnected graphs produce multiple zero eigenvalues and can distort clustering. 

> 3.Scaling and runtime are not discussed.

The RAD fixed-point (Eq. (12)) is acknowledged infeasible to invert (\\(M^2\times M^2\\)), hence an iterative update (Eq. (13)). However, per-iteration costs with \\(B\\) prompt graphs and sparse matrices (expected \\(O(B\cdot M q^2)\\)) and memory footprints are not reported

> 4.Comparisons omit crucial baselines and uncertainty.

Absent is a fair product-kernel baseline (cosine \\(\times\\) text overlap).

> 5. Evidence issues

The paper claims consistent SOTA improvements and sharper block diagonals, but (i) the evidence rests on fixed hyperparameters across datasets of very different statistics, (ii) TAC multi-prompt usage is argued via an implementation detail (footnote) rather than a principled comparison, and (iii) no statistical significance is provided.

### Questions
In addition to the issues raised above, I have the following questions:

1. Do you ever rectify \\(K_{\theta_0}\\) or clip \\(A\\) to be nonnegative before computing \\(L=I-D^{-1/2}AD^{-1/2}\\)? Do you remove the diagonal prior to eigen-decomposition? Please report results with \\(A\leftarrow\max(0,K_{\theta_0})\\), \\(A\leftarrow (K_{\theta_0}+|K_{\theta_0}|)/2\\), and with/without diagonal. (Eqs. (2), (6), (8))

2. What exactly is the algorithm for selecting “positive nouns,” especially given the definition “similar to any ID label”? What signals from the train split are used to filter nouns, and how do you prevent label-name leakage?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to take advantage of multi-modal (i.e., vision-language) representations of data for spectral clustering. Concretely, given a pre-trained vision-language model, data without any supervision can be anchored using neural tangent kernel with positive nouns, so that data representations can be further refined by considering the language modality of interest. This method can also take benefit from the ensemble of multiple language prompts. Thereafter, the refined affinity matrices can be used to do spectral clustering and better performance can be observed on various image datasets.

### Strengths
1) The problem of considering multiple modalities of data in spectral clustering is interesting and on time given the current advances in multi-modal foundation models.

2) Using neural tangent kernel to anchor the data using multi-modal information is interesting and sound.

3) Better clustering performance can be observed on various image data.

### Weaknesses
1) To consider both vision and language modalities in spectral clustering of images, this work extends existing techniques/strategies of "positive nouns", "neural tangent kernel", etc, which is making this work's technical contribution moderate.

2) It should be noted that many evaluation datasets used in this work have appeared in the training of the vision-language foundation models, which can raise a data-leakage problem since the representations under interest has been well obtained. Therefore, it is necessary to include a discussion on if the performance benefit is directly from the foundation model's representation or from the proposed method. Another potential way is using data not included in the training of the foundation model for clustering evaluation.

3) An ablation study on the contributions of each component in the proposed method would be interesting. For example, how much has been the ensemble contributing to the performance compared to using only one prompt? How much does a single prompt contribute using the proposed kernel way vs without using the kernel way.

4) Some typos can be fixed, e.g., "aim to grouping".

### Questions
Questions can be found in the above weakness section.

### Soundness
2

### Presentation
3

### Contribution
2
