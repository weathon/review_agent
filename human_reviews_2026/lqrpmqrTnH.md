# In Context Semi-Supervised Learning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
There has been significant recent interest on understanding the capacity of Transformers for in-context learning (ICL), yet most theory focuses on supervised settings with explicitly labeled pairs. In practice, Transformers often perform well even when labels are sparse or absent, suggesting crucial structure within unlabeled contextual demonstrations. We introduce and study in-context semi-supervised learning (IC-SSL), where a small set of labeled examples is accompanied by many unlabeled points, and show that Transformers can leverage the unlabeled context to learn a robust, context-dependent representation. This representation enables accurate predictions and markedly improves performance in low-label regimes, offering foundational insights into how Transformers exploit unlabeled context for representation learning within the ICL framework. Our code is available at https://github.com/Jason-fan20/ICL_Semi.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies semisupervised in-context learning via two-stage Transformer framework. The first stage is to implement representation learning via computing discrete Laplacian and Eigenmaps. While the second is for in-context learning with additional prediction head for classification problems. Experimental results have shown that this two stage framework achieves the best performance compared to the others.

### Strengths
1. This paper focuses on a realistic problem, semi-supervised learning, where a large amount of data may lack labels.
2. This paper proposed method that assigned different functional roles to different Transformer blocks, eg, representation extraction and ICL+categorial mapping. Such separation may provide insights into understanding the roles of different Transformer layers.

### Weaknesses
1. In Line115, the authors claimed "we are unaware of any prior semi-supervised implementations." However, there are numerous prior works that have explored semi-supervision ICL. For examples,
> [Agarwal et al. 2024] Many-shot in-context learning
>
> [Li et al., 2025) When and How Unlabeled Data Provably Improve In-Context Learning
>
> [Chen et al., 2025] Maple: Many-shot adaptive pseudo-labeling for in-context learning.

while there are many more relevant works. Therefore, I am surprised that the authors are unaware of any prior work on this topic.

2. In Lines 200-212, the authors construct the input $X$ from original dimension $d\times n$ to $(d+n)\times n$ where $d$ is the input feature dimension and $n$ is the sequence length. It is impractical and raises several issues:
- In sequence models, the input length $n$ is typically changing over time;
- It is computationally inefficient, especially when $n\gg d$. However, modern large models are typically capacity of long input sequences, and $n\gg d$ is common in practice. 
- According to Appendix A.1, due to such input construction, the number of model parameters have increased from $d\times d$ to $n\times n$, which greatly increases the computational and memory cost.
3. Some notations lack clarity. For examples, what are the dimensions of $\phi^{(i)}$ in Line 233, $\varphi(\phi)$ and $A$ in Line 237?
4. Due to the Weaknesses 2, the authors could only test on the fixed in-context sample size $n=100$.
5. While the authors have shown that OROIG+E2E-ICL achieves the best performance, this method require more computation and larger model parameters to train. Such comparisons may be unfair.

### Questions
1. What is $f$ in Eq(1)? It seems that the representation of each raw feature $x^{(i)}$ is independent of others $x$'s, which contradicts the nature of attention and ICL, where representation $f$ at token $i$ should depend on prior examples $(x^{(j)},y^{(j)})_{j<i}$. Otherwise, it is not aligned with standard ICL.
2. Why in Eq(4), the loss is calculated on unlabeled data given that $y^{(i)}$ is unknown? Also, could the authors clarify on this and the Line 175 "When training, this loss is fit to the n−m unlabeled samples (for which we have labels when training)."?
3. Could the author provide more details on the meaning of $w_y-\mathbb{E}(w|f_\ell)$ in Eq (6)?
4. In Eq (7), the left contains an expectation symbol but its expression does not include any expectation operation. Could you clarify on this?
5. Is the $\gamma$ in Eq  (7) equal to the parameter set {$w_c,c=1,2,\cdots C$}?
6. What does $\ell$ represent in Section 3.2?
7. Since E2E is treated as feature exactor, have the authors considered replacing it with other feature extraction methods, such as using pretrained DNN models?

### Soundness
3

### Presentation
3

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
This paper introduces IC-SSL, a novel Transformer architecture explicitly constructed to perform in-context semi-supervised learning. The core contribution is a two-stage model: 1) A representation-learning stage ($TF^{\phi}$) that is designed to leverage unlabeled data by computing Laplacian Eigenmaps, a classic spectral method for manifold learning. 2) An ICL stage ($TF_{sup}$) that uses these learned representations, along with the few available labels, to perform classification by provably mimicking functional gradient descent. The authors provide theoretical analysis (in the appendices) to demonstrate that their specialized attention modules (e.g., $Attn_{rbf}$, $Attn_{linear}$) can indeed be parameterized to execute these specific algorithmic steps. Empirical results on synthetic manifolds and real-world data demonstrate the method's superiority over baselines, especially in low-label regimes.

### Strengths
1. **Theoretical Quality**: The appendices provide a rigorous "proof-by-construction." The mathematical links between specific attention mechanisms and algorithmic steps (like $Attn_{linear}$ for power iteration) are non-trivial and well-executed.

2. **Empirical Effectiveness**: The method is shown to be highly effective, outperforming strong baselines, which validates the authors' algorithmic choices.

### Weaknesses
1. **Lack of Justification for Algorithmic Complexity**: The paper jumps directly to a sophisticated, iterative, non-linear SSL algorithm (Laplacian Eigenmaps). It doesn't first establish why this complexity is needed. The theoretical argument would be much stronger if it included an analysis (or even a citation to existing work) demonstrating that simpler, non-iterative (e.g., single-layer) or linear models fundamentally fail at this task.

2. **Specialization of the Spectral Method**: The choice of the Laplacian $\hat{\mathcal{L}}$ (via $Attn_{rbf}$) is a very specific one, tailored to non-linear manifolds. This is a strong design choice, but it's presented without robustly comparing it to simpler alternatives. A major question is whether a linear spectral method (e.g., PCA) would be sufficient. Given that the $TF^{\phi}$ module itself uses $Attn_{linear}$  (which is known to be related to matrix power computation), a standard model might more naturally learn to compute powers of the data covariance matrix ($X^\top X$) rather than the complex $\hat{\mathcal{L}}$. The paper fails to discuss the trade-offs of its specific, non-linear choice.

### Questions
1. **On the Necessity of Iteration**: Your model relies on iterative algorithms (power iteration, GD) implemented via depth/loops. What are the theoretical trade-offs of this explicit iteration versus a hypothetical non-iterative (e.g., single-layer) but perhaps very wide model? Is there a theoretical justification for why an iterative approach is fundamentally necessary for IC-SSL, thus motivating your design?

2. **On the Choice of Spectral Method (Linear vs. Non-linear)**: Your $TF^{\phi}$ module is complex: it first uses $Attn_{rbf}$ to compute the Laplacian $\hat{\mathcal{L}}$, and then uses $Attn_{linear}$ to compute its eigenmap via power iteration. A simpler approach might skip the first step. Given that stacked $Attn_{linear}$ layers are mathematically known to compute matrix powers, a standard Transformer might implicitly learn to compute powers of the data covariance matrix ($X^\top X$), a linear spectral method (i.e., PCA).
  * (a) Can you theoretically justify the necessity of your explicit, two-step, non-linear spectral method ($\hat{\mathcal{L}}$) over this simpler, linear spectral alternative ($X^\top X$)?
  * (b) Did you empirically compare your $TF^{\phi}$ module against a simpler $TF^{\phi}$ baseline that only uses stacked $Attn_{linear}$ to (implicitly) learn a linear spectral embedding?

3. **On the Algorithmic Structure (Decoupled vs. Coupled)**: You chose a 'decoupled' two-stage design (Learn Reps -> Classify), which is highly interpretable. However, many successful SSL algorithms (like EM) are 'coupled,' iteratively refining label-estimates and model parameters. What are the theoretical limitations of your decoupled approach? Did you explore or consider a more coupled, end-to-end model where the $TF^{\phi}$ and $TF_{sup}$ stages are interleaved at each layer?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper studies a setting of semi-supervised in-context learning (termed IC-SSL), where the input consists of many examples, a few of which are accompanied by gold labels. From a theory of ICL perspective, the authors construct and train small transformers on various classification tasks with synthetic data drawn from manifolds, and on one more realistic task using interpolations between images in ImageNet. They propose a specific model for this setting that outperforms baseline transformers, based on an argument about how semi-supervised learning might be performed in context mechanistically.

### Strengths
S1. The paper thoroughly explores semi-supervised ICL, presenting experiments on carefully designed synthetic data, using multiple random seeds, and reporting interesting ablations. 

S2. The comparison of performance on in-domain and out-of-domain data when training on data generated from different subsets of the classification manifolds was quite interesting. 

S3. The proposed E2E-ICL is both mathematically well-justified and empirically effective at semi-supervised ICL.

### Weaknesses
W1. It would be nice to at least mention recent [empirical work](https://arxiv.org/abs/2404.11018) on unsupervised ICL. 

W2. The claim in line 343 that the model is competitive on real data seems a bit ill-supported, as even the ImageNet task is a fairly synthetic classification task.

### Questions
Q1. Can you elaborate on line 72, the claim that the model "provably" performs ICL? 

Q2. In Figure 1, you add the labels for the few labeled examples back into the representation in step B. Why not allow representations to be learned with awareness of these labels? 

Q3. What is the transformer baseline in Figure 3/10? It's not in the list of models in 4.1.

Q4. Can you provide the parameter counts for all model types trained? It's a bit hard to tell from the descriptions alone whether some methods might have more learnable expressivity.

My work is more in the empirical direction, so some of these clarifications may be sufficiently clear already in the text for a more theory audience. 

Other comments:
- Please fix citations to use \citep{} where appropriate, especially in the introduction. 
- for the sentence on line 31 about functional gradient descent, would be nice to cite back to the prior work in this area 
- I found the contributions section 1.1 to be quite long and a bit confusing before reading the rest of the paper; I think this might be better positioned near the end of the work, with just a short summary of contributions in the introduction. 
- 4.1 and F7 are very similar text; F7 could perhaps be merged into 4.1.
- it would be nice if the line in Fig 4(b) that's a directly comparable result from 4(a) was the same color in both, though I understand the color scheme is meant to indicate the different model types.

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
This paper introduces In-Context Semi-Supervised Learning (IC-SSL), extending Transformer-based in-context learning to settings with few labeled and many unlabeled samples. It proposes a two-stage, end-to-end Transformer: the first stage learns representations from unlabeled data by constructing a Laplacian and computing an eigenmap, and the second stage performs in-context supervised learning to infer labels for unlabeled samples. This enables both representation learning and label propagation within a single forward pass. Experiments on synthetic manifolds and images show it outperforms baselines, leveraging unlabeled data to improve performance with few labels, and generalizes well in and out of distribution.

### Strengths
Strengths:

1. This paper presents a novel extension of in-context learning to the semi-supervised setting, enabling Transformers to jointly exploit labeled and unlabeled examples within a unified framework. 

2. The proposed two-stage architecture—combining Laplacian-based representation learning with an in-context gradient descent head—offers rare mechanistic interpretability and provides insight into how Transformers may implement semi-supervised reasoning. 

3. Empirically, the method demonstrates strong label efficiency and outperforms competitive baselines across synthetic manifolds, product manifolds, and ImageNet100, showing promising generalization and transfer, especially under low-label regimes.

### Weaknesses
Weaknesses:

1. Despite its contributions, the experimental evaluation is limited in scale and focuses primarily on small episodic settings, leaving open questions about robustness in large, noisy, or highly diverse real-world scenarios. 

2. The model relies on a strong geometry-driven inductive bias, which may restrict flexibility when the data does not exhibit clear manifold structure, and the architecture requires non-standard attention variants and customized modules that could hinder practical adoption. 

3. Additionally, the overall complexity introduces training sensitivity and may limit applicability beyond the studied domains.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
