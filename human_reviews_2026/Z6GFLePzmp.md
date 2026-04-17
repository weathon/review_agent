# ECAttention: Adaptive and Principled Feature Expansion-Compression with Linear Efficiency

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
The maximal coding rate reduction ($\text{MCR}^2$) objective has been proposed to learn low-dimensional subspace representations by minimizing the compression term for intra-group compactness and maximizing the expansion term for inter-group separation.Several studies have leveraged $\text{MCR}^2$ to design principled, interpretable deep models 
by following or approximating its gradient to derive layer structures.However, these approaches remain limited in achieving fully principled and effective compression and lack self-adaptive control over the strength of expansion and compression across layers.In this work, we introduce \textbf{ECAttention}, a novel attention mechanism that incorporates principled expansion and compression modules inspired by the \textbf{geometric insight} of $\text{MCR}^2$.
Geometrically, gradient-based updates of  $\text{MCR}^2$  move features along directions shaped by the underlying data structure.
Our method efficiently captures this structure using randomization combined with Cholesky decomposition to guide feature updates with \textbf{nearly linear complexity}.By introducing two trainable weights per layer, ECAttention self-adaptively regulates the strengths of compression and expansion.The resulting ECA transformer not only matches or surpasses prior methods, but also exhibits greater interpretability, with different heads focusing on distinct image regions and capturing \textbf{fine-grained structures} under simple supervised training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ECAttention, a linear-time attention mechanism inspired by the Maximal Coding Rate Reduction ($MCR^2$) objective. The authors argue that previous $MCR^2$-based models, like CRATE and ToST, are either unprincipled in their approximation or over-simplify the problem. The core idea of ECAttention is to use a geometric insight—projecting features onto their column space and null space—to perform expansion and compression. The method uses randomization and Cholesky decomposition to efficiently find the basis of these subspaces, achieving linear-time complexity relative to the number of tokens ($n$). The model also introduces two learnable parameters ($\alpha, \beta$) per layer to "self-adaptively" balance the strength of expansion and compression. The authors claim this results in a more principled and interpretable model that achieves comparable or-superior performance to baselines like ToST and CRATE.

### Strengths
1. The paper's strongest point is Figure 5. The membership visualizations for ECA-B are qualitatively excellent, demonstrating a fine-grained, semantically meaningful segmentation of objects (e.g., separating a building's floors, or a tree from a door). This provides strong evidence for the authors' claim that ToST's simplification (ignoring inter-dimensional correlations) misses important structural information, and that ECAttention's approach successfully captures it.

2. The paper plots inference time and peak memory vs. tokens and provides an appendix complexity breakdown; CRATE is quadratic, while ECA trends linearly, with an added Cholesky cost.

3. ECA-S (7.2M params) achieves 79.0% on ImageNet-ReaL, beating CRATE-L; other small/medium comparisons to ToST are “comparable.”

### Weaknesses
1. Does not clearly surpass the strongest baselines at scale. The paper itself concedes ECA “falls slightly behind ToST at larger parameter scales,” partly blaming a fixed rank (r=20). For a most-competitive venue, “comparable” is usually insufficient—particularly when the method adds nontrivial machinery (random projections + Cholesky) on top of ToST-like scaffolding. Strong, fair, large-scale sweeps (varying (r), compute-matched) are needed. 

2. Linear-time claim depends on a non-negligible Cholesky subroutine. The paper frames the overall complexity as linear in tokens with a “minor fixed overhead,” yet implements an r×r Cholesky per head/layer (plus fallbacks), which can easily become the hidden bottleneck depending on r, heads, and batch size. The current analysis underplays these constants and their training-time impact. 

3. The Cholesky path needs regularization and has a QR fallback on failure; the paper doesn’t quantify how often Cholesky fails, how ε and temperature affect training.

4. The authors note the “expansion” weight can go negative (thus compressing) when r is large; this blurs the conceptual neatness of separate expansion vs. compression and suggests under-constrained dynamics. Conditions for when expansion is beneficial vs. harmful are not established.

5. There is no evaluation on long-sequence tasks that would stress the purported advantages of linear scaling.

### Questions
The idea is clean and well-motivated, and the qualitative behavior is appealing. However, the empirical section does not yet establish a decisive advantage over the strongest baselines at relevant scales, and important practical/theoretical questions remain open (stability, overheads, and regime-dependent behavior). With stronger large-scale results, robustness analyses, this could move to an accept and I am willing to reevaluate accordingly.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Authors propose a deep (residual) neural network that, at every layer, employs two modules: an expansion module (that encourages the layer to project the latent variables onto a higher rank); and a compression module (that brings closer the representation of latent vectors that are already similar). Authors evaluate the method empirically (classification on CIFAR and Imagenet) and also qualitatively (by visualizing attention map per attention head).

Overall, the paper is potentially a very good contribution. But the presentation is not ripe yet. The authors may reconsider restructuring the paper and re-submitting at a future time. I will not recommend the paper to be accepted as-is, nor will I review a completely different version. I will unlikely change my opinion with just simple write-up improvements, as the paper (and experiments) deserve a complete re-write for consideration.

### Strengths
* There can be several motivations of this work, including:
  * Whitebox models: visualizing saliency maps can inform users what part of the inputs is the model latching on. 
  * Unsupervised learning can create some latent space that can be used later for supervised tasks.
  * Potentially useful for training compression models.
* The results are competitive, yet the training is fast.

### Weaknesses
# Notation is confusing
* Why not clearly put an equation of what $\mathbf{Q}$ equals to? The text above equation 5 says it is an orthonormal basis for $\mathbf{Z}$. In that case, shouldn't be a function of $\ell$ also?
* Line 119 defines $U_{[K]}$... What is $\in \mathbb{R}^{d \times K_p}$? Is it the entire $U_{[K]}$ or the constituents $U_s$? The notation sas the first but the intuition says the second. **please** be more exact.
* The paper mentions what is $\mathbf{Q}$ (orthonormal basis for $\mathbf{Z}$) but never mentions what is $\mathbf{Q}_k$. I spent a fair amount of time to *guess* what is $\mathbf{Q}_k$ and I think I have a great guess! It is exactly the orthonormal basis of $U_k \mathbf{Z}$. Am I right? I deserve a brownie! But why not explicitly mention it? Trade handwavey informal text for method completeness and conciseness.
* Line 210: What is "each feature" $z \in \mathbf{Z}$? Is $\mathbf{Z}$ a matrix  and you are iterating over every row $z$ of $\mathbf{Z}$?
* The paper abruptly shifts from matrix form $\mathbf{Z}$ to vector form $z$. Why not stick to matrix form? This way, your $\pi_k$'s are always vectors rather than suddenly becoming scalars, etc.

# Unsupported Claims
* "**principled**" the paper keeps mentioning that other methods are not principled but the proposed method is principled. Please be specific. Principled in what way? What properties must be satisfied by a "principled method"? The paper must present proofs to show that the proposed method is principled, per whatever way the paper chooses to define "principled", and then show why other methods do not satisfy the definition.
* "**nearly linear complexity**" mentioned in abstract is not supported. Isn't it exactly linear, if one assumes that $K << n$? Either way, you should have some note about the complexity in the main paper, even if the derivation/proof is delayed until the appendix.


# Writing is unfit for ICLR
* You use word "thus" on line 150. IU would expect some proof or reference. Otherwise, downplay and use "could" or "should".
* Equation 3 has a math bug. Last row should use subscript $K$.
* The paper uses too much handwavy text and not much math. Notation is an art: it is supposed to deliver the information without using too many words. Consider doing so especiallly in section 3.
* Writing is too low-level for first paragraph of the intro. Why are we diving so deep in related work already?
* The application domain -- "images" is only mentioned at the last sentence of the abstract, and not mentioned in intro. It makes me want to believe that the method is "more general than just images" but there are no experiments to support that.
* The word "respectively" on line 72 seems inappropriately used.
* Line 102: "stands for the probability"... please be more specific. The probability is a scalar and not a vector.

# Experiments are not compelling
* There is no experiment that pushes me to want to use this method immediately. Perhaps it is under-sold or creativy is needed for better sales pitch? When and why would I use this method?
* Figure 5 visualizes on layer of each model. Does it make sense to visualize more layers?
* Table 2 does not mention the dataset it was conducted over.

### Questions
* Why have Cholesky Decomposition in the Abstract? It seems that it is only to find some matrix decomposition as a subroutine for Randomized SVD?

* Is it that for every $i \in [n]$ we have $\sum_{j \in [K]} \Pi_{i j} = 1$? If so, why not explicitly add that ear Equation 1?

* Is it possible to show image reconstructions?

* Why is "regularization" needed for cholesky decomposition? It definitely changes the charecteristics of the gram matrix. I assumed the normalization (line 994) is sufficient to make Y^T Y full-rank i.e. succeeding the "Cholesky" and the "solve_triangle" instructions.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces ECAttention, an attention mechanism derived using the principle of Maximal Coding Rate Reduction. It models feature expansion and compression as operations that adaptively change the dimensionality of the feature subspace, where feature expansion is achieved by projecting features onto the null space of the overall data,  while compression is achieved by projecting features toward their respective class-specific column spaces.
To make this geometric operation computationally feasible, the paper uses a randomization method combined with Cholesky decomposition.

### Strengths
- Novelty of proposed attention like layer: The proposed ECAttention layer appears to be novel, fixing the flawed approximation in prior art.
- Promising Interpretability: Qualitative results provide evidence in favor of the intepretability of the proposed method.
- Potential scalability: The randomized approximation makes the proposed idea potentially scalable.

### Weaknesses
- Empirical Validation: As per table-1 ECA seems to trail TOST on the larger scale ImageNet/ImageNet Real datasets. Given the claimed superiority of the proposed method, I find this result quite surprising. While an explanation is provided in terms of the value of the parameter ‘r’, I find it unconvincing. It is on the authors to provide the relevant evidence for their claims. What is preventing the authors from exploring larger ‘r’ for larger models?
- Unclear Advantage over alternatives: While figure 2 aims to provide evidence in favor of ECA in terms of inference time and memory usage, especially in comparison to ToST, I find the evidence unconvincing. It is unclear from the presented results, if the proposed ECA method is preferable over ToST, and if yes, then in what scenarios/conditions.
- Missing comparison to non-whitebox approaches: Where does this work stand in comparison to data-driven non-whitebox approaches? It is not clear from the text how this paper pushes the frontier on the whitebox models, bringing them closer to SOTA (non-whitebox) method on the studied benchmarks.

Overall, I find the method rather ad-hoc, as it is based on a principle (MCR2) rather than being data driven. This by itself is not a major obstacle to acceptance, however, coupled with unconvincing empirical validation establishing superiority over alternatives (e.g. ToST, or even Imagenet classification in general), it is unclear why the proposed method would be of use to the community as a ready to use method or as a promising research direction.

### Questions
Please comment on the questions raised in the weaknesses section. Happy to be convinced otherwise.

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
3

### Summary
The authors propose a transformer architecture/parameterisation based on maximising the maximal coding rate reduction objective (MCR$^2$) of Chan et al. (2022). The objective posits that the transformer block encodes $K$ distinct clusters/downstream tasks, where $K$ is a user-specified hyperparameter (e.g., $K = 6$ in some of the experiments). Maximising the objective, defined as the mutual information $I[Z; \pi]$ between the representations $Z$ and their soft assignment $\pi$ to one of the $K$ classes, encourages the representations to be as informative as possible for each task.

The authors specifically assume that both $Z$ and $Z \mid \pi$ are Gaussian distributed, thus $I[Z; \pi]$ is available in closed form. In particular, the authors consider the decomposition $I[Z; \pi] = h[Z] - h[Z \mid \pi]$, which turns into a difference in the log-determinants of $Z$ and $(Z \mid \pi)$'s covariance matrices. Unfortunately, working with full covariance matrices would be computationally too expensive. Motivated by this challenge, the authors propose parameterising these covariance matrices in a way that balances representational capability and computational efficiency.

The authors perform qualitative and quantitative experiments on toy datasets. Concretely, they demonstrate that 1) their transformer architecture learns to extract more diverse and (hopefully) useful features than comparable methods and 2) for smaller model sizes, their method achieves higher accuracy than comparable methods. 

## References

Kwan Ho Ryan Chan, Yaodong Yu, Chong You, Haozhi Qi, John Wright, and Yi Ma. Redunet: A white-box deep network from the principle of maximizing rate reduction.

### Strengths
The authors' approach is reasonably well-motivated and demonstrates reasonable performance. I particularly appreciated the quantitative comparisons in Figure 5, which show that their transformer architecture extracts more diverse and meaningful representations.

### Weaknesses
The paper's two main weaknesses are the lack of "exact" baselines in the experiments and the writing.

## The baseline

The authors only compare their method against other approximate methods. Concretely, they lack comparisons to 1) "standard" multi-head self-attention (MHA) transformers and 2) to ReduNet networks that do not introduce any approximations.
As such, it is difficult to ascertain
 1. how much benefit does the authors' method bring compared to just using "standard" MHA and
 2. how severe is the approximation that they make?

Including these baselines would significantly strengthen the paper.

## The writing

Generally, there are two issues with the writing.
First, the authors assume the reader is far too familiar with ReduNet and related approaches, which is reflected in the writing in various ways:
- The Introduction section is already too much of a literature review; I was lost by the end of the first paragraph, as it essentially consists of an enumeration of competing methods.
- Abbreviations such as ToSt, CRATE, and TSSA are undefined.
- There is no discussion of the MCR$^2$ objective; where it comes from, what it means, etc. I could only understand the motivation after skimming the original ReduNet paper.
- Across the main text, there are several places where the authors explain how their method works better than competing methods, but those competing methods are never introduced. As such, these comparisons are lost on readers like me, who are unfamiliar with them.
- At the same time, several interesting and relevant details are omitted or relegated to the appendix. For example, why is the direct column-space projection more numerically unstable than subtracting the null null space projection? What purpose do the $U_k$ terms serve?

Second, despite the authors' claim that their method is principled, I did not find that their exposition bore this out. Generally, the authors should spend more time motivating and explaining the objective function and less time comparing their method to competing ones if they do not intend to describe them in sufficient detail.

Miscellaneous:
 - "inter-dimensional correlations" -- no need for the adjective
 - "first tokenized into vectors X= [x1,...,xn] ∈RD×n (i.e., tokens)" -- repetition
 - Why does "multi-head self-attention operator" abbreviate to "MSSA"?
 - Eq 3: bottom row of column vector on right-hand side: should be $U_K Z^\top$ instead of $U_1 Z^\top$.
 - Figure labels are too small and are unreadable without zooming in; please increase font size.

### Questions
n/a

### Soundness
2

### Presentation
2

### Contribution
2
