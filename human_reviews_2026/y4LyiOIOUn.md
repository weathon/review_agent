# Permutation-Consistent Variational Encoding for Incomplete Multi-View Multi-Label Classification

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
Incomplete multi-view multi-label learning is fundamentally an information integration problem under simultaneous view and label incompleteness. We introduce Permutation-Consistent Variational Encoding framework (PCVE) with an information bottleneck strategy, which learns variational representations capable of aggregating shared semantics across views while remaining robust to incompleteness. PCVE formulates a principled objective that maximizes a variational evidence lower bound to retain task-relevant information, and introduces a permutation-consistent regularization to encourage distributional consistency among representations that encode the same target semantics from different views. This regularization acts as an information alignment mechanism that suppresses view-private redundancy and mitigates over-alignment, thereby improving both sufficiency and consistency of the learned representations. To address missing labels, PCVE further incorporates a masked multi-label learning objective that leverages available supervision while modeling label dependencies. Extensive experiments across diverse benchmarks and missing ratios demonstrate consistent gains over state-of-the-art methods in multi-label classification, while enabling reliable inference of missing views without explicit imputation. Analyses corroborate that the proposed information-theoretic formulation improves cross-view semantic cohesion and preserves discriminative capacity, underscoring the effectiveness and generality of PCVE for incomplete multi-view multi-label learning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces Permutation-Consistent Variational Encoding (PCVE), a novel framework for addressing the challenges of incomplete multi-view multi-label classification (iM3C). By leveraging information bottleneck (IB) principles with a permutation-consistency regularization, the framework promotes semantic alignment across views while mitigating redundancy and over-alignment. The framework is extensively evaluated on five multi-label benchmarks under both incomplete and complete settings, achieving SOTA performance.

### Strengths
1.	The paper introduces a novel permutation-consistency objective that aligns cross-view latent representations while reducing computational complexity, offering a scalable and effective solution for semantic alignment.
2.	The proposed framework balances view-specific information preservation and cross-view semantic consistency well, avoiding over-compression and ensuring task-relevant representation adequacy.
3.	PCVE achieves SOTA results across multiple metrics in both doubly incomplete and fully observed settings, demonstrating its robustness and generality.

### Weaknesses
1.    The permutation-consistency mechanism and information bottleneck formulation are innovative, but the novelty over existing methods (e.g., DICNet, NAIM3L) is not sufficiently clarified. Additionally, the framework’s reliance on PoE-based fusion and variational inference may be similar to prior multi-view learning approaches, like SIP.
2.	Some statements are unclear, such as "we randomly select view indices from the available view set V without replacement." What does this mean?
3.	Parameter sensitivity experiments on alpha seem to be missing, do the authors have any reason to clarify it further?
4.	The dataset descriptions lack sufficient detail, making it difficult to assess their relevance to iM3C tasks. Authors are advised to provide more details about the datasets.

### Questions
See the Weaknesses.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes a PCVE framework to address the dual incompleteness problem in multi-view multi-label learning. The idea is to use the permutation consistency regularization to constrain the distribution of latent variables of different views in the encoding stage, so that the shared semantics across views are aligned. At the same time, the reconstruction term is used to retain effective information, and PoE is used to fuse to form a shared representation for multi-label prediction. The authors give the target decomposition from the IB perspective and the corresponding variational upper and lower bounds, claiming that it can outperform a variety of strong baselines at different missing rates. The highlight of the work is to pre-place the consistency constraint in the encoding stage and introduce the distribution cluster exchange of random permutation to reduce complexity.

### Strengths
1. The problem is highly motivated for the practically important dual incomplete scenario. The complexity of this problem has not been considered in existing research, and this paper grasps this difficulty and helps real-world applications.
 
2. The methods proposed in this paper have a unified form, such as encoding consistency, view-private information reconstruction, PoE fusion and masked multi-label learning, and the overall design is self-consistent.

3. This article systematically derives the objectives from IB and provides the variational bounds and training objectives that can be implemented for optimization.

### Weaknesses
1. Will the random permutation proposed by the authors introduce additional noise? Also, will swapping them at each training epoch affect the stability of the training?

2. The $z^1$ in Figure 1 does not match the description in the text, which may cause ambiguity. I suggest that the authors revise Figure 1 to keep the symbols consistent.

3. There is a lack of sufficient explanation on how missing views are handled. How did the authors handle missing views during the encoding stage? I believe that the way to handle missing view is an important step in the method of this paper.

4. What is the scope of application of PCVE proposed in this paper? In real applications, can it handle heterogeneous multimodal data?

5. Although the computational complexity has been greatly reduced thanks to the proposed random permutation strategy, I am still interested in the overall parameter count and computational efficiency because it is different from the conventional VAE framework based on single modality encoding.

### Questions
Please refer to the weaknesses for my considerations and concerns.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper tackles the double-missing problem in incomplete multi-view and multi-label (iM3C) learning by proposing a Permutation-Consistent Variational Encoding (PCVE) framework based on the Information Bottleneck principle. To suppress view-private redundancy and prevent over-alignment, the framework introduces a scalable permutation consistency regularization mechanism via "distributed cluster swapping" with random permutations. Additionally, reconstruction terms are incorporated to preserve essential information and avoid representation collapse. Extensive experiments under various benchmarks and missing ratios verify that the proposed approach achieves consistent improvements over existing state-of-the-art techniques.
.

### Strengths
1. This work innovatively advances cross-view consistency to the encoding stage and introduces a "permutation consistency" regularization strategy, which effectively achieves efficient alignment while balancing sufficiency and compactness of representations. The proposed method is intuitive with a unified framework.
2. The objective decomposition and variational lower/upper bounds are derived from the Information Bottleneck perspective, which is clearly motivated and well defined. (e.g., the motivation for the minimum sufficiency of conditional mutual information).
3. The authors conduct extensive evaluations across five standard datasets and six metrics, considering both complete and double-missing scenarios. Comparison against nine strong baselines demonstrates stable and leading performance.
4. The readability of this paper is good, with clear motivations and strong experimental results supporting.

### Weaknesses
1. Suppose the mutual information of each view about y is equal, and "the shared information is sufficient for prediction". It may not be suitable in strong heterogeneous modes or tasks (such as visual-language-audio).
2. In this paper, each view needs to construct the encoder sub-distribution $r_v^n$ to each target view, but there is a lack of analysis regarding the number of model parameters, training time, etc. I suggest the authors supplement the relevant experiments.
3. The explanation of some symbols needs to be strengthened, such as what is the relationship between $p$ and $p_i$. The lack of a clear explanation will lead to confusion among readers.

### Questions
1. In PoE fusion and consistency regularization, are unequal weights of views considered? For example, the weight of low-quality views with large variance is reduced.
2. How does the PCVE handle missing labels? It seems that this part (handle missing labels) should be clarified. 
3. There is a typo in line 59, "IIn".

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
4

### Summary
This paper explores the challenging problem of multi-view and multi-label classification with both view and label incompleteness. The authors propose a framework named Permutation-Consistent Variational Encoding (PCVE), which incorporates an information bottleneck strategy to learn variational representations that can aggregate shared semantics across views. PCVE maximizes a variational evidence lower bound (ELBO) to preserve task-relevant information and introduces a permutation-consistency regularization to encourage distributional alignment between encoded representations of the same semantic content across different views.

### Strengths
- The work defines an interesting and meaningful learning setting, incomplete multi-view missing multi-label classification (iM3C).
- The proposed universal variational encoding framework is flexible and theoretically capable of handling arbitrary missing-view and missing-label patterns.
- The idea of permutation consistency provides a potentially simple yet effective strategy for encouraging cross-view semantic consistency.

### Weaknesses
1. The experimental results are not convincing and the novelty of this paper is not supported by the experiments. In Table 1, PCVE achieve trivial gains against other methods. 
2. The paper is difficult to follow. For example, Fig. 1 is neither properly referenced nor introduced in the main text, which harms overall readability.
3. The derivations are extensive and dense; many could be moved to supplementary materials. Some portions appear overly “formulaic,” potentially obscuring the main conceptual contributions.
4 Though the paper emphasizes learning shared semantic information, there is no intuitive or visual analysis of the latent variable z(v).
5. Stochasticity Risk in Permutation Strategy. While random permutation reduces computational complexity, it may introduce training instability. This could be particularly problematic with severely imbalanced view quality (e.g., pairing a high-quality view with a low-quality/noisy one, potentially hindering convergence or leading to sub-optimal performance.
6 Limitations in Experimental Setup.  The missing data pattern is randomly generated, failing to simulate the more common real-world scenario of non-random missingness (e.g., where the absence of a view correlates with sample characteristics), which limits the demonstration of robustness in practical applications.

### Questions
1 Regarding Assumptions and Information Loss: Assumption 2.1 posits that all views havethe same mutual information with the target, When this assumption doesn't hold in practice(e.g. one view is significantly more discriminative), how does PCVE prevent valuable, unique semantic information from this "strong view" from being prematurely suppressed ordiscarded during the permutation-consistency process?
2 Comparison of Alignment Mechanisms: Compared to traditional alignment methods thatdirectly minimize distribution distance (e.g., KL divergence) in the latent space, could youprovide more direct experimental evidence (e.g., visualizations or quantitative metrics) todemonstrate the specific advantages of permutation consistency in avoiding "over-alignment" and preserving necessary diversity?
3 Real-World Missing Patterns: The current experiments use random missingness. Howwould PCVE's performance be affected if the missing pattern were non-random (e.g. theabsence of a certain view is strongly correlated with samples of specific categories)? What isyour view on the robustness of your method to such bias?
4 Applicability beyond image features: Has the approach been validated on truly multimodaldata (e.g., image-text or audio-visual tasks)? lf not, what challenges would arise whenextending PCVE to such heterogeneous domains?

### Soundness
2

### Presentation
3

### Contribution
2
