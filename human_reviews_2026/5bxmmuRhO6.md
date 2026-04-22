# Supporting Multimodal Intermediate Fusion with Informatic Constraint and Distribution Coherence

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Based on the prevalent intermediate fusion (IF) and late fusion (LF) frameworks, multimodal representation learning (MML) demonstrates its superiority over unimodal representation learning. To investigate the intrinsic factors underlying the empirical success of MML, research grounded in theoretical justifications from the perspective of generalization error has emerged. However, these provable MML studies derive the theoretical findings based on LF, while theoretical exploration based on IF remains scarce. This naturally gives rise to a question: **Can we design a comprehensive MML approach supported by the sufficient theoretical analysis across fusion types?** To this end, we revisit the IF and LF paradigms from a fine-grained dimensional perspective. The derived theoretical evidence sufficiently establishes the superiority of IF over LF under a specific constraint. Based on a general $K$-Lipschitz continuity assumption, we derive the generalization error upper bound of the IF-based methods, indicating that eliminating the distribution incoherence can improve the generalizability of IF-based MML methods. Building upon these theoretical insights, we establish a novel IF-based MML method, which introduces the informatic constraint and performs distribution cohering. Extensive experimental results on multiple widely adopted datasets verify the effectiveness of the proposed method.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper provides a theoretical analysis of multimodal representation learning (MML) with a focus on intermediate fusion (IF) and late fusion (LF) frameworks. Unlike prior studies centered on LF, the authors theoretically justify the superiority of IF under certain constraints and derive a generalization error bound based on a K-Lipschitz continuity assumption. They further propose an IF-based MML method that introduces an informatic constraint and enforces distribution coherence, achieving strong empirical results on multiple benchmark datasets. However, some parts of the theoretical derivation appear to rely on assumptions that may require further clarification or justification.

### Strengths
The paper presents a novel approach to multimodal representation learning (MML) by integrating theoretical analysis with practical model design. It provides a fine-grained theoretical investigation of intermediate fusion (IF) and late fusion (LF) frameworks, deriving a generalization error bound under a K-Lipschitz continuity assumption. Based on these theoretical insights, the authors propose an IF-based MML method that introduces an informatic constraint and performs distribution cohering to reduce distribution incoherence. This theory-driven design is validated through extensive experiments on multiple benchmark datasets, showing improved performance consistent with the theoretical predictions. The approach is innovative in that it connects theoretical guarantees with concrete architectural and training strategies, offering both conceptual and empirical contributions to the field.

### Weaknesses
While the paper presents a novel theory-driven approach to multimodal representation learning, there are some concerns regarding the theoretical assumptions. In particular, the analysis assumes that each dimension of the latent features can be strictly partitioned into task-dependent semantics and task-independent noise. This assumption provides a clear analytical framework but may be overly strong or unrealistic in practical deep learning settings, where latent features are often entangled and do not exhibit such a clean separation. As a result, some of the theoretical guarantees derived under this assumption might not fully hold in practice. Additionally, the theoretical derivations build upon prior results rather than introducing entirely new theorems, so the novelty in the theoretical contributions is somewhat incremental. Nevertheless, the empirical results indicate that the proposed IF-based method, with its informatic constraint and distribution cohering, effectively improves performance and is consistent with the theoretical intuitions, partially mitigating the concerns about the assumptions.

### Questions
1. The theoretical analysis assumes that each latent feature dimension can be strictly partitioned into task-dependent semantics and task-independent noise. Is there any evidence in the existing experimental results that indirectly supports this assumption?

2. In the proof, it is unclear how the first equality in Equation 31 connects the empirical error to the expected error. Could the authors clarify the assumptions or steps that justify this equality? Specifically, what conditions are required for this transition, and are they satisfied in the current setting?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper re-examines two mainstream fusion approaches in multimodal learning—Intermediate Fusion (IF) and Late Fusion (LF)—from a fine-grained perspective. It demonstrates that IF outperforms LF under specific constraints and further derives the generalization error upper bound of the IF method. Based on theoretical analysis, it proposes an IF method that incorporates information constraints and distribution consistency. Experimental results on a large number of datasets verify the effectiveness of this method.

### Strengths
1）The motivation is clear and the research significance is prominent. The paper identifies the issue that current theoretical research on multimodality mostly focuses on LF, while there is insufficient theoretical analysis on IF, and possesses a clear motivation to fill this gap.

2）The theoretical analysis is in-depth. It provides a comparison between IF and LF from the perspective of fine-grained dimensions, derives the generalization error upper bound, and makes significant theoretical contributions.

3）The ablation experiments are comprehensive and validate the necessity of the modules. Ablation experiments are conducted on the two core modules (information constraints and distribution consistency) to verify their respective effectiveness.

### Weaknesses
1）The experimental comparisons have gaps, which weakens the persuasiveness of the method's innovation. Current experiments mainly compare IID with LF-based SOTA methods (e.g., PDF, QMF) or general fusion models. However, direct comparisons with advanced methods specifically designed for the IF framework in recent years are lacking. It is suggested that the authors supplement relevant experiments, as such comparisons can more clearly highlight the unique advantages and contributions of IID under the specific paradigm of "Intermediate Fusion".

2）The analysis of computational complexity is insufficient. Although the paper identifies the computational bottleneck of high-dimensional Wasserstein distance and proposes an efficient dimensionality reduction method, it lacks quantitative analysis of the computational overhead of the overall IID model (including the two newly proposed modules) during training and inference. This makes it difficult for readers to comprehensively evaluate the efficiency of the method in practical applications.

### Questions
See the Weaknesses section for details.

### Soundness
3

### Presentation
3

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
This paper presents a novel framework for multimodal representation learning, with a theoretical analysis centered on intermediate and late fusion mechanisms. The proposed theory further leads to the implementation of the IID method.

### Strengths
1.Starting from a theoretical perspective, this paper systematically compares Intermediate Fusion (IF) and Late Fusion (LF) from a dimensional viewpoint, and provides rigorous proofs for both prediction error and generalization error.

2.The paper is well-structured and written in a professional academic style.

### Weaknesses
1.The errors in Figures 1 and 5 need to be corrected.

2.In the case of Intermediate Fusion (IF), it is generally assumed that Concat and Sum are equivalent, which often leads to the issue of modality laziness. However, the authors did not take this into account.

3.The notations are confusing. For instance, in the figures, w is shown as a vector, whereas in the main text it appears to be used as a scalar. It is recommended to include a notation table for clarification.

4.Although module D is theoretically derived, it appears to be meaningless from an empirical perspective, as the hyperparameter experiments (β ∈ {1e−10, …, 1e−14}) suggest negligible effects. The rationale for including module D should be further explained.

5.The proposed method is based on Intermediate Fusion (IF), yet the comparison with other methods is insufficient and somewhat unfair. Many advanced approaches within the IF framework are not considered.

6.The study lacks experiments involving additional modalities.

### Questions
1.In Table 2, the Concat method consistently outperforms IID-L. Does this indicate that modules I and D are not meaningful or contribute little to the model’s performance?

2.Taking PDF as an example, within the IID-P framework, larger weights are assigned to the dominant modality while smaller weights are given to the weaker ones. Consequently, the dominant modality converges faster whereas the weaker modality remains under-optimized. In this context, is it reasonable for module D to reduce the distribution discrepancy between features of well-converged and under-optimized modalities?

### Soundness
2

### Presentation
2

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
This paper explores why Intermediate Fusion (IF) outperforms Late Fusion (LF) in multimodal learning. The paper analyzes: (i) an existence result showing that IF performs no worse than LF under linear target mappings; and (ii) the generalization bound of IF under the K-Lipschitz classifier, where the distribution inconsistency between the feature distributions of each modality and the fusion distribution determines performance. Based on this, the authors propose the IID algorithm, which combines information constraints and distribution consistency. In six benchmark tests (visual language, RGB-D), IID outperforms the robust LF baseline and its IF variant; ablation experiments show that both modules contribute to the performance.

### Strengths
1. The authors prove that combining features from different modalities before classification (IF) is no worse than classifying them individually before weighting (LF). They then provide an upper bound on the generalization error, highlighting that the key factor affecting IF performance is the inconsistency between the feature distributions of different modalities and the combined distribution.
2. The authors use FFT sparsification, RIP projection, and unbalanced OT (Sinkhorn with KL relaxation) to approximate Wasserstein distances, the method is computationally practical and achieves consistent improvements across six benchmarks; ablations indicate both modules contribute.

### Weaknesses
1. The existence result IF better than LF assumes the target mapping 𝑔 is linear. In practice, post-fusion heads are often nonlinear. Can the claim be extended to nonlinear but K-Lipschitz heads? Please also include an experiment comparing a linear head vs. a two-layer MLP head to delineate the applicable regime.
2. The method minimizes unconditional (marginal) discrepancies between modalities, whereas decisions depend on class-conditional distributions. Under what conditions does shrinking marginal discrepancy guarantee per-class alignment and improved decision boundaries?
3. The bounds in the method involve the distance from each modality to the fusion distribution , but the objective function minimizes the distance between pairs of modes. Please specify under what assumptions minimizing the pairwise objective function is equivalent to the objective term from each mode to the fusion distribution.
4. The objective maximizes $I(z;y)-\sum_{m=1}^{M} I(z_m;y)$ and minimizes pairwise modality discrepancies, yet neither term explicitly penalizes cross-modal redundancy; indeed, alignment may amplify label-irrelevant shared patterns.
- **(i)** How does IID prevent reinforcing such redundant signals?
- **(ii)** It's better to refer to this paper for handling redundant information in related works.
- [1] X. Xiao, “Neuro-inspired information-theoretic hierarchical perception for multimodal learning,” in Proc. 12th Int. Conf. Learn. Represent., 2024, pp. 1–29.

### Questions
Same as mentioned in weakness.

### Soundness
2

### Presentation
2

### Contribution
2
