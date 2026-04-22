# Beyond DAGs: A Latent Partial Causal Model for Multimodal Learning

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Directed Acyclic Graphs (DAGs) are a standard tool in causal modeling, but their suitability for capturing the complexity of large-scale multimodal data is questionable. In practice, real-world multimodal datasets are often collected from heterogeneous generative processes that do not conform to a single DAG. Instead, they may involve multiple, and even opposing, DAG structures with inverse causal directions. To address this gap, in this work, we first propose a novel latent partial causal model tailored for multimodal data representation learning, featuring two latent coupled variables parts connected by an undirected edge, to represent the transfer of knowledge across modalities. Under specific statistical assumptions, we establish an identifiability result, demonstrating that representations learned by MultiModal Contrastive Learning (MMCL) correspond to the latent coupled variables up to a trivial transformation. This result deepens our understanding of the why MMCL works, highlights its potential for representation disentanglement, and expands the utility of pre-trained models like CLIP. Synthetic experiments confirm the robustness of our findings, even when the assumptions are partially violated. Most importantly, experiments on a pre-trained CLIP model embodies disentangled representations, enabling few-shot learning and improving domain generalization across diverse real-world datasets. Together, these contributions push the boundaries of MMCL, both in theory and in practical applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper argues that a single DAG poorly captures large-scale multimodal generative processes and proposes a latent partial causal model with coupled visual/text factors ($z_x$, $z_t$) linked by undirected cross-modal dependencies. Under distributional and invertibility assumptions, it shows MMCL representations are identifiable up to simple transforms with respect to these latent variables, implying component-wise disentanglement potential. Using pretrained CLIP, few-shot and domain-generalization experiments report consistent gains across several real datasets.

### Strengths
- This paper is well-motivated. The author systematically argues that the generation process of large-scale multimodal alignment data has heterogeneity and may have opposite causal directions. The assumption of a single DAG is too limited and needs to be relaxed to describe "partial causality", which is consistent with the actual collection process of web scale text image paired data.

- The writing is overall clear. The contribution and limitations of the theory proposed in the article are analyzed.

### Weaknesses
- The identifiability proved here is representation-level (up to simple transforms) rather than causal-effect identifiability.  I suggest the authors explicitly clarify this distinction, since the use of a "causal graph" and the term "partial causal model" may easily lead readers to misinterpret that causal-effect identification is being addressed.
- According to the paper’s theoretical analysis, the identifiability and component-wise disentanglement properties of MMCL under the proposed latent partial causal model should not be specific to the visual branch. Yet the empirical evidence focuses almost exclusively on disentangling CLIP’s image features. If the theory is modality-agnostic, we would expect similar benefits when (i) disentangling the text branch and (ii) optimizing truly multimodal objectives.
- Typos or Mistakes:
  - Line 169, $\tau$ should be a temperature hyper-parameter.
  - In the caption of Table 2, there are two extra ①.
  - Line 1730, Sec. ??.
- No additional method or component is introduced to correspond to the theory; the paper relies on existing techniques (PCA and FastICA).

### Questions
See Weaknesses.

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
The authors critique the standard assumption that a single Directed Acyclic Graph (DAG) can model the generative process of large-scale multimodal data. They argue that real-world data (e.g., image-text pairs) often result from heterogeneous processes, including conflicting causal directions like text-to-image and image-to-text. To address this, they propose a novel latent partial causal model. This model separates latent variables into latent coupled variables and modality-specific variables. The paper's core theoretical contribution is an \textbf{identifiability result}: it proves that representations learned by MultiModal Contrastive Learning (MMCL) can recover the true latent coupled variables ($z_x, z_t$) up to a simple transformation.  This theory implies that models like CLIP learn disentangled representations, which are "mixed" by a simple transformation. The authors validate this practically by showing that applying post-hoc FastICA (for the hypersphere case) or PCA + FastICA (for the convex body case) to pre-trained CLIP embeddings improves performance on few-shot learning and domain generalization tasks.

### Strengths
1. The paper clearly identifies a valid and important weakness in existing causal models for multimodal learning. The observation that web-scale data is a mixture of generative processes (e.g., text-to-image and image-to-text) is sharp and provides a strong foundation for their new model.

2. The connection between the abstract theory (identifiability of $z_x$) and a simple, practical recommendation (use FastICA on CLIP embeddings) is a significant strength. It makes the theory testable and useful.

3. The experiments are very effective. Showing that a simple, off-the-shelf, post-processing step (FastICA) on a pre-trained CLIP model can improve performance across 16+ real-world datasets is a strong validation of the theory. The qualitative disentanglement on CelebA (Figure 3) is also very illustrative.

### Weaknesses
1. The paper's core motivation is replacing DAGs with a model featuring an "undirected edge" between $z_x$ and $z_t$. However, the actual mathematical parameterizations (Eq. 4 \& 7) define a directed conditional probability, $p(z_t|z_x)$. This creates a mismatch between the conceptual claim and the formal execution.

2. The proofs rely on the conclusion that optimal encoders must be \emph{perfectly} independent of modality-specific variables ($m_x, m_t$). This is an extremely strong assumption, derived from a necessary condition for optimality, and is unlikely to hold perfectly in practice.

3. The paper provides two methods (FastICA and PCA + FastICA) based on two different theories but offers no clear heuristic to determine which is appropriate for a given model or dataset.

4.  The paper claims "component-wise" identifiability but validates it with \emph{qualitative} and \emph{semantic} disentanglement (e.g., "Smile"). It lacks standard \emph{quantitative} disentanglement metrics (MIG, SAP, etc.) to support this strong technical claim.

### Questions
1. The central modeling innovation is the "undirected edge" between $z_x$ and $z_t$. However, the generative parameterizations in Eq. (4) and Eq. (7) both define a directed conditional probability $p(z_t|z_x)$. How does this parameterization formally represent an "undirected" relationship, as opposed to a standard directed model $z_x \rightarrow z_t$? Is the "undirected" nature meant to imply that the model is a marginalization of a mixture of DAGs (like those in Figure 1), and that $p(z_t|z_x)$ is just one assumed form for this marginalized relationship?

2. In the proofs of Theorem 4.1 and 4.2 (Appendices D.1 and E.1), a critical step relies on Lemma 2. This lemma states that the optimal solution is achieved when $h_x(m_x, z_x) = h_t(m_t, z_t)$ almost surely. By differentiating, the authors conclude that the optimal encoders must be independent of $m_x$ and $m_t$. This implies that the MMCL loss perfectly enforces the encoders to discard all modality-specific information. Is this a reasonable assumption for the proof, or is it an assumed condition for optimality that may not be achievable? How does the identifiability theory hold if there is minor "leakage" of $m_x$ into the representation?

3. The paper offers two practical methods: FastICA for hyperspheres (Corollary 1) and PCA + FastICA for convex bodies (Corollary 2). Practically, how should one choose between these two methods? Is there an empirical test to determine the "geometry" of CLIP's latent space? Corollary 2 (convex bodies) implies the learned representation $f_x(x) = Pz_x + c$ is already disentangled. Why does this "simpler" case require a more complex solution (PCA + FastICA)? The paper states PCA is needed "to account for the orthogonal transformation introduced by PCA," which is circular.


4. The CelebA experiments show compelling semantic disentanglement ("Smile," "Gender"). The theory, however, promises "component-wise" identifiability. Does this result imply that the true, independent latent factors for CelebA \emph{are} these high-level semantic attributes? Have you considered quantitatively measuring disentanglement (e.g., MIG, DCI, SAP) by training a linear classifier on the disentangled representations to predict the ground-truth attributes?

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
1

### Summary
The paper proposes a latent partial causal model to generalize beyond traditional DAGs, capturing undirected multimodal dependencies and proving identifiability in contrastive learning frameworks like CLIP.

### Strengths
- The extension of the causal structure in multimodal learning beyond DAGs is a valuable conceptual exploration.
- The identifiability proofs are soundly constructed.

### Weaknesses
I don't have enough knowledge to evaluate this paper as I don't work in related domains. For me, it seems the paper did not empirically demonstrate that the learned representations encode genuine causal relations between modalities.

### Questions
Can the proposed method be tested on text domain datasets?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a Latent Partial Causal Model for multimodal data (image and text). Within this framework, the authors establish identifiability guarantees between the learned multimodal contrastive representations and the underlying generative latent factors. Specifically:

- When the true generative factors lie on a unit hypersphere, the learned representations are identifiable up to a linear transformation.

- When the true generative factors lie in a convex body, the learned representations are identifiable up to permutation, scaling, and shifting of the true generative factors.

These results imply that multimodal contrastive learning (MMCL), such as CLIP, has the potential to learn disentangled and meaningful representations rather than arbitrary embeddings.
In experiments, the authors first validate these theoretical claims through synthetic experiments. They then apply their method to pretrained CLIP embeddings and demonstrate that the resulting representations improve performance across various downstream tasks.

### Strengths
**S1.** The proposed model is interesting and suitable for multimodal settings.

**S2.** The identifiability analysis is solid, and the techniques used in the proofs are interesting.

**S3.** The downstream experiments using disentangled representations span several application areas, and the learned representations demonstrate better qualities compared to the baseline.

### Weaknesses
**W1.** The gap between the theoretical analysis and practical use is not clear. **The identifiability results rely on several assumptions** in Equations (4) and (7) (especially the probability density assumptions), along with assumptions “the latent space is the unit hypersphere” / “the latent space is a hyperrectangle.” **However, it is unclear whether the assumptions hold.** If I understand correctly, these assumptions are made on the ground-truth underlying generative factors. I suggest adding more discussion of what these assumptions mean at a high level (not just mathematically), why they are reasonable, and in which situations they may be violated.

**W2**. There is a lack of a formal definition of disentanglement. The paper highlights that a key practical use of the theory is to obtain disentangled representations using a pretrained multimodal encoder (e.g., CLIP). However, the discussion in Section 5 is high-level and lacks mathematical grounding, which could lead to misunderstandings. For example, the intended disentanglement is among the components of $z_x$ (or $z_t$), not disentanglement between $z$ and the modality-specific noise $m$, correct? I suggest including a precise definition of disentanglement in the paper and discussing it more clearly.

**W3.** The validation of the identifiability results is based on low-dimensional synthetic experiments, and the evaluation of disentangled representations on real-world data is done purely through downstream tasks. I suggest adding a semi-synthetic dataset (e.g., Causal3DIdent[1], Morph-MNIST[2]) that contains higher-dimensional ground-truth latent variables, so disentanglement can be evaluated directly rather than indirectly via downstream performance.

[1] Von Kügelgen, Julius, et al. "Self-supervised learning with data augmentations provably isolates content from style." Advances in neural information processing systems 34 (2021): 16451-16467.

[2] Castro, Daniel C., et al. "Morpho-MNIST: Quantitative assessment and diagnostics for representation learning." Journal of Machine Learning Research 20.178 (2019): 1-29.

### Questions
**Q1** Can the authors explain more about the aggregated functions h_x and h_t, and how they are used in the proof (Lines 964–977)? Is h_x(m_x, z_x) or h_x(z_x) actually h_x(m_x, z_x, m_t, z_t)? Also, are there smoothness or differentiability assumptions on h required for the proof?

**Q2**. In Corollaries 1 and 2, the learned representations are an invertible linear transformation of the ground-truth vectors, or a permutation after scaling and shifting. **Does this mean the dimensionality of the learned representations must match the dimensionality of the ground-truth latent variables?** If so, this seems impractical in real-world settings (e.g., using a pretrained encoder like CLIP, where we do not know the latent dimensionality). **Can the theorem be generalized when there is a dimensionality mismatch between the learned representations and the ground truth? Does this mismatch affect the practical applicability of the method?**

**Q3.** It appears that the proposed partial causal model (Figure 2) subsumes all the DAG structures shown in Figure 1 and can represent any setting where there is some correlation between $z_x$ and $z_t$. If so, why is this model considered “causal,” and why is it described as partial causal rather than a general latent-variable model?

### Soundness
3

### Presentation
2

### Contribution
3
