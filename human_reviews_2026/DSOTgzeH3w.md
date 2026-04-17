# On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy

- Decision: Accept (Poster)
- Scores: 8, 6, 4

## Abstract
Sparse autoencoders (SAEs) have recently emerged as a powerful tool for interpreting the features learned by large language models (LLMs). By reconstructing features with sparsely activated networks, SAEs aim to recover complex superposed polysemantic features into interpretable monosemantic ones. Despite their wide applications, it remains unclear under what conditions SAEs can fully recover the ground truth monosemantic features from the superposed polysemantic ones. In this paper, we provide the first theoretical analysis with a closed-form solution for SAEs, revealing that they generally fail to fully recover the ground truth monosemantic features unless the ground truth features are extremely sparse. To improve the feature recovery of SAEs in general cases, we propose a reweighting strategy targeting at enhancing the reconstruction of the ground truth monosemantic features instead of the observed polysemantic ones. We further establish a theoretical weight selection principle for our proposed weighted SAE (WSAE). Experiments across multiple settings validate our theoretical findings and demonstrate that our WSAE significantly improves feature monosemanticity and interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents the first theoretical framework analyzing the fundamental limits of Sparse Autoencoders (SAEs) for recovering monosemantic features from superposed polysemantic representations. The authors derive closed-form solutions showing that, under general conditions, SAEs cannot perfectly recover ground-truth monosemantic features due to feature shrinking and feature vanishing effects. They identify that full recovery is theoretically guaranteed only under extreme sparsity of the true features. To address this limitation, the paper introduces a Weighted Sparse Autoencoder (WSAE) that reweights the reconstruction loss according to the degree of polysemanticity of each input dimension. Theoretical analysis demonstrates that this weighting narrows the gap between the SAE loss and the ideal ground-truth reconstruction loss, and experiments on both synthetic data and pretrained models confirm that WSAE improves feature monosemanticity and interpretability without sacrificing reconstruction fidelity.

### Strengths
The paper’s originality lies in providing a formal, mathematical explanation for why SAEs sometimes fail to identify interpretable features, moving beyond the purely empirical understanding that dominates current mechanistic interpretability work. The authors establish clear analytical results—closed-form solutions, necessary and sufficient conditions, and a uniqueness theorem—that rigorously connect feature sparsity with successful monosemantic recovery. This theoretical grounding fills a long-standing gap in the field, where SAE behavior had been empirically impressive but conceptually opaque. The proposed reweighting principle is both simple and theoretically motivated, making it an elegant bridge between formal analysis and practical implementation.

In terms of quality and clarity, the paper is meticulously written, with precise mathematical derivations and well-structured proofs. The relationship between the SAE loss and the ideal ground-truth loss is clearly articulated and forms a compelling narrative that unifies the theoretical and empirical sections. The significance of the contribution is substantial: by formalizing the limitations of current interpretability techniques, the paper reframes the field’s expectations of what SAEs can and cannot achieve, and introduces a principled path forward through reweighted optimization. The experiments, while modest in scale, convincingly validate the theoretical predictions and demonstrate that the framework generalizes across modalities.

### Weaknesses
The main limitation is the scope of empirical validation. The experiments are primarily performed on small or medium-scale models (Pythia-160M, ResNet-18) and under controlled settings. While these choices are appropriate for validating the theory, it remains unclear how well the findings extend to large modern LLMs or to deeper, multi-layer SAE architectures that are increasingly used in practice. Furthermore, the superposition assumption in the theoretical model treats representations as linear mixtures, which simplifies the nonlinear interactions and attention-based dynamics found in real networks. As a result, the framework captures the core geometry of superposition but may not fully describe the behavior of realistic LLM feature spaces.

### Questions
1. How does the theoretical framework extend to non-linear or multi-layer autoencoders, especially when feature mixing occurs across layers? Could the authors discuss whether similar recovery limits hold in deeper settings?

2. The analysis assumes linear superposition of features. Have the authors explored how deviations from linearity—such as attention-weighted combinations or nonlinear feature interactions—affect the recovery bounds?

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
3

### Summary
This paper provides a rigorous theoretical analysis of the **feature recovery limits of Sparse Autoencoders (SAEs)**, a method widely used for interpreting polysemantic representations in large models.  
Under the **superposition hypothesis**, the authors derive a closed-form optimal solution for SAEs and prove that:
- In general, SAEs cannot perfectly recover the true monosemantic features due to *feature shrinking* and *feature vanishing* effects.
- Full recovery is guaranteed only under **extremely sparse ground-truth features** (sparsity factor \(S \to 1\)), i.e., nearly 1-sparse activations.
- To mitigate this, the paper proposes a **Weighted Sparse Autoencoder (WSAE)**, where per-dimension weights reduce the reconstruction gap between the observed polysemantic and unobserved monosemantic features.  
Theoretical results (Theorem 4–5) show that proper weighting minimizes this gap, and experiments on synthetic data and pretrained models (Pythia-160M, ResNet-18) demonstrate improved interpretability metrics.

### Strengths
- **Mathematically grounded analysis.**  
  The derivation of a closed-form SAE solution and identification of feature shrinking/vanishing phenomena clarify long-standing empirical observations in mechanistic interpretability.  
  While related to classical sparse coding theory, the explicit analytical form for the ReLU-based SAE and the formal characterization of these degradation modes constitute a novel operational understanding that was not previously formalized.

- **Bridging theory and practice.**  
  The framework connects abstract sparse recovery theory to interpretability practice in LLMs, offering insight into why SAEs sometimes fail to yield cleanly separable features.

- **Theoretically motivated improvement.**  
  The proposed WSAE provides a principled approach to reweight reconstruction according to estimated polysemanticity, improving monosemanticity without large losses in reconstruction quality (as evidenced in Fig.3(c)).

- **Transparent proofs and clear assumptions.**  
  All mathematical steps, assumptions (superposition, sparsity distribution), and limitations are clearly stated.

### Weaknesses
### 1. Relationship to dictionary learning and identifiability
The theoretical results closely parallel classical **identifiability conditions in sparse dictionary learning**, where full recovery requires incoherence or extreme sparsity of the underlying basis.  
While this correspondence is intuitive, it is not explicitly discussed in the paper.  
Clarifying this relationship would enhance the theoretical positioning of the work.  

In particular:
- Theorem 1–3 can be viewed as a **nonlinear (ReLU-based) extension of dictionary identifiability** results, where interference \(W_p^\top W_p - I\) plays the role of coherence.  
- The observed feature shrinking/vanishing could be interpreted as manifestations of partial non-identifiability under finite sparsity.  

Explicitly situating the paper in relation to established sparse coding theory (e.g., spark condition, mutual coherence) would help readers understand which parts of the contribution are new—e.g., the inclusion of ReLU nonlinearities and overcomplete encoders—and which are theoretical refinements of existing principles.

---

### 2. Implications for interpretability and practical impact
The paper’s findings—that SAEs cannot fully disentangle superposed representations except in extreme sparsity—have significant consequences for current interpretability research, though this point could be emphasized more strongly.

In particular:
- Many interpretability studies assume that increasing sparsity or width of SAEs improves feature separation indefinitely.  
  This work shows that such improvement **plateaus due to intrinsic representational interference**, meaning full disentanglement is mathematically impossible under realistic sparsity.
- Consequently, **SAE-based interpretability should be regarded as an approximation tool**, not as a faithful feature recovery mechanism.  
  This reframes SAE-derived neurons as *approximate projections of overlapping features*, rather than direct encodings of ground-truth concepts.

This reinterpretation could reshape how SAE-based analyses are used in mechanistic interpretability: rather than aiming for perfectly monosemantic neurons, practitioners might instead quantify or visualize residual interference between features.

A related empirical suggestion would be to measure the effective sparsity \(S\) of real LLM activations, to contextualize how close such models operate to the theoretical extreme-sparsity regime.

---

### 3. Sensitivity and robustness of the reweighting scheme
The WSAE introduces weights \(\gamma_i = s_i^{\alpha}\), with \(s_i\) estimated from variance or semantic consistency and \(\alpha\) controlling emphasis on monosemantic dimensions.  
The authors note (p.6) that results are “relatively robust” to α and show examples for α = 0.5 and 1.0 (Fig. 4), which supports this claim.  
Nonetheless, a small sensitivity analysis or ablation could strengthen the argument.  

Suggestions:
- A sweep of α (e.g., {0, 0.5, 1, 2}) to illustrate stability trends.  
- A comparison of different proxies \(s_i\) (variance vs. semantic metrics) to test transferability.  
- Discussion of reconstruction trade-offs: Fig.3(c) indicates that \(x_p\) reconstruction is maintained (i.e., no major Pareto penalty), but confirming this across datasets would reinforce generality.

These additions would verify that the reported improvements are not dataset-specific and that α tuning is unnecessary in practice.

---

### 4. Reproducibility and release
Appendix B.2 clearly reports compute resources and training time (e.g., 24 h on A100 for language models).  
It would still be helpful to confirm whether **code and theoretical implementations** will be released upon publication, enabling the community to replicate and extend the analysis.

---

### 5. Potential extensions
The theoretical framework appears extensible beyond reweighting—for instance, to alternative loss formulations or nonlinear encoder architectures that directly address the feature interference term \(W_p^\top W_p - I\).  
A short remark in the final version about such future directions would underscore the broader applicability of this analysis.

### Questions
1. How do Theorem 1–3 connect formally to existing identifiability results in sparse dictionary learning (e.g., spark or coherence conditions)?  
2. How sparse are real LLM activations compared to the “extreme sparsity” regime analyzed here?  
3. Could you include a small α-sweep or sensitivity plot to verify robustness?  
4. Does reweighting ever degrade reconstruction accuracy, or is the Pareto frontier generally preserved (as suggested by Fig. 3(c))?  
5. Will the theoretical framework and WSAE implementation be publicly released?  
6. Do you foresee extensions of this theoretical setup beyond reweighting—for example, alternative matrix designs that directly minimize cross-feature interference?

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
3

### Summary
This paper provides a theoretical analysis of SAEs which motivates an alternative SAE design proposal (WSAE). They claim WSAE improves feature mono-semanticity and interpretability.

### Strengths
- The paper provides a detailed theoretical analysis. 
- The paper proposes a new re-weighting strategy that may reduce polysemanticity. 
- the paper tests their new strategy on language models.

### Weaknesses
- The paper assumes an incorrect model of the underlying data distribution which is no longer considered valid by many researchers in the field. There is not one true set of non-overlapping features, but rather many kinds of features which overlap with one another. Additionally, many features, such as parts of speech are dense. While much work on SAEs has assumed sparse features in a non-overlapping basis - this was more reasonable to do a few years ago before we'd seen so much object level data. For example, see "Sparse Autoencoders Do Not Find Canonical Units of Analysis" or "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders" and indeed some SAE architectures have been proposed with this in mind "Learning Multi-Level Features with Matryoshka Sparse Autoencoders". 

- Empirical results on real data are filtered through the lens of an auto-interpretability scoring method which doesn't reflect the broader quality of an SAE. 

Figure 4 deliberately scales the x-axis to make minor differences in the semantic consistency metric look very large. It's unclear if this is meaningful or not.

### Questions
- I'd like to see theoretical analysis which assumes more complex underlying data distributions. Would this change the resulting conclusions or insights? 
- Use of established, if imperfect benchmarks like SAE bench could be used to provide a more comprehensive endorsement of the wSAE approach. For example, do wSAEs demonstrate less feature absorption?

### Soundness
2

### Presentation
2

### Contribution
2
