# Embryology of a Language Model

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 4, 0

## Abstract
Understanding how language models develop their internal computational structure is a central problem in the science of deep learning. While susceptibilities, drawn from statistical physics, offer a promising analytical tool, their full potential for visualizing network organization remains untapped. In this work, we introduce an embryological approach, applying UMAP to the susceptibility matrix to visualize the model's structural development over training. Our visualizations reveal the emergence of a clear "body plan," charting the formation of known features like the induction circuit and discovering previously unknown structures, such as a "spacing fin" dedicated to counting space tokens. This work demonstrates that susceptibility analysis can move beyond validation to uncover novel mechanisms, providing a powerful, holistic lens for studying the developmental principles of complex neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes an "embryological" lens on the training dynamics of a 3M-parameter, 2-layer attention-only language model.
The key idea is to compute **per-token susceptibilities** for each attention head, stack them into vectors (\eta_w(xy)), and then visualize these vectors with UMAP over the course of training.
The authors report (i) a long thin "rainbow serpent" manifold whose first two principal axes align with global expression/suppression (PC1) and a dorsal–ventral stratification tied to the induction circuit (PC2), and (ii) a newly described **"spacing fin"** associated with sequences of spacing tokens and their counts.

### Strengths
**Originality.**

* Using **susceptibility vectors** (rather than activations) to visualize response is an interesting perspective that complements circuit-centric methods. The "spacing fin" is a surprising, concrete emergent structure that the authors unraveled with their visualization. 
* The biological metaphor (anterior–posterior / dorsal–ventral axes) is consistant through the manuscript and helps organize observations about stratification by token pattern.

**Quality / Technical soundness.**

* The susceptibility definition and its sign interpretation (expression vs. suppression) are clearly stated, with an explicit covariance-based definition (Def. 2.1) and discussion. 
* Cross-seed visualizations (Appendix F) support claims.

**Clarity.**
The paper is easy to follow, generally well written, figures are plentiful and annotated.

### Weaknesses
**The probabilistic setup and tractability of the quenched posterior need more transparency.**

* Eq. (2) introduces the posterior ( $ p^{\beta}\_{n}(w) \propto \exp \\{-n \beta L(w) \\} \phi(w) $ ) with normalizer ( $ Z^{\beta}\_{n} $ ). The *practical* tractability of ( $ Z^{\beta}\_{n} $ ) and how its intractability propagates (or cancels) in ( $ \chi $ ) estimates are not discussed. 

**Motivation for Def. 2.1 could be surfaced earlier.**
The definition of susceptibility (Def. 2.1) appears before an intuitive build-up of why *this* covariance captures "expression/suppression." Consider moving the intuitive paragraphs ("Negative susceptibility… Positive susceptibility…") directly before the formal definition illustrating sign and magnitude. 

**Head labeling vs. permutation equivariance.**
Section 3 states (based on prior work) which heads are previous-token/current-token/induction (e.g., 0:1, 0:4, 0:5, 1:6, 1:7), but attention heads are *a priori* permutation-equivariant under reindexing. Please add a sentence clarifying **how** heads are *identified and matched across runs/checkpoints*, and how this resolves the labeling issue across seeds.

**Figures / Typo.**

* Fig. 2’s subplots don’t share a y-axis within pattern groups, making comparisons harder. Please share y-axes across rows where meaningful or include small multiples with identical scales. Also label heads (l:h) more prominently. 
* Minor: page 9 line ~450 "exhibition / exhibition" → "excitation / inhibition." (If "exhibition/inhibition" is deliberate, please justify the terminology.) 

**"Universal body plan" is stated strongly given one architecture/scale.**
The paper emphasizes universality across seeds of *one* tiny attention-only model with a specific tokenizer. Please temper claims or add more evidence: (i) a second tokenizer, or (ii) a small MLP-augmented transformer at the same scale, or (iii) at minimum, a dataset ablation showing how pattern frequencies (Fig. 3) modulate the observed geometry. Even a compact sensitivity table would help.

### Questions
**Suggestions:**

* Section 4.1 uses "Serpent" metaphors; we believe the term "Eel" (which has fins) could be better suited.
* On p. 6 you refer to "PC1/PC2" without first saying you did PCA; add a forward pointer to Appendix C. 
* On p. 9, line 450, typo "exhibition / exhibition" $\rightarrow$ "exhibition / inhibition".

In addition to the concerns raised in the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents an “embryology” lens for LM training by projecting per-token susceptibility vectors into 2D with UMAP, then tracking how the geometry evolves over training. This is meant to shed light on how language models develop their internal computational structure. The authors find the projection produces a striking rainbow serpent structure whose axes point towards the emergence of an induction circuit.

### Strengths
- The approach is unlike most views i've seen in terms of interpreting LLMs and is creative/novel.
- The paper presents a fairly holistic view of interpretability in LLMs. Instead of focusing on single circuits, the method reveals global organization and complementary expression/suppression roles across heads.
- Joint use of UMAP snapshots and per-pattern susceptibility trajectories might point to plausible temporal causal structure emergence.

### Weaknesses
- The results presented in the paper are entirely based on  a 3M, 2-layer attention-only model. It’s unclear whether the serpent structure and spacing fin persist or change in mid/large LMs with MLPs and modern tokenizers (e.g., non-whitespace-heavy merges).
- Although partly addressed, the method still relies on a nonlinear, stochastic embedding with known global-distance distortions; the work would benefit from corroboration via isometry-aware metrics in the original space.
- The approach may be sensitive to confounding, since the prominence of spacing tokens may be an artifact of the truncated GPT-2 vocab and dataset composition; more direct controls or alternate tokenizers would help.
- Overall, apart from the PC2 thickening statistic for induction, much of the case rests on visuals. More numerical results in the form of for instance statistical tests (e.g., separability indices, cluster stability, supervised recovery of pattern labels from η) would strengthen claims.

Suggestions for improvements:
- Replicate results on other architectures (with ablations) on a small-MLP transformer and a ~100–300M LM with modern BPE or unigram LM tokenizers;
- Provide chain-to-chain variance, R-hat-style diagnostics, and sensitivity to SGLD hyperparameters; test subsampling stability of η-space geometry.

### Questions
1. What exactly does susceptibility represent causally? Is χ interpretable as a local sensitivity to intervention on a head, or more akin to covariance with a loss gradient?

2. Why is the “embryology” metaphor meaningful beyond aesthetics? What else does it add in this context that we would not be able to infer otherwise?

3. Do we expect a similar 2D structure if we used t-SNE, Isomap, or diffusion maps? Is there theoretical reason (e.g., low intrinsic dimensionality of susceptibility space) to expect such a compact geometry?

4. How stable are susceptibility estimates? Did you try random seeds, sampling noise, SGLD parameters (β, ε, γ)? Is there a confidence measure per χ that could be visualized (e.g., variance across samples)?

5. Could susceptibility vectors serve as features for causal discovery (e.g., learning directed edges between heads)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes an “embryological” lens on visualization of training small language models: compute a per‑token susceptibility vector (one coordinate per attention head) that measures how changing weights in a component covaries with the loss on a specific continuation, then apply dimension reduction techniques to many such vectors over training. The resulting 2D plots form a characteristic “rainbow serpent” whose long axis (PC1) reflects global expression vs. suppression across heads and whose short axis (PC2) thickens as an induction circuit emerges. The authors also report a previously unremarked structure, a “spacing fin”, consisting of spacing tokens, especially those preceded by long runs of spaces/newlines, which they hypothesize reflects counting of spacing tokens.

### Strengths
Overall, I believe the authors have studied an interesting question and created various intriguing visualizations that could be of interest to the community. The paper presentation was good, and the figures are very nicely presented. In particular, 

1. The "embryological" analogy, framing model training as a developmental process, is conceptually intuitive yet powerful. Applying UMAP to susceptibility vectors, rather than just model activations, is a novel approach. It provides a global visualization of how the entire set of model components (attention heads) collectively organizes to handle different token patterns.

2. The theory was validated (at least in small scales) nicely by the experiments showing the emergence of induction circuits, which were intensively studied in literature, and placing it into their own contexts, offering their interpretation of the underlying mechanisms of the model.

### Weaknesses
The major limitations are already straightforwardly discussed in the paper. Here I rephrase the two most important ones in my opinion:

1. The experiments are pretty severely limited by scale, as they were only visualized with a tiny 3M model with two layers of attention-only modules. It is a pretty significant leap from even tiny-scaled language models by today's standards. This is (in my opinion) the most significant weakness of the present paper, and it is not clear what the limitation is for the authors not to report more extensive experiments.

2. Some key claims (e.g. the spacing fin’s separation) depend critically on UMAP. The paper also notes PCA fails to reveal the fin in the first three PCs and suggests it lives in higher PCs. However, as the authors have pointed out, UMAP is a non-linear visualization technique that can distort global geometric structures. This raises the question of the robustness as well as reliability of their findings when migrating to different architectures, as well as making interpreting the results harder.

A minor point is that a lot of biology references are made, which may make it hard for someone without knowledge of biological sciences to capture the analogies. Despite the above, I still have questions for the paper (see below) and will consider raising the score if the authors provide satisfying responses to the weaknesses/questions.

### Questions
1.  Following up with the scalability concern, how much computational overhead is there to compute the necessary statistics for the proposed visualization? Is training a larger base model actually the biggest computational burden, or are the tools developed for visualization demanding nontrivial computational resources beyond training the model?

2. The authors discussed the experiment results, "likely influenced by the tokenizer, and different tokenization strategies could lead to different learned structures." Can you provide more empirical evidence or insights into why this might be the case, or better, provide experiments on how tokenization changes the results?

3. While the figures are interesting, I didn't find much discussion on the impact for practitioners who are training a model or understanding a trained model. What qualitative or quantitative features of the proposed framework (e.g., emergence of induction circuits) are expected to transfer when training a (possibly much different) model or inspecting a trained model?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper introduces an "embryological" approach for studying the development of structure during the training of a small language model. The authors use UMAP on per-token susceptibility vectors to visualize how structural organization emerges through training.

### Strengths
**Originality: poor** 
Application of UMAP to high-dimensional susceptibility vectors for visualization is a straightforward combination of existing methodologies rather than a substantial advance. The framing of “embryology” and “body plan” in neural networks reads more as metaphorical novelty than genuine technical originality.

**Quality: poor**
The overall quality of the work is limited by a lack of rigorous experimental validation and an overreliance on qualitative visualization. Key claims about the “universality” and interpretability of observed structures are not robustly substantiated, and methodological choices (model size, tokenizer, architecture) are unjustified and insufficiently explored. Findings appear sensitive to experimental setup and dimensionality reduction parameters. 

**Clarity: good**
The paper is generally clear and well-organized, with visualizations that are appealing and easy to follow. Explanations of the experimental setup and the visualization process are accessible, and the writing is coherent.

**Significance: poor** 
The significance is limited, as the claims do not meaningfully advance our theoretical or practical understanding of neural network internals. The scientific insights derived from the visualizations are superficial. The potential for broader impact is unclear given the narrow experimental scope.

### Weaknesses
**Unsound/superficial application of UMAP.** 
- The paper relies heavily on UMAP for what amount to no more than qualitative visualizations. The limitations of UMAP for interpretability are identified in the appendix, and others are well-documented in various literatures. As a consequence, "anatomical" claims lack quantitative rigor to support the "serpent" as a stable, robust feature rather than a visualization artifact. The authors acknowledge that the geometry of the "serpent" should be "nterpreted cautiously." Given that the paper's claims rely almost exclusively on this geometry, all results should be interpreted *at least* as cautiously. 
- The paper's claim that UMAP "faithfully represented" aspects of the underlying high-dimensional distribution is not justified. There is little to no effort to quantify the claims of reliability or interpretability of the UMAP projections. 
- The authors remark that UMAP parameters were varied and patterns that did not persist were dismissed. However, there is no quantitative stability result to support that the observed structures are not induced by specific hyper-parameter choices. 

**Experimental limitations and poor generalization claims.** 
- Critical aspects like the "spacing fin" are demonstrated to be contingent on the tokenizer and possibly the dataset (?). But the paper does not explore how changing tokenization, data distribution, or model size / complexity alter these findings. Again, the paper's claims to have identified a persistent structure but makes no effort to rigorously test or quantify this persistence across variations. Claims to "universality" of the body plan are called into question. 
- Findings are only supported by UMAP *visualizations* and not quantitative measurements. 
- The "spacing fin" is presented as a newly discovered structure, but its mechanistic significance is left at best conjectural.

### Questions
- Quantitative support for persistence or robustness of the "body plan" and "spacing fin"? 
- Any sort of rigorous quantitative evidence that the observed structure are not artifacts of UMAP? 
- Given the dependence on tokenizer, do similar features emerge with alternative tokenization schemes and different data distributions?

### Soundness
2

### Presentation
3

### Contribution
1
