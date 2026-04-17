# Warped Latent Spaces and Traversal for Chemical Deep Generative Models

- Decision: Reject
- Scores: 2, 6, 2, 0

## Abstract
We propose a generative framework for interpretable and property-aware molecular design by learning warped subspaces within the latent space of a chemical variational autoencoder (VAE) trained on a sequential representation of small molecules. Instead of directly regularizing latent coordinates, our approach works by creating low dimensional subspaces that are smoothly warped to align with molecular property variation using a novel alignment loss. This warping provides a flexible mechanism to capture nonlinear structure in property–latent relationships while retaining interpretability. This framework enables property optimisation and traversal within a low-dimensional subspace, where directions correspond to meaningful variations in molecular properties and decode back into valid molecules in the original space. We evaluate the method on various tasks related to conditional molecular generation on standard benchmarks used in literature like QM9, ZINC250K and the Pubchem drug datasets demonstrating strong generative quality, validity, uniqueness and novelty alongside a more controllable approach molecular generation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
`method`
1 a transformer VAE.
2 non-linear warping function, achieved with an alignment loss over all pairs of molecules as eq 6 (L align).


`results`
Method evaluated on ZINC250k, on validity, novelty.
Also evaluated on property prediction.

### Strengths
the task of generative models for molecule is an important issue in ML

### Weaknesses
lack of novelty in the method.

Lack of suitable evaluation, all baselines are from before 2019 .

The results show no or minor improvements.

### Questions
not sure why we need pairs? in eq 6? is this motivated? could this be clarified?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a way to make a pretrained molecular VAE’s latent space easier to control for specific chemical properties (like QED, logP, SAS). After training a Transformer-VAE on SELFIES strings, the authors learn a tiny “warping” network for each property that maps the original latent codes into a small, property-aware subspace. In that warped space, the property varies mostly along a single direction, so you can move the latent point in a simple, controlled way to raise or lower the property. A trust-region step keeps moves near the data manifold, and an “inverse lifting” step brings the edited point back to the original latent space to decode valid molecules.

### Strengths
1. Framing property control as a lightweight warping learned on top of a pretrained VAE is a novel way. It avoids re-training the generator or baking properties into the decoder, which many prior methods require.
2. Using pairwise distance alignment and covariance whitening to keep the warped subspace well-conditioned is a sensible way to prevent degenerate mappings and encourages smooth traversals.
3. Single-direction “dials” per property are easy to reason about and integrate into interactive tools or multi-objective workflows.

### Weaknesses
1. A single linear direction per property may work only locally; globally the property landscape can be multi-modal or curved.
2. Many original latents can map to the same warped point; the inverse step may fail or land off-manifold, leading to invalid or degenerate molecules.

### Questions
1. Do you detect and swap direction when monotonicity breaks?
2. How large is the average drift in non-optimized properties after lifting and decoding?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a method for molecular property optimization, which operates within the latent space of a pre-trained Variational AutoEncoder (VAE). By employing the SELFIES representation, the VAE's latent space is reportedly free from the "dead zones" that typically plague models trained on traditional SMILES representations. This property is highly advantageous for optimization tasks. To render navigation of this high-dimensional latent space tractable, the authors propose learning property-specific transformations that project the latent vectors into a much lower-dimensional space. A covariance whitening regularizer is introduced to simplify the downstream optimization. Although the resulting optimization problem is non-convex, the authors employ a multi-restart strategy to mitigate the impact of local minima.

### Strengths
The central idea of performing property optimization within a learned, low-dimensional subspace—rather than in the VAE's full native latent space—is interesting. This approach to dimensionality reduction for targeted optimization represents the key contribution of the paper.

### Weaknesses
While the proposed low-dimensional projection is intriguing, the paper seems to concede (e.g., in the "Decoding Protocol" section) that the fundamental challenges of non-convexity and high-dimensionality persist.

A critical omission is a comparison against a baseline that predicts properties directly from the full VAE latent space (e.g., a simple regression model). Such a baseline would not only contextualize the benefit of the proposed projection but also highlight a significant weakness of the current method: its apparent inability to handle multi-property optimization, a capability that a direct-prediction model would inherently possess. The lack of multi-property optimization support is a major limitation.

Furthermore, a significant weakness of the paper is its lack of thorough contextualization. The "Related Work" section remains at a high level of abstraction, presenting lists of citations without specific discussion of how those works relate to the paper's novel contributions or limitations. Consequently, the paper fails to compare against any relevant, modern baselines. Even the VAE component's evaluation relies on a baseline from 2019, which is outdated given the rapid progress in the field.

To address this, the authors should position their work more clearly relative to:
- General structured prediction (have a look at papers like [1,2]).
- Modern representation learning for molecules, especially SELFIES-based models (e.g., [3] offers an encoder-decoder, [4] an encoder-only).
- Other contemporary methods for molecular property optimization.

[1] Amos, Brandon, Lei Xu, and J. Zico Kolter. "Input convex neural networks." International conference on machine learning. PMLR, 2017.
[2] LeCun, Yann, Chopra, Sumit, Hadsell, Raia, Ranzato, M,
and Huang, F. A tutorial on energy-based learning. Pre-
dicting structured data, 1:0, 2006.
[3] Priyadarsini, Indra, et al. "Self-bart: A transformer-based molecular representation model using selfies." arXiv preprint arXiv:2410.12348 (2024).
[4] Yüksel, Atakan, et al. "SELFormer: molecular representation learning via SELFIES language models." Machine Learning: Science and Technology 4.2 (2023): 025035.

# Typos:
1. L260: “itt” -> “it”

### Questions
1. Causal Decoder: The rationale for using a causal decoder is unclear. This architecture is typically employed for auto-regressive next-token prediction, which does not appear to be required by your non-autoregressive setup. How does the performance of this causal decoder compare to a standard (non-causal) self-attention mechanism?
2. Decoder Architecture: Could the authors please provide a diagram or pseudo-code for the decoder architecture? The description of how the property vector $m$ is repeated $T$ times and utilized within the model is currently difficult to follow.
3. Decoding Protocol Novelty: Is the "Decoding Protocol" a novel contribution of this work or standard practice? If it is novel, the justification for presenting it solely in the narrow context of molecular optimization, rather than as a general method for structured prediction, is missing. If it is standard practice, please provide relevant citations.
4. Tokenizer: What specific tokenizer is used for the SELFIES strings? Is it character-level, or does it utilize a specific SELFIES vocabulary?

### Soundness
2

### Presentation
1

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
This paper presents a VAE framework to learn property-aligned subspace via an auxiliary loss. Results show that the proposed method can learn a subspace where molecules with similar property values have similar latent representations.

### Strengths
The presentation is generally clear.

### Weaknesses
- Generally, this paper introduces nothing new to the community. The overall problem, using VAE latent space traversal for property optimization, is something that this community was exploring four or five years ago. Many more effective optimization methods have been proposed over the past five years; unfortunately, none of them are compared with or even mentioned in this paper.
- Even for vanilla molecule generation without property optimization, all the comparison methods in Table 1 are from 2020 or earlier, despite the rapid progress in this field over recent years. The lack of up-to-date baselines makes the evaluation unconvincing.
- The idea of using a Transformer-based VAE is also not new. Simply replacing SMILES with SELFIES does not contribute meaningful novelty either. The only new component, the latent space contrastive loss (Eq. 6), still does not address a valid problem. In the introduction, the authors criticize latent space disentanglement methods for suffering from property entanglement, but this issue has already been studied and mitigated by a series of works addressing latent factor correlation in VAEs.
- The properties used for evaluation, including QED and LogP, are considered toy properties with limited connection to real-world molecular design tasks, which further weakens the practical significance of the reported results.

### Questions
I don't have more questions.

### Soundness
2

### Presentation
2

### Contribution
1
