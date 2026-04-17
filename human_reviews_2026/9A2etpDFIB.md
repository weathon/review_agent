# Towards Understanding the Nature of Attention with Low-Rank Sparse Decomposition

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
We propose Low-Rank Sparse Attention (Lorsa), a sparse replacement model
of Transformer attention layers to disentangle original Multi Head Self Attention (MHSA) into individually comprehensible components.
Lorsa is designed to address the challenge of \textit{attention superposition} to understand attention-mediated interaction between
features in different token positions. Lorsa helps find cleaner and finer-grained versions of previously discovered MHSA behaviors like induction
heads, successor heads, attention sink, and a comprehensive family of arithmetic-specific Lorsa heads. Interestingly, we identify a novel
head type called \emph{subtoken induction heads} that function at character level rather than token level.
Automated interpretability analysis indicates
that Lorsa achieves parity with SAE in interpretability while Lorsa exhibits superior circuit discovery properties.
We also conduct extensive experiments on architectural design ablation, correlation to original MHSA heads and error analysis.
Our early attempt to fully sparsify a toy Transformer succeeds to reveal clean global circuits. Eventually, we hope Lorsa would help
us greatly understand attention computation and enable full sparsification of model computation along with its MLP counterparts.
Lorsa is open-sourced at https://anonymous.4open.science/r/Lorsa-5686/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper studies attention superposition by introducing **Low-Rank Sparse Attention (Lorsa)**—a per-layer module trained to **reconstruct** an MHSA layer’s output as a top-$K$ sum of many rank-1 OV “attention units,” with shared QK parameters to control cost. Training minimizes an MSE objective against the layer’s MHSA output (Algorithm 1), yielding a sparse, overcomplete basis over the attention-output space that can be analyzed with standard interpretability tooling. The authors apply Lorsa to Pythia-160M and Llama-3.1-8B, report fidelity–sparsity scaling and per-layer FVU, and present qualitative/automated evidence that Lorsa rediscovers known head families (induction, successor, sinks) and surfaces new phenomena (subtoken-induction and arithmetic-specific heads), while broadly tracking SAE trends in reconstruction and interpretability metrics. Lorsa is compatible with RoPE and GQA through straightforward adaptations.

### Strengths
First, the methodological design is original yet well-justified: the rank-1 OV constraint sharpens semantics; shared-QK binding preserves QK dimensionality where needed and avoids the degradation observed when QK is reduced below the MHSA head dimension. The top-$K$ sparsity yields monosemantic components that are easy to inspect. The paper provides clear algorithms and ablations supporting these choices.

Second, the presentation empirical analysis is substantive and yields new insights. Lorsa tracks SAE scaling and per-layer patterns, recovers canonical heads, and identifies novel mechanisms (e.g., arithmetic-specific and subtoken-induction heads) in Llama-3.1-8B, with preliminary causal probes that connect heads to behavior.

### Weaknesses
- My main concern is with the paper’s positioning: the manuscript repeatedly states that “Lorsa serves as a replacement model for Transformer attention”. However, the scope of evaluation is strictly __per-layer reconstruction__ and interpretability; there is no end-to-end evaluation or deployment-oriented analysis. As written, the “replacement” phrasing risks misleading readers about the paper’s primary contribution, which is pattern understanding of MHSA via decomposition.
    - A related, practical limitation is the scale of the Llama-3.1-8B setting: ~32K Lorsa heads per layer with (K=128) and ~512M parameters per layer (vs. 64M for MHSA) make the module expensive, and training (more like reconstructing for approximation) each Llama layer’s Lorsa takes ~24 A100-hours under the stated batch size. This underscores that the artifact is currently best viewed as an interpretability tool rather than a deployable substitute .
- A second concern is _behavioral equivalence vs. output approximation_. I am not totally convinced that closely approximating MHSA outputs implies similar underlying behavior. For instance, two architectures (A) and (B) can be trained to produce near-identical outputs while differing substantially in inductive biases and internal mechanisms. Although Algorithm 1 shows Lorsa as a simplified Top-K variant of MHSA, architectural changes e.g. rank-1 OV, shared-QK, and Top-(K) gating may alter how features are represented and composed. Thus, even if $||\mathrm{Lorsa}(x)-\mathrm{MHSA}(x)||$ is small, approximation alone does not justify studying them “as the same”; a theoretical link (e.g., bounds ensuring stability of feature-level attributions or circuit-level predicates) is needed, particularly given superposition and shared-QK.

### Questions
- On theoretical grounding, can you provide a result formalizing when Lorsa is a faithful surrogate for studying MHSA? For instance, assume a layerwise error bound $||\text{Lorsa}(x)-\text{MHSA}(x)||\le \varepsilon$ and conditions on QK similarity; then prove stability of **linear readouts** and **attribution functionals** (e.g., path-patching scores, projection-based attributions) to errors of size $\varepsilon$. A theorem of the form
   $$
   \mathbb{E}_{x\sim \mathcal{D}}|| \mathcal{A}(\text{Lorsa};x) - \mathcal{A}(\text{MHSA};x) ||_p \le C\cdot \varepsilon,
   $$for a relevant class $\mathcal{A}$ would directly justify that we can study MHSA via Lorsa. Clarifying how shared-QK and Top-K selection affect such stability (e.g., via Lipschitz or margin assumptions on head activations $z$) would be especially valuable.
- I am not an expert in this field, so Id like to refer to other reviewers for the final recommendation.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Low-Rank Sparse Attention (Lorsa), a sparse replacement model for MHSA designed to tackle the problem of attention superposition. To do that, the authors decompose the OV circuit into a large overcomplete set of sparse rank-1 heads. The work demonstrates that this method successfully rediscovers known mechanisms (like induction heads and successor heads) and uncovers novel, finer-grained units (such as subtoken induction heads and specific arithmetic heads). The paper provides quantitative analysis showing that the interpretability of Lorsa heads is comparable to that of Sparse Autoencoder (SAE) features, positioning Lorsa as a promising new tool for the mechanistic interpretability of attention layers.

### Strengths
- The paper addresses attention superposition, a critical but less-studied counterpart to feature superposition in MLPs. The application of sparse dictionary learning to decompose attention head outputs (specifically via rank-1 OV circuits) is a novel and significant methodological contribution, providing a much-needed alternative to SAEs for interpreting attention.
- The method is well-validated. The authors successfully recover a wide range of previously identified attention mechanisms, which serve as a strong sanity check. More importantly, the discovery of novel, fine-grained units like "subtoken induction heads" and families of "arithmetic-specific Lorsa heads" demonstrates the utility and resolving power of the method.
- The authors provide a strong quantitative benchmark by comparing Lorsa's interpretability directly against SAEs using automated interpretability methods. The paper also includes valuable ablation studies (in the appendix) that explore the architectural design choices, such as the impact of QK dimension and initialization.

### Weaknesses
- The notation is confusing:
    - The total number of Lorsa heads is denoted as $H$ in Algorithm 1 (line 140) but as $N_{Lorsa}$ in the main text (e.g., line 185). This is compounded in Section 4.1, where $N$ is used for the number of parameters.
    - The term $D_{QK}^{Lorsa}$ is used in Section 3.1 (line 169) without any prior definition. Its usage is also conflated: it appears to mean a dimension in line 169 ("...as $D_{QK}^{Lorsa}$ decreases") but a count of heads in line 171 and the Table 1 caption ("...across every $D_{QK}^{Lorsa}$ heads"). This makes the QK sharing mechanism difficult to parse.
- There is a direct contradiction between the pseudocode and the main text describing the model's architecture:
    - Algorithm 1 explicitly shows per-head query and key weights ($W_q^h, W_k^h$), implying no parameter sharing.
    - Section 3.1 (lines 168-183) explicitly describes the opposite: a parameter-sharing mechanism where groups of Lorsa heads share a single QK circuit. This made the paper unclear while reading, as the QK sharing mechanism is a core part of the Lorsa design.
- Some important limitations and conceptual discussions are in the Appendix. I understand that this could be because of spacing issues, so I would definitely recommend having these discussions in the camera-ready version of the paper.

### Questions
**Questions:**

1. Could the authors please clarify the difference between Algorithm 1 and the text description? 
2. Could you please clarify the notation issues mentioned in the Weakness section?
3. The FVU values, while comparable to SAEs, are non-trivial. Appendix H introduces the Lorsa Dark Matter hypothesis, suggesting this error is structured and shared with SAEs. This is a critical point for the entire replacement model paradigm. Could the authors elaborate on the implications of this? If 10-30% of the variance is unexplained and potentially uninterpretable by replacement models based methods, how does this limit the functional completeness of the discovered units and the conclusions we can draw from them?


**Minor comments:**  
There is a typo in line 126: “privleged”

### Soundness
4

### Presentation
2

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
This paper introduces Low-Rank Sparse Attention (Lorsa), a method to decompose transformer attention into sparser, more interpretable components. It effectively addresses "attention superposition," rediscovering known mechanisms like induction heads and identifying novel ones (e.g., arithmetic heads). The design is scalable and evaluation is thorough.

### Strengths
1. The paper clearly articulates the problem of "attention superposition" as a key obstacle to interpretability, drawing a compelling analogy to superposition in MLP layers. This provides a strong theoretical motivation for the work.

2. The design (rank-1 OV circuits, top-K sparsity, QK weight sharing) is effective and scalable.

3. Comprehensive experiments, including scaling laws, automated interpretability, and compelling case studies on models up to Llama-8B.

### Weaknesses
1. The primary limitation is that QK circuits are shared across heads, potentially hindering full disentanglement and causal attribution.

2. Evidence for novel head functionalities is mostly correlational; more causal intervention experiments would strengthen the claims.

### Questions
What might the unreconstructible "dark matter" in the attention output represent?

### Soundness
3

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
2

### Summary
The paper proposes a sparse replacement model of the attention layer,  Lorsa, to understand the attention   mechanisms. The authors demonstrate that Lorsa can rediscover previously reported mechanisms such as induction heads and attention sinks, and even identify new patterns such as subtoken induction heads.

### Strengths
- The proposed Lorsa method successfully rediscovers known attention mechanisms such as induction heads and attention sinks, and also discovers new interpretable phenomena (e.g. subtoken heads and arithmetic Lorsa-heads. I'm not very familiar with the literature, but the proposed approach appears to be novel. 
- The paper is well organized and is easy to follow.

### Weaknesses
- The work may be better if including motivation and ablation study of several of the design choices, especially the difference with SAE, e.g. predicting the downstream activations, the top-K operation.

### Questions
- Is it possible to use existing methods (e.g. SAE) to identify the same newly discovered attention mechanisms reported in this paper?
- Could the authors provide more details on how well autointerp performs, and clarify whether it can be trusted as a quantitative metric for comparing interpretability across different methods in Section 5.3?
- How are gradients propagated through the Top-K operation during training?

### Soundness
3

### Presentation
3

### Contribution
3
