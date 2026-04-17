# Cross-Modal Redundancy and the Geometry of Vision–Language Embeddings

- Decision: Accept (Poster)
- Scores: 2, 6, 4, 8

## Abstract
Vision–language models (VLMs) align images and text with remarkable success, yet the geometry of their shared embedding space remains poorly understood. 
To probe this geometry, we begin from the Iso-Energy Assumption, which exploits cross-modal redundancy: a concept that is truly shared should exhibit the same average energy across modalities.
We operationalize this assumption with an Aligned Sparse Autoencoder (SAE) that encourages energy consistency during training while preserving reconstruction.
We find that this inductive bias changes the SAE solution without harming reconstruction, giving us a representation that serves as a tool for geometric analysis.
Sanity checks on controlled data with known ground truth confirm that alignment improves when Iso-Energy holds and remains neutral when it does not.
Applied to foundational VLMs, our framework reveals a clear structure with practical consequences: 
**(*i*)** sparse *bimodal* atoms carry the entire *cross-modal* alignment signal; 
**(*ii*)** *unimodal* atoms act as *modality-specific* biases and fully explain the modality gap; 
**(*iii*)** removing unimodal atoms collapses the gap without harming performance; 
**(*iv*)** restricting vector arithmetic to the bimodal subspace yields in-distribution edits and improved retrieval. 
These findings suggest that the right inductive bias can both preserve model fidelity and render the latent geometry interpretable and actionable.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper investigates the geometry of the embedding space of CLIP-like models using sparse autoencoders. The authors augment SAE training by an iso-energy regularization term that encourages SAE latents to have similar spreads (i.e., second moments) for both modalities (Def. 2). They show that only a small set of features explain the modality gap and the remaining features are sufficient for cross-modal alignment. Removing former features reduces modality gap while retaining performance and the latter allows for vector arithmetic.

### Strengths
* S1: The energy penalty (Def. 2/ Eq. 1) is interesting and a simple addition to the SAE loss.

* S2: Synthetic & real experiments confirm that the proposed aligned SAE better matches the geometry of CLIP-like models. Particularly, if the SAE features can distinguish between shared or modality-gap-specific.

* S3: The paper introduced four metrics to evaluate whether the SAE variants capture the geometrical or functional properties of the VLMs.

* S4: The proposed SAE allows semantic vector arithmetic.

### Weaknesses
* W1: 3 out of the 4 key findings have been reported in previous work (see bullet points below). While the findings are reached using a different, more complex approach, the current paper seems to re-report these findings.
	* Few (unimodal) features fully explain the modality gap (Fig. 2 left, 3) ~> see Fig. 4 in [3] or Fig. 3 in [4]
	* Bimodal features carry the entire cross-modal alignment signal (Fig. 2 right, Fig. 3) ~> cross-modality transferability experiments in [2], e.g., Tab. 2.
	* Removing those modality-gap features reduces the modality gap without loss of performance (Fig. 4) ~> again, see cross-modality experiments in [2], e.g., Tab. 2.

* W2: The paper provides little to no experimental details in the main text, making it hard to understand the results without searching the supplemental.

* W3: It is assumed that bimodal atoms are semantically aligned across modalities (“bimodal atoms encode the shared conceptual backbone” l. 345) and few qualitative examples are provided in Appendix G. However, there is no quantitative evaluation for this claim.

* W4: Only contrastive models are evaluated. For example, the modality gap has been also observed in multimodal LLMs. It’d be important to include such results.

## Comment

* C1: I’d encourage the authors to include discussions on missing relevant literature [1-4].

* C2: This work’s proposition 1 seems closely related to [2]’s proposition A.1. The only difference seems to be that the modality information can be adaptive here.

* C3: The caption of Fig. 4 is partially occluded from Fig. 5.

---

[1] https://www.mlmi.eng.cam.ac.uk/files/2021-2022_dissertations/understanding_and_fixing_the_modality_gap_in_vision-language_models_reduced.pdf 

[2] https://openreview.net/forum?id=D-zfUK7BR6c 

[3] https://openreview.net/forum?id=uAFHCZRmXk

[4] https://openreview.net/forum?id=QGUju9B68Z

### Questions
* Q1: Is the standard SAE (l. 176/177, 185) the MP-SAE or is it truly standard SAE?

* Q2: How are unimodal or bimodal features separated?

* Q3: Do the unimodal features approximate the modality gap vector? Related to that, does it explain why they all have such high cosine similarities (Fig. 16b)?

* Q4: What is $\mu$ in Fig. 2 left?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the geometry of VLM embedding spaces via the Iso‑Energy Assumption—shared concepts should have domain‑invariant average squared activation. The authors train an Aligned Matching‑Pursuit SAE with a small cross‑modal alignment regularizer, yielding a dictionary that separates bimodal atoms (which carry all cross‑modal alignment) from unimodal atoms (a few high‑energy modality‑specific biases explaining the modality gap). On synthetic data and CLIP/OpenCLIP/SigLIP variants, this preserves reconstruction while markedly improving multimodality metrics, and enables interventions such as removing unimodal atoms to close the modality gap without hurting retrieval and performing in‑distribution semantic arithmetic restricted to the bimodal subspace.

### Strengths
1.  **Clear Problem Formulation and Strong Motivation:** The paper articulates a pertinent and significant problem in VLM interpretability and manipulability. By focusing on the geometric underpinnings of cross-modal alignment and the "modality gap," the work addresses a critical area for improving VLM transparency and control. 
2.  **Novel and Intuitive Hypothesis:** The "Iso-Energy Hypothesis" offers an elegant and interpretable statistical prior for identifying shared concepts within a sparse dictionary. This hypothesis provides a concrete, measurable criterion that transforms the abstract notion of "cross-modal redundancy" into an actionable constraint for dictionary learning.
3.  **Demonstrated Practical Interventions:** The ability to close the modality gap by masking uni-modal atoms and to perform "in-distribution" semantic arithmetic within the bi-modal subspace represents a significant practical contribution. These interventions offer concrete pathways for improving the robustness and interpretability of VLM applications.

### Weaknesses
1.  **Reliance on Paired Data for Alignment Regularization:** Although lines 158-160 allude to the potential of leveraging "cross-modal redundancy alone," the current formulation of the alignment regularizer explicitly requires instance-level image-text pairs. The robustness of the method to noisy or imperfect pairings, or its applicability in settings with weak or no explicit pairings (e.g., using only domain labels), remains unexplored. This dependency may limit its generality and practical scope.
2.  **Limited Assessment of Dictionary Stability and Generalizability:** While the paper aims to enhance SAE dictionary stability via the Iso-Energy Assumption and demonstrates improved recovery on synthetic data during "Sanity check", it lacks a systematic and multi-faceted analysis of this robustness on large-scale real-world VLM datasets. The reproducibility of the learned dictionary under varying conditions, such as different expansion ratios, sparsity targets, or subsets of training data, remains unexplored. Thus, the evaluation of this crucial aspect in practical scenarios is not yet comprehensive.
3.  **Scope of Evaluation and Downstream Task Relevance:** While the paper demonstrates strong results on retrieval-oriented metrics and interventions, the generalizability to other VLM tasks (e.g., visual question answering, image generation, localization, counting, spatial reasoning) is not fully explored. The claim that "masking uni-modal atoms does not hurt performance" might hold for certain tasks, but could be detrimental for tasks that rely on more modality-specific information.

### Questions
**External Validation of Atomic Concepts:** While visualizations are provided, the "semantic stability" of the atoms is largely qualitative. Is it possible to introduce quantitative measures for concept purity, namability, or alignment with human annotations to further validate the interpretability and meaningfulness of the identified bi-modal and uni-modal atoms? This would provide stronger evidence that the method is indeed recovering genuine, human-understandable concepts.

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
The paper studies the geometry of vision–language embeddings through a proposed Iso-Energy assumption, which states that shared cross-modal concepts should have equal activation energy across modalities. To explore this, the authors introduce an aligned sparse autoencoder (SAE-A) that adds a cosine-similarity–based alignment loss to a standard sparse autoencoder. The numerical experiments on CLIP, OpenCLIP, and SigLIP embeddings show that the aligned SAE could improve cross-modal alignment metrics while maintaining reconstruction quality.

### Strengths
1. The paper provides an interesting perspective on the geometry of vision–language embeddings by introducing the Iso-Energy assumption.

2. The numerical results are consistent, showing that the aligned SAE can improve cross-modal alignment metrics without damaging reconstruction quality.

### Weaknesses
1. The connection between the Iso-Energy Assumption in Definition 2 and the implemented loss in Equation (1) is not that clear. Definition 2 describes a population-level equality of per-coordinate activation energies across modalities, whereas the alignment loss in (1) simply quantifies the batch-level sum of cosine similarity between sample codes. The paper does not provide a derivation or justification showing that this cosine similarity sum term directly enforces or meaningfully approximates the Iso-Energy property.

2. The alignment loss in Equation (1) effectively reduces to a vanilla sum of cosine similarities between the latent codes from two modalities. This formulation looks too simple and somewhat ad hoc, lacking a clear connection to encourage equalized energy statistics as defined by the Iso-Energy assumption.

3. The paper introduces the aligned sparse autoencoder without providing sufficient background on the baseline SAE formulation, its reconstruction, and sparsity terms. This makes the method less self-contained and more difficult for readers less familiar with the SAE framework to follow.

4. Some of the mathematical definitions, particularly in Definition 2, are not presented rigorously. The conditional expectation is written as if conditioned on the specific sample $X$, which collapses the expectation to the outcome for that given value of $X$ in the conditional expectation of (1).

### Questions
1. Can the authors clarify the precise theoretical link between the Iso-Energy Assumption in Definition 2 and the cosine-similarity–based alignment loss in Equation (1)? 

2. As the alignment loss in (1) reduces to a simple sum of cosine similarities, did the authors experiment with other similar regularizers (e.g., the sum of the squared or absolute value of the inner products in (1)) or other regularizers that can more directly enforce the Iso-Energy property in Definition 2?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes an Iso-Energy prior for learning aligned sparse concept dictionaries on top of VLM embeddings. By mildly enforcing equal second-moment (“energy”) of a concept across image/text domains, the aligned SAE separates bimodal atoms (semantic carriers) from unimodal atoms (modality-specific bias). This yields two actionable interventions: (i) closing the modality gap by masking unimodal atoms without hurting retrieval, and (ii) performing robust semantic vector arithmetic within the bimodal subspace, reducing OOD drift.

### Strengths
1. Clear and effective framing of a testable modeling intuition.
The paper presents a well-motivated and conceptually coherent formulation. It articulates a precise inductive bias: that shared cross-modal concepts should exhibit similar activation statistics across modalities. This idea is not only intuitively appealing but also operationalized in a mathematically minimal way through second-moment constraints. The writing and structural clarity further reinforce this framing, making the contribution accessible and theoretically grounded.

2. Methodologically grounded execution with dual functionality.
The proposed method delivers more than conceptual framing. It constructs a sparse, interpretable bimodal subspace that supports both analysis and intervention. The same subspace allows for attribution-style interpretation as well as semantically coherent editing, demonstrating that the learned structure is not only intelligible but also functionally controllable. This dual capacity is rarely achieved in the interpretability literature and gives the method both analytical and practical value.

### Weaknesses
1. Sufficiency versus necessity of the Iso-Energy criterion.
Equal second moments across modalities can indicate shared concepts, but they are not required. Without invariance to modality-specific anisotropy or rescaling, genuinely shared factors may be labeled unimodal. It would be better to add invariance controls such as per-modality whitening or variance normalization, and to compare with covariance-aware baselines such as CCA or CORAL to verify that the findings are not driven by marginal variance.

2. Sensitivity to pairing noise and frequency imbalance.
The alignment term relies on paired image and text data, where long-tail frequencies and noisy matches are common. Energy equality can be confounded by corpus artifacts rather than semantics. It would be better to add two controls: a frequency-matched subsample that balances concept prevalence across modalities, and a shuffled-pairs stress test to quantify robustness to misalignment noise.

### Questions
1. To what extent do the conclusions generalize to more complex tasks and architectures, such as VQA on LLaVA-series models?

### Soundness
3

### Presentation
4

### Contribution
3
