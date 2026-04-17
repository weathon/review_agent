# Adaptive Data-Knowledge Alignment in Genetic Perturbation Prediction

- Decision: Accept (Poster)
- Scores: 4, 2, 6, 6

## Abstract
The transcriptional response to genetic perturbation reveals fundamental insights into complex cellular systems. While current approaches have made progress in predicting genetic perturbation responses, they provide limited biological understanding and cannot systematically refine existing knowledge. Overcoming these limitations requires an end-to-end integration of data-driven learning and existing knowledge. However, this integration is challenging due to inconsistencies between data and knowledge bases, such as noise, misannotation, and incompleteness. To address this challenge, we propose ALIGNED (Adaptive aLignment for Inconsistent Genetic kNowledgE and Data), a neuro-symbolic framework based on the Abductive Learning (ABL) paradigm. This end-to-end framework aligns neural and symbolic components and performs systematic knowledge refinement. We introduce a balanced consistency metric to evaluate the predictions' consistency against both data and knowledge. Our results show that ALIGNED outperforms state-of-the-art methods by achieving the highest balanced consistency, while also re-discovering biologically meaningful knowledge. Our work advances beyond existing methods to enable both the transparency and the evolution of mechanistic biological understanding.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents ALIGNED, a method for predicting the effects of gene perturbations in single-cell transcriptomic data using prior gene regulatory networks (GRNs). Unlike earlier models that also leverage GRNs—such as GEARS—the proposed approach does not treat the regulatory information as static but instead refines it dynamically throughout training. The training process itself is complex, involving a multistage optimization scheme that combines gradient descent with a Monte Carlo, ultimately aiming to align predictions based on data with the ones based on GRN.

### Strengths
The authors employ a genuinely new strategy that leverages GRN for perturbation prediction that does not treat the GRN as static during training. They also devise a complex strategy of optimizing the neuro-symbolic alignment.

### Weaknesses
In terms of the accuracy of perturbation respnse predictions, the algorithm provides only marginal gains, while being notably resource-intensive—requiring up to 12 hours of training on an 80 GB GPU. The authors, nevertheless, argue that the true advantage of their method lies not in predictive accuracy but in its improved alignment with the prior GRN and in the refinement of the GRN during optimization. However, the first claim is problematic: the reported GRN-alignment metric seems circular, as the GRN serves both as an input to the model and as a reference for evaluation. The second claim—that the optimization refines the GRN—is also inconclusive, since the authors do not quantify the number or directionality of corrected regulatory links but instead report pathway enrichment scores, which appear only tangentially related.

### Questions
**I am willing to improve the score of the paper if:**

1. The authors explain why the knowledge consistency metric is important and prove that their way of computing it is not circular.  
2. The authors conclusively show that their approach provides substantial benefits in terms of perturbation prediction as compared to other methods in the field, including the current state-of-the-art STATE.  
3. The authors describe the optimization procedure from a practical point of view, explaining how different parameters at each stage influence the accuracy of predictions.  
4. *(Minor)* The authors mention that two of the datasets were split randomly — please clarify what is meant by that.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes to tackle a ternarized version of gene perturbation prediction through incorporating prior domain knowledge given by a knowledgebase in the learning process. They suggest to alternatingly refine the predicted outcomes and the knowledge base during the learning process and propose a gradient-free algorithm for this.

### Strengths
- The presented paper is original in that it presents a novel way of introducing prior knowledge into gene perturbation prediction models.

### Weaknesses
-	This paper considers a simplified ternarized version of the underlying problem of gene perturbation prediction, **which is not useful in practice**. The extent of change introduced by a prediction is essential to know. The de facto gold-standard challenge in the field right now is the virtual cell challenge held by the Arc institute (https://virtualcellchallenge.org/, [1]), for which the evaluation metrics already disclose that continuous predictions are necessary. I have seen academic works using this binary or ternary prediction framework to be able to use language models before, but that does not make the task in anyway useful.
-	There are **strong assumptions on how inhibitory and activating effects are interacting** that are not discussed in the main paper. I.e., to perform a query on the KB (the $\delta_{KB}$), it is assumed that activating and inhibitory effects are additive, which needs clear justification.
-	It is **unclear how scalable this approach is** in practice, the adjacency matrices are #genes x # genes, which can go up to n>60k (all annotated genes) or n>24k (protein-coding genes), in both cases resulting in restrictively large matrices. It is unclear from the writing what subset of genes is used in the experiments, but it appears to be a small selection which needs justification.
-	The writing is incomplete and in its current form the paper **can not be reproduced**.
1. How is the threshold $\theta$ defined and picked in practice? Why?
2. How is weight vector $w$ constructed exactly? A vector that “contains the number of training data samples[…] and the number of annotations from gene Ontology[…]” leaves a lot of questions – these are two numbers (counts) which clearly can not make up $w$. Do you sum per gene? Then add up the two numbers (KB and GO)? What is the rationale?
3. What are the number of genes (and which ones) considered for the experiments?
4. **What are the model architectures you use exactly**?
5. **Do you evaluate on hold-out data and KBs unseen during training**? What would the performance then be, e.g. evaluating on a PPI-derived KB or other datasets on the same (or even different) cell line, which is fairly common in the literature?
-	What is the comparison to SOTA methods, including the currently not discussed Morph [2], and STATE [3] models? Fig. 4,5 and Table 1 simply **miss a comparison of the actual prediction to any SOTA**.

[1] Rohaani, Y H et al. * Virtual Cell Challenge: Toward a Turing test for the virtual cell.* Cell 188(13), pages 3370-3374, 2025.

[2] He, C et al. * MORPH Predicts the Single-Cell Outcome of Genetic Perturbations Across Conditions and Data Modalities.* bioRxiv, DOI: https://doi.org/10.1101/2025.06.27.661992, 2025.

[3] Adduri, A K et al. * Predicting cellular responses to perturbation across diverse contexts with State.* bioRxiv, DOI:https://doi.org/10.1101/2025.06.26.661135, 2025.

### Questions
-	What is the justification on the assumption made for how activating and inhibitory effects propagate and interact?
-	Given that the KB is given as binary matrices, why can’t you do gradient-based optimization on this? You can follow a similar scheme as in 3.4 as far as the explanations go – what is prohibiting this?
-	What is the (exact!) architecture you are using in the experiments?
-	How many and which genes do you consider in the experiments? What happens if you consider more/less?
-	What are the precision and recall, given that the KB is so sparse, which is hidden in the F1?
-	What is the performance on training data?
-	What is the performance on unseen (hold-out) data?
-	What is the performance of KB reconstruction on an unseen KB (e.g. PPI-derived)?
-	What is the comparison to SOTA methods using common benchmarking protocols, both regarding metrics as well as separate data (see Weaknesses)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces ALIGNED, a neuro-symbolic learning framework that integrates gene regulatory knowledge bases (KBs) with data-driven learning to improve consistency between biological prior knowledge and empirical data. The core contribution is a bi-directional alignment mechanism:
Alignment stage — adapts the model to existing biological knowledge.
Refinement stage — updates and corrects the knowledge base itself using learned patterns.
It defines a novel Balanced Consistency metric to jointly evaluate data-knowledge coherence, addressing a gap in current hybrid bioinformatics methods that often treat biological priors as static and non-adaptive. The work is motivated by the high inconsistency (14–71%) between curated knowledge bases (OmniPath, GO, EcoCyc) and observed perturbation data, and proposes a continuous refinement cycle to resolve such mismatches.

### Strengths
Timely problem framing: Tackles a critical limitation in hybrid biological modeling—static priors that cannot evolve alongside new data.
Methodological novelty: The dual-stage alignment/refinement loop and Balanced Consistency metric provide a principled way to unify knowledge and data learning.
Strong empirical evidence: Across datasets (Norman et al. 2019, Adamson et al. 2016, Precise1K, E. coli), ALIGNED consistently outperforms baselines like GEARS, scGPT, and scFoundation in both prediction accuracy and knowledge consistency metrics.
Biological interpretability: Unlike pure neural models, the system supports symbolic reasoning and interpretable GRN (Gene Regulatory Network) updates.

### Weaknesses
Limited ablation of symbolic components: While gradient-free and gradient-based updates are both introduced, their relative contributions are only superficially analyzed.
Dataset diversity: Most evaluations center on transcriptional perturbation data; extension to morphological or cross-modal biological data is not shown.
Theoretical grounding: The abductive reasoning formulation (based on ABL) could benefit from clearer formal definitions and proofs of convergence or stability.
Potential overfitting to KB biases: Since refinement still depends on noisy or incomplete KBs, it’s unclear how ALIGNED avoids reinforcing existing errors.

### Questions
1. The paper introduces “bi-directional alignment” between the neural model and symbolic knowledge base. However, the exact mathematical formalization of the alignment operator and refinement update is somewhat ambiguous.
How is the “knowledge correction” signal computed and propagated?
Does the refinement step have convergence guarantees, or can it drift over repeated iterations?

2. The model includes multiple interacting components: gradient-based learning, abductive reasoning (gradient-free correction), and symbolic consistency scoring.
What is the contribution of each?
Would ALIGNED still outperform baselines without symbolic refinement?

3. The paper’s datasets are mostly transcriptomic (gene-expression–based). Could the same approach generalize to cross-modal or spatial data (e.g., imaging, proteomics)?
Does ALIGNED require structural priors specific to GRNs, or can it adapt to other biological or symbolic domains?

4. Baseline comparisons (GEARS, scGPT, etc.) are strong, but all are data-driven. What about existing neuro-symbolic or knowledge graph–augmented models (e.g., DeepProbLog, NeuroLogic, or KG-BERT)?

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
3

### Summary
This paper proposes ALIGNED, a neuro-symbolic framework that jointly optimizes a neural predictor and a symbolic reasoning module over GRNs via abductive learning. The method introduces (i) a balanced consistency metric that merges agreement with data and with knowledge bases, (ii) an adaptive alignment mechanism (learned with REINFORCE) to select per-gene whether to trust neural or symbolic predictions, and (iii) gradient-based knowledge refinement with sparse regularization to update GRNs. Experiments on Norman, Dixit, Adamson (human/mouse) and an E. coli setting show higher “balanced consistency” than recent baselines and indicate that refinement can re-discover biologically meaningful relations.

### Strengths
Overall, I think this paper has a potentially significant contribution to perturbation-response prediction and neuro-symbolic learning, with a thoughtful framing of data-vs-knowledge trade-offs. With clarifications on theoretical assumptions, ablation coverage, and computational costs, I’d be inclined to raise my score.

### Weaknesses
1. The paper’s core ideas: balanced consistency, per-output adaptive selection, and sparse refinement—are compelling, but please delineate what is novel vs. adapted from prior ABL/REINFORCE literature (e.g., ABL variants, learnable trade-offs). A table that maps each ALIGNED component to closest prior art and states the delta would help.

2. Results implicitly require that either the neural model can approximate the true response mapping or the KB’s transitive closure (up to path length k) can approximate symbolic ground truth. Please state these assumptions explicitly and discuss failure modes when (a) GRN coverage is sparse/bias-prone or (b) δ_{KB} is systematically wrong for some modules. What guarantees (if any) are possible for convergence of the alternating align/refine loop under misspecification?

### Questions
1. How are threshold θ and weights w tuned—per dataset or globally? Any risk of overfitting via these hyperparameters?
2. For double-vs-single perturbations, does k need to scale, and does refinement overfit to short-cycle artifacts?  

Minor comments / nits
1. Typos/formatting: “Algin 2” → Align 2; “ALIGEND” → ALIGNED; ensure consistent “knowledge base (KB)” capitalization.
2. In Fig. 3–5 captions, explicitly state metric definitions (macro vs micro F1).

### Soundness
3

### Presentation
2

### Contribution
3
