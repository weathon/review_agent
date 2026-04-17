# Distilling Causal Signals for One-Shot Directed Evolution of Antibodies

- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
Improving antibody binding to an antigen without antibody–antigen complex structures or antigen-specific training data is a central challenge in therapeutic protein design. We introduce **AffinityEnhancer**, a framework for one-shot antibody affinity improvement with strong generalization: given a single lead sequence, we propose variants that increase affinity without fine-tuning on the lead and without using antigen information, epitope/paratope labels, or the lead’s structure in complex with the antigen. During training, AffinityEnhancer leverages a pan-antigen dataset of diverse binding environments (antigens) and constructs paired examples of related sequences with higher vs. lower measured binding. A shared, structure-aware module learns to transform low-affinity sequences toward high-affinity ones, distilling consistent, causal features associated with improved binding across environments. By combining pretrained sequence–structure embeddings with a sequence decoder, AffinityEnhancer generalizes to entirely unseen antibody seeds. Across multiple held-out internal and public leads, AffinityEnhancer concentrates mutations on the rim of the paratope, outperforms existing structure-conditioned and inpainting baselines, and achieves substantial in silico affinity gains in true one-shot experiments, despite never observing antigen-specific data at test time.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents AFFINITYENHANCER, a one-shot framework for antibody affinity improvement: given a single lead antibody sequence, the method learns residual mappings in a structure-aware embedding space from matched (low-affinity, high-affinity) pairs collected under the same antigen environment, and decodes improved candidates. The authors report considerable in-silico gains against other baseline methods on several antibody seeds, including Trastuzumab.

### Strengths
- The one-shot affinity-maturation problem addressed in this paper is practical and challenging in therapeutic antibody development. In real-world discovery scenarios, we frequently have only a single “lead” sequence and lack complex structural data or large matched datasets. 
- The authors propose a novel, matching-based computational framework for antibody affinity optimization that learns from matched low to high affinity pairs. This training strategy avoids relying on antigen–antibody complex structures during training and appears capable of improving performance in low-data antigens.
- In the reported in-silico evaluations, the proposed model outperforms baseline methods on several held-out antibodies.

### Weaknesses
Major
- Most of the reported improvements are evaluated using the Cortex predictor alone. Cortex itself is a deep-learning model and may introduce biases in the evaluation. The authors could supplement their validation with some other state-of-the-art predictors to provide a more comprehensive and robust assessment.
- The method depends on predicted structures (ESMFold). Antibody structure prediction, especially for the highly variable CDR loops, remains a difficult task. If the predicted structure is inaccurate, the residual mapping may learn an incorrect transformation. The paper currently lacks a failure analysis or robustness test for this critical scenario.

Minor
- Line 48: miniscule -> minuscule

### Questions
- The caption for Table 1  states: "wet-lab positives...". Please clarify: How was this "wet-lab positive" data obtained? Were any wet-lab experiments actually performed as part of this study?
- Did the authors try using sequence-only inputs (i.e., no predicted structure) for the model? Since predicted antibody structures are often inaccurate.

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
5

### Summary
The paper proposes AFFINITYENHANCER, a one-shot framework for antibody affinity maturation without requiring antigen structure or fine-tuning on target-specific data. The model learns to enhance antibody binding by training on matched pairs of antibody sequences with known differences in binding affinity, leveraging structure-aware embeddings (GearNet) and a Graph Transformer that maps low-affinity to high-affinity embeddings. The approach is motivated by the lack of antigen-specific data and aims to generalize to unseen antibody seeds through causal learning signals derived from matched improvements across multiple environments. Experimental results on public datasets reportedly show that the method outperforms existing structure-conditioned and sequence-based baselines in improving affinity in silico.

### Strengths
- Proposes a clear formulation of "one-shot affinity maturation", which addresses an under-explored scenario.

- The matching-based learning framework is well explained and connected to causal inference and preference learning.

- Integration of sequence–structure embeddings and a residual graph transformer is technically sound and leverages state-of-the-art pretrained models.

- The paper includes some theoretical grounding for causal signal isolation in matched-pair training.

### Weaknesses
- Limited novelty compared to previous works such as "Protein Design by Directed Evolution Guided by Large Language Models" (IEEE TEVC 2025) [1] and "LatentDE: latent-based directed evolution for protein sequence design" (MLST 2025) [2]. Both prior studies also model directed evolution or affinity optimization using pretrained representations and search in learned latent spaces; the main difference here is merely the inclusion of structure-aware embeddings.

- The "one-shot" setting is more of a data-split protocol than a conceptual advance: similar generalization settings were already evaluated in LatentDE and related MLDE pipelines.

- No experimental validation (either wet-lab or cross-docking) to confirm the claimed affinity improvement; results are purely in silico.

- The methodological overlap with Property Enhancer (PropEn, 2024) is substantial, with only minor modifications (structure-conditioned matching).

- The causal theory section adds mathematical formality but limited biological insight or empirical validation.

*** References:

[1] Trong Thanh Tran and Truong-Son Hy, Protein Design by Directed Evolution Guided by Large Language Models, IEEE Transactions on Evolutionary Computation, vol. 29, no. 2, pp. 418-428, April 2025, DOI 10.1109/TEVC.2024.3439690.
URL: https://ieeexplore.ieee.org/document/10628050

[2] Thanh V. T. Tran, Nhat Khang Ngo, Viet Thanh Duy Nguyen, and Truong-Son Hy, LatentDE: Latent-based Directed Evolution for Protein Sequence Design, Machine Learning: Science and Technology, Volume 6, Number 1, DOI 10.1088/2632-2153/adc2e2.
URL: https://iopscience.iop.org/article/10.1088/2632-2153/adc2e2/pdf

### Questions
How does your model differ in practice from prior ML-based directed evolution frameworks (e.g., LatentDE or LLM-guided directed evolution) beyond adding structure embeddings? Specifically, what unique biological or algorithmic insights does AFFINITYENHANCER provide that cannot be replicated by those prior approaches?

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
4

### Summary
The paper introduces AFFINITYENHANCER, a machine learning framework for one-shot affinity maturation of antibodies without requiring antigen or complex structural data. It builds on the concept of matched data (pairs of antibodies with measured affinity differences) and learns a residual mapping in embedding space from lower- to higher-affinity antibodies. The model uses GearNet for structure–sequence embeddings, a graph transformer to model causal affinity-enhancing directions, and a decoder trained on antibody sequence data. Experiments demonstrate strong in-silico improvements compared to baselines (PropEn, AntiFold, IgCraft) on SKEMPI-derived datasets and Trastuzumab.

### Strengths
Clearly formalizes one-shot affinity maturation, which is both scientifically and computationally relevant.
Provides a theoretical causal justification for why “matched improvements” capture meaningful binding signals.

Introduces a structure-aware residual map leveraging graph transformers and pretrained geometric embeddings.
The causal analysis (Theorem 1) connects local smoothness and bounded spurious movement to matched-pair supervision.

Comprehensive comparisons and ablations (matching, embedding, adjacency, transformer vs CNN).
Evaluations on multiple seeds, including a public antibody (Trastuzumab).
Consistent improvements over strong baselines.

The motivation and positioning against sequence-only and structure-conditioned baselines are clearly articulated.
The one-shot constraint is well justified with biological and data-availability arguments.

### Weaknesses
Experimental validation limited to in-silico predictions.
The study depends entirely on oracle models (Cortex) for affinity estimation.
Some discussion of potential wet-lab verification or validation strategy would strengthen impact.

Generalization claim could be quantified better.
“One-shot” evaluation is convincing qualitatively, but out-of-distribution metrics or diversity analyses are limited.
No discussion of potential failure cases (e.g., antibodies far from training distribution).

Complexity and reproducibility.
The model stack (GearNet + decoder + graph transformer) is nontrivial; ablation on computational cost or scalability would help.
The reproducibility section promises artifact release, but details like parameter counts per component or training time are missing.

### Questions
1.	Have you evaluated how the model performs on antibodies with significantly different scaffolds or germlines (i.e., distribution shift)?
2.	How sensitive is the method to the quality or number of matched pairs — could noisy or synthetic matches degrade performance?
3.	Could the model be extended to incorporate weak antigen information, e.g., paratope–epitope contact priors?
4.	Do you have plans for experimental or in-vitro validation to verify that the predicted affinity gains translate to real-world improvements?

### Soundness
3

### Presentation
3

### Contribution
3
