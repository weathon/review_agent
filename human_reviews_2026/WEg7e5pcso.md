# ABConformer: Physics‑inspired Sliding Attention for Antibody-Antigen Interface Prediction

- Decision: Reject
- Scores: 0, 6, 4, 4

## Abstract
Accurate prediction of antibody-antigen (Ab-Ag) interfaces is critical for vaccine design, immunodiagnostics and therapeutic antibody development. However, achieving reliable predictions from sequences alone remains a challenge. In this paper, we present \textsc{ABConformer}, a model based on the Conformer backbone that captures both local and global features of a biosequence. To accurately capture Ab-Ag interactions, we introduced the physics-inspired sliding attention, enabling residue-level contact recovery without relying on three-dimensional structural data. ABConformer can accurately predict paratopes and epitopes given the antibody and antigen sequence, and predict pan-epitopes on the antigen without antibody information. In comparison experiments, \textsc{ABConformer} achieves state-of-the-art performance on a recent SARS-CoV-2 Ab-Ag dataset, and surpasses widely used sequence-based methods for antibody-agnostic epitope prediction. Ablation studies further quantify the contribution of each component, demonstrating that, compared to conventional cross-attention, sliding attention significantly enhances the precision of epitope prediction. To facilitate reproducibility, we will release the code under an open-source license upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
In this paper, the authors presented ABConformer, a physics-inspired sliding attention scheme to predict paratopes and epitopes given the antibody and antigen sequence. The method is evaluated on the SARS-CoV-2 Ab-Ag dataset against other sequence-based methods.

### Strengths
It is very hard to name a strength of the paper, see my comments in the weaknesses part.

### Weaknesses
# **This paper is basically copying things from a prior publication in Nature Machine Intelligence.** 

This paper is obviously following the paper *Sliding-attention transformer neural architecture for predicting T cell receptor-antigen-human leucocyte antigen binding, Nature Machine Intelligence, 2024, 6(10): 1216-1230*. All the key technical details are similar, including the definition of the distance based attention and embedding based attention, how they are combined together with a template M, and how the sliding attention updates the 1-D coordinate of the protein sequences. The titles of the two papers are also identical. The only difference is the datasets used.

While drawing inspiration from published work is common, outright copying ideas and algorithms from a journal paper and presenting them as your own at a conference crosses ethical boundaries and undermines academic integrity.

### Questions
I am curious about what led the authors to believe that such a blatant plagiarism would escape detection by reviewers?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces ABConformer, a sequence-based framework that employs a sliding attention mechanism within a Conformer backbone for accurate antibody–antigen (Ab–Ag) interface prediction without requiring 3D structural data. The model captures both local and global sequence dependencies while incorporating a physics-inspired Gaussian sliding attention that mimics local docking interactions between VH/VL domains and antigens. This design enables fine-grained residue-level contact recovery and improves both paratope and epitope prediction accuracy and efficiency.
Experiments on a recent SARS-CoV-2 Ab–Ag dataset show that ABConformer outperforms existing sequence-based baselines and generalizes well to both antibody-specific and antibody-agnostic prediction tasks.

### Strengths
1.	Clear motivation and novelty
The paper addresses an important and challenging problem — antibody–antigen interface prediction from sequence alone — with a clear motivation and innovative approach. The sliding attention mechanism provides both computational efficiency and improved predictive power.
2.	Strong empirical performance
ABConformer achieves superior results over multiple state-of-the-art baselines on widely used benchmark datasets, demonstrating its robustness and practical relevance.
3.	Comprehensive ablation and interpretability
The ablation studies are thorough and convincing. Attention maps align well with known structural interface regions, offering biologically meaningful interpretability. The model also exhibits reasonable generalization in antibody-agnostic scenarios.
4.	High-quality presentation
The manuscript is clearly written, well-structured, and easy to follow. Figures and tables effectively communicate the results.

### Weaknesses
1. Experimental scope and robustness
The experiments are generally solid and well-structured, with clear ablation analyses and sensitivity tests that support the model’s main claims. However, although the baseline coverage is fairly comprehensive, several recent and competitive approaches using protein language models such as ESM-IF1, ProtT5, or AlphaBind are not included. Adding these stronger and more contemporary baselines would provide a fairer and more convincing comparison with current SOTA methods. Reporting computational efficiency or variance across folds would further improve the robustness and credibility of the experimental evidence.

2. Generalization across datasets
While the authors include an external evaluation on the SARS-CoV-2, its limited scale and close domain similarity to AACDB reduce the strength of the generalization argument. Additional experiments on heterogeneous datasets, such as HIV or influenza antibody antigen complexes, which would significantly enhance the credibility of the claimed generalization capability.

3. Fusion strategy justification
The separate VH and VL branches followed by fusion are biologically motivated, as VH and VL contribute differentially to paratope formation and contact distinct regions of the antigen surface. Nevertheless, the rationale for choosing this particular fusion strategy should be explicitly stated and compared with possible alternatives to enhance clarity and justification.

### Questions
NA

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
This paper presents ABConformer, a sequence-based deep learning model for predicting antibody-antigen binding interfaces. The method builds upon the Conformer architecture and introduces a sliding attention mechanism adapted from prior work to simulate molecular docking processes. The model processes antibody heavy chain, light chain, and antigen sequences separately using ESM-2 embeddings as input features. The sliding attention mechanism iteratively adjusts relative positions between sequences through Gaussian kernels for spatial proximity and attention-weighted position updates. The authors evaluate their method on AACDB and an external SARS-CoV-2 dataset, demonstrating improvements over existing sequence-based methods in antibody-specific interface prediction and competitive performance for antibody-agnostic epitope prediction, though with notable limitations in recall and methodological novelty.

### Strengths
**S1**. While the sliding attention mechanism is not original to this work, the authors make appropriate adaptations to the antibody-antigen context through a three-branch architecture. The dual sliding process and linear combination of resulting embeddings represent sensible design choices for this specific application. However, the novelty lies primarily in the application domain rather than in methodological innovation.

**S2**. The explicit separation of Ab-H and Ab-L chains is more biologically grounded than treating antibodies as monolithic entities, as paratopes are formed by complementarity-determining regions (CDRs) from both heavy and light chains. This design choice aligns well with the known structural organization of antibody-antigen interfaces.

**S3**. The paper provides thorough experimental evaluation, including extensive ablation studies examining encoding strategies and attention mechanisms, sensitivity analyses of hyperparameters, and case studies.

### Weaknesses
**W1**.The paper does not adequately justify how binding site prediction translates to practical antibody design applications. Specifically: (i) How does interface prediction assist in binding affinity estimation, which is often the ultimate goal in therapeutic antibody development? (ii) What is the relationship between predicted interface residues and functional properties? (iii) In what scenarios would researchers prefer interface prediction over structure prediction followed by docking simulations? Without addressing these questions, the practical motivation for this work remains insufficiently justified.

**W2**. The paper proceeds directly to methodology without establishing a rigorous mathematical formulation of the prediction task. A complete problem statement should specify: (i) input, (ii) output, and (iii) objective.

**W3**. The sliding attention mechanism is directly adapted from previous work, with the primary modifications being: (i) extension from two-component to three-branch architecture, (ii) sequential dual sliding operations (Ag→Ab-H, then Ag→Ab-L), and (iii) linear combination of embeddings after each sliding step. The paper should more explicitly distinguish between novel contributions and adaptations of existing methods. Given the limited algorithmic innovation, the contribution might be more accurately characterized as application engineering rather than methodological advancement.

**W4**. The assertion that ABConformer *surpass widely used sequence-based methods* is misleading. According to Table 1 and Figure 9D, ABConformer achieves superior performance only on certain metrics while showing substantially lower recall. In addition, the paper acknowledges that *the sliding-attention module has no effect* for antibody-agnostic prediction because antibody embeddings are set to zero, which reduces ABConformer to a Conformer backbone. This raises the question: why not develop and compare a dedicated Conformer-only model for antibody-agnostic prediction?

**W5**. While the paper includes AlphaFold2 Multimer v3 as a baseline, it omits several highly relevant and state-of-the-art structure-based approaches: (i) AlphaFold3 [1] has demonstrated strong performance on protein-related tasks; (ii) ESMFold [2] uses the same ESM-2 embeddings as ABConformer's encoder, making it a critical baseline for isolating the value added by the sliding attention mechanism; (iii) Boltz-1 [3] and Boltz-2 [4] are recent methods specifically designed for biomolecular interaction modeling and binding affinity prediction; (iv) PAbFold [5] is specifically designed for antibody epitope prediction using AF2. Without these comparisons, the paper cannot substantiate its implicit claim that sequence-based prediction with sliding attention offers advantages over structure prediction followed by interface extraction.

---

**Reference*

[1] J. Abramson et al. *Accurate structure prediction of biomolecular interactions with AlphaFold 3*. Nature 2024.

[2] Z. Lin et al. *Evolutionary-scale prediction of atomic-level protein structure with a language model*. Science 2023.

[3] J. Wohlwend et al. *Boltz-1 democratizing biomolecular interaction modeling*. BioRxiv 2025.

[4] S. Passaro et al. *Boltz-2: Towards accurate and efficient binding affinity prediction*. BioRxiv 2025.

[5] J. DeRoo et al. *PAbFold: linear antibody epitope prediction using AlphaFold2*. BioRxiv 2024.

### Questions
**Q1**. How does binding site prediction assist in practical antibody design workflows? Specifically: (i) Can predicted interfaces be used to estimate binding affinity, and if so, what is the correlation? (ii) How might researchers use these predictions in therapeutic antibody optimization or vaccine design?

**Q2**. How does ABConformer compare to a baseline that computes pairwise cosine or Euclidean distances between ESM-2 embeddings of antibody and antigen residues, followed by thresholding? This comparison would isolate the value added by the sliding attention mechanism beyond embedding similarity.

**Q3**. Why are results of Epi4Ab omitted in Table 1?

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
This paper presents ABConformer, a new model that predicts antibody-antigen binding interfaces using only their protein sequences. The  key innovation is a physics-inspired sliding attention mechanism, which aims to mimic the molecular docking process by sliding the antigen sequence against the antibody's heavy and light chains to find stable interaction patterns, without the need for 3D structural data.

### Strengths
- The physics-inspired sliding attention is an interesting innovation, which outperforms conventional cross-attention, especially in improving the precision of epitope prediction.
- The model is flexible, and can handle antibody-specific prediction (when all sequences are known) while also being effective for antibody-agnostic pan-epitope prediction (using only the antigen sequence), where it surpasses other sequence-based methods.

### Weaknesses
- The core innovation of this paper is the sliding attention module, which isn't used for the antibody-agnostic predictions.
- Comparisons are performed on a small external dataset of 35 SARS-CoV-2 complexes, which raises questions about generalizability. ABconformer is outperformed by DiscoTope-3.0, a structure-based method on the antibody-agnostic task. Recent structure-based methods such as AlphaFold3, Boltz-2 or Chai are also not included in the antibody-specific comparisons.
- The ablation study shows that the gap between sliding attention and conventional cross-attention is small, with cross-attention achieving slightly higher recall, but worse precision.

### Questions
- Could the authors include other benchmarks than SARS-CoV-2?
- It would be valuable to include a comparison with AlphaFold3, or other recent structure prediction models.

### Soundness
3

### Presentation
3

### Contribution
2
