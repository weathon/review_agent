# IgFlow-LM: De Novo Antibody Design via Joint Flow Matching on SE(3) and Protein Language Models Probability Flows

- Decision: Reject
- Scores: 2, 4, 2, 2

## Abstract
In this work, we present IgFlow-LM, a multi-modal deep generative model for de novo antibody design based on a flow-matching framework that integrates protein language models (PLMs). By learning the joint distribution over SE(3)-equivariant structural flows and PLM-derived probabilistic flows, IgFlow-LM enables the coordinated generation of antibody 3D structures and latent embeddings in the PLM space. Experimental results demonstrate that, in unconditional design, IgFlow-LM generates antibody structures that closely resemble naturally occurring antibodies. IgFlow-LM generates antibodies closely resembling naturally observed ones, with backbone dihedral angles exhibiting strong agreement with reference antibody distributions and overall backbone conformations adhering more closely to physical constraints. Furthermore, we benchmark IgFlow-LM against baseline models on two commonly studied conditional CDR design tasks. IgFlow-LM demonstrates superior overall performance compared to baselines and generates CDR sequences with higher diversity.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed a flow-matching-based model for antibody sequence-structure codesign. It consists of a SE(3) equivariant flow matching module for structure generation and a latent flow matching module to generate protein language model embeddings which are then decoded into antibody sequences.

### Strengths
- The formulation of sequence-structure joint formulation is natural.
- Using pretrained protein language models might improve the sequence generation quality.

### Weaknesses
- This work overlooked the most of previously published research on generative models for antibody sequence-structure codesign [1,2,3,4,5]. It lacks both discussion and benchmarking.
- Using the flow matching technique to generate antibody has been explored a lot [1,2,3,5]. Combining diffusion models and protein language models to design antibodies has also been explored [2,6]. Therefore, the core contribution of this work is not novel.
- The evaluation of this work is mostly limited to comparing the distribution of the model and the datasets (RMSD, Ramachandran, LOGO Plot). There is no metrics that measure the property of the designed antibody (e.g. binding to specific antigens, humanness, etc).
- The conditional design setting (Section 4.2) is confusing. The section first says the generation is conditioned on the regions other than CDRs, but later it shows DockQ scores. DockQ scores assess the docking quality, so what target is used for docking? Is the antigenic target used as a condition? If so, what is the protocol for docking?
- Figure 4 and 5 do not convey useful information. Figure 4 shows that tyrosine (Y) dominates the generated antibody CDRs, which is an evidence of the model learning only trivial information. In Figure 5, the data points from Sabdab are smeared. There are no informative patterns shown in the figure.

[1] Antigen-Specific Antibody Design and Optimization with Diffusion-Based Generative Models for Protein Structures. 2022

[2] A Hierarchical Training Paradigm for Antibody Structure-sequence Co-design. 2023

[3] Atomically accurate de novo design of antibodies with RFdiffusion. 2024

[4] GeoAB: Towards Realistic Antibody Design and Reliable Affinity Maturation. 2024

[5] dyAb: Flow Matching for Flexible Antibody Design with AlphaFold-driven Pre-binding Antigen. 2025

[6] Antibody Design Using a Score-based Diffusion Model Guided by Evolutionary, Physical and Geometric Constraints. 2024

### Questions
See weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
For sequence-structure co-design for de novo antibody, the authors proposed a new IgFlow-LM method that improves the existing IgFlow method by using continuous numerical representations with PLM instead of the discrete flow matching in IgFlow. Experiments showed higher diversity in the generated samples. However, continuous numerical representation PLM has already been used in the existing method, the novelty in terms of methodology seems not significant. I am also not convinced that the generated samples with higher diversity are realistic. Other than diversity, the improvements compared with IgFlow shown in Fig 3 and other tables and figures seem not significant.

### Strengths
For sequence-structure co-design for de novo antibody, the authors proposed a new IgFlow-LM method that improves the existing IgFlow method by using continuous numerical representations with PLM instead of the discrete flow matching in IgFlow. Experiments showed higher diversity in the generated samples.

### Weaknesses
1.	As presented in appendix A.1, continuous numerical representation PLM has already been used in the existing method (Ding et al 2019), the novelty in terms of methodology seems not significant.
2.	I am not convinced that the generated samples with higher diversity are realistic.
3.	Other than diversity, the improvements compared with IgFlow shown in Fig 3 and other tables and figures seem not significant.

### Questions
1.	As presented in appendix A.1, continuous numerical representation PLM has already been used in the existing method (Ding et al 2019), the novelty in terms of methodology seem not significant.
2.	I am not convinced that the generated samples with higher diversity are realistic.
3.	Other than diversity, the improvements compared with IgFlow shown in Fig 3 and other tables and figures seem not significant.
4.	The reference of IgFlow is never given.
5.	The meaning of SO(3) and SE(3) should be given. Not all the readers are familiar with these notations.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents IgFlow-LM, a novel multi-modal generative model for de novo antibody design. The core contribution is a flow-matching framework that jointly generates the 3D backbone structure of an antibody and its corresponding sequence representation in the continuous latent space of a protein language model (PLM). Specifically, it combines an SE(3)-equivariant flow for the atomic coordinates with a probability flow over the PLM embeddings. The authors claim this joint, continuous approach avoids the limitations of methods that operate on discrete amino acid sequences, leading to generated antibodies with higher sequence diversity, improved structural fidelity, and stronger adherence to biophysical constraints. The authors demonstrate the effectiveness of the mode with two design tasks.

### Strengths
1. The idea of performing joint flow matching on both the SE(3) manifold for structure and the PLM embedding is simple and an elegant solution to the challenge of co-designing sequence and structure.

2. Experiments demonstrate the efficacy of the proposed method.

### Weaknesses
1. The innovation is limited. Adding PLM into a structure-based model is common in many protein-related models. 

2. The primary baseline is IgFlow for ablation. The paper would be strengthened by more comprehensive comparisons against other state-of-the-art antibody co-design methods mentioned in the related works. In the evaluation, no sequence diversity is presented.

3. Some descriptions in the paper are not very clear. For example, the model IgFlow is not clearly defined. In table 1, the caption says RFDiffusion, but IgFlow in the table. And the evaluation of DockQ is not clear to me. In lines 407-408, is the framework aligned?

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes IgFlow-LM, a multimodal generative model for de novo antibody design that unifies SE(3)-equivariant structural flow matching (inspired by FrameFlow) with continuous flow matching in the latent space of a protein language model (PLM)—specifically IgBERT. The method jointly generates 3D backbone structures and PLM embeddings via a shared ODE-based flow-matching framework, followed by a decoder that maps embeddings back to discrete amino acid sequences. The authors evaluate IgFlow-LM on two tasks to validate the effectiveness of the proposed technique.

### Strengths
1. The integration of PLM latent flows with SE(3) structural flows is conceptually elegant and aligns with recent trends in multimodal protein generative modeling.

2. The paper demonstrates empirical results on structural validity: lower bond/angle deviations, better Ramachandran plot adherence (lower KL divergence), and high self-consistency across multiple folding predictors (ABB2, IgFold, ESMFold).

3. The use of germline-split data partitioning mitigates data leakage, enhancing result credibility.

### Weaknesses
1.	Lack of comparison with state-of-the-art diffusion-based PLM-integrated methods: In recent years, numerous works have successfully combined protein language models with diffusion frameworks for antibody design, including AbX, IgGM, and other Model such as DiffAb, dyMEAN, and RFantibody (an antibody-specific adaptation of RFdiffusion. These methods also perform joint or conditional sequence-structure co-design and report strong results on CDR generation, diversity, and structural fidelity. Without head-to-head evaluation on identical metrics and datasets, the performance advantage remains unsubstantiated.

2. Lack of functional validation: Despite claims about improved antigen binding (via DockQ), the docking experiment is based on only 20 antibodies, uses side-chain repacking only on CDRs, and reports high variance (DockQ: 0.47±0.18). Without binding affinity prediction (e.g., via deep learning or Rosetta) or wet-lab validation, functional superiority remains speculative.

3. Computational cost and scalability unaddressed: Flow matching with ODE integration is expensive. The paper does not report generation time, GPU memory usage, or scalability to full-length antibodies (only variable domains). This raises questions about practical utility.

### Questions
1. Comprehensive baseline comparison: Could the authors include comparisons with AbX, IgGM, DiffAb, dyMEAN, and RFantibody on the same conditional CDR design tasks using identical metrics (e.g., scRMSD, positional entropy, DockQ, developability scores)? This is essential to establish whether IgFlow-LM offers a genuine advance over current SOTA.

2. Functional relevance: Can the authors provide more robust evidence of improved antigen binding? For example, could they compute binding energy (ΔG) using Rosetta or a deep learning predictor (e.g., ABlooper, DeepAAntibody) on a larger set (≥100 antibodies)?

3. True co-design?: Is the PLM embedding updated during structure generation, or is it fixed from the ground-truth sequence? If fixed, doesn’t this mean the model is essentially doing structure-conditioned sequence generation, not joint co-design?

4.Generalization and robustness: How does IgFlow-LM perform on out-of-distribution antigens or non-canonical CDR lengths? The current test set is derived from SAbDab, which may not reflect therapeutic design scenarios.

5. Efficiency vs. benefit: Please report generation time per antibody and GPU memory consumption. Given the computational overhead of ODE integration, is the marginal gain in diversity/structure worth the cost compared to faster diffusion or inverse folding pipelines?

6. Tyrosine overrepresentation in HCDR3: In Figures 4, the generated HCDR3 sequences show an extreme enrichment of tyrosine (Y), especially at central positions. Is this reflective of the training data, or an artifact of the joint flow-matching objective or decoder bias? Could the authors compare the per-position amino acid frequencies in generated vs. natural HCDR3 (e.g., from SAbDab) to quantify this discrepancy? If Y is overrepresented, does this compromise the model’s capacity to generate non-canonical but functional binders (e.g., those rich in serine, glycine, or charged residues)?

### Soundness
2

### Presentation
2

### Contribution
2
