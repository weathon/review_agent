# Hierarchical Protein Backbone Generation with Latent and Structure Diffusion

- Decision: Reject
- Scores: 3, 5, 3, 6

## Abstract
We propose a hierarchical protein backbone generative model that separates coarse and fine-grained details. Our approach called LSD consists of two stages: sampling latents which are decoded into a contact map then sampling atomic coordinates conditioned on the contact map. LSD allows new ways to control protein generation towards desirable properties while scaling to large datasets. In particular, the AlphaFold DataBase (AFDB) is appealing due as its diverse structure topologies but suffers from poor designability. We train LSD on AFDB and show latent diffusion guidance towards AlphaFold2 Predicted Alignment Error and long range contacts can explicitly balance designability, diversity, and noveltys in the generated samples. Our results are competitive with structure diffusion models and outperforms prior latent diffusion models.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
**There is a pentail violation of the anonymity policy in this paper**

This paper leverages the diffusion model to generate protein structures. Sampling is conducted on the level of atomic coordinates. The high-level guidance from the protein backbone presents the effects on generation. The experiment is conducted on AFDB to verify its effectiveness.

### Strengths
1. The paper has good quality and the proposed method looks working well.
2. It is novel to sample on the level of acid coordinates.
3. The experiment is conducted on AFDB and various performance metrics are reported.

### Weaknesses
1. **Violation of the anonymity policy.** It looks like the author changed the style file in the template. The references and table numbers are highlighted, which makes the paper stand out from other submissions. It is not fair for other submissions.

2. The paper proposes to sample the atomic coordinates. However, it is not clear how to guarantee that the sampled coordinates can form a possible structure on a higher level in practice. The constraint between the two levels is missing in the training process. Other important property such as SE3 is not discussed and addressed in the method.

3. Important theoretical proof is missing. It looks like the paper intuitively leverages the reverse process of the diffusion model on the sequence. The original diffusion models such as DDPM or DDIM conduct the forward and the reverse process on the same level, which makes solving the Markov process tracable. However, it is not proven it still works when applied to two different representations.

4. Quality of experiment. The paper only introduces **only one dataset** in the experiment. It is not conclusive. Please conduct the experiment on all benchmarks in [1].
[1] ProteinGym: Large-Scale Benchmarks for Protein
Fitness Prediction and Design

5. Hierarchical Protein generation is already proposed by existing works, which may limit the contribution of this work. The author may want to integrate more baselines such as [1]
[1]	Zhilin Huang, Ling Yang, Xiangxin Zhou, Chujun Qin, Yijie Yu, Xiawu Zheng, Zikun Zhou, Wentao Zhang, Yu Wang, Wenming Yang:
Interaction-based Retrieval-augmented Diffusion Models for Protein-specific 3D Molecule Generation. ICML 2024
[2] Multi-level Protein Structure Pre-training via Prompt Learning
[3] 	Yan Wang, Lihao Wang, Yuning Shen, Yiqun Wang, Huizhuo Yuan, Yue Wu, Quanquan Gu:
Protein Conformation Generation via Force-Guided SE(3) Diffusion Models. ICML 2024
[4] Alex Morehead, Jeffrey Ruffolo, Aadyot Bhatnagar, Ali Madani:
Towards Joint Sequence-Structure Generation of Nucleic Acid and Protein Complexes with SE(3)-Discrete Diffusion.

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
the paper proposes to split protein structural generations into 2 levels, namely latent (contact map) generation and structural (coordinates) generation. 2 other techs are proposed to enhance performance via gradient-guided conditional diffusion generation: long-range-contact (LRC) and predicted-aligned-error (PAE) guidances.

### Strengths
The motivation of combining 2-level generation is novel (at least in protein design) and promising. The framework allows one to connect semantics of proteins and their rigorous structures. For example, the guidance of PAE and LRC can be more elegantly incorporated through the new paradigm.

Section 5.2 is well-organized to show the benefits of LDM+SDM paradigm and the effect of PAE and LRC.

### Weaknesses
The main results in Table 2 indicates that existing structural diffusion methods are superior to the proposed one. I respect the authors for honestly presenting results that are less competitive, while the fact seems to be that latent diffusion does not work well on structural generation. I would reckon the authors to explore their methods on more specific applications to find better evidence for reviewers to support the submission,  eg language-conditioned protein design (to excavate the advantage of LDMs), loop design for antibodies or motif-scaffolding for enzymes.

### Questions
The authors may avoid calling their method *LSD*, as it causes negative associations.

several typos: L094, L285.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
Summary: This paper focuses on the task of protein backbone generation. The authors propose a new hierarchical method, called LSD, to generate protein backbones from coarse-grained to fine-grained. The first stage of LSD is to generate protein contact map (coarse-grained), and the second stage is to generate protein backbone conditioned on the contact map. An autoencoder (map between protein backbone, latent representation, and contact map) and two diffusion models are trained in this method. The model can generate diverse, designable, and novel protein backbones, and is comparable to current SOTA.

### Strengths
1. Although this paper is based on several previous papers and requires some background knowledge, it is still well-written and not difficult to understand.
2. The tables and figures are informative. 
3. The results are comparable to current SOTA.

### Weaknesses
1. About the model design and corresponding ablation study: The current ablation study is more about hyperparameter search, missing support for the method design, for example, why do we need the contact map. One ablation is a pure latent diffusion model without contact map (similar to GeoLDM).
2. About the task: this paper only focuses on backbone generation, can it be extended to all-atom generation? 
3. About baseline methods: AlphaFold3 also has a diffusion step. How to compare this method with AlphaFold3? 
4. About the training: This method includes several training stages and is not trained end-to-end, which may limit the final performance. Can it be trained end-to-end? If so, what are the results? 
5. In addition to the efficiency, what is the advantage over existing SDM (such as GENIE2)? We see that GENIE2 achieves good results. In addition, I am wondering why this method is more efficient than other methods. I think the diffusion step should be quite slow, and this method has two diffusion parts. In addition, from section 5.2, we know that k=4. And the input feature is 3*3=9. Therefore, the latent doesn’t compress the input a lot.

Other points:
1. For LDM, how does DiT deal with different input sizes? Do we need any other design?
2. Line 285: missing reference
3. Add reference to the contact map definition used here (line 158)

### Questions
See Weaknesses

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
4

### Summary
This paper proposed a hierarchical framework for protein backbone design containing two stages. The first stage is built upon a latent diffusion model: a structure-to-contact autoencoder based on ProteinMPNN encoder and contact predicter was pretrained to encode the protein structure to $L \times K$ representations. Then a modified Diffusion Transformer was trained in the latent space to generate synthetic representations. In the second stage, a modified version of FrameFlow was trained to design the atomic coordinates from the contact map. Additionally, two further guidance factors—long-range contact and predicted alignment error—were introduced to jointly enhance designability and diversity.

### Strengths
1) The overall architecture is well organized. 
2) The clear mathematical  formulation helps to understand the real pipeline of the models.
3)  Evaluations across various aspects highlight the contributions of different factors.

### Weaknesses
1) The motivation is a bit unclear to me, as there are several approaches to get rid of equivariance issue or to develop SE(3)-equivariant models. For instance, FoldingDiff uses internal coordinates, and FrameFlow’s main architecture is SE(3)-equivariant. Intuitively, I don’t see a significant advantage to applying diffusion models on contact maps.
2) I'm unclear on certain aspects of the model design, such as why only node-wise features from ProteinMPNN are used for the latent representation, and why FrameFlow was chosen for recovering atomic coordinates, given that there are other models for structure prediction based on contact maps.
3) It would be better to apply evaluation metrics to natural samples in order to reveal any biases in the evaluation pipeline. Additional analysis on how the models perform across proteins of varying sizes, classes, or folds would also be beneficial. Besides secondary structures, other biological analysis such as torsional angles, clashes and Ramachandran plots are needed.
4) Some typos need correction, such as the citation of Chroma in Section 4.

### Questions
1) Why did you choose contact map rather than distance matrix while the latter is also SE(3)-invariant and more informative? There are also constraints on the contact map, such as neighboring residues in a sequence being in contact and certain patterns reflecting the corresponding secondary structure. Did you apply any methods to enforce these constraints?
2) While FrameFlow was developed based on continuous normalizing flows (CNF) models, why didn't you also apply CNF for the first stage?
3) Could you please explain what is the designable pairwise TM-score for and why it is the smaller the better here?

### Soundness
3

### Presentation
3

### Contribution
3
