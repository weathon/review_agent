# Data-Efficient Molecular Generation with Hierarchical Textual Inversion

- Decision: Reject
- Scores: 6, 6, 6, 5

## Abstract
Developing an effective molecular generation framework even with a limited number of molecules is often important for its practical deployment, e.g., drug discovery, since acquiring task-related molecular data requires expensive and time-consuming experimental costs. 
To tackle this issue, we introduce Hierarchical textual Inversion for Molecular Generation (HI-Mol), a novel data-efficient molecular generation method.
HI-Mol is inspired by a recent textual inversion technique in the visual domain that achieves data-efficient generation via simple optimization of a new single text token of a pre-trained text-to-image generative model.
However, we find that its naive adoption fails for molecules due to their complicated and structured nature. 
Hence, we propose a hierarchical textual inversion scheme based on introducing low-level tokens that are selected differently per molecule in addition to the original single text token in textual inversion to learn the common concept among molecules.
We then generate molecules using a pre-trained text-to-molecule model by interpolating the low-level tokens.
Extensive experiments demonstrate the superiority of HI-Mol with notable data-efficiency. 
For instance, on QM9, HI-Mol outperforms the prior state-of-the-art method 
with 50$\times$ less training data. 
We also show the efficacy of HI-Mol in various applications, including molecular optimization and low-shot molecular property prediction.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Inspired by recent textual inversion technique in the visual domain, the authors proposed Hierarchical textual Inversion for Molecular generation (HI-Mol), a novel data-efficient molecular generation method. Extensive experiments demonstrate the superiority of HI-Mol with notable data-efficiency.

### Strengths
1. Adapting textual inversion to the molecule domain is novel.
2. The method is well-introduced and convincing. 
3. The authors validated the effectiveness of HI-Mol on several downstream tasks including the molecular optimization for PLogP and the low-shot molecular property prediction on MoleculeNet.

### Weaknesses
1. It is worth mentioning some recent works on molecule generation in related works such as:

[1] Hoogeboom E, Satorras V G, Vignac C, et al. Equivariant diffusion for molecule generation in 3d, ICML 2022  
[2] Zhang Z, Liu Q, Lee C K, et al. An equivariant generative framework for molecular graph-structure Co-design. Chemical Science 2023  
[3] Flam-Shepherd D, Zhu K, Aspuru-Guzik A. Language models can learn complex molecular distributions. Nature Communications, 2022  

2. Could HI-Mol leverage the structural information of molecules? Could it be adapted for 3D molecule generation?
3. How to effectively interpret the learned tokens? Do they have chemical meanings?

### Questions
N.A.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Recently proposed molecular generation methods are mainly trained on task-related data, which are computationally expensive. The authors propose a hierarchical textual inversion method for molecular generation to overcome this issue.

### Strengths
1.	Introducing the successful textural inversion methods from the computer vision area into the molecular generation area is a good idea. 
2.	The experimental results presented in the paper demonstrate the effectiveness of the proposed method.

### Weaknesses
1.	The authors should have a clearer motivation figure in the introduction, which could be specific examples of molecules, to demonstrate that the highly complicated and structured nature of molecules makes it difficult to apply textual inversion directly.
2.	The Molecular language model part in the Section 3.2 Preliminaries should be moved to the Related Work section.
3.	Table 2 should also show the results of HI-Mol without grammar.
4.	In Table 6, Valid decreases as the token hierarchical levels increases. The authors should provide some explanations and solutions for this issue.
5.	I suggest that the authors consider rearranging the positions of the figures and tables as some of them are too far from the references in the paper.

### Questions
1.	What is the ‘simple resampling strategy’ mentioned in Section 4.2?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces Hierarchical Textual Inversion for Molecular generation (HI-Mol), a novel approach to generate molecular structures efficiently with limited datasets. It leverages a new scheme of textual inversion tailored for molecules, using multi-level tokens to capture hierarchical information. This method enables the generation of molecules with significantly less data, showing superiority in various benchmarks such as the MoleculeNet and QM9 datasets, particularly improving data efficiency and performance in molecular property prediction and optimization tasks

### Strengths
- The problem they are trying to tackle is important and interesting.
- The idea looks relatively novel and justified since molecules are constructed of similar smaller components.
- The empirical results are promising.

### Weaknesses
- The method is not described clearly and in detail. For instance, in the following paragraph of Eq. 1, it is mentioned that the intermediate tokens are "selected" during training. This is unclear and should be discussed in more detail.

-  Figure 1 is not expressive enough to outline the method.

### Questions
- The authors have mentioned that they are using the Caption2Smiles frozen model as the backbone. Can they please share just the frozen model performance on the tasks?

- Sensitivity of the model's performance to the number of k sounds very important. According to Appendix E, table 10, there is no benefit in increasing k to more than 10. Do the authors have a hypothesis for that? Aren't the molecules supposed to have a lot more "clusters", sub-components?

- I'm curious how making this approach multi-modal can be helpful. Could graph embeddings or vision embeddings of the molecules provide any benefit? I'm not a molecular properties expert, but I tried a couple of the figures (table 2 and figure 2) with GPT-4 vision, and it gave meaningful explanations. Have the authors investigated this?

- Can the authors please provide the complete Qm9 results in table 15 of Appendix G? Specifically, what are the results at the 50% and 100% ratios?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors present a hierarchical textual inversion strategy, which uniquely selects low-level tokens for each molecule. Subsequently, molecules are generated through a pre-trained text-to-molecule model by interpolating these tokens. Comprehensive testing showcases the marked data-efficiency and superiority of HI-Mol.

### Strengths
1. The tackled problem is both intriguing and holds practical significance.

2. The paper is articulate and systematically presented.

3. The introduction of multi-level token embeddings enhances the textual inversion model.

4. Very strong experiments, which clearly show the superiority of the proposed method.

### Weaknesses
1. The main concern I have with this paper is its novelty. While the ideas of multi-level molecule representation and embedding interpolation are well-established in the field, the authors merely integrate them into the newly introduced textual inversion framework. This casts doubts over the paper's genuine novelty and the depth of its technical contribution.

2. The rationale for adopting the textual inversion model appears somewhat nebulous. In my understanding, compared to SMILES, graph representations are generally more adept at modeling molecular structures. Notably, many graph-centric molecular generation methods have already incorporated hierarchical concepts. In their experiments, while the authors argue that HI-Mol surpasses several graph-based models, the underlying reasons remain unelucidated. An elucidation on how the textual inversion model specifically augments the molecular generation task, relative to its graph-based counterparts, would be greatly beneficial.

3. There's an absence of crucial baselines. The authors have chosen to compare their work with two SMILES-based baselines, one based on RNN and the other on spanning tree. However, they have not included any methods based on large-scale text-to-molecule models, many of which are mentioned in the related works section. Therefore, it is unclear whether the improved performance of HI-Mol is attributable to the utilization of large-scale text-to-molecule models.

### Questions
See weakness

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
