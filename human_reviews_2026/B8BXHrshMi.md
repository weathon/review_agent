# FlexRibbon: Joint Sequence and Structure Pretraining for Protein Modeling

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4

## Abstract
Protein foundation models have advanced rapidly, with most approaches falling into two dominant paradigms. Sequence-based language models (e.g., ESM-2) capture sequence semantics at scale, and a number of recent works incorporate structural signals into sequence encoders. MSA-based predictors (e.g., AlphaFold 2/3) achieve accurate folding by exploiting evolutionary couplings, but their reliance on homologous sequences makes them less reliable in highly mutated or alignment-sparse regimes. We present FlexRibbon, a pretrained protein model that jointly learns from amino acid sequences and three-dimensional structures. Our pretraining strategy combines masked language modeling with diffusion-based denoising, enabling bidirectional sequence-structure learning without requiring MSAs. Trained on both experimentally resolved structures and AlphaFold 2 predictions, FlexRibbon captures global folds as well as flexible conformations critical for biological function. Evaluated across diverse tasks spanning interface design, intermolecular interaction prediction, and protein function prediction, FlexRibbon establishes new state-of-the-art performance on 12 different tasks, with particularly strong gains in mutation-rich settings where MSA-based methods often struggle.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
FlexProtein unifies protein structure prediction and design. It learns to model the sequence via masked language modeling and learns to model the structure via diffusion-based denoising. This flexible framework is able to generate both structures and sequences and show impressive results on interface, interaction and function prediction.

### Strengths
The framework of co-generating sequences and structures is very interesting and an intuitive framework.
The authors show that interesting results on interface, interaction and function prediction tasks.

### Weaknesses
While the downstream tasks are very interesting, a framework that "unifies protein structure prediction and design" should evaluate on structure prediction.

The abstract mentions that "Sequence-only language models (e.g., ESM-2) capture sequence semantics at scale but lack structural grounding". However there are many works that aim to bridge this gap.

[1] Learning protein sequence embeddings using information from structure. Tristan Bepler, Bonnie Berger.

[2] ProstT5: Bilingual Language Model for Protein Sequence and Structure. Michael Heinzinger,  Konstantin Weissenow, Joaquin Gomez Sanchez, Adrian Henkel, Martin Steinegger, Burkhard Rost.

[3] Distilling Structural Representations into Protein Sequence Models. Jeffrey Ouyang-Zhang, Chengyue Gong, Yue Zhao, Philipp Krähenbühl, Adam R Klivans, Daniel J Diaz.

[4] Structure-Informed Protein Language Model. Zuobai Zhang, Jiarui Lu, Vijil Chenthamarakshan, Aurélie Lozano, Payel Das, Jian Tang.

### Questions
How does FlexProtein perform on monomeric protein structure prediction against ESMFold and AlphaFold?

How does FlexProtein compare to IgML on design when used in the sequence-only model?

### Soundness
3

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
3

### Summary
The paper presents FlexProtein, a joint sequence–structure pretrained model for proteins that removes dependence on multiple sequence alignments.
It integrates a diffusion-based structural generator and a structure-informed masked language model (SIMLM) into a unified framework.
The framework is trained on ~78M AFDB and 180k PDB structures, and a pLDDT-weighted denoising loss enables the model to learn reliable geometry from noisy predicted structures.
Experiments across 12 benchmarks show the performance of the proposed method.

### Strengths
1. This paper is well motivated, this paper addresses a well-defined and flexible problem: learning unified protein representations without relying on MSA
2. The proposed framework combines diffusion-based structure denoising and structure-informed masked language modeling achieving bidirectional supervision between sequence and structure
3. Extensive experiments across 12 heterogeneous benchmarks demonstrate the performance and generalization ability of proposed FlexProtein
4. Specifically, FlexProtein achieves consistent gains in flexible or high mutation regimes, where most MSA-based or sequence-only models degrade sharply, providing strong support for the MSA-free framework.

### Weaknesses
Overall, I find this paper well-motivated, technically mature, and experimentally comprehensive. In my view, this work represents a solid contribution to the emerging direction of general-purpose protein foundation models. I have only 1 point of technical curiosity:
* Given the scale of the training setup and the diffusion architecture, I am curious about the inference efficiency and fine-tuning cost. While the paper provides strong empirical evidence across 12 benchmarks, it would be valuable to include quantitative runtime or memory profiles. This information would help assess FlexProtein’s practicality for real-world downstream tasks.

### Questions
I found the paper technically strong and conceptually consistent, my main questions are reflected in the weakness section.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces FlexProtein, a unified framework for joint sequence-structure modeling. The method integrates a diffsuion-based structural denoising model with structure-informed masked language model to enable bidirectional learning between sequence and structure.
It consists of three modules: (1) a sequence encoder that performs MLM (2) a coarse-grained structure module that learns residue-level topology with noisy structure and sequence (3) an all-atom structure module that refines atomic coords.
The authors finetuned on several design tasks including antibody and nanobody design, intermolecular interaction prediction and protein function prediction and show great results.

### Strengths
1. This paper proposed a unified modeling of sequence and structure. The integration of structure-informed MLM with diffusion enables the model to simultaneously reason over both modalitiies.
2. This paper also unified interface for proteins and small molecules. This allows straighfoward protein-ligand binding affinity/docking prediction.

### Weaknesses
1. Lack of comparison with other models. According to the architecture, it looks like FlexProtein should be able to perform structure prediction and inverse folding. However, there is no any benchmark on that. I suggest the authors can compare against DPLM-2 on these two tasks.
2. Lack of ablation study. In this paper the authors introduce three modules, while the authors did not show any ablation study on these modules. For example, the coarse-grained structure module can be removed according to the framework. Will there be any difference? The authors did not show that.
3. The authors propose the idea of "structure-informed masked language model" but did not cite the paper that exactly used structured-informed protein language model for protein design. [1]

There are also issues with the presentation of the paper:
1. In Figure 1, the authors draw a line from $D _ \theta(R_t, t)$ to single step. I was confused when I first saw this. This should refers to the inferece stage, right? While the loss indicates it's training stage. I suggest the authors seperate the training and inference stages.
2. In line 171, the authors wrote "For small molecules, we incorporate 2D topology". While this can't be found in the framework. I suggest the authors should add the details about small molecules are added to the framework in the figure.

[1] Zheng, Z., Deng, Y., Xue, D., Zhou, Y., Ye, F., & Gu, Q. (2023, July). Structure-informed language models are protein designers. In International conference on machine learning (pp. 42317-42338). PMLR.

### Questions
See weakness above. I will raise the scores accordingly if the experiment concerns could be solved.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
FlexProtein is a pretrained protein model that jointly learns from sequences and 3D structures without MSAs, addressing limitations of existing sequence-only/MSA-based models. It captures global folds and flexible conformations via masked language modeling and diffusion-based denoising. It sets new SOTA on 12 tasks, notably in mutation-rich scenarios.

### Strengths
- This study builds a 3B protein foundation model and tests on 12 downstream tasks. The experiments are very comprehensive and the results are promising.
- FlexProtein enables codesign of sequence and structures for antibodies/nanobodies, which may surpass its MSA-free counterparts since these models may struggle due to few MSA data.

### Weaknesses
- Lack of ablation study for several key components.
	- The training objective: how does plddt-based reweighting affect the performance?
	- Pretraining tasks: there are three pretraining modes: mode 1 (seq-to-structure), mode 2 (coupled perturbation), and mode 3 (seq-masked global perturbation). How does the balance of these modes change the final results?
- Potential data leakage. In Figure 5, are the two proteins (4ake, and 2eck) already contained in the PDB dataset?

### Questions
- For all-atom coordinate denoising, how do you determine the number and type of atoms in a residue, when the sequence is masked, or perturbed? Is the structure module optmized only when the clean sequence is given?
- In Section 3.3, there is denoising diffusion loss for coordinates, why is this section named `masked language model'?
- In mode 1, do you jointly optimze the coarse grained structure module and the all-atom structure module?
- In this study, antibodies are generated in a masked-language-model style. Can the model generate longer proteins, like EvoDiff, DPLM?
- Seems that FlexProtein beats AF3 in several tasks (Fig 4, Table 5), is this due to the less usage of information in AF3?

### Soundness
3

### Presentation
3

### Contribution
3
