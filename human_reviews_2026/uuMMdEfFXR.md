# MolChord: Structure–Sequence Alignment for Protein-Guided Drug Design

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Structure-based drug design (SBDD), which maps target proteins to candidate molecular ligands, is a fundamental task in drug discovery. Effectively aligning protein structural representations with molecular representations, and ensuring alignment between generated drugs and their pharmacological properties, remains a critical challenge. To address these challenges, we propose MolChord, which integrates two key techniques: (1) to align protein and molecule structures with their textual descriptions and sequential representations (e.g., FASTA for proteins and SMILES for molecules), we leverage NatureLM, an autoregressive model unifying text, small molecules, and proteins, as the molecule generator, alongside a diffusion-based structure encoder; and (2) to guide molecules toward desired properties, we curate a property-aware dataset by integrating preference data and refine the alignment process using Direct Preference Optimization (DPO). Experimental results on CrossDocked2020 demonstrate that our approach achieves state-of-the-art performance on key evaluation metrics, highlighting its potential as a practical tool for SBDD.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes **MOLCHORD**, a large-scale framework for structure-based drug design that aligns protein structures with molecular sequences. The method couples a diffusion-based structure encoder (trained on proteins, molecules, and complexes) with a NatureLM-based sequence generator capable of producing SMILES, FASTA, and text. Training proceeds in three stages: (i) alignment pretraining on multimodal structure-to-sequence tasks, (ii) supervised finetuning on protein pocket–ligand complexes, and (iii) reinforcement learning with Direct Preference Optimization (DPO) using docking-based and drug-likeness preference signals. Experiments on the CrossDocked2020 dataset show that MOLCHORD achieves state-of-the-art performance across affinity, drug-likeness, synthesizability, and success rate, while maintaining reasonable diversity. The framework demonstrates improved generalization to unseen proteins, and ablation studies highlight the importance of both the diffusion encoder and the DPO stage. Overall, MOLCHORD advances protein-guided molecular generation by bridging protein structures, sequences, and ligand design within a unified multimodal foundation model.

### Strengths
1. **Significant research topic**: The paper tackles a highly impactful problem in structure-based drug design (SBDD), where aligning protein structures and sequences with molecular generation remains a central challenge. Given the immense search space of molecules and the importance of accurate protein–ligand modeling, this is a timely and valuable direction with broad relevance to both the ML and drug discovery communities.
2. **Interesting method**: The proposed MOLCHORD framework is novel in combining a diffusion-based structure encoder with a large autoregressive language model (NatureLM variant), aligned via multimodal pretraining and refined with Direct Preference Optimization (DPO). This design enables flexible integration of protein sequence (FASTA), structure, and pocket-level information with molecular generation, representing an innovative multimodal approach that goes beyond pocket-only baselines.
3. **Good writing and presentation**: The paper is well-structured and clearly written, with a logical flow from motivation to methodology and experiments. The staged training process is explained systematically, and the results are presented with comprehensive baselines and ablations, making the contributions accessible and convincing to the reader.

### Weaknesses
1. **Limitation to 1D/2D design**: The method only generates ligands as SMILES strings, without atomic 3D coordinates, meaning it is not a true 3D molecular generator. Therefore, it should also be compared with strong 1D/2D sequence-based baselines such as Reinvent 4 [1], which have been reported to outperform 3D approaches in SBDD [2]. Furthermore, previous 3D baselines (e.g., DecompDiff, MolCRAFT) reported Vina scores on raw generated structures without redocking, whereas MOLCHORD relies only on redocked poses, which may make direct comparisons less straightforward.
2. **Rigid protein assumption**: MOLCHORD assumes proteins are rigid and does not model flexibility or conformational dynamics. Given the importance of induced fit and protein motion in real binding processes, this limitation should be explicitly discussed.
3. **Unclear contribution of protein sequence/global structure**: Although the authors include FASTA and whole-protein features in pretraining and run some prompt-based ablations, there is no quantitative ablation study that isolates how much these inputs improve ligand generation compared to pocket-only conditioning. This weakens the justification for incorporating global protein information.
4. **Unspecified SMILES representation**: The paper does not clarify whether canonical or randomized SMILES are used during training and evaluation. Since SMILES augmentation is widely known to affect generalization and molecular diversity, the choice should be specified and justified.

[1] Reinvent 4: Modern AI–driven generative molecule design.

[2] Structure-based Drug Design Benchmark: Do 3D Methods Really Dominate?

### Questions
1. **Choice of DPO vs. direct reward optimization**: Why does the RL stage adopt Direct Preference Optimization (DPO) rather than directly using the original docking and property scores as scalar rewards, especially since those same scores are used for evaluation? Doesn’t the preference formulation risk losing information about score magnitudes?
2. **Role of whole protein structure**: Since the binding pocket is treated as static during ligand design, what is the benefit of incorporating the whole protein structure? How can atoms outside the pocket meaningfully influence the generation process?
3. **Reward design for fused rings**: Why is the number of fused rings explicitly included in the RL reward function, rather than directly using scores of QED and SA?
4. **RL training dynamics**: Could the authors provide a reward curve or training trajectory during the RL process to illustrate stability, convergence, and the trade-off between affinity, diversity, and drug-like properties?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes MOLCHORD to align protein structural representations with molecular representations in SBDD. First, MOLCHORD align protein and molecule structures with their textual descriptions and sequential representations using an autoregressive model called NatureLM. Second, the model is able to guide molecules toward desired properties. To achieve this, this work curates a property-aware dataset by integrating preference data and refine the alignment process using DPO.

### Strengths
- MOLCHORD is a unified framework aligning protein, molecule, and text representations
in target-aware molecular design.
- A property-aware dataset is curated for properties guidance and this work uses Direct Preference Optimization (DPO) to refine alignment.

### Weaknesses
- Some important SBDD methods are not discussed in this paper, such as [1][2][3]
- It’s better to also report Lipinski metric in Table 1. Also, generation efficiency is also an important factor when evaluating the practically usefulness of the model.
- For the docking metric, it’s better to also report the docking score before performing re-docking to directly evaluate the docking performance of generated molecules.
- CrossDocked2020 is a synthetic dataset and has limited quality for training and evaluating SBDD model. This was also discussed a lot within the community. Could authors elaborate more on the usage of this dataset?


[1] Zhang, Zaixi, and Qi Liu. "Learning subpocket prototypes for generalizable structure-based drug design." ICML\
[2] Zhang, Zaixi, et al. "Molecule generation for target protein binding with structural motifs." ICLR\
[3] Fu, Cong, et al. "Fragment and geometry aware tokenization of molecules for structure-based drug design using language models." ICLR

### Questions
Please refer to the weakness part

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
2

### Summary
MOLCHORD is a large multimodal model for structure-based drug design (SBDD), aiming to link protein structures with suitable small molecules. In SBDD, finding ligands that bind well to a target protein is essential but difficult because protein 3D structures and molecular properties are hard to align. MOLCHORD solves this by combining a diffusion-based structure encoder that understands 3D geometry with NatureLM, a large autoregressive language model that unifies text, protein sequences (FASTA), and molecular strings (SMILES). Through step-by-step alignment across proteins, molecules, and complexes, followed by supervised fine-tuning and Direct Preference Optimization (DPO), MOLCHORD learns to generate compounds that are drug-like, synthesizable, and high-affinity. Experiments on the CrossDocked2020 dataset show state-of-the-art performance in affinity, QED, synthetic accessibility, and diversity, demonstrating that MOLCHORD is an effective and scalable tool for modern drug discovery.

### Strengths
* It presents unified multimodal architecture aligning 3D structures with textual and sequential data (FASTA, SMILES).
* The method integrates DPO-based optimization for controllable molecule generation balancing affinity, drug-likeness, and synthesis feasibility.
* The proposed method outperforms diffusion and graph baselines on CrossDocked2020 while generating realistic, FDA-like molecules.
* It demonstrates computational efficiency suitable for high-throughput virtual screening.

### Weaknesses
* Docking-based reward still approximates true binding and ignores ADMET constraints.
* Reproducibility cannot be assessed due to the lack of released code.

### Questions
* Is there any plan to release the code, model, or dataset?

### Soundness
3

### Presentation
2

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
This paper proposes MOLCHORD, a 4-billion-parameter framework for structure-based drug design (SBDD) that addresses two core challenges in existing methods: poor alignment between protein structural representations and molecular sequence representations, and misalignment between generated molecules and desired pharmacological properties. 
MOLCHORD integrates a diffusion-based structure encoder (for capturing 3D geometric features of proteins/molecules) and an autoregressive sequence generator (a variant of NatureLM, for generating SMILES/FASTA/text), with alignment facilitated by a lightweight adapter. It adopts a three-stage training strategy: (1) Adapter pre-training for cross-modal (structure-sequence-text) alignment; (2) Supervised fine-tuning (SFT) on protein-ligand complexes; (3) Direct Preference Optimization (DPO) to refine property alignment. 

Experimental results on CrossDocked2020 show that MOLCHORD achieves state-of-the-art (SOTA) performance across key metrics (binding affinity, drug-likeness, synthesizability, diversity) and demonstrates robust out-of-distribution (OOD) generalization.

### Strengths
1. Unlike existing methods that rely solely on limited protein-ligand pairs for alignment, MOLCHORD leverages multi-task pre-training (protein-to-FASTA, molecule-to-SMILES, complex-to-text) to unify structural, sequential, and textual representations. This design effectively mitigates the data scarcity of high-quality protein-ligand pairs and enables more robust cross-modal interaction— a key innovation that addresses a long-standing bottleneck in SBDD.

2. The paper curates a property-aware dataset for DPO and introduces a reward function that balances binding affinity (Vina score) with synthetic accessibility (SA) and drug-likeness (via fused ring penalty). This avoids the common trade-off in prior RL-based methods (e.g., BindGPT, MolForm) where affinity is improved at the cost of diversity or drug-likeness. 

3. The experimental design is thorough.

### Weaknesses
1. The structure encoder is pre-trained on 78M protein structures from AlphaFoldDB/PDB, and the generator on NatureLM’s corpus—but critical pre-training details are missing: How were the 78M protein structures filtered (e.g., sequence identity thresholds, resolution constraints)? Low-quality structures could bias encoder learning. Full hyperparameters for pre-training (e.g., learning rate decay schedule, warm-up steps, batch size adjustment) are not provided, making it impossible for small labs to reproduce the 4.2B-parameter model.

### Questions
1. During the DPO dataset curation, the fused ring penalty coefficient ($\lambda$=0.5) is chosen without justification. A sensitivity analysis ($\lambda$=0.2/0.5/0.8) would validate whether this choice is optimal or arbitrary.

### Soundness
3

### Presentation
3

### Contribution
3
