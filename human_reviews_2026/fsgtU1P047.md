# Property-aware Reinforcement Learning with Retrieval Enhancement for Controllable 3D Molecule Generation

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 2

## Abstract
This paper studies the problem of controllable 3D molecule generation, which aims to design 3D molecules that satisfy given conditions. Previous methods usually incorporate the condition tokens into language models, and reconstruct molecules from the generated tokens. Despite their progress, their performance remains unsatisfactory due to the neglect of the condition during the generation process. To address this limitation, we propose a novel approach named Property-aware Reinforcement Learning with Retrieval Enhancement (POETIC) for controllable 3D molecule generation. To be specific, POETIC first tokenizes 3D molecular structures and leverages a language model (LM) for molecular generation. More importantly, it retrieves relevant samples with similar properties from an external database, which are used as prefixes to enhance generation quality. Furthermore, we pre-train a prediction model to identify the molecular properties, which in turn provides property-aware rewards for evaluation. These rewards guide reinforcement learning to optimize the LM. Extensive experiments on benchmark datasets validate the effectiveness of the proposed POETIC in comparison with state-of-the-art approaches. The source code is available at https://anonymous.4open.science/r/POETIC-BEA3.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors introduce POETIC, a framework for controllable 3D molecule generation integrating property-aware reinforcement learning and retrieval-based enhancement. 
Addressing limitations of existing methods, POETIC leverages language models for molecular generation while retrieving similar molecules from external databases to enhance generation quality. 
By pre-training a prediction model for molecular properties, POETIC provides property-aware rewards guiding reinforcement learning for precise controllability and robust generalization.

### Strengths
- This paper addresses an important challenge in controllable molecular generation and is motivated by the need to balance property alignment with generalization, a relevant and timely goal for drug and material design.

- The proposed POETIC framework is well organized, clearly separating retrieval, prefix construction, and reinforcement learning stages. The algorithms are explicitly described and supported by consistent mathematical formulation.

- Although conceptually based on existing ideas, the integration of retrieval-augmented conditioning with property-aware RL is implemented in a coherent and technically rigorous way, leading to gains in controllability and out-of-distribution robustness.

- Experimental results, ablations, and sensitivity analyses demonstrate stable trends and internally consistent improvements, indicating careful engineering and solid experimental execution, though certain evaluation aspects (such as geometry-level validation and fair baseline reproduction) would still benefit from further refinement.

### Weaknesses
- The paper frames the task as “controllable 3D molecule generation,” but the evaluation focuses almost entirely on property controllability (i.e., mean absolute error on QM9 quantum properties). There is no systematic assessment of the generated 3D conformations themselves. For example, geometric plausibility, stereochemistry consistency, bond length / bond angle distributions, structural stability after relaxation, or deviation from physically realistic conformers. Without any geometry-oriented metrics, it is difficult to judge whether the method is truly advancing 3D structure generation, as opposed to mainly optimizing a property predictor in latent space. This weakens the central claim that POETIC improves controllable 3D structure generation.

- The main quantitative table reports substantial gains over prior controllable 3D generation methods such as EDM, GeoLDM, and Geo2Seq. However, several fairness questions remain. For at least some baselines (e.g., EDM), the paper appears to reuse the numbers reported in the original work rather than retraining those models under POETIC’s specific data handling and split (e.g., the paper splits QM9 into separate subsets for training the property predictor vs. training the generator). This mismatch in data usage and training protocol makes it difficult to attribute the reported gains purely to the proposed method, and raises concerns about whether the comparison is strictly apples-to-apples.

- Both optimization and evaluation are mediated by the same frozen property predictor (an EGNN trained on QM9). The RL reward is computed from this model, and the final controllability metrics are also reported using that same predictor. This tight coupling risks overstating controllability: the generator may learn to exploit biases or blind spots of that particular evaluator, rather than truly matching the target physical property. The paper briefly acknowledges this “below-evaluator phenomenon,” but does not provide an external validation step to show that the claimed improvements persist under an unbiased evaluator. As a result, the reported MAE reductions may partially reflect evaluator overfitting rather than genuine chemical alignment.

### Questions
Please see above session.

### Soundness
2

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
4

### Summary
This paper aims to generate 3D molecular structures that satisfy given target physical and chemical properties. Existing 3D diffusion models suffer from high computational costs and limited controllability, while language model-based approaches struggle with property alignment and poor generalization to unseen properties.

To overcome these limitations, the authors propose POETIC (Property-aware Reinforcement Learning with Retrieval Enhancement) that combines retrieval-augmented conditioning, which guides generation using property- and structure-similar exemplars, with reinforcement learning, which leverages a frozen EGNN property predictor and GRPO to align molecules with target properties.
Experiments on the QM9 dataset demonstrate that POETIC outperforms state-of-the-art models such as EDM, GeoLDM, and Geo2Seq across six quantum properties. Furthermore, ablation studies confirm that the RAG and RL modules play complementary roles, contributing jointly to the overall performance improvement.

### Strengths
The paper demonstrates a strong understanding of the current landscape of 3D molecule generation research and effectively addresses a key limitation in the field — the difficulty of achieving both controllability and generalizability simultaneously. By integrating reinforcement learning (RL) to enhance property alignment and retrieval-augmented generation (RAG) to provide contextual guidance, the authors present a well-motivated and technically sound framework.

The inclusion of Figure 1 (Toy experiment result) in the introduction provides a clear and convincing motivation for combining RL and RAG, establishing the validity of the research direction early on. The ablation studies comprehensively verify the contribution of each module, demonstrating strong methodological rigor.

Furthermore, the case study (Figure 4) visually illustrates how POETIC generates molecules with controllable polarizability, offering intuitive evidence of practical effectiveness. The extended studies section goes beyond the main objectives by reporting novelty and validity evaluations, widely recognized metrics in molecular generation, reinforcing the robustness of the results. Finally, the error case analysis thoughtfully discusses the limitations of language model–based generation, showing the authors’ critical reflection on their approach.

### Weaknesses
POETIC presents an interesting approach that combines retrieval-augmented generation (RAG) and reinforcement learning (RL) for controllable 3D molecule generation. However, the paper lacks sufficient explanation and empirical validation in several key aspects.
1. Lack of ablation on RAG and RL components : Although the paper claims that the integration of RAG and RL yields a synergistic effect, baseline models using only RAG or only RL are not included in the experiments. The individual contributions of each component should be disentangled and demonstrated.
2. Limited discussion on molecule retrieval and diversity : In the RAG stage, exemplar molecules are retrieved by selecting the k nearest samples based on the mean of structure embeddings. This simple selection strategy may not simultaneously ensure structural similarity and diversity. The paper does not discuss whether clustering or diversity-aware retrieval techniques were considered. If the retrieved molecules are too similar, there is a risk of loss of generative diversity. Moreover, no quantitative comparison of diversity metrics (e.g., internal diversity, uniqueness, novelty) is provided against existing baselines.
3. Lack of analysis on the frozen EGNN property predictor : The framework employs a frozen EGNN as the property prediction module for reward computation. However, the paper does not analyze how freezing versus fine-tuning this predictor affects reward quality and the overall RL performance.
4. Insufficient discussion on scalability : The proposed framework is evaluated only on QM9, a relatively small and simple dataset. It remains unclear how well the model scales to larger or more complex molecular datasets, which raises concerns about scalability and generalizability.
5. Incomplete implementation details : Several key aspects necessary for reproducibility are missing or under-explained.

•	The structure of generated molecule tokens after the prefix is not clearly specified.

•	The edge (bond) prediction process is not explicitly described.

•	The composition and size of the RAG database are not reported.

•	The conversion process from Mamba outputs to EGNN input graphs is not detailed.

•	No embedding-space trajectory analysis or visualization is provided to show whether the generation follows a meaningful direction in latent space.

### Questions
Please refer to the questions below and the weaknesses section.

Major

1.	To quantitatively demonstrate the synergy between RAG and RL, could you provide results for RAG-only and RL-only baselines?

2.	In the RAG stage, have you considered clustering-based or diversity-aware retrieval methods to better balance structural similarity and diversity?

3.	Have you compared the diversity metrics of the generated molecules (e.g., novelty, uniqueness, internal diversity) against existing baselines?

4.	What is the performance difference between using a frozen EGNN property predictor and a fine-tuned version?

5.	Can the model be scaled to larger datasets beyond QM9 (e.g., PCQM4M, MoleculeNet) to validate its scalability and generalization?

Minor

1.	Could you specify the structure and format of the generated molecule tokens after the prefix?

2.	How are bond (edge) connections predicted and reconstructed in the generation process?

3.	What is the composition and size of the RAG database used for retrieval?

4.	Have you visualized or analyzed the embedding-space trajectory to illustrate how generated molecules evolve toward target properties?

5.	During testing, is only the target property value used as input, or is molecular information also provided?

6.	Could you explain in more detail how the outputs of the Mamba model are converted into EGNN graph inputs?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a RAG+RL framework for controllable 3D molecule generation. The proposed method uses structural information of retrieved molecules as a condition to guide the language model to generate molecules that are more aligned with desired properties.

### Strengths
- The overall idea is to use the aggregated structural information as a hint for the model to perform controllable generation. Although validated imperfectly, such an idea is promising and sound in the controllable molecule generation problem.
- The paper considers in-distribution and out-of-distribution property values. The setting of ood, which is the lowest 10% and highest 10%, is practical and aligned with real-world settings, as chemists often need to maximize/minimize properties instead of controlling them to specific values.
- The use of RAG is a future of molecule generation as it's aligned with chemists' behaviour, that is, leveraging existing database, thinking, and further innovating.

### Weaknesses
- This paper lacks a significant amount of experiment details and settings, which makes it unable to fairly evaluate the proposed method. For example, in the main experiments: 
  - 1) what are the ranges of each property value? 
  - 2) how do you set the sampling distribution for controllable generation at test time (i.e. what are the "desired" property values). 
  - 3) how does your property predictive model perform? These questions are directly related to how 'good' the proposed method is, but never answered in the paper.
- Other unclear things include:
  - why a Mamba model is used, instead a regular transformer model? 
  - how are the six properties picked for QM9 dataset?
  - in the toy experiment in introduction, what is the used dataset and the property?
  - Eq.2: why do you specifically pick ''frequency of different elements'' and ''interatomic distances'' as structural embeddings? Any evidence or preliminary studies?
  - Line 203: how is the normalized element frequencies and the most prominent distance peaks calculated?

- QM9 is a simple and toy dataset even for small molecule generation. I suggest the authors evaluate the framework on more practical datasets, like ChEMBL, and more practical properties like solubility in further revisions.
- Line 83: I think the proposed method just ''combined'', not ''unified'', RL and RAG.
- Example in Fig. 4 does not show clear trend, or does not contain much information.
- Additional qualitative study is needed to better understand the method, such as, examples of the desired properties and properties of actually generated molecules.

### Questions
see above.

### Soundness
1

### Presentation
2

### Contribution
2
