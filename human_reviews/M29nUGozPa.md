# SMI-Editor: Edit-based SMILES Language Model with Fragment-level Supervision

- Decision: Accept (Poster)
- Scores: 3, 6, 8, 8, 5

## Abstract
SMILES, a crucial textual representation of molecular structures, has garnered significant attention as a foundation for pre-trained language models (LMs). However, most existing pre-trained SMILES LMs focus solely on the single-token level supervision during pre-training, failing to fully leverage the substructural information of molecules. This limitation makes the pre-training task overly simplistic, preventing the models from capturing richer molecular semantic information. Moreover, during pre-training, these SMILES LMs only process corrupted SMILES inputs, never encountering any valid SMILES, which leads to a train-inference mismatch. To address these challenges, we propose SMI-Editor, a novel edit-based pre-trained SMILES LM. SMI-Editor disrupts substructures within a molecule at random and feeds the resulting SMILES back into the model, which then attempts to restore the original SMILES through an editing process. This approach not only introduces fragment-level training signals, but also enables the use of valid SMILES as inputs, allowing the model to learn how to reconstruct complete molecules from these incomplete structures. As a result, the model demonstrates improved scalability and an enhanced ability to capture fragment-level molecular information. Experimental results show that SMI-Editor achieves state-of-the-art performance across multiple downstream molecular tasks, and even outperforming several 3D molecular representation models.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
In this paper, the authors propose SMILES-Editor, an edit-based pre-trained SMILES language model by introducing a new pre-training strategy that randomly corrupts SMILES strings during the pre-training process and lets the LLM restore the original SMILES strings. Experimental results show that the new proposed pre-training strategy achieves better performance than the previous MLM task in various downstream tasks.

### Strengths
1. The idea of focusing on the atoms and fragments is meaningful and enables better learning of SMILES representations.
2. The new pre-training task may lead to an improvement in the valid generation of molecule SMILES representations.
3. The SMILES-Editor shows competitive performance compared to the original MLM, bringing an improvement in the pre-training of SMILES Language Models.

### Weaknesses
1. The authors claim that "most existing pre-trained SMILES language models (LMs) only provide supervision at the single-token level during pre-training and fail to fully leverage substructural information of molecules", which lacks enough support. In works like  [1], the tokenizers are initialized with SMILES Pair Encoding. Meanwhile, the sentence-piece method can also find common fragments in molecule SMILES strings, and these common fragments are usually critical functional groups. In this case, I do not agree with this claim.
2. The novelty of SMILES-Editor is limited. The main contribution of this paper is to propose a new pre-training strategy, while this strategy is not novel, and there is not a significant difference compared to the previous Masked Span Language Modeling used by T5. Furthermore, more advanced LLMs are now using decoder-only structures, while SMILES-Editor can only be adopted in LLMs with encoder structures, which further harms the novelty.
3. The experiments on the classification tasks do not incorporate the k-fold experiments, which raises concerns about the stability of the method.
4. The experiment results do not seem satisfying enough. And the comparison should include more pre-training strategies.
5. In Table 2, the performance of SMI-EDITOR-AtomsMasking in BACE is even better than SMILES-Editor.

#### References
[1] Li, X., & Fourches, D. (2021). SMILES pair encoding: a data-driven substructure tokenization algorithm for deep learning. Journal of chemical information and modeling, 61(4), 1560-1569.

### Questions
1. Could the authors find more evidence to support their claims in Weakness 1?
2. Could the authors explain more about Weakness 2?
3. I am wondering about the comparison between different pre-training strategies. Could the authors compare the auto-regressive pre-training strategies in identical decoder-only structures?
4. What is the time cost or complexity of SMILES-Editor compared to the previous MLM?

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
The paper introduces SMI-Editor, a novel edit-based SMILES language model with fragment-level supervision to improve molecular representation learning. SMI-Editor tries to address the limitations of masked language models (MLMs), such as rapid saturation and limited substructure semantics modeling, by using an LevT-based modeling.

### Strengths
- The analysis identifies interesting issues: the MLM model struggles to distinguish between random deletion and hydrophilic deletion, and it quickly saturates on single-token masking.
- The paper writing is clear and well-illustrated.

### Weaknesses
- LevT isn’t actually an MLM model; it’s a non-autoregressive generative model. From this perspective, the authors' solution for handling corrupted SMILES input is somewhat trivial, as other sequence-based generative models could also address this.
- There is a discrepancy between the motivation and solution. Neither corrupted SMILES nor fragment-level supervision is directly related to the way LevT models molecules.
- Limited effectiveness. While I understand the authors' choice of the MoleculeNet baseline without considering powerful graph-based models like Uni-Mol, DVMP_{TF} in Jinhua Zhu et al. is a SMILES-based Transformer encoder and appears to perform much better than the reported results.
- Figure 5 is confusing; if the model can distinguish hydrophilic groups, wouldn’t performance degrade noticeably after HG deletion?

### Questions
- Why not include the performance on FreeSolv and ESOL?
- What about applying MLM on fragment spans? This could also help alleviate the saturation problem in single-token prediction.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes an edit-based pre-trained SMILES language model to learn 3D molecular representations and applies the model to several downstream tasks, such as molecular property prediction. The idea of using token edit operations to replace token masking introduces a novel approach with the potential to address the rapid saturation problem. The paper provides comprehensive details on model implementation and includes extensive ablation studies.

### Strengths
Novelty: The use of token edit operations instead of traditional token masking is innovative and could offer significant advantages in terms of model performance and stability.
Comprehensive Implementation: The paper provides detailed descriptions of the model architecture and training procedures.
Extensive Ablation Studies: The authors have conducted thorough ablation studies to validate the effectiveness of their approach.

### Weaknesses
1. Figure 4: The requirement for an expert to provide training signals is inefficient and limits the size of the training set. This could be a significant drawback in practical applications.
2. Figure 5: Figures 3 and 5 should be presented together for easier comparison. Additionally, the figure's purpose is not clearly explained.
3. Figure 6: The figure's ability to demonstrate scalability is questionable. It would be helpful to show whether larger models yield better results to substantiate claims of scalability.
4. Conclusion: The statement, "ablation studies confirm the advantages of its design over traditional MLMs in modeling molecular substructure semantics and training stability," is not fully supported by the evidence presented in Figure 5. The figure does not convincingly demonstrate that the model understands molecular substructure semantics.

### Questions
1. Figure 3: The purpose of this figure is unclear. It would be beneficial to include a comparison with other methods to highlight the advantages of the proposed approach.
2. Table 1: It is important to specify the amount of data used for fine-tuning to better understand the model's performance.
3. Generative Tasks: It would be interesting to explore whether this model can be applied to generative tasks, which could broaden its applicability.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper first investigates the problem of MLM training, as train-inference mismatch. To address this, the authors propose the edit-based pre-trained language model, SMI-Editor, for SMILES. As another pre-training strategy, SMI-Editor drops the substructures in a valid SMILES sequence and tries to recover the original molecules.. The extensive experiments demonstrate the effectiveness of SMI-Editor.

### Strengths
1. The model is well-motivated to solve the train-inference mismatch problem in existing MLM paradigm.

2. Extensive experiments indicate the effectiveness of SMI-Editor.

3. The analysis of the rapid saturation problem of MLM and how SMI-Editor can solve this problem is generally convincing.

4. The paper is well-written and easy to follow.

### Weaknesses
1. The authors put a lot efforts to compared SMI-Editor with MLM paradigm. However, there are still other strategies for pre-training the model, for example, contrastive learning. Can the authors also discuss the relations between SMI-Editor and other pre-training paradigms?

2. As the authors state, increasing the mask ratios will influence the convergence in MLM. Would fragment drop ratio also influences the SMI-Editor?

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper identifies three key challenges in existing SMILES masked language models: the neglect of substructural information, overly simplistic training tasks, and a mismatch between training and inference procedures and introduce a SMILES language model employing edit-based, fragment-level supervision to address these challenges.

### Strengths
1.	This paper reveals the shortcomings of existing SMILES masked language models through an analysis of experimental results, thereby informing future research directions.
2.	This paper presents a novel SMILES language model that uses edit-based, fragment-level supervision. This approach improves performance on the molecule property prediction task.

### Weaknesses
1.	Edit-based models, as currently designed, focus solely on restoring removed substructures during pre-training. Other valuable sources of molecular information, such as correcting errors and removing extraneous components, may be neglected.
2.	The data processing and pre-training stages lack clarity. A detailed case study would significantly improve understanding of these processes.
3.	The description of the experimental tasks is inconsistent between the caption of Table 1 and the " 5.2 RESULTS ON MOLECULAR PROPERTY PREDICTION" section.

### Questions
1.	How to ensure that left-over fragments can be assembled into a valid molecule? Please provide an exact example.

### Soundness
2

### Presentation
2

### Contribution
3
