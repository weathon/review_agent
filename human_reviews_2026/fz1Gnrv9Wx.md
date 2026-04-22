# Drug-ProGO: A Gene Ontology-Enhanced Contrastive Learning Framework for Drug Virtual Screening with Multi-modality Whole-Protein Input

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Virtual screening plays a crucial role in accelerating early-stage drug discovery by efficiently identifying promising small molecule candidates. While most existing methods depend on known binding pockets, protein-level virtual screening has recently gained attention due to its broader applicability in scenarios where pocket information is incomplete or unavailable. In this work, we propose Drug-ProGO, a Gene Ontology (GO) enhanced contrastive learning framework that integrates GO information during training to enrich protein representations, enabling the model to generalize better to unseen proteins by capturing their functional similarity to known ones. This enables the model to better infer compatibility between novel proteins and small molecules. Our framework supports flexible protein inputs, including sequence, structure, and their combination. In the dual-modality setting, the two modalities are processed independently, and their prediction scores are fused using an uncertainty-aware fusion mechanism without additional trainable parameters. Extensive experiments across four virtual screening benchmarks and input settings demonstrate that incorporating GO knowledge consistently improves performance, highlighting the importance of functional knowledge integration for protein-level virtual screening.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a new method for dense retrieval basd virtual screening called Drug-ProGO. Building upon prior work including DrugCLIP and LigUnity, this work introduce a new Gene Ontology (GO) enhancement module designed to enrich protein representations by incorporating functional biological knowledge. This module is integrated into a contrastive learning framework to improve the alignment between protein and ligand embeddings. The overall training strategy, datasets, and encoder backbones largely follow those used in previous studies, but the addition of GO-based supervision aims to boost generalization to unseen proteins. Empirical results show that Drug-ProGO achieves competitive or improved performance compared to baselines across multiple virtual screening benchmarks.

### Strengths
1. The use of Gene Ontology information to enhance protein representations for virtual screening is a reasonable and meaningful idea, especially for improving generalization to unseen proteins.
2. The figures in the paper are well-designed and visually clear, helping to convey the framework and key concepts effectively.
3.  Ablation studies demonstrate that incorporating GO information does lead to measurable improvements in performance.

### Weaknesses
1. The improvements of Drug-ProGO over previous methods such as LigUnity are quite limited, especially on the three datasets other than DUD-E. Even in the sequence only setting, the performance gains of Drug-ProGO compared to LigUnity are marginal.
2. Drug-ProGO is primarily an incremental work built upon existing approaches. The main novelty lies in the integration of Gene Ontology information. While this idea is reasonable and does lead to some improvements, the overall technical novelty and contribution of the work remain limited.
3. There should be a more thorough analysis of how GO contributes to the performance improvement. For example, the visualization in Figure 3 shows better embeddings for some targets, but it does not clearly illustrate how these improvements are connected to the GO information.
4. The comparison on unseen proteins in Table 5 is a good addition, but further analysis would be valuable, such as stratifying the evaluation based on sequence identity or pocket similarity to better understand generalization across proteins with varying degrees of similarity.

### Questions
see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method that integrates Gene Ontology (GO) knowledge into contrastive learning for drug screening without using binding pocket data. The framework combines sequence and structure protein encoders with a molecular encoder and a GO-based contrastive module that links proteins and GO terms. It uses multi-modal inputs, contrastive and ranking losses, and an uncertainty-aware fusion of sequence and structure predictions to improve protein–ligand matching and generalization.

### Strengths
The paper is clearly written, and incorporating GO information is a novel attempt. It introduces additional information, and the experiments are relatively thorough.

### Weaknesses
The main issue is that the contribution of this paper is relatively incremental, as it mainly adds a new source of information on top of the previous LigUnity framework to align protein representations. How the GO data improves the model’s performance in virtual screening, and what specific information it introduces, deserve further analysis and discussion.

### Questions
1.Regarding the score fusion, have you explored other fusion mechanisms? For example, introducing some additional learnable parameters.

2. Would opening the weights of the pre-trained model for full fine-tuning lead to better results?

3.The use of SaProt in this paper differs from structure encoder in LigUnity. Compared with pocket-based approaches, what advantages does it offer?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Drug-ProGO, a novel virtual screening framework that enhances protein representations using Gene Ontology (GO) information within a contrastive learning framework. The method supports multi-modality protein inputs (sequence, structure, or both) and demonstrates sota performance across multiple benchmarks.

### Strengths
1. The incorporation of Gene Ontology (GO) terms into the contrastive learning framework is a novel approach that effectively enriches protein representations. This integration allows the model to capture functional similarities between proteins, which is crucial for generalizing to unseen proteins and improving virtual screening performance.

2. Drug-ProGO supports flexible protein inputs, including sequence, structure, or both. This flexibility is particularly valuable in scenarios where only sequence information is available or when binding pocket structures are unknown.

3. The method achieves state-of-the-art results on multiple virtual screening benchmarks, including DUD-E, LIT-PCBA, DEKOIS 2.0, and AD.

### Weaknesses
1. The method relies heavily on pre-trained models (e.g., ESM2 for sequence encoding and SaProt for structure encoding).  The freezing of most layers during fine-tuning might limit the model's ability to fully adapt to the specific virtual screening task.

2. The method also mentions that contrastive learning can fully utilize paired small molecule and protein data without the need for affinity labels. However, the ranking component of the method itself leverages these labels. This might suggest to show that for a large dataset, if using a ranking loss for the labeled portion while not using it for the unlabeled portion could potentially perform better than not using the ranking loss at all.

3. The method emphasizes that the sequence can be used as input when protein structures are unavailable. However, I think protein structures are not really an issue (thanks to AlphaFold). So, I wonder what would happen to the performance if we used the protein structures decoded by AlphaFold as input instead?

### Questions
Refer to the weakness part.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents Drug-ProGO, a protein-level virtual screening framework that integrates Gene Ontology information into a contrastive learning setup to improve generalization to unseen proteins. By leveraging functional annotations during training, the model learns representations that capture functional similarity among proteins, enabling more accurate prediction of compatible small molecules. The framework supports flexible protein inputs, including sequence, structure, and their combination, and employs an uncertainty-aware fusion mechanism to combine predictions from different modalities. Experiments on four virtual screening benchmarks demonstrate that incorporating GO knowledge consistently enhances performance, particularly for novel proteins lacking binding pocket information.

### Strengths
- Effective use of GO information to provide functional supervision and improve model generalization.  
- Flexible design supporting multiple input modalities (sequence, structure, or both).  
- Uncertainty-aware fusion mechanism offers a principled way to combine predictions without additional training.  
- Strong experimental validation across multiple benchmarks with consistent gains in generalization.

### Weaknesses
- Figure 2 is not rendered correctly.  
- In the experiments, the authors remove protein sequences identical to those in the training set and sequences with high similarity. It would further strengthen the robustness of the evaluation to also exclude proteins that are close in the GO graph.  
- Minor: The double helices in Figure 2 resemble DNA strands rather than protein structures.

### Questions
- In Figure 3, it appears that GO information primarily benefits sequence-only cases. Would incorporating AlphaFold-predicted structures provide additional improvements compared to GO features?  
- Can GO information also be utilized during inference? This could offer a significant advantage for real-world virtual screening, where drug targets are typically well-defined within human biological pathways.

### Soundness
3

### Presentation
3

### Contribution
3
