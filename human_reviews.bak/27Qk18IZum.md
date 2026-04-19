# PharmacoMatch: Efficient 3D Pharmacophore Screening via Neural Subgraph Matching

- Decision: Accept (Poster)
- Scores: 5, 6, 5, 6

## Abstract
The increasing size of screening libraries poses a significant challenge for the development of virtual screening methods for drug discovery, necessitating a re-evaluation of traditional approaches in the era of big data. Although 3D pharmacophore screening remains a prevalent technique, its application to very large datasets is limited by the computational cost associated with matching query pharmacophores to database molecules. In this study, we introduce PharmacoMatch, a novel contrastive learning approach based on neural subgraph matching. Our method reinterprets pharmacophore screening as an approximate subgraph matching problem and enables efficient querying of conformational databases by encoding query-target relationships in the embedding space. We conduct comprehensive investigations of the learned representations and evaluate PharmacoMatch as pre-screening tool in a zero-shot setting. We demonstrate significantly shorter runtimes and comparable performance metrics to existing solutions, providing a promising speed-up for screening very large datasets.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
In this paper, the authors propose a contrastive learning approach for pharmacophore screening, based on subgraph matching. The key idea is to employ approximate subgraph matching for querying conformational database, a main step in pharmacophore screening. The subgraph matching is done through a contrastive learning approach by encoding query-target relationships in the embedding space. Their model has been validated based on benchmark dataset including DUD-E.

### Strengths
It is interesting to reinterpret pharmacophore screening problem as an approximate subgraph matching problem.

### Weaknesses
From the methodology point of view, the paper lacks novelty. The contrastive learning framework and augmentation module are all rather standard approach in GNN models. The contribution is not significant. Further, the performance is not very impressive as shown in Table 1. Even though the authors have emphasized that "our goal is to achieve comparable values between our model and the alignment algorithm", the only advantage of the current model seems to be the runtime efficiency.

### Questions
1) From Methodology point of view, is there any major novelty in the current paper?
2) Other than the efficiency in terms of the runtime, any other clear advantage of the current model?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a contrastive learning approach that emphasizes augmentation strategies by incorporating the concept of pharmacophore in neural subgraph matching. Furthermore, they apply this concept to ligand-based virtual screening, demonstrating that the results are well-aligned with CDPKit’s alignment algorithm. Although the learned representations effectively capture the proposed pharmacophore concepts, the performance in virtual screening—a primary objective of the model—does not appear sufficiently strong. This is primarily due to the lack of benchmarking against other models and the use of additional evaluation metrics.

### Strengths
**S1. Alignment of Contribution and Results:**
	The study presents a coherent alignment between its contributions and the resulting outcomes. The objectives set forth by the authors are consistently addressed throughout the work.

**S2. Effective Representation Learning via contrastive learning:**
	The approach to representation learning through a contrastive learning with data augmentation appears to function as intended. The learned representations for pharmacophores are well-clustered in embedding space.

### Weaknesses
**W1. Limited Methodological Novelty:**
	The proposed methods do not introduce significant novel approaches. As mentioned in the manuscript, the concepts of Neural Subgraph Matching or neural network architectures are already proposed by other previous works. While the formulation applied to pharmacophores—particularly the augmentation strategies for contrastive learning—is noteworthy, it may not sufficiently advance the general methodologies handled in the ICLR community. If there's any other novelty compared to previous works, the points should be clearly explained in the manuscript. Authors may propose novel input processing schemes or model architectures better suited to the pharmacophore graph, or improve neural subgraph matching techniques.

**W2. Insufficient Experimental Results:**

- **W2.1 Lack of Comprehensive Benchmarks:**
	For a study proposing algorithms intended for virtual screening, it is essential to benchmark against established methods such as DrugClip or PharmacoNet to demonstrate comparative advantages. The absence of such comparisons, without a compelling justification, weakens the evaluation of the proposed method’s efficacy. Additionally, reliance on the outdated DUD-E benchmark and the arbitrary selection of only 10 targets out of 102 weakens the robustness of the experimental validation. (see Q4, 5)
	
- **W2.2 Ambiguous goal of benchmark experiment:**
	The goal of "*to achieve comparable values between our model and the alignment algorithm*" would be meaningful only if the alignment method inherently guarantees superior results compared to the previous methods, which is not adequately demonstrated in the current manuscript. (see Q5-1)
	
- **W2.3 Suitability for Virtual Screening:**
	The method appears to focus on ligand interactions with only parts of the protein pharmacophore, potentially neglecting the comprehensive information of the global protein binding site. This partial consideration might introduce bias, and it remains unclear whether the method can outperform existing techniques that utilize complete protein-ligand information. Empirical evidence showing superior performance in this regard would strengthen the study. (see Q6)

### Questions
## **Methods:**
**Q1. Clarification of Pharmacophore Representation:** In the section detailing pharmacophore representation, it is mentioned that $\mathcal{L}$ comprises only pharmacophoric descriptors. However, distances are subsequently incorporated into the representation. The notation and description should be revised for consistency and clarity.

**Q2. Model Input Specification:** Within the model input section, node labels are currently denoted as $V_p$, which is a set of node. It may be more appropriate to represent these as node element to enhance the clarity.

**Q3. Negative Data Augmentation Strategy:**
The current approach limits displacement directions to a single direction to avoid cancellation effects. Allowing for all displacement directions except for the case of cancellation can highly increase the diversity of negative training data, might lead to better model performance. Are there specific reasons why the authors chose to use only a single direction for negative displacements?

## **Results:**

**Q4. Evaluation on Recent Datasets:** To better assess the method’s applicability to real-world scenarios, evaluation on more recent datasets like LIT-PCBA[1] is recommended. Additionally, utilizing metrics beyond AUROC, such as enrichment factors (EF) (similar to BEDROC for early recognition of hit candidates), could provide a better understanding of the model’s performance in virtual screening tasks.

**Q5. Benchmarking with other methods:** Since one of the main contributions of this paper is "*fast virtual screening in the embedding space and evaluate the performance of our method through experiments on virtual screening benchmark datasets*", the authors should benchmark their virtual screening performance with other models. It seems that there are no significant differences in the objectives or methods of the compared models relative to the current work. Are there reasons why the authors did not benchmarked their model with the previous works such as DrugClip or PharmacoNet?

**Q5-1. Ambiguous criteria for showing virtual screening performance:**  The authors compared their performance with CDPKit's alignment algorithm. It only make sense that the propose method does well on virtual screening only if CDPKit alighment algorithm's performance is already good enough. Authors may provide the performance of CDPKit alignment algorithm for virtual screening explicitly in their manuscript.

**Q6: Limitation in Protein-Specific Virtual Screening Capability:**
The methodology seems similar to ligand-based approaches, where ligand structures are pre-generated and graph matching determines activity. This may limit the model's applicability, since considering only ligand structure of protein-ligand complex cannot fully consider protein pocket. For example, if protein has large binding site that various ligands with different pharmacophoric sites can interact with, considering only a single ligand might give a bias during virtual screening on the protein target. Why did authors adopted ligand-based approach instead of protein-based one, such as using protein binding site's pharmacophore instead its binding ligand?

## **References:**
[1] Tran-Nguyen, Viet-Khoa, Célien Jacquemard, and Didier Rognan. "LIT-PCBA: an unbiased data set for machine learning and virtual screening." Journal of chemical information and modeling 60.9 (2020): 4263-4273.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors introduce PharmacoMatch, a deep learning approach that reframes 3D pharmacophore screening as a neural subgraph matching problem, using a graph neural network trained through contrastive learning to encode pharmacophore structures into an embedding space. Their method achieves comparable screening performance to traditional alignment-based approaches while being approximately two orders of magnitude faster in matching speed, making it particularly valuable for screening extremely large molecular databases. The model is trained in a self-supervised manner on over 1.2 million unlabeled molecules from ChEMBL, learns to encode both structural and spatial relationships of pharmacophoric points, and demonstrates robust performance across multiple DUD-E benchmark datasets in a zero-shot setting.

### Strengths
The acceleration is impressive, with a thorough evaluation of embeddings and runtime analysis. The model provides a practical impact for screening billion-compound libraries. Besides, the reformulation of pharmacophore screening as neural subgraph matching is creative combining self-supervised training approach using augmentation strategies.

### Weaknesses
Despite mentioning recent works like DrugClip and PharmacoNet in the related work section, there are no direct comparisons with these methods. It seems the paper only compares against the traditional CDPKit alignment algorithm, missing comparisons with other more current learning approaches. Besides, there is no comparison or detailed discussion with simpler baseline models (e.g., basic GNN architectures without contrastive learning)

The discussion on the model details is not sufficient. Lacks systematic ablation studies of model architecture components (e.g., the impact of different GNN layers, the importance of skip connections). Missing analysis of the impact of different augmentation strategies on model performance and investigation of how embedding dimension and model size affect performance, as well as contrastive loss function impacts.

### Questions
1. How sensitive is the model to the choice of tolerance radius (r_T = 1.5Å)? Could you provide an analysis?
2. How did you select the 10 DUD-E targets? Could you demonstrate the method's robustness on more targets?
3. What is the impact of different augmentation strategies? For example, how much does performance degrade if you remove node deletion or displacement?
4. Could you compare PharmacoMatch with recent methods like DrugClip and PharmacoNet on the same benchmarks?
5. The speed advantage is clear, but can the author better justify the conceptual novelty? (Contrastive learning is not new, GNN for molecules is well-established and order embeddings have been used before)
6. How much 3D geometric precision is actually lost during embedding? Could you quantify the tradeoff between speed and accuracy?
I wonder if there are specific types of 3D arrangements where the method may fail.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel contrastive learning approach based on neural subgraph matching, i.e., PharmacoMatch, and the authors claim that it reinterprets pharmacophore screening as an approximate subgraph matching problem and enables efficient querying of conformational databases by encoding query-target relationships in the embedding space.

### Strengths
+ This paper is well-written and nicely organized. 
+ The proposed framework is novel and is nicely motivated.

### Weaknesses
- I suggested the authors consider conducting more experiments over other datasets instead of only DUD-E.
- I wonder if the proposed contrastive learning approach can be applied to other domain datasets?
- This paper does not provide any unique and novel insights about why the proposed architecture that works well for the current dataset.

### Questions
See comments in the Weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3
