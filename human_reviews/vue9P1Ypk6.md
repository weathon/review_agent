# MAGE: Model-Level Graph Neural Networks Explanations via Motif-based Graph Generation

- Decision: Accept (Poster)
- Scores: 6, 8, 5, 6

## Abstract
Graph Neural Networks (GNNs) have shown remarkable success in molecular tasks, yet their interpretability remains challenging. Traditional model-level explanation methods like XGNN and GNNInterpreter often fail to identify valid substructures like rings, leading to questionable interpretability. This limitation stems from XGNN's atom-by-atom approach and GNNInterpreter's reliance on average graph embeddings, which overlook the essential structural elements crucial for molecules. To address these gaps, we introduce an innovative **M**otif-b**A**sed **G**NN **E**xplainer (MAGE) that uses motifs as fundamental units for generating explanations. Our approach begins with extracting potential motifs through a motif decomposition technique. Then, we utilize an attention-based learning method to identify class-specific motifs. Finally, we employ a motif-based graph generator for each class to create molecular graph explanations based on these class-specific motifs. This novel method not only incorporates critical substructures into the explanations but also guarantees their validity, yielding results that are human-understandable. Our proposed method's effectiveness is demonstrated through quantitative and qualitative assessments conducted on six real-world molecular datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces MAGE (Motif-bAsed GNN Explainer), which improves interpretability by using motifs as core units in explanations. MAGE identifies class-specific motifs through decomposition and attention-based learning, creating clearer molecular graph explanations. This approach, validated on six datasets, provides more human-understandable results. However, there are some issues in this article that need clarification.

### Strengths
1. This paper introduces a new method, MAGE, to generate motif-based graph explanations.
2. This paper is well-organized, it is easy to follow the main idea.
3. This paper conducts lots of experiments to verify the effectiveness of this method.

### Weaknesses
1. Some symbols are confusing, such as $\mathcal{L}_T$,$\mathcal{L}_L$. It seems a trivial solution exists that the $G$ is the same as the $\mathcal{T}$ from the loss function.
2. The figures are not well expressed. For example, in Figure 3, there are two graph decoders. However, from the paper, I can only find one graph decoder.
3. It is confusing how to construct a new graph from different subgraphs. It's better to give the algorithm instructions on how to construct a new graph from subgraphs and how they share nodes.

Some minor suggestions. 
1. Notations are inconsistent.  $\mathbf{A}$ and $A$ are used interchangablelly. 
2.  It is confusing to use mathbf{} to denote matrix (A) and set (V)
3.  $N$ is used with different meanings. "N represents the total number of atoms in the graph" & "Given a dataset, denoted as G with N molecules and C classes"
4.  Line 271, z_T is not defined.
5. Format issue, cite should be \citep{}

### Questions
1. How to a construct graph from subgraphs?  Do the subgraphs share some nodes?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work proposes a novel method for generating model-level explanations by decomposing molecules into motifs and employing tree-constrained generators. To address issues of invalidity resulting from disregarding structural information, the method decomposes molecules into motif sets and uses attention-based motif identification to select key motifs for each class. A tree-constrained encoder-decoder generator and a specific loss function are introduced to ensure that the generated molecules conform to the class distribution, enhancing their validity. Experiments on six real-world datasets demonstrate that the proposed method outperforms the baselines in both effectiveness and efficiency. Qualitative results further highlight the importance of incorporating both node features and molecular structures in explanations.

### Strengths
1. This work clearly defines the problems faced by current model-level explanation methods and proposes a novel motif-based method to address them. This work conducts extensive experiments and analysis to support the claims, which are very solid and lay a good foundation for future studies in model-level molecular explanation.

2. By treating functional groups as building blocks, this method ensures that the explanations align more closely with scientifically meaningful interpretations. By considering both node features and molecular structures, the paper effectively addresses the limitations of atom-based methods.

3. The paper presents a novel approach that introduces an attention-based learning method to calculate the motif-class relationship. 

4. The paper proposes using tree-constrained generators to produce more valid explanations. The carefully and explicitly designed tree decomposition and encoder-decoder structures ensure that the model generates more valid in-class molecules.

5. The evaluation metrics used to assess explanation performance in relation to molecules are better aligned with the chemical domain. The provided explanations are user-friendly and easy to understand.

6. The authors conduct experiments on six real-world datasets, and the results clearly demonstrate that the proposed method outperforms the baselines.

7. Comprehensive experimental settings are provided to ensure the quality and reproducibility of the results, and sufficient visualizations support the findings of the work.

### Weaknesses
1. There is a typo in lines 95 and 96 : there should be full stops at the end of the sentence. 

2. In line 104, “adjacency” should be revised to “adjacency matrix.”

3. In line 168, “three methods” should be changed to “four methods.”

4. It would be beneficial to clearly state the limitations of current approaches and provide insights on how to address these limitations to better facilitate the development of this area within the community. 

5. Additionally, summarizing and highlighting the main contributions and novelties of the proposed work would enhance clarity.

6. Furthermore, it would improve the presentation to arrange the notation used in the paper more effectively.

### Questions
1) Why is the bond feature defined as $N \times D_b$? The number of edges in a graph may not correspond to the number of nodes. How does this definition account for that discrepancy?

2) What distinguishes tree-constrained methods from non-constrained methods in molecular generation?

3) In the definition $T = (A_{\tau}, X_{\tau} )$, what does $A_{\tau}$  and $X_{\tau}$ represent? Additionally, what does $Z_{\tau}$ signify? 

4) How is  $Z_{\tau}$ sampled from the graph encoder? What is the process involved?

5) What does  $f^a$ refer to in Section 3.3.5? 

6) Why does XGNN experience out-of-memory (OOM) issues? Can the authors provide intuitive explanations based on experimental observations?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces MAGE, a motif-based approach to explain GNNs, specifically focused on molecular datasets. The approach addresses limitations in previous GNN explanation methods by employing molecular substructures—as foundational motifs in model explanations. MAGE utilizes a combination of motif extraction, attention-based motif learning, and a motif-based graph generation method to yield structurally valid and interpretable explanations at the model level. Experimental results on six molecular datasets show that MAGE achieves high validity and interpretability, outperforming baselines in providing explanations that are more representative.

### Strengths
- This paper is about a highly relevant topic in ML interpretability, particularly in the context of GNNs for molecular data. With the growing importance of explainable AI in sensitive domains, especially AI4Science domains like drug discovery and materials science, providing valid, interpretable explanations is well-aligned with current research trends and practical needs.

- The use of motifs as the basis for explanations addresses the limitations of atom-level generation methods.

- The paper is clear, especially regarding the MAGE’s workflow, from motif extraction to class-wise motif generation.

### Weaknesses
- The practical contribution of model-level GNN explanation, and its pros and cons compared to relevant works, say Q1 & Q2 below.

- No contribution to the computationally intensive and challenging motif extraction task.

- No human evaluation. Although the paper claims that the generated motifs and explanations are human-understandable, this claim is not supported by any human evaluation.

### Questions
1.	The paper emphasizes the practicality of model-level explanations over instance-level ones, yet in many real-world applications, users often seek to understand individual predictions. Could the authors elaborate on why a model-level approach would be more practical in such contexts and how it aligns with the needs of end-users who focus on instance-level interpretability? 

2. How does this paper differentiate itself from MotifExplainer (Yu & Gao 2022), except for the "model-level" vs. "instance-level" difference that I am not convinced to believe is significant? MotifExplainer also utilizes motifs in GNN explanations. Can the authors clarify any unique aspects of MAGE, such as scope, or improvements in interpretability, validity, or computational efficiency?
	
3. MAGE’s approach begins by identifying all possible motifs in the dataset, which is a computationally intensive and challenging task, especially for large graphs or datasets due to the combinatorial explosion of possible motifs. However, MAGE does not contribute to addressing this fundamental challenge, as it relies on existing motif extraction methods without proposing any improvements to make motif identification more scalable. Addressing this limitation is more crucial to me, as the scalability of motif extraction represents a significant bottleneck for applying MAGE to larger datasets. Can the authors elaborate on the computational complexity of motif extraction and its impact on scalability to larger datasets?

5. Including a human evaluation would strengthen the claim by demonstrating that experts or users in the field find these explanations interpretable and meaningful. Any human evaluation to provide insights into the practical interpretability and further validate the effectiveness of MAGE in real-world applications?

6. Would MAGE be adaptable to non-molecular datasets, or are there constraints due to the specific nature of molecular motifs?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes MAGE, a motif-based explanation method for GNNs in molecular tasks, which identifies significant motifs for each class using an attention-based learning approach. The method creates model-level explanations including critical molecular structures to the predictions. Experimental results show that MAGE provides valid, human-understandable explanations, outperforming SOTA baseline methods.

### Strengths
1) The paper is well-written, and its structure is easy to follow.
2) The central idea is clear and effectively delivered.
3) SOTA methods for model-level explainability are provided and compared in the experiments, enhancing the paper's credibility and impact.
4) The method generates valid molecules, which is not well-studied and essentially missing in the literature of GNN explainability. So I find this direction of research critical.

### Weaknesses
1) The code is not shared, which reduces the paper's reliability, especially given the extensive experiments presented.
2) The paper focuses only on model-level explainability baselines; however, local explainers (especially inductive ones) could potentially be adapted to the authors' chosen metric. Some local explainer baselines can be found in benchmarking study at ICLR 2024 [1].
3) The loss function comprises two main components, but the effectiveness of each part is not analyzed.
4) The examples from qualitative study is not well explained and confusing. It is hard to understand how are the examples selected.

Small: 
- Line 244, typo: Figure 3.3.

[1] Kosan, M., Verma, S., Armgaan, B., Pahwa, K., Singh, A., Medya, S., & Ranu, S. GNNX-BENCH: Unravelling the Utility of Perturbation-based GNN Explainers through In-depth Benchmarking. In The Twelfth International Conference on Learning Representations.

### Questions
I do not have a major concern about the novelty of the paper. However, I have small but critical concerns about the paper's reproducibility and some experiment results. I'm willing to increase my score once my concerns are cleared.

1) Could you share the code to reproduce the experimental results?
2) Why does the baseline only include model-level explainers? Could local explainers be adapted to the same metrics? If not, can you explain the reasoning?
3) How does each component of the loss function impact the results?
4) How were example graphs, such as those in Table 3, selected? Is there any potential selection bias?

### Soundness
2

### Presentation
3

### Contribution
3
