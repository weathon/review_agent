# Modeling All-Atom Glycan Structures via Hierarchical Message Passing and Multi-Scale Pre-training

- Decision: Reject
- Scores: 6, 5, 6, 5, 5

## Abstract
Understanding the various properties of glycans with machine learning has shown some preliminary promise. However, previous methods mainly focused on modeling the backbone structure of glycans as graphs of monosaccharides (i.e., sugar units), while they neglected the atomic structures underlying each monosaccharide, which are actually important indicators of glycan properties. In this work, we fill this blank by introducing the GlycanAA model for All-Atom-wise Glycan modeling. GlycanAA models a glycan as a heterogeneous graph with monosaccharide nodes representing its global backbone structure and atom nodes representing its local atomic-level structures. Based on such a graph, GlycanAA performs hierarchical message passing to capture from local atomic-level interactions to global monosaccharide-level interactions hierarchically. To further enhance the model capability, we pre-train GlycanAA on a high-quality unlabeled glycan dataset in a self-supervised way, deriving the PreGlycanAA model. Specifically, we design a multi-scale mask prediction algorithm to endow the model with knowledge about different levels of dependencies in a glycan. Extensive benchmark results show the superiority of GlycanAA over existing glycan encoders and verify the further improvements achieved by PreGlycanAA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces GlycanAA, a novel model for all-atom-wise glycan modeling, which captures both atomic-level and monosaccharide-level interactions in a hierarchical manner. Unlike previous models that focus on the glycan backbone, GlycanAA uses a heterogeneous graph representation to capture detailed atomic interactions crucial for understanding glycan properties. It introduces hierarchical message passing, modeling atom-atom, atom-monosaccharide, and monosaccharide-monosaccharide interactions. To enhance the model's capabilities, the authors present PreGlycanAA, a pre-trained version of GlycanAA that uses self-supervised learning on a curated glycan dataset. The pre-training involves a multi-scale mask prediction algorithm that helps the model learn dependencies at different scales.

### Strengths
1. The paper provides a novel approach to glycan modeling by introducing GlycanAA, which captures both atomic and monosaccharide-level interactions hierarchically. The introduction of self-supervised pre-training with PreGlycanAA also adds a unique contribution to glycan modeling.
2. The hierarchical message-passing mechanism and the pre-training strategy are evaluated with extensive benchmarking, which includes ablation studies that highlight the effectiveness of the proposed techniques. The experimental results provide evidence of the model’s performance compared to existing baselines.
3. The paper is structured clearly, with explanations that are easy to follow for readers.
4. The paper addresses a notable limitation in glycan modeling by incorporating atomic-level information.

### Weaknesses
1. The technical novelty of the GlycanAA model could be considered incremental. Although the paper introduces a novel application of hierarchical message-passing for modeling all-atom glycan structures, similar hierarchical techniques have already been employed in other biomolecular modeling contexts, such as for proteins and small molecules. The concept of using graph-based models to represent complex biological structures and applying multi-level message passing has been explored in prior works (cited in the paper).

2. While the paper claims, *"Despite these advances, the potential of SSP in glycan modeling remains largely unexplored, presenting a new area of opportunity,"* it is not clear why the proposed losses from prior works, like those in [1], cannot be effectively leveraged for glycan modeling. A more comprehensive explanation of the unique challenges posed by glycan structures that necessitate a new pre-training approach would strengthen this claim.

3. This ambiguity raises additional questions about how the existing losses from prior works, if adapted for glycan modeling, would compare in performance to the proposed pre-training method. It would be valuable for the paper to explore this comparison or provide empirical evidence demonstrating the superiority of the proposed approach.

[1] Strategies for Pre-training Graph Neural Networks, ICLR 2020.

### Questions
Could you clarify why the self-supervised pre-training (SSP) losses from prior works, such as those in [1], cannot be directly applied or effectively adapted for glycan modeling? Are there specific structural properties of glycans or unique challenges that these methods fail to address? A more detailed comparison or analysis would be helpful in understanding the necessity of your custom pre-training approach.

[1] Strategies for Pre-training Graph Neural Networks, ICLR 2020.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents a novel multi-scale mask-based pretraining approach for a graph model tailored to glycans, called GlycanAA. This method introduces hierarchical message passing to capture interactions at both local atomic and global monosaccharide levels, enabling a comprehensive all-atom-wise modeling of glycan properties. The GlycanAA model extends GNN pretraining techniques to predict biologically relevant glycan properties, providing a more detailed representation than previous models that focused solely on monosaccharide-level structures. Through extensive evaluation on the GlycanML benchmark, the results demonstrate GlycanAA's superior performance over existing glycan encoders, showcasing its effectiveness in predicting complex glycan properties.

### Strengths
1. The paper represents a significant breakthrough in applying graph pretraining methods to the biological domain, particularly for glycan modeling. By incorporating essential domain-specific information—namely, atomic-level interactions and monosaccharide-level interactions—the proposed approach addresses a critical need for predicting glycan properties, which are highly dependent on these multi-level structures.
2. The paper provides a comprehensive comparison with baselines, and the experimental results are well-founded. The evaluations demonstrate the model's effectiveness and provide convincing evidence of its superior performance in glycan property prediction.

### Weaknesses
1. The novelty of the paper is limited, as it largely applies existing pretraining strategies to a new graph structure without substantial adaptation, aiming to model a hierarchical glycan property prediction scenario. The approach lacks customized design elements and theoretical innovation, which could have enhanced its contribution beyond a straightforward extension of conventional graph pretraining methods.
2. In some cases, the experimental results did not reach optimal performance (e.g., GlycanAA outperformed the best baseline in only 7 out of 11 tasks). The paper does not provide sufficient explanation for these discrepancies.

### Questions
Why is it considered unpromising to directly apply small molecule encoders or monosaccharide-level glycan encoders to all-atom glycan modeling? Has any theoretical or case analysis been provided to support this claim?

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
## Summary
The topic of this paper is modeling the glycan structure. The authors claim that the previous papers neglected the atomic structures underlying each monosaccharide. To this end, they propose GlycanAA, which models glycan as a heterogeneous graph via local and global representations. Then, the hierarchical message passing is performed to capture the atomic-level interaction and monosaccharide-level interaction. In addition, the authors pretrain the model on a high-quality unlabeled glycan dataset in a self-supervised manner. Extensive experiments demonstrate the superiority.

### Strengths
## Strength
1. The research direction is interesting and practical to life science. 
2. The method is easy to follow and understand, like Figure1.
3. The experiments are comprehensive, and the analyses are interesting. 
4. The dataset contribution is good.

### Weaknesses
## Weakness
1. From the perspective of self-supervised learning and graph neural networks, the novelty of the proposed method is limited. The techniques already existed, like massage passing, masking-recovery, pretraining-finetuning. But maybe the application scenario is new. 

2. The color presentation in Table 1 is not beautiful. Recommend using bold values and underlined values to represent the best and runner-up results. 

3. Efficiency experiments are missing, e.g., memory cost and time cost during the training and testing stages. 

4. The compared baselines are relatively old. Recommend comparing with more new methods published in 2023 and 2024, like Pre-Training Protein Bi-level Representation Through Span Mask Strategy On 3D Protein Chains.

### Questions
See weakness

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper focuses on the topic of modeling glycans. Glycans are hierarchical in structure, with each containing a number of monosaccharide and each monosaccharide containing many atoms. The authors note that previous SOTA ignores the atom-level information and only considers the monosaccharides. It has also been shown that methods that do model glycans from the atom-level tend to perform poorly. In order to properly account for both the monosaccharide and atom-level information, the authors propose to model each glycan in a hierarchical nature. Specifically, message passing is done between: monosaccharides, atoms in a monosaccharide, and between monosaccharides and it's children atoms. They further propose a pre-training strategy to enhance the downstream performance of their method. They show that their method can achieve improvement on the GlycanML benchmark.    

Overall I like the motivation and simple design of this method. However, I find the performance improvements of the proposed architecture (GlycanAA) to be marginal. Furthermore, I think the pre-training strategy should be applied to other models, as it's not unique to GlycanAA when ignoring the atoms. My general feeling is that I'm unconvinced that this paper can (a) truely leverage the power of the hierarchical structure or (b) the atom-level isn't very important to the downstream task.

### Strengths
1. The paper is written well. It is very clear and easy to understand.

2. The motivation behind their framework is good. Given the hierarchical nature of glycans, it makes sense that this information should not be ignored.

3. The model design is intuitive. It follows directly from the motivation (i.e., hierarchical modeling). I also like that it is simple, as it provides us with an easy to understand method that doesn't contain too many components or parameters that we need to tune.

### Weaknesses
1. The performance gain of GlycanAA is quite small. In terms of the weighted mean rank, it is only slightly ahead of RGCN. This suggesets that the current framework may not be able to effectively leverage the atom-level information. Or that it may not be crucial to the downstream task. This is noteworthy as I expect GlycanAA to be less efficient than other methods since it incorporates more information (the atoms in addition to the monosaccharides).

2. The pre-training strategy is not entirely unique to the proposed architecture. Essentially, a similar strategy can be used for other methods like RGCN, where we only mask the monosaccharides. Ideally this would also be included in the paper. For all we know, pre-training RGCN can even outperform PreGlycanAA. As I mentioned in my last point, it can be that considering the atom-level isn't that important. Pre-training RGCN would be helpful as it'll test us if we can achieve comparable improvements ny just masking the monosaccharides.

3. No complexity or runtime analysis is given. I only mention this as I expect GlycanAA to be more computationally expensive than monosaccharide-only methods. Furthermore, because the performance improvement is modest, it's important to consider the efficiency to see if it's "worth" considering the hierarchical structure. E.g., If it is much slower than RGCN then it may not be worth the additional time running it.

### Questions
1. Could you compare the runtime of GlycanAA and RGCN?
2. I'm curious if you can try the pre-training strategy on RGCN (where you of course only mask the monosaccharides). I think if the improvement isn't comparable to PreGlycanAA, that tells us that the hierarchical structure is important to pre-training, which would be a good motivation for this work.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces GlycanAA for all-atom modeling of glycan structures. GlycanAA uses hierarchical message passing within a heterogeneous graph for representation learning from both atomic and monosaccharide levels. Also, a pre-trained version called PreGlycanAA is developed using a multi-scale masking pre-training method, which leverages unlabeled glycan data to enhance model performance on downstream tasks. The model is evaluated on the GlycanML benchmark, showing competitive performance compared to baselines.

### Strengths
1. A hierarchical message-passing approach is proposed for modeling both local and global structures in glycans, which is crucial for complex biomolecules.
2. The pretrained version PreGlycanAA based on multi-scale masking can help utilize unlabeled data for more robust representations.
3. The paper is overall well written and organized.

### Weaknesses
1. While the model performs well on the GlycanML benchmark, there is limited discussion on what structural or relational insights it captures at the glycan level.
2. The hierarchical process may introduce extra computational complexity, but no experiment is provided regarding efficiency.
3. It seems that the most recent baseline comes from 2022, which could be a little out-of-date. Is there any more recent and competitive baselines? Besides, the improvements over a typical GNN encoder (RGCN) are not very significant, especially for the version without pre-training.

### Questions
1. Has the model been tested for scalability on larger datasets, where glycan structures may be more diverse or complex?
2. Is there any differences between the neural architecture modelings of glycan and protein? Or whether the recent advances in protein modeling can be transferred to this scenario?

### Soundness
2

### Presentation
3

### Contribution
2
