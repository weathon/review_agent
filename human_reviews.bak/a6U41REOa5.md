# Retrieval Augmented Diffusion Model for Structure-informed Antibody Design and Optimization

- Decision: Accept (Poster)
- Scores: 6, 6, 5

## Abstract
Antibodies are essential proteins responsible for immune responses in organisms, capable of specifically recognizing antigen molecules of pathogens. Recent advances in generative models have significantly enhanced rational antibody design. However, existing methods mainly create antibodies from scratch without template constraints, leading to model optimization challenges and unnatural sequences. To address these issues, we propose a retrieval-augmented diffusion framework, termed RADAb, for efficient antibody design. Our method leverages a set of structural homologous motifs that align with query structural constraints to guide the generative model in inversely optimizing antibodies according to desired design criteria. Specifically, we introduce a structure-informed retrieval mechanism that integrates these exemplar motifs with the input backbone through a novel dual-branch denoising module, utilizing both structural and evolutionary information. Additionally, we develop a conditional diffusion model that iteratively refines the optimization process by incorporating both global context and local evolutionary conditions. Our approach is agnostic to the choice of generative models. Empirical experiments demonstrate that our method achieves state-of-the-art performance in multiple antibody inverse folding and optimization tasks, offering a new perspective on biomolecular generative models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper authors propose a method to improve upon inverse folding methods by mining PDB for similar structural motifs and use an MSATransformer to take them into account when predicting the loop sequences, along side the usual structure-conditioned diffusion model as used in DiffAb. Inclusion of the mined structures improved the performance of the model in the chosen experiments inverse folding and energy optimization experiments.

### Strengths
Inclusion of the data-mined motifs from PDB with similar loop structures is a quite nice and intuitive improvement. That helps the performance of the model. The paper is clearly written and easy to follow. Ablation is performed on the key changes introduced, showing that they are all beneficial.

### Weaknesses
Authors state that other models "typically create antibodies from scratch without template constraints, leading to model optimization challenges and poor generative capability". This is a rather harsh critique and I would like such sentences to be backed either by citations or experiments directly showing that. It is true that the proposed approach did generally better than baselines in the chosen experiments, but the difference to DiffAb was marginal and there can be a multitude of reasons for that difference.

In the energy optimization experiment authors say that authors use structures "folded from the original antibody sequences" to evaluate the energy. Folding can be arbitrarilly bad in determining the loop strucutres (esp. CDR H3). So this might very much end up measuring just the agreement between the folding method and the method being tested (how common the folding method finds the sequence to be) instead of how good the energy trully would be. If sequence recovery is reasonably high, one could use grafting and MD to get higher quality/more plausable complexes. Actually, what is the RMSD to original crystal structure, of the reference structure your approach produces (RMSD of original co-crystal vs folding original Ab and runing Rosetta fast relax)?

While in general I like the idea of data mining the PDB for similar loops and using them as reference. I find that the paper is a marginal improvement on existing literature and is of somewhat limited scope for a ICLR paper (only doing inverse folding, while say DiffAb can also do complete loop sequence + structure redesign, and having only two experiments, one of which is not fully convincing to me).

### Questions
Did you investigate using a similar approach for structure and sequence re-design (as done by DiffAb)? This would expand the scope of the paper.

When you ablate the numberk of reference loop sequences fed to the model (Figure 4), you show that the optimal number of sequences to use is 15 "But when k exceeds 15, the additional information instead introduces noise to the model, leading to performance degradation.". Do you have any motivation why this would be the case? I'd imagine that as long as the mined loops satisfy some RMSD/quality threshold, they should always be beneficial to the model.

### Soundness
2

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
This paper introduces a retrieval-augmented diffusion framework for anti-body design and optimization. Its key contribution is that it retrieves structurally-similar protein segments to the CDR, and use them as conditions in the diffusion generation. Experiments demonstrates impressive results, showing improved performances in antibody CDR sequence inverse folding and antibody functionality optimization. Ablation experiments show that both the retrieval augmentation and evolutionary embedding (ESM2) contribute to the improved performance.

### Strengths
Originality: See the weaknesses below

Quality: The paper overall has good quality. The experiment is sound.

Clarity: The paper is written clearly.

Significance: The paper significantly improves upon the baseline models in terms of the metrics. It has good significance.

### Weaknesses
Originality: There has been retrieval-augmented diffusion model proposed in other applications, such as in [1] and [2] (protein-specific 3D molecule generation). Although the paper is the first to introduce it in the antibody design, the method novelty is fair due to the prior works. The paper needs to also cite previous retrieval-augmented generative models, and clearly state the contribution in terms of the method.


[1] Blattmann, Andreas, et al. "Retrieval-augmented diffusion models." Advances in Neural Information Processing Systems 35 (2022): 15309-15324.

[2] Huang, Zhilin, et al. "Interaction-based Retrieval-augmented Diffusion Models for Protein-specific 3D Molecule Generation." Forty-first International Conference on Machine Learning.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper combines the popular technique RAG with antibody design by extracting templates from PDB to enrich evolutionary information, leading to better generalization ability. To be specific, the extracted fragments are based on backbone structures and some properties using a novel retrieval mechanism to utilize both structural and evolutionary information. Experiments demonstrate the effectiveness of the proposed RADAb framework.

### Strengths
1. I really like the idea of incorporating RAG into antibody design in such an elegant manner, which not only fulfills the relatively blank space of known data but also utilizes evolutionary information. This is intuitive and turned out to be effective.
2. Nice illustrations, which clearly convey the architecture and designs of authors.

### Weaknesses
1. The experiment part is relatively weak. For example, for antibody functionality optimization task, only one simple experiment is provided.
2. Baselines are inadequate. Large efforts have been devoted to studying antibody design, such as [1][2][3]:. They utilize the same experimental settings and metrics, which should be taken into consideration.
3. If I comprehend correctly, although RADab is capable of capturing structural and sequential information, it is still a sequence design work, rather than sequence-structure co-design or even full-atom generation work, which is a major weakness compared to other methods.

[1]Kong X, Huang W, Liu Y. End-to-end full-atom antibody design[J]. arXiv preprint arXiv:2302.00203, 2023.

[2]Xiangzhe Kong, Wenbing Huang, and Yang Liu. Conditional antibody design as 3d equivariant
graph translation. arXiv preprint arXiv:2208.06073, 2022.

[3]Shitong Luo, Yufeng Su, Xingang Peng, Sheng Wang, Jian Peng, and Jianzhu Ma. Antigen-specific
antibody design and optimization with diffusion-based generative models for protein structures.
Advances in Neural Information Processing Systems, 35:9754–9767, 2022.

### Questions
1. Is the provided experimental results correct? According to previous works, can $\Delta \Delta G$ be optimized on such a huge scale?  (~100kcal/mol?) Even for the selected examples in 5.2, $\Delta \Delta G$ only changed like ~-40, far away form claimed 109.16.
2. Also, $\Delta \Delta G$ is usually expressed using negative numbers to demonstrate lower energy is achieved. Why $\Delta \Delta G$ is positive in Tab. 3? Suppose you are using the absolute value of $\Delta \Delta G$ , since lower $\Delta G$ is better, why lower $|\Delta \Delta G|$ indicates better performance? Please explain this.

### Soundness
2

### Presentation
2

### Contribution
3
