# REBIND: Enhancing Ground-state Molecular Conformation Prediction via Force-Based Graph Rewiring

- Decision: Accept (Poster)
- Scores: 8, 6, 5, 6

## Abstract
Predicting the ground-state 3D molecular conformations from 2D molecular graphs is critical in computational chemistry due to its profound impact on molecular properties. Deep learning (DL) approaches have recently emerged as promising alternatives to computationally-heavy classical methods such as density functional theory (DFT). However, we discover that existing DL methods inadequately model inter-atomic forces, particularly for non-bonded atomic pairs, due to their naive usage of bonds and pairwise distances. Consequently, significant prediction errors occur for atoms with low degree (ie., low coordination numbers) whose conformations are primarily influenced by non-bonded interactions. To address this, we propose ReBIND, a novel framework that rewires molecular graphs by adding edges based on the Lennard-Jones potential to capture non-bonded interactions for low-degree atoms. Experimental results demonstrate that ReBIND significantly outperforms state-of-the-art methods across various molecular sizes, achieving up to a 20% reduction in prediction error. The code is available in: https://github.com/holymollyhao/ReBIND

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This article proposes a method, REBIND, to enhance the performance of deep learning (DL)-based molecular ground-state conformation prediction. The authors begin by identifying a key limitation in current approaches, which primarily rely on chemical bond information and interatomic distances but fail to adequately model atomic interactions, particularly for non-bonded atom pairs influenced by van der Waals potentials. Through detailed observations and statistical analysis, the authors conclude that current methods tend to accumulate more significant errors in predicting conformations of low-degree (i.e., low coordination numbers) molecules with fewer chemical bonds. This supports their viewpoint on the limitations of existing methods. To address this shortcoming, the authors introduce an approach based on the Lennard-Jones potential, deriving a new connectivity framework (comprising two adjacency matrices, $A^{attr}$ for attractive forces and for $A^{rep}$ repulsive forces) to redefine the molecular graph. By adding these new edges, the model enhances the representation of non-bonded atomic interactions as a form of data augmentation. The overall model architecture builds upon the GMTGC [1] network, replacing its original interatomic relation modeling component, $D^{row-sub}$, with the newly proposed $A^{attr}$ and $A^{rep}$, thereby strengthening the native self-attention mechanism. The results are clear and well-documented, strongly supporting the authors’ assertions. Comparative experiments demonstrate that these improvements lead to significant performance gains across various datasets.

For me, a major strength of this work is its clear observations and straightforward improvements that lead to noticeable performance gains, with a well-organized structure throughout. However, the methodological contributions feel somewhat modest. Given that molecular ground-state conformation prediction is still an underexplored area, this work holds potential to drive further advancements in the field.

[1] Xu, Guikun, et al. "GTMGC: Using Graph Transformer to Predict Molecule’s Ground-State Conformation." *The Twelfth International Conference on Learning Representations*.

### Strengths
1. The article is well-structured, with a clear and logical flow.

2.	The authors begin by hypothesizing and questioning the accuracy of existing methods for modeling atomic interactions, then validate their assumptions through experimental observations. Building on this, they propose an improved method, making the work both robust and effective.

3. The newly proposed metric, Energy-weighted RMSD (E-RMSD), is a practical contribution, enhancing evaluation standards for this task within the research community.

4. The extension of experiments to the GEOM-DRUGS dataset is also a valuable contribution, enriching the research standards for this field.

5.	The comparison in the ablation study between $A^{near}$ and $A^{force}$ effectively demonstrates the advantage of the authors’ proposed edge-enhancement method over the traditional distance-threshold-based neighbor edge enhancement. This approach could potentially be applied as a new edge construction strategy in other molecular modeling fields.

### Weaknesses
1. Some claims in the paper are unclear:
   1. In the Introduction, the statement “To this end, we propose a novel encoder-decoder graph transformer architecture…” is ambiguous. The process of using a Transformer encoder to predict the initial conformation, followed by a decoder to refine the final conformation, is actually a contribution from the GTMGC [1] architecture and should be properly cited. The main contribution of this paper is the improvement of how atomic interactions are modeled within the GTMGC framework.
   2. The definition of ${deg}^{rel}_i$ should be included and thoroughly explained in the Notation section, as it is the key metric used to classify nodes into low-degree and high-degree categories.

2. The ablation study would be more convincing if it included an analysis of the effects of removing $A^{bond}$ and$D^{row-sub}$ . Additionally, testing combinations of $A^{bond}$, $D^{row-sub}$ , $A^{near}$ and $A^{force}$ could reveal whether certain configurations lead to better performance.

3. In my opinion, the subjective evaluation results (Table 5) should be included in the main body of the paper for better clarity.

### Questions
1. I would like to know the results of the observations in **Figure 1** on large datasets like Molecule3D.
2. I would like to understand the trade-off in terms of overall forward pass time cost after incorporating compared to GTMGC.
3. **as Weaknesses.2**

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a method to enhance the conformation prediction for non-bonded edges, inspired by the Lennard-Jones potential. Experimental results show that their method can improve the performance across multiple datasets, especially for those atoms with lower ranks.

### Strengths
In general, this paper is a well-written ML paper.

1. The background and motivation are clarified clearly. The methodology is reasonable and well-aligned with the motivation.

2. The empirical study shows a consistent improvement on multiple datasets.

### Weaknesses
My main concern is whether the "large errors" observed for low-degree atoms arise from methodological limitations or issues in the evaluation process. Additionally, it's essential to assess how significant these "errors" might be for practical applications.

There might be multiple conformations with similar energy levels, thus they would probably occur with comparable Boltzmann weights. If this is the case, the sampling bias in the training/evaluation set might be the reason for large errors for low-degree atoms.
If a predicted conformation has a large RMSD with one ground truth, maybe it is also in an energy-favored state, but not included in the evaluation. 

Furthermore, flexibility in non-bonded conformations may not adversely impact empirical applications, as real-world cases often involve ensembles of multiple conformations.

### Questions
(1) As mentioned in the "Weaknesses" section, it would be helpful if the authors could provide a detailed analysis of the large errors in non-bonded interactions. Specifically, clarifying whether these errors are primarily attributable to model limitations rather than to the incomplete representation of ground-truth conformations would strengthen the discussion.

(2) The "E-RMSD" metric combines normalized force and Boltzmann factors, which makes it challenging to distinguish the contributions of each. It would be beneficial to disentangle these components, perhaps by introducing Boltzmann factors/force alone.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes a new attention mechanism for molecular conformation prediction tasks. This additional attention bias is constructed by extracting the estimated vdw force from the conformation predicted at an intermediate point in the network. The new architecture is then applied to the task of ground state prediction of molecules.

### Strengths
The addition of the force-based bias is well motivated. The usage of supervised predictions from the center of the model into the second half of the architecture seems somewhat novel. The paper is mostly clear (although certain parts could be summarized and simplified).

### Weaknesses
Although the paper does not have major flaws, the contribution seems somewhat incremental and the experimental validation unclear. 

Regarding the contribution, while the authors present it as force-based graph rewiring, to me it appears that the authors simply add to the second half of the network a hand-crafted pairwise feature that is mapped with a linear layer to the attention bias of the transformer. Is my understanding correct? Are there key contributions I'm missing? With this in mind, while this is an interesting idea, I’m not sure it needs a 9 pages paper, especially if the experimental validation is not very strong. 

Regarding the experimental validation, my main concern is the task definition and the way that it is evaluated. Molecular conformer generation methods do not aim to generate a single pose but multiple poses that represent the different low-energy conformation that the molecule can take. For this reason, the objective function with which they are typically trained are very different than the one used by the authors and the evaluation metrics are also significantly different and look not at the accuracy in capturing one pose but the whole set. A metric like the E-RMSE that the authors propose makes no sense to me: we are optimizing for the models to learn a weighted geometric average of the different conformations. For example, if we have two poses equally energetically favorable, this evaluation function would rank best the method that predicts the geometric average of these two poses, why would this be an interesting conformation? I believe the authors should adopt the metrics regularly used in the molecular conformer generation field that look at coverage & precision of the generated poses as well as their energetic properties.

### Questions
Are the RMSD/MAE/RMSE results corrected for molecular atom perturbations? If not, it is important to add such a correction.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces REBIND for ground-state molecular conformation prediction using force-based graph rewiring. By adding edges based on Lennard-Jones potential, REBIND captures non-bonded interactions between low-degree atoms. This framework includes an encoder-decoder graph transformer to first estimate atomic forces and then integrate those interactions into refined molecular graphs. Experimental results show substantial performance improvements, demonstrating REBIND’s effectiveness and scalability across multiple benchmarks and molecular sizes.

### Strengths
This work presents a novel contribution to molecular conformation prediction, primarily through its force-based graph rewiring mechanism. By incorporating the Lennard-Jones potential for non-bonded interactions, REBIND effectively addresses the low-degree atoms, which often suffer from high prediction errors in conventional models. The proposed force-aware edges between atoms provide an extra solution for creating molecular graphs besides relying on bonds and pairwise distances. REBIND could inspire further research in both conformer generation and broader applications in molecular property prediction.

This work contains rigorous experimental design and comprehensive evaluation on well-established datasets (QM9, Molecule3D, GEOM-DRUGS). The quantitative results are compelling, showing substantial improvements over baseline methods across multiple metrics.

In terms of clarity, the paper is well-structured, with clear explanations of the related concepts.

### Weaknesses
The authors didn't provide an efficiency evaluation of the proposed method. It would be better to know about the computational burden by introducing the force-aware adjacency matrices when compared to the ablated settings in ablation study.

When introducing the prior works, the authors didn't fully cover the previous GNN works [1-3] in this field that make efforts to model non-bonded interactions. Unlike GTMGC, those works already use non-linear functions of distance with atom type-specific coefficients when modeling non-bonded interactions. It is better to mention them to strengthen the coverage of prior related works.

[1] Kosmala, Arthur, et al. "Ewald-based long-range message passing for molecular graphs." International Conference on Machine Learning. PMLR, 2023.

[2] Zhang, Shuo, Yang Liu, and Lei Xie. "A universal framework for accurate and efficient geometric deep learning of molecular systems." Scientific Reports 13.1 (2023): 19171.

[3] Li, Yunyang, et al. "Long-Short-Range Message-Passing: A Physics-Informed Framework to Capture Non-Local Interaction for Scalable Molecular Dynamics Simulation." International Conference on Learning Representations. 2024.

### Questions
Are the force-aware adjacency matrices only computed once for decoding? If so, have the authors tried recycling the predicted molecular conformation C several times, i.e. using the final predicted C to compute new force-aware adjacency matrices and doing the decoding again? The new force-aware adjacency matrices could capture more accurate atomic pairs, so as to enable even better conformation predictions.

### Soundness
3

### Presentation
3

### Contribution
3
