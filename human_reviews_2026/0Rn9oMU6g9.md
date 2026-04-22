# Subspace-Boosted Model Merging

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Model merging enables the combination of multiple specialized expert models into a single model capable of performing multiple tasks. However, the benefits of merging an increasing amount of specialized experts generally lead to diminishing returns and reduced overall performance gains. In this work, we empirically and theoretically analyze this limitation, proving that for Task Arithmetic-based methods, as more experts are merged, the common information dominates the task-specific information, leading to inevitable rank collapse. To mitigate this issue, we introduce Subspace Boosting, which operates on the singular value decomposed task vector space and maintains task vector ranks. Subspace Boosting raises merging efficacy for up to 20 experts by large margins of more than 10% when evaluated on both vision and language benchmarks. Moreover, we propose employing Higher-Order Generalized Singular Value Decomposition to quantify
task similarity, offering a new interpretable perspective on model merging.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The work tackles task-vector-based model merging, where the differences between task-specific finetunings and their base models (task vectors) are aggregated to obtain one single multi-task model that can perform all the tasks at once. In particolar, the paper investigates the problem of rank collapse when merging many tasks: the more tasks are added, the higher the singular values for the most important directions, contracting the spaces into a few dominant directions. The work proposes clamping the singular values to avoid some of the singular directions overshadowing the others. It also proposes higher-order SVD to obtain a space that is shared across the tasks, making different task-specific weights more comparable in the latter. The method is tested on ViT-B/32, ViT-B/16 and ViT-L/14 on the common 8-, 14, and 20-tasks benchmarks.

### Strengths
- The method is interesting and intuitive. Current SVD-based methods either do not consider any common subspace (TSV [1]) or obtain one just by summing the task vectors before the SVD takes place (Iso-C [2]). Producing this one through Higher-Order SVD seems principled and effective. The method is applicable to any existing task-arithmetic variant, including e.g. TIES and Consensus Merging. It is also more efficient than comparable SVD-based methods, which is by itself not a big thing in standard model merging but could be useful in applications requiring iterative merging (e.g. federated learning). Being data- and tuning-free, the method is broadly applicable and in line with the considered literature.
- The paper is well written and easy to read, with clear figures that are immediate to grasp. The formalization is intuitive and the paper structure is well-thought.
- The results are competitive with the state-of-the-art, and the experimental evidence is extensive and considers all the relevant baselines, model architectures and benchmarks.

### Weaknesses
- The evidence for the main motivation, i.e. the rank collapse, is somewhat limited: in layer 10, one of the most significant, the trend is actually the opposite: N14 and N20 have higher stable rank than N8. This suggests that it has more to do with the particular composition of the merging set rather than the number of tasks alone. For a similar analysis to be performed, one should try to average over subsets of increasing cardinality, possibly many of them so to rule out the difference in composition.
- The novelty is somewhat limited when compared to Iso-C [1]. The main contribution seems to be solving the rank collapse phenomenon, but this seems to be already tackled by Iso-C/Iso-CTS [1], although in a simpler manner. Simplicity however has its benefits, as [1] does not require a beta hyperparameter.
- From the performance standpoint, the approach only outperforms the current state-of-the-art in the ViT-B-32 setting, while remaining below Iso-CTS for larger architectures.
- The alignment method is not well motivated and discussed. I don’t fully understand what’s the intuition of e.g. MNIST and EuroSAT being most aligned to SUN397. How does this alignment measure correlate with the interference measures proposed in TSV and Iso-C? The subspace alignment studied in Iso-C would be particularly interesting as from my understanding it also measures similarity in the singular vector space, although differently.
- The whole “interpretable merging” point seems fairly oversold. It is not immediately clear to me what added interpretability stems from the proposed alignment matrices. I might have missed the point, and I would be happy to be convinced otherwise on this aspect.

Given the current strengths and weaknesses, I am inclined to reject the paper: from the performance standpoint, it does not distance itself by comparable baselines that are also similarly motivated, and the remaining benefits (e.g., interpretability) are not properly explored and discussed.   

[1 Marczak, Daniel, et al. "No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces." ICML 2025.

[2] Gargiulo, Antonio Andrea, et al. "Task singular vectors: Reducing task interference in model merging." *Proceedings of the Computer Vision and Pattern Recognition Conference*. 2025.

### Questions
- What is the proper way to look at the alignment matrix? what are its immediate implications? how does it correlate with other existing measures? (see weakness 4).
- Figure 3 does not specify the architecture, the task nor the layer.
- Why is A used in place of Delta (common choice in previous literature?) This seems like a peculiar choice, given that the alignment matrix is termed **A**.
- I don’t fully understand the I_{>1} notation. I get that it refers to the common components, but why is it expressed in this way?
- M is also somewhat confusing as might lead to think of a matrix instead of a scalar. Also some papers use it for the merged model.
- What sort of interpretability can we derive from the alignment matrices?

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
3

### Summary
This paper identifies "rank collapse" in the task vector space as a fundamental limitation in model merging, which explains why performance gains diminish as more expert models are combined. To solve this, the authors propose Subspace Boosting, a training-free method that uses Singular Value Decomposition (SVD) to decompose task vectors and explicitly enhance underutilized dimensions, thereby maintaining the effective rank and significantly improving merging efficacy by over 10% on vision and language tasks. Additionally, the paper introduces the use of Higher-Order Generalized SVD (HO-GSVD) as a novel framework to quantify task similarity, offering a new interpretable perspective on model merging and enabling principled expert selection.

### Strengths
- Unlike prior works that only observed diminishing performance with more merged experts, this study provides a mechanistic explanation from a task vector space perspective: as more experts are merged, task vectors suffer from rank collapse.

- The proposed Subspace Boosting addresses rank collapse in a highly practical manner. Operating via singular value decomposition (SVD) on merged task vectors, it boosts underutilized small singular values to maintain effective rank.

### Weaknesses
- The third part quantifies Rank collapse only relying on "Stable Rank" and "Cumulative Energy Rank" (for example, Formula 2 defines stable rank as the ratio of the sum of squares of singular values to the square of the maximum singular value). However, the universality of these two indicators for the "correlation degree of model fusion performance" has not been fully demonstrated. The manuscript only demonstrates the negative correlation between the stable rank and performance through experiments of the ViT-B/16 model (Figures 2 and 3), but does not verify: 
     - a . In different model architectures (such as the language model T5), whether the stable rank can still effectively reflect the impact of rank collapse on performance - for instance, the dimension of the weight matrix and the layer structure of T5 are significantly different from those of ViT, the distribution rules of singular values may be different, and the "effective rank" representation ability of the stable rank may fail; 
     - b. In extreme scenarios (such as fusing two models or fusing models with highly similar tasks), will there be a counterexample of "low rank but high performance" in the stable rank? If so, it indicates that this metric cannot be used alone as a criterion for determining rank collapse.

- Subspace enhancement relies on the hyperparameter β (lift threshold) to determine the singular value cutoff point to be enhanced. However, this paper only found through experiments that the performance is stable when β∈{0,0.01,0.02} (Table 3a), without providing a theoretical explanation when the task type changes, does the optimal value range of β remain stable? If β needs to be re-tuned according to the scene, the practicality of the method will significantly decline, but the documentation has not verified this boundary condition.

- The experiment only selects "same mode, same type" tasks.  All visual tasks are classification tasks and do not include non-classification tasks such as detection and segmentation. All language tasks are QA and NLP classification tasks (such as sentiment analysis), and do not include generation or translation tasks. The "cross-type task fusion" scenario - such as fusing classification and detection tasks - has not been verified to see if subspace enhancement can still improve performance. If the task vector conflicts of cross-type tasks are more significant, the method's effectiveness may drop significantly, and the generalization of existing conclusions is limited.

### Questions
- Whether Subspace Boosting can simultaneously improve performance on both visual and language sub-tasks?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper reveals the rank collapse limitation in existing model merging methods. To address this issue, the authors propose a technique called subspace boosting, whose core idea is to boost the singular values below a certain cutoff threshold.

### Strengths
1. This work identifies a critical limitation in existing model merging approaches, rank collapse, and provides empirical evidence to support this finding.

2. The paper is well-organized and easy to follow.

### Weaknesses
1. **Potential error amplification.** Directly boosting the singular values below the cutoff point may introduce noise or bias. The authors should provide further discussion to justify the rationality of the proposed subspace boosting technique.

2. **Hyperparameter sensitivity.** The method requires manual tuning of the cutoff hyperparameter, which may limit its practicality and robustness in real-world applications.

3. **Unclear connection between HO-GSVD and rank collapse.** While HO-GSVD offers a new and interpretable perspective on model merging, it is unclear how it directly relates to the core motivation of preventing rank collapse. This conceptual gap may cause readers to lose focus on the main contribution.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Subspace Boosting, a method developed to improve the effectiveness of model merging. The authors analyze the degradation in performance that occurs as the number of merged models increases, attributing it to a reduction in the effective dimensionality of the task-vector space, where variance becomes concentrated in a few dominant directions.
Subspace Boosting is implemented on top of the Higher-Order Generalized Singular Value Decomposition (HO-GSVD) framework. Additionally, the authors introduce the Alignment Matrix, which measures relationships between task vectors within the shared subspace.
Experiments on Vision Transformer (ViT-B/32, ViT-B/16, and ViT-L/14) models trained on 8, 14, and 20 tasks compare Subspace Boosting against the ISO-C and TSV baselines and show the performance obtained applying this method to some model-merging baselines ( TIES-Merging, Task Arithmetic, Consensus Merging).

### Strengths
- The topic is relevant to the community, addressing a problem in multi-task and model-merging research.
- The proposed Subspace Boosting method is conceptually clear and appears computationally efficient.
- The connection between singular value structure and model merging dynamics is insightful. The analysis of shared versus task-specific subspaces through the singular-value structure is particularly interesting.

### Weaknesses
- Some inclarities about the tables and exepriment report (see questions). 
- The notion of “rank collapse” that is central in the paper could benefit from a more formal explanation.
- Algorithm 1 applies a standard SVD step. I think it is misleading to put it in the approach instead of the algorithm of the Subspace Boosting. 

Minor
- In Figure 2 (a–c), the y-axis label “Value” should likely be “Stable rank value”, right?
- In line 268, “n” is associated with the shape of V and aslo to the  number of merged tasks, is this notation correct?

### Questions
1. Table 1 vs. Table 4: The results for your method differ between Tables 1 and 4. Could you clarify why this is the case?
2. In table 4 Subspace Boosting 4 uses LiNeS while other baselines do not, why, is the comparison fair?
3. Table 10,  Random selection: In Table 10, you average over 20 random selections when merging 8 out of 20 models. Why not also report the standard deviation to show the variance of random selection?
4.  In Figure 3, the largest singular values seem to scale roughly linearly with the number of merged experts (e.g., almost 0.05 for 4 experts,  close to 0.10 for 8, and almost 0.20 for 20).
This appears consistent with what one would expect if the overall Frobenius norm of the merged matrix increases linearly with the number of merged task vectors, meaning the curves could differ mainly by a global scaling factor rather than by a change in shape.
Did you normalize the merged weight matrices (e.g., by dividing by the Frobenius norm or the largest singular value) before plotting the singular-value distributions in Fig. 3?
If not, could the apparent steepening of the spectra be explained by such scaling rather than by true “rank collapse”?
5. $\beta$ is tuned over the set {0, 0.01, 0.02}. How this range was chosen, and whether a broader search might affect results?
6. You mention that Subspace Boosting is faster than competing methods. Could you explain which component (e.g., decomposition, projection step, or optimization) contributes most to this improvement?
7. Have you considered including the Singular Task Interference metric (Gagiurlo et al.) as an additional as a way to interpret task similarity?

### Soundness
3

### Presentation
3

### Contribution
2
