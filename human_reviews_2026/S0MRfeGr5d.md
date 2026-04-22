# StatsMerging: Statistics-Guided Model Merging via Task-Specific Teacher Distillation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
As large models are increasingly deployed across various tasks, the limited GPU memory available for storing and executing task-specific models presents a growing bottleneck. Model merging has emerged as a promising solution to accommodate multiple large models within constrained memory budgets. While traditional multi-task learning methods attempt to merge common layers, they require labor-intensive annotated labels and incur significant computational overhead. Recent merging techniques aim to address this issue by combining models at inference time; however, these approaches often rely on simplistic heuristics, ignore weight distribution characteristics, assume architectural identity, or require access to test samples to infer merging coefficients, thereby limiting generalization and scalability. We present StatsMerging, a novel lightweight learning-based model merging method guided by weight distribution statistics without requiring ground truth labels or test samples. StatsMerging offers three key advantages: (1) It uniquely leverages singular values from singular value decomposition (SVD) to capture task-specific weight distributions, serving as a proxy for task importance to guide task coefficient learning; (2) It employs a lightweight learner StatsMergeLearner to model the weight distributions of task-specific pre-trained models, improving generalization and enhancing adaptation to unseen samples; (3) It introduces Task-Specific Teacher Distillation for merging vision models with heterogeneous architectures, a merging training paradigm that avoids costly ground-truth labels by task-specific teacher distillation. Notably, we present two types of knowledge distillation, (a) distilling knowledge from task-specific models to train StatsMergeLearner; and (b) for the first time, distilling knowledge from models with different architectures prior to merging, following a distill-then-merge paradigm. Extensive experiments across vision and NLP tasks demonstrate the effectiveness of StatsMerging. Our results show that StatsMerging outperforms state-of-the-art techniques, achieving overall accuracies of 94.5% for Vision and 77.6% for NLP, while further exhibiting strong generalization to unseen tasks, and robustness to image quality variations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes StatsMerging, a method for merging pre-trained models using weight distribution statistics to guide the learning of merging coefficients.

### Strengths
1. The use of SVD singular values as a proxy for task/layer importance is a reasonable extension of existing merging techniques. This is also used in some recently published model merging methods, such as TSVM (task singular vector merging).
2. Handling heterogeneous architectures via distill-then-merge is a practical contribution.

### Weaknesses
Below are some weaknesses of the manuscript and suggestions for improve the manuscript:

1. The averaged performance of individual fine-tuned models could be shown in Figure 4 using a vertical line.
2. Lack comparison with recent state-of-the-art training-free model merging methods, such as TSVM (task singular vector merging) and RegMean++. 
3. Mirror typos: 
    - At line 371 on page 7, should "MEMoE" be "WEMoE"?
    -  At line 226-235 on page 5, please verify that all instances of the symbols $\sigma_{r}$ and $\sigma’_{r}$ are used correctly.
4. SVD rank is fixed at 3 without ablation—why not 1, 5, or 10? How sensitive is performance to rank choice?
5. Weight statistics are used but no justification for why these capture "task importance" better than alternatives like Fisher information used in Fisher merging.

### Questions
1.Why does the accuracy increase so dramatically at about 420 steps as shown in Figure 8?

### Soundness
3

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
This paper introduces StatsMerging, a statistics-guided model merging approach designed to consolidate multiple task-specific deep models—primarily in vision and NLP domains—without requiring ground truth labels or access to test samples. The architecture leverages Singular Value Decomposition (SVD)-derived statistics (mean, variance, norm, top singular values) from model weights and trains a lightweight StatsMergeLearner (SML) to predict adaptive merging coefficients through a novel task-specific teacher distillation process. The method extends to heterogeneous architectures and is validated across eight vision and seven NLP tasks, showing improvements over several baselines in average accuracy and robustness to data/label noise.

### Strengths
- The article demonstrates the absolute advantages of this method across various task sets and datasets through extensive experiments.
- The method proposed in the article does not require manually annotated labels; instead, it leverages pseudo-labels generated by existing models. This eliminates the need for manual annotation and provides certain support for the subsequent expansion of different scenarios.
- The structural design of SML is very simple with low training costs, and its computational cost is much lower compared to other model merging methods.

### Weaknesses
- There is a lack of certain theoretical or experimental explanations. For instance, it does not clarify the reason for choosing rank=3 when performing SVD.
- The experimental details are insufficient. For example, it fails to specify the exact amount of data used for training StatsMerging and StatsMerging++ respectively.
- There may be issues with the experimental results. For instance, the results of LW StatsMerging++ in Table 3 and Table 7 of the Appendix are inconsistent.

### Questions
- Why is a rank of 3 chosen for SVD? Is it necessary to add ablation experiments on rank to identify the tradeoff between the overhead caused by rank and performance?
- Can a graph showing the relationship between the amount of training data and performance be provided? Additionally, can the specific amounts of data used for StatsMerging and StatsMerging++ be confirmed?
- Can the reason for the inconsistent results of LW StatsMerging++ in Table 3 and Table 7 of the Appendix be clearly explained? Alternatively, can the corrected experimental results be provided after re-evaluation?

### Soundness
2

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
This paper proposes a lightweight, learning-based model merging method named StatsMerging. The core idea is to adaptively predict the merging coefficients based on the weight distribution information.

### Strengths
1. The paper is well-organized and easy to follow.

2. The experiments demonstrate that the proposed method achieves promising results.

### Weaknesses
1. **Unclear motivation.** The usage of weight distributions plays a central role in the proposed method and serves as its main motivation. However, the paper lacks empirical or theoretical evidence to support the importance or effectiveness of this design choice.

2. **Limited technical contributions.** The use of knowledge distillation has been extensively studied in the context of model merging and other areas of machine learning. Similarly, learning adaptive merging coefficients has also been well explored in prior model merging research, which limits the novelty of the proposed approach.

3. **Reliance on training data.** The proposed method requires access to additional training data, whereas data-free model merging approaches have already been widely studied. This reliance may weaken the practical advantage of the proposed method.

### Questions
See weaknesses above.

### Soundness
2

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
5

### Summary
The paper introduces StatsMergeLearner, a network that predicts scaling coefficients for model merging from statistical features of model weights, trained using pseudo-labels generated by teacher models. It also proposes a Merge+Distill framework to homogenise model architectures—a prerequisite for merging—and reports strong gains over prior work across vision and language tasks.

### Strengths
(1) String Performance: Consistently outperforms Adamerging and WEMoE on the reported benchmarks.

(2) Practical pipeline: The proposed Merge+Distill framework provides a workable path to architectural homogenisation, enabling broader applicability of merging.

### Weaknesses
(1) Limited task scale: Evaluation is restricted to the 8-task vision suite; it omits the widely used 14- and 20-task benchmarks from [1] on ViT-B/32, ViT-B/16, and ViT-L/14, which are important for understanding scaling with respect to task count and backbone size.

(2) Baselines: Missing comparisons to recent data-free merging methods such as Isotropic Merging[2], TSV-M[3], and KnOTS[4].

(3) Ablations: Task-level StatsMerging is introduced but not compared against layer-wise StatsMerging across settings, making it unclear what the difference in performance looks like.

References:

 [1] Wang, Ke, et al. "Localizing task information for improved model merging and compression." arXiv preprint arXiv:2405.07813 (2024).

[2] Marczak, D., Magistri, S., Cygert, S., Twardowski, B., Bagdanov, A. D., & van de Weijer, J. (2025). No task left behind: Isotropic model merging with common and task-specific subspaces. arXiv preprint arXiv:2502.04959.

[3] Gargiulo, A. A., Crisostomi, D., Bucarelli, M. S., Scardapane, S., Silvestri, F., & Rodola, E. (2025). Task singular vectors: Reducing task interference in model merging. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 18695-18705).

[4] Stoica, G., Ramesh, P., Ecsedi, B., Choshen, L., & Hoffman, J. (2024). Model merging with svd to tie the knots. arXiv preprint arXiv:2410.19735.

### Questions
(1) Effect of extra validation data: The jump from LW-StatsMerging to LW-StatsMerging++ is substantial. Do other adaptation methods (e.g., Adamerging) show similar improvements as validation data increases? Additionally, since StatsMerging does not introduce any new hyperparameters of its own, what is the role of validation data? 


(2) Data sources: What exact data are used for training StatsMergeLearner and for distillation? Is it the training samples from each task? Also, is the setup strictly unlabeled for both training and validation, or is any labelled data employed?


(3) Training cost: Line 332 states StatsMergeLearner is trained for 500 epochs, which seems heavy compared to, e.g., Adamerging’s ~500 steps, which is okay since it also outperforms it, but should not be regarded as a lightweight method. Do (a) the number of models being merged and (b) backbone size (e.g., ViT-L/14) influence the number of training steps used to train the StatsMergeLearner?

### Soundness
2

### Presentation
3

### Contribution
2
