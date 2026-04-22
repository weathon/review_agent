# AdaRank: Adaptive Rank Pruning for Enhanced Model Merging

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Model merging has emerged as a promising approach for unifying independently fine-tuned models into an integrated framework, significantly enhancing computational efficiency in multi-task learning. 
Recently, several SVD-based techniques have been introduced to exploit low-rank structures for enhanced merging, but their reliance on heuristically designed rank selection often leads to inter-task interference and suboptimal performance.
In this paper, we propose **AdaRank**, a model merging framework that replaces this heuristic selection by adaptively selecting the beneficial singular components of task vectors to merge multiple models. 
We first show empirically that **(i)** selecting only the top singular components of task vectors can cause critical interference with other tasks, and **(ii)** assigning fixed ranks does not align with the varying complexity of tasks and layers. 
AdaRank addresses both issues by adapting per-component masks, indicating the selection of the component, to the unlabeled test data with entropy minimization. 
Our experimental findings show that AdaRank consistently improves existing merging methods across diverse backbones from different modalities, largely narrowing the performance gap against individually fine-tuned models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper identifies the suboptimality of direct rank pruning and fixed number of ranks in SVD-based merging. They introduce AdaRank merging technique which dynamically selects top singular components that minimize inter-task interference, and demonstrate its effectiveness through broad settings.

### Strengths
1. Overall this paper is highly systematic, written with a good logical flow. The proposed method is tightly motivated by concrete experimental findings on top basis and ranks.
2. Proposed AdaRank is straightforward, lightweight, and compatible with many existing SVD-based methods. Useful and easy for adoption.
3. Main experiment (section 6.2) is broad and covers diverse settings, including different numbers of tasks, datasets, and architectures. Merging methods in comparison are very comprehensive.
4. Solid analysis and supporting materials. Ablation analysis (section 6.3) is well designed and provides a better understanding of the AdaRank method, as well as the behavior of ranks and cross-task singular basis interaction in general and qualitative analysis on the AdaRank’s masks. Further transparency on additional baseline comparison (Appendix D.3) is also appreciated. (though shouldn’t it be included in the main text as well?)

### Weaknesses
1. The motivational analysis on top singular components and ranks (section 3) are limited: 1) The experiment is only done on a single ViT model with a specific task set, making the generalizability of this finding toward other models, task types, and training methods unclear. 2) The analysis focuses solely on quantifying the effect of manipulating the top basis and rank, without contextualizing with other relevant information (e.g. how similar are the top singular components across different tasks to begin with, or, how much performance is when preserving only the intrinsic rank). This makes it difficult to make sense of the paper’s analysis in broader perspective. 3) By not always selecting top rank vectors, individual task’s performance would be degraded a lot more than usual, however the paper doesn’t quantify this tradeoff in isolation.
2. Time consumption and computation cost for using AdaRank on modern large-scale models like LLMs is not presented. I suspect that for larger models, the TTA time and memory requirements would be prohibitive.

### Questions
1. The comparison with AdaMerging on language models seems unfair.
2. The critical choice of minimizing Shannon entropy is not validated empirically. Its optimality isn’t specifically confirmed for AdaRank, other possible proxies are also not discussed.
3. Table 1 shows that under many settings, merging 20 tasks yields better performance than 14 tasks. Why so?
4. How is it  possible that Oracle achieves performance equal to the Individual model in ablation Figure 4? Also, what’s the models and tasks used?
5. Do the authors plan to discuss or experiment with more recent SVD-based methods (e.g. ISO [1])?
6. AdaRank builds up from the individual SVD paradigm, then how is it related to or differed from joint SVD paradigm (e.g. KnOTS [2], DRM [3])?
7. Does the finding in this paper corroborate with the result from other neighboring adaptive rank-based approaches like AdaLoRA [4]?

[1] https://arxiv.org/pdf/2502.04959v3
[2] https://arxiv.org/pdf/2410.19735
[3] https://arxiv.org/pdf/2505.23117
[4] https://arxiv.org/pdf/2303.10512

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
The paper addresses the problem of test-time adaptation for model merging, which involves learning a set of parameters on an unsupervised set to perform the merging of models. Since the task matrices exhibit varying effective ranks across tasks and network layers, using a fixed rank in SVD-based merging techniques—by selecting a constant number of top-$k$ components—can be suboptimal. To overcome this limitation, the authors propose an adaptive mechanism based on binary masks, which dynamically selects the singular components for each task matrix.

Specifically, the proposed algorithm, AdaRank, builds upon a generic data-free merging method operating on task matrices (such as Task Arithmetic, CART, or TSV). It first computes the task matrix as specified by the chosen method, then performs an SVD decomposition of the resulting task matrices for each task and layer. These binary masks are applied to the singular values and optimized during training via the Straight-Through Estimator (STE),  together with learnable task coefficients, with the objective of minimizing the entropy on the unsupervised set. The approach is evaluated on both vision and language tasks.

### Strengths
- The empirical findings showing that selecting only the top singular components of task vectors introduces interference is clearly presented. I particularly appreciated the experiments in Section 3, which illustrate how the loss evolves as singular components are progressively added to the merged task matrix.
- The proposed approach significantly improves performance over AdaMerging, the previous method that learns task coefficients for model merging on the unsupervised set.
- Extensive experiments are conducted to validate the effectiveness of the method.

### Weaknesses
-  From a methodological perspective, the novelty of the proposed approach is limited. The unsupervised loss function is identical to that used in AdaMerging. The authors introduce binary masks, which were already employed in TALL [1], although in that work they were not learnable and were applied to task matrices. The use of the Straight-Through Estimator (STE), as acknowledged by the authors themselves, is also a well-established technique. Overall, the method appears to combine existing components in a reasonable way, but without introducing a significant methodological contribution.

- While the empirical findings showing that adding top components introduces interference are interesting, the paper does not provide a clear explanation or theoretical insight into why this phenomenon occurs (see the Question section).

- As the number of tasks, network layers, and the dimensionality of the feature space increase, the computational complexity of the approach also grows. It is unclear whether the method remains effective under these conditions (see the Question section).

[1] Wang, K., Dimitriadis, N., Ortiz-Jiménez, G., Fleuret, F., & Frossard, P. (2024). Localizing task information for improved model merging and compression. In Proceedings of the International Conference on Machine Learning (ICML).

### Questions
Q1) Why does adding top principal directions introduce interference? Is this effect driven by the scale of the singular values or by the specific directions of the singular vectors? Moreover, since task singular vectors interact with one another (while the experiments only show what happens when a single task singular vector is added at a time), it would be interesting to analyze how interference emerges from the joint combination of singular vectors. Finally, the task singular vectors also interact with those of the pre-trained backbone. It remains unclear how the top singular vectors of each task matrix align or interfere with the singular directions of the pre-trained network, and whether the proposed method mitigates any of these effects.

Q2) I have seen the computational analysis performed in the Appendix on ViT-B/32 and ViT-L/14. However, for recent large language models, it may not be true that the additional overhead introduced by the method remains small compared to AdaMerging. These models have many more layers, and, more importantly, their feature space dimensionality is significantly larger than that of ViTs, leading to very large binary masks. Have the authors tested how the method scales when applied to some LLM with higher feature dimensionality? Is binary masking still effective in such settings? And does adding the top singular vectors remain detrimental as shown in Figure 1? 

Overall, as mentioned in the weaknesses, the proposed method mainly builds upon existing strategies (the entropy minimization, the STE estimator and the binary masks). However, since using learnable binary masks on the singular values of the network achieves good empirical results, the paper would benefit from providing stronger theoretical or empirical justification—particularly to explain the interference caused by the top singular vectors (as raised in Q1), which I consider more important than the computational and performance scalability issues discussed in Q2.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work present a novel method for model merging - Adaptive Rank Prunning (AdaRank). AdaRank learns binary mask for merging most useful components from SVD decomposition based on the test data. During the optimization, as a proxy objective the entropy minimization known from test-time adaptation methods is used. Comparing to the fixed SVD rank truncation for visual and language models, AdaRank presents significant improvement.

### Strengths
1. Clear motivation with a good analysis for backing the idea that (1) top singular components may benefit one task but harm others and (2) fixed rank truncation fails to account for task- and layer-specific complexity differences.

1. Significant improvements over AdaMerge.

### Weaknesses
1. I do not see the point of comparing AdaRank in Fig. 3 and putting it to the main paper. What we what to present show exactly - as the number of parameters with router-base methods are expected to be increasing with the number of task right? Maybe reconsider putting the number of the parameters and optimization of AdaRank in comparison to the AdaMerge for adaptation-time cost section. I found it more useful for presenting AdaRank method. 

1. Only single dynamic merging method is compared, and the work do not include more recent strong method, e.g. [1] that is static and present better results for some settings. 

1. AdaRank assumes access to the test data without labels, however, here the authors do not present any analysis about how the method performing with proposed entropy minimization with respect to the provided data (number of samples, covariance shift, etc.). 

[1] Marczak, D., Magistri, S., Cygert, S., Twardowski, B., Bagdanov, A. D., & van de Weijer, J. (2025). No task left behind: Isotropic model merging with common and task-specific subspaces, ICML 2025

### Questions
1. What is the performance of the Oracle model presented in Figure 4 in the experiments presented in Tab 1 and Tab 2?

1. Why in Table 2 we see different composition of the method {TA,TSV-M, CART} x {AdaMerging, AdaRank}?

### Soundness
3

### Presentation
3

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
This paper introduces a novel model merging framework named **AdaRank (Adaptive Rank Pruning)**. It is designed to overcome two
core limitations associated with heuristic rank selection in existing Singular Value Decomposition (SVD)-based model merging methods:
**inter-task interference** and **suboptimal fixed rank allocation**. The core contribution of AdaRank lies in its departure from the traditional strategy of retaining only the top $k$ largest singular components. Instead, it introduces a **learnable binary mask** for every singular component. To optimize this mask—thereby enabling the automatic selection of beneficial components and pruning of detrimental ones—the authors utilize unlabeled test data and formulate an unsupervised surrogate objective based on entropy minimization. Through the adaptive optimization of these masks, AdaRank effectively assigns an adaptive effective rank to different tasks and model layers, leading to a significant improvement in the performance of the multi-task merged model.

### Strengths
- **Fundamental Improvement on SVD-based Merging.** AdaRank is built upon established SVD low-rank approximation theory, but critically, it clearly articulates the limitations of traditional  Top-k heuristic truncation, namely, leading to inter-task interference and
suboptimal fixed rank allocation. The core innovation is replacing this rigid truncation with a learnable, per-component binary mask, enabling a dynamic, adaptive selection of each singular component's retention level. This change is backed by solid theoretical motivation and provides significant flexibility.
- **Practical Solution for Data Scarcity.** Addressing the common challenge of **unavailable training data** in model merging scenarios, the paper ingeniously avoids dependence on the original training set. It achieves this by employing Test-Time Adaptation (TTA) and minimizing the prediction entropy on unlabeled test data as a surrogate optimization objective. This successfully ensures zero reliance on training data while simultaneously bolstering the merged model's prediction confidence and performance.

### Weaknesses
- **Introduction of additional computational complexity and time overhead.** Compared to traditional algebraic operation-based model merging methods (e.g., Task Arithmetic, TSV-M), AdaRank introduces an additional test-time adaptation (TTA) optimization phase to learn binary masks. This transforms the merging process from an instantaneous pure algebraic operation into an iterative gradient descent procedure, resulting in significant training resource and time overhead. Furthermore, the method retains the preprocessing step of performing SVD decomposition on task vectors, which itself is a time- and memory-intensive operation when dealing with massive
weight matrices in large-scale models (e.g., LLMs), further limiting its applicability in scenarios that demand extreme efficiency.

- **Gradient approximation issues due to the discrete nature of binary masks.** Since the binary mask B is discrete, the Straight-Through Estimator (STE) strategy is employed for gradient approximation during optimization. This approach may lead to inaccurate gradient approximation and convergence to local minima.

- **Dependence on unlabeled test data limits generalizability.** Although AdaRank avoids dependence on original training data, it still
requires access to unlabeled test datasets. In certain extreme or confidential scenarios where test data is also scarce or completely
unavailable, the entropy minimization optimization step of AdaRank cannot be executed, rendering the method inapplicable. Moreover, if the test sample size is too small, it may not adequately represent the true distribution of the task, leading to suboptimal and less robust mask
configurations.

### Questions
- Does the AdaRank method maintain robust performance in extremely low-resource scenarios (where test data is very scarce)? Could the authors provide an ablation study analyzing the relationship between performance and the number of test samples used for entropy minimization? Specifically, determine the minimum sample size required for AdaRank to achieve performance comparable to or better than baseline methods under different task complexity levels.

- The core advantage of AdaRank lies in its ability to overcome the limitations of Top-$k$ truncation. What is the distribution pattern of the
learned binary mask $B$ after optimization?

- Does it still highly conform to the traditional Top-$k$ pattern, i.e., primarily selecting the $k$ components with the largest singular values, with only the value of $k$ adaptively varying across layers?

- Or can the mask $B$ genuinely perform non-contiguous, jump-wise selection of the most beneficial subset from all singular components,
thereby demonstrating that the method effectively prunes large, detrimental components while preserving small, beneficial ones?

### Soundness
3

### Presentation
3

### Contribution
3
