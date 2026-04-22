# REALIGN: Regularized Procedure Alignment with Matching Video Embeddings via Partial Gromov-Wasserstein Optimal Transport

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 6, 4

## Abstract
Learning from procedural videos remains a core challenge in self-supervised representation learning, as real-world instructional data often contains background segments, repeated actions, and steps presented out of order. Such variability violates the strong monotonicity assumptions underlying many alignment methods. Prior state-of-the-art approaches, such as OPEL and RGWOT, leverage Kantorovich Optimal Transport (KOT) and Gromov-Wasserstein Optimal Transport (GWOT) to build frame-to-frame correspondences but operate only on local feature similarity and pairwise relational structure, without explicit temporal priors, which limits their ability to capture the higher-order temporal structure of a task. In this paper, we introduce **REALIGN**, an unsupervised framework for procedure learning based on *Regularized Fused Partial Gromov-Wasserstein Optimal Transport* (R-FPGWOT). In contrast to RGWOT, our formulation jointly models visual correspondences and temporal relations under a partial alignment scheme, enabling robust handling of irrelevant frames, repeated actions, and non-monotonic step orders common in instructional videos. To stabilize training, we integrate FPGWOT distances with inter-sequence contrastive learning, avoiding the need for multiple regularizers and preventing collapse to degenerate solutions. Across egocentric (EgoProceL) and third-person (ProceL, CrossTask) benchmarks, REALIGN achieves up to **18.9\% (7.62pp)** average F1-score improvements and over **30\% (7.74pp)** temporal IoU gains, while producing more interpretable transport maps that preserve key-step orderings and filter out noise.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes REALIGN, a self-supervised framework for learning procedural representations from instructional videos using Regularized Fused Partial Gromov–Wasserstein Optimal Transport (R-FPGWOT). The key idea of REALIGN is to jointly model semantic and temporal relationships while allowing partial correspondences to handle irrelevant or repeated segments. Across multiple benchmarks, REALIGN achieves significant improvements based on F1 and IoU.

### Strengths
- The proposed REALIGN framework effectively integrates the semantic matching capability of Kantorovich OT, the structural consistency of Gromov–Wasserstein OT, and a relaxed partial-transport formulation. It demonstrates strong empirical performance across both egocentric and third-person instructional video datasets, even surpassing some multimodal methods despite relying solely on visual inputs.

- The unified alignment loss is well-designed to stabilize training and prevent degenerate matching. The ablation studies clearly demonstrate the contribution and necessity of each component in the framework.

- The paper includes thorough analyses of key factors and hyperparameters (Section 5), providing valuable insights into the model’s robustness and further supporting the empirical findings.

### Weaknesses
- The paper does not clearly explain the relative scales of the three loss terms in Equation 10, how their coefficients are chosen, or how sensitive the overall performance is to these hyperparameters.

- While the framework is technically solid and empirically effective, the core methodological novelty is somewhat limited. The work primarily represents a strong engineering refinement and integration of existing optimal transport formulations rather than a fundamentally new theoretical contribution.

### Questions
Please see weaknesses.

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
4

### Summary
*   The paper introduces REALIGN, a self-supervised framework designed to robustly learn procedures from noisy videos.
*   REALIGN utilizes R-FPGWOT, a formulation that jointly models visual correspondences and temporal relations while relaxing balanced constraints to allow partial alignment.
*   This partial alignment, supported by the use of a virtual frame sink, enables the robust exclusion of irrelevant background frames and redundant segments, addressing limitations where fully balanced OT methods often fail.
*   The overall objective integrates the R-FPGWOT distance with temporal smoothness priors, structural regularization, and inter-sequence contrastive learning to stabilize training and achieve substantial performance gains, reaching up to 18.9% F1-score improvements and over 30% temporal IoU gains across standard benchmarks (EgoProceL, ProceL, CrossTask).

### Strengths
The strength of the paper is the introduction of REALIGN, which jointly models visual correspondences and temporal relations. 
A key methodological advantage is the use of a flexible partial alignment scheme, achieved by relaxing balanced assignment constraints, which enables the model to handle irrelevant background frames, repeated actions, and non-monotonic step orders common in real-world videos, successfully addressing scenarios where prior fully balanced Optimal Transport methods typically fail. 

Empirically, REALIGN demonstrates performance gains, achieving 18.9% average F1-score improvements and 30% temporal IoU gains across standard egocentric (EgoProceL) and third-person (ProceL, CrossTask) benchmarks, while also proving to be data-efficient compared to competing models.

### Weaknesses
1.  The model's performance is critically reliant on a carefully selected number of key steps ($K$). Ablation studies show that setting $K$ too high (e.g., 10 or 15) leads to sharp performance drops because the segmentation fragments continuous actions and introduces temporal jitter. This leads the reader to question how robust the model is to various settings and tasks.

2.  The accurate discovery and localization of key steps are not achieved solely by the Optimal Transport (OT) alignment but require a necessary and complex post-processing step utilizing multi-label graph cut segmentation solved via the $\alpha$-Expansion algorithm. Ablation studies explicitly demonstrate that simpler, alternative clustering methods (like K-Means and Subset Selection) consistently and severely underperform, emphasizing that the overall pipeline is highly dependent on this external structured clustering technique.

3.  The formal mathematical proofs for the monotonic decrease of the objective rely on the intra-sequence structure matrices ($C_x, C_y$) either being symmetric Positive Semidefinite (PSD) (Option A, e.g., using kernelized structure) or requiring a local quadratic majorizer (Option B), thereby limiting the flexibility in using raw temporal distances if full convergence guarantees are desired.

4.  Although the approach is highly effective using only visual (RGB) frames, this limitation means that it cannot fully utilize richer sources of supervision available in egocentric contexts. The comparison tables show that the multimodal approach STEPS, which uses depth and gaze, achieves a slightly higher F1 score on the demanding EPIC-Tents dataset, indicating that restricting the input to visuals may prevent REALIGN from reaching the absolute performance ceiling on certain tasks.

### Questions
1.  Since the key-step localization relies critically on a separate multi-label graph cut segmentation (using $\alpha$-Expansion), which dramatically outperforms simpler clustering methods like K-Means and Subset Selection (Table 4), could the authors provide the specific mathematical formulation of the graph cut energy function, particularly detailing the design of the T-links and N-links that enforce temporal smoothness and structural consistency?

2.  Given the inherent subjectivity of the number of key steps ($K$) and the observation that increasing $K$ beyond the optimum ($K=7$) leads to sharp performance drops due to fragmentation (Table 5), have the authors investigated methods to dynamically learn or infer the optimal value of $K$ during training, rather than treating it as a fixed hyperparameter determined empirically?

3.  The rigorous convergence proof of the R-FPGWOT framework (Option A) relies on using kernelized temporal structure matrices ($C_x, C_y$) that are Symmetric Positive Semidefinite (PSD). Could the authors justify the empirical choice of the Laplace kernel over other standard PSD kernels (like the Gaussian kernel), and discuss the model's robustness or sensitivity to the specific scale parameter ($b$) used in the Laplace prior?

4.  The ablation study highlights that the partial penalty ($\tau$) is the most important component for performance gains (Table 3). How were the critical parameters governing partial alignment, specifically the marginal constraint penalty ($\tau$) and the virtual frame assignment threshold ($\zeta$), calibrated to optimally distinguish between noise/background frames and essential procedural steps across diverse datasets?

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
5

### Summary
The paper tackles procedure learning from instructional videos through video alignment. The key novelty is the introduction of a Optimal-Transport-based method that can handle both "background frames" and out-of-order steps in during the alignment procedure. This is achieved by combining Kantorovich-and Gromov-Wasserstein OT. The resulting alignment can be regularized and used as a loss function to train representations. The resulting model shows SoTA performance on video alignment.

### Strengths
* A single framework that can handle varying order of procedure steps and “background frames” simultaneously.
* SoTA performance on instruction step localization
* Technically involved and sound method

### Weaknesses
* The model looks over-engineered - to solve the task it combines OT objective, temporal smoothness, inter-video contrastive learning, and a dedicated inference procedure.
* The entire method looks brittle because of how many regularizers are needed to make it work. For example, despite Gromov-Wasserstein formulation already taking care of the temporal similarities, the authors add a temporal smoothness regularizer to make things work.
* The inference procedure can not be performed by the learned model alone and it uses graph-cut, which introduces discrepancy between training and inference. This is not desirable, as has been demonstrated in [a]

[a] Dvornik et al., Drop-DTW: Aligning Common Signal Between Sequences While Dropping Outliers

### Questions
* Why do the authors need temporal smoothing prior if the Gromov-Wasserstein formulation takes the position into account? Is the second one not sufficient alone?
* Is it reasonable to assume that a single “background frame” will catch all the frames that are not “foreground”?

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
3

### Summary
This paper extends Regularized Gromov-Wasserstein Optimal Transport (GWOT) [Mahmood 2025] to incorporate background segments.
The extension is done by adding a virtual frame with a threshold for alignment target videos. Inserting such dummy elements with a threshold is a common technique to allow unmatched elements in assignment problem. To implement this with GWOT, the method relaxes the strict constraint of $T1=\alpha$ and $T^T1=\beta$ of GWOT in the fourth term in Eq. (2). The experimental results show improved alignment performance in CMU-MMAC, EGTEA-GAZE+, ProceL, and CrossTask, while maintaining the performance on MECCANO, EPIC-Tents, PC Assembly, and PC Disassembly datasets.

### Strengths
- The proposed method significantly improves alignment performance on some datasets while maintaining performance on others.
- The core idea of the partial alignment schema and its formulation is technically sound.

### Weaknesses
1. The presentation is hard to follow.

The introduction describes the proposed method (R-FPGWOT) as an update from KOT rather than RGWOT, making it difficult to understand the paper's contribution against RGWOT.  Additionally, Figure 1 highlights GWOT's achievements against KOT, giving the impression that it proposes RGWOT. The actual contribution of this paper is difficult to discern. The figure should highlight the differences between traditional GWOT and the proposed R-FPGWOT, and visualize "(c) redundant frames" mentioned in the captions.

It is also difficult to follow the experimental discussion, as many score improvement claims are hard to verify without guidance from the table.
The authors should provide guidance on how the claims are calculated from the table's data or specify the exact percentage points (as they are obvious from the table data). Similar difficulties exist in L27, L356, L369, L370, and L481.

2. Inconsistency between method design and experimental results.

This reviewer understood that the method allows for background tasks that do not match segments in the opposite video.


This reviewer understood that the method relaxes RGWOT to allow background task that does not match to any segments in the opposite side video. However, there lacks discussion on dataset imbalance in score-improved datasets. For example, the same person performs food preparation in an identical CMU-MMAC kitchen and following the same recipe for each dish, suggesting a balanced dataset. Thus, the dataset is assumed highly balanced, yet performance improves significantly. A deeper analysis is necessary to enhance the reliability of the explanation of why the proposed method works better than the baselines.

### Questions
1. Is there any analysis to identify which datasets the proposed method improves and which do not benefit?

### Soundness
3

### Presentation
1

### Contribution
2
