# Interference-Isolated Elastic Weight Consolidation and Knowledge Calibration for Incremental Object Detection

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Incremental Object Detection (IOD) enables AI systems to continuously learn new object classes over time while retaining knowledge of previously learned categories. This capability is essential for adapting to dynamic environments without forgetting prior information. Although existing IOD methods have made progress in mitigating catastrophic forgetting, they usually lack explicit and quantitative modeling of information conflicts during knowledge preservation, making task boundaries ambiguous. Such conflicts often stem from the fact that a single image can contain objects belonging to previous, present, and future tasks, where unlabeled past and future objects are often mistakenly treated as background. In this paper, we propose a novel approach grounded in Elastic Weight Consolidation (EWC) to alleviate conflict knowledge preservation caused by task interference. Specifically, we introduce the Interference Knowledge Isolated Elastic Weight Consolidation (IKI-EWC) framework for IOD, which leverages the mispredictions of the old detector on new task data to estimate task conflicts and suppresses them at the parameter level. By reformulating the Bayesian posterior of model parameters, we derive a mathematical relationship between previously learned knowledge and interference knowledge, enabling targeted elimination of conflicts during model weight updates. In addition, we also propose a prototype-based knowledge calibration (PKC) mechanism to further preserve old knowledge during the training of the objector's classification head. This method employs a learnable projection layer to compensate semantic drift in old class prototypes, and then jointly trains the classification head using both calibrated prototypes and current task features, thereby mitigating forgetting caused by classifier updates. Extensive experiments on PASCAL VOC and MS-COCO benchmarks demonstrate the effectiveness of the proposed method, outperforming state-of-the-art approaches in various settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses a key issue in incremental object detection: future classes are unlabeled in early tasks and are therefore learned as background, but later must be detected as foreground. This background conflict causes strong interference and catastrophic forgetting. The proposed framework consists of IKI-EWC, which aims to isolate and down-weight conflicting background knowledge during consolidation, and PKC, which aligns stored old-class feature prototypes to the current feature space and recalibrates the current classification head without keeping raw past images.

### Strengths
1.	The paper focuses on a real problem in incremental detection: future classes are treated as background first, then must be detected later. It proposes two modules, IKI-EWC and PKC, to keep old knowledge without storing past images.
2.	Experiments on PASCAL VOC and MS-COCO are broad, and ablations show both modules are useful.

### Weaknesses
1.	Compared with current state-of-the-art methods. 
The paper claims state-of-the-art performance relative to prior incremental detection methods, but it does not compare against recent approaches such as RGR[1] and GMDP-ABR[2], which report equal or stronger final mAP on both multi-step PASCAL VOC and MS-COCO splits. The paper should include a direct comparison to RGR and GMDP-ABR in the main tables and clearly state in which regimes the proposed method is preferable, for example, no generator cost, lower complexity, or better stability on old classes.
2.	IKI-EWC formulation clarity.
IKI-EWC is presented as deriving a clean posterior by separating non-conflicting and conflicting regions and then using this to define a new importance term for an EWC-style penalty. However, the core assumptions behind this construction, for example, proposal-level independence, using the previous model to approximate past label structure on current data, and a Laplace or Gaussian approximation, are only implicit. These assumptions should be stated explicitly in the main text where the final loss is introduced.
3.	Memory usage. 
The paper emphasizes that it does not store past images, but PKC does maintain a feature memory of sampled ROI features and Gaussian prototypes for old classes. The total storage cost of this memory, for example, the number of stored features per class, their dimensionality and total size, is not reported, and there is no quantitative comparison to exemplar replay or to generative replay methods, which also claim to avoid storing raw past data but still keep some form of replay budget.  Reporting the memory footprint would make the comparison stronger and more credible.

[1] Revisiting Generative Replay for Class Incremental Object Detection 

[2] HIGH-DIMENSION PROTOTYPE IS A BETTER INCREMENTAL OBJECT DETECTION LEARNER

### Questions
1.	Please add current state-of-the-art methods to the PASCAL VOC and MS-COCO comparisons. 
2.	Please state the explicit assumptions used in the IKI-EWC derivation in the main text.
3.	Please report the prototype buffer size and memory usage.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Authors introduce IIKC, a novel framework for Incremental Object Detection (IOD) that combines interference-aware Bayesian regularization (IKI-EWC) and Prototype-based Knowledge Calibration (PKC) to tackle catastrophic forgetting and knowledge conflict during continual learning of new object classes. IIKC identifies and isolates regions in new task data that create interference, using pseudo labels from the old detector, and recalculates parameter regularization based on both retained and conflicting knowledge. The PKC module corrects semantic drift by realigning previous class features with current ones using a learnable projection, retraining the classifier on calibrated prototypes. Experiments on PASCAL VOC and MS-COCO benchmarks show IIKC consistently outperforms state-of-the-art regularization and rehearsal methods on incremental settings, with higher mAP and reduced forgetting across more challenging task splits.

### Strengths
1) Novel Theoretical Contribution: The paper makes a meaningful theoretical contribution by reformulating Elastic Weight Consolidation (EWC) in a Bayesian framework that explicitly accounts for interference knowledge — regions where unlabeled objects from future classes are mistakenly learned as background. This provides a principled mechanism to isolate and suppress task conflicts, addressing a long-standing limitation in incremental object detection (IOD).

2) Comprehensive Framework: By integrating two complementary components — IKI-EWC for parameter-level stability and PKC for feature-level calibration — the proposed IIKC framework tackles both catastrophic forgetting and semantic drift. This dual approach effectively bridges low-level model regularization with high-level feature alignment, demonstrating thoughtful architectural design.

### Weaknesses
1) Dependence on Pseudo-Labels and Sensitivity to Noise: The IKI-EWC module is heavily dependent on pseudo-labels provided by the former detector to estimate interference regions. In the incremental object detection setting, these pseudo-labels are usually noisy, particularly for classes where the previous model has low performance. Subsequently, incorrect pseudo-labels may misidentify interference regions and subsequently impact the estimation parameter importance. The manuscript does not qualify any pseudo-label accuracy or sensitivity studies to suggest feasibility under varying degrees of noise for the IKI-EWC to evaluate stability. It is difficult to assess feasibility in more realistic scenarios without assessing effect when the pseudo-labels are not accurate. 

2) Computational Complexity and Scalability Concerns: The proposed framework has several computation-heavy procedures, including running the old model on all data for new tasks for pseudo-labels, computing interference ratios (𝑘), and estimating Fisher-based parameter importance for large networks such as Faster R-CNN. All operations can be both cost-prohibitive on memory, and time. The paper does not indicate training overhead, runtime comparison, or scalability analysis for the number of classes or incremental steps are increased (e.g., COCO 80 classes or LVIS 1200+ classes), which raises concerns for practicality of the method
3) Residual interference: Future-class objects remain unlabeled and are still at risk of being treated as background, which may not be fully resolved by the current method.

IIKC offers promising advances for incremental detection, balancing stability and plasticity, but future work could improve robustness by incorporating memory replay or better foreground-background attention mechanisms.

### Questions
1) Novelty & Conceptual Soundness: The paper introduces IKI-EWC to isolate interference knowledge in incremental object detection, but it remains somewhat unclear how this approach fundamentally differs, both theoretically and algorithmically, from prior interference-aware frameworks such as BPF (Mo et al., 2024) and GMDP (Wang et al., 2025), which also attempt to mitigate background conflicts and feature drift; could the authors provide a deeper explanation of what new insight of the reformulation and posterior correction (Eq. 10) contribute beyond existing EWC-based or distillation-based methods, and whether this formulation yields measurable theoretical guarantees or only empirical improvements?

2) Methodology & Implementation: The proposed interference isolation depends heavily on pseudo-labels generated by the previous model to identify conflicting regions; since pseudo-label quality can vary widely and introduce noise, could the authors analyze how the accuracy of these pseudo-labels impacts interference estimation, describe the computational overhead of running the old detector on all new data, and clarify how the approach scales to large datasets (e.g., COCO or LVIS) where the number of proposals and IoU computations may become prohibitively expensive?

3) Equation 2 and 3 mostly focus on Bayesian posterior? Is your framework Bayesian or leverage variational inference to build the network?

4) Overall optimization is not clear.  Equation 2 is similar to EWC. The only difference is you replaced diagonal Fisher matrix with equation 12?. And, section 3.4: Equation 14 is just knowledge distillation on features? How this norm is helping for knowledge calibration? The total loss is combination of these two loss?

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
4

### Summary
The authors propose IIKC, a two-part framework for Incremental Object Detection (IOD): (1) Interference-Knowledge-Isolated Elastic Weight Consolidation leverages the old model’s mispredictions on new-task data to eliminate interference caused and rebuild the Bayesian posterior and parameter importance; and (2) Prototype-based Knowledge Calibration applies a learnable linear projection to compensate for semantic drift of old-class prototypes and then jointly retrains the classification head with current features. The approach outperforms strong baselines—covering both no-rehearsal and small-exemplar rehearsal regimes—across multiple stepwise and multi-step protocols on PASCAL VOC and MS-COCO.

### Strengths
1. The proposed IKI-EWC internalizes the IOD-specific “future-class- background” interference into computable posterior correction and importance fusion, yielding an end-to-end implementable path for conflict isolation. Compared with heuristic reweighting or soft masking, it offers an explicit probabilistic formulation with closed-form solutions (Eqs. 10 and 12).Which provides a systematic extension of EWC (new decomposition, a quantified $k$ , and a new fused importance $\tilde{I}$ ) that is both theoretically grounded and engineering-ready.

2. PKC addresses semantic drift using a lightweight projection together with prototype-based retraining, with small overhead and yielding clear gains (as confirmed by ablations). Moreover, compared with certain dual-teacher distillation schemes, the proposed method is simpler in both computation and implementation.

3. The paper is well structured and clearly written. Figures 1 and 2 intuitively illustrate the core idea and overall framework, enabling rapid understanding. It also provides an overall framework diagram and complete training pseudocode—explicitly specifying the input parameters γ, Top-K, and λ—which facilitates re-implementation and comparative ablation studies.

4. The evaluation experiments are thorough and in-depth, the results show consistent gains over strong baselines on VOC/COCO across stepwise/multi-step protocols. The authors analyze show low sensitivity to γ and that computed k beats extreme settings, and report runtime/memory costs.

### Weaknesses
1. The core of IKI-EWC is to accurately estimate interference regions, a procedure that relies entirely on pseudo-labels generated by the previous model ($M_{t-1}$ ) on the new data. If $M_{t-1}$ has degraded in performance or exhibits prediction bias, the interference estimation may be inaccurate, thereby affecting the overall performance of the framework. Although the paper notes this in the limitations section, it does not experimentally analyze how sensitive the method is to pseudo-label quality as a function of  $M_{t-1}$ performance.

2. The statistical robustness of the relative mass $k$ is not demonstrated. Since $k$  is computed as a ratio of proposal counts, it is highly sensitive to the IoU threshold, the number of proposals, class long-tail effects, and scale distributions. When sample size or the positive/negative balance fluctuates across stages, the variance of  $k$  can become large, causing the importance III to oscillate excessively. The paper lacks a systematic report of confidence intervals and sensitivity analysis for $k$ .

3. Evidence for the “approximate independence/orthogonality” assumption is limited. The derivations assume that “clean historical data” and “current data” are approximately independent, and cite an early-training gradient angle of ≈90° as support. However, orthogonality is not independence, and early mini-batches do not characterize the entire training trajectory. If representations become increasingly coupled later on—especially in multi-object/multi-scale settings—the premise underlying the posterior reconstruction weakens, potentially biasing the direction of the regularization. This point requires more extensive discussion.

4. PKC’s “prototype + linear projection” design is simple; however, compared with EFC’s anisotropic constraints with Gaussian-prototype updates, and LDC’s label-free learnable drift compensation, it might show clear shortcomings in the granularity of drift modeling, robustness and statistical sufficiency.

5. Several implementation details are insufficiently specified—for example, the confidence threshold for pseudo-labels, the method and computational cost of Hessian/Fisher estimation, and the sampling procedure used in PKC. In addition, it is not fully transparent whether data augmentation and preprocessing are exactly matched to the strong baselines in both the rehearsal-free and rehearsal-based regimes.

[1] Elastic Feature Consolidation for Cold Start Exemplar-Free Incremental Learning, ICLR 2024.
[2] Exemplar-free Continual Representation Learning via Learnable Drift Compensation, ECCV 2024.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the interference of knowledge between past/present knowledges in Incremental dense prediction problems. To mitigate this problem, the authors conduct quantative modeling based on bayesian analysis and propose novel algorithms called IKI-EWC. Also, the authors propose prototype based classifier correction algorithm to prevent the drift at the classfier level

### Strengths
- The authors well analysed the existing problem and then conducted rigorous mathematical analysis on that. This makes the motivation of the proposed method very strong
- The paper is well written and easy to follow. 
-This paper provides extensive experimental results that can empirically support the authors' claim as well

### Weaknesses
- Authors approximated non-interference knowledge only by using current data. I agree that this would be the best/practical way of doing it under this setting but I wonder does it actually approximate the goal well? Is there any way of computing ground truth goal even by using whole ground data? 

- I wonder why the authors used parameter regularisation methods. It is generally know to be poor compared to distillation methods. Can we use the IKI concept for distillation based methods as well?

- Prototype based classifier retain is quite common concept in classification CIL methods. Is it entirely novel in dense prediction tasks?

- Doesn't not regarding IKI concept on PKC levels conflicts with the other module (IKI-EWC)?

- Although the motivation is quite strong, improvements look marginal, especially when each module is solely applied.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
