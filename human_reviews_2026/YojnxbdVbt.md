# Mixing Expertise with Confidence: A Mixture of Expert Framework for Robust Multi-Modal Continual Learner

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
The Mixture of Experts (MoE) framework is widely used in continual learning to mitigate catastrophic forgetting. MoEs typically combine a small inter-task shared parameter space with largely independent expert parameters. However, as the number of tasks increases, the shared space becomes a bottleneck, reintroducing forgetting, while fully independent experts require explicit task ID predictors (e.g., routers), adding complexity. In this work, we eliminate the inter-task shared parameter space and the need for a task ID predictor by enabling expert communication and allowing knowledge to be shared dynamically, akin to human collaboration. We bridge the inter-expert knowledge sharing by leveraging the open-set learning capabilities of a multimodal foundation model (e.g., CLIP), thereby providing “expert priors” that bolster each expert’s task-specific representations. Guided by these priors, experts learn calibrated inter-task posteriors. Additionally, Multivariate Gaussians over the learned posteriors promote complementary specialization among experts. We propose new evaluation benchmarks that simulate realistic continual learning scenarios, and our prior-conditioned strategy consistently outperforms existing methods across diverse settings without relying on reference datasets or replay memory.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a Mixture-of-Experts (MoE) framework for multimodal continual learning (CL), specifically targeting the challenges of catastrophic forgetting when using foundation models like CLIP. The core ideas involve:

1.	Eliminating shared parameters between experts to prevent negative interference (like shortcut features propagating).

2.	Introducing "expert communication" via confidence-aware temperature scaling, where predictions from previous experts guide the training of the current expert.

3.	Using the multimodal foundation model (CLIP) to provide priors and calibrate expert posteriors.

4.	Employing multivariate Gaussians over embeddings and Mahalanobis distance (MD) during inference for distribution-aware expert weighting, aiming to improve task identification without an explicit router.

The method is evaluated on several benchmarks, including standard CL datasets (CIFAR-100, TinyImageNet), subpopulation shift datasets (BREEDS), and cross-domain tasks (X-TAIL), reporting improvements over various baselines.

### Strengths
1. The paper tackles important and timely problems in CL: applying CL to multimodal foundation models and addressing limitations of standard MoE architectures (shared parameter bottlenecks, reliance on routers).

2. The combination of confidence-aware temperature scaling for implicit inter-expert communication and the distribution-aware MD weighting for inference are interesting mechanisms.

3. The authors evaluate their method across a good range of CL settings (CIL, subpopulation shift, cross-domain), providing a relatively thorough empirical analysis compared to some theoretical papers.

### Weaknesses
1. While the specific mechanisms (temperature scaling based on prior expert confidence, MD weighting) are implemented in a novel combination, the underlying ideas (using confidence, modeling distributions for OOD/task-ID) don't feel particularly fundamental or surprising. The approach seems more like an intricate engineering solution built upon existing concepts rather than introducing a new core principle for multimodal CL.

2. Across several experiments (e.g., Table 1 CIL), the reported improvements over strong, recent baselines like RAIL or MoE-Adapters, while consistent, often appear marginal (e.g., ~1-4% on Avg/Last accuracy). Given the complexity of the proposed mechanisms, the practical benefit seems somewhat limited, especially compared to the zero-shot baseline or simpler fine-tuning approaches in some cases.

3. The method seems tightly coupled to the specifics of CLIP (leveraging its open-set capabilities, visual embeddings for Gaussian modeling). It is unclear how readily this approach would generalize to other multimodal architectures or different types of continual learning problems (e.g., non-classification tasks, different modalities like audio-visual). The reliance on semantic similarity signaled by CLIP priors might not hold for tasks with less semantic overlap.

4. Proposition 1 and Theorem 1 attempt to provide theoretical backing for the temperature scaling. However, the connection felt somewhat post-hoc, and the practical implications derived from this theory weren't entirely clear or rigorously tested (e.g., how the confidence gain relates to actual forgetting reduction).

### Questions
1. Could the authors elaborate on the conceptual leap their method provides beyond combining confidence measures and distributional modeling, which are known techniques? What makes this approach fundamentally different or more powerful?

2. The performance gains over recent methods like RAIL or MoE-Adapters are often small. How do the authors justify the added complexity of their proposed mechanisms (temperature scaling, Gaussian modeling, MD weighting) for these marginal gains? Are there specific scenarios or task types where a much larger benefit is expected?

3. How dependent is the proposed method on the specific properties of CLIP? Could this framework be applied, for instance, to continually learn audio classification tasks using an audio foundation model, or does it rely intrinsically on the vision-language pre-training and embedding space structure?

### Soundness
2

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
2

### Summary
This paper addresses the problem of catastrophic forgetting in continual learning within the Mixture of Experts framework. The authors identify that shared parameters become a bottleneck and that explicit task routers add complexity. They propose a method using a frozen multimodal foundation model where each task is learned by an independent, lightweight adapter. The core idea is twofold: during training, it uses confidence scores from past experts to apply an adaptive temperature scaling to the loss for ambiguous samples, forcing the new expert to learn more discriminative features. During inference, it avoids a router by calculating a sample-specific expert weighting based on the Mahalanobis distance to each expert's learned data distribution. Results are strong across varied benchmarks, and the approach of removing shared components is a practical direction.

### Strengths
-   The paper targets a well-defined and practical problem in continual learning. The issues of shared parameter interference and the complexity of routers in MoE models are significant hurdles.
-   The proposed method is clearly explained and appears to be relatively straightforward to implement. Using prior expert confidence to guide temperature scaling is an interesting mechanism for implicit knowledge transfer.
-   The experiments are thorough and the presentation is easy to follow.

### Weaknesses
-   The technical novelty of the individual components seems somewhat limited. Both temperature scaling for confidence calibration and Mahalanobis distance for out-of-distribution detection or expert gating have been explored in related fields. The main contribution lies in their specific combination and application within this router-less, parameter-isolated MoE framework.
-   The theoretical justification provided in Proposition 1 and Theorem 1 feels a bit abstract. It is not entirely clear how much of the performance gain is attributable to the formal "epistemic prior" versus the heuristic of simply up-weighting samples that previous experts found confusing. The proof sketch also relies on assumptions, such as stable posterior means, which may not always hold.
-   The method introduces a key hyperparameter, the percentile threshold (set to 90th in the appendix) used to identify the "open set error" samples for temperature scaling. The paper does not seem to include a sensitivity analysis for this threshold, which appears critical for defining how many and which samples are re-weighted.
-   The paper does not discuss the practicalities of computing the inverse covariance matrix for the Mahalanobis distance in Equation 7. CLIP embeddings are high-dimensional, which can make this computation numerically unstable and resource-intensive. This is an important implementation detail that affects feasibility.

### Questions
-   Could the authors elaborate on how this specific combination of temperature scaling and MD-based weighting is fundamentally different from prior works that might use similar components, for instance, using MD for task-ID prediction or temperature for calibration?
-   Would it be possible to provide more intuition or perhaps a visualization to support the claim in Theorem 1? For example, showing how the predictive variance or feature representations change for shortcut-activated samples after the temperature scaling is applied.
-   How sensitive is the model's performance to the choice of the percentile threshold (set to 90th) used to define the high-confidence set in Equation 3? What happens if this value is set much lower or higher?
-   How is the inverse covariance matrix for the Mahalanobis distance computed? Given the high dimensionality of CLIP embeddings, is there a regularization technique or a simplification, such as assuming a diagonal covariance, used to ensure stability and efficiency?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel method that eliminates the need for task IDs and shared parameters, and uses temperature scaling for expert communication. 

Contributions:

1. Novel confidence-based expert framework
2. Theoretical analysis under certain assumptions and approximations
3. Expert communication enhanced by a distribution-aware weighting mechanism

### Strengths
1. Reveals the observation that independent experts can be overconfident in OOD data (Figure 1)

2. Novel approach to expert coordination. This paper uses previous experts' confidence to calibrate new expert training.

3. The experiment is comprehensive, across multiple baselines (e.g., 8 CLIP-backbone-based baselines).

### Weaknesses
1. Computation complexity of confidence. Equations (3) and (4) require computing the feed-forward for all current training samples through all previous experts. When training on $D^t$, this requires $(t-1)|D^t|$ forward passes. 

2. The experiment comparison has different computation budgets, such as training cost and inference time. It can be helpful to involve the discussion of the FLOPs or feed-forward counts.


Minor concerns:

1. Each task is modeled with a single multivariate Gaussian, which may not capture multi-modal distributions, as the authors discussed in the introduction. A single Gaussian averages over these modes, potentially degrading OOD detection. The authors should justify this simplification or compare it against per-class Gaussian mixture models.

2. Results only contain the mean results without necessary statistics such as standard deviation or error.

### Questions
Refer to weakness

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel Mixture of Experts (MoE) framework for multimodal continual learning, which eliminates inter-task shared parameters and task-ID predictors by introducing dynamic expert communication via confidence-aware temperature scaling and Mahalanobis distance-based weighting. Leveraging multimodal foundations like CLIP, the method aims to mitigate catastrophic forgetting while maintaining scalability without replay buffers or reference datasets. Experimental results on benchmarks such as CIFAR-100, TinyImageNet, and BREEDS demonstrate superior performance in class-incremental, domain-incremental, and subpopulation-shift settings compared to existing baselines.

### Strengths
1. The method effectively removes shared parameters between experts, reducing catastrophic forgetting by enabling dynamic communication through temperature scaling.
2. It incorporates multimodal foundations like CLIP to provide robust expert priors, enhancing performance in diverse continual learning scenarios.
3. Comprehensive experiments across class-incremental, subpopulation-shift, and cross-domain benchmarks validate the framework's superiority over existing baselines.
4. Ablation studies rigorously justify the contributions of individual components, such as temperature scaling and Mahalanobis distance weighting.

### Weaknesses
1. The experiments rely heavily on a fixed ViT-B/16 based CLIP backbone, limiting insights into scalability and generalization across larger architectures or varied training parameters.
2. Literature review on expert-based methods in continual learning could be broadened with discussions of approaches like hierarchical expert designs [1] or other theoretical analyses [2] [3].

[1] HiDe-PET: Continual Learning via Hierarchical Decomposition of Parameter-Efficient Tuning

[2] A Theoretical Study on Solving Continual Learning

[3] Theory on Mixture-of-Experts in Continual Learning

### Questions
See the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3
