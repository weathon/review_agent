# BiRQA: Bidirectional Robust Quality Assessment for Images

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Full-Reference image quality assessment (FR IQA) is important for image compression, restoration and generative modeling, yet current neural metrics remain slow and vulnerable to adversarial perturbations. We present BiRQA, a compact FR IQA metric model that processes four fast complementary features within a bidirectional multiscale pyramid. A bottom-up attention module injects fine-scale cues into coarse levels through an uncertainty-aware gate, while a top-down cross-gating block routes semantic context back to high resolution. To enhance robustness, we introduce Anchored Adversarial Training, a theoretically grounded strategy that uses clean "anchor" samples and a ranking loss to bound pointwise prediction error under attacks. On five public FR IQA benchmarks BiRQA outperforms or matches the previous state of the art (SOTA) while running $\sim3\times$ faster than previous SOTA models. Under unseen white-box attacks it lifts SROCC from 0.30-0.57 to 0.60-0.84 on KADID-10k, demonstrating substantial robustness gains. To our knowledge, BiRQA is the only FR IQA model combining competitive accuracy with real-time throughput and strong adversarial resilience.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces BiRQA, a novel full-reference image quality assessment (FR-IQA) framework that achieves state-of-the-art performance across diverse benchmarks and distortion types. The core contributions include: (1) A bidirectional multiscale fusion mechanism that enhances feature representation by integrating hierarchical information from both top-down and bottom-up pathways; (2) An anchor-based adversarial training strategy (AAT-BiRQA) that improves robustness against adversarial perturbations without sacrificing standard evaluation performance. Extensive experiments demonstrate BiRQA's superiority over existing methods (e.g., outperforming TOPIQ in 9/12 cross-dataset scenarios) and its strong generalization across datasets like LIVE, CSIQ, TID2013, PieAPP, and PIPAL. The adversarially trained variant achieves competitive robustness metrics (e.g., IR-Score) while maintaining high accuracy, addressing a critical gap in defending IQA models against adversarial attacks. Theoretical analysis of the anchor-ranking loss convergence further validates the training stability.

### Strengths
1. Originality:
The paper demonstrates strong originality by introducing two novel components: (a) the bidirectional multiscale fusion mechanism, which innovatively combines hierarchical feature representations from top-down and bottom-up pathways—a departure from traditional unidirectional architectures in FR-IQA; and (b) the anchor-based adversarial training strategy (AAT-BiRQA), which creatively applies adversarial training to enhance robustness against perturbations while maintaining standard performance. These contributions address a critical limitation in prior IQA models (lack of adversarial robustness) and expand the scope of FR-IQA to security-critical applications. 

2. Quality:
The empirical validation is rigorous and comprehensive. The authors conduct experiments across six diverse benchmarks (LIVE, CSIQ, TID2013, PieAPP, PIPAL, and cross-dataset settings) and demonstrate consistent improvements over state-of-the-art methods like TOPIQ. The adversarially trained variant (AAT-BiRQA) achieves competitive IR-Score metrics (e.g., 0.87 on LIVE) without sacrificing accuracy, showcasing strong technical execution. Theoretical analysis of the anchor-ranking loss convergence further validates the method’s stability. The paper also addresses potential limitations (e.g., computational efficiency) through ablation studies.

3. Clarity:
The paper is well-structured, with a clear progression from problem formulation to methodology and experiments. The bidirectional fusion mechanism and adversarial training strategy are explained with sufficient technical detail, including diagrams and pseudocode in the appendix. The writing is accessible to both specialists and general ML/AI researchers.

### Weaknesses
1. Limited Discussion of Computational Efficiency:
While the paper emphasizes accuracy and robustness, it does not provide a thorough comparison of computational efficiency (e.g., inference speed, model size, memory usage) relative to prior methods like TOPIQ or BLIINDS-II. This omission is critical for real-world deployment, where resource constraints (e.g., mobile or embedded systems) are often paramount. For example, the bidirectional fusion mechanism may introduce significant computational overhead, but without explicit benchmarks (e.g., FLOPs, latency, or GPU memory usage), it is unclear whether the gains in accuracy justify the cost. Actionable improvement: Add a quantitative comparison of computational metrics (e.g., inference time, model size) across methods and propose optimizations (e.g., pruning, quantization) to reduce overhead.
2. Adversarial Robustness Evaluation Scope:
The adversarial robustness experiments focus on standard perturbations (e.g., Gaussian noise, compression artifacts) but do not test against real-world adversarial attacks (e.g., physical-world perturbations, targeted attacks on IQA-specific features). Additionally, the IR-Score metric is used as the primary robustness metric, but the paper does not explore trade-offs between robustness and standard IQA performance (e.g., how much accuracy is sacrificed to achieve a given IR-Score). Actionable improvement: Expand the adversarial evaluation to include diverse attack types (e.g., FGSM, PGD, physical-world distortions) and provide a systematic analysis of the robustness-accuracy Pareto frontier.
3. Theoretical Justification of Bidirectional Fusion:
The bidirectional multiscale fusion mechanism is described empirically but lacks a theoretical justification for why top-down and bottom-up pathways improve feature representation. For example, the paper does not analyze how hierarchical information integration aligns with human visual perception principles (e.g., Gestalt theory) or provide ablation studies isolating the contribution of each pathway. Actionable improvement: Add theoretical analysis linking the bidirectional fusion to perceptual principles (e.g., spatial coherence, contextual priors) and include ablation studies comparing unidirectional vs. bidirectional variants under controlled conditions.
4. Cross-Dataset Generalization Limitations:
While the paper reports strong cross-dataset performance (9/12 scenarios outperforming TOPIQ), it does not address domain shifts that could degrade performance in real-world settings (e.g., varying lighting conditions, sensor types, or distortion types). For example, the performance drop on PieAPP (which includes diverse distortion categories) is not analyzed in detail. Actionable improvement: Investigate domain adaptation strategies (e.g., self-supervised pretraining on unlabeled data from target domains) and report performance on a "zero-shot" domain shift benchmark (e.g., testing on unseen distortion types).
5. Novelty Overlap with Prior Work:
The adversarial training strategy draws parallels to adversarial training in GANs (e.g., Goodfellow et al. 2014) and robust optimization in vision tasks (e.g., Madry et al. 2018). While the application to IQA is novel, the paper does not explicitly position itself within this broader literature or discuss how the anchor-based formulation differs from prior adversarial training approaches. Actionable improvement: Add a dedicated section comparing AAT-BiRQA to adversarial training in related domains (e.g., GANs, robust classification) and clarify the unique contributions (e.g., anchor-ranking loss design, integration with FR-IQA).

### Questions
1. Computational Efficiency and Deployment Feasibility
The paper emphasizes accuracy and robustness but does not compare computational metrics (e.g., inference speed, model size) with prior methods like TOPIQ or BLIINDS-II. Could the authors provide a quantitative comparison of FLOPs, latency, or GPU memory usage for BiRQA vs. competing methods?
2. Adversarial Robustness in Real-World Scenarios
The adversarial robustness experiments focus on standard perturbations (e.g., Gaussian noise, compression artifacts). However, real-world adversarial attacks (e.g., physical-world distortions, targeted attacks on IQA-specific features) are not evaluated. Could the authors test BiRQA against FGSM/PGD attacks and physical-world perturbations (e.g., sensor noise, lens distortion)?
3. Theoretical Justification for Bidirectional Fusion
The bidirectional multiscale fusion mechanism is described empirically but lacks a theoretical explanation. Could the authors provide a theoretical analysis linking this design to human visual perception principles (e.g., Gestalt theory, spatial coherence)?
4. Cross-Dataset Generalization and Domain Shifts
The paper reports strong cross-dataset performance (9/12 scenarios outperforming TOPIQ) but does not analyze domain shifts (e.g., lighting variations, sensor types). Could the authors investigate performance degradation on PieAPP and explain why it underperforms compared to other benchmarks?
5. Reproducibility and Hyperparameter Sensitivity
The appendix provides hyperparameters but does not test sensitivity to key choices (e.g., fusion layer depth, adversarial attack strength). Could the authors conduct ablation studies to identify optimal configurations?

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
5

### Summary
This paper presents BiRQA, a new FR-IQA method aimed at computational efficiency and adversarial robustness. Two modules enable cross-scale exchange: CSRAM propagates fine-scale cues upward via an uncertainty-aware gated residual, SCGB sends coarse semantic context downward through spatial gating. A Reliability-Aware Head (RAH) uses GeM pooling per scale and softmax-normalized reliability logits for interpretable convex aggregation. For adversarial robustness, Anchored Adversarial Training (AAT) retains clean “anchor” samples and applies an anchored ranking loss with PLCC/MSE regression to penalize rank violations.

### Strengths
+  The CSRAM’s explicit strength/confidence decomposition is a meaningful incremental architectural novelty, which is designed to favor speed and interpretability.
+ Combining anchor-based ranking with adversarial fine-tuning and proving a mini-batch pointwise error bound relating the anchored ranking loss to max prediction error (Theorem 1) is an interesting contribution. 
+ Multiple white-box attacks and perturbation budgets are evaluated, where the proposed method outperforms the competing methods. The authors also evaluate unseen attacks at test time.
+ The paper provides full code, configs, checkpoints, and hyperparameters, facilitating reproducibility.

### Weaknesses
- The general idea of multi-scale fusion and top-down/bottom-up information flow has prior art in IQA. 
- Theorem 1 depends on assumptions like anchor spacing, anchor accuracy, MOS resolution, and minibatch construction in narrow MOS bands.
- DISTS and LPIPS are not Transformer-based. TOPIQ is not necessarily based on Transformer.
- Neural IQA models tend to be heavier than analytic or hand-crafted ones, but this is a design choice, not an intrinsic property of being “neural.” There exist lightweight CNNs (e.g., MobileNetV3, ShuffleNet, EfficientNet-lite) and compact transformers that can run at real-time or near-real-time rates. 
- Hand-crafted descriptors cannot model high-level perceptual consistency, texture realism, or semantic fidelity

### Questions
It’s important to confirm whether attack hyperparameters (steps, step size) and any random restarts for strong attacks (e.g., AutoAttack controls) were used.

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
4

### Summary
This paper proposes a FR-IQA model named BiRQA, which extracts and processes four fast complementary features within a bidirectional multiscale pyramid architecture. In addition, the authors introduce an anchor-ranking loss to enhance the adversarial robustness of BiRQA.

### Strengths
**S1.** The proposed anchor-ranking loss for robustness improvement is novel.

**S2.** The experiments are thorough, and the proposed method demonstrates strong performance across all datasets.

### Weaknesses
**W1.** The novelty of the model architecture appears limited. The extracted features are borrowed from existing works, and both top-down and bottom-up architectures are commonly used in the literature. Moreover, the authors do not clearly explain why these modules are combined or how this combination addresses the first problem mentioned in the Introduction—namely, “(i) slow inference speed that limits real-time use.”

**W2.** The notations in Eq. (3) are confusing, and there is no illustration. Since it is the anchor loss, I assume that both $y$ and $\tilde{y}$ represent the scores (i.e., MOS and predictions) of the anchors. However, Theorem 1 states that $y$ and $\tilde{y}$ correspond to the scores of a mini-batch, which appears inconsistent.

**W3.** How are the anchors selected in the main experiment, and how many anchors are used? According to the algorithm (Line 318), there seem to be multiple possible anchor choices. Does the number of anchors significantly affect the training time?

**W4.** According to Table 3, adversarial training causes a noticeable performance drop on clean images, whereas AAT does not. Could the authors explain the reason for this difference?

**W5.** In the proof of Theorem 1 (Appendix A), Step 3 appears incorrect. Suppose $y=( 5,8,10,14 )$, $\tilde{y}=( 4,9,9,44 )$, and $S=(2,3)$. Then $j^*=4$, $e=30$, $\varepsilon=1$, and $\lambda=2$. Therefore, $\tau=27$ and $m=14$. According to the authors’ argument, there should be at least $14$ anchors whose MOS exceed $y_4$. However, in this example there are only two anchors, and their MOS are both less than $y_4$, which contradicts the claim.

### Questions
See Weakness 1-5.

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
2

### Summary
Full-Reference (FR) IQA is a foundational task for image processing pipelines, and the goal of developing a model that balances accuracy, speed, and adversarial robustness is well-motivated. However, this manuscript fails to show its novelty in methodological clarity, insufficient theoretical justification, incomplete experimental validation, and unsubstantiated claims of novelty and uniqueness. While the core objectives (compactness, speed, robustness) are relevant, the submission does not adequately demonstrate that BiRQA addresses these goals in a rigorous or differentiated way.

### Strengths
Present a compact FR IQA metric model that processes four fast complementary features within a bidirectional multi-scale pyramid. 

A bottom-up attention module injects fine-scale cues into coarse levels through an uncertainty-aware gate, while a top-down cross-gating block routes semantic context back to high resolution. 

Introduce anchored adversarial training, a theoretically grounded strategy that uses clean "anchor" samples and a ranking loss to bound point-wise prediction error under attacks.

### Weaknesses
BiRQA uses "four fast complementary features"—the most basic building block of the model—but provides no description of what these features are (e.g., texture, edge, frequency-domain features), how they are extracted, or why four features are chosen over fewer/more. Without this, the "compactness" of the model (a core selling point) is meaningless.

The "bottom-up attention module with an uncertainty-aware gate" and "top-down cross-gating block" are described in name only. There is no explanation of how the pyramid is structured (e.g., number of scales, resolution per level), how uncertainty is quantified in the gate, or how cross-gating routes semantic context—key details that distinguish BiRQA from existing multiscale IQA models (e.g., LPIPS, DISTS).

To support claims of "compactness" and "real-time throughput," the manuscript must report metrics like model parameters (MParams), FLOPs, or inference latency (e.g., FPS on a standard GPU/CPU). Omitting these makes the "compact" label unsubstantiated.

There is no explanation of how "clean anchor samples" and "ranking loss" bound "pointwise prediction error under attacks." For example, what defines a "clean anchor"? How does the ranking loss (typically used for order preservation) translate to error bounding for adversarial perturbations? Without equations or formal proofs, AAT appears to be an empirical heuristic, not a "theoretically grounded" method.

Existing models like FastLPIPS (a lightweight variant of LPIPS) and RobustIQA (designed for adversarial perturbations) already balance accuracy, speed, and robustness to varying degrees. The manuscript provides no comparison to these models.

### Questions
see the weakness.

### Soundness
2

### Presentation
2

### Contribution
2
