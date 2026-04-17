# Dual-Mode Cloud-Device Collaboration for Efficient Continual Adaptation

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Continuous environmental changes induce distribution shifts, leading to significant performance 
degradation of models deployed on resource-constrained mobile devices. Existing fast adaptation 
methods fail to provide sufficient generalization to meet performance requirements, 
while cloud-device collaborative learning often relies on a considerable amount of data, 
limiting real-time applicability. To ensure both timeliness and effectiveness, we propose 
a dual-mode cloud-device collaborative framework. Specifically, the proposed mothod dynamically 
switches modes according to the degree of distribution shift: (1) Collaborative adaptation 
mode handles substantial shifts, where the cloud performs multi-level domain alignment and 
position-aware prompting to learn domain-invariant representations, which are then distilled 
to the device model; (2) Self-adaptation mode addresses minor shifts, where the device model 
performs unsupervised test-time adaptation with pseudo-label generation and quality-aware 
reweighting for fast local updates. Experimental results show that our framework achieves superior 
performance while using only 80\% of the data and incurring less than 0.5\% additional parameters 
and computation. Moreover, it consistently outperforms compared methods in both accuracy and 
single-frame inference speed.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a cloud–device collaborative framework for continual adaptation under distribution shift. This includes a quality-aware test-time adaptation (QuTTA) method for self-adaptation for minor shifts, and a multi-level domain adaptation (MuDA) method with position-aware prompting (PAP) on the cloud to conquer more severe distribution shifts. Experiments claim strong performance. However, I am concerned about the novelty, evaluations, and writings regarding the manuscript. I thus recommend Reject.

### Strengths
1.	This paper aims to address a practical scenario where both models on the cloud and the device need to adapt to the changing scenarios.
2.	The overall framework is simple.

### Weaknesses
1.	The novelty is limited. It seems that the proposed method is a simple combination of multiple techniques. For example, the proposed uncertainty estimation has been explored in ViDA [1], the proposed MuDA is heavily based on [2], the proposed PTP directly applies Coordinate Attention [3], and the proposed QuTTA shares a similar idea of existing TTA methods for update reweighting. No specific challenges or difficulties in applying these techniques to TTA are introduced, making the proposed methods look like a naïve combination.
2.	Important comparisons are missing. The compared baselines are either inappropriate or out-of-date. First, TEA/ReCAP/SAR are _not_ continual TTA methods. For continual TTA, the authors should instead compare with DPCore [4], ViDA [1], PeTTA [5]. Second, the baselines CoTTA and EATA are outdated, and it is suggested to compare with their advanced versions, _i.e._, BeCoTTA [6] and EATA-C [7]. Moreover, current evaluations are only on ResNet models. It is also important to demonstrate the versatility of the proposed method across diverse architectures, such as the Transformer.
3.	The proposed method has limited feasibility. This is because: 1)  It necessitates modification to the training pipelines, which is computation-consuming and requires access to the privacy-sensitive training data. 2) It introduces more than 12 hyperparameters, including but not limited to k, M, threshold for $V\_{unc}$, $\lambda_1$, $\lambda_2$, $\lambda_3$, $\lambda_{adapter}$, $\lambda_{cwd}$, $\tau_{max}$, $\tau_0$, $\tau_{min}$, $\gamma$. These hyperparameters are too much for proper tuning, especially for test-time adaptation, where we don’t have test data and its labels from the target domains.
4.	The proposed method is inefficient. The uncertainty estimation necessitates M extra predictions per sample with dropouts, which would significantly increase the latency.
5.	The readability of the manuscripts requires improvement. Many symbols are not properly defined, and details are missing. For example, global average pooling is a function and should be combined with inputs, but it is defined as a variable $F\_{avg}$ in the manuscript; $\mathcal{L}\_{det}$ is not defined. In addition, in $\mathcal{L}\_{img}$ and $\mathcal{L}\_{ins}$,  it is unclear how to find a corresponding source image $\boldsymbol{x}\_{i}^{s}$ for distribution alignment.  

Overall, I believe the current manuscript has strong weaknesses in method design, experiments, and writing, which are hard to be addressed in a rebuttal. I thus recommend Reject.

[1] ViDA: Homeostatic Visual Domain Adapter for Continual Test Time Adaptation. ICLR 2024.

[2] Domain Adaptive Faster R-CNN for Object Detection in the Wild. CVPR 2018.

[3] Coordinate Attention for Efficient Mobile Network Design. CVPR 2021.

[4] Dpcore: Dynamic Prompt Coreset for Continual Test-Time Adaptation. ICML 2025.

[5] Persistent Test-Time Adaptation in Recurring Testing Scenarios. NeurIPS 2024.

[6] BECoTTA: Input-dependent Online Blending of Experts for Continual Test-time Adaptation. ICML 2024.

[7] Uncertainty-Calibrated Test-Time Model Adaptation without Forgetting. TPAMI 2025.

### Questions
1.	How to accurately estimate FLOPS with backpropagation?
2.	How are $P_c^s$ and $P_c^t$ computed? Are they calculated on a single sample? If so, how can a single sample reflect class-conditional distributions?

### Soundness
2

### Presentation
1

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
This paper proposes a Dual-Mode Cloud–Device Collaboration framework for continual adaptation in changing environments. The key idea is to switch between two modes depending on how big the distribution shift is: 1.	Collaborative Mode (Cloud + Device) – When the environment changes a lot, the cloud model helps. It performs multi-level domain alignment (MuDA) and position-aware prompting (PAP) to learn domain-invariant features, then distills that knowledge back to the device model efficiently. 2.	Self-Adaptation Mode (On-Device) – When the changes are small, the device adapts itself using quality-aware test-time adaptation (QuTTA). It generates pseudo-labels, reweights samples based on their quality, and includes a self-recovery mechanism to avoid performance drop.  Experiments on Cityscapes-C and ACDC-Detection show the promise of the proposed method.

### Strengths
The explored problem of cloud–edge collaborative test-time adaptation is both practical and interesting.

The experiments on Cityscapes-C and ACDC-Detection under the collaborative TTA setting appear novel and well-motivated.

### Weaknesses
There are too many hyper-parameters in the proposed method, making it overly complex and difficult to tune during online testing. How do the authors determine these hyper-parameters, and is there any sensitivity analysis provided?

An important related work, “Towards Robust and Efficient Cloud-Edge Elastic Model Adaptation via Selective Entropy Distillation” (ICLR 2024), which also focuses on cloud-edge collaborative TTA, is missing from the discussion and comparison.

Overall, the proposed method appears to be too combinational of existing techniques. Many of the components, such as reweighting and self-recovery, have already been well studied in the TTA literature.

As a continual TTA method, the comparisons could include more continual TTA baselines. However, most baselines reported in Tables 2 and 3 (except for CoTTA) are not continual TTA approaches.

### Questions
The use of MC Dropout for uncertainty estimation is inefficient in the online TTA setting. How do the authors determine the number of drop times ($M$)? Does this approach outperform other uncertainty estimation strategies such as prediction entropy?

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
3

### Summary
The paper introduces a dual-mode on-device/cloud continual adaptation framework for mobile vision systems under distribution shift. When the shift is minor, the device enters a lightweight self-adaptation mode that performs unsupervised test-time training using pseudo-labels with quality-aware reweighting and a self-recovery mechanism to prevent catastrophic forgetting. When the shift is large, the system switches to a collaborative mode, where the cloud performs multi-level domain alignment (image- and instance-level alignment with semantic consistency and gradient-reversal) and position-aware prompting, then distills the adapted knowledge back to the device via adapters. Experiments show that this approach improves accuracy and runtime efficiency on mobile platforms while requiring limited data and negligible additional compute and parameters.

### Strengths
- The paper presents a well-motivated and timely contribution to continual adaptation for mobile vision systems, an area where robustness under real-world distribution shift remains largely unsolved. Its originality stems from a coherent dual-mode design that couples uncertainty-driven shift detection with a lightweight on-device adaptation mechanism and a cloud-assisted alignment and distillation pipeline, effectively bridging test-time training and domain adaptation ideas under resource constraints.

- The experiments are comprehensive, covering multiple distribution-shift benchmarks and real mobile hardware. The method consistently outperforms strong test-time training and cloud-based baselines in accuracy under both mild and severe shifts. Ablations on key components—such as uncertainty-based switching, pseudo-label reweighting, and adapter distillation—clearly show their contributions. Real-device latency, memory, and compute measurements further support the practicality of the approach.

### Weaknesses
- The uncertainty-based mode-switching mechanism is intuitive but relatively heuristic; a more principled analysis or comparison with alternative shift-detection methods (e.g., energy-based scores, density estimation, or prior adaptive thresholding approaches) would clarify its reliability across domains.

- Although the dual-mode system is practical, the design combines several existing components (pseudo-label filtering, gradient-reversal alignment, adapter distillation), and the incremental novelty of each piece is limited—positioning the contribution more explicitly as a system-level advance with clearer ablations isolating algorithmic novelty would improve clarity. 

- The evaluation focuses mainly on vision benchmarks with moderate resolution and a single model backbone; demonstrating robustness across model scales, backbones, or broader task types (e.g., segmentation, tracking) would strengthen generality claims. Finally, communication and energy overheads in cloud-collaboration scenarios are only lightly explored; profiling or sensitivity analysis under constrained bandwidth or intermittent connectivity would better validate real-world deployment assumptions.

### Questions
- How sensitive is the uncertainty-based mode-switching threshold to different domains and model architectures? Have the authors explored adaptive or learned thresholds, and could they share calibration results?

- Can the authors provide more analysis on failure cases where the system incorrectly stays in on-device mode or triggers cloud mode unnecessarily? Understanding these edge cases would clarify robustness.

- What is the communication overhead during cloud-assisted adaptation (frequency, data volume, latency)? How does performance degrade under constrained or unstable network conditions?

### Soundness
3

### Presentation
3

### Contribution
2
