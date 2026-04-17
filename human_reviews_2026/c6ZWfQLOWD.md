# Block-wise Adaptive Caching for Accelerating Diffusion Policy

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Diffusion Policy has demonstrated strong visuomotor modeling capabilities, but its high computational cost renders it impractical for real-time robotic control.
Despite huge redundancy across repetitive denoising steps, existing diffusion acceleration techniques fail to generalize to Diffusion Policy due to fundamental architectural and data divergences.
In this paper, we propose **B**lock-wise **A**daptive **C**aching (**BAC**), a method to accelerate Diffusion Policy by caching intermediate action features. BAC achieves lossless action generation acceleration by adaptively updating and reusing cached features at the block level, based on a key observation that feature similarities exhibit non-uniform temporal dynamics and distinct block-specific patterns. 
To operationalize this insight, we first design an Adaptive Caching Scheduler to identify optimal update timesteps by maximizing the global feature similarities between cached and skipped features. However, applying this scheduler for each block leads to significant error surges due to the inter-block propagation of caching errors, particularly within Feed-Forward Network (FFN) blocks. To mitigate this issue, we develop the Bubbling Union Algorithm, which truncates these errors by updating the upstream blocks with significant caching errors before downstream FFNs.
As a training-free plugin, BAC is readily integrable with existing transformer-based Diffusion Policy and vision-language-action models.  Extensive experiments on multiple robotic benchmarks demonstrate that BAC achieves up to 3x inference speedup for free. Project page: https://block-wise-adaptive-caching.github.io.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Block-wise Adaptive Caching (BAC), a training-free method designed to accelerate inference for transformer-based Diffusion Policy models in robotic control. The method is composed of two primary components: an Adaptive Caching Scheduler (ACS), which employs dynamic programming to compute an optimal, non-uniform schedule of cache update steps for each transformer block by maximizing feature similarity; and a Bubbling Union Algorithm (BUA). The authors identify a novel "error surge phenomenon," where naive block-wise scheduling leads to catastrophic error propagation, particularly in Feed-Forward Network (FFN) blocks. The BUA is proposed as a targeted solution to mitigate this issue by enforcing upstream blocks with high estimated errors to update their caches in synchrony with their downstream FFN counterparts. Through experiments on several robotic manipulation benchmarks, the authors claim that BAC achieves up to a 3x inference speedup with negligible, or even slightly improved, performance compared to the full-precision model.

### Strengths
- While caching in diffusion models is an existing area of research, the primary contribution of this paper—the identification, analysis, and targeted mitigation of the inter-block "error surge phenomenon"—is novel and insightful. The proposed Bubbling Union Algorithm (BUA), motivated by this analysis, is an original and simple heuristic for a previously undocumented problem.

- The use of dynamic programming within the Adaptive Caching Scheduler (ACS) to find an optimal schedule under a well-defined objective function (maximizing feature similarity) is a principled and sound approach.

- The authors validate their method across a commendable range of benchmarks (RoboMimic, Push-T, Kitchen, etc.), data sources (PH, MH), and even extend it to a large-scale Vision-Language-Action model (RDT-1B), providing evidence for the potential generalizability of the approach.

### Weaknesses
- Insufficient Evidence for the Core Assumption of Homogeneity: The entire offline-online paradigm of the proposed method hinges on the crucial assumption of "high episode homogeneity"—that a cache schedule computed from one trajectory can generalize to others. The evidence provided for this cornerstone assumption is extraordinarily weak, consisting of a visual comparison between only two demonstration instances (demo 11001 vs. 20000) in Appendix A.9. Generalizing a universal property from two samples is scientifically unsound and constitutes anecdotal evidence at best. This fails to adequately address the challenge that the stochastic nature of diffusion models can lead to fundamentally different trajectories (i.e., different action modalities), which may invalidate a statically computed schedule.
- Absence of Real-World Hardware Experiments: All experiments are conducted in simulation, but the ultimate goal of robotic policy learning is deployment on physical hardware. The paper makes strong claims about enabling "real-time robotic control", but without a single real-world experiment, these claims remain unsubstantiated.

### Questions
1. The core assumption of "episode homogeneity" is foundational to your method but is supported by only two examples. Could you provide a more systematic study across a large, diverse set of episodes and initial random noises for a given task to statistically validate that the feature similarity structure ($s_k$ curve) remains stable? How does your offline schedule contend with the explicitly multimodal, stochastic nature of diffusion policy?

2. Could you provide qualitative video results comparing the behavior of a robot arm controlled by the baseline versus the BAC-accelerated policy? This would help visually verify that the claimed speedup does not introduce detrimental artifacts. Furthermore, do you have plans for or preliminary results from deploying this method on a physical robot to validate its real-world effectiveness and truly substantiate the claims of enabling real-time control?

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
This paper proposes Block-wise Adaptive Caching (BAC), a training-free method to accelerate Transformer-based Diffusion Policies for real-time robot control. BAC adaptively caches and reuses intermediate block features using a dynamic scheduling algorithm and a bubbling mechanism to prevent error propagation. It achieves up to 3× faster inference without performance loss across multiple robotics benchmarks.

### Strengths
1. This paper proposes a method that achieves a 3× speedup, which is an important step toward making diffusion policies truly practical for real-world robotic control.
2. The paper offers an analysis of the internal feature dynamics of Diffusion Policy, revealing that feature similarity across timesteps is non-uniform among different Transformer blocks (SA, CA, FFN), which provides a solid foundation for the proposed block-wise adaptive method.
3.  Another major highlight is the discovery and explanation of the “Error Surge” phenomenon: naive block-level caching causes severe degradation in FFN blocks, and the authors convincingly attribute this to the amplification and propagation of upstream errors due to the absence of LayerNorm. This insight is broadly relevant to all Transformer-based caching approaches.

### Weaknesses
Proposition 3.1 is the paper’s only theoretical contribution, aiming to explain error propagation in the FFN blocks. While the first-order Taylor expansion (Appendix A.4) is directionally correct, it does not provide any quantitative bound, nor does it explain why the SA/CA blocks (which also contain $W_{out}$) do not exhibit the same error surge phenomenon.

### Questions
In Stage 1 of BUA, the method identifies “high-error” upstream blocks using a global $L_1$ norm (Eq. 13), which measures the average feature variation across all timesteps. However, the “error surge” analyzed in Section 3.3 is a localized event that occurs when an FFN block updates while its upstream block reuses a high-error cache at the same timestep. Why  is a global average metric $l_j$ the right choice for addressing an instantaneous error propagation problem? Wouldn’t a more appropriate measure be the instantaneous cache error at the specific timesteps where the FFN depends on the block?

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
3

### Summary
his paper introduces Block-wise Adaptive Caching (BAC), a training-free acceleration method for transformer-based Diffusion Policy models used in robotic control. Diffusion Policy models are powerful but computationally expensive due to their iterative denoising process. BAC addresses this by adaptively caching intermediate features at the block level, significantly reducing redundant computation without sacrificing performance.
Key Contributions:
1.	Block-wise Adaptive Caching (BAC): A novel caching strategy that adaptively updates and reuses features at the block level, tailored for Diffusion Policy architectures.
2.	Adaptive Caching Scheduler (ACS): Uses dynamic programming to determine the optimal update timesteps for each block, maximizing feature similarity and minimizing caching error.
3.	Bubbling Union Algorithm (BUA): Mitigates inter-block error propagation, especially in FFN blocks, by enforcing upstream updates when downstream blocks are updated.
4.	Extensive Experimental Validation: Demonstrates up to 3.55× speedup with no performance loss across multiple robotic tasks and models, including large-scale vision-language-action (VLA) models like RDT-1B.
5.	Training-Free and Plug-and-Play: BAC is easy to integrate into existing models without retraining or architectural changes.

### Strengths
This paper presents an original contribution by demonstrating that a block-level, training-free caching mechanism can significantly reduce the inference time of Diffusion Policy without compromising accuracy. Its strength lies in a clear problem statement—addressing the well-known 10 Hz to 30 Hz performance gap—and a novel insight: non-uniform temporal similarity across transformer blocks necessitates per-block caching schedules, while upstream errors can propagate and amplify into downstream FFN failures if not properly trapped. The authors support this with a concise error-propagation model and exhaustive simulations, ranging from small control tasks to a 1-B parameter VLA model, consistently achieving a 3× speedup with reproducible artifacts. Although the idea builds upon existing caching concepts, its novel reframing for Diffusion Policy, plug-and-play practicality, and immediate real-time performance gains render this work both high-quality and broadly significant for deploying generative models under strict latency constraints.

### Weaknesses
My primary concern is the absence of physical validation, as all experiments were conducted in simulation. Based on lessons from computer vision, we know that improvements in GPU-FLOPS do not always translate to reduced real-world latency. A brief hardware trace—even for a single arm and task—would help demonstrate that the acceleration persists in the presence of motor noise and timing jitter. I would also appreciate a comparison with standard diffusion acceleration techniques to validate the unique advantage of the proposed robot cache.

That said, the core idea is intuitive, the evaluation is thorough within the simulation framework, and the commitment to open-source the code is commendable. Should the authors add at least minimal real-robot validation and discuss the cache's robustness under more aggressive quantization or fewer denoising steps, I believe the paper would be of great value to the community.

### Questions
Q1. Practical Performance on Physical Hardware
The reported GPU-time improvements in simulation are promising. However, in real-world robotic systems, such gains can diminish due to factors like camera latency, USB bandwidth limitations, or motor jitter. Could the authors provide any empirical timing results—even from a single task on one robot arm—showing the actual wall-clock frequency achieved in a lab setting? This would help validate the practical applicability of the method.

Q2. Comparison with Fewer Denoising Steps
In image diffusion models, reducing denoising steps (e.g., from 1000 to 20) is a common acceleration strategy while maintaining acceptable output quality. Could the authors include a comparative analysis—such as a small plot—illustrating performance when using only 15 or 20 steps, both with and without BAC? This would clarify whether the proposed caching mechanism offers advantages beyond simply executing fewer denoising iterations.

Q3. Visualization of Cache Behavior
As a visual learner, I would find it greatly helpful to see a figure illustrating the cache's effect—for instance, an attention map or feature heatmap at a step where the cache decided not to update. Such a visualization would intuitively convey the underlying "similarity" concept without relying solely on mathematical formulations.

Q4. Hyperparameter Tuning for New Tasks
The paper mentions using S = 7 or 10. If a user applies the method to a new robot task, are there practical guidelines—such as "start with S = 8"—to avoid extensive hyperparameter search? A brief recommendation in the paper would greatly improve usability.

Q5. Code Accessibility
The commitment to open-source the code is appreciated. Would it be possible to share a GitHub link during the rebuttal period—even if kept private temporarily—so that reviewers can examine the license and potentially run a simple local example?

### Soundness
3

### Presentation
3

### Contribution
3
