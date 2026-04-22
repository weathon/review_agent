# Don't Run with Scissors: Pruning Breaks VLA Models but They Can Be Recovered

- Avg Score: 5.50
- Decision: Reject
- Scores: 10, 6, 4, 2

## Abstract
Vision–Language–Action (VLA) models have advanced robotic capabilities but remain challenging to deploy on resource-limited hardware. Pruning has enabled efficient compression of large language models (LLMs), yet it is largely understudied in robotics. Surprisingly, we observe that pruning VLA models leads to drastic degradation and increased safety violations. We introduce GLUESTICK, a post-pruning recovery method that restores much of the original model's functionality while retaining sparsity benefits. Our method performs a one-time interpolation between the dense and pruned models in weight-space to compute a corrective term. This correction is used during inference by each pruned layer to recover lost capabilities with minimal overhead. GLUESTICK requires no additional training, is agnostic to the pruning algorithm, and introduces a single hyperparameter that controls the tradeoff between efficiency and accuracy. Across diverse VLA architectures and tasks in manipulation and navigation, GLUESTICK achieves competitive memory efficiency while substantially recovering success rates and reducing safety violations. Videos, code, and additional materials are in: https://gluestick-vla.github.io/

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper is about the very important question of how to make a VLA model relevant (fast) by pruning without deteriorating the performance too much. The approach computes a correction term that is based on the un-pruned and the pruned model that is later used in inference with the pruned model. The term is only computed once and required not knowledge of the pruning method. The paper provides empirical evidence of the problem actually existing, provides insights as to why the problem arises, and demonstrates that the proposed method solved the issue.

### Strengths
- The paper addresses a relevant and important problem.

- The paper presents a good explanation and demonstration that the issue exists and is relevant.

- The paper presents an effective solution to the problem. 

- The approach has favorable properties such as only computing the correction term once and being independent of the pruning method. 

- The introduction and related work sections are well formulated. 

- The method is innovative, making use of a low-dimensional correction term (from SVD). 

- The paper contains code examples for the most important parts of the approach.

### Weaknesses
- The presentation of figure 2 is hard to read and some different way of presenting the same information would help.

### Questions
- The proposed approach computes the correction term after pruning and which happens after learning the model in the first place. Would it be possible to improve the correction performance by pruning in a certain way or learning parameters that make correction with the SVD approach easier? 

- What bias is the SVD approach introducing to the correction?

- There exist other low-rank decompositions of matrices. Why is the SVD one preferred?

### Soundness
4

### Presentation
4

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
This paper introduces GLUESTICK, a training-free, pruning-agnostic post-pruning recovery method. While pruning is an effective compression technique for LLMs, it causes drastic degradation in VLA models, leading to near-zero task success rates and increased safety violations. GLUESTICK computes a lightweight corrective term via SVD of the difference between dense and pruned weights and then applies this correction during inference.  A single interpretable hyperparameter rank $r$ is used to balance efficiency and accuracy. The overall method is simple and easy to implement. Experimental results across several VLA models and benchmarks show that GLUESTICK can help recover most of the lost performance while maintaining memory efficiency.

### Strengths
- The paper’s observation of the pruning collapse issue in Vision-Language-Action (VLA) models is quite meaningful. 
- The proposed method is simple, efficient, and easy to implement, compatible with various pruning techniques.
- It offers valuable insights for the compression, pruning, and deployment of VLA models.

### Weaknesses
- The method presented in this paper bears some similarity to adding a low-rank adapter on top of pruning to offset pruning-induced losses. It would be better for the authors to elaborate on the differences between their proposed method and approaches like LoRA.
- When utilizing different backbones and dimensions, how should the hyperparameter $r$ (rank) be determined for each scenario? Would it be feasible to assign distinct $r$ values to different weight matrices? This adjustment seems promising for further enhancing the trade-off between performance and efficiency.
- The performance of the method under different pruning sparsity levels is not explored.

### Questions
- Does the proposed method have any impact on inference speed and latency?
- In practice, VLA deployment often requires combining pruning with other methods like quantization. Does GLUESTICK still work with these techniques?

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
The paper starts from a observation from standard, LLM-validated pruning catastrophically collapses VLA policies—success drops to 0% on both manipulation (OpenVLA: 85.2% to 0%) and navigation (NaVILA: 43% to 0%). It then proposes GLUESTICK, a training-free, pruning-agnostic weight-space correction: compute the dense–pruned gap per linear layer, take a truncated SVD, and add a low-rank correction at inference to restore “lost directions”. GLUESTICK substantially recovers manipulation (≈50% of success lost to pruning) and fully restores navigation success while keeping most of the VRAM savings of structured sparsity; unsafe-episode rates return near dense baselines. The paper further diagnoses why VLAs are fragile: compared to LLM layers, VLA layers show flatter singular spectra, meaning “useful signal” is spread across many directions and is easily excised by structured pruning.

### Strengths
* Clear empirical finding: the results are quite good across two domains (manipulation/navigation) and three architectures.
* Simple method arch: GLUESTICK is training-free, drop-in, and pruning-agnostic; a single interpretable hyperparameter r controls the memory-recovery trade-off.
* Thoughtful analysis: the analysis provides a plausible reason VLAs differ from LLMs (flatter spectra -> pruning removes distributed, important directions), which aligns with the effectiveness of a low-rank “stitch-back” on top of pruned weights.

### Weaknesses
* Corrections applied only to the linear layers: for the model with heavy conv layers for vision encoders or attention projection with structured kernels, the proposed method might lose some effectiveness, like the WorldVLA case.
* The rank scheduling is empirical: with a single global r used all-way, given large per-layer variation, and the vision backbone is sensitive parameter-wise, without considering the inner difference of different layers.
* The requirement of dense-weights: the method needs the original dense checkpoint to compute the gap SVD, which might constrain its usage in some scenarios.

### Questions
* Is it possible for the method to interplay with other techniques like quantization or LoRA? Could you try with more compression baselines for more comprehensive ablation studies?
* You have mentioned that the manipulation task can only achieve ~50% of recovery; can you analyze more on the performance difference with different task settings?
* How sensitive is the method to domain shift and long-horizon tasks? 
* In Appendix D.1, the authors find that a smaller, more "targeted" calibration set for Wanda pruning yields a 2% performance gain. This is an interesting but counter-intuitive result. Does this suggest that pruning methods are highly sensitive to the quality and relevance of calibration data, and that "more data" is not always better?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The manuscript presents a study of VLA pruning, demonstrating that VLA models considerably lose performance compared to their LLM counterparts. The authors demonstrate and benchmark this behavior on manipulation as navigation tasks using the OpenVLA, WorldVLA, and NaVILA models. By analyzing the spectrum of model weights, the authors further demonstrate a difference in the weight space between VLAs and LLMs.

Based in this observation, GLUESTICK is proposed as a mitigation method. By compressing the most important components of the pruned weights using SVD, and thus, reconstructing the suppressed component of weights, the authors are able to recover part of the model performance. This behavior is demonstrated on simulation benchmarks for manipulation and navigation tasks.

### Strengths
- The analysis of performance loss is well executed, spanning multiple models and tasks.
- The proposed recovery method is grounded in the joint findings from a study considering the pruning process itself, as well as from evaluations in a robotic simulator.
- GLUESTICK is able to recover part of the lost model performance in both deployment scenarios across model architectures.
- The method is straightforward to implement without requiring target domain calibration data and shows strong improvements over the baseline.

### Weaknesses
- As a major shortcoming, the work misses a comparison to VLM pruning, where the strong performance drop of VLMs compared to LLM models is a known property [1,2]. In previous works, this behavior is especially prominent at or below 50% sparsity, the operating point chosen in this work. Since VLAs typically build on top of VLMs, rather than on a language-only LLM, this comparison and discussion of related works is required.
- Since VLMs lose considerable performance from pruning, it remains unclear if the demonstrated problem is a problem stemming from the VLM backbone or the full VLA built for the robotic task.
- The experiment in Q6 should be put in context. 200 SVD components are an extremely strong compression of the weight space, especially when compared to 200/500 residual components in GLUESTICK.
- Experiments are purely performed in simulation. Since real-world deployment of VLA policies can show considerably different performance, a small study on real robots can help the experiments in this work.

[1] Liang, Yinan, et al. "Efficientllava: Generalizable auto-pruning for large vision-language models." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

[2] Koike-Akino, Toshiaki, Jing Liu, and Ye Wang. "$\mu $-MoE: Test-Time Pruning as Micro-Grained Mixture-of-Experts." arXiv preprint arXiv:2505.18451 (2025).

### Questions
- How strong is the performance loss at different operating points of pruning, especially with lower sparsity?
- How does Figure 2 change when compared to the corresponding VLM model?
- Overall, the work should discuss the relation to VLM pruning and put the findings and novelty in the context of existing work in this area. I will reconsider my rating if this shortcoming is adequately addressed in the paper discussion, existing works, and experimental validation.

### Soundness
3

### Presentation
4

### Contribution
2
