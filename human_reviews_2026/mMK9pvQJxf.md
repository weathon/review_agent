# Improving Classifier-Free Guidance in Masked Diffusion: Low-Dim Theoretical Insights with High-Dim Impact

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Classifier-Free Guidance (CFG) is a widely used technique for conditional generation and improving sample quality in continuous diffusion models, and its extensions to discrete diffusion has recently started to be investigated. In order to improve the algorithms in a principled way, this paper starts by analyzing the exact effect of CFG in the context of a low-dimensional masked diffusion model, with a special emphasis on the guidance schedule. Our analysis shows that high guidance early in sampling (when inputs are heavily masked) harms generation quality, while late-stage guidance has a larger effect. These findings provide a theoretical explanation for empirical observations in recent studies on guidance schedules. The analysis also reveals an imperfection of the current CFG implementations. These implementations can unintentionally cause imbalanced transitions, such as unmasking too rapidly during the early stages of generation, which degrades the quality of the resulting samples. To address this, we draw insight from the analysis and propose a novel classifier-free guidance mechanism. Intuitively, our method smoothens the transport between the data distribution and the initial (masked) distribution, which results in improved sample quality. Remarkably, our method is achievable via a simple one-line code change. Experiments on conditional image and text generation empirically confirm the efficacy of our method.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper investigates Classifier-Free Guidance (CFG) in the context of masked discrete diffusion models. The authors claim to have identified a key flaw in existing CFG implementations, where high guidance strength can lead to imbalanced transitions and unmasking too rapidly, which degrades sample quality.

To address this issue, the authors propose a novel CFG mechanism based on a column normalization of the rate matrix, achievable via a one-line code change. The authors provide a low-dimensional (1D and 2D) theoretical analysis to motivate this change and to characterize the properties of effective guidance schedules.  Experimental results are also provided to validate the technique.

### Strengths
1. The paper's primary contribution is a fix for a problem in discrete CFG that can be implemented with a one-line code change. I always like seeing examples of this kind of simple and effective solution, especially when they are theoretically motivated.
2. The methodology of using low-dimensional (1D and 2D) theoretical analysis to derive insights that are relevant to the high-dimensional setting is a strong and interesting approach.
3. Based on the experimental results, the proposed normalization fix appears to be effective.

### Weaknesses
My current impression of the paper is that the authors have indeed likely identified a flaw in CFG for masked discrete diffusion models and found a simple fix for it. However, the more serious weaknesses I outline below prevented me from more deeply appreciating the authors' contributions, which has me currently sitting on the negative side of the fence. That said, I admit to being a good deal less familiar with the discrete diffusion literature than I am with the continuous diffusion literature, hence the low confidence in my score. I will keep an open mind during the discussion period, and I look forward to seeing the authors' responses and the feedback from the other reviewers.

I'll list my concerns in decreasing order of seriousness:
1. There seems to be a major contradiction in the paper's claims regarding effective guidance schedules. In Section 3.4 (lines 368-370), the 2D theoretical analysis concludes: "Therefore, effective schedules have higher guidance in the beginning and middle phases of the generation, and their effect towards the end is negligible." This directly contradicts empirical results from numerous other works as well as statements in this paper's abstract and the discussion in Section 4.2: "[S]chedules that apply stronger guidance during the middle and later stages of the sampling process, while keeping early guidance small, tend to perform better." Unless there is a serious misreading on my part, the results of Section 3.4 are never reconciled with these statements and the empirical record.
2. Considering the weight given to the paper's theoretical results, I found them very hard to follow in their current form. Notation is occasionally not defined, and it was sometimes challenging to match the theorems or lemmas with their counterparts and proofs in the appendix.
3. On the topic of notation, in most of the CFG literature, in both continuous and discrete diffusion, $x$ is a state and $y$ is a condition. But in this paper $y$ (never explicitly defined) is apparently a member of the state space (judging by the formula for the partition function) as well as a condition. I found this quite confusing.
4. There are some minor typos and other errors in the paper (e.g. *diffusion* is misspelled in the header for Section 2.1, the regionalism *smoothen* is used in the abstract instead of the standard *smooth*).
5. In the literature, the probability ratio the authors refer to as the *score* is referred to as the *concrete score*. I recommend sticking to this terminology to avoid confusion with the score as it's understood in the continuous diffusion literature.

### Questions
1. Can the authors resolve the central contradiction (Weakness #1)? Which finding is correct: the 2D theory suggesting high early guidance, or the empirical results suggesting high late guidance? Why does the theory not seem to match the practice here?

2. In the abstract, what is meant by "late-stage guidance has a larger effect"? Is this a positive or negative effect on generation quality? (This also relates to Weakness/Question #1.)

### Soundness
3

### Presentation
2

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
This paper analyzes the effect of using guidance schedules in discrete diffusion models and how such schedules can improve performance compared to constant guidance. Drawing inspiration from continuous diffusion models, the authors note that high guidance scales are necessary to enhance sample quality and condition alignment; however, using a constant high guidance scale results in suboptimal performance due to large changes during the early sampling stages. They first demonstrate that transition matrices should be normalized (via SoftMax) in the presence of guidance. Then, using a low-dimensional setup, they show that a constant guidance scale leads to poor performance, reducing diversity and increasing bias in the final generation. To address this, the authors propose an increasing weight schedule to balance the effect of guidance across sampling steps, leading to improved performance across various diffusion models for both image and text generation.

### Strengths
- Since guidance in discrete diffusion models is underexplored, this paper bridges an important gap between CFG in continuous diffusion models and discrete diffusion models. The results have potential implications for all systems that rely on discrete diffusion.

- The proposed method is simple and can be easily integrated into existing sampling pipelines.

- The theoretical results provide intuition and reasoning behind the method, although their presentation could be significantly improved.

- Experiments are conducted on both image and text generation benchmarks.

### Weaknesses
- In my opinion, the main weakness of the paper is its presentation. Several parameters are either misused in notation or not defined prior to their introduction in the text. This makes the paper difficult to follow and obscures the intuition and analysis behind the proposed method.

- Section 3.4 appears to contradict the main message of the paper. It states that “effective schedules have higher guidance in the beginning and middle phases of generation,” whereas the best performance is reported for “schedules that apply stronger guidance during the middle and later stages of the sampling process, while keeping early guidance small.”

**Minor**:
- There might be a typo in Equation 6.
- Table 1 indicates that the increasing schedule has one parameter, while Table 2 shows it has two parameters ($w$ and $r$).
- The theoretical analysis could also consider the effect of guidance on condition alignment, in addition to its impact on diversity and stability.

### Questions
1. Do normalization and time-dependent schedules also improve the performance of Simple Guidance?

2. Could you provide additional metrics, such as Precision and Recall, to separately evaluate diversity and quality, rather than relying solely on FID?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper analyzes CFG for masked diffusion, showing (in a low-dimensional setting) that strong early guidance hurts quality while late guidance helps. It also identifies an implementation flaw that causes over-eager early unmasking and proposes a one-line fix to smooth the transition from the masked prior to the data distribution.

### Strengths
1. CFG is a well researched topic in continuous diffusion. But in masked/discrete diffusion is under active exploration. Clarifying scheduling effects is valuable for both image inpainting/masked modeling and text infilling models
2. Section 3.4 provides exhaustive analysis across factors such as time parameters and guidance strength.
3. The experiment in Section 2.3 is intuitive, improving explainability.
4. The method can be performed in inferencing time.

### Weaknesses
1. **Novelty:** The importance of guidance scheduling and rescaling by conditional/unconditional norms has been reported by Kynkäänniemi et al. (2024). Can you clarify how your approach differs?
2. **Metrics/Benchmarks:** For text-to-image evaluation, you only report ImageReward. Could you also evaluate with **HPSv2** to check for aesthetic trade-offs? Additionally, please test on T2I benchmarks like **GenEval** and **T2I-CompBench**.

[1] Kynkäänniemi, T., Aittala, M., Karras, T., Laine, S., Aila, T., & Lehtinen, J. (2024). *Applying guidance in a limited interval improves sample and distribution quality in diffusion models.* NeurIPS 37, 122458–122483.

### Questions
1. Prior work reports that higher guidance weights can distort images and reduce fidelity, yet in your Figure 8 larger guidance weights yield better performance. Can you justify this discrepancy?
2. The appendix includes many generated images—could you also include the text prompts used to produce them?
3. Unified reward models are surging. Could you evaluate your method with Show-o [1] and Dual Diffusion [2] to assess its effectiveness in that setting?

[1] Xie, J., Mao, W., Bai, Z., Zhang, D. J., Wang, W., Lin, K. Q., ... & Shou, M. Z. (2024). Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528.
[2] Li, Z., Li, H., Shi, Y., Farimani, A. B., Kluger, Y., Yang, L., & Wang, P. (2025). Dual diffusion for unified image generation and understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 2779-2790).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper analyzes classifier-free guidance (CFG) in masked discrete diffusion and shows—theoretically (in low dimensions)—that strong early guidance harms quality, whereas late guidance has a larger positive impact. It identifies a flaw in common CFG implementations, where imbalanced transitions unmask tokens too quickly. To remedy this, the paper proposes a simple one-line column/softmax normalization that smooths transport. Empirically, this normalization improves ImageNet-256 FID, ImageReward text-to-image alignment, and MATH-500 text generation accuracy.

### Strengths
The paper presents a principled low-dimensional analysis of CFG for masked discrete diffusion, showing that strong early guidance is harmful while late guidance is beneficial.
Building on this insight, it introduces an elegant, theory-grounded tweak—a one-line column/softmax normalization—that corrects imbalanced early unmasking.
Importantly, by linking this tractable normalization to improved robustness and better FID/ImageReward/MATH-500 results, the paper offers a practical change likely to be widely adopted in discrete diffusion implementations.

### Weaknesses
1. The paper develops analysis and closed-form expressions only in 1–2 dimensions for masked discrete diffusion. As a result, guarantees for realistic high-dimensional CTMCs remain implicit, leaving the theoretical treatment somewhat loose.

2. Some results isolate the mechanism using a simple sampler without remasking and with fixed step counts (e.g., 50 steps on ImageNet-256), which may limit generality. Moreover, sampling schedules and samplers are crucial to implementing the guidance mechanism, yet the paper provides neither theoretical analysis nor empirical evaluation of their effects.

3. Although the evidence spans ImageNet-256 FID, ImageReward, and MATH-500 with a single LLM backbone, broader discrete domains (e.g., ASR, text, protein) and larger vocabularies are not explored. Because the approach is closely related to Unlocking Guidance and Simple Guidance, running additional experiments on the datasets used in those works is crucial.

4. Quality gains are reported, but diversity metrics are missing. Because guidance mechanisms can reduce sampling diversity, it is important to report diversity measures and quantify the proposed method’s impact.

5. It is recommended to ensure the citation style is applied uniformly across the manuscript.

### Questions
1. Since the theory is developed in 1–2D masked diffusion, could you provide bounds or a proof sketch that extends to realistic high-d CTMCs?

2. With results limited to ImageNet-256 FID and MATH-500 on a single LLM backbone, is there evidence that the approach generalizes to other discrete diffusion tasks?

3. Although normalization is cheap, schedule changes can alter unmasking rates and step counts. Therefore, reporting the resulting overhead (wall-clock, GPU-hours, memory) is important to demonstrate the strength of the proposed method.

4. The proposed framework mainly targets masked diffusion, what changes are needed for uniform or other discrete diffusions?

### Soundness
2

### Presentation
2

### Contribution
2
