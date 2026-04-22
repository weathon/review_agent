# CyclicReflex: Improving Reasoning Models via Cyclical Reflection Token Scheduling

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
Large reasoning models (LRMs), such as OpenAI’s o1 and DeepSeek-R1, harness test-time scaling to perform multi-step reasoning for complex problem-solving. This reasoning process, executed before producing final answers, is often guided by special juncture tokens that prompt self-evaluative reflection. These transition markers and reflective cues are referred to as “reflection tokens” (e.g., “wait”, “but”, “alternatively”). In this work, we treat reflection tokens as a “resource” and introduce the problem of resource allocation, aimed at improving the test-time compute performance of LRMs by adaptively regulating the frequency and placement of reflection tokens. Through empirical analysis, we show that both excessive and insufficient use of reflection tokens, referred to as over-reflection and under-reflection, can degrade model performance. To better understand this trade-off, we draw an analogy between reflection token usage and learning rate scheduling in optimization. Building on this insight, We propose cyclical reflection token scheduling (termed CyclicReflex), a training-free decoding strategy that dynamically modulates reflection token logits with a bidirectional, position-dependent triangular waveform, incurring no additional computation cost. Experiments on MATH500, AIME2024/2025, AMC2023, GPQA Diamond and LiveCodeBench demonstrate that CyclicReflex consistently improves performance across model sizes (1.5B–14B), outperforming standard decoding and recent approaches such as TIP (thought switching penalty) and S1.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper propose CycliReflex, a novel algorithm that control the generation of "reflection tokens" by adding a periodically changing penalty. Extensive experiments on math and other reasoning benchmarks are provided to show that the proposed algorithm is performing better than baselines like S1 and TIP.

### Strengths
- The paper is well-written.
- The algorithm is easy-to-follow and well-motivated.

### Weaknesses
- Only one class of models, i.e., Deepseek-distilled models are tested. Given the behavior of different model lineages could be substantially different, additional models, potentially larger ones, need to be covered.
- Certain highly related works are missing [1, 2, 3]. Specifically, [1, 2] are highly correlated papers that provide insights and potential baselines, while [3] is one highly similar paper that needs to be compared with. 
- Some implementation details that are necessary for reproducibility are missing. Please refer to some of the questions for more details.
- For test-time scaling, no pass@N or cons@N results are provided, which are the more commonly used test-time scaling metrics.

[1]. Sadhukhan R, Chen Z, Zheng H, et al. Kinetics: Rethinking Test-Time Scaling Laws[J]. arXiv preprint arXiv:2506.05333, 2025.

[2]. Kang Z, Zhao X, Song D. Scalable best-of-n selection for large language models via self-certainty[J]. arXiv preprint arXiv:2502.18581, 2025.

[3]. Chen W, Koenig S, Dilkina B. Iterative Deepening Sampling as Efficient Test-Time Scaling[J]. arXiv preprint arXiv:2502.05449, 2025.

### Questions
1. How is LiveCodeBench tested? Specifically in code, the "reflection tokens" should not be included during the code generation process in the middle. How are such linguistic restrictions considered and affect the performance?
2. According to Appendix A, a grid search is needed to find the optimal hyperparameters used in the setting. Is the search done per-model per dataset or only once? And is there any overall recommendations for parameters when trying on new settings?
3. For accuracy results like Table 4, is the experiments averaged through multiple runs? Is it still 5 times or something else?

### Soundness
2

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
3

### Summary
This paper introduces CyclicReflex, a novel, training-free decoding strategy for Large Reasoning Models (LRMs) that dynamically modulates the logits of "reflection tokens" (e.g., "wait", "but") using a cyclical, triangular waveform. The core motivation is the "resource allocation" problem for reflection tokens, where models suffer from either under-reflection (premature convergence to an answer) or over-reflection (excessive, looping reasoning). The authors draw a conceptual analogy between scheduling reflection tokens and scheduling learning rates in optimization, where under/over-reflection mirror the effects of too small/large a learning rate. Inspired by cyclical learning rates, CyclicReflex alternates between promoting and suppressing reflection tokens, balancing exploration and convergence in the reasoning "thought landscape." Extensive experiments on math and non-math benchmarks show that CyclicReflex consistently improves accuracy over baseline decoding and competing methods like TIP and S1, without significantly increasing generation length, and integrates well with other test-time scaling techniques.

### Strengths
* The core idea of treating reflection tokens as a schedulable resource and drawing an analogy to learning rate scheduling is highly original and compelling.

* CyclicReflex is simple to implement, incurs no additional training cost, and demonstrates consistent, non-trivial performance gains across a wide range of tasks and model scales.

* The experiments are thorough, covering multiple datasets (MATH, AIME, AMC, GPQA, LiveCodeBench), model families (Qwen, LLaMA), and integration with other test-time scaling methods (Best-of-N, Beam Search).

### Weaknesses
* The method introduces two new hyperparameters (amplitude A, period C). The grid search ranges provided in Appendix A are helpful, but the paper could do more to provide heuristics or guidelines for setting these parameters on new tasks or models, as optimal values seem to vary by dataset (e.g., different C for MATH500 vs. AIME).

* The set of reflection tokens V_hat is a critical component, but its definition and potential sensitivity are not deeply discussed. Is the performance robust to the specific choice of tokens? Could the method be extended to learn or identify reflection tokens dynamically?

### Questions
* How would you advise a practitioner to set the hyperparameters A and C for a new, unseen task or a different LRM architecture not tested in this work? Are they transferable across similar domains?

* How was the vocabulary of reflection tokens V_hat constructed? Is the method's performance sensitive to minor changes in this set (e.g., adding "however" or "therefore")?

* The learning rate analogy is powerful, but in optimization, the loss landscape is fixed, whereas the "thought landscape" is generated on-the-fly. How does this fundamental difference impact the interpretation of your method?

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
This paper tackles reasoning failures in LRMs caused by under-thinking and over-thinking stemming from the mismanagement of reflection tokens. The key contribution is CyclicReflex, a training-free decoding strategy that modulates reflection token logits using a periodic, bidirectional triangular waveform. This approach is inspired by cyclical learning rate schedules in optimization, aiming to dynamically balance reasoning exploration and convergence. Experiments across six benchmarks show CyclicReflex consistently improves accuracy over baselines, notably enhancing performance across all difficulty levels where prior methods failed.

### Strengths
1. The paper is well-written and logically structured, clearly motivating the problem of under- and over-reflection and leading to the proposed solution.
2.  The paper addresses the important problem of managing LRM reasoning paths. Framing reflection token generation as a resource allocation problem and drawing an analogy to learning rate scheduling in optimization is clear and novel.
3. The proposed CyclicReflex method demonstrates strong and consistent accuracy improvements across a diverse set of models and six reasoning benchmarks. It shows gains across all problem difficulty levels, validating the effectiveness of its approach.

### Weaknesses
**Weaknesses:**

1.  I have some questions on experimental setup:
    - The `max_token` limit is set to 8192. It is unclear if this is sufficiently long for R1-style CoT generation, as significant truncation could prematurely cut off reasoning paths. This could potentially confound the analysis of under-reflection (cut off before completion) and over-reflection (never fully realized), thus impacting the reliability of the results.
    - The hyperparameter search space for the period $C$ seems somewhat arbitrary. The paper uses small ranges for different methods (e.g., $[200, 1000]$ and $[1000, 2000]$ for CyclicReflex and $[100, 1000]$ for TIP) without a clear justification for these specific bounds.
2. The paper's analysis suggests an "early-explore, late-converge" pattern is beneficial (e.g., the phase shift analysis in Fig. A3d shows $\phi=0$ is optimal). It would be interesting to see a simpler, non-cyclical baseline that implements this insight, such as a linear decay schedule (i.e., starting with a high logit boost for reflection tokens and linearly decaying it to a penalty). It is therefore unclear if the *cyclical* nature of CyclicReflex is the key contributor to its success, or if a simpler decay schedule would have achieved similar results.

### Questions
Please see Weaknesses above.

### Soundness
3

### Presentation
4

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
This paper investigates the problems of "overthinking" and "underthinking" in large reasoning models. The authors treat "reflection tokens" (e.g., "wait", "but") as a cognitive resource and propose CyclicReflex, a training-free decoding strategy. Drawing an analogy with cyclical learning rates (CLR) in optimization, CyclicReflex uses a position-dependent triangular wave function to periodically adjust the logits of these reflection tokens. The goal is to dynamically balance "exploration" (encouraging reflection) and "convergence" (suppressing reflection) during the reasoning process.

### Strengths
- The paper frames the problem from a novel and interesting perspective: treating reflection tokens as a computational resource and formalizing their management as a scheduling problem.

- The analogy between reflection scheduling and cyclical learning rates is creative and highly intuitive. The resulting method is simple, training-free, and adds minimal computational overhead, which makes it very attractive for practical application.

- The paper provides good analysis, including visualizations of the "landscape of thought," which help build intuition for why the method works.

### Weaknesses
- The definition of "reflection tokens" is subjective and relies on a pre-defined, fixed set. The method's dependency on this curated list is a potential vulnerability, as it may not capture all forms of model reflection.

- The core motivation is based on an analogy. While the "landscape of thought" metaphor is appealing, it lacks a rigorous theoretical foundation and strict validation. It is not as formally defined as the "loss landscape" it draws inspiration from, making the connection feel more intuitive than theoretical.

- The method appears sensitive to its new hyperparameters. The triangular wave in Equation 3 is controlled by an amplitude ($A$) and a period ($C$). The ablation studies suggest that performance can vary significantly based on the choice of these parameters, which implies that significant tuning may be required. This undermines the practical benefit of a "training-free" approach.

### Questions
- How robust is the method to linguistic drift? If the model learns to use other phrases for reflection (e.g., "let's re-examine", "on the other hand") that are not in the pre-defined set, will the method's effectiveness degrade or fail entirely?

- Were other dynamic baselines considered? For instance, a baseline that encourages reflection (e.g., the TIP method but with a positive $\alpha$) or a baseline that applies simple random noise to the reflection token logits? Comparing against these would help clarify whether the observed benefit comes specifically from the cyclical nature of the adjustment or just from any form of dynamic modulation.

### Soundness
3

### Presentation
3

### Contribution
3
