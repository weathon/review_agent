# MUR: Momentum Uncertainty guided Reasoning for Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 6, 4

## Abstract
Large Language Models (LLMs) have achieved impressive performance on reasoning-intensive tasks, yet optimizing their reasoning efficiency remains an open challenge. 
While Test-Time Scaling (TTS) improves reasoning quality, it often leads to overthinking—wasting tokens on redundant computations.
This work investigates how to efficiently and adaptively guide LLM TTS without additional training. 
Inspired by the concept of momentum in physics, we propose Momentum Uncertainty-guided Reasoning (MUR), which dynamically allocates thinking budgets to critical reasoning steps by tracking and aggregating step-wise uncertainty over time.
To support flexible inference-time control, we introduce $\gamma$-control, a simple mechanism that tunes the reasoning budget via a single hyperparameter.
We provide in-depth theoretical proof to support the superiority of MUR in terms of stability and biases.
MUR is comprehensively evaluated against various TTS methods across four challenging benchmarks (MATH-500, AIME24, AIME25, and GPQA-diamond) using different sizes of recent Qwen3 models (1.7B, 4B, and 8B).
Results demonstrate that MUR reduces computation by over 45% on average while improving accuracy by 0.33–3.46%.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the issue of "overthinking" in Large Language Models (LLMs), where Test-Time Scaling (TTS) methods waste computation on simple reasoning steps. The authors propose Momentum Uncertainty-guided Reasoning (MUR), a training-free framework to adaptively apply TTS. The authors claim this method reduces computation by over 45% while improving accuracy on several reasoning benchmarks.

### Strengths
1. The proposed method is training-free and designed to be an orthogonal framework that can be combined with existing TTS methods, which is a good quality for a utility-focused method.
2. Efficiency is an important issue  for TTS

### Weaknesses
1. Text Entropy may not to be a good metric for confidence. 
2. I am not sure whether the simple baseline like randomly selecting steps to refine or selecting the first or the last few steps would be better or comparable with MUR.

### Questions
1. Why not consider the adaptive thinking on scaling the number of steps
2. How do you identify the intermediate steps

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
3

### Summary
This paper proposes an adaptive inference-budget allocation scheme for Test-Time Scaling (TTS). By aggregating uncertainty through a designed binary detector, the proposed method effectively reduces the number of reasoning tokens on multiple datasets without sacrificing accuracy.

### Strengths
1) The proposed scheme significantly reduces inference token consumption without sacrificing accuracy.  
2) The authors provide theoretical analyses that motivate their design, helping readers to better understand the underlying principles of the method.

### Weaknesses
1) On the theoretical side, the assumptions are overly strong. The authors should at least provide evidence that the prerequisites required by their proofs hold, or verify them experimentally. For example, Equation (7) models the fluctuations in step-level uncertainty using an independent white noise model; however, for autoregressive language models, independence clearly does not hold. Even if a simplified model is used for analysis, the authors should provide toy examples that satisfy the assumptions to demonstrate their validity. In the proof of Proposition 3, the authors “assume μ_i converges exponentially to μ_∞ as timestamp increases,” which is also a very strong assumption that is neither validated nor supported by references. Moreover, all theorems should ideally be verified under restricted conditions through carefully designed toy experiments before being extended to the LLM context. Otherwise, the theoretical analysis and the method design remain disconnected, and it is impossible to determine whether the analytical conclusions actually support the method’s effectiveness.

2) On the experimental side, although the proposed method improves inference efficiency, the additional inference latency introduced by the external detector should also be compared. The authors should include a comparison of the actual inference time across different methods.

3) Furthermore, the authors provide very little description of their component designs (e.g., the specific design of the detector) and experimental details (e.g., evaluation protocols for each task, inference parameter settings), which significantly limits the reproducibility of the paper.

### Questions
Overall, I find the motivation of this work very interesting. However, in its current form it does not appear to be sufficiently prepared.

### Soundness
3

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
4

### Summary
This paper focuses on efficient and adaptive Test-Time Scaling (TTS) methods. The authors propose MUR (Momentum Uncertainty Guided Reasoning), which uses uncertainty quantification to determine whether each reasoning step requires further scaling. The uncertainty momentum acts as a moving average of stepwise uncertainty.

MUR is a training-free approach. The authors provide theoretical proofs demonstrating that MUR outperforms simple uncertainty-based methods by emphasizing recent steps, reducing variance, and achieving better convergence.

Extensive experiments show that MUR achieves superior results on mathematical reasoning tasks and GPQA. The authors also explore its performance across different models, TTS strategies, and threshold settings.

### Strengths
1. The proposed method improves both the efficiency and performance of test-time scaling without requiring any additional training.
2. The authors conducted extensive experiments across various TTS paradigms and multiple models to validate the effectiveness of their method.
3. The authors attempted theoretical derivations to demonstrate the effectiveness of MUR.

### Weaknesses
1. The theoretical analysis presented by the authors lacks some rigor. For example, the definition of noise in the uncertainty estimation is oversimplified — modeling it directly as standard Gaussian noise seems too simplistic. Moreover, the authors did not provide supporting references to justify this assumption.
2. The efficiency improvement achieved by the proposed MUR method, compared to the naive uncertainty averaging approach, is relatively limited in terms of the average number of tokens saved.
3. The paper lacks experiments on larger-scale models and a broader range of model families, such as Llama and others.

### Questions
In Section 3.2, the paper introduces the momentum uncertainty term M_t = α* M_{t-1} + (1-α)*m_t to represent the overall uncertainty along the reasoning trajectory. Since this formulation exponentially decays earlier uncertainty values, could it potentially diminish the influence of early but crucial reasoning steps that have low uncertainty but are logically important for later inference?

### Soundness
2

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
3

### Summary
This paper addresses the "overthinking" problem in Test-Time Scaling (TTS). It proposes MUR (Momentum Uncertainty-guided Reasoning), a method inspired by the concept of "momentum" in physics, which dynamically allocates the computational budget to critical reasoning steps. MUR is claimed to reduce computation by an average of about 45% across multiple benchmarks while also yielding an accuracy improvement of 0.33% to 3.46%.

### Strengths
- The problem is well-defined. The "Per-Step Scale" baseline is demonstrably inefficient, and the pursuit of a training-free, adaptive strategy to guide TTS is an excellent and practical research objective.

- The method acts as a universal "wrapper," demonstrating compatibility and benefits when combined with various existing TTS techniques (including Guided Search, LLM as Critic, $\phi$-Decoding, and Thinking Mode). This broad applicability is a significant practical advantage.

### Weaknesses
- The reported accuracy increase ($0.33\% \text{ to } 3.46\%$) is suspicious. It is unclear if these "improvements" are simply experimental noise. The paper never reports the standard deviation from multiple runs. In LLM reasoning tasks, a $0.3\%$ improvement is easily within the range of expected stochastic variability.

- The paper claims to provide "deep theoretical proof" (Section 3.2 and Appendix A). However, these "proofs" merely reiterate the well-known properties of Exponential Moving Average (EMA), specifically that it reduces variance and bias compared to the raw signal. This is a fundamental concept in signal processing and statistics, not an innovative theoretical contribution to the field of reasoning. Furthermore, this proof only establishes EMA as a stable estimator, not that MUR provably leads to better reasoning outcomes.

### Questions
- The "accuracy improvement" is the paper's core and most surprising claim. Given the inherent randomness of LLM inference, are these gains statistically significant? Could the authors please provide the standard deviation from multiple runs for the key accuracy results presented in Tables 1 and 2?

- The paper argues that MUR avoids "overthinking." Assuming that a "Per-Step Scale" baseline is correctly implemented, why would this "overthinking" damage accuracy (rather than just waste computation)? It seems counter-intuitive that reducing work would lead to better results. Is it possible that the gain is merely the result of avoiding the accuracy degradation caused by excessive scaling?

### Soundness
3

### Presentation
2

### Contribution
2
