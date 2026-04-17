# On the Role of Temperature Sampling in Test-Time Scaling

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Large language models (LLMs) can improve reasoning at inference time through test-time scaling (TTS), where multiple reasoning traces are generated and the best one is selected. Prior work shows that increasing the number of samples $K$ steadily improves accuracy. In this paper, we demonstrate that this trend does not hold indefinitely: at large $K$, further scaling yields no gains, and certain hard questions remain unsolved regardless of the number of traces. Interestingly, we find that different sampling temperatures solve different subsets of problems, meaning single-temperature scaling explores only part of a model’s potential. We therefore propose scaling along the temperature dimension, which enlarges the reasoning boundary of LLMs. Temperature scaling enables base models to reach performance comparable to reinforcement learning (RL)-trained counterparts, without additional post-training. We further provide a comprehensive analysis of this phenomenon and design a multi-temperature voting method that reduces the overhead of temperature scaling. Overall, our findings suggest that TTS is more powerful than previously thought, and that temperature scaling offers a simple and effective way to unlock the latent potential of base models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper re-examines the Test-Time Scaling of Large Language Models for reasoning tasks, designed an efficient multi-temperature voting method with early exit for easy questions, reducing computational overhead while preserving performance gains. Experiments across multiple models and datasets validate the effectiveness of the proposed method.

### Strengths
1) The comparative experiments are relatively comprehensive; the proposed scheme requires no additional training and is orthogonal to most existing TTS methods.  
2) An entropy-based analysis and problem taxonomy are conducted to reveal the dynamic behavior of temperature scaling, constituting a key contribution of this work.

### Weaknesses
1) All models used in the comparison belong to the Qwen family, which is insufficient to demonstrate the generality of the proposed method. It is recommended to include models of different families for validation.  
2) The main text lacks a formal description of the proposed method, making its exact implementation difficult to grasp.  
3) There is no ablation study on the hyper-parameters (e.g., cross-temp threshold vs. intra-temp threshold) and their impact on performance and efficiency, hindering an understanding of how each component contributes to the gains.

### Questions
Please refer to the weakness part.

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
This paper investigates the role of temperature sampling in the test-time scaling (TTS) paradigm for large language models (LLMs). Traditionally, TTS boosts reasoning performance by generating multiple samples and selecting the best one via a verifier. The authors show that beyond a certain K, accuracy plateaus, and some problems remain unsolved regardless of further sampling. The key insight is that different sampling temperatures T lead to the solution of different subsets of problems, implying that a single-temperature TTS explores only a part of the model’s reasoning space. The authors propose multi-temperature scaling, where samples are drawn from multiple temperatures to expand the model’s “reasoning boundary.” They show empirically that temperature scaling allows base models to achieve performance comparable to RL-trained models, without additional fine-tuning.

### Strengths
- The observation is interesting and novel.
- The figures are informative and well-presented.
- The authors conduct experiments across multiple domains.
- The paper is easy to follow.

### Weaknesses
- The major concern is that the experiments are only conducted on Qwen3 series models. However, different models may have different properties regarding temperature scaling. For example, the recommended temperature for Qwen3 and DeepSeek-R1-Distill series models are different [1].
- The experiments are only restricted to models up to 8B parameters. However, larger models (e.g., 70B+) may have different behaviors with respect to temperature scaling.
- The paper lacks a theoretical framework explaining why certain temperatures preferentially solve specific hard problems.

[1] POLARIS: A POst-training recipe for scaling reinforcement Learning on Advanced ReasonIng modelS.

### Questions
- Can the authors give some high-level points of why using different temperatures helps? For example, is it because different temperatures lead to more diverse reasoning paths, or because certain temperatures are better suited for specific types of problems?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper explores the limitations of current test-time scaling (TTS) methods, which typically sample at only a single temperature. The authors propose scaling along the temperature dimension—that is, sampling at various different temperatures—to capture the union of all problems the model is capable of solving. The experiments demonstrate that this simple "temperature scaling" approach can allow a base model to match or even exceed the performance of models fine-tuned with RL.

### Strengths
- The paper's core hypothesis and observations are concise and meaningful. The idea that a model's set of solvable problems is temperature-dependent, and that the union of these sets represents the model's true capability, is a valuable insight.

- The figures in the paper are clear, well-executed, and effectively support the analysis and conclusions. Figure 3, in particular, does an excellent job of visualizing and explaining the core findings.

### Weaknesses
- High Computational Cost: The method is computationally expensive as it requires extensive sampling across a wide range of different temperatures to achieve its full effect.

- Lack of Deep Explanation: The paper lacks a deep explanation for why different problems require different temperatures to be solved. The analysis describes the resulting entropy dynamics (what is happening) but doesn't fully explain the root cause (why this dependency exists). Entropy seems to be a consequence of this phenomenon, not the cause.

- Novelty of Claims: The conclusion that a base model with sufficient repeated sampling can outperform an RL-tuned model is not an entirely novel finding in the field.

- Potential for Cherry-picking in Figure 3d: It is unclear how the specific value of $k$ (e.g., $k=128$) was chosen for the Pass@k comparison in Figure 3d. It is plausible that as $k$ increases, repeated sampling at any single temperature might eventually surpass the RL model's performance. The comparison at $k=128$ feels somewhat selective and may not present the full picture.

### Questions
There is a body of existing work focused on teaching models to perform dynamic temperature sampling (i.e., learning to adjust temperature during generation). Could the authors discuss this line of research and elaborate on its connection to their findings?

### Soundness
3

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
4

### Summary
This paper focuses on enhancing the reasoning performance of large language models through temperature scaling. The authors propose an innovative method that adjusts the temperature parameter during the testing phase to expand the model's reasoning capabilities. The study demonstrates that a multi-temperature strategy significantly outperforms a single-temperature strategy. This approach can achieve performance comparable to models trained with reinforcement learning, solely through appropriate temperature configuration during testing, without additional training. This method is both innovative and practical, warranting further exploration and discussion.

### Strengths
Introduces an innovative temperature scaling mechanism during the reasoning process.
The experimental design is sound, validating the effectiveness of the method across multiple datasets without increasing training burden.
Provides a detailed entropy analysis revealing the theoretical mechanism behind temperature scaling.

### Weaknesses
Terms like "reasoning boundary" and "upper bound" are ambiguously defined; the measurement criteria for the budget (such as the number of tokens, decoding steps, etc.) are not clearly specified, which may lead to unfair comparisons.
Insufficient comparison with strong baselines. Lacks rigorous comparisons under equivalent reasoning budgets with methods like single high-temperature self-consistency, multi-temperature grid + voting, best-of-N/majority voting, strong verifier-assisted methods, and RL models.
Key methodological details are incomplete. Lacks principles for selecting the temperature set, adaptive budget allocation, detailed voting mechanism, and sensitivity analysis for early exit strategy

### Questions
It is recommended to rigorously define terms such as "reasoning boundary" and "upper bound," and to clearly specify the metrics used for budgeting (e.g., total number of generated tokens, number of decoding steps), ensuring consistency across all comparative experiments.

### Soundness
3

### Presentation
2

### Contribution
2
