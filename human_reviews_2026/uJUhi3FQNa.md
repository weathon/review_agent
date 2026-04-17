# Reinforcement Mid-Training

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
The development of state-of-the-art large language models is commonly understood as a two-stage process involving pre-training and post-training. We point out the need for an additional intermediate stage called reinforcement mid-training with potential for strong performance gains. In this paper, we formally define the problem and identify three key challenges: (1) inefficient training due to excessive reasoning steps, (2) disregard of the imbalanced token entropy distribution, and (3) underutilization of token information. To address these challenges, we propose RMT, a framework for efficient, adaptive, and unified reinforcement mid-training with various innovative components. In particular, we first introduce a dynamic token budget mechanism that constrains unnecessary reasoning steps and mitigates model overthinking. Next, we design a curriculum-based adaptive sampling method that fosters a progressive learning trajectory from easy to hard tokens. Finally, we present a dual training strategy that combines reinforcement learning with next-token prediction, ensuring targeted learning on key tokens and full exploitation of all token information. Extensive experiments demonstrate the superiority of RMT over state-of-the-art methods, achieving up to +64.91% performance improvement with only 21% of the reasoning length in language modeling. We also show that checkpoints obtained after reinforcement mid-training can benefit the subsequent post-training, yielding up to +18.76% improvement in the mathematical domain.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Reinforcement Mid-Training (RMT), a distinct stage in the LLM development positioned between pre-training and post-training. The proposed RMT framework addresses key challenges of inefficiency, imbalanced data difficulty, and information underutilization by integrating a dynamic token budget, a curriculum-based adaptive sampling strategy, and a dual training objective. The work demonstrates improvements in both performance and computational efficiency on mathematical reasoning tasks, establishing a foundation for subsequent post-training.

### Strengths
- The paper clearly identifies three practical challenges (i.e., overthinking, entropy imbalance, token underutilization) that are highly relevant to current research.
- The proposed RMT framework is a well-designed solution where each component (DTB, CAS, dual training) directly addresses one of the identified challenges.
- The empirical results are convincing, showing substantial gains not only in accuracy but also in efficiency.

### Weaknesses
- My main concerns lie in the paper's foundational assumptions about *difficulty* and *efficiency*. The framework relies heavily on token entropy as a proxy for reasoning difficulty, which is a significant simplification. High uncertainty does not always correlate with high cognitive complexity. Similarly, the dynamic token budget enforces a *shorter is better* heuristic, which may not hold for problems that intrinsically require elaborate, long-chain reasoning. These simplified proxies could limit the method's robustness and generalizability, potentially causing it to penalize valid, complex reasoning paths in favor of concise but possibly suboptimal solutions.
- The experimental validation is predominantly confined to the domain of mathematical reasoning. This narrow focus makes it difficult to assess the framework's effectiveness on other complex reasoning tasks like code generation, legal analysis, or creative problem-solving.
- The framework introduces a considerable number of new hyperparameters (e.g., budget decay, curriculum transition points, trade-off weights), but there lacks a sensitivity analysis, leaving its practical robustness and ease of implementation in question.

### Questions
- How would the performance of the curriculum learning component change if difficulty were defined by a more semantically aware metric (e.g., reasoning step complexity) instead of token entropy?
- Could the dynamic token budget be made adaptive to the input's complexity rather than decaying with the training step, for instance, by allocating a larger initial budget for demonstrably harder problems?

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
4

### Summary
This paper proposes **RMT**, a “reinforcement mid-training” stage placed between pre-training and post-training. RMT targets three issues—**overthinking**, **imbalanced token-entropy**, and **wasted token information**—via: (1) a **dynamic token budget** with a triangular **length reward** to keep chain-of-thought concise, (2) an **entropy-based curriculum** that shifts sampling from easy→hard tokens over time, and (3) a **dual objective** that runs **RL on selected high-entropy tokens** while using **next-token prediction (NTP)** on the rest. On OmniMATH language-modeling, RMT reportedly improves accuracy by **up to +64.91%** while using **≈21%** of the reasoning length vs. baselines, and the mid-trained checkpoints further boost later post-training by **up to +18.76%** on Skywork.

### Strengths
- The paper clearly defines **reinforcement mid-training** and motivates why an intermediate stage can help later post-training. 
- The **dynamic token budget** with a **triangular length reward** is an intuitive way to curb overthinking without eliminating reasoning. 
- The **entropy-based curriculum** is straightforward to implement and aligns with progressive difficulty.

### Weaknesses
- The **evaluation scope is narrow**, focusing on math-reasoning (OmniMATH/Skywork) with only **200 evaluation samples** in the LM setup, which limits statistical confidence and domain generality. 
- The **reward is exact next-token match** (Eq. 5), which may bias learning toward surface-form correctness rather than semantic adequacy; the paper does not compare against richer RL signals (e.g., verifier or sequence-level rewards). 
- Several **strong dynamic RL baselines** are missing in experiments (e.g., recent efficient RL variants or learned curricula), even though related work mentions them; this makes the SOTA claim harder to verify.

### Questions
Have you tried **sequence-level** or **verifier-based** rewards (beyond exact token match) and, if so, how do they change the efficiency/accuracy trade-off?

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
4

### Summary
This paper introduces RMT, a framework for efficient, adaptive, and unified reinforcement mid-training. It proposes several strategies to enhance both training efficiency and performance. These include: 1) a dynamic token budget mechanism, which prevents the generation of excessively long chains of thought that can impede training efficiency; 2) a curriculum learning strategy that determines token difficulty based on entropy, allowing the model to gradually progress from simpler to more complex tokens during training; and 3) a combined training approach that utilizes both cross-entropy and Reinforcement Learning (RL) loss. The experimental results demonstrate that RMT improves next-token prediction capabilities and generalizes effectively to subsequent post-training with RL.

### Strengths
1. The method shows strong performance, significantly enhancing the model's ability for next-token prediction and demonstrating good generalization to the post-training RL phase.
2. The paper is well-written, and the proposed method is straightforward to understand.

### Weaknesses
1. I was unable to find some specific experimental details. For instance, what proportion of tokens were subjected to reinforcement learning training versus the standard cross-entropy objective? 

2. I am curious about the added value of reinforcement mid-training in a scenario where the post-training RL phase is exceptionally thorough and prolonged. Would the improvements from RMT still be significant if the model undergoes a very comprehensive post-training RL alignment?

### Questions
Could the authors elaborate on the initial stages of training? Was a "warm-up" phase or a specific set of prompts used to enable the model to learn how to reason for the next token from a pre-trained model?

### Soundness
3

### Presentation
3

### Contribution
3
