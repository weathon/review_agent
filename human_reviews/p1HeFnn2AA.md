# Deep Learning for Two-Sided Matching

- Decision: Reject
- Scores: 6, 8, 8

## Abstract
We initiate the study of deep learning for the automated design of two-sided matching mechanisms. What is of most interest is to  use machine learning to understand  the possibility of new tradeoffs between *strategy-proofness* and *stability*.  These properties cannot be achieved simultaneously, but the efficient frontier is not understood. We introduce novel differentiable surrogates for quantifying ordinal strategy-proofness and stability and use them to train differentiable matching mechanisms that map discrete preferences to valid randomized matchings. We demonstrate that the efficient frontier characterized by these learned mechanisms is substantially better than that achievable through a convex combination of baselines of *deferred acceptance* (stable and strategy-proof for only one side of the market), *top trading cycles* (strategy-proof for one side, but not stable), and *randomized serial dictatorship* (strategy-proof for both sides, but not stable). This gives a new target for economic theory and opens up new possibilities for machine learning pipelines in matching market design.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents an innovative study on using deep learning for the automated design of two-sided matching mechanisms. The core contribution is the exploration of the tradeoff between strategy-proofness and stability, two critical properties that cannot be fully achieved simultaneously in traditional matching mechanisms. The authors introduce novel differentiable surrogates to quantify ordinal strategy-proofness and stability, training neural networks to generate randomized matchings. The results show that the efficient frontier, characterized by the learned mechanisms, outperforms traditional baselines like Deferred Acceptance (DA), Top Trading Cycles (TTC), and Random Serial Dictatorship (RSD). This approach sets a new direction for economic theory, especially in the design of mechanisms that improve on current tradeoffs.

### Strengths
1 The use of deep learning to design two-sided matching mechanisms is highly innovative and represents a significant shift from traditional economic approaches. By employing neural networks, the paper opens up new possibilities for discovering previously unknown tradeoffs between stability and strategy-proofness.

2 The paper successfully demonstrates that deep learning can uncover a superior tradeoff frontier compared to traditional convex combinations of existing mechanisms (DA, TTC, and RSD). This is a substantial contribution, highlighting the potential of machine learning in economic theory and mechanism design.

3 The introduction of both fully connected and convolutional neural network architectures, especially the CNN designed for matching mechanisms using 1×1 convolutions, shows a thoughtful adaptation of neural networks for this domain. The use of permutation equivariance in the CNN design is particularly commendable, as it enhances the generalization and scalability of the approach.

### Weaknesses
1 One major limitation of the proposed method is its scalability. While the paper claims to scale efficiently to markets with up to 50 agents on each side, this size is relatively small for many real-world applications. In large-scale matching markets, such as job markets or school admissions, the number of participants can easily reach into the thousands. Therefore, the current methodology may struggle to handle the complexity of large-scale matching problems, where the number of workers and firms is significantly higher. The approach should be optimized to better handle larger scales, with at least the capability to manage thousands of participants on both sides of the market.

2 The lack of publicly available datasets for testing the approach is a notable shortcoming. While the authors attempt to capture real-world structures through simulations, real-world validation remains essential to prove the practical viability of the proposed mechanisms. Without such validation, the generalizability of the findings remains uncertain.

3 The adversarial learning approach, where defeating misreports are generated to quantify violations of first-order stochastic dominance (FOSD), may introduce significant computational overhead. This could become more problematic as the size of the market increases, further exacerbating scalability issues.

### Questions
1 How do the authors envision scaling this framework to larger markets with thousands of agents on each side? Are there specific techniques or optimizations being considered to extend the approach beyond the current limit of 50 agents? one good way should be as follows. Considering the scalability challenges, is it possible to reduce the computational complexity by focusing on truncated preference lists rather than allowing for empty sets? In real-world markets with thousands of workers and firms, where each participant's preference list is limited to a fixed number (e.g., 10 items), using truncated preferences could significantly reduce the matching time. Could this modification lead to more efficient random matching generation while maintaining strong tradeoffs between stability and strategy-proofness?

2 Given the limited availability of public datasets, what kinds of real-world matching problems do the authors envision their approach being most applicable to? Are there any ongoing efforts to test this framework in partnership with real-world markets?Have you ever considered partnering with educational institutions or job placement agencies to obtain anonymized matching data, or inquired about potential synthetic datasets that might closely mimic real-world scenarios?


3  Is there a better way to design the loss function that could more effectively balance the tradeoff between strategy-proofness and stability, or perhaps other properties? Could alternative loss formulations lead to better performance or more efficient training?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a deep learning pipeline to study the tradeoff between strategy-proofness and stability in two-sided matching markets. The paper formulates SP violation and stability violation quantitatively as a machine learning problem. Extensive experiments are conducted to verify the claim.

### Strengths
The idea of studying the tradeoff between SP and stability with deep learning is novel and intuitive. The paper opens up a new direction in mechanism design with machine learning techniques. I generally believe that this is a good paper with novel ideas and clear presentation.

### Weaknesses
The experiment is conducted on synthetic data as compared to real-world data.

### Questions
The problem has the assumption that the market is one-to-one, and there are no ties. How will the problem formulation be different if these assumptions are removed?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a deep learning-based mechanism for two-sided matching that explores the Pareto frontier between strategy-proofness (SP) and stability—two properties that are theoretically impossible to satisfy simultaneously. The authors parameterize the matching mechanism using a neural network that takes participants' reported preferences from both sides as input and outputs a randomized matching result. The loss function is designed as a convex combination of SP and stability violations. Experimental results demonstrate that the proposed method finds a more efficient frontier than classical mechanisms.

### Strengths
1. The paper initiates the study of using neural networks to discover two-sided mechanisms from data. The idea of employing neural networks for mechanism learning is compelling.
2. The learned mechanism is capable of identifying a more efficient frontier than traditional approaches.
3. The paper establishes a bridge between deep learning and traditional algorithmic game theory, which I believe is interesting to ICLR community.

### Weaknesses
1. The experiments are limited to two prior distributions of preferences; incorporating additional distributions would be beneficial.
2. There are no diagrams to illustrate the architecture of the proposed neural network.

### Questions
The market settings are unclear to me. Could you formally describe how the agents' preferences are generated? I didn't understand how to truncate the preference order, and what is the common preference order for firms (or workers).

### Soundness
3

### Presentation
3

### Contribution
3
