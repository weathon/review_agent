# Budget-aware Test-time Scaling via Discriminative Verification

- Avg Score: 4.67
- Decision: Reject
- Scores: 8, 4, 2

## Abstract
Test-time scaling is a powerful strategy for boosting the performance of large language models on complex reasoning tasks. While state-of-the-art approaches often employ generative verifiers to select the best solution from a pool of candidates, this method incurs prohibitive computational costs, limiting its practicality. In this work, we pivot the focus to a more budget-aware paradigm: discriminative verification. We conduct a thorough empirical analysis and demonstrate that while discriminative verifiers may underperform in isolation, combining them with self-consistency in a hybrid approach creates a powerful and efficient test-time scaling mechanism. Notably, under a fixed compute budget, our approach surpasses state-of-the-art generative verification by a significant margin: achieving up to 6.1\% higher accuracy on AIME2025. Our findings establish that for practical, real-world applications, budget-aware scaling with discriminative verifiers is not only a "free" upgrade over self-consistency, but also a more effective and efficient alternative to costly generative techniques. Code is available at https://anonymous.4open.science/r/Verification-ICLR2026.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
Work provides a detailed study of test-time scaling properties of different approaches to the verification of model responses. 
Results show that a combination of discriminative and self-consistency-based methods can beat generative approaches in terms of efficiency. What is more, a verifier trained on mainly math data can bring benefits in out-of-distribution settings.

### Strengths
+ usage of well-established reasoning tasks, such as AIME 2024-2025 and GPQA
+ testing verification with DeepSeek R1 distilled models ranging from 1.5B to 32B parameters.
+ practical (latency) and theoretical (FLOPS) insights into test time scaling of verification methods
+ study of the influence of solver size on methods' performance
+ practical ablations, such as omission of thinking traces in verifier output

### Weaknesses
- Figure 1: Addition of Pass@N can significantly improve presentation, as the text suggests that Pass@N is included in the figure
```
As N increases, the probability that at least one answer is correct also rises (i.e., Pass@N improves; see Figure 1)
```
- Figure 3 can benefit from the unification of  method names with Figure 1
    - DV and BoN
    - DPV and PV
- some text inconsistencies in lines 340-342
```
For example, when M = 1, hybrid
discriminative verification techniques outperform generative verification for any combination of
N ≤ 128 and M ≤ 32.
```

### Questions
Can authors identify a scenario where a verifier trained on one type of data fails on the other? 
Such a study can improve the contribution of the work. For example, it is unclear whether the verifier trained on math can help with code.

### Soundness
3

### Presentation
2

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
This paper investigates budget-aware test-time scaling for LLMs via discriminative verification. While state-of-the-art generative verifiers produce detailed chain-of-thought critiques to select solutions, they incur prohibitive computational costs. The authors demonstrate that hybrid discriminative verification methods—specifically weighted self-consistency (WSC) and pessimistic verification (PV)—combine the efficiency of discriminative verifiers with the robustness of self-consistency voting. They train a 1.5B discriminative verifier using Bradley-Terry ranking loss on 32k math problems and evaluate on AIME2024/2025, LiveBench Math, and GPQA. Key findings show hybrid discriminative methods consistently outperform self-consistency with <2% overhead and surpass generative verification by up to 6.1% under fixed compute budgets (5×10¹⁵ and 1×10¹⁶ FLOPs).

### Strengths
1. The paper addresses a critical gap in test-time scaling research by systematically analyzing computational costs alongside accuracy. 
2. Authors effectively position discriminative verification as a middle ground between pure self-consistency and expensive generative verification. 
3. The authors provide multiple efficiency analysis across different metrics with both FLOP-based and latency-based compute measurements.

### Weaknesses
1. The evaluation focuses exclusively on mathematical reasoning tasks (AIME, LiveBench Math, GPQA). Generalizability to other reasoning domains (code generation, commonsense reasoning, strategic planning) remains unclear. 
2. The paper trains a single verifier architecture (2-layer value head, Bradley-Terry loss) without comparing alternative training objectives (binary cross-entropy, contrastive learning), architectures, or verifier sizes. The choice of 0.5 for PV is validated on one held-out set but sensitivity across tasks/models is unexplored. 
3. The 22.5-minute latency and 2.2×10^16 FLOPs crossover thresholds are specific to the experimental setup (DeepSeek-R1-Distill-Qwen-32B solver, Heimdall with M=2). No analysis examines failure modes where hybrid discriminative methods underperform self-consistency, limiting practical guidance for deployment.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper analyzes the test-time scaling properties of discriminative verifiers with some commonly used selection techniques, like Best-of-N, weighted self-consistency, etc. They show that using techniques like weighted self-consistency with discriminative verifiers is a good way to improve performance over self-consistency (without a verifier) while not being as computationally expensive as using generative verifiers, which have been shown to be very expensive by prior work.

### Strengths
This paper compared various approaches (like best-of-N, weighted SC, etc.) as well as compared discriminative and generative verifiers. It compared them in terms of performance, computational cost (both FLOPs and latency), and scaling properties along various axes like the number of candidate solutions, the length of each solution, etc.

### Weaknesses
The major weakness is that most of this is already well-known and not really novel.
1. One of the primary results in the paper is that with discriminative verifiers, weighted self-consistency or GPV > self-consistency. This is pretty well-known (for example, figure 6 in [1], figure 7 in [2], figure 8 in [3], figure 1 in [4]; also see [5]).
1. The finding that discriminative verifiers can output the simple self-consistency baseline is also not new (see figure 5 in [2]).
1. The finding that discriminative verifiers are less expensive than generative verifiers is also quite expected since they don’t generate verification tokens at inference time. In terms of inference cost, they would be similar to majority voting.
1. The findings in Sections 3.2 and 3.3 are also quite expected: if the candidate solutions improve in performance, then the overall performance would increase. For example, Figure 5a is qualitatively similar to Figure 7 in [2].

## A few minor nit-picks:
- Preliminaries don’t explain what (discriminative) verifiers are.
- The Section title “Scaling Model Size for Discriminative Verification” was a bit misleading because it seemed like the focus would be on scaling verifier size (which wasn’t the case). It might be better to rephrase this for clarity.
- Section 3.3 paragraph 1 – refer to a figure or table supporting these numbers (i.e., figure 5).


## References
[1] From Decoding to Meta-Generation: Inference-time Algorithms for Large Language Models. Sean Welleck, Amanda Bertsch, Matthew Finlayson, Hailey Schoelkopf, Alex Xie, Graham Neubig, Ilia Kulikov, Zaid Harchaoui. 

[2] Large Language Monkeys: Scaling Inference Compute with Repeated Sampling. Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, Azalia Mirhoseini. 

[3] Generative Verifiers: Reward Modeling as Next-Token Prediction. Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, Rishabh Agarwal. 

[4] Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters. Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar. 

[5] Improving Large Language Model Fine-tuning for Solving Math Problems. Yixin Liu, Avi Singh, C. Daniel Freeman, John D. Co-Reyes, Peter J. Liu.

### Questions
No major questions

### Soundness
2

### Presentation
2

### Contribution
1
