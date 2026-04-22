# Cognitive Loop: Reversible Hierarchical Markov Chain for Bidirectional Self-Verifying Reasoning

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2, 6

## Abstract
Multi-step Chain-of-Thought (CoT) has significantly enhanced the mathematical reasoning capabilities of large language models by leveraging clear reasoning steps and task-specific logical structures. However, with the widespread adoption of Long CoT, the number of reasoning steps often exceeds the system's manageable limits. To address this, existing approaches attempt to reduce redundancy in KV Cache by introducing Markov chain-like reasoning structures, thereby improving inference efficiency. Nonetheless, such Markov chain-based reasoning methods introduce two critical issues: Lack of memory and Limited backward reasoning capability. To address these limitations, we propose a novel Chain-of-Thought framework based on Reversible Hierarchical Markov Chains, termed Cognitive Loop of Thought (CLoT), and a backward reasoning dataset CLoT-Instruct. In CLoT, the original problem is decomposed into sub-problems with hierarchical dependencies and modeled as a hierarchical Markov chain based on the number of dependencies. Humans typically revisit and verify their reasoning steps after reaching a conclusion to avoid errors. Inspired by this cognitive behavior, we introduce a similar backward verification mechanism at each layer. Moreover, when all higher-level (multi-dependency) sub-problems are verified as correct, we prune the remaining lower-level (fewer-dependency) sub-problems. CLoT effectively mitigates error propagation along the reasoning path and enhances the robustness of the entire reasoning process. We conduct experiments on four mathematical reasoning benchmarks, demonstrating the effectiveness of CLoT. Notably, on the AddSub dataset, when applied to the GPT-4o-mini model, CLoT achieves an accuracy of 99.0\% , outperforming traditional CoT and CoT-SC by 4.1\% and 2.9\%. Our code is publicly available at: https://anonymous.4open.science/r/CLoT-7EBD.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce Cognitive Loop of Thought (CLoT), where the authors build on chain of thought reasoning work using hierarchical markov chain through the decomposition of a problem into sub-problems. At every step, a backward verification mechanism is used to avoid errors throughout the thinking process, mimicking human thinking.

### Strengths
- Proposed framework outperforms several COT baselines proposed in the literature. 
- The framework is evaluated on 6 different reasoning benchmarks, covering three types of reasoning tasks, with consistent results across most of them. 
- Efficiency analysis shows that CLoT consumes less tokens than several other baselines, while still achieving high performance.

### Weaknesses
- Has this been tested on reasoning-tuned LLMs? It seems that the experimental set-up only looks at GPT-4. Although results are good, this does not necessarily mean that this works on other LLMs. 
- It seems that CLoT is most effective on mathematical reasoning. Why is this case? More discussion needs to be included on this point.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a new reasoning framework for LLMs.
It is designed to improve the accuracy and reliability of multi-step reasoning by mimicking the human cognitive process of verifying one's own work.
They propose to verify both forward and backward and design a hierarchical pruning to reduce token cost.
The authors validate CLoT on six mathematical and commonsense reasoning benchmarks, demonstrating better performance.

### Strengths
1. The core concept mimics human reasoning behavior and the verification is a novel method for self-correction.
2. The paper considers the trade-off between efficiency and effectiveness and CLoT creates a balance between them.
3. The empirical results prove the effectiveness of the concept.

### Weaknesses
1. The backward verification process is easy for mathematical problems, but for more complex tasks, the backward question could be vague and hard to define.
2. Although the authors propose a instruct dataset, the effectiveness of training on such dataset is not reported.

### Questions
1. How does the CLoT-Instruct dataset contribute to the training of the LLM?
2. The reversible hierarchical Markov chain relies on decomposing the problem into sub-problems with hierarchical dependencies. How are these hierarchies and sub-problems initially generated? Is this decomposition an automatic process performed by the LLM, and if so, how sensitive is the final accuracy of CLoT to the quality of this initial decomposition step?
3. How does the process differ from deductive reasoning paradigm [1][2], apart from the backward verification?
4. Typos:
    - Line 171: fforming

[1] Ling, Zhan, et al. "Deductive verification of chain-of-thought reasoning." Advances in Neural Information Processing Systems 36 (2023): 36407-36433.

[2] Zhu, Tinghui, et al. "Deductive beam search: Decoding deducible rationale for chain-of-thought reasoning." arXiv preprint arXiv:2401.17686 (2024).

### Soundness
3

### Presentation
3

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
The paper proposes a novel reasoning framework Cognitive Loop of Thought (CLoT) for large language models (LLMs) that mimics human cognitive verification. Unlike conventional Chain-of-Thought (CoT) methods that rely on forward-only reasoning, CLoT introduces a Reversible Hierarchical Markov Chain (RHMC) that alternates between forward reasoning and backward verification. Each problem is decomposed into hierarchical sub-problems; the model validates each reasoning step by reversing the logic—treating conclusions as known and re-deriving premises. A pruning strategy further skips verification of lower layers once higher-level consistency is confirmed, cutting inference cost by 41.8%. The authors also construct CLoT-Instruct, a dataset that teaches backward verification. Experiments on six benchmarks (AddSub, GSM8K, SVAMP, MATH, AQuA, CommonsenseQA) show consistent accuracy gains over CoT, CoT-SC, and other baselines, achieving 99.0% on AddSub with GPT-4o-mini and 90.5% average accuracy on GPT-4

### Strengths
1. The paper introduces a cognitively inspired, reversible reasoning paradigm, and provides rigorous mathematical formulation and efficient hierarchical pruning.
2. It demonstrates strong empirical improvements with reduced token usage.

### Weaknesses
1. The validation is only limited to reasoning benchmarks, and the experiment section lacks generalization tests on non-mathematical or other real-world tasks.
2. Does the backward verification assumes deterministic reversibility? Hierarchical Pruning assumes that if high-level reasoning passes backward verification, then all lower-level steps are also correct. Does upper-level reasoning happen to appear logically consistent while masking subtle arithmetic or semantic errors in lower layers?
3. The pruning decision depends on a calibrated threshold $\tau$ for backward consistency. If $\tau$ is too low, false positives occur (invalid reasoning passes verification); if too high, pruning rarely triggers, negating efficiency gains. Any robust and adaptive method for tuning $\tau$?
4. The overall novelty is incremental given some previous work, e.g., Atom of thoughts for Markov llm test-time scaling, and the work is missing some important references, e.g., Markov chain of thought for efficient mathematical reasoning.

### Questions
Refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes CLoT (Cognitive Loop of Thought), which is a reasoning method where the model generates forward chain-of-thought and also checks if each step can be logically reversed. If a step cannot be reversed to recover the previous information, the model identifies it as incorrect and revises only that part instead of restarting everything (a form of self-correction). The reasoning is organized from high-level plans to detailed steps, so verification begins at the top and only goes deeper if needed, making the process efficient. This approach improves accuracy and uses fewer tokens compared to standard CoT or self-consistency methods.

### Strengths
1. The work is well inspired and well motivated
2. It is good to see that the authors have not only chased after final performance, but rather showed that this improvement comes with same number of tokens
3. The dataset released would be useful to the community 
4. The idea, from what I can tell is a novel contribution

I do have some critiques, Please see weaknesses

### Weaknesses
1. The approach seems math specific ( or at least specific to deductive reasoning). While this is true, the title and abstract does not mention this explicitly. And this is a concern for me. 
2. I can understand the use of LLMs to polish a paper for grammar and typos (which the authors also declared in appendix), but I feel In the process of polishing with LLMs, this has gotten unnecessarily verbose. 
3. Further It is unclear to me why so much mathematical equations are used in the main body of the paper, while the concept can probably be explained in much simpler words ? For example I see equation 4 defined a   L_rhmc, but I cant see any algorithm directly optimizing it, or even using it as metric. 
4. It is hard to understand the method in detail unless I see the prompts, so I would request the authors to kindly share them with the reviewers, and later add to the main paper.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
