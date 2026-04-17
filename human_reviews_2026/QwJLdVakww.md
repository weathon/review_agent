# Surgical Trimming: Minimal Sufficient Chain of Thought with RazorReward-RL

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 2, 4

## Abstract
Recent advances in chain-of-thought (CoT) and post-training have improved LLMs’ reasoning abilities, but often at the cost of generating redundant steps, leading to wasted computation and increased latency in real-time applications. Existing reinforcement learning
(RL) approaches attempt to condense CoT by rewarding brevity, but they fall short in two key aspects: (1) For highly difficult queries, they waste tokens on hopeless reasoning attempts; (2) For medium-difficulty queries, models either stop too soon and miss the answer, or continue beyond the correct answer and introduce errors. To address these issues, we propose RazorReward—a novel reward scheme that sharply differentiates optimal from suboptimal reasoning. For hard queries, RazorReward penalizes unnecessary CoT steps
and encourages abstention when no solution is possible. For medium-difficulty queries, it rewards only reasoning paths that match the minimal sufficient CoT steps, heavily penalizing both under- and over-reasoning. Building on this, we introduce RazorReward-RL, a novel RL framework that segments CoT into semantically meaningful blocks, enabling more precise early stopping and targeted reward allocation. Extensive experiments on six reasoning benchmarks show that RazorReward-RL consistently outperforms previous methods, boosting accuracy by 8.3%–9.3% while reducing average token usage by 38.4%–43.8%, thus achieving a better balance between accuracy and efficiency

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes RazorReward-RL, a reinforcement learning framework designed to reduce redundant reasoning in large language models by identifying and enforcing the minimal sufficient chain of thought (CoT). The authors introduce a novel RazorReward function that imposes sharp penalties for both under- and over-reasoning relative to the optimal CoT length, improving reasoning efficiency without sacrificing accuracy. Through structural segmentation of CoT into semantic blocks, the method achieves more precise reward allocation. Experiments across six reasoning benchmarks show consistent gains in accuracy and substantial reductions in token usage.

### Strengths
The proposed method itself is coherent and easy to follow. The authors conduct experiments on several benchmarks to validate the effectiveness of the proposed method.

### Weaknesses
While RazorReward-RL presents a promising framework for mitigating overthinking in reasoning models, several weaknesses limit its theoretical depth, methodological generality, and scalability.

**(1) Incremental contribution.** Although the paper introduces a new reward shaping function, the innovation is largely an extension of existing RL-based CoT optimization methods such as DAST and ShorterBetter. The central idea—penalizing both under- and over-reasoning—is conceptually intuitive and has been partially explored in previous work. The main novelty lies in the “razor-like” sharpness of the penalty, which, while effective, may not constitute a fundamentally new learning paradigm.

**(2) Lack of theoretical justification.** The paper provides no formal theoretical analysis explaining why the minimal sufficient CoT principle should universally optimize reasoning efficiency and correctness. There is no proof or theoretical model showing how reward sharpness affects convergence stability or sample efficiency in RL training. As a result, the work remains primarily empirical, lacking a solid mathematical foundation to justify its design choices.

**(3) Heavy reliance on handcrafted components.** The segmentation lexicon for identifying reasoning blocks and the rule-based structure of the RazorReward function introduce hand-engineered bias. This reliance on manually defined linguistic markers may limit adaptability to other languages, domains, or reasoning styles. Moreover, small errors in segmentation could misassign rewards, leading to training instability.

**(4) Limited scalability and model size.** Experiments are conducted only on the 1.5B and 7B DeepSeek-R1-Distill-Qwen backbones. These are relatively small models compared to modern reasoning LLMs (e.g., 30B–70B). It remains unclear whether the proposed framework maintains its benefits or efficiency when scaled up. Larger-scale validation would strengthen the generality claim.

**(5) Narrow domain evaluation.** The benchmarks focus almost exclusively on mathematical and scientific reasoning. It is uncertain whether the same reward dynamics hold for open-domain, commonsense, or multimodal reasoning, where the notion of “minimal sufficient reasoning” is less clearly defined.

### Questions
1.	How sensitive is performance to the segmentation lexicon and its coverage across reasoning styles or languages?
2.	Can the approach be extended to open-ended or dialogue reasoning tasks, where “minimal sufficient” reasoning is ill-defined?
3.	How stable is training when applied to larger models (e.g., ≥30B parameters)?

### Soundness
3

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
2

### Summary
The paper proposes RazorReward-RL, a reinforcement learning framework designed to reduce redundant reasoning (overthinking) in large language models’ chain-of-thought (CoT). It introduces RazorReward, a novel reward function that enforces sharp penalties for both under- and over-reasoning relative to the minimal sufficient reasoning length. The method segments CoT into semantically coherent reasoning blocks using a lexicon-based approach and applies a combined reward of correctness, format, and length efficiency. Experiments on six reasoning benchmarks (GSM8K, MATH500, AIME24/25, AMC23, GPQA-D) using DeepSeek-R1-Distill-Qwen-7B/1.5B show accuracy gains of 8–9% and token reductions of 38–44%, achieving the best accuracy-efficiency trade-off compared to SOTA baselines such as DAST and ShorterBetter. Ablation and out-of-distribution tests further confirm that RazorReward-RL yields more concise, focused, and generalizable reasoning without sacrificing correctness, offering a principled solution to the LLM overthinking problem.

### Strengths
* Novel Reward Design (RazorReward): The proposed length-sensitive, sign-sharpened reward sharply separates optimal from suboptimal reasoning. It improves upon prior smooth reward schemes (e.g., DAST, ShorterBetter) by introducing exponential decay and fixed post-optimum penalties that tightly constrain CoT length.

* Balanced Trade-off Between Accuracy and Efficiency: Results show RazorReward-RL reduces average token count by ~40% while achieving up to +9% absolute accuracy gain — a compelling demonstration that reasoning brevity need not compromise correctness.

* OOD Generalization Evidence: The additional experiments on MMLU and GPQA-Diamond show transferability beyond math domains, supporting claims of generalizability.

### Weaknesses
* Reward Sensitivity & Hyperparameter Study Missing: No ablation or sensitivity analysis is provided on the exponential decay constants or penalty weights (especially within Eq. (2)). This omission limits understanding of stability and optimality during RL training.

* Limited Backbone Diversity: Experiments rely solely on Qwen-Distill backbones (7B and 1.5B). The framework’s behavior on instruction-tuned, general-purpose, or reasoning-specialized models (e.g., DeepSeek-R1-Distill-70B, OpenAI-o1) remains untested.

### Questions
See the Weaknesses

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
This paper proposes **RazorReward-RL**, a reinforcement learning framework designed to minimize redundant reasoning in large language models. The key idea is **RazorReward**, a sharp reward mechanism that penalizes both over-reasoning relative to the minimal sufficient chain of thought (CoT). The method segments reasoning into semantically meaningful blocks and applies fine-grained rewards to encourage concise yet complete reasoning. Experiments on six reasoning benchmarks show that RazorReward-RL improves accuracy by around 8–9% while reducing token usage by over 38%, achieving a strong balance between accuracy and efficiency.

### Strengths
Strengths:  
1. Experiments across multiple benchmarks demonstrate consistent improvements in both accuracy and efficiency.  
2. The proposed approach is simple to implement and could generalize well to other RL-based reasoning frameworks.

### Weaknesses
Weaknesses:  
1. The evaluation setup limits models to a 2048-token context window, which is not a fair comparison, as most recent reasoning models are trained or evaluated with much longer contexts. Efficient reasoning should emphasize the trade-off between accuracy and efficiency rather than constraining prior methods with artificially short contexts.  
2. The paper encourages shorter reasoning traces even on hard questions that the model fails to solve. The motivation for this design is unclear—if a model cannot answer a hard problem correctly with extended reasoning, it seems counterintuitive that reducing reasoning length would help.  
3. The comparison mainly focuses on quantitative improvements but provides limited **qualitative analysis** of reasoning behaviors (e.g., what kinds of redundant reasoning are trimmed).  

Statement:  
I assign a score of 2 at this stage mainly due to concerns about the current evaluation setup, which I believe has major issues affecting the fairness and validity of the results. However, if the authors can provide a more reasonable and comprehensive evaluation—demonstrating fair comparisons and stronger evidence for the claimed efficiency gains—I would be willing to reconsider and raise my score, potentially to a positive rating. I will revisit this paper carefully once the evaluation concerns are properly addressed.

### Questions
See weakness

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
4

### Summary
This paper aims to solve the "overthinking" problem for large reasoning models. The authors propose the RazorReward-RL framework. This framework segments the CoT into semantically meaningful blocks to construct training samples and mitigates overthinking through a novel reward mechanism (RazorReward). For hard queries, it penalizes unnecessary steps and encourages abstention; for medium-difficulty queries, it rewards only the paths that exactly match the "minimal sufficient CoT" and heavily penalizes both under- and over-reasoning. Experimental results show that RazorReward-RL outperforms previous methods on multiple math reasoning benchmarks, improving model accuracy while significantly reducing average token usage.

### Strengths
1. The authors propose a novel, annotation-free difficulty classification and reward mechanism. The paper devises a novel CoT truncation sampling framework and, by analyzing the answer correctness at different truncation lengths, automatically classifies queries into different difficulty levels. This classification is adaptive and does not require additional human annotation. Based on this classification, RazorReward can design highly targeted reward functions.


2. Experimental results demonstrate that it can achieve good results in accuracy with fewer tokens.

### Weaknesses
1. The reward mechanism might encourage guessing rather than sufficient reasoning. The design of RazorReward is to heavily reward the first truncated CoT that can guess the correct answer while severely penalizing all subsequent steps, assuming the remaining to be redundant. This is not very promising from some perspectives. First, the shortest CoT that can guess the correct answer does not seem to necessarily align perfectly with a logically complete "minimal sufficient reasoning." Second, many studies have shown that "Verification" steps are crucial for improving the final accuracy of complex reasoning. Although excessive verification is redundant, the paper's reward function penalizes all steps for verification, which discourages the model from performing any form of verification, even if it is helpful for making the reasoning robust. 

2. In terms of experimental setup, the current experiments for training and testing are limited to mathematical reasoning, and the models used are distilled versions of DeepSeek-R1. More experiments and models are needed to prove whether the lexicon-based structural reasoning blocks possess generalization capabilities, rather than being limited to specific tasks or the writing style of DeepSeek-R1.

### Questions
1. In the methodology section (3.1.1), the paper emphasizes that one of its core contributions is using lexicon-based "semantic chunking" to construct samples. However, in the subsequent "Reasoning Quality Analysis" (4.3.1) , the evaluation suddenly switches to the "line-level". Why not continue to use the previously defined "semantic blocks" as the basic unit of analysis?

2. Regarding the "encouragement of abstention", is the model allowed to generate a specific "I don't know" or "unsolvable" token to receive a reward?

3. What is the "Vanilla" setting in the table?

4. Does the reward designed by the authors help improve the model's accuracy? Why does it achieve a significant accuracy improvement compared to Basic RL?

### Soundness
2

### Presentation
3

### Contribution
2
