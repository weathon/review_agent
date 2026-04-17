# Critique-Coder: Enhancing Coder Models by Critique Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 4

## Abstract
Reinforcement Learning (RL) has emerged as a popular training paradigm, particularly when paired with reasoning models. While effective, it primarily focuses on generating responses and lacks mechanisms to explicitly foster critique or reflection. Several recent studies, like Critique-Fine-Tuning (CFT) and Critique-Guided-Distillation (CGD) have shown the benefits of explicitly teaching LLMs how to critique. Motivated by them, we propose Critique Reinforcement Learning (CRL), where the model is tasked with generating a critique for a given (question, solution) pair. The reward is determined solely by whether the final judgment label $c \in \{\texttt{True}, \texttt{False}\}$ of the generated critique aligns with the ground-truth judgment $c^*$. Building on this point, we introduce Critique-Coder, which is trained on a hybrid of RL and CRL by substituting 20\% of the standard RL data with CRL data. We fine-tune multiple models (Critique-Coder) and evaluate them on different benchmarks to show their advantages over RL-only models.  We show that Critique-Coder consistently outperforms RL-only baselines on all the evaluated benchmarks. Notably, our Critique-Coder-8B can reach over 60\% on LiveCodeBench (v5), outperforming other reasoning models like DeepCoder-14B and GPT-o1. 
Beyond code generation, Critique-Coder also demonstrates enhanced general reasoning abilities, as evidenced by its better performance on logic reasoning tasks from the BBEH dataset. This indicates that the application of CRL on coding datasets enhances general reasoning and critique abilities, which are transferable across a broad range of tasks. Hence, we believe that CRL works as a great complement to standard RL for LLM reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes integrating *critique data* into reinforcement learning from verifiable rewards (RLVR) for large language models (LLMs). The critique rewards are binary (1–0), indicating whether the model’s judgment is correct relative to a reference solution. The study focuses on coding tasks, where solutions can be automatically verified using unit tests. Experimental results demonstrate that incorporating critique data leads to significant performance improvements on coding benchmarks such as **LiveCodeBench**.

### Strengths
* The improvements over baseline models are substantial.
* The paper is clearly written and easy to follow.
* The experiments are carefully designed, with appropriate baselines and ablation studies.
* Overall, the proposed approach is simple yet effective, and the empirical results are convincing.

### Weaknesses
* The idea of using critique data is not entirely novel, as similar concepts have been explored in supervised fine-tuning (SFT). However, to the best of my knowledge, this is the first work to apply critique data within an RL setting with verifiable rewards.
* The contribution is somewhat incremental, as it mainly involves augmenting an existing RL framework with additional data.

### Questions
The authors note that

> Although incorporating CRL enhances the model’s reasoning and critique abilities, it does not exhibit self-critique capability.

They further present an experiment showing that having the model critique its own solutions does not improve performance. Typically, increasing compute at test time tends to yield better results. Why does this not hold in this case? Could the authors elaborate further on the intuition or mechanisms behind this observation?

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
4

### Summary
This paper introduced critique reinforcement learning (CRL). It is a training paradigm designed to improve models' reasoning and reflective abilities for code generation tasks. It  explicitly trains model to  critique and judge the correctness of generated solutions. 
So in addition to traditional outcome-based signal, they introduced a new signal regarding the correctness on question-solution paris. The experiments on several code generation tasks show improvements after leveraging critique reinforcment learning.

### Strengths
1. This paper introduced a new perspective which is to leverage critique signals to improvement the performance during reinforcement learning.
2. Tis work uses test case execution results to automatically generate judgement labels, enabling scalable data creation without human annotation.

### Weaknesses
1. The improvement of 8B model is limited. For example, on LiveCodeBench, it improves 1.2% and on Aider-Polyglot, it improves 1.1%. Setting aside noise and variance, I observe that as the model size increases, the performance gain diminishes. This raises concerns about whether the proposed method can effectively scale up.

2. For CRL process, where are the groundtruth judgement label c* from? From Figure 3 and page 4, it looks to me these judgement labels are from the text case execution and its pass rate. However, during normal RL training, we  have already had such signals from pass rate of each rollout solution. The proposed method just make it happen earlier and separately. I wonder how much gains could obtain from this process and what's the real source of improvement.

3. In the current implementation, how do you aggregate two rewards: reward from RL and reward from CRL? Since multi-objective RL is not very robust, Is there any instability or robustness issue with different aggregation methods?

### Questions
N/A

### Soundness
2

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
4

### Summary
This paper presents Critique Reinforcement Learning (CRL), a training paradigm designed to teach large language models to critique solutions rather than merely generate them. Given a (question, solution) pair, the model produces a critique containing a final binary judgment  c∈{True,False}, which is rewarded based on whether it matches the ground-truth correctness label . The authors further propose CRITIQUE-CODER, trained with 80% standard RL data and 20% CRL data. Empirical evaluations on EvalPlus, BigCodeBench, LiveCodeBench v5, and BBEH benchmarks show that CRITIQUE-CODER consistently outperforms RL-only baselines, including DeepCoder-14B and GPT-o1, especially in code reasoning tasks.
While the paper is well-executed and clearly written, its claim of being “verifier-free” and “self-improving” is only partially justified, since CRL fundamentally depends on verifiable signals derived from executable test cases and strong model–generated solutions.

### Strengths
1. Addresses a timely question: how to endow models with critique and reflection abilities.

2. Clear algorithmic presentation (Algorithm 1) and reproducible training setup.

3. Comprehensive experiments across multiple code benchmarks.

4. Demonstrates that mixing 20% CRL data improves robustness and mitigates policy collapse.

5. Writing quality, figures, and clarity of exposition are strong.

### Weaknesses
1. Dependence on verifiable signals and strong model–generated solutions.
The ground-truth labels 
are obtained through executable test cases with a pass-rate threshold (80%), and the candidate solutions are generated by a stronger model (Qwen3-30B). Although no external verifier module is used, this still constitutes a verifiable supervision pipeline, limiting applicability to domains where executable feedback is available.

2. Limited differentiation from prior critique-based methods.
CRL is conceptually close to CFT and CGD, differing mainly in the reinforcement reward design. The absence of direct experimental comparisons leaves its unique contribution ambiguous.

3. Sparse reward and lack of training dynamics analysis.
The reward is binary (0/1), yet the paper provides no discussion of variance, convergence stability, or reward sensitivity to the pass-rate threshold.

4. “General reasoning” claims rely on narrow evidence.
Improvements on the BBEH dataset are encouraging but cover only one reasoning benchmark. There is no evidence of gains in non-verifiable or natural-language reasoning domains.

5. Absence of self-critique capability (acknowledged by authors).
Section 3.7 explicitly reports that enabling the model to critique its own outputs does not improve performance, suggesting the learned skill is evaluative (“other-critique”) rather than reflective (“self-critique”).

### Questions
1. How would CRL perform if the candidate solutions were generated by a weaker or equal-capacity model?

2. Have you attempted CRL in non-code domains lacking executable verification?

3. Could you provide training curves or reward variance to assess stability under binary rewards?

4. What differences exist between your approach and CFT/CGD when controlling for compute and data volume?

5. Why do self-critique variants fail—does this relate to policy–critic decoupling or distributional mismatch?

### Soundness
3

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
This paper proposes training code generation models on a mixture of two tasks: standard code generation and critique evaluation. The critique task involves judging whether a given solution correctly solves a problem, receiving binary rewards based on judgment accuracy. The authors train Critique-Coder using GRPO on 80% code generation data and 20% critique data, fine-tuning Qwen3-4B and Qwen3-8B models. Results show improvements across coding benchmarks, with Critique-Coder-8B achieving 60.8% on LiveCodeBench (v5). The approach also improves performance on BBEH logical reasoning tasks, suggesting the critique task develops transferable reasoning skills beyond coding.

### Strengths
- Comprehensive evaluation across diverse benchmarks (LiveCodeBench, EvalPlus, Aider-Polyglot, BigCodeBench) demonstrates consistent improvements over single-task training baselines.
- Evidence of transfer to general reasoning (BBEH benchmark) suggests the critique task develops broadly applicable skills rather than narrow task-specific capabilities.
- Ablation study provides practical guidance, showing 20% critique data mixing is optimal. Too much critique data (50-100%) degrades performance due to train-test mismatch.

### Weaknesses
- Critique training data is generated using a different model (Qwen3-Coder-30B-A3B-Instruct) rather than the model's own solutions. This creates a fundamental distribution mismatch, the model learns to critique a 30B model's solution distribution, not its own, which likely explains the complete failure of test-time self-critique in Section 3.7.
- The paper is missing the obvious experiment which would be on-the-fly critique generation where models critique their own on-policy solutions during training. The paper currently takes a shortcut using pre-generated data, that too from a different model. 
- Confusing terminology—"CRL" and "RL" both refer to GRPO training on different tasks (critique vs. generation). This is fundamentally a  multi-task learning setup rather than a novel training algorithm.
- Due to lack of proper ablations, it's hard to discern the reasons for the performance boost. The observed improvements could stem from multiple confounds: simple multi-task learning effects, knowledge distillation from the 30B model, or just increased training compute. The paper doesn't isolate the actual source of gains.

### Questions
- Why generate critique data from a different 30B model instead of using the training model's own solutions? Did you consider on-the-fly critique generation during training?
- Have you investigated whether the test-time critique failure is specifically due to the distribution mismatch between training critiques (from 30B model) and test-time critiques (of own solutions)?
- Can you provide ablations isolating the source of improvements: is it multi-task learning, distillation from the larger model, or genuine critique capability?

### Soundness
2

### Presentation
2

### Contribution
2
