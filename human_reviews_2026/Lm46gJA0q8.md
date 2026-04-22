# Executable Counterfactuals: Improving LLMs' Causal Reasoning Through Code

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 6

## Abstract
Counterfactual reasoning, a hallmark of intelligence, consists of three steps: inferring latent variables from observations (abduction), constructing alternative situations (interventions), and predicting the outcomes of the alternatives (prediction). This skill is essential for advancing LLMs' causal understanding and expanding their applications in high-stakes domains such as scientific research and healthcare. However, existing efforts in assessing LLM's counterfactual reasoning capabilities tend to skip the abduction step, effectively reducing to interventional reasoning and leading to over-estimated LLM performance. To address this, we introduce executable counterfactuals, a novel framework that operationalizes causal reasoning through code and math problems. Our framework explicitly  requires all three steps of counterfactual reasoning and enables scalable synthetic data creation with varying difficulty, creating a new frontier for evaluating and improving LLM's reasoning. Our results reveal substantial drop in accuracy (25-40%) from interventional to counterfactual reasoning for state-of-the-art models such as o4-mini and Claude-4-Sonnet. To address this gap, we construct a training set comprising counterfactual code problems having if-condition and test on out-of-distribution code structures (e.g.,  having while-loop); we also test whether a model trained on code would generalize to counterfactual math word problems.  While Supervised Finetuning (SFT) on stronger models' reasoning traces improves in-distribution performance of Qwen models, it leads to a decrease in accuracy on out-of-distribution tasks such as counterfactual math problems. In contrast, reinforcement learning (RL) induces the core cognitive behaviors and generalizes to new distributions, yielding substantial accuracy gains over the base model on both code (improvement of 1.5x-2x) and counterfactual math problems. Analysis of the reasoning traces further reinforces these findings and highlights the promise of RL with scalable data generation for improving LLMs' counterfactual reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a new benchmark for evaluating counterfactual reasoning in large language models. By framing tasks as executable code and math problems, it tests whether models can perform the full causal reasoning cycle (abduction, intervention, and prediction). Using this setup, the authors find that LLMs, regardless of size, struggle especially with the abduction step. They introduce an LLM-as-a-judge method to rate reasoning quality in terms of planning and execution, revealing that reinforcement learning with verifiable rewards (RLVR) induces more consistent causal reasoning than supervised fine-tuning (SFT). The work provides a structured framework for diagnosing reasoning failures in LLMs.

### Strengths
The paper is original in defining executable counterfactuals, i.e., a new, code-based benchmark that captures causal reasoning process (abduction, intervention, prediction). The technical quality is strong, combining formal causal modeling with large-scale experiments comparing model types and training methods (SFT vs. RL).  In terms of clarity, the presentation is clear and well-structured, using intuitive examples. The significance comes from establishing a scalable framework for testing reasoning in LLMs.

### Weaknesses
The evaluation is limited to synthetic and code-based tasks, leaving unclear how the framework extends to real-world reasoning. Adding at least one natural dataset or human-grounded task would strengthen generalizability.

The LLM-as-a-judge lacks calibration against human evaluators. Without inter-rater validation or multi-judge comparison, reliability of the planning and execution scores remains uncertain.

### Questions
1. How could the proposed framework be adapted to naturalistic or real-world reasoning tasks? Any datasets/benchmarks that could be added?

2. How did the authors validate the accuracy and consistency of the o4-mini LLM-as-a-judge beyond rubric standardization? Was any human–LLM agreement test or multi-judge comparison performed to confirm that planning and execution scores reflect genuine reasoning quality rather than stylistic variance?

3. Any intuition about which parts of the reinforcement pipeline are responsible for the improved results, i.e., the emergence of structured reasoning?

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
This paper highlights that current LLMs perform poorly in identifying counterfactuals within coding tasks. Moreover, the effectiveness of supervised fine-tuning (SFT) remains limited, as it fails to generalize to unseen scenarios. Finally, the authors demonstrate that reinforcement learning (RL) achieves strong performance and shows promising potential for generating counterfactual examples.

### Strengths
Strength:
1 The topic is interesting.
2 The experiments are abundant.
3 The introduction of the template-based generation approach is clear and concise. The inclusion of a concrete example effectively clarifies the goal of the task.

### Weaknesses
Weakness:
1 Although this paper demonstrates the differences among algorithms in counterfactual reasoning, it fails to provide an in-depth analysis of the observed phenomena. In other words, the work reads more like an experimental report than a research paper. For instance, the authors claim that reinforcement learning (RL) exhibits strong generalization ability, yet offer no explanation or supporting analysis for this claim.
2 The paper lacks a clear definition of counterfactual examples, which makes it difficult to understand the exact task at first. In fact, the task only becomes clear upon reading Section 3.

### Questions
Questions and Suggestions:
1 I suggest that the authors provide additional analysis or, if possible, a theoretical guarantee to better explain and support their findings.
2 I recommend adding a preliminary section before Section 3 to clearly define the problem and formalize the task setup.
3 The paper would also benefit from including a discussion of previous work on counterfactual example generation. I list several relevant papers below for reference:
1 Mishra, Ashish, Gyanaranjan Nayak, Suparna Bhattacharya, Tarun Kumar, Arpit Shah, and Martin Foltin. "Llm-guided counterfactual data generation for fairer ai." In Companion Proceedings of the ACM Web Conference 2024, pp. 1538-1545. 2024.
2 Nguyen, Van Bach, Paul Youssef, Christin Seifert, and Jörg Schlötterer. "Llms for generating and evaluating counterfactuals: A comprehensive study." arXiv preprint arXiv:2405.00722 (2024).

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
3

### Summary
The paper investigates whether large language models (LLMs) can perform counterfactual reasoning, which requires the causal sequence of abduction, intervention, and prediction. The authors argue that existing LLM evaluations on counterfactual reasoning often overlook the abduction step, inferring hidden latent variables from factual observations, which is essential in Pearl’s framework.

To make abduction explicit and verifiable, the paper introduces a counterfactual reasoning benchmark in the form of code-based tasks. Each task corresponds to a function $Y = f(X, R_1, R_2, \ldots)$ where $X$ is the input and $R$ represents latent variables sampled inside the function. Given factual observations $X = x$ and $Y = y$, the goal is to infer the support set of the counterfactual $Y$ had $X$ been $x’$. Experiments with open-source models (1.5B–72B) and commercial reasoning models reveal that they consistently fail at counterfactual reasoning due to an inability to perform abduction.

Next, the paper studies whether fine-tuning can induce counterfactual reasoning. Two approaches are compared: Supervised Fine-Tuning (SFT) and Reinforcement Learning with Verifiable Reward (RLVR). Both methods enable strong performance on in-distribution code counterfactual tasks. However, RLVR generalizes significantly better, to functions $f$ with unseen structural patterns and to a natural language counterfactual dataset (GSM8K-style math word problems constructed from causal graphs), while SFT collapses to near-zero performance.

### Strengths
**S1**. The paper highlights the crucial role of inferring latent variables R (abduction) in counterfactual reasoning. This step is often ignored in existing LLM counterfactual evaluations. The paper provides clear motivation and concrete illustrations for why abduction must be explicitly evaluated, and how current methods implicitly avoid it.


**S2**. The proposed code-based benchmark is both systematic and innovative. By embedding latent randomness into executable coding functions, the authors create a counterfactual reasoning task with verifiable ground truth. The methodology can offer insights that could be extended to broader counterfactual reasoning tasks.

### Weaknesses
The definition of counterfactual reasoning used in the paper is narrower than the standard understanding in causal inference.

**W1**. In Pearl’s framework, abduction requires inferring the posterior distribution $P(R \mid x, y)$, followed by computing the counterfactual distribution $P(Y_{x'} | x, y) = \sum_R P(Y_{x'} | r)P(r \mid x, y)$. In contrast, the benchmark in this paper focuses on identifying the **support set** of latent variables consistent with the observation, and then predicting the **support set** of counterfactual outcomes. It is suggested to clarify that the benchmark evaluates support-set inference, not full counterfactual distributions.

**W2**.
Although the paper claims to move beyond graphical approaches, every task ultimately reduces to the functional form $Y = f(X, R_1, R_2, \ldots, R_n)$, without causal dependencies between $X$ and $R$ and other observed variables. As a result, the causal graph collapses to a simple structure $X \rightarrow Y$. Does the benchmark proposed in the work include any tasks with richer causal structures, either in the code setting or the natural language setting?

### Questions
**Q1**. In line 101, the paper states: “infer r based on the observation $y = -1$ (abduction).”
Should this instead be “infer $r$ based on the observation $x = 1, y = -1$”?
Also, does abduction always consider both x and y as observations in this benchmark?

**Q2**. See **W2**

### Soundness
3

### Presentation
3

### Contribution
2
