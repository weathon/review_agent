# ENCOURAGING CRITICAL THINKING FOR MULTIAGENT DEBATE

- Decision: Reject
- Scores: 6, 2, 2, 4

## Abstract
Large language models (LLMs) have demonstrated remarkable performance across a wide range of tasks in recent years. While prior work has explored leveraging LLMs to generate synthetic data for self-improvement, repeated iterations often suffer from diminishing returns due to the reliance on homogeneous reasoning patterns and limited exploration of alternative perspectives. In this paper, we introduce a novel framework that enriches the reasoning process by encouraging critical thinking among multiple agents. Rather than deploying an ensemble of models with identical prompts, we propose a strategy generator that produces customized instructions tailored to each individual LLM. Acting as a critical thinking agent, the generator is iteratively fine-tuned using carefully selected strategies that are both diverse and effective. This approach fosters specialization within each model while promoting diversity across reasoning paths, enabling the system to maintain varied solution trajectories and achieve sustained performance gains through iterative refinement. We demonstrate the effectiveness of our method across a variety of agentic frameworks and complex reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper shows how to improve LLM performance using multiagent debate extended with several new contributions: diversification of the reasoning paths taken by different agents, critical thinking. The diverse paths are obtained by a strategy generator that generates M strategies used to prompt M agents.

### Strengths
The approach is interesting and the experimental results seem compelling.

### Weaknesses
One overall weakness for this line of work, not just restricted to this particular contribution, is that it is not clear why this approaches lead to better performance. I do not count this remark against this paper as I think that these experimental results are important. 

The diversity metric seems to be a key element of the approach. It would be interesting to see some alternative measures of diversity and how they impact the performance of the algorithm.

Minor comments:

Line 153, where it says “answer, denoted as y_{1,i}, where the”, I believe it should say y_{i,1}

Where it says “table 4.1” it should say “table 1”

### Questions
The similarity threshold \tau to evaluate the diversity of the proposed strategies could be context dependent. In some cases, a large diversity might be needed, while in other cases it might be difficult to propose very different strategies. How do you chose this parameter and have you observed context dependent differences? The results from figure 5 are a first step in this direction. 

I am not convinced that “Critical Thinking” is what the approach is doing. Could you denote in algorithm 1, what part is the one responsible of critical thinking?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Critical Thinking with Multi-Agent Debate (CMAD), a framework for improving the reasoning capabilities of LLMs by training a strategy generator to generate diverse, undefined reasoning strategies. The framework iteratively fine-tunes the generator using feedback on correctness (based on majority voting) and diversity (based on a similarity metric), aiming to balance exploration and exploitation. Empirical results on MATH, GSM8K, and GPQA show consistent improvements across several LLMs compared to baselines such as DMAD and CoT.

### Strengths
1. The idea of using a trainable strategy generator to produce undefined reasoning paths is creative and differentiates CMAD from prior multi-agent debate like DMAD.
2. The paper evaluates across multiple benchmarks and models (GPT-4o-mini, LLaMA-3, Qwen2.5, Nova Micro), and compare with various baseline methods.
3. The introduction convincingly argues the need to move beyond homogeneous reasoning and fixed strategies.

### Weaknesses
1. The process of solution sharing and summarization risks contaminating the agents’ independent reasoning based on their given strategies. If each agent accesses others’ intermediate solutions, the resulting fine-tuning data may lose diversity and no longer reflect distinct strategies. The authors should clarify how they prevent such convergence or bias.
2. The paper focuses on the Multi-Agent Debate (MAD) setting, but it does not explain why this setting is necessary over simpler mechanisms such as majority voting or ensemble averaging. Clarifying this design choice, especially how debate interaction benefits strategy generation beyond aggregation, would strengthen the motivation.
3. Figure 1 is visually cluttered; the text overlaps and the color scheme makes it difficult to interpret. 
4. The related work section omits prior studies exploring similar concepts of using a trainable model to guide another model [1]

References:
[1] Li, Zekun, et al. "Guiding large language models via directional stimulus prompting." Advances in Neural Information Processing Systems 36 (2023): 62630-62656.

### Questions
1. The paper does not specify the underlying model for the strategy generator and solution agents.
2. How do you make sure that each strategy actually contribute to the the final solution, given that each agent can not only see its given strategy but also other's solution.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses the homogeneous reasoning patterns of complex reasoning in LLMs and proposes Critical Thinking with Multi-Agent Debate (CMAD).

CMAD uses a strategy generator that produces reasoning strategies for multiple LLM agents. After multi-round debates, a feedback loop balances correctness and diversity to select high-quality strategies for fine-tuning.

Experiments show the framework is model-agnostic and outperforms baselines on reasoning benchmarks.

### Strengths
- The paper has a good motivation to enable LLMs to generate diverse reasoning strategies instead of relying on fixed prompts (such as CoT, PoT, Step-back).

- The proposed method is simple yet effective, selecting high-quality strategies with both correctness and diversity metrics and using these data to fine-tune a strategy generator.

### Weaknesses
- Inconsistent Reporting of Results

(1) Line 355-356 says "The average improvement over the second-best method ranges from 1.2% to 9.8%." However, Table 1 shows that the performance gaps between CMAD and DMAD (the second-best method) are all less than 5%.

(2) Comparing Table 1 and Table 4, the reported results for baselines are identical, but CMAD’s results differ. What is the difference in evaluation settings between these two tables?

- Missing Important Experimental Details

(1) The paper does not explicitly specify which models were fine-tuned to produce the reported results. While Line 742–743 implies Qwen2.5-7B, Line 700–701 mentions full-model fine-tuning for Qwen1.5-7B and LLaMA-8B. 

(2) The paper does not provide the prompt used for the strategy generator or examples of its training data.

The above two concerns make the results less convincing.

### Questions
- What if we directly use the initial answers with different strategies (instead of going through the full debate process) to construct training data?

- The description of Figure 2 refers to “refine the pre-training data”—should this instead be “fine-tuning data”?

- The description of Table 3 does not align with its contents. Is the “DMAD” listed in Table 3 a typo that should be “CMAD”?

- Reference mistake: 

Line 742: Table C should be Table 4; 

Line 355: Table 4.1 should be Table 1; 

Line 375, the baseline is incorrectly cited as published in 2015; the correct publication year is 2025.

### Soundness
2

### Presentation
1

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
LLMs can be made into "agents" to solve a problem by adding a "Strategy" to the input prompt along with the problem, and then iteratively refining what this strategy is based on how well the LLM solves the problem. However, this needs a way to score answers, which may or may or may not exist.

This paper proposes a method to do so by first instantiating several different such strategies, finding the resulting answers, and using agreements and diversity between these to refine the strategies.

### Strengths
Compares agains a comprehensive set of baselines.

### Weaknesses
Main paper does not contain enough specifics of the method. It is also unclear how it differs from one of the references. (see questions below for both these points)

### Questions
It is unclear what the strategy generator is. Is it an open-weights LLM (if so which one)? It is also unclear what precisely is meant by a "strategy", and how a set of these are generated from a question. A simple example in the main text of the paper would have helped a great deal clarifying this.

What precisely is the difference between the method in this paper and the one in (Subramaniam et al 2025)? Also, it seems this method has not been compared against.

In Table 1, some methods involve no fine-tuning / training of any sort, and others (like CMAD) do. So in some sense some of these are not fair comparisons. At the very least, training-free and fine-tuned approaches should be demarcated as such.

Minor typo: A_i on line 159

How are strategies mapped to vectors (which are needed for the diverse sampling  in line 209)?

### Soundness
2

### Presentation
2

### Contribution
2
