# Make Mathematical Reasoning Adaptive

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Mathematical reasoning is a primary indicator of large language models (LLMs) intelligence.
However, existing LLMs exhibit failures of robustness and generalization.
This paper attributes these deficiencies to spurious reasoning—i.e., producing answers from superficial features. 
To address this challenge, we propose the AdaR framework to enable adaptive reasoning, wherein models rely on problem-solving logic to produce answers.
AdaR synthesizes logically equivalent queries by varying variable values, and trains models with Reinforcement Learning with Verifiable Rewards (RLVR) on these data to penalize spurious logic while encouraging adaptive logic.
To improve data quality, we extract the problem-solving logic from the original query and generate the corresponding answer by code execution and then apply sanity check.
Experimental results demonstrate that AdaR improves robustness and generalization, achieving substantial improvement in mathematical reasoning while maintaining high data efficiency.
Analysis indicates that data synthesis and RLVR function in a coordinated manner to enable adaptive reasoning in LLMs.
Subsequent analyses derive key design insights into the effect of critical factors and the applicability to instruct LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method for synthesizing mathematical reasoning training data to expand verifiable training samples for reinforcement learning. The method transforms existing mathematical reasoning data into a problem template and corresponding answer-generating code. By introducing controllable perturbations, it generates a set of similar problems with varying conditions. The model trained with reinforcement learning on the constructed dataset achieves superior performance on multiple benchmarks compared with direct SFT, RLVR, and other synthetic data approaches, demonstrating the effectiveness of the proposed method.

### Strengths
**[S1]** The idea of converting problems into templates and executable code is reasonable, as it allows for controlled generation of diverse training data.  
**[S2]** The paper is clearly written, and the proposed method is easy to follow.  
**[S3]** The paper provides detailed descriptions of the experimental setup and releases code, demonstrating good reproducibility.

### Weaknesses
**[W1] Unsubstantiated claim:** In Section 2.2, the paper claims that SFT is not suitable for training. This statement appears anecdotal and lacks empirical evidence. The authors should provide experiments to substantiate this claim.  

**[W2] Unclear data sampling strategy:** While ORCA-MATH contains 200k samples, this paper only uses 9k of them. However, the authors did not explain how these 9k samples were selected. A detailed description of the sampling strategy is necessary.

**[W3] Inappropriate title:** The current title “Making Mathematical Reasoning Adaptive” does not accurately reflect the adaptive nature of the proposed method. The authors should reconsider their title to better align with the paper’s actual contributions.

**[W4] Inconsistent performance gains:** Compared with Standard-RLVR, AdaR achieves about a 10% average improvement on Qwen2.5-MATH, but only ~2% on DeepSeekMath and ~3% on Llama3. The authors should provide a detailed explanation for these discrepancies, especially regarding how the choice of base model affects performance gains.

**[W5] Evaluation stability:** I appreciate that the authors evaluated avg@32 on AIME25 to reduce evaluation variance. However, similar evaluations should also be conducted on other datasets to ensure stability and provide more reliable conclusions.

**[W6] Lack of training dynamics analysis:** In reinforcement learning, training dynamics are often more informative than a single performance number. The paper does not analyze training dynamics, which limits the insights provided to the reader.

**[W7] Lack of comparison with existing work:** The idea of constructing problems from templates has been explored in prior work [1]. The authors should clarify the key differences between their approach and existing methods.


Ref:  
[1] Mirzadeh et al. GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models. ICLR 2025.

### Questions
**[Q1]** In Table 2, the fourth row is empty. Did the authors forget to include the experimental results?

**[Q2]** For the baseline experiments, were MetaMath and MathGenie trained using SFT or RLVR?

**[Q3]** AdaR generates more than 9k data samples through templates. In Table 1, how was data volume controlled for fair comparison? Did AdaR and Standard-RLVR use the same number of training samples and RL training steps?

**[Q4]** Could the authors provide some examples of templates and codes? This would help readers better understand the practical effect of the proposed method.

### Soundness
2

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
4

### Summary
The paper proposes the AdaR framework to enhance the mathematical reasoning abilities of LLMs. The core problem AdaR addresses is the models' tendency toward spurious reasoning, where they rely on superficial features rather than the actual problem-solving logic, leading to failures in robustness and generalization when variable values change.

The AdaR framework works by enabling adaptive reasoning, which allows LLMs to adapt to varying variable values when the underlying problem-solving logic is preserved. This is achieved through two main contributions: First, a data synthesis method generates logically equivalent problems by controllably perturbing the numerical values in the variable set. The ground-truth answers are obtained by code execution and validated to ensure high-quality data. Second, a training strategy uses RLVR on this synthetic data. This approach penalizes models that exhibit spurious reasoning by failing on the perturbed queries, thereby compelling them to learn the genuine, adaptive problem-solving logic.

Analysis confirms that AdaR successfully induces adaptive reasoning by improving models' algebraic thinking and increasing their adherence to logical order.

### Strengths
* The paper introduces a new framework which directly targets and mitigates the problem of spurious reasoning in LLMs. This focused approach enhances the robustness and generalization of LLMs in mathematical problem-solving tasks.

* A new data synthesis method that generates logically equivalent problems by perturbing numerical values, combined with sanity check. This ensures the training data accurately reflects a preserved problem-solving logic.

### Weaknesses
* The data synthesis approach, which involves perturbing numerical values to create logically equivalent problems, is not fundamentally novel. Similar methods, such as those discussed in the GSM-Symbolic paper [1], have previously highlighted models' lack of robustness to simple numerical variations. Although the authors claim their method eliminates human annotation, this is not a substantial innovation.

* The evaluation is not fully convincing because the performance gains are demonstrated on non-reasoning models, and scores on hard benchmarks like AIME remain quite low. It is highly probable that the benefits of AdaR could diminish when compared against test-time scaling or models trained on significantly larger datasets.

References:

[1] GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models

### Questions
1. The paper title is kind of confusing, the goal is to make LLMs more robust and generalizable in mathematical reasoning. I do not see how "adaptive reasoning" fits in here.

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
4

### Summary
This paper investigates the robustness and generalization failure of large language models (LLMs) in mathematical reasoning, attributing these issues to *spurious reasoning* — the tendency of models to rely on superficial features rather than true problem-solving logic.  
To address this, the authors propose **AdaR**, a framework designed to induce *adaptive reasoning* through two key components:  
1. **Data Synthesis** – generating logic-equivalent problems by perturbing variable values while keeping the underlying reasoning structure fixed.  
2. **Reinforcement Learning with Verifiable Rewards (RLVR)** – using correctness feedback on perturbed problems to penalize spurious reasoning and reward adaptive logic.  

Experiments on several mathematical reasoning benchmarks (e.g., GSM8K, ORCA-MATH, GSM-SYM, MATH, AIME) show consistent performance improvements (+8.5 points on average) across different base models (Qwen2.5-Math, DeepSeekMath, LLaMA3).

### Strengths
1. **Clear motivation** – The paper addresses an important and well-recognized problem: the lack of robustness and generalization in LLMs’ mathematical reasoning.  
2. **Intuitive framework** – The idea of perturbing variable values while preserving logic is intuitive and well-illustrated.  
3. **Automatic pipeline** – The data synthesis process is largely automated and uses executable code for answer verification, reducing annotation cost.  
4. **Interesting metric proposal** – The introduction of “Influence to Logical Order (ILO)” offers a creative, though not fully formal, metric for reasoning robustness.

### Weaknesses
1. **Limited novelty** – The main idea of combining variable perturbation with RLVR is incremental compared to prior works such as *MetaMath* and *MathGenie*. The notion of “adaptive reasoning” lacks formal definition and mainly reframes existing robustness concepts.

2. **Insufficient theoretical and empirical grounding** – The paper does not provide a rigorous formulation or proof that AdaR truly induces adaptive reasoning. Experiments focus only on numeric perturbations, without testing structural or logical variations.

3. **Lack of strong baselines (Major)** – The paper only compares with relatively early baselines such as *MetaMath* and *MathGenie*. However, after the release of *DeepSeek-R1* and other strong RLVR-based reasoning frameworks in 2025, several more competitive datasets and methods have become available. A fair comparison with these recent baselines is essential to convincingly demonstrate the contribution and significance of the proposed dataset.

4. **Scalability and practicality concerns** – The Sanity Check process, especially the validation of solution existence, is computationally expensive (~600s per 1K samples), raising doubts about the framework’s scalability to larger datasets.

### Questions
1. How does the AdaR dataset compare with datasets like Synthetic-1 or BigMath?

2. How much of the observed improvement arises from the data synthesis itself versus the RLVR training process? Would SFT on the same synthetic data yield similar gains?

### Soundness
2

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
3

### Summary
This paper proposes a refinement of Reinforcement Learning with Verifiable Rewards (RLVR, introduced in Lambert et al. 2025) for fine-tuning LLM to reduce spurious reasoning in mathematical problem solving. It introduces a framework for Adaptive Reasoning that aims to incentivize the LLM to provide a logic or program that can solve not only a given query but also similar queries with different variable parameters. The training data set for fine-tuning is created from a 9k subset of ORCA-Math, using an LLM to generate query templates and problem logics for supervised reinforcement learning. In contrast to RLVR, which rewards correct answers, AdaR also checks for the correct problem-solving logic. Qwen2.5-MATH, DeepSeekMath, and Llama3 are tested on GSM8K, ORCA-Adar-test (a 2.5k subset of ORCA-MATH without the 9k for AdaR training), GSM-SYM, MATH, CollegeMATH, CollegeMath, TheoremQA, and AIME, fine-tuned with AdaR and compared against supervised fine-tuning (SFT) with the negative log-likelihood objective, and two other methods, MetaMATH (Yu et al., 2023), and MathGenie (Lu et al., 2024). The paper reports small to moderate performance improvements with AdaR across all the different datasets.

### Strengths
Overall, the paper presents a solid approach to enhancing LLM fine-tuning for mathematical problem-solving. The summary of the method and experiments is well-structured and easy to follow. While the core idea of incentivizing LLM to mimic adaptive reasoning to improve CoT is not new, the implementation of adapting RLVR and synthesizing perturbed data for the fine-tuning appears to be well-executed.

### Weaknesses
1.  The main weakness is the lack of any proper discussion of the limitations of the proposed method. As fine-tuning on LLM-synthesized problem perturbations may improve LLM performance to some degree, the results indicate that there are still relevant limitations, either in adaptive reasoning in principle or in AdaR's ability to enforce such reasoning. In its current form, the Conclusion does not discuss the results at all, but rather provides a mere summary (which is completely redundant with the introduction). Besides a discussion of results, the paper could also include a slightly more theoretical discussion on what we can expect from such a method, attempting to get LLM to mimic adaptive reasoning.
2.  A second weakness could be the limited scope of the paper, mainly providing a fine-tuning approach with moderate results, but as I am no expert in LLM-finetuning, I am not sure about the expected scope of an ICLR paper in this area.

### Questions
Besides the addressed weaknesses of a missing discussion of results and limitations of the approach and method, the following could improve the clarity of the abstract:

1.  The abstract should be more precise and include that AdaR is an LLM fine-tuning approach.
2.  RLVR should be spelled out and properly referenced in the abstract.

Also note that Figure 1 is too small to be properly readable when printed out, and even at a reasonable zoom level.

### Soundness
2

### Presentation
3

### Contribution
2
