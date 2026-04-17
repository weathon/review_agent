# MathFimer: Enhancing Mathematical Reasoning by Expanding Reasoning Steps through Fill-in-the-Middle Task

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
Mathematical reasoning represents a critical frontier in advancing large language models (LLMs). While step-by-step approaches have emerged as the dominant paradigm for mathematical problem-solving in LLMs, the quality of reasoning steps in training data fundamentally constrains the performance of the models. 
Recent studies have demonstrated that more detailed intermediate steps can enhance model performance, yet existing methods for step expansion either require more powerful external models or incur substantial computational costs. 
In this paper, we introduce MathFimer, a novel framework for mathematical reasoning step expansion inspired by the ''Fill-in-the-middle'' task from code reasoning.
By decomposing solution chains into prefix-suffix pairs and training models to reconstruct missing intermediate steps, we develop a specialized model, MathFimer-7B, on our carefully curated NuminaMath-FIM dataset. 
We then apply these models to enhance existing mathematical reasoning datasets by inserting detailed intermediate steps into their solution chains, creating MathFimer-expanded versions.
Through comprehensive experiments on multiple mathematical reasoning datasets, including MathInstruct, MetaMathQA and etc., we demonstrate that models trained on MathFimer-expanded data consistently outperform their counterparts trained on original data across various benchmarks such as GSM8K and MATH.
Our approach offers a practical, scalable solution for enhancing mathematical reasoning capabilities in LLMs without relying on powerful external models or expensive inference procedures.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MathFimer, a novel framework designed to enhance the mathematical reasoning capabilities of Large Language Models (LLMs) by improving the granularity and completeness of training data. The core methodology ingeniously adapts the "Fill-in-the-Middle" (FIM) task from the domain of code generation. The authors first train a specialized model, MathFimer-7B, on the NuminaMath-FIM dataset, which is created by decomposing existing high-quality mathematical solutions into prefix, middle, and suffix segments. This model learns to infer and generate missing intermediate reasoning steps.

Subsequently, the trained MathFimer model is applied to expand various existing mathematical reasoning datasets by inserting more detailed steps into their solution chains. The central claim of the paper is that fine-tuning LLMs on these MathFimer-expanded datasets leads to consistent performance improvements across multiple mathematical benchmarks (GSM8K, MATH, etc.) and base models (Llama-3.1, Qwen2.5-Math). The paper supports its claims with a comprehensive set of experiments, including various ablation studies.

### Strengths
- Novel Methodological Contribution: The adaptation of the Fill-in-the-Middle (FIM) paradigm to densify mathematical reasoning chains is a novel and insightful contribution. It shifts the focus from regenerating entire solutions to surgically enhancing existing ones, which is a more targeted and potentially more efficient approach to data augmentation for reasoning tasks.

- Practicality and Scalability: The framework offers a practical and scalable alternative to common data enhancement techniques that rely on expensive API calls to proprietary models or computationally intensive search algorithms. By training a moderately-sized FIM model, the authors demonstrate a path to improve open-source models in a more accessible and cost-effective manner. This is a significant practical contribution to the community.

- Comprehensive Experimental Validation: The paper is underpinned by extensive experiments. The authors demonstrate consistent performance gains across a diverse set of base models (of varying sizes and specializations) and evaluation benchmarks, which supports the generalizability and effectiveness of their method.

### Weaknesses
> Unconvincing Comparison to Prompt-Based Baselines and Its Implications:

The paper's claim of superiority over prompt-based expansion methods (Section 5.4) is undermined by a flawed and uninformative experimental design. The comparison is made between the proposed MathFimer model (a specialized model fine-tuned on a strong math-specific base) and a general-purpose small model (Llama-3.2-3B-Instruct) in a zero-shot setting. This baseline is not representative of what is achievable with current state-of-the-art prompting techniques.

The crucial question this experiment fails to answer is whether the proposed, resource-intensive approach of creating a dataset and training a specialized FIM model is truly necessary. It is highly plausible that a modern, powerful model (e.g., GPT-4, DeepSeek-V3, Qwen3) could, through simple zero-shot prompting, generate step expansions of similar or superior quality.

The experiment does not provide meaningful evidence to rule out this simpler, more direct alternative. Consequently, the justification for the entire MathFimer training pipeline is weakened. A compelling demonstration of the method's value would require comparing it against a strong prompt-based baseline, proving that the specialized FIM training provides benefits beyond what is already achievable with existing powerful models. Without such a comparison, the practical utility of the proposed method remains an open question.


> Unaddressed Risk of Error Propagation and Lack of Verification:

The FIM model can inevitably generate steps that contain factual, logical, or calculational errors. The paper acknowledges this but only employs a similarity filter, which is insufficient for verifying logical correctness. A single erroneous insertion can invalidate an entire reasoning chain, and this risk is compounded during iterative expansion. The lack of a robust verification mechanism is a major concern for the reliability of the generated data and the models trained on it. The paper would be strengthened by a qualitative analysis of failure cases and a discussion of potential mitigation strategies (e.g., using a verifier model).

> Unacknowledged Dependence on High-Quality Seed Data and Implicit Distillation:

The paper argues for its independence from powerful external models, and the ablation study in Table 2 effectively demonstrates that MathFimer's structural enhancement provides benefits orthogonal to standard knowledge distillation. This is a strong point.

However, the paper should still discuss the provenance of the MathFimer model's own capabilities. The model is trained on NuminaMath-CoT, a high-quality dataset whose solutions are likely collected by state-of-the-art models such as GPT-4. This implies that MathFimer's ability to generate high-quality intermediate steps is itself a form of implicit distillation—it learns to master the fine-grained reasoning patterns present in its teacher's data.

This does not invalidate the paper's findings, but it reframes the contribution. The work would be positioned more precisely as a highly efficient framework for "compiling" the detailed reasoning style of a powerful teacher into a reusable FIM model, which can then be used to structurally enhance solutions from various sources. Acknowledging this dependency provides a more complete picture of the method's place within the broader ecosystem of model improvement and highlights the critical importance of the initial seed dataset's quality.


> Ambiguity in the Step Decomposition Process:

The crucial pre-processing step of decomposing continuous solutions into discrete steps is based on heuristic rules (Appendix B) that lack specificity and may not be robust. The granularity of these steps directly influences the FIM training task and the final model's behavior. Different segmentation choices could lead to vastly different outcomes. The paper would benefit from a more detailed explanation of these rules, with examples, and a discussion of the sensitivity of the method to this decomposition process.


> Limited Applicability to Non-Linear Reasoning Paradigms:

The MathFimer framework is fundamentally built on the assumption of a linear, sequential Chain-of-Thought process. However, a new generation of powerful reasoning models (e.g., DeepSeek-R1, Qwen3) often exhibit non-linear, abstract, and "jumpy" reasoning paths that defy simple sequential decomposition. For such models, the concept of "filling in the middle" between two adjacent steps becomes ill-defined. This is a limitation, and the paper should discuss the applicability of its method in the context of these emerging, more complex reasoning architectures.

> Insufficient Analysis of PRM Score Distribution:

While Figure 4 is used to argue for the quality of the inserted steps, it also reveals a subtle but important trend: the proportion of "perfect" solutions (PRM score 0.8-1.0) sometimes decreases after expansion. A more nuanced discussion of this change in the score distribution would provide a more complete and balanced interpretation of the results.

### Questions
See Weakness

### Soundness
3

### Presentation
3

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
This paper introduces MathFimer, a novel and cost-effective method to enhance the mathematical reasoning capabilities of Large Language Models (LLMs). The core insight is that the quality and detail of intermediate reasoning steps (CoT) in training data fundamentally limit model performance. MathFimer addresses this by using a Fill-in-the-Middle (FIM) task to automatically expand and enrich existing CoT steps. By employing a smaller, specialized model for this data expansion, the method avoids the high computational cost associated with using large, powerful external models for data generation. The resulting expanded datasets are used to fine-tune state-of-the-art LLMs, leading to measurable performance improvements across standard mathematical reasoning benchmarks like GSM8K and MATH.

### Strengths
* The use of the Fill-in-the-Middle paradigm for explicitly expanding reasoning steps is a creative and new application.

* The experiments show consistent and significant performance improvements across various base models and diverse benchmarks.

### Weaknesses
* The expansion process is contingent on the correctness and structure of the initial CoT steps. If the original CoT contains logical errors or fundamentally flawed reasoning, the FIM process might only elaborate on the mistake, potentially creating overly confident but incorrect training data.

* While detailed steps are generally beneficial, the FIM approach risks generating reasoning chains that are unnecessarily verbose or contain redundant intermediate steps, which could increase inference latency and potentially confuse the target model.

* The paper would benefit from a more detailed qualitative analysis showing examples of how MathFimer fixes or expands weak steps versus how it handles already correct, concise steps.

### Questions
* How does MathFimer handle expansions that introduce factual errors (e.g., via human verification or auto-checks), and what error rate was observed in generated middles?


* Have you tested MathFimer on non-mathematical reasoning tasks (e.g., logical or commonsense reasoning)?

### Soundness
3

### Presentation
2

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
Mathematical reasoning in LLMs benefits from detailed, structured intermediate steps, but expanding these steps typically requires powerful teacher models or expensive search-based methods (e.g., MCTS). The authors introduce MathFimer, which leverages the FIM paradigm to automatically insert intermediate reasoning steps into existing solution chains without external supervision. 

- They construct a large FIM training dataset by decomposing solutions from NuminaMath-CoT (853K examples) into prefix-suffix pairs, yielding around 2.5M FIM samples.
- They use this to train MathFimer-7B - a 7B-parameter model (based on Qwen2.5-Math-7B) to fill intermediate reasoning steps. This model is then used to expand reasoning datasets such as GSM8K, MATH, MetaMathQA, MathInstruct-CoT, and ScaleQuestMath.
- Given a question, prefix, and suffix of an existing solution, the model learns to generate the missing intermediate steps, effectively densifying the reasoning chain.
- Models fine-tuned on MathFimer-expanded datasets outperform those trained on original data across multiple benchmarks (GSM8K, MATH, Math Odyssey, OlympiadBench-EN)
- MathFimer achieves reasoning improvement without reliance on larger teacher models or computationally expensive inference (e.g., MCTS)

### Strengths
- The paper introduces a creative extension of the FIM objective, that was previously used in code completion, to mathematical reasoning. This adaptation is conceptually elegant and non-trivial because reasoning chains differ structurally from code.
- The idea of training a model for FIM - MathFimer - for inserting plausible intermediate steps into existing verified solutions is also elegant and computationally efficient.
- The experiments are extensive, covering multiple datasets (GSM8K, MATH, MetaMathQA, MathInstruct-CoT, ScaleQuestMath) and several base models (LLaMA-3.1-8B/70B, Qwen2.5-Math-7B/72B). The consistent improvements (typically +3–8%) across models lend credibility.
- The paper includes ablation studies disentangling FIM effects from distillation, exploring iteration effects, model scale, and prompt-based baselines. These analyses strengthen the attribution of gains to the proposed method. 
- The paper reports detailed data construction, tokenization, thresholds, and training settings, enhancing reproducibility.
- Overall, the paper is clearly written and and offers a lucid read.

### Weaknesses
- While the paper demonstrates strong results in mathematical reasoning, the scope is narrowly confined to math. That FIM-based reasoning expansion could be a general mechanism for improving structured reasoning is not empirically validated beyond this domain.
- In table 1, there is drop in performance with MathFimer in some cases - this is not discussed.
- The evaluation relies heavily on LLM-as-a-judge for correctness and PRM (process reward models) for reasoning quality. While these are reasonable proxies, substantiating with some human evaluation will strengthen the work.
- The paper briefly acknowledges error accumulation in iterative expansion but doesn’t provide detailed evidence. Multi-round expansion might insert steps that subtly shift semantics or alter problem logic
- There is some repetition and redundancy in writing that could be avoided. For instance, the description of similarity calculation on page 5.
- "fill-in-the-middle" is introduced on page 2 without citation

### Questions
Covered above in the weaknesses

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
The authors propose MathFimer, a novel framework that enhances mathematical reasoning in large language models (LLMs) by expanding intermediate reasoning steps using a Fill-in-the-Middle (FIM) approach. Inspired by code completion tasks, MathFimer trains a model to insert missing or more detailed steps into existing chain-of-thought (CoT) solutions. The method is applied to various math datasets (e.g., GSM8K, MATH), and the resulting expanded data is used to fine-tune both general-purpose and math-specialized LLMs, leading to consistent performance improvements.

### Strengths
1. The use of FIM for step expansion in mathematical reasoning is innovative and offers a scalable alternative to expensive methods like MCTS or distillation from larger models.
2. Extensive experiments across multiple datasets and model sizes show consistent improvements (e.g., +7.43% on GSM8K, +8.86% on MATH), demonstrating the method’s effectiveness.
3. The approach works well even with smaller models (e.g., 1.5B), and supports iterative expansion, making it computationally efficient and widely applicable.

### Weaknesses
1. The method is only evaluated on mathematical reasoning. Its applicability to other domains (e.g., logic, code, commonsense reasoning) remains unclear.
2. The expanded steps are generated by a model and not verified for correctness or logical consistency, which may introduce errors, especially after multiple iterations.
3. The effectiveness of MathFimer heavily relies on the quality of the initial CoT data. Poor-quality base solutions could limit or mislead the expansion process.

### Questions
1. Could you incorporate a verification mechanism (e.g., a verifier or symbolic solver) to validate the correctness of the expanded reasoning steps?
2. Have you tested MathFimer on non-mathematical reasoning tasks? What are the challenges or adaptations needed to apply FIM-based expansion to other structured reasoning domains like code or legal reasoning?
3. What is the risk of over-expansion? Could inserting too many intermediate steps lead to redundancy or confusion? Have you analyzed the trade-off between step granularity and reasoning clarity?

### Soundness
3

### Presentation
3

### Contribution
3
