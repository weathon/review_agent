# SIPDO: Closed-Loop Prompt Optimization via Synthetic Data Feedback

- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Prompt quality plays a critical role in the performance of large language models (LLMs), motivating a growing body of work on prompt optimization. Most existing methods optimize prompts over a fixed dataset, assuming static input distributions and offering limited support for iterative improvement. We introduce SIPDO (Self-Improving Prompts through Data-Augmented Optimization), a closed-loop framework for prompt learning that integrates synthetic data generation into the optimization process. SIPDO couples a synthetic data generator with a prompt optimizer, where the generator produces new examples that reveal current prompt weaknesses and the optimizer incrementally refines the prompt in response. This feedback-driven loop enables systematic improvement of prompt performance without assuming access to external supervision or new tasks. Experiments across question answering and reasoning benchmarks show that SIPDO outperforms standard prompt tuning methods, highlighting the value of integrating data synthesis into prompt learning workflows.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors deal with prompt optimization problem for the LLMs, and propose SIPDO, a closed-loop framework that integrates synthetic data generation into prompt optimization. SIPDO consists of two key components: a Data Generator that produces progressively harder synthetic samples to expose prompt weaknesses, and an Auto Prompt Optimizer that refines prompts via error analysis, recommendation, and targeted revision. The framework operates without external supervision, leveraging a feedback loop to iteratively enhance prompt robustness. Experiments on reasoning benchmarks (BIG-Bench, FOLIO, PrOntoQA) and MMLU show that SIPDO outperforms SOTA baselines (APE, PromptAgent, etc.) with multiple LLMs (GPT-4o, Gemini-1.5, etc.).

### Strengths
**Originality**
This work demonstrates originality in three folds:  (1) this work integrates synthetic data generation into prompt optimization; (2) the authors introduce to construct synthetic examples with progressive difficulty design to dynamically guide the prompt refinement; (3) The authors provide a theoretical guarantee for the proposed framework to assure the prompt error bounds.   

**Quality & Clarity**
The framework is well-structured with self-contained contents. Extensive experiments on a wide range of benchmarks (reasoning, expert knowledge, etc. ) and over 4 LLMs illustrate the effectiveness of the method. Qualitative examples and detailed prompts in appendices provide strong support for this work. 

**Significance**
This work addresses a critical point for LLM application --  unreliable prompt performance in dynamic scenarios. The introduced method has the potential to enable robust adaptation of prompts to diverse domains without human annotations. Also, this work advocates to optimize prompt by utilizing synthetic data rather than rule-based datasets in a much more practical manner.

### Weaknesses
1. The authors claim no external supervision is used in this method, but the true data including question and answer fed into the "Data Generator" module actually provide the supervision signals for the loop.

2. No quantitative demonstration on the fluctuation of LLMs' performance when the input distribution changes as mentioned in L48?

3. How to define the difficulty level L mentioned in Line 172? No investigation on the influence of this number to the overall performance.  

4. The "recommendation" step introduced in Section 3.2 might incorporate the answer of the incorrect cases into the revising recommendation, which can be regarded as a kind of information leakage?

5. Is there any context length control mechanism in the auto prompt optimizer? Since multiple patch $\mathcal{\Delta}^{(t)}$ could be applied to perform prompt revision, and the context could be extended and distracting details could also be dropped dynamically. Longer prompts increase inference latency and may exceed LLM context windows. No analysis on the context length change during the  prompt optimization. 

6. Duplicated equation indexes in Line 161 and Line 188. Also no equation indexing in Line 194.

### Questions
Several questions and doubts are raised in "Weaknesses" part. I would be willing to change my recommendation according to the authors' response.

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
3

### Summary
This paper introduces SIPDO (Self-Improving Prompts through Data-Augmented Optimization), a closed-loop framework for prompt optimization that links a synthetic data generator with a prompt optimizer. The system iteratively generates challenging synthetic examples tailored to a prompt’s current weaknesses; prompt updates are then performed in response to observed failures. The framework integrates dynamic difficulty adjustment and uses synthetic data as a feedback signal, moving beyond static prompt optimization. Across multiple reasoning benchmarks (e.g., MMLU, BIG-Bench, ProofWriter), SIPDO is shown to outperform existing prompt optimization approaches, demonstrating strong generalization and robustness improvements.

### Strengths
SIPDO introduces a closed-loop optimization framework that couples a synthetic data generator with a prompt optimizer through a dual-agent collaboration mechanism. The generator dynamically produces challenging samples targeting the current prompt’s weaknesses, while a progressive difficulty parameter ccc enables a curriculum learning strategy from simple to complex tasks. Ablation results demonstrate that this difficulty-gradient method improves average performance by 17.3%–24.3% compared to one-shot extreme sampling.

### Weaknesses
1. Although Table 2 provides a comprehensive overview of prompt optimization baselines, the current comparisons mainly cover works from 2022–2024 and lack the inclusion of more recent 2025 methods. In particular, direct comparisons with the latest closed-loop or iterative prompt optimization approaches are missing. Most existing baselines used in this paper focus on heuristic or search-based prompt engineering rather than a fully integrated feedback loop.
2. While the related work section (pp. 2–3) is thorough, it does not sufficiently engage with progress in EM-like optimization procedures or Bayesian optimization–based feedback mechanisms. This omission weakens the paper’s positioning and makes SIPDO’s core feedback-loop concept appear more novel than it actually is, as similar closed-loop designs have recently emerged.
3. Section 3.1 describes sampling from a synthetic generator regularized via KL divergence to mitigate label imbalance and mode collapse. However, key implementation details—such as how the generator is parameterized, instantiated, and updated, especially for more challenging tasks—remain underexplored and would benefit from further clarification.

### Questions
see weeknesses.

### Soundness
3

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
This paper proposes SIPDO (Self-Improving Prompts through Data-Augmented Optimization), a closed-loop prompt optimization framework that directly integrates synthetic data generation into the prompt optimization process. The system consists of two synergistic components: a Synthetic Data Generator that deliberately constructs samples with progressive difficulty to expose weaknesses in the current prompt, and an Auto Prompt Optimizer that iteratively refines the prompt through error analysis, recommendation generation, and targeted refinement. This “generate–test–repair–verify” feedback loop enables prompts to self-evolve without requiring external supervision or newly annotated data.

### Strengths
- Integrating synthetic data generation with prompt optimization into a dynamic closed-loop framework goes beyond traditional optimization methods that operate on static datasets.
- Difficulty is monotonically increased through a difficulty parameter and curriculum learning, aligning with principles of human learning.
- The ablation studies are well-designed and effectively validate the contributions of the core components.

### Weaknesses
- Each optimization iteration requires multiple LLM invocations, resulting in significantly higher computational cost compared to conventional approaches. Although the three-expert verification mechanism improves data quality, it substantially increases latency.
- Generation quality is directly constrained by the capabilities of the underlying LLM; for instance, GPT-4o-mini performs markedly worse than GPT-4o. The paper does not explore strategies to reduce reliance on powerful base models, limiting applicability in resource-constrained settings.
- Task-specific safeguards (e.g., for geometric SVG generation) demand substantial domain expertise, and the paper lacks a systematic methodology for transferring the framework to new domains, thereby restricting its practical applicability.

### Questions
- How does the computational overhead of the framework scale as task complexity increases? The authors should supplement their experiments with an end-to-end analysis that explicitly illustrates the trade-off between optimization time and performance gains.
- For tasks requiring specialized domain knowledge (e.g., medical diagnosis), how can the framework ensure the factual and professional accuracy of generated synthetic samples? Could a general-purpose domain adaptation module be designed to reduce reliance on manually crafted, task-specific safeguards (e.g., the three safeguards for geometric SVG generation)?
- At which stages would human expert intervention be most effective—synthetic data generation, error analysis, or prompt editing—and how might such human-in-the-loop feedback be integrated efficiently?
- As the input distribution continuously evolves over time, how can SIPDO be extended to support continual prompt optimization while mitigating catastrophic forgetting of previously acquired knowledge?
- When encountering a new task category, must the optimization process be restarted from scratch? Is there a mechanism to retain and transfer the general reasoning capabilities already learned by the prompt across tasks?

### Soundness
3

### Presentation
2

### Contribution
2
