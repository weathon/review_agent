# PRL: Prompts from Reinforcement Learning

- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Effective prompt engineering remains a central challenge in fully harnessing the
capabilities of LLMs. While well-designed prompts can dramatically enhance
performance, crafting them typically demands expert intuition and a nuanced understanding of the task. Moreover, the most impactful prompts often hinge on
subtle semantic cues, ones that may elude human perception but are crucial for
guiding LLM behavior. In this paper, we introduce PRL (Prompts from Reinforcement Learning), a novel RL-based approach for automatic prompt generation.
Unlike previous methods, PRL can produce novel few-shot examples that were not
seen during training. Our approach achieves state-of-the-art performance across a
range of benchmarks, including text classification, simplification, summarization,
and reasoning. On the classification task, it surpasses prior methods by 2.58%
over APE and 1.00% over EvoPrompt. Additionally, it improves the average
ROUGE scores on the summarization task by 4.32 over APE and by 2.12 over
EvoPrompt and the SARI score on simplification by 6.93 over APE and by 6.01
over EvoPrompt. On the GSM8K mathematical reasoning benchmark, PRL further
improves accuracy by 2.72% over APE and by 4.53% over EvoPrompt. We will
make our implementation publicly available upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents PRL (Prompts from Reinforcement Learning), a method for automatic prompt generation using reinforcement learning. The approach formulates prompt construction as a learning problem, enabling the model to generate few-shot examples that were not seen during training. The authors evaluate PRL on several tasks, including text classification, text simplification, summarization, and the GSM8K reasoning benchmark. Experimental results show consistent improvements over prior prompt optimization methods such as APE and EvoPrompt, with reported gains in accuracy, ROUGE, SARI, and reasoning performance. The paper claims that PRL offers a general and effective framework for enhancing LLM performance through learned prompts

### Strengths
1.	The method is clear and intuitive. The paper is well-written and easy to follow.

2.	Overall, the experimental results effectively support the authors’ claims.

### Weaknesses
1.	Compared to other approaches discussed in the paper, this method appears to be more computationally intensive.

2.	The core of the method lies in the prompt generator, but the authors only use Qwen2-7B as the base model. It would be more convincing to evaluate PRL on models of different sizes and from different model families.

3.	The generalization ability of PRL is not thoroughly studied. As different models exhibit varying levels of sensitivity and preference toward prompts, PRL may need to train separate prompt generators for different evaluation models. More tests about this could be further included.

4.	Additional ablation studies could strengthen the paper. For instance, the influence of the “thinking process” in the prompt generator is not clearly analyzed. How much would the results change if this component were removed? If the performance remains similar, the additional computation might be unnecessary.

### Questions
1.	The format of the citation at line 053 seems incorrect.

2.	The prompt generator in PRL requires a base prompt. How sensitive is the method to the quality of this base prompt? Would iterative refinement improve performance? Additionally, what would happen if prompts were generated entirely from scratch?

### Soundness
3

### Presentation
3

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
Effective prompt engineering remains a **central challenge** in fully harnessing the capabilities of Large Language Models (LLMs). While precise input prompts can guide LLMs to perform complex tasks, the most impactful prompts often rely on *subtle semantic cues* that may *elude human perception*.

To address this, the paper introduces **PRL (Prompts from Reinforcement Learning)**, a novel RL-based approach for automatic prompt generation.

**PRL's key innovations include:**

1.  **Novel Few-shot Example Synthesis:** Unlike previous methods like Automatic Prompt Optimization (APO), which are restricted to selecting few-shot examples from training data, PRL is capable of generating and selecting **novel few-shot examples** that were *not seen during training*.
2.  **Explicit Reasoning Integration:** PRL incorporates a *reasoning phase* prior to prompt generation, where the prompt generator first produces a **rationale** (enclosed within `<think>` tags) to guide the final output.
3.  **RL Optimization Cycle:** PRL trains a **Prompt Generator ($\pi_{\text{generator}}$)** (a trainable language model) to refine base prompts and generate a corresponding reasoning trace. This generated prompt is evaluated by a **frozen Evaluation Model ($\pi_{\text{eval}}$)** (an LLM used only for inference), which calculates rewards based on formatting and task performance. The generator is optimized using the *Group Relative Policy Optimization (GRPO)* update rule.
4.  **Prompt Selection Strategy:** The method uses a prompt selection strategy to mitigate training instability and noisy feedback, regularly testing generated prompts on the validation set and keeping the best overall one.

**Main Contributions and Results:**

PRL achieves **state-of-the-art performance** across a range of benchmarks: text classification, summarization, simplification, and GSM8K mathematical reasoning.

*   On the **classification task**, PRL surpasses APE by **2.58%** and EvoPrompt by **1.00%** in mean accuracy.
*   On **summarization**, it improves average ROUGE scores by **4.32** over APE and **2.12** over EvoPrompt.
*   On **simplification**, it improves the SARI score by **6.93** over APE and **6.01** over EvoPrompt.
*   On the **GSM8K mathematical reasoning** benchmark, PRL improves accuracy by **2.72%** over APE and **4.53%** over EvoPrompt.
*   The research further suggests that **RL-based optimization naturally leads to the emergence of few-shot prompting behavior**.

### Strengths
**Originality:**

*   PRL is highlighted as the **first RL-based prompt optimization method** capable of *generating and selecting novel, task-specific few-shot examples*. This is a crucial distinction, as it moves beyond the constraint of only using few-shot examples already present in the training data, a limitation faced by methods like APO.
*   The observation that few-shot examples emerge **spontaneously** during the RL training process, without explicit encouragement, is a unique and insightful finding regarding how RL shapes LLM prompting behavior.

**Quality & Significance:**

*   The method demonstrates **superior performance** consistently across all four evaluated task types (classification, summarization, simplification, and reasoning), validating its robust generalizability and effectiveness.
*   **Ablation studies confirm the value of core components:** The explicit reasoning phase proved critical, leading to a *substantial drop in accuracy* (from 75.05 to 60.12) on the SUBJ dataset when omitted. Furthermore, the Prompt Selection strategy was shown to *improve final performance and enhance training efficiency*, helping to manage the high variance inherent in RL training.
*   PRL is shown to be effective even when applied to **larger, more powerful LLMs** (e.g., Qwen2-32B-instruct), demonstrating that even these models remain *vulnerable to prompt variation* and can benefit significantly from PRL's tailored prompts.

**Clarity:**

*   The architecture, including the roles of the Prompt Generator and the frozen Evaluation Model, and the overall RL training scheme (Figure 2) are clearly described.
*   The reward function is systematically broken down into components: **formatting rewards** ($r_{token}$, $r_{structure}$) for the Generator, and **task performance rewards** ($r_{format}$, $r_{alignment}$) for the Evaluation Model, providing clarity on how the model’s behavior is guided.

### Weaknesses
1.  **Significant Computational Overhead:** The paper explicitly lists as a limitation that the improved performance is obtained at the cost of a **significantly greater computational expense** than related, comparatively simpler work. The experimental setup involved training over 48 hours using two NVIDIA A100 GPUs (40 GB each). The paper lacks concrete discussion or suggested methods for mitigating or quantifying this increased computational burden, which limits its practical accessibility.
2.  **Task-Specific Retraining Requirement:** Currently, the prompt generator must be **retrained for each new task**. The authors acknowledge that developing a *universal prompt generator* is a "desideratum" (an ideal goal), indicating that the current method’s efficiency is limited when facing a wide variety of tasks or zero-shot scenarios.
3.  **Insufficient Detail on Few-shot Synthesis:** PRL's ability to **autonomously synthesize relevant few-shot examples** not present in the training set is a major contribution. However, the paper does not delve into *how* the prompt generator, guided by RL, manages to *create* these task-aligned, non-redundant examples—specifically, the internal reasoning or constraints utilized by $\pi_{\text{generator}}$ in its thought process ($\langle \text{think} \rangle$ tag) to achieve this synthesis.
4.  **Sensitivity of Baselines to Evaluation Model Choice:** The authors note that when reproducing EvoPrompt results, the relative effectiveness of its DE and GA variants was **sensitive to the choice of the underlying language model** (Qwen2.5-7B-Instruct). Although the authors ensured a fair comparison by using the same Evaluation Model across all baselines, this inherent sensitivity suggests that the observed superiority of PRL might be conditional on the chosen model, necessitating more explicit acknowledgement of this limitation in interpreting the main results.

### Questions
1.  **Computational Efficiency and Trade-offs:** Given that PRL requires a *significantly greater computational expense*, could the authors provide a more detailed analysis of the performance gains versus the resource cost? Are there specific tuning levers (e.g., Prompt Selection frequency $t$, or the number of sampled prompts $n$) that could be adjusted to reduce training time substantially while maintaining competitive performance against baselines?
2.  **Mechanics of Task-Dependent Few-Shot Emergence:** Few-shot examples were critical for classification tasks (improving accuracy significantly, e.g., SUBJ from 66.75 to 77.95), but PRL **consistently opted *not* to include few-shot examples** for the summarization task. What drives this striking, task-dependent behavior? Which specific components of the comprehensive reward function $R$ (e.g., $r_{alignment}$ or $r_{structure}$) are responsible for prompting $\pi_{\text{generator}}$ to spontaneously generate few-shot examples for classification but omit them for generation tasks?
3.  **Path to a Universal Prompt Generator:** The paper identifies the need to *retrain the generator for each new task* as a limitation, noting that a *universal prompt generator* is a "desideratum". What are the initial conceptual steps or future research directions the authors are considering to enable the Prompt Generator to transfer or generalize prompting knowledge across different tasks without requiring full retraining?
4.  **Baseline Robustness Across Evaluation Models:** All SOTA claims are established by evaluating baselines on the Qwen2.5-7B-Instruct model. Given the noted sensitivity of EvoPrompt variants to the base model choice, how would a *complete* set of benchmark results (including APE, EvoPrompt, and APO where applicable) compare if a distinct architecture, such as LLaMA 3.1-8B-Instruct, was used as the Evaluation Model for *all* methods, similar to the setup used for the portability study?

### Soundness
3

### Presentation
3

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
The paper introduces PRL, a RL framework that trains a prompt-generation model to automatically produce effective task prompts. It can be viewed as another R1-style (GRPO RL training) approach applied to the domain of automatic prompt generation.

### Strengths
1. Framing prompt optimization as a reinforcement learning problem with a frozen evaluator is clear and straightforward.
2. The method demonstrates consistent empirical improvements across multiple benchmark tasks.

### Weaknesses
1. The paper motivates prompt optimization as an underexplored and crucial problem, but recent work (e.g., instruction tuning, preference alignment, RLHF) has substantially reduced the marginal importance of prompt optimization for strong LLMs. The paper would benefit from a deeper discussion or quantitative evidence that prompt optimization still provides meaningful gains for modern aligned models. 
2. Most baseline methods cited for automatic prompt generation date back to 2022–2023. It would strengthen the paper to incorporate or discuss more recent developments in this area.
3. The proposed method requires RL training for each dataset, which introduces significant computational overhead compared to evolutionary or heuristic prompt optimizers. The paper lacks a systematic analysis of the resulting latency and compute cost versus accuracy trade-off, making it difficult to assess practical efficiency.
4. The motivation for training a separate prompt generator model is somewhat unconvincing. The paper does not compare against stronger or larger generator models (e.g., Qwen2.5-72B, GPT) or expert-crafted prompts, which could potentially achieve comparable results without RL training.
5. Since the generator must be retrained for every dataset, the method’s scalability and generality are limited. The paper does not explore whether a single generator can generalize across multiple datasets or related domains.
6. The chosen benchmarks (e.g., GSM8K, SAMSum, SST, AGNews) are somewhat outdated and relatively simple. Evaluating on more challenging or recent datasets would better demonstrate PRL’s robustness and contemporary relevance.
7. The ablation on model size lacks a clear causal interpretation. Performance improvements when scaling from 7B to 32B evaluators could largely stem from the inherent capability gain of the larger models rather than from PRL’s prompt optimization.

### Questions
Could you include detailed statistics for the training, validation, and test datasets used in your experiments?

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
This paper introduces PRL (Prompts from Reinforcement Learning), a reinforcement learning-based
approach to automatically generating and optimizing prompts for large language models (LLMs). PRL uniquely
enables the synthesis of novel few-shot examples not seen during training and integrates explicit reasoning
steps before prompt output. This method achieves excellent empirical performance in tasks such as text
classification. The research scheme includes carefully designed reward shaping, prompt selection, and
detailed ablation experiments.

### Strengths
1. PRL devises a clear RL-based prompt optimization loop. Compared to other methods, PRL can create few-
shot prompt examples not limited to the original training data

2. PRL is evaluated across varied tasks, and multiple ablation studies dissect the contribution of prompt
selection, few-shot examples, and explicit reasoning. Additionally, its effectiveness is verified on models of
different architectures and sizes.

### Weaknesses
1. Compared to other methods, PRL requires more computational resources for training and has insufficient
generalization ability, which means that PRL needs to be retrained for different tasks, greatly limiting its
usability.

2. There is insufficient discussion on the instability, scalability, and generalization of reinforcement learning:
- Can training on a single task generalize to other tasks? Furthermore, how does the performance and
generalization of simultaneous multi-task learning compare to that of single-task learning?
- Is model training sensitive to the introduced reward function, does reward manipulation exist, and
what is the interaction between format specification rewards and task correctness rewards?
- Is it effective to use a different generator model for training than the one used to evaluation model?
Furthermore, can the same generator model be directly used for different evaluation models after
training?

3. One of the paper's cores is that few-shot prompting behavior "spontaneously emerges" from the RL setup,
yet there is little to no formal analysis or justification. A more rigorous explanation (e.g., does the reward
landscape incentivize synthesis, or is it an artifact of the prompt generator's architecture?) is sorely missing.

4. The compared methods are limited to those before 2023, lacking comparison results with new methods
from 2024-2025.

### Questions
see weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
