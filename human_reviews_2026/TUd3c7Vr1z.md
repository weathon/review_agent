# Towards Self-Robust LLMs: Intrinsic Prompt Noise Resistance via CoIPO

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Large language models (LLMs) have demonstrated remarkable and steadily improving performance across a wide range of tasks.
However, LLM performance may be highly sensitive to prompt variations especially in scenarios with limited openness or strict output formatting requirements, indicating insufficient robustness.
In real-world applications, user prompts provided to LLMs often contain imperfections, which may undermine the quality of the model's responses.
To address this issue, previous work has primarily focused on preprocessing prompts, employing external tools or even LLMs to refine prompt formulations in advance.
However, these approaches overlook the intrinsic robustness of LLMs, and their reliance on external components introduces additional computational overhead and uncertainty.
In this work, we propose a Contrastive Learning-based Inverse Direct Preference Optimization (CoIPO) method that minimizes the discrepancy between the label-aligned logits produced by the model under a clean prompt and its noisy counterpart, and conduct a detailed analysis using mutual information theory.
We augment the FLAN dataset by constructing paired prompts, each consisting of a clean prompt and its corresponding noisy version for training.
Additionally, to evaluate the effectiveness, we develop NoisyPromptBench, a benchmark enhanced and derived from the existing PromptBench. 
Experimental results conducted on NoisyPromptBench demonstrate that our proposed method achieves a significant improvement in average accuracy over the current state-of-the-art approaches.
The source code of CoIPO, pair-wise FLAN datasets, and NoisyPromptBench have already been released on https://github.com/vegetable-yx/CoIPO.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the problem of prompt robustness for large language models and introduces CoIPO (Contrastive Learning-based Inverse Direct Preference Optimization), a post-training method that enhances intrinsic prompt noise resistance without dependence on external preprocessors. CoIPO is trained on paired clean and noisy prompts from an augmented FLAN dataset, with a loss function that uses contrastive learning and inverse DPO to minimize the discrepancy between outputs for clean and semantically similar noisy prompts, while maximizing it for semantically different pairs.

### Strengths
1. This paper proposes a principled intrinsic robustness enhancement method (CoIPO) for LLMs based on contrastive learning and inverse DPO. The formulation is clear and easy to understand.
2. The authors offer an information-theoretic perspective (mutual information) to justify the approach, which is interesting.

### Weaknesses
1. The research problem of prompt optimization is important but the research scope of this work is somehow limited, since the noisy prompts in this paper primarily refer to typos (character level, word level etc.), while real-world prompt imperfections can be more varied, such as semantic ambiguity, non-standard grammar, than those benchmarked here.
2. More experiment baselines should be incorporated. For example, DPO should be considered as a baseline, since CoIPO uses both contrastive training and inverse-DPO and contrastive training baseline COIN is used, DPO should also be included as a baseline.
3. Most experiments are conducted on tasks with deterministic answers —prompts with more open-ended or generative outputs should be considered.

### Questions
1. How does CoIPO handle more nuanced and complex prompt noise types? Can it generalize to unseen perturbation types?
2. Could minimizing output discrepancies between noisy and clean prompts reduce the model's flexibility? Does CoIPO risk making the model less sensitive to subtle but meaningful variations?
3. It is not mentioned in the paper how the prompt for another task, i.e. P_2 is selected. Is it selected randomly or according to some heuristics? How does the selection method of P_2 affect the performance?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors proposed CoIPO, an novel algorithm that aims to train robust LLMs under imperfect user inputs without pre-processing. CoIPO combines contrastive learning and preference optimization by reducing the gap between clean prompt and noisy prompt while enlarging the gap between the noisy prompt and a unrelated clean prompt. Theoretical analysis on relative entropy gain is presented and experiments are conducted on Llama and Qwen-2.5 architectures. CoIPO significantly improves accuracy for both clean and noisy scenarios on PromptBench, surpassing the baselines.

### Strengths
* CoIPO tackles a significant issue, handle imperfect user input in daily use of the LLMs.
* The CoIPO algorithm leverages contrastive learning and direct alignment, backed with theoretical insights from the perspective and relative entropy gap maximization.
* Comprehensive experiments are conducted for different architectures.

### Weaknesses
* Though CoIPO alone is a novel and effective algorithm for increasing LLM robustness. My concern is that if this procedure will hurt the models performance on tasks like math reasoning and coding. Since it would appear costly to me if we sacrifice these reasoning capabilities to replace pre-processing tools for imperfect input.
* No algorithm specific hyper-parameter is presented in the paper, could the author elaborate a bit more on the details hyper-parameter if they are included in the algorithm?
* Preference optimization is relatively light weight but still requires time and compute, could the authors elaborate on the time cost for running CoIPO compared to pre-proccessing?

### Questions
Please see weakness section.

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
This paper proposes CoIPO, a post-training method to enhance LLMs' intrinsic robustness against prompt perturbations. Unlike existing approaches that rely on external preprocessing tools, CoIPO trains models to directly handle noisy prompts by minimizing discrepancies in label-aligned logits between clean and noisy prompts. The authors construct a paired FLAN dataset and develop NoisyPromptBench for evaluation. Experiments on Llama-7B and Qwen2.5-7B across five datasets demonstrate improvements over baselines, with theoretical justification provided through mutual information analysis.

### Strengths
Well-motivated problem: The paper clearly articulates limitations of external preprocessing approaches and makes a strong case for intrinsic robustness.
Solid theoretical foundation: The mutual information analysis (Equations 9-16) provides principled justification for the method.
Comprehensive evaluation: NoisyPromptBench with multiple perturbation types and the decoding radius analysis (Section 4.1, Figure 5) provide thorough robustness assessment.
Ablation studies: Table 3 effectively demonstrates the necessity of both inverse DPO and contrastive learning components.

### Weaknesses
Limited scope of evaluation:

● Only 7B parameter models tested; unclear if findings generalize to larger models (13B, 70B+)

● Only 5 datasets from GLUE-style tasks; robustness on generation tasks, reasoning, or code generation is unexplored

● Training data limited to 25 FLAN subsets; impact of training data scale not studied

Insufficient baseline comparisons:

● Only compares to COIN for intrinsic robustness methods

● Missing comparisons to recent prompt optimization methods (e.g., PromptAgent, RoP mentioned in related work)

● No comparison to instruction-tuning methods that may implicitly improve robustness

Theoretical gaps:

● The connection between Equation 15 and Equation 8 relies on several approximations that may not hold in practice

Missing analyses:

● No error analysis showing which types of errors CoIPO successfully handles vs. fails on

● Computational cost comparison not provided (training time, memory, inference latency)

### Questions
Scalability: Have you tested CoIPO on larger models (13B+)? Do the improvements scale, or do larger models already have better intrinsic robustness?

Training efficiency: What is the computational overhead of CoIPO compared to standard fine-tuning? How many training epochs are needed for convergence?

Comparison to instruction tuning: How does CoIPO compare to simply training on more diverse instruction data? Could increased data diversity achieve similar robustness?

Generation tasks: All experiments focus on classification. How does CoIPO perform on open-ended generation where output quality is harder to measure?

### Soundness
3

### Presentation
3

### Contribution
3
