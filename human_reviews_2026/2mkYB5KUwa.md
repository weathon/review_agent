# Rethinking reasoning with Masked Diffusion Models

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Masked diffusion language models (MDLMs) are trained to in-fill positions in randomly masked sequences, in contrast to traditional next-token prediction (NTP) models. Discussions around MDLMs focus on two benefits: (1) any-order decoding and 2) multi-token decoding. However, we observe that for math and coding tasks, any-order algorithms often underperform or behave similarly to left-to-right sampling, and standard multi-token decoding significantly degrades performance. At inference time, MDLMs compute the conditional distribution of all masked positions. A natural question is: How can we justify this additional compute when left-to-right one-token-at-a-time decoding is on par with any-order decoding algorithms? These findings warrant rethinking how MDLMs are utilized. First, we propose reasoning-as-infilling. By using MDLMs to infill a reasoning template, we can structure outputs and distinguish between reasoning and answer tokens. In turn, this enables measuring answer uncertainty during reasoning, and early exits when the model converges on an answer. Next, given an answer, reasoning-as-infilling enables sampling from the MDLM posterior over reasoning traces conditioned on the answer, providing a new source of high-quality data for post-training. On GSM8k, we observe that fine-tuning LLaDA-8B Base on its posterior reasoning traces provides a performance boost on par with fine-tuning on human-written reasoning traces. Additionally, given an answer, reasoning-as-infilling provides a method for scoring the correctness of the reasoning process at intermediate steps, without requiring expensive rollouts or an external model. Second, we propose multi-token entropy decoding (MED), a simple adaptive sampler that minimizes the error incurred by decoding positions in parallel based on the conditional entropies of those positions. MED preserves performance across benchmarks and leads to 2.7× fewer steps. Combined with early exits, MED leads to a 3.3× speed-up on GSM8k with a minimal (0.1%) effect on accuracy. Our work demonstrates that the training objective and compute used by MDLMs unlock many new possibilities for inference and post-training methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the parallel decoding behavior of MDLMs, revealing that even decoding two tokens per step leads to substantial performance degradation. To address this, the paper proposes an adaptive multi-token decoding algorithm, MED, which selectively decodes positions with low conditional entropy, thereby enabling controlled parallelism. Compared to entropy decoding, MED reduces the NFEs while maintaining model performance. Furthermore, the paper introduces reasoning-as-infilling, a framework that structures generation using a template with explicit reasoning and answer blocks. This approach enables the monitoring of answer uncertainty for early exits and facilitates the sampling of posterior reasoning traces given a known answer, which can be used for data generation and post-training.

### Strengths
1. The paper is well-motivated by the analysis of MDLM decoding strategies. The authors first examine both any-order and parallel decoding, revealing that left-to-right sampling remains a strong baseline and that naive parallelization substantially degrades performance. These findings provide a clear and empirical justification for developing the proposed adaptive decoding method.

2. The paper introduces the reasoning-as-infilling framework, a novel approach that structures generation by pre-filling a template with distinct reasoning and answer blocks. A key application of this framework is its ability to enable the sampling of posterior reasoning traces, which provides a more direct mechanism for data generation and model analysis compared to the conventional autoregressive models.

### Weaknesses
1. The paper lacks a direct empirical comparison with other acceleration techniques for MDLMs [1, 2], making it difficult to assess the competitiveness of the proposed methods. For instance, the reported 69% speed-up on GSM8K appears relatively modest.

2. Incomplete Empirical Evaluation. The paper's conclusions would be substantially strengthened by a more systematic evaluation. It is suggested to include results for MED on MATH and MBPP [3], and for the early-exit strategy on HumanEval and MBPP, to fully demonstrate the general applicability of the proposed methods.

3. The discussion of limitations could be more focused. Instead of addressing the general drawbacks of MDLMs, it would be more helpful to examine the scope of applicability and potential failure cases of the paper’s own contributions: the MED algorithm and the reasoning-as-infilling framework. This would clarify the specific strengths and limitations of the proposed approach.

4. The paper's clarity could be enhanced by rebalancing its structure. The analysis in Section 3 is extensive; condensing it would allow for a more thorough discussion of the core methods and experimental details, strengthening the paper's overall focus.

[1] Wu et al., Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding.

[2] Wei et al., Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles.

[3] Austin et al., Program Synthesis with Large Language Models.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper provides two techniques to make use of the parallel decoding and any-order generation attribute of MDLMs, accelerating inference: 1. predict as many tokens as possible if the entropy of their predicted distribution is low enough; 2. Set a predetermined think then answer template, the inference can exit early as soon as the entropy on the answer tokens are low enough.

### Strengths
1. The idea of predetermine a thinking template is reasonable. It bears similarity with <think> template in AR model, but in MDLMs we can control the thinking length.
2. The proposed method cleverly leverages the unique property of MDLMs—access to all masked token distributions—to enable adaptive parallel decoding. The theoretical grounding in KL divergence bounds adds rigor to the approach.
3. The paper provides thorough experiments across multiple benchmarks (GSM8K, Math500, HumanEval), demonstrating consistent speedups (up to 3.3×) with minimal accuracy loss. The post-hoc reasoning analysis is particularly insightful for model improvement.

### Weaknesses
1. The idea of fast decoding in MDLMs are already discussed in many papers. For example, Fast DLLM also decode multiple tokens according to their confidence. However, the comparisons with existing methods are ignored in this paper.
2. The template leads to human-define the length of the reasoning and answer part. I believe that for GSM8K and HumanEval, at least the answer part should be of different length. How did the authors set these hyper-parameters?
3. In table 3, it seems that we did not try accelerating a lot: the most aggressive hyper-parameter we've tried on GSM8K is still more than 0.5x steps, suggesting relatively conservative acceleration. For accelerating method, we would expect greater speedup potential.

### Questions
- How is Post-hoc reasoning evolved in the reasoning process actually? Does the post-hot reasoning lead to a second-time answer, and we actually evaluate with this second-time answer?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper reexamines MDLMs and focuses on two widely claimed advantages: multi-token decoding and any-order decoding. The authors propose Multi-token Entropy Decoding (MED), an adaptive sampler that achieves 2–3× faster inference without accuracy loss, and a Reasoning-as-Infilling framework that enables structured reasoning, early exits, and posterior sampling of reasoning traces. Extensive experiments validate the effectiveness of the proposed methods.

### Strengths
1. Multi-token Entropy Decoding (MED) is theoretically grounded and empirically validated. Its entropy-based criterion provides a principled way to control decoding errors, achieving 2–3× faster inference without accuracy loss.


2. The Reasoning-as-Infilling framework reveals a novel and previously unexplored property of MDLMs: generating correct reasoning traces for 43% of problems originally solved incorrectly.

### Weaknesses
The paper lacks sufficient discussion and comparison with closely related works. Specifically, [1] introduces a confidence-threshold-based sampling approach, and [2] proposes an entropy-bounded unmasking procedure. Both methods are conceptually similar to the proposed Multi-token Entropy Decoding (MED) and also provide theoretical analyses. Unfortunately, the authors do not discuss these works in depth or include empirical comparisons with them.

[1] Wu et al. Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding. arXiv 2025.05.

[2] Ben-Hamu et al. Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking. arXiv 2025.05.

### Questions
1. Could the proposed Early Exit mechanism terminate decoding prematurely—i.e., when the final answer has been generated but the reasoning chain is still incomplete? 

2. The Reasoning-as-Infilling framework shows that MDLMs can generate correct reasoning traces for problems originally solved incorrectly. What insights does this provide for improving MDLM training or architecture design in the future?

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
This paper investigates various inference aspects and their benefits for masked diffusion models. The authors first examine two commonly discussed advantages: any-order sampling and multi-token prediction. They find that any-order sampling offers limited gains, while multi-token prediction degrades performance. To address this, the authors propose a Multi-token Entropy Decoding (MED) strategy, which dynamically determines the number of tokens to decode at each inference step, resulting in an inference-time speedup. They further demonstrate additional efficiency improvements through early exiting based on the model’s uncertainty in its answers.

### Strengths
The proposed method MED with early exist seems to provide good inference efficiency gains on coding and reasoning tasks.

### Weaknesses
- Several claims in the paper are not sufficiently supported by experimental evidence and appear somewhat overstated.
     - For instance, to support the claim that “MDLM reasoning posterior yields high-quality traces,” the authors only provide correctness scores judged by Qwen and GPT. However, correctness alone does not offer a complete view of trace quality. It would also be more informative to include correctness scores for a baseline model, such as LLaMA 3.1 (or another model with comparable capabilities), to better contextualize the benefits of MDLM’s reasoning quality relative to other autoregressive models.
     - The claim that left-to-right sampling performs on par with any-order sampling also seems somewhat overstated and comes with a few caveats. Except for the Dream 7B model on the HumanEval task, any-order sampling with certain block sizes consistently outperforms left-to-right sampling. It is also worth noting that the Dream 7B model was adapted from an autoregressive model, which naturally introduces a left-to-right inductive bias. Additionally, tasks like GSM8K are known to rely on predominantly left-to-right reasoning patterns (see [1]). The true advantage of any-order sampling is more likely to appear in tasks requiring non-linear reasoning (e.g., Sudoku, though admittedly more synthetic in nature).

- It would also be more informative for readers if all comparisons were presented in terms of Number of Function Evaluations (NFEs) rather than the number of parallel tokens decoded (as in Tables 2 and 3). The number of parallel tokens directly reduces NFEs by a proportional factor, which results in a major change, whereas reducing the number of inference steps in diffusion impacts NFEs without such drastic changes. For instance, it is unclear how the entropy decoding method operates with 96 or 112 NFEs in Table 3.

- Finally, the paper should include a more comprehensive comparison with the EB sampler and generate_until sampler from Ben-Hamu et al. I would also like to note that the Ben-Hamu et al. paper appeared on arXiv at the end of May, and according to ICLR’s submission guidelines, papers released after July 1 are considered concurrent work.

[1] Premise Order Matters in Reasoning with Large Language Models. Chen et al. 2024

### Questions
- It would be interesting to understand the performance left-to-right decoding and any order decoding with different block sizes work on R-GSM8k proposed in [1]
- Why is HumanEval on Llada with left-to-right accuracy 15.24 in Table 1 and 11.0% in Table 7? Typo?

### Soundness
2

### Presentation
2

### Contribution
2
