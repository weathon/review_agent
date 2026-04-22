# Early Stopping Chain-of-thoughts in Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Reasoning large language models (LLMs) have demonstrated superior capacities in solving complicated problems by generating long chain-of-thoughts (CoT), but such a lengthy CoT incurs high inference costs. In this study, we introduce ES-CoT, an inference-time method that shortens CoT generation by detecting answer convergence and stopping early with minimal performance loss. At the end of each reasoning step, we prompt the LLM to output its current final answer, denoted as a step answer. We then track the run length of consecutive identical step answers as a measure of answer convergence. Once the run length exhibits a sharp increase and exceeds a minimum threshold, the generation is terminated. We provide both empirical and theoretical support for this heuristic: step answers steadily converge to the final answer, and large run-length jumps reliably mark this convergence. Experiments on five reasoning datasets across three LLMs show that ES-CoT reduces the number of inference tokens by about 41% on average while maintaining accuracy comparable to standard CoT. Further, ES-CoT integrates seamlessly with self-consistency prompting and remains robust across hyperparameter choices, highlighting it as a practical and effective approach for efficient reasoning. Implementation codes of this study are available online (hidden for peer review).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes ES-CoT, an inference-time method that reduces CoT length by detecting answer convergence and stopping generation early. By monitoring repeated step answers, the method terminates once convergence stabilizes, cutting inference tokens. Experiments on multiple datasets and LLMs show ES-secondo me CoT’s efficiency, robustness, and compatibility with self-consistency prompting.

### Strengths
- Overall, the paper is well-written.
- Efficiency in LLMs is an important and valid research challenge.

### Weaknesses
- Several related works on concise reasoning and CoT compression are missing from the discussion and experiments. The authors should compare or at least discuss their approach. Some of them (for example):
- Xu et al. – Chain of Draft: Thinking Faster by Writing Less
- Aytes et al. – Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching
- Zhang et al. – LightThinker: Thinking Step-by-Step Compression
- Fatemi et al. – Concise Reasoning via Reinforcement Learning
- Lee et al. – How Well Do LLMs Compress Their Own Chain-of-Thought? A Token Complexity Approach
- Nayab et al. - Concise Thoughts: Impact of Output Length on LLM Reasoning and Cost

Although the proposed method reduces long CoT generation, it still requires multiple-step answers. The authors should clarify how these results lead to a significant reduction in inference cost.

The mathematical framework appears somewhat isolated. The authors should better explain their technical contribution and how it supports or strengthens the experimental findings.

### Questions
Please address computational cost and theoretical issues indicated in the weaknesses section.

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
This paper introduces ES-CoT, an inference-time method to reduce chain-of-thought (CoT) reasoning length in large language models (LLMs) by stopping generation early when answer convergence is detected. ES-CoT monitors the consecutive run length of identical 'step answers' (the model's current statements of its answer at each reasoning step) and halts generation using a run-jump test—triggered when the run length makes a statistically significant increase, exceeding a minimum threshold. The paper justifies this heuristic empirically and theoretically, demonstrating that step answers typically stabilize before completion. Experiments on five reasoning datasets and three LLMs show ES-CoT reduces token usage by about 41% on average with minimal loss in accuracy, and is robust across hyperparameter settings and compatible.

### Strengths
(1)The paper provides a quantitative and visual analysis of answer convergence in LLM reasoning (see Figure 2 and Figure 3), demonstrating that step answers indeed tend to stabilize toward the end of reasoning trajectories. The convex jump in run length (Figure 3) offers a clear, observable signal for potential early stopping.
(2)The ES-CoT mechanism is straightforward to implement at inference, requiring only minor modifications (adding a 'final answer' prompt per step), and does not require retraining, parallel decoding, or auxiliary models.
(3)The method is evaluated across three competitive LLMs and five well-established reasoning benchmarks, showing substantial savings in average tokens (see Table 1) while maintaining comparable or sometimes even better accuracy.

### Weaknesses
(1):The paper does not provide head-to-head results with Speculative Rejection (Sun et al., 2024) or Early Stop Self-Consistency (ESC, Li et al., 2024), which also propose output-side early stopping methods for CoT. These missing baselines are critical for evaluating the genuine advantage of ES-CoT. Without this, it is unclear if the proposed method achieves better cost-quality tradeoffs or simply offers a simpler alternative.
(2) There is no systematic examination of what types of questions or failure modes lead to suboptimal early stopping (incorrect halts, divergence from the true answer, etc.). A deeper dive into breakdowns by problem type or error case is warranted for practical adoption.
(3) ES-CoT assumes the 'step answer' extraction at each reasoning step can be done reliably. In real-world settings, especially with less structured or conversational prompts, it may be challenging to parse step-wise answers automatically, limiting applicability to domains outside structured problem-solving.

### Questions
(1):How would ES-CoT perform or adapt in tasks where answer uncertainty remains high, or where step answers do not stabilize (violation of Assumption 1)? Can the authors provide breakdowns or error analysis for such cases?
(2):Is the t-test actually an appropriate statistical test on the run difference sequence $D$? Have the authors considered non-parametric or more tailored sequential change-detection approaches?
(3):How robust is the step-answer extraction in less-structured CoT outputs, e.g., open-domain QA or conversation? Can ES-CoT be reasonably applied in those settings?
(4):For the token counting, are the per-step answer prompts included in both ES-CoT and baseline methods? How does ES-CoT’s overhead affect the true token savings?

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
The paper proposes early-stopping CoT (ES-CoT), an inference-time method that aims to reduce reasoning length by producing a final answer at every reasoning step. The authors introduce the concept of run length, which is the number of consecutive steps yielding the same final answer, and use it to detect answer convergence. Both empirical and theoretical analyses are provided to justify run length as a reliable convergence signal. Experiments on five mathematical reasoning datasets show that ES-CoT significantly reduces the number of inference tokens.

### Strengths
- The paper is easy to read.
- Based on empirical findings, authors make two assumptions: 1) final answers are deterministic, 2) the probability of an intermediate answer being the same as the final answer monotonically increases as the reasoning progresses. Based on the two assumptions, the authors provide a theoretical justification of the run length based early stopping CoT method, which is sound.
- Authors provide sensitivity and robustness analysis of their method.

### Weaknesses
- Experiments focus on mathematical (or formal logic) reasoning datasets.
The generalizability of the method is unclear since the core assumptions (e.g., deterministic final answers) may not hold in other domains such as commonsense or open-ended reasoning.
- Accuracy is not preserved. In Table 1, ES-CoT outperforms the CoT baseline in only 4 out of 20 cases, with 3 of those improvements observed on the larger QwQ 32B model. The method performs poorly on smaller models. While token usage is reduced, this often comes at a substantial cost to accuracy (e.g., on AIME with Qwen3, accuracy drops from 0.73 to 0.50).
- Even when combined with self-consistency prompting, ES-CoT underperforms the baseline (CoT + SC) in nearly all cases.
- The method introduces an additional hyperparameter—the difference threshold between run lengths—yet the paper provides no clear guidance on how to select this parameter. The performance seems to vary a lot depending on this parameter.

### Questions
In Figure 2-(a), why is the probability that step answers match the final answer so low across all models, even at 100% reasoning progress?

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
This paper presents ES-CoT (Early-Stop CoT), an inference-time method that lets a LLM quit early while it is still doing CoT reasoning. At every step the authors force the model to output its current final answer (step answer). If that answer repeats for a suddenly longer run than before, the process stops and returns that answer. No extra training, reward model, or parallel runs are needed. Tests on 5 math/logic datasets (AIME, GPQA, MATH, Minerva and Olympiad) and 3 LLMs (QwQ 32B, Qwen3 8B, DeepSeek-R1-Distill-Llama 8B) show 41 % fewer generated tokens on average while the accuracy is close to that of standard CoT. Theory and ablations are supplied and the code is promised.

### Strengths
1. ES-CoT needs no re-training or extra GPUs, so it's convenient to reproduce the results with LLMs.  
2. The stopping signal (the "jump" in the run-length of identical step-answers) is statistically testable, giving experiments a clear, theory-backed criterion.  
3. Across 5 mathematical/logical datasets and 5 LLMs the method reduces 41% of generated tokens on average, and in several cases even raises accuracy by preventing over-thinking.  
4. When paired with self-consistency, decoding ES-CoT keeps most of its token saving and sometimes yields extra accuracy, showing that the method of Early-Stop is useful to some extent.  
5. Extensive ablation on thresholds and p-value shows stable behaviour, and the failure risk is low and controllable.

### Weaknesses
1. Every reasoning step is interrupted with the manually inserted prompt "The final answer is", which increases prompt tokens and may make some models toward stopping reasoning too early.
2. The datasets used in this paper have deterministic final answers and open-ended, creative tasks are excluded, which may lead to insufficient validation of ES-CoT's universal applicability. 
3. Since no comparison is made against distilled, same-scale short-CoT LLMs that were explicitly trained for shorter reasoning, it remains unclear how much of the 41% token savings is simply due to weak baselines.
4. ES-CoT does not achieve higher reasoning accuracy than standard CoT in most experiments, so despite its efficiency in token usage, there is limited evidence for its substantive effectiveness in reasoning.
5. The high temperature (0.6) may introduce excessive randomness, and the limited model scales tested (only 32B and 8B) restrict the generalizability of the findings. These factors weaken the reliability and applicability of the conclusions.
6. The tables are poorly formatted. For instance, in Tables 3 and 4, the word "DeepSeek" is split across rows, and there are no annotations to draw attention to significant data differences.

### Questions
See the weakness above.

### Soundness
1

### Presentation
2

### Contribution
2
