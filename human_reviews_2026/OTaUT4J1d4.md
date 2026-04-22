# SpecExit: Accelerating Large Reasoning Model via Speculative Exit

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 2, 6, 6

## Abstract
Despite their strong performance on reasoning tasks, large reasoning models (LRMs) often suffer from overthinking, producing unnecessarily long outputs and incurring high end-to-end latency, a significant limitation to their real-world deployment. To address overthinking, early-exit mechanisms have been proposed to terminate reasoning before typical completion, showing that this approach can effectively shorten generation length with minimal impact on accuracy. However, their reliance on probing mechanisms introduces a detection overhead that limits their end-to-end latency gains and compromises their generalizability across diverse problems. Inspired by the use of hidden states in speculative decoding, we propose \textbf{SpecExit}, a novel framework that predicts both future tokens and an early-exit signal directly from a lightweight draft model without probing overhead. Our method offers significant improvements, achieving up to 66% generation length reduction and 2.5× end-to-end speedup compared with the speculative decoding baseline, without compromising accuracy. Our method leverages the inherent signals from hidden states to provide effective early-exit signals, suggesting broader use of hidden states for efficient reasoning. Our code is available at: https://anonymous.4open.science/r/SpecExit-B802.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SpecExit, an early-exit framework that integrates with speculative decoding to predict both future tokens and reasoning sufficiency signals from a lightweight draft model, achieving up to 2.5× faster inference and 66% shorter reasoning without accuracy loss.

### Strengths
- Elegant integration of early-exit prediction into speculative decoding with zero probing overhead.
- Uses hidden-state signals (confidence/progress/remaining) to dynamically determine reasoning sufficiency.
- Demonstrates consistent 2×–2.5× speedup and 66% token reduction without hurting accuracy.
- Framework is general, modular, and easily deployable on vLLM or PyTorch inference pipelines.
- Well-designed ablation studies confirm the effectiveness of multi-signal fusion and signal smoothing.

### Weaknesses
- Model-specific training requirement: The MTP-based signal extraction head must be retrained for each target model due to differences in hidden-state representations, limiting plug-and-play generality.
- Limited model family evaluation: Experiments are conducted only on two reasoning model families, leaving generalization to other architectures or general-purpose LLMs unclear.
- Signal & smoothing dependency: The method relies on multiple signals and smoothing strategies (e.g., EWMA) to remain stable, indicating insufficient robustness of raw signals.

### Questions
- Since the MTP and signal heads must be retrained for each model due to architectural differences, how costly is this process in practice? Could a lightweight or partially shared training strategy (e.g., freezing most of the MTP layer or distilling from a pretrained head) reduce the retraining overhead across models?
-  Can the authors provide results or at least discussion on how SpecExit might generalize to other model architectures, such as Phi, Mistral, or LRMs?
- How sensitive is SpecExit’s performance to the chosen thresholds for confidence, progress, and remaining length? like use different thresholds to do some ablation studies. Why do you choose these thresholds in STEER method (like confidence > 0.8 && progress > 0.3 && remaining < 200)?
- Can the authors identify and analyze typical scenarios where the control signals conflict (e.g., rising confidence but low progress), and show how the EWMA controller behaves in these situations?

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
4

### Summary
The work introduces SpecExit, a reasoning-aware early exit framework for reducing the overall latency of the reasoning process. The method involves using a combination of speculative decoding and multi-task learning, to generate confidence signals from the draft model which guide the system to pre-empt the reasoning process eﬃciently. Evaluation across a range of tasks, the work claims to achieve 66% reduction in average generation length and 2.5x speedup in end-to-end latency, compared to the baseline.

### Strengths
1. The paper combines speculative decoding, multi-task learning, and confidence-based exit criteria in a coherent way to address reasoning efficiency.

2. Demonstrates significant inference latency reduction with minimal accuracy degradation across tasks.

3. The analysis of different signal types (Confidence / Progress / Remaining Reasoning Length) and smoothing methods (No Smoothing / Momentum / Sliding Window / EWMA) is thorough and informative.

4. The method offers a general framework applicable to existing reasoning models without modifying their core architecture.

### Weaknesses
- Line 267: The notion of “recover token” is unclear. How does it differ from the reasoning-end marker  ("\</think>")?
- Lines 363–364: The variant using combined signals (Confidence + Progress + Remaining Length) is denoted as SpecExit*, yet Table 1 reports results for SpecExit. Which configuration do these results correspond to?
- Table 1: Ambiguity exists between “Think” (line 333) and “Vanilla” (line 339). If they refer to the same baseline, HUMANEVAL+ accuracies (90.9 vs 88.4) are inconsistent.
- Table 1 caption should define what boldface indicates (best overall or best among comparable methods).

---
- Lines 313–314 mention that the speculative decoding baseline is trained on the same dataset as EAGLE3, but the draft model configuration is unspecified.
- What is the relationship between the draft and target models (same family, smaller variant, or identical)? Without this detail, reproducibility and interpretation of results are limited.

---
- Please quantify additional compute (e.g., FLOPs or GPU hours) for:
    - Dataset construction, which involves multiple reasoning traces per prompt to determine optimal lengths.
    - Multi-task learning with additional loss heads.
- These costs should be weighed against runtime improvements.
---

- Does the target LLM architecture affect training of the MLP heads? Since the dataset is derived from a specific model’s reasoning traces, how is overfitting avoided? Are exit heads transferable across models within the same family?
---

- The inference flow involving the MTP mechanism, lightweight draft model, MLP head, and target model is confusing. Please clarify whether:
    - The draft model includes MTP heads,
    - These predictions are passed to the target for verification, and
    - The target model then generates extended outputs via a linear layer.
- A schematic or algorithmic diagram would make the inference pipeline clearer.
---

- What is the motivation for three auxiliary losses (L_conf, L_prog, L_rem). The rationale for choosing these specific losses is unclear. What intuition or empirical observation led to defining these signals? Figure 5 shows dataset-dependent variation for Progress and Remaining signals, while Confidence remains consistently low. What happens if only Progress and Remaining signals are used?
- The abstract claims an “average generation length reduction of 66%,” but results suggest this is the maximum observed, not the mean. The phrasing should be corrected to “up to 66% reduction.”

### Questions
- Please provide detailed configurations of the draft model and its relation to the target model (architecture, size, tokenizer compatibility).

- Can the MLP exit heads trained on one model generalize to another of similar architecture?

- How large is the training overhead compared to runtime gains?

- Could the authors clarify whether the method assumes the draft and target models belong to the same family?

- Why were the three particular auxiliary losses (L_conf, L_prog, L_rem) chosen, and is there theoretical justification for their complementarity?

### Soundness
2

### Presentation
3

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
This paper introduces SpecExit, a novel framework designed to accelerate inference in Large Reasoning Models (LRMs) by tackling the "overthinking" problem, where models produce unnecessarily long reasoning chains, leading to high latency. The core idea is to integrate an early-exit mechanism directly into the speculative decoding process. Unlike prior methods that rely on costly probing of the target model, SpecExit extends the lightweight draft model to predict not only future tokens but also three auxiliary signals: confidence, reasoning progress, and remaining reasoning length. These signals are learned via multi-task training on data where the minimal sufficient reasoning length is heuristically identified. During inference, these signals are used to dynamically terminate the reasoning process at natural breakpoints (e.g., paragraph ends) without any modification to the target model or extra probing overhead. Experiments on various reasoning benchmarks with models like Qwen3-4B and DeepSeek-R1-Distill show that SpecExit can reduce generation length by up to 66% and achieve end-to-end latency speedups of up to 2.5x compared to a standard speculative decoding baseline, with minimal impact on task accuracy.

### Strengths
The primary strength of this work lies in its novel and elegant integration of two distinct lines of research: speculative decoding for per-token acceleration and early-exit mechanisms for reducing reasoning length. Instead of treating these as separate problems, the authors propose a unified framework. The key insight—to offload the exit-signal prediction to the draft model and leverage its hidden states—is highly original. This avoids the well-documented overhead of probing-based early-exit methods, which is a creative and impactful conceptual leap.

### Weaknesses
1. The author should introduce some basic knowledge of speculative decoding to facilitate readers who are not familiar with speculative decoding to understand the content of the paper.

2. Figure 2 only shows the forward process of one token. I don't understand how to predict the vocab logits and signals of multiple tokens simultaneously. The author needs to explain in detail how the MTP layer in the paper is designed. Is it exactly the same structure as Medusa or EAGLE?

3. The author combines speculative decoding with exit prediction, but I believe there is no coupling relationship between the two. According to the method designed in this paper, adding an MLP layer directly to the hidden feature of the token can also predict the exit signal without the need for an MTP layer.

4. The author can supplement the experimental results without speculative decoding (only predicting the exit signal of each token).

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces **SpecExit**, a *reasoning-aware early-exit framework* designed to reduce overthinking in large reasoning models (LRMs).
Instead of relying on explicit probing phrases (e.g., “Final Answer is”), SpecExit leverages **hidden states** from a lightweight **draft model** to jointly predict (i) future tokens and (ii) early-exit signals—**confidence**, **reasoning progress**, and **remaining reasoning length**—in a single forward pass.
The method removes probing overhead, integrates smoothly with **speculative decoding**, and dynamically stops generation at semantically coherent boundaries such as paragraph delimiters.
Experiments on GSM8K, MATH500, AIME, GPQA, HumanEval+, and ARC-Challenge show up to **66 % shorter reasoning chains** and **2.5 × latency reduction** without accuracy loss.
Ablation studies confirm that multi-signal integration with EWMA smoothing yields the best balance between efficiency and correctness.

### Strengths
* Novel integration of early-exit and speculative decoding via hidden-state supervision.
* Eliminates probing overhead while maintaining accuracy.
* Multi-signal design (confidence + progress + remaining length) improves robustness across tasks.
* Experiments on six benchmarks and two reasoning-optimized backbones.
* Ablations on signal types, smoothing strategies, and segmentation markers.
* Reproducibility and open-source release fully documented.
* Clear contextualization within reasoning-efficiency literature (DEER, EAGLE, RL-based control).

### Weaknesses
* Lacks a **formal theoretical analysis** of why hidden-state features correlate with reasoning sufficiency.
* Dependence on explicit *<think>* / *</think>* markers may limit generalization to free-form reasoning.
* Limited model family evaluation; Experiments are conducted on two reasoning model families.

### Questions
1. How sensitive is SpecExit to the chosen thresholds (confidence > 0.8, progress > 0.3)? Could these be learned dynamically?
2. Have you tested **cross-model transferability** of exit heads (e.g., trained on Qwen, applied to DeepSeek)?
3. How would SpecExit behave with **long-context**?
4. Could you quantify **energy / GPU-hour savings** corresponding to the reported 2.5× latency reduction?
5. Can you provide a theoretical account—stating assumptions and guarantees—that explains why functions of intermediate hidden states serve as reliable proxies for ‘reasoning sufficiency’ (e.g., monotonicity/calibration of the confidence and progress heads)?

### Soundness
3

### Presentation
3

### Contribution
3
