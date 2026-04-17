# Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Autoregressive decoding in large language models (LLMs) requires $\mathcal{O}(n)$ sequential steps for $n$ tokens, fundamentally limiting inference throughput. Recent diffusion-based LLMs (dLLMs) enable parallel token generation through iterative denoising. However, current parallel decoding strategies rely on fixed, input-agnostic heuristics (e.g., confidence thresholds), which fail to adapt to input-specific characteristics, resulting in suboptimal speed-quality trade-offs across diverse NLP tasks. In this work, we explore a more flexible and dynamic approach to parallel decoding. We propose **Learning to Parallel Decode (Learn2PD)**, a framework that trains a lightweight and adaptive filter model to predict, for each token position, whether the current prediction matches the final output. This learned filter approximates an oracle parallel decoding strategy that unmasks tokens only when correctly predicted. Importantly, the filter model is learned in a post-training manner, requiring only a small amount of computation to optimize it (minute-level GPU time). Additionally, we introduce **End-of-Text Prediction (EoTP)** to detect decoding completion at the end of sequence, avoiding redundant decoding of padding tokens. Experiments on the LLaDA benchmark demonstrate that our method achieves up to **22.58×** speedup without any performance drop, and up to **57.51×** when combined with KV-Cache.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Learn2PD, a learnable parallel decoding framework for dLLMs that replaces fixed heuristics with a lightweight filter model that predicts, per token, whether its current prediction already matches the final output. 
This adaptive filter enables tokens to be “unmasked” dynamically and in parallel, improving the speed–quality tradeoff. They also introduce an EoTP mechanism to detect completion and prevent redundant decoding, and experiments show significant speedups.

### Strengths
1. The problem this paper tries to address is an important one: current dLLM decoding is mainly heuristic-based. A hyperparameter needs to be used (like confidence) for unmasking. The neural predictor is indeed a plausible solution. 

2. The speedup achieved is impressive. 

3. The evaluation is qutie solid, including compatibility with KV cache methods.

### Weaknesses
1. Using a KV cache during inference can cause a misalignment with the training process, potentially altering the model's generated outputs from the training ground truth.

2. As shown in Table 2, there is a big acc drop with dual cache (5.83). Is this caused by Learn2PD? 

3. The evaluation is mainly on LLaDA. How does the method generalize to other models like Dream?

4. How much is the inference overhead is the additional predictor? 

5. How does the performance change with respect to the size of the predictor?

### Questions
Please see the weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Learn2PD and EoTP, two methods to accelerate inference in diffusion-based large language models (dLLMs), addressing inefficiencies in current parallel decoding approaches. Learn2PD trains a lightweight and adaptive filter model to predict whether the current prediction matches the final output. EoTP detects decoding completion at the end of the sequence, avoiding redundant decoding of padding tokens. With these methods, they achieve significant speedup without performance drop.

### Strengths
Decoding strategy for dllm is a very important topic. Unlike autoregressive models, dllm needs to decide which token to decode and which position to decode in each denoising step, which is much harder. This paper proposes an inspiring method by learning to predict the position that is ready to be decoded.

Experimental results show that this method can accelerate the inference significantly.

The filter model only takes the confidence score as input and exhibits very good prediction results. This means that the logits contain rich information that can be leveraged.

### Weaknesses
**Section 3.1.2 is difficult to understand.**
> we measured the amount of unnecessary and repetitive decoding, which is defined as the number of times the model continues to decode a token after that token has first matched the reference answer

What does this sentence mean? Does this mean that the model predicts the correct token but fails to unmask it?

**Figure 2 and 4 are hard to read.**

**Limited Evaluation benchmarks and baselines**

### Questions
see weaknesses

### Soundness
2

### Presentation
2

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
This paper addresses the autoregressive decoding bottleneck in large language models (LLMs) by focusing on diffusion-based LLMs (dLLMs), which enable parallel token generation via iterative denoising. However, existing parallel decoding strategies rely on fixed heuristics (e.g., confidence thresholds) that fail to adapt to input-specific characteristics, leading to suboptimal speed–quality trade-offs. The authors propose Learn2PD (“Learning to Parallel Decode”): a lightweight filter model trained in a post-training phase (requiring only minutes of GPU time) to predict whether a token’s current prediction matches the final output, thereby approximating an ideal “Extremely Greedy Parallel” (EGP) oracle that unmasks tokens only when predictions are correct—avoiding redundant re-decoding. Additionally, they introduce an End-of-Text Prediction (EoTP) mechanism to detect sequence termination and skip decoding of padding tokens. Evaluated on the LLaDA benchmark (GSM8K, MATH, HumanEval, MBPP), Learn2PD + EoTP achieves 22.58× speedup with no performance loss, and 57.51× with KV-Cache at minimal accuracy cost.

### Strengths
The method is elegantly simple and highly efficient: the filter model uses a two-layer MLP that takes block-level confidence scores as input and outputs a binary decision (remask or not) via a sigmoid-thresholded logit . Its computational overhead is negligible (<1% of inference time), avoiding complex architectures or task-specific engineering—demonstrating the principle that “simplicity works.” Moreover, the approach is highly reproducible: pseudocode is clearly provided, and inference requires no ground-truth labels, making it well-suited for open-source release and practical deployment.

### Weaknesses
- **Limited model and architecture coverage**: The method is evaluated exclusively on LLaDA (8B-Instruct) and not tested on other dLLMs such as DiffuLLaMA (Gong et al., 2025) or Dream (Ye et al., 2025), raising concerns about generalizability and model dependency.
  
- **Insufficient task and baseline comparisons**: The empirical analysis (e.g., Figure 2) focuses only on subsets like GSM8K and HumanEval, lacking comprehensive evaluation across the full LLaDA benchmark suite. Moreover, comparisons are limited to the official LLaDA baselines, omitting key alternatives such as confidence-threshold-based or slow-fast sampling strategies.

- **Oracle claims and training robustness**: The paper claims EGP achieves 15–20× optimal speedup, yet Figure 4 only reports a median of 2 steps per block without worst-case (lower-bound) analysis. Additionally, while the post-training phase is said to take “minutes,” there is no ablation on hyperparameter sensitivity (e.g., learning rate, threshold) or convergence behavior.

### Questions
- **Model details**: What are the exact specifications of \( f_\theta \)? Hidden dimension size? Activation function (e.g., ReLU)? Could incorporating confidence signals from more than one block improve performance? Additional experiments would help clarify design choices.

- **Transferability**: How would Learn2PD adapt to non-LLaDA dLLMs like Dream? Does retraining the filter incur prohibitive costs? What is the cross-task generalization gap (e.g., train on GSM8K, test on MATH)?

- **Robustness**: How does the method perform under low-resource settings (e.g., zero-shot) or with adversarial/noisy prompts? Furthermore, the “re-decoding gap” distribution in Figure 2 is based on only 10 samples—what are the mean and standard deviation across the full dataset?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes Learn2PD, a dynamic re-masking method that enables more efficient generation in diffusion language models. A learnable filter model is employed to achieve substantial acceleration in generation performance.

### Strengths
1. The paper presents a clear motivation, employing deep learning techniques to design a more principled unmasking mechanism.

2. The proposed Filter model is lightweight and easy to obtain.

### Weaknesses
1. The main experiments are conducted on the LLaDA model, and the effectiveness of the proposed method on causal diffusion models such as Dream remains to be further verified.

2. There is concern about the method’s performance on finer-grained blocks in mainstream block-level diffusion models.

### Questions
1. The proposed filter adopts a pure MLP architecture, which can be understood as a form of confidence-based feature engineering. Have you attempted to incorporate context-aware structures such as attention mechanisms?

2. There is a typo: on line 326, it should be 256 and 1024.

### Soundness
2

### Presentation
3

### Contribution
3
