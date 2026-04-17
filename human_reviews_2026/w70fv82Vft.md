# SLAKE: Softmax-Approximated Training-Free Linear Attention with KV-Cache Eviction for Long-Sequence LLMs

- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Recent advances in transformer-based large language models (LLMs) have enabled inference over contexts as long as 128K tokens. However, the quadratic computational and memory costs of full self-attention remain a fundamental bottleneck at such scales. Prior efforts to mitigate this challenge largely fall into two camps: (i) structural approximations (e.g., linear attention) that reduce asymptotic complexity but typically require costly retraining, and (ii) KV-cache optimizations (e.g., eviction or merging) that are training-free yet inevitably discard information. We introduce Softmax-Approximated Training-Free Linear Attention with KV-Cache Eviction (SLAKE), a novel framework that unifies the complementary advantages of these two paradigms. At its core, SLAKE employs Partially Taylor-Approximated Attention (PTAA), which leverages a first-order Taylor expansion to selectively linearize the Softmax attention kernel. This design enables tokens deemed low-importance via eviction scoring to be processed efficiently with linear attention, while preserving exact Softmax computation for high-salience tokens. To further improve cache efficiency, we propose Value-Aware Budget Scoring (VABS), a new allocation strategy that incorporates value contributions and overcomes key limitations of previous eviction heuristics. Extensive experiments on LLaMA-3 8B demonstrate that SLAKE delivers up to 10$\times$ inference speedup and 30.8\% peak-memory reduction on 128K-token sequences, while keeping accuracy loss below 4\%. To our knowledge, SLAKE is the first training-free approach to jointly integrate linear attention with KV-cache eviction, establishing a new state of the art among long-context, training-free methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work proposes a framework that improves the efficiency of long-context inference in large language models without retraining. It introduces Partially Taylor-Approximated Attention (PTAA), which applies a first-order Taylor expansion to partially linearize the Softmax attention kernel so that low-importance tokens can be processed with linear attention while high-importance tokens retain exact Softmax computation. To complement this, it presents Value-Aware Budget Scoring (VABS), a dynamic cache-budget allocation method that accounts for both the approximation error of PTAA and the influence of the value matrix when selecting tokens for eviction. Together, PTAA and VABS combine the advantages of linear attention and KV-cache compression in a training-free manner. Experiments on LLaMA-2-7B, LLaMA-3.1-8B, and Mistral-7B-v0.3 using the LongBench benchmark show that SLAKE achieves up to a 10× inference speedup and 30.8% peak-memory reduction for 128K-token sequences while maintaining less than 4% accuracy loss, establishing a new state-of-the-art among training-free long-context methods.

### Strengths
1. The paper introduces a novel combination of linear attention and KV-cache eviction into a single training-free framework. Its Partially Taylor-Approximated Attention and Value-Aware Budget Scoring components offer new ways to linearize Softmax and allocate cache budgets by explicitly modeling value contributions.
2. The paper is well-structured, and the main ideas and methodology are easy to follow.

### Weaknesses
1. The ablation study is not sufficiently comprehensive to fully support the individual contributions of PTAA and VABS. The current results show average LongBench scores under three configurations (eviction only, +PTAA, +VABS on LLaMA2-7B with a 128-token cache budget).
2. The experimental design lacks sensitivity analyses for the approximation hyperparameters (e.g., Taylor truncation order, scaling factors in VABS). Reporting how performance and stability vary with these parameters would clarify the robustness of the method.
3. Several experimental details are missing, including decoding hyperparameters such as temperature, maximum generation length, and top-p values. Reporting these settings would improve the reproducibility and interpretability of the experimental results.

### Questions
See the Weaknesses section.

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
4

### Summary
This paper proposes SLAKE, a training-free framework to address the quadratic complexity bottleneck in LLM long-sequence inference. SLAKE is the first training-free approach to jointly integrate linear attention with KV-cache eviction. Its core PTAA mechanism uses exact Softmax computation for high-salience tokens while processing low-importance tokens with a Taylor-approximated linear attention. Furthermore, it introduces VABS, a novel cache allocation strategy that incorporates the influence of the Value matrix to overcome limitations of previous eviction heuristics. Experiments show SLAKE outperforms existing training-free methods on LongBench, achieving up to a 10x inference speedup and a 30.8% peak-memory reduction on 128K-token sequences .

### Strengths
1. The paper introduces SLAKE, the first framework to be training-free while unifying two distinct optimization paradigms: linear attention and KV-cache eviction. Its core mechanism, PTAA, innovatively retains information from evicted tokens via Taylor approximation instead of discarding it, which directly addresses the information loss problem of standard eviction methods.
2. SLAKE achieves state-of-the-art results, consistently outperforming other training-free eviction methods like H2O and CAKE on the comprehensive LongBench benchmark. This accuracy gain is driven in part by VABS, a more insightful scoring metric that corrects a key flaw in prior methods by accounting for the Value matrix's influence.

### Weaknesses
1. Ambiguous Computational Cost: The paper is unclear about the prefill stage computational cost. Both VABS (requiring "true attention") and PTAA (requiring $x_{i,max}$) could hide an $O(N^2)$ step, which would weaken the efficiency claims.Weak Retrieval 
2. Performance: On NeedleBench (Table 4), SLAKE's performance drops significantly compared to Full KV, suggesting the approximation struggles with high-fidelity information retrieval tasks.
3. Hyperparameters: VABS introduces hyperparameters $\alpha$ and $\beta$ (Table 3), and the paper does not discuss their sensitivity or the cost of tuning them.
4. Limited Model Scale Validation: The paper's experiments are confined to 7B and 8B models. While sufficient to compare against other training-free eviction methods, cited related work (like Linearized LLM) has explored 13B models. This lack of validation on larger-scale models leaves the method's scalability in question.

### Questions
1. VABS Cost: How is the VABS score (Eq. 15), which requires the "true attention output," computed during the prefill stage without incurring an $O(N^2)$ cost?

2. PTAA Cost: How is the $x_{i,max}$ normalization term for the Taylor approximation calculated? Is it from all $N$ tokens (implying $O(N^2)$) or only the $w$ kept tokens (which would be an inaccurate normalizer for the evicted tokens)?

3. Prefill vs. Decoding: Does the 10x speedup refer only to the $O(N)$ decoding phase, or is it an end-to-end time including prefill?

4. VABS Tuning: How sensitive is VABS performance to the $\alpha$ and $\beta$ hyperparameters? What is the cost of tuning these for a new model?

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
This paper proposes SLAKE, a training-free framework combining linear attention and KV cache attention for efficient long context inference.

### Strengths
1. The idea of combining linear attention and KV cache eviction is interesting.

2. The presentation is clear and straightforward.

### Weaknesses
1. The main concern I have with this paper is the limited novelty of combining linear attention and KV cache eviction. First, the Taylor-based method to approximate linear attention has been explored in prior work [1]. Moreover, the improvement of considering value tensors seems incremental, offering marginal benefits in both algorithm and system evaluations.

2. The motivation for this paper is still unclear to me. According to Figure 4, there is no significant difference comparing prior methods in terms of memory usage and throughput.

3. More model sizes should be evaluated for scalability, such as 13B/30B.

4. Some system-related works on KV cache compression are missing [2-4].





[1] ViTALiTy: Unifying Low-rank and Sparse Approximation for Vision Transformer Acceleration with a Linear Taylor Attention, HPCA 2023.

[2] InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management, OSDI 2024.

[3] Keyformer: KV Cache Reduction through Key Tokens Selection for Efficient Generative Inference, MLSys 2024.

[4] ALISA: Accelerating Large Language Model Inference via Sparsity-Aware KV Caching, ISCA 2024.

### Questions
Please see the weaknesses.

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
4

### Summary
SLAKE proposes training-free long-context inference for LLMs by selectively mixing exact Softmax attention with a first-order Taylor linear approximation. A novel Value-Aware Budget Scoring (VABS) decides which tokens stay in the KV-cache; the rest are handled by the cheap linear path. On 128 k-token inputs SLAKE gives ≈ 10× speed-up and 30 % peak-memory reduction versus full-cache while losing < 4 % accuracy on LongBench (Llama-2-7B, Llama-3.1-8B, Mistral-7B). The method is model-agnostic, needs no retraining, and is orthogonal to FlashAttention-2.

### Strengths
1. This paper proposes the first approach to unify linear attention with KV-cache eviction without any gradient updates, delivering large wall-clock & memory gains on consumer GPUs.
2. PTAA kernel keeps pre-trained weights intact; VABS explicitly models value-matrix error amplification, yielding consistent gains across 16 long-context datasets.
3. Ablation studies, three model families, two cache budgets, needle-in-haystack stress test, and hardware numbers (H100 latency / peak mem) are all reported.

### Weaknesses
1. Taylor linearisation of Softmax and “important vs. rest” attention mixing are well-explored ideas; SLAKE’s contribution is largely combinational.
2. Only average scores are given; no per-task statistical significance, error bars, or worst-case degradation analysis—crucial for safety-critical uses.
3. Hyper-parameter fragility: VABS needs manually tuned $\alpha$, $\beta$, $\gamma$ per model & budget; no adaptive or online scheme, and no study on sensitivity to these constants.

### Questions
1. How does SLAKE behave with longer contexts (256 K–1 M) or larger models (70 B+) where the approximation error may accumulate?
2. Have you evaluated on code generation, tool-use, or multilingual tasks that may exhibit different attention patterns?
3. Can VABS be made online & input-adaptive instead of relying on fixed $\alpha$, $\beta$, $\gamma$, and what is the computational overhead of such adaptation?

### Soundness
3

### Presentation
2

### Contribution
3
