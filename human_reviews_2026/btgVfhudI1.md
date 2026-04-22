# Short Window Attention Enables Long-Term Memorization

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 2

## Abstract
Recent works show that hybrid architectures combining sliding window softmax attention layers with linear recurrent neural network (RNN) layers outperform both of these architectures taken separately. However, the impact of the window length and the interplay between softmax attention and linear RNN layers remain under-studied. In this work, we introduce SWAX, a hybrid architecture consisting of sliding-window attention and xLSTM linear RNN layers. 

A counter-intuitive finding with SWAX is that larger sliding windows do not improve the long-context performance. In fact, short window attention encourages the model to better train the long-term memory of the xLSTM, by relying less on the softmax attention mechanism for long context-retrieval. 

The issue with small sliding windows is that they are detrimental for short-context tasks, which could be solved with information from moderately larger sliding windows otherwise. Therefore, we train SWAX by stochastically changing the sliding window size, forcing the model to leverage both a longer context window and the xLSTM memory. SWAX trained with stochastic window sizes significantly outperforms regular window attention both on short and long-context problems.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper empirically evaluates how the window size of sliding window attention affects the long-context performance, when mixing sliding window attention with xLSTM linear RNN layers in a hybrid architecture. The authors find that short window attention encourages the model to better train xLSTM, leading to better long-context performance. However, short window attention may lead to worse short-context performance. The authors then propose a simple strategy to randomize the window size in training, achieving the best of both worlds.

### Strengths
1. The paper is well-written, clear, and easy to follow. The paper provides an interesting and practically insightful observation that short window attention leads to better long-context performance in hybrid models mixing window attentions and RNN layers.

2. The authors provide a simple adaption to train a model with good short-context and long-context performance.

### Weaknesses
1. The paper would be stronger if it evaluates more than one types of RNN layers to show that the findings are robust against different RNN architectures.

2. The findings are based on models without fine-tuning with long-context data. It is unclear whether the results continue to hold if the models are further fine-tuned.

### Questions
1. Could the authors comment on the generalizability of the observation to other RNN architectures?

2. Could the authors comment on whether the results continue to hold if the models are further fine-tuned?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes SWAX, a hybrid model that alternates sliding‑window softmax attention (SWA) with xLSTM (linear RNN) layers, arguing that short attention windows during training force the model to use the RNN path as true long‑term memory. Empirically, for 1.4B and 7B models trained on 150B tokens at 16k sequence length, the authors find: (i) shorter windows (e.g., 128) yield better long‑context retrieval on RULER Needle‑in‑a‑Haystack (NIAH) than longer windows (e.g., 2048) when extrapolating up to 131k tokens, and (ii) a stochastic window schedule preserves long‑context gains while recovering short‑context quality at test time with a 2048 window.

### Strengths
- Clear, counter‑intuitive empirical result: In hybrids, short windows train better long‑term memory than long windows. The heat‑map in Fig. 6 (p.8) plus the degradation curves in Fig. 5 (p.7) make the effect easy to verify.
-  The stochastic window + brief annealing yields near‑best short‑context results while keeping long‑context robustness. This is a practical recipe others can adopt with minimal code churn.

### Weaknesses
- Long‑context evaluation is narrow: RULER NIAH is synthetic. The paper lacks real‑task long‑context evaluations.
- On the short‑context suite, the Transformer baseline still leads on average. The paper should contextualize this gap with wall‑clock throughput and memory at inference.
- Design space under‑explored: The hybrid is fixed to a 1:1 SWA:xLSTM interleave and xLSTM only. It would be more valuable if more results on other setup could be added.

### Questions
N/A

### Soundness
3

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
The paper studies hybrid models that interleave sliding-window softmax attention (SWA) with linear RNN layers (xLSTM/mLSTM) in a 1:1 pattern. The core, counter-intuitive claim is that short SWA windows improve long-context recall because they push the model to train its recurrent memory instead of relying on local softmax; conversely, long windows hurt length extrapolation. The authors also propose stochastic window training—randomly switching between short and long windows during pretraining with an anneal near the end—to regain short-context quality without giving up the long-context gains. Empirically, they evaluate mainly on RULER “needle-in-a-haystack” tasks and a suite of short-context benchmarks;

### Strengths
1) Clear, testable capacity-allocation hypothesis. Long windows allow SWA “absorb” the learning signal; short windows force the xLSTM to learn long-term memory. The paper probes this across window sizes and model scales.

2) Simple, practical training recipe. Stochastic window training with a short end-of-training anneal; task-agnostic and effective at 1.4B and 7B.

3) Comprehensive comparisons. Side-by-side results for pure (xLSTM, SWA) vs. hybrid models, compute-aware reporting (FLOPs/token; Table 1), and stringent length-extrapolation stress tests (RULER NIAH up to 131k).

### Weaknesses
1) The pretraining mix is described only at a high level (“mostly web and code”); perplexity is reported on a code subset, but sources, filters, licenses, and exact proportions are not disclosed. This reduces reproducibility and external validity.

2) RULER NIAH—primarily a needle-retrieval test—is the centerpiece, but broader long-context reasoning is underrepresented (multi-hop QA over long documents, extended chain-of-thought, codebase-level tasks). The generality of the findings remains unclear. A stronger case would include evaluations across varied tasks, e.g., Babilong and long open-QA benchmarks (HotpotQA and MuSiQue in full-wiki mode, LongBench).

3) The experiments are exclusively based on the xLSTM as the linear RNN component. A brief discussion or experiment on whether this phenomenon holds for other popular models, including neural memory (for example, Titans[1]), would significantly increase the generalizability and impact of the findings.

[1] Ali Behrouz and Peilin Zhong and Vahab S. Mirrokni. Titans: Learning to Memorize at Test Time, arXiv:2501.00663

### Questions
1) How many random seeds were run per configuration? Please report mean ± standard deviation across seeds for the key metrics in Table 1 and Figures 5–8 (and include the per-seed results in an appendix or supplementary file). If only a single seed was used for any result, please state this explicitly.

2) Could you specify the full optimization setup: the optimizer used (e.g., AdamW), exact β₁/β₂ (and ε) values, weight-decay coefficient, gradient-clipping norm and threshold, all dropout rates?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes **SWAX**, a hybrid that interleaves sliding-window softmax attention (SWA) and xLSTM layers. Core claims: (i) **shorter** SWA windows surprisingly yield **better long-context recall** because they force the RNN path to learn long-term memory; (ii) a **stochastic window** training schedule (occasionally using a small window such as 128, otherwise 2048, with an anneal in the last 10% of training) recovers short-context performance while preserving long-context extrapolation. Evaluations use a 1.4B and 7B model trained on 150B tokens at 16k sequence length; long-context performance focuses on RULER NIAH, with downstream short-context benchmarks reported.

### Strengths
The stochastic window schedule (sampling 128 vs 2048; annealing off in the last 10% of training) is simple, inexpensive to try, and consistently improves length extrapolation without sacrificing short-context quality at both 1.4B and 7B scales. This is well demonstrated in Figs. 7–8 and Table 2.

### Weaknesses
1. **Limited and unbalanced evaluation of “long-context” ability.**
   The central claim rests mostly on **RULER Needle-in-a-Haystack (NIAH)** accuracy; other long-context behaviors (QA, summarization, codebase-level reasoning, tool-use traces, multi-hop retrieval) are not evaluated. As a result, it’s unclear whether the observed gains generalize beyond synthetic token recall. The paper itself emphasizes RULER NIAH and heatmaps across sequence lengths but lacks a broader long-context suite. This weakens external validity of the main claim.  

2. **Confounds in the “short vs long window” story and insufficient controls.**
   The paper argues that **longer** windows under-train the RNN path and therefore **hurt** extrapolation; however, compute is not cleanly equalized across variants (Table 1 shows **FLOPs/token** rising with window size), and the training schedule/data mix may interact with window length (e.g., document length distribution skewed to code). Stronger controls (fixed total FLOPs, matched optimization schedules, varied data mixes, and tests at matched effective receptive fields) are needed to isolate causality. As is, the causal interpretation (under-training the RNN) is plausible but not conclusively established.  

3. **Novelty is incremental relative to prior hybrid work; framing oversells.**
   Alternating xLSTM and local attention is now common; the **new ingredient** here is mainly the **stochastic window schedule** and the empirical observation that small windows train the long-term path better. The paper acknowledges closely related hybrids and prior window-size analyses (e.g., De et al.), and even positions the method as akin to **dropout** on the attention mechanism. Without broader/better-controlled evaluations or theoretical support, the contribution feels like a useful training tip rather than a substantial architectural advance.

### Questions
* Line 075: “**hybrids architecture**” → “**hybrid architectures**.” 
* Line 198: “In **constrast**” → “In **contrast**.” 
* Line 233:  “we also **train evaluate** models at the 7B parameter scale.” → “train **and** evaluate.” 
* Table/figure captions could be clearer (e.g., units/averaging in Fig. 6 heatmap; spell out metrics). 
* **Add broader long-context tasks** (e.g., LongBench-style QA/summarization/code comprehension) to test whether the NIAH gains translate to practical workloads. 
* **Tighter causal tests**: (i) equalize total compute across window settings; (ii) vary document-length distribution; (iii) report how gradients/attention mass shift between SWA and xLSTM under different windows; (iv) include ablations on the anneal schedule (you mention some in Appendix B—bring the most informative ones into the main paper).

### Soundness
3

### Presentation
2

### Contribution
2
