# Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Large reasoning models achieve strong performance through test-time scaling. However,  this comes at the cost of substantial computational overhead, particularly from excessive token generation on short input prompts. While sparse attention mechanisms can reduce latency and memory usage, existing approaches suffer from accuracy degradation due to accumulated errors during long-generation reasoning. These methods generally require either high token retention or costly retraining. We introduce LessIsMore, a {\em training-free} sparse attention mechanism for reasoning tasks. Unlike existing approaches that rely on head-specific local optimizations, LessIsMore leverages {\em global} attention patterns by aggregating token selections across heads with recent contextual information. This unified cross-head ranking enables more efficient token selection for future decoding layers, eliminating the need to maintain separate token subsets per head and improving both generalization and efficiency. Evaluation across diverse reasoning tasks and benchmarks shows that LessIsMore preserves---and in some cases improves---accuracy while achieving up to $1.6\times$ end-to-end decoding speedup compared to full attention. Moreover, LessIsMore attends to $2\times$ fewer tokens without accuracy loss and accelerates sparse attention computation by up to $1.72\times$ compared to existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper primarily introduces LessIsMore: a post-training and training-free mechanism to speed up the transformer forward pass. The LessIsMore framework targets the attention mechanism, a traditional memory and time bottleneck, in transformer-based LMs. This paper identifies the trend of spatial locality: the observation that there is substantial overlap in token-importance rankings across heads in a given decoding layer. They also identify the trend of recency locality: the observation that recently generated tokens are attended to more highly in subsequent steps. Using the observations of spatial and recency locality, they formulate cross-head unified sparse attention which combines attention head-specific local information with a cross-head global attention pattern to more accurately target relevant tokens during the decoding process. They profile the LessIsMore sparse attention method across several LLMs on several long-context reasoning tasks and show substantial speed-ups with minimal loss in task accuracy.

### Strengths
1. This paper tackles an important problem: inference efficiency. With both the size of LLMs and the tasks they are used for increasing, inference can become bottlenecked by the costly attention operation. Further research to enable faster inference with minimal impact to inference accuracy is a promising research area.
2. This paper is well written and easy to follow.
3. The observation of spatial locality is an interesting one; the authors are correct to point out that many prior works have shown that attention heads play “specialized” roles. This paper argues the contrary, and is able to defend their observation by showing that employing a “global” view of token distributions across heads in a given layer informs better token selection for their sparse attention scheme.
4. The adaptive scheme of reserving X% of token positions for the recency window is a logical method to strike the balance between accounting for both recently decoded tokens and earlier tokens (e.g., attention sink tokens [2]).

[1] Lost in the Middle: How Language Models Use Long Contexts. TACL, 2023.

[2] Efficient Streaming Language Models with Attention Sinks. ICLR, 2024.

### Weaknesses
- In figures 5.a and 5.b why don’t you compare all methods in each subplot. I want to see the speedup + latency of proposed vs. all baseline methods.
- You reference StreemingLLM [2] in section 2.2.2 as you motivate the design of adaptively sampling tokens from a recency window. Is there a reason you do not include the StreamingLLM decoding method as a baseline strategy in your analysis alongside TidalDecode, SeerAttention-r, Quest?

### Questions
- You discuss the time speed-up/latency improvements achieved by LessIsMore. In addition to time, you also cite memory as a motivating concern for the design of LessIsMore in the introduction. Can you discuss the memory footprint of LessIsMore in comparison to your comparison baselines: TidalDecode, SeerAttention-r, Quest?
- Prior work has shown that models often struggle to complete tasks that require use of the “middle” of the context [1]. By design, LessIsMore accounts for more recently decoded tokens, and as you note prior work has shown the presence of “attention sinks” where initially produced tokens hold a high amount of attention mass [2]. Due to LessIsMore’s design of explicitly accounting for more recently generated tokens, and selecting “important” tokens (which may lie at the beginning of a sequence), do you notice that tokens in the “middle” of generation are consistently not selected? Could you discuss LessIsMore in the context of [1]?
- I am not sure if I fully understand what Figure 2 is showing. How do you determine what the “ground-truth top-4K tokens” are? What do you mean by "positional token” (in the legend)? And consequently, what does it mean for a “positional token” to be selected? Some additional text in section 2.2.1/2.2.2 to explain may be helpful.

[1] Lost in the Middle: How Language Models Use Long Contexts. TACL, 2023.

[2] Efficient Streaming Language Models with Attention Sinks. ICLR, 2024.

[1] Lost in the Middle: How Language Models Use Long Contexts. TACL, 2023.

[2] Efficient Streaming Language Models with Attention Sinks. ICLR, 2024.

### Soundness
3

### Presentation
3

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
Training-free method that induces sparsity in attention via token selection/aggregation across heads during decoding. Evaluated mainly on math reasoning (e.g., AIME-24/25, MATH500, GPQA-Diamond). Shows accuracy vs. token-budget tradeoffs and latency relative to a dense baseline.

### Strengths
- Simple, training-free mechanism to enforce attention sparsity.
- Comprehensive math-reasoning evaluation with clear pass@1 reporting.
- Appears drop-in and compatible with standard decoding stacks.

### Weaknesses
- The statement on index sharing across layers (Line 296) could use further justification or empirical evidence — for example, analyzing whether top-k tokens remain stable across layers in practice.
- The main novelty appears to lie in token aggregation across attention heads; adding more discussion or ablations could help highlight its unique contribution.
- The evaluation setup (e.g., “average pass@1 over 16/8 samples”) could be elaborated — how were samples selected, and what motivates such small test sets?
- Figure 4 shows performance vs token budget for different SOTA sparse attention methods, however there is no efficiency (flops/latency saved ) analysis for different sparse attention methods , figure 5 shows latency comparison but its only relative to baseline (no sparsity). 
- It would be helpful to see how different sparse attention baselines behaves for longer context lengths (beyond 8K/16K tokens).
- Adding more sparse attention baselines (see Efficient Attention Mechanisms for Large Language Models: A Survey, arXiv:2507.19595) could provide better context.
- Since the method targets efficient reasoning, exploring broader tasks—such as code generation, tool-use, or long-context reasoning (LongBench, needle-in-a-haystack)—would test its generality and robustness. 
- The idea of focusing attention on recent tokens has been explored in earlier works (e.g., StreamLLM and related sparse attention methods)

### Questions
- What is the rationale for assuming top-k token indices remain useful across subsequent layers?
- Could the authors provide ablations isolating the effects of cross-head aggregation and token selection frequency?
- Can they include absolute compute metrics (FLOPs, latency per sequence, memory usage) for better comparison?
- How does sparsity affect performance on long-context or structured tasks?
- Are there plans to expand baselines to include more diverse sparse attention approaches?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes "LessIsMore" which is a training-free sparse-attention method tailored for long generation reasoning tasks. Instead of letting each head pick its own top-k tokens, it aggregates head-level candidates into a global, cross-head ranking and always keeps a fixed recency window of the latest tokens. On Qwen3-8B/4B across reasoning benchmarks (AIME-24/25, GPQA-Diamond, MATH500), it reports near-lossless accuracy while attending to about 2× fewer tokens and achieving roughly 1.1× average decoding speedups compared to full attention.

### Strengths
1) This is a simple, training-free method which doesn't require model changes or retraining. 

2) The paper Identifies spatial locality (overlap across heads) and recency locality (recent tokens remain important), motivating a global and recency selector. 

3) They show strong empirical results on reasoning tasks which matches or improves accuracy at higher sparsity; avoids the lengthening of generations that hurts some sparse methods.

### Weaknesses
1) Their results only focus on e.g. Qwen3 (with GQA backbone), questions remain how does it work on non-GQA/MHA models (e.g., Llama-3.x, Mistral)?

2) Some gains may be tied to custom GQA kernel support. Can you quantify benefits under other stacks like (vLLM/Flash-Attention) and with speculative decoding or KV-cache quantitation?

3) Is the cross-head overlap is something specialised to GQA or a general property? Can you show comparisons between MHA and GQA by keeping other factors fixed?

### Questions
Look at the weaknesses.

### Soundness
2

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
2

### Summary
The paper introduces LessIsMore, a training-free sparse attention mechanism tailored for reasoning LLMs (DeepSeek-R1, Qwen3).
It identifies two empirical “locality” phenomena during reasoning: (1) Spatial locality across attention heads: token-importance overlaps heavily across heads within the same layer. (2) Recency locality: recently generated tokens are consistently attended more. 


Based on these findings, the authors propose: (a) Cross-Head Unified Sparse Attention (CUSA): aggregate per-head top-k tokens globally into one shared subset. (b) Stable Recency Window: reserve a fixed ratio r (25%) of tokens for the most recent context.
They show Up to 1.6X end-to-end decoding speedup (and 1.72X kernel-level speedup) with no accuracy loss across AIME-24/25, GPQA, and MATH500, compared to TidalDecode and SeerAttention-r.

### Strengths
**1) Strong empirical motivation and analysis:** Figure 2 & Appendix A.4 genuinely show cross-head overlap and consistent recency trends.
Verified with visualizations at 10K–25K decoding steps; supports the “shared importance” hypothesis.

They correctly recognize reasoning != retrieval: long decoding amplifies error accumulation (Fig. 1b).

**2) Novel, simple design:** The Cross-Head Unified Selection (Algorithm 1, lines 11–14) is conceptually clean and implementable atop any existing sparse attention

**3) Training-free and kernel-friendly:** No retraining or finetuning is required

### Weaknesses
1)  The evaluation uses 16 samples (AIME-24/25) and 8 samples (MATH500/GPQA) to report pass@1 accuracy and compare methods (see §4.1). With such small n, 2–3 pp differences are plausibly within noise; there are no confidence intervals, seed variance, or significance tests.

2) Re-selection and layer choices ad-hoc? Section 4.1 fixes token-selection layers (12, 20) based on a “needle-in-the-haystack” test, but this heuristic is weakly justified and may bias results.


3) Sparse selection applied only at two layers, unclear if additional layers (e.g., alternating pattern) could further speed up.

### Questions
1) Layer sensitivity: Why choose Layer 12 and 20 specifically? does earlier or later selection harm performance?

2) Ratio r = 0.25: Is 25% constant across all tasks, or was it tuned? Can adaptive r perform better?

3) Have you empirically measured correlation between attention recall and benchmark accuracy across tasks?

### Soundness
3

### Presentation
3

### Contribution
2
