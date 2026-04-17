# Let's (not) just put things in Context: Test-time Training for Long-context LLMs

- Decision: Accept (Poster)
- Scores: 2, 4, 6

## Abstract
Progress on training and architecture strategies has enabled LLMs with millions of tokens in context length. However, empirical evidence suggests that such long-context LLMs can consume far more text than they can reliably use. On the other hand, it has been shown that inference-time compute can be used to scale performance of LLMs, often by generating thinking tokens, on challenging tasks involving multi-step reasoning. Through controlled experiments on sandbox long-context tasks, we find that such inference-time strategies show rapidly diminishing returns and fail at long context. We attribute these failures to score dilution, a phenomenon inherent to static self-attention. Further, we show that current inference-time strategies cannot retrieve relevant long-context signals under certain conditions. We propose a simple method that, through targeted gradient updates on the given context, provably overcomes limitations of static self-attention. We find that this shift in how inference-time compute is spent leads to consistently large performance improvements across models and long-context benchmarks. Our method leads to large 12.6 and 14.1 percentage point improvements for Qwen3-4B on average across subsets of LongBench-v2 and ZeroScrolls benchmarks. The takeaway is practical: for long context, a small amount of context-specific training is a better use of inference compute than current inference-time scaling strategies like producing more thinking tokens.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper argues that long-context failures in static self-attention stem from score dilution: as distractors accumulate, attention mass on the true “needle” vanishes unless the target–distractor logit gap grows with context length. The authors prove a logarithmic margin requirement and show that decoding-based inference-time scaling (e.g., “thinking” tokens) cannot reliably fix this. They propose query-only test-time training (qTTT): perform a single prefill to cache K/V, then run a few lightweight gradient steps updating only the query projections over short spans while reusing the KV cache.

### Strengths
1. The paper proposes a compute-aware design. The proposed prefill-once, KV-cache reuse, and FLOP matching to thinking tokens make for a fair comparison and a practical recipe.

2. The benchmarks in the paper are sufficient. It evaluates the proposed method across model sizes and multiple long-context benchmarks.

3. The idea is really cute. The inference-time updating the parameters is novel in the community.

### Weaknesses
1. The theoretical analysis is too naive to capture the main motivation. Concretely, it cannot prove that the score dilution is the main reason for the poor performance. 

* For example, whether the poor performance comes from the small number of training samples. Usually, learning more complex abilities, i.e., solving problems with longer context, requires more training samples than learning the simple ability. The poor performance can simply come from the relatively small number of samples. 

* Even we assume that the poor performance comes from the score dilution. The current analysis is oversimplified. For example, Lemma 2.3 requires a $log T$ scaling of the logits. Such can be achieved by RoPE, which is designed for the long-range decay. In addition, adapting RoPE for the proper score decaying is the cornerstone of most long-context papers. The existing analysis just ignores the role of RoPE.

2. The paper claims that the proposed method solves the score dilution problem. However, no evaluation or visualization of the attention scores are presented. 

3. Test-time scaling baseline is missing. The proposed method achieves better performance with more computation. However, no such baseline is included. For example, whether the beam search, BoN achieves the comparable performance with the same budget.

4. Ablation is missing. The learning rate for the test-time training is important. Whether this depends on the context length, context semantic is not known.

### Questions
See the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses a critical limitation of long-context Large Language Models (LLMs): while modern LLMs support context windows of millions of tokens, they often fail to reliably use information buried in long texts. Existing inference-time strategies (e.g., chain-of-thought "thinking tokens") show diminishing returns as context length grows, due to a phenomenon the authors term score dilution—static self-attention cannot sufficiently separate the "target" (relevant information) from "distractor" (irrelevant) tokens, leading to vanishing target probability.

### Strengths
*  It formally introduces "score dilution" to explain long-context LLM failures, turning vague issues (e.g., missed key info) into a quantifiable, solvable problem—filling a gap in prior research that lacked clear theoretical grounding for such limitations.

* The proposed query-only Test-Time Training (qTTT) is innovative in its frugality: it reuses frozen KV caches and only updates query projections, avoiding the high compute of full-model fine-tuning or ineffective "thinking tokens" for long texts.

* qTTT offers a low-overhead, drop-in fix for real-world long-context use cases (code analysis, EHR review), and its "score dilution" framework guides future research on improving long-context LLMs beyond incremental tweaks.

### Weaknesses
* While it highlights compute efficiency, it does not measure inference latency (critical for production) when qTTT is added—leaving unclear if its small compute overhead translates to acceptable delays for time-sensitive tasks (e.g., real-time code debugging).

* It does not explore how qTTT performs with noisy or low-quality long texts (e.g., unstructured logs, messy code), where distractors are more prevalent—limiting understanding of its robustness beyond clean benchmark datasets.

### Questions
Does this paper provide a performance comparison with alternative test-time scaling methods, given an equivalent computational budget?

### Soundness
2

### Presentation
2

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
The author proposes a test-time learning method for long context handling with ICL examples.

### Strengths
- Clever idea to learn only the test time "decoder", not the "encoder"
- Extremely strong performance improvement

### Weaknesses
- Training required (during decoding)
  - No detailed efficiency study has happened.
- No large model is tested

### Questions
- What if training the whole model from scratch using this method? (Including encoder, meta learning approach)
- Why do you need to update the query parameters? No need to finetune the MLP?
- How can we serve different query weight parameters in a real-world serving framework, such as vLLM?
  - This could be challenging due to the CUDA graph capturing.

### Soundness
3

### Presentation
4

### Contribution
3
