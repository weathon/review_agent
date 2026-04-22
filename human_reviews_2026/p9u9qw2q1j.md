# Fact or Hallucination? An Entropy-Based Framework for Attention-Wise Usable Information in LLMs

- Avg Score: 4.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4

## Abstract
Large language models (LLMs) often generate confident yet inaccurate outputs, posing serious risks in safety-critical applications. Existing hallucination detection methods typically rely on final-layer logits or post-hoc textual checks, which can obscure the rich semantic signals encoded across model layers. Thus, we propose **Shapley NEAR** (Norm-basEd Attention-wise usable infoRmation), a principled, entropy-based attribution framework grounded in Shapley values that assigns a confidence score indicating whether an LLM output is hallucinatory. Unlike prior approaches, Shapley NEAR decomposes attention-driven information flow across all layers and heads of the model, where higher scores correspond to lower hallucination risk. It further distinguishes between two hallucination types: parametric hallucinations, caused by the model’s pre-trained knowledge overriding the context, and context-induced hallucinations, where misleading context fragments spuriously reduce uncertainty. To mitigate parametric hallucinations, we introduce a test-time head clipping technique that prunes attention heads contributing to overconfident, context-agnostic outputs. Empirical results in four QA benchmarks (CoQA, QuAC, SQuAD, and TriviaQA), using Qwen2.5-3B, LLaMA3.1-8B, and OPT-6.7B, demonstrate that Shapley NEAR outperforms strong baselines, without requiring additional training, prompting, or architectural modifications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a new method for hallucination detection of LLM outputs.
The method uses the usable information provided from one layer to the next and also allows for interpretability by distinguishing between hallucinations stemming from parametric knowledge or inputs.
The method outperforms relevant baselines on a variety of datasets.

### Strengths
- The results are quite impressive with clear improvements over recent and relevant baselines on a variety of benchmarks.

- The authors also did a good job ablating their design choices and giving intuitions into the behavior of the method.

- I find the method itself quite well-motivated and intuitive to understand.

### Weaknesses
- The notation and writing are at times a bit confusing. At times it was hard to follow what variables (e.g. q, s, x), also in combination with subscripts, all stand for and some quantities like p where overloaded often which made it a bit hard to follow. Perhaps the writing of Sec. 3 could be made a bit more clearly and intuitive. There is also caligraphic V for model (which is often used for vocabulary in NLP) and V for vocabulary which is a bit confusing, though admittedly the authors inherited this confusion from (caligraphic) V-information.

- The presentation could also be improved a bit, some figures like Fig. 3 are quite small and hard to read. 

- The method is a bit slow compared to baselines (but can still be run in realtime, faster than 1s per query) which is a bit hidden in the appendix but could be moved to the main paper for transparency.

### Questions
- The overloading of p around Eq. 3 is a bit confusing, could you maybe explain p_i a bit more clearly, are q and x perhaps missing on the RHS?

- L247: "Equation equation" -> perhaps you are mixing the usage of \ref and \cref? In the paragraph from 265-269 there is only the number as a hyperlink.

- If s_i is a sentence and the score of Eq. 7 is summed over the i does it make sense to call it "sentence-level"? Shouldn't it be passage-level?

- L297+298: is it correct that the inequality is not proven rigorously by the authors but an empirical "law"? If not you might want to add a proof and make it clear that it indeed holds in general (or in a special case).

### Soundness
3

### Presentation
2

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
This paper introduces Shapley NEAR (Norm-basEd Attention-wise usable infoRmation), a hallucination detection method for large language models. The core idea is to measure entropy reduction at each attention head across all transformer layers when conditioning on context, then use Shapley values to fairly attribute this information gain to individual context sentences. The hypothesis is: if the mean Shapley-attributed information gain exceeds a threshold (Q1), the generated answer is trustworthy; otherwise, it is likely a hallucination. The method distinguishes between parametric hallucinations (model's pretrained knowledge overrides context) and context-induced hallucinations (misleading context spuriously reduces uncertainty). Experiments on four QA datasets (CoQA, QuAC, SQuAD, TriviaQA) using three models (Qwen2.5-3B, LLaMA3.1-8B, OPT-6.7B) show AUROC improvements of 8-13% over the strongest baseline (INSIDE).

### Strengths
1. It is the first to decompose information flow across all attention heads/layers using Shapley attribution, and is theoretically grounded with provable bounds.
2. It consistently outperforms 6 baselines across 12 settings, and works without fine-tuning or prompt engineering.
3. It provides sentence-level attribution and distinguishes parametric vs context-induced hallucinations.

### Weaknesses
1. The method requires 27-52 seconds per 100 samples versus 3-10 seconds for baselines (Table 24)—a 10-50× overhead due to computing entropy across all layers/heads for M=50 sampled coalitions. More fundamentally, for factual QA, why not use external search APIs + NLI verification? The paper doesn't justify why internal entropy computation is superior to external fact-checking, which is already standard in production systems and likely faster/more reliable.

2. The method averages Shapley values across all sentences (Eq. 7). This fails when a long document has only one relevant sentence:
• 100 sentences: answer sentence IG=10.0, others=0.05 → mean=(10.0+4.95)/100=0.15 (severely diluted)
• 5 sentences: same answer → mean=10.2/5=2.04 (13× higher for identical quality)
Max or top-K makes more sense: one highly informative sentence should suffice to indicate non-hallucination. Table 5 shows answer sentences have 23× higher IG, yet this signal is diluted by averaging. The paper provides no justification and critically no ablation comparing aggregation strategies. This likely works only because tested datasets have short contexts (~10-20 sentences), masking the issue for longer documents.

### Questions
1. Why Internal Detection Over External Search?
Given that external fact-checking (search APIs + NLI) is already standard in production and likely faster, what specific advantage justifies the 10-50× computational overhead of internal entropy analysis?
2. Performance on Sparse/Noisy Context?
Tested datasets have dense, relevant contexts. How does mean aggregation (Eq. 7) perform when only 1-2 sentences are relevant among 50-100 (e.g., RAG retrieval, long documents)? Does irrelevant content dilute the signal?

### Soundness
2

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
3

### Summary
This paper introduces a inner-metric based hallucination detection method, it calculate the information gain after each attention block.

### Strengths
1. Leverages internal model representations for interpretability: The method directly analyzes attention-layer outputs across all transformer layers and heads, providing fine-grained mechanistic insights into how models process contextual information, rather than relying solely on final-layer logits.
2. Clear and accessible methodology: The approach is conceptually straightforward, using well-established techniques (entropy, Shapley values) in a principled way, making it easy to understand, implement, and apply to new models without modifications.

### Weaknesses
1. Aggregation Choice Unjustified: The paper does not justify using mean aggregation (Eq. 7) over max or other functions. Since detecting hallucination depends on whether sufficient context exists, max aggregation might be more appropriate. No ablation study compares aggregation strategies.
2. Computational Overhead: Computing entropy across all L×H attention heads for M=50 coalitions per query is expensive (O(M×L×H×V) complexity), limiting scalability to real-time applications, especially for large models.

### Questions
1. Computational Cost: The method requires O(M×L×H×V) computation per query. For LLaMA-70B, this is expensive. What is the actual latency per query? How does performance degrade with fewer coalitions (M<50)? Have you explored optimizations like attention caching or layer pruning to reduce costs?
2. Aggregation Choice: Why average Shapley scores across sentences rather than taking the maximum? Since hallucination detection depends on whether any sentence provides sufficient information, wouldn't max aggregation better capture this? Please provide ablation results comparing mean vs. max vs. top-k aggregation strategies.

### Soundness
2

### Presentation
3

### Contribution
2
