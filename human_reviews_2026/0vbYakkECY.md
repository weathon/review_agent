# Draft-based Approximate Inference for LLMs

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
Optimizing inference for long-context large language models (LLMs) is increasingly important due to the quadratic compute and linear memory cost of Transformers. Existing approximate inference methods, including key-value (KV) cache dropping, sparse attention, and prompt compression, typically rely on coarse predictions of token or KV pair importance. We unify and extend recent work by introducing a framework for approximate LLM inference that leverages small draft models to more accurately predict token and KV pair importance. We provide novel theoretical and empirical analyses justifying lookahead-based importance estimation techniques. Within this framework, we present: (i) **SpecKV**, the first method to use lookahead with a small draft model to enable precise KV cache dropping; (ii) **SpecPC**, which leverages draft model attention activations to identify and discard less important prompt tokens; and (iii) **SpecKV-PC**, a cascaded compression strategy combining both techniques. Extensive experiments on long-context benchmarks demonstrate that our methods consistently achieve higher accuracy than existing baselines while retaining the same efficiency gains in memory usage, latency, and throughput.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Draft-based Approximate Inference (DAI), a framework that leverages a small draft model to generate a short sequence of lookahead tokens, which are then used to estimate token or KV importance in large language models (LLMs).
Under this framework, the authors instantiate two methods:

SpecKV — for lookahead-guided KV cache dropping

SpecPC — for prompt compression

The key idea is to use the draft model’s predicted future queries to obtain a more accurate estimate of which tokens will be attended to in future steps, enabling more effective prefill-time cache compression.
Empirical results on RULER and LongBench show that the proposed methods outperform prior works such as SnapKV, LAQ++, and SpecPrefill, with reduced memory and latency.
The paper also provides theoretical bounds relating draft embedding quality to attention approximation error.

### Strengths
1.  Clean and implementable idea:
Extends speculative decoding to approximate inference in a conceptually neat way, requiring minimal system changes.

2.  Strong empirical results:
On both synthetic and long-context benchmarks, SpecKV yields clear gains (1–2 points improvement) over existing KV compression baselines, with ~40–50% memory savings.

3.  Solid theoretical justification:
Provides error bounds connecting the draft model’s embedding deviation to importance estimation error, filling a theoretical gap that earlier methods (e.g., SnapKV, LAQ++) lacked.

4.  System efficiency:
The overhead of lookahead generation is minimal (<10% in prefill), while decoding latency and memory are significantly reduced.

### Weaknesses
1. Limited conceptual novelty:
The main idea—using a smaller model to predict future attention patterns—is a natural extension of speculative decoding (FastGen, Medusa) and prior KV-dropping methods (SnapKV, LAQ++).
The improvement lies mainly in integration and theoretical refinement, not in introducing a new paradigm.
2. Marginal accuracy gains:
Improvements on benchmarks are moderate (1–2%), suggesting the practical benefit mainly comes from efficiency rather than substantial modeling advance.

### Questions
How sensitive are the results to $n_{lookahead}$? The ablation in Appendix E.4 is informative, but further scaling analysis would be valuable.

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
This paper proposes a framework called draft-based approximate inference to optimize the efficiency of LLM in long context decoding. Existing methods, which approximate KV-cache discarding, sparse attention, and prompt compression, usually rely on rough prediction when estimating the importance of tokens or KV pairs. In contrast, the core idea of this paper is to use a small and cheap "draft model" to generate approximate future output (lookahead), so as to more accurately predict the importance of the current token or KV pair.

In this paper, two methods are introduced: 

1. SpecKV, whose goal is to use the "look ahead" capability of the draft model to more accurately discard unimportant parts in the kV cache, and combine sparse pre-filling to improve efficiency.

2. SpecPC, whose goal is to use the draft model to directly determine which tokens in the input prompt are not important, and compress the prompt before sending it to the target model.


A large number of experiments on the benchmark of rule and longbench show that speckv and specpc continuously achieve higher accuracy than the existing baseline methods under the fixed KV cache or prompt size limit.

### Strengths
1. This paper provides theoretical (Theorems 1 and 2) and experimental evidence to support the effectiveness of the "look ahead" based importance estimation. 

2. This framework unifies and extends the idea of using approximate future information to improve token importance estimation.

3. The paper claims that its method achieves the current state-of-the-art accuracy in the long context benchmark under the constraint of a fixed KV cache or prompt size.

4. Even if a weak draft model is used, this paper can perform well, far exceeding the performance of the draft model itself. Also, using a better draft model will further improve performance.

### Weaknesses
1. The core of the whole framework is to use the draft model to approximate the behavior of the target model. Both theoretical analysis (Theorem 1) and experimental results (Fig. 10) show that the accuracy of the draft model directly affects the final performance. If a draft model that is small (low overhead) and similar enough to the target model (high accuracy) cannot be found, the effect of SpecKV and SpecPC may be compromised.

2. Compared with methods such as SnapKV, which only pre-fills and compresses the target model once, the pre-filling steps of SpecKV are more complex, and the calculation cost is higher. Although the paper claims that the overall delay is reduced, this is mainly the benefit of the decoding stage.

3. This paper also introduces a new memory occupation: it needs to load and store the weight of the draft model. Although this overhead is fixed and does not increase with the sequence length as KV cache, it is still an additional memory burden compared with methods that do not require a draft model.

4. The main idea is similar to speculative decoding and previous LAQ++. It is more likely a technical extension.


I have a borderline opinion on this paper and hope to see the rebuttal and other reviewers' comments.

### Questions
How does SpecKV reduce the peak memory than LAQ++? In Algorithm 1, the target model still needs to store all the KV cache of the input sequence x and draft output y_draft.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes an effective "Draft-based Approximate Inference" framework that leverages a draft model for lookahead to more accurately estimate the importance of tokens and KV pairs, thereby improving long-context approximate inference. The authors further introduce SpecKV and SpecPC for KV cache dropping and prompt compression, respectively, and provide a thorough error analysis. A comprehensive set of experiments further demonstrates the effectiveness and application potential of the proposed method.

### Strengths
1. By using a draft model to estimate the importance of tokens in the KV cache and prompt, the method achieves strong performance under controllable complexity.
2. This work provide a clear theoretical analysis, demonstrating how embedding errors influence KV importance estimation errors (Theorem 1), and how output approximation under RIP or more general assumptions can upper bound attention approximation errors (Theorems 2 and 3).
3. The experiments on LongBench and RULER Benchmarks are solid, with results that convincingly demonstrate the effectiveness of the proposed methods.

### Weaknesses
1. For different input embeddings, are there any limitations to the applicability of Theorem 2?
2. There appear to be some typo errors in Table 2.

### Questions
1. I am curious whether using different types of draft models and target models would affect token selection (e.g., Qwen-2.5-0.5B + Llama-3-70B). Does this imply that draft models must be selected from the same model family as the target model?
2. As shown in Figure 2, the benefits gained from the draft model vary across different tasks. Any explaination?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Draft-based Approximate Inference, leveraging lightweight draft models for lookahead-based token/KV importance estimation. It introduces SpecKV for KV cache dropping with sparse prefill and SpecPC for prompt compression, both supported by theoretical proofs and extensive experiments showing superior accuracy–efficiency trade-offs over baselines.

### Strengths
1. First integration of draft-model lookahead into KV dropping and prompt compression, with theoretical justification. 
2. Strong empirical gains across diverse benchmarks, models, and compression budgets.  
3. Clear motivation, concise algorithms, and well-presented results.

### Weaknesses
1. Lacks analysis of importance score differences with/without lookahead
2. Limited breakdown of the latency trade-off
3. Unclear whether SpecKV and SpecPC can be effectively combined.

### Questions
1. The core premise is that approximate future information improves token/KV importance estimation. Could the authors present a quantitative comparison of importance score distributions obtained with lookahead versus current-token-only methods? If the distributions differ substantially, would SpecKV’s advantage over methods like SnapKV diminish when output length greatly exceeds $n_{lookahead}$?  
2. The acceleration analysis is limited. Since SpecKV and SpecPC incur additional draft-model overhead, can the authors provide a detailed latency breakdown (draft inference, dense/sparse prefill, decoding) for varying input/output lengths?  
3. Given that SpecKV employs sparse prefill, has its speed been compared directly to optimized prefill approaches such as MInference [1]?  
4. Can SpecKV and SpecPC be combined in a single pipeline, and if so, could the authors include ablation studies showing the individual and combined contributions to speed, memory reduction, and accuracy?

[1] MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention

### Soundness
2

### Presentation
3

### Contribution
3
