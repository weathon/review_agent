# ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models

- Decision: Accept (Oral)
- Scores: 6, 6, 4, 8

## Abstract
The long-output context generation of large reasoning models enables extended chain of thought (CoT) but also drives rapid growth of the key–value (KV) cache, quickly overwhelming GPU memory. To address this challenge, we propose ThinKV, a thought-adaptive KV cache compression framework. ThinKV is based on the observation that attention sparsity reveals distinct thought types with varying importance within the CoT. It applies a hybrid quantization–eviction strategy, assigning token precision by thought importance and progressively evicting tokens from less critical thoughts as reasoning trajectories evolve. Furthermore, to implement ThinKV, we design a kernel that extends PagedAttention to enable efficient reuse of evicted tokens' memory slots, eliminating compaction overheads. Extensive experiments on DeepSeek-R1-Distill, GPT-OSS, and NVIDIA AceReason across mathematics and coding benchmarks show that ThinKV achieves near-lossless accuracy with less than 5% of the original KV cache, while improving performance with up to 5.8x higher inference throughput over SoTA baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel approach to KV Cache management specifically designed for Large Reasoning Models (LRMs). Recognizing that these models generate extremely long sequences, leading to significant KV Cache demands, the authors address that existing methods often overlook the unique characteristics of reasoning models. The core contribution lies on observations about attention sparsity variations across different reasoning stages. Based on these findings, the proposed method, ThinKV, dynamically manages the KV Cache through a combination of quantization and eviction strategies. This approach aims to simultaneously reduce latency and improve peak throughput, enhancing the overall efficiency of LRM inference.

### Strengths
* **A Novel Design.** The paper introduces a thought-adaptive framework that maps attention sparsity to reasoning stages, enabling hybrid mixed-precision quantization and adaptive eviction. This design achieves fine-grained, semantically informed KV cache compression beyond traditional token-level compressions.
* **A Comprehensive Implementation.** ThinKV is supported by a strong system-level realization, featuring gather-free memory reuse via an extension from vLLM mechanism and fused dequantization–matrix multiplication inspired by KiVi. The paper further provides detailed kernel-level breakdowns and overhead analyses, demonstrating solid engineering rigor.
* **A Strong Result.** Across math and coding benchmarks, ThinKV consistently achieves higher accuracy under different memory budgets and higher throughput compared to R-KV, showcasing both algorithmic and system-level efficiency gains.

### Weaknesses
* **Lack of Latency Comparison.** Although Section 5 reports throughput under large-batch settings, the paper does not present a detailed latency analysis. For large reasoning models (LRMs), metrics especially TPOT (Time per Output Token) are critical, as generated sequence will be long and interactive responsiveness matters. A fine-grained latency breakdown across decoding steps or batch sizes would strengthen the system-level evaluation.
* **Limited on long input tasks.** ThinKV is evaluated on datasets whose input prompts average only a few hundred tokens. However, some practical workloads (e.g., SWE-bench [1]) involve very long input contexts. Since ThinKV does not compress the prefill cache, its effectiveness and efficiency under long-input regimes remain unclear.



[1] Jimenez, Carlos E., et al. "SWE-bench: Can Language Models Resolve Real-world Github Issues?." ICLR. 2024.

### Questions
* **Accuracy–Efficiency Trade-off with Non-Reasoning Modes.**
   While ThinKV improves accuracy relative to other compression baselines, it still introduces slight degradation compared to Full Cache. Recent models such as the Qwen3 series support *optional reasoning (thinking vs. direct answer)* modes. It would be valuable to compare ThinKV-enabled reasoning against no-thinking (full attention) decoding, to quantify whether the added reasoning—and compression—yields net gains in both accuracy and efficiency.
* **Sensitivity of different Thought Types to Quantization Precision.**
   Section 3.2.2 discusses precision settings on different reasoning stages. Prior work such as QAQ [2] demonstrates that Key and Value Cache have different sensitivity on precision. A similar quantization-sensitivity study across thought types would clarify the boundary of compression on different reasoning stages.

[2] Dong, Shichen, et al. "Qaq: Quality adaptive quantization for llm kv cache." arXiv preprint arXiv:2403.04643 (2024).

### Soundness
3

### Presentation
3

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
This paper presents a combination of strategies to reduce the KV cache size during inference with a decoder transformer.

The method combines quantization and deletion of blocks of tokens in the KV cache based on a category which is inferred from the attention sparsity. The authors also propose an efficient implementation derived from PageAttention.

The resulting gain in speed for a given accuracy level beats existing methods.

### Strengths
The problem addressed by the authors is key in the deployment of large language models, and more generally reducing memory footprint is a central issue: Lower memory means possibly larger batches, larger model dimension, longer contexts, and longer / more c-o-t, which ultimately translate into smarter models.

### Weaknesses
This is an highly engineering paper. The method proposed does not derive from a new principle nor rely on formal results or quantitative analysis. There are some motivating experiments that are interesting, but the bulk of the method is a recipe with lot of hyperparameters.

For the presentation, I encourage the authors to change the scaling of fig 1 and have both axes starting from 0, the current framing is IMO misleading. 

I am doubtful the inclusions of figures respect ICLR's style, and captions could be more detailed (e.g. fig 3, what data set?)

### Questions
Some questions, in no specific order:

- The clear separation of the three classes is very impressive in Fig 3, and Fig 13 provides other examples, for different models, but how does it hold on other datasets? Are all this graphs from AIME?

- There are many hyperparameters in 3.2 and 3.3, how were they chosen?

- The method is compared to other KV cache quantization or eviction, but what are the sota methods overall? What about other types of methods (MLA, YOCO)? It would be great to give a clearer sense of the pareto front accuracy vs. memory footprint across methods.

### Soundness
3

### Presentation
3

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
The paper proposes a hybrid quantization+eviction methods for reasoning model, achieve high compression ratio with minimum accuracy degradation.

### Strengths
1. The paper combines eviction with quantization, and shows that different tokens should quantized differently.
2. Experiments are run on latest reasoning models and challenging benchmarks.

### Weaknesses
1. A very unfortunate thing for post-training efficiency technique of reasoning models is the inflation of generation length. Quantization is servere as mentioned in the paper. Sparsity attention can also lead to inflation but tends to be smaller. So it is meaningless to show metrics like **throughputs** or **compression ratio** in the paper as they hide this effect. Similar to max-batch size. You will never get these high batch size as 1. larger batch size comes from larger compression ratio-> 2. larger compression ratio means more inflation of kv cache-> 3. larger kv cache per seq means lower batch size. **So from my side, the only reasonable metric for reasoning efficiency evaluation is how long you take to solve an benchmark e2e such as AIME and LCB using same GPU setting.** And this should be normalized to standard serving framework with dynamic batching like vllm or sglang so that it reflects the ablility of your kv compression method. 
2. The paper writing is not clear. It's difficult to follow. I suggest the methodology just focuses on describing what the algorithm looks like without so many definition, observation and notation. Even with so many paragraphs, I have not been convinced and motivated. For example, what if future reasoning model reasons in hidden states instead of explicit outputing tokens? Is there still going to be R, T, E? I am not sure, but I believe attention sparsity and quantization still exisit.

### Questions
1. How many times of averaged score do you use for one reasoning question?
2. Can you explain more on block_size in flg 9e?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
ThinKV is a KV-cache compression framework for large reasoning models (LRMs) that exploits attention sparsity to decompose chain-of-thought (CoT) into three semantic segments: reasoning (R), execution (E), and transition (T). It applies hybrid compression, including quantization, eviction and system co-design. Across DeepSeek-R1, GPT-OSS, AceReason (math/code benchmarks), ThinKV achieves <5% KV-cache with near-lossless accuracy and 5.8× throughput against previous SOTA methods.

### Strengths
- The overall quality of this paper is pretty good. The authors demonstrate deep insights in their observation and analysis of the problem and propose an innovative algorithmic design and operator implementation. 

- The overall presentation and expression are excellent.

- The paper demonstrates strong empirical results. It sustains >95% accuracy with 20–40× compression on AIME/LiveCodeBench, while 3× batch size and 5.8× throughput gains over R-KV/H2O.

### Weaknesses
1. The current experimental setup shows a relatively large accuracy drop. It is recommended that the authors investigate how much acceleration can still be achieved when the accuracy drop is controlled within 1%, to better demonstrate the stability and practical value of the proposed method.  

2. The paper appears to lack **ablation studies** related to the operator design. The authors are encouraged to add relevant experiments to analyze the impact of different operator designs and usages on overall throughput and accuracy, thereby enhancing the completeness and persuasiveness of the paper.

### Questions
1. The experimental results show that the proposed method achieves a significant speedup. However, according to Appendix E.2, the implementation is based on **Transformers**. It is suggested that the authors further evaluate the throughput performance on industrial deployment frameworks such as **vLLM** or **SGLang**, to verify the acceleration effect in practical scenarios and compare it with other methods (e.g., the **Double Sparse** method in **SGLang**).  

2. It is suggested to include additional KV cache compression methods beyond **H2O** and **RKV** in the experiments, so as to better demonstrate the superiority of the proposed approach.

### Soundness
4

### Presentation
4

### Contribution
3
