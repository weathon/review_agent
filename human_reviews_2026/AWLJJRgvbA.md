# A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
The rapid adoption of large language models (LLMs) has created significant challenges for efficient inference at scale. Unlike traditional workloads, LLM inference is constrained by both computation and the memory overhead of key–value (KV) caching, which accelerates decoding but quickly exhausts GPU memory. 
In this paper, we introduce the first queueing-theoretic framework that explicitly incorporates both computation and GPU memory constraints into the analysis of LLM inference. Based on this framework, we derive rigorous stability and instability conditions that determine whether an LLM inference service can sustain incoming demand without unbounded queue growth. This result offers a powerful tool for system deployment, potentially addressing the core challenge of GPU provisioning. By combining an estimated request arrival rate with our derived stable service rate, operators can calculate the necessary cluster size to avoid both costly over-purchasing and performance-violating under-provisioning.
We further validate our theoretical predictions through extensive experiments in real GPU production environments. Our results show that the predicted stability conditions are highly accurate, with deviations typically within 10%.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
Serving LLMs is an important and timely problem. One of the critical questions systems engineers face is determining the maximum load a system can serve while maintaining reasonable per-request waiting times. In practice, this is often verified empirically by incrementally increasing the workload and identifying the "inflation point" on the goodput curve. The authors propose a more theoretically-grounded method to determine this inflation point, beyond which the system becomes unstable. The authors model this behavior for a system using FCFS scheduling with a chunked prefill batching strategy, deriving a closed-form expression for the stability condition. Their experiments show that their model achieves high accuracy, with a reported error rate of up to 10%.

### Strengths
1. The authors target an extremely important and practical problem. From a systems designer's perspective, having a closed-form solution for system stability is a valuable tool that could significantly inform autoscaling strategies.

2. The choice to model available memory using the number of fittable KV cache tokens, rather than raw memory capacity, is really nice. This abstraction simplifies the analysis and makes the problem formulation more intuitive.

3. Despite a few minor grammatical issues, the paper is well-written and enjoyable to read. It presents a clear narrative, making the work's value easy to understand.

### Weaknesses
My primary concern stems from the assumption on line 184 (and Figure 1) that "batch processing time remains relatively stable in practice." I find this claim questionable, as it forms the foundation for the rest of the analysis.

While the batch processing time of LLMs is _predictable_, this does not mean it is _stable_. Figure 1 itself illustrates this point: although many requests (~90%) have a runtime near 40 ms, the distribution has a significant long tail, with runtimes extending to 700 ms. This long tail, which is characteristic of LLM serving, contradicts the notion of a stable runtime.

This runtime variance is well-understood to originate from the computationally expensive prefill (prompt processing) phase, which is orders of magnitude more intensive than the decoding phase. Furthermore, the prefill phase for a request $i$ may complete in $s_i/\hat{s}$ chunks, whereas the decoding phase requires $o_i$ iterations which is way more than number of chunks. This fundamental difference in required computation and iteration count between the two phases is the primary driver of the long-tailed runtime distribution. Therefore, while _we can estimate_ the runtime for prefill and decoding batches separately, we _cannot_ assume a single, stable runtime for an "average" batch.

This concern leads me to question the choice of using a synthetic dataset primarily with a 1:2 PD (prefill-to-decode) ratio. As Appendix Figure 4 shows, as the prefill size increases (PD ratio 4:1), the batch execution time becomes more dominated by the prefill stage, and the 90th percentile runtime diverges significantly from the median. A 1:2 ratio dataset minimizes this effect, which may artificially boost the model's accuracy.

I suspect the current model is accurate for decode-dominated workloads (i.e., requests with many output tokens), as this scenario aligns better with the "stable runtime" assumption. However, for prefill-dominated workloads, the model's estimation will likely fail because the single $\bar{b}$ term cannot capture the bimodal nature of the batch runtime. This is likely why the model's worst-case error occurs on the 2:1 PD ratio dataset, suggesting the model does not generalize well.

A potential solution, which I believe would be relatively straightforward, is to split the $\bar{b}$ term. By modeling prefill batch time and decoding batch time as two distinct, estimable parameters, the subsequent derivation would likely yield a more robust model capable of handling general-purpose workloads.

### Questions
1. As detailed in the "Weaknesses" section, could you elaborate on the expected performance and error of the model on prefill-dominated workloads, such as LongBench dataset?

2. Could you please clarify the experimental setup described around line 376, specifically regarding how the "time interval" for the experiment was measured and used?

3. How is the $\mu_{GPU}$ (stability bound) computed? One would expect this to be an arrival rate at which the system remains stable. However, Table 1 lists $\mu_{GPU}$ as 3.387 for the 1:1 dataset, while Figure 5(b) (which seems to correspond to this experiment, though it is not explicitly linked) shows an exploding median wait time of 177 seconds at this rate. A 177-second median wait time does not represent a stable system. Could you clarify this apparent contradiction?

4. How might the model need to be adapted for different scheduling strategies, such as Shortest Job First (SJF)? I am not expecting a full derivation, but rather a high-level discussion on which components of the model would need to be revisited.

### Soundness
4

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
This work presents a queueing-theoretic framework to analyze the stability of large language model (LLM) serving under memory constraints. Based on the sequence input lengths, sequence output lengths, and average batch processing time, it constitutes a performance model when chunked prefill and first-come-first-serve (FCFS) scheduling are employed. Then, it proves the conditions (w.r.t. request arrival rate) when the system will be overloaded or stable. Numerical experiments by LLM serving show that the estimated conditions are within 10% from the ground truth.

### Strengths
S1. The paper is among the first ones to rigorously analyze the stability in LLM serving. 

S2. The derived stability conditions

### Weaknesses
W1. Limited practical impact.

The analysis relies on unrealistic assumptions that the input and output lengths follow some distribution (evaluated via uniform distributions in experiments). However, in practice, the length distributions are dynamic across time (e.g., due to the request type differences across working hours and non-working hours). Such distributions cannot be known in advance, making the assessment of stability conditions infeasible.

---

W2. Over-simplification in the performance model.

The performance model makes an assumption that batch processing time remains relatively stable during the serving. However, there are two issues:
- It is known that for sequences with different lengths, the decoding speed varies substantially (e.g., see the [FlashDecoding report](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) or this [vLLM issue](https://github.com/vllm-project/vllm/issues/11286)). The assumption of stable batch processing time is suitable only when the sequence lengths are within a relatively narrow range. However, when there is a mixture of long and short sequences, the assumption fails.
- If I understand correctly, a chunked prefill step is also viewed as one batch, which implies that we need to meticulously tune the chunk size in order to ensure that the time cost of one chunked prefill step is similar to that of one decoding step. However, this may not be true in practice.

Furthermore, it lacks discussion/comparison with the performance models used in existing LLM simulators based on profiling and cost modelling (e.g., Vidur [1] and Scratchpad [2]).

[1] Vidur: A Large-Scale Simulation Framework for LLM Inference. \
[2] Scratchpad. https://github.com/eth-easl/Scratchpad/tree/main/tools/simulator 

---

W3. The experiments are conducted with a simplified setup (synthetic, constant workloads). And several key metrics like percentile latency and service level objective (SLO) are not considered.

### Questions
Please refer to the weaknesses.

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
2

### Summary
This paper presents a queueing theory framework for LLM serving that, for the first time, accounts for GPU memory constraints in addition to computation. It provides a tool for analyzing LLM serving systems, and its accuracy is validated through actual measurements, showing deviations within 10%.

### Strengths
- This is the first work to consider KV cache constraints in the queueing-theoretic analysis of LLM serving systems.
- The theoretical predictions closely match actual measurements obtained with vLLM.

### Weaknesses
- The assumption about the essential supremum (ess sup {s + o} << M) may be too strong. Section 5.1 states that $M$ is 131,000 for the 8B model, which would be smaller for larger models. Considering that context lengths can typically reach tens of thousands of tokens [1, 2], this assumption may need to be relaxed for a more realistic analysis.
- The systems considered in this paper may be overly simplified and not fully representative of real-world LLM serving environments. See the questions below for more details.

### Questions
The theoretical framework could be improved by modeling more realistic settings by considering the following points:

- What if the KV cache is not released after a request finishes? For chatbot or agent applications, it is common to store KV caches in some way for future interactions [3, 4, 5].
- What if the per-iteration latency ($\bar{b}$) exhibits high variance? I think this happens when the arrival patterns of requests are skewed.
- What if more realistic request length distributions are considered, instead of fixed PD ratios?

Please note that I am not an expert in queueing theory, so I am not entirely certain whether the above points are necessary for the paper's acceptance.

### References 

- [1] https://github.com/Azure/AzurePublicDataset
- [2] https://arxiv.org/abs/2407.00079
- [3] https://arxiv.org/abs/2311.04934
- [4] https://arxiv.org/abs/2408.12757
- [5] https://arxiv.org/abs/2510.09665

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies LLM inference as a queueing system where both compute and memory are limiting resources. The authors model prompt arrivals as a Poisson process with per-request prompt length $s$ and decode length $o$, and formalize how each request consumes KV cache over time during the prefill and decode phases. They define a cumulative memory demand function $g(s,o)$ for each request and use it to derive a closed-form expression for an effective processing rate $\mu$ as a function of the memory capacity of the GPU, the joint distribution of prompt/response lengths, the scheduler, and per-batch serving time. The main theoretical contributions are stability conditions for the LLM serving systems. The paper then validates these stability predictions on real A100 hardware running Llama-3 under vLLM-style continuous batching and chunked prefill, across several prompt/decode length ratios. The predicted stable service rate $\mu$ matches the measured throughput within 10%, suggesting this analysis can be used for capacity planning.

### Strengths
- The authors propose a novel perspective for the stability analysis of LLM inference systems considering memory constraints. Prior work largely considered compute/throughput as the bottleneck; here, KV cache usage is more explicitly modeled as a limiting resource over the lifetime of a request.
- I think that the theoretical results are simplistic and actionable; they provide basic rules for practitioners while deploying LLM inference systems.
- The authors provide comprehensive numerical results that cover single and multi-GPU settings, and they show that their estimates work.

### Weaknesses
- The model treats per-batch processing time as constant, justified empirically by showing 80% of the batches have identical processing times. I am not sure how the heterogeneity of the batches (what percentage is prefill vs decoding) affects the batch processing time for any given batch.
- The experiments force a fixed prefill/decode ratio by setting decode length to a deterministic function of the prompt length. That is convenient for control but not representative of real chat workloads, where decode length can vary and can exceed prompt length by orders of magnitude. It would also be interesting to see a robustness analysis or extension when the output lengths are estimated using the prompt information.

### Questions
- The service-rate definition Eq. (1) uses the total lifetime memory area $g(s,o)$, but the GPU memory constraints need to be satisfied instantaneously during the inference. Could the authors clarify how this aggregate memory-time simplification captures momentary memory saturation, and under what conditions the area-based estimate may under or over-estimate the real service-rate?

- The related work section could benefit from citing [1], which also analyzes queueing-based throughput models for LLM inference. This would help clarify how the present formulation differs in assumptions or theoretical guarantees.
[1] Guldogan, Ozgur, et al. "Multi-bin batching for increasing LLM inference throughput." arXiv preprint arXiv:2412.04504 (2024).

### Soundness
3

### Presentation
3

### Contribution
3
