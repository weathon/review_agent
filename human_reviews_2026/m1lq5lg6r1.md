# Understanding Efficiency: Quantization, Batching, and Serving Strategies in LLM Energy Use

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 2, 4, 8

## Abstract
Large Language Models (LLMs) are increasingly deployed in production, shifting the burden in terms of computational resources and energy demands from training to inference. While prior work has examined the energy cost of inference per prompt or per token, we highlight how \textbf{system-level design choices} - such as numerical precision, batching strategy, and request scheduling - can lead to orders-of-magnitude differences in energy consumption for the same model. We perform a detailed empirical study of LLM inference energy and latency on NVIDIA H100 GPUs, analyzing the impact of quantization, batch size, and serving configuration (e.g., with Hugging Face’s Text Generation Inference server). Our results reveal that lower-precision formats only yield energy gains in compute-bound regimes; that batching improves energy efficiency, especially in memory-bound phases like decoding; and that structured request timing (arrival shaping) can reduce per-request energy by up to $100\times$. We argue that sustainable LLM deployment depends not only on model internals, but also on the orchestration of the serving stack. Our findings motivate phase-aware energy profiling and system-level optimizations for greener AI services.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a comprehensive empirical review of how system-level factors including quantization, batch size and serving strategies impact energy efficiency of LLMs on H100 GPUs across prefill and decode stages. Authors report that quantization only improves energy efficiency in the compute-bound stage (prefill) and not in memory-bound. They also observe that batching leads to energy efficiency up to an optimal batch size. Overall, they note that sustainable llm deployment requires phase-aware energy profiling and system-level optimization.

### Strengths
- Novel framework focusing on end-to-end serving stack instead of model or prompt-level energy measurements. 
- Authors report results across multiple models and five numerical precisions. Experiments are carefully controlled and code is shared.
- The discussion around memory-bound vs compute-bound transitions, Tensor Core activation, and dequantization overheads demonstrates solid understanding of GPU behavior.
- Testing Hugging Face TGI with varied inter-arrival patterns connects lab results to production relevance. Showing that scheduling alone yields 100× savings is impactful.

### Weaknesses
- Experiments are confined to one single GPU generation. It would be great if these observations can be extrapolated to other hardware setups, including more recent GPU generation.
- The work remains largely empirical. It would be better to include a simple analytical model providing roofline-style analysis.
- CPU, DRAM, and network energy are estimated heuristically via CodeCarbon rather than directly measured, omitting parts of the total system energy footprint.
- The paper does not discuss energy–accuracy trade-offs at lower precisions, which would contextualize the value of aggressive quantization.
- Figures omit standard deviation or confidence intervals, which are important for understanding measurement variance.

### Questions
See weaknesses

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
This paper empirically studies the energy and latency efficiency of LLM inference, examining how numerical precision, batching, and request scheduling affect performance. The authors benchmark several open-source models (Qwen, Mistral, LLaMA) across five precisions on NVIDIA H100 GPUs, measuring energy via CodeCarbon and separating prefill and decode phases. While the experiments are thorough and provide useful system-level insights, the work is largely descriptive with limited novelty or analysis, offering practical observations rather than new methodological contributions.

### Strengths
1. This paper addresses an important problem—the efficiency and energy consumption of LLM inference—which has become increasingly critical as model sizes continue to grow.
2. The authors examine this issue from multiple perspectives, including **quantization**, **batching**, and **serving strategies**, all of which are highly relevant for practical deployment and provide valuable insights for practitioners seeking to better understand and optimize energy use.

### Weaknesses
1. The contributions of this work are rather limited, as most of the findings and conclusions are already known in prior literature.
2. The experimental scope is not comprehensive. While some limitations—such as prompt and output diversity and hardware variety—are acknowledged, these aspects are crucial for a complete understanding of inference efficiency and should not be deferred. Moreover, key factors such as evaluations on larger models, studies on Mixture-of-Experts (MoE) architectures, and comparisons across different inference backends (vllms, sglang, etc.) and serving frameworks are missing.
3. The paper lacks an analysis of the trade-offs between task accuracy and deployment efficiency, which would be valuable for guiding practical decisions.
4. The overall presentation could benefit from a more formal and structured scientific writing style.

### Questions
1. How do 2-bit, 5-bit, and 6-bit quantization affect energy consumption, and how do quantization methods such as GPTQ and AWQ influence those results?
2. What is the impact of FP8 precision on energy efficiency compared to lower-bit integer formats and FP16/BF16?
3. How does LLM pruning (structured and unstructured) affect inference energy use, and what are the trade-offs between pruning ratio, latency, and overall energy savings?
4. How well do the observed energy and latency patterns generalize across different GPU architectures (e.g., A100, H100, H200) given differences in memory bandwidth, tensor core design, and power efficiency?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This submission studies the GPU energy consumption impact of numerous systems-level design choices, such as adopted quantisation scheme, batching strategy and request scheduling, during LLM (inference) serving on H100 GPUs, across a diverse set of small-medium sized models.

### Strengths
- This submission conducts a timely and interesting analysis, on the increasingly important topic of quantifying the energy footprint of AI deployment, through the unique angle of incorporating system-level design aspects in the equation. 
-The results are presented with appropriate breakdowns and normalisations to allow drawing clear conclusions in the examined setup.
- Many of the presented findings are insightful (with some being more expected than others), and the results presented can act as a point of reference for future work, as well as deployment practices, revealing the energy impact of usually underestimated factors such as padding and dequantisation operations.

### Weaknesses
- The main drawback of the presented methodology is the fact that energy consumption analysis is solely focused on GPU operations, ignoring memory transfers to/from GPU memory. This aspect often has decisive impact in the power draw, as well as is notably affected by some of the examined angles such as quantisation. As such, the practical applicability of the reported findings is limited, since the analysis does not seem to capture the whole picture. 
- The presented analysis (e.g. on batching) could be strengthened if the conducted experiments and results included greater variability and breakdown in input/output length combinations, to be representative of different usecases (e.g. chat, translation and summarization demonstrate fundamentally different patterns in this aspect). 
- The presented results would be more conclusive about the key underlying trade-off between energy and latency under different system-design choices, if the latency results from the Appendix were jointly presented (e.g. overlayed) and most-importantly commented in the respective sections of the analysis.

### Questions
Please consider replying in the comments raised in the weaknesses section above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates where LLM inference energy is actually spent by decomposing generation into two phases, prefill and decode, and measuring energy for each across multiple open-weight families on H100 GPUs, including LLaMA, Qwen, and Mistral at different parameter scales. Prefill at the studied context lengths is typically compute-bound, while decode is often memory-bound. This split explains why the same knob can help in one phase and hurt in the other. Lower numeric precision helps mainly in compute-bound prefill, where Tensor Cores improve throughput and reduce energy. In memory-bound decode, weight-only int8 or int4 can increase energy due to extra small dequantization kernels and limited bandwidth gains. For batching, the paper reports energy per token under two normalizations: for input tokens, padding inflates prefill energy as batch size grows; for output tokens, energy falls quickly with batch across phases, roughly logarithmically, since completed sequences drop from the batch and longer kernels amortize overheads. Decode shows a U-shaped curve with an optimum around b=4 in their setup. Finally, the serving stack and traffic timing matter a lot: switching from a naive Transformers server to Hugging Face TGI with continuous batching cuts mean energy per request by about 12.5x, and adding fixed inter-arrival spacing (for example, 500 ms) yields reductions near 100x; similar trends appear at larger scales and on multi-GPU runs.

### Strengths
The paper tells a clear story: if you want to cut inference energy for LLMs, you must reason by phase. By decomposing generation into prefill and decode and normalizing energy by input and output tokens, the authors reveal that these phases live in different regimes (compute vs memory). This framing is simple yet novel, and it immediately explains why a single knob can help in one phase and hurt in the other. It also shifts the conversation from model-only tweaks to the full serving stack, showing that scheduling and batching policies can dominate outcomes.

The evaluation stands out by explaining results through mechanisms rather than surface-level curves. Precision is tested where it matters: lower formats help prefill when Tensor Cores are well utilized, but in decode energy can increase with weight-only int8 or int4 due to dequantization overheads and limited bandwidth relief. The batching study is explicit about normalization choices: energy per input token reveals padding waste in prefill, while energy per output token shows gains from longer kernels and sequences dropping out of the batch. Beyond microbenchmarks, the paper evaluates realistic serving backends and traffic patterns, showing that continuous batching and simple arrival pacing can deliver order-of-magnitude energy savings without altering the model.

### Weaknesses
1. Serving-stack breadth is limited. Results highlight TGI, but there is no controlled comparison vs vLLM or TensorRT-LLM under the same traffic.

2. System scope is GPU-centric, not resource-limited. The study focuses on high-end GPUs (H100) and GPU energy, with little accounting for CPU, or constrained devices.

3. Sequence-length coverage. Sequence length directly shapes which phase dominates (prefill vs decode), how much padding waste you incur, how large and active the KV-cache becomes, and how batching/continuous batching behave. Longer prompts stress prefill compute; longer outputs shift the workload toward memory traffic and cache updates; multi-turn chat effectively increases both over time. The study could benefit from including this in the experiments.

### Questions
- For weaknesses 1 and 2: while the core insights likely transfer across serving stacks and hardware, absent a controlled comparison the extent of portability is unclear. Reproducing key results with the same methodology (identical traffic traces, metrics, and normalizations) on vLLM and TensorRT-LLM, and rerunning a minimal subset on A100 or L40S as well as a CPU-heavier configuration, would materially strengthen the claims. 

- Regarding weakness 3 (sequence-length coverage): the impact of prompt/output length on phase balance, padding waste, KV-cache growth, and batching behavior is central to your claims, so targeted length-controlled experiments would strengthen external validity. Something like bucket results by prompt/output length and reporting phase energy and batch optima would go a long way to improve the study.

### Soundness
4

### Presentation
4

### Contribution
4
