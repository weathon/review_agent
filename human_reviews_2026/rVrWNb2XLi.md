# SpecOffload: Unlocking Latent GPU Capacity for LLM Inference on Resource-Constrained Devices

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Efficient LLM inference on resource-constrained devices (i.e., PCs with a single commodity GPU) presents significant challenges in compute and memory utilization. Due to limited GPU memory, existing systems offload model weights to CPU memory, incurring substantial I/O overhead between the CPU and GPU. This leads to two major inefficiencies: (1) GPU cores are underutilized, often remaining idle while waiting for data to be loaded; and (2) GPU memory has a low impact on performance, as reducing its capacity has minimal effect on overall throughput. In this paper, we propose SpecOffload, a high-throughput inference engine that embeds speculative decoding into offloading. Our key idea is to unlock latent GPU resources for storing and executing a draft model used for speculative decoding, thus accelerating inference at near-zero additional cost. To support this, we carefully orchestrate the interleaved execution of target and draft models in speculative decoding within the offloading pipeline, and propose a planner to manage tensor placement and select optimal parameters. Compared with the best baseline, SpecOffload improves GPU core utilization by 4.49× and boosts inference throughput by 2.54×. Anonymous repo is at https://anonymous.4open.science/r/SpecOffload-F3F2/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes SpecOffload for memory-restrained LLM inference scenario, where the large model has to be offloaded to CPU or disk for restrained GPU memory. It introduces speculative decoding during offloading where GPU idles, which involves 3 main modifications: batch interleave of drafting and verificaition in decoding phase, adaptive tensor planner to optimize offloading i/o latency, and ParaSpec planner to maximize throughput with given configurations. It also utilizes CPU for attention computation of the base model for more efficient pipelining. Experiments show that SpecOffload improves GPU core utilization by 4.49× and boosts inference throughput by 2.36×.

### Strengths
The topic is highly related to practical inference scenario. LLM inference on resource-restrained hardware is an important direction.

The observation of ‘Marginal utility of GPU memory’ in the introduction is inspiring: When model size far exceeds GPU memory, if a portion of memory is further ‘restricted’ from being used, the marginal throughput degradation will be marginal because there have been too much offloading operations. This guarantees the effectiveness of inserting speculation into offloading stage, as the further occupied memory causes marginal degradation to the verification but enables speculation at GPU and hence boosts overall performances.

The demonstration figures (Fig.3, 4) are clear and informative, making the main design easily understood.

### Weaknesses
My main concern is about attention and FFN pipeline of the base model in sec4.1.2: 

(1)	The paper said that, the attention computation is on CPU and FFN is on GPU, and the FFN is offloaded to GPU while attention being computed, forming a pipeline. However, the computation of attention and FFN has interleaved data dependencies: Attention Layer i+1 needs the result of FFN Layer i, and FFN Layer i+1 needs the result of Attention Layer i+1, etc.. Therefore, in my opinion, it is impossible to first run all the attention layers on CPU, then all FFN layers on GPU. 

(2)	The attention computation on CPU is parallelized with the drafting on GPU. However, running attention computation of a much larger model on CPU would be much slower than running a full smaller model on GPU (in principle), so I think there would be a huge pipeline bubble bottlenecked by the CPU computation. This could limit the effectiveness of the proposed interleaved pipeline.

Other concerns:

The compared baselines are weak. SpecOffload shows significant acceleration compared to all the mentioned baselines, while they all involve no speculative decoding, only offloading optimizations. As a result, the speedup ratios may be attributed to adding speculative decoding, rather than the effectiveness of the proposed method itself.

The contribution is relatively incremental. The contribution is mainly an engineering integration of speculative decoding into offloading pipelines. It builds upon well-established components (speculative decoding, offload scheduling, pipeline batch scheduling), making the novelty relatively incremental.

### Questions
1. Could the authors provide detailed computation workflow, to clarify how the attention and FFN layers are parallelized?
2. How do the authors mitigate the potential pipeline bubble caused by performing attention computation on the CPU while the draft model runs on the GPU? Could they provide more detailed quantitative profiling data of each stage, to show the overall pipeline efficiency?
3. Since all compared baselines lack speculative decoding, how can we disentangle the gains from speculative decoding itself versus the proposed integration?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the inefficiency of LLM inference on single-GPU or resource-limited devices, where offloading model weights from GPU to CPU memory causes severe GPU under-utilization and minimal performance scaling with additional GPU memory. The authors propose SpecOffload, a system that integrates speculative decoding directly into the offloading pipeline to reclaim idle GPU compute and unused GPU memory.
The key idea is to use a lightweight draft model that runs during GPU idle periods (while waiting for CPU–GPU data transfer) and to store it in “low-yield” GPU memory that would otherwise be wasted. SpecOffload introduces three core techniques: (1) a dual-batch interleaved pipeline allowing draft and target models to run concurrently; (2) adaptive tensor placement across GPU/CPU/disk memory tiers; and (3) a ParaSpec planner that automatically tunes batch and token parameters under GPU memory constraints.
Experiments on Mixtral and LLaMA models show up to 4.5× GPU utilization and 2.36× throughput gains over strong offloading baselines (FlexGen, DeepSpeed-Inference, Fiddler), demonstrating strong practicality on consumer GPUs.

### Strengths
1. The paper identifies an under-explored inefficiency — GPU idleness during offloading — and proposes to exploit this latent capacity through speculative decoding.
2. The dual-batch interleaved execution and adaptive tensor placement form a coherent system that effectively overlaps CPU compute, GPU compute, and I/O, where speculative decoding is repurposed not merely as a speed-up technique but as a way to hide I/O latency, which is novel in the offloading context.
3. The paper is well-written for a systems paper.

### Weaknesses
1. The speculative decoding itself and its integration into offloading is not unique. That being said, the novelty lies primarily in system integration and scheduling.
2. All experiments are single-GPU; no exploration of distributed or multi-GPU extension.

### Questions
1. The ParaSpec Planner seems tuned for specific hardware (RTX 4090). Could the authors elaborate on how much manual profiling or calibration is needed to port SpecOffload to a different GPU architecture.
2. In practice, how should users choose the draft model size relative to the target model? Nowadays many choose to use Eagle to reduce the spec execution time but it might not be too much helpful for an offloading system like SpecOffload. How should SpecOffload adapt in terms of scheduling policy?

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
The paper makes an observation that there are underutilization of GPU cores due to the memory communication between CPU and GPU. Also, the paper claims that adding more GPU has a minimal effect on overall throughput as the model parameters are too large to reside in GPU memory permanently. introduces SpecOffload which is aims to utilize the idle GPU time to run lightweight draft model for speculative decoding. It claims that storing the draft model in the GPU memory would have negligible impact on performance. The paper uses a dual-batch interleaved design enables the target model and draft model to run concurrently, increasing GPU resource utilization.

### Strengths
* Efficient LLM inference is important considering its increasing adoption in many applications. As such the paper is working on a good direction.
* The paper aims to better utilize GPUs which is important and seems to get some gains.
* Amalgamation of Offloading and Speculative Decoding seems to be a clear nice followup for the Offloading works and Speculative Decoding works.

### Weaknesses
* The paper seems to be rather shallow on the experiment depth (please refer to the questions for details). This significantly limits its generalizability.

### Questions
* If the draft model is really small. How does this compare to the works that tries to leverage CPU resources for some part of the LLM compute. How would this perform if CPU's AMX units are used for draft model inference and GPU for the target model inference?
    * Na, Seonjin, et al. "FlexInfer: Flexible LLM Inference with CPU Computations." Eighth Conference on Machine Learning and Systems. 2025
* The paper mentions that speculative decoding is not always reliable and that its acceleration effect is limited if none of the draft tokens are accepted. However, it does not provide an in-depth analysis of the draft token acceptance rate, which is a critical factor for the system's performance. Can the authors add more experimental data on how this rate varies across different models, datasets, or hardware environments?
* While the paper states the planner achieves 93.7% of optimal policy performance on average, it does not provide detailed information on its computational cost, runtime overhead, or how it performs in more varied or complex scenarios. Can you provide more details? Also, can you provide more details as to how this can be generalized to different CPU-GPU combinations + models?
* It seems as the work is only focused on increasing the throughput in single-user scenario. How would this be extended to multi-user / dynamic serving scenarios?
* A.3.3 Table 13 seems to suggest that there is only 53% improvement for even double VRAM. Could you elaborate. It would also help if utilization data are provided?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes an LLM inference system that combines offloading and speculative decoding. It executes the draft model concurrently on the idle GPU with the offloading and CPU computation of the target model. It also provides modeling of the execution pipeline and planner to search for the configuration. Evaluation show that this work has superior performance than the related offloading-based LLM inference systems.

### Strengths
1. This paper focuses on an important topic of efficient LLM serving.
2. It proposes a clever method to fill the draft model execution into the bubbles of offloading of the target model.

### Weaknesses
1. The planner part is not clear enough, nor necessary enough.
2. Some analysis and claim are not solid enough.

### Questions
1. Why put the Attention computation of the target model on the CPU? Is it only to save the GPU memory for the KV cache? Note Attention is usually memory bound and the GPU has higher bandwidth than the CPU.
2. The results in Figure 2 lack of some details. How is the increased GPU memory used? For example, one method can be making some portion of the weight persistent in the increased GPU memory and thus the performance can improve gradually.
3. In Figure 4, why the execution of the Draft model is longer than the FFN of the target model? (Note for moderate sequence length, the execution of the target model on the GPU should be dominated by its FFN.)
4. In Section 4.3, how the equation 1 is solved? Besides, equation 4 seems not accurate. For example, if $T_{draft}$ is very long, then the FFN of the target model will need to wait for the end of the draft model (according to Figure 4). In this case, the time will not be the max of the two, but seems $T_{draft} + T_{target-ffn}$.
5. It is not clear what is the search space of the planner. It seems everything is deterministic, or can be defined very easy. So the planner seems not necessary.

### Soundness
3

### Presentation
2

### Contribution
3
