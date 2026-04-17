# Prima.cpp: Fast 30-70B LLM Inference on Heterogeneous and Low-Resource Home Clusters

- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
On-device inference offers privacy, offline use, and instant response, but consumer hardware restricts large language models (LLMs) to low throughput and capability. To overcome this challenge, we present prima.cpp, a distributed on-device inference system that runs 30-70B LLMs on consumer home clusters with mixed CPUs/GPUs, insufficient RAM/VRAM, slow disks, Wi-Fi links, and heterogeneous OSs. We introduce pipelined-ring parallelism (PRP) to overlap disk I/O with compute and communication, and address the prefetch-release conflict in mmap-based offloading. We further propose Halda, a heterogeneity-aware scheduler that co-optimizes per-device CPU/GPU workloads and device selection under RAM/VRAM constraints. On four consumer home devices, a 70B model reaches 674 ms/token TPOT with <6% memory pressure, and a 32B model with speculative decoding achieves 26 tokens/s. Compared with llama.cpp, exo, and dllama, our proposed prima.cpp achieves 5-17× lower TPOT, supports fine-grained model sizes from 8B to 70B, ensures broader cross-OS and quantization compatibility, and remains OOM-free, while also being Wi-Fi tolerant, privacy-preserving, and hardware-independent. The code is available at https://gitee.com/zonghang-li/prima.cpp.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces prima.cpp, a distributed inference framework designed to run 30–70B parameter language models on heterogeneous and memory-constrained “home clusters.” The core ideas are (i) pipelined-ring parallelism (PRP), a variant of pipeline parallelism that repeatedly circulates activations around a ring of devices to overlap disk I/O with computation, and (ii) Halda, a heterogeneity-aware scheduler formulated as an integer linear program (ILP) that assigns per-device layer windows and CPU/GPU splits while enforcing RAM/VRAM constraints. Experiments on a Wi-Fi connected cluster of four to six consumer devices compare prima.cpp against llama.cpp, exo, and dllama, reporting large speedups in time-per-output-token (TPOT) and time-to-first-token (TTFT). The authors also present ablations on Halda and prefetching and discuss memory pressure considerations.

### Strengths
Originality: The paper tackles a timely and underexplored setting—distributed on-device inference across heterogeneous, low-cost consumer hardware. Integrating mmap-based offloading with pipeline parallelism and heterogeneous scheduling is an interesting direction. The attempt to capture OS-level memory reclamation in the scheduling formulation is a thoughtful touch.

Quality: The implementation effort is substantial (∼20k LOC) and the evaluation spans a range of model sizes up to 70B parameters, which is rare for “home cluster” work. The ablation that removes Halda highlights the importance of scheduling in this regime, lending credibility to the design.

Clarity: The writing is generally clear, with useful figures (e.g., Fig. 1 showing PRP) and tabular summaries of baselines and experimental results. The high-level algorithmic flow of Halda (Algorithm 1) is easy to follow.

Significance: Demonstrating sub-second TPOT for 70B models on commodity hardware, if validated, would be impactful for privacy-preserving and offline LLM deployment. The emphasis on low memory pressure addresses an important practical concern for end users.

### Weaknesses
Soundness of PRP claims: PRP is positioned as a key innovation, yet the mechanistic explanation for resolving the “prefetch-release conflict” is largely qualitative. There is no formal or empirical breakdown of the time saved per component (disk vs. compute vs. communication), and the homogeneous-cluster experiment in Fig. 2 lacks realism (uniform 8-core CPUs and SSDs) relative to the heterogeneous setting the paper targets. It remains unclear how much of the reported TPOT gains stem from PRP versus simply better scheduling or quantization choices.

Baseline fairness and completeness: The experimental section focuses exclusively on latency metrics. There is no evaluation of model quality (perplexity, task accuracy) under the aggressive quantization and speculative decoding used. Comparisons with exo and dllama are limited: exo “runs on D1–D3” but the authors disable D4 due to root access requirements, while prima.cpp does leverage D4; dllama is measured only on the 8B model and declared OOM later without empirical numbers. Additionally, llama.cpp is constrained to a single device (D3), even though its distributed variants or multiple instances could also use the other devices, making the comparison potentially unfair.

Ablations insufficient for causal attribution: The ablation “prima.cpp (w/o halda)” simply reuses exo’s partitioning rule but retains the rest of the system. There is no attempt to isolate Halda’s scheduling quality from the PRP machinery, nor to test alternative schedulers in a controlled manner. Similarly, the prefetching ablation shows 9–17% improvements, yet TPOT reductions of up to 17× over llama.cpp are claimed; the gap between these numbers is not reconciled.

Scalability and generality concerns: The entire evaluation uses a single anonymized cluster with modest Wi-Fi bandwidth. There is no analysis of how prima.cpp scales with more than six devices, higher-latency networks, or weaker hardware (e.g., devices without SSDs). The claim that Halda “selects” devices is not backed by experiments on a large pool where many must be excluded. Furthermore, the scheduler assumes access to per-device disk throughput and available memory, but it is unclear how those are measured or kept up to date in deployment.

Clarity issues and missing details: Several important implementation choices are omitted: handling of activation checkpointing or KV cache distribution, the cost of coordinating speculative decoding across devices, and how OS-specific behaviors (e.g., macOS page cache policies) are detected at runtime. The paper also references Appendix sections for critical arguments (e.g., Appendix A.1 for the prefetch-release conflict) rather than summarizing key empirical evidence in the main text.

### Questions
PRP vs. standard PP: Can you quantify the incremental benefit of multi-round PRP over single-round pipeline parallelism on the heterogeneous cluster, with and without Halda? A per-stage latency breakdown (compute, disk, communication) would help attribute the gains to specific mechanisms.

Baseline configuration fairness: Why is llama.cpp restricted to a single device while prima.cpp uses all available devices? Have you evaluated llama.cpp with manual layer partitioning (e.g., via tensor or pipeline parallel modes) to assess whether the comparison is fair?

Quality metrics under quantization: Do the aggressive quantization formats (e.g., Q4K, IQ1) and speculative decoding degrade output quality? Please report perplexity or downstream task accuracy to demonstrate that prima.cpp preserves model performance relative to the baselines.

Device selection robustness: Halda is claimed to automatically drop slow devices. Could you provide experiments on a larger heterogeneous pool (≥10 devices) showing how many devices are selected, how often the assignment changes, and the resulting impact on TPOT?

Convergence and overhead of Halda: What is the typical number of iterations before Halda converges, and how long does the scheduling phase take relative to inference time? Are there scenarios where the ILP fails to find a feasible solution or oscillates between assignments?

Network sensitivity: How does prima.cpp perform under higher-latency or lower-bandwidth Wi-Fi (e.g., 50–100 ms RTT or 100 Mbps throughput)? Since home networks can be noisy, understanding the tolerance to adverse conditions is important for practical deployment.

Memory pressure measurement methodology: Table 5 reports “memory pressure” as percentage reductions in mem_available. Can you detail the measurement interval (sampling frequency), the baseline state (idle system vs. with other apps running), and whether fluctuations during inference were observed?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents prima.cpp, a distributed on-device inference framework enabling large language models (30–70B) to run efficiently on heterogeneous consumer clusters with limited resources. It introduces pipelined-ring parallelism (PRP) to overlap disk I/O, computation, and communication, and Halda, a heterogeneity-aware scheduler optimizing CPU/GPU utilization under RAM/VRAM constraints. Experiments on four home devices demonstrate strong performance—674 ms/token for 70B and 26 tokens/s for 32B models.

### Strengths
1. Efficient large-model deployment under limited VRAM/RAM.  
2. Clear, well-structured, and easy-to-follow paper.  
3. Solid, industrial-grade implementation with strong reproducibility.

### Weaknesses
1. Not show comparison with capable baselines, like SpecExec [1], which reaches 100-250ms/token TPOT. 

2. Leveraging the Disk to offload (because of frequent writing and erasing) will cause damage to the hardware, which is not discussed in the limitation.

3. Lack of an ablation study of the impact of workload, such as context length, frequency of requests, etc.




[1] https://github.com/Infini-AI-Lab/UMbreLLa

### Questions
As those in weakness.

1. Could you provide insights on energy consumption when running prima.cpp?
2. How is the framework implemented for different hardware backends?

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
3

### Summary
The paper introduces prima.cpp which is a distributed on-device inference system to run LLMs on home clusters connected via Wi-Fi. It enables fast, private, offline-inference using heterogeneous CPUs/GPUs and limited memory and slow disks.

The paper extends pipeline parallelism to pipelined-ring parallelism where the devices form a ring, each processing a layer window of the model across multiple rounds per token. The work also includes Halda which partitions the model layers across devices considering CPU/GPU spec. It formulates the problem as a Integer Linear Fractional Program to minimize TPOT.

### Strengths
* On-device AI is an important topic considering the privacy concerns with regard to LLM inference that may handle personal sensitive data.
* Formulation of resource mapping seems to be neat and seems to provide a nice performance.
* Pipelined Ring Parallelism seems to be a little extension that seems to be effective.
* Code is open-sourced https://anonymous.4open.science/r/prima-cpp

### Weaknesses
* Experiments seem to be limited to a few devices and it does not really show the generalizability of the work.
* Only single inference request is presented whereas real use-cases may really want multi-user (multi-request) workloads. In other words, it seems that the overall serving related discussions are missing in the paper.
* Extending from previous point, it seems to lack comparisons to vLLM, SGLang, ... which are very widely used frameworks in both academia and industry.
* Power consumption and energy efficiency OR cost vs cloud inferences seem to be missing.
* More analysis of the performance breakdown would really help.
* Details about how it is designed is not really presented in the paper, really limiting the amount of insight that the readers can get from reading the paper.
* Discussion about how the work relates to various optimization approaches such as quantization, pruning are missing.

### Questions
* How well do the experiments generalize beyond the few tested devices?
* How does the system handle multi-user or multi-request workloads common in real applications?
* Why are comparisons to frameworks like vLLM and SGLang missing?
* What is the power consumption or energy efficiency compared to cloud inference costs?
* Can the authors provide a detailed performance breakdown showing compute, communication, and disk I/O contributions?
* Can more system design and implementation details be provided to give readers better insight?
* How does the proposed approach relate to optimization techniques such as quantization and pruning?

I like the paper overall. However, it has a lot of missing parts hence the score (4). I would be open to reevaluating.

### Soundness
3

### Presentation
3

### Contribution
2
