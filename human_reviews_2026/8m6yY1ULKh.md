# AUTOTRITON: Automatic Triton Programming with Reinforcement Learning in LLMs

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
Kernel development in deep learning requires optimizing computational units across hardware while balancing memory management, parallelism, and hardware-specific optimizations through extensive empirical tuning. Although domain-specific languages like Triton simplify GPU programming by abstracting low-level details, developers must still manually tune critical parameters such as tile sizes and memory access patterns through iterative experimentation, creating substantial barriers to optimal performance and wider adoption. In this work, we introduce AUTOTRITON, the first model dedicated to Triton programming powered by reinforcement learning (RL). AUTOTRITON performs supervised fine-tuning (SFT) to be equipped with essential Triton programming expertise using a high-quality data gathering pipeline, and conducts RL with Group Relative Policy Optimization (GRPO) algorithm, combining a rule-based reward and an execution-based reward to further improve Triton programming ability, sequentially. Experiments across five evaluation channels of TritonBench and KernelBench illustrate that our 8B model AUTOTRITON achieves performance comparable to mainstream large models, including GPT-5 and DeepSeek-R1-0528. Further experimental analysis demonstrates the crucial role of each module within AUTOTRITON, including the SFT stage, the RL stage, and the reward design strategy. These findings underscore the promise of RL for automatically generating high-performance kernels, and since high-performance kernels are core components of AI systems, this breakthrough establishes an important foundation for building more efficient AI systems. The model and code will be available at Github.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
AutoTriton proposes reinforcement learning (RL)-based framework to refine LLMs for triton kernel generation. Authors first collect and curate the data from various sources on Github, torch.compile, and synthetic generations using an LLM. Generated data points are validated for correctness and correct kernels are retained in the dataset. Authors initially perform supervised finetuning to enable model understand Triton language. Then RL finetuning is performed using GRPO method with verifiable rewards to enable refine the model generations. Authors have employed rule-based rewards with execution correctness component. Authors have observed and addressed reward hacking via rule-based detection of a key attribute in triton programming. Authors demonstrate the effectiveness of their approach on TritonBench-G and T and Kernelbench benchmarks. Though the fraction of correct kerrnel generations are lower via AutoTriton, speedup obtained post generations is quite on-par with other methods.

### Strengths
- Extensive data collection methodology for obtaining datapoints with wide variety.
- torch.compile data is usually ambiguous and authors have developed an approach to extract relevant code from it.
- Supervised finetuning to establish base knowledge followed by RL finetuning with verifiable rewards.
- Address reward hacking with rule-based identification of key attribute in triton programming.
- Speedup obtained by generated kernels is on par with that of frontier models.

### Weaknesses
- There have been several reported issues with TritonBench benchmark (https://github.com/thunlp/TritonBench/issues/7 and other sources). It is not clear if authors have addressed these issues in their experimentation. Authors are encouraged to discuss in detail how these issues were addressed. Without these fixes the results on TritonBench benchmark cannot be trusted.
- Some of the artifacts of TritonBench issues are observed in Table 1 results: Call and Exec accuracy are the same or very close for most models. TritonBench uses fewer and poor unit tests for Exec accuracy. So this data cannot be directly trusted.
- Though the paper shows good enough content, it is presented more as a technical report rather than a scientific study that discusses in details fundamental reasoning behind their choices in experimentation. For example, why was GRPO used for refinement? There are variants of DPO with ranked data that can be used for preferential training with the generated data instead. 
- Lines 257-259: As authors admit here, use of poor and sparse reward function in RL finetuning will not result in performance improvement of generated kernels. If performance improvement is not THE goal, then there is no incentive to generate kernels and spend so much energy in training/inference/generation.
- Rather than discussing single speedup number, e.g. fast_1 or fast_2, studying and understanding the speedup distribution usually as a function of difficulty level sheds more light on efficacy of  an approach.
- KernelBench discussion is poorly presented. It is not clear (unless one goes over the kernelbench pull mentioned) whether authors are referring to CUDA or Triton version of kernelbench. 
- Assuming KernelBench Triton version: the performance on AutoTriton is quite poor in terms of number of correct generated kernels and speedup/performance improvements. (Table 3)

### Questions
- Do results in Table 1 still hold after fixing the TritonBench issues?
- Can authors provide with a case study with "aha" moment case study on at least one kernel that shows interesting reasoning and code generation by AutoTriton?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper “AUTOTRITON: Automatic Triton Programming with Reinforcement Learning in LLMs” proposes AUTOTRITON, an RL-enhanced LLM specialized for Triton kernel programming.
The model is built upon Seed-Coder-8B and trained in two stages:
(1) Supervised Fine-Tuning (SFT) with a curated pipeline collecting and validating PyTorch–Triton kernel pairs, and
(2) Reinforcement Learning (RL) using the Group Relative Policy Optimization (GRPO) algorithm with execution- and rule-based rewards.
Experiments on TRITONBENCH and KERNELBENCH show that AUTOTRITON achieves performance comparable to GPT-5 and DeepSeek-R1 despite having only 8B parameters.

### Strengths
S1: Well-designed pipeline: The paper presents a systematic end-to-end data pipeline for high-quality Triton kernel collection and verification.
S2: Quantitative validation of RL gains: Clear ablation results show consistent improvement of RL over SFT-only baselines (Tables 1 and 2).
S3: Detailed experimental setup: Evaluation protocols and hyperparameters are thoroughly described, ensuring reproducibility.

### Weaknesses
W1: Limited novelty: Prior works (e.g., AI CUDA Engineer) already apply RL or agentic loops to CUDA kernel optimization. The main contribution appears to be applying similar ideas to Triton rather than introducing new RL methodology.
W2: Incomplete reward design: The RL reward focuses on functional correctness but does not explicitly optimize runtime speed, as the authors themselves note.
W3: Unclear difference from CUDA approaches: The paper lacks deeper analysis of what makes Triton-specific optimization fundamentally distinct from CUDA.
W4: Limited practical validation: The GitHub-derived tasks remain challenging, with relatively low success rates, suggesting real-world readiness is still limited.

### Questions
Q1: Beyond language differences, what are the key algorithmic or representational distinctions between AUTOTRITON and RL-based CUDA agents?
Q2: How effectively does the model internalize Triton-specific constraints (e.g., tiling, memory layout) during RL training?
Q3: What are the primary obstacles to integrating performance-guided rewards in future work?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper the authors propose an RL method that automatically generates a set of training samples to allow an LLM to learn how to program kernels using the Triton DSL. By using a combination of supervised fine training and RL they demonstrate that the Triton kernel produced by an 8B foundation model can be substantially improved in terms of both correctness and performance. One of the core contributions is the automatic generation of the SFT training data using two processes: instruction-guided LLM distillation and compilation with LLM-enhanced refinement. Using both processes, they show that this training set is sufficient for the LLM to internalize the structure of Triton programs and, with the modification of the reward function, encourage the LLM to produce legitimately optimized Triton kernels and avoid reward hacking. The authors claims are supported by the notable performance of AutoTriton, compared to other contemporary code generators and LLMs.

### Strengths
- The process of automatically generating a training dataset to support both the SFT and RL phases is a great direction to pursue. The SFT and RL results on the KernelBench suite demonstrate the effectiveness of the approach to impact kernel generation.
- The decomposition of the gains into those achieved through SFT and RL is interesting and acknowledges the strengths of both approaches used independently and in tandem.
- The results illustrate that although AutoTriton is based on an 8B that it can compete with Triton code generated by much larger models.
- Reward hacking seems to be especially detrimental for code generation since it may be hard to detect without inspecting the result. The author's proposal to augment the reward function was somewhat effective in ensuring Triton code was actually generated, not just wrappers.

### Weaknesses
- Although Triton is a useful DSL it wasn't clear to me why the techniques wouldn't be equally applicable to several programming languages that are of interest to the ML community.
- While the comparisons with other LLMs do support the author's claims, they compare a fine-tuned model with a model that is not tuned for Triton coding. From that perspective, I would expect AutoTriton to do better than these baseline models.
- Outside of additions to support disuade the model from reward hacking, the RL formulation and context seem rather standard. This is not a bad thing, but it detracts from a bit of the novelty.
- As the author noted, it would be interesting once support for generating a dataset to enable performance-guided training.

### Questions
It's not clear to me what is meant by collecting more high-quality training samples to support performance-guiding training. This was mentioned several times throughout the paper as an issue. Does this mean that although the samples collected are functional, they do not demonstrate the, in some sense, "optimal" way to implement a given kernel within the Triton language?

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
This paper presents AutoTriton, an automatic code generation and optimization framework that uses large language models (LLMs) to produce high-performance GPU kernels in Triton—a high-level programming language for deep learning acceleration. The goal is to automate kernel development while approaching the performance of expert-written implementations.

AutoTriton integrates a multi-stage LLM-guided pipeline to generate Triton kernels directly from high-level PyTorch functions. The framework iteratively refines the generated kernels by collecting runtime performance metrics and feeding them back into the LLM for self-improvement. The system automatically explores tile sizes, memory layouts, and parallelism strategies, learning from failed executions and performance bottlenecks. Experiments show that AutoTriton achieves up to 93.8% of the performance of hand-tuned expert kernels, outperforming prior LLM-based baselines such as AlphaTriton and OpenAI Codex-based methods, while significantly reducing human engineering effort.

### Strengths
+ It addresses a critical bottleneck in modern AI infrastructure: the manual effort and expertise required to write efficient GPU kernels. The integration of LLMs with feedback-driven autotuning aligns well with current trends in AI-assisted compiler optimization and code synthesis.

+ The framework combines static analysis, runtime profiling, and iterative LLM prompting, forming a tight optimization loop. Its error recovery and retry mechanism allows it to gracefully handle compilation or runtime failures—often a weak point in LLM code generation.

+ The performance feedback module quantitatively evaluates each kernel and guides subsequent refinement iterations, embodying a form of “RL through interaction.”
 
+ AutoTriton achieves near-expert-level performance (90–94% of hand-optimized kernels) across diverse workloads (matrix multiplications, softmax, layer norm, etc.). The benchmark suite covers multiple operator types and shapes, ensuring robustness. The performance improvements over baselines are good: up to 2.3× faster than AlphaTriton and 3.7× faster than naive LLM-generated code).

### Weaknesses
- The system still relies heavily on the LLM’s prompt design and prior exposure to Triton code examples. Although feedback improves results, semantic misunderstandings (e.g., incorrect indexing or boundary handling) remain common in early iterations.

- While practical, the paper lacks a deeper theoretical model for convergence or performance bounds of its iterative improvement process. There’s no formal guarantee that feedback-driven refinement leads to monotonic performance improvement.

- Evaluations are limited to relatively small kernels (≤1024 threads) and structured workloads.

- It is unclear how well AutoTriton generalizes to complex fused operators, multi-kernel pipelines, or non-Triton environments (e.g., CUDA directly).

- The paper primarily benchmarks against LLM-based systems. A comparison with auto-tuning frameworks such as TVM, Ansor, or DeepDSL would clarify whether LLM-guided synthesis outperforms traditional search-based compilers.

- While technically detailed, the paper is dense in several sections (especially §3.2 and §4.1).

### Questions
1) How exactly are performance metrics (e.g., latency, throughput) integrated into the prompt for LLM feedback?
A more concrete example of the textual feedback template would clarify reproducibility.
Are the prompts dynamically adapted per task, or fixed across all benchmarks?

2) How are different failure types (syntax, compilation, runtime, numerical mismatch) detected and prioritized in the retry logic?
Understanding this hierarchy would clarify how efficiently the system learns from failures.

3) How do you ensure that performance feedback (latency, occupancy, etc.) is interpretable by the LLM and leads to actionable improvements? Does the LLM truly “learn” from the feedback or just make stochastic modifications? How many refinement iterations are typically needed before convergence, and how stable are results across runs?

4) Are all kernel executions run on the same hardware and CUDA version to ensure comparability?
Performance on Triton can vary significantly across GPUs; details about environment settings would help confirm fairness.

### Soundness
2

### Presentation
3

### Contribution
2
