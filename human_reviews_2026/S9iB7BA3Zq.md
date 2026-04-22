# Bridging Non-Intrusive Tracing and Fine-Grained Cross-Layer Representations for LLM Inference Diagnosis

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
LLM inference spans the inference engine, compute backend, host operators, and GPU kernels, where asynchrony and concurrency make request-level end-to-end observability and diagnosis challenging. We present $\textbf{Truffld}$, a non-intrusive and cross-layer framework that provides fine-grained representations for diagnosis in large-scale LLM inference. For data collection, $\textbf{Truffld}$ activates NVTX markers and CUPTI callbacks to capture raw events from vertical (intra-node stack execution) and horizontal (cross-node communication) perspectives. We then propose a call-chain merging algorithm that aligns these events on a unified time base and reconstructs a per-request call-chain tree preserving both structural and temporal semantics. For anomaly detection, $\textbf{Truffld}$ adopts a two-stage pipeline. A Gaussian Mixture Model models multi-modal normal behavior and produces calibrated numeric confidences, while a large language model applies structure- and context-aware reasoning to generate step-level decisions and operator-level localization. Experiments on a multi-node GPU cluster running Qwen3-8B inference with both online and offline workloads demonstrate near-perfect step-level detection and superior operator-level performance compared to multiple baselines, with low deployment overhead and no modification to binaries. $\textbf{Truffld}$  provides a practical end-to-end solution for observability and diagnosis in large-scale LLM inference.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The core goal of this work is to provide an improved GPU diagnostic setup for tracing LLM inference events both within and across nodes. The system leverages NVTX and CUPTI to capture detailed execution traces across nodes. These traces are then analyzed using Gaussian models to detect anomalies in the running system.

### Strengths
1. Tracing systems like these are very useful in debugging event tracing issues in practice. It is very hard to analyze multi-node issues and events that could be occurring in more complex LLM serving setups
2. The anomaly detection pipeline does look like an interesting use case/extension on existing NSight.

### Weaknesses
1. The paper talks a lot about the importance of low overhead profiling. In appendix H, they display the overhead to be 10%. In practical serving system and real time serving cases this can be very high. While I appreciate it was provided, it does feel a bit hidden.
2. While I understand it is difficult to acquire larger setups, I worry this system is hard to scale/debug when working on large cluster. The evaluation is on 6 GPUs on 2 nodes, but tracing systems like this have to deal with lot of data. The overhead could scale as cluster size increases.

### Questions
1. Is it possible to run a scalability experiment with a lot more nodes while serving? Possibly via simulation
2. is there anything that can be done to further reduce the overhead? Is the overhead unavoidable?

### Soundness
4

### Presentation
4

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
The paper introduces TRUFFLE, a non-intrusive and fine-grained tracing framework fro diagnosing performance anomalies in large-scale LLM inference systems. TRUFFLD collects raw events through NVTX markers and CUPTI callbacks without modifying binaries, then constructs per-request call-chain trees that preserve both structural and temporal semantics across vertical (intra-node) and horizontal (inter-node) views with a two-stage anomaly detection pipeline. Evaluations on multi-node GPU clusters serving Qwen3-8B with online and offline workloads demonstrate step-level 0.9+ accuracy, better operator-level F1 scores, and low deployment overhead compared with several classical and modern baselines (e.g., Robustlog, LAnoBERT, MAD-GAN).

### Strengths
1. The paper addresses an interesting gap in LLM inference observability through a non-intrusive tracing framework. The use of NVTX and CUPTI enables production deployment without binary modification. 
2. The hybrid approach combining GMM-based probabilistic modeling with LLM reasoning is well-motivated. Dual-stage pipeline provides both calibrated numeric confidences modeling multi-modal normal behavior and combination of structural and semantic constraints. 
3. The evaluation of the paper show practical significance for operators of large-scale LLM serving systems, bridging a gap between low-level tracing tools and high-level diagnosis frameworks.

### Weaknesses
1. The paper is limited by generalization scope and scalability analysis. The evaluation is restricted to a single model (Qwen-8B), inference engine (vLLM), and hardware configuration (dual-node, 6× A40 GPUs). This narrow scope limits claims about generalizability to other LLMs (different sizes, architectures), frameworks (TensorRT-LLM, TGI), or hardware platforms (H100s, TPUs, heterogeneous clusters). 
2. I hold doubt on how reliable the artificial anomalies are to evaluate TRUFFLE on naturally occurring production incidents or long-term operational traces. Authors should provide stronger evidence to show that TRUFFLD captures real-world failure modes. 
3. The key design decisions of the paper lack empirical justification. The GMM stage does not compare alternative density models (kernel density estimation, normalizing flows) or centrality definitions. The LLM stage employs hardcoded thresholds (S ≤ 1 for horizontal, 10% for vertical) without sensitivity analysis or ablation of model choices. No comparison against simpler rule-based fusion methods is provided, leaving unclear whether the added complexity and cost of LLM reasoning is justified over interpretable heuristics.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

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
The paper proposes a non-intrusive tracing that has low overhead. Also, the paper claims that the approach has fine-grained cross-layer representations, where the cross-layer refers to the multiple layers in inference engine, compute backend, host operators, and device operators. Proposed work TRUFFLD uses NVTX and CUPTI to gather execution traces collecting events from all the above stacks. It merges "within-node" and "coss-node" events per-request call-chain trees. This makes it easier to understand both what and when the operators are running.
For anomaly detection the paper employs a two-stage approach. (1) Gaussian Mixture Model (GMM) and (2) LLM reasoning. (1) provides numeric confidence for each operator instance (with self-time, CUDA runtime/driver time and counts, kernel counts and totals, approximate bytes moved, stream–overlap ratios, and communication size and world size.) and (2) reads structured summaries of each step along with the semantic context to produce the final step-level abnormality decisions and operator-level localization.
The paper tests on top of vLLM serving Qwen-8B across 6-GPU dual-node cluster and uses a dataset of 3264 step-level traces from online and offline workloads. It compares against several classical, supervised and log-based baselines.

### Strengths
* Non-intrusive and Zero-modification seems like a good point from an engineering perspective. Also, that can really improve the overall debuggability in production.
* Cross-layer observation is a good contribution considering the growing complexity of the LLM SW stack.

### Weaknesses
* It seems that the work is only tested on a limited case. It is difficult to understand how well this can be generalized to other environments and models.
* The paper is rather weak on detail about how it handles the log. I see that there are some examples in the appendix, but the description are still too shallow for the work to be either reproducible by others or to be built upon.
* It is difficult to understand why the specific combination of GMM and LLM was used. The paper seems to be rather shallow on the insights it provides.

### Questions
1. It seems that the work is only tested on a limited case. It is difficult to understand how well this can be generalized to other environments and models. What if the HW changes? What if the framework changes? What if the models change?
2. How would this work if multiple models are being served simultaneously.

Minor.
It would really help if the main paper content has some outline about the usecase and the demonstration of the input and output (at least in small scale). This would help the readers benefit more from the work.

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
4

### Summary
The paper proposes TRUFFLD, a non-intrusive, cross-layer tracing and diagnosis framework for large-scale LLM inference. It leverages NVTX markers and CUPTI callbacks to collect execution events across engine, backend, host, and device layers without modifying binaries. A call-chain merging algorithm reconstructs per-request trees aligned on a unified time base. For anomaly detection, TRUFFLD combines a Gaussian Mixture Model for numeric confidence scoring with an LLM-based reasoning stage for context-aware localization. Experiments on a multi-node GPU cluster serving Qwen-8B demonstrate near-perfect step-level detection and strong operator-level performance compared to classical and log-based baselines, with low overhead.

### Strengths
Clear motivation and relevance: Addresses a critical gap in request-level, end-to-end observability for LLM inference under high concurrency.

Non-intrusive design: Uses NVTX and CUPTI without modifying binaries, minimizing deployment overhead.
Fine-grained representation: Reconstructs per-request call-chain trees that preserve structural and temporal semantics.
Two-stage anomaly detection: Combines statistical modeling with LLM reasoning for robust and interpretable diagnosis.
Strong empirical results: Outperforms multiple baselines on both step-level and operator-level metrics; overhead analysis shows practical feasibility.
Comprehensive evaluation: Includes fault injection across software, CUDA, hardware, and communication layers.

### Weaknesses
Combination of existing techniques: The approach is mainly based on a combination of existing GMM and LLM methods.
Limited generalization evidence: Experiments focus on Qwen-8B with vLLM; applicability to other models or inference engines is not demonstrated.
LLM reasoning details: Prompt design and schema enforcement are described, but robustness to unseen anomalies and cost implications of LLM inference could be discussed more thoroughly.
Scalability concerns: While overhead is reported, the impact on large clusters or multi-tenant environments is unclear.
Interpretability trade-offs: The reasoning stage relies on textual context; failure cases or misdiagnoses are not deeply analyzed.
Ablation studies: Missing analysis of the contribution of each component (e.g., GMM vs. LLM reasoning) to overall performance.

### Questions
What are new in the proposed use of GMM and LLM?
How much does each component contribute to the anomaly detection improvement?
How does TRUFFLD scale when deployed on clusters with hundreds of GPUs and thousands of concurrent requests?
How sensitive is the anomaly detection pipeline to the choice of GMM hyperparameters?
How sensitive is the anomaly detection pipeline to the choice of LLM?
Could you provide examples of failure cases where TRUFFLD misdiagnoses anomalies and explain why?

### Soundness
3

### Presentation
4

### Contribution
3
