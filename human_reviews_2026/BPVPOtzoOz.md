# NetArena: Dynamic Benchmarks for AI Agents in Network Automation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
As AI agents expand into high-stakes domains like network system operations, evaluating their real-world reliability becomes increasingly critical. However, existing benchmarks risk contamination due to static design, show high statistical variance from limited dataset size, and fail to reflect the complexity of production environments. We present NetArena, a dynamic benchmark generation framework for network applications. NetArena introduces a novel abstraction and unified interface that generalize across diverse tasks, enabling dynamic benchmarking despite the heterogeneity of network workloads. At runtime, users can generate unlimited queries on demand. NetArena integrates with network emulators to measure correctness, safety, and latency during execution. We demonstrate NetArena on three representative applications and find that (1) NetArena significantly improves statistical reliability across AI agents, reducing confidence-interval overlap from 85% to 0, (2) agents achieve only 13–38% average performance (as low as 3%) for large-scale, realistic queries, and (3) it exposes more fine-grained behaviors that static, correctness-only benchmarks miss. NetArena also enables use cases such as SFT and RL fine-tuning on network system tasks. Code is available at https://github.com/Froot-NetSys/NetArena.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
The authors propose a dynamic benchmarking framework named NETARENA, designed to evaluate the performance of LLMs in network system applications. Unlike traditional static benchmarks, NETARENA can dynamically generate unlimited queries and integrates with high-fidelity network emulators to assess correctness, safety, and latency.

### Strengths
1. The paper introduces a dynamic LLM benchmark generation framework specifically for the networking domain, demonstrating clear innovation.
2. Beyond traditional correctness metrics, the benchmark incorporates safety and latency as key evaluation dimensions, which better align with the needs of high-stakes systems.
3. The paper is well written and clearly presented.

### Weaknesses
1. Although the paper defines safety and latency evaluation standards, it lacks explicit quantitative formulas or threshold specifications.
2. The evaluation focuses on three types of network tasks, but broader validation across more diverse scenarios is missing. The authors could further discuss potential directions for future evaluation (additional experiments are not necessary).
3. While correctness, safety, and latency often involve trade-offs, the paper does not provide corresponding quantitative analyses to characterize these relationships.

### Questions
1. Can NETARENA support cross-task generalization testing? For example, can a model trained on routing tasks generalize to microservice policy troubleshooting tasks?
2. In practice, how much time and computational resources are required to complete a full-scale evaluation?
3. Were the dynamically generated natural language task templates reviewed or validated by human experts? How do you ensure consistency between task descriptions and the simulated system states?

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
The paper introduces a framework, NETARENA, for evaluating LLM-based agents on realistic, execution-time network/system tasks. Instead of relying on small, static, possibly contaminated benchmarks, it dynamically generates tasks over a unified state–action interface, runs them in emulators (Mininet/K8s/DC simulator), and scores agents on correctness, safety, and latency. Experiments across several network-style applications show current LLM agents perform much worse in these realistic, dynamic settings than static benchmarks suggest.

### Strengths
- Clear unified state–action abstraction that works across three concrete network apps (DC capacity planning, Mininet routing, K8s policy troubleshooting), not just a toy demo.
- Dynamic, on-demand query generation with stochastic sampling and emulator-backed ground truth, explicitly to cut contamination and widen coverage
- Execution-time evaluation on correctness, safety, and latency inside real emulators (Mininet, K8s, DC simulator), which exposes failure modes that static, correctness-only benchmarks miss

### Weaknesses
- RL/SFT “use cases” are proof-of-concept and on small models (Qwen2.5-0.5B, limited SFT splits), so the “can be used for rl training” claim is ahead of the evidence.
- All results are still in three networking-style environments; claims of generality beyond these domains are argued but not empirically shown.
- The dynamic generation relies on hand-designed templates and app-specific state equivalence/safety checks; portability to other operators’ emulators may be non-trivial.

### Questions
1. In 5.1 you show a GRPO run with Qwen2.5-0.5B in Mininet and note it “does not fully solve routing issues.” Can you clarify whether NETARENA currently supports stable, long-horizon RL runs (multiple episodes, curriculum, failure replay), or whether this is mainly a demonstration of feasibility? If it’s the latter, please make the scope explicit and report at least learning curves / success-per-episode to show the environment is not too sparse.

2. You claim the unified state–action abstraction “generalizes across applications,” but all experiments are DC capacity planning, Mininet routing, and K8s policy. Can you point to a non-network/system domain where you tried to plug in the same pipeline?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents NETARENA, a dynamic benchmarking framework for evaluating LLMs in realistic network and system environments. It addresses critical limitations of existing static benchmarks, including data contamination risks, high statistical variance from limited dataset sizes, and inadequate representation of production environment complexity.. The framework defines a unified state–action abstraction that enables automatic query and ground truth generation across applications such as datacenter capacity planning, routing misconfiguration, and microservice policy troubleshooting. By integrating high-fidelity network emulators like Mininet and Kubernetes, NETARENA provides runtime feedback on correctness, safety, and latency. Experiments with models such as GPT-4o and Qwen-72B demonstrate low average correctness (13–38%), underscoring the complexity of real-world network tasks. NETARENA also supports supervised fine-tuning and reinforcement learning, enabling scalable, dynamic, and contamination-resistant evaluation of LLM agents in safety-critical network operations.

### Strengths
1. This paper effectively solves data contamination risk through dynamic generation, eliminates statistical unreliability of small datasets , and captures real-world complexity missing in existing benchmarks
1. It integrates with production-grade emulators (Mininet, Kubernetes), and provides execution-grounded assessment beyond simple correctness, including safety and latency metrics.
1. It supports 9,250+ queries with unlimited generation, while maintaining diversity across complexity levels and task types that static benchmarks cannot achieve.
1. The framework provides an environment that supports RL training and evaluation of LLMs in realistic network applications.

### Weaknesses
1. Limited agent diversity: The evaluation only includes baseline prompting strategies (CoT, Few-shot, ReAct), which may not fully represent the capabilities of advanced LLM-based agents in network reasoning tasks.
1. The integration with high-fidelity emulators may introduce significant setup challenges, potentially reducing the reproducibility and accessibility of the framework.
1. While correctness, safety, and latency are meaningful metrics, the evaluation could be enriched with additional dimensions.
1. Although RL post-training is mentioned, the paper does not include experimental results or analysis for RL-based fine-tuning.

Minors:
1. QWen -> Qwen

### Questions
1. What is the complexity of running the emulators and evaluation process at scale?
1. How is the ground truth constructed for the SFT dataset described in Section 4.3?

### Soundness
3

### Presentation
3

### Contribution
4
