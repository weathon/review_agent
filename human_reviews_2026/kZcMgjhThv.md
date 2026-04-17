# ConsumerBench: Benchmarking Generative AI Applications on End-User Devices

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
The recent shift in Generative AI (GenAI) applications from cloud-only environments to end-user devices introduces new challenges in resource management, system efficiency, and user experience. This paper presents ConsumerBench, a comprehensive benchmarking framework designed to evaluate the system efficiency and response time of GenAI models running on end-user devices. Unlike existing benchmarks that assume exclusive model access on dedicated GPUs, ConsumerBench simulates realistic multi-application scenarios executing concurrently on constrained hardware. Furthermore, ConsumerBench supports customizable workflows that simulate complex tasks requiring coordination among multiple applications. ConsumerBench captures both application-level metrics, including latency and Service Level Objective (SLO) attainment, and system-level metrics like CPU/GPU utilization and memory bandwidth. Through extensive experiments, ConsumerBench reveals inefficiencies in resource sharing, unfair scheduling under greedy allocation, and performance pitfalls of static model server configurations. The paper also provides practical insights for model developers and system designers, highlighting the benefits of custom kernels tailored to consumer-grade GPU architectures and the value of implementing SLO-aware scheduling strategies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces ConsumerBench, a benchmarking framework designed to evaluate the runtime and system efficiency of Generative AI (GenAI) applications running on end-user devices (e.g., laptops, smartphones). Unlike existing benchmarks that assume dedicated hardware, ConsumerBench emulates multi-application, resource-constrained environments, measuring both application-level SLOs (latency, throughput) and system-level metrics (CPU/GPU utilization, memory bandwidth, power).

Experiments on consumer GPUs (RTX 6000) reveal:
1. Greedy GPU allocation causes starvation for lightweight, latency-sensitive applications.
2. Static GPU partitioning improves fairness but lowers utilization.
3. Shared inference servers with fixed configurations can cause conflicting SLO satisfaction.

The authors distill design insights for architecture-aware kernels, SLO-aware scheduling, and configurable inference servers, highlighting gaps in current GenAI runtime systems.

### Strengths
1. The paper identifies a critical and underexplored problem: efficient concurrent execution of heterogeneous GenAI workloads on consumer devices.
2. This paper provides an end-to-end benchmarking suite with user-configurable DAG-based workflows, automated metric collection, and extensible APIs for custom apps.
3. This paper is easy to follow. Well-structured experiments reveal non-trivial interactions between applications.
4. The authors provide implementation details, configuration examples, and supplementary materials for replication.

### Weaknesses
1. I’m not entirely sure, but NVIDIA MPS might support dynamic resource allocation rather than static partitioning. You can start the NVIDIA MPS and run multiple jobs on the same set of GPUs without additional configuration. I think this could serve as a new baseline.
2. The evaluation only considers greedy and static partitioning, without implementing or comparing against dynamic schedulers (e.g., kernel-level preemption or SLO-prioritized scheduling). This limits the depth of system insights. In fact, several resource allocation papers have explored inference scheduling, for example, Fairness in Serving Large Language Models (OSDI). The authors could consider including additional baselines from this line of work.
3. The study measures latency and utilization but lacks user-centric Quality of Experience (QoE) metrics or perceptual thresholds that better represent real usage.
4. The title formatting and the “Anonymous authors” placeholder in the template look a bit unusual.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents ConsumerBench, a benchmarking framework for evaluating Generative AI applications running concurrently on end-user devices. Users specify applications, models, service-level objectives (SLOs), and inter-application dependencies in a YAML configuration file, which the system compiles into a directed acyclic graph (DAG) of application instances. ConsumerBench then executes and measures these workloads to expose inefficiencies in resource management, GPU sharing, and SLO-aware scheduling under multi-application concurrency.

### Strengths
1. Novel concurrency focus: The paper addresses multi-application inference, a relatively unexplored yet practically important problem for end-user AI systems.
2. The experimental results yield clear, practical takeaways for developers seeking to improve performance and fairness when multiple generative AI applications share hardware resources.
3. The use of YAML-based configuration and the DAG-based task execution model make the framework easily extensible — new applications, models, or metrics can be integrated with minimal effort.

### Weaknesses
1. Despite claiming a focus on “end-user devices,” all experiments are performed on a single workstation with an RTX 6000 GPU. Evaluations on consumer-class GPUs (e.g., RTX 4060/4070) or integrated accelerators would strengthen the paper’s external validity.
2. Each task uses a fixed model configuration. Demonstrating results across multiple models per modality would better validate that the benchmark’s findings generalize beyond specific architectures.
3. The paper discusses SLO-aware resource allocation but does not evaluate any dynamic or adaptive scheduling strategies. This weakens the central argument and leaves open the question of how ConsumerBench would perform under more optimal schedulers.

### Questions
1. Have you considered expanding the benchmark to include more task types and custom metrics, allowing users to choose flexible workload combinations for evaluation?
2. What was the reasoning behind using only one hardware setup? Would you consider evaluating on more representative consumer devices or cross-vendor platforms to demonstrate generality?

### Soundness
2

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
3

### Summary
This paper presents a novel benchmark called ConsumerBench, which helps evaluates performances of multi-application scenarios executed concurrently on constrained hardware. This is a useful benchmark given the intensive requirements of applications these days. The paper also showcases an example that helped provide potential insights that can be drawn from using the benchmark.

### Strengths
The paper identifies an useful, practical gap in benchmarking applications under constrained on-device resources. It also presents careful thought out workflow and analysis, along with an example usecase where the use of such a benchmark/framework might help bring more insights on the concurrent execution and where the limits might occur. The findings are well written and presented with clarity.

### Weaknesses
The paper provides a benchmark that is useful in terms of software aspects for applications running concurrently on constrained resources. However, it hasn't mentioned any consideration for hardware impacts, which can further influence the performance of the applications on the end-user devices. Furthermore, the paper could explain more on the usecases and usefulness of the benchmark, such as how the workflow/findings scale and provide systematic insights across different architecture and types of end-user devices.

### Questions
- Can this benchmark be used in a container or sandbox setting, such as to simulate and understand the performance of target applications on a device before actual implementation/purchase? The main usecase seems to be for an existing device.
- The paper identified several factors affecting performance. To what extent can the bottleneck and processes identified in the analysis be used to better schedule the applications?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces consumerbench, a benchmarking application for four GenAI workloads. The benckmark is used to benchmark the performance of the four applications on a 2018 RTX600 GPU, with an Intel Xeon Gold 6126 CPU (2.60GHz, 24 cores) and 32GB of system memory (DRAM). The results show that three out of the four applications tested, achieve their required SLOs, with the third one not achieving the slo for 1.5% of the audio samples.

### Strengths
Benchmarking hardware is an important problem. This paper aims to improve the current State-of-the-art of edge/small device benchmarking by benchmarking workflows of tasks.

### Weaknesses
My main issue is with the novelty and depth of the work. For example, as a benchmark, when comparing the suggested benchmark with PalmBench (cited in the paper), the authors have a much more limited set of applications/Models. When it comes to insights from the experiments, the results are very well known. For example, the KTransofrmer project (open-sourced with a paper in SOSP 2025) has been setup to solve many of the insights discussed. 

Another issue, for a benchmarking paper, one typically needs to run on many configurations. Going back to the PalmBench paper, they test with three operating systems on nine different hardware platforms. Testing on only two platforms does not allow for a good enough evaluation.


Writing:
1- The paper almost exclusively cites Arxiv papers/versions of the paper with a handful of exceptions. Please fix this!

### Questions
1. Can you expand your experiments to more models, scenarios, and more hardware/OS configurations?
2. Besides the wokflow, what else is different from PalmBench?

### Soundness
2

### Presentation
3

### Contribution
1
