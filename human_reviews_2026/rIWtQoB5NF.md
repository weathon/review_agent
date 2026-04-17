# Towards Generalizable Context-aware Anomaly Detection: A Large-scale Benchmark in Cloud Environments

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Anomaly detection in cloud environments remains both critical and challenging. Existing context-level benchmarks typically focus on either metrics or logs and often lack reliable annotation, while most detection methods emphasize point anomalies within a single modality, overlooking contextual signals and limiting real-world applicability. Constructing a benchmark for context anomalies that combines metrics and logs is inherently difficult: reproducing anomalous scenarios on real servers is often infeasible or potentially harmful, while generating synthetic data introduces the additional challenge of maintaining cross-modal consistency. We introduce CloudAnoBench, a large-scale benchmark for context anomalies in cloud environments, comprising 28 anomalous scenarios and 16 deceptive normal scenarios, with 1,252 labeled cases and roughly 200,000 log and metric entries. Compared with prior benchmarks, CloudAnoBench exhibits higher ambiguity and greater difficulty, on which both prior machine learning methods and vanilla LLM prompting perform poorly. To demonstrate its utility, we further propose CloudAnoAgent, an LLM-based agent enhanced by symbolic verification that integrates metrics and logs. This agent system achieves substantial improvements in both anomaly detection and scenario identification on CloudAnoBench, and shows strong generalization to existing datasets. Together, CloudAnoBench and CloudAnoAgent lay the groundwork for advancing context-aware anomaly detection in cloud systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CloudAnoBench, a benchmark for context-aware anomaly detection in cloud systems that jointly contains metrics and logs across 28 anomalous and 16 deceptive normal scenarios. It also proposes CloudAnoAgent, a multi-agent LLM system with Fast/Slow Detection and a symbolic verifier that correlates metrics and logs to reduce false positives and identify scenarios. On CloudAnoBench, CloudAnoAgent improves F1 and lowers FPR versus ML baselines and vanilla LLM prompting. It also adapts to log-only datasets (HDFS v1, Thunderbird, BGL) with competitive results.

### Strengths
1. Positioning anomalies as contextual interactions between metrics and logs is well-motivated and distinct from point-anomaly setups. The benchmark includes deliberately deceptive normal scenarios to stress false-positive robustness.


2. The scenario taxonomy spans resource, network, software/app, malicious, and subtle cases; normal scenarios are enumerated with plausible operational events.

3. CloudAnoAgent improves F1 and reduces FPR over ML-based vanilla-LLM methods. It further shows competitive performance on log-only datasets.

### Weaknesses
1. The dataset generation relies heavily on LLMs. Though with human review for verification, the paper lacks quantitative evidence of how realistic the generated metric dynamics and log semantics are.

2. There is a lack of more recent DL-based and LLM-based anomaly detection baselines that are metric-only. The paper only evaluates ML-based metric-only methods.

3. The paper provides ablation with/without the symbolic verifier but does not isolate the contributions of Fast vs Slow stages or the Integrated Agent, leaving unclear where most of the gains arise.

### Questions
1. Dataset creation is heavily dependent on LLMs generating both metrics and logs with “code execution,” followed by GPT-4o verification and manual review. Are the metrics and logs generated from scenarios created by the LLMs but running on real hardware setups? If it is running on real hardware, how to manage the software environments of different anomaly scenarios?

2. In addition, how is each anomaly scenario in the 5 categories being selected into the benchmark? How do we ensure that they reflect the realistic cloud anomalies or incidents? How much proportion of the LLM-generated examples are inspected by human experts? Is the system based on any real-world incident traces?

3. The paper mentions that LO2 is a dataset with logs and metrics from a microservice system. How is it different than CloudAnoBench in terms of data collection, workload realism, etc?

4. Can you provide a comparison with state-of-the-art DL-based and LLM-based metrics-only methods? Current ML-based methods seem to be too simple.

5. Can you further provide an ablation study comparing the system without slow detection and without integrated agent? Currently, only one ablation with a symbolic verifier is provided.

6. What is being learned and refined in the symbolic verifier for the refinement process described in Section 4.2? Are they rules used in the metric verifier and log verifier?

7. How to deal with the context window limit for the metric and log agent if the input data is long?

### Soundness
3

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
3

### Summary
The paper introduces CLOUDANOBENCH, a large-scale benchmark for context-aware anomaly detection in cloud environments. It emphasizes realistic, multimodal context anomalies and challenging normal cases that often cause false positives. The authors propose CLOUDANOAGENT, an LLM-based multi-agent system with symbolic verification, which integrates metric and log data for robust anomaly detection and scenario identification. Experiments show CLOUDANOAGENT outperforms traditional machine learning, log-only, and vanilla LLM baselines.

### Strengths
+) A comprehensive and realistic benchmark that combines both metrics and logs

+) Extensive experimental results showing significant improvements over LLM baselines

### Weaknesses
-) The benchmark, while large, is still limited to the scenarios and data sources curated by the authors, which may not be fully covering the network anomalies

-) It might be better to analyze failure cases

### Questions
a) The symbolic verification part seems remedy of the agent system. Is it possible to make the agent detector stronger and eliminate symbolic verification in the future versions?

b) What is the overhead of the proposed anomaly detection system?

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
3

### Summary
The paper introduces two main contributions for context-aware anomaly detection in cloud environments. The first is CLOUDANOBENCH, a new large-scale benchmark dataset that combines both metric and log data. A key feature of this benchmark is its inclusion of 28 anomaly scenarios and 16 "deceptive normal" scenarios, which are designed to look like anomalies but are benign, making the detection task more challenging. The second contribution is CLOUDANOAGENT, an LLM-based, multi-agent system designed to work with this kind of multi-modal data. The agent uses a "Fast and Slow" detection mechanism to process metrics and logs, and critically, it incorporates a symbolic verifier to validate the LLM's findings, aiming to improve accuracy and reduce false positives. The experimental results show that CLOUDANOAGENT outperforms traditional machine learning methods and standard LLM prompting on the new benchmark, particularly in reducing the false positive rate.

### Strengths
1. The benchmark itself is a solid contribution. The focus on "context anomalies" that require both metrics and logs is well-motivated, and the inclusion of "deceptive normal scenarios" addresses a common and practical challenge in real-world operations where false alarms are costly.
2. The design of the CLOUDANOAGENT, which combines LLM-based reasoning with a symbolic verifier, is a strong point. This hybrid approach directly tackles the reliability issues often associated with LLMs, using a rule-based system as a check. The significant drop in the FPR shown in Table 2 for the agent with the verifier supports the value of this design.
3. The paper provides a very clear explanation of the problem and its proposed solution. The data generation process, agent architecture, and experimental setup are well-documented. The public release of the benchmark is also a valuable service to the community.

### Weaknesses
1. The benchmark is generated synthetically using LLMs. While this is a practical necessity for scenarios that are dangerous to reproduce, it raises questions about the data's realism and potential biases. An LLM-based evaluation model (CLOUDANOAGENT) might perform very well on data generated by another LLM, but it's unclear if this performance would hold on real-world, non-synthetic data.
2. The claim of "strong generalization" seems a bit of a stretch. The agent is tested on older, log-only datasets (BGL, Thunderbird, HDFS_v1) in Table 4. While it performs competitively, these datasets are for point-anomaly detection from a single modality. This doesn't fully validate the agent's core strength, which is meant to be context-aware detection on multi-modal data. Generalization should ideally be shown on a different context-aware dataset.
3. The practical aspects of the CLOUDANOAGENT system, such as latency and computational cost, are not discussed. The multi-step process involving multiple agents and a verifier (Figure 3) seems like it could be slow, which is a critical factor for a real-time anomaly detection system.

### Questions
1. Regarding the dataset generation: The use of an LLM to generate the benchmark data is clever, but I'm curious about the potential for model-specific artifacts. Could the authors comment on whether the CLOUDANOAGENT (which also uses LLMs) might have an inherent advantage on this dataset compared to non-LLM methods? Were any steps taken, beyond the manual review mentioned in Section 3.3, to ensure the data distribution truly mirrors real-world operational data?
2. Regarding the evaluation of baselines: In Section 5.3 (Line 385), it's stated that the ML methods were restricted to using only metric data. Given that the benchmark's main premise is the combination of metrics and logs, this seems to put those baselines at a significant disadvantage. Was there any attempt to incorporate log features for the ML models, for instance through log parsing and feature extraction, to allow for a more direct comparison?
3. Regarding the symbolic verifier: This component appears to be key for the performance improvement, especially in reducing false positives. Could the authors elaborate on the process for creating the verification rules for each of the 28 anomaly scenarios? Does this require significant manual effort and domain expertise for each new anomaly type the system needs to detect? How might this approach scale in a real-world environment where new and unexpected failure modes can emerge?
4. Regarding latency: The agent's architecture involves several sequential steps. Could the authors provide some information on the end-to-end latency of the detection process, from receiving the data to producing a final decision? How does this potentially compare to the much simpler vanilla LLM or ML baselines, and what are the trade-offs for the improved accuracy?

### Soundness
3

### Presentation
3

### Contribution
2
