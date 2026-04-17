# Towards General Agentic Intelligence via Environment Scaling

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 2

## Abstract
Advanced agentic intelligence is a prerequisite for deploying Large Language
Models in practical, real-world applications. Diverse real-world APIs demand
precise, robust function-calling intelligence, which needs agents to develop
these capabilities through interaction in varied environments. The breadth of
function-calling competence is closely tied to the diversity of environments
in which agents are trained. In this work, we scale up environments as a step
towards advancing general agentic intelligence. This gives rise to two central
challenges: (i) how to scale environments in a principled manner, and (ii) how
to effectively train agentic capabilities from experiences derived through inter-
actions with these environments. To address these, we design a scalable frame-
work that automatically constructs heterogeneous environments that are fully
simulated, systematically broadening the space of function-calling scenarios.
We further adapt a two-phase agent fine-tuning strategy: first endowing agents
with fundamental agentic capabilities, then specializing them for domain-
specific contexts. Extensive experiments on agentic benchmarks, τ-bench,
τ2-Bench, and ACEBench, demonstrate that our trained model, AgentScaler,
significantly enhances the models’ function-calling capability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces AgentScaler, a framework for advancing general agentic intelligence through environment scaling and teacher-free experience learning. The authors propose an automatic pipeline that constructs diverse tool-use environments by clustering 30k+ APIs into functional domains, materializing them as executable code with mock database schemas, and generating synthetic agent–human trajectories through simulated interactions. Agents are then fine-tuned in two stages: a general foundation phase and a domain-specific specialization phase.
Empirical evaluations on τ-Bench, τ²-Bench, and ACEBench show consistent improvements over Qwen3 baselines, with further cross-lingual tests on ACEBench-zh. The work positions itself as a scalable, verifiable alternative to teacher-model distillation.

### Strengths
- The environment construction pipeline is genuinely novel. It derives “gold” supervision signals directly from environment logic (database state transitions and tool dependency graphs) rather than relying on stronger teacher models or human annotation. Allowing it to scale nicely

- Representing tools as read/write operations over structured domains is an elegant unifying abstraction that allows programmatic  environment generation.

- The three-stage trajectory filtering process (validity → state alignment → exact match) provides a credible mechanism for ensuring quality and consistency in synthetic data.

### Weaknesses
- The paper never explicitly states which model generated the simulated trajectories. It appears that the base model under training were used for generation, but this should be made explicit to rule out hidden distillation.

- Limited notion of “generalization”. The only OOD test (ACEBench-zh) evaluates cross-lingual robustness, not transfer to unseen tool domains or schema structures. Because τ-Bench–derived tools appear in both training and evaluation, true domain generalization remains untested.

- While overall metrics improve, per-domain results fluctuate substantially. For example, in ACEBench-zh the “Special” subset consistently drops across model sizes, and in τ²-Bench some domains improve dramatically while others regress. No analysis is offered to explain these discrepancies.

- The pass^k curves show higher pass^1 accuracy but a faster decline as k increases, ultimately converging with the baseline. This indicates better one-shot precision, not improved stability or consistency across runs.

- The paper offers no insight into why certain domains or subsets improve more than others. Missing are per-domain data distribution analyses, qualitative error cases, or ablations on environment size or filtering stages.

- The environment builder reproduces τ-Bench-like schemas (“high consistency with official implementations”), suggesting potential overlap between training and benchmark environments, which weakens the validity of reported gains.

### Questions
- Which model was used as the agent during synthetic experience generation — the same Qwen3 base model or a stronger teacher? It would be helpful clarify explicitly.

- Data–evaluation overlap: How do you ensure that APIs, schemas, or tool sequences from τ-Bench or τ²-Bench are not directly included in the training environments? Do you plan to evaluate on unseen tool domains or unseen function schemas to substantiate claims of “general agentic intelligence”?

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
The paper introduces AgentScaler, a pipeline for developing general agentic intelligence through large-scale simulated tool-use environments and two-stage training (general → domain-specialized). It automatically constructs verified tool-use trajectories from 30K+ APIs, clusters them into domains, and trains models (4B–30B) that outperform comparable open LLMs on τ-Bench, τ²-Bench, and ACEBench.

### Strengths
- The paper presents a novel and automated approach to transform large-scale APIs into executable and verifiable environments, significantly improving reproducibility and coverage for agentic training.
  - The two-stage training strategy, which separates general capability building from domain specialization, is well-motivated and empirically validated to enhance performance across benchmarks.
  - The results demonstrate strong effectiveness, with compact models (e.g., 30B) achieving performance close to proprietary systems, showing that the proposed pipeline is efficient and scalable.

### Weaknesses
- The methodological novelty of the paper is relatively limited. The two-stage training strategy (general-to-domain specialization) closely resembles existing agent fine-tuning pipelines such as AgentFlan, ToolAce
  - The paper does not compare its environment construction approach or data generation pipeline with other existing frameworks, nor does it analyze in detail how environment scaling quantitatively affects model performance.

### Questions
- Could the authors provide a more detailed quantitative analysis of how environment scaling influences model performance? For example, how does expanding the number or diversity of simulated environments affect the model’s tool-use accuracy, generalization, or stability across benchmarks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper, the authors argue that agents need to interact with diverse and scale environments to gain real world function calling abilities. Therefore, the authors build a scaled up environment framework to automatically construct simulated heterogeneous environments for agents to interact with. Then they finetuned agents on trajectories collected from their simulated environments and then evaluated on benchmarks.

### Strengths
- The authors provided a useful tool of scaling up simulated environments for agent to interact with.
- The authors performed comprehensive experiments by training on models of different sizes and evaluated on multiple benchmarks.
- The authors compared their models with various baselines.

### Weaknesses
- Seed-OSS-36B (which has similar model size with AgentScaler-30B-A3B) achieves higher performance on ACEBench-en (both normal and overall). Yet the authors still claimed that they achieved state of the art performance on all three benchmarks they evaluated on.
- In Table 2, it appears that after training, the performance of AgentScaler-4B dropped 15.3 on special, AgentScaler-8B dropped 5.1 on normal, AgentScaler-30B-A3B dropped 3.4 on special, yet the authors didn't explain why this might be happening. However, the authors still claimed that their trained model, AgentScaler, significantly enhances the models' function-calling capability.
- With the abundant existing environments that agents could interact with, the authors didn't make it clear why simulated environments are useful.

### Questions
- Why does performance dropped after training?

### Soundness
2

### Presentation
3

### Contribution
2
