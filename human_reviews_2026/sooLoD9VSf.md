# EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Current Large Language Model (LLM) agents show strong performance in tool use, but lack the crucial capability to systematically learn from their own experiences. While existing frameworks mainly focus on mitigating external knowledge gaps, they fail to address a more fundamental limitation: the inability to iteratively refine problem-solving strategies. In this work, we introduce EvolveR, a framework designed to enable agent to self-improve through a complete, closed-loop experience lifecycle. This lifecycle comprises two key stages: (1) Offline Self-Distillation, where the agent's interaction trajectories are synthesized into a structured repository of abstract, reusable strategic principles; (2) Online Interaction, where the agent interacts with tasks and actively retrieves distilled principles to guide its decision-making, accumulating a diverse set of behavioral trajectories. This loop employs a policy reinforcement mechanism to iteratively update the agent based on its performance. We demonstrate the effectiveness of EvolveR on complex multi-hop question-answering benchmarks, where it achieves superior performance over strong agentic baselines. Our work presents a comprehensive blueprint for agents that learn not only from external data but also from the consequences of their own actions, paving the way for more autonomous and continuously improving systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces EvolveR, a self-evolving LLM agent framework that learns from its own interactions through a closed-loop experience lifecycle. Offline, the agent self-distills past trajectories into abstract, reusable strategic principles and maintains a curated experience base via semantic deduplication, integration, and dynamic scoring. Online, it retrieves these principles to guide multi-turn reasoning and tool use, generating higher-quality trajectories that feed back into learning. The loop is closed with reinforcement learning (GRPO), optimizing the policy on experience-guided trajectories to internalize effective strategy use. Across seven QA benchmarks and multiple Qwen2.5 model scales, EvolveR consistently outperforms strong baselines, with the 3B model achieving the best results; ablations show experience retrieval is indispensable and that self-distillation surpasses distillation by a stronger external teacher at larger scales, supporting the importance of cognitive alignment.

### Strengths
1. The paper is in general well-written.
2. The results are strong and surpass search-R1 by 5.7% on average.

### Weaknesses
1. This paper claims to propose a novel framework to leverage distilled experiences in online interaction. I feel this core idea lacks novelty, as it is similar to existing work that learns from experience (https://arxiv.org/pdf/2504.06821?, https://arxiv.org/pdf/2402.12317) and the data synthesis from existing experiences (https://arxiv.org/pdf/2501.10893?)
2. The evaluated datasets are limited to QA tasks. It would be better if the authors could extend to more general scenarios.
3. It is unclear that whether the generated data would be effective for larger models.

### Questions
Have you compared the difference between SFT and RL? What happens if you use SFT instead of RL.

### Soundness
3

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
This study proposes EvolveR, a framework that enables large language model (LLM) agents to self-evolve through their own interactive experiences.
Unlike previous approaches that rely on teacher models for knowledge distillation or external memory for storing prompts, this work introduces a human-inspired learning lifecycle consisting of two key stages.

The offline experience self-distillation phase allows the agent to summarize abstract strategic principles from its own interaction trajectories, performing semantic and quality evaluations to form an updatable experience base.
The online interaction phase enables the agent to retrieve its past strategic principles as guidance to enhance reasoning, while also generating new trajectories to further refine the experience base.

Experiments on several commonly used QA benchmarks demonstrate that EvolveR exhibits solid performance, and the ablation studies highlight the crucial role of the experience self-distillation mechanism.

### Strengths
1. The paper proposes a novel learning framework that models a human-inspired self-reflective learning mechanism for LLM-based agents. This framework enables agents to learn from their own experiences and enhance reasoning capabilities. The authors also include ablation studies demonstrating the effectiveness of the self-distillation component, and the framework shows competitive performance across multiple QA benchmarks.

2. The design of a continuously updatable experience database and a corresponding reward mechanism tailored to this reflective reasoning paradigm represents a useful contribution that could inspire future research and applications in related domains.

3. The writing is clear and fluent, and the figures and diagrams are well-organized, making the overall methodology easy to follow.

### Weaknesses
1. In the main experiments, the authors compare EvolveR primarily against Search-R1. However, other concurrent approaches mentioned in the related work—specifically Auto-Refine and O2-Searcher—are not directly compared in the experiments.
Based on available preprints from May 2025 using the same Qwen2.5-3B backbone, those methods achieved average Exact Match scores of 0.405 and 0.391, both higher than the 0.382 reported for EvolveR.
The paper only briefly explains this omission by stating:

“However, these systems either store raw, unstructured data or rely on memory mechanisms that are not designed for the systematic, long-term distillation and refinement of abstract strategic knowledge...”
While this provides conceptual differentiation, the lack of experimental comparison weakens the empirical persuasiveness of the work.

2. The paper does not provide a detailed discussion of the computational cost or resource consumption associated with maintaining and updating the strategy trajectory database. As the experience base grows, this could introduce scalability concerns and potential efficiency challenges for lifelong learning, even with pruning mechanisms.

3. A minor limitation is the absence of experiments on other LLM architectures such as LLaMA. Without such comparisons, it remains unclear whether the framework is specific to Qwen2.5 or generalizable to other base models.

### Questions
1. Could the authors clarify how EvolveR is positioned relative to Auto-Refine and O2-Searcher?
Are there specific scenarios where EvolveR performs better or offers complementary advantages?
Providing either empirical evidence or a clearer theoretical justification would help strengthen the experimental credibility.

2. Is there any analysis of computational efficiency, including training time or the impact of the cosine similarity threshold (θ_sim) used to decide when to add or merge trajectory principles?

3. Have the authors attempted experiments with larger-scale models (e.g., 7B or above)?
If not, could they clarify whether this was due to computational limitations or other design considerations?

### Soundness
2

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
EvolveR addresses the context forgetting problem in LLM agents (where each interaction is treated independently) through a self-evolution paradigm that alternates between offline self-distillation of trajectories into strategic principles and online interaction with principle retrieval. The framework employs GRPO-based policy updates and a dynamic experience base with semantic deduplication and quality scoring (Eq 2). Evaluated on seven QA benchmarks, EvolveR achieves 0.382 average score on Qwen2.5-3B, outperforming Search-R1 (0.325) and demonstrating that self-distillation surpasses teacher distillation at the 3B scale.

### Strengths
1.  Empirical results looks solid outperforming strong baseline like Search-R1

     a. The paper demonstrates consistent improvements across 7 diverse QA benchmarks (both in-domain and out-of-domain), with EvolveR achieving 0.382 average score on Qwen2.5-3B versus Search-R1's 0.325, showing robust generalization.

2. Ablation experiments (Table 2, 3) show that experience retrieval is crucial, however lack further analysis (see Q1). I personally like the additional of distillation experiments as well.

### Weaknesses
1. While the paper describes deduplication and quality control (Eq 2, line 302 ), there's insufficient analysis of long-term scalability. How does performance change as the experience base grows to thousands of principles?

2. The claim (lines 427-431) that self-distillation surpasses teacher distillation at 3B due to “cognitive alignment” is not concrete enough as it is only seen at 3B scale while smaller scales show reverse result.

3. Some incomplete analysis of limitations and failure modes:

    a. Dependency on GPT-4o for cold-start (Table 5) somewhat contradicts "self-evolving" narrative.
 
    b. Appendix A.4's finding that absorbing principles hurts performance deserves main paper discussion

### Questions
Major:
1. How does the use of different actions ( search_knowledge, search_experience, think ) change during the RL process, does search_knowledge or search_experience increase with more steps? 

   a. Could you provide a longitudinal analysis to show how principle quality evolves across RL iterations?

    b. Also, please provide qualitative examples of high-scoring vs low-scoring principles to understand what makes principles effective.

Minor:

2.  Can you provide computational costs? Training time, memory requirements, and inference overhead compared to baselines would help assess practical viability.

3. In the distillation experiment at 3B scale, would scaling more data helps with the performance? I would assume the data size in the distillation is fixed for all 3 scales but for bigger models, we would expect it needed more data to perform better?

4. Although as listed in one of the limitations which stated experiments only include QA, I would still curious would non QA tasks ( as out of domain ) such as coding or multi-turn conversations benefit from EvolveR?

### Soundness
4

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
The paper proposes EvolveR, a closed-loop lifecycle for agent self-evolution with two alternating stages: (i) Offline Self-Distillation that converts raw trajectories into reusable principles with deduplication and dynamic scoring, and (ii) Online Interaction where the agent retrieves principles to guide reasoning; the loop is closed with policy evolution via RL (GRPO). Experiments on seven QA datasets (NQ, HotpotQA, TriviaQA, PopQA, 2Wiki, Musique, Bamboogle) using Qwen2.5 backbones show gains over prompting, SFT, and RL baselines; ablations analyze self-distill vs teacher-distill and the role of principle retrieval.

### Strengths
1.	Clear lifecycle design that operationalizes “experience as principles” rather than raw cases; includes deduplication and dynamic scoring for maintenance. 
2.	Integrated RL (GRPO) ties the use of retrieved principles to policy updates via a composite reward (outcome + format), providing a concrete optimization path. 
3.	Broad evaluation across seven QA benchmarks with competitive improvements; main table shows best average at 3B. 
4.	Insightful ablations.

### Weaknesses
1.	All experiments are QA-centric; it is unclear whether principle distillation generalizes to planning, tool orchestration, or embodied tasks. The paper frames a general “agent evolution” paradigm but evaluates only text QA. 
2.	The format reward is hand-crafted with specific thresholds (e.g., counts of <think>/<search>), which could bias behaviors; robustness to alternative reward shapes or automatic shaping is not shown. 
3.	The score s(p)= (csucc+1)/(cuse+2) is simple; its sensitivity to retrieval noise, task difficulty, or distribution shift is not analyzed. Limited justification for thresholds (θ_sim, θ_prune) and no calibration study. 
4.	Table 1 averages improve (0.382), but per-dataset effect sizes vs best baselines are sometimes modest; variance/error bars, seed sensitivity, and cost/throughput are not systematically reported. 
5.	Limited analysis of failure modes Although examples/prompts are in Appendix, the main paper provides few qualitative failures where principles mislead or conflict; Appendix notes a negative result for “exp-absorb,” but broader diagnostics remain thin.

### Questions
1.	Can you show results on non-QA agent tasks (multi-step tool use, program synthesis, web navigation, or planning) to validate the paradigm beyond QA?
2.	How sensitive are results to θ_sim, θ_prune, and the format reward hyperparameters? Any ablation on these knobs?

### Soundness
2

### Presentation
3

### Contribution
3
