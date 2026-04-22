# LUMINA: Long-horizon Understanding for Multi-turn Interactive Agents

- Avg Score: 4.40
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 6, 4, 6

## Abstract
Language models can excel at a variety of tasks (e.g., mathematical reasoning and coding) which are fundamental to solving more general goal-oriented feedback-driven agentic problems.
However, based on recent findings, two key points are evident:
(a) agentic problems require a variety of skills such as long-context reasoning, planning and decision making, and efficient exploration;
(b) even large frontier models under-perform in these family of tasks, especially in problems requiring long-horizon understanding. For example, GPT-4o has a 48.8% success rate on the AppWorld benchmark. 
In this paper, our goal is to understand the relation between the two, by examining which skills are necessary for solving multi-turn problems.
We work towards this goal using an oracle counter-factual framework that allows us to answer the question: what if the agent could leverage a specific oracle skill to achieve its goal?
To enable this framework, we introduce a set of procedurally-generated game-like tasks whose complexity can be controlled. For these controlled environments, we can provide accurate oracle interventions to guide the agent towards the goal.
Our findings suggest that while most interventions (e.g., planning) are generally beneficial, for some interventions the utility depends on the intricacies of the benchmark (e.g., ability to track state while iteratively modifying python lists).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
To study how agents perform in long-horizon tasks given a specific oracle skill, the authors construct three text-based worlds. They also focus on long-horizon tasks for agents that can be modeled as Partially-observable Markov Decision Processing, where the oracle intervention is that  agent can recover the belief state of the POMDP accurately under such intervention. They have found out that while the skills can improve LLM's policies, the effectiveness of each skill is influenced by the model size and environment.

### Strengths
1. Construction of three worlds to enable oracle skill control contributes to the research community. The worlds enable faithfully constructed oracle skills to study the behavior and performance of LLM agents, which is helpful to understand what affects LLM's performance.
2. The finding about LLMs excel at each step but performs relatively poorly in the entire horizon is interesting.

### Weaknesses
1. I would like to see stronger models' performance, like Qwen3-235B you have mentioned in the abstract, and also GPT-4o, maybe GPT-5. I would also like to see how o3 or o4-mini models performs. I am concerned about those tasks maybe only hard enough for small open source models (Qwen3-4b can get 86% with state tracking and planning in grid world). If this is the case, those worlds are still useful but limited.
2. The oracle formulation accommodates hints, planning, state tracking and history pruning. Those are reasonable and common considerations. However, Since hint, planning, history pruning are somehow "common practice" to augment LLMs in different tasks,  I would like to see some oracle interventions that is tailored to POMDP specially other than state tracking.

### Questions
1. Planning, State Tracking and History Pruning are common skills. Any other skills that can be considered? (Connection to Weakness 1)
2. Would you mind providing results of Qwen3-235B/GPT-4o and o3/o4-mini/Deepseek R1 on three worlds?
3. Can we view reasoning models (e.g. o3) as one LLM agent equipped with $O^{\text{plan}}$?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates the performance gap between LLMs on single-turn tasks and their underperformance on multi-turn tasks. The authors attribute this gap to additional skills required in multi-turn settings, including long-context reasoning, planning and decision making, and state tracking. Experiments conducted in controlled environments shows that compounding errors constitute a primary source of failure.

### Strengths
This paper provides a clear analysis of compounding errors in long-horizon tasks, showing how small step-wise mistakes accumulate to reduce overall success. It further disentangles specific agentic skills and introduces controlled environments to assess their individual contributions. Experiments across multiple skill combinations and model scales reveal that larger models can leverage longer contextual dependencies more effectively.

### Weaknesses
1.  The proposed environments are symbolic and fully rule-defined, omitting key challenges of real-world tasks such as parsing unstructured feedback. It is therefore unclear whether the identified bottlenecks generalize to real-world tasks.

2. Some of the findings have been reported in other works, which may limit the novelty of the results. For instance, recent studies on memory-augmented and planning-based agents have shown that these components can substantially influence performance. 

3. All experiments are conducted with the Qwen-3 model family, limiting the findings to Qwen-3’s scaling behavior. Other model families, such as Llama or Mistral, may exhibit different scaling dynamics and bottleneck characteristics.

### Questions
1. The study reports binary success rates as the primary metric. Could the authors provide additional measures, such as the ratio of actual-to-optimal path length, to better assess the impact of agent skills?

2. The results suggest that larger models can leverage longer contextual dependencies more effectively than smaller models. Could the authors evaluate how interaction history length affects performance across model scales, as shorter histories may benefit smaller models while excessively long histories could reduce performance for larger models?

3. The paper notes that performance declines when in-context examples are not aligned with oracle feedback, suggesting sensitivity to prompt design. Ablation studies on prompt wording or structure could clarify this.

4. The paper mentions that $O^{plan}$ provides a description of a single-turn subtask, but its precise nature is unclear. Could the authors clarify whether this corresponds to step-by-step guidance or a lower-level specification of the immediate action, and how each type affects model performance?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
LUMINA offers a principled way to analyze why LLM agents fail in complex, multi-step interactions — by decomposing agent behavior into modular skills and testing them systematically. To address this, the authors introduce LUMINA, a controlled evaluation framework using oracle counterfactual interventions: They design procedurally generated game-like environments where agent goals and task complexity can be precisely controlled. The oracle can intervene to provide specific “skills” (e.g., planning, exploration, state tracking), allowing researchers to test how each skill contributes to final performance.

### Strengths
The paper identifies a crucial and underexplored limitation in current LLM-based agents—their inability to maintain robust long-horizon reasoning across multiple turns. The motivation is well-grounded in empirical evidence (e.g., low success rates despite high per-step accuracy), and the authors effectively position long-horizon understanding as a distinct capability beyond standard reasoning or planning.

The introduction of an oracle counterfactual intervention framework is a major methodological strength. By isolating specific skills (e.g., planning, tracking belief state, context reformulation) and testing their contribution to success, the paper provides a systematic, interpretable way to analyze agentic competence—something rarely achieved in prior multi-turn benchmarks that often rely on end-to-end success metrics.

### Weaknesses
While the proposed environments (ListWorld, TreeWorld, GridWorld) are carefully controlled and effective for isolating individual skills, they remain relatively synthetic and detached from widely adopted agentic benchmarks such as ScienceWorld, OSWorld, or TravelPlanner. As a result, the paper provides valuable mechanistic insight but lacks direct evidence that the identified skill bottlenecks generalize to real-world multi-turn tasks. This limitation weakens the practical applicability and external validity of the findings.

The paper’s focus is primarily diagnostic rather than improvement-oriented. Although the oracle intervention analysis yields interpretive insights, it does not translate into a clear enhancement of actual agent performance. The study stops short of proposing or validating concrete training or inference strategies that could operationalize these insights to improve long-horizon reasoning capabilities. Thus, the contribution remains more analytical than actionable.

### Questions
as weakness

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This studies what specific skills make LLMs effective as multi-turn agents. The authors build an oracle intervention framework that can add idealized capabilities to an LLM policy during rollouts, such as planning hints, belief-state summaries, and context pruning. They introduce 3 generated environments—ListWorld (iterative list edits), TreeWorld (graph traversal), and GridWorld (2D navigation)—with controllable complexity and computable optimal actions.

### Strengths
Three procedurally generated environments enable controllable complexity.  They are designed with simple action spaces and trajectory-level annotations, supporting accurate measurement of optimal actions.

### Weaknesses
The idea of using oracle-based counterfactual interventions to dissect agent capabilities is interesting.
However, I have some concerns. The three oracle modules are treated as independent switches, but they interact tightly. Oplan converts the decision into a one-step optimal subtask, inherently reducing the need for state inference or history recall. Ostate summarization may already encode most of the historical trajectory.
Also  the simplification of Ohistory as truncate earlier steps is questionable, making it unclear whether the observed effects are from history or from the artificial deletion of essential cues.

### Questions
-  The abstract claims that “LLMs perform well on mathematics and code generation” are too broad; these capabilities were acquired after domain-specific fine-tuning.
- The reported 44.5% accuracy of Qwen3-235B on the BFCLv3 multi-turn benchmark lacks citation or reproducibility details.
- The notation alternates between (O_{state}), (O_{belief}), and (O_{context}). Consistent naming would improve readability.
- Several grammatical errors should be fixed, e.g., *“a oracle” → “an oracle”*, *“recieves” → “receives”* (line 139).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Language models have demonstrated strong performance across a wide range of tasks, such as mathematical reasoning and coding, which are fundamental to solving more general goal-oriented and feedback-driven agentic problems. However, such agentic problems require a diverse set of capabilities, including long-context reasoning, planning and decision making, and efficient exploration. Even large frontier models still underperform on this family of tasks, particularly those involving long-horizon understanding. For example, Qwen3-235B achieves only 44.5% accuracy on the BFCLv3 multi-turn benchmark. This paper aims to investigate which specific skills are essential for effectively solving multi-turn problems. To this end, it introduces an oracle intervention framework that evaluates the importance of different skills by posing counterfactual questions. The study finds that while most interventions, such as improving planning, are generally beneficial, the utility of certain interventions depends on the nuances of the benchmark, for example, the ability to accurately track state while iteratively modifying Python lists.

### Strengths
Originality: The paper designs three procedurally generated multi-turn environments, which facilitate the study of which skills have the greatest impact on agent capability.

Significance: Analyzing which skills, or combinations of skills, constitute the main bottlenecks to advancing capable multi-turn agents is highly meaningful, as it provides guidance for targeted improvements.

Clarity: The paper is clearly written.

### Weaknesses
Quality: The capability improvements observed in simulation environments may not necessarily transfer to real-world settings.

Significance: Can the conclusions drawn from simulation environments be applied to benchmarks in real-world scenarios?

### Questions
1. How are the different agent skills defined and categorized? Why does the paper focus only on the three skills: planning, state tracking, and history planning?
2. Are ( $O^{state}$ ) and ( $O^{belief}$ ) referring to the same concept in line 158?
3. In the experiments comparing the impact of different skills on performance, the trajectories are multi-step. Is the specific skill intervention applied at every step of the trajectory?

### Soundness
3

### Presentation
3

### Contribution
3
