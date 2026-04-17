# MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning

- Decision: Accept (Poster)
- Scores: 2, 2, 6, 6

## Abstract
Recent vision-language models have strong perceptual ability but their implicit reasoning is hard to explain and easily generates hallucinations on complex queries. Compositional methods improve interpretability, but most rely on a single agent or hand-crafted pipeline and cannot decide when to collaborate across complementary agents or compete among overlapping ones. We introduce MATA (Multi-Agent hierarchical Trainable Automaton), a multi-agent system presented as hierarchical finite-state automaton for visual reasoning whose top-level transitions are chosen by a trainable hyper agent. Each agent corresponds to a state in the hyper automaton, and runs a small rule-based sub-automaton for reliable micro-control. All agents read and write a shared memory, yielding transparent execution history. To supervise the hyper agent’s transition policy, we build transition-trajectory trees and transform to memory-to-next-state pairs, forming the MATA-SFT-90K dataset for supervised finetuning (SFT). The finetuned LLM as the transition policy understands the query and the capacity of agents, and it can efficiently choose the optimal agent to solve the task. Across multiple visual reasoning benchmarks, MATA achieves the state-of-the-art results compared with monolithic and compositional baselines. The code and dataset are available at https://github.com/ControlNet/MATA.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a multi-agent framework for visual reasoning tasks, where a learnable hyper agent coordinates three types of reasoners. During training, only the hyper agent is optimized using synthetic data. Experiments are conducted on two VQA benchmarks (GQA and OK-VQA) and several referring expression comprehension (REC) datasets.


The general idea of multi-agent collaboration has been extensively explored in prior works such as [1] and [2]. While this paper introduces a slightly different collaboration mechanism, the overall conceptual novelty remains limited. Therefore, the main contributions appear to be: (1) applying multi-agent collaboration to visual reasoning; and (2) learning the hyper agent using synthetic data.


Regarding (1), the technical contribution seems incremental. The framework largely mirrors existing multi-agent reasoning pipelines, with LLMs replaced by VLMs and visual tools. Moreover, the experimental evaluation is restricted to GQA and OK-VQA, which are insufficient to demonstrate general effectiveness on broader visual reasoning tasks.


Regarding (2), based on Table 5, the results show minimal difference between models trained with and without SFT. This weak empirical signal makes it difficult to conclude that the proposed learning scheme provides substantial benefits.


Overall, given the limited technical novelty and unconvincing empirical validation, I lean toward a negative recommendation for this submission.

[1] Weize Chen et al., AGENTVERSE: FACILITATING MULTI-AGENT COLLABORATION AND EXPLORING EMERGENT BEHAVIORS

[2] Sirui Hong et al., METAGPT: META PROGRAMMING FOR A MULTI-AGENT COLLABORATIVE FRAMEWORK

### Strengths
1. A multi-agent framework for visual reasoning
2. Empirical verification of the effectiveness of the proposed framework

### Weaknesses
See the comments in "Summary"

### Questions
N/A

### Soundness
2

### Presentation
3

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
MATA introduces a hierarchical finite-state automaton for visual reasoning where multiple specialized agents (specialized perception, stepwise reasoning, oneshot reasoning) collaborate and compete. The key innovation is a trainable hyper agent that learns transition policies between agents using supervised fine-tuning on a generated dataset (MATA-SFT-90K). The dataset is created by expanding transition trajectory trees, scoring leaf nodes based on task performance, and generating memory-to-next-state training pairs. MATA improves over the base internvl model on GQA, OK-VQA, RefCOCO/RefCOCO+/RefCOCOg, and Ref-Adv benchmarks.

### Strengths
- The combination of trainable high-level transitions with rule-based sub-automata is elegant, focusing learning on the ambiguous agent selection problem while preserving reliable execution within agents.
- The paper provides experiments across multiple benchmarks (VQA and visual grounding), demonstrating consistent improvements over the base model.
- The transition trajectory tree expansion provides a principled approach to generating supervision for the hyper agent, though scalability concerns remain.

### Weaknesses
- The gains appear modest (e.g., 75.2% base internvl25 used as vlm vs 76.5% theirs on AOKVQA) considering the 90K in-domain training examples generated. The paper doesn't isolate whether improvements come from multi-agent collaboration or simply additional task-specific training data.
- Table 5 reveals that removing SFT causes performance to drop below the base internvl25 model, suggesting the architecture itself may be detrimental without training. A crucial missing experiment is training a monolithic model on the same MATA-SFT-90K data to isolate the architectural contribution.
- The paper claims zero-shot generalization but the base models may have been pre-trained on these datasets. This undermines claims about generalization capabilities.
- The authors admit their trajectory tree expansion becomes intractable as agents increase, yet provide no computational overhead analysis or comparison with simpler methods.
- While claiming to address competition between functionally overlapping agents, the paper only uses three distinct agents with clearly separated roles, not truly competitive alternatives.

### Questions
- What happens when you train InternVL2.5-8B directly on MATA-SFT-90K to output answers without the multi-agent architecture? This would isolate the contribution of the architecture versus the training data. Or did distillation of a large VLM to internvl2.5?
- What is the computational overhead (inference time, memory) compared to monolithic models and single-agent methods like HYDRA?
- How does performance scale when adding more agents? Given the acknowledged exponential growth in trajectory trees, is this approach practical beyond 3-4 agents?
- Can you provide evidence that the models used (InternVL2.5, Florence2-L) were not trained on GQA/OK-VQA/RefCOCO to substantiate zero-shot claims?
- Why does the architecture without SFT perform worse than the base VLM used (internvl2.5)? Why not use a VLM instead of an LLM for the state controller?
- Have you considered more efficient alternatives to near-exhaustive tree expansion, such as Monte Carlo tree search or learned value functions to prune unpromising branches?

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
This paper proposes **MATA (Multi-Agent Trainable Automaton)**, a hierarchical framework for **multi-agent visual reasoning**. Instead of executing a fixed modular pipeline, MATA organizes several reasoning agents (one-shot, step-wise, and specialized perception agents) as states of a **finite-state hyper-automaton**. A **trainable hyper-agent** which is an LLM-based controller then learns to select the next agent state from a shared memory, enabling dynamic collaboration and competition among agents.

To supervise this transition policy, the authors construct **MATA-SFT-90K**, a dataset of memory-to-next-state pairs extracted from expanded transition-trajectory trees across visual reasoning datasets (GQA, OK-VQA, RefCOCO series). The learned controller yields interpretable reasoning traces and achieves state-of-the-art results across multiple visual reasoning and grounding benchmarks.

### Strengths
– **Clear conceptual motivation:** The paper identifies an important limitation of existing VLMs and compositional systems that the lack of a learned, flexible orchestration mechanism among reasoning agents and recasts it elegantly as a finite-state automaton control problem.

– **Novel hierarchical formulation:** Treating each agent as a sub-automaton and learning high-level transitions through a hyper-agent is a conceptually clean, interpretable design that unifies rule-based micro-control with data-driven macro-control.

– **Strong empirical results:** MATA attains new SOTA accuracy on GQA, OK-VQA, and RefCOCO series, outperforming both monolithic VLMs and compositional baselines.

– **Generalizability and ablation rigor:** Cross-dataset transfer (Table 6) shows < 1 % drop in zero-shot settings; ablations confirm that supervised fine-tuning of the hyper-agent contributes most gains (Table 5).

### Weaknesses
– **Incremental algorithmic novelty:** While the integration is elegant, many components (agent orchestration, SFT, trajectory trees) extend known concepts from HYDRA and NAVER. The work’s originality lies more in *system design* than in theoretical innovation.

– **Limited discussion of scalability:** The near-exhaustive transition expansion is tractable for 3 agents but may explode combinatorially as more states are added.

– **Computational cost analysis:** Wall-clock training times and GPU usage are only qualitatively stated; quantitative comparisons would clarify efficiency relative to HYDRA or DWIM.

### Questions
Please see the above weakness.

1. Do you encounter issues with the shared memory growing unboundedly during long reasoning sequences? If so, how is this mitigated?
2. It would also be helpful if the authors could provide qualitative visualizations comparing MATA’s reasoning paths against other systems, to better illustrate how its hierarchical controller differs in practice.

### Soundness
3

### Presentation
3

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
This paper addresses a key challenge in current multi-agent systems: how to train a policy to select among multiple sub-agents, rather than relying on manually handcrafted pipelines. To tackle this, the authors propose MATA (Multi-Agent hierarchical Trainable Automaton), a novel system for visual reasoning that organizes inference as a hierarchical finite-state automaton. The authors construct a transition-trajectory dataset, MATA-SFT-90K, which is used for supervised training and evaluation. The method demonstrates strong performance on a range of visual reasoning tasks.

Overall, this is a solid paper with clear motivation. The method is well designed, and the experiments are conducted rigorously. One major concern is that the three-agent design is somewhat simplistic and limited in scope, as the authors themselves acknowledge in the limitations section.

### Strengths
- **Clear motivation**: The paper addresses an important limitation in current multi-agent systems: leveraging the power of multiple agents typically requires manual pipelining, which becomes unwieldy as task complexity grows. The proposal to learn a hyper-policy for agent selection is both reasonable and interesting.
- **Principled and extensible design**: MATA’s architecture is well aligned with its motivation. It is technically sound and, importantly, not narrowly restricted to the specific visual reasoning setting or the particular sub-agents used in this paper. In principle, the approach could extend to other tasks and larger agent pools, opening up many potential research directions.
- **Rigorous experiment design and evaluation**: The experiments are well designed and executed. For example, the three SFT configurations and their results convincingly demonstrate the benefits of the proposed method and its generalization ability, rather than mere overfitting.

### Weaknesses
### Major

- **Limited applicability**: As noted in the limitations, the use of only three agents is a restricted setting. There is also a lack of detail regarding how these three agents were selected and the rationale behind their design.
- **Unclear attribution of performance gains**: It is not clear whether the learned state transition policy is truly responsible for the observed performance improvements. For example, if all three agents were simply called exhaustively, would performance improve regardless, making the learned policy less critical?
- **Competition mechanism not fully justified**: While the collaborative aspect of the system is well motivated, the competitive aspect is less convincing. The paper claims that “a competition mechanism where functionally overlapping agents for the same subtask work together is under-explored,” but in the current three-agent setup, the agents seem to serve distinct roles with little actual overlap. It would be more compelling to see experiments with a larger pool of agents, including multiple agents with overlapping capabilities, to better demonstrate the value of competition.

### Minor
- Line #352: “... target dataset for evaluated; ...” should be “... target dataset for evaluation; ...”

### Questions
- How were the three agents chosen, and what was the rationale behind their selection?
- How is the competition aspect justified, given that the three agents in the current setup do not appear to have significant functional overlap?
- Although the exponential growth of the transition search space is discussed as a limitation, do the authors have any thoughts on extending the approach to more complex scenarios with significantly more agents?
- Would it be possible that we would observe the same performance gains if the three agents are called in an exhaustive way?

### Soundness
3

### Presentation
3

### Contribution
3
