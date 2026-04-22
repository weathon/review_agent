# Dual-Scale World Memory for LLM Agents towards Hard-Exploration Problems

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
LLM-based agents have seen promising advances, yet are still limited in hard-exploration tasks which require agents to perform sustained exploration under sparse feedback. We present GLoW, a novel approach leveraging a dual-scale textual world memory, maintaining a trajectory frontier of high-value discoveries at the global scale, while learning from local trial-and-error in exploration through a Multi-path Advantage Reflection mechanism which infers advantage-based progress signals to guide exploration. To evaluate our framework for hard-exploration, we tackle the Jericho benchmark suite of text-based games, where GLoW achieves a new state-of-the-art performance for LLM-based approaches. Compared to state-of-the-art RL-based methods, our approach achieves comparable performance while requiring 100-800× fewer environment interactions. When scaled to stronger LLMs, GLoW surpasses all prior methods on 4 out of 6 difficult and extreme Jericho games.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces GLoW (Global-Local World Models), a new framework for Large Language Model (LLM) agents designed to tackle hard-exploration problems. The core idea is a dual-scale mechanism. At the global scale, GLoW maintains a "trajectory frontier" of the most successful past episodes. An LLM analyzes this frontier to decompose state values into achieved rewards and future potential, guiding the selection of promising states to explore from (similar to the "select" phase of Go-Explore). At the local scale, it employs a "Multi-path Advantage Reflection" (MAR) mechanism, where the agent makes several exploration attempts from a single state. An LLM then compares these attempts to infer "semantic advantages" for specific actions, guiding the subsequent exploration policy (the "explore" phase).

### Strengths
The paper's central concept of a dual-scale world model is interesting. At the global scale, it improve the Go-Explore simple state archive with a value-ranked trajectory frontier. This allows an LLM to reason across successful episodes to identify bottlenecks and infer a state's "potential value," providing a more principled method for balancing exploration and exploitation than the vague heuristics of prior work. The local-scale Multi-path Advantage Reflection (MAR) uses the LLM to compare divergent outcomes from multiple exploratory paths starting from the same state. This generates dense, semantic advantage signals, creating an interesting solution for credit assignment in sparse-reward environments and a clear improvement over single-trajectory reflection methods.

### Weaknesses
In the established literature of model-based reinforcement learning and robotics, a world model is understood to be a learned, predictive model of an environment's dynamics, specifically its transition function p(s' | s, a) and reward function r(s, a) (Ha & Schmidhuber (2018)). The "world models" presented in this paper function as structured, retrospective representations of experiential knowledge: the global model is a curated repository of successful trajectories, and the local model is a synthesized summary of inferred action advantages. The paper's clarity and contribution would be significantly strengthened by either explicitly justifying this terminological choice or, preferably, by adopting a different terminology.

The experimental setup provides a clear comparison by standardizing on a single LLM (GPT-4.1-mini). While we recognize the practical and budgetary reasons for this choice, a targeted ablation study, even on a subset of the Jericho games with a different LLM, would greatly enhance the paper's contribution. This would provide valuable evidence that the dual-scale mechanism is a robust and generalizable principle, rather than one that relies on the specific idiosyncrasies of the chosen model.

### Questions
Could the authors first clarify their choice to use the term 'world model', given that the mechanism is not a predictive dynamics model, as is standard in the RL literature, and this term was notably absent in its predecessor, Go-Explore?

The MAR method is interesting but I guess it would fail in a stochastic setting, do you have any intuition if there is a way to extend it to the stochastic setting?

Have the authors considered the potential of adapting the Multi-path Advantage Reflection mechanism as a powerful reward shaping or exploration module for traditional, sample-based RL agents?

### Soundness
2

### Presentation
2

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
This paper addresses the limitation of LLM agents in "hard-exploration" tasks, which are characterized by sparse rewards, deceptive local optima and large state spaces. The authors propose GLOW, a novel framework that learns at two scales to solve this. GLOW addresses the global learning challenge by maintaining a "trajectory frontier" of high-value discoveries, allowing an LLM to analyze these trajectories and select the most promising states for future exploration. To tackle local trial-and-error, its Local World Model uses a Multi-path Advantage Reflection (MAR) mechanism. MAR explores multiple paths from the state selected by the global world model and uses an LLM to infer advantage-like signals. On the Jericho benchmark, GLOW achieves SOTA performance among LLM agents and shows performance comparable to SOTA RL-based methods, with 100-800x fewer environment interactions.

### Strengths
- Clear and specific motivation
- Novel framework, combining in-context learning with established RL theories.
- Much higher sample efficiency without compromising performance, a critical advantage in environments where interaction is costly or slow.
- Effective use of LLM reasoning, leveraging LLMs for high-level reasoning to provide dense, semantic feedback from sparse rewards.

### Weaknesses
- Heavy reliance on the LLM's reasoning abilities, especially the inferred "semantic advantages" are not as rigorous as traditional, mathematically-derived advantage functions.
- The analysis of efficiency is one-sided, focusing on sample efficiency (environment steps) while obscuring the total LLM calls.
- The choice of experimental environment is not well justified.

#### Minor
- The format of reference is inconsistent.

### Questions
- How does the framework's performance differ when using LLMs with different capabilities?
- Can the authors provide a qualitative analysis of failure cases?
- Can the authors justify the reason why the experimental environments (e.g., Game of 24, Put Next To (PN) from BabyAI-Text, and The Cooking Game (CG) from TextWorld), which are used in the closely related work IGE, are not used in this work?
- Can the authors discuss the potential of the proposed method for LLM reasoning? As there are some existing works that also propose different methods to enhance the exploration ability of RLVR algorithms.

I am willing to update my evaluation if my questions and concerns are well addressed during the discussion stage.

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
5

### Summary
This paper introduces GLoW, an LLM agent framework for tackling hard-exploration problems through the combination of a global and local world model. The global world model keeps track of a trajectory frontier of states to return back to, while the local world model performs Multi-path Advantage Reflection (MAR). The authors evaluate their method on the Jericho benchmark, a well-established benchmark for text-based games, and find their method to be state-of-the-art among LLM-based approaches (but not among pure RL approaches). Finally, the authors also provide ablation studies to see how different components of their method contribute to the final performance.

### Strengths
1. The authors provide strong empirical results: they find their method to be state-of-the-art among LLM-based approaches. While they are still a bit behind the strongest RL-based approaches, the proposed method uses a lot less environment steps.
2. Conceptually, I like the authors' proposal to think about *semantic* optimism to describe exploration potential, despite being only loosely implemented in prompts.
3. I commend the authors for providing a data contamination analysis in Appendix C.
4. I like Algorithm 3 in the appendix, which shows the differences and similarities between various Go-Explore-based algorithms.

### Weaknesses
1. Why is Algorithm 1 in the main text the Go-Explore Algorithm? This feels a bit strange as this is an already existing algorithm and is nothing new.
2. Page 3: “…. preserving complete trajectories enables accurate credit assignment and value estimation in sparse-reward environments where success depends on precise action sequences” → Could the authors explain why this is the case? There seems to be no evidence or reasoning provided for this.
3. [Minor] Page 3: this sentence needs rewriting: “This is an effective choice for Jericho’s sparse reward structure, and the possibility of encountering negative rewards or terminal failures.”
4. Page 3: “to extract a set of critical global states with value annotations” → what are “critical” global states?
5. Page 3: “Once a state is chosen, we replay the stored sequence of actions to return to the state” → does this mean the method only works in the deterministic setting? If so, I think the authors should mention this as a limitation since some of the prior works (e.g. DRRN, XTX) can operate in both settings.
6. Page 5: how is the actual advantage identification that’s visualized in Figure 2 done? It doesn’t seem to be described as far as I could see. From the appendix it seems it's all done via prompting, but that's not made clear in the main text.
7. Page 9: “While traditional world models in model-based RL focused on transition dynamics (Ha & Schmidhuber, 2018; Hafner et al., 2024), recent works show that in the context of LLMs, world models are usefully expanded as mechanisms for extracting task-sufficient state representations (Tang et al., 2024; Li et al., 2024).” → I disagree with the authors here that the current work can really be thought of as world models, which are usually grounded in transition models (even in the mentioned work of Tang et al., 2024). The current work does not build any such thing, and hence I would recommend the authors change the phrasing from world model to something else that more accurately depicts the proposed algorithm.
8. The writing is somewhat weak in its presentation and contains quite a few places with spelling or grammatical errors.
9. Figure 1: what does align_{LLM} do? It doesn’t seem to be explained.
10. Method seems to require quite a bit of hand crafted prompt design based on the prompts in Appendix D.
11. Since some of the standard errors are a bit large (e.g. 20.1 for Enchanter), it would be good to run a few more seeds.

### Questions
n/a

### Soundness
3

### Presentation
1

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
The paper introduces GLoW, a dual-scale framework designed to improve exploration efficiency of LLM-based agents in hard-exploration environments such as the Jericho benchmark. It combines a Global World Model, which maintains a trajectory frontier and uses LLM reasoning to estimate both achieved and potential state values for principled state selection, with a Local World Model, which applies Multi-path Advantage Reflection (MAR) to infer semantic advantage signals from multiple rollouts under sparse rewards. Experiments show that GLoW achieves new state-of-the-art performance among LLM agents.

### Strengths
- The proposed framework achieves strong empirical performance and high sample efficiency on multiple games from benchmark.
- The proposed semantic advantage representation enhances interpretability compared to scalar rewards.

### Weaknesses
Overall, I like this paper, I just have a few minor questions:
In the global world model, how exactly is a trajectory defined? If an episode does not reach a terminal state (e.g., success or failure), does it default to a fixed-length rollout of 50 steps?
Would excessively long interactions degrade the reasoning quality of the LLM? Conversely, if the LLM is capable of longer reasoning chains, can the trajectory length be adaptively extended? Simply truncating trajectories might risk losing key contextual information.

### Questions
The paper mentions using GPT-4.1-mini for trajectory analysis. I wonder whether smaller models could achieve comparable effects. It would be insightful to include a few case studies exploring how different model influences the quality of world-model reasoning and trajectory evaluation.

### Soundness
3

### Presentation
3

### Contribution
3
