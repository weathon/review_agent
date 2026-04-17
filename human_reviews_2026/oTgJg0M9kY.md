# Controllable Logical Hypothesis Generation for Abductive Reasoning in Knowledge Graphs

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 4

## Abstract
Abductive reasoning in knowledge graphs aims to generate plausible logical hypotheses from observed entities, with broad applications in areas such as clinical diagnosis and scientific discovery. However, due to a lack of controllability, a single observation may yield numerous plausible but redundant or irrelevant hypotheses on large-scale knowledge graphs. To address this limitation, we introduce the task of controllable hypothesis generation to improve the practical utility of abductive reasoning. This task faces two key challenges when controlling for generating long and complex logical hypotheses: hypothesis space collapse and hypothesis reward oversensitivity.
To address these challenges, we propose **CtrlHGen**, a **C**on**tr**ollable **l**ogcial **H**ypothesis **Gen**eration framework for abductive reasoning over knowledge graphs, trained in a two-stage paradigm including supervised learning and subsequent reinforcement learning.
To mitigate hypothesis space collapse, we design a dataset augmentation strategy based on sub-logical decomposition, enabling the model to learn complex logical structures by leveraging semantic patterns in simpler components.
To address hypothesis reward oversensitivity, we incorporate smoothed semantic rewards including Dice and Overlap scores, and introduce a condition-adherence reward to guide the generation toward user-specified control constraints.
Extensive experiments on three benchmark datasets demonstrate that our model not only better adheres to control conditions but also achieves superior semantic similarity performance compared to baselines. Our code is available at https://github.com/HKUST-KnowComp/CtrlHGen.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Abductive reasoning in knowledge graphs facilitates the creation of plausible logical hypotheses to explain observed entities. This article presents a novel and controllable abductive reasoning approach to overcome the limitations of existing methods, which often lack flexibility in real-world applications. To address challenges including overly lengthy hypotheses due to hypothesis space collapse and the high sensitivity of generated outputs, the authors introduce a sub-logic decomposition-based data augmentation technique, combined with meticulously crafted reinforcement learning reward signals. Comprehensive experiments show that the proposed CtrlHGen method effectively resolves these issues, significantly improving the controllability and semantic quality of hypothesis generation.

### Strengths
1.The controllable abductive reasoning approach for knowledge graphs introduced in this article is both innovative and practically significant. By generating plausible hypotheses tailored to user-specific interests, this method holds substantial value for applications such as recommendation systems and scientific discovery.

2.The writing of this paper is clear and easy to understand. This paper naturally puts forward the significance of controllability, the challenges of achieving controllability and how to solve these challenges.

3.The paper presents convincing experimental results. Evaluations across five distinct settings convincingly demonstrate the proposed method's ability to achieve controllability. Furthermore, ablation studies robustly validate the effectiveness and contributions of the approach.

4.The case studies reveal unexpectedly robust results. When presented with control conditions unrelated to the observed entities, the model adeptly introduces logical disjunctions (OR) to concurrently meet both the control constraints and maintain semantic interpretability. This behavior is both intriguing and highly significant.

### Weaknesses
1.The author did not explore whether some naive methods could be applied to controllable abductive reasoning tasks. For instance, models based on link prediction in KG, or search methods based on subgraph matching.

2.It seems to me that the poor performance of LLMs is very interesting. But the authors should provide a specific case, including the 2-hop subgraph obtained by the large model and its generation results, so as to more clearly demonstrate the reasons for the poor performance of the large model.

3.Following W1 and W2, the poor performance of LLMs may be limited by some naive searches. The author should make an analysis on this point. For example, whether the 2-hop subgraph can contain valid answers.

### Questions
My questions have already been listed in Weakness.

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
4

### Summary
This paper addresses the lack of controllability in abductive reasoning on knowledge graphs (KGs), where existing methods often generate numerous plausible but irrelevant hypotheses. The authors introduce the new task of controllable hypothesis generation, which allows users to specify constraints on both the semantic content (e.g., focusing on "pathology" or "treatment") and the structural complexity (e.g., the length) of the desired hypotheses. They identify two primary challenges in generating long, complex hypotheses: "Hypothesis Space Collapse" (a sharp drop in the number of valid complex hypotheses) and "Hypothesis Reward Oversensitivity" (unstable reinforcement learning due to strict rewards like the Jaccard score). To solve these problems, they propose CtrlHGen, a framework trained in two stages (supervised learning and reinforcement learning). CtrlHGen uses a "sub-logical decomposition" strategy to augment the dataset, helping the model learn complex structures. It also introduces a novel reward function that combines "smoothed semantic rewards" (using Dice and Overlap scores) with a "condition-adherence reward" to ensure generated hypotheses are both accurate and follow the specified constraints. Experiments on three benchmark datasets demonstrate that CtrlHGen significantly outperforms baselines, including modern LLMs, in both adhering to control conditions and maintaining high semantic similarity.

### Strengths
1. Novel and Practical Problem Formulation

The paper is the first to formally introduce and tackle the task of "controllable abductive reasoning" in KGs. This addresses a significant and practical limitation of prior work (AbductiveKGR ), where a single observation can lead to an unmanageable number of "plausible but irrelevant hypotheses". By defining clear control mechanisms for semantic content and structural complexity , the paper moves the field toward greater practical utility in real-world applications like clinical diagnosis or scientific discovery.

2. Insightful Diagnosis of Core Challenges

A key contribution is the clear identification and illustration (in Figure 2) of two non-obvious challenges: "Hypothesis Space Collapse" and "Hypothesis Reward Oversensitivity". Diagnosing why generating complex, controlled hypotheses is difficult (i.e., fewer valid complex examples exist, and minor errors in them cause massive reward drops) provides a strong, logical foundation for the paper's proposed solutions.

3. Novel and Effective Augmentation Strategy

The "dataset augmentation strategy based on sub-logical decomposition"  is an innovative solution to the "Hypothesis Space Collapse" problem. By programmatically breaking down complex hypotheses into simpler, semantically related sub-components, the authors create a richer training environment. This allows the model to learn the building blocks of complex logic, which the ablation study confirms significantly improves performance, especially for complex logical patterns.

4. Robust and Well-Justified Reward Function

The paper intelligently designs a reward function to overcome "Hypothesis Reward Oversensitivity". Recognizing that the Jaccard score is too "strict" , they smooth the reward landscape by incorporating Dice and Overlap scores. Furthermore, the inclusion of a "condition-adherence reward"  directly optimizes for the new task's constraints. The ablation study (Table 3) effectively demonstrates that this dual-objective reward function successfully balances semantic accuracy with constraint adherence.

5. Extensive and Solid Experimentation

The authors validate their CtrlHGen model thoroughly. They experiment on three standard KG datasets , test against the relevant SOTA baseline (AbductiveKGR) , and include a comparison against four modern LLMs (GPT-4o, Kimi K2, etc.). Their evaluation is comprehensive, using distinct metrics for both semantic similarity (Jaccard, Dice, Overlap) and condition adherence (Accuracy). The ablation studies (Figure 5, Table 3) are particularly strong, as they isolate and confirm the positive impact of their two main technical contributions.

### Weaknesses
1. Limited Scope and Definition of "Control"

The paper defines "semantic control" as providing a specific entity or relation from the target hypothesis and "structural control" as a predefined pattern or count . This is a good first step, but it's a fairly rigid and limited form of control. In a practical application (like clinical diagnosis ), a user might want to provide more abstract semantic guidance (e.g., "focus on infectious diseases but not viral ones") or compositional constraints that are not covered by this framework.

2. High Complexity of the Training Pipeline

The proposed CtrlHGen framework involves a complex, multi-stage pipeline: (1) sampling pairs, (2) sub-logic augmentation, (3) unconditional supervised training, (4) conditional supervised fine-tuning, and (5) reinforcement learning with a specific optimizer (GRPO). This pipeline has numerous components and hyperparameters (e.g., $\lambda_{1}, \lambda_{2}, \lambda_{3}, \alpha$), which could make the method difficult to reproduce, tune, and apply to new KGs or different control types.

3. Baseline Comparison Is Not Perfectly "Apples-to-Apples"

The primary baseline, AbductiveKGR, is run in an "unconditional" setting  and is then compared to CtrlHGen's conditional results. While this demonstrates the benefit of adding control, it doesn't fully isolate the benefit of CtrlHGen's method. A stronger baseline would have been to adapt AbductiveKGR to be conditional (e.g., by prepending the condition tokens to its input) and then showing that it fails, thereby proving that the sub-logic augmentation and novel reward function are necessary.

4. Scalability to Truly Large-Scale KGs is Untested

The paper notes that on large KGs, the number of plausible hypotheses "grows dramatically". The experiments are conducted on standard benchmarks (DBpedia50, WN18RR, FB15k-237), which are small-to-medium in size. The paper does not provide evidence that the sampling, augmentation, and RL-tuning processes would be computationally feasible or effective on modern, web-scale KGs containing billions of facts.

5. Analysis of LLM Failures is Plausible but Incomplete

The paper correctly identifies that LLMs perform "very poor" and offers plausible explanations, such as a lack of understanding of structured data and knowledge conflicts. However, the LLMs were only provided with a "2-hop subgraph" as context. This is a very limited and simple prompting strategy. It's possible that the LLMs' failure is at least partially due to this weak context-providing method, rather than an inherent inability to perform the task. More advanced prompting or retrieval-augmentation strategies would be needed for a more definitive conclusion.

### Questions
The current system is one-shot. A more practical system would be interactive. A user could receive a generated hypothesis, provide feedback (e.g., "This is too complex," "This part is wrong," "Explore this entity more"), and the model would use this feedback to refine the hypothesis, making the process a collaborative dialogue. Would it be possible to achieve the interactive system?

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
This paper introduces CtrlHGen, a framework for controllable logical hypothesis generation in abductive reasoning over knowledge graphs. The authors identify two key challenges in this setting. (1) Hypothesis space collapses when generating long and complex logical hypotheses. (2) Reward oversensitivity when reinforcement learning overly biases toward specific metrics. To address these, the paper proposes sub-logical decomposition for dataset augmentation and smoothed semantic rewards to balance semantic quality with user-specified control constraints. Experiments on three benchmark datasets demonstrate improved controllability and semantic similarity compared to baselines.

### Strengths
- The paper clearly motivates the need for controllable abductive reasoning, which is underexplored in KG reasoning.


- The introduction of smoothed semantic rewards and condition-adherence reward is a thoughtful way to mitigate oversensitivity.

### Weaknesses
- Unclear problem definition. It is unclear to me the difference between abductive KG reasoning and rule-based link prediction. It would be better to add more explanations.

- Method limitation. It seems that the data sampling method relies on predefined logical patterns. How can the proposed method be generalized to other different KGs?


- Evaluation limitations. While semantic similarity and adherence metrics are reported, the paper could benefit from more qualitative analysis of generated hypotheses (e.g., case studies, error analysis). It is not entirely clear whether the chosen baselines represent the strongest possible competitors in controllable generation or abductive KG reasoning.

### Questions
- How does CtrlHGen handle contradictory or mutually exclusive hypotheses under control constraints?

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
4

### Summary
In this paper, the authors introduce a model called CONE, which aims to generate logically consistent hypotheses from a given premise. They are allowing fine-grained control over the desired logical relationship. The key innovation is the integration of natural logic into the generation process. In this proposed method, rather than relying solely on learned patterns or black-box transformers, they decompose logical relationships at the token level and apply compositional rules to reason about the entire sentence. They have trained a RL model using the curated data. The results demonstrate superiority compared to raw LLMs.

### Strengths
1-	This paper is the first to formally define controllable abductive reasoning over knowledge graphs (KGs). It is an important task as it can be used in many downstream tasks where KG statements, such as causal-effect or factual statements, exist or could be inferred.
2-	The proposed framework is novel and addresses the challenges of the task. Authors have introduced a two-stage framework for sub-logic decomposition for data and an RL with a dual reward design that balances semantic alignment and control adherence.
3-	The paper is well-structured, uses clear formulations, and provides a detailed training setup, metrics, and open-source code, supporting reproducibility (Appendices A & B).

### Weaknesses
1-	More baselines are needed. Current baselines cannot really demonstrate the true contribution of the paper. I recommend that authors include methods that try to address the same challenge. For instance, Logic-Gen [1] can be a good candidate:
Asai, A., & Hajishirzi, H. (2020). Logic-guided data augmentation and regularization for consistent question answering. arXiv preprint arXiv:2004.10157.
2-	Some of the LLMs that are used for comparison do not have reasoning capability. For instance, while GPT 4o may have general reasoning power, the thinking mode is not included, the same for Kimi K2. For such a task where reasoning is important, it is better to use more advanced models. GPT 5, GPT 5-mini, Qwen 3 with Thinking tokens enabled, etc., are all good options. I am not sure what the rationale was behind choosing Kimi K2, as it is mostly good at coding benchmarks. I recommend that authors briefly explain the reason for their choice of LLMs.
3-	The way context is provided to baseline LLMs is questionable. It seems the entire 2-hop subgraph is dumped into the prompt, and a non-thinking LLM was supposed to reason based on that. It would be much better if at least a RAG set-up were used to make it more realistic. Or if raw LLM was the purpose, as mentioned in the second point, a more advanced LLM should be used.

### Questions
Is the thinking mode activated for Grok-3? It would be great if authors explained the experimental setup of the LLMs in detail. Often in such tasks, specific hyperparameters can change the performance. For instance, in deterministic scenarios, like diagnosis, it is recommended to use temperature 0 to avoid randomness. It would be great if authors include such consideration in their tests.

### Soundness
2

### Presentation
4

### Contribution
3
