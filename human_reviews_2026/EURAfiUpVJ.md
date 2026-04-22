# MARS-SQL: A Multi-Agent Reinforcement Learning Framework for Text-to-SQL

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Translating natural language to SQL remains a significant challenge for complex queries requiring environmental interaction and self-correction. To address this, we introduce MARS-SQL, a novel multi-agent framework that combines principled task decomposition and interactive reinforcement learning (RL). Our system comprises three specialized agents: a Grounding Agent for schema linking, a Generation Agent for query generation, and a Validation Agent for final selection. The core of our framework is the Generator agent, which is trained via a multi-turn RL policy. Adopting a ReAct-style Think-Act-Observe loop, the agent iteratively generates thoughts, executes SQL actions against a live database, and revises its strategy based on execution feedback, enabling dynamic, stateful reasoning and self-correction. At inference time, we generate multiple interaction trajectories to explore diverse reasoning paths. The Verifier agent, then selects the optimal trajectory by modeling verification as a next-token prediction task and choosing the solution with the highest generation probability. This structured workflow, which pipelines specialized agents and combines interactive RL for generation with generative modeling for verification, proves highly effective for robust and accurate SQL generation. Experiments show that **MARS-SQL** achieves state-of-the-art Execution Accuracy of 77.84\% on the BIRD dev set and 89.75\% on the Spider test set.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents MARS-SQL, a multi-agent framework for Text-to-SQL. The system decomposes the task into three distinct stages: (1) a Grounding Agent for reasoning-driven schema linking, (2) a Generation Agent trained with RL (specifically GRPO) to interactively build queries in a multi-turn, ReAct-style loop, and (3) a Validation Agent trained via SFT to select the best query from multiple generated trajectories. The authors report SOTA execution accuracy for a 7B parameter model on the BIRD-dev set. The model also demonstrates strong out-of-domain generalization on the Spider test set, despite being trained exclusively on the BIRD dataset.

### Strengths
- Most part of this paper is clearly written and easy to follow
- The proposed multi-agent methodology itself appear to be technically sound and presents a principled approach to managing the complexity of the NL2SQL task.
- The execution accuracy, if can be further verified by the BIRD test set, seems to be a strong result on the BIRD bench for 7B models.

### Weaknesses
- The contribution and novelty of this paper is limited. LLM-based schema linker + multi-turn generator + LLM verifier is a well-established NL2SQL pipeline, the subset of which, if not full, has been discussed in many prior papers. Vanilla GRPO and SFT is used for tuning the agents, which are also well known methods by the research community and industry.
- Missing important technical details and insights for the research community to learn and reproduce the results, especially on how to overcome the challenges of multi-turn RL fine-turning, why the synergy between schema linker and verifier have such huge accuracy gain, and the insights into generalizability from BIRD to Spider. 
- Given the surprisingly strong results in BIRD-Bench (potentially top 1 on BIRD-bench leaderboard), it is essential to cross-validate the findings on the official BIRD test set, which is missing in this paper.

### Questions
- Given the surprisingly good results in this paper that surpass many much larger models, it will be beneficial to further include the official BIRD test results to cross-validate the generalizability of the methodology.
- Why does the grounding agent only perform table-level schema linking instead of the column-level?
- Is there any scenario in multi-turn generation where there is no error or empty results returned from database execution but the generator still decides not to finalize the answer?
- It is notoriously challenging to train LLMs with multi-turn RL. Can authors share any insights or lessons learnt from overcoming the fundamental challenges of multi-turn RL training, especially in NL2SQL tasks?
- What is the accuracy of the verifier on BIRD dev and Spider dev?
- Can authors provide more qualitative or quantitative analysis to explain *why* training on BIRD leads to SOTA performance on Spider. What specific, robust reasoning patterns did the agent learn from BIRD that transfer so effectively? Without this analysis, the extraordinary result is just a number on a table, offering no insight or lesson for the research community.
- From table 3, it looks like the grounder and verifier work extremely well when used together on BIRD-dev. However, the verifier contributes to the majority of the EX improvement on spider datasets. Can authors explain why? It looks counter-intuitive because the verifier was SFTed only on the BIRD train set and it doesn’t have schema information from the databases. How can it understand the nuances in schema and value linking errors?
- Can authors clarify the setting for “Generator Only (Base)” in table 3? Is it using the base model or simply RL tuned generator as a baseline?
- Can the pipeline generalize to more challenging datasets, such as spider 2?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces MARS-SQL, a multi-agent reinforcement learning framework designed to improve Text-to-SQL task. The authors model Text-to-SQL as a multi-agent decision process involving three specialized agents: Grounding Agent identifies relevant tables and columns from the schema using GRPO; Generation Agent employs an interactive ReAct-style reasoning loop to generate SQL queries that adapt to database feedback; Validation Agent evaluates multiple candidate SQLs and selects the best one through a learned judgment process. Extensive experiments on BIRD, Spider, and Spider-DK benchmarks demonstrate new state-of-the-art execution accuracy among open-source systems. Ablation and scaling studies confirm the contribution of each agent and the benefits of multi-round interaction.

### Strengths
1. The Grounding Agent is trained with GRPO to optimize schema selection via explicit reward feedback, rather than relying solely on supervised matching heuristics. This RL-based grounding approach allows the model to dynamically balance precision and recall of relevant tables and columns, improving robustness on complex database schemas.
2. The Generation Agent integrates a multi-turn reasoning–action–feedback loop, where each SQL generation step is conditioned on execution feedback from the database. 
3. Experiments on BIRD-dev, Spider-test show that MARS-SQL consistently outperforms prior state-of-the-art open-source systems,

### Weaknesses
1. The overall system design closely resembles XiYan-SQL, following a multi-agent pipeline with schema linking, SQL generation, and selection modules. Moreover, the use of GRPO for training the Text-to-SQL model has already been explored in Arctic-Text2SQL-R1. The conceptual innovation beyond these prior frameworks is limited, and the contribution mainly lies in engineering refinement rather than new theoretical insights.
2. The framework requires training two RL-based agents and one SFT verifier, making it resource-intensive. The paper does not quantify training time, or compute cost, nor does it compare against non-trained multi-agent systems (e.g., prompt-based approaches). 
3. The paper benchmarks primarily against Arctic-Text2SQL-R1, which is a single-model system. Comparing a multi-agent system to a single-agent baseline underrepresents the true performance gap. A fairer comparison should include multi-agent baselines, such as entries on the BIRD leaderboard or Spider 2.0-Snow benchmark.
4. The abstract and discussion emphasize that RL training improves robustness, especially in schema linking. However, the paper provides no dedicated experiment or metric explicitly evaluating robustness (e.g., under schema perturbations, synonym replacements, or unseen database structures). As a result, the robustness claim lacks empirically supported.

### Questions
1. Could you clarify how many RL training steps were used for the Grounding Agent and Generation Agent respectively? Additionally, how did the reward and rollout length evolve during training? Were there stability or convergence patterns observed?
2. For the multi-turn SQL Generation Agent, how did the actual number of reasoning turns used during training vary across examples? Beyond the maximum turn limit T, was there any minimum turn constraint or adaptive stopping rule applied?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces MARS-SQL, a novel multi-agent reinforcement learning framework for Text-to-SQL, designed to handle complex queries requiring environmental interaction and self-correction. The system decomposes the task into three stages, each handled by a specialized 7B agent: (1) a Grounding Agent for schema linking, trained with GRPO; (2) a core Generation Agent, trained with multi-turn RL, which adopts a ReAct-style (Think-Act-Observe) loop to interact with a live database, enabling it to dynamically revise its SQL based on execution feedback; and (3) a Validation Agent, trained via SFT, to select the best final query from multiple generated trajectories. This Validation Agent reframes selection as a next-token prediction task, choosing the trajectory with the highest probability of being correct. The framework demonstrates state-of-the-art results, achieving 77.84% execution accuracy on the BIRD dev set and 89.75% on the Spider test set.

### Strengths
- **S1.** The interactive Generation Agent is the main strength of this work. By training it with RL in a "Think-Act-Observe" loop that interacts with a live database, the agent can learn to dynamically self-correct from execution errors (e.g., "no such table" or "empty result"), which is a significant step beyond static, single-pass generation models.
- **S2.** The design of the Validation Agent is a clever and efficient approach to selection. Reframing verification as a generative next-token prediction task (P("Yes")) avoids the need to train a separate, complex classifier and effectively leverages the generative model's own understanding to rerank trajectories.
- **S3.** The empirical results are state-of-the-art. Achieving 77.84% on BIRD and 89.75% on Spider (without Spider training data) significantly outperforms prior open-source methods, demonstrating the effectiveness of the proposed framework.

### Weaknesses
- **W1. Missing Efficiency and Practicality Analysis:** A significant weakness is the complete omission of any efficiency analysis. Text-to-SQL is often a real-time, user-facing task where latency is critical. MARS-SQL is a complex, multi-stage workflow involving numerous LLM calls (Grounder calls, multi-turn Generation rollouts, and multi-sample Validation calls). This implies a potentially high end-to-end latency and token cost. Without experiments analyzing this cost-accuracy trade-off against other SOTA methods, the practical applicability of the system is unknown.
- **W2. Misleading Parameter Count:** The presentation of model size in Tables 2 and 3 is misleading. The paper lists "MARS-SQL" under "7B" parameters. However, the framework trains and utilizes three separate 7B models (Grounder, Generator, Verifier), making the total parameter count 21B. This is an unfair comparison to other methods listed as 7B or 14B and incorrectly suggests higher parameter efficiency.
- **W3. Clarity Issues in Experimental Reporting:** The paper's presentation suffers from a lack of clarity and missing details, which hinders reproducibility. (a) The terms used in the ablation study (Section 4.3, Figure 3) are ambiguous. The paper does not provide explicit definitions for the evaluation metrics "Greedy", "Selected", and "Best of N". (b) The experimental setup (Section 4.1) lacks crucial hyperparameters for each agent. For example, the number of rollouts (G) for the Generation Agent and the number of stochastic reasoning rounds (M) for the Validation Agent's probability estimation are not specified.

### Questions
- **Q1. Efficiency and Cost-Benefit Trade-off:** My primary concern is the practical viability of this multi-agent workflow. Could the authors provide a detailed efficiency analysis? Specifically, what is the end-to-end latency and total token cost for an average query, and how does this cost-accuracy trade-off compare to SOTA baselines?
- **Q2. Parameter Count:** In Table 2, MARS-SQL is listed as "7B". To confirm, does this framework use three separate 7B models (Grounder, Generator, Verifier), making the total parameter count 21B? If so, this should be corrected for a fair comparison.
- **Q3. Clarification of Experimental Details:** Could you please clarify the experimental details mentioned in W3?
- **Q4. BIRD-Test Generalization:** The paper claims strong out-of-domain generalization by testing on the Spider and Spider-DK benchmarks. However, the Spider dataset is often considered relatively simple. Since the model was trained on the BIRD dataset, the most crucial measure of its generalization is its performance on the BIRD-Test set. Have the authors submitted this model to the official BIRD Leaderboard? If so, could you please report the BIRD-Test execution accuracy to fully validate the model's performance?

### Soundness
3

### Presentation
3

### Contribution
3
