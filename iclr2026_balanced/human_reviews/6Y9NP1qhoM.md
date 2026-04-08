## Human Reviewer 1

### Summary
This paper studies how misinformation affects Multi-Agent Systems. Specifically, the paper introduces MISINFOTASK which evaluates how MAS defends against misinformation injection. It also proposes ARGUS, a training free defense that (i) locates the most critical communication channels where misinformation is likely flowing, then (ii) performs “goal-aware” persuasive rectification using CoT-style reasoning to counter and correct it.

### Strengths
1. The paper is studying a meaningful question that how misinformation propagate within MAS after information injection attacks.
2. The paper builds a complete benchmark including the dataset, setup and evaluation, and also propose a method to tackle such problem.

### Weaknesses
1. The paper first uses LLM to generate tasks, and then manually filter out tasks. However, it is unclear whether these tasks align with real-world tasks and whether they are diverse enough to be used, since the same prompt is being used to generate tasks over and over.
2. The dataset only contains 108 tasks, which is small, especially the main content is crafted by LLM.
3. The evaluation employs an LLM judge. Although LLM judge could be useful, the paper doesn't have any sanity check of it, for example, compare it with manual scores.
4. While the paper states that one limit of ARGUS is efficiency and cost, there is no such measurement of its cost compared to other methods. This call into question whether such method is practical, that it could be trading massive resource for performance boost.

### Questions
The questions below correspond to each point of the weakness.

1. Can you explain how you ensure diversity and real-world utility of these tasks?
2. (Please see weakness)
3. Do you have any analysis of the LLM judge (human agreement)? 
4. Can you provide how much additional cost of the proposed method compared to other methods?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
2

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper studies how LLM-based MAS are vulnerable to hidden misinformation attacks. It introduces MISINFOTASK, a dataset with 108 complex tasks used to test the robustness of MAS. It also presents ARGUS, a defense framework that does not need extra training.

In the first stage, called Adaptive Localization, ARGUS finds the key communication channels. In the second stage, called Goal-aware Persuasive Rectification, it places a corrective agent on these channels. The agent uses chain-of-thought reasoning to break down messages, detect suspicious claims, compare them with its own knowledge, and create persuasive corrections.

Tests show that ARGUS lowers misinformation toxicity by 28.17% on average and raises the TSR by 10.33% compared to systems without defense.

### Strengths
-  The design is original. The synthesis of static topological analysis with dynamic, semantic re-localization based on an inferred misinformation goal is a clever and novel approach. Furthermore, the concept of a "persuasive" corrective agent using CoT reasoning is more advanced than simple fact-checking or edge-pruning defenses. 

- The MISINFOTASK dataset is a useful contribution. 

- Experiments are thorough. This paper tests across multiple LLM families, multiple modern attack vectors and defense baselines. The analysis is detailed. 

- The paper is written with clarity. 

- Code is attached. Seems well

- This work addresses a challenge to the adoption of MAS.

### Weaknesses
- The paper acknowledges the limit of computational overhead & cost but does not quantify the overhead.

- The initial localization step relies on Edge Betweenness Centrality, which is computationally expensive ($O(V \cdot E)$) and does not scale well to large graphs. This would be a bottleneck for MAS with tens or hundreds of agents.

- The paper's definition of misinformation is "content that contradicts the factual knowledge implicitly stored in the parameters of an LLM." Thus "Internal Knowledge Resonance" relies on this. This makes ARGUS vulnerable to misinformation about dynamic, time-sensitive information pre-training data. 

- The choice of $k=M-1$ seems to be a brute-force approach to ensure all critical channels are covered, but it maximizes the cost. How performance degrades with a more sparse and realistic $k$.

### Questions
1. See weakness

2. How sensitive is the adaptive re-localization mechanism to the weights $\alpha=0.2, \beta=0.2, \gamma=0.6$? Was a sensitivity analysis or sweep performed to arrive at these values?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper introduced MISINFOTASK, a novel dataset  featuring complex, realistic tasks designed to evaluate MAS robustness against such threats. In addtion, they proposed ARGUS, a two-stage, training-free defense framework leveraging goal-aware reasoning for precise misinformation rectification within information flows. Experiment results show that ARGUS exhibits significant efficacy across various injection attacks for misinformation alleviation and task success rate improvement.

### Strengths
1. The paper introduces a novel and practical dataset that enables rigorous evaluation of misinformation robustness in multi-agent systems.
2. The proposed ARGUS framework effectively mitigates misinformation without additional training and improves task completion.
3. The work addresses an important and timely problem of misinformation security that has been largely overlooked in prior MAS research.
4. The paper is clearly written, well organized, and supported by comprehensive experimental validation.

### Weaknesses
1. It is recommended that the authors revise the structure of Table 1 so that the model names appear in the first column and the defense types in the second column, making the table layout clearer and improving comparability across models.

2. The paper would benefit from a clearer specification of the threat model, detailing attacker goals, capabilities, and assumptions, which would help strengthen the discussion on the security significance of misinformation propagation in MAS.

3. In MAS, various intelligent agents have different motivations, or mindsets. They have competitive, compromising, and accommodating personalities to achieve their goals. Therefore, how does this information and misinformation spread? In addition, how can the spread of illusions and misinformation be distinguished?

4. The contribution and self-containment of the paper could be improved if the authors provided a more detailed description of the structure and content of MISINFOTASK. In particular, including concrete examples or a summary of task types and misinformation patterns would help readers better understand the dataset’s design and relevance.

5. In the experimental analysis section, it is recommended that the authors add a discussion on the MAS topology

6. How does ARGUS resist the attack of Misinformation in existing MAS? 

7. it is recommended that the author reflect the results and discussion of the Appendix in the main text.

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
5

---

## Human Reviewer 4

### Summary
This paper makes two primary contributions. First, they introduce MISINFOTASK, a new benchmark dataset comprising 108 tasks designed specifically to evaluate MAS robustness against covert misinformation. The dataset includes plausible but fallacious arguments for each task, along with ground truth information. Second, they propose ARGUS, a training-free, two-stage defense framework.

### Strengths
1.  This paper evaluates the robustness of MAS against misinformation using  Tool Injection, Prompt Injection and RAG Injection.

2.  This paper builds a corrective agent to guard the misinformation in MAS.

### Weaknesses
1. The initial localization phase relies exclusively on Edge Betweenness Centrality, a purely topological metric, to identify critical channels. This strategy assumes that misinformation will travel along the most central paths. However, a sophisticated adversary could easily evade this initial detection by injecting misinformation through less central, seemingly unimportant edges, allowing it to propagate for at least one full round before content-aware mechanisms are activated.

2. The effectiveness of the adaptive re-localization hinges entirely on the corrective agent's ability to accurately infer the misinformation's intent-driven goal $g_{mis}$. If this inference is flawed or inaccurate, the subsequent calculation of information relevance $Score_{rel}$ will be based on an incorrect premise.

3. The entire defense rests on the strong assumption that the corrective agent's internal, parameterized knowledge is factually superior to the incoming information.

4. The comprehensive score for channel importance is calculated using fixed weights (α=0.2, β=0.2, γ=0.6). These static values may not be optimal across different MAS topologies, task types, or evolving adversarial strategies.

5. The paper states that the dataset underwent a rigorous manual review process after AI generation. However, the methodology lacks transparency. Crucial details are omitted, such as the qualifications of the human experts, the number of reviewers per entry, the specific guidelines for filtering content, and any inter-annotator agreement metrics. This makes it difficult to independently assess the quality, consistency, and objectivity of the final dataset.

6. The dataset's size of 108 entries is relatively small for a benchmark intended to test broad generalization. Furthermore, since the initial data was generated by a single LLM (GPT-4o), there is a risk that the misinformation is inadvertently tailored to the specific failure modes of that model family.

7. The dataset generation prompt explicitly targets misinformation that contradicts well-established facts likely learned during pre-training. This approach overlooks the challenge of dynamic misinformation related to recent events, evolving topics, or time-sensitive data that falls outside an LLM's static knowledge base. This limits the dataset's applicability to real-world scenarios where misinformation is often timely and ephemeral.

### Questions
1. The number of monitored channels was set to k=M-1, meaning nearly every channel was monitored. This implies significant computational and financial costs. Have you considered or experimented with more efficient implementations, such as a sampling strategy for monitoring or a lightweight initial check to triage messages before triggering the full, resource-intensive CoT analysis?

2. The corrective agent's rectification strategy relies on a CoT prompting method guided by heuristic principles like "root cause analysis" and "cognitive reframing". Could you discuss the generalizability of this prompting strategy?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
4