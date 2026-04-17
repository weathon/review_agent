# Refining Bias and Reward in LLM Recommender Agents through Meta-Controlled Tool Invocation

- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
Large language model (LLM) agents have recently been brought to recommender systems given their flexible capability of tool use. Although existing approaches adopt the reasoning and acting paradigms for profiling, planning, and memory augmentation, they remain ad hoc and overlook core recommendation challenges in agent-environment interactions, including debiasing and reward estimation in offline learning scenarios. In this paper, we introduce BARO (Bias And Reward Optimization), a meta-controlled, tool-augmented LLM agent framework that explicitly addresses these challenges. BARO employs a two-stage recommendation process: a coarse recommender generates a candidate slate based on user history, and a meta-controller adaptively invokes three specialized tools to refine the recommendation results: a bias detector assesses and mitigates bias in the candidate set, a reward estimator calibrates noisy offline rewards, and an action grounder selects final recommendations from the candidate pool. This design injects bias correction and reward refinement directly into the agent’s decision loop in the recommendations. Empirical results on two benchmark datasets demonstrate that BARO achieves consistent improvements over state-of-the-art methods in metrics such as accuracy, diversity, and fairness. The code will be made publicly available upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes BARO (Bias And Reward Optimization), a meta-controlled, tool-augmented LLM agent framework for recommender systems. The framework consists of a two-stage pipeline: a frozen LLM acts as a coarse recommender to generate a candidate slate, while a meta-controller adaptively invokes three tools—bias detector, reward estimator, and action grounder—to refine recommendations. The goal is to address bias mitigation and reward calibration in offline learning scenarios. Experiments on Steam and Amazon Book datasets are provided, showing improvements in some metrics over baselines.

### Strengths
1. The paper addresses an important problem — how to mitigate bias and improve reward reliability in LLM-based recommendation.

2. The general idea of combining LLM agents and meta-control for recommendation is interesting and in line with recent trends of using agentic reasoning for RecSys.

### Weaknesses
1. The paper criticizes post-training as “computationally costly,” but its own framework introduces both SFT and RL stages, which are even more expensive. There is no analysis or discussion showing that BARO reduces cost compared to prior post-training methods. Hence, the motivation is not convincingly addressed.

2. The design choice—using a frozen LLM for coarse recommendation and a downstream collaborative recommender to generate reward signals—is questionable. Typically, LLMs lack global-level item ranking capability, while collaborative recommenders are more effective in retrieval. The proposed division of labor thus appears reversed and unintuitive, weakening the conceptual justification.

3. There are several critical missing details in the method and experimental settings.
(1) It is unclear how the coarse recommender prompt is constructed, and how item candidates are ensured to belong to the dataset’s valid item set. 
(2) How are LLMs prompted when the item pool exceeds tens of thousands? Are items pre-filtered or indexed? How is fairness ensured when comparing LLM-based models with sequential baselines that have different candidate pools?
(3) Metrics such as IoI/IoR lack calculation details within this framework.
(4) It is unclear how sequential recommendation baselines are implemented in the current setup.

4. Both bias and reward checkers improve results, but the paper also emphasizes that the meta-controller can decide whether to invoke them. Without experiments comparing “always-check” versus “meta-controlled” usage, the proposed meta-decision mechanism seems unnecessary and unvalidated.

5. The reported results show SASRec performing drastically worse than Caser and GRU4Rec, which contradicts prior literature and suggests improper baseline setup or evaluation. The authors do not analyze or justify this anomaly.

6. The reported accuracy values are extremely high (e.g., Caser reaching 0.977), which raises concerns about whether the dataset or task setup is meaningful.

7. The refined reward estimation modules are under-explained. It remains unclear how refined rewards are obtained or why this additional deisgn is necessary.

### Questions
See weakness

### Soundness
2

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
5

### Summary
The paper proposes a BARO method to solve the problems of LLM-based recommender systems: debiasing and reward estimation. Empirical results show the method works well.

### Strengths
The paper is well-motivated.

The paper is readable.

### Weaknesses
1. One of the motivations of the paper is that LLM-based recommender systems suffer from *interaction bias*, but the paper does not give examples, quantitative analysis, a definition or qualitative explanation, nor does it explain the harms caused by interaction bias.
2. Similarly, *reward inaccuracies* are not illustrated with examples or quantitative analysis, and their harms are not shown.
3. What is the role of the “world model” in Section 4?
4. Figure 1 needs improvement. Many arrows in the figure will confuse readers — I suggest adding explanatory labels on the arrows.
5. The abstract claims “BARO achieves consistent improvements over state-of-the-art methods in metrics such as accuracy, diversity, and fairness,” but I do not see diversity or fairness metrics in the experiments.
6. The paper aims to solve “interaction bias” and “reward inaccuracies” in LLM-based recommender systems, but I find no analysis of these problems in Experiments. Therefore, I am skeptical that the proposed method actually addresses these two issues.

### Questions
see weakness.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes BARO (Bias And Reward Optimization), an agentic recommendation framework for addressing the debiasing and reward estimation in recommender systems. Specifically, a meta-controller dynamically calls three tools (i.e., bias detector, reward estimator, and action executor) to correct bias and stabilize reward estimation. Experiments on extensive datasets verify the effectiveness of the proposed method.

### Strengths
**Timely and well-motivated study for introducing agentic AI in recommender systems.** 
The paper addresses a pertinent issue at the intersection of *agentic AI* and recommender systems. Moreover, the unique question of bias is an important issue in recommender systems, which is a reasonable motivation for adopting agents in recommendation.

**Novel modular design.** 

The two tools (i.e., bias detector and reward estimator) proposed in this work are practical for addressing the debiasing and reward estimation issues in recommendation. 

**Solid experiments and clear presentation.** 

Experiments on multiple datasets, together with detailed ablation studies, convincingly show the effectiveness of each module. The writing is also clear and coherent, making the whole paper easy to understand and easy to follow.

### Weaknesses
**Lack of discussion about the user simulator.**

It would be important to give more details about how the user simulator is trained. Additionally, it would be important to add more details about the reliability of the user simulator.

**Lack of case study and demonstration.**

It would be better to illustrate how the proposed method works in practice, especially with some demonstration and examples. For example, how does this method call the two tools under which condiction?

### Questions
Refer to the weakness part.

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
3

### Summary
The paper proposes BARO, a meta-controlled tool-invocation framework that aims to reduce recommendation bias and calibrate reward learning for LLM-driven recommender agents. The system adds a controller that chooses when and how the agent calls tools that (i) diversify or rebalance candidate sets and (ii) reshape reward signals used for agent tuning.

### Strengths
1. The paper presentation is clear and easy to follow.

2. The paper focuses on two pain points for LLM recommender agents: exposure/popularity bias and reward misspecification. The former is widely documented to degrade user experience and diversity, while the latter can lead to unstable offline-to-online transfer.

### Weaknesses
1. The paper states that “most agentic frameworks were not designed specifically for RecSys,” yet the proposed method mainly adds two recommendation-specific tools while keeping a generic ReAct-style agent loop otherwise unchanged. It is not very convincing that the proposed agentic framework is recommendation-specific as replacing the tools with other domain tools seems to not affect the framework's functionality.

2. The design appears to host and orchestrate multiple models/modules (planner LLM, bias tool, reward tool, possible rerankers), but there is no cost/latency analysis and comparison with baseline methods. Therefore it is not clear how much performance gain comes from the agentic framework design instead of the use of more test-time compute.

3. The bias handling module lacks novelty. The so-called “bias detector” is just a frozen LLM used as a tool without any new learning or calibration mechanism. The evaluation focuses narrowly on exposure/popularity bias and does not cover other key aspects in RecSys such as fairness, calibration, or long-tail coverage. Relying on a frozen LLM for bias detection also introduces its own biases and subjectivity.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
