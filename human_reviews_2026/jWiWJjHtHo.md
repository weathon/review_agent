# RLRF: Competitive Search Agent Design via Reinforcement Learning from Ranker Feedback

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Competitive search is a setting where document publishers modify them to improve their ranking in response to a query. Recently, publishers have increasingly leveraged LLMs to generate and modify competitive content. We introduce Reinforcement Learning from Ranker Feedback (RLRF), a framework that trains LLMs using preference datasets derived from ranking competitions. The goal of a publisher (LLM-based) agent is to optimize content for improved ranking while accounting for the strategies of competing agents. We generate the datasets using approaches that do not rely on human-authored data. We show that our proposed agents consistently and substantially outperform previously suggested approaches for LLM-based competitive document modification.  We further show that our agents are effective with ranking functions they were not trained for (i.e., out of distribution) and they adapt to strategic opponents. These findings provide support to the significant potential of using reinforcement learning in competitive search.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a framework, RLRF, for training LLM agents to engage in competitive search—a task where agents strategically edit documents to achieve higher search rankings. The method involves simulating a multi-agent competitive environment to generate a preference dataset, which is then used to fine-tune an agent using DPO. The experiments demonstrate that this dynamically trained agent (RA) outperforms prompt-based baselines and shows an ability to generalize to unseen rankers. The authors also report that their agent, despite being trained solely on a ranking objective, exhibits better content faithfulness than the baselines.

### Strengths
1.	Clear Problem Formulation and Comprehensive Empirical Validation: The paper addresses a timely and interesting problem by formalizing strategic content generation as a learnable, multi-agent task. The experimental setup is comprehensive, providing a solid benchmark for future work in this area. 

2.	Identifies Interesting Emergent Phenomena: The study successfully identifies two non-obvious and valuable phenomena: (1) the agent's emergent tendency to preserve content faithfulness despite a purely competitive objective, and (2) the asymmetric nature of knowledge transfer between different ranking functions. These findings could inspire follow-up research.

### Weaknesses
1.	Insufficient Conceptual and Methodological Novelty: The primary concern lie in the paper's lack of significant novelty. The core idea—applying a reinforcement learning-style algorithm to a game-theoretic problem—is a well-established paradigm. The methodology is a straightforward combination of existing techniques: using multi-agent simulation (akin to self-play) to generate data, followed by a standard preference alignment algorithm (DPO). While this "recipe" is effective, it does not introduce a novel algorithmic component, or a fundamentally new way of thinking about the problem. The central hypothesis that training on dynamic, competitive data (DG) is superior to training on static, isolated data (SG) is also obvious and confirms an existing intuition rather than revealing a new insight. 

2.	Superficial Analysis of Key Findings: While the paper reports interesting emergent behaviors (faithfulness and asymmetric transfer), it fails to provide a deep investigation into mechanism behind these phenomena. For instance, explaining the emergent faithfulness could lead to profound insights about the implicit biases of modern neural rankers or the nature of strategic alignment. Without this deeper analysis, the findings remain intriguing observations. I recommend the authors could explore more on this aspect.

3.	Unaddressed Sim-to-Real Gap: The entire evaluation is conducted in a highly idealized simulation that abstracts away the most difficult aspects of the real-world problem: non-stationary and adversarial rankers, black-box algorithms, and sparse, delayed, and noisy feedback. The paper's claims of robustness are based on transferring between fixed, known-type rankers, which does not adequately address the challenges of a constantly evolving, real-world search ecosystem. A more critical discussion of these limitations is needed to properly contextualize the work's practical applicability.

### Questions
See Weaknesses

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
2

### Summary
The paper studies the problem of editing documents via LLMs to sequentially improve their ranking in a search engine given a query, a setting defined as "competitive search". They propose to use the Ranking Feedback (RF) to finetune agents for the task, based on past rankings. Concretely, at each round, the highest and lowest performing document edits are stored and used to finetune the LLMs via the DPO algorithm. Experiments are performed in multi-agent simulations (where multiple agents compete for the highest rank) and show that agents aligned with RF outperform other baseline prompt-based approaches.

### Strengths
The paper formalizes a novel problem in information retrieval, and shows Reinforcement Learning from Ranking Feedback (RLRF) to be an effective algorithm to maximize ranking, with a clear hedge over simpler prompting baselines.

The experiments are sound and the RLRF agents seem able to generalize also to unseen ranking functions, making it more realistic for real-world scenarios where the ranking functions are unknown.

### Weaknesses
The following are the main weaknesses of the paper: 

- The paper lacks novelty in terms of methods: the used approach is quite standard in the LLM literature: collect documents preference pairs --in this case using the ranking function-- and finetune the model via the DPO algorithm.

- The aligned agents are evaluated after DPO finetuning was completed. I think it would be interesting to see the evolution of their performance as a function of the funetuning steps. In particular, how much data (i.e. ranked document pairs) are required for them to outperform the other baselines?

Unfortunately, I cannot judge the relevance of the problem and whether it is of interest to the ICLR community.

### Questions
See weaknesses.

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
This paper introduces Reinforcement Learning from Ranker Feedback (RLRF), a novel framework for training large language models (LLMs) as strategic agents in competitive search environments. The agents optimize document content to improve rankings against competitors by learning from preference datasets derived from ranking competitions. The authors propose two data generation methods: Static Generation (SG) (independent document modifications) and Dynamic Generation (DG) (simulated multi-agent competitions). Experiments using the LEMSS simulator show that RL-aligned agents (RA agents) consistently outperform non-aligned prompt-based agents (NA agents) across diverse ranking functions (e.g., E5, Contriever, BM25). RA agents also generalize to unseen rankers and adapt to strategic opponents. Key contributions include formalizing competitive search as a learning problem, demonstrating RLRF’s effectiveness, and highlighting transfer learning capabilities.

### Strengths
- Originality: Novel formulation of competitive search as an RL problem, with DG simulating strategic opponent adaptation.
- Quality: Rigorous experiments across 4 LLMs, 4 rankers, and homogeneous/heterogeneous settings. RA agents (e.g., Mistral+DG+LSW) achieve up to 75% win-rate (Ho) and 60% (He).
- Clarity: Clear problem definition (Section 3) and accessible methodology (e.g., DPO for alignment).
- Significance: Demonstrates transfer learning to unseen rankers (Table 2), with asymmetric generalization (e.g., Contriever-trained agents generalize better than E5-trained).

### Weaknesses
- Synthetic Limitations: Experiments rely on simulated competitions (LEMSS) and MS MARCO data. Real-world deployment (e.g., dynamic user queries) is unexplored.
- Scalability: Training RA agents requires multi-round simulations (450 games × 30 rounds), but computational costs are not quantified.
- Strategy Stability: Convergence analysis (Appendix G) shows agents gravitate toward similar documents over time, but robustness to adversarial opponents (e.g., non-LLM agents) is untested.

### Questions
See weakness.

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
2

### Summary
This paper addresses the problem of "competitive search," a dynamic setting where document publishers strategically modify their content to improve their ranking on search engines. Recognizing that publishers increasingly leverage Large Language Models (LLMs) for this task, the authors introduce "Reinforcement Learning from Ranker Feedback" (RLRF), a novel framework for training these LLM-based publisher agents. The methodology involves training an LLM—termed an "RL-aligned agent" or "RA agent" —using preference datasets. These datasets are synthetically generated from simulated ranking competitions, thereby encoding signals from the ranker's outputs (i.e., the rankings). The agent alignment is performed at training time, notably using Direct Preference Optimization (DPO) , which allows the finalized RA agent to operate at test time via simple prompting without further optimization.

### Strengths
The primary novelty is the RLRF framework itself—the conceptual reframing of alignment-RL for a competitive, game-theoretic task. The agent trained on SG learns to hit a static target. The agent trained on DG learns to win a dynamic game. For any competitive, multi-agent domain, training agents on static preference data is fundamentally insufficient. One must incorporate multi-agent simulation (like DG) into the training loop to learn the "meta-game."

### Weaknesses
1. The paper dismisses Proximal Policy Optimization (PPO) as "less stable"  and adopts DPO. While DPO is a strong and modern choice, this justification is brief. A more detailed explanation or a preliminary experiment comparing them would have strengthened this methodological choice.
2. The paper's claim from Figure 2 (that RLRF improves faithfulness) is true only when controlling for model size. The paper fails to address the critical question: "How does an 8B-RA agent compare to a 70B-NA agent?" This weakness badly muddies the conclusions about faithfulness. It is plausible that the primary benefit of RLRF is rank promotion, and faithfulness is almost entirely a function of model scale.

### Questions
1. Could you please elaborate on your hypothesis for the "asymmetric" transfer learning observed in Table 2? Why does Contriever appear to be a "better teacher" than E5-unsupervised? Does this imply that the choice of generation ranker is a critical, first-class component of the RLRF framework that requires its own line of research?
2. could you provide a brief sensitivity analysis for a key parameter, such as the DPO beta? This would help confirm the "robustness" interpretation and allay fears that the results are specific to the chosen default.

### Soundness
3

### Presentation
3

### Contribution
2
