# ArchPilot: A Proxy-Guided Multi-Agent Approach for Machine Learning Engineering

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Recent LLM-based agents have demonstrated strong capabilities in automated ML engineering. However, they heavily rely on repeated full training runs to evaluate candidate solutions, resulting in significant computational overhead, limited scalability to large search spaces, and slow iteration cycles. To address these challenges, we introduce ArchPilot, a multi-agent system that integrates architecture generation, proxy-based evaluation, and adaptive search into a unified framework. ArchPilot consists of three specialized agents: an orchestration agent that coordinates the search process using a Monte Carlo Tree Search (MCTS)-inspired novel algorithm with a restart mechanism and manages memory of previous candidates; a generation agent that iteratively generates, improves, and debugs candidate architectures; and an evaluation agent that executes proxy training runs, generates and optimizes proxy functions, and aggregates the proxy scores into a fidelity-aware performance metric. This multi-agent collaboration allows ArchPilot to prioritize high-potential candidates with minimal reliance on expensive full training runs, facilitating efficient ML engineering under limited budgets. Experiments on MLE-Bench demonstrate that ArchPilot outperforms SOTA baselines such as AIDE and ML-Master, validating the effectiveness of our multi-agent system.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents ArchPilot, a multi-agent system designed to automate machine learning engineering and neural architecture search under limited compute budgets. The framework decomposes the problem into three interacting agents: an orchestration agent that manages search via Monte Carlo Tree Search (MCTS), a generation agent that produces and refines runnable ML scripts, and an evaluation agent that estimates model performance using multiple lightweight proxy signals rather than full training runs. The proxies are adaptively reweighted through ridge regression, and when their weighting changes, the system restarts the search tree to realign exploration with updated evaluation semantics. Experiments on MLE-Bench show that ArchPilot outperforms existing LLM-based AutoML agents such as AIDE and ML-Master in terms of valid-submission rate and leaderboard ranking.

### Strengths
1.	This paper tackles an important issue in LLM-driven AutoML, i.e., the prohibitive cost of repeated full-training evaluations. The proposed proxy-based evaluation strategy is practical and aligns well with real resource constraints.
2.	The system design is modular, separating generation, evaluation, and orchestration into independent agents, which improves clarity and extensibility. This architectural decomposition is well motivated and technically sound.
3.	The adaptive proxy weighting and MCTS mechanism is a good contribution. It provides a way to correct for drift in proxy fidelity over time and avoids getting trapped in search trajectories.

### Weaknesses
1.	The experimental analysis is quite shallow. For example, there is no ablation study showing the contribution of each component, such as proxy reweighting, restart policy, or the multi-agent structure itself. It is unclear which part of the framework actually drives the improvement.
2.	All experiments use a single LLM (GPT-4.1).
3.	The paper does not report token usage, API cost, or wall-clock time, which are crucial for verifying whether the approach is truly more efficient than baselines.
4.	Implementation details are largely missing. The paper does not report prompt templates, MCTS parameters, or how the agents communicate. This omission severely limits reproducibility and makes it difficult to verify the technical correctness of the system.
5.	Only two baselines are included. Some recent and stronger systems are missing. Moreover, stronger reasoning models such as GPT-5, DeepSeek-R1 are not included.
6.	All experiments are conducted on MLE-Bench, which is an AutoML benchmark. No true NAS experiments are provided. As a result, the paper does not substantiate its claimed contribution to neural architecture search.
7.	The multi-agent idea itself is not new. Similar decompositions have been explored in earlier works. The novelty lies mainly in the proxy-guided evaluation and restart mechanism, which is useful but incremental rather than fundamental.

### Questions
See weaknesses. The authors should provide more implementation details, such as the used prompts.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a multi-agent system containing three agents: AO, GA, and EA. These agents performed different jobs: searching, generating, and evaluating, respectively. The most significant contribution is that the authors use the proxy module to avoid full retraining of the ‘small’ ML model.

### Strengths
- The idea of using a proxy to avoid retraining is technically sound and novel.
- Three designed agents play basic roles in the agent system, which forms into a loop to enable the system with self-evolving capability.
- The paper is well-written and well-organized, which makes it easy to follow.

### Weaknesses
- The experiments are weak. Only two baseline agents are compared on a single benchmark, MLE-Bench, and only one LLM backbone is used.
- As the only benchmark used in this paper, it lacks a detailed illustration, such as how many instances are in each task.
- The paper lacks ablation studies, which cannot demonstrate the contributions from different modules or agents.
- There are no quantitative results of the time ArchPilot reduces using the proxy module. I believe the author should at least display a “proxy vs full training cost” dialog.

### Questions
- Though this paper uses proxies to reduce full training, it highly depends on the calling of LLMs like GPT, which is also consuming. Can authors roughly judge this comparison of computational consumption?
- Will the authors open-source the code if the paper is accepted?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes ArchPilot for automated ML engineering. It has three agents (Orchestration, Generation, Evaluation) and uses an MCTS-type search over the search space of scripts. The key ideas are to guide the search through 1) adaptive combination of cheap proxy evaluations and occasional full training, and 2) adaptively fitting a proxy-to-true-score weights to reweigh the nodes in the tree.  Experiments are conducted on the MLE-Bench, where ArchPilot demonstrates improvements over AIDE and ML-Master.

The work represents a largely engineering effort with limited conceptual or algorithmic novelty, and an unnecessary agentic framing.

### Strengths
- The paper seeks to address a core challenge in AutoML, the prohibitive cost of searching over large search spaces, and improving performance over limited search budgets. 
- The modular design is clean, although this is the case in many prior NAS approaches too.
- Evaluation on MLE-Bench is better than that on standard NAS benchmarks.

### Weaknesses
- The agnetic framing of ArchPilot is disingenuous. There is no action space, nor any decision making going on. The OA is simply calling tools (LLM in this case), GA and OA are largely manually designed processes. Existing NAS methods also do the same. 

- The primary novelty of ArchPilot is combining existing components (MCTS/UCT, proxy training, ridge-fitting, LLM-based code generation, restart). The paper over states the contributions.

- Only two baselines methods are considered, AIDE and ML-Master. Comparisons to Non-LLM based NAS methods (e.g., DARTS) are needed to understand the gains benefits beyond LLM-agent literature.

- The problems in MLE-Bench are towards the smaller scale. It is unclear if the benefits translate to more realistic problems.

- ArchPilot has many design choices (restart, proxy reweighting, etc.). However, there is no discussion of their contributions to overall performance.

### Questions
- The paper reports ranking metrics. However, there is no discussion of the actual performance differences. 
- Quantify the contribution of design choices through ablations (without restart, without proxy reweighting i.e., static weighting, without GA improvements i.e., random improvements, and with different proxies).
- Aggregating proxies by simple convex combination is convenient, but does it improve rank-consistency of solutions?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a LLM-assisted MCTS approach for neural architecture search. The authors use three agents to decompose the functional parts in a NAS approach: candidate proposal, evaluation and search strategy. The overall search strategy is controlled by a orchestration agent (OA), which is basically a MCTS process that use UCB funtion to credit the expansion and resources allocation of the searching tree. Given a node selected, the candidate proposal depends on a LLM-based agent (generation agent, GA) that directly generates codes of network architecture and corresponding training script. A novel evaluation method is proposed to provide proxy evaluation with adaptive update to reduce full training inefficiency. The authors validate their approach on MLE bench where diverse ML tasks serve as good testbed for measuring general effectiveness of the proposed approach. The comparison results with two latest baseline show the improvement of the proposed tri-agent system.

### Strengths
Novelty: I think this paper hold certain novelty in combining the strength of MCTS and LLM for automated NAS.

### Weaknesses
While novel to some extends, I still think this paper is not prepared well for publication.

1. Writing: I have to say that te writing and content organization of this paper is not very ideal. Where is the introduction or preliminary of MCTS-based NAS, why the authors expect the readers very familiar with this field? There are many NAS works while the related work section only review those that focus on how to do efficient evaluation. 

2. Contribution: I can not understand why the three proxy functions can truly profile the true performance together? Provide detailed explanation on the rationale behind. More importantly, provide an significance or R2 score analysis on the correlation between the proxy validation and full training validation. A nother concern is that the authors claim this proxy-base validation is the major contribution of this paper, however, it is still an integration of existing proxy techniques, making the controbution somehow weak.

3. Significance: I suggest the authors at least add some NAS baselines into your comparison, without which one can not claim that their method is SOTA. In particular, I must say that according to your results on MLE, the portion of task your approach and other two are not very idea, e.g., even some submissions your approach generates are not valid (0.893), and valid submissions that achieved median of the performance is only 0.293, how can one tell such results are SOTA? Does this observation suggests that LLM-based NAS is not a good choice?

4. Ablation: with no ablation on your proposed designs, I can not tell whether your approach is truly useful, provide such results, please.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
