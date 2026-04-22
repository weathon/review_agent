# Free-MAD: Consensus-Free Multi-Agent Debate

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
Multi-agent debate (MAD) is an emerging approach to improving the reasoning capabilities of large language models (LLMs). Existing MAD methods rely on multiple rounds of interaction among agents to reach consensus, and the final output is decided by majority voting in the last round. However, this consensus-based design faces several limitations. First, multiple rounds of communication increases token overhead and limits scalability. Second, due to the inherent conformity of LLMs, agents that initially produce correct responses may be influenced by incorrect ones during the debate process, causing error propagation. Third, majority voting introduces randomness and unfairness in the decision-making phase, and can degrade the reasoning performance. To address these issues, we propose Free-MAD, an alternative and novel MAD framework that eliminates the need for consensus among agents. Free-MAD introduces a novel score-based decision mechanism that evaluates the entire debate trajectory rather than relying on the last round only. This mechanism tracks how each agent's reasoning evolves, enabling more accurate and fair outcomes. In addition, Free-MAD reconstructs the debate phase by introducing anti-conformity, a mechanism that enables agents to mitigate excessive influence from the majority. Experiments on eight benchmark datasets demonstrate that Free-MAD significantly improves reasoning performance while requiring only a single-round debate and thus reducing token costs.  We also show that compared to existing MAD approaches, Free-MAD exhibits improved robustness in real-world attack scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes FREE-MAD, a consensus-free multi-agent debate framework for LLM reasoning.  Instead of the traditional rounds of debate, followed by a majority vote at the end, as presented in MAD; the authors present present a method that: (1) redesigns the debate stage with an anti-conformity, CoT-driven prompt that asks agents to only change their answer when they can justify that their previous answer was wrong; and (2) replaces final-round voting with a score-based decision mechanism that aggregates all intermediate answers across rounds. Experiments on 8 benchmarks shower higher accuracy results than the baselines, with added rousted and reduced token cost.

### Strengths
1. The paper a new MAD protocol decomposed in 2 phases. The method moves away from consensus and integrates anti-conformity.
2. The method shows achieving competitive performance for single-round debate, and better performance in some cases for multiple rounds. 
3. The paper clearly presents the method, including accurate descriptions and the algorithm's underlying principles and implementation details

### Weaknesses
1.The authors assume that answer switch between rounds leads to an improvement. However, it doesn't take into account the change could lead to worse performance. The algorithm could take into account the quality improvement of those answer switches: fraction of switches that move toward ground truth per round
2. The robustness analysis focuses on communication attacks, but it misses to analyze some more realistic attacks, such as contamination, adversarial attack or colluding agents. The robustness claims seem to be overstated given the lack of analysis.
3. Similarly, the authors make claim regarding scalability of their method, but only perform experiments for up to 2 rounds, how does this method behave with larger number of rounds.

### Questions
- The baselines are not clear. The paper should include a paragraph describing the baselines. To make it clearer, it is better to refer to them by their name rather than by writing Baseline 1 or Baseline 2.
- Why doesn't Figure 4 clearly state the number of rounds for that experiment?
- Why is Figure 3 presenting the results Token Consumption vs Accuracy? If the purpose of this plot is to show scalability as mentioned in Section 6.2, wouldn't it be better to present Token Consumption and Accuracy vs Number of Rounds. I think the current plot is not presenting new information. 
- Concerns about robustness. The authors have selected the communication attack, but have neglected some more real attacks such as contamination, adversarial attack or colluding agents. Can they prove their method is robust against other type of attacks?
- How much does the method scales? How many agents or rounds can the method scale to and what are the benefits?

### Soundness
2

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
3

### Summary
The paper presents FREE-MAD, a framework for multi-agent debate that improves how LLMs reason together without needing consensus. Instead of using majority voting, it scores all agents’ responses across the debate to find the most consistent and well-reasoned answer. By encouraging anti-conformity, agents critique rather than copy one another, reducing groupthink and errors. Experiments on eight benchmarks show that FREE-MAD improves accuracy, scalability, and robustness while needing only a single debate round

### Strengths
1. By leveraging anti-conformity, FREE-MAD encourages agents to question and critique others’ reasoning rather than follow the majority, reducing groupthink and improving the overall accuracy 

2. It achieved notable performance gains while reducing token consumption through just a single round of debate.

3. Good writing and easy to understand.

### Weaknesses
1. The score-based decision mechanism employs a fixed set of weighting coefficients derived from theoretical analysis. However, it seems to lack a clear analytical rationale or process explaining how these values were determined.
2. In the agent groups, when N = 3 or N = 4, the setup is based on only two models — Qwen1.5-7B-Chat and DeepSeek-V3, or Qwen1.5-7B-Chat and Qwen2.5-72B-Instruct. This design is somewhat confusing, as it’s unclear which model each agent corresponds to. In addition, the variety of agent types is limited, since the configuration relies on only two model architectures.
3. The paper doesn’t include a clear analysis showing how key parameters—like the scoring weights or number of agents—affect performance or stability.

### Questions
1. Is there a clear analytical or empirical basis for choosing the specific coefficient values used in the model?
2. Additional experiments should be conducted using a more diverse set of models and a larger range of N values.

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
3

### Summary
This paper argues that standard multi-agent debate (MAD) suffers from conformity, majority-vote randomness, and multi-round token overhead, and proposes FREE-MAD—a consensus-free framework that evaluates the entire debate trajectory instead of only the last round. The method reconstructs both stages of MAD: (i) a debate stage that can encourage anti-conformity via a CoT-style prompt to reduce error propagation, and (ii) a score-based decision stage that maintains a matrix of agents’ answers across rounds and updates a score dictionary to reward newly adopted answers and downweight abandoned ones, with a round-dependent factor to limit late-round conformity. The authors formalize the MAD protocol, compare against majority voting and LLM-as-judge designs, and introduce two practical variants—FREE-MAD-N (anti-conformity debate) and FREE-MAD-C (conformity debate)—that share the same scoring mechanism. Across eight benchmarks, FREE-MAD delivers sizable gains, with average improvements of roughly 13–16.5% over baselines, and remains competitive or better even when baselines use more rounds.

### Strengths
* Conformity is a critical challenge in the development of MAD, which motivates FREE-MAD well.
* FREE-MAD achieved significant improvement on diverse benchmarks

### Weaknesses
* There are too many hyperparamters in FREE-MAD, including w1-w4 and f to manage the behavior of scoring. The selection of hyperparameters are not analyzed.
* The scoring mechanism proposed in this paper is not "consensus-free". It is more like an advanced consensus mechanism considering longer context. The name is misleading and overclaiming.
* In Section 6, "Consequently, in such scenarios, conformity may lead to more effective outcomes." However, FREE-MAD is motivated by mitigating the conformity during MAD to improve the outcome. This observation seems contradictary to the motivation, and is not well explained.
* From the analysis, I cannot be convinced that the new "consensus-free debate stage" is the root cause for the improvement. There is a significant lack of detailed anaylsis to validate the effectiveness of consensus-free debate stage, including find-grained ablation study and comparison to baselines. For instance, is it possible that the improvement is simply brought by shifted focus during the debate due to the updated prompts, which makes LLMs aware of reasoning gaps?

### Questions
See weaknesses.

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
4

### Summary
The paper proposes FREE‑MAD, which replaces final‑round majority voting with trajectory‑level scoring. During debate, prompts ask agents to revise only when they can falsify their prior answer. During decision, the method treats whether an agent revised or persisted in each round as signals for weighted scoring, with a late‑round decay, and returns the highest‑scoring answer. Experiments on math/logic/QA benchmarks show gains over majority‑vote baselines.

### Strengths
The method is simple: it uses revision vs. persistence as signals to realize a multi‑round weighted voting scheme.

Empirical results show improvements.

### Weaknesses
Code is not open-sourced, hindering reproducibility.

The motivation (that consensus/conformity may dilute or derail correct reasoning) is supported mainly by illustrations and anecdotal observations, not a direct statistical analysis.

The core method rewards revisions (up‑weighting the new answer and down‑weighting the old answer) and gives a smaller reward to persistence. In essence, this encourages changing answers. Although the authors claim the algorithm promotes evidence‑based rather than conformity‑based revisions, it does not actually detect the cause of a change; meanwhile the motivation discourages conformity—this creates a conceptual inconsistency.

The approach applies to tasks with a single, explicit correct answer (e.g., QA, MATH). It does not directly transfer to more complex coding or agentic tasks.

### Questions
For Limitation 1, is there a transfer path to coding/agentic tasks?

Will you compare against Self‑Consistency [1]? There are recent studies showing that Self‑Consistency typically outperforms MAD in both cost and accuracy [2, 3]. 

[1]  Self‑consistency improves chain‑of‑thought reasoning in language models. 

[2] If multi‑agent debate is the answer, what is the question. 

[3] Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs

### Soundness
2

### Presentation
3

### Contribution
2
