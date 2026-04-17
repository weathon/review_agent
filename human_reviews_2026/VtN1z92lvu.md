# Decision Aggregation under Quantal Response

- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
The effectiveness of collective decision-making is often challenged by the bounded rationality and inherent stochasticity of individual agents. We investigate this by analyzing how to aggregate decisions from $n$ experts, each receiving a private signal about an unknown state. Assuming signals are conditionally independent and identically distributed, we depart from the fully rational paradigm and model expert behavior using quantal response—a stochastic choice model capturing bounded rationality. Within a minimax regret framework, we show that majority voting is the optimal robust aggregator when individual rationality falls below a certain threshold. Interestingly, such groups can outperform perfectly rational agents, as their decision randomness encodes weak but informative signals lost in deterministic behavior. We validate these findings using large language models (LLMs), which naturally exhibit quantal response via their temperature parameter. Aggregating moderately stochastic LLM outputs significantly improves accuracy on complex reasoning tasks, highlighting bounded rationality not as a limitation, but as a potential strength in collective intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper examines information aggregation under bounded rationality. A decision maker gathers reports from agents receiving noisy signals about a ground truth. Instead of sincere voting, agents follow a quantal response modeled by a logistic link between their posterior and report. The authors show that (1) majority voting with an optimal threshold minimizes regret, and (2) with more agents, quantal responders can outperform sincere ones. Experiments further confirm that (1) large language models exhibit quantal response behavior via temperature, and (2) the theoretical results hold empirically.

### Strengths
The paper offers a valuable contribution to the information aggregation literature by applying a rational framework to analyze boundedly rational agents. The results are both elegant and insightful, providing a simple yet positive perspective on a complex problem. The experimental findings are also convincing.

### Weaknesses
I am not fully convinced by the second main result (bounded rational advantage). The idea behind it is not as surprising as it may seem. Quantal response preserves randomness even when rational (I'd rather call sincere) agents become deterministic. This randomness helps the majority vote to operate processes like the Condorcet Jury Theorem, constructing another form of informative voting (voting as their signal suggests) and reaching the ground truth with high probability.

The thing is, this observation is based on the assumption that agents vote sincerely. Personally, I don't quite buy this assumption. On the simplicity side, conducting a Bayesian update and maximizing expected utility is not an easy task for voters who don't really know Bayesian statistics. On the rational side, sincere voting is not necessarily a Nash Equilibrium [Austen-Smith and Banks, 1996; Han et. al, 2023]. Consequently, I am curious to see if the same result extends to various settings. Nevertheless, I am overall positive about this paper. 

Austen-Smith D, Banks J S. Information aggregation, rationality, and the Condorcet jury theorem[J]. American Political Science Review, 1996, 90(1): 34-45.
Han Q, Schoenebeck G, Tao B, et al. The Wisdom of Strategic Voting[C]//Proceedings of the 24th ACM Conference on Economics and Computation. 2023: 885-905.

P.S. You might want to update some of the references, whose status seems out-of-date.

### Questions
1. How do you justify sincere voting?
2. Do the same results extend to various settings?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents theoretic results on how decisions are aggregated under uncertainty while quantal response is considered. The main theorem is in 3.1, which proves that (1) if the rational parameter is bounded under a threshold, then majority voting is the optimal aggregation rule and (2) the group collective decision can be benefited from bounded rationality of individuals.

### Strengths
The technical contribution is sound. The results are quite interesting.

### Weaknesses
It is not very clear why LLMs are generally considered as quantal best responder

### Questions
The theoretical results in this paper generally look great and insightful. However, I wonder the authors could comment more about why LLMs are considered quantal best responder, as section 4.1 only concerns on one domain. Also, is here every LLM considered to have the same \lambda?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work shows that when experts make rational, stochastic decisions (modeled by quantal response), majority voting optimally aggregates their judgments under minimax regret. Surprisingly, such partially random groups can outperform perfectly rational agents, as their randomness preserves informative variation. Experiments with LLMs confirm that aggregating moderately stochastic outputs improves reasoning, revealing bounded rationality as a strength in collective intelligence.

### Strengths
1. Theoretical results: the authors proved interesting theoretical results showing that bounded rationality can outperform perfect rationality;
2. Interesting experiments: the authors leverage LLMs as conditional independent agents, which provides a scenario where the condition of the theoretical results can be easily satisfied.

### Weaknesses
1. It seems not straightforward to calculate g(n) in the main theorem. Appendix A2 provides a plot of g(n), but mostly observations. May provide more insight on why g(n) = g(n-1) for even n's. 
2. (minor) the authors refer to Fig 5 in line 340, but Fig 5 is in the appendix. Better refer to both the appendix section and the figure.

### Questions
See above

### Soundness
4

### Presentation
4

### Contribution
3
