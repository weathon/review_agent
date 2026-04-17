# Conformal Risk-Controlled Routing for Large Language Model

- Decision: Reject
- Scores: 4, 4, 4, 0

## Abstract
Recent advances in small-scale large language models have shown that compact models can successfully handle an expanding range of natural language and reasoning tasks. This progress opens the door to more affordable AI inference services by enabling broader use of cost-efficient models. However, existing approaches often fail to fully exploit small models due to fuzzy boundaries of their capabilities. In this paper, we propose a risk-controlled routing framework that dynamically selects among models of different scales, with a strong emphasis on maximizing the utility of smaller models. Our framework integrates supervised contrastive learning to enhance the separability of smaller-model capabilities and grounds its routing mechanism in conformal risk control, providing theoretical guarantees on system-level routing risk. Across benchmarks, our method delivers cost–accuracy performance that is comparable to or better than strong baselines, with an absolute accuracy improvement of $\sim3.49\%$ at equal cost and up to $\sim36\%$ cost reduction at comparable accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose Conformal Risk-Controlled Routing, a framework that integrates capability-aware representation learning with principled risk control and cost-aware selection.

### Strengths
1. The proposed method is notably simple and efficient.

### Weaknesses
1. The readability of the paper could be improved. As an example, the variable l introduced in Line 193 is used without a clear definition, creating confusion for the reader.
2. The dependence of the proposed SCL method on the specific sentence encoder (all-MiniLM-L6-v2) is unclear. Its performance may be sensitive to the choice of this base model, which limits the understanding of the method's generalizability.
3. The proposed SCL method is specifically designed for question-answering tasks and its applicability to other problem types, such as text generation, is not demonstrated. Furthermore, its reliance on an IID calibration set may limit its practicality in real-world scenarios where such data is not readily available.

### Questions
How does the framework handle queries that are inherently unanswerable? Does it ultimately consume the most resources by routing them to the largest model, or does it correctly identify and handle them with the most economical option?

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
4

### Summary
This paper introduces Conformal Risk-Controlled Routing (CR^2), a method for routing between models while obtaining provable guarantees on the tradeoff between cost and accuracy. A binary classifier is used to decide the smallest and cheapest model is able to handle a given query $q$ based on a fixed threshold $t_1$. If now, the the query is passed through a multiclassifier that scores models 2,...,K on the probability $\hat{p}_i(q)$ that model $i$ answers $q$ correctly. Then We construct a conformal candidate set $C_\lambda(q)=\{i|\hat{p}_i(q) \geq \lambda\}$ for $i=2,...,K$ and then tune the $\lambda$ (while holding $t_1$ routing on the first model fixed) using conformal risk control to find the ``right'' choice of $\lambda$ that presumably pushes the Pareto frontier on cost and accuracy as much as possible. The authors provide empirical support for this framework

### Strengths
The paper addresses a highly practical problem of LLM routing for pushing performance on the cost-accuracy tradeoff, particularly doing it in a way that doesn't involve retraining of the base LLMs. The pipeline of the framework from the SCL and binary classifier to determine the cheapest model's ability to answer, to the use of CRC in the next stage appears carefully orchestrated. The authors make clever use of conformal risk control to choose amongst models to form a candidate set of models in the second stage with a loss that basically seems to measure (number of incorrect models in the set) / (total number of actually incorrect models). In other words this seems to be like the proportion of bad models that the router mistakenly certified as capable of answering. The use of CRC allows the authors to do better than a purely heuristic router approach in that they can tune an $\alpha$ tolerance to this loss, and is using CRC in a particular way for this setting. The paper attempts to carefully make the case for the efficacy of this framework in the empirics, providing a number of figures for visualization and presenting results across multiple benchmarks. A particular strength is that the authors include benchmarks such as MBPP that are not limited to MC questions. Overall I find the approach of tackling this routing problem with provable guarantees to be valuable and thus a central strength of this paper.

### Weaknesses
It seems that this framework will be limited to settings restricted to exact binary notions of correctness, not a wider range of possible "scores" on correctness that one might expect when using CRC rather than vanilla CP. 

Generally I just find the composite risk of $\alpha$ to be hard to interpret. Same for the fixed value of $t_1$ for the first model. It feels like there could be a tunable parameter in that place as well, which moves along with $\lambda$ to measure a potentially more meaningful notion of system-wide risk. Also the choice of $\alpha=0.08$ doesn't seem well motivated and misses one of the general use cases I would expect from CP or CRC which is showing value across a range of $\alpha$. 

Cost analysis seems a bit simplistic and limited to input tokens, while fine for MC questions since output is small, might limit usefulness of this analysis for other settings. It seems to me that the the CRC guarantee is weakened by the fallback mechanism. Figure 5b shows that ~32% of escalated queries result in an empty candidate set which triggers a heuristic arg max selection that is not covered by the conformal risk guarantee.

Finally, it appears to me that one of the major claimed contributions of this paper is in fact false, specifically that this is the ``first work to introduce CRC into LLM routing.'' The paper "Conformal Arbitrage: Risk-Controlled Balancing of Competing Objectives in Language Models" appears to address the cost-accuracy tradeoff as s special case of competing objectives and also does so using CRC. Although from inspection the approach to using CRC differs significantly, and that Conformal Arbitrage only compares 2 models not $K$, this still puts into question the claim of this paper to be the first to use CRC for LLM routing.

### Questions
The correctness label appears to rely on exact match with a single ground truth. How does or could this framework handle generative tasks with potentially multiple valid answers or even more generally a range of scores. 

Figure 5b suggests that the empty-set fallback (which seems to not be covered by the CRC guarantee) is triggered for >30% of escalated queries. How can we interpret this and its relation to the overall guarantees? 

Could the authors provide substantive comparison of their work to the aforementioned paper "Conformal Arbitrage" on LLM routing using CRC?

### Soundness
2

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
3

### Summary
This paper presents Conformal Risk-Controlled Routing (CR2), a two-stage framework for routing queries to a pool of language models of varying scales. The primary contribution is a system that aims to maximize the use of smaller, cost-efficient models while providing theoretical guarantees on performance.

### Strengths
The paper addresses the important and practical problem of optimizing the cost-accuracy trade-off in deploying large language models. The primary strength is the novel application of conformal risk control to this domain, which provides a principled, distribution-free method for managing routing decisions, a clear improvement over the heuristics common in prior work. The use of supervised contrastive learning to create "capability-aware" embeddings that distinguish between a model's ability to answer a query and the query's general semantics is another good contribution.

### Weaknesses
The overall presentation could be improved to meet the standards of a top-tier conference. Figure 2 is lower quality than it should be, and Figure 4 should take up half the space it does. I'm point this out not to nitpick but to suggest that the authors have much more room than they have properly used to develop their ideas better. Certainly the paper could have benefited from more substantive experiments and analysis. While the methodology is sound, the composite risk function defined in Equation 13 seems less defended and a more specific formulation would be beneficial. Furthermore, the system's performance relies on a fixed threshold t1 for the first-stage filter, which is selected heuristically. Given that a core contribution is the move towards principled risk control, the dependence on this fixed gate could be seen as a limitation.

### Questions
The first-stage threshold, t1, is treated as a fixed hyperparameter set on a validation set. How sensitive is the system's overall performance to the choice of t1?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The authors apply conformal risk control to a hierarchical routing framework. Compared with other approaches, their methodology applies supervised contrastive learning to yield an embedding in which a small LLM’s correctness label is the same for nearby points. The method seems to perform well on MMLU but yields mixed performance on other benchmarks.

### Strengths
- Supervised contrastive learning of text embeddings based on a small LLM's correctness labels is a compelling innovation!
- Applying CRC for global risk control is useful in practice.

### Weaknesses
- Unrealistic model pool: the largest model under consideration has only 14B parameters, which is very small. In practice, routing is often applied with frontier models. For this paper to be meaningful, the approach should be tested on LLM pools containing models with 100B+ parameters.
- The experimental data is strange. Typically, larger models outperform smaller models, and within a routing pool containing a strong frontier (or near-frontier) model, always querying the largest model usually sets an upper bound on performance. In the authors' data, however, the largest model appears to perform poorly (see Figure 3a). This makes interpretation of results confusing.
- Authors' interpretation of their methodology's performance borders on research misconduct. In Figure 3a, it is apparent that their method's performance is at or below baselines on 4/5 benchmarks. Only on a single benchmark, MMLU, does their method outperform the baselines. Against this empirical backdrop, the authors brazenly report that their method "consistently outperforms state-of-the-art baselines." This is a major red flag. 🚩
- The Pareto frontier in Figure 3b is unclear. Does the figure aggregate results from all benchmarks? There should be by-benchmark splits. In addition, baseline methods should also have their Pareto frontiers shown, not just the average performances.
- Given known stability issues with supervised contrastive learning when nearby points map to different labels, the paper should give clearer evidence that contrastive learning works for this use case. To give an example, reporting the test performance of linear classifiers for correctness prediction—trained separately on the contrastively tuned vs original embeddings—would constitute effective validation.
- Code is not available. It would be especially useful given the issues noted above.

### Questions
- Please give more details on the ablation study. To what one-stage process exactly do you compare your two-stage process?
- Contrastive learning based on correctness tends to suffer from stability problems. In the paper's setting, correct/incorrect labels are discontinuous in the original embedding space at the start of training, as nearby points have different correctness labels. Could you comment on training stability of your contrastive methodology?
- A central challenge in LLM routing is robustness to shifts in the query distribution. Could you comment on your method's robustness under distribution shifts? Empirically, how task-specific is your methodology?

### Soundness
1

### Presentation
3

### Contribution
3
