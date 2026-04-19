# Learning Linear Utility Functions From Pairwise Comparison Queries

- Decision: Reject
- Scores: 6, 6, 6, 3

## Abstract
There is increasingly widespread use of reward model learning from human preferences to align AI systems with human values, with applications including large language models, recommendation systems, and robotic control.  Nevertheless, a fundamental understanding of our ability to successfully learn utility functions in this model remains limited. We initiate this line of work by studying learnability of linear utility functions from pairwise comparison queries. In particular, we consider two learning objectives. The first objective is to predict out-of-sample responses to pairwise comparisons, whereas the second is to approximately recover the true parameters of the utility function. We show that in the passive learning setting, linear utilities are efficiently learnable with respect to the first objective, both when query responses are uncorrupted by noise, and under Tsybakov noise when the distributions are sufficiently "nice". In contrast, we show that utility parameters are not learnable for a large set of data distributions without strong modeling assumptions, even when query responses are noise-free. Next, we proceed to analyze the learning problem in an active learning setting. In this case, we show that even the second objective is efficiently learnable, and present algorithms for both the noise-free and noisy query response settings. This qualitative learnability gap between passive and active learning from pairwise comparisons suggests that the tendency of conventional alignment practices to simply annotate a fixed set of queries may fail to yield effective reward model estimates, an issue that can be remedied through more deliberate query selection.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper explores the feasibility of learning linear utility functions from pairwise comparison queries, proposing that this method can be applied to improve AI alignment with human values. The study addresses two levels of the problem: predicting out-of-sample pairwise comparisons and accurately estimating utility parameters. In the passive learning setting, it demonstrates that predicting comparisons is feasible, but estimating utility parameters is challenging without strong structural noise assumptions. The negative result, indicating that the linear utility function is not PAC-PC learnable, is presented in this setting. In the active learning setting, where comparisons can be strategically selected, estimating utility parameters becomes much more feasible, provided there is the ability to invert the embedding.

### Strengths
Please find the strengths below:
1. The problem of learning the linear utility function is important in AI. The paper addresses both passive and active learning settings, which are commonly encountered in practice.
2. For passive learning, the paper presents a negative result regarding the learnability of linear utility function parameters.
3. The proofs in the main paper are clear and intuitive to follow.

### Weaknesses
Please find the weaknesses below:
1. The paper lacks intuitive explanations for these assumptions, such as examples of cases where the assumptions are satisfied.
2. The paper does not sufficiently discuss the practicality of assumptions, such as those on well-behaved isotropic distributions, $F'(z)^2 - F''(z) \cdot F(z) \geq \gamma$, and the approximation of the inverse of $\phi$. It is unclear how to determine parameters (e.g., $c$ and $\gamma$) in these assumptions in practice.
3. In Theorems 3, 7, and 8, the paper only establishes the existence of an efficient algorithm but does not explain how to find the algorithm within the main text (though pseudo code for Theorems 7 and 8 is provided in the appendix).
4. The paper lacks experiments assessing the complexity of the algorithms. Even some numerical experiments would be beneficial, as would verification of these assumptions on practical datasets.

### Questions
Refer to the weaknesses section above and find the additional questions below:
1. Is there a lower bound for the sample complexity in Theorem 3? It seems that the complexity could be exponentially high with respect to $1/\alpha$.
2. Is it possible to extend beyond linear utility functions, particularly for the simpler problem of predicting pairwise comparisons?

### Soundness
3

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
4

### Summary
Driven by applications in LLM alignment, this paper studies the problem of learning linear utility functions from pairwise comparisons. The authors provide theoretical results that suggest the following.

- Efficiently PAC-learning utilities that predict pairwise comparison outcomes is possible, except in degenerate cases.
- Learning the utility parameters can fail when the RUM does not satisfy certain conditions. The noiseless case is an example of failure case.

The authors then consider the active learning setting, where queries can be chosen adaptively based on previously observed outcomes. In that setting, it is possible to efficiently PAC-learn utility parameters in the noiseless case.

### Strengths
- The paper is particularly well written and easy to follow. The organization is clear, the progression makes sense, and the decisions on what to include in the main text and what to defer to the supplementary material are well thought-through.
- The results appear to be rigorous. I did not check every single proof, but the ones I checked appear to be correct and, in general, the arguments make sense.
- Research on this topic is timely, given the increasing interest in practical applications, and highly relevant to the ICLR community.

### Weaknesses
- While the exact problem studied in the paper does not have a lot of prior work (to my knowledge), there is an extensive literature on learning RUMs from pairwise comparisons over a fixed set of items with an independent utility for each item in the set. This can be recast as sparse "one-hot" features in the model studied in the paper.
    - some of the insights the paper provides are well-known in that literature. For example, sample complexity results for the Bradley-Terry model typically depend on the noise via the dynamic range, i.e., the maximum ratio between item utilities. The case of unbounded dynamic range has been studied before. Some references: [1], [2], [3].
    - making connections to this literature by comparing & contrasting results would improve the paper significantly.
- In general, I find the connection between the theory developed in this paper and LLM alignnment a bit weak. The paper could be incredibly impactful if there were concrete take-aways that could inform practice. Are there any?
    - l 082: "Our results thus suggest that conventional reward model [...] may yield potentially misleading estimates". Backing this claim up with real data or at least realistic experiments would be helpful.
- The motivation for the active learning section could be improved.
    - The authors say: "first [...] curates a dataset of input pairs, and only then obtains preference data for these". This is the exact _opposite_ of active learning, and indeed the two papers cited (OpenAI & Anthropic) are examples of passive learning.
    - However there is work showing that active learning can improve LLM alignment techniques, see e.g. [4] and [5], so I see an opportunity to make the point much more convincingly.

Minor feedback:

- l 046: "... over a vector space of outcomes" -> what does this mean?
- l 095: "the restriction of our utility function is qeuivalent to ..." -> I could not make sense of this sentence.
- l 154: "with probability $1-\delta$ such that ..." I think this should be flipped: "such that ... with probability $1-\delta$"
- Theorem 2 seems kind of obvious: if one can make outcomes arbitrarily noisy, it can get arbitrarily hard to learn. Is there a deeper insight I am missing here?

References:

1. Generalized Results for the Existence and Consistency of the MLE in the Bradley-Terry-Luce Model <https://proceedings.mlr.press/v162/bong22a/bong22a.pdf>
2. l-infty-Bounds of the MLE in the BTL Model under General Comparison Graphs <https://proceedings.mlr.press/v180/li22g/li22g.pdf>
3. Active Ranking in Practice: General Ranking Functions with Sample Complexity Bounds <https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=52f514508329c951ef9b417ce5243cf40db136bd>
4. Active Preference Learning for Large Language Models <https://arxiv.org/pdf/2402.08114>
5. Direct Language Model Alignment from Online AI Feedback <https://arxiv.org/pdf/2402.04792>

### Questions
1. What concrete take-aways should a practitioner doing preference-based LLM alignment get from the paper? Which theoretical findings can infrom practice?
2. Much of the theory seems to exploit the fact that learning from pairwise comparisons in linear RUMs is equivalent to linear binary classification. Which insights are unique to the fact that we observe pairwise comparisons?

### Soundness
4

### Presentation
4

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
This paper studied the learnability of linear utility functions from pairwise comparison queries. The paper considered two settings, including the passive learning setting and the active learning setting, with two objectives: 1) predicting the out-of-sample pairwise preferences; 2) estimating the true parameters of the utility function. Theoretical analysis show that in the passive learning case, the first objective is achievable both when query responses are uncorrupted by noise / under Tsybakov noise when the distributions are sufficiently nice. However the utility function parameters may not be learnable. In contrast, in the active learning setting, both objectives can be efficiently achieved.

### Strengths
- This paper studied a significant and practical problem of reward learning, when the observations are pairwise comparison queries. This can be a common scenario in real-world examples such as online recommendation platforms. Overall the paper is presented with a good clarity on the model, assumptions and results.
- The theoretical analysis justified an interesting learnability gap between the passive and active learning settings, where the active queries significantly helped with the utility function estimation. This result may be helpful to provide guidance for real-world applications where the active queries may be obtained via incentivization.

### Weaknesses
In the active learning setting, the estimation of the utility function relies on the ability to invert the embedding function, and obtaining the inverse of the embedding can be computationally challenging or infeasible particularly with neural network models.

### Questions
- in the active learning setting, is the sample complexity optimal?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper addresses PAC learning of linear functions using comparison queries. It considers two objectives: predicting the outcome of a comparison between two inputs and learning the true underlying linear parameters. Both noiseless and noisy settings are explored, as well as both active and passive learning.

### Strengths
The problem studied is very clean and interesting to the learning theory community.

The writing is very clear.

### Weaknesses
I believe the primary weakness of this paper lies in its technical contributions. I didn't find anything particularly novel or technically interesting.

For instance, the results in passive learning don't appear to be new. Theorems 1 and 3 have been previously addressed, and the observation in Theorem 2 seems straightforward—naturally, certain noisy models prevent learning. Similarly, the conclusion in Theorem 6 also seems obvious.

In the active learning setting, the problem of active learning for halfspaces with comparison queries appears to have already been solved by [1].

[1] Kane, Daniel M., Shachar Lovett, Shay Moran, and Jiapeng Zhang. "Active classification with comparison queries." In 2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS), pp. 355-366. IEEE, 2017.

### Questions
It would be nice if the authors could explain the technical novelty of this work.

### Soundness
3

### Presentation
3

### Contribution
2
