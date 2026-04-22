# Birch SGD: A Tree Graph Framework for Local and Asynchronous SGD Methods

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4

## Abstract
We propose a new unifying framework, Birch SGD, for analyzing and designing distributed SGD methods. The central idea is to represent each method as a weighted directed tree, referred to as a computation tree. Leveraging this representation, we introduce a general theoretical result that reduces convergence analysis to studying the geometry of these trees. This perspective yields a purely graph-based interpretation of optimization dynamics, offering a new and intuitive foundation for method development. Using Birch SGD, we design eight new methods and analyze them alongside previously known ones, with at least six of the new methods shown to have optimal computational time complexity. Our research leads to two key insights: (i) all methods share the same iteration rate of $\mathcal{O}\left(\frac{(R + 1) L \Delta}{\varepsilon} + \frac{\sigma^2 L \Delta}{\varepsilon^2}\right)$, where $R$ the maximum ``tree distance'' along the main branch of a tree; and (ii) different methods exhibit different trade-offs---for example, some update iterates more frequently, improving practical performance, while others are more communication-efficient or focus on other aspects. Birch SGD serves as a unifying framework for navigating these trade-offs. We believe these results provide a unified foundation for understanding, analyzing, and designing efficient asynchronous and parallel optimization methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a unified framework to represent various distributed SGD algorithms in the form of a computation tree. Each branch of the tree captures different versions of the model (potentially at different workers) and how they are updated with fresh or stale gradients. Some of the algorithms that are subsumed in the framework include local SGD, asynchronous SGD, and several other variants of these methods. The paper provides a unified analysis that matches previous results for these algorithms and sometimes gives improved analysis and optimal communication/computation complexity bounds.

### Strengths
- The elegance of the unified framework and analysis is a clear strength of this paper. Even if the bounds are not improved or new algorithms are not proposed, a unified representation is valuable for researchers working in this field to put algorithms in perspective.
- The addition of experimental results to complement the theory is well appreciated. Since this is a theory-focused paper, I do not have high expectations about the size and scale of experimental results. Nevertheless, I appreciated that the authors included these comparisons.

### Weaknesses
- The paper introduces eight new local SGD methods. While presented as a strength, this seems excessive and might confuse system designers choosing an algorithm. The insights section doesn't clearly guide which method to use under different communication and computation constraints. The comparisons in Section 3 are somewhat informative but could be clearer. Can you identify specific regimes—such as (1) communication-delay-bounded, (2) compute-bounded, and (3) bandwidth-bounded—and recommend the best algorithm for each?

- While the writing is generally clear, it needs improvement in organization and conciseness. I suggest spending more time explaining the computation graph structure, specifically how the number of workers $M$ relates to the tree branches. Currently, the paper gives minimal explanation of this aspect (see my question below) and jumps too quickly to theoretical results and new algorithm proposals.

- The writing comes across as overselling and condescending. I recommend toning down some claims to avoid alienating readers. For example, "virtually all local SGD algorithms" sounds too strong—please clarify which algorithms are not covered. The paper frequently uses adjectives and superlatives that add little information.

- The results apply only to homogeneous data distributions across workers and would not hold in heterogeneous data settings (like federated learning). I don't consider this a major weakness since the paper provides a valuable unified framework and analysis for the homogeneous setting. However, I would like the authors to clearly acknowledge this limitation and provide some ideas on whether and how the framework could be extended to heterogeneous data settings.

- The literature review is mostly thorough but omits other papers that provide unified analyses of different SGD algorithms, such as https://arxiv.org/pdf/1808.07576, https://arxiv.org/abs/2007.07481, https://arxiv.org/abs/2011.02828, https://arxiv.org/abs/2207.03730. I suggest broadening the literature survey to cover more key papers in the field rather than limiting references to papers from a few selected authors.

### Questions
- In local SGD and its variants, gradients are aggregated using AllReduce to update all models. Where does this step appear in the computation graph? At first glance, each edge seems to represent a single stochastic gradient (per-sample or mini-batch) rather than an average of gradients from multiple workers. However, upon closer reading and examining Figure 4, it appears that edges can represent averages of multiple gradients computed at different model versions. Is the gradient variance (now reduced due to taking the average) captured by $\eta$? How does the aggregation step of local SGD occur in the computation graph?

- Related to the above question, where does the number of workers appear in the convergence bound? Which term(s) explicitly depend on the number of workers?

- The framework subsumes asynchronous SGD and its variants. A crucial aspect affecting these algorithms and their analysis is gradient staleness—the mismatch between the model index at which the gradient is computed and the model updated using this gradient. I initially thought $R$ would capture a bound on this staleness, but I don't understand how it does this based on the distance metric dist(x,z) defined in the paper. Please clarify.

- How does this paper relate to https://arxiv.org/abs/2207.03730? That paper also models local SGD variants using a computation graphs with edges representing stochastic gradient computations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a unified theoretical framework for analyzing stochastic gradient based optimization method. The tree visualization is interesting. By using Lipschitz smoothness to bound how wrong the stale gradients can be the authors propose a unified analysis for convergence guarantees.

### Strengths
The unified framework provides a helpful tool to analyze many different implementations of SGD. The paper is clearly written with many helpful visualizations. The convergence guarantee theory is based on topology alone and seems quite powerful.

### Weaknesses
How tight is Theorem 2.4? It would be interesting to include some numerical experiment where R grows over time and show that it fails. Or include some correlated noise. I suspect that would be less harmful.

It seems quite conservative to use the upper bound on R rather than an average distance. I suspect most of the proof will go through using average distance but changing to a weaker convergence guarantee (in expectation maybe). 

Some of the new methods (cycle SGD) was not tested in numerical studies. When they do get added, please also include the total time, as some of them need to keep track of/communicate additional meta data/information and might take time in distributed settings. Though this part could be tricky to simulate.

### Questions
What about non-iid data? Could we add a measure of heterogeneity in the analysis?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents a unified framework to frame, analyze, and design distributed SGD algorithm. Basically, the framework models any distributed SGD algorithm as a weighted directed tree. The convergence properties of the algorithm only rely on the geometric properties of the directed tree. The authors provide a unified analysis and discussed many concrete examples, including both old and new distributed SGD variants.

### Strengths
- The idea of framing distributed SGD algorithms as a directed tree is very novel and interesting.
- The unified convergence analysis is also solid and clean.

### Weaknesses
The writing is not super clear. From algorithm 1, it is hard to tell why local SGD can be treated as an instance of birch SGD. I did not see any part of the algorithm to handle the communication (ie averaging gradients or parameters).

### Questions
1. From algorithm 1, it is hard to tell why local SGD can be treated as an instance of birch SGD. I did not see any part of the algorithm to handle the communication (ie averaging gradients or parameters).
2. The convergence rate is established in terms of the main branch? If so, then K is not the gradient computation times as in previous literature. As shown in figure 10, the main branch only has 5 iterates but there are total 10 gradient computations.
3. In table 1, why does local SGD listed as new? It is not a new algorithm.

### Soundness
2

### Presentation
2

### Contribution
3
