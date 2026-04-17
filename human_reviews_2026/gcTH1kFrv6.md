# Gold Panning: Turning Positional Bias into Signal for Multi-Document LLM Reasoning

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 4

## Abstract
Large language models exhibit a strong position bias in multi-document contexts, systematically prioritizing information based on location rather than relevance.
While existing approaches treat this bias as noise to be mitigated, we introduce GOLD PANNING BANDITS, a framework that leverages position bias as a diagnostic signal: by reordering documents and observing shifts in the model's responses, we can efficiently identify the most relevant content. 
We frame the problem of choosing reorderings as a bipartite matching problem.
While an optimal assignment can be computed at each iteration with the Hungarian algorithm in $O(N^3)$ time, we propose a greedy $O(N \log N)$ strategy that achieves comparable performance by prioritizing the placement of the most uncertain documents in the most informative positions.
Our approach identifies relevant documents using up to 65\% fewer language model queries than random permutation baselines on knowledge-intensive NLP tasks, substantially reducing computational cost without model retraining.
This work demonstrates that inherent LLM biases can be transformed from liabilities into assets for efficient, inference-time optimization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel method to more efficiently rank documents’ relevance. The main contribution is demonstrating that an LLM's positional bias can be transformed from a weakness into a strength.

It introduces the GOLD PANNING algorithm, which strategically reorders documents in the context. By placing the most uncertain documents in the most informative positions, it more efficiently identifies relevant information, significantly reducing the number of queries required without any model retraining.

### Strengths
1. The perspective is very novel. This represents the first application of bandit algorithms to long-context problems, and the first time of leveraging the position bias as a strength.
2. A rigorous theoretical analysis of the algorithm, showing the solid mathematical and algorithmic foundation of the author.
3. The problem description is clear.

### Weaknesses
1. The practical significance is relatively small. If you only want to determine each document's relevance with the query, why not just judge them using LLM (such as gpt-5), which should be faster, cheaper, more direct and more accurate. For example, you can input one document and the query each time and ask gpt-5 whether it's relevant; or input the document list and ask gpt-5 to rank every document's relevance.
    
2. Some important content is not detailed in main content, making it hard to comprehend the whole method. For example, I wonder how to transform the LLM's output into a signal to reflect relevance, but this information is in appendix instead of in the main content.
    
3. As mentioned by the author in section 5.2, high demands are placed on the model selection: powerful models don't have obvious position bias while weak models can't follow instructions. This limits the generalization of this method in real world applications.

### Questions
Have you tried to directly ask powerful LLMs such as gpt-5 to do the relevance ranking? Is it a faster, cheaper and easier way?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Large language models exhibit significant position bias when processing multiple documents, meaning the models tend to make decisions based on the position of information in the context rather than its actual relevance. This paper proposes transforming position bias from noise that needs to be eliminated into a signal that can be actively utilized, thereby enabling more efficient identification of relevant content in documents. The paper introduces the GOLD PANNING algorithm, which employs a greedy strategy to pair documents with the highest entropy with the most diagnostic detectors. This approach demonstrates significantly higher efficiency compared to the Hungarian algorithm.

### Strengths
The method proposed in this paper is more efficient than the Hungarian algorithm while delivering comparable results.

### Weaknesses
1. The authors claim that "To our knowledge, this is the first work to demonstrate that systematic LLM biases can be exploited rather than mitigated for inference-time optimization." However, several existing studies have already explored related directions [1,2,3].

2. The experiments are relatively limited—only two methods are compared, one of which dates back 70 years. The presentation is also simplistic, relying solely on a few curve graphs. The experimental results do not sufficiently demonstrate the superiority of the proposed method.

3. There is a lack of experiments evaluating the computational overhead of the proposed GOLD PANNING strategy. Theoretically, the token consumption and response time imposed by this method on LLMs appear prohibitively high.

Refs: [1] Zhang, Zhenyu, et al. "Found in the middle: How language models use long contexts better via plug-and-play positional encoding." Advances in Neural Information Processing Systems 37 (2024): 60755-60775.

[2] Alexander Peysakhovich and Adam Lerer. Attention sorting combats recency bias in long context language models. CoRR, abs/2310.01427, 2023. 

[3] Zhining Liu, et al. Selfelicit: Your language model secretly knows where is the relevant evidence. CoRR, abs/2502.08767,2025.

### Questions
See Weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a greedy algorithm that exploits LLM positional bias as signals to identify relevant documents in multi-document reasoning tasks. The authors formalize the problem as a combinatorial bandit model and derive theoretical guarantees on information gain and convergence. Empirical results, including simulations and small-scale experiments on GPT-4o-mini, show modest improvements over random baselines.

### Strengths
This paper presents an alternative perspective on the multi-document problem and, building on this view, proposes a greedy algorithm with theoretical guarantees. It also provides empirical results to support the proposed approach.

### Weaknesses
See the ``Questions`` section for specific points of weakness. In addition, the presentation could be improved. The concepts are introduced in a rather abstract manner, with few concrete examples to aid reader understanding.

### Questions
1. I have serious concerns about the problem setting considered in this work. In my view, the setting is neither realistic nor broadly applicable. Specifically,
    - If I understand correctly, the approach assumes the same (or at least highly similar) prompts and an identical collection of documents, which seems rather artificial. 
    - Moreover, the algorithm requires access to a verifier for the questions in the prompts, so that the ordering can be updated accordingly. However, such a verifier is typically unavailable in most practical scenarios. Without it, there is no reliable way to validate the answers or to determine what constitutes a meaningful "signal."
2. It also appears that the algorithm requires the true positive rate (TPR) and false positive rate (FPR) as inputs, which are critical for diagnosticity and belief updating. Yet again, it is unclear how these metrics can be obtained in the absence of a verifier.
3. The algorithm assumes a symmetric $N$-to-$N$ assignment, and while the authors claim that other cases can be reduced to this one, the proposed reduction for the $N>M$ case (essentially selecting $M$ items first) is unconvincing. The selection of items can substantially affect overall LLM performance, often even more than positional bias.
4. The definition of *detectors* is presented in an overly abstract manner and is difficult to interpret in the context of multi-document LLM settings.
5. The numerical experiments are rather weak. I recommend that the authors include additional experiments (e.g., more large-scale) to provide a more comprehensive empirical evaluation.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper argues that large language models' systematic positional bias in multi-document contexts, which prior works try to mitigate, can in fact be exploited in determining relevancy. The work formalizes the process of multi-document relevancy assessment in long-context settings as a bandit problem, and proposes a novel greedy algorithm with Bayes-update that approximates the true optimum, while reducing complexity from O(n^3) to O(n \log n). Evaluation on a dataset derived from MonoRel shows that the method proposed correctly identifies a correct ground truth fact among distractors while performing up to 65% less LLM calls than baselines.

### Strengths
This work presents a novel view on positional bias; instead of attempting to mitigating it, the work proposes to leverage such biases in obtaining better relevancy signals. In addition to introducing the mental framework, the work also keeps efficiency in mind, presenting a greedy algorithm that renders the naive O(n^3) Hungarian algorithm to a tractable O(n \log n) approximation. Further, the work establishes theoretical guarantees that the approximation is myopically optimal.

### Weaknesses
- As the authors indicated, the work makes several uneasy assumptions: (1) a good, known, and stable TPR/FPR per position, and (2) the relevancy of multiple candidates are independent of each other.
- The authors noted that the approach is only effective when positional bias is strong (heterogenous detectors), and therefore only presented experimental results for GPT-4o-mini, as the other 5 LLMs tested were not suitable for the argument. This makes the problem of "leveraging positional bias" seem rather artificial.
- The work only tested on 1 real dataset derived from MonoRel, which is to predict exactly 1 ground truth fact candidate from remaining distractors. This evaluation seems narrow.
- Unlike the PSC baseline, the proposed method, framed as a bandit problem, must be iterative. Thus, while it achieves less queries, this may not directly lead to efficiency gains.

### Questions
- Should the evaluation be expanded to more tasks? Since the work compares to PSC as a baseline, it may be a good idea to incorporate similar evaluations (ranking tasks, etc.)
- To what extend does the proposed method work on LLMs other than GPT-4o-mini? It was noted that 5 other LLMs were not suitable for experiments, but some additional justification should be used to demonstrate the method's robustness.

### Soundness
2

### Presentation
2

### Contribution
2
