# Learning Distributions over Permutations and Rankings with Factorized Representations

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4

## Abstract
Learning distributions over permutations is a fundamental problem in machine learning, with applications in ranking, combinatorial optimization, structured prediction, and data association. Existing methods rely on mixtures of parametric families or neural networks with expensive variational inference procedures. In this work, we propose a novel approach that leverages alternative representations for permutations, including Lehmer codes, Fisher-Yates draws, and Insertion-Vectors. These representations form a bijection with the symmetric group, allowing for unconstrained learning using conventional deep learning techniques, and can represent any probability distribution over permutations. Our approach enables a trade-off between expressivity of the model family and computational requirements. In the least expressive and most computationally efficient case our method subsumes previous families of well established probabilistic models over permutations, including Mallow's and the Repeated Insertion Model. Experiments indicate our method significantly outperforms current approaches on the jigsaw puzzle benchmark, a common task for permutation learning. However, we argue this benchmark is limited in its ability to assess learning probability distributions, as the target is a delta distribution (i.e., a single correct solution exists). We therefore propose two additional benchmarks: learning cyclic permutations and re-ranking movies based on user preference. We show that our method learns non-trivial distributions even in the least expressive mode, while traditional models fail to even generate valid permutations in this setting.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors approach the problem of learning probability distributions over permutations. Although this task might look trivial, the number of probability models over permutations is limited (Mallows, Plackett-Luce, Bradley-Terry,…), and each of them imposes a certain structure in the learnt distribution (strong unimodality, …). In this sense, given a sample of permutations, learning the probability distribution that best resembles the features of the samples is challenging. In this work, the authors consider the inversion-vector type representation to transform the permutations, and learn the distributions on top of using masked language models.

### Strengths
S1. Although the problem dates back 50 years, it is still a relevant and unsolved problem today.

S2. I did not read about Fisher-Yates draws previously, and it was definitely relevant to consider as an alternative representation of permutations.

S3. The paper is very well written.

### Weaknesses
W1. Many relevant references are missing. For me, it is surprising not to refer to general literature on this topic as:
—> Marden 1995, Analyzing and Modeling Rank Data.
—> Critchlow et al.1991, Probability models on rankings.
—> Fligner et al.1998, Multistage ranking models.

In fact, Critchlow et al. (1991) analyze four properties of probability models on rankings (strong unimodality, L-decomposability, …) that could be interesting to consider in this paper. What can the authors say about the properties of the distributions learnt?

W2. At times, the authors present the use of alternative representations as if it were novel, which it is not. In fact, Mallows' Generalized model itself uses V_j to decompose permutations (incidentally, I have not seen any mention of this model; I recommend that the authors review Fligner 1998).

W3. Furthermore, there is a group of authors who have worked on the use of alternative representations to propose algorithms (some based on NN+RL) that learn probability distributions to sample solutions to permutation problems. A recent work that uses feed-forward networks to estimate probabilities on the inversion vectors is:
—> Malagon et al. 2024, A Combinatorial Optimization Framework for Probability-Based Algorithms by Means of Generative Models. ACM TELO.
In fact, in this work, it is shown that one of the strong points is the possibility to infer in parallel a great number of permutations. So, contribution 5 in Section 4.2 is not entirely new.

W4. Regarding Theorem 4.3, I have to say that this result has been published too; actually, even a more general outcome is proposed. See Table 1 in the work below:
—> Malagon et al. 2025, Una Visión Unificada de Transformaciones Biyectivas en la Optimización de Problemas de Permutaciones, MAEB2025 (English version accepted for ECAI 2025).
In that work, the authors consider inversion vectors, Lehmer codes, and the RIM as a part of the same framework and use them with a unified notation. So, the published version is even more general than that in the present work.

W5. Several times, the authors mention the idea of unconstrained learning. I think it would help if the authors first refer to the mutual exclusivity constraint that appears naturally in permutations, and then suggest that the concept disappears in the inversion vectors space. However, in some sense, inversion vectors have also "constraints" since the integer numbers that can appear at each position are strict. However, the authors do not mention this.

### Questions
Taking into account the comments above:

Q1. What is the novelty of the work? Are you merely proposing to use masked language models (MLM) with alternative representations?

Q2. Can you describe any properties of the masked models learned, following Critchlow 1991?

Q3. Would it be possible to extend the idea of MLM as adapted from Uria 2016 in this paper to solve optimization problems?

If the authors consider the suggested comments and make changes to the paper so that it provides a much more comprehensive overview of the literature and its possibilities, I would be willing to reconsider my score.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose three other parameterizations of permutations as opposed to the typical inline notation when modeling distributions over permutations, and perform experiments using generative models to confirm the efficacy of these parameterizations.

### Strengths
The core idea seems novel and natural, as there are seemingly many ways to represent permutations with no intuitive reason to prefer the inline notation.  On the synthetic experiments the results are also quite good.

### Weaknesses
For this last task in particular, there are some issues.  The measurement is NDCG@k which as I understand it doesn’t require a distribution?  Even if one explicitly wanted a distribution over rankings for the sake of, say, uncertainty quantification, I also feel a somewhat straightforward baseline is missing from this task.  Namely, one could learn a distribution over each individual ranking of a film conditioned on a user, and then implicitly induce a distribution over rankings by sampling scores for each film and sorting them.

### Questions
Can the authors offer specific settings where one would like to learn a distribution over permutations in practice, using a MLM or any other generative model?  I feel there could be more motivation for the generative element of the paper.

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
4

### Summary
This paper addresses distribution learning over permutations by replacing the usual inline permutation representation with three factorized representations — Lehmer codes, Fisher–Yates draws, and Insertion-Vectors. The key idea is that these representations have element-wise conditional independence, so masked language-modeling (MLM) or autoregressive (AR) training can produce valid permutations with much more expressivity at low numbers of forward passes (NFEs) than inline notation.

### Strengths
1) **Clear, well-motivated idea.** 
Replacing inline notation with factorized bijections is simple but powerful: it directly addresses invalid outputs in fully-factorized masked models and provides a tradeoff knob between expressivity and compute.  
2) **Theoretical insight.** 
The paper formally characterizes why inline representations collapse to degenerate distributions under 1 NFE and proves a useful identity (Theorem 4.3) that enables efficient batched insertion-vector decoding.  
3) **Remarkable gains on benchmarks.** 
Method outperforms diffusion and relaxation baselines on jigsaw tasks; The paper also extend two new benchmarks: the cyclic and MovieLens experiments. Both results further prove that the choice of representation can be used to model beyond the delta distribution and to handle conditional generation tasks via  insertion-vector.

### Weaknesses
1. **Scalability to large n.**  The results of training/serving for very large rankings (e.g., thousands of items) remains unknown. Jigsaw and cyclic (n=10) are illustrative but small; MovieLens experiments are more realistic but limited to subrankings (n=50). Behavior at larger scale is untested.

2. **Representation selection requires prior knowledge.**
Choosing the best factorized representation (e.g., Fisher–Yates for cyclic structure) is sensible but introduces an extra design choice and possible brittleness when task priors are unknown.

### Questions
1. **Scaling to larger n.**  While the proposed approach shows strong results on moderate-sized permutations (e.g., n ≤ 50), its scalability to larger sets remains unclear. It would be helpful to include preliminary experiments on larger-scale settings (e.g., n = 100 or 200) to assess both accuracy and computational cost. Prior work such as Zhang et al. (2024) has demonstrated scaling on sorting tasks; including similar experiments would make the empirical evaluation more comprehensive.

2. **Representation selection and adaptability.**  The paper currently relies on manual choice of representation (e.g., Fisher–Yates for cyclic structures). It would strengthen the contribution to discuss or explore automatic strategies for selecting or combining representations—such as heuristic rules, task-dependent priors, or meta-learning approaches. Providing such guidance would make the method more practical and easier to adopt across diverse domains.

3. **Clarification on non-delta distributions (MovieLens benchmark).**  
The paper argues that most prior benchmarks (jigsaw puzzles) effectively correspond to learning *delta distributions* over permutations, whereas the proposed MovieLens re-ranking benchmark involves learning a *non-delta* target distribution. However, the paper does not clearly explain why the MovieLens setup yields a non-degenerate distribution. Could the authors elaborate on how the stochasticity or user-item sampling procedure induces a genuine distribution rather than a single ground-truth permutation? More details or visualizations of this distributional diversity would help clarify the distinction.

4. **Conditional independence under Lehmer codes.**  
The authors mention that Lehmer code representations exhibit conditional independence across indices, which supports efficient factorization. However, this property seems to hold only when the target permutation distribution is a delta (i.e., deterministic). For non-delta or multimodal permutation distributions, the dependencies across indices should reappear. Could the authors clarify under what assumptions the conditional independence claim holds, and whether it still approximately holds for stochastic targets such as those in the MovieLens setting?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Defining distributions over permutations is a useful machine learning primitive for applications that involve ranking, assigning, etc.

This paper provides new, interesting ways to represent distributions over permutations by leveraging a variety of classic methods for encoding permutations (e.g., using swaps, insertions, etc). Each of these can be coupled with a language model (masked, autoregressive) for defining a distribution over permutations.

The representational capacity of various representations is discussed and experiments on both synthetic and real data are used to compare performance.

### Strengths
The paper invokes a number of interesting, classical formulations of permutations that I was not familiar with. I learned a lot from reading the paper. 

The paper is a good example of how to combine classic algorithmic formulations with neural networks to get novel/flexible distributions over structured objects.

### Weaknesses
I found the  (num function evaluations) NFE formalism confusing. This is used to provide a tradeoff between compute cost and expressivity of the resulting distribution. I understand the NFE=1 and NFE=k regimes, but I don't know how to interpret a value in between and I don't understand how the partitioning was chosen. I also don't know why this variable was so central in the experiments. To me, I was most interested in the different representations, not sweeping over values of NFE.

I have some questions about the experiments. See below.

### Questions
I found the MLM vs. AR exposition confusing. It took me a while to realize that things are sampled independently within a masked region. It may have been helpful to define MLM

I found the setup for experiment 5.2 (cylic permutations) confusing. In what sense are you assessing whether the uniform distribution was learned? Can't these models be used to assess the likelihood of a given permutation? Could you check that the likelihood scores are approximately uniform (suggesting that the distribution is approximately uniform)?

In sec 5.3 (MovieLens), why were only certain representation approaches used? I would have expected to see a comparison of all the approaches in Figs 5 and 6. Also, MovieLens has been used for benchmarking a wide variety of learning-to-rank models. I don't expect you to necessarily get SOTA performance, but it would have been useful to understand roughly how well your methods compare.

### Soundness
3

### Presentation
3

### Contribution
3
