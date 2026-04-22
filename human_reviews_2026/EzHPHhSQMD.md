# Competition is the key: A Game Theoretic Causal Discovery Approach

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 2

## Abstract
We introduce a \textbf{game-theoretic reinforcement learning framework} for causal discovery in which a DDQN agent \emph{competes} with a strong baseline (GES or GraN-DAG) and always \emph{warm-starts} from the opponent’s graph. This yields three key guarantees: the learned graph is \emph{never worse} than the warm start, warm-starting \emph{accelerates convergence}, and with high probability the method selects the best candidate when $n$ is large enough. Formally, if 
$
n \ge \tfrac{8L^2}{\Delta_n^2}\log\bigl(\tfrac{2|C|}{\delta}\bigr),
$
then with probability $1-\delta$ the algorithm recovers the population-optimal graph. Here $L$ is a Lipschitz constant of the score function, $\Delta_n$ is the empirical gap between the best and second-best candidate scores, $|C|$ is the number of candidate graphs considered, and $ \forall\delta \in (0,1)$ is the failure probability. To our knowledge, this is the first finite-sample consistency result for an RL-based causal discovery method. Empirically, DDQN-CD matches or outperforms GES and GraN-DAG on standard benchmarks (Sachs, Asia, Alarm, Child, Hepar2) and scales to large graphs (Dream $\sim$100, Andes $\sim$220 nodes). Our results demonstrate that RL-based discovery can be simultaneously \emph{provably safe}, \emph{sample-efficient}, and \emph{scalable}, helping bridge the gap between theoretical guarantees and practical performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Idea. The authors propose a reinforcement learning based approach to causal discovery.  In particular, they use an RL-based approach to improve on an initial solution: they define their reward through a BIC score function, and the possible actions consist of adding, removing, or reversing an edge from a given solution. They benchmark their method on several real-world datasets and compare the performance to other existing algorithms in causal discovery.

### Strengths
The paper propose a causal discovery algorithm with reinforcement learning. Their method comes with some theoretical guarantees. I found particularly valuable having high probability results with finite samples, something that is often overlooked in the literature, which mostly focuses on theoretical guarantees at population level. Moreover the authors produced a nice collection of real world datasets over which to benchmark their algorithm. This strengthen the value of their empirical analysis, especially knowing that the field mostly focuses synthetic experiments which may not be relevant for real world applications.

### Weaknesses
**Presentation and clarity.** 

- The paper lacks any technical introduction to reinforcement learning, graph theory, and the problem of causal discovery. These are the key ingredients of the inference problem at hand and the algorithm the authors design to solve this problem. Readability would be positively impacted if the terminology used in the paper were appropriately introduced.
- I am a bit confused by the title in relation to the content of the paper: where is the *game-theoretic* part? My understanding of the method is that you take an initial solution (it could be randomly taken, or from some other method) and then you search the space of solutions, optimizing a score function.
- I would ask the authors to conduct a thorough comparison of their method with GES. The underlying ideas are very similar. It seems to me that the main difference is that while in GES first you remove all edges and then you add some back (so the available *actions,* in RL language, change across these two phases), here the authors allow themselves to take any action without splitting the search into two phases. But at the end of the day, they both optimize a BIC score, adding and removing edges to explore the search space. Can the authors elucidate points of contact and differences between these two methods?
- In the definition of the score, and hence of the reward, how is the likelihood function defined? E.g., in GES, we know that if the data are linear Gaussian, you want to encode that assumption explicitly in the likelihood. In this case, I didn’t find a definition of the optimized likelihood. Can you clarify that?
---

**Soundness and contribution**

- An important point in the causal discovery literature is identifiability: we know that without any functional assumptions, the unique graph is not identifiable. In this case, identifiability reduces to a Markov equivalence class, represented by a CPDAG. In the case of this paper, however, the interpretation of the inferred object remains unclear.
- Theorem 1 says that the solution found by DDQN-CD is always *better than or equal* *to* the initial solution. This is trivially achieved by any algorithm that returns the initial solution itself. In this sense, I wouldn’t feel confident in calling this a *safety guarantee* or even a Theorem. Another point about this is that it is unclear how the BIC score the authors define translates in terms of accuracy (e.g. their weighted score, or SHD)  of the inferred graph. Is it possible to conclude that a better score implies better SHD? Or something like that?
- The authors claim that their method always improves on the warm start solution. I have two concerns in this sense
    - If you look at the experimental results, the method basically never improves with respect to GES, both in terms of the weighted score and in terms of SHD accuracy.
    - However, I imagine that running their method on top of GES, grandag, or whatever initial algorithm is chosen, is time-consuming. In this sense, it seems that DDQN-CD is not really a good solution, compared to GES. One can only run GES and save time.

---
**Comparison to the literature**. The authors declare that algorithms with strong empirical performance lack finite sample guarantees. This is actually not precise. Giving an example that I know, the SCORE algorithm (Rolland et al., 2022) has finite samples guarantees (see Zhu et al., 2023) while being — in my experience and in several experimental setups, see e.g. Montagna et al., 2023 — better than e.g. GraNDAG, chosen by the authors as a method to compare to.

### Questions
Please refer to the weaknesses section

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes DDQN-CD, a reinforcement learning framework for causal discovery that refines graphs from baseline methods (GES or GraN-DAG) through sequential edge operations. The authors provide three theoretical guarantees: the output is never worse than the baseline, warm-starting accelerates convergence, and with sufficient samples, the method selects the best candidate with high probability. Experiments span small to large benchmark datasets.

### Strengths
The game-theoretic framing of causal discovery as RL-guided refinement of classical baselines is novel, and providing finite-sample guarantees for RL-based causal discovery addresses an important gap between empirically strong but theoretically ungrounded methods. The experiments span a large variety of real-world benchmarks of different sizes.

### Weaknesses
Main Concerns:

1. Unclear necessity of RL: Why is RL specifically needed for refinement rather than any other iterative improvement method? The theoretical proofs do not seem to explicitly leverage properties unique to RL algorithms, raising questions about what the RL component contributes beyond a standard local search procedure. Authors do not make clear what the added benefit of leveraging an RL approach is.

2. Marginal empirical improvements: For Asia, Sachs, Lucas, and Hepar2, there is negligible improvement over the original GES or GraN-DAG baselines. For Child and Alarm, the method only outperforms baselines under the authors' composite Score metric, while standard SHD shows competitive or worse performance. For large datasets, only NOTEARS and GraN-DAG are compared—polynomial-time methods like DirectLiNGAM and GOLEM should be included. Additionally, the absolute Score values (~0.10) are very low, questioning whether improvements are practically significant.

3. Loose hitting time bound (Theorem 2): The bound only considers the probability of following the shortest improving path. Why not include the probability of reaching G* via suboptimal paths? The current bound appears extremely loose and should be tightened. A similar issue may apply to Theorem 3, although its unclear from the proof description given in the main text.

Minor Concerns:

1. The theoretical results assume linear-Gaussian SEMs (Assumption A5), but this is introduced late. This should be stated upfront, as it significantly limits the applicability of the guarantees.

2. The authors should clarify in the introduction that continuous optimization methods typically use score functions derived from additive noise models, making them similar to functional causal model approaches.

3. Section 3 defers all methodology details to the appendix. While the high-level summary is adequate, key algorithmic details should be in the main text, with an expanded summary.

### Questions
1. Distinction from standard warm-start methods: How does warm-start + RL agent differ from simply warm-starting GES with the opponent graph? Both approaches start from an initial point and iteratively improve, so what specific advantage does the RL formulation provide?

2. Relationship to RL-BIC2: How does this work differ from RL-BIC2? Both use RL for causal discovery, and the main difference appears to be warm-starting from an existing method. Could the three theorems be proven similarly for RL-BIC2 applied to an initial graph?

3. Identifiability concerns: Linear-Gaussian SEMs only identify the Markov equivalence class (MEC), not unique DAGs [1]. Why do experiments evaluate DAG-specific metrics (e.g., SHD on exact edges) rather than MEC-invariant properties like the skeleton? This raises concerns about the validity of the experimental comparisons

[1] Causal Discovery with Continuous Additive Noise Models, Peters et al., JMLR 2014.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors propose a DDQN-based causal discovery method. Their method learns to generate causal graphs that improve the BIC score. Furthermore, they incorporate design choices in their algorithm that provide some theoretical assurances: 1) the output of the agent will have a better or equal score than the original one, 2) an initial graph estimate improves search for good solutions, and 3) good candidates are selected with high probability. They verify the 3rd theorem empirically and put the overall empirical performance of their algorithm in context with existing baselines.Soundness:

### Strengths
-Tackling theoretical questions in Reinforcement Learning (RL) is a significant and sparse area of research, and the paper's engagement with this aspect is a clear strength. This focus on theoretical underpinnings helps to advance the field by providing a more robust foundation for future work in RL for causal discovery.

-It's a promising idea to enhance RL for causal discovery by initiating the process with a graph determined by another algorithm. As the authors highlight in their theoretical discussion, this approach could significantly reduce the number of learning episodes required, making the process more efficient.

-The paper's inclusion of an extensive baseline comparison is a notable strength, particularly with the introduction of a comprehensive score.  The use of a comprehensive score further enhances the clarity and interpretability of these comparisons.

### Weaknesses
-While the algorithm effectively combines existing elements, its originality is somewhat constrained by the straightforward application of established concepts. 

-While the paper touches upon foundational and some contemporary literature, a more comprehensive exploration of recent advancements—such as GFlow Nets [1], RL for causal discovery [2], and other state-of-the-art methods [3]—would greatly enhance the understanding of its impact and relevance within the field.

-Given the inherent design of the algorithm, Theorem 1 seems to be a rather straightforward outcome.

-The current bound in Theorem 2 might be perceived as less impactful due to its considerable looseness (e.g., Amax=10, d=10, 1/π = 1e^23, even for a small graph). It would be great to rephrase the theorem to underscore the critical role of a better initial graph in significantly tightening this value. For improved understanding, it would also be beneficial to include Lemma 1 within the main text. 
The soundness of Theorem 3 is challenging to verify. While other theorems also lack some detail, the extent of this issue here makes both the theorem and its proof incredibly difficult to follow. For example, the loose introduction of A^diamond_n significantly obscures the proof.

-Refining the presentation of the mathematical concepts, background information, and general structure is recommended. For example, a more thorough introduction of symbols within the main text, rather than solely relying on an appendix table, would greatly enhance clarity.

-The assumptions seem to largely mirror the algorithm's design, which might prompt questions about their broader purpose and the generalizability of the theoretical findings beyond this particular algorithm. Clarification here would be beneficial.

-While it is beneficial to include numerous benchmarks, the claim of state-of-the-art (SOTA) performance appears to be based on inconclusive results, as several prominent SOTA methods (e.g., DCDI and ENCO) are not included in the comparison.

-In the results section, several statements appear to lack direct support from the presented data. For instance, the method does not seem to outperform GES on the Asia dataset as stated, and on smaller datasets, its performance often mirrors that of GES. Additionally, in some Gran-DAG configurations, the proposed method occasionally leads to a decrease in results. 

[1] Deleu, T., Góis, A., Emezue, C., Rankawat, M., Lacoste-Julien, S., Bauer, S. &amp; Bengio, Y.. (2022). Bayesian structure learning with generative flow networks. Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence, in Proceedings of Machine Learning Research.
[2] Sauter, A., Botteghi, N., Acar, E., & Plaat, A. (2024, May). CORE: Towards Scalable and Efficient Causal Discovery with Reinforcement Learning. In Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems 
[3] Lippe, P., Cohen, T., & Gavves, E. Efficient Neural Causal Discovery without Acyclicity Constraints. In International Conference on Learning Representations.

### Questions
Q: I would be curious about your insights into why sometimes adding this algorithm on top of Gran-DAG leads to worse results. Do you think this could mean that either BIC or the composite score you define are not suitable for evaluation downstream performance?

### Soundness
3

### Presentation
1

### Contribution
2
