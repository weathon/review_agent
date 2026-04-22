# Improving constraint-based discovery with robust propagation and reliable LLM priors

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 4, 8

## Abstract
Learning causal structure from observational data is central to scientific model-
ing and decision-making. Constraint-based methods aim to recover conditional
independence (CI) relations in a causal directed acyclic graph (DAG). Classical
approaches such as PC and subsequent methods orient v-structures first and then
propagate edge directions from these seeds, assuming perfect CI tests and exhaustive search of separating subsets—assumptions often violated in practice, leading
to cascading errors in the final graph. Recent work has explored using large lan-
guage models (LLMs) as experts, prompting sets of nodes for edge directions,
and could augment edge orientation when assumptions are not met. However,
such methods implicitly assume perfect experts or predictable error rates, which
is unrealistic for hallucination-prone and unstable LLMs. We propose MosaCD,
a causal discovery method that propagates edges from a high-confidence set of
seeds derived from both CI tests and LLM annotations. To filter hallucinations, we
introduce shuffled queries that exploit LLMs’ positional bias, retaining only high-
confidence seeds. We then apply a novel confidence-down propagation strategy
that orients the most reliable edges first, and can be integrated with any skeleton-
based discovery method. Across multiple real-world graphs, MosaCD achieves
higher accuracy in final graph construction than existing constraint-based methods, largely due to the improved reliability of initial seeds and robust propagation
strategies.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a causal discovery algorithm called MosaCD. MosaCD improves constraint based causal discovery methods using LLMs and non-collider-identification first approach. Techniques such as shuffling queries are employed to address hallucination problems in LLMs. Experiments are performed on several datasets to show the advantages of MosaCD compared to baselines.

### Strengths
1. The paper tackles an important problem of causal discovery
2. The paper is mostly written well and easy to understand

### Weaknesses
Major:

1. In the abstract and related work, the paper states that existing methods using LLMs for causal discovery assume perfect experts. However, methods that treat LLMs as imperfect/noisy experts do exist (see [1–3]).

2. As stated in Section 3, MosaCD relies on the idea that “estimated conditional independences can indeed be due to limited power of statistical tests.” However, the MosaCD algorithm starts from a skeleton obtained using conditional-independence tests (beginning with a complete graph). How is reliance on the skeleton obtained from CI tests justified?

3. In Line 5 of the algorithm, when there exists a semi-directed path between $X$ and $Y$ and also  an undirected edge $X-Y$, what is the intuition behind orienting $X\to Y$

4. In the MosaCD algorithm, LLMs are provided with p-values and conditional-independence statements. How can an LLM effectively make use of p-values?

4. In the experiments, MosaCD is not compared against other families of causal-discovery methods such as score-based and continuous optimization–based methods. Also, some LLM-based methods (e.g., [1,2]) are not included as baselines.

5. Why is the structural Hamming distance (SHD) metric not reported in the results? Why is only one base LLM (GPT-4o-mini) used? Why is “Meek” considered a separate baseline—shouldn’t it be part of the PC algorithm?


Minor:

1. Explicitly state the assumptions about the underlying causal graph, such as the no-hidden-confounding assumption.

2. In the introduction, it is written that constraint-based methods are computationally efficient. However, constraint-based methods can be computationally inefficient: when the graph size is large, the number of conditional-independence tests required for graph recovery can be prohibitively large. Also, it is written that constraint-based methods are state-of-the-art. Due to computational challenges, alternatives such as score-based methods and optimization-based methods are often used.


References:

[1] Long, Stephanie, et al. "Causal discovery with language models as imperfect experts."

[2] Vashishtha, Aniket, et al. "Causal inference using LLM-guided discovery."

[3] Ban, Taiyu, et al. "From query tools to causal architects: Harnessing large language models for advanced causal discovery from data."

### Questions
See the weaknesses section and provide the responses.

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
** Summary

The paper proposes a hybrid causal discovery method that combines constraint-based methods and LLM-based methods, where the authors claim the potential to mitigate the error of collider-first order in traditional constraint-based methods. The proposed method, termed MosaCD, first initialises a skeleton by the same process as PC, then, orients high-confidence edges based on LLMs’ generation, and lastly, determines the other edges’ directions by propagation principles, conditional independence test results, and LLMs’ generations. The paper provides theoretical analysis on the correctness of MosaCD, and the error rates of traditional constraint-based methods. Lastly, it provides experimental analysis to show MosaCD’s efficacy.

** Recommendation 

I would like to recommend a weak rejection to this paper as I am unclear about whether their method efficiently addresses the collider-first error. The paper raises an interesting problem of the traditional constraint-based causal discovery methods, and provides a formal proof, though it is based on strong assumptions. However, I am not clear whether the method can largely solve the collider-first issue because it basically uses the same conditional independence test results as PC, which means for each pair of nodes, there may exist at most one sepset. I am happy to increase my score if the authors clarify I am wrong on the following comments.

### Strengths
1. The paper studies the collider-first error of the traditional constraint-based methods, and provides a proof for that.
2. The proposed method tries to mitigate the error with some special design.

### Weaknesses
1. My main concern is that the method may fail to mitigate the claimed issue. Basically, since the sepsets are obtained by the same process as PC, each node pair may have at most one sepset. This means that MosaCD orients a v-structure is the middle node is not in this sepset. Then, this has the same problem. I understand that LLMs are used first, and this may reduce the error, however, LLMs can also bring much noise for its bad performance in causal reasoning.
2. The assumptions, e.g., 5.3 and 5.4, are quite strong. For instance, 5.3 is close to an asymptotic assumption, however, asymptotically, PC is guaranteed to orient correct v-structures. 5.4 then largely simplifies the graph.
3. Bnlearn benchmarks may not be good choices for evaluation because in general, LLMs have good memory on them. This undermines the convincingness of the work. Additionally, the authors may consider to compare to more recent LLM-based or hybrid methods.

### Questions
Questions see the weakness section.

** Minor comments

1. Define E_{seed} in Line 267
2. Line 156: dependence -> independence?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces MosaCD, a new causal discovery algorithm that integrates traditional constraint-based methods (like PC, CPC, and PC-Stable) with LLM priors to improve the accuracy and robustness of learning causal graphs from observational data. Classical approaches rely heavily on conditional independence (CI) tests and v-structure orientations, which are error-prone with finite samples and can cause cascading mistakes. MosaCD mitigates these issues by using LLMs to propose initial edge orientations, while filtering out hallucinations via shuffled multiple choice queries that exploit positional bias, keeping only consistent, high-confidence edges. It further introduces a confidence-prioritized propagation strategy that orients non-colliders before colliders, shown theoretically and empirically to reduce errors. The work seems promising and well grounded, however it can benefit from more empirical analysis, given high potential of data leakage on the benchmarks the paper evaluates on.

### Strengths
1. The paper makes a clear conceptual leap by embedding LLMs directly inside the causal discovery pipeline, rather than using them only as post-hoc annotators or priors.

2. The new confidence-down propagation rule orienting edges with higher CI confidence first and prioritizing non-collider identification over colliders is theoretically grounded and empirically validated.

3. The approach is modular and model-agnostic, thus enabling generalizable adoption.

4. Paper shows strong empirical improvements.

### Weaknesses
1. Shuffling order and re-querying is a well known way of averaging out positional bias which has been used in many prior papers [1]. So i feel the claim that the proposed framework removes hallucination is heavily overloaded and lacks novelty.

2. The datasets on which the framework is evaluated belongs from the well known BNLearn benchmark, with a very high chance of leakage during model's training. Recent studies like [2] confirm that LLMs have memorized the datasets from BNLearn that this study uses for evaluation. *I would strongly suggest tackling this claim*. 

3. Computational overhead from repeated LLM queries due to shuffling of query order seems like a big issue which is not properly addressed. This will become significantly higher when costly reasoning models would be used for inference for large causal graphs. 

4. Although MosaCD outperforms PC variants and recent LLM-based methods (ILS-CSL, SCP), it does not compare against hybrid or differentiable structure learning approaches (e.g., NOTEARS, DAG-GNN) that could also integrate priors. Thus, it remains unclear how its benefits scale relative to more flexible score-based frameworks. Also lack of comparison with recent work on integrating LLMs with traditional discovery methods [3][4][5] is a problem, since this paper proposes lacks comparison with recently proposed frameworks and fails to highlight how it stands out in comparison to those work. 

5. F1 score is a limited metric, especially when dealing with structural evaluation of a graph. Try incorporating Structural Hamming Distance and Topological Divergence [3] which are more graph specific metrics and relate to real world causal inference tasks like effect estimation.

6. Qualitative analysis of the LLM reasoning behavior will help understand the impact of the question framing effect, which is not provided currently. 

7. The paper emphasizes on PC, whose performance is heavily conditioned on the provided sample size. I am curious if their framework will help get significant improvement under scarce data settings, especially when the data samples are very low (<1,000). 

8. Please incorporate more for evaluation, as current analysis only limits to GPT-4o-mini. Try to incorporate open source models from Qwen, Llama family for wider adoption of your framework. 

References:
[1] Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions (Pezeshkpour et al., 2023)

[2] Realizing LLMs’ Causal Potential Requires Science-Grounded, Novel Benchmarks (Srivastava et al., 2025)

[3] Causal Order: The Key to Leveraging Imperfect Experts in Causal Inference (Vashishtha et al.,2025)

[4] Efficient Causal Graph Discovery Using Large Language Models (Jiralerspong et al., 2024)

[5] Causal Structure Learning Supervised by Large Language Model (Ban et al., 2023)

### Questions
Please try to answer each point enlisted under weaknesses. I believe the work has potential, but lacks in empirical rigor. I'll be happy to increase my scores if the author can show proof and tackle all points I mentioned under weaknesses. I believe the framework's effectiveness cannot be judged until the data leakage issue is resolved, since recent papers show that the graphs used for evaluation in this work from BNLearn are memorized, thus potentially skewing the results heavily.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
MosaCD is a constraint-based causal discovery algorithm that augments PC-style orientation with LLM-based seeding and a confidence-prioritized propagation strategy. It mitigates error cascades from imperfect CI tests by (1) using LLM queries to propose initial edge directions, (2) applying a shuffle-and-vote strategy to filter hallucinations, and (3) prioritizing orientations supported by reliable CI evidence—specifically orienting non-colliders first.

Theoretical analysis demonstrates that this reversal reduces orientation error rates compared to collider-first methods under noisy CI tests. Empirically, MosaCD achieves strong gains across 10 real and synthetic BNLearn graphs (5–76 nodes), outperforming PC, PC-Stable, CPC, and recent LLM-assisted methods (SCP, ILS-CSL). The method remains competitive even with noisy variable descriptions and across different LLM backbones.

### Strengths
- Conceptual elegant way of designing the algorithm and rethinking how LLMs should contribute to constraint-based discovery, avoiding naïve replacement.
-  Formal  analsys of soundness and an elegant asymptotic argument showing that non-collider orientation reduces expected errors.
- Rigorous empirical evaluation with 10 benchmarks, 3 skeleton variants, detailed ablations on hallucination filtering and seed robustness.
- Performance degrades gracefully under uninformative variables and different LLMs.
- Compatible with any PC-style skeleton learner.

### Weaknesses
- Dependence on large commercials models, no comparison to open-source smaller or distilled models presented. 
- The confidence-down propagation weights are heuristic; a probabilistic interpretation or adaptive β schedule would strengthen robustness.
- Evaluation metrics relies on F1 of edge orientations only; additional causal sufficiency or equivalence-class metrics (e.g., SID, SHD) would provide a fuller view.

### Questions
1) What is the computational cost of the shuffle-and-vote procedure relative to the total CI test time?
2) Could confidence propagation be formalized probabilistically (e.g., Bayesian edge belief updating)?
3) How does MosaCD handle contradictory LLM votes or inconsistent priors across iterations?
4) Can the approach scale to 100–500 nodes or use learned embeddings to approximate CI filtering?
5) Can the method work with smaller or distilled language models that could be run locally?

### Soundness
4

### Presentation
3

### Contribution
4
