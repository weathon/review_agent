# A Genetic Algorithm for Navigating Synthesizable Molecular Spaces

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
Inspired by the effectiveness of genetic algorithms and the importance of synthesizability in molecular design, we present SynGA, a simple genetic algorithm that operates directly over synthesis routes. Our method features custom crossover and mutation operators that explicitly constrain it to synthesizable molecular space. By modifying the fitness function, we demonstrate the effectiveness of SynGA on a variety of design tasks, including synthesizable analog search and sample-efficient property optimization, for both 2D and 3D objectives. Furthermore, by coupling SynGA with a machine learning-based filter that focuses the building block set, we boost SynGA to state-of-the-art performance. For property optimization, this manifests as a model-based variant SynGBO, which employs SynGA and block filtering in the inner loop of Bayesian optimization. Since SynGA is lightweight and enforces synthesizability by construction, our hope is that SynGA can not only serve as a strong standalone baseline but also as a versatile module that can be incorporated into larger synthesis-aware workflows in the future.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents SynGA, a synthesis-constrained genetic algorithm that operates directly on molecular synthesis routes. By using custom crossover and mutation operators, it ensures all generated molecules are synthesizable and can be efficiently optimized for both analog search and property objectives in 2D and 3D. The method is further enhanced with a lightweight ML-based building block filter, and its variant SynGBO achieves state-of-the-art performance on benchmark property optimization and docking tasks. Overall, the work demonstrates a flexible and effective approach for synthesis-aware molecular design.

### Strengths
1. The idea of measuring synthesizability directly from synthesis trees is insightful.

2. The paper is well-structured and clearly organized, making the methodology and results easy to follow.

3. The experiments are solid and thorough, and the results look pretty convincing.

### Weaknesses
1. Some parts of the method feel a bit engineering-heavy, and the paper is missing some discussion. For example, when they say “we couple the NAM with a more powerful predictor that filters the samples post-hoc from SynGA”, it’s not clear how much the extra predictor actually contributes beyond the NAM. Also the five mutation operations.
2. The algorithm relies on expert-defined SMARTS strings (templates), and here they only use 91 reactions templates. If more templates were used, like those extracted in Retro*, the algorithm might face more pressure. That said, this may not be a big issue since related work also uses a similar number of templates.

### Questions
1. In line 166, $R(M(S1),M(S2)) \not= \emptyset$ is a bit hard to follow, though the meaning can still be understood.

2. I wonder what would happen if the building block filtering considered all the building-block molecules used in multiple feasible synthesis paths for a target, instead of just a path.

3. The table reports averages over 5 seeds, but the variance is not shown.

4. In Table 2, it would be interesting to know how the results of SynGA (MLP) would change if its runtime were also 80m, matching SynFormer, and how its performance varies under different computational budgets.

### Soundness
3

### Presentation
4

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
This paper proposes SynGA, a novel genetic algorithm that directly models and manipulates synthesis routes to efficiently explore the combinatorially vast space of synthesizable molecular pathways. The method demonstrates promising performance across multiple benchmarks, including docking and property optimization tasks.

### Strengths
- The proposed method, SynGA, enables direct control of synthesis routes without ML tuning, while ensuring synthetic feasibility.
- The paper is well structured and clearly explains the methodology for designing synthesis routes directly.
- SynGA and SynBO show good performance across various benchmarks.

### Weaknesses
- Although one of the key advantages of SynGA is its ML-free design, its performance drops without ML-guided building block filtering. In practice, meaningful results appear only when combined with ML filtering. Moreover, In both PMO and docking tasks, the performance improvement is achieved only when using the computationally expensive BO algorithm, which weakens the claim of being ML-free.
- While the method performs well across benchmarks, results are reported for only a few targets.

### Questions
- One of SynGA’s strengths is its ML-free structure, but its performance drops when ML-guided building block filtering is not used (probably because the search space is too large). While the process of constructing synthesis routes directly through custom genetic operators is novel, the performance seems to rely heavily on the filtering method. Could additional datasets such as Enamine[1] or ZINC250[2] be included for comparison? Evaluating the effect of the proposed approach (especially the ML-guided building block filtering) across different datasets in the synthesizable analog search task would help demonstrate the generality of SynGA.

- In molecular design, diversity can be as important as chemical validity [3]. Would it be possible to evaluate diversity in the analog search experiments?

- Could you clarify why only a subset of tasks is reported in Table 3? According to [4], the LIT-PCBA dataset includes 15 protein targets.

- Similarly, Table 5 seems to report results for only a few targets. Prior works such as RXNFLOW[5] and 3DSynthFlow[6] include results not only for ALDH1 and ESR_ant, but also for ADRB2, ESR_ago, and FEN1. Could you clarify the selection criteria for the reported targets? If there was no specific reason for selecting only these three, results for additional targets—as in previous studies—would make the evaluation more comprehensive.

- Could optimization curves, similar to those shown in previous PMO papers (e.g., PMO[7], MOLLEO[8], DyMol[9]), be included for additional analysis? These curves might help better illustrate the sample efficiency of SynGA and SynGBO.

[1] Lee, Seul, et al. "Rethinking Molecule Synthesizability with Chain-of-Reaction." arXiv preprint arXiv:2509.16084 (2025).
 
[2] Irwin, John J., et al. "ZINC: a free tool to discover chemistry for biology." Journal of chemical information and modeling 52.7 (2012): 1757-1768.

[3] Renz, Philipp, Sohvi Luukkonen, and Günter Klambauer. "Diverse hits in de novo molecule design: Diversity-based comparison of goal-directed generators." Journal of Chemical Information and Modeling 64.15 (2024): 5756-5761.

[4] Luo, Shitong, et al. "Projecting molecules into synthesizable chemical spaces, 2024." URL https://arxiv.org/abs/2406 4628.

[5] Seo, Seonghwan, et al. "Generative flows on synthetic pathway for drug design." arXiv preprint arXiv:2410.04542 (2024).

[6] Shen, Tony, et al. "Compositional Flows for 3D Molecule and Synthesis Pathway Co-design." arXiv preprint arXiv:2504.08051 (2025).

[7] Gao, Wenhao, et al. "Sample efficiency matters: a benchmark for practical molecular optimization." Advances in neural information processing systems 35 (2022): 21342-21357.

[8] Wang, Haorui, et al. "Efficient evolutionary search over chemical space with large language models." arXiv preprint arXiv:2406.16976 (2024).

[9] Shin, Dong-Hee, et al. "Dynamic many-objective molecular optimization: Unfolding complexity with objective decomposition and progressive optimization." Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI). 2024.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a template-based genetic algorithm for molecular generation. Furthermore, they propose a model-based fragment filtering step that improves the performance of the method. The approach is evaluated on multiple benchmarks.

### Strengths
- The approach is sound, and presents an interesting contribution.
- The area of focus (synthesizability-constrained generation) is of high importance.
- The paper is clearly written and, for the most part, easy to follow.
- The experiments are comprehensive and measure the performance of the proposed approach in multiple relevant tasks.

### Weaknesses
- The compared baselines are not standardized in terms of reaction set, fragments/synthons, and the number of allowed reactions used. The choice of chemical vocabulary greatly affects the performance. It is not clear to what extent reported performance improvement compared to baselines is due to a better algorithm, or the vocabulary.

### Questions
- I am familiar with an earlier version of the paper submitted to a previous conference. I noticed that the performance for 3DSynthFlow in Table 5 has decreased compared to the previous version. This is a substantial change, since previously this was the best performing method, outperforming SynGA and within the standard deviation of SynGBO. As a result, the change affects the conclusions to some extent. Can authors clarify why the results were changed?
- More of a suggestion, but in my opinion it would make more sense to put the paragraph describing SynGBO in a separate subsection.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
very solid paper proposing a new genetic algorithm for molecular design combined with ML based building block selection; with convincing evaluation and a good contribution to the field.

### Strengths
- clear description of the work
- good, well-motivated method contribution
- proper evaluation in line with prior work

### Weaknesses
No major weaknesses.

if I had to really pick some, I'd say the language of the paper is maybe in parts unnecessarily formal, and the contribution isn't paradigm-shifting (but nevertheless very solid!).

### Questions
- do the authors think that number of oracle calls is really important in practice?

### Soundness
4

### Presentation
3

### Contribution
3
