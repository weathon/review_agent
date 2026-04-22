# Neuro Symbolic Graph Generative Modeling

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
We challenge the prevailing deep generative paradigm for graphs, exemplified by diffusion models, which is computationally intensive, lacks formal guarantees, and offers little user control. We introduce Neuro-Symbolic Graph Generative Modeling (NSGGM), which recasts graph generation as sequence modeling plus constraint satisfaction. NSGGM learns a vocabulary of subgraph tokens and uses an autoregressive sampler to propose token sequences, which an SMT solver then assembles into valid graphs by enforcing learned structural rules and user-defined constraints. This hybrid design avoids costly iterative refinement while providing correctness by construction and interpretable control. Across molecular and general graph benchmarks, NSGGM achieves state-of-the-art quality with fine-grained, user-steerable control, which current methods lack. Thus, NSGGM offers a practical path to trustworthy, targeted graph synthesis with broad applicability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper is motivated by 3 key challenges faced by diffusion method: 
high computational cost, the need for post-hoc filtering to ensure valid outputs, and limited interpretability.

The proposed model, NeurSymb, introduces a neuro-symbolic approach to address these issues. 
It decomposes a graph into substructures, treating each as an atomic “word” in a symbolic vocabulary. 
These substructures can then be assembled to generate larger graphs, much like constructing sentences from words.

Unlike previous sub-block–based approaches, NeurSymb introduces two innovations:

1 Structure-guided generation through the use of structural prompts provided by users.

2 Validity by construction, ensuring that all generated outputs are guaranteed to be valid without the need for post-processing.


The model employs an autoregressive decoder, which generates intermediate outputs that are subsequently passed to a symbolic solver. 
This solver deterministically enforces logical and structural constraints, ensuring correctness.

In the preliminary section, the authors discuss the theoretical foundation based on Satisfiability Modulo Theories (SMT), which underpins the symbolic reasoning process.

NeurSymb’s approach to splitting graphs into subgraphs resembles the JT-VAE framework, 
but extends it by allowing multi-tree motifs, offering greater flexibility and structural expressiveness.

The experimental results show that NeurSymb achieves perfect validity (100%), 
demonstrating the strength of its symbolic constraints. 
However, the model performs poorly on most other metrics, 
indicating a trade-off between validity and other aspects of generative quality such as diversity or fidelity.

### Strengths
the idea of using neurosymbolic method is well motivated and interesting.

### Weaknesses
the idea is not novel, there is much overlap with jt-vae.

experimental results are quite weak, especially on metrics other than validity.

We compare NSGGM against state-of-the-art graph generator"
baselines from 2019/ 2020 are not SOTA, the field moves quickly.

### Questions
what are the limitations / tradeoffs of your method?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a new paradigm for generating graphs through Neuro-Symbolic Graph Generative Modelling (NSGGM), which has three main steps: learning a vocabulary of subgraph tokens, autoregressively sampling from that vocabulary, and using an SMT solver to assemble the sampled tokens into a graph. The key claimed advantages are guaranteed validity and fine-grained user-controllable constraints on output graphs.

Unfortunately, the experimental validation has significant weaknesses. First, the evaluation is incomplete and inconsistent: metrics differ across tables without justification, confidence intervals are missing for larger datasets, and critical information such as inference time and vocabulary coverage is not reported. Second, NSGGM shows poor performance on distributional metrics across all larger molecular datasets—for instance, achieving 0.0 scaffold diversity and FCD of 41.26 on MOSES compared to ~1.0 for diffusion methods. Third, while the paper claims controllability as a key advantage over existing methods, this is demonstrated only through qualitative examples without any quantitative comparison to conditional generation baselines.

While guaranteed validity through constraint satisfaction is conceptually appealing, the paper does not make a convincing case for when this justifies the substantial degradation in distributional fit. Overall, NSGGM presents a novel framework but fails to demonstrate practical scenarios where it outperforms existing methods.

### Strengths
It is laudable that the paper introduces a new framework (NSGGM) and proposes an alternative to currently prevailing diffusion-based models. The NSGGM framework is well explained and Section 3 is well structured, clearly presenting the decomposition, tokenization, and SMT-based assembly process. The paper provides formal guarantees (Propositions B.1-B.4) for correctness-by-construction, which is a notable theoretical contribution. The demonstrations in Figure 2 showcase a higher level of user control through scaffold completion and constraint-driven synthesis, representing capabilities that are difficult to achieve with purely neural methods.

NSGGM achieves perfect validity (100.0%) across all benchmarks and very high novelty on MOSES (100.0%), though the latter should be interpreted in context with the poor distributional fit metrics (FCD 41.26, 0.0 scaffold diversity).

### Weaknesses
My principal concern with the paper is threefold, all aspects related to the evaluation of the framework.

__1. Rigour__
The model is evaluated both on QM9 with and without implicit hydrogen. While this is a good approach, the metrics reported are minimal and not the same, for no clear reason. 
- Training time is only reported for implicit hydrogens for example. 
-  In table 2, the legend claims diffusion-based baselines "retain higher Unique/Novelty" but novelty is not reported in the table.
- In table 3 and 4, no confidence intervals are specified.
- In table 3, a higher FCD score is indicated to be preferable, in table 4 it is the opposite. This should be corrected or explained.
- Differing metrics are evaluated in table 3 and 4, despite the task, namely larger molecule generation, being the same.

__2. Performance__
The model scores by definition 100.0 on validity and has decent performance on uniqueness/novelty on the larger datasets, albeit poorer than the models it compares to. However, performance on distributional metrics is poor across the board. The Kullback-Leibler divergence is weak and the FCD performance (which from context I assume to be the Fréchet ChemNet Distance) is not competitive. Both SNN and Scaf metrics are poor, and also not explained. On MOSES, the scaffold diversity is 0.0, indicating the model generates from a significantly narrower distribution than the training data. While the authors acknowledge this as a trade-off, they do not provide analysis of vocabulary coverage or demonstrate that the learned tokens can adequately represent the chemical space of the training data.

__3. Evaluation of constraint satisfaction__
The paper's primary claimed advantage over existing methods is fine-grained, user-steerable control through constraint satisfaction. However, this is demonstrated only qualitatively through examples in Figure 2. There is no quantitative evaluation comparing NSGGM's constraint satisfaction capabilities against relevant baselines. Without such evaluation, it is difficult to assess whether the claimed controllability advantage justifies the significant performance degradation on standard metrics.

In addition, ablations on the different novel aspects of the paper would be helpful.

Minor points:

* Metrics like FCD, SNN, Scaf should if not described, at least appear in their spelled out form at some point.
* Atomic and molecular stability cannot be assumed to be known in the graph community and should be defined.
* Figure 2a.: I think it should say "row", not "column" in the legend.

### Questions
* In practice, high but not perfect validity is not a major problem in applications of graph generation, since invalid graphs can simply be removed and more generated if novelty is high enough. Can the authors clarify the usefulness of guaranteeing validity?
* Are there ways that the performance of the model can be improved on the chosen benchmarks?
* The authors should discuss how their approach relates to fragment based molecular design and older genetic algorithm approaches such as: https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c.
* In figure 6 in the appendix, large ring structures are generated that look very different from what real molecules are expected to look like. Can the author comment on that?
* How does vocabulary size scale with dataset size? Can the vocabulary size be reported for the datasets in the paper?
* How sensitive are results to the minimum support threshold in frequent subtree mining?
* Can you provide quantitative evaluation comparing constraint satisfaction rates and output quality against conditional generation baselines? What complex constraints has NSGGM successfully satisfied that existing methods cannot handle?

Although answering these questions would improve the paper, the main shortcoming lies in the evaluation performance of NSGGM.

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
4

### Summary
The paper proposes a novel generative model for molecules using a symbolic approach. Specifically, it encodes user/application constraints explicitly and uses an SMT solver (in combination with an autoregressive decoder generating samples) to check them.

### Strengths
- It's an interesting idea to use a logical representation and tools in the molecule representation setting.

### Weaknesses
- The questioning of diffusion models seems somewhat wrong-placed given the recent rather impressive results [1].
- To me, it is unclear if SMT solvers are really needed/useful here. Even the older, existing methods achieve rather high validity rates. Moreover the solver is only applied, for checking, so the logic does not seem to be encoded within the model.
- Apart from UniGEM, which is dropped after Table 1 (why?), the considered baselines are all rather outdated (2019-2023). There are various more recent ones, e.g., [2-4].
- Because of the former, it is impossible to judge the potential impact and advantages of the approach.

[1] https://hannes-stark.com/assets/boltzgen.pdf

[2] Ketata et al. Lift your molecules: Molecular graph generation in latent euclidean space. ICLR'25

[3] Wang et al. Learning-Order Autoregressive Models with Application to Molecular Graph Generation. ICML'25.

[4] Lee et al. GenMol: A Drug Discovery Generalist with Discrete Diffusion, ICML'25

### Questions
--------------------------------

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
The authors introduce Neuro-Symbolic Graph Generative Modeling (NSGGM), a novel neural-symbolic framework reframing graph generation as a dual problem of sequence generation and constraint-satisfaction problem solving. An autoregressive decoder proposes a blueprint sequence of subgraph tokens, which a symbolic SMT solver then assembles into a final graph, providing the advantages in controllability and verifiability. Extensive experiments shows that the superiority of NSGGM in metrics Validity and Stability.

### Strengths
- The authors propose a novel neural-symbolic framework to treat graph generation as a compositional task.
- The proposed neural component, a lightweight autoregressive decoder, achieves faster training compared to diffusion-based approaches.
- The experimental evaluation is comprehensive, benchmarking against several state-of-the-art graph generation models across multiple graph-related datasets.

### Weaknesses
- While NSGGM ensures validity and stability in molecular generation tasks, it struggles to encode certain molecular constraints (e.g., uniqueness or stereochemical rules) that SMT solvers cannot easily capture. This limits its applicability to novel drug design, where such constraints are critical.
- The generative capacity of NSGGM appears limited by the fixed vocabulary size, which constrains the diversity and novelty of generated graphs. As reflected in the results, diversity and novelty metrics degrade substantially, even though validity and stability remain high.

### Questions
- How do the authors determine the granularity of subgraph tokens, *i.e.*, the smallest blueprint or atomic unit used in graph decomposition?
- Could the authors elaborate on the meaning of hard and soft constraints? It is unclear why these constraints are categorized as structural constraints.
- The paper would benefit from an analysis of how vocabulary size and subgraph decomposition affect the trade-off between validity, diversity, and novelty in the generated graphs.
- What is the inference latency of NSGGM compared to diffusion-based methods such as DiGress?

### Soundness
3

### Presentation
2

### Contribution
2
