# Rethinking LLM Evaluation: Can We Evaluate LLMs with 200× Less Data?

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Benchmark suites for large language models are growing faster than our ability to pay for them. Even when training is already expensive, many use cases require repeated evaluation across many checkpoints, variants, and competing systems, and the steady expansion of benchmark suites increasingly turns evaluation into a bottleneck in tokens and compute. This scale changes what ``useful data'' means. Instead of asking whether an instance is good for training one model, we ask **which instances are necessary to keep the collective ordering of many models stable.** We analyze redundancy at the instance level and find repetition in both the text and the ranking patterns induced across models. Based on this observation, we formulate benchmark compression as a subset optimization problem that targets accurate score reconstruction and ranking preservation at the same time. We propose EssenceBench, a coarse-to-fine framework with three stages: redundancy-aware filtering with text and ranking signals, fitness-driven subset search with an iterative genetic algorithm and a fixed surrogate predictor, and attribution-guided refinement for better coverage under tight budgets. Across multiple leaderboards, EssenceBench achieves lower reconstruction error and stronger ranking preservation than prior approaches while reducing selection time. On HellaSwag with 10K instances, EssenceBench preserves 95\% of model rankings within a 5\% shift using only 50 instances, a 200$\times$ compression.  The source code will be made available upon acceptance of the paper.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes EssenceBench, a coarse‑to‑fine framework for compressing LLM benchmarks so that rankings and overall scores can be reconstructed with far fewer evaluation items. The pipeline (i) performs coarse filtering to remove redundant items using text similarity (embedding cosine) and ranking similarity (correlation of model outcomes), (ii) runs a Genetic Algorithm to select subsets that minimize a GAM‑predicted reconstruction error (RMSE) of full‑set accuracies across models, and (iii) applies attribution‑guided selection using an Explainable Boosting Machine (EBM) to score per‑item contribution, group items (high/low/random attribution), and re‑search within the best group; this is iterated to improve diversity and convergence. Experiments on five benchmarks (GSM8K, ARC, HellaSwag, WinoGrande, MMLU) report lower RMSE and better rank preservation than MetaBench and other baselines, with large reported reductions (e.g., 25×–200× fewer samples on HellaSwag) and faster compression.

### Strengths
1. Casting benchmark compression as minimizing an error between full‑set and subset‑based scores is crisp and aligns the objective with leaderboard reconstruction
2. The paper quantifies both text redundancy and ranking redundancy and visually demonstrates non‑trivial redundancy on popular datasets
3. The three‑stage design is easy to follow and practically implementable.

### Weaknesses
1. The duplication filter “keeps the earlier item and discards the later one if similarity exceeds thresholds”, which introduces order bias and potential instability across random permutations of the dataset. No robustness check to ordering is reported.
2. The paper mentions bge‑m3 embeddings and ranking correlations, but concrete threshold values, normalization, and sensitivity are absent
3. Fig 2C shows wall clock hours, but the hardware, eval setup is not clear. How do you compare the eval time?

### Questions
1. In Section 3.3, the fitness uses a GAM g that maps subset accuracy s_j to full‑set accuracy y_j. It is unclear whether g is retrained per candidate subset or pre‑trained on a pool of subsets. The compute/variance implications differ drastically and affect the efficiency claim.

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
3

### Summary
For addressing the growing cost and redundancy of large-scale LLM benchmarking, the study presents a benchmark compression framework, EssenceBench. It aims to preserve ranking fidelity while substantially reducing evaluation cost for large language models. The approach combines redundancy-aware filtering with a genetic algorithm and attribution-guided refinement to identify representative subsets of benchmark data. Experimental results on five datasets show consistent improvements over prior methods. It highlights the potential of EssenceBench to make large-scale LLM evaluation more efficient and scalable.

### Strengths
1. The proposed method addresses an increasingly important and practical challenge in LLM evaluation concerning benchmark cost without losing ranking reliability.
2. The integration of fitness-based subset selection with attribution-based sample selection provides an effective paradigm for benchmark compression while maintaining the intrinsic properties of datasets (e.g., diversity).
3. Experimental results across five benchmarks and multiple baselines demonstrate the strong empirical performance of EssenceBench.

### Weaknesses
1. The sample redundancy phenomenon has been widely observed in prior studies on dataset quality assessment. The paper should more clearly differentiate EssenceBench from these existing efforts to highlight the necessity of Genetic Algorithms.
2. The study lacks the computational or time complexity analysis of the proposed framework, which may better demonstrate the feasibility of EssenceBench for datasets with different scales.
3. The parameter sensitivity analysis is limited, which only investigates the effects of the number of generations (gens) and refinement rounds (rounds). Other key hyperparameters (e.g. $\tau_{text}$, $\tau_{ranking}$ and $N_{\mathcal{P}}$) that likely have a substantial impact on compression quality are omitted. It raises the concern regarding parameter search complexity and the overall robustness of the proposed framework.
4. The experimental evaluation narrowly focuses on the extrinsic metric of prediction error (RMSE) and ranking consistency. The experiments lack some quantitative metric to evaluate intrinsic quality and characteristics of the compressed datasets (e.g., diversity, bias and so on).

### Questions
1. Could the evaluation be extended with objective metrics that assess the intrinsic properties of the compressed datasets (e.g., diversity [1], bias [2] and so on)
2. An analysis of the genetic algorithm's computational complexity and a justification for the number of generations required for convergence are needed.
3. A more comprehensive sensitivity analysis is required for the hyperparameters to verify the robustness of the proposed framework, along with a discussion on the parameter search strategy.
4. The definition 4 lacks sufficient details about how the ranking scores $\mathbf{r}$ are computed from several LLMs. 
5. The case study in Table 3 offers an interesting glimpse, but a broader analysis is needed to fully support the claims about model behavior. Could you include a dedicated appendix section with more examples across different datasets to better illustrate the duplication-aware filtering in EssenceBench?
6. In Figure 2, the legends of MetaBench and EssenceBench are mislabeled.
7. Part of notations are inconsistent. For example, Equation 12 defines $j \in \mathcal{I}(m)$ while the main text refers to $j \in \mathcal{I}(\mathcal{D}_{filtered})$

[1] On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey. 2024.

[2] Bias and fairness in large language models: A survey. 2024

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
The paper argues that sample-level redundancy makes many LLM benchmarks wasteful. The authors present EssenceBench, a coarse-to-fine pipeline that first removes redundant items, then searches for a compact subset using a genetic algorithm steered by a learned accuracy predictor, and finally refines the selection with attribution-based grouping to preserve coverage. The selection goal is to use a k-item subset to reconstruct full-benchmark scores. EssenceBench reports lower reconstruction error and strong rank fidelity.

### Strengths
- well-motivated problem. this paper addresses what I believe is an important and practical problem of expensive LLM evaluations
- the experimental results are thorough and consistent
- paper is written clearly with good explanation and clear pseudo-code and a good appendix. also has thorough ablations and analysis.
- I like the formalization of LLM benchmark compression as an optimization problem and use GA + attribution refinement

### Weaknesses
- experiments focus on static, multi-choice benchmarks. I think applicability to open-ended, interactive, or multi-modal tasks may be limited or at the very least is untested
- the GA + EBM pipeline involves several hyperparams and the thing costs should be discussed more explicitly as its likely to be expensive. Other methods like SMART filtering [1] don't require this tuning but seem to enable accurate compression. The filtering method could be adapted to your setup as a baseline or complementary approach.
- the coarse filtering step relies on manually set thresholds τ_text and τ_ranking. How sensitive is the method to these choices? Different benchmarks may require different thresholds, limiting generalizability.

[1] Gupta, Vipul, et al. "Improving model evaluation using smart filtering of benchmark datasets." arXiv preprint arXiv:2410.20245 (2024).

### Questions
- I think generalizability is my biggest question/concern. Will the compressed benchmarks work for new model architectures not in the train set? How often do these benchmarks need to be recomputed, etc?
- I would like to see a computation cost trade-off? How many GPU hours to compress each benchmark? It might be helpful to add a clear table showing benchmark size, compression time, compressed size, and number of evaluations needed to break even. This will be nice for practitioners wondering if they should use this method. 
- How were τ_text and τ_ranking chosen? Is there a principled way to set these automatically?
- Is there a theoretical justification for using GA specifically? Did you compare against other optimization approaches? Why is this particularly suitable?
- I'm curious if this approach can be extended to eval tasks where scoring is more complex (e.g. using LLM as a judge)

### Soundness
3

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
The paper propose using genetic algorithm to compress LLM evaluation set. Experiments show ranking is preserved with a small shift.

### Strengths
1. Paper is clearly written, well motivated, and generally easy to follow.
2. Authors identify and analyze the redundancy of the widely used LLM benchmarks.
3. Compressing LLM benchmark by 25x (lossless) to 200x (5%) shows the potential for a far smaller “validation set” or “test set”.

### Weaknesses
1. The main benchmarks being used (GSM8K, ARC, HellaSwag, WinoGrande, MMLU) are considered saturated for the frontier LLMs these days. Newer LLMs are challenged with harder benchmarks, and harder benchmarks contain fewer instances since they are more challenging to come up with. I encourage authors to address why their overall method may adapt to modern and future benchmarks.
2. Similar to above, authors did not test newer benchmarks. For example, how would authors method extend to agentic benchmarks (SWEBench / tau2-bench) and multimodal benchmarks (MMMU etc.) I believe a compression of those benchmarks are needed at the moment. And after all benchmark compression is a one-off effort, releasing compressed benchmarks on author’s end would be a good contribution to the community.

### Questions
1. Original benchmark performance is treated as ground truth in the paper. However, there remains doubts on that. For example, suppose if a benchmark has 50% of problems as 1+1=2, this would definitely mean that the compressed benchmark will behave differently than the original benchmark, but this isn’t a necessarily bad thing. Maybe a compressed benchmark at a certain level could better reflect an LLM’s capability.

### Soundness
3

### Presentation
3

### Contribution
2
