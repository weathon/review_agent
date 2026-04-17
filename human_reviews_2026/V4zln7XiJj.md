# Scheduling Your LLM Reinforcement Learning with Reasoning Trees

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 2

## Abstract
Using Reinforcement Learning with Verifiable Rewards (RLVR) to optimize Large Language Models (LLMs) can be conceptualized as progressively editing a query's 'Reasoning Tree'. This process involves exploring nodes (tokens) and dynamically modifying the model's policy at each node. When combined with data scheduling, this process yields further gains in data efficiency and accuracy.
However, existing RLVR data scheduling methods typically rely on path-based metrics to rank queries, overlooking the reasoning tree structures of these queries.
In this paper, we introduce a novel metric, namely **Reasoning Score (r-score)**, which measures the query's learning difficulty based on the structure of its reasoning tree.
Based on the r-score, we propose the **Reasoning Tree Schedule (Re-Schedule)**, a scheduling algorithm that constructs a curriculum progressing from structurally simple (high r-score) to complex (low r-score) queries.
Experiments on six math-reasoning benchmarks show that Re-Schedule significantly improves average accuracy, achieving gains of up to 3.2\%.
These strong results validate our approach and demonstrate that a structural understanding of the reasoning tree provides a more powerful and principled foundation for RLVR data scheduling.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces R-Score, a novel metric to quantify the learnability of queries in RL. R-Score measures how many node modifications are required in a reasoning tree to achieve a correct answer, reflecting how suitable a sample is for training.
In experiments, curriculum learning based on R-Score achieves better performance than accuracy-based and random scheduling methods, demonstrating that R-Score can effectively guide sample selection during RL training.

### Strengths
1. The paper proposes a new metric, R-Score, to evaluate whether a question is appropriate for RL training. The design aligns well with the intuition that queries that require fewer reasoning corrections are easier to learn, and the empirical results consistently support this idea.

2. The paper is clearly written, with detailed explanations of the R-Score computation, experimental setup, and ablation studies. Figures and tables effectively illustrate the trends and improvements.

### Weaknesses
1. The method is only validated on the Qwen2.5 and math task, which is a special combination that makes the results less general [1]. To establish broader applicability, more models and diverse reasoning tasks should be included.

2. The computation of R-Score depends heavily on several hyperparameters such as the branching factor, depth, and node modification budget. These choices can dramatically increase computational requirements, making large-scale use impractical without further optimization or approximation.

[1] Wu, Mingqi, et al. "Reasoning or memorization? unreliable results of reinforcement learning due to data contamination." arXiv preprint arXiv:2507.10532 (2025).

### Questions
Is the R-Score static during training, or do you recompute it as the model improves?
Since the model’s competence evolves, a query that was previously hard may later become easy. Would it be beneficial or necessary to periodically update the R-Score to reflect this changing learnability?

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
3

### Summary
This paper challenges the use of path-based accuracy for RLVR data scheduling, arguing it is a poor proxy for true learning difficulty. The authors reframe the problem around the "Reasoning Tree" structure, proposing the "r-score": a novel metric that quantifies a query's "learning potential" based on the maximum accuracy gain achievable under a limited "node editing budget". They introduce "Re-Schedule", a curriculum learning algorithm that uses the r-score to prioritize structurally simple queries first. Experiments on six math benchmarks show that Re-Schedule significantly outperforms strong baselines, achieving state-of-the-art results.

### Strengths
1.  Insightful Problem Definition: The paper provides a valuable reframing of the scheduling problem, moving from simple accuracy to the more nuanced "structural complexity" of a query's reasoning tree. The "node-editing" analogy is highly intuitive.

2.  Principled Metric (r-score): The r-score is a well-defined metric that attempts to directly quantify "learning potential" rather than just static difficulty. This is a more principled approach to capturing the learning dynamics of a query.

3.  Strong Empirical Results: The Re-Schedule method achieves state-of-the-art results across six math benchmarks. The significant accuracy gains (up to 3.8%) over strong RLVR and scheduling baselines compellingly demonstrate the method's effectiveness.

### Weaknesses
1.  Limited Experimental Domain: The method's effectiveness is only validated on math-reasoning tasks. It is unclear if the r-score framework will generalize to other domains. Code generation would be a good benchmark with verifiable rewards.

2.  Insufficient Discussion of Related Work: The paper needs to better position its contributions relative to the broader curriculum learning (CL) field. A deeper comparison with other methods that also estimate 'learning potential' (e.g., via gradients or uncertainty) is warranted.

### Questions
3.  High Computational Overhead: The method relies on a substantial offline pre-computation stage to build approximate trees and calculate r-scores for all queries, which may be computationally infeasible for very large-scale datasets.

4.  Hyperparameter Sensitivity: The r-score's quality depends on tree approximation hyperparameters ($k$, $d$, $w$). As Table 3 and Table 4 suggests, performance is sensitive to these choices, but the paper lacks a principled guide for setting them.

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
3

### Summary
This paper focuses on designing a more effective data scheduling strategy that leverages the structural relationships among data samples through reasoning trees. The proposed R-score captures these internal structural dependencies, while traditional path-based metrics such as accuracy fail to do so. Building on this metric, the Re-Schedule algorithm prioritizes high R-score samples early in training, using curriculum learning principles to balance exploration and exploitation and enhance generalization.

### Strengths
The reasoning tree formulation is intuitive and well motivated. The authors clearly articulate the limitations of path-based metrics and provide convincing evidence supporting the structural view of reasoning. Empirically, the results demonstrate consistent improvements: both Qwen2.5-Math-7B and Qwen2.5-7B models outperform all baselines across multiple mathematical reasoning datasets. The ablation studies and supporting analyses further validate the soundness of the approach.

### Weaknesses
1. Since reasoning trees can grow exponentially with depth, the authors approximate them using a fixed structure defined by (k,d,l). While this keeps the computation tractable, it may oversimplify the actual reasoning process. Although ablation studies explore nearby settings, all tested configurations remain within the same order of magnitude. How sensitive is performance to these parameters? Would larger or deeper trees provide further gains? Or would they introduce diminishing returns? Some intuition for choosing these parameters in advance would be valuable.
2. The paper does not fully discuss the computational cost of sampling k^d branches, even with KV-cache reuse. A quantitative analysis of runtime or memory cost would help assess scalability to larger models or non-mathematical reasoning tasks.
3. Writing and clarity: In Section 4, the meaning of the blue line in Figure 2 is unclear. Does it correspond to queries with more complex reasoning structures? Likewise, the terms potential samples and stagnant samples are not well explained in the caption. Clarifying these details would improve readability.
4. The R-score metric is intuitive but relies on several hyperparameters \eta,w_{\min},w_{\max},k,d,l, which may need careful tuning.

### Questions
1. In Figure 3(a), do the illustrated parameters correspond to k,d=3? It would be helpful to align the figure with the numerical example in Lines 256–257 for consistency.
2. What would happen if training started with lower R-score (i.e., more difficult) samples? Would the model still converge, and how would this affect generalization?
3. Since the evaluation temperature is set to 1.0, which introduces more randomness, how many evaluation runs were averaged? Are the reported improvements statistically significant?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work proposes Reasoning Tree Schedule (Re-Schedule), a scheduling algorithm for reasoning tasks that constructs a curriculum from structurally simple to complex problems. The complexity is defined as r-score, a metric that measures the difficulty for learning based on the structure of the reasoning tree. Experiments on six math benchmarks show that Re-Schedule achieves better performance than existing methods, with an average improvement of up to 3.2%.

### Strengths
* This work introduces a novel metric, r-score, to measure the difficulty of reasoning tasks in terms of the structure of the reasoning tree. The proposed metric shifts the focus from only accuracy to structure-based difficulty assessment.
* With the proposed metric, the authors construct a curriculum from structurally simple to complex problems. This method is potentially applicable to other tasks and domains.
* The curriculum is used to train a reasoning model, which shows improved performance on six math benchmarks.
* The ablation studies show the importance of the proposed hyperparameters.

### Weaknesses
* Missing detailed analysis of the variation of the results. For example, since AIME 24/25 only has 30 problems each, the results may be not stable. The variation can be more than the 1.7% improvement on Qwen2.5-7B in Table 2.
* Adding to the point above, while prior works such as SimpleRL-Zoo (which was cited in Table 1 and 2) uses metrics such as Avg@32 to make the result more stable, the authors did not mention whether averaging is performs for Re-Schedule. If not performed, the results may be not stable.
* This work performs experiments with a fixed token interval (l=200). However, no ablations are performed to investigate the impact of the token interval, which determines how coarse the reasoning tree is.
* Comparisons with prior works are not performed in a fair way. For example, SimpleRL-Zoo makes use of GSM8k and MATH datasets only, while this work makes use of DAPO training set. Since the proposed method is a training schedule algorithm, the comparison should be performed in the same training set with the baseline (or with a reproduced result using the DAPO training set).
* Authors did not provide theoretical analysis of the proposed method. The proposed node-editing motivation is intuitive, but not formally justified or validated beyond empirical observation in Figure 2.

### Questions
* Are results obtained with averaging over multiple runs? If not, the authors are encouraged to provide additional experiments with averaging to make the results more stable. This affects the validity of the method and is an important factor of the score and recommendation.
* Does the optimal token interval depend on the model type and the difficulty of the question set (e.g., Qwen2.5-7B-Math vs Qwen2.5-7B, GSM8k vs DAPO training set)?

### Soundness
2

### Presentation
2

### Contribution
2
