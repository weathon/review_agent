# MAGO: Beyond Fixed Hyperparameters with Multi-Objective Pareto Optimization for Hybrid LLM Reasoning

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Large language models (LLMs) with advanced step-by-step reasoning capabilities have achieved remarkable performance in complex problem-solving through chain-of-thought (CoT) reasoning. However, uniformly applying elaborate reasoning to all queries creates substantial computational inefficiency, as many problems can be solved directly without extended reasoning chains. Current hybrid reasoning approaches rely on static hyperparameters and heuristic single-objective optimization, leading to suboptimal trade-offs and poor adaptation to varying task complexities. To address these limitations, we propose a multi-objective adaptive generation optimization (MAGO) framework, which integrates multi-objective optimization with dynamic adaptive weighting into hybrid reasoning. MAGO optimizes three competing objectives simultaneously: accuracy (maintaining solution correctness), efficiency (minimizing computational costs through appropriate mode selection), and calibration (ensuring mode selection aligns with model capabilities). The framework employs Pareto frontier maintenance with correlation-aware optimization to automatically explore the full trade-off space, avoiding the spatial constraints that limit fixed-weight approaches to narrow cone-shaped regions of the objective space. Unlike existing methods requiring manual hyperparameter tuning, MAGO's Pareto optimization dynamically adapts weights based on task complexity and training progress, achieving principled and adaptive decision-making across varying problem complexities. Comprehensive evaluation on mathematical reasoning benchmarks including AIME, Minerva Algebra, MATH-500, and GSM-8K shows $2.2\times$ to $3\times$ token-efficiency gains and relative accuracy improvements of $0.6\%$ to $9.4\%$ over heuristic baselines, while remaining competitive with the strongest task-specific models. Additional experiments on CommonsenseQA and MedQA further confirm the framework's generalizability beyond mathematics, achieving $1$ to $2\%$ higher accuracy and approximately $2\times$ efficiency improvement without additional fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes MAGO, a multi-objective adaptive generation optimization framework for hybrid LLM reasoning that learns when to use short versus think modes. MAGO formulates training as a three-objective problem—accuracy, efficiency, and calibration—combining a Pareto frontier over weight vectors with a correlation-aware selection rule and exploration to avoid “cone entrapment” inherent in fixed scalarizations. The adaptive weight mMAGO(x) scales the control-token loss in a GRPO-style objective, aiming to stabilize mode selection and prevent collapse. On math benchmarks (AIME 2024, Minerva Algebra, MATH-500, GSM8K), the paper reports 2.2–3× token savings and small-to-moderate accuracy gains over several baselines.

### Strengths
MAGO is described with explicit equations: adaptive objective (Eq. 5), calibration via confidence binning and exponentially decayed historical accuracies (Eqs. 9–13), Pareto frontier maintenance and selection (Eqs. 14–21), and integration into the GRPO-style loss where mMAGO(x) scales control-token terms (Eq. 22).

Figures 4A–C provide interpretable evidence that MAGO mitigates mode collapse and reduces variance relative to vanilla GRPO and DeGRPO during RL, with later, more stable crossovers between modes.

### Weaknesses
The text states that MAGO achieves the highest accuracy on AIME, Minerva Algebra, and MATH-500, but Table 1 shows DeepSeek-R1-1.5B surpasses MAGO on Minerva Algebra (0.9577 vs 0.9483), and QMath-1.5B surpasses MAGO on GSM8K (0.8572 vs 0.8469). The abstract and Section 5 claim accuracy improvements of 2.6%–9.4%; the largest improvements visible in Table 1 over strong baselines appear smaller on some tasks (e.g., GSM8K +0.5 points over Thinkless). These discrepancies need reconciliation.

DeGRPO is analyzed qualitatively in Figure 4 but not reported as a quantitative baseline in Table 1. Router baselines include a 7B router, while MAGO uses a 1.5B single model; compute-normalized comparisons (e.g., wall-time, FLOPs/token including routing overheads) are not provided. It is unclear whether all baselines share the same SFT data, RL steps, and inference constraints.

The contributions highlight logical inference and general problem solving, but experiments focus on math-only datasets. The conclusion acknowledges this limitation; however, the earlier claims of breadth may mislead.

Overall, I think the most important thing is the proposed method still lack of the novelty and the performance is mainly tested on math problem which increases the concerns that whether the proposed method can be generalized to other domains.

### Questions
Could you add a quantitative DeGRPO baseline in Table 1 using your training setup, given that MAGO is positioned as an advancement over GRPO/DeGRPO?

Expectations in Eqs. (6)–(8): During training, how many (c,a) samples per x are used to estimate Saccuracy, Sefficiency, and Scalibration? Are these single-sample estimates, and if so, how do you control variance for the correlation matrix Ct (Eq. 16)?

Since the contributions mention logical inference and general problem solving, do you have held-out non-math evaluations (even small-scale) using the same trained model to substantiate cross-domain claims? If not, please limit claims or provide such experiments.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper targets hybrid thinking, driven by the commonly known fact that reasoning-enabled LLMs often waste compute by using long CoT on easy inputs. With the emergence of deep reasoning models such as DeepSeek-R1, hybrid thinking is a trending topic with prevalent work such as Thinkless. Existing hybrid reasoning methods rely on fixed hyperparameters (e.g., a single controller weight) and single-objective heuristics, which lock the system into narrow, suboptimal regions of the accuracy–efficiency trade-off. 

This paper takes a different approach along this line. The authors propose to reframe hybrid reasoning as multi-objective optimization over three goals: accuracy, efficiency, and decision calibration. They call the framework MAGO. In short, MAGO introduces a dynamic weighting mechanism over those three objectives, learned via Pareto frontier maintenance plus correlation-aware weight selection. Experiments on AIME-24, Minerva Algebra, MATH-500, GSM8K show that MAGO brings good improvements by 2.6% - 9.4%.

### Strengths
- ``S1``:  The paper is very well motivated. The authors use a section (Sect. 2) with pilot experiments to demonstrate their motivation. In particular, it shows that fixed weights cause mode imbalance and dataset-specific sensitivity (Fig. 2A–B), and that scalarization constrains search to a cone-shaped region of objective space (Fig. 2C–D), missing better trade-offs. This strongly motivates the proposed methodology.

- ``S2``: The proposed MAGO makes sense. Experiments on several math reasoning benchmarks show both token savings and accuracy gains.

### Weaknesses
- ``W1``: The scope of the evaluated benchmark is a bit limited and narrow (math). Additional experiments on SuperGPQA and CommonsenseQA would strengthen the contribution by showing the generalization of MAGO in extended domains.

- ``W2``: In Tab. 1, the authors are suggested to include the reference for each method.

- ``W3``: Computational overhead is one of my concerns. I notice that the authors have admitted the limitations of extensive overhead by Pareto optimization in the last section. It would be great if the authors could show some computational complexity analysis.

### Questions
- ``Q1``: Why choose answer-token max-prob as confidence instead of sequence-level confidence or verifier-based scores?
- ``Q2``: Does MAGO generalise to other domains rather than math reasoning?

### Soundness
3

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
4

### Summary
This paper proposes MAGO (Multi-objective Adaptive Generation Optimization), a framework for training hybrid reasoning systems in LLMs that dynamically choose between short (direct) and think (chain-of-thought) modes. The key innovation lies in reformulating the hybrid reasoning optimization as a multi-objective optimization problem balancing three competing objectives: accuracy, efficiency, and decision calibration. Rather than using fixed hyperparameters as in existing methods, MAGO employs Pareto frontier maintenance with correlation-aware weight selection to dynamically adapt optimization weights during training. Evaluation on mathematical reasoning benchmarks (AIME, Minerva Algebra, MATH-500, GSM-8K) shows efficiency improvements of 2.2x-3x while simultaneously improving accuracy by 2.6%-9.4% compared to baseline methods like Thinkless and CoT-Valve.

### Strengths
1. The paper identifies a genuine limitation in existing hybrid reasoning systems—the dependence on static hyperparameters that vary significantly across datasets. The reformulation as a multi-objective optimization problem is novel for this domain, and the motivation clearly demonstrates why fixed-weight approaches fail (Figure 2)

2. The Pareto frontier maintenance approach is mathematically well-motivated. The use of correlation-aware weight selection (Equation 17) to handle interdependencies between objectives addresses a real challenge in multi-objective optimization. 

3. The paper evaluates on multiple mathematical reasoning benchmarks and includes several comparison baselines (router-based methods, model merging, CoT-Valve)

### Weaknesses
1. While accuracy and efficiency are straightforward objectives, the decision calibration objective (Equation 8) lacks rigorous justification. The approach relies on binned historical accuracy statistics, but several concerns arise: (1) How are the bins (Equation 10) initialized during early training when insufficient historical data exists? (2) The exponential decay (Equation 13) may create staleness issues when task distributions shift. (3) The paper doesn't analyze how this objective theoretically prevents miscalibration or what formal guarantees exist. 

2. DeepSeek-R1-Distill-Qwen-1.5B. This is a relatively small model, and generalization to larger models (7B, 13B, or larger) remains unclear. Does the Pareto frontier maintenance scale computationally with model size?

3. The router baselines (Router Random, Router Q-7B) seem surprisingly weak—Router Q-7B achieves only 14.8% on AIME while Thinkless achieves 25%. This brings up there might be some possible implementation or configuration issues.

4. The paper mentions diversity filtering (Equation 21) but provides insufficient detail. How quickly does the frontier grow? What happens when |F_t| approaches |F_max|? The exploration mechanism (Equation 19-20) could lead to unbounded frontier growth early in training.

### Questions
1. Why is calibration defined as the difference between model confidence and empirical accuracy rather than other common calibration metrics (expected calibration error, Brier score)?

2. Can you evaluate on non-mathematical reasoning tasks (e.g., commonsense reasoning, multi-hop QA, code generation)? The current scope limits the paper's impact.

3. Visualize how the Pareto frontier evolves during training. How does frontier size grow? Do solutions converge or continue diversifying? When does the optimal weight selection stabilize?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the Multi-Objective Adaptive Generation Optimization (MAGO) framework, designed to address inefficiencies in hybrid reasoning approaches for large language models (LLMs). ​Current methods often rely on static hyperparameters and single-objective optimization, which fail to adapt to varying task complexities and result in suboptimal trade-offs between accuracy, efficiency, and calibration. ​MAGO leverages multi-objective optimization and dynamic adaptive weighting to optimize these three competing objectives simultaneously. ​By maintaining a Pareto frontier and employing correlation-aware optimization, MAGO explores the full trade-off space, avoiding the limitations of fixed-weight approaches. ​The framework eliminates the need for manual hyperparameter tuning and achieves principled decision-making across diverse problem complexities. ​Experimental results on mathematical reasoning benchmarks (AIME, Minerva Algebra, MATH-500, GSM-8K) demonstrate that MAGO improves computational efficiency by 2.2x to 3x while enhancing accuracy by 2.6% to 9.4%. ​The paper also highlights MAGO's ability to prevent mode collapse during training and achieve stable optimization dynamics.

### Strengths
- The paper addresses a critical limitation in hybrid reasoning systems by reformulating the problem as a multi-objective optimization task, which is a significant advancement over static hyperparameter-based methods.

- The introduction of dynamic weight adaptation and Pareto frontier maintenance is innovative and eliminates the need for manual hyperparameter tuning, making the framework more adaptable to diverse problem complexities.

- The authors provide extensive experimental results on multiple mathematical reasoning benchmarks, demonstrating significant improvements in both accuracy and efficiency compared to existing methods.

- MAGO effectively prevents mode collapse during training, ensuring balanced reasoning mode selection and stable optimization dynamics.

### Weaknesses
- The framework is primarily evaluated on mathematical reasoning tasks, which may limit its generalizability to other domains. ​Broader validation across diverse reasoning tasks is necessary to establish its applicability.

- While the training overhead is claimed to be amortized over inference, the paper acknowledges that this may pose challenges for resource-constrained scenarios. ​Further analysis of the trade-offs between training cost and inference efficiency would strengthen the paper.

- The proposed framework involves intricate mechanisms such as Pareto frontier maintenance, correlation-aware weight selection, and dynamic weight adaptation. ​This complexity may hinder practical adoption and require significant expertise for implementation.

- The paper does not provide concrete examples or case studies of how MAGO could be applied in real-world scenarios, which would help demonstrate its practical utility.

### Questions
Line 216-218: Why do we take the logits corresponding to the final answer token to compute the raw confidence score instead of all the tokens in the generated answer.

### Soundness
3

### Presentation
3

### Contribution
2
