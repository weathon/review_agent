# Beyond Binary Preferences: A Principled Framework for Reward Modeling with Ordinal Feedback

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Reward modeling is crucial for aligning large language models with human preferences, yet current approaches lack a principled mathematical framework for leveraging ordinal preference data. When human annotators provide graded preferences on a Likert scale (e.g., significantly better, better, slightly better, negligibly better), existing methods typically apply ad-hoc heuristics, such as margin terms or scaling factors, to loss functions derived from binary preference models like Bradley-Terry. These approaches lack an underlying mathematical model for how ordinal preference data is generated. We present a theoretically grounded framework that formulates reward modeling with Likert scale preferences as a discrete ordinal regression problem. We derive two loss functions from this formulation: a negative log-likelihood loss and an all-threshold loss, both of which learn threshold parameters that naturally capture the ordinal structure of preferences. Unlike existing heuristic methods that manually specify fixed margins or scaling weights, our approach learns these parameters directly from data within a coherent probabilistic framework. Experimental results on multiple benchmarks demonstrate that our ordinal regression approach consistently achieves competitive or superior performance compared to existing heuristic methods across diverse evaluation categories including chat, reasoning, and safety tasks. Our work provides the first principled mathematical framework for incorporating Likert scale preferences into reward model training, moving beyond ad-hoc modifications of binary preference models to enable more effective utilization of fine-grained human feedback.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a principled mathematical framework for training reward models from ordinal preference data, such as 7-point Likert scales. The authors argue that existing methods for handling such data—like adding margin terms or scaling factors to the standard binary Bradley-Terry loss—are ad-hoc heuristics.

The primary contribution is to reframe reward modeling with ordinal feedback as a discrete ordinal regression problem. From this established statistical framework, the authors derive two loss functions:
1.  A probabilistic Negative Log-Likelihood loss based on the ordered logit model.
2.  A margin-based All-Threshold loss inspired by large-margin classifiers.

A key advantage of this framework is that the thresholds partitioning the reward-difference space are learned directly from the data, rather than being manually specified. The paper also provides two theoretical results: Theorem 3.1, which proves the necessity of regularization to ensure a finite optimal solution when learning thresholds, and Theorem 3.2, which provides a theoretical justification for using symmetric thresholds.

### Strengths
1.  **Principled Formulation:** The paper's core strength is its reframing the problem as discrete ordinal regression connects LLM alignment to a mature field of statistics and provides a solid theoretical foundation, moving beyond the intuitive but "ad-hoc" heuristics currently in use.
2.  **Learned Thresholds:** The shift from manually-specified margins to data-driven, learned thresholds is a significant conceptual improvement. This makes the method more general and removes a key hyperparameter tuning burden from practitioners.
3.  **Generalizability:** The provided extension to DPO in Appendix A also demonstrates the generalizability of the framework.

### Weaknesses
1.  **Notational Inconsistency in Formulation:** There is a critical notational contradiction in the paper's core formulation. Section 3.3 explicitly states that the threshold $\zeta_0$ is *excluded*. However, the probabilistic model in Equation (14) implicitly requires $\zeta_0$ to compute the probabilities for $z=-1$ (which uses $\zeta_{z+1} = \zeta_0$). As written, this makes the central NLL loss function ill-defined.
2.  **Missing Ordinal Evaluation of Baselines:** The paper's entire motivation is that its method *better leverages ordinal information*. However, the main experimental results (Tables 1 & 2) only report binary preference accuracy, which fails to measure this. The paper introduces excellent ordinal metrics (MAE and Accuracy-within-k) in Appendix H, but reports them only for its own method. To substantiate the paper's core claim, it is essential to compare these ordinal metrics against the baselines.
3.  **Potentially Unfair Baseline Comparison:** The baseline methods (Margin BT, Scaled BT) are highly sensitive to their specific hyperparameters (the margin values $m(z)$ or scaling/probability values $p(z)$). While Appendix F details the hyperparameter search for the proposed methods (e.g., learning rates, regularization $\lambda$), it is not stated that the crucial $m(z)$ or $p(z)$ hyperparameters for the baselines were also tuned. If the authors used fixed, non-optimized values from prior work for the baselines while carefully tuning their own model, the comparison is not a fair one.
4.  **Lack of Experimental Robustness:** All results in Tables 1 and 2 are reported from a single experimental run (using "fixed random seed 0"). Given the known high variance in LLM fine-tuning, single-seed results are insufficient to make strong claims of superiority, especially when many of the reported gains are small (1-3%). Reporting the mean and standard deviation over multiple seeds is necessary to demonstrate robust and statistically significant improvements.

### Questions
See the weaknesses.

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
The paper makes the case that human preferences are inherently ordinal and current reward modeling approaches do not incorporate this in a principled manner (these rely on naive margins or scaling terms). The authors introduce two different methods for this (a) the first uses the CDF over the rating levels to ensure that preference ratings learned are well calibrated with annotations in existing datasets, and (b) the second ensures that model score differences fall within the desired intervals (of levels). These are formulated as loss functions that are compared against the more common Bradley-Terry-based formulations with 3 LLMs (Llama 3.1 8B, Zephr 7B, Mistral 7B) on two datasets (HelpSteer 2 and 3) that come with rating levels for preferences. In general, ordinal reward modeling outperforms the baselines to demonstrate the validity of the proposed technique on real data and confirm the theory behind the design choices (asymmetric rewards, etc.).

### Strengths
1) It's very intuitive to incorporate a measure of 'strength' of preference into the reward modeling. The authors do a good job of creating a mathematical formulation to support this, backed by empirical results. 

2) The experiments are very well done with care taken for reproducibility and robustness of takeaways. 

3) The formulation reduces back down to something very much like DPO when you set two levels of preference, which has a nice 'backward compatibility' to existing methods.

### Weaknesses
1) The method assumes a consensus among annotators about different levels of ratings. Human preference datasets are already noisy, and this creates an additional source of noise (calibration among annotators), so while we do better at benchmarks, it is unclear if these reward models successfully capture the true distribution of user preferences well.

2) I think some fine-grained error analysis and qualitative examples would really help provide some context for some of the results. For example, do the ordinal methods make fewer 'large-margin' errors (option was strongly preferred but scored wrongly) than the Bradley-Terry models? Can you plot something like a calibration curve for 'strength of preference' vs accuracy? Are there any examples where the labels are consistently flipped from incorrect to correct, or vice versa, by the ordinal ranking methods?

### Questions
1) Are humans well calibrated in providing Likert-style preference judgments on pairs of examples? I imagine the proposed method is less tolerant to this noise than a more aggregated binary preference judgement. This is independent of your method, but I think the premise is that this is a natural way we make judgments (which I agree). It's just that I suspect it leads to more disagreement when we create datasets. 

2) Does the performance of the method vary with the granularity of levels provided i.e. K=3 vs 5 vs 7? Is this purely empirical/dependent on the model/dataset?

3) Does HelpSteer filter out examples with high disagreement? I wonder if this works in favour of your model in that many examples might be high-margin? 

4) It's not a 'weakness', but there isn't a clearly preferred loss formulation among the variants proposed. It is not clear to me if this could be due to variance in optimization error or something like noise in the training data. Do you have a clear recommendation to practitioners using your technique?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper argues that treating human feedback as binary throws away signal when labels come in strengths (e.g., “slightly” vs. “much” better). It reframes reward modeling as ordinal regression on the reward difference, learning a set of thresholds that separate preference levels instead of hand-tuning margins or weights. The paper introduces two principled objectives—an ordered-logit NLL and an all-threshold (margin) loss—that jointly fit the reward and these thresholds, with L2 regularization to keep training stable and interpretable. Empirically, the paper shows that across different base model families and HelpSteer-style data, the approach trains RM with competitive performance as existing methods measured by RM/Rewardbench.

### Strengths
originality: The paper introduces a new theoretical framework that uses **ordinal regression** on the reward difference to do reward model training. It introduces 2 major variations of such objective – ordered-logit NLL and an all-threshold (margin) loss, and also provide the practical recipe for training with these objectives stably.

quality & clarity: The theoretical analyses and proofs are quite thorough and the experiment part comprehensively include different model backbones and benchmarks. The table presentation is clear and readable.

significance: I find the idea of leveraging the difference in rated score as a proxy of strength is interesting. The paper also provides different potential methods on how this signal can be leveraged. It’d have more significance if the experimental result shows more obvious and consistent gain.

### Weaknesses
1. Overall weakness: It would be helpful if the authors provide more explanation of the method's motivation. From the experimental result itself (e.g. Table 2), the proposed method doesn't produce better benchmark results than other cited methods for quite a lot of the combinations tested. If that's the case, the it indicates that learning the ordinal relationship doesn't buy much gain in the performance? 
2. Overall weakness: The paper introduces 3 different objective designs. Although in the theory part, the authors provide some design principles for each objective, I think the paper lacks a general link between the theoretical proposal and empirical results. E.g. the experimental results indicate that symmetric NLL is better but I don't see a strong theoretical illustration of why that should be better. 
3. Overall weakness: Both datasets used in the paper are Helpsteer series of work. I wonder how common is the exact score rating in popular preference dataset. If the method needs to first assume that preference dataset contains the exact score, this may significantly limit where the method can be applied.

### Questions
1. Regarding weakness 1: could authors provide more motivations on why this method is better if it doesn't produce a better performance?
2. Regarding weakness 2: could authors provide more theoretical support of why we see a experimental result like that among the 3 proposed objectives?
3. See weakness 3:

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a framework for learning reward models from ordinal (Likert-scale) preference data, replacing ad‑hoc extensions of Bradley-Terry with discrete ordinal regression. The threshold between different levels are now also learnable parameters. Two families of losses are considered from the literature of ordinal regression: (i) a probabilistic ordered logit negative log-likelihood (NLL) over learned thresholds that partition reward differences, and (ii) an all‑threshold (AT) margin-based loss. The $L_2$ regularizer is introduced by considering the limit behavior of scaling going to infinity. Symmetric and asymmetric derivations are discussed. An optimization reparameterization is adopted to guarantee ordering. Experiments on HelpSteer2/3 training data with 7B-8B backbones evaluate on RM‑Bench and RewardBench, suggesting the NLL-Symmetric variant generally performs the best.

### Strengths
1. Modeling of ordinal feedback from the perspective of discrete ordinal regression is natural. This enables the authors to adopt the literature already exists and makes more sense than simply heuristic choices.

2. Theory is insightful. Theorem 3.1 justifies the choice of regularizer. Theorem 3.2 characterizes one sufficient condition of symmetry.

3. The paper is clearly written and well organized.

4. Empirical experiments show gains on several datasets/models. The NLL‑Symmetric variant is often best or competitive. The additional ordinal metrics go beyond binary accuracy and are also valuable points.

### Weaknesses
1. Joint learning of threshold and reward is still challenging. Scale identifiability remains under‑addressed. Regularizing thresholds cures the unbounded loss, but the joint scaling of reward head and thresholds can still be weakly identifiable. Anchoring strategies (e.g., fixing one threshold gap, adding mild L2 on reward head, or a temperature/variance parameter) and a short calibration section would improve interpretability.

2. Positioning requires more contrast. The claim of being the first principled ordinal framework for reward modeling should be more carefully justified against existing works that already explore ordinal feedback for RMs (the paper itself cites several heuristic and non‑heuristic directions).

### Questions
1. Have you considered per-annotator (or random-effects) thresholds to capture personalized/individualized ordinal scales?

2. Can you show performance vs. fraction of ordinal data and under controlled label noise?

### Soundness
3

### Presentation
3

### Contribution
2
