# Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation

- Decision: Accept (Poster)
- Scores: 2, 6, 2, 6

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) offers a robust mechanism for enhancing mathematical reasoning in large models.
However, we identify a systematic lack of emphasis on more challenging questions in existing methods from both algorithmic and data perspectives, despite their importance for refining underdeveloped capabilities.
Algorithmically, widely used Group Relative Policy Optimization (GRPO) suffers from an implicit imbalance where the magnitude of policy updates is lower for harder questions.
Data-wise, augmentation approaches primarily rephrase questions to enhance diversity without systematically increasing intrinsic difficulty.
To address these issues, we propose a two-dual MathForge framework to improve mathematical reasoning by targeting harder questions from both perspectives, which comprises a Difficulty-Aware Group Policy Optimization (DGPO) algorithm and a Multi-Aspect Question Reformulation (MQR) strategy.
Specifically, DGPO first rectifies the implicit imbalance in GRPO via difficulty-balanced group advantage estimation, and further prioritizes harder questions by difficulty-aware question-level weighting.
Meanwhile, MQR reformulates questions across multiple aspects to increase difficulty while maintaining the original gold answer.
Overall, MathForge forms a synergistic loop: MQR expands the data frontier, and DGPO effectively learns from the augmented data.
Extensive experiments show that MathForge significantly outperforms existing methods on various mathematical reasoning tasks.
The code and augmented data are all available at https://github.com/AMAP-ML/MathForge.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper targets mathematical reasoning by emphasizing harder problems from both the optimization and data sides within a unified framework, MathForge. MathForge addresses this with a new training algorithm and a targeted augmentation pipeline designed to push models toward robust, generalizable reasoning.

Algorithmically, DGPO (Difficulty-Aware Group Policy Optimization) introduces (i) difficulty-balanced group advantage estimation (DGAE) and (ii) difficulty-aware question-level weighting (DQW), which upweights questions with lower mean accuracy within a batch.

On the data side, MQR (Multi-Aspect Question Reformulation) augments each problem—while preserving the gold answer—across three axes: adding narrative background (noise tolerance), introducing abstract terminology (conceptual grasp), and nesting sub-problems (multi-step, cross-domain reasoning).

### Strengths
1. The paper is generally well-written and easy to follow, with a clear description of the method.
2. The paper provides intuitive visual demonstrations to help better understand the paper.

### Weaknesses
1. **Limited novelty.** Several design choices appear incremental relative to prior work. For example, restricting the token-level loss averaging to only the number of valid queries in the batch (`Equation (3)`) is conceptually very close to DAPO’s online filtering, which also discards uninformative samples to stabilize updates. Likewise, the proposed Difficulty-Aware Question-Level Weighting overlaps with established difficulty-prior methods that upweight harder items [1]. 

2. **Claims that lack sufficient evidence or run counter to intuition.** The discussion around “sharp gradient fluctuations” on `Line 144–145` seems at odds with the usual motivation for valid-query filtering (faster training and fewer zero-gradient issues). It is also unclear why all-correct/all-incorrect groups (zero-gradient) would *induce* fluctuations rather than increase variance. Please clarify this with quantitative diagnostics: gradient-norm/variance traces across steps, per-batch valid-token ratios, and an ablation toggling the valid-query criterion to show its true effect on stability. In addition, the argument on `Line 178–181` that samples with “at least one correct response” are especially valuable may conflate genuine learnability with chance hits (guessing). A more reliable analysis could incorporate token-level entropy, answer-consistency across rollouts, and response-pattern statistics (e.g., self-consistency or calibration curves) to disentangle “solvable but hard” from “lucky” cases.

3. **Training cost and efficiency.** From the ablations in `Tab.1`, MQR appears to drive a substantial share of the gains, yet the efficiency implications are unclear. Please report a training cost–performance gain analysis across multiple augmentation data scales, including training time, needed memory. It would also be helpful to control for total training data size (or time) when comparing proposed method with others.

4. **Fairness and completeness of comparisons.** In this paper, comparisons are only provided on GEOQA-8k, which raises two concerns: (i) the works for other algorithms may not conduct experiments on this dataset, and (ii) there is no control for equal data volume or sampling, which can confound conclusions. Please add broader and fairer comparisons: match total training data sizes across methods, use more base training dataset (e.g., beyond GEOQA-8k) to demonstrate robustness. 

5. **Title–claim alignment (“HARDER IS BETTER”)**. While the paper introduces Multi-Aspect Question Reformulation, it does not provide solid evidence that the augmented items are *actually harder*. Beyond the qualitative examples and the accuracy gains, please include direct hardness validation to substantiate this claim. 

References:

[1] GRPO-LEAD: A Difficulty-Aware Reinforcement Learning Approach for Concise Mathematical Reasoning in Language Models. https://arxiv.org/abs/2504.09696

### Questions
See `Weaknesses` part.

### Soundness
2

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
4

### Summary
This paper argues that Group Relative Policy Optimization (GRPO) and its variants exhibit a critical limitation: their advantage estimation introduces an implicit bias where the magnitude of policy updates is lower for harder questions. And the authors propose a Difficulty-Aware Group Policy Optimization (DGPO) algorithm to debias the magnitude of policy updates. For data-centric viewpoint, this paper points out that existing augmentation approaches primarily rephrase questions to enhance diversity, without systematically increasing their intrinsic difficulty and propose a Multi-Aspect Question Reformulation (MQR) strategy. Extensive experiments demonstrate the method outperforms existing methods on various mathematical reasoning tasks.

### Strengths
1. This paper proves that traditional GRPO has an implicit bias and could be resolve with DGPO in small modification. 
2. This paper propose a data synthesis method MRQ which shows performance improvement.

### Weaknesses
Lack of depth analysis between GRPO and DGPO under different settings, e.g. queries in different difficulties, different types of datasets. More is discussed on questions section.

### Questions
1. The definition of DGAE is similar to GRPO but replace second central moment to first moment of denominator and total magnitude is 1 proved by B.3, what is the meaning of multiplying G.
2. could GRPO has better performance then DGPO if given queries near 50% accuracy since it has the largest advantage magnitute.
3. Notice this paper use a strong model gpt-o3 to augment dataset in MQR, what is the success rate and cost to generate new queries and is it capable to employ on less capable models. A comparison with baseline rephrasing method is preferred. 
4. what is the potential explaination in Fig. 1(b) the output length is reduced significantly than GRPO.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper mainly introduces 3 techniques on top of GRPO:
1. Difficulty-Balanced Group Advantage Estimation (DGAE), which guarantees the sums of the estimated advantage magnitude across different groups are the same.
2. Difficulty-aware Question-level weighting (DQW) to encourage training on hard questions, which is similar to the AD technique(Advantage-reweighting for Difficulty) in arxiv:2504.09696.
3. Multi-Aspect Question Reformulation (MQR) strategy to use a prompt asking OpenAI O3 to rewrite the prompt problem in three different ways that maintain the groundtruth answer.

The authors do experiments and ablation studies to show that the models trained with the three techniques on the MATH dataset are better in performance on the benchmarks. However, from my point of view, the training and evaluation setups are questionable, and do not match the mainstream practice. As a result, the conclusions drawn by this paper are not completely reliable.

### Strengths
1. The idea of DQW is intuitive and reasonable. Compared to the advantage-reweighting for difficulty technique used in arxiv:2504.09696, DQW is simpler and has fewer hyperparameters for tuning. 

2. The evaluation is comprehensive in the aspect that it is done on various benchmarks, including AMC, AIME, MATH500 and the Olympiad.

### Weaknesses
1. The three techniques introduced in the paper are all not completely novel. The biased estimation issue is well-known in the community and has been handled in the Dr.GRPO work. The difficulty-aware policy optimization is also a topic already touched in arxiv:2504.09696. The Multi-Aspect Question Reformulation (MQR) strategy belongs to question augmentation methods via LLMs that have been investigated in depth since arxiv:2210.11610. 

2. The theoretical analysis of DGAE is not convincing to me. In subsection 3.1.1, the authors say "... $|\hat{A}_{GR,i}|$ determines the corresponding update magnitude. Therefore, the total update magnitude for a single question q can be defined as the sum of these individual magnitudes across all G responses". However, it is obvious that the total magnitude is not equal to the sum of the individual magnitudes. As a result, using the sum of the advantage estimation magnitude as the measure for the estimation bias is not convincing. In contrast, Dr.GRPO solves the biased estimation problem by simply removing the standard variation normalization.

3. The presentation is not clear enough. For example, in the introduction part, it is said that "MathForge comprises two key components: a Difficulty-Aware Group Policy Optimization (DGPO) algorithm and a Multi-Aspect Question Reformulation (MQR) strategy". However, in Table 2, it is indicated that Mathforge is a framework more than DGPO+MQR, and section 3 provides no further details on what Mathforge is.  Also, DGPO differs from GRPO not only in DGAE and DQW, but also in integrating part of the techniques of DAPO, which confuses the readers about the contributions of DGPO. 

4. The experimental setup is not appropriate, which leads to some conclusions contradictory to previous work, such as the performance degradation of Dr.GRPO and GSPO. The base model series, Qwen2.5-Math models, are known to be continual-pretrained on Qwen Math Corpus v1&v2 with a context length of 4096. However, this paper performs training and evaluation with a 1024 maximum completion length, which (1) makes the evaluation performance significantly lower than that reported in the Qwen2.5-Math technical report (e.g. 35.0% on MATH500 in this paper vs. 49.8% on MATH in Qwen's report for Qwen2.5-Math-1.5B) (2) forces the model to reason with fewer tokens as shown in Figure 1(b), which is contrary to the common sense that longer thinking length indicates better reasoning ability.

### Questions
My questions have been listed in the weaknesses. I appreciate if the authors can clarify the confusions and provide more experimental details mentioned in weakness 4.

### Soundness
1

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
This paper proposes MathForge, a framework that enhances mathematical reasoning in large language models by focusing on harder questions from both algorithmic and data perspectives. The authors identify a critical limitation in Group Relative Policy Optimization (GRPO): its advantage estimation introduces an implicit bias where policy updates are suppressed for both easier and harder questions, peaking only at moderate difficulty. To address this, they introduce Difficulty-Aware Group Policy Optimization (DGPO), which employs difficulty-balanced group advantage estimation (DGAE) using L1 normalization and difficulty-aware question-level weighting (DQW) to prioritize challenging questions. Complementing this algorithmic improvement, they propose Multi-Aspect Question Reformulation (MQR), which generates more difficult question variants by adding story backgrounds, introducing abstract terminology, and nesting sub-problems—all while preserving the original answer. Experiments across multiple models (1.5B-7B parameters) and six mathematical reasoning benchmarks demonstrate consistent improvements, with the full MathForge framework achieving a 4.56% average improvement over the strong GRPO baseline on Qwen2.5-Math-7B.

### Strengths
The paper makes several notable contributions. First, it provides rigorous mathematical analysis with formal proofs (Theorems 1 and 2) demonstrating that GRPO's advantage estimation creates an inherent bias favoring moderate-difficulty questions over harder ones, which is a valuable theoretical insight into a widely-used algorithm. Second, the proposed solutions are well-designed and complementary: DGPO corrects the algorithmic bias while MQR enriches training data with harder questions, creating a synergistic framework. Third, the experimental validation is comprehensive, spanning multiple model sizes, architectures, both text-only and multimodal domains, and six benchmarks, with consistent improvements demonstrating strong generalizability. Fourth, the MQR strategy cleverly preserves answer equivalence (avoiding costly solution regeneration) while targeting distinct reasoning skills through three reformulation aspects. Finally, the demonstration that DGPO is compatible with other policy optimization methods (GPG, DAPO, GSPO) and improves them further suggests it addresses fundamental aspects of the learning process rather than being narrowly applicable.

### Weaknesses
Despite its strengths, the paper has several limitations that warrant consideration. The central premise that prioritizing harder questions is universally beneficial lacks deeper justification—while GRPO under-weights them, the paper doesn't definitively establish when or why this is detrimental, or whether all types of difficulty contribute equally to learning. The MQR approach's reliance on OpenAI o3, an exceptionally powerful and potentially expensive model, raises concerns about reproducibility, scalability, and accessibility for the broader research community. Although MQR achieves 97-99% equivalence rates, the 1-3% of potentially incorrect reformulations could introduce noise, yet the impact of this corruption on training is not analyzed. Some improvements, particularly DGPO alone (+2.18%), are modest relative to the added computational complexity and implementation overhead. The temperature hyperparameter T=2.0 for DQW appears somewhat arbitrary despite ablations, and guidance on setting this across different scenarios would strengthen practical applicability. Finally, the paper doesn't discuss potential failure modes or scenarios where prioritizing harder questions might harm performance, such as when models still struggle with fundamental concepts.

### Questions
Difficulty distribution sensitivity: How does DGPO's effectiveness depend on the difficulty distribution of the training data? If the dataset already skews heavily toward hard or easy questions, would DGPO still provide similar benefits, or is there an optimal difficulty distribution where this method works best? Have you analyzed the interaction between dataset composition and DGPO's prioritization mechanism?

### Soundness
3

### Presentation
3

### Contribution
3
