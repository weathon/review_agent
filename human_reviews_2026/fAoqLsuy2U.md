# Efficient Bayesian Inference from Noisy Pairwise Comparisons

- Decision: Reject
- Scores: 0, 4, 4, 6

## Abstract
Evaluating generative models is challenging because standard metrics often fail to reflect human preferences.  
Human evaluations are more reliable but costly and noisy, as participants vary in expertise, attention, and diligence.  
Pairwise comparisons improve consistency, yet aggregating them into overall quality scores requires careful modeling.  
Bradley-Terry-based methods update item scores from comparisons, but existing approaches either ignore rater variability or lack convergence guarantees, limiting robustness and interpretability.  
We introduce BBQ, a Bayesian Bradley-Terry variant that explicitly models rater quality, downweighting or removing unreliable participants, and provides guaranteed monotonic likelihood convergence through an Expectation-Maximization algorithm.  
Empirical results show that BBQ achieves faster convergence, well-calibrated uncertainty estimates, and more robust, interpretable rankings compared to baseline Bradley-Terry models, even with noisy or crowdsourced raters.
This framework enables more reliable and cost-effective human evaluation of generative models.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
This paper extends the Bayesian Bradley-Terry model to jointly estimate item quality and rater reliability, called BBQ. Its main contribution is deriving closed-form EM updates, which give the model two key advantages:
(1) The EM updates guarantee monotonic likelihood increase and fast convergence;
(2) The model jointly estimates item skill and rater reliability, reducing the impact of unreliable raters.
Extensive experiments show that BBQ outperforms Bayes-BT and Crowd-BT in ranking accuracy, stability, and computational efficiency.

### Strengths
The problem is important: the Bradley-Terry model is widely used, and modeling rater reliability is often necessary.

BBQ shows clear advantages over other variants: (1) guaranteed and fast convergence; (2) higher accuracy.

Experiments are thorough, with proper baselines and comprehensive evaluation on both accuracy and efficiency.

Writing quality is acceptable.

### Weaknesses
# The authors violate the double-blind review policy in the supplementary materials, where the license file reveals their identities. I recommend desk rejection for this reason.

Aside from that, I have only minor comments:
It would be helpful to show BBQ’s performance on downstream tasks as a complement to simulations. For example, in automated peer review which is an area of growing interest in the ML community, LLM-based tools can produce pairwise quality comparisons between papers. Comparing BBQ and the original BT model in such a setting would be insightful.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new metric for generative models based on human evaluations with rater quality.

### Strengths
1. Evaluations span diverse real and synthetic datasets (e.g., HUMAINE, MT-Bench, HiFiC, WD, CLIC/IHQ).
2. Unlike prior approaches, it introduces closed-form EM updates, ensuring monotonic likelihood convergence — a notable theoretical contribution.

### Weaknesses
1. No example evaluation results (the evaluation score for image generation and LLM)
2. Following Bradley & Terry (1952), it shows the method is suitable for larg sample test. However, the dataset testing in the experiment, only HUMAINE is the large sample. Then the other results may not be reliable.
3. The application is limited. The main benefits of BBQ appear only when rater quality is heterogeneous. In curated or expert datasets (e.g., MT-Bench, WD, IHQ-screened), the paper shows that simpler models (like standard BT or Bayes-BT) achieve equivalent performance.
Also, the evaluation only handles pairwise comparison data.

### Questions
1. In sec.3.1, it says 'two items i and j of this set'. It should introduce what it is. In the paper, it has introduced many different input, the skill, quality score or the input images/texts. 
2. In table 1, What does scr. and unscr. mean?
3. Limited Validation Beyond Benchmark Correlations. The evaluation relies mostly on ranking correlation (Kendall’s τ) and Top-1 agreement, without deeper qualitative analysis or ablation on uncertainty calibration. While the model’s uncertainty estimates are discussed, there’s no clear downstream validation of whether these uncertainties improve decision-making or leaderboard robustness in real settings.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents BBQ, a Bayesian Bradley-Terry model variant that incorporates rater quality into the aggregation of noisy human-generated pairwise comparison data. By introducing rater-specific reliability parameters and employing a conjugate prior Bayesian formulation, the methodology leverages an EM algorithm with closed-form updates, ensuring monotonic convergence of the likelihood. The method addresses instability and interpretability issues in crowdsourced or heterogeneous rater settings. Empirical studies across multiple evaluation datasets for generative models, including image compression and language modeling, demonstrate that BBQ yields faster convergence, greater robustness to unreliable raters, and well-calibrated uncertainty estimates compared to standard BT and other recent extensions.

### Strengths
1. The paper develops a Bayesian Bradley-Terry model that explicitly incorporates rater reliability, featuring a mathematically grounded EM algorithm with full derivations. This formulation ensures stable, monotonic convergence, in contrast to gradient-based approaches like Crowd-BT.

2. The approach provides Bayesian credible intervals for estimating item skills, enhancing interpretability and yielding more dependable statistical inferences. Also, despite its Bayesian formulation, BBQ is demonstrated to be computationally efficient.

### Weaknesses
1. BBQ is strictly designed for pairwise comparison data, whereas much of the recent literature on preference learning and robust aggregation seeks to handle more general forms like full rankings or ratings. This restricts its applicability for community studies increasingly collecting multi-way ore scalar judgments.

2. The current formulation of BBQ assumes that unreliable raters behave as random guessers, ignoring more realistic cases where low-quality participants exhibit systematic biases or coordinated behavior. Furthermore, the experiments evaluate only random noise scenarios, without testing adversarial or colluding raters that often arise in real-world crowdsourcing or online evaluation platforms. As a result, both the modeling assumption and the experimental design fall short of demonstrating robustness under structured or adversarial noise conditions.

3. While BBQ improves preference aggregation quality, it remains unclear whether this leads to measurable gains in downstream policy learning. Evaluating how BBQ-derived preferences affect model performance in DPO or RLHF setups would clarify its practical impacts.

### Questions
See weaknesses.

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
2

### Summary
The paper introduces BBQ, a Bayesian Bradley-Terry variant that models item skill and rater quality jointly and optimizes them with an EM algorithm that guarantees monotonic likelihood improvement. The model down-weights unreliable raters and yields calibrated uncertainty for rankings. Experiments on LLM and image-compression preference datasets show faster convergence and stronger robustness than Bayes-BT and Crowd-BT, especially when rater quality is heterogeneous.

### Strengths
- Clear modeling of rater noise with principled estimation
- Empirical robustness across diverse datasets
- Particularly strong when raters are mixed-quality
- Well-calibrated uncertainty
- Practical efficiency
- Clarity and completeness

### Weaknesses
- Scope limited to pairwise comparisons
- Diminishing returns with uniformly high-quality raters
- Assumptions and design choices
- Scalability caveat

### Questions
Your framework is currently restricted to pairwise comparisons. What is your concrete plan (modeling and inference) to support K-way choices and absolute ratings, and what identifiability or prior choices would change? Do you have any preliminary results (e.g., 3AFC, 5-point ratings)?

### Soundness
4

### Presentation
3

### Contribution
3
