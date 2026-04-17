# Multi-Preference Optimization: Generalizing DPO via Set-Level Contrasts

- Decision: Reject
- Scores: 6, 0, 4, 6

## Abstract
Modern post-training pipelines for LLMs frequently involve on-policy generation to produce multiple candidate responses per prompt. However, popular alignment methods like Direct Preference Optimization (DPO) are restricted to pairwise comparisons, discarding valuable supervisory signal. In this setting, we propose Multi-Preference Optimization (MPO), a generalization of DPO that optimizes over entire sets of selected and rejected responses. This set-level contrastive approach is theoretically grounded: we first prove that leveraging $n$ responses achieves a $\mathcal{O}\bigl(\tfrac{1}{\sqrt{n}}\bigr)$ convergence in TV-distance to the true preference distribution. We then prove, under a formal model with spacing-scaled Gaussian noise ($\Delta, \sigma = \mathcal{O}(1/n)$), that MPO's 2-bin partition reliability remains bounded away from zero, in contrast to full-ranking methods which degrade exponentially ($\exp(-\mathcal{O}(n))$). To further enhance learning, MPO employs a deviation-based weighting, which emphasizes outlier responses to induce an implicit curriculum. Empirically, as we show over multiple models and benchmarks,  MPO achieves state-of-the-art performance, with an improvement of up to $\sim 17.5$\% WR on AlpacaEval2 in the on-policy iterative setting, and state-of-the-art results in off-policy settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a set-level, contrastive preference optimization framework that generalizes DPO with groupwise. 
Empirically, MPO and W-MPO deliver state-of-the-art results on AlpacaEval2 (WR, LC-WR), Arena-Hard, and MT-Bench across both off-policy and on-policy training regimes, with performance in some cases approaching that of GPT-4o on AlpacaEval2.

### Strengths
MPO retains the simplicity and elegance of DPO while naturally extending it to handle multiple responses per prompt, without the extra cost of full ranking or reward calibration. 
It achieves consistent state-of-the-art performance across model sizes and training regimes, scaling effectively with more responses per query and remaining competitive even under limited data or compute budgets.

### Weaknesses
Like DPO, MPO’s objective relies on the log-ratio between the policy and the reference model. It would be useful to analyze how sensitive performance is to the choice or vintage of the reference model (e.g., Llama vs. Qwen families).

W-MPO weights samples by their absolute deviation from the mean. Why use absolute deviation, and why the mean specifically? Including ablations with alternative robust statistics (e.g., median, trimmed mean, quantiles) could clarify stability under skewed or noisy reward distributions.

In on-policy settings, top-k and bottom-k responses depend on a particular reward model. How robust is MPO/W-MPO to miscalibration or domain shift in that scorer? Cross-reward or human-labeled validation would strengthen the reliability of the findings.

The main paper appears to be 10 pages long, while the conference strictly limits submissions to 9 pages. Please ensure the paper adheres to the page limit requirements.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The main text of this paper is 10 pages, which exceeds the page limit for ICLR submissions. According to ICLR policy, this paper should be desk-rejected.

### Strengths
N/A

### Weaknesses
N/A

### Questions
N/A

### Soundness
2

### Presentation
1

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
Traditional DPO methods only allow a pair of preference data to be trained. In this paper, they propose MultiPreferenceOptimization(MPO), a generalization of DPO that optimizes over entire sets of selected and rejected responses. Within the paper, authtors provide some theoretical evident on why MPO works better than other DPO-style methods. Experiments are conducted on several open-ended benchmarks and the results show that MPO achieves better performance than other methods.

### Strengths
1. This paper provides theoretical evidence on why MPO works better than other DPO-style methods.
2. The experiments are conducted and MPO are compared with several strong baselines.

### Weaknesses
1. What is the difference between the "Off-policySetting" and "On-policySetting"? It seems that they only differ in the initial model (off-policy uses a weaker sft model, while the on-policy uses a stronger open-sourced instruct model). If so, why they get this name? Based on my understanding, Off-policy and On-policy should be different in how they are trained (sample from base model or from current policy model).
2. In the experiments, it seems some of strong baseline models are missing -- SimDPO [1]、BMC[2] etc. The authors should enrich their baseline comparsion.
3. Except for performance improvement, authors should also conduct some analysis on the strength of their approaches (e.g., case study). What is the additional advantages of MPO? (the authors provide results on data efficiency)

### Questions
see weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper tackles the important challenge of aligning LLMs in the post-training and introduces  Multi-Preference Optimization (MPO), a generalization of Direct Preference Optimization (DPO) that extends beyond pairwise comparisons. The approach optimizes over the entire sets of selected and rejected responses, potentially capturing valuable supervisory signal. The method is supported by a theoretical analysis proving that using n responses leads to a faster convergence rate (O(1/√n)) and that MPO's 2-bin partitioning is more robust to reward model noise than full-ranking methods like Plackett-Luce. The paper also introduces Weighted MPO (W-MPO), which uses reward score deviations to create an implicit curriculum by up-weighting informative outlier responses. The approach shows strong empirical results with an improvement of up to 17.5% win rate on AlpacaEval2 in the on-policy iterative setting, and state-of-the-art results in off-policy settings and mostly equivalent or better results on Arena-Hard and MT-Bench.

### Strengths
- The paper addresses an important challenge in the alignment process, and the presented method is a clean and intuitive generalization of DPO, moving from pairwise to set-wise comparisons.
- The method is shown to achieve state-of-the-art results across a variety of models, benchmarks, and training paradigms (off-policy, on-policy, iterative), demonstrating its robustness and effectiveness, especially on AlpacaEval2.
- The paper provides both theoretical motivation (Theorems 1 & 2) and strong empirical validation for its core claims, particularly the benefits of using more responses and the robustness of 2-bin partitioning over full-ranking.
- Strong selection of benchmarks like AlpacaEval 2.0, Arena-Hard, and MT-Bench indicates generalizability of the approach.

### Weaknesses
- The on-policy results depend on a single reward model. It is possible that MPO is particularly good at optimizing for the specific reward distribution of the Skywork RM, and its gains might be less pronounced with other RMs.
-  The theoretical result on noise robustness (Theorem 2) relies on a specific "spacing-scaled" noise model. It is unclear how realistic the assumption is.
- Relying on RM leaves actual alignment to be questionable; the study would benefit from human evaluation and more qualitative case studies.

### Questions
- Could the authors elaborate on potential failure modes for MPO? For instance, how would the mean-based partitioning perform if the reward distribution for a prompt is strongly bimodal?
- Regarding the On-Policy Data: In your on-policy experiments, you discard the median-reward response. What was the rationale for this decision?
- You mentioned skipped sets. How common are these in your experiments?

### Soundness
3

### Presentation
3

### Contribution
3
