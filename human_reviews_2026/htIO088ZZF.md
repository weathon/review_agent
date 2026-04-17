# SURE: Shift-aware, User-adaptive, Risk-controlled Recommendations

- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
Although Sequential Recommender Systems (SRS) have been well developed to capture temporal dynamics in user behavior, they face a critical gap in formal performance guarantees under preference shifts. When preferences change, predictions often become unreliable, undermining user trust and threatening long-term platform success. To address this challenge, we introduce **SURE** (**S**hift-aware, **U**ser-adaptive, **R**isk-controlled R**E**commendations), a dataset- and model-agnostic framework that provides adaptive recommendation sets with formal coverage guarantees while remaining compact under preference shifts. Specifically, SURE (i) ensures validity through a loss-based change-point mechanism that adaptively updates calibration thresholds upon detecting preference shift, (ii) maintains compact recommendation sets by stabilizing predictions with a Hedge-weighted ensemble of bootstrapped experts, preventing validity from degenerating into impractically large outputs, and (iii) guarantees robustness under non-stationarity by deriving finite-sample bounds that ensure the ensemble’s expected set size remains close to the best expert while controlling the utility-based risk in recommendation. Extensive experiments across multiple datasets and base models validate the effectiveness of the proposed framework, which aligns with our theoretical analysis.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This article proposes the SURE framework to address the problem of unreliable predictions caused by user preference shifts in sequential recommendation systems. A statistically guaranteed adaptive recommendation set generation method is proposed. It combines:
1) Loss-based change-point detection for preference-shift identification.
2) Hedge-weighted ensemble of bootstrapped models.
3) Dynamic conformal prediction to generate recommendation sets with a formal coverage guarantee coverage.
Theoretically, SURE bounds prediction set size and controls risk under preference shifts. Experiments across 5 datasets (e.g., MovieLens, Taobao) and 4 base models (e.g., SASRec, FMLP-Rec) show SURE outperforms preference-aware baselines (e.g., TiSASRec) and conformal methods (e.g., EnbPI) in recommendation metrics (Recall, NDCG) while maintaining coverage with compact sets.

### Strengths
1. Clear Motivation: This paper points out the problem of the lack of statistical guarantees in existing SRS in preference drift scenarios, which has not been systematically addressed from a statistical learning perspective in the field of recommendation systems. 
2. Model-Agnostic Method: Works with any SRS backbone (e.g., transformers, CNNs) and adds minimal overhead (+1.5 min training time).
3. Strong Reproducibility: Code/data released with detailed documentation.
4. Theoretical Contributions: Finite-sample guarantees for ensemble set size and risk control under distribution shifts.

### Weaknesses
1. Ceiling@25 score lacks justification: The use of "the maximum achievable value under its own ranking when limited to 25 items" as an upper bound is not well-motivated, and the expression is not clear enough. True Recall/NDCG should approach 1 if all relevant items are captured.
2. Missing Baselines: Omits key *uncertainty-aware SRS*: variational autoencoders (ContrastVAE WWW [1], HNVM WWW [2]) and hierarchical attention (HVAM DASFAA [3]). These methods directly model uncertainty in sequences—critical for novelty claims. 
3. Opaque Method Design: The intuition behind why the proposed change point detection module (Eqs 10–13) detects preference shifts better than existing shift detection methods is unclear. Intuitively, simply detecting preference shifts based on the change of training loss may yield satisfactory results [4]. Moreover, the difference with existing conformal prediction methods [5,6] has not been deeply explained. 

References:
[1] Zhe Xie, Chengxuan Liu, Yichi Zhang, Hongtao Lu, Dong Wang, Yue Ding: Adversarial and Contrastive Variational Autoencoder for Sequential Recommendation. WWW 2021: 449-459
[2] Teng Xiao, Shangsong Liang, Zaiqiao Meng: Hierarchical Neural Variational Model for Personalized Sequential Recommendation. WWW 2019: 3377-3383
[3] Jing Zhao, Pengpeng Zhao, Yanchi Liu, Victor S. Sheng, Zhixu Li, Lei Zhao: Hierarchical Variational Attention for Sequential Recommendation. DASFAA (3) 2020: 523-539
[4] Wenjie Wang, Fuli Feng, Xiangnan He, Liqiang Nie, Tat-Seng Chua: Denoising Implicit Feedback for Recommendation. WSDM 2021: 373-381
[5] Margaux Zaffran, Olivier Féron, Yannig Goude, Julie Josse, Aymeric Dieuleveut: Adaptive Conformal Predictions for Time Series. ICML 2022: 25834-25866
[6] Isaac Gibbs, Emmanuel J. Candès: Adaptive Conformal Inference Under Distribution Shift. NeurIPS 2021: 1660-1672

### Questions
Please refer to the weaknesses.

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
This paper formulates the sequential recommender systems problem as an SURE framework and then proposes the DAUO algorithm, with a Hedge component, to address it. Theoretical and empirical results were also presented.

### Strengths
1. The model formulation of SRS is new and of practical interest. 
2. A theoretical guarantee is also presented for the DAUO algorithm.

### Weaknesses
1. The algorithmic novelty is not strong. Figure 1 basically shows a standard application of Hedge, and the detailed DAUO algorithm is deferred to the appendix, which, from the reviewer’s perspective, reads like a combination of prior works’ methods with Hedge. 
2. Although the DAUO algorithm has a better performance than other baselines, the improvements are marginal, say in Table 2, it only improves around 4%, and the deviations are not reported. So, the paper needs more experiments to support the advantage of DAUO.
3. The notations of this paper are hard to follow. There are many other confusions on the notations.
    - For example, in Eq.(6), the capital $L$ appears, but with no physical meaning explanation. Is it the number of experts? What’s the relation between experts and ensemble models? What is an ensemble model? 
    - Later in Eq(10), $L_t$ appears. What’s the relation between $L$ , $L_t$ and $\mathcal{L}$ in Eq.(5).  
    - Another example, in Eq.(15), sometimes the $\ell$ appears in subscript, in superscript, and the index $t$ in the reverse way, what’s the reason for swapping the sub/super scripts? 
    - Also, Eq.(16) and Eq.(17) use $\mathbf{E}$ and $\mathbb{E}$ to represent exceptions. Is there a special reason to differentiate the notations? 
    - Line 198, it should be $L-1$ dimension simplex.

### Questions
Listed in Weaknesses.

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
4

### Summary
This paper introduces SURE, a model-agnostic framework for sequential recommender systems that addresses the challenge of user preference shifts. SURE provides formal performance guarantees by using a loss-based change-point mechanism to adaptively update recommendations when it detects a shift in user preferences. The framework utilizes a Hedge-weighted ensemble of bootstrapped models to maintain compact and robust recommendation sets.

### Strengths
1. The paper addresses an interesting and real-world problem in recommender systems, i.e., adapting to dynamic shifts in user preferences.
2. The proposed SURE reasonably integrates established techniques from several fields.
3. The authors provide rigorous theoretical analysis to back their claims.

### Weaknesses
1. The paper’s central motivation, i.e., adapting to preference shifts, is not empirically validated on the selected datasets. It provides no analysis to show that such shifts are prevalent or significant, making it unclear if the proposed complex solution is addressing a demonstrated problem or merely a hypothetical one.
2. The framework's technical contribution is more an integration of existing methods than a fundamental innovation. It assembles well-established techniques: adaptive conformal prediction, Hedge ensemble algorithm, and standard change-point detection concepts. 
3. The evaluation is not conducted against current SoTA sequential RS. The absence of stronger, more recent baselines (e.g., TIGER-based RS or HSTU-based RS) makes it difficult to assess the true value of SURE. The reported improvements may not be significant when applied to a more powerful SoTA model. Several sequential RS can be considered like TIGER [1], OneRec [2], HSTU [3], MTGR [4].

Ref: 

[1] Rajput, Shashank, et al. "Recommender systems with generative retrieval." Advances in Neural Information Processing Systems 36 (2023): 10299-10315.

[2] Deng, Jiaxin, et al. "Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment." arXiv preprint arXiv:2502.18965 (2025).

[3] Zhai, Jiaqi, et al. "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations." International Conference on Machine Learning. PMLR, 2024.

[4] Han, Ruidong, et al. "MTGR: Industrial-Scale Generative Recommendation Framework in Meituan." arXiv preprint arXiv:2505.18654 (2025).

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the challenge of achieving accurate preference predictions in the context of dynamic changes in user preferences. The proposed method promptly detects preference shifts and adjusts decision criteria by real-time monitoring of user behavior through applying change point detection to the loss function. Meanwhile, it aggregates predictions from multiple models via weighted recommendations to enhance prediction performance while ensuring the conciseness of the recommendation set. To achieve stable and reliable recommendations under evolving user preferences, the approach establishes a performance baseline: on one hand, it controls the risk of irrelevant recommendations; on the other hand, it ensures that the performance of the final ensemble model approximates that of the best expert model. The advantages of this method include: model-agnostic applicability across various scenarios and base models, provision of theoretically grounded recommendation sets with guaranteed user satisfaction, and maintenance of set compactness to avoid information overload.

### Strengths
1. Rigorous theoretical proofs and a comprehensive experimental design (encompassing multiple datasets, models, and baselines). Time efficiency analysis validates its practicality, and the results are highly reproducible (with an anonymous code repository provided).
2. A scalar loss-based shift metric is designed, integrating change point detection with dynamic threshold calibration to overcome the limitations of traditional conformal methods. The Hedge weighting integration strategy naturally optimizes set compactness while ensuring coverage.
3. Fills the research gap of lacking strict performance guarantees in sequential recommendation under non-stationary preferences. The proposed model-agnostic property enables easy deployment in practical systems, offering strong generalization value.
4. A dedicated loss function for change point detection is proposed, facilitating the mitigation of preference drift issues.

### Weaknesses
1. The conformal prediction baselines fail to include post-2024 adaptive methods, which may undermine the comprehensiveness of the comparative analysis.
2. The time efficiency analysis only reports total training time without separating the time consumption of the change point detection and threshold update modules, making it difficult to evaluate real-time performance in online scenarios.
3. The manuscript lacks graphical illustrations and overrelies on formula derivations, resulting in poor readability for certain audiences.
4. The work lacks a high degree of innovation: its implementation principle is analogous to Ensemble Learning, offering limited contributions to overall novelty and leading to relatively insufficient academic contributions.
5. The summary of limitations in existing work (Lines 041~047 of the Introduction) is neither detailed nor in-depth enough.
6. Starting from the fourth paragraph of the Introduction, conformal prediction or adaptive conformal approaches appear to be the core idea of this paper (though, as noted later, they cannot be directly applied to sequential recommendation systems). However, the paper fails to clarify which specific problems of existing work these approaches address and why they can solve them. This information is crucial for understanding the paper’s core contributions and rationale. Consequently, while the authors propose a relatively complex and effective model, the reasoning behind its effectiveness remains unclear.

### Questions
1. Could you supplement the results with visualizations illustrating the dynamic process of preference drift (e.g., changes in thresholds, set sizes, and risk values at different timestamps) to more intuitively demonstrate the framework’s adaptive capability?
2. In scenarios with extreme data sparsity (e.g., cold-start users) where the predictive stability of base models is poor, can SURE’s ensemble strategy still function effectively?
3. In the experiments, the maximum recommended set size was fixed at 25. If this constraint were removed, could SURE’s set size distribution still maintain compactness?
4. If multiple ensemble models collectively exhibit selection bias, can the method still provide accurate preference-based recommendations?
5. Could you incorporate 1–2 relevant works published in 2025 into the experiments to enhance the persuasiveness of the experimental results?

### Soundness
2

### Presentation
2

### Contribution
2
