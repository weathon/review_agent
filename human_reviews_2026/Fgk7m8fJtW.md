# CARE-RFT: Confidence-Anchored Reinforcement Finetuning for Reliable Reasoning in Large Language Models

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Reinforcement finetuning (RFT) has emerged as a powerful paradigm for unlocking reasoning capabilities in large language models. However, we identify a critical trade-off: while unconstrained RFT achieves strong reasoning performance, it severely compromises model trustworthiness by amplifying hallucination and worsening calibration; conversely, RKL-constrained RFT preserves trustworthiness but limits reasoning gains due to its unbounded penalty on exploratory deviations. To resolve this tension, we introduce CARE-RFT (Confidence-Anchored Regularized Reinforcement Finetuning), a novel method that replaces standard reverse KL regularization with a skew reverse KL divergence. CARE-RFT provides a confidence-sensitive penalty—bounded for confident, consistently-rewarded explorations to enable reasoning, while unbounded elsewhere to preserve calibration. Extensive experiments across multiple model scales and RFT algorithms show that CARE-RFT achieves a superior balance, matching the reasoning performance of unconstrained RFT while recovering the trustworthiness and calibration of the base model. Our work establishes that careful, confidence-aware regularization is key to building both capable and trustworthy reasoning models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces CARE-RFT, a method that replaces the standard reverse-KL regularization used in RLwith a skew reverse-KL divergence. The goal is to achieve a better trade-off between reasoning performance and trustworthiness in large language models. Experiments on Qwen2.5-3B/7B models show that CARE-RFT matches unconstrained RL on reasoning benchmarks (e.g., MATH, GSM8K) while recovering the calibration and factual reliability of the base model (e.g., TruthfulQA).

### Strengths
- Empirical results are promising, showing consistent improvements across reasoning and calibration metrics.

- The proposed SRKL regularization is simple and easy to integrate into existing RFT frameworks like GRPO or DAPO.

### Weaknesses
- The paper attributes miscalibration in RL-trained models primarily to sparse rewards and credit-assignment issues. I would argue this is incomplete. RL objectives themselves are not proper scoring rules. Even with dense rewards, an unregularized RL objective will not produce calibrated probabilities, since it optimizes expected reward rather than likelihood alignment. KL regularization helps mainly by anchoring the policy to a calibrated base model, not by densifying the reward.

- There are some inconsistencies and missing clarifications in Section 2. In Equation (2), $GC_{div}$ appears as the coefficient of $\nabla_\theta \log\pi_\theta$, while in Equation (4), $GC_{div}$ of RKL is multiplied by $\nabla_\theta \pi_\theta$. This inconsistency likely comes from the fact that there are two ways to implement RKL regularization in RL: as part of the reward, or as an additional objective term. It would help if the authors clarified which version is used here, and how it relates to the common Schulman approximation [1,2], which is the de facto implementation of RKL-based constraints.

- The idea of using SRKL instead of RKL seems promising, but I feel like the paper lacks depth that would convince the reader this is the right direction to go:

  - The solution to KL-regularized RL is known and well studied [3,4]. How does the optimal policy look when we replace the KL with SRKL regularization? What are the tradeoffs?

  - The paper presents SRKL as a way to address the credit assignment problem. However, there have been multiple papers recently [5,6 and others] that try to do the same. Is CARE-RFT better than these methods, or complementary to them?

[1] http://joschu.net/blog/kl-approx.html

[2] Amini, Afra, Tim Vieira, and Ryan Cotterell. "Better Estimation of the KL Divergence Between Language Models." arXiv preprint arXiv:2504.10637 (2025). 

[3] Korbak, Tomasz, Ethan Perez, and Christopher L. Buckley. "RL with KL penalties is better viewed as Bayesian inference." arXiv preprint arXiv:2205.11275 (2022). 

[4] Vieillard, Nino, et al. "Leverage the average: an analysis of kl regularization in reinforcement learning." Advances in Neural Information Processing Systems 33 (2020): 12163-12174. 

[5] Wang, Shenzhi, et al. "Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning." arXiv preprint arXiv:2506.01939 (2025). 

[6] Qu, Yuxiao, et al. "Optimizing test-time compute via meta reinforcement fine-tuning." arXiv preprint arXiv:2503.07572 (2025).

### Questions
- In your definition of ECE, why do you consider only the majority-voted answer? Seems like considering all answers in the N responses will lead to a better estimator of ECE. 

- Please include confidence intervals in your results. Especially for datasets like truthfulQA that are on the smaller side (<1000 questions) it is important in order to understand how significant an increase or drop in metrics are.

### Soundness
2

### Presentation
3

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
The paper introduces CARE-RFT, a reinforcement fine-tuning method that preserves calibration. The paper starts by identifying that unconstrained RL is great for exploration but leads to hallucinations and loss of calibration, while KL-regularized RL limits exploration but is much better calibrated. The authors present a method to get the best of both worlds by replacing standard reverse KL regularization with a skew reverse KL divergence. CARE-RFT applies confidence-sensitive penalties that encourage reliable (calibrated) yet exploratory learning. Experiments show that it is competitive in accuracy to unconstrained RFT while preserving the calibration and low hallucinations of the base model.

### Strengths
- **Potentially impactful method**: KL-based RL training can prevent exploration, while unconstrained training can lead to loss of calibration and hallucinations. The proposed method strikes a balance between the two and can be generally useful. 
- **Intuitive writing**: The paper motivates the problem well, and the method is presented in an intuitive way. The paper would be even stronger if intuition is backed with solid theory on why the divergence chosen by the authors is the correct choice.

### Weaknesses
- **Limited results**: RL training is performed only on a single dataset (MATH), which introduces doubts about generality. This is further compounded by the fact that the MATH training dataset is small (7000 examples), and actual practice is to use much larger datasets for math training (>20K questions). Very limited ablations and analysis are presented. The authors analyze the entropy curves and find that entropy collapses in unconstrained GRPO, but do not try a method with an entropy loss. 
- **Design decisions**: It is unclear why the frequency of an answer is used as the confidence for the answer. More justification and experiments with alternate choices (verbalized confidence, log-prob based confidence) need to be provided (see questions section). Additionally, while the authors tuned the $\alpha$ parameter for their proposed divergence, they did not seem to tune the default $\beta$ parameter for the default reverse KL objective. It is possible that a well-tuned reverse KL setting matches their method.

### Questions
- Please move at least some part of the related work into the main paper, it should not be deferred to appendix. 
- What is the bolding scheme of Table 2? DAPO (no constraint) Math has higher accuracy but is not bolded. Baselines with higher accuracy are not bolded throughout the table. 
- For the entropy collapse of GRPO (no constraint), did authors try a variant with entropy regularization, which has been considered by some prior works [1] ?
- Why is skewed KL the correct divergence choice? There are so many divergences to choose from (some with bounding), what theoretical insights make this the correct choice? Is there any intuition for doing so? Comparing to other divergences (JSD for example) is important to understand the actual benefits from the chosen divergence.
- If $\alpha$ parameter of skew-KL was tuned, then why wasn’t $\beta$ for the simple reverse KL divergence tuned? Increasing/decreasing this parameter is a way to directly control tradeoffs between accuracy and calibration as well. Just like the ablation study on  $\alpha$ (Sec 5.4), it would be nice to see an ablation study on $\beta$. 
- I think a better way to visualize the accuracy-calibration tradeoff is a graph with accuracy (higher better) on y axis and calibration (higher better) on x axis. Then points from different approaches can be plotted on it. Figure 1 can be improved as it is currently difficult to extract insights from it. 
- Why is the frequency of the answer the correct way to determine confidence? It does not seem to be the optimal proxy. The optimal answer to output in RLVR is the one which the model is most confident in (highest expected). If the model thinks answer A has 70% chance of being correct and answer B has 30%, this does not mean that it should output answer B 30%of the time. In fact, it should always output answer A to maximize the binary correctness reward. I believe there are other calibration proxies which should also be considered -  such as using log-prob of answer to get confidence for tasks like MMLU, or asking for a verbalized confidence from the models? 
- Why is pass@4 used for the reasoning tasks, the choice seems arbitrary and no justification is provided anywhere. 
- Please provide details on the TruthfulQA and Self-Aware datasets used, the exact task and how it is evaluated. It is okay for these to be in the appendix. 

[1]: He, J., Liu, J., Liu, C. Y., Yan, R., Wang, C., Cheng, P., ... & Zhou, Y. (2025). Skywork open reasoner 1 technical report. arXiv preprint arXiv:2505.22312.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper suggests Confidence-Anchored Regularized Refinement Fine-Tuning (CARE-RFT),  a regularization technique proposed to solve the overconfidence and hallucination problems that arise in existing RFT-series methods (like GRPO, DAPO), which are trained solely on rewards. To address the limitations of standard RKL (Reverse KL), which suppresses exploration due to excessive constraints on the reference policy ($\pi_{\text{ref}}$), CARE-RFT introduces Skewed Reverse KL (SRKL). This SRKL uses a mixture distribution of the current and reference policies ($\alpha \cdot \pi_{\theta} + (1-\alpha) \cdot \pi_{\text{ref}}$) as an anchor. This approach achieves both exploration and reliability by applying only a finite penalty when the model confidently increases probabilities, while strongly suppressing probability decreases in uncertain directions. Mathematically, it possesses a finite upper bound, which prevents gradient explosion. Experimentally, on benchmarks like MATH, GSM8K, and TruthfulQA, CARE-RFT was shown to maintain reasoning performance similar to existing RFT methods while significantly improving the Expected Calibration Error (ECE), demonstrating that it achieves a balance between performance and calibration.

### Strengths
The strength of this paper is that it raises a highly relevant problem within the current research and temporal context.

Recently, reinforcement learning-like techniques such as RFT, GRPO, and DAPO have been actively studied to enhance the reasoning abilities of LLMs. However, most of these are trained solely on rewards based on correct/incorrect answers, which has exposed a problem where models become progressively overconfident and lose calibration.

CARE-RFT diagnoses the root cause of this phenomenon as the asymmetry of reward propagation and the rigidity of the KL constraint. In proposing a confidence-aware regularization (SRKL) to mitigate this, the study is both very timely and necessary.

### Weaknesses
The paper's main weakness is the lack of mathematical justification linking the proposed regularization term (SRKL) and calibration, specifically regarding its connection to **proper scoring rules**.

CARE-RFT claims to mitigate overconfidence via "confidence-anchored regularization." However, it provides no theoretical rationale for whether this regularization actually ensures "properness"—that is, consistency between the predicted probabilities and the true answer distribution.

In other words, while the finite upper bound and asymmetric penalty structure of SRKL might improve training stability, this is fundamentally different from the "probabilistic-regularity" or "truth-consistent calibration" guaranteed by proper scoring rules like the Brier score or log score.

Consequently, CARE-RFT's "improved calibration" appears to be more of an **empirical correlation** rather than a mathematically justified outcome. The causal link—*Relaxed KL bound $\rightarrow$ Stabilized confidence $\rightarrow$ Improved calibration*—remains formally unproven.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses a key limitation of reinforcement finetuning (RFT) for reasoning LLMs: unconstrained RFT improves accuracy but harms calibration and factual reliability. It proposes CARE-RFT, a variant that introduces a skew reverse KL (SRKL) regularizer to anchor uncertain token updates while allowing confident deviations. Through experiments on Qwen2.5-3B/7B across several reasoning and trustworthiness benchmarks, CARE-RFT shows a more balanced trade-off between reasoning gains and calibration loss compared to standard RKL. The method is simple, well-motivated, and empirically effective within math-based reasoning settings.

### Strengths
The paper tackles a practically important issue in reinforcement finetuning (RFT) — the trade-off between reasoning performance and trustworthiness. The idea of using a skew reverse KL regularizer to control confidence-dependent updates is conceptually sound and adapts classical divergence theory in a clear, incremental way.

Empirically, the study is solid within its scope: it evaluates two model scales (3B, 7B), three representative RFT algorithms (GRPO, DAPO, GSPO), and several reasoning and factuality benchmarks. The diagnostic analysis of reward updates (+Reward/–Reward/Full) provides an intuitive understanding of how unconstrained RFT destabilizes model calibration, justifying the need for regularization.

The paper is clearly written and easy to follow, with clean mathematical exposition and well-organized figures illustrating the relationship between entropy, confidence, and divergence skew.

Overall, while the contribution is incremental, CARE-RFT presents a practical and interpretable regularization method that meaningfully improves the reasoning–reliability trade-off in current RFT practice. Its value lies in providing an empirically grounded refinement rather than a radical conceptual advance.

### Weaknesses
1.Regularizer-strength trade-off is under-explored. The paper provides a clear ablation over the skew parameter α for SRKL (Table 3), showing that α≈0.8 yields a good balance between reasoning and trustworthiness. However, the divergence strength β is fixed at 0.04 for both RKL and CARE-RFT, and its effect is not studied. This leaves open whether CARE’s advantage persists across a broader range of constraint strengths, or whether a well-tuned RKL baseline could close much of the gap in the reasoning–calibration trade-off. A β-sweep for both RKL and SRKL (e.g., plotting MATH vs ECE as β varies) would make the case for SRKL much more convincing.

2.Scope and external validity. All training is conducted with an outcome-level correctness reward on MATH, and evaluation focuses on math reasoning (MATH, GSM8K) plus two factuality / self-awareness benchmarks (TruthfulQA, SelfAware) for the same Qwen2.5-3B/7B family.  While this is a reasonable starting point, the framing in the abstract and introduction sometimes reads as if the conclusions apply broadly to “trustworthy reasoning models” under RFT. It would be helpful to either (i) slightly temper these claims, or (ii) add at least one non-math or non–outcome-reward setting (e.g., step-wise or preference rewards) to support broader generality.

3.Strength of the “recovering trustworthiness” claim. The paper repeatedly states that CARE-RFT “matches the reasoning performance of unconstrained RFT while recovering the trustworthiness and calibration of the base model.”  However, Tables 2 and 4 show a more nuanced picture: CARE often lies between unconstrained and RKL variants on TruthfulQA and ECE, and on Qwen2.5-3B it does not fully return to base-model calibration (e.g., ECE 0.132 vs 0.102 for GRPO).  The trade-off it achieves is still quite favorable, but the empirical evidence supports “approaching” or “maintaining strong trustworthiness” more than complete “recovery” — especially given the absence of multi-seed variance or significance reporting

### Questions
1.On β and the regularizer–performance trade-off. Have you run any preliminary experiments varying the KL strength β for both RKL and CARE-RFT (e.g., β ∈ {0.01, 0.02, 0.04, 0.08})? If so, does CARE-RFT continue to dominate RKL across a range of β, or can a well-tuned RKL baseline close most of the reasoning–calibration gap?

2.On scope and generality beyond MATH outcome rewards. Do you have any preliminary results, or at least concrete expectations, for how CARE-RFT would behave under different reward structures (e.g., step-wise/process rewards, pairwise preference rewards) or on non-math tasks such as open-domain QA or code writting?

3.On the strength of the “recovering trustworthiness” claim. Could you report multi-seed results or at least standard deviations for the main tables (MATH, TruthfulQA, ECE), so readers can judge whether the observed 1–2 point differences are statistically meaningful?

### Soundness
3

### Presentation
3

### Contribution
2
