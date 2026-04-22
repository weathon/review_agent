# Probabilistic Uncertain Reward Model

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Reinforcement learning from human feedback (RLHF) is a critical technique for training large language models. However, conventional reward models based on the Bradley-Terry model (BTRM) often suffer from overconfidence when faced with inconsistent labels or out-of-distribution samples, leading to reward hacking, where the policy model blindly optimizes for proxy rewards while degrading true performance. This paper proposes the Probabilistic Uncertain Reward Model (PURM), which generalizes the Bradley-Terry model to learn the reward distributions that emerged from the preference data. We theoretically derive the loss function of PURM and introduce a novel method that uses the overlap between distributions to define and derive the quantify uncertainty. Empirical results show that PURM outperforms existing methods with more accurate reward and sound uncertainty estimations, and sustains effective learning for more optimization steps and obtain higher maximum win rate in RLHF. The data and code of this paper are released at https://anonymous.4open.science/r/Probabilistic-Uncertain-Reward-Model/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the Probabilistic Uncertain Reward Model (PURM), a generalization of the
classical Bradley–Terry Reward Model (BTRM) that represents rewards not as scalars but as Gaussian distributions.
The authors derive the corresponding loss function for PURM and propose a new metric to quantify uncertainty.
Empirically, they show that PURM matches existing reward models performance when predicting the reward, pro-
vides a sometimes more reliable measure of uncertainty, and seems to help mitigate reward hacking when used to train an LLM policy
while improving win rates.

### Strengths
1. The challenge of addressing both uncertainty and reward hacking in reward models is interesting, and this paper makes a step in this direction. Modeling the reward as a normal variable seems novel.
2. Proofs are written step by step and are easy to follow.

### Weaknesses
1. Conclusions about hyperparameter choice seems to be drawn from experiments by running a single seed, which
is far from ideal given the relatively small differences shown in the figures. This makes the results far less convincing. In particular, there seems not to be a large (or any?) difference between using the overlap-measure of uncertainty or just simply the sigma (empirically, that is).
2. Some metrics, such as length-controlled win rate, are not clearly defined or explained.
3. Some inconsistencies between loss in the equation and the one coded up, see below in questions.

### Questions
1. What is reason for changing the judge in section 3.3 between Figures 5(a) and 5(b) ?
2. In your implementation of the PURM loss in the code and in Appendix B, you used the average of log sigmoid(z),
whereas the straightforward loss from equation 4 is using the log of average of sigmoid(z). This seems to be a discrepancy that you do not mention. The loss you seem to use is an upper bound via Jensen’s inequality, can you explain why not simply use the log of average of sigmoid(z) instead ?
3. In section 3.2 you claim that your measure of uncertainty performs better, but figure 5(a) shows that using
just sigma gives comparable performance (given the experiment uses only one seed). How does the average of sigma
or the average of sigma divided by mu behave ? Can you make a convincing case that your measure performs better?
4. In figure 10, why do the curves with w=1e5 and w=1e6 differ even at early optimization steps < 400 when
there isn’t supposed to be any difference as the list size still hasn’t reached 1e5 ? And if this difference comes
from randomness during optimization then how and why did you conclude that 1e6 is the best hyperparameter
choice ?

[Extra question out of curiosity] DPO rewrites the BT reward to arrive at a loss that gets rid of the reward model alltogether. Can you do something similar for your case, when you assume the rewards are Gaussian normals?

### Soundness
3

### Presentation
3

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
This paper proposes the Probabilistic Uncertain Reward Model (PURM) as an extension of the classic Bradley–Terry reward model. Instead of producing a deterministic scalar reward, PURM outputs the parameters of a Gaussian distribution (μ,σ). The Bhattacharyya coefficient is further employed to measure the overlap between reward distributions, which serves as an uncertainty quantification mechanism. The authors claim that this approach improves the stability of RLHF training and mitigates reward hacking.

### Strengths
1. Clear motivation: The deterministic outputs of BTRM indeed lead to overconfidence and reward hacking risks. The paper addresses a practically relevant issue.

2. Simple and intuitive method: Modeling rewards as distributions and incorporating overlap-based uncertainty is straightforward and low-cost to implement.
3. Interesting uncertainty quantification: Using the Bhattacharyya coefficient, rather than variance alone, better aligns with the intuition of distributional separability.
4. Reasonable experimental coverage: Includes tests on public preference datasets and RLHF settings, along with ablation studies.
5. Practical applicability: Minimal code changes are required to integrate PURM into existing RLHF frameworks.

### Weaknesses
1. Limited novelty: Distributional reward modeling and uncertainty quantification are not new ideas. For example, URM[1] already proposed modeling reward uncertainty via probabilistic distributions. PURM is conceptually similar but does not clearly articulate its unique theoretical or empirical contributions.
2. Strong Gaussian assumption: Assuming reward distributions follow a Gaussian lacks justification. Real-world preference data may be skewed or multi-modal, raising concerns about robustness.
3. High sensitivity to λ: The effectiveness of the method strongly depends on the penalty coefficient λ, yet the paper provides no principled guidance or adaptive mechanism for its selection, limiting practical usability.

[1] Lou, Xingzhou, et al. "Uncertainty-aware reward model: Teaching reward models to know what is unknown." arXiv preprint arXiv:2410.00847 (2024).

### Questions
See weaknesses.

### Soundness
2

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
This paper proposes Probabilistic Uncertain Reward Model (PURM) as an extension of BTRM used in RLHF. Instead of modeling scalar rewards, PURM represents each reward as a Gaussian distribution parameterized by a mean $\mu$ and a standard deviation $\sigma$. This probabilistic formulation aims to capture uncertainty and mitigate reward hacking.

Parts of this review were discussed with a colleague to ensure clarity and accuracy.

**Contributions:**
1. Introduces a probabilistic variant of BTRM that outputs a Gaussian reward distribution rather than a scalar value.
2. Derives a tractable Monte Carlo–based training objective for learning from preference data.
3. Proposes a novel use of the Bhattacharyya Coefficient to quantify uncertainty in reward modeling.
4. Integrates uncertainty into RLHF by penalizing uncertain rewards to mitigate reward hacking.

### Strengths
1. The experimental results are strong;
2. The idea of modeling reward uncertainty in RLHF is interesting and conceptually aligns with the intuition that reward confidence should guide policy learning.

### Weaknesses
1. The method makes the reward modeling problem much more complicated than necessary. The probabilistic formulation and uncertainty estimation introduce large computational overhead. It would be helpful if the authors could justify whether such improvements are worth the added cost in real-world RLHF settings.
2. The derivation in Eq. (7)–(9) treats the pairwise reward difference $r_1 - r_2$ as Gaussian, but the validity of this assumption is not discussed. Since both $r_1$ and $r_2$ are modeled as independent Gaussian variables, this implicitly assumes independence between responses, which is unrealistic in preference data (they are conditioned on the same prompt $x$).
3. The paper approximates the intractable sigmoid–Gaussian integral using Monte Carlo sampling, but does not discuss the computational cost of this approximation during large-scale training.
4. The proposed use of the Bhattacharyya coefficient as a global uncertainty measure (Eq. 14–15) is questionable. Averaging pairwise overlaps with a random subset of the dataset (Eq. 16) is computationally heavy and not theoretically grounded as an uncertainty estimator.
typographical issues:
1. Inappropriate citation styles: all references are in `\citep` form. Mixing `\citet` and `\citep` properly would improve readability.
2. Differential notation `d` in `dz`, `dw` and `sigmoid` should be typeset as an operator (e.g.`\mathrm{d}z`, `\mathrm{d}w` and `\sigma(z)`).

### Questions
See weaknesses.

### Soundness
3

### Presentation
1

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
This paper proposes the *Probabilistic Uncertain Reward Model (PURM)*, which extends the traditional Bradley–Terry Reward Model (BTRM) by introducing a probabilistic framework that models reward distributions rather than point estimates. PURM further quantifies uncertainty via the Bhattacharyya Coefficient, allowing uncertainty-aware reward penalties during RLHF to mitigate reward hacking. Empirical results show that PURM improves stability and achieves higher win rates over BTRM and other uncertain reward models.

### Strengths
1. The proposed method is simple, intuitive, and easy to implement. It appears effective in practice and introduces little additional latency compared to BTRM.
2. The *Uncertainty Evaluation* section presents particularly interesting observations, especially that PURM can adjust its reward uncertainty when training data labels are randomly flipped, while other baselines cannot.
3. The paper is well-written, with smooth logical flow and clear presentation of both the intuition and methodology.

### Weaknesses
1. **Conceptual novelty and related work.**
    The core idea, i.e. replacing a scalar reward with a Gaussian distribution and introducing an uncertainty-based penalty, is quite straightforward. I am surprised that such a distributional approach to reward modeling has not been explored before. I am not an expert in this subarea, but I found several potentially related works that are not discussed in the paper:

    - *Bayesian Reward Models for LLM Alignment*, ICML 2024 (Workshop)
    - *Active Preference-Based Gaussian Process Regression for Reward Learning*, RSS 2020
    - *Know What You Don’t Know: Uncertainty Calibration of Process Reward Models*, arXiv:2506.09338
    - *Aligning Crowd Feedback via Distributional Preference Reward Modelling*, ICLR 2025 (Workshop)

    If these works are relevant, the authors should position PURM more clearly relative to them, clarify its distinctive contributions, and include comparative experiments. If they are not directly related, it would still be valuable to explain *why* distributional reward modeling has received little prior attention.

2. **Empirical claims need stronger support.**

    - The choice of the Bhattacharyya Coefficient (BC) as the uncertainty measure is insufficiently justified. In Appendix C.1, its performance is not substantially better than simply using standard deviation, and this difference could likely be offset by tuning the hyperparameter λ.
    - Line 264 (“We attribute this to the fact that …”) asserts a causal interpretation that is not supported by explicit ablation or visualization.
    - Section 3.3 shows improved RLHF performance, but it does not clearly demonstrate that PURM mitigates *reward hacking* per se; it could simply reflect better learned reward.

3. **Limited exploration of downstream behavior.**
    The empirical analysis in Section 3.2 is incomplete. More fine-grained studies would greatly strengthen the paper, examples are:

    - Does PURM also improve policy model's robustness to noisy preference data or out-of-distribution (OOD) evaluation tasks?
    - How does it perform on tasks with inherently low reward noise, such as code or math reasoning where unit-test-based rewards are nearly deterministic?

    Exploring such aspects would definitely help the community understand the broader implications of probabilistic reward modeling.

------

### **Minor Issues**

1. Line 382: *“GPT-4o Hurst et al. (2024) is used …”* should read *“In Hurst et al. (2024), GPT-4o is used …”*.
2. The figures contain text that is too small to be readable after printing; font sizes should be increased for accessibility.

------

### Questions
See the concerns noted in the *Weaknesses* section. My current rating is deliberately conservative, but I would be happy to engage in discussion and to raise my ratings accordingly it if the authors address these issues.

### Soundness
2

### Presentation
3

### Contribution
3
