# Off-Policy Evaluation for Ranking Policies under Deterministic Logging Policies

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Off-Policy Evaluation (OPE) is an important practical problem in algorithmic ranking systems, where the goal is to estimate the expected performance of a new ranking policy using only offline logged data collected under a different, logging policy. Existing estimators, such as the ranking-wise and position-wise inverse propensity score (IPS) estimators, require the data collection policy to be sufficiently stochastic and suffer from severe bias when the logging policy is deterministic. In this paper, we propose novel estimators, Click-based Inverse Propensity Score (CIPS) and Click-based Doubly Robust (CDR), which exploit the intrinsic stochasticity of user click behavior to address this challenge. Unlike existing methods that rely on the stochasticity of the logging policy, our approach uses click probability as a new form of importance weighting, enabling low-bias OPE even under deterministic logging policies where existing methods incur substantial bias. We provide theoretical analyses of the bias and variance properties of the proposed estimators and show, through synthetic and real-world experiments, that our estimators achieve significantly lower bias compared to strong baselines, particularly in settings with completely deterministic logging policies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies off-policy evaluation (OPE) for ranking policies under deterministic logging, where existing estimators such as IPS, IIPS, and RIPS suffer from severe bias due to the lack of exploration. The authors propose a new estimator, Click-based Inverse Propensity Score (CIPS), which leverages the stochasticity of user click behavior instead of the logging policy. The paper provides theoretical analyses proving that CIPS is unbiased under mild assumptions and demonstrates through synthetic and real-world experiments that CIPS substantially reduces bias compared to existing methods.

### Strengths
1. The paper addresses an important and realistic problem in OPE that has been largely overlooked — deterministic logging policies.
2. The idea of exploiting user click stochasticity as a new source of importance weighting is conceptually novel.
3. Theoretical results are clearly stated and well justified.
4. The paper is clearly written and easy to follow.

### Weaknesses
1. The effectiveness of CIPS depends on the accuracy of estimated click probabilities, which may be difficult to obtain in real systems.
2. The independence of potential rewards assumption may not hold in many practical ranking scenarios.
3. Real-world validation relies on simulated click probabilities (Eq. 13), limiting the strength of empirical evidence.
4. No statistical significance or confidence intervals are reported for the experimental results.

### Questions
1. Could the authors clarify the rationale behind the reward design, particularly the two-stage structure of “clicks” and “potential rewards”? How is this modeling choice connected to real-world ranking scenarios, and to what extent does it affect the theoretical guarantees or the empirical behavior of CIPS? For example, would CIPS still perform similarly if the downstream reward R were not conditioned on the click C?
2. In the real-world experiments (KuaiRec), the click probabilities were generated using a simulated noise model (Eq. 13).
Could the authors provide more discussion or evidence about how well this setting reflects real deterministic logging systems?
3. While CIPS focuses on bias reduction, could the authors elaborate on its variance behavior in practical settings where click probabilities are estimated with noise? How sensitive is CIPS to misspecification of the click model?

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
3

### Summary
This paper investigates the severe bias introduced by existing Off-Policy Evaluation (OPE) methods when applied under deterministic logging, and illustrates this problem through concrete examples and theoretical analysis. To address this issue, the authors propose a Click-based Inverse Propensity Score (CIPS) estimator, which exploits the intrinsic stochasticity of user click behavior through click importance weights, substantially relaxing the support condition and enabling unbiased or low-bias estimation under deterministic logging. The effectiveness of the proposed method is thoroughly validated through rigorous theoretical analysis that establishes its unbiasedness and consistency properties, and extensive experimental results that empirically confirm its superior performance over existing OPE approaches under deterministic logging.

### Strengths
1. The paper is motivated by a well-founded research question — existing OPE methods are primarily developed for stochastic logging policies, which inherently limit their reliability under deterministic settings. This motivation is both sound and clearly articulated.
2. To alleviate the bias commonly induced by the OPE methods under deterministic logging, the paper relaxes the stochasticity assumption of existing OPE methods and takes a novel perspective by leveraging the inherent randomness in user click behaviors. By leveraging click probabilities for reweighting, the proposed CIPS method represents a clever and conceptually elegant reformulation of the problem.
3. The paper clearly illustrates the bias issue of IPS-based estimators under deterministic logging through intuitive examples, making the problem easy to understand. Moreover, the theoretical analysis is thorough and well-justified, and the experimental results further validate that CIPS effectively mitigates bias across varying degrees of determinism in logging policies.

### Weaknesses
1. The CIPS method implicitly assumes consistency in the action space between the old and new policies. It remains unclear whether CIPS can maintain its robustness when new actions appear under the new policy.
2. In the synthetic data section, the exact forms of functions appearing in Equations (9) and (10) are not clearly specified.
3. The paper lacks a dedicated related work section. The relation between this work and related work should be discussed. 
4. Baselines are limited. More recent baselines should be used. The strongest baseline, RIPS, was proposed in 2020; it is unclear whether more recent OPE methods should be considered for comparison.

### Questions
1. In real-world scenarios, although the IPS exhibits relatively large bias compared to other approaches, why does this bias remain unchanged when the logging policy shifts from stochastic to more deterministic?
2. In real-world scenarios, why does the CIPS method appear less robust to bias across varying degrees of deterministic policies compared to synthetic data? What might be the underlying causes of this discrepancy?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the challenge of Off-Policy Evaluation (OPE) for ranking policies, which aims to estimate a new policy's performance using data collected by a different, existing policy. The authors highlight that standard estimators, like Inverse Propensity Score (IPS), fail and suffer from severe bias when the logging policy is deterministic.
The paper's contribution is an estimator called the Click-based Inverse Propensity Score (CIPS). The key finding is that reliable OPE is possible even with deterministic logging by exploiting a different source of randomness: the intrinsic stochasticity of user click behavior. Instead of weighting by the policy's action probabilities, CIPS uses the ratio of marginalized click probabilities as a new form of importance weighting. Theoretical analysis shows CIPS is unbiased under specific conditions. Experiments on synthetic and real-world data demonstrate that CIPS achieves lower bias and more accurate estimations than existing methods in deterministic settings.

### Strengths
* The paper's primary strength is proposing an approach to address the Off-Policy Evaluation (OPE) issue when the data-logging policy is fully deterministic, a setting where existing methods fail.

* The CIPS estimator method interestingly shifts the source of randomness away from the deterministic policy and instead exploits the intrinsic stochasticity of user click behavior for importance weighting. This approach is shown to "significantly" reduce the severe bias of standard estimators in this challenging but practical setting.

### Weaknesses
The following are some of the items that it make the paper better to see them developed/explained/addressed further:

* The theoretical unbiasedness of CIPS depends on Condition 3.2, which assumes that the expected potential reward for an item (e.g., a purchase after a click) depends only on that item, not on the other items shown in the ranking. The paper's own experiments show that as this condition is more strongly violated, the Mean Squared Error (MSE) of CIPS, while still the best, does increase. The experiment section didn’t develop much on this aspect and how to potentially address it.

* The CIPS estimator relies on knowing the true click probabilities for both the logging and new policies. In practice, these are unknown and must be estimated from data. As Theorem 3.2 shows, any inaccuracy in the ratio of these estimated click probabilities will introduce bias into the final policy evaluation.

* While CIPS is designed for deterministic logging policies, it can still suffer from "non-negligible bias" in the most challenging scenarios. Specifically, the real-world experiment (which used a deterministic new policy in addition to a deterministic logging policy) showed this bias. The authors state this is due to violations of the click-wise common support (Condition 3.1), which can happen when both policies are deterministic. It would be great to develop further how to address this.

### Questions
1- Your theoretical analysis for CIPS's unbiasedness relies on the "Independence of Potential Rewards" (Condition 3.2). How robust is the CIPS estimator in practical scenarios where this assumption is strongly violated, such as systems with significant item complementarity or competition, and what is the expected impact on its bias?

2- Theorem 3.2 indicates that the bias of CIPS depends on the accuracy of the ratio of estimated click probabilities. How sensitive is the estimator's performance to errors in this ratio, particularly in low-data regimes, and what are the practical implications for model selection when estimating these click probabilities from logged data?

3- Your real-world experiment noted non negligible bias when both the logging and new policies were deterministic. This suggests violations of the clickwise common support (Condition 3.1). Could you elaborate on the performance degradation in this specific scenario and discuss potential extensions or clipping methods to mitigate this bias when even clickbased support is deficient?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the off-policy evaluation for ranking policies. Existing approaches require the data collection (logging) policy to be sufficiently stochastic and do not perform well when it is deterministic. This paper has overcome this challenge by proposing novel estimators, referred to as Click-based Inverse Propensity Score (CIPS), based on the intrinsic stochasticity of user click behavior. This paper proceeds as follows: the considered off-policy evaluation problem is formulated in Section 2, and the CIPS estimator is proposed and analyzed in Section 3. Preliminary experimental results are demonstrated in Section 4.

### Strengths
- This paper has considered a very practical problem: how to do off-policy evaluation for ranking policies when the data collection (logging) policy is deterministic (or not sufficiently stochastic). This is a practical and important problem in algorithmic ranking systems, such as recommendation systems.

- The proposed algorithm, CIPS, is natural and simple.

Overall, I think this is an interesting paper, and recommend accepting it.

### Weaknesses
- I have concerns about some key math notations in this paper. In particular, notations like $C(a)$, $R(a)$, $C(k)$, and $R(k)$ can be misleading. Specifically, $C(k)$ hints that the click event **only** depends on the position $k$, while $C(a)$ hints that the click event **only** depends on the action $a$. I do not think this is what this paper has assumed. I recommend that the authors clean up such notations.

- The idea of this paper is very interesting. However, given the idea and the CIPS algorithm, the analyses for the CIPS algorithm (Theorems 3.1-3.3) are relatively straightforward based on existing literature and techniques. If the authors disagree, please explain the key challenges and novelties in the proofs for Theorems 3.1-3.3, compared to existing literature.

- In the experiments, this paper has only considered a fixed logging policy defined in equation 11 and a fixed "new" policy defined in equation 12. First, please better motivate and explain why choosing such a logging policy and such a new policy. Second, I strongly recommend that the authors also include experimental results with a different logging policy and a different new policy to justify that the advantages of CIPS are not limited to such policies.

- In Section 4.2, the click probability model for real-world data defined in equation 13 seems pretty arbitrary. Please better motivate and justify it.

### Questions
- Please try to address the weaknesses listed above. I will consider increasing my score if some of them are addressed.

- If possible, I recommend moving the Click-based Doubly Robust estimator to the main body of this paper. I think it is important.

### Soundness
3

### Presentation
3

### Contribution
3
