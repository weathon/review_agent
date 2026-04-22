# From Curiosity to Caution: Mitigating Reward Hacking for Best-of-$N$ with Pessimism

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 8, 4, 6

## Abstract
Inference-time compute scaling has emerged as a powerful paradigm for improving language model performance on a wide range of tasks, but the question of how best to use the additional compute remains open.  A popular approach is *Best-of-$N$* (BoN) sampling, where $N$ candidate responses are generated, scored according to a reward model, and the highest-scoring response is selected.  While this approach can improve performance, it is vulnerable to *reward hacking*, where performance degrades as $N$ increases due to the selection of responses that exploit imperfections in the reward model instead of genuinely improving generation quality. Prior attempts to mitigate reward hacking---via stronger reward models or heavy-handed distributional regularization---either fail to fully address over-optimization or are too conservative to exploit additional compute.  In this work, we explore the principle of *pessimism* in reinforcement learning (RL), which uses lower confidence bounds on value estimates to avoid out-of-distribution (OOD) actions with uncertain reward estimates. Our approach, termed as *caution*, can be seen as the *reverse* of *curiosity*: where curiosity (e.g., via Random Network Distillation, RND) rewards prediction error as a signal of novelty, caution penalizes prediction error as a signal of distributional uncertainty. Practically, caution trains an error model on typical responses and uses its prediction error to lower reward estimates for atypical ones. Our extensive empirical evaluation demonstrates that caution is a simple, computationally efficient approach that substantially mitigates reward hacking in BoN sampling.  We also provide a theoretical analysis in a simplified linear setting, which shows that caution provably improves over the standard BoN approach.  Together, our results not only establish caution as a practical solution to reward hacking, but also provide evidence that curiosity-based approaches can be a general OOD detection technique in LLM settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Best-of-N with pessimism to mitigate potential reward hacking issues in test-time scaling. Specifically, they capture "caution" by training a lightweight predictor network to reconstruct reward model features, and the pessimistic score is larger when the prediction error is lower (the reward model is more familiar with the response pattern). Experiment results on GSM8K, Math-500 and BBH-hard demonstrate that their methods are stronger than the Best-of-N baseline.

### Strengths
- The proposed pessimistic score is reasonable, which accounts for the reward model's familiarity with the input. The reward design thus becomes the original reward minus the prediction error, and is clear and easy to implement.
- Experiment results demonstrate that the proposed pessimistic score is effective when applied to Best-of-N method. What's more, it is also useful for OOD domain test-time scaling.

### Weaknesses
- The authors did not compare with other strong test-time scaling methods, such as self-consistency. So it is unsure whether their method is widely applicable to other test-time scaling methods, as well as state-of-the-art inference time scaling methods.
- The proposed method is not as lightweight as the authors claimed. In fact, it requires extra training for each reward model. That means for every reward model on a specific task, they need to train a predictor network again based on the specific reward model.
- The layer number L is a hyperparameter, and it is unclear how they select it. The authors only mentioned in appendix that they use L=10. It is unsure whether this hyperparameter can be applied to other settings / domains / reward models.

### Questions
See weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses reward hacking in Best-of-N (BoN) sampling. The authors introduce an elegant, lightweight, and practical solution called "caution." 

The method works by training a lightweight predictor network to mimic the internal features of a given reward model on in-distribution data. At inference time, the predictor's error on a new candidate response serves as an uncertainty penalty, which is subtracted from the original reward score. Extensive experiments demonstrate that this approach effectively mitigates reward hacking, leading to improved performance with N.

### Strengths
- A core strength is the paper's excellent presentation and framing of the problem. Drawing a parallel between curiosity-driven exploration in RL and "caution" for avoiding OOD reward exploitation is an insightful and generally useful contribution.
- The proposed solution is compelling in its simplicity and practicality. Training the auxiliary predictor network is a one-time offline cost (with unlabeled data), and the inference-time overhead is minimal as it leverages the same internal representations as the reward model. 
- The experiments are extensive and well-designed.

### Weaknesses
W1. This is both a perceived weakness and a question. Please correct me if I misunderstood the setup. The "caution" mechanism defines "in-distribution" by learning the typical feature patterns of the base reward model. This is effective at penalizing novel responses that exploit weird loopholes, but it may fail to mitigate, and could even reinforce, the reward model's own systemic biases which you mention in Lines 81-83 and in Figure 4. For instance, if a reward model was trained on data where humans consistently preferred verbose answers, its internal features would encode verbosity as a "normal" characteristic of high-quality responses. The caution predictor would learn this pattern and would consequently fail to penalize a verbose, incorrect, reward-hacking response. It might even penalize a concise, correct response for being stylistically "out-of-distribution." In essence, the method regularizes generations toward the central tendencies of the reward model, and if those tendencies are themselves flawed, the method risks preserving those flaws. If doing so is indeed shown to mitigate hacking, why do you think hacking happens to begin with?

W2. What is the most fundamental way to combine a reward signal and an uncertainty signal with BoN? Subtracting the uncertainty from the reward feels very natural from an RL / training point of view. However, BoN only cares about the ranking of the responses according to their rewards, making it insensitive to any shifts or rescaling. Is there a way we can introduce uncertainty in the ranking itself and how would you justify subtraction for inference-time alignment? I think this aspect is worth discussing in the paper and makes your decision of running a z-score transformation followed by using a global hyperparameter to balance the subtraction of the reward with the uncertainty more convincing.

Other work considered, for example, a multiplicative exponential term to model uncertainty in reward scoring and they motivate it well for policy optimization [1].

[1] Houliston, Sam, et al. "Uncertainty-penalized direct preference optimization." arXiv preprint arXiv:2410.20187 (2024).

### Questions
- How do you expect the "caution" mechanism to perform if the base reward model is systematically biased? 

- The paper's discussion of Huang et al. (2025b) is very clear. A direct empirical comparison on the same experimental setup would be appreciated, as it represents the most relevant alternative approach.

- Why do you not consider calibrated reward as done in InfAlign (Balashankar et al., 2024)? Since Best of n only cares about the ranking of the reward scores, such a calibration is theoretically motivated. Instead, you opt for a z-score transformation. Can you elaborate on lines 300-302? In particular, did you perform the z-score transformation per prompt or across prompts?

- You may consider citing [2]. They use the reward model's learned representation of the prompt response pair to model its uncertainty. They argue when a data point is (almost) in distribution, there is less uncertainty.

[2] Zhang, X., Ton, J. F., Shen, W., Wang, H., & Liu, Y. (2024). Overcoming reward overoptimization via adversarial policy optimization with lightweight uncertainty estimation. arXiv preprint arXiv:2403.05171.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In this work, the authors mitigate reward hacking by adding a pessimism or caution term to a reward model. Specifically, they focus on Best-of-N sampling and mitigate the selection of high proxy-reward, low true-reward generations by penalizing out-of-distribution samples where the proxy-reward estimate is uncertain and inaccurate. They do so by training a second network to reconstruct the proxy-reward model's latent representation. Then, on future data, they use the error of this reconstruction as a measure of the certainty of the reward model. They regularize their reward with this certainty term, penalizing OOD samples which result in low reconstruction while having abnormally high reward.

### Strengths
* The method is intuitive and interesting in its own right: quantifying OOD-ness of data using a second model trained with reconstruction loss
 * The paper is clear and well-written and provides a thorough study of their method and demonstrates its efficacy.
 * The empirical results demonstrate hacking mitigation on mathematical reasoning tasks.

### Weaknesses
* The authors say they see "monotonic" performance improvements for all N, yet in my opinion the curves look more like a plateauing reward. Indeed, past work on hacking often frames the approach as avoiding over-optimization and recovering an "optimal" reward rather than expecting gains as N increases forever [1]. I don't think this claim is necessary for the impact of the paper, and I think some may disagree with it.
 * Notationally, it was a little confusing introducing $T$ as separate from $r$. I understand that you take some intermediate layer from $r$, but I might define that more explicitly in the paragraph on line 202 or streamline notation. 

[1] Khalaf, H., Verdun, C. M., Oesterling, A., Lakkaraju, H., & Calmon, F. D. P. (2025). Inference-Time Reward Hacking in Large Language Models

### Questions
* In Table 2, it seems you use $\lambda$ as a convex combination of reward terms, $\lambda r(x) + (1-\lambda) u(x)$. This is different than your regularization term introduced in line 236. Can you standardize this?
 * Intuitively why do we expect any performance gains when using caution only? We aren't using any reward signal at all, and are just picking completions that are more in distribution. What are the costs in terms of coherency or diversity in completion?
 * In Table 3, for Lightweight + Separate Emb.: How can Peak Acc be lower than Final Acc?
 * In your proofs, you condition on $\theta$, but isn't $\theta$ not a random variable? Why are you conditioning on it?
 * Can you explain in Prop. 2 where you use the $E[r^*|\hat{r}]$?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
* The paper aims to improve the degradation in performance in the Best-of-N sampling scheme for large values of $N$ ("reward hacking") by constructing a weighted sampling scheme which combines reward and uncertainty scores.
* In the model, a language model stochastically maps a prompt $x$ to a response $y$, a reward model $r$ is a mapping from prompt-response pairs $(x,y)$ to scores. Given reward model $r$, uncertainty estimate $u$ and hyper-parameter $\\lambda$, the proposed approach samples candidates according to the “pessimistic” reward $r_\\mathrm{LCB}=r-\\lambda u$. The uncertainty estimate is inspired by random network distillation, aiming to detect OOD datapoints. 
* A theoretical analysis is briefly described in the body of the paper, and presents a simplified setting in which pessimistic sampling converges towards the optimal-reward response but the absolute gap between naive and pessimistic sampling grows with $N$.
* The empirical analysis evaluates performance on three datasets, showing favorable performance compared to the Best-of-$N$ baseline.

### Strengths
* Addresses a well-motivated and timely issue.
* Proposed method seems relatively simple to implement, suggesting potential practical applicability.
* The presentation is generally clear and easy to follow.

### Weaknesses
* Despite the focus on empirical analysis, the analysis code doesn’t seem to be attached.
* While related prior work is mentioned (e.g., Huang et al. 2025, Jinaai et al. 2024), empirical evaluation doesn’t seem to include it as a possible baseline.
* The theoretical analysis underlying section 2.3 seems to be relatively intricate, but the body of the paper presents it only briefly. In particular, key assumptions, their limitations, implications of the analysis, and proof techniques are do not seem to be discussed.

### Questions
* Is it possible to elaborate on the intuitive assumptions and limitations underlying the theoretical analysis?
* What are the general limitations of the proposed approach, and when is it expected to fail? 
* How does the method handle cases in which the ground truth reward has “true" label noise? (e.g., topic in which annotators have typically diverse preferences).
* Is it possible to formulate the relation the proposed method as a "soft distance constraint" between the training set and candidate response distributions? For example, if I understand correctly, it seems that only "in-distribution" candidates will be sampled when $\\lambda \\to \\infty$ and $u$ is a perfect OOD detector, because any OOD candidates will be penalized heavily. Does this intuition hold? And if so, what can be said about intermediate values of $\\lambda$?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the problem of reward hacking in Best-of-N sampling, where due to biases of the reward model sub optimal preferences might be made. The authors propose "caution," an inference-time technique that instantiates the principle of pessimism from reinforcement learning. It works by training a lightweight "predictor" network to predict the internal feature representations of the frozen reward model on a dataset of typical responses. The prediction error of this network is then used as an uncertainty score. This score is subtracted from the reward model's output to create a "pessimistic" reward, which penalizes responses that are out-of-distribution (OOD) from the perspective of the reward model. The authors demonstrate this approach with significant gains over GSM8K, MATH-500, BigBench-Hard reasoning tasks.

### Strengths
- The paper addresses an important and persistent problem in reward hacking that exists despite a large amount of papers on the subject.
- The experimental evaluation is comprehensive and well-designed. The testing across different distributions and domains (GSM8K, MATH-500, BBH) convincingly demonstrates the robustness of the approach.
- The ablation studies in Section 3.2 are thorough and successfully validate the key design decisions, especially the critical importance of using reward model features over random ones.
- The proposed method is computationally efficient and practical. The predictor network is trained fully offline, and at inference time, its forward pass can be parallelized with the reward model's, adding minimal overhead. This makes it a readily applicable solution for practitioners already using BoN sampling.

### Weaknesses
- Weak theoretical justification, It is unclear how the insights from this linear-Gaussian model translate to the complex, high-dimensional geometry of transformer feature spaces.
- Stronger baselines. The paper could be strengthened by comparing "caution" to other plausible inference-time uncertainty estimation or OOD detection techniques. There has also been a lot of work in reward hacking mitigation, so a comparison with more techniques would be appreciated.
- Qualitative examples - the paper would benefit from more qualitative study of the hacking mitigations
- Human alignment - more human evaluation would strengthen the usefulness of the approach for most tasks.

### Questions
- The main comparison is to standard BoN. How might "caution" compare to other classes of OOD/uncertainty detection methods applied at inference time, such as using Monte Carlo dropout on the RM's final layers or calculating Mahalanobis distance in the feature space of the RM to penalize outliers?
- The predictor Pθ is trained on typical responses from one policy (Llama-3.2-3B). How well would this pre-trained predictor generalize if used to score responses from a different, more capable base model?
- he case study highlights the detection of formatting errors. Can you provide examples of more semantic or stylistic deviations that the method successfully identifies and penalizes?

### Soundness
3

### Presentation
4

### Contribution
3
