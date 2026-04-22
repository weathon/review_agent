# Reward Model Routing in Alignment

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Reinforcement learning from human or AI feedback (RLHF/RLAIF) has become the standard paradigm for aligning large language models (LLMs). However, most pipelines rely on a single reward model (RM), limiting alignment quality and risking overfitting. Recent work explores RM routing—dynamically selecting an RM from a candidate pool to exploit complementary strengths while maintaining \(O(1)\) RM calls—but existing methods suffer from cold-start and insufficient exploration. We propose {\name}, a hybrid routing framework that combines offline RM strengths learning with online Bayesian selection. In the offline stage, a multi-task router is trained on preference data to estimate per-RM reliability. In the online stage, a Bayesian Thompson sampling router performs per-query RM selection, initializing RM-specific weight vectors with offline embeddings as Gaussian priors and adaptively updating their posteriors with online rewards to adapt to the evolving policy distribution. Extensive experiments on instruction-following (AlpacaEval-2, Arena-Hard, MT-Bench) and reasoning (GSM8K, MMLU) benchmarks show that {\name} consistently outperforms individual RMs, RM ensembling, and existing routing methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a Bayesian approach to online reward model selection to improve LLM performance.
The approach combines offline and online data. With offline data, a neural network representation of preference pairs alongside two prediction heads for prompt-response preference pairs is trained. Based upon this model, the reward model with best prediction performance for a given prompt-response preference pair is selected.   With online data, the authors use a Bayesian linear regression model of reward model "utility" (based upon the representation obtained offline). A sample of reward models from the posterior distribution is used to select the model better suited for a given a prompt-response preference pair (Thompson-sampling).
The proposed approach is evaluated on experimental testbed with alternative reward model selection models such as majority opinion, uncertainty-weighted optimization and linear upper confidence bound (as in Nguyen et al., 2024).

### Strengths
1. The proposed approach aims to provide the RLHF training paradigm with the ability to adapt according to changing tasks while minimizing additional computational effort.

2. The numerical results indicate the proposed that reward model selection mechanism outperforms alternative approaches for model selection.

### Weaknesses
1. The choice of a linear model for utility is presented in ad-hoc fashion. A linear model has significant limitations for example it fails to capture interactions between features (e.g., "helpfulness is only valued if tone is polite") or hierarchical semantics (e.g., "if response is correct, then prefer shorter length; otherwise, correctness dominates").

2. The performance of the linear model of utility relies heavily on the representation quality (which is obtained offline).  Any attempt to jointly re-train representation and utility will take away the simplicity of Bayesian updating of a linear-gaussian model.

3. The choice of Gaussian noise is also ad-hoc. One can not represent multimodal preference beliefs (e.g., two equally plausible user preference patterns) with such model.

### Questions
1. I encourage the authors to provide arguments to support ad-hoc modeling choices (linear model of utility, robustness of neural representation for different tasks, Gaussian noise).

2. I find the alternative model selection models may not provide a competitive benchmark. Perhaps the linear upper confidence bound proposed in Nguyen et al., 2024 duly extended to account for preference pairs might be a more decent comparison.

2. Thompson sampling optimizes immediate expected gain under a single posterior sample. This can lead to suboptimal long-term exploration (over-exploitation) and inefficient learning when preferences are high-dimensional or multimodal.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes BayesianRouter, a hybrid reward-model (RM) routing framework for preference-based alignment (DPO family). Experiments on instruction following (AlpacaEval-2, MT-Bench, Arena-Hard) and reasoning (GSM8K, MMLU) claim consistent gains over: best single RM, ensembling (majority, UWO), and the prior routing baseline LASER

### Strengths
1. Clear problem framing: RM routing is a compelling middle ground between costly O(N) ensembles and single-RM training.
2. Using offline BT embeddings as priors for a Bayesian linear bandit seems a pretty interesting design.
3. Main results across five benchmarks show better performance over existing works
4. The paper argues O(1) RM calls with lightweight extra overhead, and compares wall-clock training to single-RM, LASER, and majority vote

### Weaknesses
1. To my understanding, the main novelty of this work is the prior injection from offline embeddings into online TS. The contribution is solid engineering + good fit to RM routing, but may feel unsurprising conceptually.
2. The online update uses (normalized) negative DPO loss per routed pair as the reward. This couples router training tightly to the current policy’s loss landscape and may conflate pair difficulty and RM quality.
3. Concerns on evaluation choices & fairness (see questions for details)

### Questions
1. Judge LLM and length-controlled AlpacaEval are used for instruction following evaluation. Would this risks judge-model bias? Results should report variance across judges.
2. Were their hyperparameters tuned per RM ($\beta$, LoRA rank, sampling temperature), or held constant—potentially under-optimizing some RMs?
3. For the computation cost, the router adds encodings + bandit updates; wall-clock plots are described, but exact GPU hours and per-step RM latency distributions (small vs large RMs) would clarify cost/quality trade-offs.
4. Offline router shows a non-trivial gap in OOD case; the paper argues online adaptation bridges it, but I don’t see stress tests where online distribution shifts during training. Is it possible to include such an experiment study?
5. Since reward is a function of policy loss, how stable is routing across training (do posteriors collapse to 1–2 RMs)? Please include posterior traces $(\mu,\Sigma)$ over steps and the share of traffic per RM if possible.
6. With larger pools (>8 RMs), do we still get O(1) advantages in practice? Any diminishing returns or exploration debt? (You mention scaling, but numbers are brief.)

Overall I think the paper tackles a real pain point in RLHF/RLAIF with a clean method and interesting results. Addressing my above concerns could further strengthen the paper.

### Soundness
3

### Presentation
3

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
This paper addresses the limitations of using a single, fixed Reward Model (RM) in alignment pipelines (RLHF/RLAIF), such as limited generalization and the risk of overoptimization. While RM ensembles improve robustness, they incur high computational costs (1$O(N)$ calls).2 Existing routing methods (like LASER) attempt to maintain $O(1)$ cost but suffer from cold-start issues and insufficient exploration

### Strengths
The paper tackles an important problem. The proposed hybrid approach, integrating offline pre-training with online adaptation, is in the context of RM routing and directly addresses limitations of prior work (cold-start in LASER, distribution shift in purely offline methods).

Recognizing the alignment between the offline Bradley-Terry head and the online Bayesian router (both acting as linear models over the preference-pair embedding) allows for the effective injection of offline embeddings as priors. This is empirically validated as superior to heuristic weighting schemes.

### Weaknesses
The most critical weakness lies in the definition of the reward signal used to update the online Bayesian router (Section 3.3). The reward is defined as the negative DPO loss: $\tilde{r}_n^i = -\mathcal{L}_{DPO}^i$ (L303). The DPO loss measures how well the policy aligns with the preference provided by the RM. A high reward (low DPO loss) indicates that the policy already agrees with the RM or can easily model that preference.Crucially, this is not a measure of the RM's quality or the correctness of the preference itself. If an RM provides an incorrect preference, and the policy easily learns this incorrect preference, the bandit receives a high reward. This methodology conflates the ease of policy alignment with the quality of the alignment signal, creating a perverse incentive that does not encourage the selection of the most reliable RM, and may instead accelerate confirmation bias or reward hacking.

Inconsistent Empirical Results and Exaggerated Claims: The paper claims that "Bayesian Router consistently outperforms all baselines on both instruction-following and reasoning benchmarks" (L354-356). This claim is demonstrably false according to the results presented in Table 1.On AlpacaEval-2, Bayesian Router (63.23) is outperformed by Majority vote (63.40) and UWO (63.60).On MT-Bench, Bayesian Router (58.75) is significantly outperformed by the single RM1 (64.80), UWO (61.74), and even Random router (61.20).Furthermore, based on the SFT baseline scores (54.29 for GSM8K, 67.63 for MMLU), it appears the columns for GSM8K and MMLU might be swapped in Table 1. If this is the case, the performance of Bayesian Router on MMLU (57.39) is drastically worse than RM1 (74.22), UWO (74.30), and LASER (74.00).The proposed method fails to consistently beat the strongest baselines, significantly undermining the central claims of the paper.

The experiments utilize a small pool (N=4) of relatively small, homogeneous RMs (ranging from 0.6B to 3B parameters, L328). The benefits of routing are typically most apparent when the pool includes diverse models with clear specializations (e.g., safety vs. reasoning) or vastly different scales. The limited diversity makes it difficult to assess the router's ability to effectively exploit complementary strengths.

### Questions
In addition to the weakness section, I have the below questions:

How do you justify using the negative DPO loss as the reward signal for the bandit? Given that this reward signal measures the policy's ability to learn a preference (right or wrong), rather than the correctness of the RM's preference, how does the proposed feedback loop ensure the selection of high-quality RMs and avoid reinforcing easily-learned but incorrect preferences?

The "Full Advantage" reward design mentioned in Appendix C seems more conceptually aligned with measuring RM quality than the negative DPO loss, although it is more expensive. Why was this conceptually stronger reward signal not adopted as the primary method, perhaps using the "Light Advantage" approximation to manage the cost?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses a key limitation in aligning large language models (LLMs): the reliance on a single reward model (RM) in pipelines like Reinforcement Learning from Human Feedback (RLHF). Using a single RM can lead to overfitting, reward hacking, and poor generalization across diverse tasks.


To solve this, the authors propose the Bayesian Router. This novel hybrid framework dynamically selects the most suitable RM from a candidate pool for each query during Direct Preference Optimization (DPO) training.


The method consists of two main stages:


* Offline Router Training: An offline model is first trained on existing preference datasets to learn the "strengths" or reliability of each RM in the pool. This router encodes the full preference pair (prompt, response 1, response 2). It uses a multi-task objective, combining a Bradley-Terry (BT) head (to learn relative RM abilities from disagreement samples) and a Classification (CLS) head (to predict an RM's correctness on any given sample).

* Online Bayesian Selection: The learned offline embeddings (specifically from the BT head) are used as a Gaussian prior to initialize an online Bayesian router. This online router, based on Thompson sampling, performs instance-level (per-query) RM selection during the DPO process. It adapts to the evolving policy by updating its posterior distribution using rewards derived from the DPO loss .





The paper claims that this hybrid approach effectively solves the "cold-start" problem (via the offline prior) and poor exploration (via Thompson sampling), achieving state-of-the-art results. Experiments on instruction-following (AlpacaEval-2, MT-Bench) and reasoning (GSM8K, MMLU) benchmarks show that Bayesian Router outperforms single RMs, ensemble methods, and the previous SOTA routing method, LASER.

### Strengths
* Well-Motivated Problem: The paper tackles a well-known and significant problem in LLM alignment—the overoptimization and brittleness of single-RM pipelines. The goal of efficiently leveraging a diverse pool of RMs is a clear and valuable research direction.
* Novel & Principled Method: The hybrid offline-online design is the paper's key strength. It correctly identifies the weaknesses of purely offline approaches (distribution shift) and purely online approaches (cold-start, inefficient exploration). Using the offline-learned BT embeddings as the prior mean $\mu_n^{(0)}$ for the online Thompson sampler is an elegant and principled way to integrate both components.
* Strong Design Choices: The method improves upon prior work (LASER) in several well-justified ways:
    * It uses instance-level routing, which is more granular than LASER's batch-level routing.
    * It employs Thompson sampling for better, uncertainty-aware exploration, directly addressing the potential for premature exploitation in LinUCB (used by LASER).
    * It encodes the full preference pair $(x, y_1, y_2)$ rather than just the prompt $x$, which is a more informative context for RM selection.

* Comprehensive Experiments: The evaluation is exceptionally rigorous.
    * It uses a good mix of instruction-following and reasoning benchmarks.
    * It compares against a strong set of baselines, including single-best RMs, ensemble methods (Majority Vote, UWO), and the main competitor (LASER).
    * The ablation studies are excellent, clearly demonstrating the individual contributions of the offline prior (BR vs. w/o offline) and the online adaptation (BR vs. w/o online).

* Insightful "Controlled Simulation": The analysis in Section 4.3 (Tables 3 and 6) is a standout. By using a dataset with ground-truth labels (RewardBench 2), the authors directly measure the router's annotation accuracy and show it correlates with downstream performance. This provides strong, direct evidence that the router is actually selecting the correct RM and that this is the source of the performance gains.

### Weaknesses
* Practical Complexity: The proposed system is significantly more complex to deploy than the baselines. It requires a full, separate pre-training stage to (1) run all $N$ RMs over a large offline dataset to create $\mathcal{D}_{beh}$, and (2) train the multi-task offline router. This represents a non-trivial barrier in terms of computation and engineering.
* Confounding Online Reward Signal: The reward signal for the online router, $\hat{r}_n^i$, is derived from the negative DPO loss, $-\mathcal{L}_{DPO}^i$. This signal seems potentially confounded. An RM that strongly challenges the current policy (e.g., by correctly identifying a "preferred" response that the policy currently disfavors) might induce a high DPO loss, resulting in a low reward for the bandit. Conversely, a weak or "yes-man" RM that agrees with the policy's (potentially flawed) preferences might get a low loss and a high reward. The paper does not fully justify why this heuristic signal is optimal, although the empirical results suggest it works.
* Scalability of the RM Pool: The main experiments use a pool of $N=4$ RMs, and the efficiency analysis uses $N=8$. It is unclear how the framework scales to a much larger pool (e.g., $N=50$). While $O(1)$ inference cost is maintained, the offline training (especially the BT head, which relies on disagreement pairs) and the online update of $N$ posteriors could become bottlenecks.

### Questions
1. As mentioned in the "Weak Points," the online reward $\hat{r}_n^i$ derived from $-\mathcal{L}_{DPO}^i$ seems confounded. Could a "good" but "difficult" preference from a high-quality RM be unfairly penalized with a high loss (low reward) simply because it is far from the current policy? Could you elaborate on the justification for this signal and discuss why it doesn't just reward RMs that are "easy" for the policy?
2. In Appendix C, you introduce a "Full Advantage" reward (based on comparing the selected RM to all other RMs) and show it outperforms the w/o offline variant. This seems like a more robust (though expensive) reward. How does this "Full Advantage" reward perform when combined with the full Bayesian Router (i.e., with the offline prior)? Is it possible that the main results are constrained by the heuristic reward, and not the router itself?
3. Could you provide more details on the practical overhead of the offline training stage? Specifically, what was the computational cost (e.g., in GPU-hours) to (a) generate the $\mathcal{D}_{beh}$ dataset by running all 4 RMs over the 50k offline pairs, and (b) train the offline router, relative to the cost of the main online DPO training run? This would help in assessing the method's practical utility.

### Soundness
3

### Presentation
3

### Contribution
2
