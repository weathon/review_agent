# LaSeR: Reinforcement Learning with Last-Token Self-Rewarding

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4, 8

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a core paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs). To address the lack of verification signals at test time after RLVR, prior studies incorporate the training of model's self-verification capabilities into the standard RLVR process, thereby unifying reasoning and verification capabilities within a single LLM. However, previous practice requires the LLM to sequentially generate solutions and self-verifications using two separate prompt templates, which doubles the inference cost per sample and significantly reduces efficiency. In this work, we theoretically reveal that the closed-form solution to the RL objective of self-verification training can be approximately reduced to a remarkably simple form: the true reasoning reward of a solution is equal to its last-token self-rewarding score, which is computed as the difference between the policy model's next-token log-probability assigned to any pre-specified token at the solution's last token and a pre-calculated constant, scaled by the KL coefficient. Based on this insight, we propose LaSeR (Reinforcement Learning with Last-Token Self-Rewarding), an algorithm that simply augments the original RLVR loss with a Mean Squared Error (MSE) loss that aligns the last-token self-rewarding scores with the verifier-based reasoning rewards, and jointly optimizes the reasoning and self-rewarding capabilities of LLMs. The optimized self-rewarding scores serve as auxiliary reward signals in both training and testing to enhance model performance. Notably, our algorithm derives these scores from the predicted next-token probability distribution of the last solution token immediately after solution generation, thereby incurring only the minimal extra cost of at most one additional token inference. Experimental results show that our method not only improves the reasoning performance of the model also equips it with remarkable self-rewarding capability, thereby further boosting its inference-time scaling performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a self-verification calculation method. It expands on the implicit reward and provides an expression through approximation to eliminate the expensive partition function evaluation, enabling the model to estimate the reward through a simple single token probability. The derived expression simplifies reward estimation and allows the model to self-evaluate its reasoning confidence. Empirical results show improvements on mathematical reasoning benchmarks, particularly in F1 scores for self-verification tasks.

### Strengths
1. Approximating implicit rewards through single-token probability is elegant and efficient. Theoretically it is novel. The derivation can be extended to future works for self-verification since it is simple enough to apply.

2. There are empirical performance improvements in self-verification F1 scores on math reasoning tasks. Table 1 F1 scores show clear improvements over the self-verification.

### Weaknesses
1. The key observation of negligible $log Z$ does not have a strict guarantee that it will be approximately 0 when the model weights are changed. It would be beneficial to add additional analysis like Figure 4 for the post training model.

2. It is unclear why the self-reward is added on top of the true reward instead of being used directly for optimization. This design choice should be justified. The claim could be stronger to have an ablation study with only self-verification as the reward signal, separating it out from GRPO advantage.

3. Equation 9 would benefit to have more explanations. 

4. Appendix F appears to evaluate only on the model’s own outputs. It would be informative to test cross-model generalization, for example, using Qwen2.5-7B-Base to verify outputs from Open-Reasoner-Zero-7B.

### Questions
n/a

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
3

### Summary
LaSeR proposes a way to make LLMs “self-reward” by showing that, under its RL objective, the verifier reward can be approximated by a last-token self-rewarding score, the KL-scaled difference between the policy’s next-token log-probability on a pre-specified token at the final solution token and a precomputed constant. It augments RLVR with a simple MSE loss aligning this score to verifier rewards and then uses the learned score as an auxiliary signal at train and test time. The experiment shows this method can improve model reasoning ability and self-verification ability for test-time scaling.

### Strengths
- Detailed analysis of the proposed approach, combining both theoretical and empirical analysis.
- LaSeR is lightweight & efficient. Adding an MSE term aligning “last-token self-reward” to verifier rewards; computing the score costs at most one extra token. 
- Weighting majority-vote by the learned self-reward scores improves solution selection at test time.

### Weaknesses
- Potential issue in the partition function expansion (Eq. 9)
  - The paper defines the verification reward as **one-hot** over the *ground-truth* label: reward = 1 **only** when the predicted token equals the correct label (e.g., ($z_+\in{z_c,z_i}$) depending on the task’s truth), and 0 otherwise (Eq. 7). 
  - However, when expanding the partition function (Z(x,y)) in Eq. (8), Eq. (9), it adds both $(z_c)$ and $(z_i)$ terms with the factor $(\exp(1/\beta_v))$, treating *both* as rewarded. I think only the *true* label ($z_+$) should carry reward 1 while the other label $(z_-)$ gets reward 0. Then
 $  Z(x,y)=\sum_{z\notin{z_c,z_i}}\pi_{\text{ref}}(z|x,y) e^{0}+\pi_{\text{ref}}(z_+|x,y)e^{1/\beta_v}+\pi_{\text{ref}}(z_-|x,y) e^{0}
  =1+\pi_{\text{ref}}(z_+|x,y)\big(e^{1/\beta_v}-1\big)$
- 0.5 threshold on a non-probability score (L318). The score ($r_s=\beta_v\log\pi_\theta(z_c\mid x,y)-\beta_v c_{\text{ref}}$) is a scaled/shifted log-probability (unbounded, not a probability). Thresholding it at **0.5** is arbitrary—changing ($\beta_v$) or ($c_{\text{ref}}$) shifts the decision boundary. It also ignores the competing label ($z_i$) even though verification is binary.
- The experiment is limited. Only base model and GRPO compared, and some setup (Open-Reasoner-Zero-7B x AIME25) shows limited advantage or is just not better than baseline at all. As the margin is so small, the statistical interval should be included.

### Questions
- As the primary focus is to integrate with GRPO using verifiable reward, why MSE loss is used for Eq. 12 instead of BCE loss?
- How is the weighted majority voting for RM@K?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents **LaSeR**, a reinforcement learning algorithm that augments the standard RLVR objective with an additional Mean Squared Error (MSE) loss aligning last-token self-reward scores with verifier-based rewards, enabling joint optimization of reasoning and self-verification. By approximating the reference log-probability as a constant, applying class-balanced re-weighting, and integrating the self-reward into advantage estimation, LaSeR provides an efficient improvement over vanilla GRPO with minimal computational overhead. Experiments on multiple mathematical reasoning benchmarks and three base models show consistent gains in both reasoning accuracy and self-verification performance, as well as the effectiveness of self-verification on test-time scaling and its generalization to other domains.

### Strengths
- **Theoretical grounding and originality:** The derivation of the self-reward term is well-grounded in established RL literature, with a clear and rigorous justification for approximating the partition term as zero using both theoretical analysis and empirical evidence. While related works exist, LaSeR presents a coherent and novel formulation of self-verification that links last-token modeling with verifier-based rewards.
- **Efficiency and practicality:** LaSeR introduces minimal training and inference overhead compared to prior self-verification or RLVR methods. The approximation of the reference log-probability as a constant further simplifies implementation without sacrificing performance, making the approach highly practical for large-scale use. Additional training techniques grounded in empirical observation further enhance the method’s engineering value.
- **Comprehensive experimentation:** The study conducts extensive experiments across three base models and five mathematical reasoning benchmarks, with an additional evaluation on a science dataset. Detailed ablations on hyperparameters and the constant reference log-probability assumption validate the design choices. Together, these experiments strengthen the reliability of the conclusions.

### Weaknesses
- **[Motivation]** The motivation could be more clearly articulated. The paper frames LaSeR mainly as an efficiency improvement over existing joint reasoning–verification training paradigms but does not sufficiently justify *why* integrating these two abilities within a single model is necessary, instead of using a separate generator–verifier system. A clearer discussion of the conceptual or practical advantages of joint training would strengthen the motivation and emphasize the broader significance of the approach.
- **[Significance]** The reasoning accuracy gains over the GRPO baseline are modest, and the performance difference between LaSeR and LaSeR w/ SWA is small. This limits the overall perceived significance and makes it difficult to assess the impact of the SWA module.
- **[Experimentation]** The experimental analysis is somewhat limited. For instance, the generalization evaluation on GPQA reports only inference-time scaling results, without presenting zero-shot reasoning or verification performance, making it difficult to fully assess out-of-domain robustness. The comparison between SFT and MSE reward curves also omits the final performance results, leaving the claimed advantage of MSE insufficiently supported. Furthermore, the paper does not investigate why reasoning ability improves under LaSeR or whether the auxiliary objective influences the model’s reasoning dynamics or solution style. A deeper analysis in these aspects would offer clearer insight into the mechanisms driving the observed improvements.
- **[Presentation]** Section 3 could be more condensed, as some of the detailed derivations and techniques may be moved to the appendix to improve readability and maintain focus on the key contributions.

### Questions
- How is the weighted majority voting implemented?
- How are the special tokens chosen for representing correctness (e.g., `<vision_start>`)? Are they selected randomly?
- Figure 6 shows that many correct solutions have self-reward scores below 0.5, even though the mean score for correct cases is higher than for incorrect ones. Does this suggest that the self-reward signal works better for re-ranking rather than binary verification? It would be helpful if the authors could clarify this behavior or provide more results regarding this point.

### Soundness
3

### Presentation
2

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
The paper proposes LaSeR, a novel policy optimization method that enables a model to self-verify its own reasoning outputs efficiently by leveraging the next-token probability of a pre-specified special token as a self-rewarding score, thereby improving test-time scaling. The authors report improved reasoning accuracy, self-verification F1, and test-time scaling efficiency across multiple reasoning benchmarks.

### Strengths
1. The paper is well-written and easy to follow, with clear theoretical motivation and solid empirical validation.

2. Extensive experiments across multiple reasoning benchmarks and ablation studies demonstrate robustness

3. The proposed idea of last-token self-rewarding is conceptually simple but empirically effective.

### Weaknesses
1. The proposed idea overlaps with prior work such as ReVISE [1], which also learns to self-verify using a special-token confidence and applies it for weighted voting at test time. 

2. The comparison with external verifiers (in Appendix F) may not be fair without training the verifier on the same dataset and backbone.

3. Although several components are introduced (re-weighting, warm-ups, advantage integration), ablation studies are insufficient to isolate their effects.

4. Evaluation is limited to math reasoning; generalization to other domains (code [2,3], general-reasoning [4]) is unclear.
---
[1] Lee et al., Revise: Learning to refine at test-time via intrinsic self-verification, ICML 2025\
[2] Austin et al., MBPP: Program Synthesis with Large Language Models\
[3] HumanEval:Evaluating Large Language Models Trained on Code\
[4] AlpacaEval: An Automatic Evaluator of Instruction-following Models

### Questions
1. How does LaSeR perform under Best-of-N [5] evaluation without weighting?

2. What is the actual inference-time gain (in token count or latency) compared to prior self-verification approaches?
---
[5] Snell et al., Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
LaSeR trains a policy to encode a self‑evaluation signal in the final step of its answer: a last‑token score that tracks whether the just‑generated solution is correct. Instead of running a separate verifier sequence, the method adds a tiny auxiliary loss so that this one‑token score matches a standard verifier’s binary reward during RLVR. Inference then needs only the policy’s next‑token distribution at the last position, which also supports inexpensive weighted voting across samples. Experiments on several math benchmarks show small but consistent accuracy gains over GRPO baselines and strong self‑verification F1.

### Strengths
1. Elegant and practical idea. This method compresses self‑verification into a single last‑token readout, avoiding a second generation pass and adding almost no training or inference cost.

2. Near‑zero overhead: this method requires no second pass, just a next‑token query; training-time cost is essentially free since token log‑probs are already computed.

3. Empirical Justification: The method improves Pass@1 across multiple backbones and yields high self‑verification F1; the score boosts inference‑time scaling via simple weighted voting.

4. The argument about length bias in cumulative implicit rewards is valid and demonstrated (Appendix B), aligning with new empirical results suggesting shorter chains often outperform longer ones.

### Weaknesses
1. Limited breadth: The evaluation centers on math; results on a general reasoning set show only modest separation. More domains (code, QA, open‑ended tasks) would strengthen the case.

2. Calibration analysis of self-rewarding: the reward model here is essentially a binary classification model. F1 is a good metric but not comprehensive enough. It would strengthen the paper if the authors could include reliability diagrams / ECE / AUROC for the last‑token score vs. correctness, and analyze length effects (short vs. long CoT) in line with recent findings that shorter chains can be preferable.

3. This work misses discussions of prior RLHF (not RLVR) papers that incorporates reward signals explicitly in RL training.

### Questions
1. Reference‑free test time: How stable is the pre‑computed constant across model updates or decoding settings? Would a quick online calibration (or an ensemble of special tokens) reduce drift?

### Soundness
3

### Presentation
3

### Contribution
3
