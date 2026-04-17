# Uncalibrated Reasoning: GRPO Induces Overconfidence for Stochastic Outcomes

- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Reinforcement learning (RL) has proven remarkably effective at improving the accuracy of language models in verifiable and deterministic domains like mathematics and coding. However, it is unclear if current RL methods are similarly effective at optimizing language models to reason about the probability of uncertain events from stochastic data, a valuable capability for decision-making and scientific discovery. Here, we demonstrate that Group Relative Policy Optimization (GRPO) induces highly overconfident probability predictions across three proper scoring rule rewards, while Proximal Policy Optimization (PPO) and REINFORCE Leave-One-Out (RLOO) yield well-calibrated models. We show that removing group standard normalization in GRPO fixes its miscalibration and provide a theoretical explanation for why GRPO's biased advantage estimate causes overconfidence. Our results demonstrate the negative impact of GRPO's standard normalization on probabilistic prediction and highlight an important design consideration for RL algorithms: while unbiased advantage estimates provide a consistent optimization signal across tasks, biased advantage estimates must be aligned with the structure of the target objective to be effective.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper claims that standard GRPO makes language models become overconfident, and compares with PPO and RLOO with several experiments. Emprically, it shows that on multiple datasets, GRPO creates overconfidence, while PPO, RLOO, and GRPO w/o std norm can keep model calibration. Theoretically, it shows that the expected average of standard GRPO is biased, and creates a positive feedback loop that pushes the model to become more extreme in its predictions.

### Strengths
- The paper shows a clear experiment result that GRPO causes model overconfidence on three different datasets.
- They show that a simple fix can be done by removing the standard‑normalization in GRPO.
- They also give a theoretical explanation that standard GRPO's advantage estimation biases updates toward overconfident predictors.

### Weaknesses
- The main problem is that this paper does not include a related work section. There is a lack of discussion of previous literature on model calibration, different post-training methods, etc.
- In fact, many previous papers have pointed out that other post-training methods (PPO, DPO) can also lead to model overconfidence, which conflicts with this paper's experiments. The authors do not discuss these relevant works and their connections with this paper.
- The experiment lacks enough generality. The authors only experiment with Qwen3-4B, without considering other architectures, such as OctoThinker based on Llama models. Dataset-wise, the authors only use three uncommon datasets for evaluation, without reporting on reasoning benchmarks such as MATH, AIME-24/25, etc.

### Questions
How sensitive are results to group size G? Your theoretical analysis assumes a large group size G, but you use a small G=4 in your CRISPR experiments.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors claim that Group Relative Policy Optimization (GRPO) produces overconfident probability estimates when trained with a log-likelihood reward, whereas PPO and RLOO yield well-calibrated predictions.

Experiments are conducted on synthetic probability prediction, a CRISPR biological dataset, and the MedMCQA medical question-answering dataset. Across all tasks, GRPO models show poor calibration (high ECE), even though classification accuracy remains similar. The authors attribute this issue to the group standard normalization in GRPO’s advantage estimate, arguing theoretically that it introduces policy-dependent bias amplifying overconfidence.

### Strengths
1. Interesting empirical observation. The finding that GRPO causes overconfidence in stochastic prediction tasks is novel and may interest researchers exploring calibration or uncertainty in RL algorithms.

2. Practical implication. The fix—removing standard normalization—is simple and easy to test. If correct, it could inform best practices for future reasoning RL work.

3. Link theory and practice. The discussion about group normalization introducing a bias in the advantage estimate points toward a principled mechanism, not just an empirical finding.

### Weaknesses
1. Limited originality and contribution. The main claim—removing normalization from GRPO improves behavior—closely parallels Dr. GRPO [1], which already proposed removing the same term. The paper essentially re-validates known ideas rather than introducing a distinct algorithmic or theoretical innovation. Calibration of GRPO is certainly a bug worth fixing, but the contribution is incremental—basically a note explaining why an already known fix works—rather than delivering a substantial new methodological or conceptual advance.

2. Shallow theoretical analysis. The “theoretical explanation” seems mostly heuristic: it sketches the bias term but does not mathematically prove the direction or magnitude of the effect. Many simplifying assumptions severely undermine rigor. 

3. Weak experimental design. The synthetic dataset is a toy-like one and nearly trivial; calibration differences there do not convincingly extend to real settings. The Qwen 3‑4B experiments use very small training sizes, few seeds, and no standard deviation reporting. Metrics are reported without statistical significance or uncertainty intervals. 

4. Writing and presentation issues. The paper is poorly organized, with the main content spanning only seven and a half pages; it is more like a lab note rather than a polished paper.

[1] Liu, Zichen, et al. "Understanding r1-zero-like training: A critical perspective." arXiv preprint arXiv:2503.20783 (2025).

### Questions
Please refer to the **Weaknesses** part. 

I am looking forward to the authors’ response, and may reconsider this paper based on that.

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
4

### Summary
The paper studies 3 types of RL algorithms, PPO, RLOO, and GRPO (with and without group std normalization), for uncertainty estimation in stochastic settings. By evaluating across 3 different scenarios, synthetic, scientific, and medical data, the paper shows that GRPO with group std normalization induces overconfident probability prediction for categorical stochastic outcomes. The paper also provides a theoretical explanation that the std normalization introduces a policy-dependent bias in the advantage estimation, which over-reinforces already confident predictions. A simple solution is to remove the std term.

### Strengths
- This paper is well-written and easy to follow. The motivation is well-supported by evidence, including visualizations of the miscalibration and explanations.
- The experimental setup is clear. Although the experiments are not large-scale RL, they cover 3 different datasets and clearly demonstrate the overconfidence phenomenon across different stochastic settings.

### Weaknesses
While I appreciate that the authors provide a new perspective on the impact of group std normalization from the lens of uncertainty and overconfidence, the theoretical discussion itself does not bring substantial new insights. Similar ideas, namely that group std normalization can lead to overconfidence, and the corresponding solution of removing the term have also been discussed in [1]. Furthermore, the role of group std normalization has been extensively discussed in prior work, such as Dr. GRPO [2] (discussed in the main text) and other related works [3, 4]. In addition, recent large-scale RL works have also moved away from using group std normalization [5, 6, 7], adopting batch-level normalization or removing the std term entirely. Although the motivations may differ, these existing words nonetheless limit the practical contribution of this paper.

[1] Outcome-based Reinforcement Learning to Predict the Future, arxiv 2025

[2] Understanding R1-Zero-Like Training: A Critical Perspective, COLM 2025

[3] GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning, arxiv 2025

[4] REINFORCE++: An Efficient RLHF Algorithm with Robustness to Both Prompt and Reward Models, arxiv 2025

[5] The Art of Scaling Reinforcement Learning Compute for LLMs, arxiv 2025

[6] Magistral, arxiv 2025

[7] Kimi k1.5: Scaling Reinforcement Learning with LLMs, arxiv 2025

### Questions
The main text frequently refers to the appendix for key information. Since there appears to be space left in the main body, it might be better to move some of those into the main text to improve readability.

Overall, while this is a clearly written paper that demonstrates a real and relevant phenomenon, its scope and theoretical discussion are quite narrow, focusing almost entirely on the group std term. Moreover, it does not provide a substantially new or meaningful solution, as simply removing the std term has already been widely adopted in recent RL works. Given the current state of the field, where this topic has already been extensively studied (albeit from different perspectives) and addressed, the paper feels somewhat outdated in scope and timing, and its contribution therefore appears incremental.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates how reinforcement learning methods for training reasoning language models behave when predicting stochastic outcomes, rather than deterministic domains like math. The authors compare GRPO, PPO, and RLOO across synthetic probability prediction tasks, biological perturbation data, and a medical multiple-choice QA dataset. They find that GRPO produces highly overconfident and poorly calibrated probability estimates, whereas other RL algorithms yield well-calibrated results. The paper attributes the miscalibration to GRPO’s group standard normalization step and demonstrates that removing this normalization substantially improves calibration. A theoretical analysis supports that standard normalization introduces a policy-dependent bias that reinforces overconfident predictions.

### Strengths
- The paper clearly identifies and isolates the role of group standard normalization in GRPO, providing both empirical and theoretical evidence that this design choice induces systematic overconfidence in stochastic decision settings.

- The experimental evaluation spans synthetic, biological, and clinical knowledge domains, demonstrating that the observed miscalibration behavior persists across qualitatively different tasks and data regimes.

- The theoretical explanation of how standard normalization biases the advantage estimate is well-motivated and aligns with the empirical findings, strengthening the validity and interpretability of the presented results.

### Weaknesses
- The empirical evaluation relies only on a single model (Qwen3-4B) across all experiments, which makes it difficult to determine whether the observed calibration differences generalize beyond this specific architecture and scale. Including additional models would substantially strengthen the empirical claims. Furthermore, some details of the experimental setup are under-specified in the main text.

- The model is required to generate explicit natural language tokens to represent  probability, rather than deriving numeric probabilities from token logits. The motivation for this design choice is not sufficiently discussed, and the direct generation approach may conflate reasoning about uncertainty with format adherence. Extracting probabilities from token logits is a common baseline in LM calibration and uncertainty estimation research, including this would provide more comprehensive results.

- The practical benefits of calibrated stochastic outcomes are not fully demonstrated in the presented tasks, since accuracy remains comparable across all methods. While the paper emphasizes calibration as the primary evaluation signal, it is not clearly shown how improved calibration leads to different or better decision-making in downstream applications. Stronger justification or concrete use cases illustrating how calibrated uncertainty materially improves task outcomes would help clarify the broader impact.

### Questions
- It would be helpful to elaborate in more detail what is meant by using the log-likelihood of the observed answer under the model’s predicted probability as the reward (Line 151~153). Since this is central to the RL setup, providing a mathematical expression for the reward function in the experiments section would make the methodology clearer for readers.

### Soundness
2

### Presentation
2

### Contribution
2
