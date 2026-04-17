# Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4, 6

## Abstract
Reinforcement Learning with verifiable rewards (RLVR) has emerged as a primary learning paradigm for enhancing the reasoning capabilities of multi-modal large language models (MLLMs). However, during RL training, the enormous state space of MLLM and sparse rewards often leads to entropy collapse, policy degradation, or over-exploitation of suboptimal behaviors. This necessitates an exploration strategy that maintains productive stochasticity while avoiding the drawbacks of uncontrolled random sampling, yielding inefficient exploration. In this paper, we propose CalibRL, a hybrid-policy RLVR framework that supports controllable exploration with expert guidance, enabled by two key mechanisms. First, a distribution-aware advantage weighting scales updates by group rareness to calibrate the distribution, therefore preserving exploration. Meanwhile, the asymmetric activation function (LeakyReLU) leverages the expert knowledge as a calibration baseline to moderate overconfident updates while preserving their corrective direction. CalibRL increases policy entropy in a guided manner and clarifies the target distribution by estimating the on-policy distribution through online sampling. Updates are driven by these informative behaviors, avoiding convergence to erroneous patterns. Importantly, these designs help alleviate the distributional mismatch between the model’s policy and expert trajectories, thereby achieving a more stable balance between exploration and exploitation. Extensive experiments across eight benchmarks, including both in-domain and out-of-domain settings, demonstrate consistent improvements, validating the effectiveness of our controllable hybrid-policy RLVR training. Code is available at https://github.com/zhh6425/CalibRL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge of entropy collapse and inefficient exploration in Reinforcement Learning with Verifiable Rewards (RLVR) for enhancing multi-modal large language model (MLLM) reasoning. Existing methods often suffer from policy degradation or over-exploitation, while hybrid approaches combining RL with supervised fine-tuning (SFT) can lead to distributional mismatch and accelerated entropy collapse. To overcome these issues, the authors propose CalibRL, a hybrid-policy RLVR framework that enables controllable exploration by treating expert demonstrations as a calibration baseline rather than a strict imitation target. CalibRL employs two key mechanisms: distribution-aware advantage weighting, which emphasizes rare yet informative responses to preserve exploration diversity , and an asymmetric LeakyReLU activation function that uses expert log-probabilities to moderate overconfident updates while amplifying underrepresented correct reasoning paths. By calibrating policy updates relative to expert knowledge, CalibRL aims to maintain productive policy entropy and guide exploration effectively, balancing exploration and exploitation. Extensive experiments on eight benchmarks demonstrate that CalibRL consistently outperforms standard GRPO and state-of-the-art hybrid-policy baselines in both in-domain and out-of-domain settings.

### Strengths
1. The paper addresses the crucial challenge of balancing exploration and exploitation within Reinforcement Learning with Verifiable Rewards (RLVR), a key paradigm for training reasoning capabilities in multi-modal large language models (MLLMs) which often suffers from entropy collapse.
2. The proposed CalibRL framework introduces intuitive mechanisms, namely distribution-aware advantage weighting and asymmetric activation using LeakyReLU calibrated by expert knowledge, which are conceptually straightforward to understand.
3. The effectiveness and generalization ability of CalibRL are rigorously validated through extensive experiments across eight diverse benchmarks, encompassing both in-domain and out-of-domain (OOD) reasoning tasks.

### Weaknesses
1. The comparison is limited to GRPO and a few hybrid methods (LUFFY, RL-PLUS), omitting recent advancements in GRPO variants like DAPO which could offer a more competitive benchmark .
2. While supplementary experiments are conducted on two additional models, the primary evaluation and analysis rely heavily on a single MLLM architecture (Qwen2.5-VL-7B), leaving the method's broad applicability across diverse MLLMs less certain .
3. The paper does not quantify the potential additional computational overhead introduced by calculating the log-probability gap relative to expert responses and applying the asymmetric activation and advantage weighting compared to the standard GRPO baseline.

### Questions
Please see paper weaknesses.

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
The paper proposes **CalibRL**, a hybrid-policy RLVR framework for MLLM reasoning that reformulates expert supervision as a distributional baseline rather than a hard imitation target. For each prompt, the method defines a log-likelihood gap $\Delta \mathcal{l}$ between the model’s on-policy trajectory and an expert trajectory, multiplies it by a correctness indicator $s_i$, and **passes $−s_i\Delta \mathcal{l}$ through a LeakyReLU gate**. The resulting exploration term is further scaled by the **group-normalized advantage magnitude $∣\hat{A}_i∣$** (interpreted as group-wise rarity), and added to a GRPO-style objective. Intuitively, CalibRL **reinforces underweighted correct** responses and **suppresses overconfident incorrect** ones while keeping policy entropy from collapsing. Across eight benchmarks (in-domain and OOD) and multiple backbones, the method reports consistent gains over GRPO and several hybrid-policy baselines, with ablations on $\alpha$ (LeakyReLU slope) and $\lambda$ (trade-off).

### Strengths
i. **Clear and practical idea:** Uses the expert as a relative reference to guide exploration instead of enforcing imitation; integrates into GRPO with minimal code changes.

ii. **Readable and structured:** Algorithm and ablation setups are clear and reproducible.

iii. **Well-motivated gating:** The four-quadrant behavior (boost underconfident correct, damp overconfident wrong) neatly matches RL intuition and helps preserve entropy.

iv. **Consistent empirical gains** across diverse reasoning benchmarks and model sizes; ablations on $\alpha$ and $\lambda$ illustrate controllability of exploration.

v. **Rare-event emphasis** via $∣\hat{A}_i∣$ often improves signal quality in sparse-reward, group-sampled settings.

### Weaknesses
i. **$∣\hat{A}_i∣$ is under-specified** near Eq. (9): the paper states it “captures group-wise rarity” but does not restate the exact computation (presumably GRPO’s group z-score) nor the degenerate case handling when the group reward variance is zero (all-correct or all-wrong).

ii. **Length bias in $\Delta \mathcal{l}$:** using sequence-level log-likelihood differences without explicit length normalization can penalize longer (yet correct) solutions; no targeted analysis is provided.

iii. **Sensitivity/fairness:** Results depend on group size $G$, sampling temperature/top-p, and the presence/absence of KL to a reference policy. These choices for baselines (e.g., LUFFY, RL-PLUS) are not exhaustively justified or tuned for parity; statistical significance over multiple seeds is limited.

iv. **Potential double-counting of advantage:** GRPO already weights updates by $∣\hat{A}_i∣$,$t$; the exploration term scales again by $∣\hat{A}_i∣$. Without gradient-norm breakdowns, it’s unclear whether this over-amplifies certain samples.

v. **Theory is light:** The claim that $∣\hat{A}_i∣$ encodes “rarity” is intuitive but lacks formal characterization (e.g., relation to group frequency, robustness to reward noise).

### Questions
1. Please **explicitly define $∣\hat{A}_i∣$** used in Eq. (9): is it exactly the GRPO group z-score at sequence level (shared across tokens)? Do you apply clipping or normalization? How do you handle **std = 0** groups (all-correct/all-wrong)?

2. How do you address **length effects** in $\Delta \mathcal{l}$? Any experiments with token-average log-prob or length-normalized variants, and their impact on long CoT tasks?

3. Does CalibRL remain effective if the **expert baseline is weaker** (e.g., a smaller SFT model)? Please report performance vs. strength of the expert.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
I am not an expert in this domain, I will try to review this paper from my limited understanding.

This work introduces CalibRL, a reinforcement learning framework that improves exploration in multi-modal large language models. Unlike prior RLVR methods prone to entropy collapse, CalibRL uses advantage weighting and a LeakyReLU-based asymmetric activation to guide exploration with expert supervision while preserving diversity. This balanced approach stabilizes training and enhances reasoning performance, achieving consistent gains over GRPO, LUFFY, and RL-PLUS across multiple benchmarks.

### Strengths
1. The paper is clearly written and easy to follow.
2. It addresses an important and widely existing problem, that under the SFT-then-RL paradigm, the policy becomes tightly anchored to the expert distribution during the SFT stage. This causes exploration to be restricted within the local neighborhood of expert behaviors, making it difficult to adapt to reward signals or discover more optimal reasoning trajectories.
3. The paper provides comprehensive and convincing ablation studies that support its claims.

### Weaknesses
1. Most evaluations are math benchmarks. The claim of general multi-modal reasoning would be more convincing with benchmarks involving richer visual, linguistic, or commonsense reasoning modalities (e.g., ScienceQA, MMMU, or multimodal dialogue tasks).

### Questions
1. The proposed method does not seem inherently limited to VLMs. It could also apply to LLMs. Why is the paper specifically focused on VLMs?
2. Why LeakyReLU-based activation yields controllable entropy growth? Do other similar activations have a similar effect?

### Soundness
3

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
4

### Summary
CalibRL reframes demonstrations as a calibration baseline inside RL with verifiable rewards, adding two simple mechanisms to a GRPO pipeline: a relative log‑probability gap between the on‑policy and expert responses passed through an asymmetric (LeakyReLU‑style) gate, and a group‑rarity weighting that scales updates by how uncommon a response is within a rollout group. The result is exploration that stays stochastic yet directed—rare correct behaviors are strengthened while overconfident mistakes are damped. Across eight visual‑math and multimodal reasoning benchmarks and several base models, the method consistently improves over a GRPO baseline and avoids the entropy collapse or aimless randomness observed in common SFT‑then‑RL and prior hybrid setups.

### Strengths
1. This proposed method introduces a clear hybrid‑policy objective that treats expert trajectories as a relative reference, which helps maintain entropy while steering updates toward verified behaviors. The design uses an intuitive pair of knobs—an asymmetric gate on the policy‑vs‑expert log‑prob gap and a group‑rarity magnitude—that integrates cleanly with GRPO and is easy to implement.

2. Simple, general mechanism: The LeakyReLU‑gated is an elegant way to use demonstrations for relative guidance. It respects the on‑policy signal and works as a plug‑in to GRPO.

3. Experiments demonstrate consistent gains on eight benchmarks, with especially large improvements on the hardest geometry split and solid out‑of‑domain boosts. The approach generalizes across different model sizes/architectures, and the ablations probe the important controls (gate slope and balance weight) to show how exploration is tuned.

### Weaknesses
1. GeoEval split clarity: The paper constructs GeoEval from validation failures of GPT‑4o CoT filtering, then reports it as a test benchmark with the largest deltas. Please clarify whether this split was ever used for hyper-parameter tuning or early stopping. If yes, results could be optimistically biased; if no, state this explicitly and detail safeguards.

2. Baselines for entropy control: Since the contribution is controllable exploration, it misses comparisons to standard entropy‑regularized GRPO (fixed/annealed entropy bonus) or token‑level entropy‑regularized updates. This would isolate whether your calibrated, expert‑relative update is better than “just add entropy.”

3. Ablation on the activation & reference. It would strengthen claims to compare LeakyReLU vs hinge, Huber, or sigmoid saturations, and to test “expert baseline” vs “reference policy baseline” (e.g., KL to SFT policy), to confirm that expert pairing and asymmetry specifically drive the effect.

4. Compute/efficiency reporting. Please quantify incremental overhead (extra forward passes to score expert trajectories), wall‑clock time vs GRPO, and sample efficiency (accuracy vs total rollouts). This is particularly important for practitioners training on LLMs / VLMs.

### Questions
1. Why do LUFFY/RL‑PLUS trail GRPO here? Any hints from your logs (entropy trends, clipping ratios, off‑policy mismatch) as to why these hybrids underperform? Sharing failure modes would be valuable to the community.

2. Calibration evidence. Beyond entropy curves, could you report ECE or other calibration diagnostics on answer probabilities, to substantiate “distributional calibration”?

3. Generalization beyond geometry. Have you tried training CalibRL outside geometry (e.g., coding or algebraic word problems) with appropriate verifiers (e.g., test oracles for code)? It would help demonstrate domain generality.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the RLVR algorithm. The authors start by analyzing the drawbacks of existing paradigms of incorporating expert data and reward feedbacks into foundation models, and then propse a novel algorithm, CalibRL, that alleviate the problems. Finally, they show via visual-reasoning tasks, that the proposed algorithm improves the performance of reinforcement learning.

### Strengths
1. The paper is well written and easy to follow. It provides a nice summary of existing methods as well.
2. The proposed algorithm is simple and novel.
3. The empirical performance is strong.

### Weaknesses
1. The algorithm seems to apply further beyond visual reasoning domain, but is not discussed.
2. The effectiveness of the algorithm is not verified for larger scales, like 30B. This seems too much to ask for though especially if the paper comes from academia.

### Questions
1. Does the algorithm apply to other areas, such as math reasoning (without multi-modality)?
2. In formula 9, why do we need to define another signal that labels the correctness? Cannot we just use a normalized version of the actual reward?

### Soundness
3

### Presentation
4

### Contribution
3
