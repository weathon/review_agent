# Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4

## Abstract
Masked diffusion models (MDMs) have recently emerged as a novel framework for language modeling. MDMs generate sentences by iteratively denoising masked sequences, filling in [MASK] tokens step by step. Although MDMs support any-order sampling, performance is highly sensitive to the choice of which position to unmask next. Prior work typically relies on rule-based schedules (e.g., max-confidence, max-margin), which provide ad hoc improvements. In contrast, we replace these heuristics with a learned scheduler. Specifically, we cast denoising as a KL–regularized Markov decision process (MDP) with an explicit reference policy and optimize a regularized objective that admits policy-improvement and convergence guarantees under standard assumptions. We prove that the optimized policy under this framework generates samples that more closely match the data distribution than heuristic schedules. Empirically, across four benchmarks, our learned policy consistently outperforms max-confidence: for example, on SUDOKU, where unmasking order is critical, it yields a 22% gain over random and a 12% gain over max-confidence.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents an algorithm to train a unmasking-position policy for masked diffusion models. While rule-based unmasking policies based on confidence or probability margin are popular,  the authors propose reinforcement learning (more concretely, GRPO) with KL-divergence regularization starting from such rule-based reference policies. The proposed algorithms have theoretical guarantees under technical assumptions, and show practical improvements on puzzle and math benchmarks.

### Strengths
Adding KL-divergence regularization in the RL of unmasking policy yields theoretical tractability and improvement in ablations. It also proposes a KL-divergence inequality regarding among the data distribution, reference policy, and optimized policy, which can be more informative than the reward inequality.

### Weaknesses
The weakness of the paper primarily lies in the mismatch of theory and practice.
- [W1] While the analyzed algorithm in Section 3.2-3.3 is based on 0-1 reward, it is totally different from what is described in Section D.2; the hitting rate in puzzle benchmarks, and more heuristic additional dense reward in math reasoning benchmarks. The readers might think that the practical performance is just based on the heuristic choise of rewards rather than theoretical insights.
- [W2] Theorem 2 suggests the KL divergence is closer (or equal to) the reference policy's KL divergence, but I feel that it only reflects the reward/rate of correct answer. For example, if there are multiple answers in each query, then there is a case where $L_{\text{MDM}}>0$ but $r_{\text{ref}}=1$, with the model only outputting a small subset of possible answers. In such a case, the policy is already converged, and the KL divergence is the same. Though this example does not violate the inequality (as it is not a strict inequality), the message after Theorem 2 (L281) "Theorem 2 guarantees the closer sampling to $p_{\text{data}}$ or ideal MDM $θ^*$ than $g_{\text{ref}}$" is somewhat misleading to me.
- [W3] In Theorem 2, the assumption $\text{supp}(p_{\theta*}(\cdot|q))\subseteq \text{supp}(p_{g\text{ref}}(\cdot|q))$ looks too strong, especially when considering the deterministic nature of the "confidence" unmasking policy.
- [W4] While $g_{\text{conf}}$ is deterministic, since you have a sampling randomness with $\pi_\theta$ in addition to the positional choice $g$, the sampling results should still be random. I am not sure why Max-Confidence is treated as if a constant in Figure 1, which might be misleading.
- [W5] The proofs are not written in a careful way. For example:
    - [W5-1] At least in Equations 30, 31, 36, 38, 39, 71, and 73, the parentheses are not appropriately closed. Some are just trivial mistakes but some make the equations ambiguous. At this rate of obvious mistakes left, I'm not sure if the draft is appropriately checked by the authors themselves.
    - [W5-2] In Eq. 82, why is it a strict inequality rather than "$\leq$"?
    - [W5-3] The equality from Eq. 92 to Eq. 93 is valid only when $p_{g\phi}$ is abosolute continuous w.r.t. $p_{g\phi\text{old}}$. Is it an additional assumption?

While the proposed method consitently improves upon existing methods, the current manuscript has a large room for improvement as discussed above, which makes me lean to rejection.

Typos:
- L95: $\sum T_{x'\in X}$ -> $\sum_{x'\in X}T$
- L888: {1, L} -> {1, ..., L}?
- L937: Let let -> Let

### Questions
In addition to the points raised in the Weaknesses section, I have the following question:
- [Q1] As the authors write in +, Huang et al. (2025) applies RL jointly to the masked diffusion model and unmasking policy so the existing method is heavy. Doesn't it mean that, by fixing the diffusion model, the existing method can train only the policy model for lighter computational cost? Can the authors discuss a bit more on the difference and the preposed method's original contribution? (I assume it is basically about initialization and KL regularization though.)

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper explores learning the unmasking order in MDMs. People have realized that a pretrained MDM has an any-order sampling property, which led them to explore heuristic unmasking policies. These use the information of per-index logits, so they don't fully leverage the information and also rely on heuristics. This work focuses on learning the unmasking policy in fine-tuning the LLaDA using RL methods, achieving better performance than baselines.

### Strengths
First of all, this paper is clearly well-written. The authors explain the background clearly and then the problem that they want to solve in the paper. Also, examples in Sections 2 and 3 assist the reader in grasping the context better. Moreover, the experiment design (loss design, architectural modification) is really clear to follow. I believe this paper takes a step toward finding an effective unmasking strategy beyond prior heuristic strategies.

### Weaknesses
Perhaps the most significant weakness of this paper is the less-than-ideal experimental results: We can see that on several tasks, the performance of the trained model's sampling strategy is comparable to the Margin sampling strategy from (Kim et al, 2025). Given that the method proposed in the paper requires additional training (while the Margin strategy is training-free), it might be questionable to some people that if we really need an approach that requires training.

### Questions
- In Figure 1, we can see that the Pass@N performance can actually go far beyond the Max-Confidence baseline, indicating that there's indeed a decent sampling path (that has a much higher accuracy even than Max-Confidence). The results in the paper, however, aren't that significant. I'm curious how the authors could discuss the discrepancy, i.e., what was the main factor that reduced this expected huge gap?

- I wondered what the authors meant by `Meanwhile, on GSM8K, the much larger search space makes Top-K preferable for stability. In future work, we plan to develop an unmasking policy optimization algorithm that explores the full mask tokens even when the search
space is large.'

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method to improve the token generation order of pretrained Masked Diffusion Models (MDMs) through RL. Specifically, the authors train a learned unmasking scheduler using GRPO. Compared to heuristic orderings such as confidence-based or margin-based selection, the proposed approach achieves better performance across several benchmarks.

### Strengths
**Novelty:**
The idea of learning the generation order of a non-autoregressive model through reinforcement learning has been previously explored in classical non-autoregressive language models (see [1]). However, applying this idea to **modern diffusion-based generation models** represents an original and meaningful contribution.

**[1]** Insertion-based Decoding with Automatically Inferred Generation Order, [arXiv:1902.01370](https://arxiv.org/abs/1902.01370)

### Weaknesses
**Generalizability of the Method:**
Since the optimization is based on RLVR (GRPO), the method can only be applied when a **verifiable reward signal** is available. This limits its general applicability to domains where rewards can be explicitly computed.

**Suggestion:**
Could the authors explore the use of a **preference reward model** to enable training in more general settings? Alternatively, using the **generative perplexity** of a strong language model such as GPT-2 Large as a reward could be an interesting direction for learning generation order without explicit ground-truth supervision.

**Incremental Performance Gain:**
Despite requiring additional RL-based training, the performance improvements over strong baselines appear modest—except for the SUDOKU task, where the generation order plays a particularly critical role. The paper would benefit from discussing why the proposed method does not yield stronger improvements on other benchmarks.

### Questions
* **Pass@K:**
  Could the authors report *pass@K* results? It would be informative to know whether the proposed method sacrifices diversity (e.g., generating overly similar samples) in pursuit of higher accuracy.

* **NFE (Number of Function Evaluations):**
  How many diffusion reverse steps are used during inference? Please specify the number of denoising iterations to facilitate fair comparison with other methods.

* **Qualitative Analysis:**
  It would be helpful to visualize the learned unmasking order on SUDOKU and compare it to existing baselines such as max-margin or max-confidence. A qualitative comparison could clarify what the learned policy actually discovers beyond heuristics.

### Soundness
4

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes learning an unmasking position-selection policy for masked diffusion language models (MDMs) by casting denoising as a KL-regularized Markov Decision Process and optimizing a GRPO-style objective against a strong reference scheduler (e.g., max-confidence or Top-K). The authors prove (i) fixed-point convergence that improves expected reward over the reference and (ii) a “reference-KL tightening” result showing the terminal output distribution becomes closer to the data than that induced by the reference policy [Thm. 1–2]. To make training tractable, they derive a surrogate token-level objective (LUPO) whose gradient aligns with the intractable output-level objective and instantiate realizations for different references [Eq. (19), Table 1]. Empirically, a lightweight policy model (≈134M) plugged into a frozen 8B MDM improves accuracy across SUDOKU, ZEBRA, GSM8K, and MATH500 over random, margin, entropy, and max-confidence schedules, and adds gains on top of diffu-GRPO post-training.

### Strengths
- Clear problem focus: unmasking order materially affects discrete diffusion inference; paper formalizes learning this policy rather than relying on heuristics.

- Theoretical contributions: output-level GRPO formulation with stated convergence and a KL tightening guarantee relative to a reference policy; proofs sketched in main text and detailed in Appendix C.

- Practical surrogate: token-level objective with clipping and an empirical divergence estimator, plus realizations for max-confidence/softmax/Top-K with explicit training recipe.

- Solid empirical signal: consistent wins vs. strong baselines on four benchmarks (e.g., SUDOKU +11.2% over max-confidence; GSM8K +1.9%) and compatibility with diffu-GRPO (GSM8K +1.3% over max-confidence).

- Efficient design: trains a small policy while keeping the 8B MDM frozen; memory-efficient algorithm described (Algorithm 2).

### Weaknesses
- Sec. 2 reverse-process mismatch is under-explained. The section claims that there exist states where the true reverse process and the model’s reverse process diverge, but the accompanying example is hard to follow and it’s unclear what concrete training pathology this implies. Specifically, how this mismatch arises under the stated pretraining objective, how it manifests during inference-time unmasking (e.g., biased token selection vs. compounding calibration error), and whether it is an artifact of modeling choices (masking schedule, likelihood factorization) are not clearly connected.

- Evaluation breadth. Benchmarks are primarily small-format puzzles and math QA; no natural-language generation or long-form tasks to test the generality of unmasking policy beyond structured reasoning.

### Questions
1. In Sec. 2, please (i) formalize the exact training objective under which the “true vs. model reverse-process” divergence occurs, (ii) explain the mechanism by which this leads to inference-time errors in the unmasking policy (with a minimal worked example), and (iii) discuss whether the issue can be mitigated during pretraining (e.g., via alternative noise schedules, calibration terms, or reverse-process consistency regularization) rather than deferred to policy learning at inference.

2. Does the learned policy transfer across MDM backbones (e.g., Dream-7B), or between tasks without re-training? Any negative transfer?

### Soundness
3

### Presentation
3

### Contribution
2
