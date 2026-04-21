Now I have all the information needed to write a comprehensive meta-review. Let me calibrate the score based on the anchors.

## Summary

The paper proposes **label annealing**, a method that mitigates catastrophic forgetting during LLM finetuning by adding a KL divergence penalty between the finetuned model's predictions and those of a frozen copy of the initial model on the finetuning data. The method requires only the model weights (no pretraining data). Experiments on Llama 3 8B across math finetuning, code finetuning, instruction tuning, and niche domain adaptation show label annealing can reduce forgetting on source benchmarks while preserving most target-domain gains. The paper also provides a theoretical analysis in over-parameterized linear regression, showing label annealing preserves pretrained knowledge both within and orthogonal to the finetuning data span.

## Strengths

- **Clear empirical forgetting mitigation on math and code finetuning (Tables 1–2):** In math finetuning, label annealing restores TriviaQA from 53.80 (direct finetuning) to 65.87 (vs. 67.99 base), while maintaining target gains (MATH 17.94 vs. 17.10 direct; GSM8K 61.78 vs. 62.01 direct). In code finetuning, it recovers MATH from 1.19 to 17.16 (near baseline 15.92) while preserving most HumanEval gains (51.06 vs. 54.53 direct). These are practically meaningful results.

- **Principled evaluation framework:** The target/source benchmark decomposition provides a clean methodology for evaluating both learning and forgetting simultaneously, and the paper evaluates across four distinct finetuning scenarios (math, code, instruction tuning, niche domain).

- **Honest presentation including tradeoff scenarios:** Section 3.3 shows cases where label annealing produces a smooth tradeoff rather than a free lunch (Figure 2), and Table 3 includes unfavorable comparisons with replay.

- **Theoretical analysis provides geometric insight (Theorem 1, Section 4.3):** The decomposition showing that direct finetuning discards pretrained knowledge within the finetuning data span while label annealing interpolates between old and new knowledge within that span offers meaningful intuition, even if limited to linear models.

## Weaknesses

### Fatal
None.

### Major

- **Missing comparison with LoRA and other PEFT methods:** LoRA (Hu et al., 2022) is the most widely used approach for finetuning LLMs while constraining weight drift, yet the paper does not compare against it. LoRA structurally prevents weight drift by freezing the pretrained weights, directly addressing the problem this paper targets. Other continual learning baselines (EWC, layer freezing) are also absent. Without comparison to these standard approaches, the paper cannot establish that label annealing offers anything beyond what practitioners already use. This gap is especially notable because the paper's own scope—"methods that require only access to the weights"—includes LoRA.

- **Limited novelty of the core method:** Label annealing is KL-regularized finetuning against a frozen reference model. This technique is well-established: it is the KL penalty used in RLHF/PPO (Schulman et al., 2017; Ouyang et al., 2022), and it appears in knowledge distillation literature. The paper acknowledges connections to distillation (Section 1.1) but frames the goal as different (forgetting vs. compression). While the application context is valid, the core technique itself—adding KL divergence toward a frozen copy of the initial model—is not new regardless of the stated goal. The paper would benefit from more precisely articulating what aspects (if any) are novel beyond the application context.

- **Table 3 substantially undermines the core motivation:** The paper's stated motivation (Section 1) is that pretraining data is unavailable for open-weight models, "necessitating" methods requiring only weights. Yet Table 3 shows that approximate replay using RedPajama (a public reconstruction of Llama training data) outperforms label annealing on target benchmarks (MATH: 22.40 vs. 17.94; GSM8K: 69.52 vs. 61.78) while also mitigating forgetting. The paper's response—that data reconstruction becomes harder as training strategies get more complex—is speculative and not empirically supported; indeed, Table 3 itself shows even a rough approximation works remarkably well. The claim that label annealing is "necessary" because replay is infeasible is directly contradicted by the paper's own evidence. The paper's contribution would be stronger if it honestly acknowledged replay as a strong alternative and positioned label annealing more precisely (e.g., as a complementary method or as useful when no approximate data is available).

### Minor

- **Loose connection between theory and method:** Section 4 analyzes label annealing in linear regression by replacing KL divergence on softmax distributions with L2 loss on pre-softmax linear outputs (Equation 8), discarding the softmax nonlinearity, calibration effects, temperature focusing on high-probability tokens, and KL asymmetry. The "orthogonal complement" geometric intuition derived from this simplification may not transfer to transformers. The paper acknowledges this simplification (Section 4.2: "If we simplify the transformer neural network to a single linear layer... If we again simplify the KL divergence penalty with L2 loss"), but no empirical validation confirms the linear model's qualitative predictions hold for the neural network setting.

- **No ablation on temperature T:** The temperature parameter T is introduced as a key component of the method (Equation 2) but no ablation study examines its effect. The main experiments appear to use a single T value selected via the same sweep procedure as λ. It is unclear whether T>1 provides benefit over the simpler T=1.

- **No variance estimates:** Tables 1–2 report single numbers without standard deviations or confidence intervals. Given that HumanEval is known to have high variance and some differences (e.g., MMLU 64.62 vs. 62.54) are modest, statistical significance cannot be assessed.

- **Misleading naming:** The method is called "label annealing" but uses a fixed λ—there is no annealing schedule. The name suggests a dynamic process that does not exist in the proposed method.

### Trivial
None.

## Nice-to-Haves

- Combine label annealing with LoRA to test whether they are complementary (function-space vs. parameter-space regularization).
- Include continual learning baselines (EWC, SI, MAS) that also require only model weights.
- Investigate the MATH 15.92→1.19 collapse under code finetuning—is this truly forgetting or a formatting/prompt issue?

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Replay is impossible" strawman (Harsh Critic #1 partial):** The harsh critic claimed the paper states replay is "impossible." The paper actually says "making it impossible to mix the finetuning data with the original training data," which is a statement about mixing with the *original* data, not about approximate replay. However, the paper does overstate the necessity of label annealing given Table 3's results—this concern is kept in Major weaknesses above but with the correct framing.

- **Asymmetric hyperparameter selection (Harsh Critic):** The claim that label annealing gets an unfair advantage by sweeping over (λ, T) while L2 only sweeps λ. This is a minor concern—both methods select the best hyperparameter configuration, and a 2D sweep over (λ, T) is not dramatically more expressive than sweeping λ alone when the selection criterion is fixed. Moved to a weakened form in Minor.

- **Formatting issues in equations/tables:** These are parser artifacts, per hard rules.

- **Missing references to specific prior work (Harsh Critic):** Per hard rules, I cannot verify whether specific cited works exist, so these are removed.

- **Strength about "minimal implementation overhead" (Strength Finder):** This is a generic strength that applies to any simple method. While true, it's not a distinguishing contribution. Removed.

## Novel Insights

The Theorem 1 decomposition revealing that direct finetuning in the over-parameterized linear regime discards pretrained knowledge *within* the finetuning data span (not just in orthogonal directions) is a genuine and non-obvious insight. This provides a crisp explanation for why L2 regularization toward initialization fails in the LLM setting—L2 regularization has no clean geometric separation between old and new knowledge in the data span, while label annealing achieves a convex interpolation. This insight could inform future work irrespective of the method's novelty concerns.

## Suggestions

- Add LoRA as a baseline across all experiments. Even if label annealing is complementary (operating in function space vs. parameter space), showing how they compare—and potentially combine—is essential for practitioners.
- Reposition Table 3 results from the limitations section to the main experiments and discuss replay honestly as a strong alternative, with label annealing positioned as useful when approximate pretraining data is unavailable or insufficient.
- Add an ablation on temperature T to clarify its role; if T=1 suffices, the method is even simpler; if T>1 helps, this is an additional contribution worth highlighting.

## Calibration

**Anchors used for scoring:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| LDIFS (VQ7Q6qdp0P) | VQ7Q6qdp0P | 4.75 | Most similar paper—simple distillation-based regularization for finetuning forgetting. Rejected due to novelty concerns and missing baselines. This paper has similar novelty issues but better theory. |
| HFT (WzgcreQaNV) | WzgcreQaNV | 4.00 | Freezing half parameters to prevent forgetting. Similar "simple method" positioning. Rejected due to lack of novelty and missing LoRA comparison. |
| LiNeS (J5sUOvlLbQ) | J5sUOvlLbQ | 5.75 | Layer-increasing scaling to prevent forgetting. Accepted poster despite simple method, but had broader baselines and experiments. |
| Dissecting forgetting (tmsqb6WpLz) | tmsqb6WpLz | 5.75 | Empirical analysis of forgetting, accepted poster. |
| FV-guided regularization (gc8QAQfXv6) | gc8QAQfXv6 | 9.00 | Novel function-vector guided method for LLM forgetting, oral. Far stronger novelty and depth. |
| GradSimCore (cHy00K3Och) | cHy00K3Och | 2.50 | Essentially a baseline method with overclaimed novelty, rejected. |

This paper falls between the LDIFS/HFT anchors (4.0–4.75, rejected with similar novelty concerns) and the LiNeS anchor (5.75, accepted poster with broader evaluation). The key differences from LDIFS that push this slightly higher are: (1) the theoretical analysis in Section 4 provides genuine geometric insight, (2) the experimental scope is broader (4 finetuning scenarios), and (3) the results are clearly presented with source/target decomposition. However, the missing LoRA baseline, limited novelty, and self-undermining Table 3 prevent a higher score.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>