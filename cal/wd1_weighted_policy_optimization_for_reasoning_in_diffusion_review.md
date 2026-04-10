=== CALIBRATION EXAMPLE 11 ===

# Final Consolidated Review
## Summary
This paper proposes *wd1*, a reinforcement learning method for diffusion-based large language models (dLLMs) that replaces policy‑ratio estimation with a weighted log‑likelihood objective. By avoiding the approximation of multiple policy likelihoods, *wd1* reduces computational overhead and mitigates error amplification. The authors provide a theoretical interpretation of *wd1* as energy‑guided discrete diffusion training combined with negative‑sample unlearning, and extend the method to *wd1++* which leverages intermediate denoising steps. Experiments on LLaDA‑8B show substantial gains over the baseline *d1* on reasoning benchmarks (e.g., +58.8% on Sudoku) and state‑of‑the‑art results on MATH500 and GSM8K with few training steps.

## Strengths
- **Novel, ratio‑free formulation**: The weighted log‑likelihood objective eliminates the need to compute policy ratios, which for dLLMs can exponentially amplify likelihood‑approximation errors (Sec. 3.1, Eq. 15‑16). This is a direct and well‑motivated solution to a known bottleneck in diffusion‑based RL.
- **Strong empirical gains and efficiency**: Without supervised fine‑tuning, *wd1* outperforms the baseline *d1* by large margins on planning tasks (Sudoku, Countdown) and matches or exceeds it on math reasoning (Table 1). The method also reduces per‑step FLOPs and function evaluations (Table 2). The extended *wd1++* achieves competitive state‑of‑the‑art results with far fewer rollouts (Table 3).
- **Theoretical grounding and interpretation**: The paper formally derives the objective from reverse‑KL regularized policy optimization (Theorem 2) and provides a novel interpretation as energy‑guided diffusion training combined with negative‑sample unlearning (Theorem 1, Remarks 1‑2), connecting the method to established diffusion concepts.
- **Thorough ablation studies**: Ablations validate the importance of both positive and negative weighting (Table 4, Table 9) and show that the balanced combination is most effective (Fig. 4). The study also examines the effect of the weight‑scaling parameter ψ (Fig. 4).

## Weaknesses
### Major
- **Uncertainty in baseline comparison**: The reproduced *d1* baseline underperforms the numbers reported in the original paper on several tasks (Table 7). While the paper still shows large improvements over the original *d1* on Sudoku and Countdown, the discrepancy raises questions about the magnitude of gains on other tasks (e.g., GSM8K, MATH) and the robustness of the comparison.
- **Insufficient details for state‑of‑the‑art claims**: The comparison with concurrent methods (Table 3) lacks critical details—e.g., whether *wd1++* uses full fine‑tuning or LoRA, the exact dataset sizes, and hyperparameter settings—making it difficult to assess the fairness and significance of the claimed state‑of‑the‑art results.
- **Reliance on a biased likelihood approximator**: *wd1* uses the same biased, single‑timestep likelihood approximation as *d1* (t=1). Although the ratio‑free objective reduces error amplification, the underlying bias remains. The paper does not ablate the choice of approximator, leaving open whether the benefits stem from the weighted objective or from this particular biased estimator.

### Minor
- **Limited evaluation scope**: Experiments are confined to mathematical reasoning and puzzle‑solving tasks (GSM8K, MATH, Sudoku, Countdown), with only a brief coding experiment in the appendix. Broader evaluation on diverse language generation tasks (e.g., dialogue, summarization) would better demonstrate general applicability.
- **Incomplete handling of degenerate reward cases**: The method halts when all completions in a group receive identical rewards (because *w⁺ = w⁻*). The paper notes this but does not analyze how often it occurs in practice or propose mitigations (e.g., reward shaping, dynamic group sizing).
- **Under‑analyzed training dynamics**: While reward and length dynamics are plotted (Figs. 3, 5), key phenomena—such as the sharp early reward drop in MATH500 training—are not explained. A deeper analysis of stability, convergence, and hyperparameter sensitivity (beyond ψ) is missing.
- **Missing ablations on key design choices**: The paper does not ablate the contribution of intermediate denoising steps in *wd1++*, the sensitivity to group size *G* (critical for the group‑relative advantage), or the effect of using more accurate (but costly) likelihood approximators within the *wd1* framework.
- **Lack of empirical connection to theory**: The energy‑guided diffusion interpretation, while theoretically novel, is not empirically validated (e.g., by tracking how the advantage distribution shifts during training or visualizing the effect of the guidance).
- **No variance reporting**: Results are reported from single runs without standard deviations across multiple seeds, which is expected for rigorous empirical evaluation and affects reproducibility.

### Trivial
- **Theory‑practice simplification**: The theoretical derivation employs a geometric mixture of old and reference policies, but the implementation sets β=0 and λ=1, effectively removing the reference policy. This is a valid special case of the theory, and the paper explicitly states the choice, so it does not invalidate the contribution.

## Nice-to-Haves
- Compare *wd1*‑tuned dLLMs to RL‑tuned autoregressive models of similar scale on the same tasks, to better contextualize the progress in dLLM reasoning.
- Measure inference efficiency (latency/throughput) after fine‑tuning, since one advertised advantage of dLLMs is faster generation.
- Provide qualitative examples of completions before/after *wd1* training, especially for tasks with large gains, to illustrate how the method improves reasoning steps.
- Expand the limitations discussion to address performance with sparse/binary rewards and in noisy‑reward settings.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Strength**: “The paper is well‑written” – generic, removed.
- **Weakness**: “The theoretical interpretation is loosely connected to the practical algorithm” – the paper does connect the theory to the objective (Remarks 1‑2) and the implementation is a special case (β=0, λ=1), so this is overstated.
- **Weakness**: “The method requires extensive hyperparameter tuning” – the paper provides ablation on ψ and uses standard hyperparameters for dLLM RL; no evidence that tuning is excessive.

## Suggestions
- **Address baseline reproduction**: Re‑run the *d1* baseline with the same hyperparameters and data as the original paper (or clearly justify any differences) to ensure a fair comparison. Report both the reproduced and originally reported numbers side‑by‑side.
- **Clarify SOTA comparison details**: In Table 3, explicitly state the fine‑tuning strategy (full vs. LoRA), dataset sizes, and any other relevant experimental conditions for *wd1++* and the compared methods. If possible, run a controlled comparison under identical settings.
- **Ablate the likelihood approximator**: Compare *wd1* using the *d1* approximator against a more accurate ELBO‑based approximator (with multiple t samples) to disentangle the benefits of the ratio‑free objective from the choice of approximator.
- **Report variance**: Run key experiments (e.g., Sudoku, GSM8K) with multiple random seeds and report mean ± standard deviation to assess reproducibility.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0]
Average score: 6.7
Binary outcome: Accept
