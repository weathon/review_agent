So QLoRA (Dettmers et al., 2023) is cited in the references but not discussed or compared against. The paper cites the 2022 Dettmers et al. (int8) in the text but the 2023 QLoRA paper only appears in references without being discussed. Now let me also look at the zero-valued quantized weights issue more carefully.

Looking at the paper again, in Q-SPSA (Eq. 5), the perturbation is $(\Delta + \epsilon z) \odot \bar{\theta}$, so $\theta_{\text{perturbed}} = (\Delta + \epsilon z) \odot \bar{\theta}$. When $\bar{\theta}_i = 0$, then $\theta_{\text{perturbed},i} = 0$ regardless of the perturbation, meaning entries with zero quantized weights receive no gradient signal. In 4-bit quantization with groups of 128, having many zero entries is common. This is a valid concern but it's a design choice limitation rather than an error.

Let me now write the final review.</think>Now I have all the information I need. Let me compile the final review.

## Summary

The paper proposes Quantized Zeroth-order Optimization (QZO), which enables memory-efficient fine-tuning of quantized LLMs by perturbing continuous quantization scales rather than discrete weights (Q-SPSA), combined with a Directional Derivative Clipping (DDC) technique for training stability. QZO is demonstrated on 4-bit (GPTQ) and 2-bit (AQLM) models across OPT and Llama families, showing significant memory reduction over full fine-tuning and MeZO while maintaining competitive performance on downstream tasks.

## Strengths

- **Core idea (Q-SPSA) is sound and elegant**: Perturbing continuous quantization scales $\Delta$ while keeping discrete weights $\bar{\theta}$ frozen naturally resolves the fundamental incompatibility between ZO optimization and discrete weight representations. This is a simple, principled design with no de/re-quantization cycles needed, making it practically plug-and-play with existing PTQ methods. Table 1 demonstrates this works across three model families.

- **Real practical impact**: QZO genuinely enables fine-tuning 13B models on a single 24GB GPU (Table 3: 5.78GB for Llama-2-13B in 2-bit). The measured ~3× memory savings over MeZO and ~5× over 16-bit full fine-tuning (Table 1) are meaningful capabilities, and the 2-bit extreme quantization results beating zero-shot baselines by large margins (e.g., SST-2: 80.5 vs 57.6) underscore practical value.

- **Empirical evidence that DDC prevents training collapse**: Figure 2 provides clear evidence that without DDC, training produces NaN losses, while Figure 3 shows robustness across a range of thresholds ($C \geq 75$). This is a practically important finding even without theoretical justification.

- **Generality across quantization paradigms**: Demonstrating compatibility with both GPTQ (scalar-based, 4-bit) and AQLM (codebook-based, 2-bit) in Section 4.2 broadens the method's applicability.

## Weaknesses

### Fatal
None.

### Major

- **Theorem 1 (unbiasedness of clipped gradient estimate) is very likely incorrect**: The theorem claims $\hat{\nabla}'_{\Delta} = \text{clip}(d, -C, C) \cdot z$ is an unbiased estimate of $\nabla_{\Delta} L(\Delta \odot \bar{\theta})$, but clipping $d$ is a nonlinear operation on a function of the random vector $z$, and in general $\mathbb{E}[\text{clip}(d, -C, C) \cdot z_k] \neq \mathbb{E}[d \cdot z_k] = \nabla_{\Delta_k} L$. This undermines the variance reduction argument in Equation (8), which requires unbiasedness to conclude that $\text{Var}[\hat{\nabla}'_{\Delta_k}] \leq \text{Var}[\hat{\nabla}_{\Delta_k}]$. DDC's practical effectiveness is real (Figure 2), but its theoretical framing as a bias-free variance reduction method is unjustified. It should be presented as gradient clipping applied to ZO—a known stabilization technique—rather than a theoretically novel insight.

- **The "more than 18× memory reduction for 4-bit LLMs" claim is misleading**: The abstract and introduction prominently state "QZO can reduce the total memory cost by more than 18× for 4-bit LLMs," but the measured data in Table 1 shows approximately 5× reduction for 4-bit models (e.g., 26.0GB/5.0GB for Llama-2-7B). The 18× figure appears to arise from comparing theoretical peak memory for 13B Adam fine-tuning (~104GB) against measured 2-bit QZO memory (5.78GB), conflating 4-bit vs 2-bit settings and theoretical vs measured baselines. The paper's own conclusion states "~3× less memory" than MeZO, which is accurate. The 18× claim is the paper's most prominent headline and it overstates the evidence.

- **"On par with MeZO" overstates performance on important model-task combinations**: On Llama-3.1-8B, QZO trails MeZO by 21.5 points on CB (69.6 vs 91.1) and 5.2 points on BoolQ (78.2 vs 83.4). On OPT-6.7B, QZO is 5.4 points behind on SST-2 (87.6 vs 93.0). While QZO wins on some tasks (e.g., Llama-2-7B SQuAD: 85.5 vs 80.7), the performance is highly variable across model-task combinations, and the paper does not acknowledge the large gaps or analyze when and why QZO struggles.

### Minor

- **Missing comparison with QLoRA**: The paper references QLoRA (Dettmers et al., 2023) in its references but does not compare against it. QLoRA is a directly relevant baseline that also fine-tunes quantized LLMs with low memory, and comparing against it would help readers understand the performance-memory trade-off.

- **No standard deviations or multiple runs reported**: ZO methods are notoriously high-variance. All results appear to be single runs, making it impossible to assess statistical significance of performance differences.

- **Only ~0.075% of parameters are updated**: QZO fine-tunes only group-level quantization scales. This extreme parameter restriction likely explains some of the performance gaps (e.g., the 21.5-point drop on Llama-3.1-8B CB). The paper does not analyze what types of adaptations the scale-only training enables or where it fails.

- **Zero-valued quantized weights receive no gradient signal**: When $\bar{\theta}_i = 0$ (common in 4-bit quantization), the perturbation $(\Delta + \epsilon z)_i \cdot \bar{\theta}_i = 0$, so these entries are completely insensitive to perturbation. This could degrade gradient quality, especially for models with many zero weights.

### Trivial
None.

## Nice-to-Haves

- Comparison with concurrent ZO+quantization methods (Feng et al. 2024; Zhou et al. 2025; Bar & Giryes 2025) mentioned in the related work section
- Ablation comparing Q-SPSA alone vs. Q-SPSA + regular SPSA for un-quantized parts (mentioned in Section 3.2.1 but never evaluated)
- Training dynamics/loss curves comparing QZO, MeZO, and full fine-tuning to understand convergence behavior

## Removed Points

- **Formatting/parser artifacts**: Various notation issues (e.g., garbled symbols in equations) are parser artifacts, not author errors, and are removed.
- **Missing appendix/proof for Theorem 1**: The proof is in Appendix A which was stripped by the parser. This exists in the original submission. However, the concern about Theorem 1's correctness is substantive and retained above.
- **Different optimizer settings across baselines**: The harsh critic flags that full fine-tuning uses SGD while MeZO/QZO use ZO-SGD with different batch sizes and learning rates. This is a standard comparison setup in the ZO literature (following MeZO itself) and not an unfair comparison — it reflects the actual use case for each method.
- **The "1% of FLOPs" claim being misleading**: This is accurate relative to the measured numbers in Table 2 and reflects a legitimate efficiency advantage of updating only scales.
- **Concern about naming collision with Zhou et al. 2025 "QZO"**: This is a real issue but is a presentation matter, not a methodological flaw. The authors should note it but it doesn't affect evaluation.

## Novel Insights

The key observation underlying QZO—that quantization scales provide a natural continuous parameter space within quantized models that ZO methods can perturb—is genuinely useful and resolves a practical bottleneck. However, DDC's theoretical contribution is illusory: Theorem 1 is likely incorrect, and DDC reduces to standard gradient clipping on ZO estimates (a known stabilization trick). The paper's real novelty is Q-SPSA, not DDC. The paper would be stronger if it honestly positioned DDC as an empirical stabilization technique rather than wrapping it in incorrect theory.

## Suggestions

- Retract or correct Theorem 1. If the proof cannot be fixed, acknowledge that DDC introduces bias and remove the variance reduction claim. Present DDC as an empirical stabilization technique.
- Correct the abstract and introduction to replace "more than 18× for 4-bit LLMs" with an accurate claim based on measured data (approximately 5× over full fine-tuning, 3× over MeZO for 4-bit models).
- Provide a more nuanced characterization of performance vs. MeZO, acknowledging the significant gaps on some tasks (especially Llama-3.1-8B CB and BoolQ) and analyzing when/why QZO struggles.
- Add QLoRA as a comparison baseline to contextualize the performance-memory trade-off.
- Report results across multiple random seeds.

## Calibration

I compared this paper against the following anchors:

- **LOZO (avg 7.0, Accept Poster)**: Zeroth-order fine-tuning with low-rank structure + correct convergence theory. More rigorous theory than this paper, similar practical motivation. This paper is clearly below LOZO due to the incorrect Theorem 1 and overclaimed results.
- **Sparse MeZO (avg 5.5, Reject)**: Incremental MeZO modification with empirical improvements but questionable theoretical justification and missing baselines. Similar profile to QZO—practical ZO contribution with theoretical gaps.
- **MeZO-A³dam (avg 4.75, Withdrawn/Reject)**: ZO optimization for LLMs with theoretical claims that reviewers found incorrect (Lemma C.6). Similar pattern of incorrect theory overclaiming.
- **Papers with overclaimed results (lf8QQ2KMgv, avg 3.75; edx7LTufJF, avg 2.50)**: Overstated experimental claims that didn't match data. QZO's 18× claim is similar but less severe (the actual savings are real, just ~5× not 18×).
- **Papers with incorrect theorems (C9BA0T3xhq, avg 2.0; Hh0Cg4epYY, avg 2.33)**: These had fundamental theoretical problems making core claims invalid. QZO's incorrect theorem undermines DDC's theoretical contribution but the Q-SPSA contribution stands independently, and empirical results are genuine.

QZO is better than the 2.0-2.5-scoring papers because its core practical contribution (Q-SPSA) works and is novel, even though the theorem is wrong. It's comparable to Sparse MeZO (5.5) but weaker on theory and has overclaimed metrics. It's below LOZO (7.0) which has both correct theory and strong experiments. I place it at 5.0—borderline, with a real practical contribution undermined by overclaims.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>