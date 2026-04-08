# ICLR Benchmark Results

Date: 2026-04-07 02:23
Critic/Merger: claude:claude-sonnet-4-6 (OpenRouter)
Neutral: qwen/qwen3.6-plus:free, Related Work: qwen/qwen3.6-plus:free:online (OpenRouter)

## b6qQmQ2F13

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary

This paper investigates how to optimally allocate a fixed GPU memory budget across model weights, KV cache, token budgets (serial scaling), sampling group sizes (parallel scaling), and KV cache compression for reasoning model deployment. Through systematic experiments spanning over 1,700 configurations across Qwen3, DeepSeek-R1-Distill, and OpenReasoning-Nemotron model families on AIME25, GPQA-Diamond, LiveCodeBench, and MATH500 benchmarks, the authors identify a scale-dependent threshold: models with effective size below ~8-bit 4B benefit more from allocating memory to larger/higher-precision weights, while larger models benefit more from longer generations and parallel scaling. The work also finds that 4-bit weight quantization, established as memory-optimal for non-reasoning tasks, is inefficient for mathematical and code reasoning, and that KV cache eviction outperforms quantization for small models.

## Strengths

- **Comprehensive and systematic empirical scope**: The evaluation covers an extensive search space—model scales from 0.6B to 32B, weight precisions of 4/8/16-bit, token budgets from 2k to 30k, sampling group sizes up to 16, and multiple KV compression strategies (R-KV, StreamingLLM, HQQ quantization)—across three model families and four diverse benchmarks. The 1,700+ configurations provide robust coverage of the design space.

- **Actionable, principled deployment guidelines**: The paper successfully translates empirical observations into clear heuristics. The identification of the ~8-bit 4B effective size threshold (Section 4, Figure 2) provides a concrete inflection point for practitioners deciding between model capacity vs. test-time compute. The task-dependent precision findings—mathematical reasoning and code generation favor higher precision (8/16-bit) while knowledge-intensive tasks favor 4-bit—offer nuanced, evidence-based advice beyond one-size-fits-all quantization prescriptions.

- **Validates across multiple model families and quantization methods**: The authors replicate key findings on DeepSeek-R1-Distill and OpenReasoning-Nemotron (Appendix C.6, Figures 6 and 16) and verify that conclusions hold across GPTQ, AWQ, and FP8 quantization (Appendix C.2, Figure 12). This cross-validation strengthens confidence that findings are not artifacts of a specific architecture or quantization scheme.

- **Clear methodology with reproducibility support**: The Pareto frontier analysis is well-executed with precise memory cost equations (Appendix B), transparent accounting of batch size effects on weight amortization (Appendix C.3), and publicly available code. The latency and throughput analysis (Appendix C.1) provides additional practical context.

## Weaknesses

- **No statistical significance measures reported**: All accuracy figures are pass@1 averages over 32 generations with no confidence intervals, standard errors, or error bars on the Pareto curves. Given that AIME25 has only 30 problems (where a single problem correct/incorrect swing represents 3.3 percentage points) and GPQA-Diamond has 52 problems, the absence of variance reporting makes it difficult to distinguish meaningful accuracy differences from stochastic noise. This is particularly concerning for Pareto frontier comparisons where configurations near the boundary may not be statistically distinguishable.

- **Inconsistent threshold reporting between introduction and Section 5**: The introduction (line 131) states that "KV cache eviction provides a better memory–accuracy trade-off than KV cache quantization for models with an effective size smaller than an 8-bit 4B model," while Section 5 (line 679) states "For models with an effective size smaller than an 8-bit 8B model, eviction consistently provides the best memory–accuracy trade-off." This discrepancy—roughly a factor of 2 in the stated threshold—is never reconciled and creates confusion about which threshold the evidence actually supports.

- **Limited mechanistic explanation for the scale-dependent threshold**: The paper identifies that the ~8-bit 4B threshold is task-dependent (shifting for MATH500 in Appendix C.4) but offers no theoretical or architectural account of why this inflection point emerges. Is it related to when KV cache memory first dominates weight memory? A capacity change in model representations? The finding is empirically solid but lacks explanatory depth that would help practitioners extrapolate to future architectures.

- **PRM comparison limited to a single verifier model**: The conclusion that "external verifiers are consistently memory-inefficient compared to self-contained majority voting" (Section 4.1, Figure 7) rests on experiments with ActPRM-X (7B, 13.28 GB) only. A smaller PRM or a verifier with better accuracy-per-parameter efficiency might yield different conclusions. The paper acknowledges this limitation but the claim is stated more strongly than the evidence warrants.

## Nice-to-Haves

- Analysis of how the threshold shifts with model architecture (e.g., MoE models where KV cache dynamics differ significantly from dense models) to establish broader applicability.

- Confidence intervals or bootstrapped error bars on the primary Pareto curves to distinguish meaningful accuracy gaps from noise.

- A brief theoretical or mechanistic hypothesis for why the effective size threshold appears where it does, beyond empirical observation.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **StreamingLLM not used as baseline**: This criticism is factually incorrect. StreamingLLM results are reported in Appendix C.7 (Figures 17, 18), where it is compared against R-KV and quantization across model sizes.

- **Temperature 0.6 choice unexplained**: The paper cites Muennighoff et al. (2025) for the temperature choice, which is standard practice in reasoning model evaluation. This is not a weakness.

- **Real-world memory fragmentation not measured**: Theoretical memory calculations are standard practice in this research area. Accounting for fragmentation and kernel overheads is beyond scope.

- **MoE model validation required**: While valuable, evaluating MoE architectures would expand the paper beyond its stated scope of studying dense reasoning models. The paper explicitly scopes to Qwen3 and related families with similar architectures.

- **Budget forcing ecological validity concerns**: Budget forcing with "Wait" prompts is the standard methodology established in Muennighoff et al. (2025) and used across recent test-time scaling literature. While the concern about reasoning coherence degradation is reasonable, it is not unique to this work.

- **Abstract presentation preference (threshold buried)**: This is a stylistic preference, not a substantive weakness. The abstract clearly states the contribution.

- **SOTA KV eviction baselines (SnapKV, H2O) missing**: R-KV (Cai et al., 2025) is a recent method specifically designed for reasoning models, making it a highly relevant baseline. Including additional methods would strengthen but is not required.

## Novel Insights

The paper's central insight—that the established wisdom of "4-bit quantization is memory-optimal" fails for reasoning models—is genuinely novel and practically significant. The decomposition of memory allocation into five interacting factors (model size, weight precision, token budget, sampling group size, KV compression) and the identification of scale-dependent optimal strategies provides a principled framework for deployment decisions that was previously lacking. The finding that mathematical reasoning tasks are more sensitive to weight precision degradation than knowledge-intensive tasks challenges assumptions about task-agnostic quantization strategies and suggests that future work on reasoning-optimized quantization should account for the computational vs. retrieval nature of the task. The observed synergy between model scale and optimal KV compression strategy (eviction for small models, quantization competitive for large models) offers a new dimension for efficient reasoning model design.

## Suggestions

- Reconcile the stated thresholds in the introduction and Section 5, either by correcting the discrepancy or explicitly explaining why different thresholds apply to different analyses.

- Add bootstrapped confidence intervals or standard errors to accuracy figures, particularly for the Pareto frontier plots where small accuracy differences drive allocation recommendations.

- Include a brief discussion hypothesizing why the effective size threshold emerges—whether it correlates with KV-to-weight memory ratios, attention head capacity, or other architectural factors—even if speculative.

---

## Pa6ak2B9jJ

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (4.4/10)
- Match: N/A

### Final Review

## Summary

AUTO-RT is a reinforcement learning framework for automatic jailbreak strategy exploration in red-teaming large language models. The key contributions are: (1) a hierarchical decomposition separating strategy generation (AM_g) from strategy-conditioned query rephrasing (AM_r), enabling reusable attack logic; (2) Dynamic Strategy Pruning (DSP), which terminates unpromising exploration branches early via constraint violations; and (3) Progressive Reward Tracking (PRT), which uses progressively degraded models and a novel First Inverse Rate (FIR) metric to provide denser reward signals. Experiments across 16 white-box and 2 black-box LLMs demonstrate improvements in attack success rate, strategy diversity, and defense generalization compared to RL and imitation learning baselines.

## Strengths

- **Hierarchical strategy decomposition (AM_g / AM_r)**: The insight that attack generation can be separated into reusable strategy learning and intent-conditioned rephrasing is a genuine conceptual contribution. By fixing AM_r and training only AM_g via RL, the approach reduces the optimization problem and enables strategies that generalize across toxic intents—empirically supported by high Defense Generalization Diversity scores and transferability results (Table 6).

- **Principled techniques for sparse-reward RL**: Dynamic Strategy Pruning grounds early termination in constrained MDP theory (Sun et al., 2021), and Progressive Reward Tracking with FIR-guided downgrade selection provides a systematic approach to reward shaping. The FIR metric is novel and empirically validated across multiple models (Figure 4).

- **Comprehensive evaluation scope**: The paper evaluates 18 target models across multiple families (Llama, Mistral, Yi, Gemma, Qwen, R2D2), plus black-box experiments on Llama3-70B and Qwen2.5-72B, and even proprietary models in Appendix G. Three complementary metrics (ASR, SeD, DeD) are reported with full ablations across all models (Tables 7–9).

## Weaknesses

- **Missing state-of-the-art baselines in main comparison**: PAIR, TAP, and AutoDAN-Turbo are cited in Related Work as relevant strategy-exploration methods but are excluded from Table 1. Table 3 compares against AutoDAN and human templates but omits TAP and AutoDAN-Turbo entirely. This omission weakens the positioning of AUTO-RT against current state-of-the-art methods, particularly for black-box settings where PAIR and TAP are specifically designed.

- **Table 3 reveals a significant ASR tradeoff**: AutoDAN achieves ASR_tst = 55.23% versus AUTO-RT's 38.38%—a 17 percentage point gap. While AUTO-RT outperforms in DeD (38.19% vs. 17.88%) and semantic diversity, the abstract's claim of "significantly improves success rates" should acknowledge this tradeoff. The paper states AUTO-RT achieves "near-human-level sustained attack capabilities," but this framing somewhat obscures the substantial raw ASR gap with template-based methods.

- **FIR selection criterion is not operationalized algorithmically**: The rule "select the last model before a sharp increase in FIR" requires visual inspection of Figure 4. The paper provides no automated elbow-detection procedure, threshold, or discussion of edge cases where no sharp increase exists. Only 6 of 18 models are shown in Figure 4; the consistency of this selection rule across all models is unverified.

- **Ablation non-monotonicities are unexplained**: Combining DSP+PRT does not consistently outperform individual components. For example, Vicuna 7B DeD: +DSP = 43.02, +PRT = 47.02, AUTO-RT = 46.80 (worse than +PRT alone). Qwen1.5-7B DeD: +DSP = 42.37, +PRT = 32.56, AUTO-RT = 34.25 (worse than +DSP alone). Additionally, +PRT consistently raises semantic similarity (hurts diversity) in Table 9, contradicting the claim that PRT improves exploration. These inconsistencies warrant discussion.

- **R2D2 failure mode under-analyzed**: On R2D2, Few-Shot (27.18%) outperforms AUTO-RT (12.45%)—the one adversarially trained model in the evaluation. The paper attributes this to R2D2's defense mechanism but provides no dedicated ablation or analysis of why the learned strategy model fails where random sampling succeeds.

- **Downgrade model construction is underspecified**: Reproducing the M1–M6 degradation pipeline requires details on fine-tuning method, learning rates, steps, and toxic data mixing ratios. The paper states "tuning or in-context learning" (Section 3.1) without specifics, and the Appendix does not fill this gap. White-box and black-box downgrade constructions (fine-tuning vs. ICL) differ qualitatively but receive no comparative analysis.

- **Efficiency claims lack compute-normalized metrics**: Figure 3 compares methods by training stages (1,000 episodes each), but early termination in DSP alters episode length, making stage-count comparisons uninformative about actual compute. The paper claims "accelerated discovery" but reports no wall-clock time, GPU-hours, or API query counts. This matters because the PPO training budget (8×A100, 9,000 episodes) is substantial.

## Nice-to-Haves

- Statistical significance tests or confidence intervals for ASR improvements would strengthen quantitative claims, particularly for smaller gains (e.g., Llama 3 8B: 14.55 → 15.00).

- Human verification of attack success rates beyond Llama-Guard-2 classification would address potential reward hacking concerns.

- A systematic FIR sensitivity analysis (selecting downgrade models one step earlier or later) would establish robustness.

- Comparison to CRT and Diver-CT (constrained red-teaming methods cited in Related Work) in Table 1 would clarify AUTO-RT's position relative to the broader literature.

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **"Exploitability never operationally measured"**: The paper conceptualizes exploitability as ease-of-triggering and argues that strategy-level attacks (like past-tense framing) are inherently more exploitable because they can be applied by any user. While a formal exploitability metric would strengthen the contribution, the conceptual framing is coherent and the empirical diversity/diversity metrics indirectly support the claim. Requesting a separate exploitability measurement is scope creep beyond the paper's stated goals.

- **"Statistical significance absent"**: While confidence intervals would be nice, single-run evaluations are standard for RL red-teaming papers at ICLR, and the paper's gains over baselines are often large enough (e.g., Vicuna 7B: 31.95 → 56.40) that statistical tests are unlikely to change conclusions. This is a nice-to-have, not a core flaw.

- **"Theoretical guarantee for penalty C unverified"**: The paper correctly cites Sun et al. (2021) for the theoretical foundation of early-terminated CMDPs. The practical choice of penalty values is standard implementation detail; while specific values would aid reproducibility, the absence is not a fundamental theoretical flaw.

- **"LLM usage statement"**: The reviewer's concern about LLM usage in writing is not a substantive weakness; the paper transparently discloses LLM assistance for editing in the appendix.

## Novel Insights

The hierarchical decomposition of attack generation into strategy learning and rephrasing is a genuinely novel conceptual contribution that the field has largely overlooked. Most prior work optimizes prompts directly; AUTO-RT instead learns *how to think about attacks* (strategies) rather than the attacks themselves. This mirrors the distinction in human red-teaming between learning adversarial principles versus memorizing specific exploits. The DeD metric (Defense Generalization Diversity) is also innovative—measuring sustained attack capability after defense adaptation directly tests whether discovered strategies reflect fundamental vulnerabilities versus shallow classifier artifacts. The finding that strategies transfer partially across models (Table 6) but with substantial decay suggests learned strategies capture some model-agnostic vulnerabilities while retaining model-specific components.

## Suggestions

- Add an automated FIR selection procedure (e.g., gradient-based elbow detection or a threshold rule) and evaluate its consistency across all 18 models.

- Include PAIR and TAP as baselines in black-box experiments, and AutoDAN-Turbo in white-box comparisons, to properly position against SOTA.

- Report compute-normalized efficiency (wall-clock time, GPU-hours, or total target queries) to substantiate "accelerated discovery" claims.

- Analyze and discuss the ablation non-monotonicities: why does PRT hurt semantic diversity, and why does DSP+PRT sometimes underperform individual components?

- Expand the R2D2 analysis: does AUTO-RT fail because learned strategies overfit to non-adversarially-trained models, or is there a deeper interaction with the defense mechanism?

---

## Kw2mvnzCoc

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (6.2/10)
- Match: N/A

### Final Review

## Summary
TSPulse introduces a family of compact (~1M parameter) pre-trained time-series models with disentangled temporal, spectral, and semantic embeddings, achieving strong performance across anomaly detection, classification, imputation, and similarity search tasks. The key innovations include multi-objective reconstruction across time and frequency domains, register tokens for semantic representations, hybrid masking for training robustness, and task-specific post-hoc fusers.

## Strengths
- **Strong empirical results across four diagnostic tasks:** TSPulse achieves state-of-the-art performance on TSB-AD anomaly detection (ranking #1 with VUS-PR 0.48 vs. prior best 0.42), strong classification accuracy on UEA benchmarks (73.3% mean accuracy, 5–16% over baselines), competitive imputation MSE under both block and hybrid masking protocols, and substantial gains in similarity search over forecasting-focused pre-trained models.
- **Practical efficiency with small model footprint:** At 1M parameters, TSPulse delivers performance competitive with models 10–100× larger (MOMENT: 40–341M, Chronos: 8–709M). Table 3 shows TSPulse achieves 14–100× faster CPU inference and 9–125× faster GPU inference than baselines, enabling deployment in resource-constrained environments.
- **Well-designed hybrid masking for robustness:** The hybrid masking strategy (combining point and block masking) addresses a genuine issue in existing pre-trained models that overfit to fixed masking patterns. The ablation in Table 1(c) shows a 79% MSE drop when hybrid masking is removed under the hybrid evaluation protocol, demonstrating the method's effectiveness.
- **Ablation studies demonstrate component contributions:** Table 1 systematically shows drops from removing each component (8% classification accuracy loss without disentangled embeddings, 11–16% loss without TSLens, 9% loss without identity-initialized channel mixers), providing evidence that design choices matter.
- **Reproducibility and transparency:** Models and code are publicly available, hyperparameters are detailed in Appendix A.9, pre-training datasets are clearly listed in Table 10, and evaluation follows established benchmarks (TSB-AD, UEA, LTSF).

## Weaknesses
- **Unfair imputation comparison framing:** The paper prominently claims "+50% improvement on imputation" but this result comes from evaluating TSPulse (pre-trained with hybrid masking) against baselines (MOMENT, UniTS) that were pre-trained with block masking, under a hybrid masking evaluation protocol. This comparison is structurally favorable to TSPulse. The more honest comparison in Figure 13 (block masking protocol) still shows TSPulse outperforming baselines by substantial margins, and should be the primary result highlighted.

- **"Zero-shot" anomaly detection terminology is misleading:** The TSPulse-ZS variant for anomaly detection uses labeled tuning data to select the best scoring head via "Multi-Head Triangulation" (Appendix A.11.3). While this follows the TSB-AD benchmark protocol and is legitimate, calling it "zero-shot" obscures the use of labels. The true zero-shot Headensemble achieves VUS-PR 0.44 vs. 0.48 for the tuned variant—a meaningful drop that should be transparently reported in the main text.

- **Similarity search lacks specialized representation learning baselines:** The comparison includes only MOMENT and Chronos, both forecasting-focused models. Time-series representation learning methods specifically designed for similarity/retrieval tasks (TS2Vec, BTSF, TimeDRL, T-Rep) are absent despite being cited in the related work. This creates an asymmetric comparison that favors TSPulse's retrieval-oriented training.

- **"Disentanglement" claims exceed the methodology:** The paper uses "disentangled representations" terminology but implements multi-head multi-task training on spatially separated embedding segments, without formal independence constraints or mutual information regularization. While Section 6's perturbation analysis shows useful behavioral differences between embeddings, this is post-hoc characterization rather than structural disentanglement. The dimensional confound (Time/FFT at 1536 dims vs. Register at 256 dims) in distortion metrics (Table 2) also complicates the interpretation—lower distortion for semantic embeddings could partially stem from fewer dimensions.

- **Incomplete statistical validation:** Classification ablations (Table 18) use only 17 of 29 UEA datasets without stated selection criteria. No confidence intervals, standard deviations, or statistical significance tests are reported throughout the paper. Given modest improvements (e.g., 5% over VQShape on classification), readers cannot assess reliability across random seeds or splits.

- **Synthetic-only disentanglement validation:** The embedding sensitivity analysis (Section 6, Appendix A.3–A.4) uses only synthetic sinusoidal signals. It remains unclear whether the claimed disentanglement properties transfer to real-world time series with non-stationary, multimodal, or domain-specific characteristics found in the actual benchmark datasets.

## Nice-to-Haves
- Formal disentanglement metrics (DCI, MIG) on real benchmark data to complement synthetic perturbation analysis
- Comparison to specialized representation learning baselines (TS2Vec, BTSF, TimeDRL) for similarity search
- Statistical significance testing and confidence intervals across all reported metrics
- Discussion of failure cases or domains where TSPulse underperforms baselines

## Removed Points
*These points are flagged to be removed, treat them with caution*

- **Claim that "+20% on TSB-AD" is numerically overstated:** The calculation is approximately correct. TSPulse-ZS achieves 0.48 VUS-PR vs. Sub-PCA's 0.42, which is roughly 14% relative improvement. The "+20–30%" claim includes the fine-tuned variant and refers to multiple baselines, which is reasonable framing.

- **"First to unify and triangulate multi-space outputs" priority claim:** This is qualified in the paper as applying specifically to "a single lightweight framework" for pre-trained time-series models. The related work discussion of TF-C and BTSF provides sufficient context.

- **Register tokens "borrowed" from vision transformers:** This adaptation to time-series is a valid contribution. Citing prior work is standard practice and doesn't diminish the contribution.

- **FFT normalization sensitivity to outliers:** This is speculative without empirical evidence of failure. The paper uses standard normalization techniques.

- **Pre-training data dominated by electricity/traffic/weather domains:** The critic notes this as a limitation, but the evaluation benchmarks cover diverse domains including ECG, motion, and other types. This is a generalization concern, not a factual error.

## Novel Insights
The hybrid masking design addresses a previously underappreciated "pre-training mask bias" in existing foundation models: models trained exclusively with fixed-length block masking perform well on block-missing patterns but struggle with irregular point-and-block missingness common in real-world scenarios. The 79% MSE degradation when block-trained models face hybrid missingness (Table 23) quantifies this gap and justifies hybrid masking as more than a simple augmentation technique—it's a pre-training distribution correction.

The multi-head triangulation mechanism for anomaly detection reveals a complementary property: different anomaly types (sudden spikes, periodicity breaks, trend violations) are better captured by different reconstruction heads (temporal, spectral, predictive). This suggests that explicit multi-view architectures provide intrinsic robustness for anomaly detection that single-head models lack.

## Suggestions
- Lead with the block masking imputation comparison (Figure 13) in the main paper rather than the hybrid masking results, as it provides a fairer comparison. Report both prominently but distinguish the evaluation protocols clearly.
- Rename TSPulse-ZS for anomaly detection to clarify that head selection uses labeled tuning data, or report Headensemble results (true zero-shot) alongside the tuned variant.
- Add TS2Vec, TimeDRL, or T-Rep baselines for similarity search to strengthen the comparison for that task.
- Include confidence intervals or standard deviations for classification results across multiple runs, or clarify the dataset subset selection criteria for ablations.
- Extend the perturbation analysis in Section 6 to include at least one real benchmark dataset to validate that disentanglement properties transfer beyond synthetic signals.

---

## sJxBWDc8SM

- GT: Reject (avg 3.5)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary

This paper systematically compares State-Space Models (SSMs) and Transformers on associative recall (MQAR) and copying tasks, demonstrating that SSMs exhibit critical optimization instabilities—specifically, success confined to extremely narrow learning rate windows—that can confound expressivity evaluations. Through extensive experiments (~3,000 runs, ~20,000 GPU hours), the authors show that proper hyperparameter tuning substantially improves SSM performance, reveal contrasting scaling behaviors (SSMs favor width; Transformers favor depth), and identify 1D convolutions as a key architectural component enabling single-layer task solution.

## Strengths

- **Comprehensive experimental investigation with meaningful practical implications:** The paper provides substantial evidence across MQAR and copying tasks that prior conclusions about SSM expressivity limitations were confounded by suboptimal learning rate selection. Figure 1 convincingly demonstrates that Mamba's effective learning rate window is dramatically narrower than Transformers', and that grids from prior work (Arora et al., 2023) could miss this window entirely.

- **Actionable architectural insights with clear ablations:** The convolution ablations (Table 2, Table 3) provide concrete mechanistic evidence: removing conv1d from Mamba drops accuracy from 99% to 2%, while adding convolution to 1-layer Transformers enables task solution. The DeltaNet comparison (Figure 7) further suggests architectural pathways to improved stability.

- **Important empirical discovery about single-layer dynamics:** The observation that 1-layer Transformers exhibit a loss bump resembling induction head formation—previously only documented in multi-layer settings—is genuinely novel. The paper correctly hypothesizes this represents an "attempt" at circuit formation that cannot be completed without additional layers.

## Weaknesses

- **Scope limited to synthetic benchmarks with unclear generalization to language modeling:** All experiments use MQAR and copying tasks. The authors acknowledge this limitation, but it remains significant: the central thesis that optimization instability (not expressivity) is the "crucial differentiator" depends on whether these findings transfer to realistic language modeling. Without validation on any natural language tasks, the practical significance for SSM development remains uncertain.

- **Narrow learning rate window lacks mechanistic explanation:** While the paper documents LR sensitivity extensively, it provides no direct empirical analysis of why SSMs exhibit this instability. The discussion of vanishing gradients and decay rates references Trockman et al. (2024) but offers no gradient norm measurements, loss landscape curvature analysis, or eigenvalue tracking that would substantiate the proposed mechanism. The DeltaNet/Householder hypothesis remains untested.

- **Striking convolution asymmetry left unexplored:** Table 3 shows that convolving K or V alone achieves 99% accuracy, but convolving Q achieves only 2%. This asymmetry has a natural mechanistic interpretation (K and V carry stored content; Q must match that content), yet the paper does not develop this insight at all—a significant missed opportunity.

- **Induction head claim remains speculative without mechanistic probes:** The hypothesis that single-layer Transformers "attempt" to form induction heads is plausible but unverified. No attention pattern analysis, head probing, or representational similarity metrics are provided to support this interpretation of the loss bump phenomenon.

## Nice-to-Haves

- Validation on at least one language modeling benchmark (even small-scale) to demonstrate whether LR instability persists in natural text settings.

- Analysis with alternative optimizers (AdamW alternatives) to test whether the instability is architecture-intrinsic or optimizer-coupled.

- Gradient norm tracking or loss landscape measurements to provide direct empirical support for the vanishing-gradient hypothesis.

## Removed Points

These points are flagged to be removed, treat them with caution:

- *Critic's claim that 1-layer Transformer failure on MQAR is "largely a known theoretical result"* — The paper properly cites Sanford et al. (2024) and contextualizes this finding. The empirical demonstration that Mamba can solve the single-layer task while Transformers cannot, and the associated training dynamics analysis, remains a contribution even if the Transformer limitation was previously known.

- *Critic's request for "how narrow" quantification ratio* — While precise numerical ratios of effective LR ranges would strengthen the paper, Figure 1 visually demonstrates the point clearly enough for an empirical contribution. This is a presentation preference, not a substantive gap.

- *Critic's demand for significance tests and confidence intervals* — The paper reports means with max-min ranges over 5 seeds. While standard deviation would be preferable, max-min provides variability information, and the extensive sweep (3000 runs) provides statistical robustness. This is not a critical methodological flaw.

- *Spark finder's request for "optimizer agnosticism" testing* — The paper's contribution is specifically about LR sensitivity in standard training (AdamW). Testing alternative optimizers is a reasonable future direction but not required for the current contribution to be valid.

- *Spark finder's request for "theoretical stability bound"* — ICLR publishes strong empirical papers without requiring theoretical proofs. The empirical findings are valuable even without formal derivation.

## Novel Insights

The discovery that single-layer Transformers exhibit a loss bump "reminiscent of induction head formation"—previously only observed in multi-layer settings—suggests that even shallow attention architectures undergo a phase transition attempting to form retrieval circuits. That this "attempt" succeeds in Mamba (with convolution) but fails in vanilla 1-layer Transformers provides a concrete lens into how architectural inductive biases interact with optimization dynamics. The convolution asymmetry (K/V convolutions help; Q convolution does not) points toward a mechanistic explanation: convolving query representations fails to provide useful locality when the query's role is to match stored content rather than to store it.

## Suggestions

- Add a small-scale language modeling experiment (e.g., WikiText-103 perplexity or a downstream task) using the identified hyperparameter best practices to demonstrate external validity.

- Provide at least cursory gradient norm measurements across training for Mamba vs. DeltaNet to empirically ground the vanishing-gradient discussion.

- Add 2-3 sentences analyzing the K/V vs. Q convolution asymmetry, even if speculative, to close this mechanistic loop.

---

## RpDJz00zNh

- GT: Reject (avg 4.5)
- Predicted: N/A (5.2/10)
- Match: N/A

### Final Review

## Summary

The paper proposes ConciseHint, an in-reasoning intervention framework that reduces token usage in large reasoning models by dynamically injecting concise hints during generation. The method adaptively controls injection interval and position based on current reasoning length, and extends to a trainable variant (ConciseHint-T) that learns soft prompt embeddings from concise reasoning data. Experiments across GSM8K, AIME24, and GPQA-Diamond with Qwen3 and DeepSeek-R1 models demonstrate 30-60% token reduction with minimal accuracy loss, plus compatibility with existing efficiency methods.

## Strengths

- **Novel intervention paradigm:** The shift from pre-reasoning interventions (prompting/SFT) to during-generation intervention is a meaningful conceptual contribution. The dual-adaptive mechanism (interval scaling via Eq. 1 and dynamic position via Eq. 3) is well-motivated by the accuracy-efficiency trade-off, and ablations in Tables 3-4 convincingly demonstrate its necessity—fixed high-intensity injection drops AIME24 accuracy from 67.00 to 45.33 on Qwen3-4B.

- **Strong empirical coverage with compatibility demonstration:** The paper evaluates across model scales (1.7B to 30B), multiple benchmarks (mathematical reasoning, code generation, commonsense), and demonstrates plug-and-play compatibility. Table 1 shows that combining ConciseHint with Deer, NoWait, and Prompt baselines consistently yields additional token reductions (e.g., 40.1% further reduction when combined with Deer on GSM8K/Qwen3-4B).

- **Rigorous latency analysis:** Section A.2 provides both theoretical and empirical analysis of KV cache invalidation overhead, showing relative per-injection latency <0.3% (Figure 6). The end-to-end latency reductions in Figure 7 (e.g., GSM8K latency from 3.23s to 1.68s for Ours(Ori)) demonstrate practical utility beyond token counts.

- **Controllability via interpolation:** The embedding interpolation mechanism (Eq. 4) provides a smooth control knob for compression intensity, empirically validated in Figure 3's controllability curves. The learned embeddings in ConciseHint-T generalize from GSM8K training data to out-of-domain benchmarks (AIME24, GPQA-Diamond), suggesting the captured patterns transfer.

## Weaknesses

- **Insufficient statistical rigor on small benchmarks:** AIME24 contains only 30 problems. With accuracy differences of 1-3% translating to sub-1-problem effects, the reported comparisons (e.g., Table 3: Ours(adaptive) 67.00 vs. Fixed-128 63.33) lack confidence intervals or significance tests. The 10-run averaging helps but does not substitute for proper statistical reporting.

- **Unexplained hyperparameter constants:** Equation (3) introduces constants 1024 and 0.8 without principled motivation. Why 1024 specifically for normalization? Why cap position at 0.8 of τ_k rather than 0.7 or 0.9? The ablations explore α and β (Appendix A.1) but not these constants, making the formula appear heuristic rather than grounded.

- **Self-referential complexity proxy:** Equation (1) uses current output length l_k as a proxy for query complexity, but l_k is itself influenced by prior hint injections. A query that triggers verbose early exploration (before hints take effect) may be misclassified as complex, while an easy query heavily-hinted early appears shorter than its true difficulty warrants. The paper notes the correlation prior but provides no analysis of failure cases where this breaks down.

- **Missing comparison to mechanistically similar work:** The s1 paper (Muennighoff et al., 2025), cited by this work, uses budget forcing to inject tokens during generation to extend reasoning—the inverse objective but identical mechanism. The distinction from and comparison to this prior art is absent, making the novelty claim of "largely unexplored direction" require more careful qualification.

- **Implementation reproducibility gap:** Algorithm 1 uses `client.completions.create` suggesting black-box API usage, yet the method requires continuous embedding injection and KV-cache manipulation (Section A.2) that necessitate white-box access. The exact mechanism for integrating soft prompts into the forward pass is not detailed, creating a reproducibility barrier.

- **Limited reasoning quality evaluation beyond accuracy:** Section A.4 uses GPT-4o-mini to evaluate reasoning quality but only on correct responses, missing the critical failure cases where hints may induce superficial reasoning paths. The HumanEval pass@10 regression (98.78→96.34 in Table 7) is also unaddressed despite being a meaningful coverage reduction.

## Nice-to-Haves

- Analysis of failure modes where complexity proxy breaks down (queries with short early exploration but high true difficulty)
- Comparison to budget forcing methods that inject during generation for opposite purposes
- Batched inference implementation details for practical deployment
- Calibration analysis to verify hints don't suppress necessary uncertainty expression

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Budget forcing comparison reveals inverted goals"** — The critic frames the absence of s1 comparison as concealing similarity, but s1's budget forcing EXTENDS reasoning while ConciseHint SHORTENS it. These are opposite objectives with opposite mechanisms despite sharing the mid-generation intervention concept. A comparison would be informative but the gap is not a concealment.

- **"Cumulative KV cache overhead is hidden"** — The per-injection analysis is supplemented by end-to-end latency in Figure 7, which directly measures cumulative effects. The critic's claim that cumulative overhead is unreported is factually incorrect.

- **"Paper claims performance maintenance but severe degradation occurs without adaptive mechanism"** — The abstract states the adaptive mechanism ensures performance maintenance. The critic's claim that this qualification is hidden is overstated; it's explicitly the core contribution.

- **"Batched inference incompatibility is unaddressed"** — This is a valid practical concern but represents deployment engineering beyond the paper's scope. The method is demonstrated on standard inference; batched integration is a natural extension, not a core flaw.

## Novel Insights

The transition word analysis in Table 5 reveals that ConciseHint primarily eliminates entire thought-checking cycles rather than compressing individual reasoning steps—the transition interval barely changes (113.42→118.66 tokens between transition words on GSM8K/Qwen3-4B), but the count drops from 14.97 to 4.39. This suggests the mechanism operates at the level of meta-cognitive control (suppressing self-correction loops) rather than linguistic compression, an important distinction for understanding intervention effects.

## Suggestions

- Add confidence intervals or statistical significance tests, particularly for AIME24 where sample size is smallest
- Provide principled motivation or ablation for the 1024 and 0.8 constants, or acknowledge them as empirically tuned
- Include failure case analysis: examine incorrect predictions to verify hints don't induce systematic reasoning shortcuts
- Release inference wrapper code to bridge the gap between API-style Algorithm 1 and the actual implementation
- Discuss the HumanEval pass@10 regression and whether it represents a meaningful trade-off for code generation tasks

---

## 1EdAn5gMVv

- GT: Reject (avg 5.0)
- Predicted: N/A (5.3/10)
- Match: N/A

### Final Review

## Summary

SpatialBoost proposes a framework to enhance pre-trained 2D vision encoders with 3D spatial awareness by injecting linguistically structured spatial knowledge via LLM-guided fine-tuning. The approach constructs a multi-turn Chain-of-Thought spatial reasoning dataset (pixel→object→scene hierarchy) from off-the-shelf depth, segmentation, and 3D reconstruction models, and uses a dual-channel attention mechanism to prevent catastrophic forgetting while learning new spatial features. The method is evaluated across depth estimation, segmentation, 3D scene understanding, robotics, and image classification benchmarks.

## Strengths

- **Comprehensive empirical validation across diverse tasks and backbones:** The paper evaluates SpatialBoost on dense prediction (depth, segmentation), 3D-centric tasks (Lexicon3D), embodied robotics (CortexBench), and general vision (ImageNet, retrieval) using multiple backbones (OpenCLIP, SigLIPv2, DINOv2, DINOv3). Tables 3–5 show consistent improvements, and the ablations in Tables 6–8 and Appendix F systematically isolate design choices.

- **Effective mitigation of catastrophic forgetting:** The dual-channel attention mechanism is well-motivated. Table 17 and Figure 6 clearly demonstrate that full fine-tuning degrades ImageNet performance (86.3→79.5), while dual-channel attention preserves it (86.3→87.6) while improving spatial metrics. This addresses a critical practical concern when adapting frozen encoders.

- **Hierarchical CoT structure validated:** Table 7 shows that forward hierarchical ordering (pixel→object→scene) outperforms shuffled or reverse ordering, and Appendix F.1 (Table 15) demonstrates that combining all three levels yields the best performance. This validates the design principle of progressive spatial reasoning.

- **Detailed methodology and reproducibility:** Appendices A–F provide training protocols, hyperparameter ranges, data filtering criteria (LPIPS constraints, CLIP-based scene filtering), and explicit prompt templates (Tables 10–14). The construction of the spatial reasoning dataset is well-documented.

## Weaknesses

- **Incomplete reporting in main result tables:** Tables 1 and 2 are missing rows for SigLIPv2+SpatialBoost and DINOv3+SpatialBoost. The abstract explicitly claims "SpatialBoost improves SigLIPv2 from an RMSE score of 0.51 to 0.39 on NYUd linear probing," yet this result does not appear in Table 1. Similarly, Section 4.2 states "DINOv3's mIoU on ADE20K increases from 55.9% to 59.7%" without a corresponding table row. This inconsistency between text claims and presented data undermines confidence in the reported improvements.

- **Potential data contamination in geometric understanding evaluation:** The paper uses ScanNet for multi-view VQA training data (Appendix C), then evaluates geometric understanding on ScanNet-based benchmarks from Lexicon3D (Registration Recall, RTE). While the task-specific heads are trained separately, the encoder has seen images from ScanNet during Stage 2 alignment, potentially inflating geometric understanding scores. Table 3 shows exceptionally large gains (e.g., OpenCLIP Registration Recall: 22.6%→78.8%, SigLIPv2: 47.8%→86.4%) that warrant explanation beyond "spatial knowledge injection."

- **Misleading comparison with frontier models:** Appendix B presents SpatialRGPT results where Vicuna-1.5-7B + SpatialBoost DINOv3 achieves 58.7, described as "surpassing frontier models like GPT-4o (39.7) and Gemini-2.5-Flash (42.5)." The paper notes these models "are not directly compared" but immediately makes the comparison. This is misleading because SpatialBoost is trained on spatial reasoning data designed for this benchmark, while GPT-4o and Gemini are generalist models without such specialization.

- **Incorrect attribution of dual-channel attention as novel:** The paper states "we introduce a dual-channel attention mechanism" and presents it as part of its contribution, but this mechanism is directly adopted from Hong et al. 2023a (CogVideo). While properly cited, claiming introduction of an adopted technique overstates the methodological novelty.

- **No computational cost analysis:** The paper does not report training time, GPU-hours, memory footprint, or inference latency. Stage 2 fine-tunes a 7B LLM, and dual-channel attention adds 25–30% parameters. Without efficiency metrics, practical adoption cannot be assessed, especially given the complexity of the three-stage pipeline.

## Nice-to-Haves

- **Analysis of pseudo-label error propagation:** While Table 19 shows VFM-based and GT-based data yield similar results (100K ScanNet samples), a deeper analysis of how noise from Depth Pro, SAM, or VGGT affects learned representations—particularly on challenging scenes (occlusions, reflections)—would strengthen confidence in the approach.

- **Isolation of language encoding contribution:** Table 6 compares LLM supervision against pixel-level heads (linear, SAM decoder, VGGT decoder), but the LLM approach uses different data (multi-turn CoT vs. single-task). A cleaner ablation controlling for data would clarify whether the gains come from language encoding specifically or simply from richer supervision.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"DINOv3 is an arXiv preprint and less established"** — This is not a weakness. DINOv3 is properly cited and represents the current state-of-the-art in vision encoders; its recency does not diminish the validity of benchmarking against it.

- **"Need zero-shot 3D benchmarks"** — This is scope creep. The paper uses linear probing on NYUd and KITTI for depth, which is the standard evaluation protocol for vision encoder quality. Requesting zero-shot evaluation is an additional experimental burden not standard in this literature.

- **"Need comparison with 3D foundation models (DUSt3R, Depth Anything V2)"** — The paper's goal is enhancing existing 2D encoders with spatial knowledge, not competing with specialized 3D models. This would require a different methodological scope.

## Novel Insights

Beyond the paper's contributions, an important insight emerges from Table 6: LLM-based supervision yields +2.32% ImageNet accuracy while pixel-level depth/segmentation heads cause -1.39% to -1.74% degradation. This suggests that language encoding provides a form of *regularization* that prevents task-specific overfitting—a hypothesis the paper does not explore but which has implications for how we should think about multi-task representation learning. Additionally, the combination of scene captions with spatial CoT (Appendix F.1) suggests that maintaining general visual knowledge alongside specialized spatial knowledge is crucial for avoiding catastrophic forgetting—a design principle worth emphasizing.

## Suggestions

1. **Complete all tables:** Add SigLIPv2+SpatialBoost and DINOv3+SpatialBoost rows to Tables 1 and 2, or explain their absence. If results are weaker or inconsistent, report them transparently.

2. **Address potential data contamination:** Clarify whether ScanNet images appear in both training splits and evaluation splits, and if so, provide analysis on held-out scenes to ensure fair evaluation.

3. **Reframe frontier model comparisons:** Either remove the GPT-4o/Gemini comparison or clearly frame it as a specialized vs. generalist comparison rather than a claim of superiority.

4. **Correct dual-channel attention attribution:** State clearly that this mechanism is adopted from CogVideo (Hong et al. 2023a) rather than introduced in this work.

5. **Add computational overhead reporting:** Report GPU-hours, memory usage, and inference latency for Stage 2 and Stage 3 training to enable practical assessment.

---

## NfO2Lt2WY7

- GT: Reject (avg 2.0)
- Predicted: N/A (4.6/10)
- Match: N/A

### Final Review

## Summary

This paper investigates whether the complex GRPO loss function is necessary for training LLMs to reason mathematically. Through systematic ablations, the authors find that (1) negative feedback via centered advantage estimation is essential for training stability, and (2) PPO-style clipping is dispensable. They propose RGRA (REINFORCE with Group Relative Advantage), a simplified variant that removes clipping while retaining group-relative advantages, and demonstrate that it matches or exceeds GRPO performance across mathematical benchmarks on models up to 1.5B parameters.

## Strengths

- **Systematic ablation methodology:** The paper cleanly isolates GRPO components—testing positive-only advantages, raw REINFORCE (no advantage normalization), and clipping removal—with training dynamics (Figure 1) tied directly to downstream outcomes (Tables 1–3). The finding that advantage centering prevents collapse while clipping is optional is clearly supported by the experiments.

- **Evidence connecting stability to reasoning emergence:** The qualitative analysis in Figure 2 demonstrates that stable training methods (GRPO, RGRA) induce explicit reasoning traces, while collapsed methods output degenerate responses. This ties algorithmic properties to emergent behavior.

- **Multi-architecture, multi-benchmark evaluation:** Experiments cover Qwen2.5 (0.5B, 1.5B) and Llama3.2 (1B) across 9 benchmarks spanning English math, Chinese math, and STEM. Within the small-model regime, this breadth supports generalization claims.

- **Reproducibility:** Complete hyperparameters (Table 4), training details, and a code repository are provided.

## Weaknesses

- **Limited model scale undermines generalization claims:** All experiments use models ≤1.5B parameters. The claim that PPO-style clipping is unnecessary for LLM reasoning is only demonstrated on small models, where gradient dynamics and policy drift differ from the 7B+ models commonly used in reasoning research. The authors acknowledge this limitation, but it fundamentally bounds the paper's contribution—ICLR reviewers reasonably expect findings about LLM post-training to hold at scales where reasoning emergence is non-trivial.

- **No statistical validation of performance claims:** Results are single-seed runs without variance estimates. The "17 out of 27 tasks" claim could reflect training noise rather than genuine advantage. For example, RGRA vs. GRPO differences are often 1–4 points on benchmark averages—within plausible seed-to-seed variance. Standard ICLR expectations include multiple seeds with significance testing for such claims.

- **Efficiency claims are unsubstantiated:** The abstract and conclusion frame RGRA as "more efficient," but no wall-clock time, memory footprint, FLOPs, or sample efficiency metrics are reported. This is not a minor omission—it is a core claimed benefit that lacks any evidence.

- **Critical efficiency consideration ignored:** RGRA samples from the current policy (π_θ) per Equation 2, while GRPO samples from the old policy (π_θ_old), enabling off-policy reuse across updates. This distinction means RGRA may require fresh rollouts each gradient step, potentially increasing wall-clock cost. The paper never addresses this practical concern.

- **RGRA's novelty unclear relative to prior work:** The paper acknowledges inspiration from Ahmadian et al. (2024), which studied REINFORCE-style optimization for LLMs, including leave-one-out baseline variants (RLOO). RGRA with group-relative advantages is conceptually close to RLOO, but no direct comparison or differentiation is provided. Without this, the contribution's novelty is ambiguous.

- **Key hyperparameters not ablated:** The KL penalty (β=0.005) and group size (G=8) are held constant. Whether RGRA's stability depends on specific values of these parameters—particularly whether KL regularization substitutes for clipping—is not investigated.

## Nice-to-Haves

- **Compare RGRA against RLOO directly:** Since Ahmadian et al. (2024) already proposed REINFORCE with leave-one-out baselines, clarifying what RGRA adds (if anything) would strengthen the contribution.

- **Ablate KL penalty and group size:** Testing RGRA without KL regularization, and with varied group sizes, would clarify whether the "clipping is unnecessary" finding is robust or contingent on other stabilizing mechanisms.

- **Report efficiency metrics:** Even basic wall-clock time per step and total training time would substantiate the efficiency framing.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **GSM8K memorization concern:** The harsh critic suggested training on GSM8K train split while testing on GSM8K test constitutes memorization. This is standard ML practice and overstates the issue—the paper tests on 8 other benchmarks beyond GSM8K.

- **"Three models not two" minor error:** The harsh critic noted a sentence says "two separate models" but lists three (policy, reward, value). This is a trivial phrasing issue, not a substantive criticism.

- **Missing DAPO baseline:** While DAPO is a relevant recent GRPO variant, its absence is a reasonable omission given the paper's focus on systematic ablation rather than comprehensive baseline comparison. The paper already compares to GRPO, RAFT, and vanilla REINFORCE.

- **Formatting complaint about equation placement:** The placement of the positive-only formulation in the Results section rather than Methods is a minor presentation issue, not a scientific flaw.

## Novel Insights

The paper provides a clear empirical dissection of GRPO's components, correctly identifying that advantage centering—not PPO-style clipping—is the essential ingredient for training stability in this setting. The training dynamics in Figure 1 (showing reward collapse for positive-only methods) coupled with the reasoning trace analysis (Figure 2) offer a mechanistic view: methods that ignore negative feedback fail to induce reasoning behavior. However, this insight is bounded by the small model scale and lack of theoretical analysis explaining *why* clipping becomes unnecessary when advantages are properly normalized.

## Suggestions

- **Run multi-seed experiments (at least 3 seeds)** and report means with standard deviations. Perform significance testing on key comparisons (RGRA vs. GRPO).

- **Address the on-policy sampling cost:** Either measure wall-clock time for RGRA vs. GRPO, or discuss whether the claimed simplification has hidden computational costs from requiring fresh samples each update.

- **Scale validation:** Even one experiment at 3B or 7B would substantially increase confidence that findings transfer to scales where reasoning research typically operates. If hardware prohibits this, provide theoretical discussion of how gradient variance and policy drift scale.

- **Clarify the RGRA vs. RLOO relationship:** A brief comparison (even theoretical) to Ahmadian et al.'s formulation would establish novelty or acknowledge equivalence.

- **Add one ablation of KL penalty:** If RGRA without KL also works well, the "simplification" claim is stronger. If it fails, then RGRA's success relies on substituting one stabilizer (KL) for another (clipping).

---

## sh1hWO9RHo

- GT: Reject (avg 4.5)
- Predicted: N/A (5.7/10)
- Match: N/A

### Final Review

## Summary
The paper introduces the Agent GPA (Goal-Plan-Action) framework, a multi-dimensional evaluation paradigm that decomposes LLM agent assessment into specialized LLM-as-a-Judge metrics aligned with the agent's operational loop. The framework proposes six judges (Plan Quality, Plan Adherence, Logical Consistency, Execution Efficiency, Tool Selection, Tool Calling) to detect, localize, and categorize agent errors. Experiments on TRAIL/GAIA, an internal production dataset, and TRAIL/SWE-bench demonstrate that specialized judges achieve high human agreement (82-95% coverage), localize 86% of errors, and show strong inter-run consistency (Krippendorff's α > 0.7 for most metrics).

## Strengths
- **Practically motivated and structurally coherent framework:** The GPA taxonomy directly addresses limitations of monolithic or outcome-only evaluations by aligning assessment with the agent's operational loop (Goal-Plan-Action). The decomposition into interpretable dimensions provides actionable debugging signals rather than aggregate scores alone.
- **Strong empirical results on error localization and coverage:** The judges collectively capture 95% (267/281) of TRAIL-annotated errors on the test set with higher coverage on medium/high-impact errors (Tables 2, 5), and localize 86% of errors to specific trace spans—demonstrating concrete utility for targeted debugging.
- **Comprehensive evaluation methodology:** The paper reports coverage, F1/F2, localization rates, correlation with human scores, and Krippendorff's α for inter-rater reliability across multiple runs (Tables 3-7), providing a thorough assessment protocol that aligns with rigorous evaluation standards.
- **Demonstrated adaptability via GEPA optimization:** Section 4.1.5 shows that automated prompt refinement (GEPA) can match or exceed manually-tuned prompts, and cross-domain transfer to SWE-bench (Table 9) indicates promising generalizability with appropriate adaptation.

## Weaknesses
- **Goal Fulfillment judge is introduced but absent from experiments:** The abstract and Section 3 list Goal Fulfillment (GF) as one of five primary metrics, but GF does not appear in any experimental table. The paper offers no explanation for this omission, which is a significant gap given that goal fulfillment is the ultimate success criterion for agents. This inconsistency between framework description and empirical validation undermines completeness claims.
- **Limited validation across agent architectures and model families:** All primary experiments use Claude-4-Sonnet (with GEPA using Claude-Sonnet-4.5), and agent traces come from only two architectures (Open Deep-Research for GAIA, CodeAct for SWE-bench). Without testing across additional model families (e.g., GPT-4, open-source LLMs) or agent architectures, the framework's generalizability cannot be established. Appendix D mentions gpt-4o and gpt-4.1 but does not provide comparable experimental results.
- **Human-human agreement baseline is underreported:** Human annotator agreement (consensus rate 0.67-0.70) appears only in Appendix E, yet this baseline is critical context for interpreting LLM-human agreement. If humans agree only ~67% of the time, LLM judges achieving ~80% agreement may be operating within the inherent noise ceiling. This should be discussed prominently in the main text.
- **Dataset scale limits statistical robustness:** The primary test set has only 59 traces (TRAIL/GAIA), the internal production dataset has 17 traces, and SWE-bench test has 16 traces. Several error categories have very few examples (e.g., PQ has only 14 test-set errors; PA/PQ have n ≤ 2 for low-impact errors), making per-category performance estimates unreliable.
- **No intervention study validating the "targeted improvement" claim:** The paper's core motivation is enabling "targeted improvement of agent performance" (Abstract), but no experiment demonstrates that agents modified based on GPA diagnoses actually improve. This leaves the practical utility claim empirically unsubstantiated.
- **Execution Efficiency judge shows weak human alignment:** Table 4 shows EE's 3-point bucketed accuracy at only 0.356 on the test set (disagreeing with humans on ~64% of traces), which the paper attributes to EE "flagging errors not strictly related to efficiency" without detailed analysis. This weak alignment is underemphasized for a metric that features prominently in the framework.

## Nice-to-Haves
- **Ablation study on judge necessity:** An analysis testing whether all 6 judges are required or whether a subset achieves comparable coverage would help justify the computational overhead of running multiple specialized judges.
- **Cost and latency analysis:** No discussion of inference cost or scalability for running 6 LLM judges per trace—important for practical deployment considerations.
- **Guidance on judge selection for different debugging scenarios:** Given varying precision/recall trade-offs (TC is conservative, TS is liberal), a decision matrix for practitioners would enhance practical utility.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **"100% coverage claim is tautological/circular"** (Harsh Critic): While the error-to-dimension mapping was human-annotated, the empirical claim is that LLM judges can *detect* those mapped errors (95% detection rate), which is a meaningful empirical result. The taxonomy coverage claim is separate from detection coverage.
- **"Double-counting across judges inflates recall"** (Harsh Critic): The paper explicitly notes that errors can map to multiple judges by design (multiple failure dimensions for a single error). This reflects the framework's intent to capture multi-dimensional failures, not inflation.
- **"4 decimal places for small-n cells"** (Harsh Critic): Standard practice in ML papers; the paper appropriately acknowledges small sample sizes for low-impact PA/PQ errors.
- **"Meta-judge circularity for GEPA results"** (Harsh Critic): The meta-judge was validated against human agreement (159/198 vs. 177/198), providing reasonable assurance. While not ideal, this is a preliminary case study appropriately labeled as such.
- **"Baseline comparisons are incomplete"** (Harsh Critic): The paper does compare against TRAIL's LLM judge with and without control flow (Tables 2, 5), establishing a meaningful baseline. While additional baselines would strengthen the paper, this is a nice-to-have rather than a critical flaw.

## Novel Insights
The cross-GPA metrics agreement analysis (Appendix F) reveals genuinely interesting orthogonality: the six metrics show consistently low agreement (Jaccard 0.04-0.45) and weak correlations (negative for PQ-EE), confirming they capture distinct failure modes. This supports the core design principle that multi-dimensional evaluation is necessary—no single metric can capture the full spectrum of agent failures. Additionally, the finding that TC acts as a "conservative but accurate" judge while TS operates as a "high-recall specialist" suggests practical deployment strategies: use TS for comprehensive debugging and TC for automated processes requiring precision.

## Suggestions
- **Explicitly explain Goal Fulfillment's absence:** Add a clear statement in Section 4 explaining why GF was not evaluated experimentally (or add GF to the evaluation).
- **Move human-human agreement to main text:** Include the 0.67-0.70 human baseline in Section 4.1.3 or 4.1.4 to contextualize LLM-human agreement figures.
- **Add at least one intervention experiment:** Demonstrate that GPA-guided debugging improves agent success rates on held-out tasks, even on a small scale, to substantiate the "targeted improvement" claim.
- **Test with at least one non-Anthropic LLM:** Run the same evaluation protocol with GPT-4 or an open-source model to demonstrate framework generalizability beyond the Claude family.
- **Provide actionable judge selection guidance:** Add a table or flowchart recommending which judges to prioritize based on debugging objectives (e.g., high-recall discovery vs. high-precision automation).

---

## tswBfpkwHn

- GT: Reject (avg 5.0)
- Predicted: N/A (6.3/10)
- Match: N/A

### Final Review

## Summary

This paper provides the first theoretical analysis of in-context learning (ICL) training dynamics for Mamba models, specifically characterizing convergence, sample complexity, and robustness to additive outliers in prompts. By decomposing a one-layer Mamba into linear attention plus a nonlinear gating layer, the authors prove that the gating mechanism actively suppresses outlier-containing examples while inducing recency bias, enabling outlier tolerance approaching α → 1 (fraction of corrupted examples) when training corruption is proportional—versus α < 1/2 for linear Transformers. The theoretical results are validated through synthetic experiments and a real-world sentiment classification task.

## Strengths

- **Novel two-phase training dynamics analysis:** Lemmas 4–5 introduce a rigorous two-phase analysis of how the gating parameter w evolves during training, tracking the transition from initial outlier suppression to stable pattern selection. This provides the first mathematical foundation for understanding why gated SSMs behave differently from linear attention during ICL.

- **Mechanistic decomposition with empirical validation:** Corollary 1 (attention focuses on query-matching patterns) and Corollary 2 (gating suppresses outliers and decays exponentially with distance from query) offer interpretable theoretical predictions that are directly verified in Figures 3–4. The exponential decay of gating values for clean examples and near-zero values for outlier examples are visually compelling.

- **Provably superior robustness bound under controlled conditions:** Theorem 2 establishes that Mamba can tolerate α < min(1, p_a · ltr/lts) versus α < 1/2 for linear Transformers (Theorem 4). When p_a · ltr/lts > 1/2, Mamba provably outperforms linear attention. This difference is clearly demonstrated in Figure 2 across three outlier labeling schemes.

- **Transparent acknowledgment of position-sensitivity failure:** Table 1 honestly documents that Mamba achieves only 82.73% accuracy when outliers are closest to the query (CQ), compared to 93.96% for linear Transformers. The paper does not hide this genuine weakness and provides an initial mitigation (curriculum training in Appendix B.1).

## Weaknesses

- **Strong structural simplification of Mamba:** The analysis sets A = −I_m, removing Mamba's defining selective state expansion mechanism (Gu & Dao, 2023). The resulting model (Equation 3) is effectively gated linear attention. While this is standard for theoretical analysis and noted in the paper, it limits claims about "Mamba" specifically versus "gated linear attention architectures" broadly.

- **Test-time outlier generalization requires span overlap:** Condition (a) of Theorem 2 requires test outliers v_s'^* to be positive linear combinations of training outliers. This is more restrictive than "unseen outliers" suggests—outliers orthogonal to the training outlier span are not covered by the theory. The theory also requires outliers to be orthogonal to all relevant and irrelevant patterns (v_s^* ⊥ µ_j, v_s^* ⊥ ν_k), which represents an "easy" robustness case where outliers cannot mimic informative features.

- **Robustness bound circularity not prominently flagged:** The bound α < p_a · ltr/lts means α → 1 requires either p_a → 1 (training on nearly-entirely-corrupted data) or lts → 0 (very short test prompts). This creates a practical limitation—significant test-time robustness requires proportional training-time corruption—that is embedded in a condition rather than prominently discussed as a limitation.

- **Softmax Transformer comparison relegated to appendix:** Appendix Tables 3–5 show that softmax Transformers achieve robustness comparable to Mamba (99.40% vs. 99.73% in FQ at α = 0.5) and actually outperform it in CQ (99.28% vs. 82.73%). These findings directly complicate the paper's core message but appear only in the appendix. While the linear Transformer baseline is methodologically justified (Remark 6: "to isolate the effect of nonlinear gating"), the softmax results should be discussed in the main text to contextualize whether the advantage is specific to linear attention.

- **Lower bound on outlier magnitude:** Condition (ii) of Theorem 1 requires κ_a ≳ Vβ^{-4}, meaning outliers cannot be arbitrarily weak. The theory does not cover subtle data poisoning where corruptions have small magnitude—a realistic threat model.

## Nice-to-Haves

- Include standard deviations or confidence intervals for all experimental results (synthetic and real-world).

- Extend theoretical analysis to multi-layer Mamba architectures to bridge the gap between single-layer theory and three-layer experiments.

- Visualize training loss convergence curves to empirically validate the iteration complexity comparison (T_M vs. T_T).

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh claim that the model is "not Mamba":** The simplification A = −I is explicitly acknowledged and justified (citing Gu & Dao, 2023 Theorem 1) as a theoretical convenience. The model remains recognizable as a Mamba variant with specific parameterization, and the core insight about gating behavior is preserved. This is standard practice in theoretical deep learning.

- **Demand for softmax Transformer in the main comparison throughout:** While the appendix placement is suboptimal, the paper's stated goal is "isolating the effect of nonlinear gating" (Remark 6). A linear Transformer is the correct baseline for this controlled experiment. The issue is visibility, not methodology.

- **Criticism that no code is provided:** ICLR papers do not require code release. The experimental settings are described in sufficient detail to reproduce.

- **Demand for proof guides/diagrams:** The appendix proofs are dense but complete with standard techniques. Adding intuition boxes is a presentation preference, not a methodological flaw.

- **Criticism about the single-layer scope being insufficient:** Single-layer analysis is the prevailing standard in theoretical ICL work (cited Transformer papers: Zhang et al., 2023; Li et al., 2024a). Multi-layer extension is a natural future direction, not a current weakness.

- **Claim that the CQ failure "contradicts the robustness claim":** The robustness theorems assume random outlier placement (Definition 2). The CQ vulnerability is an empirically observed boundary case that the theory does not claim to cover. The paper acknowledges this in Table 1 and Remark 8.

## Novel Insights

The two-phase gating dynamics (Lemmas 4–5) reveal a temporal structure to robust ICL acquisition: Phase 1 (t ≲ η^{-1}β^{-2}κ_a^{-1}(1-p_a)^{-1}V) involves the gating parameter growing along outlier directions at rate O(−ηβ²tκ_a(1-p_a)^{-1}/V), while Phase 2 (t ≳ η^{-1}(1-p_a)^{-1}β^{-2}M₁) involves pattern selection along relevant directions. This sequential learning explains why training requires more iterations for Mamba than Transformers (T_M = Θ(ltr) · T_T)—the gating must first establish outlier suppression before reliable attention learning can proceed. The exponential gating decay (Corollary 2, equation 18: G_{h^{(j)}, lts+1} ≥ Θ(2^{-j})) provides a mechanistic explanation for Mamba's recency bias as a feature, not a bug: it naturally prioritizes temporally proximal clean examples, which aligns with standard ICL intuition that recent demonstrations are most informative.

## Suggestions

- Move the softmax Transformer comparison (Tables 3–5) from Appendix B.1 to a dedicated subsection in Section 4, discussing the implications for practical model choice. Explicitly state whether Mamba's advantage lies primarily in computational efficiency (linear complexity) versus robustness relative to softmax attention.

- Add a "Limitations" paragraph in the main text explicitly acknowledging: (1) the A = −I simplification removes Mamba's selectivity; (2) the test-time outlier generalization requires span overlap with training outliers; (3) the robustness bound's circular dependency between training and test corruption fractions; (4) the CQ positional vulnerability and its theoretical open status.

- Include a brief theoretical discussion of what happens when test outliers are orthogonal to the training outlier span, even if only to state that robustness guarantees do not extend and empirical investigation is left to future work.

---

## qSak1Hjfdq

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary

This paper formalizes the All-Day Multi-Scenes Lifelong Vision-and-Language Navigation (AML-VLN) problem, addressing catastrophic forgetting when adapting VLN agents across diverse scenes and degraded visual environments (low-light, scattering, overexposure). The authors propose Tucker Adaptation (TuKA), which represents multi-hierarchical navigation knowledge as a 4th-order tensor via Tucker decomposition, decoupling it into shared core components and scene/environment-specific expert factors. Coupled with a Decoupled Knowledge Incremental Learning (DKIL) strategy combining EWC-style regularization on shared subspaces with orthogonality constraints on task-specific experts, the resulting AlldayWalker agent demonstrates strong continual navigation performance on an extended Habitat benchmark with synthesized degradation and real-world validation.

## Strengths

- **Principled architectural innovation for multi-hierarchical knowledge:** TuKA's use of 4th-order Tucker decomposition to explicitly decouple shared navigation skills, scene priors, and environmental factors into structured tensor factors (G, U₁, U₂, U₃, U₄) is a technically coherent extension of MoE-LoRA approaches. The key insight—selecting rows from scene and environment expert matrices to reconstruct 2D weight updates—elegantly solves the dimension-alignment problem between high-order tensors and LLM backbone matrices (Eq. 3).

- **Comprehensive ablation isolating tensor structure from hierarchy:** The paper constructs ABC-LoRA, a hierarchical matrix baseline that mirrors the scene-environment branching without using higher-order tensors (Appendix I). The finding that TuKA outperforms ABC-LoRA (65% vs 55% avg SR) provides meaningful evidence that gains stem from tensor interactions rather than mere architectural branching, strengthening the core claim.

- **Strong empirical results with real-world validation:** AlldayWalker achieves 65% average SR across 24 tasks vs 52% for the best baseline (SD-LoRA), with forgetting rates of 11% vs 23%. The 55% SR on 6 held-out unseen tasks (vs ~40% for baselines) and real-world robot deployment (Table 5, Appendix F) provide external validity beyond synthetic benchmarks.

- **Contribution of practical benchmark:** Extending Habitat with physics-based degradation models (atmospheric scattering, low-light sensor noise, overexposure) and releasing code/videos adds utility to the embodied AI community for studying robust lifelong navigation.

## Weaknesses

- **Missing upper-bound baselines:** The paper does not report joint training (all tasks simultaneously) or per-task oracle LoRA (storing and loading the correct adapter) baselines. Without these, it is unclear whether 65% SR approaches the achievable ceiling or leaves significant headroom. Given that the paper explicitly dismisses "storing all adaptation weights" as trivial (§2), reporting this upper bound is essential to contextualize improvements.

- **Non-standard Fisher computation:** Appendix C states Fisher information is computed "using the first 10% of the data before adaptation to each task." Standard EWC computes Fisher *after* training on a task to identify parameters important for that task. Computing Fisher before training biases the estimate toward the prior model rather than the task-relevant parameters, potentially confounding comparisons with EWC-LoRA.

- **Undefined loss term in core formulation:** Equation 9 includes L_sk in the total loss, but this term is never explicitly defined in the paper. While L_ewc (Eq. 4), L_co (Eq. 7), and L_es (Eq. 8) are defined, L_sk appears only in Eq. 9 without clarification, impeding reproducibility.

- **No statistical significance reported:** Results report single-point metrics across 100 episodes per task without variance, standard deviation, or confidence intervals. With only one task ordering tested, statistical robustness of SOTA claims cannot be verified.

- **Inference efficiency unquantified:** While TuKA uses ~0.3M parameters, Tucker reconstruction (computing U₁·(G ×₃ U₃[s,:] ×₄ U₄[e,:])·U₂ᵀ at every layer) and explicit expert matching via CLIP feature retrieval introduce computational overhead. Latency, FLOPs, and memory footprint comparisons against standard LoRA are not provided—critical for real-time robotic deployment claims.

- **Expert retrieval mechanism underspecified:** Scene and environment experts are retrieved via cosine similarity between a single CLIP observation feature and stored feature banks (§3.4). It is unclear whether (a) the same feature reliably encodes both scene identity and lighting condition for independent retrieval, and (b) how matching degrades under novel or ambiguous observations. No ablation tests retrieval failure modes.

## Nice-to-Haves

- **Alternative tensor decompositions:** Comparing Tucker against CP or Tensor-Train decompositions would strengthen the justification for Tucker specifically. While the paper ablates 3rd- vs. 4th-order tensors, it does not compare alternative factorization strategies that may offer different expressivity/parameter trade-offs.

- **Multiple task orderings:** Evaluating across several random task sequences would address the well-known sensitivity of continual learning methods to ordering.

- **Analysis of forgetting correlation:** Quantifying how forgetting rates correlate with scene/environment overlap across tasks would directly validate the decoupling hypothesis.

## Removed Points

- *Title being "unwieldy"* — This is a style nitpick; the title accurately describes the contribution.
- *"Matrix-based representation inherently limited" being unproven* — The paper provides both theoretical motivation and empirical evidence (3rd vs 4th order ablation, ABC-LoRA comparison); this is standard contribution framing, not an unproven assertion.
- *Task-ID assumption during training being problematic* — This is standard in class-incremental/continual learning formulations; the paper explicitly states task IDs are available during training and not during inference.
- *Duplicate Figure 2 caption* — This appears to be a parsing artifact in the extracted text, not an actual paper issue.
- *Benchmark being entirely synthetic* — The paper includes real-world validation (Tasks 21–24 in Tables 1–2, plus unseen real-world generalization in Table 5).
- *Typos like FeeM vs FeeN* — Minor notation errors that do not affect technical correctness.

## Novel Insights

The DKIL strategy—applying EWC only to shared components (G, U₁, U₂) while using orthogonality constraints for task-specific experts (U₃, U₄)—represents a clean separation of concerns: shared subspace stability vs. task-specific disentanglement. This hybrid regularization design is broadly applicable to other multi-hierarchical continual learning problems beyond VLN. The finding that sharing the core tensor G and encoder U₂ contributes more than sharing the decoder U₁ (Table 3) suggests asymmetry in which components encode transferable vs. scenario-specific knowledge, offering intuition for future adapter designs.

## Suggestions

1. **Add joint training and per-task oracle baselines** to establish upper bounds and contextualize absolute performance.
2. **Report variance across multiple seeds and/or task orderings** to support statistical claims.
3. **Define L_sk explicitly** in Eq. 9 or replace it with the correct defined term for clarity and reproducibility.
4. **Provide inference latency and FLOP measurements** comparing TuKA vs. standard LoRA and MoE-LoRA baselines.
5. **Ablate expert retrieval robustness** by testing matching accuracy under noise or analyzing similarity score distributions during inference.

---

## 1E4Bltg6Xb

- GT: Accept (Poster) (avg 4.7)
- Predicted: N/A (3.9/10)
- Match: N/A

### Final Review

## Summary
The paper proposes Dynamics Feature Representation (DFR), a hierarchical framework for compressing global traffic dynamics into compact state representations for reinforcement learning-based dynamic path planning (DPP). DFR uses a pre-trained static shortest-path policy to extract a task-relevant subgraph (policy attention) and refines it via n-hop neighborhoods around the current agent position, reducing state dimensionality while attempting to preserve decision-relevant information.

## Strengths
- **Well-motivated problem formulation:** The trade-off between global dynamics (computationally prohibitive) and local dynamics (information incomplete) in RL-based path planning is clearly articulated. The hierarchical refinement from global to task-relevant to local features addresses a recognized practical challenge in state design for RL agents.

- **Substantial empirical efficiency gains:** DFR achieves significant reductions in planning time (up to 85.59% faster than All Dynamics baselines) while improving Success Rate and reducing Mean GAP across three urban road networks. Compactness Rates below 6% demonstrate effective dimensionality reduction without catastrophic performance loss.

- **Systematic ablation study:** The analysis of k (proportion of top-k shortest paths) and n (neighborhood depth) provides practical deployment guidance, including observations about performance saturation with moderate n and the complex impact of k that requires careful tuning.

## Weaknesses
- **Static prior vs. dynamic objective mismatch:** The policy attention mechanism extracts subgraphs based on static shortest paths (minimizing distance), but the DPP objective is to minimize travel time under dynamic congestion. When congestion patterns diverge from static topology—exactly the scenario DPP is meant to address—the fixed subgraph may exclude critical edges. The paper acknowledges distance is a "fundamental constraint" but provides no empirical analysis of failure cases where time-optimal paths substantially differ from distance-optimal paths.

- **Unproven theoretical claims:** Equations 6–8 assert that the compressed representation preserves optimality (using ≈) without formal bounds or empirical verification. The Predictive State Representations connection is invoked as theoretical grounding, but the sufficient conditions are neither established nor tested; it functions as conceptual motivation rather than a provable guarantee.

- **Baselines limited to All Dynamics variants:** The comparison is restricted to DQN, PPO, and GCN+DQN with and without DFR. There is no comparison to other state compression methods (attention-based GNNs, variational state abstractions, learned representations) that could isolate whether gains come from the specific policy attention mechanism versus generic dimensionality reduction.

- **Underspecified dynamics model:** The congestion factor β ∈ [0.1, 1.5] is presented without describing its temporal evolution, spatial correlation structure, or generation methodology. Without this specification, the experimental setup's realism cannot be assessed, and reproducibility is hampered.

- **Unreported graph sizes undermine scalability claims:** The paper claims applicability to "large-scale urban networks" and "massive urban networks" but does not report node/edge counts for the three test regions, making scalability claims unsubstantiated.

- **Missing statistical rigor:** No standard deviations or confidence intervals are reported for main results. The triangle area visualization (Figure 5) combines three incommensurable quantities (1−GAP, SR, 1−CR), which could obscure performance trade-offs.

## Nice-to-Haves
- Using classical k-shortest-paths algorithms (e.g., Yen's algorithm) instead of RL pre-training for policy attention would be simpler and provably correct; the paper does not justify why RL-based static policy learning is preferable to established graph algorithms.

- Analysis of how errors or suboptimality in the static policy π_d^* propagate to final dynamic policy quality would strengthen robustness claims.

- Evaluation under varying traffic volatility levels (mild congestion vs. severe incidents) would test robustness to the "unexpected events" cited in the Introduction.

## Removed Points
*These points are flagged to be removed, treat them with caution*

- **"Single source-goal pair per city":** The paper states "source and goal nodes are randomly sampled from a subgraph" and "each episode corresponds to a new scenario," indicating multiple source-goal pairs are used during training. The harsh critic's interpretation appears to misread the experimental setup.

- **"No comparison to non-RL baselines":** The paper explicitly scopes to RL methods (footnote 3: "advantages of RL-based approaches over traditional methods in DPP have been well established in the literature"). Criticizing the absence of classical planners is scope creep beyond the stated contribution.

- **"Deterministic MDP makes RL unnecessary":** The paper's contribution concerns state representation within RL. Questioning whether RL is the right tool targets the paper's premise rather than its execution, which falls outside evaluation scope.

- **"Temporal indexing overload":** The paper includes a clarifying footnote (fn. 2) distinguishing the two uses of t. While potentially confusing, this does not affect correctness.

- **"Triangle area visualization is non-standard":** While unusual, the visualization provides an intuitive summary and is accompanied by specific metric values. This is a presentation choice rather than a substantive flaw.

## Novel Insights
The key insight is that structural priors from static shortest-path topology can effectively compress dynamic state representations for RL agents. Rather than learning state abstractions end-to-end, DFR demonstrates that pre-computed subgraphs based on distance-optimal paths capture much of the decision-relevant information while dramatically reducing dimensionality. This suggests a broader design principle: for structured decision problems, injecting domain knowledge about likely action sequences (via k-shortest paths) can substitute for expensive representation learning. The trade-off is between compression efficiency and adaptability—when dynamics favor routes outside the static prior, the method's fixed subgraph becomes a liability.

## Suggestions
- **Quantify static-dynamic divergence cases:** Report how often and by how much the time-optimal path under actual congestion differs from the distance-optimal path, and analyze DFR performance specifically on these cases to establish failure boundaries.

- **Add representative state-compression baselines:** Include at least one alternative state abstraction method (attention-based pooling, autoencoder, random projection) to isolate whether gains come from policy attention specifically or from dimensionality reduction generally.

- **Report graph statistics:** Provide node counts, edge counts, and average degree for each test region to substantiate scalability claims.

- **Provide statistical measures:** Report mean ± standard deviation across multiple random seeds for main metrics (GAP, SR, planning time).

- **Document dynamics generation explicitly:** Specify how β is sampled over time, including any spatial/temporal correlation structure or lack thereof.

---

## xFo13SaHQm

- GT: Accept (Poster) (avg 6.5)
- Predicted: N/A (6.6/10)
- Match: N/A

### Final Review

## Summary

This paper identifies and formalizes the "copy-paste" artifact in identity-consistent image generation, where models excessively replicate reference images rather than synthesizing identities with natural variations. The authors contribute MultiID-2M, a large-scale paired multi-identity dataset; MultiID-Bench, a benchmark with a novel copy-paste metric (MCP); and WithAnyone, a FLUX-based model that uses paired training, a GT-aligned ID loss, and an extended-negative contrastive objective to mitigate copy-paste while preserving identity fidelity.

## Strengths

- **Principled problem formalization and metric design**: The paper rigorously defines the copy-paste artifact and introduces MCP (Eq. 2), which explicitly measures whether generated images skew toward the reference rather than ground truth. This addresses a fundamental evaluation gap where prior work reported only Sim(Ref), inadvertently rewarding trivial copying. Fig. 5 clearly demonstrates that baselines cluster along a fidelity-copying trade-off curve while WithAnyone deviates toward the desired upper-right region.

- **High-quality dataset curation**: MultiID-2M provides 500k paired multi-ID images with hundreds of diverse references per identity, filling a documented gap in multi-identity customization research. The pipeline (Sec. 3, Appendix C) includes ArcFace clustering, automated filtering, caption generation, and quality scoring—resulting in a resource likely to have lasting community value.

- **Clever training innovations**: The GT-aligned ID loss (Sec. 5.1, Appendix E.1) exploits flow-matching's velocity prediction to compute identity supervision at all noise levels without full denoising overhead, and the four-phase pipeline logically transitions from reconstruction to controllable generation. Table 3 shows Phase 3 paired tuning reduces copy-paste by ~30% while maintaining Sim(GT).

- **Comprehensive empirical validation**: The paper evaluates on both single-person and multi-person subsets, averages three face recognition models (Table 5), conducts user studies with correlation analysis (Table 7), and provides extensive qualitative comparisons. The ablation in Table 3 validates each component.

- **Reproducibility commitment**: Training hyperparameters, optimizer settings, and phase durations are explicitly documented (Appendix F.1), and the authors commit to open-sourcing model, data, and benchmark.

## Weaknesses

- **ArcFace dependency creates evaluation circularity**: The same ArcFace embedding model is used for dataset construction (identity clustering and assignment), training loss computation (GT-aligned ID loss and InfoNCE), and evaluation metrics (SimGT, SimRef, MCP). This bootstrap loop means the model is optimized to match ArcFace embeddings, then evaluated on ArcFace similarity—potentially inflating performance. Using an orthogonal face descriptor (e.g., a VLM-based face encoder) as a held-out evaluator would strengthen credibility.

- **Threshold inconsistency in dataset construction**: The main text (Sec. 3) states the ArcFace matching threshold is 0.4, while Appendix C.1 states it is 0.5. This discrepancy must be resolved for reproducibility.

- **Inference-time spatial control for multi-ID generation is unexplained**: The architecture uses GT bounding boxes to create attention masks during training (Appendix E), but the inference protocol for spatial assignment when GT is unavailable is not described. This is central to the "controllability" claim and should be clarified.

- **Non-celebrity generalization is insufficiently validated**: Appendix F.2 shows only four qualitative examples from OmniContext, claiming generalization to non-celebrity identities. Given that training data consists entirely of ~3k celebrities with skewed US/China distribution (Fig. 13b), proper quantitative evaluation on a held-out non-celebrity dataset is needed to support broad applicability claims.

- **MCP metric has uncharacterized edge cases and modest human correlation**: When reference and GT are similar (small θ_tr), the denominator in Eq. 2 approaches ε, causing large MCP swings—yet the distribution of θ_tr across test cases and sensitivity to ε choice are not analyzed. Additionally, Table 7 shows only 0.44 Spearman correlation with human copy-paste judgments, indicating the metric captures real artifacts but leaves substantial variance unexplained. Failure case analysis is missing.

- **GT-aligned ID loss behavior under pose mismatch is unexplored**: The loss uses GT landmarks to align the generated face. When generated pose differs substantially from GT pose (the very case where controllability matters), applying GT landmarks produces misaligned crops, corrupting embedding computation. The ablation in Fig. 7 shows noise-level behavior but not extreme pose discrepancy cases.

- **Training-inference landmark mismatch**: Training leverages GT landmarks for the ID loss, but inference must rely on predicted landmarks from generated images. The paper does not analyze how landmark detection errors on generated outputs affect identity consistency during deployment.

## Nice-to-Haves

- Hyperparameter sensitivity analysis for λ_ID and λ_CL (both fixed at 0.1) to clarify tuning requirements for practitioners.

- Dataset size ablation (training on 50k, 100k, 250k subsets) to demonstrate whether scale alone drives improvements.

- Failure case gallery showing where MCP disagrees with human judgment, improving transparency about metric limitations.

- Compute cost and training time reporting (8×H100 across 100k+ steps) for reproducibility and feasibility assessment.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Incremental architectural novelty"**: The reviewer critiques that the core model is "just" a cross-attention adapter. However, data-centric contributions and training recipe innovations are legitimate research contributions at ICLR; architectural novelty is not a requirement.

- **"GPT-4o evaluation anomaly requires removal"**: The paper already acknowledges GPT's prior knowledge issue in Sec. 6.1 and Appendix F.3, italicizing scores with a footnote. This is appropriately handled.

- **"Concurrent method comparison timing"**: The reviewer questions whether cited 2025 works were available at submission. Papers cited should be assumed available unless proven otherwise; this is an AI reviewer knowledge limitation, not an author misclaim.

- **"Licensing verification impossible for celebrity images"**: While valid as a practical concern, this requires external verification beyond reviewer knowledge. The ethics statement describes CC filtering; we cannot fact-check without external sources.

- **"Demographic imbalance requires disaggregated evaluation"**: While the dataset has US/China skew, requesting per-ethnicity performance analysis goes beyond the paper's scope. A benchmark paper need not solve all downstream fairness questions.

- **"Statistical significance testing for Sim(GT) improvements"**: The improvement margins are modest (0.460 vs 0.464 vs 0.452 in Table 1), but this is addressed in the trade-off analysis—the key result is not just Sim(GT) but the Sim(GT) vs MCP trade-off curve in Fig. 5.

- **"User study too small (10 participants)"**: While 10 participants is limited, the correlation analysis (Table 7) provides statistical grounding. This is a limitation worth noting but not a fatal flaw.

## Novel Insights

The copy-paste artifact formalization is the paper's key conceptual contribution. Prior work focused narrowly on maximizing Sim(Ref), implicitly encouraging models to memorize and replicate reference appearances. By introducing Sim(GT) as the primary metric and MCP as an explicit penalty for reference bias, the paper reframes the evaluation paradigm: identity fidelity should mean synthesizing the *person* across variations, not reproducing a specific reference photo. The insight that reconstruction-only training creates this failure mode—and that paired data with GT-aligned supervision can break the cycle—is well-motivated and empirically supported by the trade-off curve deviation in Fig. 5.

## Suggestions

1. Add a brief paragraph in Sec. 5 or Appendix E explaining inference-time spatial control for multi-ID generation (e.g., user-provided bounding boxes, automatic face detection with assignment, or attention-based spatial conditioning).

2. Resolve the 0.4 vs 0.5 threshold inconsistency between Sec. 3 and Appendix C.1, or clarify if different thresholds apply to different pipeline stages.

3. Include at least one quantitative evaluation on non-celebrity faces (e.g., FFHQ-held-out or OmniContext full benchmark) to substantiate generalization claims beyond four qualitative examples.

4. Add analysis of θ_tr distribution in MultiID-Bench and MCP sensitivity to ε to validate metric robustness under edge cases.

5. Discuss the ArcFace dependency explicitly in the limitations section, noting that evaluation circularity is partially mitigated by averaging ArcFace, AdaFace, and FaceNet but that an orthogonal evaluator would be preferable for future work.

---

## ZBhZT307xx

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (5.9/10)
- Match: N/A

### Final Review

## Summary

This paper presents a systematic analysis of verifiers used in reinforcement learning with verifiable rewards (RLVR) for mathematical reasoning. The authors demonstrate that rule-based verifiers suffer from significant false negative rates (14% on average, worsening with stronger policy models), and investigate model-based alternatives. They find that while model-based verifiers improve static classification accuracy, fine-tuned verifiers become vulnerable to reward hacking during RL training—a counterintuitive finding where improved static accuracy correlates with worse dynamic robustness. The study includes adversarial probing across 13 attack patterns and proposes a hybrid verifier architecture.

---

## Strengths

- **Timely and Relevant Problem Formulation:** The paper addresses a critical bottleneck in RLVR pipelines, which underpin recent reasoning models like DeepSeek-R1 and o1. The observation that false negative rates increase as policy models strengthen (Figure 2) is a genuinely important insight with direct implications for scaling.

- **Comprehensive Empirical Scope:** The evaluation spans multiple rule-based implementations (VERL, Qwen-Math, HuggingFace), off-the-shelf LLMs (1.5B–7B, short-CoT and long-CoT), fine-tuned verifiers (R1-Distill-Verifier, xVerify, general-verifier), and four datasets (Math, DeepScaleR, ORZ-Math, Skywork-OR1) plus WebInstruct-Verified for generalization.

- **Counterintuitive Finding with Practical Significance:** The discovery that fine-tuned verifiers with higher static accuracy (R1-Distill-Verifier-1.5B) are *more* susceptible to reward hacking than off-the-shelf models is surprising and actionable. This challenges the prevailing assumption that verification accuracy alone should guide verifier selection.

- **Oracle-Based Reward Hacking Detection:** Using GPT-4o as an independent oracle to track divergence between training reward and actual correctness during RL training (Figure 3, Right) provides a clean methodological contribution for diagnosing reward hacking that future work can adopt.

- **Systematic Adversarial Taxonomy:** The construction of 13 distinct hacking patterns (empty symbols, gibberish, adversarial prefixes, etc.) and systematic evaluation across verifiers (Table 3) provides actionable guidance—showing that discriminative verifiers (xVerify) are significantly more robust than generative ones.

---

## Weaknesses

- **Single Policy Model for RL Experiments:** All RL training experiments use Qwen2.5-7B-Base exclusively. Whether the reward hacking phenomenon generalizes to other model families, sizes, or RL algorithms (beyond GRPO) remains unvalidated. Given that the core claims about verifier reliability depend on policy-verifier dynamics, this limits confidence in generalizability.

- **Limited Statistical Rigor:** The main results (Table 2) report single-run evaluations for most benchmarks (only AIME24/AMC23 use Avg@32). The claimed "2.3 point improvement" lacks confidence intervals. RL training is known to exhibit high variance, making single-run results difficult to interpret reliably.

- **GPT-4o Used in Multiple Roles with Potential Circularity:** GPT-4o serves as: (1) the ground-truth annotator for the static evaluation dataset (§3.1), (2) the training data annotator for R1-Distill-Verifier-1.5B (Appendix K), and (3) the oracle judge during RL training (§5.2). For the trained verifier specifically, training targets and evaluation oracle share the same source, potentially inflating perceived quality. The paper does not verify that GPT-4o itself is robust to the hacking patterns identified.

- **Unexplained Mechanism for Fine-Tuning Vulnerability:** The paper demonstrates that R1-Distill-Verifier-1.5B is more hackable than its base model but provides no mechanistic explanation. Does fine-tuning cause overfitting to specific output patterns? Does it reduce semantic understanding? Without this analysis, the finding remains empirical rather than principled, limiting its utility for designing robust verifiers.

- **xVerify Not Tested in RL Despite Strong Robustness:** Table 3 shows xVerify achieves near-zero attack success rates (0.0–0.4%), yet no RL training experiment uses xVerify as the verifier. If xVerify is both accurate and robust, demonstrating its RL performance would significantly strengthen the paper's practical guidance. The absence is unexplained.

- **No Mitigation Strategies Tested:** The paper concludes with diagnostic findings but no experiments on potential solutions. Simple interventions—adversarial fine-tuning, input sanitization, ensemble voting, or using discriminative verifiers as safety filters—are not evaluated, leaving practitioners without actionable remediation paths.

- **Incomplete Cross-Dataset Results:** Table 8 (WebInstruct-Verified) shows only the HF verifier baseline. The R1-Distill-Verifier and general-verifier conditions are omitted from the table despite being discussed in the text. This fragmentation makes cross-dataset comparison difficult.

---

## Nice-to-Haves

- **RL Experiments with Discriminative Verifiers:** Run full RL training with xVerify to confirm whether its static robustness translates to dynamic training stability.

- **Human Validation of Oracle Labels:** Manually verify a sample of GPT-4o's "oracle" judgments on the hacked responses to confirm the oracle itself isn't being fooled by the same patterns.

- **Analysis of Why Fine-Tuning Increases Vulnerability:** Investigate whether fine-tuned verifiers develop surface-level token biases (e.g., toward `\boxed{1}` or reasoning-chain patterns) that adversarial inputs exploit.

- **Multiple Policy Model Sizes:** Test whether a stronger policy model (e.g., 32B) can hack the currently "robust" hybrid setup to establish robustness boundaries.

- **Compute-Overhead Analysis:** Quantify the training latency overhead of model-based verifiers in the hybrid setup to assess practical viability.

---

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **Abstract statistic context:** The harsh critic argued the 86% recall figure lacks context. However, Table 4 shows VERL achieves 0.92, 0.86, 0.89, and 0.78 across datasets—averaging to ~0.86 for that specific verifier. The paper provides this breakdown. The abstract's framing is defensible.

- **Conditional evaluation criticism:** The harsh critic suggested Table 1's conditional evaluation setup (only on rule-based failures) is misleading. The paper explicitly states this design choice in §3.3, and Table 5 provides global performance. This is transparent methodology, not deception.

- **Adversarial dataset size (471 samples):** The probing study uses 471 samples, which the harsh critic called "quite small." For a systematic vulnerability analysis across 13 patterns, this sample size is reasonable—the goal is pattern discovery, not statistical power.

- **Precision-recall framing:** The suggestion that "high precision at the cost of recall" is poorly framed because precision is high by design is technically correct but tangential. The paper's framing accurately describes the empirical finding.

---

## Novel Insights

The paper's most distinctive insight is the **accuracy-robustness mismatch for fine-tuned verifiers**: training a verifier to improve classification accuracy can *increase* its susceptibility to reward hacking during RL optimization. This contradicts the intuitive assumption that better static performance translates to better dynamic performance. The mechanism appears to be that fine-tuning overfits verifiers to surface patterns that adversarial policies can exploit (gibberish, empty symbols, adversarial prefixes), whereas off-the-shelf models retain more robust representations. Additionally, the finding that discriminative verifiers (xVerify) are substantially more robust than generative ones—while achieving comparable accuracy—suggests that output format (direct judgment vs. chain-of-thought reasoning) significantly impacts vulnerability. The hybrid architecture (rule-based first, model-based second) offers a practical partial solution but does not eliminate the fundamental tension between verification flexibility and adversarial robustness.

---

## Suggestions

1. **Add at least one additional policy model configuration** (different size or family) to the RL experiments to establish generalizability.

2. **Run and report RL training with xVerify** to demonstrate whether discriminative robustness transfers to dynamic settings, addressing the most notable gap between static probing and practical deployment.

3. **Include 2–3 sentences in the limitations section** explicitly acknowledging: (a) the single-policy-model scope, (b) the multiple roles of GPT-4o, (c) that short-answer mathematical verification may not generalize to open-ended reasoning.

4. **Add confidence intervals or multiple seeds** for at least the primary benchmark metrics to address statistical reliability concerns.

5. **Test one simple mitigation** (e.g., input length filtering or adversarial pattern pre-screening) to transform the paper from purely diagnostic to partially prescriptive.

---

## WhO6Km5Rku

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (3.4/10)
- Match: N/A

### Final Review

## Summary

QubitCache proposes a KV-cache compression framework that shifts from discrete token eviction to continuous probabilistic preservation of attention patterns. The method retains ~15% of tokens classically (anchors, recent, and high-attention tokens) while encoding the remaining tokens' attention distributions into quantum-inspired states, using soft attention reconstruction via inverse-distance-weighted interpolation during inference. The core contribution—using probabilistic attention weights rather than binary decisions for evicted tokens—is interesting, but the paper's quantum framing and several empirical claims require careful scrutiny.

## Strengths

- **Problem reframing is well-motivated**: The paper correctly argues that attention patterns between tokens encode essential relational information, and that binary eviction decisions irreversibly discard this structure. Appendix A.2 provides reasonable grounding from attention interpretability literature.

- **Attention-based token selection is empirically validated**: Table 4 ablation shows that removing attention-selected critical tokens causes a 20.4% F1 drop, while removing position-based anchors/recent tokens causes minimal degradation. This directly validates the core hypothesis that attention patterns—not position—determine semantic importance.

- **Qualitative analysis shows reduced hallucination**: Tables 6-9 demonstrate that QubitCache produces fewer factual hallucinations and less topic drift compared to StreamingLLM and H2O on summarization tasks, consistent with preserving contextual relationships.

- **Practical memory savings demonstrated**: The method achieves 7× compression (15% retention) while maintaining 92-97% of baseline performance across multiple models and benchmarks.

## Weaknesses

- **"Beyond classical information-theoretic limits" claim is misleading**: The abstract claims logarithmic compression "beyond classical information-theoretic limits," but all experiments use classical Qiskit simulation on GPUs. A 9-qubit statevector requires storing 2⁹ = 512 complex amplitudes classically—there is no information-theoretic advantage from quantum formalism when running on classical hardware. This claim should be retracted or precisely qualified.

- **Missing proof for bounded error claim**: The abstract states "We prove QubitCache preserves rank-r attention structure with bounded reconstruction error," but no theorem, lemma, or formal derivation appears in the paper or appendix. This is an unsupported theoretical claim.

- **Quantum encoding provides minimal benefit**: The ablation (Table 4) shows "Full QubitCache" (0.491) vs "No Quantum" (0.472)—only a 3.9% gain. More strikingly, "Random + Quantum" (0.335) vs "Random No Quantum" (0.334) shows essentially zero difference (0.3%). This strongly suggests the performance gains come from attention-based token selection and the soft-weighting/interpolation mechanism, not from quantum encoding. A proper classical soft-attention baseline would isolate this, but is absent.

- **Unfair compression ratio comparison**: Primary baselines (H2O, ScissorHands) operate at 50% retention (2× compression) while QubitCache operates at 15% retention (7× compression). The headline "15-25% F1 improvement" compares methods at different compression ratios. A fair evaluation would fix compression ratio and compare methods at equivalent settings.

- **No latency or throughput analysis**: For a method targeting production deployment, the absence of wall-clock inference time, tokens-per-second, or computational overhead metrics is a significant gap. Running Qiskit statevector simulation during autoregressive generation could introduce substantial latency.

- **"No Quantum" ablation is underspecified**: The paper does not clarify what "No Quantum" actually implements—does it use the same IDW interpolation with uniform weights? With attention-based weights computed classically? Without this clarification, the 3.9% improvement cannot be attributed to quantum encoding.

## Nice-to-Haves

- **Classical soft-attention baseline**: Implement identical token selection with purely classical soft-weighting (normalized attention scores directly) to isolate any genuine quantum benefit.

- **Matched compression ratio experiments**: Report QubitCache at 50% retention vs baselines at 50% retention for fair comparison.

- **Latency/throughput metrics**: Include inference speed benchmarks to evaluate practical deployment viability.

- **Error propagation analysis**: Analyze how probabilistic reconstruction errors accumulate over long sequences (100K+ tokens), which is the stated use case.

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **Claim that citations are mischaracterized**: Reviews claim Choromanski et al. (2020) doesn't support attention topology claims—this requires external verification beyond what can be determined from the paper itself.

- **Table formatting nitpicks**: Duplicate paragraph in Section 4.3 and confusing table organization—these are proofreading issues, not substantive weaknesses.

- **PG19 metric labeling**: While the reviewer notes PG19 is typically perplexity, the paper may be reporting F1 for a downstream task variant; this is unclear but not definitively wrong.

## Novel Insights

The most important insight—partially obscured by the quantum framing—is that soft probabilistic attention reconstruction for evicted tokens can preserve relational structure better than binary eviction. The ablation data strongly suggests that attention-based token selection (the classical component) drives nearly all performance gains, while the quantum formalism contributes minimally. This implies a simpler classical method—keeping attention-weighted tokens plus interpolating values from preserved neighbors—may achieve comparable results without quantum simulation overhead. The paper's theoretical claims about "rank-r preservation" and "bounded error" remain unproven, and the actual mechanism of benefit appears to be classical smoothing rather than quantum information processing.

## Suggestions

1. **Retract or qualify the information-theoretic claims**: Remove "beyond classical information-theoretic limits" unless running on actual quantum hardware, and either provide the missing rank/error proof or reframe the claim as empirical observation.

2. **Add a proper classical soft-weighting baseline**: Define precisely what "No Quantum" implements and compare against a classical method that uses identical soft weighting derived from attention scores directly.

3. **Report matched-compression-ratio experiments**: Show performance when all methods use the same retention rate.

4. **Include inference latency metrics**: Measure and report tokens-per-second and wall-clock time relative to FullKV and baselines—essential for assessing practical viability.

---

## Ksvv8x00eo

- GT: Withdrawn (treated as Reject) (avg 3.5)
- Predicted: N/A (5.9/10)
- Match: N/A

### Final Review

## Summary

CaTS-Bench introduces the first large-scale, multimodal benchmark for context-aware time series captioning, comprising 20k samples derived from 11 real-world datasets across climate, health, demographics, and commerce domains. Each sample includes a numeric series segment, rich metadata, a line-chart visualization, and reference captions generated through an oracle-LLM pipeline with extensive validation (manual checks, human indistinguishability studies, diversity analyses). The benchmark also provides 460 diagnostic multiple-choice questions and proposes two novel numeric fidelity metrics for evaluation.

## Strengths

- **Rigorous multi-pronged validation of semi-synthetic captions:** The paper validates oracle-generated captions through three complementary studies—manual verification achieving 98.6% accuracy on statistical and trend claims (Section 3.2, Table 9), a blind human study showing 41.1% detection accuracy (near random), and embedding-based diversity analysis showing only 2.3% of caption pairs exceed 0.95 cosine similarity (Section H.4). The paraphrasing robustness experiment (Table 11) demonstrates that model rankings remain stable across different ground-truth linguistic styles (Spearman ρ=0.9266), confirming that evaluation captures semantic content rather than stylistic alignment.

- **Task-specific evaluation metrics addressing numeric fidelity:** Beyond standard N-gram metrics, the paper introduces Statistical Inference Accuracy and Numeric Score (with 5% tolerance) to separately measure numeric hallucination vs. omission. This provides actionable diagnostics for where models fail—whether by inventing incorrect statistics or omitting key values (Section 3.5, Table 4).

- **Key empirical finding on visual modality neglect:** The visual ablation experiments (Figure 4, Table 15) reveal that removing the plot image causes negligible performance drops or even slight gains for most VLMs. The attention analysis (Figure 7, Appendix I.2) shows models focus on axis labels and titles rather than trend lines. Alternative visual encodings (GAFs, recurrence plots) also fail to improve results (Table 16), isolating the bottleneck to model integration rather than visualization design.

- **Comprehensive reproducibility provisions:** Full temporal split strategies, exact cropping ranges (Appendix C), detailed finetuning hyperparameters (Table 6), complete prompt templates (Appendix N), explicit metric definitions with tolerance justifications (Appendix F), and variance validation across multiple runs (Figure 6) are all provided. Human baseline results for Q&A tasks are included.

## Weaknesses

- **Gemini 2.0 Flash serves simultaneously as oracle and evaluated baseline:** The semi-synthetic (SS) ground-truth captions are generated by Gemini 2.0 Flash, which is also evaluated as a baseline model (Table 3). When Gemini is evaluated against its own generated references, the comparison reflects self-consistency rather than independent captioning quality. While the paper mitigates this with human-revisited (HR) captions and paraphrasing experiments, the circularity should be explicitly acknowledged in the main text—not only in Appendix A—so readers can interpret SS-column results appropriately.

- **Small Q&A test sets without confidence intervals:** The Q&A evaluation uses only 40 questions per comparison sub-task (amplitude, peak, mean, variance) and 100 questions per matching task. At these sample sizes, confidence intervals are approximately ±8–10%, yet no error bounds or statistical significance tests are reported. Claims about models performing "near random chance" on plot matching (Section 4.2) need quantified uncertainty to be credible.

- **Apparent experimental anomaly for QwenVL finetuning:** In Table 8, QwenVL's finetuned results are identical to its pretrained results across all metrics (DeBERTa F1=0.643, ROUGE-L=0.249, Numeric=0.504). This suggests a potential bug (wrong checkpoint loaded or configuration error) and requires clarification, as other models show clear finetuning gains.

- **Limited scale of human-revisited subset:** Only 579 samples (~14% of the test set) receive human revision, restricting statistical power for fine-grained conclusions about model performance against genuinely human-authored references. The paper relies primarily on semi-synthetic ground truth with the attendant oracle concerns.

- **Potential sample correlation from window cropping strategy:** Random windows are cropped from the same source time series entities. Two overlapping or nearby windows from the same city's air quality data, for example, may produce highly correlated samples. While the temporal train/test split prevents leakage, within-split correlation could inflate apparent performance. The paper does not quantify overlap frequency or its potential impact.

## Nice-to-Haves

- **Sensitivity analysis for numeric tolerance threshold:** Testing model rankings at tighter tolerances (1%, 2%) alongside the proposed 5% would demonstrate metric robustness and reveal whether high numeric scores reflect precise reasoning or approximate guessing.

- **Confidence intervals for Q&A results:** Given small per-task sample sizes, reporting standard errors or binomial confidence intervals would strengthen claims about model capabilities on reasoning tasks.

- **Cross-architectural attention analysis:** The visual grounding analysis is restricted to LLaVA. Extending to one additional architecture (e.g., Qwen-VL or Idefics) would strengthen the conclusion that visual neglect is a general VLM limitation.

- **Expanded human validation with domain experts:** The human study uses university students without domain expertise. Validation by climate scientists, public health analysts, or financial domain experts would strengthen claims about caption quality for "high-stakes sectors."

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Large-scale" claim disputed via TACO comparison:** The critic notes TACO has 2.46B timesteps vs. CaTS-Bench's 570k. However, Table 1 shows TACO is synthetic/templated while CaTS-Bench is multimodal with real-world sources. Scale is large relative to prior TSC benchmarks specifically. The differentiation is meaningful.

- **BLEU scores "extremely low" criticism:** The critic states BLEU values of 0.02–0.13 are "extremely low by any NLP standard." This misunderstands captioning evaluation—low BLEU is typical for open-ended generation where multiple valid outputs exist. The paper appropriately uses multiple complementary metrics.

- **Missing medical, industrial, financial domains as weakness:** This is scope creep. The paper includes 11 diverse real-world datasets. Demanding specific additional domains evaluates the paper against an unstated contribution.

- **"Author-as-annotator bias" as separate weakness:** This is acknowledged in Appendix A as a limitation. The human-revised captions were edited by the authors, which the paper explicitly notes.

- **Cross-domain generalization experiments missing:** The paper's stated contribution is a benchmark, not demonstrating cross-domain generalization. This is a suggestion for future work, not a weakness of the current contribution.

- **Pure LLM vs. VLM baseline suggested:** The spark finder requests this comparison, but it already exists in the paper. Section 4.3 and Table 15 show "L (text-only)" vs. "VL (vision-language)" performance deltas. The paper directly addresses this.

- **Temporal bias in test sets as flaw:** Using the final 20% for testing is standard practice for time series to simulate forecasting scenarios. This is a methodological choice, not a weakness.

## Novel Insights

The most significant insight from the evaluation is the **systematic visual modality neglect in current VLMs for time series reasoning**. Despite being provided with both raw numeric sequences and line-plot visualizations, models show negligible performance gains from visual input—sometimes performing better without it. The attention analysis reveals that when models do attend to visual elements, they focus on textual annotations (axis labels, titles) rather than the line trends themselves. This finding has implications beyond time series: it suggests that current VLM architectures may be heavily dependent on text-reading capabilities rather than visual pattern recognition for structured data. The paper's GAF and recurrence plot experiments further isolate this as a model integration limitation rather than a visualization choice. This should inform future VLM design for scientific and analytical domains.

## Suggestions

- **Add explicit acknowledgment in Section 4 or the experimental setup** that Gemini 2.0 Flash serves as both oracle and baseline, explaining how to interpret SS-column results and why HR-column results provide an independent evaluation. This transparency strengthens rather than weakens the contribution.

- **Investigate the QwenVL finetuning anomaly:** Verify whether the identical pretrained and finetuned scores represent a configuration error or genuine finding. If confirmed as error, correct the table; if genuine, explain why this model shows no finetuning benefit.

- **Report confidence intervals for Q&A tasks:** Even a simple binomial confidence interval (e.g., Wilson interval) would contextualize accuracy claims on small test sets.

---

## cZFgsLq8Gs

- GT: Accept (Poster) (avg 4.0)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary

DeepScientist presents an LLM-based autonomous research system that models scientific discovery as a goal-directed search process, using a persistent Findings Memory and UCB-inspired selection to balance exploration and exploitation. The system achieved measurable improvements over human-designed SOTA baselines on three AI research tasks (Agent Failure Attribution, LLM Inference Acceleration, AI Text Detection) within one month of computation.

## Strengths

- **Clear architectural contribution:** The three-stage workflow (Strategize & Hypothesize, Implement & Verify, Analyze & Report) with a cumulative Findings Memory provides a principled structure for multi-cycle research. The system explicitly learns from both successful and failed attempts, addressing a key limitation of prior AI Scientist systems that operated in isolation.

- **Empirical improvements on contemporary baselines:** The paper demonstrates genuine SOTA improvements: +7.9% AUROC on AI Text Detection (RAID benchmark), +183.7% accuracy on Agent Failure Attribution, and +1.9% throughput on LLM Inference Acceleration. The baselines are recent, competitive methods (ICML 2025 Spotlight, ACL 2025 Outstanding, ICLR 2024 Best Paper).

- **Transparent failure analysis:** The paper honestly reports that ~5,000 ideas yielded only 1,100 implemented trials and 21 progress findings (~0.42% success rate), with 60% of failures attributed to implementation errors. This candid bottleneck analysis provides valuable insight into the current limitations of autonomous research systems.

- **Progressive discovery demonstration:** The AI Text Detection trajectory (T-Detect → TDT → PA-TDT) shows evidence of genuine methodological building: each method builds on limitations identified in the previous one, rather than random recombination. The t-SNE visualization of the conceptual search space (Figure 5) supports the claim of directed exploration.

## Weaknesses

- **Misleading "Bayesian Optimization" framing:** The paper uses standard BO terminology (surrogate model, acquisition function, UCB) but the surrogate is simply an LLM prompted to produce integer scores (0–100) for utility, quality, and exploration. There is no probabilistic posterior, no uncertainty quantification, and no calibration against outcomes. Calling this "Bayesian Optimization" without these properties will mislead readers familiar with the formal literature. The system should be described as "UCB-inspired selection with LLM-based scoring" rather than proper BO.

- **Inconsistent autonomy claims:** The abstract claims "fully autonomous scientific discovery," but Section 4 states "three human experts supervise the process to verify outputs and filter out hallucinations," and Appendix F documents a secondary verification step because "approximately 50% of initial implementation attempts failed to complete." The paper does not quantify the extent of human intervention (number of hallucinations filtered, veto rate, time commitment), making the actual degree of autonomy unclear.

- **Unfair selection ablation:** The paper claims "without selection, randomly sampling 100 ideas yields success rate effectively zero" compared to the actual ~1,100 selected ideas. This compares different sample sizes, confounding selection quality with trial count. A fair ablation would compare the same number of randomly selected vs. UCB-selected ideas.

- **Surrogate model calibration not validated:** The UCB selection depends on the LLM surrogate's scores (v_u, v_q, v_e), but the paper never reports the correlation between these scores and actual experimental outcomes. Without demonstrating that the surrogate predicts value, the selection mechanism's effectiveness remains unverified.

- **Statistical rigor insufficient:** No confidence intervals, standard deviations, or significance tests are reported for benchmark improvements. The 1.9% throughput gain and 7.9% AUROC improvement lack error bounds, which is concerning given the stochastic nature of LLM-based experiments.

- **"Near-linear scaling" claim is statistically weak:** The scaling experiment (Figure 6) shows only 5 data points (1, 2, 4, 8, 16 GPUs) with high variance expected from stochastic discovery. Claiming a "near-linear relationship" from this sparse data is premature.

- **"3 years vs 2 weeks" comparison is misleading:** Figure 1 compares human research timeline (calendar years) with DeepScientist's compute time, but human researchers did not have access to 20,000 GPU hours. The efficiency claim requires a compute-matched baseline.

- **No ablation of core hyperparameters:** The UCB weights (w_u = w_q = κ = 1) are set without justification or sensitivity analysis. The retrieval count K = 15 is similarly unvalidated.

## Nice-to-Haves

- Analysis of failed hypotheses beyond the 60% implementation errors — understanding why scientifically plausible ideas still fail would strengthen the system's scientific reasoning claims.

- Broader task evaluation beyond NLP/AI domains to validate the "scientific discovery" framing across domains with different feedback characteristics.

- Open-source the "Analyze & Report" module for full reproducibility, or quantify the human effort required for paper generation.

## Removed Points

These points are flagged to be removed, treat them with caution:

- "The title is redundant" — This is a minor stylistic nitpick not relevant to scientific merit.

- "Human reviewers were not domain experts in the specific tasks" — The reviewers were LLM researchers with ICLR reviewing experience, which is appropriate for evaluating AI-generated research papers. This criticism demands expertise beyond what's reasonable.

- "Evaluation using DeepReviewer-14B is circular due to author overlap" — This IS disclosed in the paper. While the circularity concern has merit, the paper includes both automated evaluation and separate human expert review, mitigating this concern.

- "Citation error for Wolters et al." — The paper uses Wolters et al. for a retrieval model; this appears to be a legitimate reference to a compute-in-memory architecture paper for efficient retrieval, not necessarily an error.

- "Micronano-DeepScientist lacks proper baseline comparison on ALGOTUNE" — This is Appendix material, not central to the main claims. The concern about missing AlphaEvolve comparison is valid but peripheral.

## Novel Insights

The paper's honest accounting of the discovery funnel (5,000 ideas → 1,100 implementations → 21 progress findings → 5 papers) reveals that autonomous research systems currently succeed at ideation but struggle at execution and validation. The 60% implementation failure rate suggests the bottleneck has shifted from hypothesis generation to reliable code execution — a finding with implications for how the field should prioritize future development (stronger execution agents vs. smarter ideation mechanisms). The progressive discovery trajectory in AI text detection, where each method explicitly addresses the limitations of its predecessor (T-Detect's heavy-tailed normalization → TDT's wavelet analysis → PA-TDT's phase congruency), provides evidence that the Findings Memory mechanism genuinely guides exploration rather than merely logging random trials.

## Suggestions

- Run a proper selection ablation: compare N UCB-selected ideas against N randomly selected ideas (where N is the actual number selected) to isolate the contribution of the selection mechanism.

- Quantify surrogate calibration: report Pearson/Spearman correlation between LLM surrogate scores (v_u, v_q, v_e) and actual experimental outcomes for a representative sample of implemented ideas.

- Add confidence intervals or run multiple seeds: the stochastic nature of LLM-based discovery requires statistical significance testing, especially for small-percentage improvements.

- Precisely bound human intervention: report the number of hallucinations filtered, human override frequency, and estimated human hours to clarify the true autonomy level.

- Replace "Bayesian Optimization" terminology with "UCB-inspired selection" or validate the probabilistic properties that would justify the BO framing.

---

## iaoAKDRAJQ

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.1/10)
- Match: N/A

### Final Review

## Summary
This paper extends the theory of adaptive smoothness from convex to nonconvex optimization, showing that it governs convergence rates for a broad class of adaptive optimizers (Adam, AdaGrad, one-sided Shampoo) with well-structured preconditioners. The authors prove that adaptive smoothness enables Nesterov acceleration in convex optimization—yielding $O(\Lambda_H/T^2)$ rates—and introduce "adaptive gradient variance" to obtain dimension-free stochastic convergence guarantees, contrasting with dimension-dependent lower bounds under standard variance.

## Strengths
- **Unified nonconvex analysis with non-commutativity handling:** The extension of convergence analysis to general (non-diagonal) well-structured preconditioner sets via the novel matrix inequality in Lemma 3.3 is a substantive technical contribution. The explicit $\log d$ factor for non-commutative cases, compared to the commutative (diagonal) case, correctly identifies the technical overhead from matrix non-commutativity.

- **Clear algorithmic benefit with matching lower bounds:** The separation result—adaptive smoothness enables $O(1/T^2)$ acceleration (Theorem 4.3) while standard $\ell_\infty$ smoothness cannot beat $\Omega(1/T)$ (citing Guzmán & Nemirovski 2015)—is a concrete theoretical contribution. The parallel separation in stochastic settings (Theorems 4.5 and 4.7) shows dimension-free rates are achievable under adaptive variance but not under standard variance, with matching lower bounds for SignGD.

- **Technical novelty in Lemma 3.3 and Lemma C.1:** The matrix inequality bounding $\mathbf{V}_t^{-1/2}(\mathbf{V}_t^2 - \beta \mathbf{V}_{t-1}^2)\mathbf{V}_t^{-1/2}$ using operator concavity of the matrix logarithm appears novel and enables the general preconditioner analysis.

## Weaknesses
- **Logarithmic factor tightness is unaddressed:** The $\log d$ factor in convergence rates for non-commutative preconditioner sets (Theorem 3.1, Eq. 27) lacks discussion of whether this is a fundamental limitation or a proof artifact. Without a matching lower bound, the optimality of the noncommutative analysis remains open.

- **The lower bound in Theorem 4.7 is algorithm-specific:** The dimension-dependent lower bound applies specifically to SignGD with momentum under standard $\ell_1$ variance. This does not establish that *all* algorithms under standard variance must incur dimension dependence—only that NSD/SignGD does. The paper should clarify this scope limitation and whether an information-theoretic lower bound remains open.

- **Quantitative comparison with concurrent work is incomplete:** The paper states its rate is "strictly better" than Kovalev & Borodich (2025) due to the relationship between standard and adaptive smoothness, but the comparison remains qualitative. An explicit inequality or table showing the relative magnitudes of constants in both rates would strengthen the claim.

- **Conditions for acceleration benefit lack explicit demonstration:** The acceleration result requires $\Lambda_H(f) \leq \sqrt{L_{\|\cdot\|_H}(f) T}$ to beat the non-accelerated NSD rate. Since Proposition 2.5 gives $\Lambda_H \leq d \cdot L_{\|\cdot\|_H}$, the crossover requires $T > \Lambda_H/L_{\|\cdot\|_H}$. The paper would benefit from a worked example (even synthetic) demonstrating when adaptive smoothness genuinely yields computational advantage over standard methods.

- **Practical interpretability of adaptive variance (Definition 4.1) is limited:** The definition requires minimization over all $\mathbf{H} \in \mathcal{H}$ with $\text{Tr}(\mathbf{H}) \leq 1$ of uniform variance bounds, which may be difficult to verify or estimate for practical problems. While Proposition B.10 shows it is weaker than bounded covariance, no concrete example computes $\sigma_{\mathcal{H}}$ for a representative optimization landscape.

## Nice-to-Haves
- Empirical validation on synthetic functions demonstrating the theoretical rate separations would strengthen practical relevance but is not essential for a theory-focused contribution.
- Discussion of when adaptive smoothness constants $\Lambda_H(f)$ are substantially smaller than $d \cdot L_{\|\cdot\|_H}(f)$ in deep learning settings would help connect theory to practice.

## Removed Points
These points are flagged to be removed, treat them with caution:
- *Claim that examples for Shampoo/Muon are missing:* The paper explicitly covers this on page 5 ($\mathcal{H} = \mathcal{S}_+^d \otimes I_d$ recovers one-sided Shampoo).
- *Formatting concerns about appendix organization:* Standard for theory papers; not a substantive weakness.
- *Demand for empirical experiments:* Nice-to-have for theory papers, not a core flaw.
- *Claim that contributions list refers only to appendix:* Theorem 3.2 (cumulative variant) is stated in the main text; stochastic results are appropriately referenced.
- *Concern about the $\log d$ factor being a "proof artifact":* While tightness is unaddressed, the factor correctly arises from non-commutativity in the proof and is explicitly characterized—labeling it an "artifact" is speculative.

## Novel Insights
The paper's key conceptual insight— that adaptive optimizers and NSD exploit non-Euclidean geometry through fundamentally different smoothness/noise assumptions (adaptive vs. standard) rather than through the same mechanism—provides a clean theoretical explanation for why these algorithm families exhibit different empirical behaviors despite their mathematical connections. The demonstration that adaptive smoothness enables acceleration while adaptive variance enables dimension-free rates reveals a systematic pattern: stronger assumptions on smoothness/noise geometry translate into algorithmic benefits that are provably unattainable under weaker standard assumptions.

## Suggestions
- Add a brief discussion (even speculative) on whether the $\log d$ factor for non-commutative preconditioners is likely tight, or whether alternative proof techniques might eliminate it.
- Provide a simple worked example (e.g., diagonal quadratic) computing both $\Lambda_H(f)$ and $L_{\|\cdot\|_H}(f)$ to illustrate when the adaptive assumption yields genuine computational benefit.
- Include an explicit quantitative comparison table with Kovalev & Borodich (2025), showing both rate formulas side-by-side with the relationship between their noise/smoothness assumptions and yours.

---

## 7yvz93kBw9

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (4.9/10)
- Match: N/A

### Final Review

## Summary
The paper addresses sparse-view 3D Gaussian Splatting (3DGS) by identifying and mitigating two failure modes: overfitting in near-field dense Gaussian regions and underfitting in far-field sparse regions. The proposed D²GS framework combines a Depth-and-Density Guided Dropout (DD-Drop) mechanism that probabilistically regularizes redundant Gaussians, and a Distance-Aware Fidelity Enhancement (DAFE) module that boosts supervision in distant regions using monocular depth priors. Additionally, the work introduces Inter-Model Robustness (IMR), a Wasserstein-distance-based metric to quantify training stability across independent runs.

## Strengths
- **Diagnostically motivated design:** The empirical analysis in Figure 1 and Section 3.1 provides clear visual and quantitative evidence of spatial imbalance—near-field shows 11,450 vs. 6,112 Gaussians (overfit), while far-field shows 3,082 vs. 5,224 Gaussians (underfit) compared to dense-view settings—directly motivating the dual-component solution.
- **Principled probabilistic dropout:** DD-Drop addresses a specific failure mode of hard-threshold dropout in DropGaussian by using soft, depth-and-density-aware dropout probabilities, with ablations (Table 5) demonstrating that balanced weights (ω_depth = ω_density = 0.5) achieve optimal performance.
- **Novel robustness metric:** IMR provides the first distribution-level metric for 3DGS that quantifies training stability across independent runs using optimal transport theory, complementing traditional image-space metrics like PSNR/SSIM.
- **Consistent empirical improvements:** D²GS achieves consistent gains across LLFF (3-view, 6-view), MipNeRF360, and DTU datasets, with improvements of ~0.35-0.59 dB PSNR over DropGaussian (Tables 1, 8, 9).

## Weaknesses
- **DD-Drop depth score direction ambiguity:** Equation (1) defines S_i = ω_depth · d̃_i + ω_density · ρ̃_i, where d̃_i is the min-max normalized Euclidean distance to camera. The paper states "high-scoring Gaussians would be dropped with higher probability" but never clarifies whether d̃_i directly encodes distance (larger = farther) or is inverted (larger = nearer). If d̃_i directly encodes distance, the local scoring mechanism would bias dropout *toward* far-field Gaussians—contradicting the stated motivation of preserving these regions. While Eq. (2) attenuates dropout for far-field layers (λ_far = 0.3 < λ_middle = 0.7 < 1), this operates multiplicatively on S_i and may not fully correct the local scoring bias. This ambiguity in a core equation requires explicit clarification.

- **DAFE depth prior misalignment risk:** The DAFE module uses monocular depth estimates (DepthAnything V2) to construct far-field supervision masks. Monocular depth predictions are affine-invariant (relative, up-to-scale and shift), while 3DGS Gaussian primitives are positioned in metric/COLMAP space. The paper does not discuss alignment or rescaling between these depth representations. For scenes with strong depth discontinuities, the τ·D_max threshold may select inconsistent regions across training views.

- **IMR lacks correlation with rendering quality:** IMR measures inter-model Gaussian distribution consistency but does not directly measure rendering quality or stability. Two models could yield identical renderings (high PSNR) but have geometrically diverse Gaussian distributions (high IMR), or conversely, nearly identical distributions (low IMR) could render poorly. No experiment correlates IMR values with PSNR variance or perceptual stability across seeds, leaving IMR's practical interpretation unclear.

- **Baseline reproduction concern:** Appendix E states "we found it difficult to reproduce the results reported in [DropGaussian's] paper, and thus, we report the results obtained from our training." Since DropGaussian is the primary baseline against which gains are measured (~0.59 dB improvement), using self-run baseline numbers raises concerns about fair comparison. The paper does not detail whether hyperparameters, random seeds, or initialization protocols were matched.

- **Ablation baseline attribution:** Table 4's ablation study starts from vanilla 3DGS (PSNR 19.22) rather than DropGaussian (PSNR 20.76). Since D²GS is explicitly "built on DropGaussian" (Section 4), the ablation conflates DropGaussian's own contributions with those of the proposed DD-Drop and DAFE modules, making it difficult to isolate the specific gains from each component.

- **No statistical significance for modest PSNR gains:** The reported improvements over DropGaussian (~0.35-0.59 dB PSNR) are modest. Given the method already requires 10 independent training runs for IMR computation, variance or confidence intervals should be reported for PSNR/SSIM to contextualize whether these margins are statistically meaningful.

## Nice-to-Haves
- **Comparison with feed-forward methods:** PixelSplat, MVSplat, and HiSplat are discussed in related work but not compared experimentally. Even a single table noting their performance/efficiency trade-offs would strengthen SOTA claims.
- **Depth prior sensitivity analysis:** Experiments injecting controlled noise or bias into monocular depth estimates would clarify DAFE's robustness when depth priors are unreliable (transparent surfaces, reflective objects).
- **IMR computational cost reporting:** The time/memory required to compute IMR for a standard scene would help practitioners assess its practical utility.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Table 2 incomplete in main text:** The harsh critic claimed Table 2 is incomplete. However, full MipNeRF360 comparisons are provided in Table 8 (Appendix E), and the paper references this. This is a layout choice, not a missing comparison.
- **Generic novelty criticism:** The claim that DD-Drop and DAFE are "incremental adaptations" is weakened—the combination is well-integrated and the IMR metric is genuinely novel for 3DGS evaluation.
- **Hyperparameter count as weakness:** While the method has several hyperparameters, Table 5 provides systematic ablations over key parameters (ω_depth, ω_density, r_min, r_max, τ, λ_DAFE). Fixed parameters like λ_far and λ_middle are standard in such regularization schemes.
- **"Well-written paper" as strength:** Removed per rules—too generic.
- **"Important topic" as strength:** Removed per rules—applies broadly to sparse-view reconstruction papers.

## Novel Insights
The diagnostic framing—identifying overfitting in near-field dense regions and underfitting in far-field sparse regions—provides a principled explanation for why uniform dropout (DropGaussian) underperforms: it indiscriminately discards Gaussians in regions that need different treatment. The insight that far-field regions need enhanced supervision (DAFE) rather than reduced regularization is particularly valuable and contrasts with the common assumption that sparse views uniformly require regularization. The IMR metric formalizes a previously informal observation—that 3DGS training is unstable across seeds—and provides a quantitative tool for future work to measure progress on training robustness.

## Suggestions
- Explicitly state in Eq. (1) whether d̃_i encodes normalized distance or inverted distance; if inverted, show the formula; if not, explain how the local score combined with global layering preserves far-field Gaussians.
- Add a correlation analysis between IMR values and per-seed PSNR variance to establish IMR's practical relevance for rendering quality.
- For ablation studies, include a baseline row showing DropGaussian (not just vanilla 3DGS) to isolate D²GS's specific contributions from DropGaussian's base improvements.
- Include confidence intervals or standard deviations for PSNR/SSIM metrics across multiple runs where available (especially since IMR already requires 10 runs).

---

## FlcMckO6x5

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (6.2/10)
- Match: N/A

### Final Review

## Summary

This paper establishes theoretical foundations for separable neural networks (SepNNs) by proving a universal approximation theorem for CP, TT, and Tucker variants, deriving NTK regimes that distinguish deterministic (infinite-rank) from stochastic (fixed-rank) cases, and proposing SepPGD—an efficient preconditioned gradient descent method that exploits grid structure to reduce complexity from O(n^D) to O(nD). The work provides the first unified theoretical treatment of SepNNs with empirical validation across kernel regression, implicit neural representations, and physics-informed neural networks.

## Strengths

- **Unified approximation theory for SepNNs.** Theorem 1 establishes universal approximation for CP, TT, and Tucker SepNNs by verifying the Stone-Weierstrass conditions (closed under addition/multiplication, separates points, contains constants) and connecting to classical MLP approximation. This unifies prior bivariate results (Cho et al., 2023; Yu et al., 2024) to general multivariate settings with a clean proof technique.

- **Clean separation of NTK regimes.** Theorem 2 and Corollary 1 provide a principled distinction: infinite width and rank yield a deterministic NTK (enabling convergence analysis), while fixed rank yields a stochastic kernel. This correctly identifies the theoretical gap between asymptotic analysis and practical deployment.

- **Genuine algorithmic efficiency with theoretical grounding.** SepPGD achieves O(nD) preconditioner application complexity for n^D grid samples, compared to O(n^D) for standard NTK-PGD. Lemma 2 formally establishes equivalence to classical PGD for D=2, validating the eigenvalue modulation strategy through Kronecker structure exploitation rather than heuristic design.

- **Comprehensive empirical validation across domains.** Experiments span kernel ridge regression, 2D/3D image and surface representation via INRs, and 3D PDEs via PINNs. Convergence is plotted against wall-clock time, sensitivity analyses examine modulation functions (Table 2), rank (Table 3), width (Table 6), and noise levels (Table 7), and the code repository is publicly available.

- **Transparent limitations disclosure.** Appendix A.1.2 explicitly acknowledges gaps: quantitative approximation rates, fixed-rank convergence guarantees, and extension to TT/Tucker NTK analysis are listed as future work.

## Weaknesses

- **Fixed-rank regime lacks convergence guarantees.** Corollary 1 correctly characterizes the NTK as stochastic under fixed rank, but Remark 3 admits "the training dynamic cannot be characterized uniformly using a fixed NTK matrix." Practical SepNNs operate at small ranks (R=64–500 in experiments), yet the paper's convergence analysis (Eq. 5) relies on the infinite-rank deterministic NTK. The theoretical-practical gap is acknowledged but unaddressed.

- **SepPGD equivalence proof limited to D=2.** Lemma 2 establishes that SepPGD equals classical NTK-PGD for bivariate SepNNs. The paper states this "can be readily extended to multivariate cases D > 2" without proof. The method is applied to D=3 PINN experiments without theoretical grounding for that dimensionality.

- **Missing key baseline: SepNN with standard NTK-PGD.** Experiments compare (vanilla MLP, SepNN+Adam, SepNN+SepPGD) but omit SepNN+classical-PGD (Geifman et al., 2024; Shi et al., 2025). This makes it impossible to isolate whether gains come from preconditioning itself versus the separable decomposition of the preconditioner.

- **No wall-clock timing comparison.** The central claim of O(nD) versus O(n^D) efficiency would be most directly validated by wall-clock timing per iteration. Tables 1 provides asymptotic complexity but no actual runtime comparison against MSK/IGA baselines.

- **NTK analysis covers CP only.** Theorem 2 and its corollaries address CP-SepNN, leaving TT and Tucker variants without NTK characterization despite all three being covered in the approximation theorem. The paper notes extension as future work (Appendix A.1.2).

## Nice-to-Haves

- Visualization of NTK eigenvalue distribution before and after SepPGD to directly validate spectral flattening claims
- Comparison against strong INR baselines (SIREN, WIRE, Fourier features) to contextualize PSNR improvements
- Explicit discussion of when SepPGD's grid requirement fails and potential workarounds for scattered data
- Extension of SepPGD to PDE residual losses in PINNs, where spectral bias is most consequential

## Removed Points

*These points were flagged for removal as they misunderstand the paper or demand unreasonable scope:*

- **"Existence result only, no quantitative rates."** While true, this criticism mischaracterizes the contribution's intent. The universal approximation theorem establishes representation completeness—the foundational question of whether SepNNs can represent functions. Quantitative rates are a separate research direction that the paper explicitly acknowledges as future work (Appendix A.1.2). Demanding rates would require a fundamentally different theoretical framework.

- **"Routine proof technique via Stone-Weierstrass."** The proof technique being "standard" is not a weakness. The contribution lies in unifying three architectures (CP, TT, Tucker) under one framework and extending bivariate results to multivariate cases. The mathematical machinery being established does not diminish the result.

- **"Two-layer MLPs only in NTK analysis."** The paper correctly states (Remark 1) that extension to multi-layer networks is straightforward using Arora et al. (2019b). The recursive NTK formulation for deep networks is well-established. While proving it would be complete, claiming this is a "gap" overstates the issue—the theoretical machinery exists.

- **"Single image in main experiments raises cherry-picking concerns."** This factual error misreads the paper. Figures 2-3 and Appendix Figs. 6-12 include multiple images (Plane, Peppers, Baboon) and 3D surfaces. The sensitivity analysis (Table 3) explicitly shows varying performance across rank and k values.

- **"Overstating that NTK stays fixed during training."** The analysis correctly bounds weight movement to show NTK change vanishes as W,R→∞. The "infinitely small learning rate" assumption is standard in NTK literature (Jacot et al., 2018; Arora et al., 2019). Calling this a "purely asymptotic statement with limited practical content" applies the same critique to the entire NTK research program.

## Novel Insights

The identification that SepNNs exhibit fundamentally different NTK behavior depending on rank scaling—infinite rank yields deterministic kernels amenable to standard convergence analysis, while fixed rank yields stochastic kernels with no uniform training characterization—reveals a previously unrecognized theoretical boundary in separable architectures. This explains why empirical SepNN practice (small fixed ranks) may diverge from theoretical predictions (infinite-rank assumptions) and pinpoints exactly where future theoretical work should focus: developing fixed-rank convergence bounds via random feature or stochastic NTK frameworks.

## Suggestions

- Add wall-clock timing comparison against classical NTK-PGD applied to SepNNs to isolate the contribution of separable preconditioner decomposition versus preconditioning itself. Even a single table comparing iterations/second for D=2 and D=3 cases would substantiate efficiency claims.

- Provide a brief discussion or visualization of how the condition number of the preconditioned NTK relates to rank R, since practical deployment requires understanding the spectral adjustment at finite R rather than the asymptotic regime.

- State prominently in the introduction (not just footnotes and appendix) that SepPGD's efficiency advantage requires grid-structured inputs, as this is a key practical limitation for potential users.

---

## Rt9SeEAMWv

- GT: Reject (avg 4.8)
- Predicted: N/A (5.7/10)
- Match: N/A

### Final Review

## Summary

This paper introduces **random set stability**, a new framework for deriving worst-case generalization bounds over data-dependent random sets (such as parameter trajectories from stochastic optimization). The key insight is to replace intractable mutual information terms in prior fractal and topological generalization bounds with an interpretable stability parameter β_n. The authors prove that their framework recovers classical stability bounds (as a special case with J=1) and classical Rademacher complexity bounds (with J=n), while enabling the first fully computable topological generalization bounds without information-theoretic terms.

## Strengths

- **Addresses a genuine bottleneck**: The intractability of mutual information terms in prior topological generalization bounds (Simsekli et al., 2020; Birdal et al., 2021; Andreeva et al., 2024) is a well-recognized limitation. This paper provides a principled stability-based alternative that eliminates these terms entirely.

- **Unifies multiple learning-theoretic frameworks**: Lemma 3.4 introduces a parameter J that interpolates between classical algorithmic stability (J=1, Corollary 3.5) and data-independent hypothesis set complexity (J=n, β_n=0, Corollary 3.6). This demonstrates the framework's mathematical consistency and conceptual generality.

- **First computable topological bounds**: Theorems 4.3 and 4.4 provide explicit bounds in terms of fractal dimension and weighted lifetime sums / positive magnitude that contain no information-theoretic terms. The experiments actually compute these bounds, which has not been possible with prior IT-based approaches.

- **Strong theoretical grounding**: Lemma 3.2 shows that classical uniform argument stability implies random set stability, and Corollary 3.3 establishes explicit stability parameters for projected SGD under standard smoothness/Lipschitz assumptions.

## Weaknesses

- **Slower convergence rate**: The optimized bound scales as O(β_n^{1/3}), yielding approximately O(n^{-1/3}) when β_n = O(1/n). This is slower than the classical O(n^{-1/2}) Rademacher rate. While the paper acknowledges this as a "deliberate trade-off to maintain boundedness," it does not analyze whether the exponent 1/3 is fundamental or an artifact of the proof technique. No lower bound is provided to show tightness.

- **Expectation-only guarantees**: All main results (Lemma 3.4, Theorems 4.3 and 4.4) provide bounds in expectation only, not high-probability bounds. For practical worst-case guarantees, high-probability statements would be substantially more useful. The limitations section acknowledges this but does not explore whether McDiarmid-type concentration arguments could extend the results.

- **Theory-practice gap in stability verification**: Corollary 3.3 establishes random set stability for projected SGD on smooth, Lipschitz losses with explicit β_n = O(T²/n). However, experiments use Adam optimizer on large ViT and GraphSAGE models fine-tuned from pretrained checkpoints—settings far from the theoretical assumptions. The estimated β_n values (~3×10⁻⁴) are empirically small, but the theoretical bound for T=500 iterations would be orders of magnitude larger. No comparison between theoretically derived and empirically estimated β_n is provided.

- **Bound looseness**: Table 1 shows estimated bounds approximately 10× larger than actual generalization error (e.g., bound ~1.04 vs. error ~0.10 for ViT). It is unclear how much looseness comes from the stability framework itself versus the Massart bound used to estimate Rademacher complexity versus the optimistic β_n estimation.

- **Lipschitz assumption not empirically verified**: Assumption 4.1 requires local Lipschitz continuity of the loss on W_{S,U} with constant L_{S,U}. This constant is never estimated in experiments, leaving a gap between theoretical requirements and empirical practice.

## Nice-to-Haves

- Discussion of whether high-probability extensions are obtainable via concentration inequalities
- Comparison with IT-based bounds in regimes where IT terms are provably finite
- Direct estimation of Rademacher complexity (rather than Massart upper bounds) to identify where bound looseness arises
- Larger-scale experiments (current maximum n=10,000) to better validate scaling predictions

## Removed Points

These points are flagged to be removed, treat them with caution:
- **"PMag experimental results are entirely absent"** (from Harsh Critic): This is factually incorrect. Appendix D.1 explicitly presents PMag results with Figures 4–7 and 10–13, analyzing both weighted lifetime sums E¹ and positive magnitude PMag.
- **"Tan et al. (2024) limitations not engaged with"**: The Tan et al. paper critiques fractal dimension as a generalization measure, but this paper's contribution is a stability framework that subsumes multiple complexity measures. The critique applies to using fractal dimension alone, not to this broader framework.
- **"Exponentially many runs required"**: The criticism that β_n estimation requires multiple runs overlooks that (1) the estimation is a one-time cost for a given architecture/hyperparameters, and (2) prior IT-based bounds were fundamentally intractable, not merely expensive.

## Novel Insights

The interpolation parameter J in Lemma 3.4 reveals a fundamental trade-off between stability and complexity: choosing J small emphasizes the stability term (requiring stronger algorithmic stability for tighter bounds), while choosing J large emphasizes the complexity term (recovering data-independent generalization bounds). This structural insight—that trajectory-based generalization can be analyzed at any point along this spectrum—unifies previously disparate approaches and clarifies why certain bounds are tighter in different regimes. The empirical finding that topological complexity increases with sample size n (contrary to the intuition that larger datasets should yield "simpler" trajectories) suggests that the relationship between generalization and trajectory geometry is more subtle than previously appreciated.

## Suggestions

- Provide a sensitivity analysis showing how bound tightness changes under different estimates of the local Lipschitz constant L_{S,U}, or directly compute Rademacher complexity for a subset of trajectories to isolate sources of looseness.
- Discuss whether the O(n^{-1/3}) rate is tight by considering alternative choices of J or proof techniques, or provide a concrete example where this rate cannot be improved.
- For the stability estimation procedure (Algorithm 1), explicitly quantify how the finite-sample approximation (M=500 held-out points) biases β_n estimation downward and discuss whether correction factors are applicable.
- Add an explicit comparison between the theoretical β_n from Corollary 3.3 and the empirically estimated values, even if only to illustrate the conservativeness of the theoretical bound.

---

## khBHJz2wcV

- GT: Accept (Poster) (avg 3.0)
- Predicted: N/A (5.9/10)
- Match: N/A

### Final Review

## Summary

This paper introduces a post-training fine-tuning framework for flow-matching generative models that enforces PDE constraints via weak-form residuals and jointly infers latent physical parameters through a learned surrogate base flow. By reformulating fine-tuning as a stochastic optimal control problem via Adjoint Matching, the method produces physically consistent samples and parameter estimates without requiring paired state-parameter training data. Experiments across four PDE families demonstrate residual reductions, distributional fidelity trade-offs, and inverse-problem capabilities.

## Strengths

- **Addresses a genuine data-scarcity bottleneck**: The joint state-parameter evolution formulation enables physics-constrained generation when parametric labels are unavailable—a real limitation in scientific domains. The surrogate base flow construction (Section 3.2) provides a principled mechanism for this.

- **Methodologically sound constraint formulation**: The weak-form PDE residuals with randomly sampled test functions (Section 3.1, Appendix D.3) align with established computational mechanics practice and provide numerical stability under noisy or misspecified data. Wendland-wavelet test functions with compact support reduce derivative order and avoid boundary artifacts.

- **Theoretical contribution**: Lemma 1 (Appendix D.4) proves that the scaled memoryless noise schedule σ²(t) = (1−κ)²·2ηₜ remains consistent with the Adjoint Matching framework, providing practitioners with a tunable stabilization parameter. The proof is clean and the extension is novel.

- **Controllable trade-offs via regularization**: The λ_f parameter enables smooth interpolation between pure physics constraint enforcement and fidelity to the base distribution. Figures 3 and Appendix F.1 demonstrate this tunability empirically across residual vs. MMD trade-offs.

- **Comprehensive empirical evaluation**: Four PDE families (Darcy, elasticity, Helmholtz, Stokes) with deliberately varied forms of model misspecification (noise, boundary conditions, damping, forcing) test robustness. Detailed architecture specifications, hyperparameter tables, and a public repository support reproducibility.

## Weaknesses

- **Systematically disadvantaged baseline comparison**: PBFM is included as a training-time physics-constrained baseline, but all experiments deliberately introduce physics mismatch between training data and fine-tuning targets (Appendix E.2). The paper acknowledges this disadvantages PBFM, yet no experiment tests the regime where training and fine-tuning physics match. This leaves unclear whether the proposed method would remain competitive without this structural advantage.

- **No oracle upper bound**: There is no comparison to a model trained directly on clean data under the correct PDE specification. The reference set D_ref is used only as a metric target. Without an oracle baseline, it is impossible to assess how much performance is lost relative to the ideal scenario of having access to clean, correctly-specified training data.

- **The κ hyperparameter lacks sensitivity analysis**: The scaled noise schedule uses κ=0.9 in all PDE experiments, representing 81% noise attenuation. While Lemma 1 provides theoretical justification, there is no ablation showing how results vary with κ. Given that κ has substantial effects on noise injection during fine-tuning, practitioners have no guidance on appropriate values for different PDE regimes.

- **Improvements over ablations have overlapping uncertainty bounds**: In Tables 1-2 and Appendix F.1, the full AM method achieves lower residuals than Base AM ablations, but error bars frequently overlap (e.g., Helmholtz: AM achieves R_weak 4.3±1.29 vs Base AM 4.9±1.85). The Stokes experiment (Figure 5) shows clearer gains in MMD_α, but the residual improvements across methods are comparable.

- **Ground-truth parameter recovery is not directly evaluated**: The inverse-problem framing claims to infer latent parameters, yet MMD_α measures distributional shift rather than pointwise recovery accuracy. Reporting RMSE between inferred α and ground-truth α would directly validate the inverse-problem capability—currently left unverified.

- **Computational overhead is not quantified**: Weak-form residuals require numerical integration over stochastically sampled test functions at each fine-tuning step. The paper states fine-tuning takes "under 15 minutes on a single NVIDIA L40S" for Darcy, but provides no breakdown of per-step cost, FLOP counts, or comparison to training PBFM from scratch.

- **Natural image experiment diverges from core contribution**: Section 4.6 demonstrates parametric color transformation with PickScore rewards, which tests whether the joint evolution framework couples latent variables in non-scientific settings. This does not validate physics-constrained generation claims and dilutes focus from the primary scientific ML contribution.

## Nice-to-Haves

- **Comparison to inference-time guidance methods**: While FM+ECI is included for elasticity, comparing to classifier-free guidance or score-based posterior sampling on at least one inverse-problem experiment would better isolate the contribution of joint fine-tuning versus inference-time steering.

- **Analysis of residual vs. solution error correlation**: Low weak residuals do not necessarily imply accurate solutions in ill-posed settings. Demonstrating that reduced residuals correlate with actual solution error |u_pred − u_true| would strengthen the physical validity claims.

- **Hyperparameter selection guidance**: Practitioners must tune λ_x, λ_α, λ_f, and κ, but the paper provides no guidance on principled selection beyond the ablation curves in Figure 3.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Claim that ECI should appear in all experiments**: The paper omits FM+ECI from Helmholtz and Stokes experiments. While the harsh critic questions this, the paper does include ECI in the elasticity experiment where it performs poorly (residuals orders of magnitude larger). Including ECI where it fundamentally cannot perform parameter inference would not provide meaningful comparison.

- **Demand for multimodality/diversity metrics beyond SSIM**: The paper already measures distributional fidelity via MMD and diversity via SSIM-based diversity. Additional multimodality metrics would be nice-to-have rather than essential.

- **Criticism that test function variance is unanalyzed**: While "low-variance learning signal" is claimed without formal analysis, the use of N_test randomly sampled test functions with averaging is standard practice in weak-form methods. The variance critique is valid but not a core flaw.

## Novel Insights

The joint state-parameter evolution mechanism (surrogate base flow for α) is the most distinctive technical contribution. Rather than treating parameter inference as a separate post-hoc optimization, the method couples α and x through a shared vector field, enabling coherent joint sampling. The Stokes experiment reveals the key advantage: comparable residual reduction across methods, but substantially lower MMD_α for the joint model—suggesting that joint evolution preserves parameter-distribution structure that independent fine-tuning disrupts. The scaled noise schedule (Lemma 1) is a genuine, if modest, theoretical extension that provides practical stabilization without abandoning memoryless consistency.

## Suggestions

1. **Add a PBFM comparison in the non-misspecified regime**: Include at least one experiment where the training data PDE matches the fine-tuning target PDE to fairly assess whether the proposed method maintains competitiveness when baselines are not structurally disadvantaged.

2. **Report ground-truth parameter recovery metrics**: Compute RMSE between inferred α and true α on a held-out test set for the inverse-problem experiments. This directly validates the core claim of "addressing ill-posed inverse problems."

3. **Add κ ablation**: Vary κ ∈ {0.5, 0.7, 0.9, 1.0} on at least one PDE to show sensitivity and guide hyperparameter selection for practitioners.

4. **Quantify computational cost**: Report wall-clock time breakdown between residual evaluation and model forward/backward passes, and compare total fine-tuning FLOPs to training PBFM from scratch.

5. **Remove or reframe the natural image experiment**: Move to appendix or explicitly reframe as "demonstrating framework generality" rather than implying it validates the physics-constrained generation methodology.

---

## USyGD0eUod

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (6.1/10)
- Match: N/A

### Final Review

## Summary
This paper applies a fundamental sanity check—comparing trained models against randomized baselines—to sparse autoencoder (SAE) evaluation metrics in mechanistic interpretability. Across Pythia models (70M–6.9B parameters), the authors find that aggregate auto-interpretability scores (fuzzing/detection AUROC) frequently fail to distinguish SAEs trained on trained transformers from those trained on randomly initialized or re-randomized models. The paper proposes token distribution entropy as a preliminary measure of feature "abstractness" and provides toy model experiments suggesting random networks may preserve or amplify input superposition.

## Strengths
- **Comprehensive empirical design with meaningful controls:** The study spans five model scales and four randomization variants (trained, step-0, re-randomized with/without embeddings, and a Gaussian-embedding control). The control condition—where token embeddings are replaced with i.i.d. Gaussian noise at inference time—provides an essential baseline, confirming that auto-interpretability scores fall to chance (~0.50 AUROC) when no real structure exists. This methodological choice strengthens confidence that the observed scores on random weights reflect genuine preserved structure rather than metric artifacts.
- **Important and alarming scaling finding:** The core result—that larger models exhibit *more* overlap between trained and randomized SAE metrics—is demonstrated across Figures 1–2. At Pythia-6.9B, the randomized variants (AUC ~0.87–0.88) actually *outperform* the trained model (AUC 0.79) on fuzzing AUROC. This reversal, where random activations become *easier* to auto-interpret than trained activations at scale, is a surprising and practically significant finding.
- **Robustness experiments and reproducibility:** Appendices C, E, and F verify results across training data scales (1B tokens), multiple random seeds (for Pythia-70M), and SAE hyperparameters (expansion factors 16–128, sparsity k=16,32). Compute costs are transparently reported (~435 GPU-hours total). The paper uses open frameworks (EleutherAI/delphi, EleutherAI/sparsify) and open models (Pythia suite).
- **Constructive diagnostic proposal:** The token distribution entropy analysis (Figure 2, bottom row; Figure 20 in appendix) reveals a genuine difference: trained model latents show increasing entropy across layers (consistent with more abstract, multi-token features), while randomized variants remain at low entropy. This provides a concrete starting point for improved metrics.

## Weaknesses
- **Theoretical explanation is incomplete and architecture-mismatched:** Section 4 uses toy MLP models to argue that random networks "preserve or amplify superposition," but transformers differ fundamentally from 2-layer ReLU MLPs—they include attention mechanisms, layer normalization, and residual streams. The paper acknowledges the toy model shows "plausibility" but does not explain why transformer-specific architecture should exhibit the same behavior. Furthermore, the Pareto frontier evidence in Figure 5a shows that the gap between superposed inputs and Gaussian inputs *narrows* after passing through a random MLP—this could indicate the MLP *destroys* superposition structure rather than preserving it, which partially contradicts the authors' interpretation.
- **No statistical significance testing for the core AUROC comparisons:** The visual overlap between trained and randomized ROC curves is the paper's central evidence, but no confidence intervals, standard errors, or hypothesis tests are provided for the larger models (only Pythia-70M has multi-seed uncertainty plots in Appendix E). With only 100 features sampled per SAE, whether the AUROC differences at Pythia-6.9B (e.g., trained 0.79 vs. randomized 0.87) are statistically meaningful remains unclear.
- **The auto-interpretability pipeline itself is not analyzed:** The paper does not disentangle whether high scores on random models stem from SAE-learned structure in the activations, or from the LLM explainer/simulator pipeline assigning plausible explanations to sparse token-frequency patterns. Ablations that shuffle token order or apply the pipeline to random activations directly would clarify the source of inflation.
- **Token entropy is proposed but not operationalized or validated:** While entropy shows separation between trained and random variants, the paper does not define how to use this measure in practice (threshold? combined score?). There is no experiment testing whether entropy-filtered features are more causally relevant or whether entropy-weighted AUROC better discriminates trained from random models.
- **Step-0 outperforming trained models on AUROC is noted but not explained:** At nearly all model scales, the "Step-0" (initialization) variant achieves higher fuzzing AUROC than the fully trained model. This counterintuitive finding—that a model before any training would be *more* interpretable—is mentioned only in passing with brief speculation about parameter norms, leaving a significant phenomenon unexplained.

## Nice-to-Haves
- Causal intervention experiments (activation patching, steering) comparing trained vs. random features would strengthen the claim that random features lack "computational relevance," but this extends beyond the paper's scope as a sanity-check study.
- Correlation between token entropy and established measures of computational relevance (e.g., feature utility in downstream tasks) would validate the proposed metric.
- Testing alternative SAE architectures (Gated SAE, JumpReLU) to verify the finding generalizes beyond TopK.

## Removed Points
- *Claim that entropy analysis is "relegated to appendix"* — Token distribution entropy appears in Figure 2 (main results) and is discussed in the main text. Only the scatter plot analysis is in Appendix H, which is appropriate for supplementary detail.
- *Criticism that title incorrectly generalizes to "all metrics"* — The title specifies "Automated Interpretability Metrics," which accurately refers to auto-interpretability scores. The paper does not claim reconstruction metrics fail; it explicitly notes CE loss distinguishes trained models but only applies to them.
- *Demand for human evaluation of feature interpretations* — This would strengthen claims about interpretability quality but is not required for a sanity-check paper focused on automated metrics. The paper's contribution is showing that automated metrics have a failure mode, not evaluating human judgments.
- *Request for intervention experiments proving random features lack computational relevance* — While valuable, this asks the paper to solve the problem it identifies rather than diagnose it. The contribution is the diagnosis.
- *Demand for multiple seeds on all large models* — Valid concern, but practical compute constraints are acknowledged. Appendix E provides uncertainty quantification for Pythia-70M as a representative case.

## Novel Insights
The most striking insight—larger randomized models *outperform* trained models on auto-interpretability—is documented but under-analyzed. The paper speculates about feature specificity with SAE size but does not probe why training would *reduce* interpretability scores relative to random initialization at scale. This reversal deserves deeper investigation: does training systematically produce more distributed, harder-to-explain features that lower aggregate scores, while random activations retain simple token-frequency structure that auto-interpretability pipelines easily capture? The entropy analysis hints at this (trained = higher entropy, randomized = lower entropy), but the paper stops short of a unified explanation.

## Suggestions
- **Operationalize entropy as a practical metric:** Define a concrete procedure (e.g., compute entropy-weighted AUROC or filter features above an entropy threshold) and test whether it improves discrimination between trained and random models.
- **Add statistical testing:** Report confidence intervals for AUROC values across multiple random seeds for at least one larger model, or compute p-values for the trained vs. randomized comparisons using existing data.
- **Clarify the theoretical section's scope:** State explicitly that the MLP toy model demonstrates plausibility for the superposition-preservation hypothesis but does not constitute a mechanistic explanation for transformer behavior. Consider relabeling Section 4 as "Preliminary Analysis" or similar.
- **Analyze the pipeline directly:** Run a simple ablation where token order is permuted before explanation generation; if AUROC remains high, the explainer (not the SAE) is responsible for inflation.

---

## GiaF5cFIpI

- GT: Reject (avg 3.5)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary

This paper presents a unified streaming framework for designing neural stimulations that drive latent neural dynamics along desired directions. The approach integrates three components: (1) streaming dimensionality reduction methods including a novel sjPCA variant, (2) online nonparametric kernel regression to learn state-dependent stimulus-response mappings that adapt to non-stationarity, and (3) constrained optimization for selecting high-dimensional stimulation patterns under biological feasibility constraints (non-negativity, sparsity). The method is evaluated on simulated dynamical systems, calcium imaging data, and electrophysiological recordings, with demonstrated end-to-end runtimes under 100 ms.

## Strengths

- **Integrated real-time pipeline with practical constraints**: The framework successfully chains streaming manifold construction, latent dynamics filtering, and closed-loop stimulation targeting into a unified pipeline (Algorithm 1). The optimization (Eq. 8) explicitly encodes non-negativity and L₁-relaxed sparsity constraints that directly mirror optogenetic experimental limits, and Appendix H provides empirical benchmarking showing average latencies <10 ms with worst-case <100 ms on standard hardware.

- **Adaptive response mapping handles non-stationarity**: The kernel regression model Ŝ (Eq. 7) captures state-dependent, potentially time-varying stimulus-response relationships without assuming stationarity. Figures 2d–e demonstrate recovery from abrupt discontinuities (180° map flip) and continuous drift within ~15 seconds, consistently outperforming "blind" predictors that ignore stimulation effects.

- **Comprehensive modular evaluation across methods**: The paper systematically evaluates multiple latent space constructions (proSVD, sjPCA, mmICA) and dynamical models (Kalman filter, VJF, Bubblewrap) across two neural modalities (calcium imaging, electrophysiology). Tables 1–12 provide full ablation results, demonstrating architectural robustness and revealing that proSVD + KF generally performs best.

## Weaknesses

- **Real data experiments use simulated stimulations**: While the paper accurately describes its experiments as using simulated stimulations on real neural recordings (Section 4.1), this is a significant limitation. The simulated stimulation model (aₜ = 0.8·aₜ₋₁ + uₜ) is substantially cleaner than actual optogenetic responses, which depend on opsin expression, network connectivity, and state-dependent biological factors. Appendix C includes two real-stimulation datasets with favorable error comparisons, but these analyses are limited to scalar prediction errors without spatial or trajectory analysis. The core claim of "driving latent dynamics" remains incompletely validated in the target experimental setting.

- **No comparison to prior stimulation design methods**: The only baseline is a "blind" model that ignores stimulation events. The paper does not compare against MiSO (Minai et al., 2024), Bayesian optimization approaches, or active learning methods (Wagenmaker et al., 2024), all cited as related work. Without such comparisons, the relative sample efficiency and alignment precision of the proposed method cannot be assessed.

- **Curse of dimensionality in stimulus kernel**: The RBF kernel K₂(u, Uᵢ) operates on stimulus vectors that can be hundreds of dimensions (592 neurons in the calcium data). In high dimensions, RBF kernel values become nearly uniform, potentially rendering the regression degenerate. The paper neither discusses bandwidth selection for this setting nor how the sparsity of u might alleviate this concern.

- **Incomplete methodological details for novel sjPCA contribution**: The sjPCA method is presented as novel but the Sherman-Morrison update derivation is only alluded to without details. For a claimed novel contribution, the lack of explicit rank-1 update formulas and stability analysis reduces reproducibility and theoretical grounding.

- **Memory complexity of kernel regression grows unboundedly**: The estimator stores all past stimulation-response pairs, with prediction cost O(N_stim · N) per query. No truncation, forgetting mechanism beyond the temporal kernel, or approximation strategy is discussed, raising scalability concerns for long closed-loop experiments.

- **Hyperparameter sensitivity not analyzed**: The pipeline depends on kernel bandwidths (K₁, K₂, K₃ in Eq. 7) and regularization coefficient λ₁ (Eq. 8). No ablation or sensitivity analysis is provided, leaving unclear how robust performance is to hyperparameter choices.

## Nice-to-Haves

- Quantification of the reachable volume in latent space under biological constraints (sparsity, non-negativity) to inform users which perturbations are theoretically feasible.

- Extension to nonlinear latent spaces (e.g., VAEs) as acknowledged in the Discussion, which would broaden applicability to non-linear neural manifolds.

- Comparison of sample complexity scaling with number of neurons N and latent dimensions k.

## Removed Points

These points are flagged to be removed, treat them with caution:

- *Equation formatting issues (Eqs. 4-5 showing "if u_t = 0" in both branches)*: This appears to be a PDF rendering artifact where ≠ was corrupted to =. The intended logic is clear from context and this is a production issue, not a methodological flaw.

- *Equation numbering disorder*: Eq. 6 appearing before Eqs. 4-5 is a minor organizational issue that does not affect comprehension.

- *Statistical rigor with N=10 runs*: While marginal, this is within typical practice for ML methodology papers; the paper shows error bars and variance.

- *Missing theoretical convergence guarantees for sjPCA*: Demanding theoretical analysis goes beyond the paper's empirical contributions scope.

- *Missing behavioral correlation*: The authors explicitly acknowledge this limitation in the Discussion.

- *Demanding real in vivo closed-loop experiments*: While this would strengthen the paper, the authors are transparent about their experimental setup, and Appendix C provides preliminary validation with real stimulation data.

## Novel Insights

The paper makes an interesting observation that different latent space constructions (proSVD, sjPCA, mmICA) can be run in parallel with adaptive selection based on predictive performance (Fig. 1c). This suggests that the "optimal" neural manifold for stimulation design may vary over time depending on the current neural state and task demands. The streaming formulation of sjPCA, which combines online skew-symmetric estimation with Orthogonal Procrustes stabilization, provides a novel approach to real-time rotational dynamics tracking that could find applications beyond this specific stimulation context. The L₁ relaxation using ‖u‖_max − ‖u‖₁ for sparsity, while not rigorously justified, offers an alternative to standard sparsity penalties that may be worth further theoretical investigation.

## Suggestions

- Add direct quantitative comparison with MiSO or Bayesian optimization methods on shared metrics (sample efficiency, alignment error) to establish relative performance.

- Expand the Appendix C real-stimulation analysis beyond scalar error metrics to include trajectory visualizations and alignment quality.

- Add hyperparameter sensitivity analysis for λ₁ and kernel bandwidths to demonstrate robustness or provide guidance for tuning.

- Include a brief discussion of kernel bandwidth selection in high dimensions or implement adaptive bandwidth strategies to address the curse of dimensionality concern.

- Clarify the memory complexity and discuss potential approximation strategies (e.g., budgeted kernel regression, inducing points) for long-duration experiments.

---

## 0cbUKCyBsH

- GT: Reject (avg 3.5)
- Predicted: N/A (6.9/10)
- Match: N/A

### Final Review

## Summary

This paper argues that time series forecasting faces a fundamental performance ceiling due to the "self-stimulation" paradigm, where models predict future values using only historical observations while ignoring external influences. Through a control-theoretic lens, the authors formally prove that unobserved influences create an irreducible error bound (Proposition 2.1) and that incorporating measurable influences systematically reduces this bound (Proposition 3.1). They operationalize this paradigm through a new leak-free, temporally-synced benchmark incorporating textual influences, and propose FIATS, a lightweight LLM-free model with channel-aware sensitivity mechanisms (CASM and CAPS). Empirical results demonstrate consistent improvements over self-stimulated baselines across synthetic, atmospheric, traffic, and business datasets.

## Strengths

- **Rigorous theoretical framework:** The control-theoretic derivation (Propositions 2.1 and 3.1, Appendix B) provides mathematical justification for why self-stimulated models converge to conditional expectations rather than true dynamics, establishing an irreducible error floor proportional to influence variance and system sensitivity. The proofs are technically sound and directly support the claimed paradigm shift.

- **Carefully designed benchmark with explicit leak-free constraints:** The Temporal-Synced IATSF benchmark addresses documented issues in existing multimodal TS datasets (Section 4.1). By restricting influences to independently evolving factors and requiring contemporaneous alignment, the benchmark prevents information leakage that has plagued prior work. The four-dataset suite spans controlled toy systems to real-world business scenarios.

- **Principled architecture translating theory to practice:** The CASM (Channel-Aware Adaptive Sensitivity Modeling) and CAPS (Channel-Aware Parameter Sharing) mechanisms directly implement theoretical insights about channel-specific influence sensitivity (Section 5). Ablation studies (Table 3, Figure 6) confirm that removing channel descriptions or corrupting influence text degrades performance, validating the design rationale.

- **Strong empirical validation with interpretable analysis:** Results across four datasets (Table 1, Table 2) show substantial MSE reductions: 36.0% on Atmospheric Physics and 44.3% on NYC Traffic vs. PatchTST. The FM Toy dataset result (FIATS: 0.003 vs. baselines: 0.282-0.909) directly validates the theoretical prediction that self-stimulated models should approach conditional expectations. Attention visualizations (Figures 3, 5, 10) show layer-wise progression from temporal context to channel-specific influence sensitivity.

## Weaknesses

- **Evaluation assumes known future influences during testing:** The main experiments use ground-truth future textual influences ($U_f$), as acknowledged in Appendix B.3.2: "we assume the influence forecaster is highly accurate, with negligible error." While this isolates model capability from influence-forecasting quality, it sidesteps a critical deployment challenge. In practice, predicting future textual descriptions introduces error propagation that could dominate overall performance. The noise robustness experiment (Figure 6) partially addresses this but uses synthetic noise rather than realistic influence-forecasting errors.

- **Missing comparison against modern exogenous-variable models:** The baseline comparison focuses on self-stimulated models (DLinear, PatchTST, iTransformer) and foundation models (Chronos, MOIRAI, Time-MoE), but excludes architectures specifically designed for exogenous variables (e.g., TiDE, TimeXer, ChronosX, or PatchTST variants with explicit exogenous conditioning). This makes it difficult to determine whether gains stem from the textual modality itself or from simply incorporating any external signal. TimeLLM is included as an LLM-based baseline but differs substantially in approach.

- **Independence assumption limits theoretical generality:** Propositions 2.1 and 3.1 assume $U_t \perp X_h$ (influences independent of historical states) and instantaneous impact. Real-world systems often exhibit state-dependent influences, delayed effects, or feedback loops. The paper acknowledges this limitation but does not analyze performance degradation when independence is violated, nor does the architecture explicitly handle lagged influence effects.

- **Insufficient detail on textual influence generation for reproducibility:** The Atmospheric Physics dataset uses LLM-generated weather summaries (Appendix O.4.4), but the exact prompts, temperature settings, and validation procedures are not specified. Since FIATS's performance depends on semantic alignment between text and time series dynamics, variations in the captioning pipeline could materially affect results. The ablation in Table 6 shows text quality matters, but the generation methodology remains opaque.

## Nice-to-Haves

- **Text vs. numerical exogenous ablation:** A comparison between FIATS using textual weather summaries and a variant using raw numerical weather data would clarify whether the textual modality provides unique signal or merely acts as a proxy for available numerical exogenous variables.

- **Cross-domain generalization testing:** Evaluating FIATS trained on one domain (e.g., Weather) and tested on another (e.g., Traffic) would demonstrate whether the model learns semantic influence dynamics versus dataset-specific correlations.

- **Analysis with delayed influence effects:** Extending the CASM mechanism to handle lagged onset (e.g., weather affecting traffic with delay) would strengthen claims about real-world applicability beyond the instantaneous-impact assumption.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Criticism about "text modality providing no unique value":** This is speculative and contradicted by Table 6, which shows FIATS with good text achieves MSE 0.186 while random/no text yields 0.249-0.724, demonstrating clear semantic value beyond mere additional input dimensionality.

- **Criticism about foundation model scaling not addressing the core problem:** The paper's FM Toy experiment (Table 1, Figure 1) directly addresses this by showing billion-parameter models fail on systems with strong influence sensitivity, supporting the theoretical argument that scale cannot overcome missing influence information.

- **Demand for confidence intervals across multiple runs:** Standard practice for large-scale TS benchmarks is single-run evaluation. Table 10 in Appendix M provides variance estimates (e.g., FIATS Toy: 0.027±0.001), showing stable performance.

- **Criticism that the benchmark is "unfairly designed for FIATS":** The benchmark design (independent influences, temporal synchronization) is motivated by fundamental methodological requirements for influence-aware modeling (Section 4.1), not model-specific optimization. Self-stimulated baselines fail by design on FM Toy because they cannot incorporate influence information—this validates the theory rather than indicates unfairness.

## Novel Insights

The paper's core insight—that TSF performance plateaus stem from a fundamental theoretical ceiling rather than architectural limitations—provides a principled explanation for why scaling model parameters has yielded diminishing returns. The FM Toy results offer compelling evidence: when influence sensitivity is high (large $\nabla_U F$), even billion-parameter foundation models produce collapsed, averaged predictions matching the theoretical bound. This reframes the field's trajectory from "build better sequence models" to "explicitly model the control inputs that drive system dynamics." The channel-aware sensitivity design (deriving $dU/dx_i$ from linear systems analysis) successfully translates control-theoretic intuition into a learnable architecture that generalizes across domains with fundamentally different dynamics (physics-based weather vs. human-driven markets).

## Suggestions

1. **Extend experiments with predicted influences:** Add a realistic evaluation setting where future influences are forecasted (not oracle-provided), even using simple baselines like persistence or autoregressive models for influence prediction. Report performance degradation curves to quantify error propagation sensitivity.

2. **Include TiDE/TimeXer/ChronosX baselines:** These models explicitly handle exogenous variables and would isolate whether FIATS's gains come from textual representations or principled influence integration.

3. **Document the text generation pipeline:** Specify exact LLM prompts, sampling parameters, and quality validation steps in an appendix. Include a few raw examples showing the input data and generated captions.

4. **Analyze lagged influence scenarios:** Either discuss how the framework extends to delayed effects, or add an experiment with synthetically delayed influences to demonstrate robustness or identify limitations.

---

## Iq1fNZus2W

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (5.2/10)
- Match: N/A

### Final Review

## Summary

The paper addresses the quadratic computational cost of multi-condition control in Diffusion Transformers (DiTs) by proposing PKA (Patch-wise and Keyword-Aware Attention). PKA decomposes full attention into two specialized modules: Position-Aligned Attention (PAA) for spatial-aligned conditions and Keyword-Scoped Attention (KSA) for subject-driven conditions, along with a condition KV cache mechanism and an early-timestep sampling strategy for training. Experiments on FLUX.1 demonstrate significant efficiency gains—up to 10× inference speedup and 5.12× attention-module VRAM reduction—while maintaining or improving generative quality compared to baselines OminiControl2 and UniCombine.

## Strengths

- **Empirical motivation through attention analysis**: Figures 2 and 3 provide concrete visualizations of attention sparsity patterns—spatial conditions show strong diagonal concentration while subject-driven conditions exhibit keyword-localized activation. This empirical grounding directly justifies the proposed decomposition strategy and distinguishes the work from generic efficiency approaches.

- **Substantial efficiency gains with maintained quality**: Table 1 demonstrates that PKA matches or outperforms baselines on generative quality (FID, SSIM), controllability (F1, MSE), subject consistency (CLIP-I, DINOv2), and text fidelity (CLIP-T) across three multi-condition tasks, while achieving significant efficiency improvements (Figures 7–8). The efficiency-quality tradeoff is favorable.

- **Training insight from perturbation analysis**: Figure 5 and Appendix A.2 provide evidence that visual conditioning is most influential during early (high-noise) denoising steps, which motivates the early-timestep sampling strategy. Figure 13 shows this strategy yields faster convergence and better final SSIM compared to standard or late-biased sampling, offering a practical training heuristic beyond the architectural contribution.

## Weaknesses

- **Keyword extraction procedure is unspecified**: KSA fundamentally depends on identifying a "keyword set K" of 1–2 tokens from the text prompt (Equation 3), but the paper never describes how this extraction is performed—whether via manual annotation, a parser, LLM-based extraction, or heuristic. This is a critical implementation detail that affects reproducibility and applicability. Prompts without explicit subject keywords (e.g., "a beautiful scene") may fail silently.

- **Condition KV cache validity is unexamined**: The paper caches condition K/V projections after the first denoising step and reuses them throughout, assuming timestep-invariance. However, in DiT architectures like FLUX, condition tokens pass through transformer blocks with potential timestep-dependent modulation (e.g., via AdaLN). The paper provides no ablation comparing cached vs. freshly-computed K/V across timesteps, leaving open whether this optimization introduces approximation error.

- **PAA's softmax degenerates to a constant**: Equation 2 describes PAA as computing `Softmax(Q_i K_i^T / √d) V_i` for a single key-value pair. With only one key, the softmax output is identically 1 (the exponential divided by itself), reducing the operation to a simple projection `V_i` with no attention weighting. The paper presents this as "attention" when it mathematically functions as patch-wise fusion. While the efficiency gain remains valid, the framing obscures what computation is actually occurring.

- **Headline efficiency claims require contextualization**: The 10× speedup and 5.12× VRAM reduction apply specifically to the attention module and scale with condition count. Figure 7 shows speedup increases from ~3.9× (at lower condition counts matching Table 1's tasks) to 10× at higher condition counts. Additionally, VRAM claims are isolated to attention—not total model memory. For practitioners, the practical gains may be smaller than headlines suggest.

- **Scalability claims lack quantitative support for 3+ conditions**: The introduction and abstract emphasize "multi-conditional" and "high number of conditions" scenarios, yet Table 1 evaluates only 2-condition tasks. Results for 3 and 4 conditions appear only as qualitative figures in Appendix A.4 without corresponding speed, memory, or quality metrics. The core scalability claim is thus partially unsupported.

- **Generalizability limited to single backbone**: All experiments use FLUX.1 as the backbone. The proposed PKA exploits structural priors (diagonal attention patterns for spatial conditions, keyword-localized attention for subject conditions) that may vary across DiT architectures and training procedures. Without evaluation on SD3 or other DiT variants, generalizability cannot be assumed.

## Nice-to-Haves

- **End-to-end latency and total VRAM metrics**: Reporting only attention-module efficiency may obscure whether other components become new bottlenecks. Full pipeline timing and total model memory would give practitioners a complete picture.

- **PAA ablation with quantitative quality metrics**: Figure 9 shows qualitative results for PAA vs. sliding window baselines but reports only latency/VRAM. FID/SSIM/F1 metrics would verify that quality is maintained.

- **Evaluation on additional spatial condition types**: Testing on semantic segmentation maps or bounding box layouts (beyond Canny/Depth) would strengthen the claim that PAA generalizes across spatial-aligned conditions, since high-level layout conditions may require more global context than edge maps.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Missing baseline PixelPonder"**: PixelPonder is mentioned in related work. The paper compares against OminiControl2 and UniCombine, which are directly relevant DiT-based multi-condition methods. Requesting additional baselines beyond what is already compared is scope creep.

- **"Fair training comparison concerns"**: The paper states it fine-tunes FLUX.1 with LoRA and compares against OminiControl2 and UniCombine. While explicit confirmation of baseline retraining vs. published checkpoints would be helpful, this is standard practice and not a substantive flaw.

- **"Hyperparameter values not reported"**: The early-timestep sampling parameters (µ = 0.5, δ = 1.5) are explicitly stated in Figure 13 in the appendix. This criticism incorrectly claims they are absent.

- **"Test set size not specified"**: The paper mentions curating a subset from Subject200K with train/test partition. While exact test set size would improve reproducibility, this is a presentation gap rather than a fundamental flaw.

- **"No standard deviations or confidence intervals"**: Common practice in generative model papers. Would be nice to have, but not required by community standards for this venue.

- **"Large-subject images would negate efficiency"**: This hypothetical failure mode is not demonstrated experimentally. The paper's qualitative examples include varied subject sizes, and the KSA threshold provides user control over this tradeoff.

## Novel Insights

The perturbation analysis in Figure 5 and Appendix A.2 reveals an asymmetry in how visual conditions influence the denoising trajectory: perturbations applied in high-to-low (early-to-late) order rapidly degrade condition adherence, while low-to-high perturbations preserve fidelity longer. This suggests visual conditions establish global structure early, with later timesteps primarily refining details—a finding that could inform conditional generation research beyond this specific method. Additionally, the observation that condition tokens need not attend to the noisy image (enabling the KV cache) is a structural insight that challenges the implicit assumption in full concatenation-and-attend architectures, though its validity across architectures warrants further study.

## Suggestions

- **Add explicit description of keyword extraction**: Specify whether this is rule-based, model-based, or manual in the method section. If automated, describe the algorithm or model used.

- **Include ablation for condition KV cache**: Compare generation quality and efficiency with cached vs. freshly-computed K/V across timesteps to validate the timestep-invariance assumption.

- **Add quantitative results for 3+ conditions**: Include speed, memory, and quality metrics for at least one 3-condition setup to substantiate the scalability claims.

- **Clarify PAA's mathematical operation**: Either describe it as patch-wise gated fusion (removing softmax framing) or explain whether a local window is used instead of strict one-to-one alignment.

---

## bH5M0ts8Y6

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary

VINCIE proposes learning in-context image editing from native video sequences rather than curated paired image data. The authors construct 10M interleaved multimodal sessions from videos using VLM-generated transition descriptions and SAM2-derived segmentation masks, then train a DiT model on three proxy tasks: next-image prediction, current segmentation prediction, and next-segmentation prediction. The model achieves state-of-the-art performance on multi-turn editing benchmarks and demonstrates emergent capabilities in chain-of-editing, story generation, and multi-concept composition.

## Strengths

- **Innovative and scalable data paradigm:** Replacing manually curated or synthetically generated paired datasets with native video sequences is conceptually strong. The pipeline demonstrates clear log-linear scaling of later-turn success rates with training data size (Fig. 5), validating video as a high-signal supervision source.

- **Well-motivated proxy task design:** The trio of tasks (NIP, CSP, NSP) meaningfully addresses different aspects of in-context editing. Table 3 shows the chain-of-editing strategy (CS→NS→I) improves Turn-5 success from ~10% to ~17% compared to image-only prediction.

- **Clear empirical improvements over academic baselines:** On MSE-Bench, VINCIE achieves 22% Turn-5 success vs. <2% for prior open methods, demonstrating the practical value of video-derived training for multi-turn coherence.

- **Comprehensive architectural ablations:** Table 10 systematically evaluates attention mechanisms and RoPE configurations, showing text-then-image RoPE with full attention performs best for early turns while interleaved RoPE excels at longer contexts—a useful design insight.

## Weaknesses

- **Major evaluation metric discrepancy:** The paper reports 22% Turn-5 success on MSE-Bench via GPT-4o evaluation (Table 2), but human evaluation (Table 6) shows only 7% success—a three-fold gap. This discrepancy fundamentally undermines confidence in the benchmark results and the magnitude of claimed improvements. A 0.48 Pearson correlation between GPT-4o and human judgment (Table 7) is only moderate, leaving substantial variance unexplained.

- **MSE-Bench has methodological circularity:** GPT-4o generates the editing prompts for MSE-Bench and also serves as the evaluator. While this is disclosed, the circularity risks biasing evaluation toward instructions matching GPT-4o's internal preferences. With only 100 test instances and no confidence intervals reported, statistical reliability is limited.

- **Core claim attribution is unclear:** The paper emphasizes "trained exclusively on videos" and "learned solely from videos," yet the model is initialized from a pretrained MM-DiT (Section 4.1) that already encodes strong generation priors from T2I/T2V training. The video sessions teach context-conditioning, but the contribution of video-derived supervision versus foundation model initialization is conflated. Table 5 compares sequence vs. pairwise data using the same base model, but does not isolate what video uniquely provides beyond a random or T2I-only initialization.

- **SFT contribution to final performance is underexplored:** VINCIE+SFT achieves the best results on MagicBrush (Table 1), but SFT incorporates substantial pairwise editing data (OmniEdit, SEED-Edit, X2I2). The paper should disentangle whether the video-pretraining provides a meaningful scaffold versus simply being a different initialization for SFT.

- **VLM annotation pipeline unreproducible:** The visual transition annotation relies on an unspecified "in-house LMM." Human evaluation (Table 8) reports 75% accuracy and 69% recall, meaning ~1 in 4 annotations is incorrect. No analysis is provided on how annotation errors propagate through multi-turn sessions or affect final performance.

- **No failure mode characterization:** With 75-78% failure rate at Turn-5 on MSE-Bench, characterizing what instruction types or visual contexts cause failure would be scientifically valuable. The paper shows qualitative successes (Figures 15-18) but no systematic failure analysis.

## Nice-to-Haves

- Inference latency and memory profiling for the full-attention model over interleaved sequences would clarify practical deployment trade-offs.

- Quantitative metrics for positional drift mitigation (Figure 7 claims segmentation prediction helps, but provides only qualitative examples).

- External multi-turn benchmark validation beyond the authors' own MSE-Bench.

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **"Solely from videos is imprecise"** — The paper explicitly discloses in Section 4.1 that the model is "initialized from our in-house MM-DiT (3B and 7B), pre-trained on text-to-video tasks." The abstract's "trained exclusively on videos" refers to the session data after initialization, which is accurate.

- **"Equation indexing discrepancy (T₀ vs T₁)"** — Upon examination, Eq. 1 and Eq. 2 use consistent notation where I₀ is the initial frame and T₀ is the first instruction. The indexing is internally coherent; this is a minor notational concern inflated beyond its importance.

- **"Paragraph duplication in Section 4.1"** — This appears to be a PDF parsing artifact in the extracted text, not a paper writing error. The duplicated sentences describe the same implementation details and would not appear twice in the actual typeset paper.

- **"Novelty overlap with RealGeneral/UES"** — The paper adequately distinguishes itself by using long session sequences (2-20 frames) rather than 2-frame pairs, and by using native video without task-specific pipelines. This is sufficient differentiation.

- **"Baseline unfair comparison"** — Comparing VINCIE against methods with smaller models and less compute is standard practice. The paper provides appropriate context about the foundation model initialization and scales.

## Novel Insights

The most significant finding that emerges from cross-reviewing the evidence is the **human-evaluation gap**: the ~3x discrepancy between GPT-4o evaluation (22% Turn-5 success) and human evaluation (7% success) reveals that automatic metrics substantially overestimate multi-turn editing capability. This has implications beyond this paper—it suggests the community needs more robust evaluation protocols for generative editing tasks. Additionally, the log-linear scaling of long-horizon performance with training data (Fig. 5) is a valuable empirical finding that video data provides progressively more signal for context-dependent tasks as sequence length increases.

## Suggestions

- Report all MSE-Bench results using both GPT-4o and human evaluation, with confidence intervals, to establish the true performance bounds.

- Add an ablation comparing: (a) video session pretraining → SFT, (b) T2I/T2V pretraining → SFT directly, and (c) SFT-only, to isolate the video-specific contribution beyond foundation model initialization.

- Release the in-house VLM annotation code/specification or use a publicly available VLM to enable reproducible data construction.

- Include a failure mode analysis categorizing Turn-5 errors by type (e.g., identity drift, instruction misunderstanding, artifact accumulation) to guide future improvements.

- Provide the "video-only" checkpoint separately from the SFT version to allow the community to evaluate the core contribution independently.

---

## ZNAY3ivd62

- GT: Reject (avg 4.0)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary

GUI-Spotlight introduces an iterative visual grounding model for GUI agents that uses coordinated tool calls (extract, crop, find_color) to progressively narrow focus on target screen elements. The model is trained via a three-stage pipeline combining supervised fine-tuning with a modified Group Sequence Policy Optimization (GSPO) objective that includes an auxiliary cross-entropy loss on correct trajectories to prevent training collapse. The key result is achieving 52.8% accuracy on ScreenSpot-Pro using only 18.5K curated training samples—substantially less than comparable baselines trained on millions of examples.

## Strengths

- **Strong data efficiency with competitive accuracy**: The model achieves 52.8% on ScreenSpot-Pro with 18.5K samples, surpassing V2P-7B (50.6% with 9.6M samples) and GTA-1-7B (50.1% with 1.56M samples). This directly challenges the paradigm that GUI grounding requires massive-scale supervised data, making the method accessible to labs without large-scale annotation resources.

- **Comprehensive RL algorithm exploration with documented negative results**: The paper systematically compares seven GRPO/GSPO variants and explicitly reports which modifications degrade performance (e.g., continuous reference policy updates, top-p uncertainty sampling). The auxiliary J′(θ) term that filters to format-valid, result-correct trajectories effectively prevents policy collapse—a practical contribution for multi-turn tool-use RL that would benefit the broader community.

- **Rigorous data curation pipeline**: The three-stage LLM-audited filtering (Instruction Quality, Bounding Box Accuracy, Consistency) retains approximately 50% of the UGround dataset, providing a reproducible standard for cleaning noisy GUI grounding data. The transparency about filtering criteria (IoU ≥ 0.40 for consistency, clarity scores ≥ 6/10) is valuable for future work.

- **Clear agentic inference framework with offset tracking**: Algorithm 1's image registry design that maintains (image, offset) pairs across successive crops correctly handles coordinate recovery—a non-trivial engineering detail that ensures final coordinates map back to the original screenshot.

## Weaknesses

- **No inference cost or latency analysis**: An iterative, multi-turn approach that repeatedly crops and re-encodes images has inherent computational overhead. The paper reports no average tool calls per sample, latency measurements, or comparison to single-shot baselines on inference cost. For a method designed for practical GUI agents, this gap makes it difficult to assess deployment feasibility. The paper should at minimum report the distribution of tool calls (e.g., % samples resolved in 1, 2, 3+ steps) and latency overhead.

- **The `find_color` tool's input specification is underspecified**: The tool requires a "target RGB" to minimize perceptual color distance (ΔE in CIE Lab space), but the paper does not explain how this RGB triplet is obtained from the natural language instruction. Does the LLM infer it from visual features? Is there a color extraction module? Without clarification, the tool's applicability to arbitrary prompts and its failure modes remain unclear. The paper also lacks analysis of how frequently each tool is invoked and their individual success rates.

- **Limited improvement on OSWorld-G benchmark**: On OSWorld-G, GUI-Spotlight achieves 62.7% versus its UI-TARS-1.5-7B base model at 61.9%—a gain of only +0.8 absolute points. Meanwhile, GTA1-7B achieves 67.7% on the same benchmark. The paper does not discuss why the iterative approach provides substantial gains on ScreenSpot-Pro but minimal gains on OSWorld-G. This matters for assessing the method's generality.

- **Underperforms comparable model on UI-Vision, not acknowledged**: On UI-Vision, GUI-Spotlight achieves approximately 23.4% (as stated in the conclusion) while UI-Venus-Ground-7B achieves 26.5%—a meaningful gap of ~3 points. The paper discusses only gains over backbone base models but does not acknowledge this underperformance relative to a directly comparable baseline trained on similar data scale (107K samples vs. 18.5K). Honest engagement with this result would strengthen the paper.

- **No tool-level ablation or usage analysis**: The claim that "multi-tool coordination" drives improvement is not substantiated. The paper does not report what happens when individual tools are removed, nor does it show the learned distribution of tool calls across test examples. Without this, it's unclear whether the model learns meaningful tool selection or relies on a dominant tool with others providing marginal value.

- **Reward weights appear hand-tuned without sensitivity analysis**: The five reward components use fixed weights (α = 0.30, 0.25, 0.05, 0.20, 0.20) without justification beyond "we combine five rewards into a weighted sum." Section 4.2 shows ablations on reward types (dense vs. sparse) and Crop/Extract ratios but not on the primary weight vector. Readers cannot assess robustness.

- **No failure mode characterization**: The paper does not analyze cases where iterative refinement diverges or converges to incorrect locations. Given that the method involves multiple sequential steps, characterizing failure modes (e.g., off-target crops, cascading errors) is essential for understanding reliability in real deployment.

## Nice-to-Haves

- Statistical significance testing across multiple random seeds for RL experiments, though single-run evaluation appears to be the norm for these benchmarks.

- Ablation on maximum tool call limits (1, 2, 3, 5+ steps) to show accuracy-compute tradeoffs.

- Cross-backbone generalization on a third architecturally distinct model beyond UI-TARS-1.5-7B and Qwen2.5-VL-7B.

- Visualization of example trajectories showing screenshots at each refinement step.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Section 5.2 is missing"** — This is a PDF parsing artifact. The section content exists; the table placement in the parsed output confused the layout.

- **"Equation for J′(θ) appears displaced"** — Also a parsing artifact. The equation is correctly placed in the original paper.

- **"Figure 5 caption is duplicated"** — Parser artifact in extraction.

- **"No confidence intervals reported"** — Single-run evaluation is standard for large-scale benchmarks like ScreenSpot-Pro and UI-Vision; demanding multi-run statistics is not the field norm.

- **"Dataset composition doesn't sum correctly"** — Upon verification: 2,561 (Stage 1 trajectories) + 12,000 (Stage 2 samples) + 4,000 (Stage 3 samples) ≈ 18.5K. The numbers do sum correctly.

- **"High-resolution web data from narrow consumer websites may not generalize to CAD/creative tools"** — The model achieves strong performance on ScreenSpot-Pro's CAD and Creative categories (Table 3: 44.0% and 51.0% respectively), suggesting generalization beyond the training domain. The criticism overstates the limitation.

- **"Benchmark contamination risk from web-crawled data"** — This is speculative; the paper does not address it, but there's no evidence of contamination. Without external confirmation, this remains ungrounded speculation.

## Novel Insights

The most interesting finding that emerges from the paper but isn't explicitly highlighted: the **Extract reward being more beneficial than Crop reward** (Figure 4, right panel) suggests that coarse spatial reasoning is easier to learn than fine-grained coordinate prediction. This implies that GUI grounding models may benefit from a curriculum that first rewards approximate region identification before precise localization—a finding that could inform architecture and reward design beyond this specific method. Additionally, the observation that dense Answer rewards perform *worse* than sparse rewards (Figure 4, left) challenges the intuition that shaping rewards should always help; in multi-turn settings, dense rewards may introduce conflicting gradient signals when intermediate steps don't directly contribute to final correctness.

## Suggestions

- Add a table reporting average tool calls per sample, distribution of multi-turn episode lengths, and per-sample latency comparison to single-shot baselines.

- Explicitly describe how the `find_color` tool obtains target RGB values—if the LLM generates them, report success rates; if there's a color extraction heuristic, document it.

- Acknowledge the UI-Vision underperformance and OSWorld-G minimal gains in the text, with hypotheses about why ScreenSpot-Pro shows larger improvements (e.g., resolution differences, UI density).

- Add per-tool ablation results (model trained without each tool) to substantiate the claim that multi-tool coordination matters.

- Include 3-5 trajectory visualizations in the appendix showing success and failure cases with intermediate crops.

---

## JEN4nsDgh9

- GT: Reject (avg 3.5)
- Predicted: N/A (4.3/10)
- Match: N/A

### Final Review

## Summary

This paper proposes a benchmark for evaluating text-to-image (T2I) models on generating images for WordNet taxonomy concepts. The authors introduce nine evaluation metrics—combining preference-based ELO rankings (human and GPT-4), taxonomy-specific CLIP-derived similarity measures (Lemma, Hypernym, Cohyponym Similarity, Specificity), and standard metrics (FID, Inception Score)—to assess 12 open-source T2I models against a retrieval baseline across multiple dataset splits. The key finding is that model rankings differ from standard T2I benchmarks, with Playground-v2 and FLUX performing best.

## Strengths

- **Addresses an underexplored and legitimate research problem:** The paper systematically evaluates T2I models on structured lexical taxonomies, testing generalization across hierarchical, abstract, and rare concepts rather than standard object-centric datasets. This is a genuine contribution to understanding how well diffusion models can support knowledge base enrichment.

- **Comprehensive empirical infrastructure:** The evaluation triangulates multiple signals—human ELO rankings (4 annotators, Spearman correlation ρ≈0.8), GPT-4 pairwise evaluations, a preference-aligned reward model (Xu et al., 2024), and CLIP-derived taxonomy metrics—across 12 models and three dataset splits totaling ~3,370 items. The commitment to release the generated image dataset and preference data enhances reproducibility and community value.

- **Validates taxonomy-specific metrics against human judgment:** The paper demonstrates strong correlation between the proposed Hypernym/Cohyponym CLIP-Score metrics and human evaluation rankings (ρ≈0.87–0.91, p<0.001 in Section 4.2), providing empirical grounding for using these hierarchy-aware metrics.

- **Informative error analysis:** Appendix I identifies systematic failure modes—struggles with abstract concepts, rare/specific lemmas, functional roles, and parent-child confusion—which usefully characterizes current T2I limitations for taxonomy visualization.

## Weaknesses

- **Internal inconsistency in reported correlation values:** The abstract states the Spearman correlation between human and GPT-4 model rankings is 0.92 (p≤0.05), while Section 5 reports 0.88 with definitions and 0.73 without. The source of the 0.92 figure is unclear and the inconsistency erodes confidence in the precise numerical claims.

- **GPT-4 evaluator has near-zero correlation with humans at the individual battle level:** The paper documents that "we found no correlation between raw scores for individual battles" (Section 5), with strong positional bias toward the first option (Figure 5, Figure 12). This is a fundamental reliability concern—the aggregate ELO correlation may smooth over noise rather than reflect consistent judgments. The paper would be strengthened by analysis showing how many battles are needed for rankings to stabilize.

- **Overclaiming metric novelty:** The abstract claims "9 novel taxonomy-related text-to-image metrics," but FID, Inception Score, and the Reward Model are standard or existing. Only 4 metrics (Lemma, Hypernym, Cohyponym Similarity, Specificity) are genuinely taxonomy-specific contributions.

- **FID computed against retrieval images is misaligned with the task goal:** FID measures distance from the retrieved Wikimedia Commons distribution, which the paper demonstrates is low-quality. Models that mimic these suboptimal images score well on FID despite potentially poor semantic correctness. While acknowledged in Section 4.3, FID remains prominent in the results tables (Tables 2, 8, 10) without sufficient caveats.

- **Retrieval baseline is weak:** Using only top-1 Wikimedia Commons API search without reranking or dense retrieval limits the comparison. The finding that generation outperforms retrieval may partly reflect baseline limitations rather than a principled conclusion about retrieval versus generation for taxonomy coverage.

- **Missing statistical significance testing for CLIP-based metrics:** Tables 3–6 report ELO scores with confidence intervals, but Tables 11–15 (CLIP-based similarity metrics) present point estimates only, making it difficult to assess whether observed differences between models are meaningful or within noise.

## Nice-to-Haves

- **Inclusion of closed-source SOTA models (DALL-E 3, Midjourney):** Would strengthen practical utility, though acknowledged in Appendix A as a limitation. Practitioners seeking the best model for taxonomy visualization cannot directly apply the findings to state-of-the-art proprietary systems.

- **Ablation study isolating taxonomy metrics from standard CLIP Score:** Would clarify whether Hypernym/Cohyponym Similarity provides signal beyond general text-image alignment.

- **Analysis of performance versus taxonomy depth:** Quantifying correlation between synset depth and generation quality would strengthen the claim that models understand hierarchical structure rather than surface-level concepts.

## Removed Points

*These points were flagged for removal as they are factually incorrect, overly harsh, or outside scope:*

- **"Full WordNet-3.0 coverage claim unsubstantiated":** The abstract states they "publish the dataset of the images generated by the best Text-to-Image approach from the benchmark that fully covers WordNet-3.0." The evaluation uses ~3,370 items; the full dataset release is a separate contribution. The critic misread this as an evaluation claim.

- **"Conclusion overstates Playground's dominance":** Tables 5–6 confirm Playground ranks first in human preference ELO across conditions. The critic's confusion arose from conflating Figure 8 (GPT preferences without definitions, where FLUX leads) with human preferences.

- **"Theorems are trivial or circular":** While Theorem 1 is indeed MAP under uniform prior and Theorem 2 makes simplifying assumptions, the metrics themselves are empirically validated. Formal theoretical concerns are less critical given the empirical grounding.

- **"Missing closed-source models invalidates the paper":** This is acknowledged in Appendix A as a deliberate scope limitation (open-source models allow fine-tuning and reproducibility). It's a nice-to-have, not a flaw.

- **"CLIP trained on ImageNet biases metrics":** This concern is speculative without empirical demonstration. The Easy Concepts subset tests common nouns, while the Random and LLM-predicted subsets test rare/abstract concepts. The error analysis shows models struggle more with abstract concepts, suggesting the metrics capture meaningful variation.

- **"SuperGLUE citation error":** The critic correctly identified that Sarlin et al. (2020) is SuperGlue (image feature matching), not SuperGLUE (NLP benchmark). However, this is a minor reference error that doesn't affect the paper's technical content.

## Novel Insights

The finding that T2I model rankings for taxonomy visualization diverge from general T2I benchmarks is significant: aesthetic-focused models (SDXL-turbo) lead on CLIP alignment while preference-trained models (Playground, FLUX) lead on human judgment. This suggests that optimizing for visual appeal versus semantic precision creates different capability profiles. The positional bias in GPT-4 image evaluation (Figure 5) and near-zero individual battle correlation with humans is an important negative result for automated evaluation methodologies—the Bradley-Terry aggregation may mask fundamental reliability issues.

## Suggestions

1. **Reconcile correlation values:** Either correct the abstract to report 0.88 (the value from the main evaluation) or clarify what the 0.92 figure represents.

2. **Add confidence intervals to CLIP-based metric tables:** This would enable readers to assess statistical significance of model comparisons on Hypernym/Cohyponym Similarity and Specificity.

3. **Downweight or reframe FID:** Given that the retrieval reference distribution is demonstrably poor, explicitly frame FID as measuring "distributional similarity to web retrieval" rather than "quality" and de-emphasize it in conclusions.

4. **Quantify the rank shift from standard T2I benchmarks:** Compute Kendall's Tau or Spearman correlation between the reported ELO rankings and public leaderboards (e.g., GenAI Arena) to substantiate the claim that taxonomy visualization yields different model priorities.

5. **Strengthen the retrieval baseline:** Add a dense retrieval baseline using CLIP-ViT-L/14 embeddings over a curated image corpus to isolate whether generation beats modern retrieval or just naive keyword search.

---

## kMfVTka2WB

- GT: Reject (avg 2.0)
- Predicted: N/A (2.4/10)
- Match: N/A

### Final Review

## Summary

The paper proposes Covariance-Adjusted Support Vector Machine (CSVM), which incorporates class-conditional covariance structure into SVM classification. The authors argue that traditional SVM's margin formulation implicitly assumes Euclidean geometry, while data in "statistical space" should use Mahalanobis distance. They derive modified margin conditions showing margins scale with class covariance ratios, propose an iterative "SM Algorithm" to estimate population covariance from training data, and demonstrate empirical improvements over standard SVM kernels and global whitening methods on five benchmark datasets.

## Strengths

- **Clear geometric motivation:** The paper correctly identifies that standard SVM treats all dimensions equivalently regardless of class-specific variance structure, and provides an explicit derivation (Equation 14, Lemma 2.3) showing how margin width should scale with class covariance. This offers a transparent geometric interpretation of why class-conditional whitening can improve classification.

- **Class-wise whitening formulation:** Unlike PCA/ZCA whitening which applies a single global transform, the paper's approach of computing separate Cholesky decompositions per class is a principled choice when classes have genuinely different covariance structures. The empirical results (Tables 1-4) show CSVM outperforms global whitening baselines on 4 of 5 datasets across accuracy, recall, and F1 metrics.

- **Comprehensive metric reporting:** The experiments report accuracy, precision, recall, F1, and AUC across multiple datasets from diverse domains (healthcare, astronomy, quality control), providing a multi-faceted view of performance beyond single-metric evaluation.

## Weaknesses

- **Theoretical framing contains conceptual errors:** The paper's central claim that "statistical space is non-Euclidean" (lines 47-48, and throughout Section 2) conflates non-Euclidean *spaces* with non-Euclidean *metrics*. R^n with Mahalanobis distance is still a vector space—it simply uses a different inner product. The KKT conditions are first-order optimality conditions applicable to any differentiable constrained optimization problem, not exclusively to Euclidean-space formulations. The paper would be stronger if it framed the contribution as "metric adaptation for SVM" rather than "correcting invalid SVM theory."

- **Two optimization problems lack resolution mechanism:** Equations 10-13 derive separate optimization objectives for each class (minimizing θᵀ(Σ_{y=1})⁻¹θ versus minimizing θᵀ(Σ_{y=-1})⁻¹θ), both involving the same parameter θ. The paper never specifies how these competing objectives are combined into a single well-posed optimization problem. Without this, the theoretical derivation is incomplete—the algorithm cannot be derived from the lemmas as stated.

- **Intercept adjustment θ₀' is not specified:** Step 2(e) of the SM Algorithm states "adjust θ₀ to θ₀' so that the modified classifier divides the margin in the input space in ratio [covariance ratio]," but the exact formula for computing θ₀' is never provided. This makes the algorithm underspecified and raises reproducibility concerns.

- **SM Algorithm has data leakage:** Steps 2(f)-(g) assign labels to test points and incorporate them into the training pool for covariance re-estimation. This transforms the method into a transductive semi-supervised approach, yet the evaluation reports inductive generalization metrics (accuracy, AUC on held-out test data) as if it were a standard supervised classifier. Either the evaluation protocol should use a separate hold-out set never touched during training, or the paper should explicitly position the work as transductive learning and compare against appropriate baselines (e.g., Transductive SVM, self-training methods).

- **Experimental evaluation lacks statistical rigor:** All results use a single 80:20 train/test split without cross-validation, confidence intervals, or significance testing. Differences between methods are often small (e.g., CSVM 0.786 vs. SVM-Linear 0.760 on Diabetes accuracy), and without statistical tests, it is unclear whether observed improvements are genuine or due to split variance.

- **Inconsistency between Lemma 2.2 and implementation:** Lemma 2.2 states that an N-class problem produces N distinct classifiers in the input space, yet the implementation uses a single adjusted decision boundary. The paper does not reconcile this theoretical claim with the practical algorithm.

## Nice-to-Haves

- Comparison with LDA/QDA or Mahalanobis metric learning methods (LMNN, ITML) would strengthen the positioning, as these directly address class-conditional covariance in classification.

- Evaluation on higher-dimensional datasets would test scalability, since Cholesky decomposition incurs O(d³) complexity.

- Ablation study separating the contribution of class-conditional whitening from the iterative SM pseudo-labeling would clarify where performance gains originate.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Demand for LDA/QDA baselines:** While valuable, the absence of these comparisons does not invalidate the paper's core contribution. The paper already compares against relevant whitening baselines (PCA, ZCA) and standard kernels.

- **Claim that cited works are "misrepresented":** The paper cites prior work on Mahalanobis-based SVM variants (Tsang et al., Wang et al., Zafeiriou et al.); claiming novelty over these is reasonable if the dimensional consistency and vector-space formulation are genuinely new, which would require domain expertise to fully adjudicate.

- **Demand for theoretical proof of convergence:** Convergence analysis would strengthen the work, but the SM Algorithm is explicitly described as "heuristic" (line 520), and empirical demonstration of convergence behavior could substitute for formal proof.

- **Criticism about small datasets:** The five datasets, while not large-scale, span multiple domains and are reasonable for initial validation. This is a scope question rather than a fundamental flaw.

## Novel Insights

The most interesting insight in this paper is the explicit quantification of how margin width should scale with class covariance (Equation 14). While metric learning and covariance-adjusted classifiers exist, the specific derivation showing that margins split in ratio θᵀ(Σ_{y=-1})⁻¹θ : θᵀ(Σ_{y=1})⁻¹θ provides a clean geometric interpretation. The observation that class-wise whitening differs fundamentally from global whitening—because each class may belong to a different population with distinct covariance structure—is a practical insight that the empirical results support.

## Suggestions

1. **Correct the theoretical framing:** Replace "non-Euclidean space" language with precise statements about metric geometry. Frame the contribution as adapting SVM's implicit L₂ metric to a Mahalanobis-type metric informed by class-conditional covariances. State clearly that KKT conditions apply to any smooth constrained optimization—what changes is the metric, not the validity of optimality conditions.

2. **Resolve the optimization formulation:** Specify how the two class-conditional objectives combine. One natural approach would be a single weighted objective; alternatively, explain if the sequential algorithm implicitly handles this through iteration.

3. **Specify θ₀' explicitly:** Provide the formula for intercept adjustment to ensure reproducibility.

4. **Fix evaluation methodology:** Either (a) use a three-way split (train/validation/test) where test data is never used during iterative relabeling, or (b) explicitly classify the method as transductive and compare against appropriate transductive baselines.

5. **Add statistical testing:** Report means and standard deviations over multiple random splits, and use significance tests to support claims of improvement.

---

## c2ozZYoZFd

- GT: Reject (avg 2.7)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary

This paper provides a detailed case study critiquing Nguyen et al. (2024), a high-visibility ICLR 2025 Oral paper that introduced min-p sampling. Through systematic re-analysis of four evidence streams—human evaluations, NLP benchmarks, LLM-as-a-Judge evaluations, and community adoption claims—the authors demonstrate that the original paper's conclusions are not supported by its own data. The paper introduces a "Best-of-N" methodology for equalizing hyperparameter search budgets and derives six lessons for more rigorous empirical ML research.

## Strengths

- **Best-of-N hyperparameter control framework (Sec. 3.1):** The paper introduces a principled subsampling methodology to compare methods fairly by equalizing the volume of hyperparameter space searched. The extensive sweep (~6000 A100-hours across 9 models, 4 samplers, 31 temperatures, multiple seeds) provides strong reproducible evidence that min-p's apparent advantage disappears under equalized tuning conditions. This is the paper's most technically novel contribution.

- **Rigorous statistical re-analysis of human evaluations (Sec. 2.2):** The paper correctly identifies that the original pooled data across heterogeneous conditions to perform a single t-test and failed to correct for multiple comparisons. The re-analysis applying Bonferroni correction (reducing significant comparisons from 5/12 to 1/12 at α=0.05) and using an Intersection-Union Test to operationalize "consistently outperforms" is methodologically sound and directly contradicts the original claims.

- **Discovery of omitted evaluation data (Sec. 2.1):** The finding that 1/3 of collected human evaluation data (the basic sampling condition) was silently excluded from analysis—confirmed through public correspondence with the original authors—is a serious documented flaw that substantively changes the conclusions.

- **Exceptional transparency:** All raw data, annotations, analysis code, and sweep configurations are publicly linked. The manual annotations of qualitative responses are posted in full, enabling independent verification. This sets a strong reproducibility standard.

## Weaknesses

- **Qualitative annotation lacks methodological rigor (Sec. 2.3):** The manual annotation of human evaluators' qualitative preferences was performed by the re-analysis authors themselves without reported inter-rater reliability, blinding procedures, or annotation guidelines. Given that the annotators already believed the original paper was flawed before annotation, confirmation bias is a substantial concern. This section would be stronger with independent annotators or at minimum reporting Cohen's κ or similar agreement metrics.

- **Claims equivalence without formal testing (Sec. 2, 3):** The conclusion that min-p offers "no apparent advantage" rests largely on failing to reject null hypotheses. Without equivalence testing (e.g., Two One-Sided Tests) or power analysis, the paper cannot rule out that non-significance stems from limited sample size or metric noise rather than true parity. For human evaluations (n=53 per condition after exclusions), power may be insufficient to detect small but meaningful differences.

- **Narrow empirical scope for benchmark counter-evidence (Sec. 3):** The extensive Best-of-N evaluation is restricted to GSM8K CoT despite the original paper also testing GPQA. More critically, min-p's claimed utility is for creative generation—yet the paper provides no evaluation on open-ended, instruction-following, or creative writing tasks. A smaller-scale evaluation on these task types would address the exact use-case the original paper targeted.

- **Selective reporting accusation under-substantiated (Sec. 4.3):** The claim that Table 3(b) reported the higher of two scores for min-p but the lower for top-p relies on a Telegram link from the first author. While concerning if true, the presentation lacks: (a) clear numerical values in the main text, (b) explanation of what experimental condition generated "two scores," and (c) consideration of benign explanations. The win rate differences are small (52.01 vs. 50.14 for min-p; 50.07 vs. 50.43 for top-p), and without confidence intervals, it's unclear whether these are even distinguishable from chance (50%). A serious accusation of selective reporting warrants more rigorous documentation.

- **Blueprint lessons are not novel:** The six lessons—control for hyperparameter tuning, correct for multiple comparisons, practice data transparency, scrutinize qualitative summaries, ensure methodological clarity, watch for selective reporting—are established best practices already advocated in prior work (e.g., Agarwal et al., 2021; Biderman et al., 2024). The contribution is the demonstration of how violating these practices led a specific paper astray, not the principles themselves. The framing should acknowledge this.

- **Single case study limits generalization:** The "blueprint" is derived entirely from one paper. Without applying the audit framework to additional papers, readers cannot assess whether the methodology is broadly applicable or specifically tuned to detect min-p's particular flaws.

## Nice-to-Haves

- Formal equivalence testing with power analysis to support "no advantage" claims
- Evaluation on creative generation or instruction-following tasks where min-p claims benefits
- Application of the Best-of-N framework to at least one additional sampling paper to demonstrate generalizability
- Compute/latency comparison between samplers to assess practical deployment differences

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Why min-p was chosen" selection bias concern:** The paper explicitly states min-p was chosen for high visibility (ICLR 2025 Oral, 18th-ranked submission). This is a reasonable motivation for a case study; the selection bias concern is speculative without evidence.

- **Focus on high-diversity setting is not independent:** The authors disclose that they focused on high-diversity because the original authors stated low-diversity was "experimental" and had poorly-chosen hyperparameters. This is transparent disclosure, not a methodological flaw.

- **Temperature hyperparameter implementation unclear:** The paper describes the sweep (31 temperatures, 6 hyperparameters per sampler) and the subsampling procedure (150 random draws). The implementation details are adequately specified for reproduction.

- **Venue appropriateness concern:** Critique papers have precedent at ICLR. The paper offers methodological contributions (Best-of-N framework, statistical correction protocols) beyond pure rebuttal, making ICLR appropriate.

- **Section 2.4 "confounded changes" criticism:** The new human evaluation conducted by Nguyen et al. involved multiple changes. The paper correctly notes this limits direct comparison, but this is inherent to re-analysis work—the authors cannot control what the original authors choose to change.

## Novel Insights

The Best-of-N subsampling methodology provides a concrete, implementable technique for detecting potential cherry-picking in hyperparameter-heavy empirical work. Rather than simply complaining about unfair comparisons, this framework offers a constructive way to audit claims: if Method A outperforms Method B, does this hold when both methods receive equal search budgets? The approach could be standardized into a benchmarking protocol where reviewers request Best-of-N curves alongside standard performance tables. The paper also demonstrates that omitted data—the silent exclusion of 1/3 of human evaluation results—can flip paper conclusions, suggesting data release norms in ML may need enforcement mechanisms beyond author self-reporting.

## Suggestions

- Add power analysis and equivalence testing to Sections 2 and 3 so that "no advantage" claims are statistically grounded rather than relying on failed significance tests alone.
- Run at least one creative generation benchmark (e.g., AlpacaEval creative writing subset) under Best-of-N conditions to directly address min-p's claimed use-case.
- Have at least one additional case study (even briefly) to demonstrate the framework generalizes beyond this specific paper.
- Strengthen Section 4.3 by including the exact numerical values in the main text and confidence intervals around the win rates; if the selective reporting claim cannot be rigorously substantiated, frame it more cautiously.

---

## khHNHzRjMy

- GT: Reject (avg 3.0)
- Predicted: N/A (3.8/10)
- Match: N/A

### Final Review

## Summary

EmoSign introduces the first dataset of American Sign Language (ASL) videos annotated for sentiment and emotion, labeled by 3 Deaf native ASL signers with professional interpretation experience. The dataset comprises 200 video clips with 7-point sentiment ratings, intensity scores for 10 emotion categories, and open-ended descriptions of emotional cues. Baseline experiments across 4 multimodal LLMs reveal that current models struggle to integrate visual emotional cues and rely heavily on text captions, exhibiting systematic biases toward positive emotions.

## Strengths

- **Linguistically valid annotation methodology:** The decision to recruit Deaf native ASL signers as annotators is a significant strength. This ensures annotations correctly disentangle grammatical facial markers (e.g., raised eyebrows for yes/no questions) from affective expressions—a core technical challenge that hearing annotators would likely conflate. The paper explicitly motivates this in Sections 1 and 3.2, noting that "facial expressions simultaneously serve grammatical and emotional functions" in sign language.

- **Qualitative cue descriptions provide unique value:** Unlike prior datasets with only categorical labels, EmoSign includes open-ended descriptions of specific emotional cues (signing speed, facial expressions, body movement). Section 3.4 documents themes such as modified sign size and speed for emotional emphasis, providing linguistically grounded ground truth that future work can leverage for interpretability and grounding tasks.

- **Well-designed ablation experiments:** The three-condition evaluation (caption-only, video-only, video+caption) cleanly isolates the extent to which models rely on text versus visual input. The finding that video-only performance substantially lags behind caption-only performance (e.g., GPT-4o sentiment wF1: 5.97 vs 18.23) quantitatively demonstrates the visual grounding gap that the paper claims.

- **Inter-annotator agreement contextualized against established benchmarks:** The paper reports Krippendorff's α (average 0.593) and notes that established emotion datasets MELD (κ=0.43) and IEMOCAP (κ=0.48) have comparable or lower agreement. This transparency about label quality is appropriate and informative.

## Weaknesses

- **VADER-based video selection creates methodological tension with the paper's core claim:** The dataset was constructed by selecting the 100 most-positive and 100 most-negative utterances according to VADER scores computed on *English text captions*. However, the paper's central motivation is that visual emotional cues in sign language diverge from textual content. Using a text-based sentiment filter to select "emotional" videos for a dataset meant to capture visual emotion introduces a potential selection bias: the dataset may over-represent cases where text and visual emotion align, and under-represent cases where they diverge—the very cases most valuable for testing visual grounding. The paper acknowledges this obliquely in Section 6 ("VADER results differed from the annotators' results") but does not quantify the divergence. Reporting the fraction of videos where VADER sentiment differs from annotator sentiment would directly address this concern.

- **Low inter-annotator agreement for several emotion categories undermines ground-truth validity:** Table 2 reports Krippendorff's α of 0.119 for surprise (negative) and 0.166 for disgust—well below conventional thresholds for acceptable agreement (typically α ≥ 0.667). The paper presents benchmark accuracy on these categories (Table 4) as "ground truth" despite near-random agreement among annotators. The paper should either exclude these unreliable categories or explicitly caveat performance on them. The asymmetry between positive emotions (α ≈ 0.55–0.70) and negative emotions (α ≈ 0.12–0.37) also deserves deeper discussion—this pattern suggests that negative emotions may be genuinely harder to perceive in ASL, or that the emotion taxonomy itself may need refinement.

- **Dataset scale limits statistical reliability and model development:** At 200 clips (~16 minutes) from only 4 signers, the dataset is small even for a specialized benchmark. Several emotion categories have fewer than 20 examples (e.g., surprise-negative: 15, anger: 15 in the single-expression set). At this scale, accuracy percentages are highly sensitive to individual predictions—a difference of 1-2 clips can shift per-class accuracy by 5–15%. The paper appropriately cites comparable small datasets in niche domains, but this does not fully address the limited generalizability. The reliance on a single source corpus (ASLLRP, which uses controlled lab recordings) further limits diversity of signing styles and contexts.

- **No train/test split prevents supervised learning evaluation:** All experiments are zero-shot evaluations on the full 200 clips. While this characterizes current MLLM capabilities, it does not demonstrate whether the dataset can support model development via fine-tuning. The paper should explicitly state that EmoSign is currently positioned as a zero-shot evaluation benchmark and discuss plans for supervised splits if the intent is to support training.

- **Emotion cue grounding task lacks quantitative evaluation:** Section 4.1 introduces emotion cue grounding as a task, and Section 5.3 provides only qualitative analysis by "manually inspecting several randomly selected videos." The paper collects rich qualitative descriptions of emotional cues from annotators (a genuine contribution) but does not use them to quantitatively evaluate model grounding—for example, by computing overlap between model-identified cues and annotator-described cues, or by evaluating whether models' reasoning aligns with ground-truth descriptions. Without quantitative grounding evaluation, the claim that "models fail to integrate visual cues" rests on indirect evidence (classification accuracy) rather than direct grounding analysis.

- **Unclear number of annotations per clip and tie-handling details:** The paper states "minimally 1, maximally 3 annotators" but does not report how many clips received fewer than 3 annotations, how often ties occurred in majority voting, or how frequently the "most confident annotator" tie-breaker was invoked. Given the small annotator pool, these details affect the reliability of the aggregated labels.

## Nice-to-Haves

- **Specialized computer vision baselines:** Including traditional facial expression or pose-based classifiers (e.g., OpenFace feature extraction with a simple temporal model) would help distinguish whether poor video-only performance is due to fundamental task difficulty versus MLLM architectural limitations. The paper concludes that "specialized visual encoders" are needed—demonstrating this with a baseline would strengthen the claim.

- **Multi-label emotion classification:** The paper notes that 37 clips contain multiple emotions but excludes them from evaluation. Even a preliminary multi-label analysis (macro-F1, subset accuracy) would better reflect the compositional nature of real-world affect and increase benchmark comprehensiveness.

- **Quantitative analysis of text-visual mismatches:** Cases where VADER sentiment differs from annotator sentiment are precisely where the dataset is most valuable for testing visual grounding. Analyzing model performance specifically on these mismatch cases would directly test whether models can rely on visual cues when text is misleading.

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **"Generalization to other sign languages not discussed"** — The paper is explicitly titled "American Sign Language" and focuses on ASL throughout. Criticizing the absence of BSL, LSF, etc. is scope creep; the paper should be evaluated on what it claims to contribute, not on covering all sign languages.

- **"Broader impact section missing risks of emotion recognition on Deaf individuals"** — This is a valid consideration for deployment papers, but the current work is a dataset/benchmark contribution. Ethical deployment concerns, while important, are outside the paper's stated scope.

- **"Formatting issues and scrambled text in Section 5.3"** — The instructions explicitly note that formatting artifacts are parser issues, not paper problems. This is not a legitimate criticism.

- **"MiniGPT4 is a dated baseline"** — MiniGPT4-video (Ataallah et al., 2024) is a reasonable contemporary baseline; including stronger models would be a nice-to-have improvement, not a core flaw.

## Novel Insights

The paper's most striking empirical finding is the asymmetry in how models handle text versus visual modalities: when video+caption inputs are provided, models often use text to *post-hoc rationalize* visual interpretations rather than genuinely integrating both modalities. Figure 3's examples show that the same visual cue (e.g., signing style) is interpreted oppositely depending on caption availability—"frustration" without caption versus "worry" with caption. This suggests that MLLMs may not be performing grounded visual reasoning on sign language content, but rather using text as a semantic anchor around which to hallucinate plausible-sounding visual justifications. This phenomenon parallels findings in other visual grounding domains but is particularly consequential for sign language, where non-manual markers systematically differ from spoken-language facial expressions.

## Suggestions

- Report the explicit fraction of clips where VADER sentiment differs from annotator sentiment, and analyze model performance on this subset to directly test visual grounding capability.

- Either exclude or explicitly caveat the two emotion categories with unacceptably low inter-annotator agreement (surprise-negative, disgust), explaining that ground-truth labels for these categories are unreliable.

- Add a train/test split (even a small held-out test set) and demonstrate that the dataset can support supervised fine-tuning, not just zero-shot evaluation.

- Provide quantitative evaluation of the emotion cue grounding task by comparing model reasoning outputs against annotator cue descriptions (e.g., keyword overlap, or human evaluation of whether cited cues are present).

---

## vGkXf8nvt9

- GT: Reject (avg 4.7)
- Predicted: N/A (5.3/10)
- Match: N/A

### Final Review

## Summary

The paper introduces *Forget-to-Focus (F2F)*, a two-stage protocol where LLMs first undergo targeted unlearning on general-domain data (with an optional retain set for stability), then fine-tune on domain-specific data. The authors argue that proactively suppressing irrelevant pretraining knowledge improves subsequent specialization by reducing negative transfer. Experiments across coding, medical, and mathematical domains on models from 0.6B to 72B parameters show consistent improvements over standard SFT, DAPT, and PEFT baselines (e.g., HumanEval improving from 31.71 to 42.07 on Qwen-0.6B), along with calibration improvements and representational geometry shifts analyzed via CKA, SVCCA, and Fisher information.

## Strengths

- **Novel conceptual reframing**: Repurposing machine unlearning from a privacy mechanism to a preparatory optimization step for domain adaptation is conceptually innovative and addresses a real pain point (negative transfer and spurious priors from pretraining).

- **Comprehensive empirical evaluation**: The paper tests across five model families (Qwen-0.6B, Gemma-2B, LLaMA-8B, LLaMA-13B, Qwen-72B), three domains (coding, medicine, mathematics), and multiple adaptation methods (SFT, LoRA, DAPT, CurlLoRA). The gains are substantial and consistent—for LLaMA-8B HumanEval improves from 33.54 to 60.37.

- **Multi-faceted mechanistic analysis**: Beyond aggregate metrics, the paper employs CKA, SVCCA, Fisher information, and PCA-shift analyses to show that unlearning reshapes representational geometry and dampens shallow-layer sensitivity, providing evidence for a capacity-reallocation mechanism.

- **Calibration improvements**: Table 7 shows ECE dropping from 0.277 (base tuned) to 0.050 (F2F) on MedMCQA, a meaningful improvement for high-stakes domains like medicine.

- **Systematic ablations**: The appendix includes ablations on forget-set quality (BC-Select vs. BC-Mixed vs. BC-Cosine), retain-set size, gradient weighting (λ/σ), learning rates, and multi-seed robustness (Table 9 shows low variance across 3 seeds).

## Weaknesses

- **Retain set creates a data exposure confound**: The paper states (Section 3.3) that "The retain set is a small subset of the fine-tuning data." This means F2F exposes the model to domain-specific examples during the unlearning phase (via the retain set), whereas standard fine-tuning baselines see this data only during fine-tuning. This additional exposure—not the unlearning mechanism—could partially explain performance gains. A proper control would use a retain set from out-of-domain data.

- **No compute-normalized baseline comparison**: F2F requires T_u unlearning steps + T_retune fine-tuning steps, while baselines use only fine-tuning steps. The paper does not compare against standard fine-tuning run for the same total optimization budget. Without this control, gains may stem from additional training rather than the proposed mechanism.

- **Forget set specificity is underspecified**: The method uses 100–1000 samples from BookCorpus as the forget set, claiming to remove "harmful" knowledge. However, BookCorpus is generic fiction text—it is unclear why gradient ascent on fiction would selectively remove coding- or medicine-interfering features. The paper lacks an ablation comparing curated forget sets against random data or other out-of-domain corpora, leaving the "negative transfer mitigation" hypothesis insufficiently validated.

- **Statistical rigor is inconsistent**: Main results (Tables 1–3) report single-run point estimates without error bars. While Table 9 provides multi-seed robustness checks in the appendix, the headline comparisons lack variance estimates, making it difficult to assess significance for small improvements (e.g., the 1–3pp gains over CurlLoRA).

- **Model instability during unlearning phase**: Table 1 shows that UnlGA+GD achieves 0.00 on HumanEval for Gemma-2B, and UnlGA achieves 0.00 for LLaMA-2-13B, before fine-tuning recovery. These intermediate collapses indicate the unlearning phase can destabilize some models, which is a practical concern for deployment.

- **Theoretical analysis does not bridge to LLM reality**: The Proposition and Corollary assume convex linear models, orthogonal feature decomposition, and strong convexity—assumptions explicitly acknowledged as not holding for LLM training (lines 132–133). While the paper is transparent about this, the theory remains disconnected from the empirical results and provides no actionable insight for hyperparameter selection or mechanism validation.

## Nice-to-Haves

- **Direct probing of negative transfer**: Include gradient interference analysis or feature probing to demonstrate that specific spurious correlations (e.g., web-text patterns conflicting with medical terminology) are actually reduced after unlearning, rather than relying solely on representational drift metrics.

- **Cost-benefit analysis of compute overhead**: Report GPU-hours or FLOPs for the unlearning stage to help practitioners assess whether the accuracy gains justify the additional training cost versus simply fine-tuning longer.

- **Dynamic forget-set construction**: Propose a method to automatically identify harmful pretraining data based on target-task validation gradients, making the approach more actionable for new domains.

## Removed Points

These points are flagged to be removed, treat them with caution:

- *Abstract claim inconsistency about 11.95%*: While the exact comparison basis for the Qwen-72B improvement claim is unclear, the overall gains are empirically verified in Table 1. This is a presentation issue, not a methodological flaw.

- *Claim that improvements may reflect benchmark contamination*: HumanEval and MBPP are widely used, but the same concerns apply to all fine-tuning papers. There is no evidence of contamination specific to this work, and the claim is speculative.

- *Demand for attribution or logit-lens analysis*: These would strengthen the paper but are not standard requirements for empirical fine-tuning papers. The current mechanistic analysis (CKA, SVCCA, Fisher) exceeds typical practice.

- *Criticisms about missing related work references*: Without external verification, claims about missing citations cannot be substantiated and may be incorrect.

- *Complaint about Qwen-72B using QLoRA with quantization*: The paper transparently reports this in Section 3.4. This is a practical adaptation for large-scale experiments, not a hidden flaw.

## Novel Insights

Beyond the paper's own contributions, the reviews surface an important observation: F2F may function similarly to optimization landscape regularization methods (akin to sharpness-aware minimization or adversarial training), where perturbing parameters via gradient ascent creates a smoother optimization trajectory. The strong baseline performance of CurlLoRA (40.91 HumanEval without unlearning vs. 42.07 with F2F) suggests that much of the benefit may come from parameter-space regularization rather than precise knowledge removal. This reframes F2F less as a targeted unlearning mechanism and more as a principled initialization strategy—a distinction that matters for how we interpret the results and design follow-up experiments. The calibration improvement (ECE 0.277 → 0.050) is an underexplored strength with practical implications for deployment in high-stakes domains.

## Suggestions

- Add a compute-normalized baseline: report results for standard fine-tuning run for T_unlearn + T_finetune total steps to isolate the effect of the unlearning mechanism from extended training.

- Control for retain-set exposure: include an experiment where the retain set is drawn from out-of-domain data rather than the fine-tuning data, ensuring fair comparison with baselines.

- Add forget-set specificity ablation: compare against gradient ascent on random Gaussian noise or shuffled data to test whether the semantic content of the forget set matters.

- Report standard deviations or confidence intervals for main table results to establish significance, even if multi-seed results exist in the appendix.

---

## ZMzha5gbnF

- GT: Accept (Poster) (avg 7.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Final Review

## Summary

This paper identifies the "priming vulnerability" in Masked Diffusion Language Models (MDLMs): affirmative tokens appearing at intermediate denoising steps can steer subsequent generation toward harmful responses, even in safety-aligned models. The authors propose First-Step GCG, an efficient attack deriving a tractable lower bound via a monotonicity assumption, and Recovery Alignment (RA), an RLHF-style defense that trains models to recover from contaminated intermediate states. Experiments across three MDLMs show RA substantially mitigates vulnerability while preserving utility.

## Strengths

- **Architecture-specific vulnerability formalization:** The paper correctly identifies that MDLM inference differs mechanistically from autoregressive generation—tokens fixed at intermediate steps remain fixed and bias subsequent denoising. The "anchoring attack" cleanly parameterizes attack strength via the intervention step $t_{inter}$, and Figure 2 shows a striking increase from 2% to 21% ASR at $t_{inter}=1$ alone.

- **Efficient attack derivation:** Theorem 4.1 derives a tractable lower bound on the attack objective, avoiding high-variance Monte Carlo sampling. First-Step GCG achieves ~20× speedup over MC-GCG while substantially improving ASR (e.g., 58% vs. 20% on LLaDA Instruct, Table 1).

- **Principled defense with curriculum scheduling:** Recovery Alignment is well-motivated: standard alignment minimizes $P(\text{harmful} | q, r_0)$ but does not constrain $P(\text{harmful} | q, r_t)$ for contaminated $r_t$. The linear intervention schedule enables gradual curriculum learning. Ablations in Figure 3b confirm linear scheduling outperforms constant and uniform alternatives.

- **Comprehensive experimental scope:** The paper evaluates three MDLMs, seven attack types (intervention-based and prompt-optimization), three safety evaluators (GPT-4o, guardrail model, keyword matching), and eleven utility benchmarks. Code and detailed configurations are provided.

## Weaknesses

- **Training data inconsistency between method and baselines:** Section 6.1 states RA uses BeaverTails, while Appendix D.6 states baselines (SFT, DPO, MOSA) use PKU-SafeRLHF. Though both from Ji et al. (2023), they differ functionally—RA uses (query, harmful response) pairs while baselines use preference pairs. Section D.4.2 further notes RA was trained on the *entire* BeaverTails dataset including harmless pairs to avoid over-refusal. This asymmetry could confound comparisons and requires explicit acknowledgment or ablation.

- **Monotonicity assumption validated only in the assumed regime:** Theorem 4.1 relies on $\log \pi_\theta(\tilde{r}_{t+1}=r|q,r_t) \geq \log \pi_\theta(\tilde{r}_1=r|q,r_0)$ for all $t$. Appendix C.2 validates this empirically using the anchoring attack to construct contaminated states—precisely the same regime where the attack exploits the vulnerability. This creates a circular validation: the assumption holds because priming vulnerability exists, and the attack works because the assumption holds.

- **ReNeLLM robustness worsens on MMaDA after RA:** Table 3 shows ReNeLLM ASR increases from 79.3% to 81.7% on MMaDA after RA, while improving on other attacks. This anomaly is unexplained and undermines the claim that RA "improves robustness against conventional jailbreak attacks" (Abstract).

- **Residual vulnerability at late intervention steps:** At $t_{inter}=32$ (25% of generation length), RA still yields ~50% ASR across models. The paper acknowledges this but does not bound the limits of RA's protection or propose mitigation beyond the current curriculum.

- **Reward model choice underexplored:** RA uses DeBERTaV3 "without additional fine-tuning" as the reward model. The sensitivity of RA to reward model choice is not evaluated, and whether a stronger safety-specific reward model would improve results remains unexamined.

- **Denoising intervention threat model realism:** The first threat model (attacker intervening in denoising process) assumes access to intermediate states unavailable in deployed APIs. While useful for vulnerability characterization, this represents system-level compromise rather than typical adversarial access. The paper labels this "hypothetical" but could more explicitly bound practical implications.

- **No benign refusal rate reported:** Safety alignments often cause over-refusal on benign inputs. Table 4 reports capability benchmarks but not false positive rates on harmless queries.

## Nice-to-Haves

- **DPO-style supervised alternative:** The Limitations section notes a DPO-style instantiation would reduce data-construction costs. A prototype evaluation would clarify practical trade-offs.

- **Adaptive attack evaluation:** Testing RA against attackers aware of the defense (e.g., targeting intervention steps outside $[t_{min}, t_{max}]$) would strengthen security claims beyond the fixed attack suite.

- **Qualitative examples of reward hacking:** The paper mentions "meaningless responses" emerge with large $t_{max}$ but provides no examples, leaving the practical limits of the RLHF objective underspecified.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Claim that "discovery" is overclaimed given concurrent work:** The paper acknowledges Zhang et al. (2025) and Wen et al. (2025) and positions its contribution as providing clearer formalization and quantitative evaluation. This is adequately handled in Section 2.2.

- **Request for MC-GCG Pareto frontier or cost-controlled comparison:** The 20× speedup with higher ASR is already strong evidence; additional Pareto analysis is a nice-to-have extension.

- **Omission of GSM8K/MATH benchmarks:** The evaluation includes HumanEval for code generation, and the 11 benchmarks span reasoning, knowledge, and utility. Math benchmark inclusion would be informative but is not a critical gap.

- **Request to extend to continuous DLMs:** The paper scopes to MDLMs throughout; extending to continuous DLMs is out of scope for this work.

- **Request for inference latency analysis:** RA is a training-time alignment method that does not modify inference. Latency overhead is not applicable.

- **Request to add math reasoning benchmarks:** The capability evaluation is already comprehensive; this is a generic expansion request.

## Novel Insights

The "recovery" framing—training on contaminated intermediate states rather than only clean initializations—provides a principled approach to safety that may generalize beyond MDLMs. The insight that standard alignment minimizes $P(\text{harmful} | r_0)$ but ignores $P(\text{harmful} | r_t)$ for adversarially perturbed states could inform safety research across generative architectures. The connection between intervention step scheduling and curriculum learning is a practical contribution that warrants broader exploration in adversarial training.

## Suggestions

1. **Provide ablation with unified training data:** Train baselines on BeaverTails or RA on PKU-SafeRLHF to isolate the method's contribution from data differences.

2. **Investigate and explain the ReNeLLM regression on MMaDA:** This anomaly requires at minimum acknowledgment and preferably diagnosis.

3. **Report benign refusal rates:** Evaluate on harmless prompt datasets to quantify over-refusal, a standard safety metric.

4. **Add an adaptive attack experiment:** Run First-Step GCG against RA-aligned models with intervention steps beyond $t_{max}$ to test generalization.

5. **Clarify reward model calibration:** Report reward score distributions for safe vs. harmful responses under DeBERTaV3, or ablate with alternative reward models.

---

## pNpnqsn0Si

- GT: Reject (avg 3.0)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

# Final Meta-Review

## Summary
Thoughtbubbles introduces a Transformer variant that learns to dynamically fork and prune residual streams during pretraining using only language modeling loss. Tokens that require more computation can spawn "bubbles" of cloned residual streams for additional parallel processing, with allocation controlled by learned cumulative scores. The method achieves consistent perplexity improvements over parameter-matched baselines and computation-matched "copy-N" baselines across three model scales (150M–772M) on OpenWebText and peS2o.

## Strengths
- **Novel architectural contribution to unsupervised adaptive computation:** The score-attenuated forking mechanism enables dynamic compute allocation during standard pretraining without auxiliary supervision, explicit thinking tokens, or specialized fine-tuning. This is genuinely different from prior approaches like pause tokens (which are non-adaptive) and chain-of-thought (which requires serial generation and supervised training). (Evidence: Section 2.3–2.4; the model is trained solely on cross-entropy loss while learning to allocate computation via cumulative scores.)

- **Consistent empirical improvements across scales and datasets:** The method achieves lower perplexity than both parameter-matched baselines and computation-matched copy-N baselines at all three scales on two distinct corpora. The 319M κ=4L model achieves comparable perplexity to the 772M baseline while using ~40% fewer parameters, suggesting meaningful efficiency gains. (Evidence: Table 1; Figure 3 shows consistent perplexity improvements at each scale.)

- **Mechanistic evidence that computation is allocated meaningfully:** Fork count correlates with token entropy (moderately high-entropy tokens receive more forks), and the CLUTRR analysis shows forks concentrate at interpretable decision boundaries (coreferences, query boundaries). Parent tokens attend strongly to their forks (Fig. 4), indicating forks contribute meaningfully to parent computation rather than being ignored. (Evidence: Figures 4, 5, 7; Section 5 analysis.)

## Weaknesses
- **Unexplained gradient flow through non-differentiable top-k selection:** The forking decision uses hard top-k operations (Eqs. 5–6), which are non-differentiable. The paper never explains how gradients propagate through this bottleneck during training. While Section 8 acknowledges gradient issues at deeper layers, no mechanism (e.g., straight-through estimator, Gumbel-softmax relaxation) is described, making the training procedure non-reproducible. This is a significant methodological gap.

- **Missing comparison to the most natural baseline—Mixture-of-Depths:** MoD (Raposo et al., 2024) adaptively routes tokens through or around transformer layers with no auxiliary loss, and is explicitly cited in Related Work. Its omission from experiments is a serious gap; comparing against MoD would establish whether the proposed parallel forking approach offers genuine advantages over adaptive layer routing. (Evidence: MoD is cited in Section 6 but excluded from Table 1.)

- **Sub-Chinchilla training budget limits conclusions about scale:** All models train for only 2.5B tokens, which is significantly below Chinchilla-optimal for 319M (~6.4B tokens) and 772M (~15.5B tokens) scales. Results at this training budget may not reflect relative architecture performance at standard pretraining scales, particularly for an architecture that must learn non-trivial scoring behavior from scratch. (Evidence: Section 3.2; standard Chinchilla scaling laws.)

- **Contradiction in Appendix B overforking ablation:** Table 4 shows "Ours (extended forking)" with perplexity 28.02 versus "Ours" with 29.84—indicating extended forking performs better. Yet the text claims "extended forking approach is slightly worse than forking only in the beginning." This direct contradiction requires correction. (Evidence: Appendix B, Table 4 vs. surrounding text.)

- **Approximate FLOP-matching without precise accounting:** The paper claims κ=4L is "roughly FLOPs-matched" against copy-5 but provides no FLOP counts. Attention complexity is O(N²), and dynamic sequence expansion makes the accounting non-trivial. Without precise measurements, the computation-matched claim remains unverified. (Evidence: Section 3.4 states "roughly FLOPs-matched" without derivation.)

- **No statistical significance across single-run results:** Each configuration is trained once. Several comparisons show small differences (e.g., PIQA 772M OpenWebText: baseline 62.3 vs. ours κ=4L 61.9), and without error bars, it's unclear whether observed improvements exceed random variation.

## Nice-to-Haves
- Evaluation on multi-step reasoning benchmarks (GSM8K, ARC-C) would strengthen the claim that adaptive compute benefits complex reasoning, even if conducted at smaller scales.
- A scaling law fit or experiments beyond 772M parameters would help establish whether benefits persist at production scales.
- Explicit algorithm pseudocode in addition to equations would improve reproducibility, particularly for the forking loop and top-k routing.
- Training loss curves to demonstrate stability across the full training run.

## Removed Points
*These points are flagged to be removed, treat them with caution*

- **"LAMBADA absolute scores are surprisingly low compared to GPT-2":** This comparison requires verifying that evaluation protocols match (e.g., word-level vs. token-level scoring, exact match vs. probability-based). Without confirming protocol equivalence, this criticism may be comparing incompatible metrics. (Harsh critic)

- **"Autoregressive generation latency and KV cache complexity":** While valid engineering concerns, these are implementation optimization issues rather than core methodological flaws for a research contribution. The paper already acknowledges wall-clock inefficiency in Limitations. (Harsh critic)

- **"No test-time scaling experiments despite claiming to unify train/test-time scaling":** This abstract framing is aspirational; the core contribution is showing unsupervised adaptive computation is learnable during pretraining. The lack of test-time scaling experiments doesn't invalidate the actual contribution made. (Harsh critic)

- **"High-entropy tokens being low-information words is an alternative explanation for the entropy-fork pattern":** This is not an alternative to the authors' explanation—it's consistent with their claim that high-entropy regions include clause boundaries (punctuation, prepositions). The critique doesn't undermine the observed correlation. (Harsh critic)

- **"Forking only at layers 3, 7, 11 means 70% of network has no adaptive behavior":** This is a design choice. The ablation in Appendix B shows that placing forking at earlier layers is preferred over extended forking (despite the text contradiction). The criticism is that a different design could have been chosen, not that the chosen design fails. (Harsh critic)

- **"Copy-N baseline is naive":** This is explicitly the point—copy-N provides a non-adaptive, computation-matched reference. The baseline fairly tests whether adaptive allocation provides benefits over simply expanding sequence length uniformly. (Spark finder)

## Novel Insights
Beyond the paper's own contributions, the analysis reveals an important tension: Thoughtbubbles demonstrates that *meaningful* adaptive computation allocation is learnable from LM loss alone, but the entropy-fork relationship shows a concave (not monotonic) pattern—the model allocates *less* compute to highest-entropy tokens. This suggests a sophisticated learned heuristic: moderate uncertainty (e.g., choosing among plausible continuations) benefits from additional computation, while extreme uncertainty (e.g., genuinely ambiguous tokens) does not. This insight—that not all uncertainty is equally worth investing compute in—deserves further investigation.

## Suggestions
1. **Add explicit description of the gradient estimation method** for the top-k routing decision in Section 2.3, even if only a straight-through estimator is used.

2. **Include Mixture-of-Depths as a baseline** to compare adaptive parallel forking against adaptive layer routing, which is the closest competing approach in the literature.

3. **Correct the Appendix B text** to match Table 4, or clarify the apparent contradiction between the stated conclusion and the reported perplexity values.

4. **Provide explicit FLOP counts** for all configurations to substantiate the computation-matching claim, noting that attention complexity is quadratic in sequence length.

5. **Consider evaluating on at least one reasoning benchmark** (e.g., CLUTRR, since the paper already uses it for analysis) to validate that adaptive compute benefits structured reasoning tasks.

---

## CTEXdHB1BB

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Final Review

## Summary

CANON introduces a conditional advantage estimation framework for RLVR that partitions sampled responses into high/low metric groups and computes inter-group advantages (to identify beneficial metric trends) and intra-group advantages (to select better responses within each trend). The method avoids hand-crafted directional priors (higher/lower is better) by letting the data reveal which metric direction correlates with success. Empirical results show consistent improvements over DR.GRPO across six math benchmarks and three logic tasks on three LLMs, with particularly compelling efficiency gains that establish a superior Pareto frontier.

## Strengths

- **Clear motivation addressing a genuine pain point:** The paper correctly identifies that existing reward/advantage shaping methods require hand-crafted directional priors (e.g., "higher-entropy-is-better") that are brittle and context-dependent. The core insight—regrouping by metric values and comparing across groups to discover beneficial trends—is conceptually clean and well-formalized in Equations 3–5.

- **Theoretical grounding with meaningful connections:** Theorems 1 and 2 provide formal justification for equal-sized grouping (showing it maximizes advantage signal ratio relative to DR.GRPO when |Cq⁺| is held fixed) and selective amplification (CANON based on metric c₁ won't amplify influence of independent condition c₂). The unification that DR.GRPO equals CANON with µ = 0.5 is an insightful theoretical connection.

- **Comprehensive empirical validation across benchmarks and models:** CANON is evaluated on six math benchmarks (including AIME, Olympiad, MATH-500) and three complexity levels of ZebraLogic across Qwen2.5-Math-7B/1.5B and Llama3.1-8B. Results consistently show gains over DR.GRPO, GRPO, RLOO, ReMax, REINFORCE++, and recent entropy/length-shaping methods (Tables 1–3).

- **Compelling efficiency results with stable Pareto frontier:** The CANON-Eff experiments (Table 3, Figure 4) demonstrate stable exploration of the performance-efficiency tradeoff. Notably, Length Reward(+) collapses when its coefficient changes from 0.004 to 0.005 (performance drops from 54.8 to 22.5), while CANON-Eff remains stable across α ∈ {0.5, 0.7, 0.8, 0.88, 0.96}. This practical robustness is a significant contribution.

- **Random regrouping ablation validates design:** Table 12 shows that random regrouping produces the same accuracy as DR.GRPO with slightly longer responses, confirming that meaningful metric-based grouping is essential to CANON's gains.

## Weaknesses

- **CANON-Dynamic results rely on model-specific strategy selection without transparent validation protocol:** Section 5.2 tries four scheduling strategies and selects different ones for different models (Cosin-First-Inter-Later-Intra for Qwen2.5-Math-7B and Llama3.1-8B; First-Inter-Later-Intra for Qwen2.5-Math-1.5B). The paper states results are "derived from one of the tried scheduling strategies that achieve strong performance in both scenarios" but does not explain how this selection was made without access to test performance. This is effectively reporting best-case hyperparameter configurations without held-out validation, undermining reliability of the aggregate CANON-Dynamic comparison in Figure 3 and Table 2.

- **Theorem 2's independence assumption is likely violated in practice:** The theorem assumes P(o ∈ C₁ ∩ C₂ | q, θ) = P(o ∈ C₁) · P(o ∈ C₂) for independent conditions c₁ and c₂. However, entropy and response length of LLM outputs are substantially positively correlated (longer chains-of-thought tend to have higher token-level entropy). The paper does not empirically verify this independence assumption nor discuss consequences when it is violated, which weakens the theoretical guarantee that CANON-based-on-entropy doesn't inadvertently affect length.

- **Models tested are ≤8B parameters despite "Large Reasoning Models" framing:** The paper title references "Large Reasoning Models" and cites Gemini 2.5 Pro, DeepSeek-R1, and OpenAI o1, but all experiments use Qwen2.5-Math-7B/1.5B and Llama3.1-8B. While these are reasonable proof-of-concept models, the scalability to models where RLVR is most impactful (32B+, 70B+) remains unexplored.

- **Training runs are short (150 steps) with some methods still improving:** Figure 2 shows CANON-Intra's performance on logic tasks is still growing at step 150. The reported performance figures may not reflect convergence, and it is unclear whether CANON's relative advantages persist at longer training horizons.

- **No statistical significance assessment despite small benchmark sizes:** AIME 24 and AIME 25 contain approximately 30 problems each. Reported gains of 1–5 points (e.g., AIME 24: 27.7 → 29.7 for CANON-Eff with α=0.96) translate to roughly 1–2 additional correct answers. Without confidence intervals or multi-seed evaluation, it is unclear whether these gains exceed noise.

- **Additional hyperparameters introduce tuning burden:** Beyond DR.GRPO's existing hyperparameters, CANON introduces µ (balancing inter/intra-group advantages) and α (weighting longer responses for efficiency). While the paper argues these have "rich physical meaning" and smooth effects (Appendix D.4–D.5), practitioners must still tune µ-min/µ-max values and scheduling strategies, adding complexity compared to the baseline.

## Nice-to-Haves

- **Split ratio ablation:** Theorem 1 justifies equal-sized grouping, but empirical verification with unequal splits (e.g., 60/40, 70/30) would strengthen the theoretical claim.

- **Compute overhead analysis:** The paper should report wall-clock training time and GPU hour comparisons between CANON and baselines, quantifying the cost of sorting/regrouping rollouts each iteration.

- **Mechanism analysis for intra-group advantage on complex tasks:** The claim that intra-group advantage aids complex logic through "exploration" is asserted but not substantiated. Gradient variance or policy entropy change rate analysis would clarify why intra-group comparison helps harder tasks.

- **KL divergence ablation:** The paper removes KL loss following prior work (DAPO, DR.GRPO), but an ablation showing CANON's behavior with KL regularization would demonstrate robustness.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Abstract imprecision about direction selection" (Harsh Critic):** The abstract states CANON "amplifies the impact of the target metric without presuming its direction." The critic claims CANON *does* select a direction data-adaptively. However, the distinction is correct: CANON does not *presuppose* a fixed direction (higher-is-better or lower-is-better) but discovers it from the data. This is precisely what "without presuming" means. The abstract is accurate.

- **"Binary rewards only in theoretical analysis" (Harsh Critic):** The paper's theoretical analysis assumes binary 0-1 rewards, which matches the experimental setup using Math-Verify. This is consistent, not a limitation.

- **"Compute-matched baselines concern" (Spark Finder):** The paper explicitly states "We sample 16 responses per prompt and use temperature=1.0 for rollout generation" for all methods. The baselines are compute-matched.

- **"Constant group size assumption validity" (Spark Finder):** CANON groups by sorting responses by metric values and splitting at the median. Group size is constant by design (half of samples per group). This is not a concern.

- **"Generalization to non-verifiable tasks" (Spark Finder):** The paper's stated scope is RLVR (Reinforcement Learning with Verifiable Rewards). Demanding evaluation on non-verifiable reward tasks (RLHF preferences) is scope creep.

- **"Writing quality and formatting nitpicks" (Harsh Critic, various):** The reviewer noted "Section 5.2 is confusing in its current structure." While the scheduling strategy presentation could be clearer, this is not a substantive flaw warranting rejection.

## Novel Insights

The reflection gain analysis (Figure 2f, Figure 6) provides a meaningful behavioral interpretation of why inter-group and intra-group advantages serve different functions. The paper shows that CANON-Intra achieves positive gains from rethinking patterns while CANON-Inter maintains higher training rewards—CANON-Dynamic with scheduled µ achieves both. This connects the advantage formulation to concrete reasoning behaviors (verification, sub-goal setting, backtracking) rather than treating performance gains as a black box. However, the rethinking patterns are identified via keyword matching (Appendix C.1), which is semantically fragile; future work should validate these findings with more robust pattern extraction.

## Suggestions

- **Establish a transparent hyperparameter selection protocol for CANON-Dynamic:** Define a clear rule (e.g., "select the strategy that maximizes validation accuracy on held-out data" or "use cosine scheduling with µ-min=0.4, µ-max=1.0 across all models") rather than model-specific best-picking. Report results for all tested strategies to enable fair comparison.

- **Empirically verify or discuss the independence assumption in Theorem 2:** Report the correlation between entropy and response length in your rollout samples. If correlation exists (as is likely), discuss how this affects selective amplification guarantees and whether additional normalization or conditioning would help.

- **Add statistical significance reporting for AIME benchmarks:** Report results across multiple random seeds (at least 3) with mean and standard deviation, or use bootstrap confidence intervals for accuracy estimates on small benchmarks.

---

## c7OsKOOZo8

- GT: Withdrawn (treated as Reject) (avg 4.5)
- Predicted: N/A (4.2/10)
- Match: N/A

### Final Review

## Summary

The paper proposes an end-to-end framework for multi-view diabetic retinopathy (DR) grading that eliminates dependency on costly external annotations (vessel maps, lesion segmentations). The core contribution is a Grade-Activated Lesion Proposal (GALP) module that self-generates lesion proposals via stage-wise auxiliary classifiers and CAM-derived evidence maps, combined with a Cross-View Lesion Expert-Guided Regional Fusion (LGRF) module that uses Mixture-of-Experts routing for selective cross-view feature integration. Experiments on MFIDDR (four-view) and DRTiD (two-view) demonstrate competitive or state-of-the-art performance without external supervision.

## Strengths

- **Practical annotation reduction:** The framework achieves strong performance (83.9% accuracy on MFIDDR, 76.0% on DRTiD) without requiring vessel/lesion annotations at inference time, directly addressing a real clinical bottleneck. The "w/o lesion" variant matches or exceeds several externally-informed methods (Table 1), demonstrating that self-derived proposals can effectively substitute for expert cues.

- **Cohesive dual-purpose GALP design:** The auxiliary classifier serves two complementary roles—enforcing grade-discriminative intermediate representations via focal loss (Eq. 1-2) and generating spatially sparse proposal tokens via Top-K selection on GEMs (Eq. 3-7). This efficiently combines representation learning with proposal generation without additional supervision.

- **Systematic hyperparameter analysis:** Figure 3 provides tuning curves for retention ratio α and MoE configuration (M, K₂), demonstrating that α=50% and K₂=2 experts provide the best trade-off. This supports the architectural choices with empirical evidence.

## Weaknesses

- **No spatial validation of lesion proposals:** The core premise is that GALP generates "lesion proposals" that act as surrogates for external annotations. However, the paper provides no quantitative evidence that the Top-K GEM regions correspond to actual anatomical lesions. MFIDDR includes lesion segmentation masks that could enable IoU or spatial recall metrics. Without this validation, the claim that proposals "target small, low-contrast lesions" (Abstract) remains unsubstantiated.

- **Poor Grade 4 (proliferative DR) performance:** Table 2 shows Grade 4 F1 scores of 36.0% (w/o lesion) and 51.6% (with lesion)—the worst across all grades and notably below CVSA (64.1% F1). Since Grade 4 represents the most severe, vision-threatening stage, this performance gap has direct clinical implications and is not discussed in the paper.

- **Missing critical ablations:** Table 4 removes GALP or LGRF entirely, but does not isolate component contributions. Key missing ablations: (a) GALP with random region selection vs. Top-K GEM selection to quantify the value of grade-conditioned proposals; (b) LGRF with standard cross-attention vs. MoE routing; (c) a pure backbone baseline (Swin-B with multi-view concatenation) to isolate the contribution of the proposed modules from backbone improvements.

- **Cyclic fusion design unjustified:** Section 3.3 restricts cross-view fusion to adjacent views (j = i+1 mod N). For the four-view MFIDDR dataset, views 1 and 3 never directly interact despite potentially containing complementary lesion patterns. No rationale or ablation comparing cyclic vs. all-pairs fusion is provided.

- **No statistical significance testing:** Reported improvements are small (e.g., 76.0% vs. 75.6% on DRTiD). With dataset sizes of 8,613 and 3,100 eyes and single train/test splits, confidence intervals or significance tests are essential to establish that differences are not due to run-to-run variance.

- **Missing reproducibility details:** Section 4.1 omits critical training hyperparameters: optimizer type, learning rate schedule, batch size, number of epochs, weight decay, and data augmentation pipeline. Algorithm 1 provides pseudocode but lacks specifics for reproducibility.

- **No computational cost analysis:** The framework adds four auxiliary classifiers, MoE routing with 6 experts, and Top-K weighted attention. No comparison of FLOPs, parameter count, or inference latency vs. baselines (MVCINN, CVSA, etc.) is provided, making it difficult to assess the efficiency trade-off.

## Nice-to-Haves

- **External dataset validation:** Testing on APTOS 2019 or EyePACS would strengthen generalization claims beyond MFIDDR and DRTiD.

- **Annotation scarcity experiments:** The paper claims reduced annotation dependency but only compares "no external" vs. "full external" conditions. Experiments with limited annotation budgets would better demonstrate practical utility.

- **Failure case analysis:** Understanding when self-generated proposals diverge from ground truth lesions would clarify the method's reliability boundaries for clinical deployment.

- **Expert routing interpretability:** Analyzing which experts specialize in which lesion types or grades would substantiate the claim that LGRF enables "contextual corroboration."

## Removed Points

*These points are flagged to be removed, treat them with caution:*

- "Title is cumbersome" — Pure style nitpick, irrelevant to technical merit.

- "Three-stage taxonomy is self-serving" — Nonsensical criticism; taxonomies are standard framing devices.

- "Interpretability not quantitatively demonstrated" — The paper shows qualitative GEM visualizations in Figure 1; quantitative interpretability evaluation is not the paper's claim.

- "No robustness experiments under distribution shift/missing views" — Scope creep; the paper's contribution is annotation reduction, not robustness.

- "Stage 4 features bypass GALP/LGRF" — This is intentional architectural design (deeper semantic features need different handling), not a flaw.

- "Concurrent with lesion variant undermines annotation-free claim" — The paper clearly presents both variants and appropriately discusses the "w/o lesion" variant as the main contribution; the "+lesion" variant is an optional enhancement.

- "Stage-wise auxiliary loss weight λ_aux=1.0 seems arbitrary" — This follows naturally from λ_aux having the same scale as the classification loss; sensitivity analysis would be nice-to-have, not a flaw.

- "CAM at 224×224 is too coarse for microaneurysms" — Valid concern about spatial resolution, but the paper acknowledges this limitation in the introduction; the question is whether the method still works despite coarse proposals.

## Novel Insights

The observation that self-generated grade-conditioned CAMs can achieve comparable performance to external lesion annotations represents a meaningful shift for medical imaging. The key insight is that for DR grading—where clinical labels are inherently tied to lesion presence—auxiliary supervision naturally drives intermediate features to encode lesion-relevant information without requiring explicit localization supervision. This suggests a broader principle: in strongly label-correlated medical tasks, internal supervision may substitute for expensive annotation-dependent pipelines. The ablation results (Table 4) showing synergistic benefits from both GALP and LGRF indicate that proposal quality and fusion strategy are jointly important, not independently optimizable.

## Suggestions

1. **Add spatial validation:** Compute IoU/recall between GALP Top-K regions and MFIDDR lesion masks. Even without perfect alignment, demonstrating meaningful overlap would validate the proposal mechanism.

2. **Add Grade 4 analysis:** Discuss the poor Grade 4 performance explicitly—whether it's due to class imbalance, proposal limitations, or architectural factors—and propose mitigation.

3. **Add pure backbone baseline:** Report Swin-B with simple multi-view concatenation to isolate the contribution of GALP and LGRF from backbone improvements.

4. **Add statistical testing:** Report confidence intervals across multiple runs or bootstrap significance for key comparisons.

5. **Provide complete training details:** Specify optimizer, LR schedule, epochs, batch size, and augmentation to ensure reproducibility.

6. **Justify cyclic fusion:** Either provide a rationale for the adjacent-view-only design or add an ablation comparing to all-pairs fusion.

---

## ey7CXUBn1g

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary

AdaSVD proposes two improvements to SVD-based LLM compression: (1) adaComp, an alternating update scheme using Moore-Penrose pseudoinverses to compensate for truncation errors, and (2) adaCR, a layer-wise adaptive compression ratio assignment based on input-output activation similarity. The method is evaluated on LLaMA-2-7B, OPT-6.7B, Mistral-7B, Vicuna-7B, and LLaVA-7B, demonstrating consistent perplexity reductions compared to SVD-LLM and other SVD baselines, with integration into GPTQ quantization also shown.

## Strengths

- **Technically sound compensation mechanism (adaComp):** Reformulating the post-truncation update as a Least Squares Estimation problem solved via Moore-Penrose pseudoinverse addresses the numerical instability of direct matrix inversion. Figure 3(a) provides empirical evidence that the proposed update (MPPU) converges smoothly while the naïve update (NU) diverges, validating the design choice.

- **Strong empirical improvements over SVD baselines:** Tables 1-2 show consistent perplexity reductions across multiple LLM families. At 60% compression, AdaSVD achieves WikiText-2 PPL of 50.33 vs. SVD-LLM's 89.90 on LLaMA2-7B—a 44% relative improvement. Results generalize across OPT, Mistral, Vicuna, and LLaVA architectures.

- **Clear ablation of both contributions:** Table 3 isolates the effects of adaComp and adaCR. The ablation shows adaCR provides additional gains on top of adaComp at most compression ratios, and the iteration analysis in Table 3(c) demonstrates the trade-off between more updates and overfitting to calibration data.

- **Orthogonality to quantization demonstrated:** Table 4 shows AdaSVD integrates with GPTQ-INT4, achieving better perplexity than SVD-LLM+GPTQ at all compression ratios. This demonstrates practical relevance for hybrid compression pipelines.

## Weaknesses

- **VLM evaluation is purely qualitative:** Figure 5 presents only visual captioning examples without quantitative metrics (CIDEr, BLEU, METEOR). The abstract's claim of "extensive experiments across multiple LLM/VLM families" is not substantiated for VLMs—standard captioning benchmarks should have been reported.

- **Conceptual tension in importance metric for adaCR:** The layer importance is defined as cosine similarity between input X and output WX (Eq. 17). A layer with high cosine similarity preserves input direction—intuitively, this could indicate a near-identity transformation that is *more* redundant and compressible. The paper equates high similarity with high importance (giving such layers higher retention ratios) but never addresses this conceptual tension. While empirical results (Figure 4) show the first layer has highest importance—which aligns with practical intuition—the direction of the metric's relationship to importance deserves justification.

- **No computational cost or inference latency analysis:** The paper lacks compression time, peak memory during compression, and inference throughput measurements. adaComp requires iterative SVD computations per layer, potentially making it significantly slower than one-shot methods like SVD-LLM. Without timing data, the practical efficiency claim cannot be verified.

- **Ablation shows adaComp alone can hurt performance at moderate compression:** Table 3(a) reveals that at 50% compression, adaComp without adaCR yields WikiText-2 PPL of 30.00—worse than SVD-LLM's 27.19. This inconsistency (adaComp helps at 40% and 60% but hurts at 50%) is noted only obliquely ("more iterations may lead to overfitting") without adequate explanation. The mechanism by which adaComp sometimes degrades performance deserves analysis.

- **Perplexity values remain high at ≥50% compression despite improvements:** At 50% compression on LLaMA2-7B, PTB perplexity is 593.14 (vs. original 8.35)—a severe degradation even with AdaSVD's improvements over baselines. The paper frames results as "narrowing the performance gap" but should acknowledge practical deployability limits at high compression ratios.

## Nice-to-Haves

- **Experiments on larger models (≥13B):** While the method should theoretically scale, demonstrating effectiveness on LLaMA-2-13B or 70B would strengthen claims about practical deployment scenarios mentioned in the introduction.

- **Convergence stopping criterion:** The iteration number is fixed (τ ∈ {1, 3, 15}) with Table 3(c) showing overfitting risk at more iterations. An adaptive stopping rule based on loss plateau or validation perplexity would improve robustness.

- **Calibration data sensitivity analysis:** All experiments use 256 WikiText-2 samples. Understanding how results vary with calibration set size and source domain would strengthen robustness claims.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"No comparison to non-SVD compression methods":** The paper's stated scope is improving SVD-based compression. It demonstrates orthogonality to quantization in Table 4. Demanding comparison to pruning or distillation at equivalent model sizes is scope creep—evaluate whether AdaSVD improves SVD compression, not whether SVD is the best paradigm overall.

- **"Theoretical convergence guarantee absent":** Most empirical compression papers at ICLR do not provide convergence proofs. The empirical evidence in Figure 3(c) and ablation studies is acceptable. A convergence theorem would be nice but is not a standard expectation.

- **"Abstract claim of 'significantly reduced memory requirements' is vague—no MB/GB numbers":** Compression ratio directly maps to parameter reduction. The improvement is in reducing perplexity at given compression ratios, not claiming novel memory efficiency. This criticism misreads the contribution.

- **"All experiments on 7B models—scalability untested":** The method operates on weight matrices and should theoretically scale. This is a valid concern but belongs in Nice-to-Haves rather than Weaknesses for an ICLR paper.

- **"Stack-of-batch strategy lacks theoretical justification":** Figure 3(b) empirically demonstrates benefit. The strategy is a practical engineering contribution for memory-constrained calibration. Deep theoretical justification is not required for empirical contributions.

- **"Cosine similarity metric not compared to Hessian/Fisher alternatives":** The paper proposes a simple metric that works empirically. Comparing to all alternative importance metrics is a substantial expansion of scope. The conceptual tension (preserving direction vs. importance) is the real issue, not lack of metric comparison.

## Novel Insights

The observation that alternating Moore-Penrose pseudoinverse updates can effectively compensate for SVD truncation errors without training is methodologically interesting. The stack-of-batch calibration strategy addresses a real practical constraint (GPU memory limits on calibration data) that many compression papers gloss over. The finding that layer-wise importance varies substantially—with the first layer consistently most important (Figure 4)—is practically useful for future layer-adaptive compression methods. However, beyond these contributions, no fundamentally new insight emerges; the work is a careful engineering of known techniques (alternating least squares, pseudoinverse solutions, layer importance) into an effective pipeline.

## Suggestions

- **Quantify compression overhead:** Report wall-clock time and peak GPU memory for running AdaSVD compression on LLaMA2-7B at 40%/50%/60% ratios. Compare to SVD-LLM's one-shot compression time.

- **Add quantitative VLM metrics:** Run standard captioning benchmarks (CIDEr, BLEU) on LLaVA-7B to substantiate the VLM claim. Three qualitative examples are insufficient.

- **Explain adaComp's inconsistent behavior:** Investigate and explain why adaComp alone degrades performance at 50% compression. Is there an interaction with the specific singular value distribution at that retention ratio? A brief analysis would strengthen the paper.

- **Add inference latency measurements:** Report tokens/second for the compressed models on a standard GPU to demonstrate practical efficiency gains beyond perplexity improvements.

---

## iIEEgI6WsF

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (6.3/10)
- Match: N/A

### Final Review

## Summary

This paper proposes On-Demand Communication (ODC), a distributed training scheme that adapts the parameter server paradigm to modern Fully Sharded Data Parallel (FSDP) systems. ODC replaces FSDP's per-layer collective operations (all-gather, reduce-scatter) with point-to-point RDMA primitives, relaxing synchronization from the layer level to the minibatch level. This decouples device execution and mitigates straggler effects caused by variable sequence lengths in LLM post-training. Empirical evaluation on SFT and RL tasks across 1.5B–32B models demonstrates throughput improvements of up to 36% over standard FSDP.

## Strengths

- **Insightful problem framing and architectural contribution:** The paper correctly identifies that collective communication's efficiency fundamentally relies on balanced workloads—an assumption violated in LLM post-training due to variable sequence lengths. The reframing of FSDP as a "decentralized parameter server" (Section 3.1, Figure 6) is conceptually clean and provides a principled basis for the design.

- **Substantial empirical validation across diverse settings:** The evaluation spans two SFT datasets (LongAlign, SWE-Smith), one RL dataset (AIME), and models from 1.5B to 32B parameters. The bubble rate analysis (Tables 4, 6) demonstrates strong correlation between predicted idle time reduction and actual throughput gains, providing compelling mechanistic evidence.

- **Transparent acknowledgment of limitations with proposed mitigations:** Section 6.1 honestly characterizes ODC's inter-node bandwidth disadvantage (Figure 11) and proposes hybrid sharding as mitigation, with preliminary validation in Appendix E. This demonstrates mature systems engineering practice.

- **Practical implementation with reproducible artifacts:** The use of CUDA-IPC and NVSHMEM for non-intrusive communication is well-documented, and the open-sourced code (https://github.com/sail-sg/odc) lowers adoption barriers. The integration into FSDP requires minimal code changes.

## Weaknesses

- **Convergence verification is insufficiently rigorous:** Correctness is validated only on a 1.5B model trained from scratch on 8K samples (Appendix F, Figure 14). Given that ODC changes the timing of gradient accumulation through asynchronous scatter-accumulate operations, and floating-point addition is non-associative, the potential for subtle numerical drift cannot be ruled out by this limited test. Convergence verification on 7B+ models with longer training runs, or gradient norm comparisons between ODC and Collective baselines, would strengthen confidence in correctness.

- **Inter-node scalability is not validated at production scales:** Experiments are limited to 32 GPUs (4 nodes). The paper acknowledges (Section 6.1, Figure 11) that ODC's inter-node bandwidth is "significantly slower" than NCCL collectives because point-to-point RDMA forgoes hierarchical optimizations. The proposed hybrid sharding mitigation (Appendix E) is only evaluated on truncated 8K sequences, not on the actual LongAlign/SWE-Smith datasets where ODC's value is highest. Production post-training often runs on 64–256+ GPUs; the absence of validation beyond 4 nodes is a gap.

- **Regression cases at certain minibatch sizes are unexplained:** Tables 3 and 5 show ODC can be slower than collective at Minibs=1 (14B: -2%, 32B: -2%) and Minibs=2 for RL (14B AIME: -5%). The paper mentions that "methods perform similarly when the minibatch size is one" but does not address the Minibs=2 regression or explain why ODC overhead dominates at small batch sizes. Understanding the break-even point where ODC becomes beneficial would clarify the method's applicability domain.

- **RL evaluation is constrained by framework limitations:** The authors note (Section 5.2) that verl requires identical samples per device, preventing LB-Mini from taking full effect. Since RL is a primary target use case, evaluating ODC with an unconstrained implementation would better validate its claimed advantages. The reported 5–10% RL gains are modest compared to SFT, but separating framework limitations from ODC's inherent limitations is not fully achieved.

- **No comparison against concurrent workload balancing approaches:** The paper cites WLB-LLM (Wang et al., 2025) as related work addressing sequence length imbalance through 4D parallelism, but does not provide an empirical comparison. If similar throughput gains can be achieved through alternative parallelism strategies without requiring RDMA-based point-to-point primitives, this would weaken the case for ODC's adoption.

## Nice-to-Haves

- **Evaluate hybrid sharding on actual long-sequence datasets:** The hybrid sharding experiments in Appendix E use truncated sequences (max 8K) rather than the original LongAlign/SWE-Smith data. Testing hybrid sharding on native long-context workloads would clarify whether it effectively mitigates inter-node overhead at the scales where ODC provides maximum benefit.

- **Provide theoretical runtime bound for ODC:** Equation (1) formalizes FSDP's runtime as a function of per-layer maxima, but no analogous expression is derived for ODC. Even a simplified analytical model relating expected speedup to workload variance and device count would strengthen the theoretical contribution.

- **Add convergence verification on 7B+ models:** Training curves or gradient norm comparisons for larger models would provide stronger evidence that the relaxed synchronization semantics preserve training dynamics at production scales.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Up to 36% is misleading":** The paper clearly presents this as a maximum in the abstract and provides full tables showing varying speedups. This is standard practice in systems papers; the phrase "up to" properly signals the best-case nature.

- **"Per-device computations being 'independent' is overstated":** The paper correctly states that forward/backward computations are independent and that gradient synchronization is what requires coordination. ODC moves the synchronization point but preserves synchronous semantics. The statement is technically accurate.

- **"Confounded contributions (ODC + LB-Mini vs Collective + LB-Micro)":** The paper does include ODC LB-Micro in all tables, which isolates the communication improvement from the load balancing algorithm. For example, Table 5 shows 14B LongAlign Minibs=4: Collective LB-Micro = 45.1, ODC LB-Micro = 49.9 (+11%), ODC LB-Mini = 61.4 (+36%). The ablation is present, though the paper could more explicitly highlight it in text.

- **"No confidence intervals or variance estimates":** Single-run throughput benchmarks are standard practice in distributed systems papers; GPU performance on fixed workloads is highly deterministic after warmup. The bubble rate analysis provides a theoretical predictor that correlates well with measured speedups, offering independent validation.

## Novel Insights

Beyond the paper's contributions, the bubble rate analysis (Tables 4, 6) provides a novel diagnostic tool: by estimating idle fraction before execution, practitioners can predict when ODC will provide benefit versus when its overhead will dominate. The correlation between predicted bubble rate and actual speedup across all configurations (e.g., bubble rate dropping from ~67% to near 0% at higher minibatch sizes) validates the mechanistic hypothesis that ODC's gains stem directly from eliminating per-layer synchronization barriers rather than from opaque optimization.

## Suggestions

- Provide convergence loss curves for at least one 7B model configuration over meaningful training duration, and report gradient norm distributions or weight checkpoint differences between ODC and Collective to rule out numerical drift.

- Explicitly state the break-even conditions: characterize the minimum minibatch size, sequence length variance threshold, or device count where ODC transitions from regression to improvement.

- Add a single comparison against WLB-LLM or another concurrent workload balancing approach on the same hardware to contextualize ODC within the broader solution landscape.

---

## D5PJX02Jki

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary

This paper identifies that standard Rotary Position Embeddings (RoPE) discard the imaginary component of the complex-valued dot product when computing attention scores. The authors propose RoPE++, which recovers this imaginary component as a parallel set of attention heads, introducing two variants: RoPE++EC (equal cache, double heads) for improved performance and RoPE++EH (equal heads, half cache) for memory efficiency. The work includes theoretical analysis of characteristic decay curves and empirical validation across 376M, 776M, and 1.5B parameter models.

## Strengths

- **Mathematically grounded insight**: The core observation—that RoPE's real-vector rotation formulation discards the imaginary component—is genuine and non-trivial. The derivation in Equations 2–4 showing that imaginary attention is equivalent to applying a −π/2 rotation to query vectors before standard RoPE computation is elegant, and the characteristic curve analysis (Equation 5, sine integral vs. cosine integral) provides principled justification for why imaginary attention preferentially captures long-range dependencies.

- **Practical architectural design**: The dual-variant approach directly addresses real engineering constraints. RoPE++EH halves KV cache while maintaining head count (Table 4, Figure 4), offering tangible inference-time benefits for long-context deployment. The compatibility with FlashAttention (Section 3.3) and existing interpolation methods (Table 3: YaRN, Linear PI) indicates thoughtful integration with current infrastructure.

- **Empirical validation across scales**: The paper trains models at 376M, 776M, and 1.5B scales with 50B+ tokens, demonstrating that RoPE++ generally improves over vanilla RoPE on both short-context tasks (Table 1) and long-context benchmarks (Table 2). The convergence analysis (Tables 7–9) shows training stability comparable to RoPE.

- **Mechanistic investigation**: The noise perturbation experiment (Figure 5, Section 5.2) provides evidence that imaginary attention heads are more critical for long-context retrieval than real heads. Adding Gaussian noise to imaginary components degrades RULER scores more severely than equivalent noise to real components (5–8 point gaps at σ=1.0), supporting the theoretical claim about their functional specialization.

## Weaknesses

- **Inconsistent results at 1.5B scale**: Table 6 shows troubling cross-benchmark inconsistency: RoPE++EC achieves 37.5 vs. RoPE's 35.1 on RULER (improvement) but only 22.9 vs. 29.5 on BABILong (substantial regression). RoPE++EH shows the opposite pattern. RULER tests retrieval while BABILong tests reasoning; this divergence suggests the method's benefits may be task-dependent in ways not analyzed. The abstract's claim of "consistent improvements" is overstated given these 1.5B results.

- **Missing baseline comparisons on long-context benchmarks**: Tables 1 includes FoPE, Pythia (partial RoPE), and ALiBi for short-context tasks, but these baselines are absent from Table 2 (RULER, BABILong). FoPE in particular is designed for long-context extrapolation; excluding it from the long-context evaluation leaves a significant gap when claiming "RoPE++ achieves the best performance on average" in Section 4.3.

- **RoPE++EC training throughput cost**: Table 11 shows RoPE++EC reduces training throughput by 30–40% at 32k context (22,631 vs. 29,019 TGS for 776M). The paper mentions this is acceptable because "long-context inference is primarily IO-bounded," but this justification applies to inference, not training. The substantial training overhead should be prominently discussed as a cost of the RoPE++EC variant.

- **RoPE++EH incompatibility with some interpolation methods**: Table 3 shows RoPE++EH with YaRN achieves 24.7 vs. vanilla RoPE's 28.2 on RULER at 376M—a regression. Similarly, RoPE++EH with PI achieves 19.6 vs. RoPE's 25.1. The paper states RoPE++ "consistently achieves the highest scores," but this is driven entirely by RoPE++EC; RoPE++EH underperforms vanilla RoPE when combined with these interpolation techniques, which is an important practical limitation underemphasized in the main text.

- **No parameter-matched ablation for RoPE++EC**: RoPE++EC doubles the output projection (W_o) dimensions (Section 3.3), introducing additional parameters. The paper does not include a control experiment comparing RoPE++EC against a vanilla RoPE baseline with similarly increased W_o or head dimensions. Without this, gains cannot be cleanly attributed to the imaginary rotation mechanism versus simply having more representational capacity in the output projection.

## Nice-to-Haves

- **Validation at 7B+ scale**: Position embedding behaviors can shift non-linearly at larger scales. Validation on a standard open-weight model (e.g., Llama 3 8B) via continued pre-training would strengthen claims about scalability.

- **Instruction-tuning evaluation**: Long-context utility is primarily realized after instruction tuning. Evaluation on IFT checkpoints would verify the method transfers to usable models.

- **Explicit comparison with LongRoPE/Dynamic-NTK**: While YaRN and PI are tested, comparison with more recent position embedding extensions would situate RoPE++ within current best practices.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Statistical significance testing absent"**: Standard practice concern. ICLR typically does not require multiple runs with error bars for pre-training experiments at this scale. The convergence tables (7–9) showing consistent trends across checkpoints provide robustness.

- **"Collapse back to standard RoPE claim is confusing"**: The paper's reasoning in Section 3.3 is actually correct. If imaginary attention simply applies −π/2 rotation to the query, and you have separate parameters for it, the model could learn representations that effectively undo this rotation, yielding equivalent computations to standard RoPE. The constraint ensures the imaginary component provides genuinely complementary signal.

- **"i.i.d. assumption for characteristic curve is restrictive"**: This expectation-level approximation is the same one used in the original RoPE analysis (Su et al., 2024). While imperfect, it provides useful theoretical intuition, and the empirical attention pattern analysis validates the qualitative prediction.

- **"Absence of gradient conflict analysis"**: Scope creep beyond the paper's stated contributions. The empirical results demonstrate training stability without requiring additional mechanistic justification.

## Novel Insights

Beyond the paper's own contributions, the characteristic curve analysis reveals that imaginary attention's sine-integral decay (Figure 1, Equation 5) provides a qualitatively different inductive bias than real attention's cosine-integral. Real attention exhibits strong locality (rapid decay), while imaginary attention maintains elevated attention weights at longer distances. This complementary decay structure suggests a principled basis for designing "dual-stream" attention mechanisms where one stream specializes in local semantics and another in global retrieval—a pattern that echoes biological visual systems with separate foveal and peripheral processing.

## Suggestions

1. **Add control ablation for RoPE++EC**: Compare against a baseline with doubled W_o dimensions (without imaginary rotation) to isolate the contribution of the imaginary component from parameter count effects.

2. **Prominently acknowledge 1.5B inconsistencies**: Discuss why RULER and BABILong performance diverges at 1.5B in the main text rather than Appendix D.2. Analyze whether imaginary heads differentially benefit retrieval versus reasoning tasks.

3. **Include FoPE in long-context benchmark tables**: At minimum, report FoPE results on RULER and BABILong to provide a more complete comparison with competing long-context position embeddings.

4. **Clarify training costs in main text**: Move the throughput reduction for RoPE++EC from Table 11 to the main experimental section and discuss the training-time vs. inference-time tradeoff explicitly.

---

## Mz98kwANpF

- GT: Reject (avg 4.5)
- Predicted: N/A (4.9/10)
- Match: N/A

### Final Review

## Summary

This paper challenges the prevailing multi-task LoRA paradigm by demonstrating that architectural isolation (via multi-component designs) may be unnecessary. Through empirical analysis, the authors show that (1) M-LoRA—a simplified multi-head variant with high inter-head similarity—outperforms complex routed architectures, and (2) simply increasing the rank of standard LoRA matches multi-component performance. Motivated by these findings, they propose Align-LoRA, which adds a KL-divergence or MMD alignment loss to encourage task-shared representations, achieving superior multi-task generalization with zero inference overhead.

## Strengths

- **Counter-intuitive empirical finding:** The observation that M-LoRA achieves superior performance despite exhibiting high inter-head similarity (>0.85) directly challenges the prevailing assumption that component diversity is essential for multi-task LoRA. Table 1 shows consistent improvements over HydraLoRA and R-LoRA across five tasks, with the strongest baseline (M-LoRA at 75.45) outperforming R-LoRA (74.67) by a meaningful margin.

- **Practical deployment advantage:** Unlike routed multi-component methods that cannot be merged into the backbone, Align-LoRA retains full mergeability, incurring zero inference latency. Appendix D confirms the method achieves lower FLOPs and faster training time than multi-head variants, making it highly attractive for real-world deployment.

- **Comprehensive model and benchmark coverage:** The evaluation spans Qwen2.5 (3B, 7B, 14B), LLaMA2 (7B, 13B), and LLaMA3-8B, with both out-of-distribution generalization (BBH) and in-domain adaptation benchmarks (8-task). Tables 4 and 5 show consistent improvements across all configurations, with Align-LoRA-K achieving the highest average scores.

- **Mechanistic hypothesis with empirical support:** The paper advances a clear hypothesis—that learning task-shared representations outperforms architectural isolation—and validates it through representation visualizations (Figure 5 in Appendix I.1), which show Align-LoRA indeed brings task representations closer together.

## Weaknesses

- **Theoretical derivation contains an error:** Appendix F claims that the identity Σ_i Δ(D_i, D̄) = (1/2M) Σ_i,j Δ(D_i, D_j) holds for KL divergence by invoking "convexity of KL divergence and linearity of the global centroid." This is **not generally true**. KL divergence is asymmetric and does not satisfy the triangle inequality; the decomposition holds for squared Euclidean distance and MMD, but not for KL without additional assumptions about the distribution structure. This invalidates the formal claim that "Align-LoRA's bound is tighter," though the empirical results remain valid. The theory section should be either corrected or substantially revised to reflect the actual mathematical claims that can be supported.

- **Inconsistent performance of A-LoRA-M undermines generality claim:** In Table 5 (Qwen2.5-3B), A-LoRA-M achieves 78.35 while M-LoRA achieves 78.51; similarly on 7B, A-LoRA-M (82.31) trails M-LoRA (82.46). The paper frames both alignment variants as validation that "explicit representation alignment is effective," but A-LoRA-M's underperformance relative to the *ablation baseline* contradicts this narrative. The paper should explain why MMD underperforms KL-divergence in this setting or qualify the generality claim.

- **Incomplete ablation of the dropout mechanism in M-LoRA:** Section 3.3 argues that multi-head dropout combined with router removal transforms heads into "collaborators" and enables task-general learning. However, Table 1 lacks a critical cell: "M-LoRA without dropout" (i.e., no router, no dropout). Without this, the mechanism attributed to dropout cannot be definitively isolated from other architectural differences.

- **No statistical significance testing reported:** Performance margins are consistently small (often 0.5-2 points), yet no error bars, standard deviations across seeds, or statistical significance tests are provided. Given the claimed "paradigm shift," this empirical rigor gap is notable.

- **Rank mismatch in key comparisons:** In Table 4, A-LoRA-K uses rank=8 (0.20% params) while M-LoRA uses rank=4 (0.22% params). While A-LoRA achieves this with *fewer* parameters—which strengthens the efficiency claim—the rank difference introduces a confound: we cannot determine whether the alignment mechanism works equivalently well at the same rank, or whether the benefit partially comes from a larger representation capacity.

## Nice-to-Haves

- **Iso-rank ablation for Align-LoRA:** Adding Align-LoRA at rank=4 (matching M-LoRA's rank) would cleanly isolate the alignment contribution from capacity effects.

- **Hardware latency benchmarks:** While theoretical mergeability is claimed, actual throughput measurements (ms/token) comparing merged Align-LoRA against routed methods would substantiate the efficiency contribution.

- **Task conflict matrix:** Reporting per-task performance deltas would reveal whether alignment causes negative transfer on any specific tasks, particularly for highly dissimilar task pairs.

- **Scalability to many tasks:** The O(M²) pairwise KL computation becomes relevant beyond 10 tasks; evaluating at M=20+ tasks would clarify the scalability limits.

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **"Abstract claims 'significant' without statistical tests":** While statistical testing is lacking, the margins are consistent across tables and models. This concern is already captured above and doesn't need separate emphasis in the abstract analysis.

- **"Introduction's four contributions are redundant":** This is a stylistic preference. Points 2 and 3 are distinct (one is empirical observation, the other is the proposed hypothesis), and inflation is a minor issue.

- **"Gaussian assumption for batch distributions is unaddressed":** The paper addresses this by also testing MMD, which requires no distributional assumptions. While A-LoRA-M underperforms, the existence of two variants partially mitigates this concern.

- **"Batch composition sensitivity not studied":** This is a practical concern but represents scope creep for a methods paper. The batch-level estimation is standard practice in similar alignment methods.

- **"Theoretical section too compressed in main paper":** This is a clarity observation, not a substantive weakness. The full derivation is available in Appendix F (though with errors noted above).

## Novel Insights

The paper's central insight—that multi-component LoRA architectures may be solving the wrong problem—is genuinely novel and empirically grounded. The "paradox of diversity" finding (Figure 2, Table 1) is striking: methods explicitly designed to increase head diversity (R-LoRA) underperform against a variant that naturally converges to high similarity. This suggests the field's focus on task-specific specialization through architectural isolation may have been misguided, and that the capacity allocated to routing mechanisms could be more effectively deployed toward simpler representation alignment. The Align-LoRA method itself is simple and practical, avoiding the inference latency that plagues routed methods.

## Suggestions

- Correct the theoretical derivation in Appendix F: either remove the KL-specific claims and focus on MMD (for which the decomposition is valid), or provide the correct mathematical treatment for KL divergence under appropriate distributional assumptions.

- Add a "M-LoRA without dropout" ablation cell to definitively attribute the performance mechanism to dropout vs. other architectural changes.

- Explain why A-LoRA-M underperforms relative to M-LoRA in some settings, or revise the claim that "both alignment methods validate the approach" to reflect the inconsistency.

- Report standard deviations across at least 3 random seeds for the main results (Tables 1, 4, 5) to substantiate the significance of the observed margins.

- Include an iso-rank comparison (Align-LoRA at rank=4) to isolate the alignment benefit from the rank advantage.

---

## d2pUyiXwcm

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (6.3/10)
- Match: N/A

### Final Review

## Summary
The paper introduces SCaSML, a framework that improves pre-trained scientific ML surrogates for high-dimensional PDEs at inference time without retraining. By deriving a "defect PDE" that describes the error of any approximate solution, the method applies Multilevel Picard Monte Carlo simulation to compute targeted corrections. The authors prove convergence guarantees showing multiplicative error reduction and demonstrate 20–80% error improvements across PDE benchmarks up to 160 dimensions.

## Strengths
- **Mathematically sound core insight:** The key observation that the defect PDE preserves the semi-linear structure of the original PDE, enabling efficient Monte Carlo solvers like MLP, is correctly derived and non-trivial. Lemma D.11 proves that the modified nonlinearity inherits the same Lipschitz constant as the original, which is crucial for complexity bounds.
- **Comprehensive experimental validation:** The paper tests on four PDE types using two surrogate architectures (PINN and GP), with statistical significance tests (p ≪ 0.001) and consistent error reductions of 20–80%.
- **Improved convergence rate theoretically and empirically verified:** Theorem 2.5 and Corollary 2.6 establish a provably faster rate, and Figure 4 provides empirical validation with clear log-log slope differences between SCaSML and base surrogates.
- **Fixed-computational-budget comparisons provided:** Appendices G.7 and G.8 compare SCaSML against larger PINNs under equal total compute, showing consistent efficiency gains.

## Weaknesses
- **Theory-experiment gap for general PDEs:** The main theoretical results are proven only for µ=0 and σ=sI_d (isotropic heat equation, Appendix D.2.2), while experiments cover more general cases (LQG, DR with full Laplacians). This creates a disconnect between the scope of proofs and empirical claims.
- **Strong assumption on defect regularity:** Assumption 2.4 requires that the W^{1,∞} norm of the defect is bounded by the same scalar e(û) that bounds the L^∞ residual. This presupposes stability of the defect PDE solution in terms of its gradient, which requires maximum-principle-type arguments not provided. For PINNs with spectral bias, gradient errors can exceed function errors.
- **Unequal hyperparameters in baseline comparisons:** The paper uses different clipping thresholds for naive MLP (threshold 10) versus SCaSML (threshold 0.1 for LQG, 0.01 for DR). Since clipping affects Monte Carlo variance and stability, the comparison may be partially biased.
- **Limited empirical scaling validation:** All tabulated results use M=10 samples and N=2 levels. The scaling law (Figure 4) is based on only ~4–5 data points per curve, making slope estimates fragile without confidence intervals.
- **LCD benchmark lacks meaningful difficulty:** The linear convection-diffusion problem has exact solution u(r,y) = Σy_i + r, a simple linear function. Results on this benchmark may not generalize to genuinely challenging high-dimensional PDEs.

## Nice-to-Haves
- Stress tests with under-trained surrogates to identify failure regimes where the defect PDE becomes harder than the original.
- Wall-clock time breakdowns separating gradient/Hessian evaluation costs from Monte Carlo simulation costs.
- Analysis of break-even surrogate accuracy: how poor can û be before SCaSML becomes computationally wasteful?

## Removed Points
- Criticism about "inference-time scaling" terminology: The paper correctly positions this as variance reduction via structured Monte Carlo; the core contribution stands regardless of terminology.
- Criticism about missing baselines (Deep BSDE): Without external verification of appropriate baselines, this criticism cannot be validated.
- Criticism about Pareto front missing: The paper addresses this in Appendices G.7 and G.8.
- Minor notation inconsistencies between main text and appendix are editorial, not substantive.

## Novel Insights
The most insightful observation, under-emphasized in the paper, is Lemma D.11's proof that F̃ inherits the exact same Lipschitz constant as the original F. The cancellation of surrogate-dependent terms (F(û, σ^T∇û) subtracts out in F̃) makes the defect PDE "no harder" than the original in terms of regularity requirements, enabling MLP complexity reduction. This theoretical nuance is the hidden foundation of the method's efficiency.

## Suggestions
1. Clearly state in the main text that Theorems 2.5/E.6/F.5 apply to the isotropic heat equation, and discuss how empirical results on more general PDEs suggest broader applicability that remains to be theoretically established.
2. Add a discussion of when Assumption 2.4 may fail, particularly for surrogates with large gradient errors.
3. Report naive MLP results with matched clipping thresholds to isolate the benefit of the defect PDE formulation.
4. Provide confidence intervals on scaling law slopes in Figure 4 to convey empirical uncertainty.
5. Add a "regime analysis" figure showing error reduction vs. initial surrogate error to help practitioners understand when SCaSML provides substantial vs. marginal benefits.

---

## wSbVv6xaRr

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (4.3/10)
- Match: N/A

### Final Review

## Summary
FedMPDD introduces a federated learning algorithm that compresses client gradients via multi-projected directional derivatives along m random Rademacher vectors, reducing uplink communication from O(d) to O(m) bits while providing inherent privacy against gradient inversion attacks through rank deficiency of the projection operator. The method achieves O(1/√K) convergence matching FedSGD and provides explicit reconstruction error lower bounds, with empirical validation across image classification benchmarks.

## Strengths
- **Unified theoretical framework:** The paper rigorously proves that the multi-projected estimator is unbiased, derives convergence guarantees via the Johnson-Lindenstrauss lemma, and provides explicit bounds on gradient reconstruction error (Lemma 1) and data reconstruction error (Lemma 2). The decomposition of variance into client sampling, gradient stochasticity, and projection-induced components is clean and instructive.
- **Tunable privacy-communication-accuracy trade-off:** The parameter m provides a principled knob for balancing competing objectives. Tables 1-2 demonstrate that FedMPDD achieves target accuracy with substantially fewer transmitted bytes while maintaining low SSIM (≈0.14-0.28) under gradient inversion attacks, outperforming QSGD, Top-k, and LDP variants under strict byte budgets.
- **Direct comparison to FedSGD rates:** The paper correctly shows that FedMPDD matches FedSGD's O(1/√K) convergence rate for non-convex smooth objectives, with the distortion parameter ε from the JL lemma appearing in the bound's constant rather than the rate itself.

## Weaknesses
- **Convergence rate discrepancy in abstract:** The abstract claims "converges at a rate of O(1/K)" but Theorem 2 establishes O(1/√K) (the bound's dominant term scales as K^{-0.5}). While this matches standard non-convex FedSGD rates and is a meaningful result, the abstract overstates the guarantee—this should be corrected for accuracy.
- **Privacy is empirical/geometric, not formal:** The paper's "inherent privacy" is based on rank-deficiency and reconstruction error bounds, not a formal privacy definition such as (ε,δ)-differential privacy. This provides no composability guarantees across rounds and no protection against computationally unbounded adversaries with sufficient observations. The comparison to LDP is meaningful but should acknowledge that LDP provides formally composable guarantees that FedMPDD does not.
- **Privacy collapses for long training:** Remark 2 and Appendix D establish that privacy is preserved only while T × m < d. For m=600 and typical model dimensions (d≈60K–300K), this limits training to ~100–500 rounds before gradient recovery becomes theoretically possible. The paper acknowledges this as a "conservative bound" but the practical implications for realistic training durations are understated.
- **LDP baselines appear poorly calibrated:** In Tables 1-2, FedSGD+Laplace(var=1) achieves ≈11% accuracy on CIFAR-10 (near random), while FedSGD+Laplace(var=0.1) fails to protect privacy (SSIM≈0.8-0.96). A properly calibrated LDP baseline achieving reasonable accuracy and moderate privacy would strengthen the comparison.
- **No statistical significance reporting:** Results report single values without standard deviations or confidence intervals across the five seeds mentioned in Appendix H.2, making it difficult to assess reliability.
- **JVP computational efficiency claimed but not implemented:** Remark 1 discusses Jacobian-vector products to reduce client-side computation, but this is marked as "future work" (Appendix F). The current implementation computes full gradients before projection, adding O(dm) overhead. The efficiency claims should be empirically validated or stated as aspirational.
- **Experiments limited to small models:** All models have d<300K parameters. The claim that communication savings "become even more substantial for large-scale models" remains unverified. A single experiment on a larger architecture (e.g., ResNet-18 or ViT-Small) would strengthen practical relevance.

## Nice-to-Haves
- **FedAvg/local-SGD comparison:** Nearly all practical FL systems use multiple local SGD steps per round. While orthogonal to compression, showing how FedMPDD integrates with FedAvg would improve real-world applicability.
- **Secure Aggregation baseline:** SecAgg is a standard FL privacy mechanism; comparing against it would contextualize FedMPDD's privacy-efficiency trade-off.
- **Larger model validation:** ResNet-18 or similar would validate the O(log d) scaling of m and demonstrate practicality for modern model sizes.
- **Non-vision domain:** FL is increasingly applied to language modeling and recommendation; evaluation on at least one non-CV task would demonstrate generality.

## Removed Points
*These points are flagged to be removed, treat them with caution*

- **"ResNet-18 experiments missing" (Harsh Critic)** — The paper uses ResNet-18 only as motivation for communication costs (42MB per round), not as a claim. Testing on larger models is a nice-to-have, not a weakness.

- **"No FedAvg baseline" (Harsh Critic, Review 3)** — The paper specifically addresses gradient compression for FedSGD variants. Local update methods (FedAvg) are orthogonal to compression; requiring them is scope creep.

- **"Adaptive attacks not considered" (Review 3)** — The paper tests against published GIA methods (Yu et al., 2025; Zhu et al., 2019). Expecting adaptive adversarial analysis is beyond standard requirements for an initial method contribution.

- **"Non-IID heterogeneity insufficient" (Review 3)** — The paper tests both IID and non-IID (2-class partition per client) settings. More realistic heterogeneity is a nice-to-have, not a missing requirement.

- **"SSIM is not formal privacy" (Harsh Critic)** — This conflates the metric with the guarantee. SSIM is appropriately used to evaluate reconstruction quality under empirical attacks, consistent with the privacy framing (reconstruction error bounds, not DP). The lack of formal DP guarantees is a valid separate concern, kept above.

## Novel Insights
The paper's key insight is that the rank deficiency inherent in random projections provides *dimension-proportional* obfuscation: the relative reconstruction error (d-1)/m depends on model size, meaning larger models automatically get stronger protection at fixed m. This differs fundamentally from additive noise methods where privacy degrades for large gradients. The multi-projection averaging mechanism to overcome dimension-dependent variance (which would otherwise yield O(√d/K) convergence for single projections) is a clever application of JL-type concentration to FL.

## Suggestions
- Correct the abstract to state O(1/√K) convergence explicitly; clarify that this matches standard non-convex FedSGD rates.
- Add mean ± standard deviation or confidence intervals to Tables 1-2 across the five reported seeds.
- Re-run LDP baselines with calibrated noise levels that achieve comparable accuracy-privacy trade-offs, or acknowledge that FedMPDD's advantage is specific to strict byte-budget scenarios.
- Implement and benchmark the JVP-based forward gradient computation, or clearly reframe FedMPDD as communication-only compression assuming gradients are computed for other reasons.
- Discuss practical guidance for selecting m given model dimension d and expected training rounds K, addressing the T < d/m bound more prominently.

---

## Vit5M0G5Gb

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (6.4/10)
- Match: N/A

### Final Review

## Summary

This paper presents a unified theoretical framework explaining "saddle-to-saddle" learning dynamics—where networks learn increasingly complex solutions through sequential plateaus—as a consequence of embedded fixed points, invariant manifolds, and timescale separation. The authors show that fixed points of narrow networks embed in saddles of wider networks (Theorem 1), that weight constraints defining these embeddings are preserved under gradient flow (Theorem 3), and identify two distinct mechanisms: data-induced timescale separation in linear networks (yielding low-rank weights) and initialization-induced timescale separation in quadratic networks (yielding sparse weights). Predictions about network width, data distribution, and initialization effects are validated experimentally.

## Strengths

- **Unified architectural framework**: The parameterization in Equation (1) cleanly encompasses fully-connected, convolutional, and self-attention layers under a single formalism, enabling theorems (particularly Theorem 1 on embedded fixed points and Theorem 3 on invariant manifolds) to apply broadly. This unification is a genuine conceptual advance over prior architecture-specific analyses.

- **Rigorous mathematical foundation for core results**: Theorems 1 and 3 are properly proved (Appendices E, F) with clear distinctions between cases (equal weights, zero weights, proportional weights, linear dependence) that correspond to different architectural properties. The identification that cases (5)-(7) in Theorem 1 are the ones actually visited during training, while case (4) is not, is an important empirical observation with theoretical motivation.

- **Novel mechanistic disentanglement**: The distinction between data-induced timescale separation (Theorem 4 for linear networks, where components grow at rates determined by singular values of Σ_yz) and initialization-induced timescale separation (Proposition 5 for quadratic networks, where one unit dominates via "rich-get-richer" dynamics) is conceptually sharp and empirically supported. This explains why different architectures exhibit different plateau structures—a genuine insight.

- **Non-trivial predictive success**: The prediction that increasing attention heads shortens plateaus while increasing FC network width does not (Figure 2A) follows directly from the theory and represents a concrete, validated architectural insight. The predictions about initialization structure (Figure 2C) and scale (Figure 2D) are similarly grounded and tested.

- **Clear discussion of limitations**: The Discussion section explicitly identifies failure modes (tanh networks, large initialization), restricts rigorous dynamics analysis to two-layer networks, and acknowledges that exhaustiveness of fixed points remains open.

## Weaknesses

- **Gap between early-phase dynamics and full trajectory**: Theorem 4 rigorously establishes that weights become approximately rank-r in the early phase near small initialization. However, the subsequent claim that the network "approaches a fixed point on that manifold" (Section 5.1, para 2) before escaping to the next saddle is not proved. The paper acknowledges (Appendix A.1) that "visiting subsequent saddles is not [well understood]" for full trajectories. This gap between early-phase analysis and complete saddle-to-saddle dynamics is significant for a paper claiming to explain the full phenomenon.

- **Heuristic treatment of ReLU networks**: Despite prominent placement in Figure 1(D,E), ReLU networks lack rigorous theoretical support within the paper. The argument relies on Taylor expansion (ReLU is "approximately linear near zero") and citation of prior empirical observations about "condensing" and "quantizing." The discussion in Appendix C acknowledges ReLU "probably" exhibits saddle-to-saddle dynamics, but the central dynamics mechanism is not worked out for ReLU's piecewise-linear structure.

- **Unjustified assumptions in Proposition 5**: The assumption that Σ_yZ "is symmetric and has both positive and negative eigenvalues" is necessary for the timescale separation argument in quadratic networks, but the paper does not explain what data or task conditions ensure this. For practical self-attention trained on realistic tasks, this assumption may or may not hold, limiting the scope of this analysis.

- **Deep network claims are conjectures**: While Theorems 1 and 3 apply to deep networks, the dynamics analysis (Section 5) is explicitly two-layer. The discussion of deep networks (Section 7, Eq. 17) presents an interesting conjecture about which layers recruit additional units, but this is supported only by simulation, not theory. The paper should more clearly distinguish conjecture from derived result.

- **Limited empirical validation scope**: Experiments primarily use synthetic data with controlled spectral properties and MNIST digits. While appropriate for theory validation, the claim of a "universal mechanism" would benefit from validation on more complex datasets where singular value gaps may be less pronounced or the clear stage-like transitions may not manifest.

## Nice-to-Haves

- **Connection to generalization**: The paper focuses on training dynamics; connecting saddle-to-saddle progression to test performance or implicit regularization would strengthen impact but extends beyond the stated scope.

- **Trajectory proximity to invariant manifolds**: The theory claims trajectories evolve "near" invariant manifolds. A quantitative metric measuring this distance over training would strengthen the empirical validation.

- **Boundary between stage-like and smooth dynamics**: Systematic experiments mapping where saddle-to-saddle dynamics breaks down (initialization scale, data dimensionality, architecture choices) would clarify practical applicability.

## Removed Points

These points are flagged to be removed, treat them with caution:
- *Demand for "real-world dataset validation" (Spark Finder)*: This is scope creep. The paper is a theoretical contribution with controlled experiments for theory validation—appropriate for ICLR theory track.
- *Criticisms about softmax vs. linear self-attention (Harsh Critic)*: The paper explicitly states it analyzes linear self-attention (Eq. 2, main text). The Harsh Critic's claim that the abstract "doesn't qualify" this is overstated—the main text is clear.
- *Demand for comparison with NTK/spectral bias theories (Spark Finder)*: Not within the paper's scope. The paper positions itself as explaining a different phenomenon.
- *Demand for proofs extending to depth > 2 (Spark Finder)*: The paper explicitly scopes rigorous dynamics to two-layer networks and discusses deep networks as conjecture with empirical support. This limitation is stated clearly.
- *Criticisms about square loss assumption (Harsh Critic)*: The paper notes it considers squared error loss, which is a standard choice for analysis. Cross-entropy dynamics is a known open question but not a flaw of the presented work.

## Novel Insights

The disentanglement of data-induced versus initialization-induced saddle-to-saddle dynamics represents a genuine conceptual advance. Prior work observed plateaus and low-rank/spurious patterns but did not clearly identify *why* linear networks develop low-rank weights (data covariance spectrum) while attention mechanisms develop sparse head activation (initialization randomness). The paper correctly identifies that the order of the activation function in the weights (linear vs. quadratic) determines which mechanism dominates—a principled architectural prediction that extends beyond empirical observation.

## Suggestions

- Add a brief analysis or simulation showing how robust the Proposition 5 mechanism is when Σ_yZ eigenvalues are not perfectly symmetric or have no negative eigenvalues. This would clarify the boundary conditions for the quadratic case.

- In Figure 2C, the "large low-rank initialization" result is interesting but described qualitatively. Adding theoretical justification for why dynamics "first drops exponentially" before exhibiting plateaus would strengthen this novel observation.

- Clarify in the abstract that rigorous dynamics analysis is limited to two-layer networks, while fixed point and invariant manifold theorems apply generally. This accurately represents the contribution without overclaiming.

---

## dCtkwjkK0E

- GT: Reject (avg 2.0)
- Predicted: N/A (4.2/10)
- Match: N/A

### Final Review

## Summary

This paper presents a dataset-centric active learning framework for flow matching models with continuous conditions. By analyzing flow matching through a piecewise-linear neural network lens, the authors derive two competing query strategies: one that maximizes diversity by selecting label-consistent data, and one that improves accuracy by selecting label-diverse data. A hybrid strategy with adjustable weights enables trade-off control. Experiments on synthetic and shape design datasets demonstrate improvements over discriminative active learning baselines.

## Strengths

- **Clear problem framing for an underexplored domain**: Active learning for *generative* models (rather than using generative models to aid discriminative tasks) is genuinely understudied. The paper correctly identifies this gap and motivates it with real-world applications where labeling costs dominate (CFD simulations, medical imaging).

- **Intuitive theoretical insight with practical consequence**: The core finding—that data sharing labels with the training set expands the convex hull of possible interpolations (enhancing diversity), while label-distant data tightens approximation bounds (improving accuracy)—provides a transparent mechanism for the diversity-accuracy trade-off. This cleanly motivates the distinct objectives in Q_D and Q_A (Eqs. 4–6).

- **Computationally efficient query mechanism**: Both proposed strategies operate directly on dataset statistics and use lightweight RBF networks for label prediction, bypassing repeated training of the flow matching model. This addresses a practical bottleneck in active learning loops where each query round typically requires expensive model retraining.

## Weaknesses

- **Theoretical assumption lacks empirical validation**: The entire analytical framework (Section 2.2) rests on the hypothesis that trained flow matching networks exhibit piecewise-linear interpolation behavior, motivated by the "condensation phenomenon" cited from Luo et al. (2021); Xu et al. (2025). However, the paper provides **no empirical verification** that the 8-layer fully-connected networks used in experiments actually exhibit this behavior. The theory (Eqs. 1–3, Lemmas 1–2) applies strictly to closed-form flow matching models, and the connection to trained neural networks remains an unsubstantiated leap. This significantly weakens the claimed "rigorous theoretical characterization."

- **Critical hyperparameters unspecified**: The diversity query strategy Q_D (Eq. 4) introduces three weighting coefficients (α, β, γ) that balance label proximity, entropy increase, and data-space distance. These values are **never specified** in the paper. The ablation study (Fig. 9) reports relative importance but provides no absolute values or selection procedure, creating a substantial reproducibility gap.

- **Scale inconsistency in hybrid strategy**: The hybrid strategy Q_hybrid = ωQ_D + (1−ω)Q_A (Eq. 7) combines Q_D (a multi-term weighted sum) and Q_A (a single distance term). These quantities will have different scales across datasets, yet no normalization is described, making the weight ω difficult to interpret or tune in practice.

- **No statistical rigor in experimental results**: Results in Fig. 4 are presented as single lines without error bars, confidence intervals, or information about the number of random seeds. Active learning trajectories are known to have high variance depending on initial random selection; single-run results are insufficient for ICLR standards.

- **RBF label prediction quality unexamined**: Both Q_D and Q_A require predicting labels for unlabeled data using RBF neural networks. The accuracy of these predictions directly determines query quality, yet the paper provides **no analysis of RBF prediction error** or its downstream impact on selection quality.

- **Counterintuitive diversity result unexplained**: The paper claims "Q_D achieves the highest diversity, even outperforming the model trained on the full dataset" (Section 3.2). This striking claim—that strategically selecting ~6% of data yields higher diversity than using all data—receives no substantive analysis. Possible explanations (data imbalance effects, metric artifacts, overfitting to balanced subsets) are not explored.

## Nice-to-Haves

- **Validation of piecewise-linear assumption**: Before building theory on this assumption, empirically verify that trained networks actually exhibit piecewise-linear interpolation in the condition space (e.g., by analyzing activation patterns or interpolation error across conditions).

- **Principled guidance for ω selection**: The hybrid weight ω controls the diversity-accuracy trade-off (Fig. 7), but no method for selecting it is provided. A Pareto-front analysis or marginal-gain heuristic would enhance practical utility.

- **Generative-native uncertainty baselines**: Compare against uncertainty measures derived from the flow matching model itself (e.g., ensemble variance, dropout-based uncertainty) rather than only discriminative-AL methods adapted to this setting.

- **Standard generative metrics**: Supplement the custom pairwise-distance diversity score with established metrics like the actual Vendi score or precision/recall for generative models to facilitate comparison with prior work.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **Citation mismatch (diffusion vs. flow matching)**: The harsh critic claims the paper incorrectly cites diffusion papers as flow matching. While the examples (DALL-E-3, Veo3) are indeed diffusion-based, the methodology itself correctly implements flow matching. This is a minor framing issue, not a substantive flaw.

- **Missing modern AL baselines (BADGE, BatchBALD, BALD)**: The paper explicitly scopes its contribution to comparing against methods designed for discriminative models. While including modern deep AL baselines would strengthen comparison, their absence is not a critical flaw given the paper's focus on generative-specific strategies.

- **Requirement for non-shape-design benchmarks**: The paper explicitly focuses on continuous-condition shape design. Requesting validation on image generation benchmarks is scope creep beyond the stated contribution. If the theoretical framework generalizes, that's a bonus, not a requirement.

- **Demand for theoretical proofs of interpolation properties**: The harsh critic demands rigorous proofs, but this is an empirical methods paper with theoretical motivation. The theory serves to motivate and design the query strategies, which are then validated experimentally. Perfect proofs are not the standard for this type of contribution.

- **Excessive proof-level criticism**: Lemma proofs have gaps in exposition but the key insight (interpolation in label space leads to interpolation in data space) is empirically testable and the strategies are evaluated regardless.

## Novel Insights

The paper surfaces a non-obvious tension: in active learning for conditional generative models, maximizing diversity and accuracy require *opposite* data selection strategies. Label-consistent samples (which intuition might suggest as redundant) are actually essential for diversity—they expand the interpolation space—while label-diverse samples improve accuracy by tightening error bounds. This reframes diversity-accuracy trade-offs from a model-optimization problem to a data-composition problem, with practical implications for annotation budget allocation in scientific design domains.

## Suggestions

1. **Add empirical validation of the piecewise-linear assumption**: Include an experiment showing that trained networks produce piecewise-linear interpolation behavior (or acknowledge it as a modeling approximation if not strictly true).

2. **Specify all hyperparameters**: Provide exact values for α, β, γ in Q_D, along with the selection procedure or sensitivity analysis.

3. **Add statistical reporting**: Report mean ± standard deviation across at least 3–5 random seeds, and specify exact dataset sizes (initial pool, per-round budget, final labeled counts).

4. **Analyze the full-dataset diversity anomaly**: Explain why Q_D can exceed full-dataset diversity, or acknowledge this as a potential metric artifact requiring investigation.

5. **Include scale normalization for Q_hybrid**: Describe how Q_D and Q_A are normalized before combination, or reformulate to ensure comparable scales across datasets.

---

## GMP1S4R6Ke

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (5.1/10)
- Match: N/A

### Final Review

## Summary
LoRA-Mixer applies Mixture-of-Experts routing to LoRA adapters at the linear projection layers of attention/SSM modules, rather than at FFN blocks as in prior work. The paper introduces a Routing Specialization Loss (RSL) that augments standard MoE load-balancing with an entropy regularization term to promote input-aware expert specialization while avoiding expert collapse. Experiments across 15 benchmarks on three base models (LLaMA3-8B, Mistral-7B, Falcon-Mamba-7B) demonstrate improved performance over LoRA-MoE baselines with fewer trainable parameters.

## Strengths
- **Architecture-agnostic design:** By targeting linear projection layers (Q, K, V, O projections), the framework applies uniformly to Transformer and SSM architectures. Falcon-Mamba-7B experiments demonstrate genuine cross-architecture compatibility (Table 2).
- **Theoretical grounding for RSL:** The paper provides convergence analysis (strong convexity via entropy regularization) and generalization bounds (Theorem 2), going beyond typical PEFT papers' theoretical depth. The insight that entropy regularization provides curvature on the routing simplex is technically sound.
- **Comprehensive empirical evaluation:** Testing across 15 benchmarks spanning medical QA, mathematical reasoning, code generation, and NLU provides broad coverage. The inclusion of cross-model transfer (Mistral→LLaMA, Table 5) and internet-sourced LoRA reuse (Table 3) demonstrates practical applicability.
- **Parameter efficiency verified:** Appendix A.4 confirms LoRA-Mixer uses 48% of MixLoRA's trainable parameters (3.88% vs 8.08%). Inference time comparison (Table 12) shows LoRA-Mixer at 0.574s vs MixLoRA at 0.597s, supporting efficiency claims.

## Weaknesses
- **Missing architectural ablation:** The central claim that projection-layer routing outperforms FFN-layer routing is never directly tested. Section 1 claims FFN-based designs yield "shallow output fusion and weak integration" but provides no controlled experiment comparing LoRA-Mixer applied to attention projections vs. FFN layers with identical RSL. Without this ablation, the source of empirical gains (architecture vs. loss function) remains ambiguous.
- **Confusing RSL formulation:** Equation 5 defines $L_{RSL} = \alpha \sum \bar{p}_i \bar{f}_i - \lambda \mathbb{E}[H(p(x))]$. Since minimizing this loss would *maximize* entropy, Section 3.3's statement that "minimizing $H(p(x))$ reduces token-conditional uncertainty" contradicts the formula. The gradient derivation (Eq. 9) shows $+\lambda(\log p_i(x))$ pushing toward peaked distributions, which is correct for specialization, but the sign in Eq. 5 should be $+\lambda \cdot \mathbb{E}[H(p)]$ for consistency. This confusion undermines theoretical clarity.
- **Abstract claims misaligned with tables:** The abstract reports "+3.79%, +2.90%, and +3.95% on GSM8K, CoLA, and ARC-C" but Table 2 shows LoRA-Mixer vs. MixLoRA on LLaMA3-8B yields only +1.09%, +1.55%, and +0.34% respectively. The larger percentages appear to compare against different baselines or use different calculation methods, which is not clearly specified.
- **No standard deviations reported:** The paper states "all experiments are run three times and the average reported" but provides no error bars. Many improvements are in the 0.3–1.5% range (e.g., ARC-C: +0.34%), making statistical significance unverifiable.
- **Non-monotonic data efficiency:** Table 9 shows RSL underperforms auxiliary loss at 4K data before recovering at higher data. Appendix A.16's explanation ("trigger exploration but not yet rich enough to stabilize") is post-hoc and not predicted by the theoretical framework. If entropy regularization were principled, theory should anticipate this instability.
- **Cross-model transfer limited to architecturally similar models:** Appendix A.10 shows Mistral-7B and LLaMA3-8B have identical hidden dimensions (4096), layer counts (32), and activation functions (SwiGLU). The transfer claim is valid but modest—it would be more impressive across models with different dimensions.

## Nice-to-Haves
- **Dynamic Top-K routing:** The conclusion notes fixed Top-K "may limit adaptability." Implementing adaptive K per token or layer would strengthen the efficiency narrative.
- **Expert quality sensitivity analysis:** Testing how RSL handles LoRAs of varying quality (e.g., injecting noisy or conflicting experts) would validate robustness for internet-sourced adapters.
- **Scaling beyond 6 experts:** Current experiments use 6 experts; understanding performance scaling with expert count would inform large-scale modular model design.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Harsh critic: "Missing weight-space merging baselines (TIES-merging, DARE, Model Soups)":** The paper focuses on routing-based LoRA composition, which is a distinct paradigm from weight-space merging. Comparing against LoRAHub, MoLE, MixLoRA, LoRA-LEGO, PHATGOOSE, GMoE, DS-MoE, and AESL provides appropriate baselines for the routing-based approach. Demanding weight-merging comparisons is scope creep.

- **Harsh critic: "Missing full fine-tuning upper bound and multitask LoRA baseline":** Single-task LoRA is already included as "LoRA" in all tables, serving as a natural baseline. Full fine-tuning would require substantially more compute and is not the appropriate comparison for a PEFT method.

- **Harsh critic: "Medical-QA uses DeepSeek-R1 as evaluator which is a black-box":** While a methodological concern, using LLM-as-judge for open-domain QA is increasingly standard practice. The paper documents the evaluation methodology transparently.

- **Harsh critic: "Table 3 shows 4 tasks but names 5":** Table 3 correctly shows 5 columns (SST-2, CoLA, MRPC, RTE, QQP). This is a minor proofreading observation, not a substantive weakness.

- **Balanced reviewer: "Gradient derivation introduces Lagrange multiplier μ without explaining practical computation":** Standard gradient computation through softmax handles the simplex constraint implicitly. The Lagrange multiplier analysis provides theoretical insight but doesn't require explicit computation in implementation.

- **Spark finder: "Cross-model transfer handles vocabulary mismatch":** LoRA adapters only modify weight matrices, not embeddings. The paper correctly targets projection layers (A.10 shows matching dimensions), so vocabulary alignment is irrelevant to the transfer mechanism.

## Novel Insights
The entropy regularization in RSL serves dual purposes beyond specialization: (1) it provides strong convexity on the routing simplex (Lemma 1), enabling faster convergence rates than standard auxiliary losses which yield only convex objectives; (2) the generalization bound (Theorem 2) shows RSL reduces hypothesis complexity via the $\lambda$ parameter, formally explaining improved data efficiency. This dual theoretical contribution—optimization stability and generalization—distinguishes RSL from prior MoE auxiliary losses.

## Suggestions
1. **Add a projection-layer vs. FFN-layer ablation:** Apply the identical RSL routing to FFN layers and compare against projection-layer placement. This single experiment would validate the architectural contribution independently of the loss function.
2. **Clarify the RSL sign convention:** Either change Eq. 5 to $L_{RSL} = \alpha \sum \bar{p}_i \bar{f}_i + \lambda \mathbb{E}[H(p(x))]$ or revise Section 3.3 to correctly state "maximizing $H(p(x))$ encourages exploration." Ensure the paper is internally consistent.
3. **Add standard deviations to all tables:** Report mean ± std from three runs. Statistical significance claims require error bars, especially for improvements under 1%.
4. **Investigate the 4K instability:** If the explanation in A.16 is correct, the theory should predict why entropy regularization causes temporary performance drops at specific data regimes. A learning curve with finer granularity (e.g., 1K, 2K, 3K, 4K, 5K, 6K, 8K, 10K) would clarify the behavior.
5. **Clarify abstract performance claims:** Explicitly state which baseline each percentage improvement refers to, or align the abstract numbers with the main table comparisons.

---

## j3htU5i01r

- GT: Reject (avg 4.0)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary

This paper proposes a compositional meta-learning framework that learns a probabilistic generative model of tasks by separating within-module dynamics (implemented as module RNNs) from between-module transition statistics (captured by a gating RNN). Once trained, new tasks are solved not through parameter updates but through probabilistic inference using particle filtering to identify the optimal module sequence. The approach is validated on synthetic rule-learning and motor-control tasks, demonstrating one-shot inference, robustness to sparse feedback, and generalization to longer sequences than seen during training.

## Strengths

- **Principled probabilistic formulation**: The paper provides a clean mathematical framework that treats compositional meta-learning as inference in a learned generative model. The particle filtering approach (Equations 5-8, Appendix A.2) with Gumbel-softmax reparameterization for discrete module selection is well-grounded and correctly derived.

- **Strong architectural inductive bias**: The explicit separation of modules ("syllables") from gating ("grammar") is well-motivated and empirically validated. The ablation experiments (Figure 3) convincingly show that the gating network is essential for sparse-feedback inference—the model without gating fails precisely where the full model succeeds (Figure 3c vs. 3d).

- **Length generalization capability**: The model successfully infers solutions for test tasks that are 4× longer than training tasks (Figure 2f), demonstrating genuine generalization of the learned transition statistics rather than mere memorization of fixed-length sequences.

- **Clear empirical recovery of ground truth structure**: In controlled synthetic domains, the model recovers the true shift operations (Figure 2b) and learns history-dependent transition statistics (Figure 2c) that capture non-Markovian structure—something a standard HMM cannot achieve.

- **Solid reproducibility commitment**: The authors provide complete code, trained weights, and scripts to regenerate all figures, meeting high open-science standards.

## Weaknesses

- **Evaluation limited to low-dimensional synthetic tasks**: Both domains (6D vector shifts and 2D motor chunks) are specifically designed for the method, with known module structure and clean boundaries. The paper does not evaluate on any standard meta-learning benchmarks, making it impossible to assess whether the approach generalizes to higher-dimensional problems, real-world data, or settings where compositional structure is less clean.

- **Computational cost of particle filtering not analyzed**: Training uses 250 particles, but the paper provides no analysis of how performance degrades with fewer particles, nor how inference cost scales with module count, sequence length, or observation dimension. A core claim is avoiding parameter updates, but the wall-clock cost of running 250 particles forward per timestep versus taking gradient steps is never quantified (Figure 3e shows episode count, not compute time).

- **Fixed number of modules is a significant constraint**: The architecture requires specifying N in advance, and Figure A1 shows that mismatched N leads to either unused modules or degraded performance. There is no principled mechanism for model selection or dynamic module addition—this limits applicability to real problems where compositional structure is unknown.

- **Train-time vs. inference-time particle filtering mismatch not investigated**: During training, the paper uses guided particle filtering (Equations 10-14) which conditions on future observations; at test time, it uses bootstrap filtering without this lookahead. The paper acknowledges this difference but does not analyze whether it introduces systematic bias or how many particles are needed at test time to match training-time performance.

- **Sparse-feedback claims supported only by qualitative examples**: Figures 2e and 4e show single episodes with sparse feedback. There is no systematic study of how inference accuracy varies with feedback density, what the minimum viable feedback is, or how robust the method is to timing noise in the feedback signal.

- **Training instability acknowledged but not deeply analyzed**: The paper mentions a "chicken-and-egg" problem between module specialization and gating consistency (Discussion), and Appendix A.1 notes that small initial weights (w_init = 0.01) are needed for stability. However, there is no analysis of failure rates, convergence sensitivity to hyperparameters, or mitigation strategies beyond "use small initialization."

- **Task design choices make module identification artificially easy**: In both domains, different modules/skills have different durations (3, 4, or 5 timesteps), providing a strong signal for identifying module boundaries. The paper does not test whether recovery succeeds when all modules have identical duration, which would be substantially harder.

- **No comparison to modern in-context learning baselines**: The paper compares to MAML and MLDG (Figure 3e) but not to transformer-based meta-learning or attention-gated modular networks, which similarly achieve rapid adaptation without gradient updates. This leaves the relative advantage of particle filtering versus contemporary inference-based methods unclear.

## Nice-to-Haves

- **Observation noise robustness analysis**: Testing performance under varying noise levels would strengthen the probabilistic inference claims.

- **Automatic module count determination**: A model selection criterion (e.g., held-out likelihood as a function of N) would reduce reliance on pre-specifying the number of modules.

- **Comparison to Markov gating**: An ablation testing whether a simpler Markov chain suffices versus the proposed RNN gating would validate the necessity of the architectural choice.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"One-shot claim is misleading"**: The paper explicitly states "single episode" and conditions on learned modules; this is the intended claim scope. The abstract correctly describes achieving rapid inference "from just single examples" in the context of test tasks, not learning from scratch.

- **"Test tasks are not genuinely novel"**: This misunderstands compositional generalization. The paper's test tasks are novel *combinations* of learned modules, which is precisely what compositional meta-learning aims to achieve. Comparing to MAML on a vanilla RNN is appropriate because MAML is architecture-agnostic by design.

- **"Motor learning variant is a fundamentally different model"**: The changes (removing input x_t, resetting hidden states, module-specific projections) are practical adaptations for a different domain. The core probabilistic framework and inference procedure remain identical.

- **"Equation flow disrupted by figure descriptions"**: This is a minor formatting issue that does not impede understanding.

- **"No Related Work section"**: This is a stylistic preference. ICLR does not mandate a dedicated Related Work section, and relevant citations are integrated throughout the Discussion.

- **"No cross-domain transfer demonstrated"**: This would strengthen the paper but is outside its stated scope of demonstrating compositional inference within domains.

## Novel Insights

The framework's core insight—that meta-learning can be reformulated as inference in a learned generative model rather than gradient-based adaptation—is conceptually significant. The paper demonstrates that learning "task grammar" (transition statistics) and "task syllables" (module dynamics) separately creates a powerful constraint for test-time inference, particularly when feedback is sparse. The sparse-feedback results (Figures 2e, 4e) reveal an interesting property: the gating network constrains the hypothesis space even when no feedback is available, allowing inference to "continue" from the last confirmed hypothesis. This is a genuine contribution that standard gradient-based methods cannot replicate. However, without evaluation on non-synthetic tasks or comparison to transformer-based in-context learning, the practical significance remains an open question.

## Suggestions

- **Quantify inference cost**: Report wall-clock time and FLOPs for particle filtering at test time versus gradient steps for baseline methods. Even a simple comparison (e.g., "250 particles × T timesteps costs X ms vs. Y gradient steps cost Z ms") would substantiate the efficiency claim.

- **Systematic sparse-feedback study**: Plot inference accuracy as a function of feedback density (e.g., 10%, 25%, 50%, 75% feedback) with confidence intervals across multiple episodes and seeds. This would replace qualitative examples with quantitative evidence.

- **Add particle count sensitivity analysis**: Show how performance degrades as K decreases from 250. This is critical for understanding whether the method remains viable when computational resources are constrained.

- **Consider standard benchmark evaluation**: Even a single experiment on Meta-World or a procedural task suite would substantially strengthen claims about real-world applicability.

---

## rBj2iVyrhh

- GT: Reject (avg 2.0)
- Predicted: N/A (3.8/10)
- Match: N/A

### Final Review

## Summary

This paper proposes Classifier-Constrained Alternating Training (CCAT), a two-stage framework to mitigate modality imbalance in multimodal learning. The authors first pretrain an "unbiased" classifier using bidirectional cross-attention fusion and a mutual-information-based regularization term that penalizes contribution disparities between modalities. This classifier is then frozen, and alternating training proceeds with modality-specific LoRA adapters to bridge the distribution shift between fused-feature pretraining and unimodal-feature inference. Sample-level secondary updates further optimize underperforming modalities. The method achieves consistent accuracy improvements on CREMA-D, Kinetic-Sound, and MVSA benchmarks.

## Strengths

- **Clear problem diagnosis with empirical grounding:** The paper correctly identifies that existing alternating training methods (like MLA) decouple encoder gradients but fail to address classifier bias toward faster-converging modalities. Figure 1 provides empirical evidence that contribution disparities persist even under alternating training, substantiating the core motivation.

- **Modular and well-motivated architecture:** The two-stage design is conceptually sound—pretraining a classifier with contribution-aware regularization to obtain a balanced initialization, then freezing it to provide stable optimization targets. The addition of sample-level secondary updates for extreme-imbalance samples provides a targeted refinement mechanism.

- **Comprehensive ablation study:** Table 2 systematically validates each component (classifier freezing, alternating training, secondary updates, LoRA), showing that all components contribute positively on CREMA-D. The validation of the frozen classifier's benefit for weak modalities is clearly demonstrated.

- **Consistent empirical gains:** The method shows improvement across three benchmarks with diverse modality combinations (audio-video for CREMA-D/KS, image-text for MVSA), demonstrating some generalizability of the approach.

## Weaknesses

- **Theoretical claims are overstated:** Section 3.1 claims to establish a "profound theoretical isomorphism" between class and modality imbalance, but provides only gradient-level analogies. Critically, Eq. (3)'s decomposition **f** = γ₁**f**^(1) + γ₂**f**^(2) assumes linear fusion, yet the paper's own BiCross attention module uses nonlinear attention-based fusion—making the theoretical analysis inapplicable to the proposed architecture. The "isomorphism" should be characterized as a conceptual parallel, not a theoretical framework.

- **Missing key baseline comparisons:** The paper explicitly lists MLA, MMPareto, and LFM as "Recent SOTA" baselines in Section 4.1, but Table 1 does not contain results for these methods—it jumps from OGM-GE/QMF directly to CCAT. Additionally, SMSL (Zhou et al., 2025b), which provides the core MI-based contribution formula adopted by this paper, is not compared against. This is a significant gap for claims of consistent SOTA improvement.

- **MI regularization forces artificial balance:** With softmax normalization (Eq. 6), c¹ + c² = 1, so the regularization L_reg = |c¹ - c²| penalizes deviation from equal contributions regardless of genuine modality informativeness. On MVSA, where unimodal accuracies are ~70% (text) vs. ~27% (image), forcing equal MI contributions could suppress genuinely informative signals. The paper does not ablate this regularization on datasets with severe modality quality imbalance.

- **Pretraining-inference distribution mismatch is inadequately addressed:** The classifier pretrained on cross-attention fused features **f**_i must process unimodal features z^m during alternating training. LoRA is proposed to bridge this gap, but the ablation shows LoRA removal only drops performance by ~1.2% (84.68% → 85.89% on CREMA-D), suggesting the distribution mismatch is either minor or LoRA's benefit comes primarily from added parameters rather than principled distribution alignment.

- **No computational efficiency analysis:** The framework involves: (1) classifier pretraining with cross-attention and MI regularization, (2) alternating encoder updates, (3) sample-level secondary updates. No training time, GPU hours, FLOPs, or memory comparisons against baselines are provided, making practical adoption trade-offs unclear.

- **Limited evaluation scope:** All three benchmarks are relatively small-scale classical datasets. Absence of validation on larger, noisier, or modern multimodal benchmarks (e.g., AudioSet, vision-language datasets) limits claims about scalability and applicability to contemporary multimodal systems.

## Nice-to-Haves

- **Compute-normalized baseline comparison:** Compare CCAT against baselines with equivalent total training budget to ensure gains stem from methodological innovation rather than additional compute.

- **LoRA vs. full fine-tuning ablation:** Compare LoRA adapters against full per-modality classifier fine-tuning to determine whether low-rank constraints are truly beneficial versus simply adding learnable parameters.

- **Cross-dataset classifier transfer:** Validate whether the "unbiased" pretrained classifier transfers across datasets, demonstrating its generalization capability rather than dataset-specific tuning.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Notational inconsistency with L variable:** The harsh critic flagged L being used for both batch size (Eq. 11 context) and retrain subset size (Eq. 12). While true, this is a minor notation issue that doesn't affect methodological correctness.

- **Eq. 2 derivation precision:** The critic noted that the gradient approximation for minority classes applies only to the correct class index j=y. While the derivation could be clearer, this doesn't invalidate the core insight about gradient suppression dynamics.

- **KS ablation anomaly claim:** The critic claimed removing alternating training improves audio accuracy on KS, but the extracted ablation table shows CREMA-D results only. Cannot verify this claim from available content.

- **t-SNE visualization concerns:** The critic questioned the reliability of clustering metrics on t-SNE embeddings. While t-SNE has limitations, the CH/SH/DB scores provide supplementary quantitative evidence that complements the qualitative visualization.

- **LoRA on single FC layer:** The critic argued LoRA on a small classifier is equivalent to adding a rank-r linear pathway. While architecturally true, the ablation does show LoRA provides measurable benefit; whether this comes from "anchoring" or expressivity remains an empirical question but not a methodological flaw.

## Novel Insights

The paper surfaces an important observation: in multimodal learning, interventions that address encoder-level gradient conflicts (like alternating training) may be insufficient because the classifier itself can develop entrenched preferences for dominant modalities early in training. This is analogous to class imbalance where majority classes bias decision boundaries early, creating persistent suppression of minority classes. The empirical tracking of contribution values in Figure 1 substantiates this—the gap persists despite encoder decoupling. However, the insight that classifier freezing (borrowed from long-tailed learning) could address multimodal imbalance is the paper's strongest conceptual contribution, even if the theoretical justification has gaps.

## Suggestions

- **Include missing SOTA baselines:** Add MLA, MMPareto, LFM, and SMSL to Table 1 for fair comparison. If computational resources limit this, provide at least representative comparisons on one benchmark.

- **Scale back theoretical claims:** Revise Section 3.1 to present the class/modality imbalance connection as a "conceptual parallel" or "analogous dynamics" rather than "theoretical isomorphism." Acknowledge that the linear fusion assumption does not match the BiCross architecture.

- **Add computational overhead analysis:** Report training time, GPU hours, and parameter counts for CCAT vs. baselines. This is essential for practical adoption decisions.

- **Ablate MI regularization on imbalanced datasets:** Test CCAT with and without L_reg on MVSA (where text is much more informative than image) to determine whether forced contribution balancing harms performance.

---

## OuMNJoKJBQ

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (3.8/10)
- Match: N/A

### Final Review

## Summary
This paper investigates why LLM safety alignment remains vulnerable to jailbreak attacks, hypothesizing that current alignment relies on shallow pattern recognition rather than deep reasoning. Through causal intervention experiments targeting reasoning-critical attention heads, the authors argue that safety refusals operate largely independently of reasoning capabilities. They propose a two-stage approach: (1) fine-tuning on a new Chain-of-Thought (CoT) safety dataset, and (2) Alignment-Weighted DPO (AW-DPO), which assigns distinct optimization weights to reasoning versus response segments based on harmfulness scores. Experiments across multiple model families and attack types show improved safety with maintained utility.

## Strengths
- **Error-Analysis-Driven Method Design:** AW-DPO is explicitly motivated by empirical analysis of failure modes (~15% of jailbreak cases where reasoning and response safety diverge). This principled connection between observed failures and algorithmic intervention is a strong methodological approach.
- **Comprehensive Benchmark Coverage:** The paper evaluates against 20 jailbreak attack types from SorryBench across four model families (Llama-2-7B, Llama-3.2-3B, Llama-3.1-8B, Mistral-7B), plus comparisons with recent methods (SAFECHAIN, Representation Rerouting, STAIR). This breadth is substantial.
- **Open-Science Contribution:** The authors release their CoT safety fine-tuning dataset, which combines safety-critical and utility-oriented prompts with reasoning traces—addressing a gap where prior work often did not release such datasets.
- **Efficiency Over Iterative Methods:** AW-DPO achieves competitive results with single-round SFT+DPO, compared to methods like STAIR-DPO-3 that require multiple training rounds. This is a practical advantage for resource-constrained settings.

## Weaknesses

- **Causal Intervention Methodology Is Flawed:** The paper claims that ablating "reasoning-critical neurons" leaves safety unaffected, proving alignment is "superficial." However: (1) The "reasoning task" uses CommonsenseQA questions concatenated with correct/incorrect answers—a semantic discrimination task, not multi-step reasoning; (2) The "alignment task" distinguishes harmful prompts from benign Natural Questions, which differ in obvious surface lexical features, making near-100% early-layer accuracy expected; (3) The two tasks differ in intrinsic difficulty, making comparison invalid; (4) Showing that CommonsenseQA-specific neurons don't affect safety only proves the circuits are different, not that safety is "shallow." Table 6 shows the probing accuracy on reasoning before ablation is only 41-51% (barely above chance), undermining confidence that these are even "reasoning-critical neurons."

- **Mathematical Formulation Has Unaddressed Issues:** Equation 3 defines $w_{s_t} \in \{0, 1\}$ as binary masks for partitioning tokens, while Equation 4 applies continuous weights $w_\text{reasoning}$ and $w_\text{respond}$ derived from harmfulness score differences. The paper does not clearly explain this two-step process. More critically, if $d_\text{reasoning} = h^\text{chosen}_{rs} - h^\text{rejected}_{rs}$ is negative (i.e., chosen response has higher reasoning-harm than rejected), the weighting scheme can produce values outside [0,1] or invert the optimization signal. No mechanism to handle this edge case is described.

- **CoT Safety SFT Paradox Unexplained:** Table 1 shows that on Llama-2-7B, CoT SFT achieves 0.68% Base ASR while CoT Safety SFT achieves 60.50% Base ASR—much worse. Similar inversions appear on other models. The paper claims CoT Safety SFT "outperforms standard SFT baselines" (Vanilla SFT, Safety SFT) but does not compare to CoT SFT, which performs better. This counterintuitive result—that adding safety data to CoT training degrades safety—needs explanation.

- **LLM-as-Judge Reliability for Weight Assignment:** AW-DPO's core mechanism relies on GPT-4o scoring reasoning and response harmfulness separately. Table 8 reports Pearson correlation of only 0.5761 for reasoning-only scores between perturbed prompt variants. Since weighting is the key differentiator from standard DPO, moderate reliability in the scoring mechanism directly affects method precision.

- **Limited Utility Evaluation:** Utility is measured solely via MMLU (multiple-choice knowledge retrieval), which does not capture instruction-following quality, conversational ability, or open-ended generation quality—all commonly affected by safety fine-tuning. Reported standard deviations are high (e.g., ±13.50% for some configurations), and no formal significance testing is provided.

- **Missing Ablation on Weighting Mechanism:** The paper compares AW-DPO to standard DPO (equal weighting) but does not include a random-weighting baseline. Without this, it is unclear whether the specific harmfulness-derived weights matter or whether any non-uniform segmentation provides similar benefits.

- **Scaling Factor α Appears Without Motivation:** Appendix H introduces a scaling factor α (set to 0.2) that is absent from Equations 3-4. The main formulation should include this parameter or explain its role.

## Nice-to-Haves
- Human evaluation of reasoning trace quality (current reliance on GPT-4o to judge GPT-4o-generated content risks circular validation)
- Evaluation on additional utility benchmarks (MT-Bench, IFEval, AlpacaEval) beyond MMLU
- Adaptive attack evaluation where adversaries explicitly target the reasoning structure
- Testing on larger models (70B+) to verify scaling
- Ablation of AW-DPO without prior CoT-SFT to isolate contribution of each stage

## Removed Points
*These points are flagged to be removed, treat them with caution*

- **"AW-DPO doesn't outperform DPO in some cases"**: Upon checking Table 1 and Table 12, AW-DPO consistently shows lower ASR than standard DPO across models. The critic appears to have misread the table format.

- **"STAIR-DPO-3 comparison lacks depth"**: The paper acknowledges STAIR-DPO-3 achieves better metrics but requires three training rounds versus one. This tradeoff discussion is adequate.

- **"GPT-5 mention in Appendix P"**: The reviewer critic mentions GPT-5 usage, but the paper states "LLMs (e.g., GPT-4o) are used for grammar checking"—not GPT-5. This appears to be a misreading.

- **"Utility standard deviations too high"**: While variance is reported, this is standard for safety benchmarks where ASR can vary across attack categories. This does not invalidate the safety improvements observed.

- **"Reasoning traces may be decorative"**: This speculation is not substantiated by experiments in the paper and would require additional investigation beyond the current scope.

## Novel Insights
The error analysis identifying "correct reasoning + unsafe response" and "incorrect reasoning + safe response" as distinct failure modes is a valuable diagnostic contribution. It suggests that standard preference optimization, which treats responses monolithically, may miss fine-grained misalignments between reasoning and output. This decomposition could inform future work on segment-aware alignment. However, the 15% figure means the theoretical improvement ceiling from this mechanism is inherently bounded—and the paper does not explain why AW-DPO sometimes improves on the remaining 85% of cases.

## Suggestions
1. **Resolve the weighting formulation**: Clarify whether token-level weights are binary masks or continuous, and explicitly handle negative weight cases (e.g., through clamping or absolute values).
2. **Explain the CoT Safety SFT paradox**: If adding safety CoT data degrades safety relative to pure CoT SFT, this should be analyzed and discussed—otherwise the baseline comparison appears selective.
3. **Tighten causal claims**: Replace "alignment is superficial" with more measured language such as "safety classification relies on early-layer features that operate largely independently of CommonsenseQA-reasoning circuits" and acknowledge the limitations of the probing methodology.
4. **Add random-weighting ablation**: Include a baseline where reasoning and response weights are randomly assigned (but fixed per sample) to isolate whether harmfulness-derived weights specifically matter.
5. **Broaden utility evaluation**: Add at least one open-ended or instruction-following benchmark to demonstrate that safety gains do not come at the cost of general model capabilities.

---

## zKQSyT7a7n

- GT: Reject (avg 6.0)
- Predicted: N/A (5.7/10)
- Match: N/A

### Final Review

## Summary
VT-WM introduces a multi-task visuo-tactile world model combining exocentric vision with fingertip tactile sensing for contact-rich robot manipulation. Using pre-trained Cosmos and Sparsh-X encoders with a transformer predictor, the model demonstrates improved object permanence (33% average Fréchet distance reduction) and physical compliance (29% improvement) in imagined rollouts, with zero-shot planning achieving up to 35% higher success rates on real-robot contact tasks.

## Strengths
- **Clear motivation and problem formulation**: The paper correctly identifies that vision-only world models fail under occlusion and visual aliasing for contact states, with tactile sensing providing the missing local grounding. Evidence from Figs. 5, 7, and 18 shows concrete cases where V-WM hallucinates object disappearance while VT-WM maintains object permanence during grasp and transport.
- **Novel evaluation metrics for contact perception**: The use of CoTracker-based normalized Fréchet distance to quantify object permanence and causal compliance provides objective, quantitative metrics rather than relying on visual inspection alone. Paired t-tests demonstrate statistical significance across multiple tasks (place fruits, push fruits, cube stacking).
- **Real-world hardware validation**: Zero-shot transfer to Franka Panda + Allegro Hand with Digit 360 sensors across five contact-rich tasks validates the approach beyond simulation. The data efficiency experiment (77% vs 22% success with 20 demos of a new task) demonstrates practical sample efficiency for the world model paradigm.

## Weaknesses
- **Insufficient statistical power in planning experiments**: The planning evaluation uses only 5 trials per task, meaning the finest resolution is 20 percentage points. A claimed 35% improvement on reach & push corresponds to 3 vs 1 success out of 5 trials—this difference cannot be statistically distinguished from noise. No significance tests are reported for any planning results, undermining confidence in the headline claims.
- **Confounded comparison in data efficiency experiment**: VT-WM uses open-loop CEM planning while ACT (BC) is deployed closed-loop. The comparison conflates the world model architecture with the planning paradigm. Open-loop execution may be advantageous for goal-image-conditioned tasks but problematic for reactive contact correction, making it unclear whether gains come from the visuo-tactile model or execution mode.
- **Missing ablations on modality fusion**: The paper concatenates vision and tactile tokens without comparing alternative fusion strategies (cross-attention, late fusion, modality gating). The asymmetric temporal windows (1.5s vision, 0.16s tactile) are stated but not justified through ablation. Without these studies, it is unclear whether the design choices are optimal or arbitrary.
- **Unexplained regression in scribble task**: Causal compliance (Fig. 6) shows VT-WM performing worse than V-WM on the scribble task (t = -1.22, p = 0.23). The paper does not explain whether tactile signals are uninformative, noisy, or actively harmful for this task type, which undermines claims of universal benefit.

## Nice-to-Haves
- **Tactile-aware planning cost function**: The planning objective uses only visual latents (ℓ₂ distance to goal image). A tactile-augmented cost (e.g., penalizing predicted slip or rewarding predicted contact during goal-reaching) could leverage tactile predictions more directly.
- **Test-time tactile ablation**: Experiments removing tactile input during inference (while keeping it during training) would clarify whether tactile is strictly necessary for deployment or primarily acts as a training regularizer.
- **Failure mode characterization**: Detailed analysis of when VT-WM fails to predict contact correctly would strengthen reliability claims for safety-critical manipulation.
- **Multi-task BC baseline**: Comparing against a multi-task ACT policy (rather than single-task) would isolate whether gains come from world model structure or multi-task training itself.

## Removed Points
- "Missing related work references (RoboPack, Tian et al. 2019, Sutanto et al. 2019)" — Per instructions, I do not have external sources to confirm what related work should be cited and could risk making incorrect claims about existence or relevance of works not discussed.
- "32 A100 GPUs is substantial computational cost" — This is standard for systems robotics papers and not a weakness of the method itself; CEM planning latency concerns are retained below.
- "CoTracker may be unreliable under occlusion" — The metric compares model rollouts to ground truth trajectories using the same tracking method. Since both V-WM and VT-WM predictions are evaluated against ground truth using identical CoTracker-based metrics, any unreliability applies equally. The key comparison is the relative improvement, not absolute accuracy.
- "Dataset too small (124 demonstrations)" — This is a deliberate design choice; the data efficiency results demonstrate the model works with limited data, which is a strength, not a weakness.
- "Training/test task overlap limits generalization claims" — The paper explicitly acknowledges this limitation in Appendix D and does not claim out-of-distribution generalization. Criticizing absence of OOD evaluation when not claimed is scope creep.
- "PDF formatting issues (equation placement, garbled tables)" — Per instructions, formatting nitpicks should be removed.
- "ACT with tactile input not tested" — The comparison is between VT-WM planning and standard BC. Testing ACT with tactile would be informative but the paper's focus is world models, not policies, making this a nice-to-have rather than core weakness.

## Novel Insights
The core insight—that tactile sensing disambiguates visually identical contact states (hand hovering vs. touching a cloth)—is genuinely valuable and demonstrated through both quantitative metrics and qualitative rollouts showing reduced hallucination. The observation that world models trained on contact-rich tasks transfer contact dynamics priors to new tasks (data efficiency result) suggests tactile representations capture transferable physical knowledge beyond task-specific features. The deliberate design choice to use tactile only as context (not as planning goal) reveals an interesting hypothesis: tactile's primary role may be improving rollout fidelity rather than goal specification.

## Suggestions
1. **Increase planning trial counts**: Report confidence intervals or conduct significance testing. Even n=10 per task would substantially improve reliability of percentage claims.
2. **Add closed-loop VT-WM comparison**: Execute VT-WM with replanning at each step to isolate whether open-loop vs. closed-loop execution explains the BC comparison gap, or whether gains genuinely come from the world model structure.
3. **Analyze the scribble task regression**: Discuss why VT-WM underperforms on certain tasks and what this reveals about limits of tactile signal utility for different manipulation types.
4. **Report CEM planning latency**: Practical deployment requires understanding whether CEM optimization can run at sufficient frequency for real-time contact tasks.

---

## WwDNiisZQm

- GT: Accept (Poster) (avg 4.0)
- Predicted: N/A (6.3/10)
- Match: N/A

### Final Review

## Summary
The paper proposes Content-Aware Mamba for Learned Image Compression (CMIC), which addresses two limitations of applying vanilla Mamba to image compression: rigid content-agnostic raster scanning and strict unidirectional causality. The authors introduce Content-Adaptive Token Permutation (CTP), which reorders tokens by feature-space similarity rather than spatial proximity, and Global-Prior Prompting (GPP), which injects cluster-derived statistics into the SSM's output matrix. CMIC achieves state-of-the-art BD-rate performance (-15.91%, -21.34%, -17.58% on Kodak, Tecnick, CLIC vs. VTM-21.0) with favorable efficiency compared to prior Mamba-based LIC methods.

## Strengths
- **Well-motivated architectural design**: The paper correctly identifies that vanilla Mamba's fixed raster scan order and strict causality are mismatched with image compression's need to model long-range dependencies between semantically similar but spatially distant regions. Both CTP and GPP directly address these limitations with clear theoretical justification tied to redundancy elimination.

- **Strong empirical performance with efficiency gains**: CMIC achieves consistent improvements over VTM-21.0 and recent Mamba-based methods (MambaVC, MambaIC) across three benchmarks. Importantly, it does so with 57% fewer FLOPs, 39% lower decoding latency, and 78% less peak memory than MambaIC (Table 1, Section 4.4), demonstrating a favorable complexity-performance trade-off.

- **Thorough ablation and visualization analysis**: The component ablations (Table 2) isolate CTP and GPP contributions, showing they are complementary (combined gain 2.7-3.6% BD-rate). The Effective Receptive Field visualizations (Figures 7, 9, 16) compellingly demonstrate content-adaptive, global dependency capture versus the fixed spatial patterns or cross-shaped artifacts of prior methods.

- **Stable training with non-differentiable components**: The use of EMA-updated codebooks for K-Means clustering provides empirical training stability (Figure 18), and the overhead is quantified at ~5% of training time (Section 4.1). This practical design choice avoids the instability of per-sample online clustering.

## Weaknesses
- **Imprecise causality framing**: The paper describes GPP as "mitigating strict causality" and enabling "non-causal long-range modeling." However, examining the SSM equations (Section 3.4), the hidden state h_i still depends only on h_{i-1} and x_i—GPP modulates the output projection C with a pre-computed global prompt. What GPP actually provides is **global output conditioning**, not bidirectional information flow within the SSM itself. While the ERF visualizations (Figure 9) show activations beyond the raster-scan boundary, this reflects the shared global prior P influencing outputs, not future tokens influencing past hidden states. The paper should correct this terminology to accurately describe the mechanism's contribution.

- **Non-differentiable optimization lacks theoretical grounding**: Token permutation via K-Means sorting is non-differentiable. The paper acknowledges "gradients are biased" (Appendix A.8) but relies solely on empirical stability curves without analyzing how this affects the rate-distortion optimization landscape. No comparison to soft relaxation alternatives (e.g., Gumbel-Softmax routing) or formal analysis of EMA convergence is provided. While empirical stability is demonstrated, the theoretical gap remains unaddressed.

- **Missing training reproducibility details**: Section 4.1 specifies the optimizer (Adam), initial learning rate (10^-4), and λ values, but omits total training steps, learning rate decay schedule, warmup details, and batch size. These are essential for reproducibility in a methods paper.

- **No random permutation baseline**: The ablations compare CTP (content-aware permutation) against no permutation, but do not include a random permutation control. Without this, one cannot conclusively attribute gains to content-awareness specifically versus simply breaking spatial locality. The claim that *content-awareness* drives improvements would be stronger with this isolation.

- **Sorting complexity overhead not discussed**: The paper emphasizes Mamba's linear complexity but does not explicitly acknowledge that sorting N tokens by cluster assignment is O(N log N). While this may be negligible in practice, it should be addressed when making efficiency claims. The paper does state that clustering adds only ~5% training overhead, which implicitly addresses this, but explicit complexity analysis would be more rigorous.

- **Decoder-side permutation reconstruction unclear**: For practical deployment, the decoder must reconstruct the inverse permutation π^{-1} to restore spatial layout. The paper states clustering is "deterministic" at inference (Section 3.3), implying the decoder runs identical K-Means on quantized features ŷ. However, quantization noise may perturb cluster assignments differently at encoder and decoder, potentially causing misalignment. The paper should clarify whether permutation indices are transmitted as side information or implicitly reconstructed, and whether this introduces any bitrate overhead.

## Nice-to-Haves
- **Cross-dataset generalization test**: Testing the fixed codebook on domain-shifted datasets (e.g., medical, satellite imagery) without fine-tuning would validate whether centroids capture universal features versus overfitting to Flickr2W.

- **Visualization of permuted scan paths**: Displaying the actual 1D sequence order after CTP on a sample image would directly verify that semantically similar tokens are grouped consecutively as claimed.

- **Latency component breakdown**: A breakdown separating SSM compute, clustering, and sorting overhead would strengthen the efficiency claims.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **"Within-cluster ordering remains causal"** (Harsh Critic): This is disclosed in Appendix A.10 and does not undermine the method—the claim is that CTP prioritizes inter-cluster interactions between similar tokens; within-cluster processing remains sequential, which is acceptable.

- **"Cluster ordering is arbitrary"** (Harsh Critic): While cluster order is fixed by codebook order, this is a design choice with negligible practical impact. The speculative concern about inter-cluster dependencies is not substantiated by evidence.

- **"Codebook utilization suggests collapse"** (Harsh Critic): Table 5 shows 23-38% of centroids active per image. Unlike VQ-VAE where this indicates training issues, the EMA-based clustering here naturally adapts to image content. Framing 64 as an "upper bound" is appropriate.

- **"Encoder latency not in main text"** (Harsh Critic): Table 10 in the appendix provides full latency metrics. This is standard placement; encoding latency is correctly noted but the claim of selective reporting is overstated.

- **"BD-rate anchor range concerns"** (Harsh Critic): Using different quality point counts is common in LIC literature. Without evidence that this disadvantages baselines, this is not a substantive critique.

- **"Why larger gain on Tecnick?"** (Harsh Critic): Speculative concern without evidence of overfitting. Tecnick is a standard benchmark; differences in resolution or content distribution may explain performance variations.

- **"Need uniform baseline retraining"** (Spark Finder): Standard practice in LIC comparisons is to use reported results from established methods. The comparison includes MambaVC and MambaIC, which are recent works likely trained on similar datasets. Demanding full retraining of baselines is beyond typical reproducibility expectations.

## Novel Insights
The key insight of this work is that Mamba's sequential processing creates a fundamental tension for 2D image data where redundancy is content-driven rather than spatially-ordered. The CTP mechanism's elegance lies in recognizing that SSMs can process tokens in any order—the scan order is not fundamental to the architecture, only to its conventional application. By reordering by feature-space proximity, the method allows distant but semantically similar regions to interact naturally within the sequential model. The GPP mechanism's contribution is more subtle: by pre-computing global statistics from clustering and injecting them as prompts, the model gains awareness of global structure without expensive multi-directional scanning. This suggests a broader design pattern for adapting sequential models to non-sequential data: decouple the scanning order from the spatial structure, and use global statistics to provide context that sequential processing cannot naturally acquire.

## Suggestions
1. **Correct the causality terminology**: Replace "non-causal" and "mitigating strict causality" with "global output conditioning" or "augmenting with global priors." The mechanism's contribution is accurately described as injecting pre-computed global statistics into each SSM step, not as enabling bidirectional information flow.

2. **Add a random permutation ablation**: Include a baseline where tokens are randomly permuted (with consistent inverse at decoder) to isolate whether content-awareness specifically drives gains versus simply breaking spatial locality.

3. **Provide complete training details**: Specify total training steps, learning rate schedule (decay policy, warmup if any), and batch size in Section 4.1 or Appendix.

4. **Clarify decoder-side permutation**: Explicitly state whether permutation indices are transmitted or deterministically reconstructed, and discuss any potential quantization-induced assignment mismatches between encoder and decoder.

---


# Summary

Papers: 50 | Accuracy: N/A
