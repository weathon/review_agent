# ICLR Benchmark Results

Date: 2026-04-07 21:43
Critic/Merger: qwen/qwen3.5-plus-02-15 (OpenRouter)
Neutral: qwen/qwen3.5-flash-02-23, Related Work: qwen/qwen3.5-flash-02-23:online (OpenRouter)

## Xn33bU71m4

- GT: Withdrawn (treated as Reject) (avg 1.3)
- Predicted: N/A (2.2/10)
- Match: N/A

### Final Review

## Summary

This paper systematically evaluates nine open-source mid-sized LLMs (12B–16B range) on two binary reverse engineering tasks—name recovery and type inference from stripped binaries—across four architectures and multiple optimization levels. Using an automated pipeline with an auxiliary LLM for output normalization, the authors find that without fine-tuning, all models achieve near-zero F1 scores, and even with LoRA fine-tuning, performance remains low (type inference F1 < 0.1; name recovery F1 up to 0.33 in isolated cases). The paper's primary contribution is an empirical benchmark demonstrating that current LLMs are fundamentally limited on these tasks.

## Strengths

- **Systematic multi-model, multi-architecture benchmark at unusual scale.** Evaluating 9 LLMs across 4 ISAs, 4 optimization levels, and 40,000 functions with both zero-shot and fine-tuned settings provides a breadth of comparison that exceeds typical ad-hoc evaluations in the reverse engineering literature. This scale enables meaningful cross-model and cross-architecture comparisons (e.g., Finding 4 on training quality vs. model size, Finding 6 on within-family scaling).

- **Honest and transparent reporting of negative results.** The paper does not cherry-pick favorable configurations. It openly reports that assembly-based function name recovery yields 0.00 F1 across all models, that type inference remains below 0.1 even after fine-tuning, and that fine-tuning sometimes degrades performance. This is a valuable corrective against community overestimation of LLM capabilities in low-level program analysis.

- **Automated output normalization pipeline addresses a real methodological challenge.** Different LLMs produce responses in heterogeneous formats, and the use of an auxiliary LLM to standardize outputs is a pragmatic solution that enables fair, large-scale comparison. The placeholder normalization (FUNC, VAR, TYPE) to eliminate decompiler artifacts is also well-motivated.

## Weaknesses

- **Critical numerical inconsistency in the abstract.** The abstract states: "after fine-tuning, DeepSeek-R1 demonstrates a notable improvement, reaching an F1 score of 0.37." Yet examining Tables 2–4, no model achieves 0.37 on any task on x86-64. DeepSeek-R1's highest fine-tuned F1 is 0.17 (Table 3, variable name recovery, O0). No score across any architecture in the appendix tables approaches 0.37 either. This discrepancy between the abstract and the actual reported results is a serious issue that undermines confidence in the experimental rigor of the entire study. If this reflects a calculation error or a result from a different experimental configuration, it must be reconciled; if it is simply wrong, it suggests insufficient verification before submission.

- **No comparison with specialized reverse engineering baselines.** The paper motivates its work by discussing SYMGEN (Jiang et al.), RESYM (Xie et al., 2024), VARBERT (Pal et al., 2024), SYMLM (Jin et al., 2022), and DEBIN (He et al., 2018)—all systems specifically designed for name recovery or type inference. None of these are included as baselines. Without knowing whether existing specialized methods achieve F1 of 0.01 or 0.60 on comparable data, the reported LLM scores (0.01–0.33) are uninterpretable in context. The paper's central claim—that LLMs are "not yet" ready for reverse engineering—requires showing that they underperform relative to the state of the art, not just that their absolute scores are low.

- **Auxiliary LLM normalization is unvalidated.** The evaluation pipeline depends critically on a second LLM to parse and standardize outputs from all target models (§3.2). If the auxiliary model misparses a response—extracting a wrong name, dropping a prediction, or hallucinating a value—the resulting F1 score is corrupted. The paper provides no accuracy metric for this normalization step: no extraction success rate, no manual validation on a sample, no comparison with rule-based parsing. Since the target models often produce "extraneous explanations" and "vary between short tokens and full sentences," the risk of extraction error is non-trivial, and models whose outputs are easier for the auxiliary LLM to parse may be systematically advantaged.

- **Insufficient analysis of failure modes and error types.** The paper reports aggregate F1 scores but provides almost no breakdown of what goes wrong. For type inference (all scores < 0.1), the paper attributes difficulty to "strict evaluation criteria" and "user-defined types," but does not analyze whether models fail on pointers, structs, function pointers, or primitives. For name recovery, there is no categorization of whether errors stem from semantic misunderstanding, hallucination, or incomplete output generation (§4.5 notes models "often miss some" requested names). Without an error taxonomy, the findings—while empirically valuable—cannot guide targeted improvements. Table 8 shows four successful fine-tuning examples but no representative failures.

- **Fine-tuning sometimes degrades performance, with no investigation.** Table 3 shows DeepSeek-V2 variable name recovery dropping from 0.10 to 0.05 after fine-tuning at O1; Table 4 shows many models scoring 0.00 post-fine-tuning where they had 0.01–0.02 before. This is counterintuitive and could indicate overfitting, data contamination, LoRA instability, or evaluation artifacts. The paper acknowledges the inconsistency (Finding 1: "gains are inconsistent and not guaranteed") but treats it as an observation rather than investigating root causes. Given that the paper's other main finding is that fine-tuning helps, the conditions under which it *hurts* are at least equally important.

- **Strict type matching metric without near-miss analysis obscures actual capability.** The type inference evaluation requires exact string matching (§3.4), so `uint32_t` vs. `unsigned int` counts as a complete miss. While the paper justifies this choice, it provides no complementary analysis of near-misses. A simple secondary metric (e.g., type category match, or relaxed matching) would reveal whether models are reasoning correctly about types but differing in nomenclature, or whether they are fundamentally confused. The name recovery task uses CodeWordNet for partial credit, but no analogous concession exists for types, creating an asymmetry that may make type inference appear worse than it actually is.

## Nice-to-Haves

- Comparison with non-LLM baselines (SYMLM, RESYM, VARBERT, or even simple heuristic approaches) to contextualize LLM performance.
- Error taxonomy for type inference failures (e.g., primitive vs. pointer vs. struct vs. function pointer).
- Ablation on auxiliary LLM normalization accuracy (manual validation on a sample of 100–200 outputs).
- Evaluation on binaries compiled from C++, Rust, or Go, or with different compilers (MSVC, Clang), to test generalizability beyond GNU C software.
- Prompt variation ablation, given the paper's own finding that prompt templates significantly affect both training time and output compliance.
- Cross-architecture generalization analysis for fine-tuned models (e.g., train on x86-64, test on ARM).

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Incomplete citations ("Jiang et al." without year, "cha" for ChatGPT).** These are formatting/style issues. The references are present in the bibliography and identifiable. Removed per formatting nitpick rule.

- **Weakness: Missing Finding 5 (findings numbered 1,2,3,4,6,7,8).** This is a minor editorial oversight with no impact on content. Removed per formatting nitpick rule.

- **Weakness: No fine-tuning hyperparameters disclosed (LoRA rank, learning rate, epochs).** Removed per hard rule on reproducibility nitpicks about undisclosed hyperparameters.

- **Weakness: Table 1 formatting makes model counting ambiguous.** Table 1 clearly lists 9 models; the "ambiguity" is a parser artifact. Removed as factually wrong per the paper content.

- **Weakness: "Similar sizes" claim contradicted by CodeLlama 7B/13B/34B size analysis.** The main comparison uses 12B–16B models; the size analysis (Figure 2) is a separate experiment. No contradiction exists. Removed as factually wrong.

- **Weakness: Where do the extra 120K training functions come from?** The paper explicitly states "10,000 functions per architecture and optimization pair" for training. With 4 architectures × 4 optimization levels = 16 combinations × 10,000 = 160,000. The math is correct. Removed as factually wrong.

- **Weakness: Only C programs evaluated, no C++/Rust/Go.** The paper clearly scopes itself to C binaries. Criticizing the absence of other languages is scope creep. Moved to Nice-to-Haves.

- **Weakness: CodeWordNet ablation only on x86-64.** The paper states this is to "illustrate general trends without requiring a full-scale analysis." This is a reasonable scope decision for an ablation. Removed as generic weakness.

- **Weakness: Assembly code experiment is worthless because all scores are 0.00.** The negative result itself is informative—it shows that assembly representation is fundamentally harder than decompiled code. The paper does state this as Finding 3. Whether more analysis of *why* is needed is addressed above under failure mode analysis. Removed as the experiment has legitimate value.

- **Weakness: Grammar errors ("promp" for "prompt", "an unstripped binaries").** Removed per formatting nitpick rule.

## Novel Insights

The most revealing finding across the reviews and the paper itself is the tension between what the paper claims to show and what the data actually supports. The paper frames its contribution as proving "LLMs are not yet ready for reverse engineering," but the data more precisely shows that **zero-shot LLMs with a single prompt and LoRA fine-tuning on decompiled C code produce low F1 scores under strict metrics**. This is a much narrower claim. The variable name recovery results for fine-tuned models (up to 0.33 F1 at O0) are non-trivial and suggest that with better fine-tuning strategies, richer context, and more relaxed evaluation, LLMs may be closer to practical utility than the paper's title suggests. The sharp gap between O0 performance (where decompiled code preserves more structure) and O2/O3 performance (where optimization destroys it) is itself a diagnostic: the bottleneck may be decompiler output quality rather than LLM reasoning ability. This suggests a productive research direction—improving the interface between decompilers and LLMs—rather than simply concluding LLMs are insufficient.

## Suggestions

- **Immediately reconcile the 0.37 F1 claim in the abstract with the actual data.** If this number is incorrect, correct it; if it comes from a different experimental configuration, document it. This is the single most important fix.
- **Add at least one specialized baseline** (e.g., RESYM or SYMLM on a subset of your dataset) so readers can contextualize whether LLM F1 of 0.05–0.33 is below, at, or above the state of the art.
- **Validate the auxiliary LLM normalization** on a manually annotated sample (even 200 examples) and report extraction accuracy per model. If some models' outputs are systematically harder to parse, this biases the comparison.
- **Add a near-miss analysis for type inference.** Report what percentage of incorrect type predictions are in the right category (e.g., integer family, pointer family) versus completely wrong. This costs little but dramatically increases the informativeness of the results.
- **Investigate why fine-tuning degrades performance in specific cases.** At minimum, report training loss curves and check for overfitting on the models where fine-tuning hurts.

## Axis Evaluations

- **Novelty:** Low to moderate. The contribution is empirical benchmarking rather than a new method or insight. The automated normalization pipeline has modest methodological novelty, but the core experimental design applies standard practices (LoRA fine-tuning, F1 evaluation) to a new domain.

- **Technical soundness:** Compromised. The 0.37 discrepancy in the abstract is a serious error. The unvalidated normalization pipeline and the unexplained fine-tuning degradation cases introduce uncertainty about whether the reported numbers accurately reflect model capabilities.

- **Empirical support:** Broad but shallow. The scale (9 models × 4 architectures × 4 optimization levels × 2 settings) is impressive in coverage, but the analysis stops at aggregate F1 scores. The absence of baselines, error categorization, and near-miss analysis means the empirical contribution, while useful, is less actionable than it could be.

- **Significance:** Moderate if the methodological issues are addressed. A rigorous negative result about LLM capabilities in reverse engineering would be valuable, but without baselines for comparison and without understanding failure modes, the practical impact is limited. The paper is more significant for the reverse engineering community than for the core ICLR audience.

- **Clarity:** Adequate with a critical exception. The paper is generally readable and well-structured, but the abstract's numerical error and the inconsistent finding numbering suggest insufficient proofreading. The table formatting in the extracted version is challenging, though this may be a parser artifact.

---

## eu3PwSle8J

- GT: Reject (avg 4.7)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary

The paper proposes Augmented Intermediate Representations (AIR), a defense against indirect prompt injection attacks that injects layer-specific trainable embeddings encoding privilege levels into every decoder layer of an LLM, rather than only at the input layer. The core hypothesis is that input-level instruction hierarchy (IH) signals degrade as they propagate through the network; by recurrently injecting these signals, AIR achieves 1.6×–9.2× reduction in attack success rate (ASR) against gradient-based attacks compared to prior IH-based defenses (Delimiters, ISE), with minimal utility degradation across Llama-3.2-3B, Qwen-2.5-7B, and Llama-3.1-8B.

## Strengths

- **Strong mechanistic motivation with direct empirical validation.** The paper doesn't just hypothesize signal degradation—it demonstrates it via two complementary analyses: cosine similarity between privilege-level representations increases across layers for Delimiters/ISE (Figure 3), and linear probe accuracy for predicting privilege level drops from perfect to ~91% for ISE by the final layer, while AIR maintains near-perfect separability throughout (Figure 10, Appendix E). This dual evidence makes the motivation substantially more convincing than a purely intuitive argument.

- **Substantial robustness improvements on gradient-based attacks.** The ASR reductions are large and consistent: on GCG against Llama-3.2-3B with SFT, AIR achieves 4.1% ASR vs. 38% (Delim) and 48.1% (ISE); on Astra, 0.1% vs. 14.5% and 25.8%. These are not marginal improvements—they represent order-of-magnitude gains on the strongest attack category, which is where defensibility matters most.

- **Minimal architectural overhead.** The method adds only 0.4M parameters (0.005%) for Llama-3.1-8B and requires only a simple table lookup and vector addition per layer per token, making it straightforward to implement and deploy atop existing model stacks.

## Weaknesses

- **Insufficient attack optimization budget for gradient-based attacks.** The paper uses 50 steps (SFT models) or 200 steps (DPO models) for GCG/Astra optimization (Section 5.4). Standard GCG evaluations typically employ 500–1000+ steps. Fifty steps is exceptionally low and likely insufficient for attack convergence, especially against defended models where the loss landscape may be more complex. This risks substantially underestimating ASR and inflating the perceived robustness of all defenses, including AIR. The 1.6×–9.2× improvement claims rest on these under-powered attacks; their validity at higher budgets is unknown and critical to assess.

- **No adaptive attack evaluation.** All evaluated attacks treat the model as a fixed system. An attacker with white-box access who is aware of AIR's per-layer embedding tables could potentially optimize adversarial prefixes to counteract or cancel the injected IH signals (e.g., by learning perturbations that align with the embedding structure). This is a specific vulnerability of additive architectural defenses that the paper neither evaluates nor discusses, despite claiming robustness against "gradient-based (white-box)" attacks.

- **Missing ablation studies on AIR's own design.** The paper does not investigate key design choices: (a) whether injecting at every layer is necessary versus a sparse subset (e.g., every 4th layer), (b) whether layer-specific embeddings outperform a single shared embedding table across layers, or (c) the sensitivity of robustness to embedding dimensionality. Without these ablations, it is unclear whether the recurrent injection per se is the key factor, or whether simply adding more trainable parameters at any layer suffices.

- **Initialization sensitivity across architectures is a practical concern.** Appendix B.2 reveals that the default initialization ($\sigma=0.02$) failed for Qwen-2.5-7B, requiring a 5× larger $\sigma=0.1$. The paper attributes this to Qwen's larger activation magnitudes and provides heuristic guidelines, but no systematic ablation of $\sigma$ is performed. This sensitivity raises questions about out-of-the-box applicability to new architectures and whether the method requires per-model hyperparameter tuning that undermines its generality.

- **Residual attack success rates are unanalyzed.** Even with AIR, GCG achieves 22.6% ASR on Qwen-2.5-7B with SFT and 11.3% on Llama-3.1-8B with SFT. The paper does not analyze what characterizes the remaining successful attacks—whether they exploit specific token patterns, privilege conflicts, or architectural blind spots. Understanding these failure modes is essential for trusting the defense in high-stakes deployments.

- **No variance reporting across random seeds.** Given the acknowledged sensitivity to embedding initialization (Appendix B.2), the absence of multi-seed variance reporting for both robustness and utility metrics leaves the reliability of the reported numbers unverified. A single training run per configuration is insufficient to establish that the improvements are robust to initialization randomness.

## Nice-to-Haves

- Evaluation on reasoning benchmarks (e.g., GSM8K, MATH) to more sensitively detect capability degradation from modifying the residual stream at every layer; MMLU (Appendix G) is a start but tests factual knowledge rather than reasoning.
- Multi-turn conversational evaluation, which the paper explicitly scopes out but which is critical for agentic deployments—the primary motivating scenario.
- Comparison or combination analysis with detection-based defenses (guard models, perplexity filters) to clarify whether AIR is complementary to or redundant with that defense class.
- Sparse injection ablation (injecting at every $N$-th layer) to determine if full recurrent injection is necessary or if a more efficient variant suffices.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Inference latency concern** (harsh critic): Claimed that per-layer embedding lookups could disrupt kernel fusion and increase HBM bandwidth pressure. However, the operation is a single vector lookup from a tiny table (3 entries of dim 4096) plus an addition per layer—this is genuinely negligible compared to the attention and FFN computations. Without evidence of actual latency impact, this is speculative engineering nitpicking.

- **Architectural parity concern** (harsh critic): Whether the "None" baseline shares the same architectural modifications as AIR. The 0.005% parameter difference is too small to meaningfully conflate results with capacity differences, and the structural change (table lookup) does not alter the computational graph in a way that would independently improve performance.

- **Interference risk during Stage 1 training** (harsh critic): Speculation that AIR embeddings could "learn noise" during non-adversarial instruction tuning before adversarial training. This is purely hypothetical—the utility results show no degradation, which directly contradicts the concern. Without evidence of interference, this is unfounded.

- **Zero ASR on static attacks as evidence of memorization** (harsh critic): The paper explicitly acknowledges in Section 6.1 that Naive and Ignore attacks are "in-distribution as they are seen during adversarial training." The reviewer raised this as a concern, but the paper already addresses it.

- **Scalability to 70B+ models** (harsh critic): This is scope creep beyond the paper's stated experimental range. The limitation is acknowledged in Appendix A.

- **Missing related works / detection baseline comparison** (multiple reviewers): Hard rule—cannot flag missing related works. The comparison with detection methods is a nice-to-have, not a core flaw, since the paper explicitly positions itself within the IH defense framework.

- **Quantization compatibility** (spark finder): Not standard in this field for defense evaluation; scope creep.

- **Cross-architecture transfer of embeddings** (spark finder): Outside the paper's stated scope; the paper trains per-model.

## Novel Insights

The analogy between IH signal injection and positional encoding evolution (Section 4) is genuinely insightful. Just as the field moved from input-only sinusoidal/learned positional embeddings to RoPE's per-layer injection of relative position—and found this architectural choice critical for length generalization and performance—AIR applies the same principle to privilege information. This parallel suggests a broader meta-pattern: any signal that must persist through deep computation (position, privilege, task identity) may benefit from recurrent injection rather than input-only provision. The linear probing results in Appendix E provide causal-ish evidence that this is not merely an inductive bias but an observable phenomenon—ISE's probe accuracy degrades from 100% to 91% across layers while AIR stays near-perfect—making this one of the clearer demonstrations of "signal dilution" in intermediate representations that the community has produced.

## Suggestions

- **Re-run gradient-based attacks with substantially higher optimization budgets** (≥500 steps for all models) and report the resulting ASRs. This is the single most important action to validate the core claims. If AIR's advantage persists at higher budgets, the contribution is strong; if it shrinks significantly, the narrative needs revision.

- **Add a sparse injection ablation**: Test AIR with injection at every 2nd, 4th, and 8th layer. If every-other-layer injection matches full injection, the method becomes more efficient and the "recurrent" claim is refined; if robustness drops sharply, it validates the every-layer design.

- **Report results across at least 3 random seeds** for the primary configurations (especially AIR-DPO on Llama-3.1-8B and AIR-SFT on Qwen-2.5-7B) with standard deviations for both ASR and utility.

- **Discuss adaptive attacks explicitly** in the limitations section: acknowledge that an attacker aware of AIR could optimize against the embedding tables, and characterize what additional robustness (if any) the per-layer injection provides over input-only injection under such an adaptive threat model.

- **Analyze failure cases**: For configurations where ASR remains above 10% (e.g., AIR-SFT on Qwen-2.5-7B at 22.6%), examine the successful attack prefixes and model outputs to identify patterns, even qualitatively. This would strengthen the paper's contribution from "it works better" to "here is when and why it works, and here is where it doesn't."

---

## mHRuCmc9lo

- GT: Accept (Poster) (avg 7.3)
- Predicted: N/A (5.2/10)
- Match: N/A

### Final Review

## Summary

This paper develops a minimax-optimal decision-making framework for forecasters that satisfy partial calibration guarantees (H-calibration) rather than full calibration. Using Lagrangian duality, the authors characterize the optimal robust decision rule in closed form (Theorem 3.1) and prove a striking "sharp transition" result: once the test class H contains the decision-calibration indicators, the robust policy collapses to the simple plug-in best response (Theorems 4.1–4.2), recovering the trustworthiness semantics of full calibration at a substantially weaker and more tractable condition.

## Strengths

- **Novel duality characterization of robust decision rules (Theorem 3.1):** The reduction of the minimax problem over policies to a finite-dimensional concave maximization over dual variables, with pointwise computation of the worst-case belief q*(v), provides a concrete and implementable recipe. This is a non-trivial structural result that cleanly separates the global (dual multiplier optimization) from the local (pointwise best-response) computation.

- **Decision calibration collapse result (Theorems 4.1, 4.2):** The finding that the minimax-optimal policy reduces to plug-in best response under decision calibration—and that this is stable under enrichment of H—is the paper's most significant insight. It upgrades decision calibration's previously known swap-regret guarantees to minimax optimality over *all* forecast-based policies, not just swap-type policies. The proof insight is clean: decision calibration makes the expected utility of a_BR invariant to the adversary's choice of q ∈ Q, so the adversary cannot degrade its performance.

- **Self-orthogonality from standard training (Proposition 4.4):** Identifying that first-order stationarity of squared-loss training with a linear head yields a free H-calibration guarantee is practically useful—it means practitioners can apply the robust rule without any post-hoc recalibration, leveraging structure that already exists in standard pipelines.

- **Simultaneous optimality across decision problems (Corollary 4.3):** The result that a single forecaster satisfying combined decision-calibration tests yields plug-in optimality for *all* downstream decision makers simultaneously is a strong practical upshot that goes beyond what prior work on decision calibration established.

## Weaknesses

- **The linearity assumption on utility (Assumption 2.1) is a fundamental restriction.** The proof of Theorem 4.1 critically uses linearity to establish invariance of a_BR's utility under adversarial tilting (Eq. 9: E[u(a, q(f(X)))] = u(a, E[q(f(X))])). For concave (risk-averse) utilities, Jensen's inequality breaks this equality, and the adversary could exploit curvature within decision regions R_a to degrade plug-in performance while respecting calibration constraints. The paper acknowledges this in Section 6 and cites linearization over bases (Gopalan et al., 2024b; Lu et al., 2025), but notes these bases "are not always low dimensional enough to be practical." This gap matters because the introduction explicitly motivates the work for healthcare and finance—domains where risk aversion is the norm. The paper should more prominently flag this as a scope limitation of the core "trustworthiness" claim, not just a future direction.

- **The empirical evaluation does not test the paper's central theoretical result.** The most important contribution—the collapse of the robust policy to plug-in best response under decision calibration—is never empirically demonstrated. All experiments use the self-orthogonality H = {h(v) = v}, which falls short of decision calibration. An experiment post-processing a forecaster to satisfy decision calibration and then verifying that the robust rule matches plug-in best response would directly validate Theorem 4.1 and is an obvious missing piece.

- **Experiments use only constructed adversaries, not real distribution shifts.** The adversarial evaluations in Table 1 are distributions mathematically derived to satisfy H-calibration constraints, which circularly validates the duality theory but does not demonstrate that the robustness helps against *naturally occurring* distribution shifts (e.g., temporal drift on Bike Sharing across years). Whether calibration-preserving adversaries align with real-world failure modes is the key empirical question left unanswered.

- **Limited experimental scope and missing baselines.** Only two 1D regression datasets with 3-action decision problems and a single MLP architecture are tested. There is no evaluation in the high-dimensional multiclass setting that motivates the paper (the intractability of full calibration in high d), and no comparison to other robust decision-making methods (e.g., Wasserstein DRO, conformal prediction-based decisions). Without such baselines, it is unclear whether the calibration-specific robustness structure offers advantages over generic distributional robustness.

- **Gap between population-level self-orthogonality and finite-sample practice.** Proposition 4.4 assumes the model reaches a population-level first-order stationary point. In practice, training involves finite data, early stopping, and approximate SGD, so the self-orthogonality moments hold only approximately. While Appendix B provides epsilon-slack theory, no experiment or analysis connects the epsilon to sample size, network architecture, or training duration, leaving it unclear how large the violation might be in practice and how it scales.

## Nice-to-Haves

- Evaluation under authentic temporal or domain shifts (e.g., testing Bike Sharing on held-out years) to probe whether calibration-preserving adversaries correlate with real distributional changes.
- Comparison against standard DRO or conformal prediction-based decision baselines to contextualize the calibration-specific robustness advantage.
- An experiment in a multiclass (d > 1) setting to demonstrate the framework in the regime where full calibration is actually intractable and the paper's partial calibration approach is most needed.
- Error bars or confidence intervals on the utility numbers in Table 1, particularly since some differences (e.g., 0.474 vs. 0.463) are small.
- Computational overhead analysis comparing the latency of solving the dual + pointwise minimization versus a single forward pass.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: "The claim that decision calibration is 'tractable' conflates verifying vs. learning the guarantee."** The paper cites Noarov et al. (2023) for achieving decision calibration in practice. Whether enforcing it during training is expensive is a reasonable concern, but the paper explicitly references prior work showing how to accomplish this. This is more of a nuance than a flaw in the paper's claims.

- **Weakness: "The dual problem's convergence rate and conditioning in high dimensions are not discussed."** The paper describes a standard concave maximization with projected subgradient methods and provides the structure of the dual. For a theory paper, demanding a full convergence rate analysis is scope creep beyond the stated contributions. Moved to nice-to-have territory.

- **Weakness: "The pointwise computation of q*(v) may be prohibitive in latency-sensitive systems."** This is speculative without evidence of actual runtime. The pointwise problem is a small convex program over [0,1]^d, which for moderate d and finite A is fast. Without measurements, this is not a demonstrated weakness.

- **Weakness: "Missing discussion of societal impact / potential under-provisioning in healthcare."** This is a generic responsibility demand not standard for a theoretical ML paper at ICLR. The paper studies decision rules given forecasts; it does not deploy systems.

- **Weakness: "Figure 2 is only schematic; a plot with actual data would be more convincing."** This is a formatting/visual nitpick. The schematic effectively communicates the conceptual point.

- **Weakness: "No comparison to top-label calibration or weaker variants of decision calibration."** This is asking the paper to expand its scope. The paper already studies the hierarchy via H-classes and identifies the sharp transition. Requesting analysis of every intermediate notion is scope creep.

## Novel Insights

The most striking insight emerging from the synthesis of these reviews is that the paper contains an **internal tension between its theoretical center of gravity and its empirical evaluation**. The theory's crown jewel—the decision calibration collapse—receives zero experimental validation, while the experiments exclusively test the weakest instantiation (self-orthogonality under 1D regression). This means the paper empirically demonstrates only the least surprising prediction of its framework (robustness helps under adversarial shifts consistent with weak guarantees) while leaving its most surprising and significant prediction (the sharp transition) as a purely theoretical claim. A single experiment with a decision-calibrated forecaster would have closed this gap and dramatically strengthened the paper. Additionally, the self-orthogonality result creates an interesting practical asymmetry: it gives practitioners robustness "for free" from standard training, but this free robustness is precisely the regime where it is least needed (1D, low-stakes regression), while the high-dimensional, high-stakes settings that motivate the paper require explicit decision-calibration enforcement—exactly the regime left untested.

## Suggestions

- Add one experiment post-processing a forecaster to satisfy decision calibration (e.g., via the batch multicalibration algorithm from prior work) on a multiclass task, and show that the robust rule matches plug-in best response, directly validating Theorem 4.1.
- Evaluate under a real temporal or covariate shift (e.g., train Bike Sharing on 2011–2012, test on 2012–2013 held-out data) to establish whether the robustness helps beyond mathematically constructed adversaries.
- Include at least one baseline from the DRO or conformal prediction literature to contextualize whether the calibration-structured ambiguity set offers practical advantages over generic distributional uncertainty sets.

---

