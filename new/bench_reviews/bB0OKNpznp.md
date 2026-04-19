Now let me read more of the paper and search for calibration anchors simultaneously.Now I have enough to write the review. Let me synthesize everything.Now I have sufficient calibration data. Let me write the final review.

---

## Summary

This paper introduces Quantum Parameter Adaptation (QPA), a method that uses parameterized quantum circuits (PQCs) combined with a classical MLP mapping model to generate parameters for PEFT methods (LoRA, DoRA, Prefix-Tuning, Feed-Forward Adapters) when fine-tuning LLMs. The central claim is that PQC-based generation enables polylogarithmic parameter scaling because N×L quantum angles can govern 2^N measurement probabilities. Experiments on GPT-2 and Gemma-2 show that QPA can match or slightly improve perplexity with fewer trainable parameters than vanilla PEFT baselines on WikiText-2 text generation.

---

## Strengths

- **Batched parameter generation (Section 3.2, Eq. 5–8):** The chunking mechanism that reduces required qubit count from ⌈log₂ m⌉ to ⌈log₂(⌈m/n_mlp⌉)⌉ is a concrete, practical engineering contribution. For m=10⁹ and n_mlp=1024, this drops from 30 to 20 qubits while cutting quantum-state memory by a factor of 1024. This extends the prior quantum parameter generation framework to LLM-scale targets in a non-trivial way.

- **Scale-up over prior quantum parameter generation work (Section 1):** The paper applies quantum parameter generation to Gemma-2 (2B), targeting a layer with 0.52B parameters — roughly 1,785× larger than any previously studied quantum parameter generation setting (0.28M). This is a genuine step toward demonstrating applicability at meaningful scale.

- **Generality across PEFT methods (Figures 2–3, Table 2):** QPA is evaluated on four distinct PEFT methods — LoRA, DoRA, Prefix-Tuning, and Feed-Forward Adapters — demonstrating that the quantum generation scheme is not tied to a single adaptation strategy.

- **Inference decoupling:** Quantum circuits are used only during training; the deployed model is fully classical. This is a principled design choice that removes the need for quantum hardware at inference time, which is a known barrier to conventional QML.

---

## Weaknesses

### Fatal

- **Quantum advantage claim is unsupported by the presented experiments.** The paper's core theoretical argument — that PQCs reduce parameters to a polylogarithmic scale because N×L quantum angles control 2^N Hilbert-space amplitudes — is valid on quantum hardware. However, Section 4 states explicitly: *"The experiment is conducted using quantum circuit simulation via PyTorch and TorchQuantum… the quantum state amplitudes (probabilities) are obtained exactly."* On a classical simulator, computing 2^N measurement probabilities requires O(2^N) classical operations and memory; the exponential cost is fully paid by the CPU/GPU. QPA-as-evaluated is therefore a classical hypernetwork: a parameterized function (the PQC angles) composed with an MLP to produce PEFT weights. There is no computational mechanism here unavailable to a purely classical approach of the same parameter count. The conclusion claims QPA "marks the first example of quantum computing applied to fine-tuning classical LLMs at a practical scale" — but since all computation is classical simulation, this is an overclaim the results cannot support.

- **No classical hypernetwork baseline.** Given the above, the fundamental empirical question is whether the PQC structure contributes anything over an equivalent classical parameterization. The paper never tests this. A minimal ablation would replace the PQC with a small classical MLP or random embedding of identical dimensionality (N×L input → n_ch intermediate points) and check whether performance is distinguishable from QPA. Without this control, every observed result — including any perplexity improvement — could be entirely attributable to the MLP mapping model acting as a regularizer, with the quantum circuit serving no essential role. This is not a missing-dataset problem or scope creep; it is the comparison needed to falsify the paper's central claim.

### Major

- **DoRA baseline appears misconfigured, calling the DoRA+QPA results into question.** Table 2 reports DoRA achieving PPL 5.003 (GPT-2) and 5.504 (Gemma-2), versus LoRA's 1.595 and 1.418 on the same models. DoRA is a strict generalization of LoRA that subsumes LoRA as a special case and should, in any correctly configured run, match or outperform LoRA. A PPL gap of ~3.5 points in the same lmhead setting strongly suggests a DoRA configuration error (e.g., incorrect learning rate, improper magnitude-vector initialization, or inconsistent rank selection relative to the LoRA comparison row). The paper reports QPA-DoRA improving on this broken DoRA baseline (PPL 4.955 / 5.487), but improvements over a misconfigured baseline are not interpretable. Roughly one-quarter of Table 2's entries are thus unreliable.

- **Non-standard and narrow experimental scope limits generalizability.** All experiments fine-tune only the `lmhead` layer. Section 4 states: *"we simplify the PEFT setup by freezing all layers of Gemma-2 and GPT-2, and fine-tuning only the final linear layer."* LoRA and other PEFT methods are canonically applied to the attention projections (Q, K, V, O) distributed throughout all transformer blocks; the lmhead is rarely a target in practice. Fine-tuning only the lmhead is an unusual, arguably degraded setup in which nearly any fixed parameterization can compete with LoRA because the lmhead has relatively unconstrained plasticity. Whether QPA generalizes to the standard PEFT target (attention matrices across all blocks) is entirely unknown from the presented evidence. Evaluation on a single dataset (WikiText-2) and a single task (perplexity) further limits the scope.

### Minor

- **Perplexity gains are marginal and lack statistical testing.** The headline results are a 0.75% perplexity improvement for GPT-2 (1.595 → 1.583) and 0.07% for Gemma-2 (1.418 → 1.417). No standard deviations, confidence intervals, or multi-seed results are reported anywhere. Differences at the third decimal place are well within typical training noise; the single-run results do not provide interpretable evidence that QPA is better than, rather than equivalent to, the baselines.

- **MLP vs. PQC contribution is not disentangled.** The MLP mapping model has a fixed architecture [32, 64, 128, 128, 64, 32, n_mlp] (Table 1). For larger n_mlp values (e.g., 65536 in FFA experiments), the final MLP layer alone (~2M parameters) dwarfs the PQC parameters. The paper does not separately report PQC parameter counts vs. MLP parameter counts, making it impossible to assess whether performance is driven by the MLP, the PQC, or their combination.

- **Barren plateau analysis is limited.** The claim that gradient variance does not exhibit exponential vanishing (Appendix H) is supported only up to 11 qubits. Barren plateau analyses at this scale are not sufficient evidence for the absence of the phenomenon at larger qubit counts; the paper acknowledges "a slight downward trend with increasing L" but does not develop this into a rigorous analysis.

### Trivial

- The abstract's characterization of a 0.07% perplexity difference as a "marginal performance improvement" could mislead a casual reader; the text should clarify this is within experimental uncertainty without multi-seed replication.
- Minor inconsistency: n_mlp = 16258 appears in the DoRA chunk size list — this appears to be a typo for 16384 (a power of 2).

---

## Nice-to-Haves

- Downstream task evaluation (e.g., a generation-quality benchmark, MMLU subset, or commonsense reasoning) with the fine-tuned models would make perplexity improvements interpretable in terms of actual utility.
- A small-scale experiment on real quantum hardware (e.g., IBM Eagle or similar) — even at 4–6 qubits on a tiny model — would be a first validation that the polylogarithmic scaling argument has any practical import beyond classical simulation.
- Applying QPA to the attention matrices across multiple transformer blocks (the standard PEFT setting) rather than only the lmhead would substantially strengthen generalizability claims.

---

## Removed Points

*These points are flagged as removed — treat them with caution.*

- **"The polylogarithmic claim is correct as stated"** (Strength Finder): We retain the observation that batched generation is a genuine contribution; however, the polylogarithmic scaling framing as a *quantum advantage* is removed from strengths because it does not apply in the classical simulation context used for all experiments.
- **"Figure 1 provides clear conceptual diagrams"** (Strength Finder): Dropped as a generic presentation strength without sufficient evidence it contributes to the paper's core claims.
- **"Comprehensive hyperparameter disclosure aids reproducibility"** (Strength Finder): Dropped per the rule against nitpick/reproducibility-focused strengths not tied to core claim.
- **Barren plateau demand for large-scale statistical rigor** (Harsh Critic): Retained as a minor concern but downgraded from structural weakness; the paper does cite the limitation explicitly.
- **Missing related works** (implicitly raised): Removed per hard rule — we cannot verify existence of external works without search tools.

---

## Novel Insights

The core insight that a quantum circuit's Hilbert-space dimensionality could serve as a compression basis for PEFT parameters is conceptually interesting and the batched generation technique (chunking) that dramatically reduces qubit requirements is a genuine engineering innovation. However, the insight is obscured by the absence of the one experiment that would validate or refute it: a matched classical hypernetwork baseline. If such a baseline showed QPA-unique gains, the paper would make a compelling case; if it showed parity, the paper would honestly reframe as a "classical hypernetwork for PEFT" contribution. Either outcome would be more scientifically valuable than the current framing.

---

## Suggestions

1. **Add a matched classical hypernetwork baseline.** Replace the PQC with a small classical MLP or learned embedding of identical parameter count (N×L → n_ch values) and re-run all four PEFT experiments. Report whether QPA-specific gains survive. This single experiment would resolve the central question of the paper.
2. **Fix or re-run the DoRA baseline.** At rank r=4 with proper initialization and learning rate, DoRA should produce perplexity ≤ LoRA's perplexity. Diagnose the misconfiguration and report corrected DoRA baselines.
3. **Report at least 3 seeds for the headline results.** The differences at the third decimal place require variance estimates to interpret.
4. **Run QPA on standard LoRA targets (attention Q/K/V/O projections)** in addition to the lmhead to establish that the method generalizes beyond this non-standard setup.
5. **Separate the MLP and PQC parameter counts** in all tables and figures, allowing readers to assess the relative contribution of each component.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Topic | Score |
|---|---|---|
| vrBVFXwAmi (LLM4QPE) | QML applied to LLMs, strong experiments | 8,8,8,8 |
| gnexAe3kjx (QNF-Net) | Quantum circuits in simulation only, no classical baseline advantage shown | 6,1,8,5 (avg ~5, rejected) |
| YHUOaIbFby | Quantum advantage claimed without proper validation | 3,5,1,3 (avg ~3, withdrawn) |
| dfEuojp0rX | QML on toy dataset, overclaimed practical utility | 1,5,3,3 (avg ~3, rejected) |
| JsJGd0xfgv | Quantum architecture search, all simulation, borderline | 5,5,5,5 (avg 5, withdrawn) |
| s7DkcgpRxL (LoRAM) | PEFT compression with solid experiments | 8,6,3,6,8 (accepted) |

**Positioning:** This paper is closest to gnexAe3kjx (quantum simulation only, novel application domain) and JsJGd0xfgv (quantum circuits for ML, all simulation, borderline). Both were rejected/withdrawn. This paper is worse than gnexAe3kjx in one respect: QNF-Net at least compared against classical baselines and showed superiority; QPA lacks the equivalent classical hypernetwork comparison, which is arguably the most important control experiment. On the other hand, the paper is better than pure toy-dataset quantum papers (dfEuojp0rX, YHUOaIbFby) because it applies to real LLMs at meaningful scale and the engineering contributions (batched generation, 4 PEFT methods) are genuine.

The two fatal issues — unsupported quantum advantage framing and missing classical hypernetwork baseline — are not addressable in rebuttal (they require new experiments), and they directly undermine the paper's framing. The DoRA misconfiguration further damages trust in a substantial portion of results. The marginal perplexity differences without statistical testing mean even the positive results are hard to interpret.

**Score: 3.0**

This places it slightly below the gnexAe3kjx cluster (~5) due to the missing classical baseline (the single most important missing experiment for any paper claiming quantum advantage), and aligns with the YHUOaIbFby/dfEuojp0rX cluster of rejected quantum papers with unsubstantiated advantage claims. The paper does show more engineering substance than those (real LLMs, batched generation), preventing a score of 1-2, but the fatal issues place it solidly below the acceptance threshold.

---

**Originality:** Low-to-moderate. The application of quantum parameter generation to LLMs is new, but the method is a direct extension of prior work (Liu et al., 2024b) to a new domain using standard PEFT methods. The batched generation is the most original technical element.

**Importance of research question:** Moderate. If quantum hardware genuinely offered the claimed polylogarithmic compression, it would matter. In the current simulation-only form, the question reduces to "can a particular parameterization of a hypernetwork compress PEFT parameters?" — a narrower and less impactful question.

**Support for claims:** Poor. The core quantum advantage claim is unsupported by the experimental setup. The empirical performance differences are marginal and lack statistical validation.

**Soundness of experiments:** Poor. Non-standard setup (lmhead only), suspicious DoRA baselines, no multi-seed evaluation, and missing the critical classical control condition.

**Clarity of writing:** Adequate. The method is explained clearly, but the distinction between what is demonstrated in simulation vs. what requires quantum hardware is not adequately flagged in the main text.

**Value to the research community:** Low in current form. The batched generation engineering trick has potential value, but it is buried under a quantum advantage framing that the results cannot support.

MY FINAL SCORE: <pineapple>3.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>