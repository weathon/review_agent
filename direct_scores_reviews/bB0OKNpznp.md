## Summary
Quantum Parameter Adaptation (QPA) is a hybrid quantum-classical framework that uses a Parameterized Quantum Circuit (PQC) coupled with a classical MLP mapping model to generate trainable parameters for PEFT methods (LoRA, DoRA, Prefix-Tuning, Feed-Forward Adapters) applied to LLMs. By decoupling quantum hardware from the inference stage, QPA reduces the trainable parameter count at training time while producing a fully classical deployed model. Experiments on GPT-2 and Gemma-2 (WikiText-2) show QPA reduces trainable parameters to 52.06% (GPT-2) and 16.84% (Gemma-2) of standard LoRA parameter counts, with marginal perplexity improvements of 0.75% and 0.07% respectively.

---

## Strengths

- **First quantum parameter generation applied at LLM scale:** Prior work on quantum parameter generation is limited to models ≤0.28M parameters. QPA targets a 0.52B-parameter lmhead, representing a genuine scaling milestone (~1785× the largest prior target layer) that demonstrates the method is not confined to toy settings.

- **Pragmatic architecture decoupling quantum from inference:** The design choice to restrict quantum circuit execution to the training phase only, leaving inference entirely classical, is well-motivated and addresses a real obstacle to QML deployment. This is a concrete and non-trivial engineering contribution relative to standard QML approaches.

- **Breadth of PEFT integration:** QPA is demonstrated across four distinct PEFT families (LoRA, DoRA, Prefix-Tuning, FFA), including both additive and decomposition-based methods. The generalization across PEFT paradigms is substantiated with results in Figures 2–3 and Table 2, showing consistent parameter reduction.

- **Strong low-parameter-budget regime performance:** Figure 2 shows QPA consistently dominating LoRA and DoRA in the low-parameter regime (e.g., fewer than ~150K parameters for Gemma-2), where standard rank-based PEFT cannot access fine-grained intermediate points. This "between-integer-rank" flexibility is a specific, concrete benefit with empirical backing.

- **Practical qubit count (4–11 qubits):** The batched generation scheme in Section 3.2 reduces qubit requirements to 4–11, which is achievable on near-term quantum hardware and classically simulable at modest memory cost, providing a practical pathway.

---

## Weaknesses

- **No classical hypernetwork ablation — the central missing comparison:** The core claim of the paper is that a quantum circuit-based generator enables efficient parameter compression. However, all trainable parameters consist of two parts: PQC angles θ (N × L, e.g., 9 × 8 = 72 for the largest GPT-2 setting) and MLP mapping parameters **b** (architecture [32, 64, 128, 128, 64, 32, n_mlp], with n_mlp up to 8192). The MLP contains the overwhelming majority of QPA's trainable parameters. The paper never compares against a classical-only hypernetwork (a small MLP taking a learned low-dimensional embedding as input, outputting PEFT parameters) at the same total parameter budget. Without this ablation, it is impossible to isolate whether the performance gains stem from quantum properties of the PQC or simply from the compression architecture built around the MLP. This is the most significant gap and directly undermines the paper's central claim. Reviewers 1, 2, and 3 all flag this independently.

- **Experimental scope restricted to one layer (lmhead) on one dataset (WikiText-2 main text):** Section 4 explicitly states that only the final linear layer is fine-tuned, with all other layers frozen. While the authors justify this as "isolating the effects of QPA," standard PEFT methods are applied to attention projections across all transformer blocks — the setting for which LoRA and DoRA were designed and evaluated in the literature. Fine-tuning only lmhead is a legitimate research design choice, but it means: (a) the baselines are not operating in their native regime, making external comparison impossible; (b) the task reduces to head-level adaptation of WikiText-2 perplexity, not representative of downstream task performance that motivates PEFT research; and (c) generalizability to standard PEFT workflows remains undemonstrated. This severely limits the practical significance of the results.

- **Performance margins are tiny and statistically unvalidated:** The headline results — a 0.75% improvement for GPT-2 and 0.07% for Gemma-2 in perplexity — are single-run results with no standard deviations, confidence intervals, or multi-seed statistics reported. Given the narrow margins, these could easily lie within run-to-run noise. This is especially critical given that the paper's core claim is that QPA achieves "comparable or improved performance" — a claim that cannot be substantiated for sub-percent differences without statistical testing.

- **No training time or compute cost comparison:** QPA requires running a PQC simulation on every forward pass, plus gradient computation via the Jacobian through the quantum circuit. The paper reports no wall-clock training time, FLOPs, or per-step overhead relative to standard LoRA/DoRA. A method paper claiming "efficiency" must show that reduced parameter count translates to practical training efficiency; it is entirely plausible that circuit simulation overhead makes QPA slower than direct LoRA optimization, even for 4–11 qubits.

- **DoRA results are anomalous and may indicate a configuration problem:** Table 2 shows DoRA achieving PPL 5.003 for GPT-2 and 5.504 for Gemma-2 — substantially worse than LoRA (1.595 and 1.418). DoRA was specifically designed to match or exceed LoRA. The dramatically degraded DoRA performance in this unusual lmhead-only setting (and at different parameter counts per rank than LoRA) raises a concern about whether DoRA was correctly configured, which would make the "QPA DoRA" comparison unreliable.

- **Barren plateau analysis relegated to appendix:** The trainability of PQC-based generators with increasing qubit count is a well-known fundamental challenge. The paper mentions gradient variance in Appendix H (unavailable for review) and briefly notes "a slight downward trend observed with increasing L" in Section 4.2 — but this is a core concern for any PQC-based method and warrants more than a passing sentence in the main text, especially given that the paper claims scalability as a contribution.

- **QNN depth L fixed at 8 independent of N in main experiments:** The polylogarithmic compression claim requires L = O(poly(N)). In the main experiments, L is fixed at 8 for all N ∈ {4,...,11}, meaning the circuit is not necessarily in the claimed complexity regime for N = 4 or N = 5. The Section 4.2 analysis with varying L is helpful but uses fixed N, leaving the joint (N, L) scaling regime unexplored.

---

## Nice-to-Haves

- **Extend QPA beyond lmhead to attention projection layers:** Demonstrating QPA on standard attention matrices (Q, K, V, out projections) in at least one representative setting would substantially increase the paper's significance and connect it to mainstream PEFT workflows.

- **Training convergence curves (loss vs. steps/wall-clock time):** Plotting QPA versus LoRA training curves would reveal whether QPA converges in comparable steps and at comparable computational cost, which is essential context for the efficiency narrative.

- **Noise robustness in main text:** Appendix G discusses shot noise and gate errors. Moving at least a summary of this analysis into the main text (e.g., one table showing PPL degradation at realistic noise levels) would strengthen the claim of NISQ-era practicality.

- **Visualization of generated vs. standard PEFT parameter distributions:** A comparison of weight heatmaps or singular value spectra between QPA-generated and standard LoRA matrices would provide insight into what structural properties QPA's generator is learning.

- **Multi-seed validation:** Running each configuration across 3+ seeds and reporting mean ± std is a low-effort addition that would significantly strengthen the credibility of the small performance gains reported.

---

## Removed Points
*These points are flagged for removal; treat them with caution.*

- **"1785× scaling claim is misleading"** (Harsh Critic): The paper explicitly states the comparison is between the target layer size (0.52B) and the largest previously studied target model (0.28M). This is a legitimate and consistent claim about the scope of the problem addressed by quantum parameter generation — the paper is not claiming it directly optimizes 0.52B parameters.

- **"Abstract overpromises broad fine-tuning applicability"**: The abstract does state case studies on GPT-2/Gemma-2, and Section 3.3 of the paper references that PEFT is applied to one layer, with Section 4 as the full explanation. While more upfront disclosure would be ideal, this is a clarity issue rather than a factual misrepresentation.

- **"Fair parameter-count comparison is asymmetric"** (Harsh Critic): The comparison in Figure 2 sweeps across both n_mlp and rank r independently and plots performance vs. total parameter count, which is a standard and fair way to present Pareto-style comparisons. This is not a genuine asymmetry issue.

- **"DoRA designed to beat LoRA so the gap proves misconfiguration"**: DoRA's underperformance here may reflect the unusual lmhead-only fine-tuning setup rather than a configuration error. DoRA's advantage is established on standard multi-layer PEFT benchmarks, not single-head adaptation. This should be noted as an unexplained result but not definitively flagged as a bug.

---

## Novel Insights

The most genuinely interesting conceptual contribution of QPA — partially obscured by the paper's framing — is the "between-integer-rank" capability enabled by the batched quantum generator. Standard LoRA is constrained to integer rank r with discrete parameter jumps; QPA's n_mlp parameter provides a continuous knob that spans parameter budgets between conventional rank steps. This is a specific mechanism by which quantum-circuit-based generation provides a structurally different parameterization space than discrete PEFT methods, independent of any quantum speed-up claim. If this flexibility is the true source of empirical gains (rather than quantum properties of the PQC per se), it would argue for designing a classical hypernetwork with the same continuous-budget property — which is exactly what the missing ablation would reveal.

---

## Suggestions

1. **Add the classical hypernetwork ablation immediately.** Replace the PQC with a small classical MLP or embedding table of the same parameter count as θ, keeping the mapping MLP **G_b** identical. If performance matches QPA, the paper should reframe its contribution as "continuous-budget compression via a hypernetwork generator" rather than quantum advantage. If QPA outperforms, this becomes the paper's strongest result.

2. **Report training time per step for QPA vs. LoRA at matched perplexity.** Even informal timing results (GPU-hours to convergence) would allow readers to assess the practical trade-off.

3. **Include a one- or two-configuration experiment on standard multi-layer PEFT** (e.g., QPA applied to all Q/V projections of GPT-2 with rank 4), even as a preliminary result, to establish scope beyond lmhead-only adaptation.

4. **Move barren plateau and gradient variance analysis to the main text.** A one-paragraph summary with a figure showing gradient variance vs. N and L belongs in Section 4.2, not in an inaccessible appendix, given it directly addresses a core scalability concern.

5. **Run the key comparison configurations (Table 2) with at least 3 seeds** and report mean ± std. The core empirical claims rest on differences of 0.07%–0.75%, which require statistical support to be meaningful.

---

## Evaluation Summary

**Novelty:** Moderate. The idea of applying quantum parameter generation to PEFT at LLM scale is new, and the scale-up from 0.28M to 0.52B-parameter target layers is a meaningful milestone. However, the quantum circuit's specific role is not isolated from the classical MLP, and the novelty of the compression mechanism over a classical hypernetwork is undemonstrated.

**Technical soundness:** Weak-to-moderate. The formalism is consistent and clearly presented, but the central mechanism (quantum vs. classical contribution) is not disentangled. The claim of polylogarithmic compression is only informally stated and not rigorously verified in the experimental regime.

**Empirical support:** Weak. Experiments are limited to single-layer (lmhead) fine-tuning on WikiText-2, with single-run results at sub-percent performance differences. The DoRA anomaly is unexplained. No training cost data is provided.

**Significance:** Limited in current form. The lmhead-only restriction disconnects results from standard PEFT practice. Demonstrating the method on attention layers across a broader benchmark suite is needed to establish practical significance.

**Clarity:** Adequate for the quantum circuit mechanics; the experimental scope limitation is under-emphasized and should be foregrounded earlier.

MY FINAL SCORE: <pineapple>4.2</pineapple>