Now I have a complete picture of the paper. Let me synthesize the reviews carefully.

---

## Summary
STD-Former is a dual-branch video action recognition architecture that processes spatiotemporal and temporal features in parallel. The spatiotemporal branch uses a Parallel Transformer Module (PTM) combining MHA, FFN, and 2D convolutions; the temporal branch uses a Cross Transformer Module (CTM) with cross-attention queries from PTM. A Spatio-Temporal Diffusion Module (STDM) feeds temporal branch features back into the spatiotemporal branch via lightweight convolutions, and a Salient Motion Excitation Module (SMEM) correlates adjacent-frame features for fine-grained motion capture. Experiments on SSV1 and SSV2 show competitive results, with STD-Former outperforming UniFormerV2-B on SSV1 (+0.5%) but trailing it on SSV2 (−0.3%), both using CLIP-400M pretraining.

---

## Strengths

- **State-of-the-art on SSV1 with the same pretraining regime as the best baseline.** STD-Former achieves 57.3% Top-1 on SSV1 vs. UniFormerV2-B's 56.8%, using identical CLIP-400M pretraining and input configuration (16×3×1), making the gain directly attributable to architecture rather than pretraining advantage.
- **Informative ablation structure.** Tables 2–4 isolate PTM, STDM, and SMEM contributions individually and test design choices (2D vs. 3D convolution placement, fusion strategy in SMEM), providing more ablation granularity than many competing papers that only report full model comparisons.
- **SMEM is the most technically differentiated module.** The correlation-based adjacent-frame excitation (element-wise multiplication over reshaped temporal features at t, t+1, t+2) draws a clear conceptual connection to motion excitation and provides a concrete lightweight mechanism for fine-grained action capture.

---

## Weaknesses

### Fatal
*(None identified — the paper is not fatally flawed, but has several overlapping major issues that collectively weaken the case for acceptance.)*

### Major

- **Misleading "Diffusion" terminology undermines the paper's central branding.** STDM (Section 3.4) consists of three convolutions (1×3×3, 3×1×1, 1×1×1), batch normalization, and ReLU. No diffusion equation, stochastic process, noise schedule, or connection to physical or probabilistic diffusion is provided. The paper claims inspiration from "the advantage of the diffusion principle for capturing long-distance relevant information" but never cites, formalizes, or justifies this claim. The terminology actively misleads readers about the module's mechanism. Either a rigorous theoretical link to a diffusion formalism must be established, or the module should be renamed (e.g., "Temporal Feedback Convolution Module") and the motivation reframed around inter-branch feature propagation. As written, this looks like buzzword adoption rather than principled design.

- **CLIP-400M pretraining contribution is not disentangled from the architecture's contribution.** The ablation (Table 2) is conducted entirely under CLIP-400M pretraining, which is the most powerful pretraining regime in Table 1. The individual module gains in the ablation are 0.2–0.4% Top-1. Without a baseline trained from ImageNet or K400, it is impossible to know whether these marginal increments are real architectural effects or noise attributable to training variance under a very strong pretrained initialisation. No repeated runs or variance estimates are reported. This is a significant confound that undermines the ablation's evidentiary value.

- **No computational efficiency analysis.** A dual-branch architecture with 12 PTMs, 12 CTMs, multiple STDMs and SMEMs, all built on a CLIP-400M backbone, carries substantial computational overhead. Without FLOPs, parameter counts, or throughput (FPS) figures, the marginal accuracy gains over UniFormerV2-B on SSV1 (+0.5%) cannot be evaluated in context. If STD-Former requires substantially more compute, the net value proposition is unclear.

- **Evaluation is restricted to SSV1/SSV2.** While these are the standard temporal reasoning benchmarks, the paper claims STD-Former is a general improvement in spatiotemporal representation. Testing exclusively on two temporally-biased datasets of the same family leaves the question of generalization entirely open. Results on at least one appearance-dominated dataset (e.g., Kinetics-400) would allow the community to understand whether the design choices improve temporal reasoning specifically or video recognition broadly.

### Minor

- **Abstract and conclusion overclaim superiority.** The abstract states STD-Former "more accurately identify the fine-grained action and has favorable robustness than the current state-of-the-art action recognition models." However, STD-Former is 0.3% below UniFormerV2-B on SSV2 (the larger and more widely-used benchmark). The claim should be qualified.

- **STDM injection mechanism is not formally specified.** Section 3.4 states the module "diffuses [features] to the spatiotemporal branch," but does not provide the operation: is the STDM output added residually to PTM outputs? Concatenated? At which layers? The informal phrase "feeds back the feature" is insufficient for reproducibility.

- **Asymmetric PTM fusion is unexplained.** Equation (1) is y = y₁ + αy₂ + βy₃, where y₁ is the MHA output without a learnable weight, while y₂ (FFN) and y₃ (2D Conv) have learnable scalars α and β. Why is MHA treated as the fixed anchor while other branches are reweighted? This design choice has no stated justification and is not ablated.

- **CTM depth notation is inconsistent.** The text says the temporal branch "is mainly composed of twelve cross transformer modules (CTM) (where m+n=11 in Figure 1)," but neither m nor n is ever defined, and the claimed count (twelve) conflicts with the expression (11). This should be clarified.

- **"Conventional transformer module" baseline undefined.** In Section 4.4, the ablation baseline replaces PTM with "a conventional transformer module" without specifying which module (standard ViT block? TimeSformer block?). This makes the baseline comparison ambiguous.

### Tiny

- **Duplicate citation in Related Work.** "Lee et al. (2023) constructed a cross-spatiotemporal attention module and proposed a new network based on transformer to enhance video spatiotemporal comprehension" appears twice in the same paragraph in Section 2.2.

- **PTM description of 2D Conv as "temporal feature" extraction is potentially misleading.** The paper says the 2D convolutional layer "extracts temporal feature from adjacent frames," but a standard 2D conv applied to a spatial frame cannot span across temporal dimension unless the feature tensor is reshaped. The exact feature shape entering the 2D Conv branch should be specified to clarify how temporal information is accessed.

---

## Nice-to-Haves

- **Visualization of attention maps.** Showing where the model attends (vs. UniFormerV2-B) on background-heavy SSV clips would substantiate the claim of improved background suppression, which is currently asserted but not demonstrated visually.
- **Sensitivity analysis for α and β.** The learnable weights in PTM's fusion (Eq. 1) are presented as a design feature, but their converged values and sensitivity are not reported. A brief analysis would confirm these parameters are learning meaningful weightings rather than collapsing to trivial values.
- **Kinetics-400 results**, as a nice-to-have generalization check beyond temporally-biased datasets.
- **Fusion baseline in STDM ablation.** Comparing STDM against a simple addition or concatenation of temporal features into the spatiotemporal branch would better isolate whether the specific convolutional propagation in STDM matters, or whether any feature injection strategy would suffice.
- **Temporal sensitivity analysis.** Testing performance under different frame sampling rates (e.g., 8 vs. 16 vs. 32 frames) would validate whether STDM's claimed long-term dependency modeling actually scales with temporal input length.

---

## Removed Points
*These points are flagged to be removed — treat them with caution.*

- **Statistical significance testing demanded (Harsh Critic).** The critic demands confidence intervals and multiple-run statistics for the SSV1/SSV2 benchmarks. Single-run evaluation is the universal norm in video action recognition (as in virtually all SOTA papers in Table 1), making this requirement non-standard. **Removed** as scope creep from community norms.

- **Comparison unfairness against models with missing SSV1 numbers (Harsh Critic).** The critic argues that AIM-B/16 or MTV-B "would score higher than 57.3% on SSV1 if those numbers existed." This is speculative; the paper cannot be faulted for missing results in baseline papers it did not produce. **Removed** — the comparison uses what the baselines reported.

- **Dual-branch design is "well-trodden territory" (Positive Reviewer / Harsh Critic).** Broad characterizations of novelty as "incremental" without engagement with how this specific combination differs from SlowFast (which uses different-rate RGB streams, not transformer branches) or MTV (which uses heterogeneous encoder designs) are too generic to be actionable. This is **weakened** to the major point above about ablation quality.

- **Demand for theoretical proofs for diffusion connection (Harsh Critic).** While renaming/reframing the STDM is warranted (kept as major weakness), demanding a formal diffusion-theoretic derivation for what is essentially an empirical systems paper is an unreasonable rigor bar. The core issue is terminological honesty, not missing theory.

- **General "add more baselines" request (Positive Reviewer).** The request to add "most recent SOTA methods from 2023–2024" is a generic one-size-fits-all weakness and not specific to this paper's claims. **Removed.**

---

## Novel Insights

None beyond the paper's own contributions. The three reviewers collectively confirm that the most interesting design decision is the cross-branch query mechanism in CTM (PTM provides queries, CTM provides K/V), which differs from standard cross-attention between branches and warrants more careful exposition. The SMEM's element-wise correlation of three adjacent-frame features is a lightweight motion proxy that could generalise, but no experiment probes this. The reviewers converge on the finding that the "diffusion" framing is the paper's central conceptual liability: stripping it away reveals a competent but incrementally novel engineering contribution that achieves a modest gain on one of two tested benchmarks.

---

## Suggestions

1. **Rename STDM or formally ground it.** Either rename to "Temporal Feedback Propagation Module" (or similar) and rewrite Section 3.4 to motivate it as inter-branch feature injection, or provide a concrete mathematical analogy to a diffusion process (e.g., iterative smoothing, heat kernel, message passing on graphs) with supporting analysis.
2. **Run the ablation with a weaker pretraining (ImageNet or K400) baseline.** This would separate the architectural contribution from CLIP-400M's dominance and make the 0.2–0.4% module gains interpretable.
3. **Add efficiency table.** Report GFLOPs, parameter count, and inference throughput for STD-Former alongside UniFormerV2-B, TimeSformer-L, and AIM-B/16 to contextualise the accuracy trade-off.
4. **Specify STDM injection operation precisely** (which layer output is transformed, what operation connects STDM output to PTM features — additive residual? gated?) with a formal equation.
5. **Specify the ablation baseline.** Name the exact transformer block used when PTM is replaced (e.g., "standard ViT attention + FFN block identical to the setup in UniFormerV2").
6. **Fix duplicate Lee et al. (2023) citation** in Related Work.
7. **Clarify the m+n=11 notation** — define m and n explicitly and reconcile with the "twelve CTMs" description.
8. **Qualify abstract claims** to accurately reflect that STD-Former leads on SSV1 but not on SSV2 among CLIP-400M-pretrained models.

---

**Evaluative summary:**
- **Novelty:** Low-to-moderate. The dual-branch cross-attention design is incremental over existing work (UniFormer, SlowFast, MTV), and the "diffusion" module is a standard conv block in misleading clothing. SMEM has some genuine differentiation.
- **Technical soundness:** Moderate. The described components are functional, but key architectural details (STDM injection, PTM temporal-dimension handling, CTM depth alignment) are underspecified. The asymmetric PTM fusion is unexplained.
- **Empirical support:** Weak-to-moderate. Results on only two closely-related datasets, no efficiency metrics, no pretraining ablation, and module improvements of 0.2–0.4% without variance estimates make it hard to draw strong conclusions.
- **Significance:** Limited. A 0.5% gain on SSV1 over the nearest comparable baseline, with a regression on SSV2, is unlikely to shift practice in the field. The dual-branch overhead is uncharacterized.
- **Clarity:** Below par for ICLR. Several critical architectural details are absent or inconsistent, the "diffusion" framing actively obscures what the model does, and the related-work section contains errors.