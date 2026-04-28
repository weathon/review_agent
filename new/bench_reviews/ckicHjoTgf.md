Now let me do calibration searches.Now I have sufficient calibration material. Let me write the final review.

---

## Summary

This paper proposes a threat model for automated AI accelerator generation platforms (e.g., Gemmini), in which a malicious platform developer embeds adversarial code across three components: (1) the exploration/DSE unit, to identify sensitive model parameters via the Cross-layer Sensitive Filter Exploration (C-SFE) algorithm; (2) the hardware generation flow, to insert a hardware Trojan (HT) into the accelerator RTL; and (3) the software stack, to inject covert attack parameters into unused bits of RoCC instruction fields. The attack is validated on real FPGA hardware (Xilinx U50, Gemmini + Rocket Core), achieving >97% targeted misclassification across ResNet-18, VGG-16, and YOLOv8m-cls with negligible hardware overhead (0.34% LUT increase).

---

## Strengths

- **System-level threat model with real hardware validation**: The paper provides end-to-end implementation on Gemmini/RISC-V Rocket running Linux on a Xilinx U50 FPGA, including actual synthesis results (Table 2), FPGA layout (Fig. 6), and quantized-model inference. This is substantially stronger evidence than purely software-simulated attacks.
- **Covert channel via unused RoCC instruction bits**: The observation that fields like `pool_size` and `kernel_dim` in the `LOOP_CONV_WS_CONFIG_x` commands use only 3–4 bits out of a 16-bit field, and the identification of 15 such usable fields across seven RoCC commands, is a specific, non-obvious implementation insight grounded in the actual Gemmini ISA (Section 3.3).
- **C-SFE algorithm with K-SIM**: The cross-layer kernel selection heuristic, which infers kernel positions in earlier layers from a chosen kernel in the current layer by following feature-map channel paths, directly reduces the multi-layer search to a single-layer traversal (Section 3.4, Fig. 4). This is technically concrete and addresses a real engineering challenge (fitting attack parameters within the 15 available instruction fields).
- **Dual synthesis strategy for stealth**: Table 2 shows that applying an area-optimization synthesis strategy to the malicious design yields fewer total LUTs than the clean default-strategy design (−3.90%), providing a concrete mechanism to evade resource-differential-based detection.

---

## Weaknesses

### Fatal
*None.*

### Major

- **Unmotivated attack complexity under the stated threat model**: Section 3.1 establishes that "the adversary is the developer of the AI accelerator platform," who controls the hardware generation pipeline, the software stack (compiled to `.so` or Python wheels), and has access to the user's model and a calibration dataset. Under this threat model, the attacker has direct write access to the weight tensors that the `.so` library stores and serves at inference time. A direct weight-modification attack via the compromised software stack would be simpler, equally powerful, and not require a hardware Trojan at all. The paper never explicitly argues why the transient, HT-based weight modification is preferable to direct, persistent weight modification — the HT's key advantage (no static trace in the stored weights) is implicit and never stated. Without this argument, the complex HT + covert-channel mechanism appears unmotivated, and the paper's core design choices lose their justification. This is not merely a presentation gap; it leaves the threat model inconsistent with the proposed mechanism.

- **No evaluation against hardware Trojan or model-level detection methods**: The paper's headline stealth claim rests on two metrics — 0.34% LUT overhead and "near-original performance when untriggered." Neither is validated against any detection baseline. The reference list cites MERO (Chakraborty et al., 2009), gate-level information-flow tracking (Hu et al., 2016), and model-level bit-flip defenses (He et al., 2020; Guo et al., 2021), but none of these are used in experiments. Whether 0.34% LUT variation is detectable by state-of-the-art HT tools, or whether the covert instruction pattern is anomaly-detectable, is entirely unaddressed. For a paper whose central security claim is concealment, this is a critical evidentiary gap.

### Minor

- **Unexplained discrepancy between Table 1 and Figure 5 (ResNet-18, honeycomb)**: Table 1 reports a 99.2% targeted classification rate with 12 total attacked parameters (4/4/4 across three layers). Figure 5's own data table, described as "targeting the same kernel positions as detailed in Table 1," shows 95.8% at 12 parameters — a 3.4-point gap. The text above Figure 5 states "the minimum bit flips needed to achieve the highest classification rate… targeting the same kernel positions as detailed in Table 1," which implies they should converge. No explanation (different random seed, different optimization run, different test split) is offered. The abstract repeats the 99.2% figure as a headline result, and the inconsistency leaves this unreliable without clarification.

- **Single target class per model**: Each model is attacked toward one target category (panpipe for VGG-16, honeycomb for ResNet-18 and YOLOv8). Figure 5 shows that attack performance is highly sensitive to the number of parameters around a "threshold" (a 24% drop from 8 to 7 parameters, a 42% drop from 7 to 6). Whether this threshold is favorable or unfavorable depends on the specific class–model pair. The generality of C-SFE across different target classes cannot be assessed from single-class experiments.

- **Constrained T-BFA comparison**: Section 4.3 explicitly restricts T-BFA to "the same three layers as C-SFE." The legitimate comparison question — whether unrestricted T-BFA (searching all intermediate layers) can also achieve concentrated kernel targeting within the 15-field limit — is never answered. Since the paper's design constraint (intermediate layers only) is shared by both methods in the comparison, the comparison demonstrates that C-SFE concentrates kernels better under the shared constraint, but does not show that the constraint itself is necessary for T-BFA to exceed 15 fields.

- **K-SIM efficiency not ablated**: The claim that K-SIM "eliminates the overhead of repeated exploration across multiple layers" (Section 3.4) is stated qualitatively. No complexity analysis or controlled ablation comparing C-SFE with K-SIM vs. naive sequential layer search is provided, making it impossible to quantify the exploration time benefit.

### Trivial

- **"Grey-box" label mismatch**: The paper calls the attack a "grey-box" attack in Section 3.1, but the adversary controls the entire platform (hardware generation, software stack, DSE) and has the user's full model. The "grey-box" description applies to the user's perspective (opaque platform), not the adversary's capability, which is effectively white-box with respect to all platform components. Clarifying this distinction would prevent reader confusion.

---

## Nice-to-Haves

- A direct comparison between the proposed HT-based attack and a pure software-stack attack (direct `.so` weight modification) in terms of detectability would significantly strengthen the threat model motivation.
- An ablation studying attack performance across ≥5 different target categories per model would validate generality.
- K-SIM complexity analysis (or at least a wall-clock timing comparison between K-SIM and naive layer-by-layer search) would substantiate the efficiency claim.
- Analysis of C-SFE behavior on multi-branch or depthwise-separable architectures (e.g., MobileNetV3, EfficientNet), where channel correspondence is more complex, would clarify scope.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh critic Issue #2 as stated ("Figure 5 reports 95.8% at 12 parameters"—framing it as a contradiction of the abstract)**: The discrepancy is real (verified: Figure 5 data table row 12 = 95.8%, Table 1 = 99.2%) but the framing that "99.2% is therefore unreliable" overclaims — stochastic algorithms naturally vary across independent runs. Kept as a Minor weakness asking for clarification, not as structural evidence of falsified results.
- **"The 1-to-1 attack variant is described but never demonstrated experimentally"**: Figure 1 labels the attack modes but the body describes N-to-1 throughout. While a complete omission, the paper's contribution is the HT mechanism and C-SFE algorithm, which apply equally to both modes. This is a nice-to-have, not a fatal omission.
- **Missing related work criticism**: Per hard rules, no external sources were confirmed to verify specific missing citations.
- **Reproducibility/hyperparameter concerns**: Per hard rules, undisclosed hyperparameters (GA population size, threshold values) are not penalized.
- **Strength Finder claim — "high attack efficacy with negligible perturbation" as a general strength**: Kept only in the context of the concrete numbers (0.000102% perturbation rate for ResNet-18). Removed generic framing about "demonstrating concealment" without the detection evaluation.

---

## Novel Insights

The paper's most underappreciated contribution is the concrete mapping between bit-flip attack parameter counts and instruction-field capacity: by identifying that Gemmini's seven-command convolution protocol exposes exactly 15 exploitable fields and that each kernel position requires 2 fields while each bit-flip mask uses 1, the paper establishes a hard information-theoretic budget for covert-channel-triggered hardware attacks. This budget framing — attack feasibility as a function of instruction-protocol bandwidth — is a specific and transferable insight for evaluating security margins in any co-processor ISA with overprovisioned field widths.

---

## Suggestions

1. **Explicitly argue the HT stealth advantage over direct `.so` weight modification** — even a one-paragraph treatment (transient vs. persistent weight presence, detectability by memory integrity checkers) would resolve the threat model motivation issue.
2. **Run one detection tool (e.g., MERO or a simple gate-level netlist differential) against the malicious bitstream** to substantiate the stealth claim with evidence rather than assertion.
3. **Report attack performance across 5+ target classes per model**, at minimum noting success rate variance, to support the generality claim.
4. **Clarify the Table 1 vs. Figure 5 discrepancy** in a footnote (e.g., "Table 1 reports the best of three GA runs; Figure 5 reports a fixed-seed ablation").

---

## Score and Decision

**Calibration anchors used:**

| Path | Avg Human Score | Comparison to paper under review |
|---|---|---|
| `NI0RsRuFsW.md` (DNN Trojan evasion) | 4.0, Reject | Similar pattern: hardware/model trojan paper, novel angle, rejected for no adaptive defense evaluation and small experiments — directly analogous to this paper's missing detection evaluation |
| `mFzpBaTLGK.md` (Acoustic backdoor) | 3.5, Reject | Rejected for unrealistic threat model and outdated defenses; paper under review has a more grounded threat model and real hardware, placing it above this anchor |
| `41uZB8bDFh.md` (Q-Misalign, quantization attack) | 6.0, Accept | Accepted for novel attack angle and end-to-end validation; paper under review matches on hardware validation novelty but has more significant gaps (missing detection baseline, unmotivated mechanism, narrow evaluation) |
| `hzu5luG4DC.md` (Contrived threat model) | 3.0, Reject | Rejected for confusing threat model — paper under review's threat model is clearer but shares the weak motivation gap |
| `S5JCqTJyKj.md` (Deferred backdoor) | 3.0, Withdrawn | Rejected for weak threat model and experiments — paper under review is clearly stronger on the implementation side |
| `uDNP1q5aZq.md` (Backdoor poisoning) | 5.5, Reject | Borderline; similar level of experimental scope to paper under review |

**Assessment:** The paper is stronger than the 3.0–3.5 anchors in terms of concrete hardware implementation and specific technical novelty (the RoCC covert channel insight, K-SIM). However, it falls short of the 6.0 Q-Misalign anchor mainly due to: (1) the unresolved threat model motivation gap (a logical problem, not just a presentation one), and (2) the complete absence of detection evaluation for a paper claiming stealth. The closest anchors cluster around 4.0 (NI0RsRuFsW). The paper is above the median anchor of the reject cluster due to its hardware grounding, but the two major weaknesses are real and substantive.

**Axis evaluation:**
- *Originality*: Moderate — the system-level framing is novel; C-SFE is incremental over prior bit-flip work.
- *Importance of research question*: High — accelerator supply-chain security is underexplored.
- *Claims well-supported*: Weak — headline stealth claim is unvalidated; Table 1/Figure 5 inconsistency.
- *Soundness of experiments*: Fair — real hardware implementation is a strength; single-class evaluation and missing detection tests are gaps.
- *Clarity of writing*: Fair — structure is clear but threat model motivation is implicit.
- *Value to community*: Moderate — the RoCC covert-channel analysis and dual-synthesis stealth observation are useful; C-SFE is of limited standalone ML value.

**Final score: 4.0** (positioned at the DNN Trojan evasion paper anchor, reflecting genuine novelty in the hardware-level framing that pushes it slightly above the 3.0–3.5 rejected backdoor papers, but held down by two major unresolved weaknesses that the Q-Misalign anchor (6.0) did not have).

MY FINAL SCORE: <pineapple>4</pineapple>
MY FINAL DECISION: <orange>Reject</orange>