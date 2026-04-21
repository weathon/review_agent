Now I have enough information to write the final review. Let me carefully verify the key claims from the harsh critic against the paper text before proceeding.

**Verification of Harsh Critic Claims:**

1. **"Threat model trivializes contribution"**: Section 3.1 clearly states the adversary is the platform developer with access to model parameters, calibration data, and control over the entire generation flow. The "grey-box" label is used, but the attacker has far more capability than typical grey-box models. This is a valid concern, though the paper *does* identify specific, concrete attack vectors (unused instruction bits) that go beyond "if you control the compiler, compile malicious code."

2. **"Unfair T-BFA comparison"**: Section 4.3 explicitly says "we restricted the layers explored by T-BFA to match those of C-SFE." This does advantage C-SFE. However, the paper's purpose is to show C-SFE achieves the same attack effectiveness with *fewer hardware resources* (fewer instruction fields needed), which is the relevant metric for HT design. The restriction is explained by the 15-field instruction limit. Still, the comparison should acknowledge that T-BFA exploring all layers might find more effective parameters, and the comparison only shows C-SFE is more HT-compatible, not more effective overall.

3. **"Stealthiness claims unsupported"**: The paper claims "near-original performance as in uncompromised designs" (Abstract) and "concealment" (Conclusion). Table 1 shows clean accuracy is maintained when HT is dormant, but when triggered, accuracy drops to 0.1-0.2%. The "stealthiness" claim rests on: (a) dormant HT (trivial property of any HT), (b) small area overhead (genuine, 0.34% LUT), and (c) targeting intermediate layers (claimed to avoid attention). The absence of defense evaluation against HT detection methods is a real gap.

4. **"Missing hyperparameters"**: Algorithm 1 mentions threshold, population size, β, and x without specifying values. This is a minor reproducibility concern.

5. **"K-SIM limitations with skip connections"**: The paper actually addresses this in Section 3.4, showing that for residual structures, K-SIM restricts the kernel index selection. It does work for ResNet-18, though less flexibly.

Now let me write the final consolidated review.

---

## Summary

This paper proposes a threat model for automated AI accelerator generation platforms, where the platform developer (as adversary) leverages the exploration unit to identify sensitive model parameters, embeds a hardware Trojan (HT) in the generated accelerator, and triggers it via malicious information concealed in unused instruction bit-fields. The paper introduces the Cross-layer Sensitive Filter Exploration (C-SFE) algorithm that targets intermediate convolutional layers with kernel-level regularity to minimize HT resources, and validates the full attack chain on the Gemmini accelerator platform running on a Xilinx U50 FPGA, achieving 97.3–99.2% targeted misclassification rates across VGG-16, ResNet-18, and YOLOv8m-cls with minimal parameter perturbation (as low as 9 parameters) and 0.34% LUT overhead.

## Strengths

- **End-to-end demonstration on real FPGA hardware**: The paper implements a complete attack chain—from exploration algorithm through HT insertion to instruction-level triggering—on a Gemmini/Rocket Core SoC running on a Xilinx U50 Alveo card (Section 4.1). This goes beyond simulation-only results, which is rare and valuable in hardware security literature.

- **Novel and concrete attack vector using unused instruction bit-fields**: The observation that RoCC instruction fields like `pool_size` and `kernel_dim` reserve 16 bits but only use a few for typical workloads (e.g., ResNet-18 kernels need at most 3 bits for dimension 7) creates a specific, novel attack surface. The paper identifies 15 usable fields across 7 commands (Section 3.3) and demonstrates how malicious trigger data can be embedded without changing instruction counts or formats.

- **C-SFE's kernel-level regularity is well-motivated for HT design**: By constraining the attack to one kernel per intermediate layer (avoiding first and final layers), C-SFE directly addresses the hardware design constraint that HTs should minimize resource usage and detection risk. The K-SIM method (Section 3.4) provides an efficient heuristic for cross-layer kernel selection, and Figure 7 concretely shows C-SFE needs only 6 instruction fields versus T-BFA's 23 for the same layer.

## Weaknesses

### Fatal
None.

### Major

- **Overclaimed novelty of the threat model — the strong attacker assumption substantially reduces the conceptual contribution**: The adversary is the platform developer with white-box model access, full control over the exploration unit, the software stack, and hardware generation (Section 3.1). Under this model, inserting malicious behavior into any part of the design pipeline is trivially possible—the real question is *how* to do so in a resource-efficient, stealthy way. The paper's genuine novelty lies in the specific attack vector (unused instruction bits) and C-SFE's HT-compatible parameter selection, not in the threat model itself. Calling this a "novel threat model" (Abstract, Conclusion) and labeling it "grey-box" overclaims the conceptual contribution: the attacker has far more capability than a typical grey-box setting implies. The paper would be stronger if it acknowledged this and reframed its contribution around the concrete attack mechanisms rather than the threat model framing.

- **Unfair comparison with T-BFA weakens the claimed advantage of C-SFE**: Section 4.3 restricts T-BFA "to match the layers explored by C-SFE," then concludes C-SFE is superior because T-BFA requires 23 fields for one layer (exceeding the 15-field limit) while C-SFE needs only 6. However, T-BFA was designed to explore *all* layers, and restricting it to intermediate layers removes its ability to find scattered sensitive parameters across the full network. The comparison demonstrates C-SFE's HT compatibility within the same layer set, but it does not show C-SFE is a better attack algorithm overall—it shows C-SFE is *more suitable for HT implementation* under a specific instruction field budget, which is a narrower claim than the paper suggests.

- **Stealthiness claims are overstated without defense evaluation**: The paper repeatedly claims the attack is "stealthy" (Abstract, Section 3.3, Conclusion) and has "near-original performance." When dormant, the HT indeed preserves accuracy and incurs only 0.34% LUT overhead (Table 2)—genuine stealth properties. However, "near-original performance as in uncompromised designs" (Abstract) is misleading because triggered accuracy drops to 0.1–0.2% (Table 1). More importantly, the paper provides no evaluation against any hardware Trojan detection method (e.g., side-channel analysis, path delay testing, or weight integrity verification). For a security paper asserting stealthiness, the absence of defense evaluation leaves the practical threat level uncertain.

### Minor

- **Generality claim is supported by only one platform**: The assertion that the approach is "broadly applicable to any similar automation platform" (Section 3.2) rests entirely on Gemmini, whose specific RoCC instruction format provides the 15 unused fields that make the attack possible. Whether other platforms (e.g., NVDLA, which uses MMIO instead of RoCC) provide comparable instruction redundancy is not analyzed. This limits the generality claim.

- **Missing sensitivity analysis for calibration set size**: The attack uses 50 randomly selected validation images (Section 4.1) with no analysis of how attack effectiveness varies with this parameter, which affects the practicality of the attack under different data access constraints.

- **The "grey-box" designation is inaccurate**: With full model parameter access, platform source code control, and validation data access, the attacker has essentially white-box access to the platform. The "grey-box" label refers only to the user's perspective (the platform internals are not public), which describes the trust boundary but not the attacker's information level.

### Trivial
None.

## Nice-to-Haves

- Evaluation against at least one hardware Trojan detection method would substantiate stealthiness claims and significantly strengthen the security contribution.
- Demonstration on a second accelerator platform (e.g., NVDLA) would substantiate the generality claim.
- Analysis of 1-to-1 targeted attacks (misclassifying specific classes rather than collapsing all inputs to one class) would address practical threat scenarios beyond the demonstrated N-to-1 attacks.
- Discussion of potential defenses (e.g., instruction field auditing, weight integrity verification, platform attestation) would advance the paper from "identifying a problem" toward "advancing toward solutions," as the Conclusion itself notes the "necessity for protection."

## Removed Points

These points are flagged to be removed, treat them with slight caution:

- **"The threat model trivializes the contribution completely"** — While the threat model is strong, the paper's specific contributions (unused instruction bits, C-SFE, FPGA implementation) go beyond "controlling the compiler means you can compile malicious code." The attack vector is concrete and non-obvious. Kept as a Major weakness but softened from Fatal.

- **"K-SIM breaks down for architectures with skip connections"** — The paper actually addresses this in Section 3.4, explaining how residual structures constrain kernel index selection. This is a limitation, not a failure.

- **"Missing hyperparameters for reproducibility"** — Algorithm 1 leaves some parameters unspecified (threshold, β, population size), but the paper provides a GitHub link and these are implementation details commonly omitted in security papers.

- **"No evaluation of 1-to-1 attacks"** — The paper explicitly targets N-to-1 attacks as its primary demonstration, and Figure 5 shows results across different numbers of attacked parameters. Requesting 1-to-1 attacks is scope expansion.

- **"Formatting and notation issues"** — These are parser artifacts, not author errors.

- **"Missing related work on HT detection"** — Per rules, I cannot confirm whether specific missing references exist, so this is excluded.

## Novel Insights

The most insightful observation from this work is that *instruction field redundancy in RISC-V accelerator interfaces creates a practical covert channel for hardware Trojan triggering*—a genuine attack surface that the hardware security community should be aware of. The tension between the paper's genuine technical contribution (the C-SFE algorithm's kernel-level regularity and the instruction-field embedding technique) and its overclaimed threat model novelty is instructive: the concrete attack mechanisms are novel and valuable, but framing them under a "novel threat model" where the attacker controls the entire platform undermines the perceived contribution by making the vulnerability seem inevitable rather than surprising.

## Suggestions

- Reframe the contribution: lead with the specific attack mechanisms (unused instruction bits, C-SFE's HT-compatible regularity) rather than the threat model. The threat model is a necessary setup, not the main novelty.
- When comparing with T-BFA, add a clear statement that the comparison demonstrates C-SFE's *HT compatibility* advantage (fewer instruction fields per layer), not overall attack superiority. Consider adding an "apples-to-apples" comparison under the same HT area budget without restricting T-BFA's layer exploration.
- Qualify "stealthy" claims: distinguish between the genuine stealth advantages (small LUT overhead, dormant state preservation) and the aspects that are trivially true of any HT (dormant = no degradation). Remove or soften claims that are not evaluated (e.g., resistance to detection methods).
- Change "grey-box" to "strong adversary with platform-level access" or similar, since the attacker's information level is effectively white-box for the platform.

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Decision | Comparison |
|-------|-----------|----------|------------|
| qJkCEcd50n (Influence manipulation) | 3.0 | Reject | Weaker: trivially simple attack under unjustifiable threat model. Our paper has real technical depth (FPGA implementation, novel attack vector). |
| NqpdT8DwGc (Stealing Invisible) | 2.5 | Withdrawn | Weaker: unrealistic assumptions make results impractical. Our paper has a stronger attacker but a more concrete attack. |
| 41uZB8bDFh (Q-Misalign) | 6.0 | Accept Poster | Stronger: similar structure (exploiting a deployment step), better-scoped claims, multiple models evaluated. Our paper has similar technical depth but overclaims more. |
| NI0RsRuFsW (Evasive Trojans) | 4.0 | Reject | Similar scope: Trojan attack on DNNs. Our paper has real hardware demonstration but also overclaims and lacks defense evaluation. |
| 2ov9RiAkxE (LLM-integrated vulnerabilities) | 4.5 | Reject | Similar: novel attack surface with limited technical depth. Our paper has more concrete technical work (FPGA implementation, specific algorithm). |
| VmGRoNDQgJ (Influencer Backdoor) | 7.5 | Spotlight | Stronger: genuinely novel threat model with clean evaluation. Our paper overclaims novelty of its threat model. |

This paper has genuine technical substance—the unused instruction bit-field attack vector, the C-SFE algorithm, and the full FPGA demonstration are real contributions. However, the overclaimed novelty of the threat model (strong attacker with platform control), the unfair T-BFA comparison, and the unsubstantiated stealthiness claims are significant issues. Compared to the Q-Misalign paper (6.0, accepted as poster) which has a similar attack structure but better-scoped claims, this paper overclaims more. Compared to the Evasive Trojans paper (4.0, rejected), this paper has stronger technical work (real FPGA) but similar overclaiming issues. The score should be in the 4-5 range.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>