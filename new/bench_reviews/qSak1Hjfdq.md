Now let me search for calibration anchors.Now I have enough information to write the final meta-review. Let me synthesize the findings.Now I have enough information to write the final meta-review.

---

## Summary

This paper introduces the All-day Multi-Scenes Lifelong Vision-and-Language Navigation (AML-VLN) problem, formalizing sequential navigation adaptation across two axes (scenes and environments). The proposed Tucker Adaptation (TuKA) represents multi-hierarchical knowledge in a 4th-order tensor using Tucker decomposition, with separate scene-specific and environment-specific expert rows, a shared core tensor, and a shared encoder-decoder. A Decoupled Knowledge Incremental Learning (DKIL) strategy handles catastrophic forgetting through EWC regularization on shared components, expert consistency constraints, and orthogonality pressure. Built on TuKA, the AllDayWalker agent achieves 65% avg SR across 24 tasks vs. 52% for the best complete baseline.

---

## Strengths

- **Well-motivated and structured problem formulation.** The 2D scene × environment decomposition of lifelong VLN is more principled than treating scenarios as flat task IDs, and directly maps to real-world robotics deployment needs. The 24-task benchmark spanning 5 simulation scenes, 2 real-world scenes, and 4 imaging conditions (normal, low-light, overexposure, scattering) is substantially broader than typical continual VLN evaluations.

- **Clean factored-expert architecture.** TuKA's separation of scene experts U³[s,:] and environment experts U⁴[e,:] enables a combinatorially efficient parameterization: M + N expert vectors cover M × N scenario combinations rather than requiring M·N independent adapters. This is a genuine and non-trivial structural innovation over MoE-LoRA variants.

- **Empirically solid across most tasks.** AllDayWalker achieves 65% avg SR vs. the next best complete baseline at 52% (O-LoRA), and a dramatically lower forgetting rate (F-SR = 11% vs. 18% for SD-LoRA and 36% for BranchLoRA). Results hold across SR, SPL, and OSR.

- **Generalization to unseen scenarios.** Table 5 shows AllDayWalker achieves 55% avg SR on six completely unseen scenario combinations, surpassing SD-LoRA (39%) and BranchLoRA (40%) by large margins, indicating the factored expert structure enables useful compositional generalization.

---

## Weaknesses

### Fatal
None.

### Major

- **Incomplete comparison with SD-LoRA.** Table 1 is missing SD-LoRA results for T22, T23, and T24, and its average SR is entirely absent. Crucially, where SD-LoRA data exists, it outperforms AllDayWalker on T8 by a wide margin (74% vs. 38%) and also leads on T12 (75% vs. 67%). The paper provides no explanation for these anomalies nor for the missing entries. Without SD-LoRA's full average, the headline "consistent superiority" claim is incomplete with respect to the strongest baseline. A reader cannot determine whether AllDayWalker's gap over SD-LoRA is robust or narrow. The paper needs to either include the missing results or explicitly explain why they are absent (e.g., a methodological constraint of SD-LoRA on real-world tasks).

- **The inference-time expert retrieval mechanism is load-bearing but completely unanalyzed.** At test time, the system selects scene and environment experts by cosine similarity between CLIP embeddings of current observations and stored training embeddings (§3.4). Whether this retrieval is correct or not determines whether the right adapter is applied. The paper reports no retrieval accuracy, no oracle upper bound (using ground-truth expert identity), and no analysis of failure modes (e.g., a dark normal-lit room being misidentified as low-light). Figure 7's legend ("Recall," "Task2Vec," "CLIP") suggests some comparison of retrieval strategies exists, but it is confined to the appendix with no numerical summary in the main text. Given that this module mediates the entire multi-expert benefit, its absence from ablation analysis is a substantive gap.

- **The central technical claim—that Tucker decomposition itself (beyond factored expert indexing) drives the gains—is not established.** From Eq. (3), the actual weight update ΔW_t = U¹·(G ×₃ U³[s,:]×₄ U⁴[e,:])·(U²)ᵀ is a standard low-rank 2D matrix; the Tucker tensor is a parameterization, not the deployed operator. The genuine structural innovation is maintaining *separate, independently indexed* scene and environment expert vectors (U³[s,:] and U⁴[e,:]). This factored structure could in principle be implemented with a simple outer-product LoRA using two separate expert vectors, without a Tucker tensor. The ablation comparing 3rd-order vs. 4th-order tensors (Figure 8) confounds tensor order with the number of expert vectors (20 joint experts for 3rd-order vs. M+N=11 decoupled experts for 4th-order), so it does not cleanly isolate the Tucker structure's contribution. While the 4th-order result is empirically superior, the reason is left ambiguous.

### Minor

- **Equation (6) contains circular notation.** The Fisher update F_{θ,t} = ω·F_{θ,t-1} + (1-ω)·F_{θ,t} has F_{θ,t} on both sides. The right-hand F_{θ,t} presumably refers to the freshly computed task-t Fisher before EMA mixing, but this is not notated distinctly. This should use F̂_{θ,t} or similar to avoid ambiguity.

- **Single task ordering used throughout.** The caption of Figure 6 states "the order of tasks is randomized," but results are reported for a single fixed sequence. Continual learning performance is known to be sensitive to task order. Averaging over multiple orderings would substantially strengthen the reliability of the reported results—especially the forgetting metrics.

- **Table 3 ablation contains a near-duplicate row.** The 3rd row (✓✓✓, SR=65, F-SR=11, SPL=58, OSR=69) and the 6th row (✓✓✓, SR=65, F-SR=11, SPL=58, OSR=68) both show the full model configuration. The two rows are nearly identical (OSR differs by 1), strongly suggesting a copy-paste error. One row should presumably cover a different ablation condition (e.g., ✓✗✓) to complete the design space.

---

### Trivial

- Eq. (8) as written applies the orthogonality loss to the full Ũ³ matrix, which may appear to constrain previously-learned frozen rows. The text should clarify that frozen rows receive no gradients, so the constraint operates only on the active row in practice.

---

## Nice-to-Haves

- **Joint training upper bound.** A multi-task oracle (training simultaneously on all 24 tasks) would establish the performance ceiling and contextualize how much lifelong learning costs vs. the upper bound.

- **Expert retrieval accuracy in the main text.** Even a simple table showing per-environment retrieval accuracy (scene identification rate and environment identification rate separately) would be informative and easy to include.

- **Effect of task ordering.** Report performance under 2–3 additional random orderings to assess robustness of the lifelong learning conclusion.

- **Expert specialization visualization.** Showing what scene experts U³[s,:] and environment experts U⁴[e,:] capture—via attention maps or representation similarity—would provide qualitative evidence that the decoupled experts indeed specialize as hypothesized.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **FSTTA/FeedTTA comparison criticism.** The harsh critic argues that including these test-time adaptation baselines is misleading because they are designed for single-step distribution shifts. However, the paper explicitly describes their purpose and includes them to show contextual contrast, which is informative rather than manipulative. *Removed as scope creep.*

- **Benchmark asymmetry (real-world lacks overexposure/scattering).** While technically true, the paper explicitly notes these conditions are impractical to generate in real-world settings, and the benchmark is clearly described. This is an inherent practical constraint, not a methodological flaw. *Removed as scope creep.*

- **Figure 7 label mismatch ("Recall," "Task2Vec," "CLIP").** The harsh critic flags this as a content inconsistency. However, these labels likely describe different expert retrieval strategies compared in Figure 7's ablation (CLIP-based retrieval, Task2Vec-based retrieval, recall-based retrieval), which is consistent with §3.4's focus on expert selection. The parser may also have garbled figure-level legend text. *Removed as likely parser artifact and plausible content explanation.*

- **AllDay-Habitat uses parametric degradation.** The critic questions whether the degradation is realistic. The paper uses physically grounded atmospheric scattering and sensor noise models—standard practice in simulation environments. *Removed: not a methodological flaw.*

- **Strength: "paper addresses an important problem."** Generic; dropped per filter rules.

---

## Novel Insights

The most genuinely interesting observation emerging from this review is the combinatorial efficiency of the factored expert structure: maintaining M scene + N environment expert vectors (M + N parameters of width r) covers M×N scenario combinations that would require M×N independent adapters under a flat MoE-LoRA approach. This combinatorial compression implicitly encodes the assumption that "scene s under environment e" decomposes into separable components—a strong and potentially useful inductive bias for structured lifelong learning. However, the degree to which this inductive bias actually drives the empirical gains (as opposed to the DKIL regularization strategy) remains to be isolated by future work.

---

## Suggestions

1. **Include complete SD-LoRA results for T22–T24** or provide an explicit explanation (e.g., "SD-LoRA by design does not generalize to real-world tasks T22–T24") along with a discussion of the T8 anomaly where SD-LoRA substantially outperforms.

2. **Report a retrieval accuracy experiment**: what fraction of test episodes correctly identify (a) scene and (b) environment, broken down by environment type. Report at least an oracle upper bound (correct expert selected by ground truth) to quantify the retrieval module's practical cost.

3. **Add a factored-expert LoRA ablation** (outer-product of scene and environment vectors, no Tucker tensor) to cleanly separate the Tucker structure's contribution from the factored expert indexing concept.

4. **Fix Eq. (6) notation**: use F̂_{θ,t} for the freshly computed task-t Fisher before EMA.

5. **Report results under multiple task orderings** (2–3 random permutations) to validate that gains are not sensitive to sequence effects.

---

## Score and Decision

**Calibration anchors:**
- **SD-LoRA** (5U1rlpX68A, avg 7.5, Oral): the strongest baseline in this very paper. SD-LoRA was accepted at Oral for rigorous theoretical and empirical CL-LoRA contributions. The paper under review is less rigorous and has incomplete comparison against SD-LoRA itself.
- **TAIL** (RRayv1ZPN3, avg 6.2, Poster): adapter-based continual adaptation for imitation learning. Accepted with modest novelty but complete experiments.
- **C-CLIP** (sb7qHFYwBc, avg 6.5, Poster): multimodal VL continual learning with benchmark + method, accepted.
- **FLoRA** (OALIb8oNfl, avg 5.75, Poster): Tucker decomposition for PEFT across dimensions, accepted as a poster. Closest thematic analog; has more complete experiments in its narrower scope.
- **ModalPrompt** (04TRw4pYSV, avg 3.5, Reject): multimodal CL, rejected for weak baselines and insufficient novelty. Clearly weaker than this paper.
- **Online Weight Approximation** (HCCkCjClO0, avg 3.0, Reject): weak CL paper—clearly weaker.

**Positioning:** The paper sits between the accepted medium-tier (C-CLIP, TAIL at 6.2–6.5) and FLoRA (5.75). It has a broader problem scope than FLoRA and solid empirical results overall, but the three major weaknesses—incomplete SD-LoRA comparison (including cases where SD-LoRA outperforms), zero analysis of the critical retrieval module, and inadequately isolated Tucker claim—keep it below the clearly accepted papers. The overall result profile (if SD-LoRA missing data were explained) would support acceptance, but as-is, the experimental gaps represent genuine reviewer blockers.

**Score: 5.0** (Borderline Reject / Weak Accept)

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>