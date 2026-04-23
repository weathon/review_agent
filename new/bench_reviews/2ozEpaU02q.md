Now I have all the information I need to write the final consolidated review. Let me synthesize everything.

## Summary

The paper proposes two strategies—randomized global initialization (RGI) and dual examples (DE)—to improve adversarial transferability by leveraging multiple random-start optimization trajectories. RGI pre-computes global momentum by averaging momentum from N random-start MI-FGSM runs, while DE maintains K parallel I-FGSM trajectories whose gradients are averaged into the main attack's momentum. The method integrates plug-and-play with existing attacks and achieves consistent improvements (+3–4% average ASR) across single-model, SIA-integrated, and ensemble settings on ImageNet.

## Strengths

- **Consistent, substantial empirical improvements across all experimental settings.** The method achieves 72.7% average ASR in single-model attacks (+4.0% over PGN, Table 1), 84.0% with SIA integration (+4.3% over DTA, Table 2), and 77.7% under ensemble settings (+4.3% over Anda, Table 3). Improvements are consistent across both CNN and transformer victim models, including the typically hard-to-transfer Swin (+5.4% in Table 1).

- **Broad plug-and-play compatibility demonstrated.** Table 4 shows RGI improves all 5 tested base methods (mean +4.92% over GI, Table 4a), and the DE strategy improves all 5 base methods (up to +25.9% on I-FGSM, Table 4b), confirming the generality of both proposed strategies.

- **Evaluation against advanced defense methods.** Table 3 evaluates against AT, HGD, RS, and NRP defenses, showing the method maintains improvements even under robust defenses (e.g., 38.2% vs RS, 48.6% vs AT).

## Weaknesses

### Fatal
None.

### Major

- **Algorithm 1 contains a clear indexing bug and missing core mechanism, making the pseudocode incorrect and incomplete.** Line 16 reads `m_t ← (1/N) Σ_{n=1}^N g_{k,t} + γ · m_{t-1}`, but at this point the inner loop over k=1…K has completed, so the sum should be over the K dual-example gradients `g_{k,t}` normalized by 1/K, not over n normalized by 1/N. The index n is from the earlier RGI loop (lines 3–9) and is out of scope here. Additionally, the restart mechanism—described in Section 4.1 as "5 as the number of epochs to restart" and referenced in the algorithm comment (line 11)—is never specified in the pseudocode. These errors mean a reader cannot correctly implement the method from Algorithm 1 alone.

- **Significant computational overhead (~2× gradient evaluations) is unreported and uncontrolled for.** The proposed method adds N=5 RGI pre-computation runs (5×5=25 gradients) plus K=20 dual-example gradients per main-loop step on top of VMI-FGSM's own 20 variance-estimation samples. This totals roughly 435 gradient evaluations versus ~210 for VMI-FGSM or ~220 for PGN—a ~2× cost increase. The paper never reports this cost, never provides wall-clock times or total gradient queries, and never performs a compute-adjusted comparison. If baselines were given an equivalent budget (e.g., more iterations or more variance-estimation samples), the headline improvements could shrink or disappear. Without this control, the gains cannot be confidently attributed to the proposed mechanisms rather than additional compute.

- **Multiple inconsistencies between the algorithm, methodology text, and implementation description.** (1) Step-size: Section 4.1 says "decreasing step size (log)" while Section 3.2, Algorithm 1's header, and the hyper-parameters all say "increasing step size" / "ln sequence as the scheduled increasing step size." (2) Dual example optimizer: Section 3.2 says "optimize the dual example by I-FGSM," but Section 4.1 says "optimized by MI-FGSM"; Algorithm 1 (lines 12–15) shows I-FGSM with no momentum. (3) Section 3.2 says "we randomly generate N perturbations" for the dual example phase, but the algorithm initializes K dual examples (line 1), not N. These inconsistencies collectively undermine confidence in which configuration was actually used in experiments.

### Minor

- **Core motivation relies on t-SNE visualizations (Figure 2), which are unreliable for the structural claims being made.** The foundational claim—that different random initializations lead adversarial optimization to "converge to distinct local optima"—rests entirely on t-SNE projections. t-SNE is well-known to manufacture cluster structure from noise and distort inter-cluster distances. Trajectories appearing to reach "distinct local optima" in t-SNE space may occupy nearly the same basin in the original loss landscape. Quantitative validation (e.g., loss values or gradient cosine similarities at convergence points) would substantially strengthen this claim.

- **The paper does not explain why combining trajectories with similar individual ASR yields improved ASR.** Figure 1 shows different random starts achieve comparable individual ASR (max difference 1.6%), yet combining them improves ASR. This apparent paradox is not addressed—the mechanistic story of "exploring unexplored loss landscape" doesn't explain why different trajectories with similar attack success should combine to greater effect. Is this an ensemble/diversity effect? Does the averaged gradient reduce overfitting to the surrogate? The paper would benefit from an explicit explanation.

- **Ablation study uses only ResNet-18 as surrogate.** Table 4 evaluates RGI and DE effects using only ResNet-18, while main results use three surrogates (RN18, DN121, ViT). This limits the generalizability of the ablation findings. Additionally, no ablation isolates the restart mechanism or provides detailed analysis of the step-size schedule effect.

### Trivial
None.

## Nice-to-Haves

- Compute-adjusted comparison: give baselines (PGN, VMI-FGSM) the same total gradient-evaluation budget and compare. This would either confirm or refute whether the improvements stem from the proposed mechanisms vs. raw compute.
- Quantitative validation of the "multiple local optima" claim via loss values, gradient cosine similarities, or cross-trajectory transfer rates at convergence points.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Clear and complete algorithm specification" (Strength Finder):** Conflicts with the verified algorithm bug in Line 16 and missing restart mechanism. The algorithm is neither correct nor complete.

- **"PiT and Visformer never introduced" (Harsh Critic):** Section 4.2 does list all eight victim models including PiT and Visformer. The inconsistency is only in Section 4.1's model listing, which is trivial.

- **"White-box attack rates inflating averages" (Harsh Critic):** Including surrogate models as victim models and averaging across them is standard practice in adversarial transferability literature. This is not a weakness of the paper.

- **"Demand for larger ablation or more models" (implicit in Harsh Critic):** The model zoo (8 victim models, 3 surrogates) is already adequate for this setting.

- **"Dual example strategy is conceptually similar to VMI-FGSM/EMI-FGSM" (Harsh Critic Section-by-Section):** While there are surface similarities, the key difference—sampling from random starts rather than perturbations of the current adversarial example—is noted and the empirical comparison in Table 4 shows different behavior. This is not a weakness per se, though the paper should more explicitly discuss this relationship.

- **"Increasing step size argument not coherent" (Harsh Critic):** The paper argues that increasing step size allows dual examples to spend more steps near the benign sample (because small steps early → more iterations before leaving the neighborhood). The logic is: small initial steps keep the dual example close to x, gathering near-sample gradients; the step size then grows to escape. This is a reasonable, if imperfectly articulated, design choice. The Figure 3 observation about degradation with large iterations under fixed decreasing step size provides some empirical support.

## Novel Insights

The paper reveals an interesting empirical tension: different random initializations yield optimization trajectories that diverge visually (Figure 2) yet achieve nearly identical attack success rates individually (Figure 1, max difference 1.6%). The key insight is not that these trajectories reach "different local optima" (which is unverified beyond t-SNE), but that combining gradient information from diverse trajectories improves transferability—an effect more akin to ensemble diversity than to landscape exploration per se. The ablation in Table 5 showing that different base attacks benefit from different step-size schedules (log for MI, linear for PI and VMI) suggests the proposed strategies interact non-trivially with the base method's optimization dynamics, which would benefit from deeper analysis.

## Suggestions

- Fix Algorithm 1: correct Line 16 to `m_t ← (1/K) Σ_{k=1}^K g_{k,t} + γ · m_{t-1}`, add explicit restart mechanism (re-initialize δ_{k,0}^{dual} every R epochs), and clarify the step-size schedule (resolving the "decreasing" vs "increasing" contradiction and specifying the exact ln sequence used).
- Report computational cost (total gradient evaluations and/or wall-clock time) for all methods in the main tables, and ideally provide a compute-adjusted comparison where baselines are given an equivalent gradient-evaluation budget.
- Add quantitative analysis to support the t-SNE-based motivation: report loss values and gradient cosine similarities at the convergence points of different random-start trajectories.

## Score and Decision

**Calibration anchors compared:**

| Paper | Avg Score | Decision | Comparison |
|-------|-----------|----------|------------|
| Boosting Ray Search (tIBAOcAvn4) | 7.50 | Accept (Spotlight) | High anchor: has strong theoretical foundations + empirical results. This paper is clearly below — it lacks theoretical grounding and has algorithm bugs. |
| Rethinking Invariance in AT (M9SKazbVkJ) | 7.00 | Accept (Poster) | High anchor: novel insight with empirical support. This paper has weaker motivation rigor. |
| Enhancing Transferable Attacks on ViTs (1BuWv9poWz) | 5.33 | Accept (Poster) | Medium anchor: strong empirical results but opaque presentation. This paper is comparable in empirical strength but has worse algorithm/presentation issues (actual bugs, not just opacity). |
| Activating Neurons for Transferability (VSidzaTzpd) | 5.25 | Withdrawn | Medium anchor: correlation-vs-causation concern similar to this paper's t-SNE concern. This paper has stronger empirical results but also has algorithm errors. |
| Efficient Diversified Attack (qpuxHL9X3d) | 5.25 | Reject | Medium anchor: multi-restart diversification, computational overhead. Very similar profile — both use multiple trajectories with compute cost concerns. |
| MAA (iR5qF9N1Ge) | 5.80 | Reject | Medium anchor: comprehensive experiments but computational cost and reproducibility issues. |
| Certified Defense Dynamic Smoothing (85Eej2kUHQ) | 2.33 | Withdrawn | Low anchor: has a provably incorrect core theorem. This paper is above — its bugs are in pseudocode, not in the method itself. |

This paper sits below the borderline accepted paper (5.33, ViT attacks) because it has actual algorithm bugs and ~2× unreported compute cost, and sits alongside the rejected papers (5.25, EDA; 5.80, MAA) which had similar patterns of reasonable ideas with significant methodological gaps. The consistent empirical improvements are a genuine strength, but the combination of algorithm errors, missing cost analysis, and weak motivation justification places it below the acceptance threshold.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>