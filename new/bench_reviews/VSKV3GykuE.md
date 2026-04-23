Now I have enough information to write the consolidated review. Let me synthesize everything.## Summary
RAC-LoRA proposes a randomized asymmetric chain of LoRA updates where one LoRA matrix is randomly sampled per chain step and only the other is trained. This restores Lipschitz smoothness in the trainable subproblem and converts the update to a projected gradient step in the ambient weight space. The paper provides O(1/T) nonconvex and linear-rate convergence under the PL condition, extending to SGD, random reshuffling, and a federated learning variant.

## Strengths

- **Concrete convergence failure demonstration (Figure 1, Section 3):** The paper provides a clear numerical counterexample using a 9-dimensional quadratic, showing that LoRA and COLA diverge at step size 1/L, and that AsymmLoRA converges to a suboptimal stationary point even with smaller step sizes, while RAC-LoRA converges to the global optimum.

- **Elegant reduction to randomized sketched gradient descent (Equations 3–4, Section 5.1):** Fixing one LoRA matrix randomly converts the bilinear update into a projected gradient step W^{t+1} = W^t − γ H_B^t ∇f(W^t) with H_B^t a rank-r orthogonal projection. This is a principled and clean derivation that unifies the asymmetric structure and chaining mechanisms.

- **Genuine convergence guarantees (Theorems 5.3 and 5.5):** O(1/T) convergence in the nonconvex smooth setting and linear convergence under the PL condition are established, with the rate explicitly depending on λ_min^H = r/n, meaning convergence degrades gracefully with rank and recovers standard GD at full rank. This is a meaningful improvement over COLA's theory, which completely ignores the rank structure.

- **Framework generality (Table 1):** The analysis covers GD, SGD, and Random Reshuffling under one framework, with a federated learning extension (Fed-RAC-LoRA, Algorithm 2). This breadth adds value beyond a single-optimizer result.

- **Empirical validation of the r/n convergence rate (Figure 2):** Linear regression experiments confirm that convergence speed scales proportionally with r/n across different ranks, directly validating Theorem 5.3.

## Weaknesses

### Fatal
None.

### Major

- **Empirical results fail to demonstrate practical superiority over existing methods.** On GLUE (Table 2), RAC-LoRA achieves 77.0 average, below standard LoRA (78.5) and COLA (77.6). The authors explicitly concede this in the Discussion, attributing it to GLUE being "too easy" (FPFT and LoRA are already close), but no harder benchmark where FPFT significantly outperforms LoRA is then tested. The MNIST experiment (Table 3) isolates the chaining benefit by artificially constraining capacity to rank 1 — this demonstrates that chaining in general helps, not that RAC-LoRA specifically outperforms alternatives. COLA (Gaussian/Zero init) outperforms RAC-LoRA in Table 3 (92.6% vs 92.0%) in the most directly comparable setting. The core practical claim — that RAC-LoRA closes the FPFT gap — remains experimentally unsupported on any real and competitive task.

### Minor

- **Self-critique in Section 2.2 partially applies to the paper's own theory.** The paper criticizes COLA's analysis for "replac[ing] low-rank optimization over matrices A and B with full-rank matrix optimization (ΔW)." After Equation (3), RAC-LoRA's own convergence analysis is entirely in terms of the full weight matrix W (not A and B), and is essentially a convergence analysis of a randomized sketched gradient descent method on W. The key distinction — that RAC-LoRA's rate explicitly captures the r/n factor through λ_min^H — is real and meaningful, but the paper would benefit from acknowledging this parallel rather than presenting COLA's approach as categorically different.

- **Growing effective rank of the accumulated update is unaddressed.** After T chain steps, ΔW = Σ_t (α/r) B_S^t Â^t has effective rank up to T·r. For T=10, r=2 (as in Table 2), the effective rank of the adaptation is up to 20 — comparable to a rank-20 single LoRA. This growing-rank phenomenon has implications for how RAC-LoRA compares to higher-rank single LoRA, and for whether the chaining procedure provides benefits beyond simply using a higher-rank adaptation from the start.

- **Theory covers GD/SGD/RR, but all practical experiments use AdamW.** No theoretical result is proven for Adam-type optimizers. The paper claims theoretical findings are "supported by experimental results" in the abstract, but the optimizer mismatch is never acknowledged. This gap weakens the claim that the theory explains the empirical behavior.

- **The convergence rate 2(f(W^0) − f*) / (λ_min^H · γ · T) can be very slow in the practical low-rank regime.** With λ_min^H = r/n, for r=2, n=768 (RoBERTa), this gives λ_min^H ≈ 0.003, which means the constant in the rate is ~333× worse than full-rank GD. The paper shows this dependence in Figure 2 but does not discuss its practical implication for large-model fine-tuning.

- **The abstract does not qualify that "convergence to the same solution as FPFT" requires the PL condition.** Theorem 5.5 requires Assumption 5.4 (PL), which is not guaranteed in practice. The abstract's unqualified claim modestly overclaims.

### Trivial
None that qualify under reviewer guidelines.

## Nice-to-Haves

- A competitive benchmark where FPFT substantially outperforms LoRA (e.g., domain adaptation with a larger model) would properly stress-test the claim that chaining closes the FPFT gap.
- A fair comparison controlling for total adaptation capacity (e.g., comparing RAC-LoRA with T chains of rank r against single LoRA with rank Tr) would clarify whether chaining offers benefits beyond simply increasing effective rank.
- An extension of the theoretical analysis to Adam-type optimizers would close the most obvious gap between theory and practice.
- A loss landscape or trajectory visualization for the toy quadratic would make the intuition behind Section 3 more concrete.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"The theory misrepresents what it establishes" (Harsh Critic §1):** While technically accurate that after Equation (3) the analysis operates in W-space, this is not a misrepresentation — the paper's own notation makes this explicit. Furthermore, the low-rank structure IS captured through H_B^t and λ_min^H = r/n, which is a genuine improvement over COLA's W-space analysis. The observation that the results are "essentially known for sketch-based methods" is removed under the rule against citing missing related works.

- **"Connection to sketch-and-project / randomized Kaczmarz literature missing" (Harsh Critic):** Removed under the hard rule prohibiting mention of missing related works.

- **"LoRA and COLA fail to converge is overstated" (Harsh Critic §3):** Partially valid, but Figure 1 demonstrates both (a) divergence at the standard step size 1/L and (b) convergence to a suboptimal stationary point at smaller step sizes. The paper presents both behaviors honestly and does not exclusively rely on the 1/L setting to make the case.

- **"Subproblem approximation gap between derivation and algorithm" (Harsh Critic §Section 5.1):** The paper states that Step 4 of Algorithm 1 involves approximate solving, and the gap is handled in the appendix. Removed as a missing-appendix criticism (the hard rule excludes such criticisms).

- **"Convergence failure framing" regarding step-size sensitivity (Harsh Critic):** The paper explicitly tracks this distinction ("even with smaller step sizes") so the concern is already addressed in the text.

- **Strength Finder's "Communication efficiency for federated learning":** Generic — no quantitative evidence comparing communication cost to COLA or other methods. Dropped per the rule on unsupported generic strengths.

## Novel Insights

The most genuinely novel observation in this paper is the precise identification of *why* standard LoRA updates lose Lipschitz smoothness (the bilinear coupling of A and B) and the clean algorithmic fix: randomizing one factor per chain step reduces the update to a projected gradient step in the ambient weight space, with convergence rate explicitly governed by λ_min^H = r/n. This establishes a smooth interpolation between rank-r adaptation (slow by a factor of r/n) and full-parameter fine-tuning (r = n, recovering standard GD), providing a theoretically grounded explanation for why LoRA underperforms FPFT from an optimization perspective — not just from a representational capacity perspective.

## Suggestions

1. **Provide a competitive benchmark where the chaining benefit is demonstrable**: Test on a task where FPFT substantially outperforms single-LoRA (e.g., domain-shift fine-tuning of a 7B model). Without this, the central practical claim cannot be evaluated.
2. **Explicitly compare at equal effective rank**: Show whether T chains of rank r outperform a single LoRA of rank Tr, to separate chaining benefits from rank benefits.
3. **Acknowledge and discuss the growing-rank nature of ΔW**: After T steps, the adaptation is of rank up to Tr — state this explicitly and discuss the memory implications.
4. **Be transparent in the abstract and Section 2.2** about the parallel between RAC-LoRA's own theory (which operates on W) and COLA's theory (same ambient space); distinguish them on the basis of the r/n dependence, not on the use of W-space.

## Score and Decision

**Calibration anchors consulted:**

| Paper | Path | Avg Score | Comparison to RAC-LoRA |
|---|---|---|---|
| GoLore | udtrtwkvk5.md | 5.25 (Reject) | Most similar: finds convergence failure in GaLore, proposes provably convergent variant with O(1/T) guarantee, modest empirical improvement over baseline; slightly stronger empirical results than RAC-LoRA |
| Stiefel LoRA | c2OtbtZXFC.md | 4.75 (Reject) | Similar: LoRA optimization theory with theory-practice gap; weaker in both theory and experiments |
| NEAT | l3oE5vBjDs.md | 5.0 (Reject) | Similar: PEFT convergence theory-practice gap; weaker theory but similar experiment quality |
| LoRA Done RITE | VpWki1v2P8.md | 8.67 (Accept Oral) | High anchor: strong LoRA theory with 4.6% accuracy improvements on large models; clearly superior empirical results |
| VeRA | NjNfLdxr3A.md | 7.25 (Accept) | High anchor: novel PEFT method with solid experiments on real LLM benchmarks |
| Weak papers | cPmLjxedbD, JNZ3Om6NPS | 1.0–2.0 (Reject) | Low anchors: fundamentally flawed or unsupported; RAC-LoRA is clearly above these |

**Assessment relative to anchors:** RAC-LoRA is most comparable to GoLore (5.25). Both identify convergence failures in existing methods and propose provably convergent alternatives with O(1/T) guarantees. However, RAC-LoRA is somewhat stronger theoretically (covers GD/SGD/RR/federated learning under one framework, captures r/n dependence explicitly) but weaker empirically (RAC-LoRA *underperforms* baseline LoRA on the main NLP benchmark, while GoLore shows marginal positive improvements). Balancing the richer theoretical framework against the empirical regression, RAC-LoRA sits at the same level as GoLore.

The paper has genuine technical merit — the key insight is sound and the theoretical framework is clean — but the empirical shortfall (underperforming LoRA on GLUE with no compensating experiment at scale) is a significant barrier. This is a Reject at the current calibration level.

**Originality:** Moderate. The randomization trick restoring smoothness is a clean idea, but the convergence analysis is essentially sketched GD applied in a LoRA context.
**Importance:** Moderate. Theoretical grounding for LoRA optimization is a real need, but not demonstrated to matter empirically.
**Claim support:** Weak. The core practical claim (closing the FPFT gap) is unsupported experimentally.
**Soundness:** Good. The mathematics appears correct.
**Writing:** Clear and honest about limitations.
**Value to community:** Limited in current form without strong empirical demonstration.

**Score: 5.0 — Reject**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>