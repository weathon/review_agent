Now let me look at the calibration papers mentioned by the human finder reviewer:## Summary
This paper develops constraint-based causal discovery for acyclic stochastic differential equation (SDE) models, making three core contributions: (1) a "lifted dependence graph" construction with a proved global Markov property that exploits temporal directionality; (2) sound and complete causal discovery algorithms (Algorithm 1 via h-local CIs, Algorithm 2 via symmetric CI + post-processing) that recover the full DAG—not just an equivalence class—assuming a CI oracle; and (3) a consistent signature kernel-based CI test for path-valued random variables that serves as a practical oracle replacement. The framework simultaneously handles irregular sampling, partial observations with unobserved confounders, path-dependence, and diffusion dependence—no prior method addresses all four.

---

## Strengths

- **Elegant lifted graph construction:** The lifted dependence graph (Section 3.1, Proposition 3.1, Figure 2) provides a clean and technically sound way to exploit temporal directionality. The insight that V₀ (past) and V₁ (future) nodes yield an acyclic auxiliary graph that enables separation of CI relationships is conceptually novel and well-executed. Unique full-DAG recovery—not just CPDAG—is a notable theoretical advance over static constraint-based methods.

- **Sound and complete theory:** Theorem 3.2 proves soundness and completeness under parent faithfulness (weaker than strong faithfulness, as explicitly acknowledged). The completeness proof via the lifted graph and global Markov property, combined with the impossibility result for cyclic graphs (Appendix A.3), reflects genuine mathematical depth. Corollary 3.3 for orienting remaining CPDAG edges via initial-condition independence is an elegant additional result.

- **Consistent CI test for path space:** The signature kernel CI test fills a genuine gap. The existing KCIPT/SDCIT consistency proofs rely on density assumptions that break for path-valued random variables. The novel consistency proof in Appendix A.14 using different arguments is a contribution of independent interest. Figure 7 (robustness to missingness) and Figure 8 (fractional Brownian motions, outside the semi-martingale framework) usefully demonstrate breadth beyond the SDE model.

- **Honest acknowledgment of limitations:** The paper explicitly discusses the acyclicity constraint, provides the impossibility result for cyclic graphs, acknowledges that s/h choices affect reliability, and discloses that SCOTCH wins in specific configurations. The "proof-of-concept" framing of the trading experiment is appropriately humble.

- **Genuinely broad experimental scope:** The paper tests across linear, path-dependent, nonlinear, and diffusion-dependent SDEs; multiple dimensions (d=3 to 50); partial observations; functional data; and fractional Brownian motions. The power analysis in Figure 3 (against the only prior path-space CI test baseline) is convincing, showing near-1 power already at n=40.

---

## Weaknesses

### Fatal
*None.*

### Major

- **Missing irregular sampling experiments despite being listed as a primary claimed advantage.** The paper lists criterion (a)—handling irregularly and differently-sampled observations—as a key differentiator, and the introduction, abstract, and limitations section foreground it. Yet every experiment uses regularly sampled paths. Figure 7 shows robustness to *missingness* (randomly dropped observations), which is distinct from genuine irregular/asynchronous sampling across variables. The central advantage of not relying on the "discrete-time" assumption is stated but never directly demonstrated against a baseline operating on a shared regular grid. This is a structural gap between the paper's stated scope and its evidence.

- **Partial observation evaluation is too thin for a major claimed contribution.** The partially observed setting is advertised as one of four core criteria. The evaluation consists of one 4-node example (Figure 4) with a single latent structure, reporting 88 vs. 8 errors over 100 runs. There is no systematic variation over graph sizes, numbers of latent nodes, sample sizes, or missingness regimes. Given that partial observation handling is a primary selling point of the constraint-based approach and is explicitly positioned as a reason to prefer this method over SCOTCH, the evidence is not commensurate with the claim.

- **High-dimensional performance gap (d ≥ 20) is underacknowledged.** Table 2 shows that at d=20 and d=50, SCOTCH(200,2k) substantially outperforms all SigKer variants: 370±174 vs. 1026±40 (⊥_{s,h}^+) and 538±70 vs. 4593±93 (⊥_sym+pp) at d=50. The paper's main text says the method "clearly outperform[s] PCMCI and SCOTCH up to d=10"—this qualification is honest—but the conclusion and abstract use broader language ("outperforming existing approaches"). The paper provides little analysis of *why* performance degrades so severely at d≥20 (whether due to CI test power loss from larger conditioning sets, or combinatorial issues in the skeleton step), leaving practitioners without guidance on the method's practical range of applicability.

### Minor

- **Lack of ablation isolating CI test contribution from discovery algorithm contribution.** Table 2 compares ⊥_{s,h}^+, ⊥_sym, and ⊥_sym+pp variants against each other, which is informative. However, there is no experiment that plugs the signature kernel into a standard PC algorithm (with Euclidean snapshot features) to disentangle whether gains in causal discovery come from the CI test design, the interval-based independence criterion, or the algorithm's edge orientation logic. This would strengthen the case for the specific algorithmic contributions.

- **No theoretical guidance on s and h selection; empirical choice is non-trivial.** The paper reports s=0.1T "performed best" (based on Table 5 in the appendix) and acknowledges that ⊥_{s,h}^+ reliability may be "negatively affected" by the choice of s and h. No guidance is given on how to set these parameters for different SDE types or time horizons. Since Algorithm 1's practical behavior depends directly on these choices, this is a genuine usability gap, though the paper is honest about it.

- **Real-data trading experiment cannot validate causal discovery accuracy.** The authors explicitly call it a "proof-of-concept study" and use P&L as a "substitution for ground truth," which is honest. However, the text states the experiment demonstrates "broad applicability and potential downstream impact," which overstates what P&L performance can show—trading returns conflate many design choices with test quality. The section is illustrative at best.

- **Faithfulness discussion deferred entirely to appendix.** Theorem 3.2's completeness relies on "parent faithfulness." The main text says only that "a faithfulness type assumption" is used and points to Appendix A.8. Given that faithfulness violations are a known practical concern for constraint-based methods, a brief discussion of what parent faithfulness excludes in the SDE context would help readers assess the real-world scope of the theorem.

### Trivial

- The paper acknowledges computational complexity is "deferred to Appendix B.9 due to space constraints." For a method being applied to d=50, a one-sentence summary of the dominant complexity term in the main text would be useful without taking significant space.

---

## Nice-to-Haves

- Experiments on truly irregularly (asynchronously) sampled data, where different variables are observed at different time points (not just missing data), to directly validate criterion (a).
- A partial-observation benchmark over multiple graph structures and latent configurations, with comparison to FCI equipped with a standard kernel CI test as a natural competitor.
- Analysis of type I error calibration and empirical test power as a function of conditioning set dimensionality, to characterize when the CI test remains reliable in practice.
- Exploration of approximate methods (screening, sparse structure assumptions) to improve scalability at d≥20.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**Harsh Critic – Baseline comparison asymmetry (inflated):** The critic argues that judging SCOTCH by "worst case performance over a set of reasonable hyperparameter settings" is invalid when the proposed method also benefits from internal model selection (s=0.1T, test variant choice). However, the paper's defense is grounded: SCOTCH's optimal hyperparameters shift between n=200 (best: SCOTCH 50,2k at 25±13) and n=400 (best: SCOTCH 100,2k at 9±8), demonstrating genuine instability that cannot be resolved without ground truth. The authors' selection of s=0.1T is a single fixed choice validated across all settings, not a per-instance tuning. The asymmetry is real but the paper's framing is not merely rhetorical—it reflects a principled distinction between a hyperparameter-free method and one with unstable, unselectable hyperparameters. This point is weakened to the extent that it is addressed.

**Harsh Critic – "Fundamentally limited" claims about neural/variational approaches are overclaiming rhetoric:** The paper asserts that neural SDE methods like SCOTCH "are fundamentally limited" in partially observed settings because they require parameterizing the unobserved variables. This is substantively true—one cannot enumerate infinitely many possible latent variables in a neural network prior—and is backed by the paper's discussion. This is not unfair rhetorical positioning.

**Harsh Critic/Spark – Missing related works:** Per policy, criticisms about missing citations are excluded as we cannot confirm existence of specific works.

**Spark – No per-edge accuracy breakdown:** While useful, SHD is the standard metric for graph recovery in this community and its use is not a weakness.

---

## Novel Insights

The lifted-graph construction is the paper's most intellectually distinctive contribution. By doubling the variable set into past/future nodes and proving that the resulting acyclic auxiliary graph satisfies a global Markov property, the authors transform the SDE causal structure problem into a form where time directionality provides strict identifiability advantages unavailable to static methods—recovering the full DAG rather than its equivalence class without additional distributional assumptions. This is qualitatively different from prior discrete-time approaches that also use temporal direction, because it works directly with path-valued random variables and handles diffusion dependence, not just drift dependence. The pairing of this construction with the impossibility result for cyclic graphs (which rules out extending the framework naively) demonstrates that the acyclicity scope is not a limitation of ambition but a principled boundary. Together, these constitute a coherent and genuinely novel theoretical framework for continuous-time causal discovery.

---

## Suggestions

1. **Add at least one irregular-sampling experiment:** Demonstrate the method on data where variables are genuinely asynchronously observed (different timestamps per variable), which is the cleanest way to validate criterion (a). Even a simple bivariate comparison suffices.
2. **Expand partial observation evaluation:** Test 3–4 different graph structures with different latent configurations; compare to FCI with a standard kernel CI test as a baseline; report skeleton precision/recall in addition to error counts.
3. **Add a diagnostic for d≥20 failures:** Report per-dimension type I/II error rates for the CI test at large conditioning sets, and discuss whether approximate permutation tests or variable screening could recover reasonable performance.
4. **Provide one or two sentences on faithfulness in the main text:** Briefly characterize what "parent faithfulness" excludes in SDE models to help readers assess practical scope without needing to consult the appendix.

---

## Score and Decision

**Calibration papers:**

| Paper | Topic | Scores | Decision |
|---|---|---|---|
| V1GM9xDvIY (SCOTCH) | Neural SDE causal structure learning | 6,6,8,8 (avg ≈ 7) | Accept (poster) |
| lrQlLqQase | Causal reasoning in stochastic processes | 5,6,3,8 (avg ≈ 5.5) | Accept (poster) |
| GPcSYm89wK | Kernel-based CI test | 5,5,3,5 (avg ≈ 4.5) | Reject |
| or8wkKoBP4 | Constraint-based methods, faithfulness | 3,5,3,5 (avg ≈ 4) | Reject |
| orD5t7blqV | Constraint-based algorithm efficiency | 3,6,5,3 (avg ≈ 4.25) | Reject |

**Reasoning:** This paper compares favorably to SCOTCH (avg 7, accepted), offering more theoretical depth (soundness/completeness proofs, novel Markov property, consistency proof), broader scope (partial observation, irregular sampling framework), and comparable empirical breadth. The rejected papers (avg 4–4.5) either lack experiments, have technical errors, or offer only incremental improvements—none of these apply here. The paper's real weaknesses (missing irregular-sampling demo, thin partial-observation evaluation, poor high-dim performance) are genuine but do not invalidate the core contributions; they are gaps in empirical coverage rather than flaws in the theoretical framework. This places the paper clearly above the rejected papers and in a comparable range to SCOTCH, though the missing experiments and high-dim scalability gap prevent a higher score.

**Axis ratings:**
- *Originality:* High — lifted graph, path-CI Markov property, unique DAG recovery without equivalence class ambiguity are genuinely new
- *Importance of research question:* High — continuous-time causal discovery with irregular/partial observations is an important unsolved problem
- *Claims supported:* Moderate — theory is well-supported; empirical claims on partial observations and irregular sampling are understated
- *Soundness of experiments:* Moderate — comprehensive coverage of SDE types but missing the irregular-sampling demonstration and thin partial-observation benchmark
- *Clarity of writing:* Good — well-organized, limitations honestly disclosed
- *Value to research community:* High — CI test for path space is of independent interest; framework opens new directions

**Final score: 7.0**

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>