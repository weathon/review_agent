## Summary
DSpodFL proposes a unified algorithmic framework for decentralized federated learning (DFL) in which both per-client local SGD steps and pairwise model exchanges are modeled as arbitrary binary indicator random variables, capturing heterogeneous and time-varying computation/communication availability. The authors derive convergence guarantees for strongly convex and non-convex loss functions under mild graph connectivity and data heterogeneity assumptions, show that DGD, DFedAvg, and Randomized Gossip emerge as special cases, and demonstrate empirically that DSpodFL achieves better accuracy-under-delay than those baselines.

---

## Strengths

- **First joint treatment of sporadic SGD and sporadic aggregation in fully decentralized FL.** Prior work addresses either sporadic SGD (centralized, e.g., Anarchic FL) or sporadic aggregation/gossip (decentralized), but not both simultaneously in the decentralized setting. The coupling of these two processes introduces genuinely new analytical challenges around heterogeneous aggregation periods and uncorrelated consensus epochs.

- **Comprehensive convergence analysis.** Theorems 4.11 and 4.12 cover strongly convex and non-convex cases with both constant and diminishing learning rates, yielding geometric and O(ln K/√K) rates respectively. The paper also provides explicit specializations that recover known DGD rates (d_min = 1, ζ = 0), giving a verifiable consistency check.

- **Milder graph connectivity assumption.** Assumption 4.4 requires only asymptotic connectivity (the union graph is connected and each edge appears infinitely often), strictly relaxing the static-graph or B-connected-graph requirements used in most prior DFL analyses (Sun et al. 2022; Mishchenko et al. 2022; Nedić & Ozdaglar 2009). This is not a cosmetic difference: it permits genuinely time-varying topologies.

- **More general data heterogeneity parameterization.** The (δ, ζ) gradient diversity assumption in Assumptions 4.1(c)/4.2(b) is tighter than constant gradient-norm bounds because it allows the deviation to vanish near the optimum/stationary point. This leads to provably tighter bounds in low-heterogeneity regimes.

- **Empirical advantage widens with heterogeneity.** Figure 3d shows that the performance gap between DSpodFL and baselines grows as α = β decreases (higher resource heterogeneity), which is exactly the regime the framework is designed for. This targeted behavior supports the core motivation rather than showing uniform blanket improvement.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Time-varying experiments are relegated to the appendix.** The paper's title, abstract, and contributions explicitly emphasize *time-varying* computation and communication as a central novelty. Yet all main-body experiments (Sec. 5, Figures 2–4) use d_i and b_ij that are "held constant over iterations k." Time-varying results appear only in Appendix O. This creates a direct disconnect between the headline claim and the evidence presented in the main paper. Readers cannot assess the core dynamic claim without chasing the appendix.

- **All baselines are special cases of DSpodFL.** DGD, Randomized Gossip, Sporadic SGDs, and DFedAvg are all identified in Figure 1 as special cases of the proposed framework. No baseline is drawn from outside the DSpodFL family (e.g., asynchronous DFL methods). This makes it impossible to attribute the empirical gains unambiguously to the framework's generality: the advantage could arise purely from DSpodFL having more degrees of freedom in choosing d_i and b_ij, while each baseline is constrained to a degenerate configuration. At least one non-special-case baseline is needed to validate that the framework itself provides value beyond its own restricted variants.

- **The delay metric is self-referential and not calibrated to real systems.** The key empirical metric τ_total^(k) = τ_trans^(k) + τ_proc^(k) is defined as normalized sums involving 1/b_ij and 1/d_i — the very parameters that define DSpodFL's flexibility. This means the delay model is constructed around the framework's own parameterization. The paper does not show what real physical quantity this corresponds to, nor validate it against an actual network simulator or testbed. The "consistently improved training speeds" claim in the abstract therefore rests on a metric that may inherently favor the proposed method by design. Appendix P.3 provides some discussion, but the justification is not grounded in a concrete communication model.

### Minor

- **Independence assumptions (Assumption 4.3) may be overly idealized.** The framework requires that gradient noises ε_i^(k) are uncorrelated across clients, indicator variables v_i^(k) are uncorrelated across clients, and link indicators v̄_ij^(k) are uncorrelated across links. In wireless or resource-constrained environments — the paper's stated motivation — node and link availabilities are often spatially and temporally correlated (e.g., a congestion event knocks out multiple nearby links simultaneously). The paper does not discuss how severe violations of this assumption would affect the guarantees, nor does it acknowledge this as a limitation despite it being the most consequential assumption for real-world applicability.

- **d_min-dominance in convergence bounds.** Both Theorem 4.11 and Theorem 4.12 depend on d_min = min_{k,i} d_i^(k), the minimum SGD probability over all clients and all iterations. If even one client rarely computes (low d_i), the optimality/stationarity gap can be large and the allowed learning rate shrinks. This means the bound is controlled by the weakest client, which is analytically conservative and, more importantly, is never discussed. The paper should characterize the practical regime (concrete values of d_min, ρ̃) where the bounds remain non-vacuous, and acknowledge when the guarantee becomes too loose to be actionable.

- **No concrete guidance for choosing sporadicity parameters.** The framework models v_i^(k) and v̄_ij^(k) as exogenous, and experiments simply sample d_i and b_ij from fixed distributions. The paper never proposes a rule, heuristic, or optimization criterion for how a client should choose its participation probability given resource constraints. Given that the practical motivation is resource-aware adaptation, this is a notable gap between the theoretical framework and any deployable system.

### Tiny

- **Non-convex theorem constants are opaque.** Theorem 4.12 features constants w_1, ..., w_5 involving multiple free scalars Γ_i. While the existence of such constants is typical for non-convex results, the paper provides no intuitive interpretation, and practitioners cannot easily translate the bound into actionable choices of α or network design parameters.

- **The "arbitrary indicator random variables" framing slightly overstates generality.** Footnote 2 confirms that v_i^(k) and v̄_ij^(k) are Bernoulli for each k, and Assumption 4.3(b) requires independence across clients and links. "Arbitrary" more precisely means "time-varying Bernoulli with independently chosen expectations" — a clarification that would prevent misreading.

---

## Nice-to-Haves

- **Move time-varying experiments to the main body** and use them to illustrate where DSpodFL uniquely excels over static-heterogeneity baselines.
- **Include at least one asynchronous DFL baseline** (e.g., from Srivastava & Nedic 2011 or Even et al. 2024) to demonstrate that gains extend beyond comparisons within the framework's own special cases.
- **Provide a simple resource-aware policy** (even a threshold heuristic such as "skip SGD if estimated processing time exceeds T") to illustrate how d_i could be set in practice.
- **Replace or supplement the custom delay metric** with a network-simulator measurement or at minimum a clear mapping to a standard queuing model (e.g., M/D/1 link delay).
- **Provide a side-by-side rate comparison table** between DSpodFL's bounds and exact rates for DGD/DFedAvg under matched assumptions to make the "cost of sporadicity" quantitatively transparent.
- **Show failure modes** — e.g., configurations where DSpodFL does not improve over baselines (very low d_min, poorly connected graph) — to give a more complete picture and build trust that results are not cherry-picked.

---

## Removed Points

*These points are flagged to be removed; treat them with caution — they may reflect misreadings of the paper or standards not appropriate for this setting.*

- **"Arbitrary sporadicity" is overstated because of Bernoulli structure:** Binary indicator variables are, by definition, Bernoulli {0,1}. The paper correctly uses "arbitrary" to mean time-varying expectations d_i^(k), b_ij^(k), not that the marginal distribution is non-parametric. This criticism misreads the framework.
- **Step-size condition requiring knowable constants:** The harsh critic asks whether Proposition 4.10's constants are estimable in practice. For a theory paper establishing convergence existence, this is not a required standard; practitioners tune learning rates empirically regardless.
- **Spectral radius argument for time-varying matrices:** Under the constant step size of Theorem 4.11, Φ^(k) is replaced by worst-case constant Φ, making the product convergence argument valid. The concern about time-varying Φ^(k) does not apply to the stated theorem.
- **No statistical significance tests:** Single-run or small-replicate evaluation is standard in distributed learning system papers at ICLR. The paper does report mean ± 1σ across multiple runs, which is consistent with community norms.
- **FMNIST/CIFAR10 being "modest" for ICLR:** These are standard benchmarks for decentralized FL convergence papers; the contribution is primarily theoretical and the experimental design is appropriate for the scope.
- **Eq. (10) being dimensionally incorrect:** This is most likely a PDF parsing artifact (the vector/bracket rendering); it should not be treated as a mathematical error without access to the original LaTeX.
- **DFedAvg baselines not being fairly tuned:** The paper explicitly sets D = ⌈(1/m)Σ_i 1/d_i⌉, which is the natural round-trip time under the delay model and constitutes a principled, if arguable, tuning choice. Labeling this as unfair requires showing a better D, not merely raising the question.
- **Eq. (5) missing a caveat about masked gradients:** The paper does note that ḡ^(k) = (1/m)Σ_i g_i^(k) v_i^(k), which is an average of masked stochastic gradients. The subsequent analysis handles this correctly; the "missing caveat" is not actually missing.

---

## Novel Insights

The most practically important insight not fully exploited in the paper is the *d_min tension*: the framework's headline strength — accommodating extremely weak or rarely-participating clients — is simultaneously its theoretical Achilles heel, since the convergence bound degrades as d_min → 0, precisely the clients the framework is supposed to help. This creates a regime where the framework is most relevant empirically (high heterogeneity, some very weak clients) but where the theoretical guarantee is weakest. Articulating the crossover point — what d_min, ρ̃, and system size make the bound non-vacuous — would be a concrete theoretical contribution that none of the reviewers developed but the paper would benefit greatly from.

---

## Suggestions

1. **Move time-varying experiments to the main body** (e.g., replace or supplement Figure 4 with Appendix O content). The time-varying case is the core novelty; it must appear in the main paper.
2. **Add one non-special-case baseline** — an asynchronous DFL method — to demonstrate that gains are not simply artifacts of within-framework comparisons.
3. **Ground the delay metric** in a concrete physical model (e.g., link capacity and packet size → transmission time), or validate with a network simulator on at least one scenario. Alternatively, include an iteration-count plot alongside the delay plot so readers can decouple algorithmic benefit from delay-metric effects.
4. **Add a "practical regime" subsection** characterizing specific d_min and ρ̃ values for which Theorem 4.11's bound is non-vacuous (e.g., optimality gap < ε for given ε) with concrete numerical examples.
5. **Acknowledge independence limitations** in the main limitations section (Sec. 6), and briefly discuss whether the empirical results suggest robustness to correlation (e.g., if all clients in one neighborhood simultaneously drop out).
6. **Provide at least a sketch of a practical decision rule** for d_i and b_ij (e.g., an expression that trades off the optimality gap reduction from larger d_i against the delay cost in τ_proc). Even an informal discussion would meaningfully bridge the theory–deployment gap.

---

**Overall evaluation:**
- *Novelty:* High — the joint treatment of sporadic SGD and aggregation in decentralized FL is new and technically non-trivial.
- *Technical soundness:* Good — the proof structure (average model error + consensus error coupled via 2×2 linear system) is elegant and the recovery of known rates is verifiable. Minor concerns around d_min-dominance and independence assumptions do not invalidate the core results.
- *Empirical support:* Moderate-to-weak — the experimental design has two significant gaps (time-varying dynamics not in main body; all baselines are in-family) that prevent the experiments from fully supporting the claimed contributions. The custom delay metric compounds this concern.
- *Significance:* Moderate — the unification is useful for the community and the milder graph connectivity assumption is a genuine improvement, but the practical deployment pathway is underspecified.
- *Clarity:* Good in the algorithmic and theoretical sections; the experimental section needs restructuring to align with the paper's stated contributions.