## Summary
PFM-Net is a learning-based automated mechanism design framework that achieves exact truthfulness by construction. The core theoretical contribution is Theorems 3.4–3.5, which show that the class of truthful direct mechanisms is equivalent (up to measure zero under the prior) to the class of full-menu mechanisms with convex pricing functions satisfying a pricing-rule decomposition — a generalization of Rochet (1987) and Hammond (1979) to the multi-player quasi-linear setting with regularization terms and flexible allocation constraints. This characterization motivates parameterizing the pricing function using convex neural network architectures (PICNN, GroupMax, MoA, LSE), enabling optimization over the truthful class without regret-penalty heuristics. Experiments on single-buyer revenue maximization (up to 20 items) and a social-planner market-design problem demonstrate improvements over discretization-based baselines, particularly at higher dimensionality.

---

## Strengths

- **Exact truthfulness by construction, not by penalty.** Unlike RegretNet-style approaches that approximate DSIC via penalty terms with residual violations, PFM-Net's parameterization lies entirely within the characterized truthful class by design. This is a principled and economically meaningful distinction that most learning-based AMD papers cannot claim.

- **Non-trivial generalization of the classical Rochet/Hammond characterization.** The paper extends the convex-pricing-function characterization to multi-player settings where players interact through a shared mechanism conditioned on others' types, where allocations can be negative, where a player-specific regularization term $c_i(x_i)$ is embedded in the valuation, and where the platform's utility may depend on the true type profile. These are concrete and non-trivial additions over prior single-player/simpler-type results.

- **Architecture-level support for convex constraint enforcement.** The use of PICNN and GroupMax architectures — which produce functions convex in $x_i$ by construction while conditioning arbitrarily on $t_{-i}$ — is an elegant and practically sound choice that bridges mechanism design theory and modern deep learning.

- **Empirical evidence of scaling beyond discretization baselines.** Table 1 shows GroupMax-3 consistently and increasingly outperforming UM-GemNet as $m$ grows (e.g., matching Bundle-OPT for $m \leq 5$ but diverging meaningfully at $m = 15, 20$), directly supporting the claim that discretization-based menus plateau while continuous convex parameterizations can capture additional structure. The observation that UM-GemNet collapses to near-Bundle-OPT behavior for large $m$ is an interesting and specific finding.

- **Broader applicability to non-auction settings.** The social-planner experiment (Table 2) goes meaningfully beyond standard revenue maximization, demonstrating the framework applies to welfare-maximizing settings with market-clearance penalties and two-sided allocations — a design space that most prior AMD learning papers do not address.

---

## Weaknesses

### Fatal
None.

### Major

- **No comparison with regret-based baselines (e.g., RegretNet).** The introduction explicitly and centrally positions PFM-Net against regret-based methods, characterizing them as producing "unpredictable and unstable" mechanisms. Yet no regret-based method appears in Tables 1 or 2. Without this comparison, it is impossible to evaluate the practical trade-off between the paper's core selling point (exact truthfulness) and the potential utility cost of restricting to the characterized class. This gap directly undermines the paper's main positioning argument. At minimum, the regret of PFM-Net (theoretically zero) should be compared against the IC violations and revenues of RegretNet in standard settings.

- **No empirical verification that truthfulness holds numerically after training.** The exact-truthfulness guarantee holds for the parameterized mechanism class in theory, but in practice, inference requires solving $\arg\max_{x_i \in \mathcal{X}_i} u_i(\cdot)$ numerically, and training uses finite samples with a penalized bi-level objective. Any approximation in either step introduces IC/IR violations. The paper never measures max regret or IR violations on a held-out type grid after training. For a method whose primary differentiator is exact truthfulness, the absence of numerical IC verification is a critical omission.

- **PICNN is excluded from the universal approximation result (Proposition 5.1) without acknowledgment.** The main text lists MoA, LSE, PICNN, and GroupMax as implemented architectures, and Section 5 argues universal approximation covers the method. But Proposition 5.1 explicitly lists MoA, LSE, GroupMax, and GroupLSE — omitting PICNN. If PICNN does not share the same approximation guarantee, this must be stated clearly. As written, a reader cannot tell whether PICNN results in Tables 1–2 have the same theoretical backing as GroupMax results.

### Minor

- **Multi-player experiments are limited to $n \leq 3$.** The paper claims "general multi-player mechanism design," and the single-buyer experiments (Table 1) test only $n=1$. The social-planner experiment reaches $n=3$ at most. Whether the method scales gracefully in the number of agents — a central challenge in AMD — remains unanswered empirically.

- **Training details and convergence analysis are almost entirely in the appendix.** For an ML venue, the training procedure — the objective function, the alternating optimization between platform and player allocations, the penalty schedule, and convergence criteria — should be accessible in the main paper. The alternating optimization has no convergence guarantee, and the main text provides no empirical evidence of stable convergence across seeds or settings.

- **Efficiency claim is unsupported without computational cost data.** The paper claims "efficiency" as one of three headline properties, yet no training time, inference time, or memory comparison against baselines is reported. Inference requires a per-player convex optimization at test time, which is more expensive than the forward pass of discretization-based methods. Without wall-clock comparisons, the efficiency claim cannot be assessed.

- **Table 2 column headers appear inconsistent.** The table caption describes experiments with 1, 2, and 3 players, but the header row lists six columns as $P_{2,5}^U, P_{2,5}^N, P_{2,5}^U, P_{2,5}^N, P_{3,5}^U, P_{3,5}^N$ — repeating $P_{2,5}$ twice. The first two columns are almost certainly $P_{1,5}^U$ and $P_{1,5}^N$. This presentation error makes the table difficult to interpret.

- **VCG yielding 0 under uniform distribution in Table 2 requires explanation.** VCG = 0 in the $P_{n,5}^U$ columns while being positive under normal distribution. It is plausible that under symmetric types ($U([-1,1])$) and the market-clearance-penalized social welfare objective, VCG (which ignores the clearance penalty in its allocation rule) consistently loses all gained welfare to penalty. But this behavior is surprising and unexplained, potentially suggesting a mismatch between the VCG implementation and the problem definition rather than a genuine failure of VCG in this domain.

### Tiny

- **The normalization trick enforces $f_i(\mathbf{0}; \cdot; \theta) = 0$ (equality) when no-buy-no-pay only requires $\leq 0$.** The paper should explicitly note this is without loss of optimality since the normalization is a free additive constant. This is technically harmless but leaves a small logical gap in the presentation.

- **Consistency of notation between $p(t;\theta)$ (Section 2) and $p_i(x_i; t_{-i}; \theta)$ (Section 4)** — the parameterization notation shifts form without explicit reconciliation, which could confuse a reader tracking how the AMD formulation connects to the PFM-Net representation.

---

## Nice-to-Haves

- **Empirical IC violation heatmap over type space.** Even if truthfulness holds by construction in theory, visualizing the max utility gain from misreporting over a grid of types for a trained PFM-Net (which should be near zero) — compared to a trained RegretNet (which will have residual violations) — would powerfully illustrate the paper's core claim and provide reassurance that numerical approximation errors are negligible.

- **Visualize learned pricing functions in the main text.** The paper's method is defined by convex pricing surfaces, but no such visualization appears in the main body. Showing $f_i(x_i; t_{-i}; \theta^*)$ as a function of allocation for representative type profiles (especially alongside the analogous discretized menu from UM-GemNet) would vividly communicate what "non-trivial components" the method learns.

- **Ablation on convex architecture choice.** Table 1 shows PICNN-1, GroupMax-1, and GroupMax-3 behaving differently (e.g., GroupMax-1 dominates at $S_2, S_3$ while GroupMax-3 wins at $S_{10}+$), but no analysis explains why. Architecture guidance for practitioners is absent.

- **Clarify formal sense of generalization beyond Rochet/Hammond in the main text.** The paper claims this in a footnote (footnote 8) and refers to Appendix A. A brief proposition or 2–3 sentence argument in the main text explaining the specific dimensions of generalization (regularization term $c_i$, negative types/allocations, multi-player conditional menu, compact convex type spaces) would strengthen the theoretical contribution.

- **Expand multi-player experiments** to $n \geq 5$ (e.g., market-design settings) to test whether the method degrades with strategic complexity.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Critic's concern about measure-zero equivalence conflicting with pointwise DSIC/IR.** The equivalence in Definition 3.1 is standard in mechanism design theory, and using prior-induced measure-zero exceptions is a well-established convention in the Myerson/Rochet tradition. The theorems clearly restrict to IC/IR classes as inputs and outputs, so the measure-zero treatment is appropriate and not a logical gap.

- **Critic's concern about correlated types not discussed.** The paper explicitly states the prior $\mathcal{F}$ may be correlated and notes dominant-strategy IC is used. Since DSIC is independent of the prior's correlation structure, no further discussion is needed.

- **Critic's concern about the platform's utility depending on true types being problematic.** Under a truthful mechanism, reported types equal true types in equilibrium, so the platform observing $\mathbf{t}$ and computing $v_0(\mathbf{x}; \mathbf{t})$ is unambiguous. This is not a modeling error.

- **Critic's demand for confidence intervals and multi-seed error bars.** Single-run evaluation is standard practice in learning-based mechanism design (RegretNet, GemNet, and all cited baselines report point estimates). Requiring error bars would be imposing norms not standard in this field.

- **Critic's concern about player-by-player menu decomposition under joint platform utility.** In the paper's model, each player's utility depends only on $(x_i, p_i, t_i)$, so the player-by-player IC decomposition is valid. The platform's utility $v_0(\mathbf{x}; \mathbf{t})$ depends on the full allocation profile but this does not affect the IC characterization per player. The critic's concern is not a real gap.

- **Criticism about Theorem 5.4's strong convexity assumption being impractical.** The paper explicitly addresses this: "we note that this is only a technical condition, which is not strong because $\varepsilon_1$ can be chosen so small that strong convex function can be arbitrary close to any convex function in bounded domain." This is a reasonable (if informal) addressal; the assumption is standard in approximation theory contexts.

- **Critic's concerns about "new paradigm" / "mainstream method" / "unpredictable and unstable" being informal language.** These are presentational choices, not substantive errors.

- **Critic's concern about model scope being narrower than claimed.** The quasi-linear model with bilinear hidden-type structure is explicitly stated and footnoted in Section 2. Criticizing it as an unstated restriction is incorrect.

---

## Novel Insights

The most genuinely novel observation from the reviews is the following: the paper's true differentiation may not be adequately captured by the main experiment (single-buyer revenue maximization), where gains over UM-GemNet are modest. The more compelling evidence of the method's value is the *qualitative* observation that GroupMax learns pricing rules that vary non-trivially with other players' types (Section 6.3, Appendix G.3), while UM-GemNet collapses to near-bundle behavior in high dimensions. This structural learning — pricing that adapts to the type context of the multi-player interaction — is precisely what the theory predicts a full-menu conditional mechanism should do, and is something that neither AMA-based nor discretization-based approaches can represent. A stronger framing of the paper should center this insight, supported by pricing function visualizations comparing the learned conditional pricing structure across methods, rather than leading with modest revenue improvement numbers.

---

## Suggestions

1. **Add RegretNet as a baseline across all experiments.** Report both platform utility and empirical max-regret for all methods. This is the single most important missing element given the paper's framing.

2. **Report empirical IC verification (max regret) for PFM-Net after training,** using a dense grid or adversarial search over misreports. If numerical optimization is sufficiently accurate, max regret should be near machine epsilon; demonstrating this would powerfully distinguish the method.

3. **Resolve the Table 2 header error** (repeated $P_{2,5}$ columns — likely $P_{1,5}$) and add a brief explanation of why VCG = 0 under uniform distribution in that experiment.

4. **Add PICNN to Proposition 5.1 with a citation or proof**, or explicitly state it does not satisfy the same universal approximation guarantee and explain implications for the experimental results.

5. **Report wall-clock training and inference times** for PFM-Net and UM-GemNet under the same hardware, especially for $S_{15}$ and $S_{20}$, to back the efficiency claim.

6. **Move key training algorithm details** (objective function, penalty schedule, convergence criterion) into the main text, even if condensed, to enable reproducibility without relying on the appendix.

7. **Include a pricing function visualization in the main text** showing how $f_i(x_i; t_{-i}; \theta^*)$ varies with allocation and conditioning type, contrasted with UM-GemNet's discrete menu — this is the most intuitive demonstration of PFM-Net's structural advantage.