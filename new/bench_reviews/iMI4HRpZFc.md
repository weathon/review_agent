## Summary

This paper introduces a diagnostic taxonomy of delusions (distinct from hallucinations) in target-directed RL, categorizing problematic targets as G.1 (nonexistent/invalid) and G.2 (temporarily unreachable), and estimator failures as E.0–E.2. Building on this framework, the authors propose two novel hindsight relabeling strategies—“generate” (JIT generator-sampled relabeling) and “pertask” (cross-episode relabeling)—and a hybrid two-slotted approach that feeds different relabeled data to the generator and estimator. Experiments on a custom gridworld environment (SwordShieldMonster, or SSM) demonstrate that hybrid strategies improve OOD success rates over standard HER baselines.

## Strengths

- **Conceptual taxonomy distinguishing generator and estimator failures.** The paper provides a useful diagnostic vocabulary (G.1 vs. G.2 targets; E.0–E.2 estimator delusions) that decomposes target-directed failures beyond the standard “hallucination” framing, grounded with clear SSM visualizations (Sec. 3, Fig. 2).
- **Novel relabeling strategies with a principled hybrid architecture.** The “generate” and “pertask” strategies target specific delusion types, and the two-slotted design—decoupling generator and estimator training data—is a sensible response to their conflicting needs (Sec. 4.1–4.3, Table 1).
- **Clean diagnostic testbed with ground-truth measurement.** SSM enables precise quantification of delusional estimation errors and behavior frequencies by equivalence class, which is difficult in standard benchmarks where target feasibility is opaque (Sec. 2, Fig. 2).

## Weaknesses

### Fatal
None.

### Major
- **Figure 3 contains critical mismatches between text claims and figure contents that undermine the causal claims.** Fig. 3(f) is captioned as showing only atomic strategies (F-E, F-P, F-G), yet the main text states that hybrid strategies “F-(E+P)” and “F-(E+P+G)” achieved “particularly significant improvement in accuracy in E.2 delusional estimates in f)” (Sec. 5.5). Similarly, Fig. 3(g) (“E.2 Behavior Ratio”) is captioned as showing F-(E+G) and F-(E+P+G), but the text claims that “F-(E+P) … addressed the most delusional behaviors (in g))” (Sec. 5.5). Because F-(E+G) is explicitly described elsewhere as a strategy “mostly for E.1” (Sec. 5.4), its presence in the E.2 behavior panel and the absence of F-(E+P) there break the logical link between proposed strategies and their purported behavioral effects. These discrepancies prevent readers from verifying the paper’s core causal claims at the behavior level.
- **Empirical validation in the main text is too narrow to support claims about general target-directed agents.** The main body presents only a single experimental set (Skipper on SSM). While the paper mentions four total sets across two environments and two methods (Sec. 5.6), three are relegated to the appendix. OOD generalization is operationalized solely as varying lava density within the same grid topology. Without at least one additional main-text experiment on a standard or distinct domain, broad claims about applicability to “general target-directed agents” (Sec. 1, Sec. 4) are inadequately supported.

### Minor
- **Hand-tuned mixture proportions lack justification or sensitivity analysis.** The hybrid strategies use fixed proportions (50/25/25, half/half) without ablation or analysis of how mixture weights affect the tradeoff between delusional and non-delusional accuracy (Sec. 4.2, Sec. 5.4).
- **No statistical significance testing reported for OOD performance.** Fig. 3(h) shows confidence intervals for OOD success rates, but the text does not report formal significance tests (e.g., t-tests, Mann-Whitney U) for the gaps between hybrid and baseline strategies (Sec. 5.5).
- **G.2 taxonomy lacks formal grounding beyond irreversible environments.** The paper notes that G.2 targets “do not exist in all MDPs” (Sec. 3.1.2) and derives the category from SSM’s irreversible state structure. Without formal necessary conditions or validation on reversible environments, the G.2 concept’s generality remains uncertain.

### Trivial
None.

## Nice-to-Haves
- Evaluate on standard goal-conditioned benchmarks (e.g., Fetch) to validate that G.2 delusions arise and that the proposed strategies help outside gridworlds.
- Replace hand-tuned mixture weights with online selection (e.g., bandits) based on estimation error signals.
- Sensitivity analysis on the mixture proportions used in hybrid strategies.
- Visualize estimator prediction landscapes on a continuous-control domain to demonstrate interpretability beyond gridworlds.

## Removed Points
These points are flagged to be removed; treat them with caution.

- **“The psychiatric analogy in Fig. 1 adds no technical content”** — Removed as a pure style nitpick; the analogy is presentation, not a technical flaw.
- **“Zhao et al. (2024) undermines novelty of SSM”** — Removed as a misread. The paper never claims SSM is the hardest possible environment; it explicitly positions SSM as a diagnostic testbed with ground-truth delusion cases.
- **“G.1/G.2 split feels arbitrary; G.2 is functionally a hallucination”** — Removed as misunderstanding the paper. The paper explicitly distinguishes G.1 (invalid/nonexistent targets) from G.2 (valid but temporarily unreachable targets), which is a substantive conceptual difference with different training implications.
- **“Train-test mismatch on initial state distribution”** — Removed. Fixing evaluation initial states to maximize difficulty is by design; the OOD variable is lava density, not initial state distribution.
- **“Missing related works (RAPID, HAC, etc.)”** — Not raised, per instructions.
- **Comments on formatting, typos, or appendix-deferred proofs** — Removed per hard rules.

## Novel Insights

The paper’s core insight—that generator and estimator have *conflicting* data needs and should be fed from independent relabeling distributions—is a genuinely useful design pattern for hierarchical RL training (Sec. 4.3). The two-slotted approach is simple but underappreciated in hindsight relabeling literature, and the paper’s granular decomposition (estimation error → behavior frequency → OOD success) provides a template for diagnosing target-directed failures more broadly.

## Suggestions
- Correct the Fig. 3 caption or text references so that behavior panels (c, g) align with the strategies they are meant to demonstrate, and ensure panel (f) is accurately described in the text.
- Move at least one additional experimental set (e.g., LEAP or the second environment) into the main body, even if abbreviated, to broaden empirical support.
- Add formal statistical tests for the OOD performance gaps in Fig. 3(h).

## Context for Scorer

- **Original reviewer signal:** The Harsh Critic found the empirical scope too narrow, Fig. 3 mismatched, and the taxonomy SSM-specific. The Strength Finder praised the diagnostic taxonomy, novel relabeling strategies, two-slotted architecture, and SSM diagnostic value.
- **What was dropped and why:** Several criticisms were removed for misreading the paper: (1) the claim that Zhao et al. undermines SSM’s novelty (the paper never claims SSM is the most challenging environment); (2) the claim that G.2 is “functionally indistinguishable from hallucinations” (the paper explicitly distinguishes them on valid-state status and temporal reachability); (3) the “train-test mismatch” on initial states (the OOD variable is lava density, and fixing evaluation initial states is by design); (4) formatting/typos and missing-related-work critiques.
- **Cross-checks performed:** I directly read Fig. 3’s caption and Sec. 5.4–5.5 to verify the Harsh Critic’s claim about mismatched strategy-behavior comparisons. The caption lists Fig. 3(f) as showing only F-E, F-P, F-G; the text claims hybrids F-(E+P) and F-(E+P+G) improve in panel (f). The caption lists Fig. 3(g) as showing F-(E+G) and F-(E+P+G); the text claims F-(E+P) performs best in panel (g). These mismatches are real and substantive.
- **Severity read:** The surviving weaknesses are primarily one Major structural issue (Fig. 3 text/figure mismatches breaking causal verification) and one Major scope issue (narrow main-text experiments relative to broad claims). Neither alone invalidates the entire paper—Fig. 3(h) still supports overall OOD improvement—but together they significantly weaken the evidentiary chain for the core mechanism claims. The paper is a reasonable conceptual contribution hampered by serious presentation/figure errors.
- **Anything else load-bearing:** The paper explicitly scopes itself as a conceptual framework with “necessary conditions” rather than a fully general theory, and acknowledges that G.2 targets do not exist in all MDPs. The appendix contains 3/4 experimental sets that may partially address the narrow-empiricism concern, but they are inaccessible in the parsed version.