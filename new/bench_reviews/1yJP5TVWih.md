## Summary
This paper studies rank collapse in sequence models through a unified framework covering both transformers and SSMs, and proposes a parametrized residual pathway, the lambda-skip connection, as a mechanism to mitigate it. Its main technical contribution is a lower bound on the rank-collapse metric under conditions on \(\lambda\), together with analyses of collapse in ablated settings and experiments probing skip strength and gating in transformer/Mamba-style architectures.

## Strengths
- The paper tackles an important and timely problem. Rank collapse has become a meaningful theoretical and practical concern, and extending the discussion beyond transformers to SSMs is a worthwhile objective.
- The unified formulation in Section 3 is conceptually useful: expressing both attention and SSM blocks through \(O^{(k)} = M^{(k)}V^{(k)}\) gives the work a coherent cross-architecture lens rather than treating transformers and SSMs as unrelated cases.
- The paper does provide a nontrivial theoretical result in Theorem 4.1: a sufficient condition on \(\lambda\) yielding a layerwise lower bound on \(\mu(Y^{(K)})\). This is accompanied by additional analysis rather than a single isolated theorem.
- The necessity-oriented discussion is more substantive than usual for papers of this kind: the manuscript includes collapse results in ablated settings, constructive examples where some \(\lambda\) still lead to collapse, and a tightness discussion via Proposition 4.3.2.
- The empirical observation that gating mechanisms correlate with avoiding rank collapse in Mamba-2-style models is interesting and potentially valuable, even though the current evidence is not yet fully conclusive for causal practical claims.
- The paper is generally clear about at least some of its limitations, especially the omission of gating from the theoretical treatment and the conservativeness of the sufficient condition.

## Weaknesses

###: Fatal

### Major:
- **The central “prevention” claim is overstated relative to what Theorem 4.1 actually proves.**  
  The theorem is a finite-depth lower bound, not a blanket architectural guarantee that rank collapse is eliminated. It requires an explicit input condition, \(\mu(Y^{(0)})^2 \ge b\), where \(b\) itself depends on \(K\), \(\lambda\), and model constants. Moreover, the theorem only guarantees \(\mu(Y^{(K)})^2 \ge a^K \mu(Y^{(0)})^2\). In the paper’s own Remark 4.1, for the important Mamba case, “the only possible choice for the collapse rate is \(a<1\),” which still permits exponential decay with depth. So while the theorem is meaningful, statements such as “guarantee prevention of rank collapse” and “rank collapse does not occur” are too strong unless carefully qualified as finite-depth, input-conditioned, and architecture-constant-dependent.
- **The theoretical treatment of SSMs omits a central component of the practical architecture—gating—creating a real theory/practice gap.**  
  Section 3.1 explicitly says gating is ignored in theory “for simplicity,” yet Section 5.2 later argues empirically that gating plays a crucial role in preventing rank collapse. Since gating is not incidental in Mamba-style architectures, the paper’s broad SSM-facing claims are weaker than presented: the analysis is about an ablated/simplified SSM block, not the full architecture practitioners actually use. The authors do acknowledge this in the limitations, which is good, but it remains a substantive limitation of the paper’s claimed generality.
- **The empirical evidence does not adequately support the stronger practical narrative that lambda-skip connections are a useful architectural intervention in trained real systems.**  
  The main probing experiments modify pretrained models at inference time: e.g., Section 5.1 uses a pretrained Mamba-2, removes gating, inserts additive skip connections with various \(\lambda\), and measures \(\mu\) on Wikipedia excerpts. This is informative as a diagnostic probe of the mechanism, but it is not strong evidence that the proposed architecture is beneficial when actually trained as such. Likewise, the gating ablations in Section 5.2 show that removing gating or LayerNorm from a pretrained model changes \(\mu\), but that is confounded by the fact that the model was trained with those components. Table 1 is the only training-based evidence, yet it reports task accuracy rather than rank-collapse dynamics and gives mixed results.
- **The practical usefulness of the main bound is limited by its conservativeness and input dependence.**  
  The paper itself states in Section 5.1 that “our condition on \(\lambda\) in Theorem 4.1 is too conservative, in practice much lower values of \(\lambda\) are good enough.” This is an honest admission, but it also means the main theorem currently offers weak guidance for choosing \(\lambda\) in practice. The dependence of \(b\) on \(1/a^K\) also makes the input condition harder to interpret or verify as depth grows.

### Minor
- **The experimental evaluation is narrow for the scope of the practical claims.**  
  Table 1 includes only two tasks, and the results are mixed rather than consistently favorable. For example, variable \(\lambda\) slightly helps some settings but hurts others, including Mamba-2 on LRA Image. This does not negate the theoretical contribution, but it does weaken claims of broad practical benefit.
- **The necessity framing in Section 4.2 is somewhat stronger than what is actually established.**  
  To the paper’s credit, it explicitly says it does not provide a formal necessary condition. Still, the section title and surrounding narrative can leave a stronger impression than warranted, because the evidence consists of imported transformer results, restricted SSM results, and hand-crafted examples rather than a genuine necessity theorem.
- **The theoretical setup uses a simplified LayerNorm model, while the experiments involve practical architectures with more complex normalization behavior.**  
  This simplification is not unreasonable for analysis, but it should be framed more carefully when making claims about real architectures.

### Trivial
- The notation around Eq. (6) is a bit awkward, with layer indexing shifted relative to the earlier per-layer definitions, which mildly hurts readability. This is not a substantive flaw.

## Nice-to-Haves
- Include training-time measurements of rank collapse, gradient norms, and loss stability for models trained with different fixed/learned \(\lambda\), to connect the theory more directly to optimization behavior.
- Report the learned \(\lambda\) values in the variable-\(\lambda\) experiments and compare them to the theoretical sufficient condition.
- Quantify the gap between the theorem’s lower bound and the empirical \(\mu\) curves to make the conservativeness of the theory more precise.
- Extend the theory to include gating, even under a simplified multiplicative model, since that is the most relevant missing architectural component for SSMs.
- Clarify more explicitly when the input condition \(\mu(Y^{(0)})^2 \ge b\) is expected to hold in practice.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Lambda-skip connections are not novel because parametrized skip connections existed before.”**  
  Removed as a core weakness. The paper does not really claim the architectural primitive itself is wholly novel; its claim is the rank-collapse analysis and guarantee around that mechanism. This is better treated as scope clarification than a substantive criticism.
- **Strong criticism of missing variance/confidence intervals or more exhaustive reproducibility details for Table 1.**  
  Removed. These are not central flaws here, and such requests are not essential for assessing the paper’s main claims.
- **Any criticism doubting the existence/availability of cited models such as Mamba-2 or benchmarks used in the paper.**  
  Removed per instruction.
- **Overly strong complaint that Eq. (6) indexing is “wrong.”**  
  Removed. The indexing is awkward but interpretable; this is a readability issue, not a factual or fatal flaw.

## Novel Insights
The paper’s most interesting underlying tension is that its strongest practical empirical observation is about gating in Mamba-style models, while its strongest theory is about lambda-weighted additive skips in architectures where gating is omitted. That mismatch suggests the deeper scientific contribution may not yet be “lambda-skip connections prevent rank collapse” so much as “rank collapse in modern sequence models is governed by identity-preserving pathways more broadly, with additive residuals and multiplicative gating playing partially analogous roles.” If developed explicitly, that broader framing could unify the paper’s theory and experiments much better than the current presentation.

## Suggestions
- Reframe the main claim more precisely: say Theorem 4.1 gives a **finite-depth sufficient condition for avoiding severe collapse** or maintaining a nonvanishing lower bound, rather than a general unconditional prevention guarantee.
- Tighten the theorem if possible, or at minimum analyze the conservativeness gap quantitatively and explain how practitioners should use the result despite that gap.
- Add training-based experiments that measure both rank-collapse metrics and downstream optimization behavior for models actually trained with fixed or learned \(\lambda\).
- Incorporate gating into the theoretical framework, even if only under simplifying assumptions.
- Discuss mixed results in Table 1 directly rather than leaving them uninterpreted.
- Report learned \(\lambda\) values and layerwise trends to show whether practice aligns with the theory.

## Score and Decision
**Assessment by axis:**  
- **Originality:** Moderate to good. Extending rank-collapse analysis to SSMs and studying lambda-weighted skips in this context is a real contribution, even if parametrized residuals themselves are not new.  
- **Importance:** Good. Rank collapse is a meaningful problem, and a unified transformer/SSM view is valuable.  
- **Claims support:** Mixed. The core theorem exists, but the presentation overclaims relative to the actual conditions and guarantees. Practical claims are under-supported.  
- **Experimental soundness:** Moderate to weak for practical conclusions. The probing experiments are suggestive but not decisive, and the training-based validation is limited.  
- **Clarity:** Fairly good overall, though some claims should be more carefully stated.  
- **Value to the community:** Moderate. The paper has useful ideas and some interesting evidence, but it falls short of fully convincing support for its headline claims.

**Calibration against retrieved human reviews:**  
I compared this paper primarily against:
- **X6xzYP2cMk, “Mind the Gap: a Spectral Analysis of Rank Collapse and Signal Propagation in Transformers”** (scores 5,5,6,3; reject): another theory-heavy rank-collapse paper whose scope was limited relative to practical architectures. The current paper is somewhat stronger because it broadens to SSMs and includes more cross-architecture synthesis, but it has a similar theory/practice mismatch.
- **hgjpO0H0id, “On the interplay between learning and memory in deep state space models”** (scores 3,3,6; reject): another SSM theory paper criticized for limited applicability to practical models. The current submission is stronger and more relevant, but still shares the gap between simplified analysis and practical architectures.
- **cxKLRM3KhC, “Residual Connections Harm Generative Representation Learning”** (scores 6,5,5,6; reject): related in spirit because it studies modified residual pathways with practical claims that require strong empirical support. The present paper has stronger theoretical substance but weaker direct practical validation.
- On the positive side, I also considered accepted theory-oriented papers like **KlxK4ncqWZ** and **wYxOMEzpkl**, which earned accepts because their claims and evidence were better aligned. This paper does not reach that level of evidential closure.

Overall, this submission is **interesting and nontrivial**, but the combination of overclaimed headline guarantees, omission of gating from the theory, and limited practical validation keeps it below the acceptance bar for me.

**Score: 4.5 / 10**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>