Now let me search for calibration papers to anchor my score.Now I have enough information to write the consolidated review. Let me synthesize all three reviewer inputs against the verified paper content.

---

## Summary

This paper studies the learnability of almost-full degree parities ($k = d - O(1)$, including full parity) by gradient descent on two-layer ReLU networks. The authors demonstrate that initialization is decisive: Rademacher initialization enables efficient learning (even in one GD step for the correlation loss by training only output weights), while Gaussian or perturbed-Rademacher initialization with large enough constant $\sigma$ prevents learning. The core technical contribution is the introduction of the Gradient Alignment (GAL) measure, a novel loss-dependent hardness criterion applicable to single target functions, which is used to prove exponential-time lower bounds for Gaussian-initialized noisy-GD.

---

## Strengths

- **Resolves a genuine open problem.** Prior work (Abbe & Sandon 2020) showed hardness for dense parities when both $k$ and $d-k$ grow, but the almost-full regime ($k = d - O(1)$) was explicitly left open. This paper settles it with a crisp separation based on initialization.

- **Novel hardness measure with broader potential.** The GAL measure is a genuine conceptual contribution: unlike cross-predictability or SQ dimension, it applies to a *single* target function (e.g., the full parity itself), bypassing the need to construct orbit classes. The junk-flow coupling technique for propagating GAL bounds through training dynamics is elegant.

- **Strong and clean positive result.** Theorem 4 shows that under Rademacher initialization, the hidden layer embedding makes almost-full parities linearly separable, and perfect accuracy follows in *one GD step* (output layer only) for correlation loss, or in poly($d$) SGD steps for hinge loss. This is a clean mechanistic insight.

- **Rigorous Gaussian negative result.** Theorem 7 and Proposition 1 give a solid lower bound: any Gaussian-initialized two-layer ReLU network with poly($d$) neurons requires exponential time to learn high-degree parities under noisy-GD with correlation loss, for any poly($d$) time budget. This is among the strongest results in the paper.

- **Honest scope and thorough experiments.** The paper explicitly delineates what is proved versus what is left to future work, and the experimental section genuinely extends beyond the theory to deeper architectures, different losses, and discrete/continuous perturbations.

---

## Weaknesses

### Fatal
*None identified.*

### Major

- **Incomplete perturbed-Rademacher lower bound.** This is the most significant gap. The paper explicitly states (Section 5.2.2, paragraph before Theorem 8): *"Together with a similar bound for the output layer weights (which we omit from this version of the paper) that would give the statement of Theorem 3 also for the σ-perturbed initialization."* This means the extension of the negative result to perturbed Rademacher—a key piece of the "Rademacher is a special case" narrative—is **not fully proved** in the submission. Theorem 8 as stated appears to present a complete result, but the surrounding text makes clear the argument is incomplete. Since this narrative is central to the paper's contribution, this missing piece is a real gap. The authors should either complete the proof (even in an appendix) or retitle and reframe the perturbed-Rademacher results as partial.

- **Asymmetry between positive and negative training regimes.** The main positive result (Section 4.1, Theorems 4–5) trains **only the output layer** with correlation loss, whereas the negative result (Theorem 7) applies to **all-layer** noisy-GD with correlation loss. This means the paper does not have a clean apples-to-apples separation: the same architecture, same loss, same trainable parameters, changing only initialization. The hinge-loss positive result (Section 4.2) does train both layers, but it is deferred almost entirely to the appendix and restricted to full parity only. The paper is candid about this (Section 4.1 explicitly states output-only training), but it substantially weakens the narrative of the introduction and abstract, which present a broad "initialization determines learnability" claim.

### Minor

- **Negative result limited to correlation loss.** The GAL-based hardness proof (Step 3 of the junk-flow argument) is currently restricted to correlation loss, as the authors acknowledge in Remark 3. Since the experiments use hinge loss throughout, and since standard practice uses cross-entropy or hinge losses, the theoretical scope is narrower than the practical message. Figure 2 provides suggestive empirical evidence that GAL remains small under hinge loss as well, but no theory currently covers this.

- **Large gap in the σ regime.** The positive result holds for $\sigma = O(d^{-1})$; the negative result requires $\sigma = \Omega(1)$. A substantial intermediate regime remains unresolved. While the paper acknowledges this and Figure 2 (right) provides some qualitative evidence, no conjecture about the threshold location (e.g., $\sigma = \Theta(d^{-1/2})$?) is offered, leaving the "threshold phenomenon" discussion without grounding.

- **Experimental architecture mismatches theory.** All experiments use a 4-layer MLP with hinge loss trained on all layers, while the theory concerns 2-layer ReLU networks. The paper does say it "explores settings beyond the theoretical analysis," but there are no theory-matched experiments on the 2-layer architecture. This means the experiments support qualitative plausibility but cannot validate or challenge the specific theoretical claims.

### Trivial

- **Large width requirement.** Corollary 1 requires $\Omega(d^4)$ hidden neurons for ReLU and $\Omega(d^2)$ for clipped ReLU. While not tight, this is a consequence of the approach and does not affect the existence/separation result.

- **Loose exponent in Corollary 3.** The bound in Eq. (6) involves $\text{GAL}_f(\theta^0)^{1/18}$, which arises from technical smoothing arguments and is likely quite loose. For moderate $d$ this weakens the bound's practical applicability.

---

## Nice-to-Haves

- **Provide a concrete conjecture (and supporting evidence) for the σ threshold.** A GAL computation or scaling experiment at intermediate σ values could show whether the critical σ scales as $d^{-1/2}$, guiding future work substantially.

- **Phase diagram of learnability vs. (d, σ).** A 2D heatmap varying both $d$ and $\sigma$ would directly test whether the critical $\sigma$ scales as $1/d$ as claimed. Current main-text experiments only show $d = 50$.

- **Scatter plot of GAL at initialization vs. final accuracy across many initializations/seeds.** This would validate whether GAL is genuinely predictive of learning success, making it a stronger empirical tool.

- **Discussion of practical implications.** Standard practice uses Kaiming/He/Glorot initialization. A brief discussion of whether any practical scheme shares the favorable properties of Rademacher for structured targets would increase impact.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **"Extraneous E_x in Definition 2 of GAL"** (Harsh Critic, Section 3): The paper writes $\text{GAL}_f(\theta) := \mathbb{E}_x \|\Gamma_f(\theta) - \Gamma_r(\theta)\|_2^2$, where $\Gamma_f(\theta) = \mathbb{E}_x[\nabla_\theta L(\cdot)]$ is already a full expectation. The outer $\mathbb{E}_x$ is indeed notational redundancy (the argument is constant in $x$), which the harsh critic flags as a "potential presentation issue." However, the paper explicitly attributes this to PDF parsing artifacts in the problem statement, and this is a trivial notational issue that does not affect any theorem or proof. *Removed as pure formatting/notation nitpick.*

- **"Broader 'most typical initializations' claim in conclusion"** (Harsh Critic, Conclusion): The harsh critic notes the conclusion says the full parity is "challenging with most typical initializations" while only Gaussian and partially perturbed Rademacher are rigorously covered. Looking at the paper, the conclusion actually says "gradient descent on neural networks with most typical initializations, with the Rademacher being a special case." This is a judgment call, but the paper includes extensive experiments with discrete and continuous non-Rademacher initializations (Figure 3), making this a broadly supported empirical claim even if the theory is narrower. *Removed as the concern is addressed by the experimental evidence.*

- **"Spark reviewer's claim: no scaling experiments with d"**: The main text uses $d=50$ only, but the paper explicitly says "in the Appendix, we report experiments with larger input dimensions" (Section 6). The claim that the paper does not address scaling is a strawman. *Removed as the paper addresses this in the appendix.*

- **"Spark reviewer's claim: no experiments with correlation loss"**: The paper provides theoretical results for correlation loss and Figure 2 (right) explicitly computes GAL under correlation loss. The demand for full learning experiments under correlation loss is a scope extension beyond what the theory itself requires for validation. *Weakened to nice-to-have.*

---

## Novel Insights

The most genuinely novel insight emerging across all reviewers is the observation—supported by Theorem 4 and Section 4.1—that the Rademacher initialization is "special" not primarily because of optimization dynamics but because it induces a random feature embedding that *already makes almost-full parities linearly separable*. This means the learning phenomenon is more about the structure of the random feature map than about GD finding a good solution; the gradient step is just reading off what the initialization gave. This interpretive point, while somewhat underemphasized by the paper itself, sharpens the distinction from "initialization helps optimization" to "initialization determines representational capacity of the induced random feature map." It also suggests that the GAL measure's power comes from detecting whether the initialization already approximately aligns with the target, not from tracking optimization dynamics per se.

---

## Suggestions

1. **Complete Theorem 8** by including the output-layer GAL bound in the appendix, or reframe Theorem 8 as a partial result and adjust the abstract/introduction accordingly.

2. **Add a matched experiment on 2-layer networks** to directly validate the theoretical predictions for the correlation loss and Gaussian/Rademacher initializations.

3. **Discuss or prove sharpness of the 1/18 exponent** in Corollary 3. Even a discussion of whether $1/2$ is achievable would help contextualize the bound.

4. **Strengthen the σ-gap discussion** with a conjecture backed by GAL computations at intermediate σ values.

5. **Rewrite Section 4.1 framing** to be explicit upfront that output-only training is used, and distinguish this from the full-training setting in the conclusion of the section.

---

## Score Calibration

**Anchor papers compared:**

| Paper | Topic | Key features | Score | Decision |
|---|---|---|---|---|
| HgOJlxzB16 (XOR SGD) | 2-layer ReLU, GD theory | Clean complete result, near-optimal sample complexity, both layers trained | 8,8,8,6 | Accept spotlight |
| ARPrtuzAnQ (Hardness under symmetries) | Hardness results for GD | Solid SQ lower bounds, well-executed | 8,6,8 | Accept spotlight |
| LEuuOaZNOT (Learning Boolean functions) | Boolean parities | Output-layer only, poor presentation | 3,3,3 | Reject |

The paper under review is clearly above LEuuOaZNOT—it has novel measures, open-problem resolution, and sound theory. Compared to HgOJlxzB16, it is comparable in scope (novel positive+negative theory for parity learning) but weakened by: (a) the positive/negative regime asymmetry, (b) the incomplete perturbed-Rademacher proof, and (c) the experiment-theory mismatch. Compared to ARPrtuzAnQ, it is comparable in theoretical novelty (GAL vs. CSQ bounds), with similar gaps (limited loss coverage). Both ARPrtuzAnQ and HgOJlxzB16 have cleaner, fully proved main claims.

Given these anchors, the paper deserves a solid accept but below spotlight level: the open problem resolution and the GAL measure are genuine contributions worthy of the community's attention, but the headline narrative is fragmented and the submission has an explicitly incomplete proof. I place this at **6.5**.

## Score and Decision

**Originality:** High. The GAL measure is novel, the almost-full parity setting is a genuine gap, and the junk-flow technique is a creative proof approach.

**Importance:** High. Initialization-induced computational separations, especially for a fixed function against SQ, are important for understanding the limits of gradient methods.

**Claims vs. support:** Moderate. The Gaussian negative result is fully proved; the perturbed-Rademacher case is incomplete; the positive/negative settings are partially mismatched.

**Soundness:** Good for the proved claims; the perturbed Rademacher case has an acknowledged gap.

**Clarity:** Good overall; some fragmentation between sections creates narrative confusion.

**Value to community:** Solid. The GAL framework and the resolution of the almost-full parity problem add real value.

**Final score: 6.5 — Weak Accept**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>