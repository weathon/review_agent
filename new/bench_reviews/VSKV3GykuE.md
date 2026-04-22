Now I have a good understanding of the paper and the calibration anchors. Let me synthesize the review.

Let me verify a few key claims from the harsh critic against the actual paper text:

1. **"Convergence guarantees require computing the full gradient"** - The paper says the update is $W^{t+1} = W^t - \gamma H_B^t \nabla f(W^t)$, which indeed uses $\nabla f(W^t) \in \mathbb{R}^{m \times n}$. However, the paper also says in Section 2.3 that "one can thus choose essentially any optimizer for approximately solving" the subproblem, meaning one doesn't need to explicitly compute $\nabla f(W^t)$—one just needs an approximate minimizer for $\min_A f(W^t + \frac{\alpha}{r} B_S^t A)$, which can be done via backprop with respect to $A$ only. The full gradient appears in the theoretical analysis but not necessarily in the practical algorithm. This is actually a valid point—the theory requires the full gradient, but the algorithm can be implemented efficiently. Let me check more carefully...

Actually, looking at the derivation more carefully: The update $W^{t+1} = W^t - \gamma H_B^t \nabla f(W^t)$ is derived by minimizing the majorization. The algorithm (Algorithm 1, Step 4) says "Using some iterative solver, approximately solve the subproblem" $\hat{A}^t \approx \min_A f(W^t + \frac{\alpha}{r} B_S^t A)$. So in practice, you don't compute the full gradient—you optimize with respect to $A$ only. The theoretical analysis derives what happens when you solve the subproblem exactly (which is equivalent to taking a projected gradient step), but the algorithm allows approximate solvers. The harsh critic's claim that "the theoretical derivation explicitly relies on the full gradient" is partially correct—theorems assume exact projected GD, not approximate solutions. But the claim that you need to compute the full gradient in practice is incorrect—you can use any solver for the subproblem.

Actually wait, for the specific convergence theorems (5.3 and 5.5), they're stated for "RAC-LoRA (Algorithm 1) with GD updates (Equation 3 or 4)." So the theorems specifically cover GD updates, not arbitrary subproblem solvers. The "any optimizer" claim in Section 2.3 is broader than what the theorems actually cover. This is a fair criticism.

2. **"The convergence rate degrades catastrophically with low rank"** - The remark after Assumption 5.1 shows $\lambda_{\min}^H = r/n$ for isotropic distributions, giving a rate of $(1 - \gamma \mu r/n)^T$ for PL, and $O(n/(rT))$ for non-convex. For $r=1, n=768$, this means $768\times$ slower. The paper mentions in the experiments that "the convergence speed is proportional to $r/n$" but does NOT explicitly state the $n/r$ slowdown factor. The abstract claims "provable guarantees of convergence to the same solution as FPFT" without mentioning the rate degradation. This is a legitimate concern about overclaiming.

3. **"The algorithmic contribution is randomized sketch descent"** - The paper does use "sketch" terminology (Left Sketch, Right Sketch) and the update $W^{t+1} = W^t - \gamma H \nabla f(W^t)$ where $H$ is a random projection is indeed a randomized sketch descent / randomized Kaczmarz type method. The paper mentions Gower et al. (2019) and Richtárik in the introduction but doesn't explicitly connect to the sketch descent literature. The remark connecting to isotropy is brief. This is a valid point about related work.

4. **"Experiments don't support bridging FPFT gap"** - On GLUE (Table 2), RAC-LoRA (77.0 avg) underperforms even the authors' own LoRA reimplementation (78.5). The paper acknowledges this: "RAC-LoRA performs competitively with other low-rank adaptation methods, but does not outperform Asymmetric LoRA despite having greater capacity." They explain this by saying GLUE tasks don't need high-rank adaptation. On MNIST (Table 3), RAC-LoRA achieves 92.0/96.1 vs AsymmLoRA's 62.3/81.6, clearly showing the chaining benefit. But COLA achieves 92.6/96.2 without convergence guarantees. The paper's main claim is about theoretical guarantees + practical utility, not necessarily beating all baselines.

5. **"Theory doesn't cover experiments"** - The theorems cover GD, SGD, and random reshuffling, but experiments use AdamW. This is a valid gap. The paper addresses this indirectly by claiming the framework supports "essentially any optimizer."

Now let me assess the strength of each criticism:

**Harsh Critic Issue 1 (Full gradient requirement):** This is partially valid but partially incorrect. The practical algorithm doesn't require computing the full gradient—you optimize the subproblem in $A$ only. However, the convergence theorems are specific to the projected GD form (Equations 3-4), which is equivalent to solving the subproblem exactly. The gap between theory and practice here is real but not as severe as stated. The real issue is that the convergence rate degrades with low rank.

**Harsh Critic Issue 2 (Theory doesn't cover AdamW experiments):** This is valid. The theorems assume GD updates specifically, and the SGD/RR extensions in the appendices still don't cover AdamW. The "essentially any optimizer" framing is overclaimed relative to what's proven. However, this is common in optimization papers—theory often lags behind practice. I'll include this as a minor-to-moderate weakness.

**Harsh Critic Issue 3 (Algorithm is sketch descent):** The connection to randomized sketch descent is real and worth noting, but the paper does use the terminology "sketch" and references Gower and Richtárik (2019). The actual contribution is recognizing that LoRA's problems can be addressed through this lens and applying it in the LoRA context. I'll note this as a minor weakness—acknowledgment of the connection should be more explicit, but it's not a fatal flaw.

**Harsh Critic Issue 4 (Empirical results don't support bridging claim):** Valid. The GLUE results are not convincing for the "bridging" claim. The MNIST results show clear improvement over AsymmLoRA, which is the relevant comparison given that both have convergence guarantees (or lack thereof). But COLA beats RAC-LoRA without guarantees, which complicates the narrative.

Let me now write the consolidated review.

Now for calibration scoring, let me compare:

- **GoLore (5.25, Reject)**: Similar paper—identifies convergence failure in an existing method (GaLore), proposes a randomization fix with convergence guarantees, but empirical gains are marginal. RAC-LoRA has a clearer theoretical framework and a more explicit convergence rate analysis, but also has the issue that its theory-based algorithm doesn't clearly outperform heuristics in practice.

- **LoRA-RITE (8.67, Oral)**: Strong theoretical analysis of LoRA with consistent empirical improvements across multiple benchmarks. This is much stronger than RAC-LoRA.

- **SRO (6.50, Poster)**: Sketch-based optimization with convergence guarantees for least squares. More focused theoretical contribution with empirical validation.

- **Non-differentiability paper (3.0, Reject)**: identifies real problems but solutions are not compelling enough.

- **MAST (7.0, Poster)**: Novel optimization with random sketch, convergence guarantees, practical improvements.

This paper is somewhere in the 5-6 range. It identifies a genuine problem (LoRA convergence failures), provides a theoretically clean framework, but:
- The convergence rate scales poorly with low rank (r/n factor barely acknowledged)
- Empirical gains are modest (RAC-LoRA doesn't even match basic LoRA on GLUE)
- The theory-experiment gap exists (AdamW vs GD theorems)
- The algorithm has deep connections to existing randomized sketch methods that aren't adequately discussed

The contribution is real—the convergence analysis of LoRA/COLA failure modes is valuable, and the RAC-LoRA framework is principled. But the practical impact is limited, and the bridging claim is overreaching. 

I'll score this around 5.0-5.5. It has genuine theoretical contribution but the practical significance is limited.</think>Now I have enough context to write the consolidated review. Let me verify one more detail about the COLA comparison on MNIST.

The paper (Table 3) shows: LoRA (Gaussian, Zero) = 83.8, COLA (Gaussian, Zero) = 92.6, AsymmLoRA (Gaussian, Zero) = 62.3, RAC-LoRA (Gaussian, Zero) = 92.0. And LoRA (Zero, Gaussian) = 87.0, COLA (Zero, Gaussian) = 96.2, AsymmLoRA (Zero, Gaussian) = 81.6, RAC-LoRA (Zero, Gaussian) = 96.1. FPFT = 98.0.

So COLA slightly beats RAC-LoRA in both configurations (92.6 vs 92.0 and 96.2 vs 96.1), but RAC-LoRA has convergence guarantees while COLA doesn't. The key comparison is RAC-LoRA vs AsymmLoRA (same asymmetric structure, RAC-LoRA adds chaining), where RAC-LoRA dramatically outperforms (62.3→92.0 and 81.6→96.1).

## Summary

RAC-LoRA proposes a randomized asymmetric chain of low-rank adaptations that provides provable convergence guarantees for LoRA-style fine-tuning. The paper demonstrates that standard LoRA, AsymmLoRA, and Chain-of-LoRA (COLA) can diverge or converge to suboptimal points, and derives a projected gradient descent interpretation of the RAC-LoRA update, establishing O(1/T) non-convex and linear PL convergence rates parameterized by λ_min^H = r/n under isotropic sketch distributions.

## Strengths

- **Concrete demonstration that existing LoRA methods can fail to converge.** The 3×3 quadratic counterexample (Eq. 2, M=Diag(10,1,...,1)) and Figure 1 clearly show LoRA and COLA diverging at theoretical step sizes and converging to suboptimal points at smaller step sizes, while AsymmLoRA converges to a different stationary point. This is a genuine and valuable contribution that identifies a real theoretical gap.

- **Principled convergence framework with explicit rates.** Theorems 5.3 and 5.5 provide explicit convergence guarantees (O(1/(λ_min^H γ T)) for non-convex and (1−γμλ_min^H)^T for PL) where no prior LoRA variant has such guarantees. The key insight—that randomizing one sketch matrix per step yields a positive expected projection eigenvalue enabling convergence—is clean and correct.

- **The framework provably interpolates between LoRA and full-parameter fine-tuning.** The remark after Assumption 5.1 (λ_min^H = r/n for isotropic distributions, recovering λ_min^H = 1 at full rank) formally establishes this interpolation. The linear regression experiments (Figure 2) confirm the predicted r/n scaling of convergence speed.

- **Dramatic improvement over AsymmLoRA.** On MNIST (Table 3), RAC-LoRA achieves 92.0 vs AsymmLoRA's 62.3 (Gaussian B, Zero A) and 96.1 vs 81.6 (Zero A, Gaussian B), demonstrating the value of the chaining mechanism when capacity is limited.

## Weaknesses

### Fatal
None.

### Major

- **The convergence rate scales as n/r, which is a severe practical limitation that is insufficiently acknowledged.** Under isotropic distributions, λ_min^H = r/n, so for typical LoRA ranks (r=1–16) and dimensions (n=768–4096), the PL convergence rate (1−γμr/n)^T requires n/r ≈ 48–768× more iterations than full GD to achieve comparable accuracy. The abstract and introduction claim "provable guarantees of convergence to the same solution as FPFT" without ever stating this rate penalty explicitly. The paper mentions convergence being "proportional to r/n" in the linear regression section but does not present the full implication: that in practice, with low rank, convergence may require orders of magnitude more outer iterations. This significantly tempers the contribution of the "bridging" claim.

- **The experiments do not validate the theory and do not show RAC-LoRA meaningfully outperforming existing methods.** All experiments use AdamW with learning rate schedules, while all convergence theorems (5.3, 5.5) and their extensions (SGD, RR in appendices) assume projected gradient steps. The paper's claim in Section 2.3 that "one can thus choose essentially any optimizer" is overstated—the proven guarantees only apply to specific subproblem solvers. On the primary NLP benchmark (Table 2), RAC-LoRA (avg 77.0) underperforms the authors' own LoRA reimplementation (78.5) and matches AsymmLoRA (76.8), while remaining far from FPFT (80.9). COLA (77.6) slightly outperforms RAC-LoRA despite having no convergence guarantees. The MNIST results show RAC-LoRA benefits over AsymmLoRA but do not close the gap to FPFT (96.1 vs 98.0) or clearly outperform the simpler COLA (96.1 vs 96.2).

### Minor

- **Insufficient engagement with the randomized sketch/projection optimization literature.** The update W^{t+1} = W^t − γH∇f(W^t) where H is a random projection matrix is precisely randomized sketch descent (Gower & Richtárik, 2019). The paper uses "sketch" terminology and cites them in passing, but never explicitly states this connection or discusses how the convergence rates relate to standard results in that literature. The PL-convergence rate and non-convex rate are standard for such methods. The novelty lies in applying this framework to LoRA, not in the optimization itself, and this should be stated transparently.

- **The "any optimizer" framing in Section 2.3 is misleading relative to what is proven.** Algorithm 1 Step 4 says "Using some iterative solver, approximately solve" the subproblem, and the contributions list claims "one can thus choose essentially any optimizer." However, all convergence theorems require the exact projected GD solution (Equations 3–4). The gap between what is claimed as flexible and what is actually analyzed should be clearly delineated.

## Nice-to-Haves

- Experiments using GD/SGD on neural network tasks to directly validate the convergence rate predictions from Theorems 5.3/5.5, rather than only AdamW experiments.
- Explicit discussion of the n/r convergence rate penalty and practical strategies to mitigate it (e.g., importance sampling for the sketch distribution, acceleration techniques).
- Comparison at matched compute budgets (total FLOPs or wall-clock time) rather than just matched step counts.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic's claim that convergence guarantees "require computing the full gradient," undermining efficiency.** The paper's Algorithm 1 says to "approximately solve the subproblem" min_A f(W^t + α/r B_S^t A), which can be done via backpropagation with respect to A only. The full gradient ∇f(W^t) appears in the theoretical analysis to derive the projected GD form, but the practical implementation only requires optimizing over the trainable matrix A or B per step. The theoretical equivalence to projected GD is what guarantees convergence; the implementation need not compute the full gradient explicitly.

- **Harsh critic's claim that the derivation is "circular" or "reverse-engineered."** Deriving an algorithm from a majorization-minimization perspective and then proving it converges is a standard and valid approach in optimization, not circular reasoning.

- **Harsh critic's demand for comparison with LoRA* using a different implementation/setup.** The paper clearly marks LoRA* with an asterisk indicating results from Hu et al. (2021) and provides their own LoRA implementation for fair comparison. This is a standard practice.

- **Strength finder's claim about "generality across optimizers" (Table 1) as a core strength.** While the table lists GD, RR, SGD, and FL extensions, these are all variants of gradient-based subproblem solvers and don't cover adaptive methods like AdamW. The generality is real but narrower than claimed—removed from core strengths since it's undermined by the theory-practice gap noted above.

- **Harsh critic's criticism that LoRA "converges with smaller step sizes" undermines the divergence claim.** The paper itself acknowledges this at the end of Section 3: "When smaller step sizes are applied to LoRA and COLA, these methods do converge, but to a stationary point that is significantly distant from the optimal solution." So the convergence issue is not just divergence but also convergence to wrong solutions, which the critic partially acknowledged but then dismissed.

## Novel Insights

The paper's most impactful contribution is not the convergence rates themselves (which follow from standard randomized sketch descent theory) but rather the formal demonstration that practical LoRA/COLA methods can fail in predictable ways, combined with the insight that freezing one randomly-sampled matrix and training the other avoids the non-smoothness issue identified by Sun et al. (2024) while enabling theoretical analysis through the random projection lens. The λ_min^H = r/n tradeoff—formalizing the intuitive "lower rank = slower convergence" as an exact quantitative relationship—is valuable even though the practical implications of this scaling are understated.

## Suggestions

- Add explicit quantitative statements about the convergence rate scaling, e.g., "For rank-1 LoRA adaptation of a layer with dimension d, our method converges to the same optimum as full fine-tuning but requires approximately d times more iterations." This allows readers to assess practical implications.
- Run at least one experiment using GD on a neural network task to validate the theoretical convergence rate predictions directly.
- Acknowledge the connection to randomized sketch descent explicitly and discuss what is novel beyond applying this framework to LoRA (the convergence failure analysis of existing methods being the main novelty).

## Score and Decision

### Evaluation

**Originality:** The convergence failure analysis of LoRA/COLA is novel and valuable. The RAC-LoRA algorithm itself recovers a known optimization form (randomized sketch descent) applied to a new domain, with the main novelty being the LoRA-specific analysis and the asymmetric+chaining design insight.

**Importance of research question:** Important—LoRA is widely used and its theoretical convergence properties are indeed understudied.

**Claim support:** The theoretical claims are correct (convergence rates, PL results). The "bridging" claim is formally true (convergence to the same solution as FPFT) but practically overstated given the n/r rate penalty and the lack of empirical demonstration that RAC-LoRA closes the FPFT gap on standard benchmarks.

**Soundness of experiments:** Limited. Convex experiments validate the theory but neural network experiments use AdamW (not covered by theorems) and don't show clear practical advantages over simpler baselines on the main benchmark.

**Clarity:** Generally well-written and well-organized, with clear notation and a logical progression from problem statement to theory to experiments.

**Value to community:** Moderate. The formal characterization of LoRA convergence failures is valuable; the practical impact of RAC-LoRA is limited by the convergence rate scaling and the lack of clear empirical advantage.

### Calibration Anchors

1. **GoLore** (avg 5.25, Reject): Most similar—identifies convergence failure in existing method, proposes randomization fix with guarantees, but marginal empirical improvement. RAC-LoRA has clearer theory and a more dramatic demonstration of failure modes, but similarly limited practical gains. RAC-LoRA is somewhat stronger.

2. **LoRA-RITE** (avg 8.67, Oral): Strong theoretical analysis of LoRA with significant empirical improvements. RAC-LoRA is notably weaker on both the empirical side and the originality of the optimization contribution.

3. **Non-differentiability paper** (avg 3.0, Reject): Identifies important theoretical issues but provides inadequate practical solutions. RAC-LoRA is clearly stronger than this.

4. **MAST** (avg 7.0, Poster): Random sketch optimization with convergence guarantees. RAC-LoRA has less novel optimization content (standard sketch descent applied to LoRA) but addresses a more timely application.

5. **SRO** (avg 6.5, Poster): Focused sketch-based optimization contribution. RAC-LoRA's contribution is comparable in nature—the LoRA convergence failure analysis adds value beyond the standard optimization content.

Relative to anchors: RAC-LoRA is stronger than GoLore (5.25) due to cleaner theory and a more dramatic convergence failure demonstration, but weaker than LoRA-RITE (8.67) due to limited practical gains and overstated bridging claim. It sits in the 5–6 range alongside papers with genuine but limited contributions.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>