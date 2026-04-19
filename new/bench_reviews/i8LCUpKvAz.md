Now I have enough calibration data. Let me write the final review.

## Summary
This paper proposes EQO (Exploration via Quasi-Optimism), a tabular reinforcement learning algorithm that achieves minimax-optimal regret bounds using a simple c/N exploration bonus without empirical variance estimation. The key technical innovation is "quasi-optimism," a relaxed optimism condition that allows controlled underestimation while maintaining theoretical guarantees. The paper claims both theoretical advances (sharpest regret bounds under "weakest" assumptions) and practical superiority over existing minimax-optimal algorithms.

## Strengths

1. **Minimax-optimal regret without empirical variance estimation**: The algorithm achieves the optimal $\tilde{O}(H\sqrt{SAK})$ leading term with $\tilde{O}(HS^2A)$ non-leading terms matching Zhang et al. (2021a), while eliminating the need for empirical variance computation in the bonus. **Table 1** and **Theorem 1** confirm EQO is the only listed minimax-optimal method marked "Not required" for empirical variance.

2. **Novel quasi-optimism analysis framework**: The relaxed optimism condition (**Lemma 2**, Section 4.4.2) where $V_h^k(s) + \frac{3}{2}\lambda_k H \geq V_h^*(s)$ represents a genuine conceptual departure from standard UCB-style proofs. This allows the variance and 1/N terms to be decoupled in the Freedman-based analysis (**Lemma 1**), enabling the simpler bonus design.

3. **Sharp theoretical guarantees**: The regret bounds achieve improved logarithmic factors compared to Zhang et al. (2021a), and the PAC bounds (**Theorems 3-4**) match the minimax lower bounds $\tilde{O}(H^2SA/\varepsilon^2)$ for $\varepsilon < H/S$. The proof sketches in Section 4.4 are unusually informative for a main text.

4. **Algorithmic simplicity**: EQO uses a single scalar bonus $b^k(s,a) = c_k/N^k(s,a)$ per state-action pair (**Algorithm 1, Line 9**), avoiding the multi-term Bernstein-Freedman bonuses of UCBVI-BF, EULER, and ORLC. This reduces per-step computational complexity.

5. **Strong empirical performance on RiverSwim**: **Figure 1** shows EQO achieving lower cumulative regret than UCRL2, UCBVI-BF, EULER, ORLC, and MVP on two RiverSwim configurations ($S=30, H=120$ and $S=40, H=160$).

## Weaknesses

### Fatal
None.

### Major

- **Inconsistent "weakest assumptions" claim**: The paper repeatedly emphasizes operating under the "mildest" or "weakest" boundedness assumptions as a key contribution (Abstract, Introduction, **Table 1** footnote). However, **Assumption 1** (Section 4.1) requires BOTH $0 \leq V_h^*(s) \leq H$ (value boundedness) AND $0 \leq R_h^k \leq H$ (per-step reward boundedness). The standard "bounded return" assumption in cited works (e.g., Zhang et al. 2021a) assumes non-negative rewards with $\sum_{h=1}^H R_h \leq H$, which already implies each $R_h \leq H$ when rewards are non-negative. The paper's claim on line 152 that "our bounded value condition is weaker than the bounded return assumption" is misleading—the analysis still requires bounded rewards per step, and the value-boundedness condition does not replace it. This overclaim undermines a central advertised novelty.

- **Empirical evidence insufficient for broad claims**: The abstract and introduction assert EQO "consistently outperforms existing algorithms" and demonstrates "practical superiority." However, experiments are restricted to a **single environment family** (RiverSwim chain MDP) with only two configurations. There is no evaluation on environments with different dynamics (e.g., stochastic gridworlds, random MDPs with varying branching factors, sparse vs. dense rewards). Additionally, **Figure 1** appears to show single-run curves without confidence intervals or variance bands, and the paper lacks discussion of hyperparameter tuning procedures for $c_k$ and baseline methods. These limitations do not invalidate the theoretical contribution but fail to support the strong empirical superiority narrative.

### Minor

- **Overstated "first c/N bonus" claim**: The paper claims to be "the first to use an exploration bonus of the form $c/N$ for the reinforcement learning setting and attain regret guarantees" (Section 1.1). However, this overlooks RL algorithms using Hoeffding-style fixed bonuses or count-based confidence sets (e.g., UCRL2-style confidence intervals) that also avoid explicit empirical variance scaling. The distinction between "no variance estimation in the algorithm" versus "variance not needed conceptually" is not adequately clarified—the analysis still relies on Freedman-type variance-dependent concentration (**Lemma 1**), just without computing empirical variances online.

- **Parameter tuning complexity understated**: While the paper promotes "single parameter $c_k$" as practically convenient (Section 3, footnote 4), the theoretically recommended values require knowledge of $H, S, A, K, \delta$, and involve complex expressions (e.g., $c = \max\{7H\ell_1, 1.4H\sqrt{K\ell_1/(SA\ell_{2,K})}\}$ in **Theorem 1**). The PAC-optimal choice differs substantially from the regret-optimal one. No empirical sensitivity analysis demonstrates robustness to mis-specified $c$, which is critical if parameter simplicity is a selling point.

- **Notation error in Proposition 1**: The proposition states $\sum_{k=1}^K \text{Regret}(K) \leq \dots$, which should be $\text{Regret}(K)$ (without the sum) or $\sum_{k=1}^K \text{Regret}(k)$. This is likely a typo but causes confusion.

### Trivial
None beyond the notation error above.

## Nice-to-Haves

- Include learning curves with uncertainty bands (multiple seeds, shaded confidence intervals) to make empirical claims more robust.
- Add an ablation comparing EQO's $c/N$ bonus against a standard Hoeffding-style $\sqrt{(\log t)/N}$ bonus within the same algorithmic structure to isolate the effect of quasi-optimism.
- Provide per-episode runtime comparisons with implementation details (language, library, hardware) to substantiate computational efficiency claims.
- Include a brief intuitive example or toy demonstration showing how quasi-optimism changes exploration behavior compared to full optimism.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **Harsh Critic Point 4 (Regret/PAC constants mismatch)**: Moved to Minor weaknesses. The concern about $c_k$ tuning complexity is valid but does not invalidate the theoretical result—many theoretical RL papers have similar gaps between worst-case parameter choices and practical tuning. This is a presentation/clarity issue rather than a methodological flaw.

- **Harsh Critic point about pseudocode reproducibility (ties in argmax, N=0 handling)**: Removed as trivial implementation detail. The pseudocode (**Algorithm 1, Line 10**) does specify the $N^k(s,a)=0$ case explicitly ($Q_h^k(s,a) = H$), and tie-breaking is standard practice not requiring specification.

- **Strength Finder strength about "superior empirical regret performance"**: Weakened—the RiverSwim results are positive but the single-environment limitation prevents claiming broad "superiority." Moved to Nice-to-Have for more diverse evaluation.

- **Strength about "reduced computational complexity per step"**: Kept but noted that actual runtime comparisons are only in the appendix (Table 4), not the main text.

- **Harsh Critic's skepticism about "first minimax optimal RL with 1/N bonus"**: Partially valid—the paper should better position against UCRL2 and other count-based methods. Kept as Minor weakness with nuance.

## Novel Insights

The quasi-optimism framework represents a genuinely novel analytical technique that may have broader applicability beyond this specific algorithm. By allowing $V_h^k(s)$ to underestimate $V_h^*(s)$ by a controlled amount ($O(\lambda_k H)$) rather than enforcing strict optimism, the analysis sidesteps the need to fully bound estimation errors with variance-dependent bonuses at each step. The key technical innovation—using a difference-type variance bound (**Lemma 27** reference in Section 4.4.2) to telescope the sum of variances without requiring bounded returns—could inspire similar relaxations in other RL settings where strict optimism is difficult to maintain (e.g., function approximation, model-free algorithms). However, the practical impact of this insight remains to be demonstrated beyond the tabular setting.

## Suggestions

1. **Correct the boundedness assumption narrative**: Revise claims about "weakest assumptions" to accurately reflect that Assumption 1 requires both value boundedness and per-step reward boundedness. Either remove the "weaker than bounded return" claim or clarify that the value-boundedness condition is an additional (not replacement) assumption that broadens applicability in a different sense.

2. **Temper empirical claims or expand evaluation**: Either soften language like "consistently outperforms" to "outperforms on RiverSwim benchmark" or add experiments on at least 2-3 additional environment families (e.g., stochastic gridworlds, randomly generated MDPs) with multiple random seeds and confidence intervals.

3. **Clarify the "no empirical variance" contribution**: Distinguish more precisely between "no variance estimation in the algorithm" versus "variance not needed in the analysis." Acknowledge that Freedman-type variance terms appear in the proof, but the algorithm avoids online variance computation.

4. **Add sensitivity analysis for $c_k$**: Include a small ablation showing how regret varies with different choices of $c$ (e.g., $0.5c^*, c^*, 2c^*$ where $c^*$ is the theoretically recommended value) to demonstrate practical robustness.

5. **Fix the Proposition 1 notation**: Correct $\sum_{k=1}^K \text{Regret}(K)$ to $\text{Regret}(K)$.

## Score and Decision

**Calibration comparison**:
- **High-scoring theoretical RL papers**: The adversarial linear MDP paper (6yv8UHVJn4.md) with rate-optimal $\tilde{O}(\sqrt{K})$ regret received scores 8,8,8,6 (spotlight). The average-reward MDP sample complexity paper (jOm5p3q7c7.md) that closes a gap between upper/lower bounds received 6,6,8,6 (poster). Both had strong, uncontroversial theoretical contributions without significant overclaiming.
- **Papers with overclaiming issues**: The RetNet paper (UU9Icwbhin.md) with "impossible triangle" overclaims received 3,5,5,6 (rejected). The LM self-play paper (tCfvktlrHI.md) with limited empirical scope received 3,5,6,5 (rejected).
- **Borderline theoretical papers**: Papers with solid theory but limited experiments or overstated claims typically scored 5-6.

**Positioning**: This paper has a genuinely novel theoretical contribution (quasi-optimism framework) and achieves state-of-the-art regret bounds. The core theory appears sound and the algorithmic simplicity is valuable. However, the inconsistent "weakest assumptions" claim is a significant overstatement that undermines a key advertised novelty, and the empirical evidence is too narrow for the strong superiority claims made. Compared to the average-reward MDP paper (6,6,8,6) which had cleaner contributions without overclaiming, this paper is slightly weaker due to the assumption inconsistency. Compared to rejected papers with overclaiming issues, this paper is stronger because the core theory is solid and the overclaim is correctable.

The appropriate score is **5.5** (borderline). The theoretical contribution is genuinely interesting and would be strong enough for acceptance if the framing were more accurate and empirical claims more modest. However, the current overclaiming on assumptions and empirical scope prevents a clear accept recommendation without revision.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>