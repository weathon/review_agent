I already have the full paper text in the user message. Let me re-examine the key issues:

**Issue 1: "Weakest boundedness assumption" claim**

The paper's Assumption 1 states:
- 0 ≤ V_h^*(s) ≤ H for all s ∈ S and h ∈ [H]
- 0 ≤ R_h^k ≤ H for all h ∈ [H] and k ∈ N

The paper claims this is weaker than the "bounded return" assumption (0 ≤ Σ R_h ≤ H with non-negative rewards). Let me think carefully:

- Under "bounded return": Σ R_h ≤ H, and R_h ≥ 0. This means each R_h could be as large as H (if all other rewards are 0). So bounded return does NOT imply bounded per-step rewards.

However, the paper's Assumption 1 requires 0 ≤ R_h^k ≤ H, which means each individual reward is bounded by H. This is the SAME bound as what bounded return gives (since under bounded return, R_h ≤ H follows from R_h ≥ 0 and Σ R_h ≤ H, because the maximum any single R_h can be is H when all others are 0).

Wait, actually under bounded return: Σ_{h=1}^H R_h ≤ H and R_h ≥ 0. This means each R_h ≤ H is automatically implied (since the sum is at most H and each term is non-negative). So the paper's per-step bound 0 ≤ R_h^k ≤ H is actually implied by bounded return.

Now, the key question: does bounded value + bounded reward (as in the paper) imply something weaker than bounded return?

Under Assumption 1: 0 ≤ V_h^*(s) ≤ H and 0 ≤ R_h^k ≤ H.

Under bounded return: Σ R_h ≤ H and R_h ≥ 0, which implies each R_h ≤ H.

So actually the R_h bound is the same in both cases. The question is what other constraint each imposes.

Bounded return: Σ R_h ≤ H (almost surely, or in expectation depending on formulation). Some formulations say the total reward is bounded by H, meaning the random total reward is at most H.

The paper's Assumption 1 has: 0 ≤ V_h^*(s) ≤ H (value function is bounded) and 0 ≤ R_h^k ≤ H.

Under bounded return with R_h ≥ 0: The total random reward is at most H. Since V_h^* is the maximum expected total remaining reward starting from state s at time h, and each R_h ≤ H, we have V_h^*(s) ≤ H * (H-h+1) which could be much larger than H. But bounded return also gives V_h^*(s) ≤ H. Wait, actually if each episode's total reward is at most H, then V_1^*(s) ≤ H. And since V_{h+1}^* ≥ 0, we have V_h^*(s) ≤ H - (minimum of h-1 rewards) ≤ H. So indeed bounded return implies bounded value.

Now, does bounded value + bounded reward (R_h ≤ H) imply bounded return? No! Consider: R_h = H for some h, and R_j = 0 for j ≠ h. Then total return = H, which is bounded. But what about: R_h = H for two different time steps h? Then total return = 2H, which violates bounded return. But does this violate bounded value? V_1^*(s) could be at most... well, it depends on the transition structure. If the optimal policy can collect H at each of two time steps, then V_1^*(s) ≤ 2H. But Assumption 1 requires V_1^*(s) ≤ H. So in practice, Assumption 1 is constraining the MDP so that you CANNOT collect big rewards at every step.

So the comparison is:
- Bounded reward (per-step): R_h ∈ [0,1]. This gives V_h^* ≤ H-h+1. The strongest assumption.
- Bounded return: Σ R_h ≤ H. This gives V_1^* ≤ H, and R_h can be as large as H. Weaker than bounded reward.
- Paper's assumption: V_h^* ≤ H and R_h ≤ H. 

The question is: is "V_h^* ≤ H AND R_h ≤ H" weaker than "bounded return"?

Under bounded return (Σ R_h ≤ H, R_h ≥ 0): 
- V_h^* ≤ H (since optimal policy gets at most H total reward from any point)
- Each R_h can be up to H (when only one step gets reward and rest get 0)
- But R_h ≤ H follows from Σ R_j ≤ H and R_j ≥ 0

So bounded return implies the paper's Assumption 1. Does the paper's assumption imply bounded return? No—in the paper's setting, you could have R_h = H at two different time steps. The total return could be 2H, which violates bounded return. But V_1^* ≤ H would prevent the optimal policy from collecting both rewards.

Wait, that's the key insight. The paper's Assumption 1 says V_h^* ≤ H and R_h ≤ H. This means:
1. The optimal value is bounded by H (same as bounded return)
2. Each individual reward is bounded by H (this is ALREADY guaranteed by bounded return since R_h ≥ 0 and sum ≤ H implies R_h ≤ H)

So the paper's assumption is EQUIVALENT to (or weaker than) bounded return in terms of the value bound, and the R_h ≤ H condition doesn't add anything beyond what bounded return already implies.

Actually wait. Let me reconsider. Under the "bounded return" assumption used by Zhang et al. (2021a), the condition is: 0 ≤ Σ_{h=1}^H R_h ≤ H almost surely. This means the random total reward is bounded by H. This already implies each R_h ≤ H (since all rewards are non-negative). And it also implies V_1^*(s) ≤ H.

Under the paper's Assumption 1: V_h^*(s) ≤ H for ALL h (not just h=1), and 0 ≤ R_h^k ≤ H. The fact that the paper requires V_h^* ≤ H for all h is actually also implied by bounded return (since from any time step h, the remaining total reward is bounded by H).

So where is the paper's assumption actually weaker? The paper says bounded value is weaker because:
- Under bounded reward (R_h ∈ [0,1]): V_h^*(s) ≤ H-h+1 ≤ H. V is bounded.
- Under bounded return (0 ≤ Σ R_h ≤ H): V_h^*(s) ≤ H for all h. V is bounded.
- Under bounded value (V_h^*(s) ≤ H): This is the weakest condition among the three.

The key difference: bounded value allows per-step rewards that are NOT bounded by H. For example, R_h could have very large variance or even be unbounded (in the sense that E[R_h] could be up to H but individual R_h^k samples could exceed H), as long as V_h^* stays bounded.

BUT the paper's Assumption 1 explicitly requires 0 ≤ R_h^k ≤ H for all h, k. This is a per-step bounded reward constraint (just scaled by H instead of 1). This means:

Paper's assumption = bounded value + bounded per-step reward (scaled by H)

If we consider "bounded return" as: 0 ≤ Σ R_h ≤ H almost surely (with R_h ≥ 0), then:
- This already implies R_h ≤ H for all h
- This already implies V_h^* ≤ H for all h

So actually, bounded return IMPLIES the paper's Assumption 1, and the paper's Assumption 1 does NOT obviously imply bounded return (since you could have R_h = H for multiple h under the paper's assumption as long as V_h^* ≤ H).

Wait, I'm confused now. Let me re-read the paper's Assumption 1 more carefully.

Assumption 1: 
- 0 ≤ V_h^*(s) ≤ H for all s ∈ S and h ∈ [H]
- 0 ≤ R_h^k ≤ H for all h ∈ [H] and k ∈ N

The paper's claim in Section 4.1: "We further relax this boundedness assumption by constraining only the optimal values V_h^*(s) to be within the interval [0, H], along with the conventional boundedness on the random rewards within [0, H]."

And the paper notes: "Since the value function is the expected return, our bounded value condition is weaker than the bounded return assumption"

So the paper is saying: V_h^* ≤ H (which is implied by bounded return) + R_h ≤ H (which is also implied by bounded return) = our condition, which is weaker.

The logic would be: their condition doesn't require that the total random return is almost surely bounded by H. It only requires that the EXPECTED optimal value is bounded by H and each individual reward is bounded by H. Under bounded return (a.s.), both conditions hold, so bounded return implies the paper's condition. But the converse need not hold: you could have R_h^k with large variance where the total return in some episodes exceeds H, but V_h^* ≤ H on average.

Actually, hmm. Under bounded return (a.s.): Σ R_h ≤ H. This implies R_h ≤ H. And it implies V_1^*(s) ≤ H. But does it imply V_h^*(s) ≤ H for all h? Yes, because from time step h, the remaining reward is also bounded by H a.s., so V_h^*(s) ≤ H.

Under the paper's condition: R_h^k ≤ H and V_h^*(s) ≤ H. Note that R_h^k ≤ H is a per-step a.s. bound. This does NOT imply Σ R_h ≤ H a.s. So the total return can exceed H.

So the paper's Assumption 1 is indeed WEAKER than bounded return in the following sense: bounded return implies both V_h^* ≤ H (all h) and R_h ≤ H, but the reverse is not true. You can have examples where V_h^* ≤ H and R_h ≤ H but the total return exceeds H.

Actually, wait. Let me reconsider the "bounded return" assumption. Zhang et al. (2021a) state: "the total reward of each episode is bounded in [0, 1]." If we scale by H, this becomes "the total reward is bounded in [0, H]." And they may allow individual rewards to exceed 1 (or H after scaling) as long as the sum is bounded. The paper references Jiang & Agarwal (2018) and Zanette & Brunskill (2019) for the bounded return setting.

Hmm, actually in the "bounded return" setting specifically used in prior work (like Zhang et al. 2021a), what they assume is: Σ_{h=1}^H R_h ∈ [0, H] almost surely, AND R_h ≥ 0 for all h. This implies R_h ≤ H for all h (since all rewards are non-negative and sum is at most H).

The paper's assumption: V_h^*(s) ∈ [0, H] and R_h^k ∈ [0, H].

The comparison:
1. Bounded return (Σ R_h ∈ [0, H] a.s., R_h ≥ 0) → implies V_h^* ∈ [0, H] and R_h ∈ [0, H]. ✓
2. Paper's assumption (V_h^* ∈ [0, H], R_h^k ∈ [0, H]) → does NOT imply Σ R_h ∈ [0, H] a.s. ✗

So the paper's Assumption 1 IS weaker than bounded return. The harsh reviewer's criticism that "R_h^k ≤ H is just bounded rewards (up to scaling), which is stronger than bounded return" is WRONG. Bounded return already implies R_h ≤ H! (Because non-negative rewards summing to at most H means each individual reward is at most H.)

So the harsh reviewer's Critical Issue 1 is incorrect. The paper's Assumption 1 IS genuinely weaker than bounded return, because it doesn't require the total return to be almost surely bounded by H, only the expected (value-bound) and per-step reward bounds. The per-step reward bound is already implied by bounded return, so it's not adding a constraint beyond what bounded return already imposes.

Actually, let me reconsider one more time. Is it possible that in the "bounded return" assumption, individual rewards are allowed to exceed H? If they define "bounded return" as just requiring Σ R_h ∈ [0, H] a.s., they might not require R_h ≥ 0. In that case, R_h could be negative at some steps (with positive rewards at others). 

Let me re-read the paper. The paper says about bounded return: "the return, defined as the total reward of an episode, is bounded as 0 ≤ Σ R_h ≤ H, and that each random reward is non-negative." So they explicitly require R_h ≥ 0 and Σ R_h ∈ [0, H]. This means each R_h ∈ [0, H].

So the paper's Claim 1 ("our bounded value condition is weaker than the bounded return assumption") is CORRECT:
- Bounded return ⟹ bounded value + bounded per-step reward: YES (trivially, since bounded return implies each R_h ≤ H)
- Bounded value + bounded per-step reward ⟹ bounded return: NO (you could have an MDP where R_h = H at every step but V_1^*(s) is still at most H because poor connectivity)

The harsh reviewer was WRONG on this point. The paper's assumption is genuinely weaker.

**Issue 2: Algorithm not specified for unvisited pairs**

Looking at Algorithm 1:
- Line 3-5 define N^k, r̂^k, P̂^k using sums over data
- Line 9-10 define b^k and Q_h^k with a piecewise definition for N^k(s,a) = 0 and N^k(s,a) > 0

When N^k(s,a) = 0:
- r̂^k and P̂^k are undefined (0/0)
- b^k is undefined (c/0)
- But Q_h^k is defined as H (the optimistic initialization)

This is a standard convention in RL algorithms. When N^k(s,a) = 0, Q is set to H (the maximum possible value), and the agent won't use r̂^k or P̂^k for that pair because it goes through the N^k > 0 branch. The agent will pick actions with N^k(s,a) = 0 because Q = H is maximally optimistic.

The harsh reviewer is technically correct that r̂^k and P̂^k are undefined for unvisited pairs, but this is a minor presentation issue, not a structural gap. The algorithm is well-defined in the sense that whenever these quantities are used in computing the policy, the branch with N^k > 0 is taken. The analysis also properly handles this because unvisited state-action pairs have N^k = 0 and the proof doesn't need well-defined empirical estimates for them—the bonus and estimates are only used when N^k > 0.

This is a TRIVIAL/MINOR issue, not a "structural specification gap." Standard RL algorithms handle this the same way.

**Issue 3: "No empirical variance" overclaimed**

The paper mentions Tiapkin et al. (2022) in the Related Work section. Let me check what it says:

"In the time-inhomogeneous setting... Tiapkin et al. (2022) propose a posterior-sampling algorithm and achieve the minimax bound without computing empirical variances."

The paper discusses this. They acknowledge Tiapkin et al. (2022) achieves minimax bounds without empirical variance. However, Tiapkin et al. is for the TIME-INHOMOGENEOUS setting, which as the paper notes "is often viewed as a special case of the time-homogeneous setting with HS states." The paper's claim is about the TIME-HOMOGENEOUS setting specifically.

Looking at Table 1: it clearly says "Comparison of minimax optimal algorithms for tabular reinforcement learning under the time-homogeneous setting." Tiapkin et al. (2022) works on the time-inhomogeneous setting and doesn't appear in that table. The paper's claim that "all minimax optimal algorithms [for the time-homogeneous setting] rely on empirical variances" is correct given their scope.

However, the abstract says "Unlike existing minimax optimal approaches, EQO avoids reliance on empirical variances" without specifying the time-homogeneous setting. This is a valid concern—the abstract's claim is broader than what's actually shown. But the body of the paper is clear about the time-homogeneous setting.

The point about variance still appearing in the proof is valid but somewhat semantic. The algorithm doesn't compute empirical variances; the analysis uses them. The paper does acknowledge this in Section 4.4. The key contribution is showing that you don't need to compute empirical variances in the bonus, which simplifies the algorithm significantly.

**Issue 4: Limited experiments**

The paper only evaluates on RiverSwim with two configurations. This is indeed limited. Let me consider the claim: "EQO consistently outperforms existing algorithms in both regret performance and computational efficiency." This is too strong for just one environment.

**Issue 5: "Sharpest known regret bound"**

The paper claims the tightest regret bound with improved logarithmic factors compared to Zhang et al. (2021a). The leading term matches and the logarithmic factors are claimed to be tighter. The paper should explicitly compare the logarithmic factors side by side, but the claim appears valid for the time-homogeneous setting.

Now let me also look at the neutral reviewer's points and the spark:

The neutral reviewer's weaknesses:
1. Limited experiments (same as harsh reviewer) - VALID
2. Practical value of c parameter is oversimplified - VALID, the theoretical c depends on many quantities
3. Quasi-optimism gap could be large - worth noting but the analysis handles it
4. No formal computational complexity comparison - nice to have but not critical
5. Connection to bandit literature - minor

The spark suggests:
1. Test on more environments - VALID
2. Include variance-free baselines like PSRL - VALID
3. Ablation on c_k - VALID
4. Evaluate under weaker boundedness assumption - VALID
5. Characterize quasi-optimism gap empirically - VALID

Let me now write the consolidated review.</think>Now I have enough calibration context. Let me carefully verify the key claims against the paper and write the final review.

## Summary

The paper introduces EQO (Exploration via Quasi-Optimism), a tabular reinforcement learning algorithm that achieves minimax-optimal regret Õ(H√(SAK) + HS²A) using a simple bonus term c/N(s,a) rather than the empirical-variance-based bonuses employed by all prior minimax-optimal algorithms. The key analytical innovation is "quasi-optimism," which relaxes the standard requirement that estimated values be fully optimistic, instead allowing bounded underestimation. The paper also achieves these results under weaker boundedness assumptions (bounded value function rather than bounded returns) and provides PAC guarantees.

## Strengths

- **Quasi-optimism is a genuine conceptual contribution.** The relaxation from full optimism (V_h^k ≥ V_h^*) to quasi-optimism (V_h^k + (3/2)λ_k H ≥ V_h^*) in Lemma 2 is novel and opens a new direction for algorithm design. This enables a simpler bonus structure while still controlling regret, and the analysis technique (decoupling variance and 1/N terms via Freedman's inequality in Lemma 1) is elegant and likely reusable.

- **Sharper regret bound with improved logarithmic factors.** The leading term matches the lower bound Ω(H√(SAK)), and the logarithmic factors are tighter than the prior state-of-the-art (Zhang et al., 2021a). The non-leading term Õ(HS²A) matches the best known. This is a meaningful improvement in the time-homogeneous tabular setting.

- **Simpler algorithm structure.** The c/N(s,a) bonus eliminates the need to compute empirical variances at each step. This is a practical simplification over UCBVI-BF, EULER, ORLC, and MVP, all of which require variance estimation. The single-parameter {c_k} design also simplifies implementation.

- **Weaker boundedness assumption.** Assumption 1 only requires V_h^* ∈ [0,H] and R_h^k ∈ [0,H]. This is genuinely weaker than the bounded return assumption used in prior work (since bounded return implies both conditions, but not vice versa), broadening the class of MDPs covered.

- **Complete PAC guarantees.** Matching mistake-style PAC and best-policy identification bounds complement the regret analysis, providing a more complete statistical characterization.

## Weaknesses

### Fatal
None.

### Major

- **Overclaimed practical superiority based on limited experiments.** The abstract states EQO "consistently outperforms existing algorithms in both regret performance and computational efficiency," and the conclusion claims "practical applicability." However, all experiments are on a single environment family (RiverSwim) with just two configurations (S=30/H=120 and S=40/H=160). RiverSwim is well-known to be a hard-exploration environment that heavily favors exploration-heavy algorithms. No experiments on other standard tabular benchmarks (random MDPs, deterministic MDPs, gridworlds, sparse-reward settings) are provided. The empirical claims should be significantly toned down or the evaluation substantially expanded. This matters because a paper claiming to resolve a "theory–practice gap" needs diverse evidence.

- **The "no empirical variance" narrative is overclaimed.** The paper frames its core contribution as showing that empirical variance in the bonus is "not necessary" or "no longer the only viable or practical approach" (Introduction). However: (1) the proof in Section 4.4 still fundamentally relies on variance (Freedman's inequality, Var(V_{h+1}^*) terms), so the variance dependence hasn't been eliminated from the analysis—it has been moved from the algorithm to the proof; (2) Tiapkin et al. (2022) already achieves minimax bounds without empirical variance computation (albeit in the time-inhomogeneous setting, which the paper acknowledges). The actual contribution is more precisely stated as: a count-based bonus suffices for minimax-optimal regret in the time-homogeneous tabular setting, enabled by a quasi-optimism analysis. The broader claims about variance being "no longer necessary" should be scaled back.

### Minor

- **Algorithm specification for unvisited pairs is implicit but standard.** When N^k(s,a) = 0, the quantities r̂^k, P̂^k, and b^k are undefined, but Q_h^k is set to H (maximally optimistic), making the algorithm well-defined in practice. The proofs also correctly handle the N = 0 case. This is a presentation clarity issue, not a correctness issue—common in the literature—but making it explicit would improve the paper.

- **The theoretical constant c depends on knowledge of K, S, A, H, δ.** Theorem 1 requires c that depends on K, making it not truly an "anytime" algorithm. Theorem 2 provides an anytime version but with a doubling-trick-style c_k and worse leading constants (75 vs. 38). The claim of "convenient control through a single parameter" (Section 3 and footnote 4) somewhat understates the practical complexity of setting c, since practitioners need to choose K or adopt the anytime version.

- **Non-leading term Õ(HS²A) dominates when K < S³A.** The paper states the bound matches the lower bound for K ≥ S³A but does not discuss the practical implications. For moderate-sized MDPs (e.g., S=50), this threshold is in the millions of episodes. A brief discussion of when the leading term dominates would help readers assess practical relevance.

### Trivial

- The discussion of how W_h functions are derived in Section 4.4.2 (with c_1=2, c_2=1/2 appearing) is somewhat abrupt; more intuition for these specific constants would aid readability.

## Nice-to-Haves

- Experiments on additional environments (random MDPs, gridworlds, deterministic chains) and inclusion of variance-free baselines like PSRL would substantially strengthen the empirical claims.
- Ablation or sensitivity analysis on the choice of c_k would help substantiate the "single parameter" practical advantage.
- A formal per-episode computational complexity comparison (O(S²A) vs. O(S²A + SA) with variance computation) would formalize the efficiency claims.
- Visualization of V_h^*(s) - V_h^k(s) during learning would illuminate whether quasi-optimism meaningfully differs from full optimism in practice.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic's Issue 1 ("Weakest boundedness assumption" claim is incorrect)**: After careful verification, the claim is actually CORRECT. Bounded return (Σ R_h ∈ [0,H] a.s. with R_h ≥ 0) implies both V_h^* ≤ H and R_h ≤ H, but the converse does not hold. For example, one can have R_h = H at multiple time steps (violating Σ R_h ≤ H) while V_h^* ≤ H due to MDP structure. The paper's assumption IS genuinely weaker. The reviewer incorrectly claimed R_h ≤ H is "bounded rewards," but bounded return already implies R_h ≤ H, so this is not an additional constraint.

- **Harsh Critic's Issue 2 (Algorithm not specified for unvisited pairs)**: This is downgraded to a minor presentation issue (see above). The Q_h^k definition explicitly handles N^k=0 by setting Q=H, and the proofs don't rely on r̂^k/P̂^k being defined for unvisited pairs. Standard practice in the RL literature.

- **Requests for additional related work**: Not verified against external sources; removed per rules.

- **Formatting/style nitpicks**: Removed per rules.

## Novel Insights

The quasi-optimism concept is genuinely interesting beyond this paper's specific use. By allowing bounded underestimation rather than enforcing full optimism, it decouples the algorithm's exploration signal from a worst-case guarantee on the estimate's relationship to the true value. This could potentially be applied in settings where full optimism leads to overly conservative behavior (e.g., in gap-dependent analyses or function approximation settings). The key technical trick—using Freedman's inequality to separate variance and 1/N terms, then absorbing the variance terms into a recursive bound that leverages the structural property 2HV^* - (V^*)² ≤ H²—transforms what would normally require explicit variance estimation into a purely count-based bonus.

## Suggestions

- Tone down the abstract and conclusion to reflect that practical superiority is demonstrated on one benchmark family and that the variance independence claim applies specifically to the time-homogeneous tabular setting.
- Add at least 1-2 additional environments (random MDPs, a deterministic MDP) to experiments and include PSRL as a variance-free baseline.
- Explicitly state how unvisited state-action pairs are initialized in Algorithm 1 (e.g., "for pairs with N^k(s,a)=0, we set Q_h^k(s,a)=H without computing r̂^k or P̂^k").
- Include a side-by-side comparison of the logarithmic factors with Zhang et al. (2021a) to substantiate the "sharper" claim.

## Score and Decision

**Calibration**: I compared against: (1) the SdBApv9iT4 paper (horizon-free RL in linear MDPs, scores 5-8, accepted as poster) which had a similar profile of strong theory with limited practical implications; (2) the en3NwykrHW paper (minimax optimal regret with trajectory feedback, scores 3-8, rejected) which had overclaimed theory and lacked experiments; (3) the 8y5vlBuRll paper (action robust RL, scores 5-6, rejected) which was incremental with limited novelty. The current paper has genuine theoretical contributions (quasi-optimism, sharper bounds, weaker assumptions) but overclaims on practical impact and the novelty of variance-free bonuses. It is stronger than en3NwykrHW (which had more serious presentation issues and weaker claims) and comparable in theory quality to SdBApv9iT4 (which was accepted at 6-8 range). The key issue is the overclaiming, not the theory itself.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>