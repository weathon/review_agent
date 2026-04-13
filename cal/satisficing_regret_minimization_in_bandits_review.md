=== CALIBRATION EXAMPLE 22 ===

# Final Consolidated Review
## Summary
This paper studies satisficing regret minimization in bandits, where the goal is to play arms whose mean exceeds a threshold \(S\) as often as possible. The main contribution is **SELECT**, a black-box reduction that wraps any bandit oracle with sublinear standard regret and, in the realizable case \(r(X^*) \ge S\), obtains a **time-horizon-independent satisficing regret bound** depending on the **exceeding gap** \(\Delta_S^* = r(X^*)-S\), while in the non-realizable case preserving the oracle’s standard regret rate. The most interesting aspect is that this extends constant satisficing regret guarantees beyond finite arms to settings like concave and Lipschitz bandits where the usual satisficing gap \(\Delta_S\) can be zero.

## Strengths
- **Targets a real conceptual limitation of prior satisficing-bandit analyses and resolves it in continuous-arm settings.** The paper directly addresses the fact that prior bounds depending on the satisficing gap \(\Delta_S=\min\{S-r(x):r(x)<S\}\) become vacuous when \(\Delta_S=0\), which indeed happens in the paper’s own examples of concave and Lipschitz bandits (Remark 4). Replacing this by dependence on the exceeding gap \(\Delta_S^*=r(X^*)-S\) is the key conceptual advance.
- **The algorithmic decomposition is genuinely insightful and not a routine wrapper.** SELECT combines: (i) sampling a candidate arm from an oracle trajectory, (ii) forced sampling of that candidate, and (iii) an LCB-based continuation/restart rule. Remark 2 gives a convincing reason for each component, especially the point that using an LCB test rather than UCB/empirical-mean testing is what avoids reintroducing \(1/\Delta_S\)-type dependence.
- **The realizable/non-realizable split is handled in a strong “best-of-both-worlds” way.** Theorems 1 and 2 together claim an attractive fallback property: constant satisficing regret when a satisficing arm exists, and no asymptotic loss relative to the base oracle when it does not.
- **The framework is instantiated beyond finite-armed bandits in a way that matches the paper’s motivation.** The concave and Lipschitz corollaries are not decorative; they are exactly the settings where \(\Delta_S=0\) makes prior finite-arm-style analyses inadequate.
- **The lower bounds strengthen the paper’s conceptual message.** Theorems 3 and 4 show \(\Omega(1/\Delta)\) lower bounds in finite-armed and 1D concave settings with \(\Delta=r(X^*)-S\), supporting that dependence on the exceeding gap is not merely an artifact of the analysis.

## Weaknesses

### Fatal
None.

### Major:
- **The central black-box reduction is under-justified in the main paper relative to how broad the claim is.**  
  The paper’s headline claim is quite general: under Condition 1, any oracle with expected regret bound \(C_1 t^\alpha \log^\beta t\) can be converted into constant satisficing regret in the realizable case. But in Section 3, Step 1 is justified only at the level of an *expected* reward gap for a uniformly sampled arm from the oracle trajectory (“we find an arm whose mean reward is at most \(\tilde O(\gamma_i)\) below \(r(X^*)\) in expectation”). For Proposition 2 and the geometric-round intuition to work, the analysis needs a sufficiently strong conditional probability statement that the sampled arm is actually above threshold often enough after accounting for estimation error—not just a small expected suboptimality. This may well be provable from expected regret plus concentration/Markov-style reasoning, and the appendix may contain it, but in the main paper this is the crucial logical step and it is not explained at a level commensurate with the generality of the theorem. For a theory-led paper, this is a substantive presentation weakness because the reduction itself is the core contribution.
- **The empirical section is too thin to substantiate the breadth of the claims.**  
  The experiments are limited to a single hand-crafted instance per problem family, horizons only up to 5000, and do not probe the regimes most relevant to the theory. In particular:
  - there is **no sweep over \(\Delta_S^*\)**, despite the entire theoretical message being about replacing \(1/\Delta_S\) with dependence on \(1/\Delta_S^*\);
  - there is **no ablation** of the three claimed essential ingredients in Remark 2 (trajectory sampling, forced sampling, LCB testing);
  - the finite-arm experiment does **not** test a near-zero satisficing-gap regime where SELECT’s advantage over \(\Delta_S\)-dependent methods should be clearest;
  - the non-realizable experiments set \(S=1.5\) while rewards are in \([0,1]\), which makes non-realizability easy and does not stress the claimed fallback behavior near the boundary \(S \gtrsim r(X^*)\).  
  As a result, the experiments function mainly as sanity checks rather than serious empirical support for a broad framework.
- **Some claim framing is broader than what is actually established.**  
  There are places where the exposition overstates what the paper shows. Most notably, Remark 3 says the finite-arm result is a “major improvement” over Michel et al. (2023). What is clearly established is a different dependence—removing \(\Delta_S\) and replacing it by \(\Delta_S^*\). That is highly meaningful in settings where \(\Delta_S=0\), but it is not a uniform dominance statement for finite arms because \(\Delta_S\) and \(\Delta_S^*\) are different quantities and neither uniformly controls the other. Similarly, the paper sometimes gestures toward a broadly complete picture, but the lower-bound discussion is much more limited: finite-armed and 1D concave only, with no corresponding Lipschitz lower bound and no tightness discussion for the dimensional dependence in Corollaries 2–3.

### Minor
- **The boundary case \(\Delta_S^*=0\) is not discussed clearly enough.**  
  Theorem 1’s constant term scales like \((1/\Delta_S^*)^{\alpha/(1-\alpha)}\), so the result is meaningful only when \(S<r(X^*)\). The paper says the exceeding gap is “positive in general,” but the important boundary case \(S=r(X^*)\) is not “general” and is exactly where this bound becomes vacuous. This does not invalidate the paper, but the scope of the theorem should be stated more explicitly.
- **The practical cost of forced sampling is not discussed much.**  
  Step 2 uses \(T_i=\lceil \gamma_i^{-2}\rceil\) forced samples, and Remark 2 explains why this is analytically useful. But the paper does not quantify the practical tradeoff or discuss whether milder schedules might work empirically or with slightly weaker theory.
- **The instantiated upper bounds leave notable gaps to the provided lower bounds.**  
  For finite arms, the upper bound is \(O(K/\Delta_S^*)\) up to polylogs, while the lower bound is only \(\Omega(1/\Delta_S^*)\). For concave bandits, the upper bound has \(\mathrm{poly}(d)/\Delta_S^*\) while the lower bound is only one-dimensional. This is not a flaw in correctness, but the paper would be stronger if it discussed whether these factors are artifacts of the reduction or potentially necessary.
- **The wording “Since each round of SELECT runs independently” in the proof sketch is imprecise.**  
  What is really needed is a uniform conditional bound on round continuation/termination, not literal probabilistic independence of rounds in the strongest sense. This seems likely to be just imprecise exposition, but for a proof sketch of the main theorem it is a confusing statement.
- **A few notation/presentation inconsistencies make Section 3 harder to follow.**  
  The prose alternates between \(\tilde X_i\) and \(\hat X_i\) for the candidate arm, and Step 2’s prose refers to validating \(\hat X_i\) immediately after Step 1 introduces \(\tilde X_i\). Minor, but noticeable in the most important section.

### Trivial
None.

## Nice-to-Haves
- Add a systematic experiment varying \(\Delta_S^*\) and, for finite arms, also varying \(\Delta_S\) toward zero to directly validate the paper’s main scaling claim.
- Include ablations replacing the LCB test with UCB/empirical mean, and removing forced sampling, to empirically support Remark 2.
- Test harder non-realizable cases with \(S\) just above \(r(X^*)\), rather than far beyond the reward range.
- Provide a short discussion of the crossover point in Theorem 1 between the “constant” term and the \(T^\alpha\) fallback term, since the constant can be quite large when \(\Delta_S^*\) is small.
- Discuss computational overhead, especially when the oracle itself is expensive and must be rerun across rounds.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“No confidence intervals / variability are shown in experiments.”**  
  Removed as a core weakness. For this style of bandit paper, single-curve averaged experiments are standard sanity checks; lack of confidence intervals is not by itself a substantive flaw.
- **“Comparisons to SAT-UCB/SAT-UCB+ in continuous settings are unfair because those are heuristic discretizations.”**  
  Weakened/removed as a direct criticism. The paper explicitly presents these as heuristics (“we also use SAT-UCB and SAT-UCB+ as heuristics by viewing the problem as Lipschitz bandits”), so this is not a hidden unfair comparison. It is still fair to say these baselines do not strongly validate broad superiority.
- **“Theorem 2 is contradicted because in Figure 3b SELECT has higher regret than SAT-UCB/SAT-UCB+.”**  
  Removed as a misunderstanding. Theorem 2 says SELECT matches the **oracle’s** regret order in the non-realizable case, not that it must outperform every heuristic baseline.
- **Generic requests for more baselines or justification of using a particular bandit rule.**  
  Removed in generic form. The paper’s contribution is a reduction around an arbitrary oracle, not advocacy of one specific base bandit algorithm.

## Novel Insights
The paper’s most important contribution is not just “another satisficing algorithm,” but a change in what quantity controls difficulty: it reframes realizable satisficing from being limited by the *distance from below-threshold arms to the threshold* (\(\Delta_S\)) to being controlled by the *slack of the optimum above the threshold* (\(\Delta_S^*\)). That reframing is exactly what makes continuous-arm settings tractable. The algorithmic structure reflects this shift: Step 1 only needs to find arms near-optimal in the standard-regret sense, and Step 3 only needs to reject candidates whose empirical evidence fails to certify they are above threshold. This is a neat conceptual bridge between ordinary regret minimization and satisficing. At the same time, the experiments do not yet illuminate where this bridge is practically strongest—namely near the hard regime of small \(\Delta_S^*\) and vanishing \(\Delta_S\).

## Suggestions
- Strengthen the main-paper proof sketch around Step 1 \(\rightarrow\) Proposition 2: explicitly show how an expected regret guarantee for the oracle implies a sufficiently large probability that a uniformly sampled point on its trajectory is near-optimal / satisficing.
- Narrow several claims in the framing: say the method gives a new dependence on \(\Delta_S^*\), rather than implying uniform improvement over prior finite-arm results.
- Add experiments that vary \(S\) to sweep \(\Delta_S^*\), including near-boundary realizable and non-realizable regimes.
- Add ablations for the three components emphasized in Remark 2.
- Discuss whether the \(K\), \(\mathrm{poly}(d)\), and Lipschitz-dimensional factors are likely artifacts of the generic reduction or inherent.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0]
Average score: 6.7
Binary outcome: Accept
