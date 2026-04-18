## Summary

The paper proposes Classifier-Free Diffusion Generation (CFDG), a data augmentation method for Offline-to-Online RL that uses a conditional diffusion model with classifier-free guidance to separately generate offline-style and online-style synthetic transitions. The key insight is that offline and online data serve different roles in O2O RL and have different distributions, so they should be augmented independently via conditioning on a binary data-type label. CFDG is evaluated on D4RL Locomotion and AntMaze tasks with three base O2O algorithms (IQL, PEX, APL), showing improvements over baselines and existing diffusion augmentation methods (SynthER, EDIS).

## Strengths

- **Well-motivated problem framing**: The analysis of why offline and online data should be treated differently during augmentation is intuitive and well-argued. The critique that EDIS generates online-policy-aligned data from offline sources (Section 3.1) while online data itself is more aligned with the current policy is compelling and clearly articulated.

- **Clean methodological design**: Using classifier-free guidance with a binary label (offline vs. online) to generate both data types from a single diffusion model is an elegant solution that avoids training separate models. Algorithm 1 and the two data-usage paradigms (concatenation vs. OORB-based) provide clear integration recipes.

- **Multi-algorithm evaluation**: Testing CFDG on three qualitatively different O2O RL algorithms (IQL, PEX, APL) that follow two distinct data utilization paradigms demonstrates the generality of the approach and goes beyond many similar works that evaluate on a single base algorithm.

- **Improvements on challenging settings**: The most notable gains appear on medium-replay and random datasets (e.g., IQL hopper-mr-v2: 66→86; PEX walker2d-mr-v2: 101→112; APL walker2d-mr-v2: 70→109), which are typically the hardest scenarios in O2O RL.

## Weaknesses

### Fatal
None.

### Major

- **Statistical significance concerns for claimed improvements** — Several entries in Table 1 show very large standard deviations that overlap substantially between baseline and CFDG, and some tasks show performance *degradation*. Examples: APL halfcheetah-medium-replay goes from 76±40 to 96±2 (overlapping error bars given the baseline's enormous variance); IQL hopper-random drops from 16±13 to 10±1; PEX walker2d-random goes from 18±10 to 65±37 (the latter's huge variance makes interpretation unreliable). The claim of "15% average improvement" appears to be computed from the sum of normalized scores divided by the number of tasks, which can be distorted by a few large improvements masking no change or degradation elsewhere. No per-task statistical tests or even straightforward counts of wins/losses/ties are provided. This significantly weakens the core empirical claims.

- **Missing ablation on classifier-free guidance vs. alternatives, especially two separate unconditional models** — The paper's central architectural claim is that classifier-free guidance with data-type labels is beneficial. Yet the ablation (Section 4.3) only compares "CFDG augmenting online data" vs. "CFDG augmenting both." The most natural alternative—training two separate unconditional diffusion models, one on offline data and one on online data—is never tested. This directly tests whether classifier-free guidance adds value beyond simply conditioning on data type. Additionally, the guidance scale $w$ (Equation 7) is never varied despite being a key hyperparameter. Without these ablations, the paper cannot attribute improvements to classifier-free guidance specifically rather than to data augmentation in general or to having two data sources.

### Minor

- **Negative and high-variance results are not discussed** — Several tasks show degradation with CFDG (IQL hopper-random: 16→10; PEX hopper-random: 8→8, no change; APL hopper-medium: 103→99). The paper never acknowledges these cases or discusses when/why CFDG might fail. This makes the overall assessment one-sided.

- **Hyperparameter sensitivity is acknowledged but not analyzed** — The synthetic data ratio $r=1/3$, the generated online-to-offline ratio of 8:2, and the generation frequency $T_{\text{diff}}$ are fixed across all tasks. The conclusion itself notes that "the ratio of offline to online data can significantly impact performance," yet no sensitivity analysis is provided in the main experiments. This is especially concerning given the 8:2 ratio strongly biases toward online-style generation.

- **Computational overhead is not reported** — Diffusion model training and periodic data generation add non-trivial computational cost. The paper claims this approach "greatly reduces time costs" compared to training separate models, but provides zero quantitative evidence (wall-clock time, FLOPs, etc.). As periodic diffusion generation is known to be expensive, this gap limits practical assessment.

- **Limited evaluation scope** — APL results are missing for AntMaze tasks, and no Kitchen or Adroit tasks are evaluated. Given that the paper claims CFDG is "versatile" and "can be integrated with existing offline-to-online RL algorithms," testing only on MuJoCo locomotion and a subset of AntMaze is insufficient to fully establish this claim.

- **t-SNE distribution analysis is qualitative and limited to one environment** — The core motivation rests on Figure 1 (t-SNE), which is non-metric and shown for a single task. No quantitative distributional metrics (e.g., MMD, Wasserstein distance) or multi-environment analysis are provided to support the claim that offline and online data have systematically different distributions.

### Trivial
- The paper uses "O2O RL" in the abstract but "OZO RL" in Section 3.1 (likely a typo from character similarity between "2" and "Z").

## Nice-to-Haves

- Ablation on the guidance scale $w$ and comparison with two separate unconditional diffusion models to isolate the contribution of classifier-free guidance.
- Hyperparameter sensitivity analysis for $r$, the 8:2 ratio, and $T_{\text{diff}}$.
- Quantitative evaluation of generated data quality (e.g., dynamics consistency errors, reward prediction accuracy).
- Per-task win/loss statistics and confidence intervals to substantiate the improvement claims.
- Wall-clock time comparison between CFDG and baselines.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"SynthER and EDIS comparisons are unfair because adaptation details are unspecified"** (Harsh Critic #2): While the paper could be more explicit about how SynthER/EDIS were adapted to O2O, Section 4.2 describes the conceptual differences and Figure 2 shows comparisons on the same IQL base. The asymmetry, if any, would make SynthER/EDIS *easier* to beat (since they were not designed for this setting), not harder. Per the hard rules, criticisms about unfair comparisons that favor the authors' method are removed.

- **"The paper overclaims novelty"** (Harsh Critic #1, partially): While the conceptual step is incremental, the paper honestly positions itself as a data augmentation method for O2O RL, not as a fundamentally new framework. The novelty is in the *application insight* (separating offline/online generation in O2O), not the technique itself. This concern is already reflected in the novelty-related minor weakness above.

- **"Missing simple non-diffusion augmentation baselines (Gaussian noise, mixup)"** (Harsh Critic #5): The paper compares against SynthER, which is an unconditional diffusion baseline. This partially addresses the question of whether any data augmentation helps vs. whether the conditional design helps. Non-diffusion baselines are somewhat outside scope for a paper specifically about diffusion-based augmentation.

- **"EDIS using offline data is counterintuitive"** (Neutral Reviewer): The paper itself addresses this argument in Section 3.1, noting EDIS's rationale (energy-guided correction) while arguing online data is more aligned with the current policy. The paper already engages with this perspective.

## Novel Insights

The most insightful observation from the review is that the 8:2 online-to-offline generation ratio implicitly encodes an assumption that online-style data is more valuable than offline-style data during fine-tuning—which directly mirrors the well-known O2O RL insight that online data improves convergence while offline data prevents suboptimal convergence. If this ratio were studied systematically, it could reveal whether the optimal generation ratio shifts during training (e.g., more offline-style generation early for diversity, more online-style later for stability), potentially yielding an even stronger adaptive method.

## Suggestions

- Run an ablation comparing CFDG against two separate unconditional diffusion models (one per data type) on at least 4-6 tasks. If the conditional model with classifier-free guidance outperforms separate models, it validates the architectural contribution; if not, it clarifies what actually drives the gains.
- Provide a per-task win/loss/tie table across all 16 tasks and 3 base algorithms (48 comparisons total), making it transparent where CFDG helps and where it doesn't.
- Sweep at least two values for the 8:2 ratio (e.g., 6:4 and 9:1) and report the sensitivity to demonstrate robustness or identify task-specific tuning needs.

## Score and Decision

**Calibration**: I compared against similar diffusion-based RL augmentation papers and O2O RL papers:
- GODA (diffusion data augmentation for offline RL): avg score ~4.0 — had novelty concerns, missing baselines, hyperparameter issues
- ATraDiff (diffusion for RL data augmentation): avg score ~4.6 — similar concerns about compute, generated data quality
- RTDiff (diffusion augmentation for offline RL): avg score ~5.75 — stronger empirical validation
- SERA (O2O RL with data augmentation): avg score ~4.75 — similar concerns about evaluation scope and variance

CFDG is stronger than GODA on evaluation breadth (3 base algorithms, 2 augmentation baselines) but weaker than RTDiff on empirical rigor (missing key ablations, high variance). It sits in the middle of this range. The incremental novelty and the significant empirical gaps (missing ablations on core claims, high variance undermining "15% improvement") pull it below borderline accept.

**Score: 4.5**

The paper proposes a reasonable idea (conditional diffusion with data-type labels for O2O augmentation) with encouraging but inconsistently strong empirical results. However, the novelty is incremental (applying an established diffusion technique with a binary label), the core claims about classifier-free guidance are not isolated via ablation, and several results have high variance or show degradation. These issues substantially weaken the evidence supporting the main claims.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>