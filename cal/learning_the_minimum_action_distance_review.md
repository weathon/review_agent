=== CALIBRATION EXAMPLE 53 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title "Learning the Minimum Action Distance" is accurate and appropriately scoped. The abstract is mostly well-calibrated, though the claim of "significantly outperforms existing state representation methods" is overstated given only two baselines are evaluated (QRL and Hilbert). Notably, Table 1 shows TDMadDist, not MadDist, achieves the best result on one of six environments (PM Giant Navigate: 0.99 ± 0.05 for TDMadDist vs. 0.93 ± 0.17 for MadDist), which slightly contradicts the framing that MadDist is uniformly dominant. The claim of learning from "neither reward signals nor the actions executed by the agent" is a meaningful practical constraint that is understated and deserves more emphasis.

---

### Introduction & Motivation

The motivation is clear and the problem is well-scoped. The observation that prior MAD approximation methods impose symmetric distance metrics that cannot capture the inherently asymmetric nature of the MAD is a legitimate and well-stated gap. The three claimed contributions are specific and verifiable. One issue: the introduction states prior work has not been "systematically evaluated on their ability to approximate the MAD function itself," yet the most directly comparable prior work — Steccanella & Jonsson (2022), which the paper explicitly builds upon — is not included as a baseline in the experiments. This disconnect is a significant problem.

---

### Background & MAD Formulation (Sections 3–4)

The MDP formalism is standard. The constrained optimization formulation (Equation 1) for MAD is elegant and well-motivated. The proof of uniqueness in Appendix A is correct for finite state spaces: the induction establishes that any feasible d satisfies d(s,s') ≤ d_MAD(s,s') pointwise, and since d_MAD is itself feasible, it is the unique maximizer.

**Concern:** For continuous state spaces, the uniqueness and optimality arguments are informal. The sum over S² is uncountably infinite, and the measure-theoretic justification for invoking "maximum sum" arguments is absent. The paper acknowledges continuous state spaces are important but only fully proves the finite case.

**Concern:** For stochastic MDPs, the Bellman-style recursion used in TDMadDist (Section 6.2, Eq. 8) relies on d_MAD(s_i, s_j) = 1 + d_MAD(s_{i+1}, s_j), which is exact only for deterministic dynamics. In stochastic settings, d_MAD is a lower bound on any policy's actual step count, not an expected value, and the "next state" s_{i+1} on the trajectory is one realization drawn from the transition distribution. The paper does not formally justify that this Bellman equation holds for MAD under stochastic dynamics — it would hold for the value of the *support-based shortest path*, but s_{i+1} is sampled, not chosen optimally. This needs clarification.

---

### Asymmetric Distance Metrics (Section 5)

The proposed d_simple quasimetric (Equation 3) is a genuine contribution: a simple, computationally cheap quasimetric that the paper proves satisfies the triangle inequality and positive homogeneity (Appendix B, Propositions 1–3, which are correct). The comparison of d_simple with d_WN (Wide Norm) and d_IQE (Interval Quasimetric Embedding) is valuable.

**Concern:** The α hyperparameter in d_simple (balancing max and mean aggregation) is not discussed in terms of how it is set or whether it is tuned per environment. Tables 2–3 show the hyperparameters used but do not list α, which is notable given that the ablation only varies quasimetric type and not this parameter.

**Concern:** The paper does not provide intuition for *why* d_simple outperforms d_WN and d_IQE (Appendix E.2). d_WN applies a learned transformation, which might be expected to be more expressive. The empirical finding that a simpler, fixed quasimetric outperforms learned ones deserves more analysis — is it an optimization difficulty issue? An inductive bias advantage?

---

### Learning Algorithms (Section 6)

**MadDist:** The composite loss (Eq. 4) extends Steccanella & Jonsson (2022) in three ways: (1) scale-invariant normalization of the squared error by (j−i)² in Eq. 5 — which is well-motivated to prevent long-horizon pairs from dominating; (2) a contrastive repulsion term L_r (Eq. 6) that serves as a proxy for the maximize objective; and (3) an upper-bound penalty L_c (Eq. 7) applied only to pairs within H_c steps. These are sensible modifications.

**Concern:** The key hyperparameter H_c = 6 for L_c appears fixed across all environments. This means only pairs within 6 steps have the upper bound enforced. For large mazes (OGBench Giant Maze can span hundreds of steps), this is a severe restriction. Why is H_c not tuned or at least ablated across environments? This seems like a critical hyperparameter whose effect on the quality of long-horizon estimates is unexplored.

**Concern:** The d_max hyperparameter in L_r (set to 100 or 500 depending on environment, Table 2) and wr (set to 1 or 10) indicate environment-specific tuning. This weakens the comparison if baselines did not receive equivalent tuning. The paper should clarify the tuning protocol.

**TDMadDist:** The bootstrapped target (Eq. 8) uses min(j−i, 1+d_{θ'}(s_{i+1}, s_j)) as the regression target for d_θ(s_i, s_j). This is a natural TD extension. However, TDMadDist **consistently underperforms MadDist** in most settings, and the Discussion (line 635) only notes this without explanation. In the RL literature, TD instability, value overestimation, and slow propagation of bootstrapped signals are well-studied failure modes. Is TDMadDist suffering from any of these? The lack of analysis is a missed opportunity.

---

### Experiments & Results (Section 7)

**Environments:** The benchmark suite is a genuine contribution — having ground-truth MAD values allows a systematic evaluation that is rare in this literature. The diversity of environments (symmetric/asymmetric, discrete/continuous, stochastic/deterministic, noisy observations) is appropriate.

**Concern — Missing Baseline:** Steccanella & Jonsson (2022) is the most directly relevant prior work and the algorithm from which MadDist is derived. It is not included as a baseline. This is the most significant experimental gap in the paper. Without this comparison, it is impossible to attribute the observed improvements to the specific algorithmic novelties (scale-invariant loss, quasimetric, contrastive term) rather than to other implementation details.

**Concern — Inconsistent Seed Reporting:** The paper states (line 614) "All reported results are means over five independent runs (random seeds)." However, Figure 3's caption says "Shaded regions minimum and maximum values across **three** random seeds," as do Figures 11 and 12. This discrepancy is unexplained and undermines confidence in the reported variances.

**Concern — Partial Results in Main Paper:** Figure 3 shows results for only 3 of the 8+ evaluation environments (KeyDoorGridWorld, CliffWalking, OGBench Giant Maze). While the full results are in Appendix F, the selection of which environments to show in the main paper should be transparent. Are these representative, or are they cherry-picked? A summary table in the main paper (akin to Table 1 for planning) would be stronger.

**Concern — Planning Experiment (Appendix H):** Table 1 shows impressive success rates for MadDist on OGBench. However, the planning algorithm (random shooting MPC, described in Appendix H) requires access to the **true environment simulator** at test time (step 2: "use the true environment simulator to roll out the corresponding state trajectory"). This is a strong assumption that significantly limits real-world applicability. The paper acknowledges "the true simulator" in the appendix but does not discuss this limitation in the main text or conclusion. For many practical applications (robotics, real-world systems), simulator access at planning time is not available.

**Concern — No Comparison with GCRL Methods for Planning:** Table 1 compares MadDist against QRL and Hilbert for planning, but does not include vanilla GCRL methods (e.g., GCSL, HIQL) that are standard baselines in the OGBench benchmark. The comparison is limited to methods that specifically learn distance representations, which may not represent the full landscape of competitive approaches.

---

### Writing & Clarity

The paper is generally well-written. Two specific issues:

- Line 2258: "The results appear in Figure 12 and in Figure 12" — clearly a self-reference typo (should reference two distinct figures).
- The relationship between MadDist and Steccanella & Jonsson (2022) is primarily described in Section 4 but the precise delta (which modifications are new vs. inherited) could be stated more explicitly in Section 6.1.

---

### Limitations & Broader Impact

The paper's conclusion section gestures at limitations but misses several important ones:

1. **Simulator access for planning:** Not acknowledged.
2. **Coverage bias from random policy:** For hard-exploration environments (large mazes), a random policy provides poor coverage. The learned metric is constrained to pairs observed in trajectories. For the OGBench "Stitch" dataset (short trajectories of at most 4 cells), long-horizon distances must be inferred by generalization — the paper shows this works (Table 1) but doesn't analyze when/why.
3. **MAD vs. practically achievable distances:** The paper acknowledges that in highly stochastic environments, MAD may be an overly optimistic lower bound. This is a fundamental tension that limits the utility of MAD as a reward shaping signal in such settings, and deserves more discussion.
4. **Missing Steccanella & Jonsson baseline:** Not acknowledged as a limitation.

---

### Overall Assessment

This paper addresses a real and underexplored problem — learning the Minimum Action Distance from action-free state trajectories — and makes genuine contributions: the d_simple quasimetric (with formal proofs), the MadDist and TDMadDist algorithms, and a benchmark suite with ground-truth MAD. The MadDist results on OGBench planning (Table 1) are impressive. However, the paper has significant structural weaknesses that limit its evaluative credibility. The most serious is the **absence of Steccanella & Jonsson (2022) as a baseline**, despite it being the paper from which MadDist is directly derived — without this comparison, the source of empirical improvements is ambiguous. The **seed inconsistency** (5 seeds claimed, 3 shown in figures) undermines reproducibility claims. The **planning evaluation relies on true simulator access**, a major assumption not discussed prominently. The **TDMadDist underperformance is unexplained**, and the theoretical justification of the TD update in stochastic settings needs strengthening. In its current form, the paper is borderline for ICLR — the benchmark and d_simple quasimetric contributions are valuable, but the experimental rigor and theoretical completeness fall short of the ICLR bar without revision.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes two algorithms, MadDist and TDMadDist, to learn state representations where distances approximate the Minimum Action Distance (MAD) using only state trajectories without reward signals. The authors introduce a simple asymmetric quasimetric and provide a benchmark suite with ground-truth MAD for various environments, demonstrating superior performance in representation accuracy and downstream planning compared to symmetric and other asymmetric baselines. By leveraging trajectory path lengths as upper-bound supervision, the methods effectively capture the directional structure of environments with irreversible dynamics.

### Strengths
1.  **Clear Formulation of MAD Approximation:** The paper provides a rigorous mathematical derivation of the MAD as an optimization problem and connects it directly to the learning objectives via quasimetrics. Appendix A proves the uniqueness of the MAD solution to the constrained optimization problem, providing strong theoretical grounding.
2.  **Addressing Asymmetry:** A significant strength is the explicit focus on learning asymmetric quasimetrics. The paper correctly identifies that existing methods (e.g., Hilbert representations) often rely on symmetric metrics that fail in environments like CliffWalking or KeyDoorGridWorld. Empirical results confirm this advantage, with MadDist outperforming Hilbert baselines by margins >0.50 in correlation in asymmetric settings (Figure 3).
3.  **Comprehensive Benchmark with Ground Truth:** The introduction of a diverse suite of environments (NoisyGridWorld, PointMaze, OGBench) where ground-truth MAD is computable is highly valuable for the community. This facilitates rigorous evaluation of representation learning techniques, which are often opaque compared to standard RL benchmarks.
4.  **Strong Downstream Utility:** The evaluation extends beyond correlation metrics to include a downstream planning task using MPC. Table 1 shows MadDist achieves near-perfect success rates in complex mazes (e.g., 1.00 in Giant Navigate), suggesting the learned metric is practically useful rather than just mathematically accurate.

### Weaknesses
1.  **Incremental Novelty Regarding Prior Work:** The core idea of learning MAD-like distances is not new. Steccanella & Jonsson (2022) and Wang et al. (2023b) (QRL) explicitly address this. The distinction between `MadDist` and QRL is somewhat narrow (trajectory path length supervision vs. Lagrangian locality constraints), yet `MadDist` claims a "novel quasimetric" (Equation 3) which is a simple convex combination of ReLU reductions. The paper may overstate the novelty of the quasimetric component compared to QRL's IQE approach.
2.  **Inconsistent Performance of TDMadDist:** While `MadDist` performs well, `TDMadDist` (which aims to improve robustness via bootstrapping) often underperforms `MadDist` and sometimes QRL in Table 1 (e.g., 0.74 vs 0.95 in PM Giant Stitch). The paper mentions this in discussion but lacks a deep analysis of why TD bootstrapping degrades performance here compared to direct trajectory supervision.
3.  **Limited Computational Efficiency Analysis:** The paper focuses on representational quality but does not thoroughly discuss the computational costs of the proposed `d_simple` quasimetric versus competitors like IQE or Wide Norm. IQE reshapes embeddings into matrices; does `d_simple` offer efficiency gains sufficient to justify its use over potentially richer quasimetrics?
4.  **Simplistic Planning Baseline:** While the planning task is valuable, the "random shooting" MPC planner is computationally expensive (100 candidate sequences per step) and relies on a true simulator. The evaluation does not compare against more efficient planners or standard RL baselines to show if MAD actually accelerates *learning* (sample efficiency) rather than just planning success with a fixed metric.

### Novelty & Significance
The **novelty** is moderate. The framing of MAD learning via supervised quasimetric embedding is an extension of prior work (Wang et al., 2023b) rather than a radical departure. However, the specific combination of the trajectory-length supervision signal and the simple ReLU-based quasimetric offers a clean, effective alternative to more complex IQE or Hilbert approaches. The **significance** lies primarily in the benchmark creation and the demonstration that asymmetric supervision significantly aids learning in directional environments. It provides a clear resource for other researchers to evaluate distance-learning methods. The methods have practical significance for goal-conditioned RL where rewards are sparse or unavailable.

### Suggestions for Improvement
1.  **Deepen Novelty Discussion:** Explicitly contrast the optimization landscape of `MadDist` (maximizing distances subject to trajectory upper bounds) with `QRL` (Lagrangian optimization of locality/separation). Clarify why `MadDist` finds better global structure than `QRL` in the text, as this distinguishes the algorithmic contribution beyond just the quasimetric choice.
2.  **Analyze TDMadDist Failure Modes:** Investigate why the TD update (Equation 8) underperforms the direct supervision loss in asymmetric tasks. Is it due to error propagation or incompatibility with the strict upper-bound constraints? Addressing this would strengthen the algorithmic contribution.
3.  **Include Sample Efficiency Results:** To align with ICLR's focus on RL contributions, consider adding a sample efficiency experiment (e.g., planning success vs. dataset size or training steps) to show if the learned representations enable faster convergence in downstream RL tasks, not just better planning with a fixed dataset.
4.  **Clarify Quasimetric Design:** Provide more intuition or ablation on why `d_simple` outperforms `IQE`. Since `IQE` is designed specifically for interval semantics, a strong empirical result suggests `d_simple` captures the relevant structure differently. A geometric interpretation of `d_simple` would be beneficial.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Pixel-based environments (e.g., DMControl, Atari):** The current evaluation relies on low-dimensional state vectors; ICLR representation papers require validation on high-dimensional visual inputs to prove generalization beyond coordinate access.
2. **Action-free contrastive baselines:** Compare against standard representation methods like CPC or DRIM that also use state trajectories, isolating whether quasimetrics specifically drive gains versus general contrastive learning.
3. **Asymmetry-specific error metrics:** Pearson/Spearman correlations aggregate over all pairs and obscure directional errors; add metrics specifically measuring $|d(s, s') - d(s', s)|$ against ground truth asymmetry.
4. **Downstream model-free RL sample efficiency:** The planning experiment uses the true simulator for rollouts, undermining the "unknown environment" claim; show actual RL training curves where dynamics are unknown.
5. **True continuous ground truth validation:** The "continuous" experiments rely on discretized graphs for ground truth; evaluate on environments where MAD is defined analytically without grid approximation to verify continuous space claims.

### Deeper Analysis Needed (top 3-5 only)
1. **TDMadDist failure mode analysis:** TDMadDist consistently underperforms MadDist in results; provide an analysis of bootstrap error accumulation in distance metrics to justify including it as a contribution.
2. **Triangle inequality violation quantification:** Despite using quasimetric heads, neural networks may violate constraints during training; measure and report the frequency/magnitude of triangle inequality violations.
3. **Computational complexity comparison:** The paper claims efficiency but lacks FLOPs or wall-clock time comparisons against IQE and Hilbert baselines to substantiate the computational benefit.
4. **Sensitivity to dataset coverage density:** Analyze how performance degrades as the state space coverage becomes sparser, which is critical for justifying usability in hard-exploration domains.
5. **Embedding capacity ablation:** Determine if the success of $d_{simple}$ is due to the quasimetric structure or simply having sufficient embedding capacity compared to the baselines.

### Visualizations & Case Studies
1. **Asymmetry scatter plots:** Plot predicted $d(s, s')$ vs. $d(s', s)$ for asymmetric environments (KeyDoor, Cliff) to visually confirm the model captures directionality rather than collapsing to symmetry.
2. **Embedding topology vs. graph structure:** Use t-SNE/PCA on embeddings colored by ground-truth graph distance to reveal if the learned geometry aligns with the environment's connectivity.
3. **Planning failure heatmaps:** Visualize specific regions in the maze where the planner gets stuck due to metric inaccuracies, exposing where the representation fails to guide navigation.
4. **Constraint violation training curves:** Plot the magnitude of the constraint loss $L_c$ over training steps to show whether the model successfully enforces the upper bound constraints.

### Obvious Next Steps
1. **Integrate into model-free RL loop:** Demonstrate that using MAD as a reward shape or heuristic actually accelerates policy learning in standard RL benchmarks, not just planning with known dynamics.
2. **Extend to visual domains with distractors:** Validate the robustness claims by adding visual distractors or camera shifts to test if the metric remains invariant to irrelevant observation changes.
3. **Provide theoretical error bounds:** Derive bounds on the approximation error relative to dataset coverage density to move beyond empirical observation to theoretical grounding.
4. **Remove or justify TDMadDist:** Either improve the TD formulation to match MadDist performance or relegate it to an appendix, as it currently weakens the core algorithmic contribution.

# Final Consolidated Review
## Summary

This paper proposes two algorithms, MadDist and TDMadDist, for learning the Minimum Action Distance (MAD)—the minimum number of actions required to transition between states—from state trajectories alone, without requiring reward signals or action labels. The authors introduce a novel simple quasimetric (d_simple) that provably satisfies the triangle inequality, and provide a benchmark suite of environments with ground-truth MAD values, enabling systematic evaluation across deterministic/stochastic dynamics, discrete/continuous state spaces, and symmetric/asymmetric transition structures.

## Strengths

- **Theoretical grounding:** The paper provides a clean constrained optimization formulation of MAD (Eq. 1) with a uniqueness proof in Appendix A, establishing that MAD is the unique solution maximizing pairwise distances subject to one-step reachability and triangle inequality constraints.

- **Addresses a fundamental limitation of prior work:** The paper correctly identifies that existing methods rely on symmetric distance metrics, which cannot capture the inherently asymmetric MAD in environments with irreversible dynamics. Empirical results (Figure 3) show MadDist outperforming symmetric Hilbert baselines by >0.50 correlation in asymmetric environments like CliffWalking and KeyDoorGridWorld.

- **Benchmark with ground-truth MAD is valuable for the community:** The suite of environments where MAD is exactly computable (discrete grids) or approximable (mazes via Floyd-Warshall) provides a principled evaluation framework that is rare in representation learning literature.

- **d_simple quasimetric is a genuine contribution:** The proposed ReLU-based quasimetric (Eq. 3) is computationally efficient and, unlike learned quasimetrics, requires no additional parameters. Appendix B proves it satisfies triangle inequality and positive homogeneity. Ablation (Appendix E.2) shows it outperforms both Wide Norm and IQE alternatives.

- **Strong downstream planning results:** MadDist achieves near-perfect success rates (Table 1) on OGBench PointMaze planning tasks, demonstrating practical utility beyond correlation metrics.

## Weaknesses

- **Missing baseline from most directly comparable prior work:** Steccanella & Jonsson (2022) is the paper from which MadDist is derived—Section 4 explicitly builds on their loss function—but it is not included as an experimental baseline. Without this comparison, it is impossible to attribute observed improvements to the specific algorithmic novelties (scale-invariant loss, quasimetric choice, contrastive term) rather than implementation differences.

- **Inconsistent reporting of experimental seeds:** The paper states (line 614) "All reported results are means over five independent runs (random seeds)," yet Figure 3 and Figures 11–12 captions specify "Shaded regions minimum and maximum values across three random seeds." This discrepancy undermines reproducibility claims and should be clarified or corrected.

- **TDMadDist underperformance is unexplained:** Table 1 shows TDMadDist achieving only 0.70–0.74 success rates on Stitch environments versus MadDist's 0.99–1.00, yet the paper provides no analysis of why bootstrapped targets degrade performance. In RL literature, TD instability and error accumulation are known failure modes—the paper should investigate whether these explain the gap.

- **Planning evaluation assumes access to true simulator:** The MPC planner (Appendix H, step 2) "use[s] the true environment simulator to roll out the corresponding state trajectory." This is a strong assumption that limits real-world applicability (e.g., robotics, physical systems). The limitation is not discussed prominently in the main text.

- **Theoretical gaps for continuous state spaces:** The uniqueness proof in Appendix A covers finite state spaces. For continuous state spaces, the "sum over S²" is not well-defined, and the measure-theoretic justification is absent. The paper acknowledges the continuous case is important but leaves this gap.

- **Key hyperparameter H_c = 6 fixed across all environments:** The constraint loss L_c (Eq. 7) enforces upper bounds only for state pairs within H_c = 6 steps on trajectories. For OGBench Giant Maze, which spans hundreds of steps, this may be insufficient for long-horizon distance accuracy. No ablation over H_c is provided.

- **The α hyperparameter in d_simple is not discussed:** Equation 3 defines d_simple with a weight α balancing max and mean aggregations, but Tables 2–3 do not list its value, and no ablation studies explore its effect.

## Nice-to-Haves

- Analysis of why d_simple outperforms learned quasimetrics (Wide Norm, IQE)—optimization dynamics, inductive bias, or expressivity trade-offs would strengthen the empirical narrative.

- Computational efficiency comparison (wall-clock time, FLOPs) between d_simple, Wide Norm, and IQE to substantiate claimed efficiency benefits.

- Ablation over H_c across environments with different horizon lengths to assess constraint enforcement sufficiency.

- Comparison against action-free representation learning baselines (e.g., CPC, time-contrastive representations) to isolate whether gains come from quasimetric structure versus trajectory-based supervision.

## Removed Points

*These points were flagged to be removed; treat them with caution.*

- **"TDMadDist achieves the best result on one environment"** — While factually correct (PM Giant Navigate: 0.99 for TDMadDist vs. 0.93 for MadDist), this is a minor point. MadDist achieves best on 5/6 environments, and the abstract's claim of "significantly outperforms existing methods" refers to baseline comparisons (QRL, Hilbert), not TDMadDist.

- **"Pixel-based environments required"** — Scope creep. The paper's contribution is learning MAD from state trajectories; low-dimensional state vectors are a valid setting.

- **"Embedding capacity ablation missing"** — Already addressed in Appendix E.1, which shows performance saturates around latent dimension 10 and remains stable for larger dimensions.

- **"Triangle inequality violation measurement"** — The quasimetric formulation enforces triangle inequality by construction; this is a theoretical guarantee, not an empirical concern.

- **"Sample efficiency in downstream RL"** — Beyond the paper's stated scope, which focuses on representation quality and planning with learned metrics, not policy learning.

## Novel Insights

The paper makes a valuable conceptual contribution by reframing MAD learning as maximizing pairwise distances subject to quasimetric constraints, rather than simply fitting temporal distances. The insight that d_simple—arguably the simplest possible quasimetric construction—outperforms more expressive alternatives like IQE suggests that the inductive bias of a fixed, simple distance function may be more valuable than learning the quasimetric itself. This raises an interesting question for future work: whether the success of quasimetric learning lies more in the optimization landscape of the loss function than in the flexibility of the distance parametrization.

## Suggestions

- Add Steccanella & Jonsson (2022) as a baseline, or explicitly justify its exclusion with a description of how MadDist's objective differs algorithmically.

- Correct the seed inconsistency—either report results consistently over 5 seeds or update the text to reflect 3 seeds.

- Add a brief analysis of TDMadDist's failure modes, even if speculative (e.g., bootstrap error accumulation, incompatibility with strict upper-bound constraints).

- Discuss the true-simulator assumption for planning in the main text or conclusion, noting it as a limitation for real-world deployment.

- Report the α hyperparameter value used for d_simple and optionally ablate its effect.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 2.0, 4.0]
Average score: 4.5
Binary outcome: Reject
