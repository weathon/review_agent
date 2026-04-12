## Summary
The paper studies learning the Minimum Action Distance (MAD) from state-only trajectories, without rewards or actions, by fitting state embeddings whose induced distances approximate shortest-path distance on the support graph of the MDP. Its main technical contributions are two learning objectives (MadDist and TDMadDist), support for asymmetric distances via quasimetrics, and a particularly simple ReLU-based quasimetric (`d_simple`) that appears empirically strong across several navigation-style environments with known ground-truth MAD.

## Strengths
- **The paper identifies and directly targets an important mismatch in prior MAD approximation work: symmetry.** The core motivation is well-founded in the paper itself: MAD is generally asymmetric in environments with irreversible transitions, and the experiments include exactly such cases (e.g., KeyDoorGridWorld and CliffWalking). This is not a generic “asymmetry matters” claim; the environments are constructed so symmetry is a real failure mode, and the reported gains over the symmetric Hilbert baseline are correspondingly large.
- **`d_simple` is a genuinely useful technical contribution, not just a minor variant.** Equation (3) defines a very lightweight quasimetric based on ReLU coordinate differences and max/mean aggregation, and Appendix B gives a self-contained proof of the triangle inequality and positive homogeneity. The ablations in Appendix E further support that this simple construction is not merely cheaper, but often better than IQE and Wide Norm within the authors’ own framework.
- **The paper provides a principled optimization view of MAD and connects it to learnable objectives.** Section 4 and Appendix A/C formulate MAD as the maximal feasible distance under identity, one-step, and triangle constraints, then derive tractable surrogates using quasimetric embeddings. Even if the final objectives are imperfect proxies, the conceptual bridge from shortest-path constraints to representation learning is clear and technically substantive.
- **The benchmark design is thoughtfully aligned with the paper’s claims.** The environments cover deterministic/stochastic settings, discrete/continuous observations, noisy observations, and asymmetric dynamics, with known or constructed ground-truth MAD. This controlled setup is useful for evaluating a representation that is otherwise difficult to assess.
- **The downstream planning experiment is well-chosen to isolate heuristic quality.** Appendix H explicitly uses random-shooting MPC with the true simulator so that success depends on whether the learned distance provides a useful progress signal, rather than conflating the result with learned dynamics or policy optimization quality. As a test of “is this a useful goal heuristic?”, this is a sensible and informative choice.

## Weaknesses

###: Fatal
- **The central claim that the method learns the *minimum* action distance is not convincingly supported by the actual training objective.**  
  This is the most serious issue. MadDist’s main objective explicitly regresses distances toward the observed trajectory separation:
  \[
  L_o = \mathbb{E}_{(s_i,s_j)\sim \tau}\left(\frac{d_\theta(s_i,s_j)}{j-i}-1\right)^2
  \]
  while the constraint term only penalizes values *above* that same upper bound:
  \[
  L_c = \mathbb{E}\left[\mathrm{relu}(d_\theta(s_i,s_j)-(j-i))^2\right].
  \]
  Since \(j-i\) is only an upper bound on \(d_{\text{MAD}}(s_i,s_j)\), these losses encourage matching the trajectory path length, not discovering the shortest feasible path across trajectories. The paper states in Section 4 that \(j-i\) “is an upper bound on \(d_{\text{MAD}}(s_i,s_j)\),” but the objective in Section 6.1 then uses that upper bound as the regression target. There is no explicit mechanism that would systematically prefer a smaller value when the sampled trajectory is suboptimal under the random behavior policy used in the main experiments.  
  This is not a minor wording problem: it strikes at whether the method is actually estimating MAD or a trajectory-induced temporal distance surrogate. The planning results and correlations show the learned metric is useful, but they do not fully validate the stronger claim of recovering minimum action distance.

### Major:
- **The evaluation metrics are too insensitive to absolute calibration to substantiate the main claim as stated.**  
  The primary metrics are Pearson correlation, Spearman correlation, and the ratio CV. These are useful for assessing ordering and consistency, but they do not directly measure whether predicted distances equal the true MAD in action-count units. A method could be strongly correlated with MAD while systematically overestimating or underestimating it. This matters especially because the loss already tends to target trajectory lengths \(j-i\), which are upper bounds. If the paper’s core claim were framed more modestly as learning a useful geometry aligned with MAD, the current metrics would be more adequate; for claiming accurate MAD estimation, they are insufficient on their own.
- **TDMadDist is under-motivated and under-analyzed relative to its empirical behavior.**  
  The TD variant is presented as a second main algorithm, but the paper itself reports that it often underperforms MadDist (“While TDMadDist underperforms the MadDist and QRL algorithm...” in Section 7). The bootstrapped target
  \[
  \min(j-i,\;1+d_{\theta'}(s_{i+1},s_j))
  \]
  is plausible, but the paper does not analyze when it should help, when it should hurt, or why it fares poorly in some settings. Given that the next state \(s_{i+1}\) comes from an arbitrary dataset trajectory rather than a shortest path, the backup may inherit substantial bias toward behavior-policy paths. As written, this variant feels exploratory rather than well-justified.
- **The continuous-environment ground-truth protocol is underspecified in a way that affects interpretability of the reported correlations.**  
  In PointMaze and OGBench, the model consumes continuous 4D states \((x,y,\dot x,\dot y)\), while Appendix G defines ground-truth MAD by discretizing maze positions and running shortest path on the resulting graph. The paper does not clearly explain how evaluation pairs are mapped from continuous states to ground-truth distances, nor how velocity components are treated when assigning labels. Since two observations with the same position but different velocities may receive the same discretized “ground-truth MAD,” the correlation results are harder to interpret than they should be. This is not necessarily fatal—the approximation may still be reasonable—but the protocol needs to be specified much more carefully.
- **The paper does not isolate which parts of the composite loss are actually responsible for the gains.**  
  MadDist combines three ingredients: regression to trajectory upper bounds (\(L_o\)), a contrastive separation term (\(L_r\)), and local upper-bound enforcement (\(L_c\)). Appendix E ablates latent size, quasimetric choice, and dataset size, but not the loss components themselves. Without this, it is difficult to know whether performance gains come from the MAD-inspired structure, from the random-pair contrastive term, or from the constraint penalty. This weakens the paper’s mechanistic understanding.

### Minor
- **The empirical scope is narrower than the breadth of the paper’s claims.**  
  The environments are varied in asymmetry/stochasticity/noise, but they are all still navigation/maze-style tasks with low-dimensional state vectors. This is enough to support the paper’s claims within that regime, but not enough to strongly support broader implications about state representation learning more generally.
- **The paper claims `d_simple` is computationally efficient, but gives no quantitative efficiency comparison.**  
  The claim is plausible from the construction, and the paper does provide hardware details, but there is no training-time, throughput, or memory comparison against IQE/Wide Norm or the main baselines. Since efficiency is one of the advertised advantages of the new quasimetric, some empirical support would strengthen the contribution.
- **The role of the short constraint horizon \(H_c=6\) in long-horizon tasks is not well examined.**  
  The method relies on the quasimetric triangle inequality plus the learned embedding to propagate local constraints globally, but the paper does not directly study how well this works as horizon increases, despite evaluating on very long-horizon mazes. A horizon-stratified error analysis would be helpful.

### Trivial
- **The downstream planning evaluation is informative but limited.**  
  It validates heuristic usefulness via random-shooting MPC, but it is not evidence that the learned metric plugs effectively into a full RL training pipeline. This is not a flaw in the stated experiment, just a limit on how far one should generalize from it.

## Nice-to-Haves
- Add **scale-sensitive error metrics** such as MAE/RMSE or exact step-count error against ground-truth MAD, not only correlation-based measures.
- Include a **loss-component ablation** over \(L_o\), \(L_r\), and \(L_c\).
- Provide **error-by-horizon plots** to show whether long-range distances are actually estimated well.
- Clarify the **continuous-state evaluation protocol**, especially how continuous \((x,y,\dot x,\dot y)\) states are mapped to discretized ground-truth shortest-path labels.
- Add **runtime / memory comparisons** to substantiate the efficiency claim for `d_simple`.
- If space permits, include a **downstream RL use case** (e.g., shaping or goal-conditioned policy learning) to complement the planning heuristic evaluation.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Missing related work” criticisms.** Per instruction, I am not including complaints about omitted prior work. The paper already positions itself relative to several relevant families (QRL, Hilbert, successor features, time-contrastive methods, Laplacian methods), and I cannot verify external omissions.
- **Complaint that Hilbert is an unfair baseline because it is symmetric.** This is not a valid weakness here. The asymmetry mismatch actually favors the authors’ argument: showing large gains over a baseline that is structurally unable to capture asymmetric MAD is a legitimate and even strong comparison.
- **Reproducibility concern based on code only being released upon acceptance.** The paper provides substantial implementation detail in Appendix D; withholding code until acceptance is not, by itself, a substantive review weakness.
- **Requests for formal statistical significance testing / confidence intervals.** The paper reports multiple seeds and standard deviations/ranges, which is reasonably standard for this type of empirical RL/representation-learning evaluation. Formal hypothesis testing would be nice but is not essential here.
- **Criticism that NoisyGridWorld is invalid because observation noise breaks the Markov assumption.** The paper is explicit that it learns from observed state trajectories and that noise dimensions are nuisance features; Appendix G clearly defines the underlying latent coordinates and the intended ground-truth MAD ignoring observation noise. This is a legitimate robustness test, not a conceptual error.
- **Criticism that the planning setup should use a learned dynamics model instead of the true simulator.** Appendix H explicitly chooses the true simulator to isolate the quality of the learned metric as a heuristic. Demanding learned dynamics would change the question being evaluated.

## Novel Insights
The paper is strongest when interpreted not as exact MAD recovery, but as learning a quasimetric geometry that preserves directed reachability structure well enough to support planning. Under that interpretation, several pieces line up unusually well: the shortest-path-inspired constraints, the emphasis on asymmetry, the strong results in irreversible environments, and the usefulness of `d_simple` as a lightweight inductive bias. The main tension in the paper is therefore not whether the method learns something valuable—it clearly does—but whether the current objectives and metrics justify the much stronger claim that it learns the *minimum* action distance itself.

## Suggestions
- Reframe the main claim unless you can directly address the objective mismatch: either show why the current losses recover MAD despite using \(j-i\) targets, or soften the claim to learning a MAD-aligned quasimetric.
- Add **absolute-error evaluation** against ground truth MAD, alongside the current correlation metrics.
- Include a **component ablation** removing \(L_r\) and \(L_c\) in turn, to identify what truly drives performance.
- Provide a **theoretical or empirical analysis of TDMadDist**, especially why the TD target helps in some environments and hurts in others.
- Specify the **continuous-state label mapping** in detail for PointMaze/OGBench and discuss the effect of velocity/state discretization.
- Add a **horizon-wise error analysis** and possibly vary \(H_c\), especially on giant mazes.
- Quantify the **efficiency advantage** of `d_simple` with wall-clock time and memory usage relative to IQE/Wide Norm.