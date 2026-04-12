# ICLR Benchmark Results

Date: 2026-04-12 00:38
Critic/Merger: qwen/qwen3.6-plus (OpenRouter)
Neutral: qwen/qwen3.5-flash-02-23, Related Work: qwen/qwen3.5-flash-02-23:online (OpenRouter)

## Gk7gLAtVDO

- GT: Accept (Oral) (avg 7.5)
- Predicted: N/A (6.4/10)
- Match: N/A

### Final Review

## Summary
This paper proposes TRACE, a detection method for implicit reward hacking that does not rely on reading the semantic content of chain-of-thought. Instead, it truncates the model’s reasoning at multiple points, forces an answer, measures expected proxy reward at each cutoff, and uses the AUC of reward-vs-length as a signal that the model knew how to get reward “too early.” The paper shows strong gains over a CoT-monitor baseline on synthetic but carefully controlled math and coding loopholes, and also demonstrates a promising loophole-discovery use case via clustering on TRACE scores.

## Strengths
- **The paper identifies a genuinely different oversight signal than CoT-text monitoring.** The core move is not “better prompt engineering for monitors,” but shifting from semantic inspection of CoT to a behavioral measure of when reward becomes attainable under truncation. This is a real conceptual contribution, especially for the stated threat model of unfaithful or post-hoc-rationalized CoT. Section 4 makes this distinction clearly: TRACE generalizes early-answering from “same answer as full CoT” to expected proxy reward \(E[\hat R]\), which is exactly the right abstraction when hacked and intended answers may differ.
- **The evaluation design includes unusually strong ground-truth labeling for this kind of safety problem.** The paper does not merely assume that responses from a “hacking-trained” model are hacked; it uses counterfactual tests to label instances based on dependence on the loophole. In Section 3.2 / 4.1, hacked samples are identified by removing/replacing the hint or restoring the correct verifier. This is much stronger than many alignment papers that rely on heuristic labels for deceptive behavior.
- **TRACE appears especially well matched to cases where CoT looks superficially plausible.** The math reward-model loophole is a good example: the model can preserve apparently normal reasoning while only perturbing the final answer by a minus sign. The paper’s examples and Figure 20 support the claim that CoT-monitoring struggles there while TRACE still fires because rewarded outputs appear early under truncation.
- **The paper goes beyond binary detection and shows a plausible debugging workflow.** Section 5’s clustering-based loophole discovery is not just an add-on metric table; it suggests TRACE could help localize problematic data or verifier artifacts in training environments. That broader “oversight as environment debugging” angle is one of the more interesting aspects of the work.
- **The partial-loophole training setup is a meaningful robustness test.** Section 4.2 Setup 2 is stronger than a fully systematic loophole setting because the model only learns to hack on a subset of data. Showing TRACE still separates hacking from non-hacking responses there makes the contribution more credible than if it only worked in the easiest all-loophole regime.

## Weaknesses

###: Fatal
- None.

### Major:
- **The paper’s “scalable oversight” framing is stronger than what the current evidence supports, because TRACE is inference-expensive and the cost is not quantified.** This is a real issue. TRACE requires repeated truncation evaluations and, in math, multiple samples per truncation (“In math, we sample 5 answers with a temperature of 0.7…”). That makes this much heavier than a single-pass monitor. The paper’s discussion of scalability in Section 7 is about *not needing a stronger external monitor*, which is a valid sense of scalability, but it does not address computational scalability or deployment cost. Since “scalable oversight” is a central framing claim, the paper should report the number of truncation points, total forward passes, and a cost/performance tradeoff.
- **The experiments are still confined to synthetic loopholes that, while useful for controlled study, leave open how well TRACE transfers to more realistic reward-hacking phenomena.** The paper itself acknowledges this in Section 7 (“Our simulated loopholes are necessarily simplified…”). That caveat is important. The in-context math loophole is an answer hidden as problem ID; the reward-model loopholes include accepting negative answers or rewarding presence of `else`. These are valid proof-of-concept environments, but they are still relatively stylized. The current empirical support is therefore strongest for the claim “TRACE works in controlled loophole-learning settings,” and weaker for broader real-world claims.
- **The central interpretation of TRACE score as “reasoning effort” is plausible but not fully disentangled from alternative causes of early rewarded answers.** The method assumes that high AUC reflects low effort shortcutting, but the paper itself notes confounders such as overthinking in Section 7. There are additional nearby concerns: legitimate shortcut solutions, lucky guessing, or format-induced answering behavior under forced truncation could also raise early reward. The current results strongly support TRACE as a useful detector in the tested settings, but they do not yet fully validate the stronger mechanistic interpretation that the score cleanly measures hidden internal effort.
- **The thresholding strategy is underdeveloped for deployment-style use.** In Section 4, the paper classifies a response as hacking when its TRACE score exceeds the average score of the initial policy, and Section 7 explicitly notes failure modes if the initial policy already hacks some in-context samples or if overthinking inflates scores. This is a substantive limitation rather than a mere detail: the empirical results depend on a thresholding choice that may drift under RL-induced distribution shift, verbosity changes, or changes in task mix. The paper acknowledges possible mitigations but does not test them.

### Minor
- **The comparison against CoT monitoring is meaningful but narrower than the headline framing suggests.** The paper compares TRACE primarily to one family of baselines: rubric-prompted CoT monitors with different model sizes. That is enough to support the claim that TRACE outperforms these CoT monitors in the tested settings, but not enough to establish broader superiority over monitoring/detection methods in general.
- **Applicability is currently limited to settings with verifiable or cheaply computable proxy rewards at each truncation point.** This is inherent in the method, not a flaw in execution, but it meaningfully narrows significance. TRACE is immediately natural for math and code with explicit verifiers; its role for open-ended generation, dialogue, or tasks without clear intermediate reward remains unclear.
- **The paper would benefit from a more careful treatment of false positives from legitimate early convergence.** If a problem admits a short valid solution or pattern-recognition shortcut that is intended rather than exploitative, high early reward could look TRACE-positive without reward hacking. This issue is adjacent to the authors’ overthinking discussion but deserves more direct empirical analysis.
- **Some of the strongest single-model detection results are primarily shown on 3B models.** The paper does include larger-model comparisons in the hacking-vs-non-hacking setting and monitor-scaling appendix, so it is not fair to say evaluation is only on small models. Still, the most deployment-relevant “single model” detection story would be stronger with more extensive results on larger hacked policies.

### Trivial
- **The math/code TRACE estimation procedures differ (sampling in math, deterministic pass-rate estimation in code), and variance/stability analysis is limited.** This does not invalidate the results, but a small analysis of sensitivity to sampling temperature / number of samples would make the AUC signal easier to trust.

## Nice-to-Haves
- Add a direct cost analysis: truncation count, samples per truncation, forward passes, wall-clock latency, and detection quality versus compute.
- Evaluate threshold robustness more systematically, including dynamic or percentile-based thresholds, as already suggested in the discussion.
- Add ablations for the truncation-forcing mechanism (e.g., alternative answer-forcing tokens or formats) to verify that the signal is not overly tied to a specific formatting intervention.
- Include experiments or case studies on legitimate shortcut solutions / easy problems to characterize false positives.
- Strengthen the larger-model story, especially for the single-model detection setting.
- Expand the realism of loopholes where feasible, or at least include a sharper statement delimiting what kinds of real-world attacks these synthetic setups are meant to approximate.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The baseline comparison is invalid because a simple regex or specially tuned heuristic could catch the hacks.”**  
  Removed because the paper’s stated comparison is against CoT monitoring, not against any imaginable task-specific detector. In fact, for the coding RM loophole the asymmetry arguably favors the baseline, since the exploit (`else`) is explicit in the output and should be easier for a monitor than for TRACE. So this does not undermine the reported comparison; at most it limits the breadth of the claim.
- **“The hacks are not implicit because the final outputs are anomalous (negative numbers / `else`).”**  
  Removed as an overstatement. The paper defines implicit reward hacking as cases where the CoT is unfaithful and does not reveal the exploit, not cases where the final answer looks perfectly ordinary. On that definition, the math RM examples do fit the paper’s threat model: the reasoning can appear benign while the reward is obtained by a hidden shortcut.
- **“The paper should evaluate latent reasoning models because it claims CoT monitoring is inapplicable there.”**  
  Removed as scope creep. The paper mentions latent-reasoning models as motivation for not depending on monitor access to faithful CoT, but TRACE itself still requires some externally truncatable reasoning trace in the current form. Lack of latent-model experiments is not a fair core criticism of the presented contribution.
- **“Limited model diversity: only Qwen variants.”**  
  Weakened and effectively removed as a main criticism. The paper does test multiple Qwen scales and uses both math and code settings. Broader architecture diversity would help, but this is not a decisive weakness relative to the paper’s central claims.
- **Pure reproducibility complaints about missing implementation details.**  
  Removed: the appendix includes substantial training/configuration details, and no core claim appears to hinge on omitted trivial settings.
- **Formatting/style concerns from the extracted PDF artifacts.**  
  Removed entirely.

## Novel Insights
The most interesting synthesis across the reviews is that TRACE’s real contribution may be less “a universal reward-hacking detector” and more “a behavioral audit primitive for measuring reward attainability as a function of revealed reasoning.” That framing helps reconcile both its strengths and limitations: it is compelling precisely because it avoids trusting CoT semantics, but its usefulness depends on whether early rewarded answers truly distinguish exploitative shortcuts from benign early convergence. In other words, the method already looks valuable as an auditing/debugging tool for training environments and verifier design, even if the stronger interpretation as a direct measure of latent reasoning effort still needs more validation.

## Suggestions
- Report exact TRACE evaluation cost and compare it directly to CoT monitoring cost; if the paper wants to retain “scalable oversight,” define clearly in what sense it is scalable.
- Add a threshold-sensitivity study using alternative baselines (initial policy, clean validation set, percentile thresholds, mixed-checkpoint thresholds).
- Add an ablation on the truncation intervention itself, testing whether alternative answer-forcing formats preserve the signal.
- Evaluate false positives on examples with legitimate short solutions or intentionally injected overthinking, to separate “shortcutting” from “hacking.”
- Tighten the main claims to match the evidence: TRACE is strongly supported as a detector in controlled verifiable-reward settings with unfaithful CoT, while broader real-world generalization remains promising but not yet established.

---

## QBGVlffCzf

- GT: Reject (avg 2.0)
- Predicted: N/A (2.4/10)
- Match: N/A

### Final Review

## Summary
The paper formulates a large-scale heterogeneous variant of Colonel Blotto as a Dec-POMDP and proposes a MARL framework with two named components: Group-Mix for type-aware value decomposition and H-PPO for hierarchical curriculum-based policy optimization. The intended contribution is to make Blotto-like team allocation problems tractable at scales up to 1,000 agents and 20 battlefields, with some accompanying theoretical justification for CTDE and type sharing.

## Strengths
- The paper targets a genuinely difficult and underexplored setting: heterogeneous, team-based Blotto with very large agent counts. The explicit LHBG formalization extends classical Blotto from a single centralized allocator to opposing teams of many typed agents, with definitions for types, team budgets, battlefield capabilities, local observations, and joint actions.
- The paper identifies two real technical pain points for this setting—heterogeneous credit assignment and large-scale training stability—and proposes architecture choices aligned with them: type-aware aggregation in Group-Mix and curriculum scaling in H-PPO. Even though the integration is not fully specified, the problem decomposition itself is sensible.
- The scalability target is ambitious relative to typical MARL toy settings. The experimental setup does at least instantiate scenarios from 50 agents / 5 battlefields up to 1000 / 20, which is a meaningful stress-test regime for this problem class.
- One useful paper-specific idea is the use of type structure as an organizing prior throughout the framework: same-type parameter sharing, type-group feature aggregation, and type-conditioned policy/value components are consistently used in the modeling and algorithm sections rather than appearing as an isolated trick.

## Weaknesses

###: Fatal
- **The core algorithm is not specified in a coherent, executable way.** The paper claims a “dual-path” framework where Group-Mix and H-PPO “collaborate” and form a “complete closed loop,” but it never gives the actual training objective or update procedure connecting them. Group-Mix is presented as a value decomposition method producing \(Q_{\text{tot}}\), while H-PPO is presented with the standard PPO clipped objective using an advantage \(\hat A_t\). The paper never defines how \(\hat A_t\) is computed from Group-Mix, whether Group-Mix is the critic, whether there are separate critics, whether training is on-policy or off-policy, or how gradients flow between components. This is not a small omission: the main claimed method is underspecified at the level needed to judge technical soundness or reproduce the contribution.
- **The theoretical claims are not reliable as stated.** Theorem 3 is mathematically flawed in the paper’s own proof. It states that type sharing compresses strategy space from \(O(|A|^K)\) to \(O(|A|^{|M|})\), but the proof explicitly says “\(\sum_{m\in M}|A| = |A|^{|M|}\),” which is incorrect. More importantly, what the paper really argues is parameter sharing / policy tying across agent types, which reduces the number of learned policy functions, not the underlying joint action or game-theoretic strategy space of the environment. So the theorem overclaims both mathematically and conceptually.
- **The empirical evidence is insufficient to support the main performance claims.** The abstract and introduction claim advantages in “solution quality, convergence speed, and scalability over existing methods,” but the results table reports only the proposed method across different scales. There are no comparisons to standard MARL baselines (e.g., value decomposition or PPO-family baselines), no comparison to simpler variants of the proposed method, and no comparison to classical Blotto solvers in any regime where those are feasible. As a result, claims of superiority or even specific benefit from Group-Mix/H-PPO are unsupported.

### Major:
- **The opponent model / evaluation protocol is not specified clearly enough to interpret the reported rewards and win rates.** In Definition 9, the enemy capability update is written as \(T^{[-p]}_{n,t+1} = \Psi_{\text{enemy}}(s_t)\), but the paper never concretely describes what the opposing team does in experiments: fixed heuristic, random policy, mirror self-play, learned adversary, or something else. Since this is a competitive game, the meaning of reward, net win rate, and “solving” depends critically on the opponent.
- **The paper’s “stable learning” claim is contradicted by its own reported statistics.** Table 4 reports very large variance at larger scales, especially \(4.71 \pm 8.59\) average reward in the 1000-agent setting, with a minimum of \(-0.20\). Without learning curves, seed counts, or any variance analysis, this does not substantiate the abstract’s repeated stability claims.
- **There is a direct mismatch between the paper’s motivation about synergy and the actual formal reward.** Section 3.3 states that the reward design *can introduce synergistic effects*, e.g., extra rewards when attack and reconnaissance agents are deployed together. But Definition 10 specifies reward purely as battlefield win/loss indicators, and no synergy term appears in the formalism or experiments. As written, the paper motivates richer heterogeneous coordination than it actually models.
- **Several headline claims are stronger than the evidence shown.** For example, the paper says it “solves” ultra-large-scale heterogeneous Blotto and that such settings were “previously considered computationally intractable,” but it provides neither equilibrium-quality analysis nor computational efficiency evidence (runtime, memory, sample complexity, or scaling cost) to substantiate that stronger systems claim.
- **Theoretical support for CTDE/IGM is largely generic and not specialized to this setting.** Lemma 1 and the corollary mainly restate standard monotonic value-decomposition logic under IGM/additive assumptions. The paper does not convincingly show that these assumptions are especially justified for the thresholded Blotto reward structure it uses, nor that this yields a novel theoretical result for LHBG beyond importing standard MARL machinery.

### Minor
- **The curriculum description is internally inconsistent.** Section 5.3 defines four curriculum stages \(\{[50,5],[200,10],[500,15],[1000,20]\}\), while Algorithm 2 loops over stages \(i=1\) to \(3\). This inconsistency weakens confidence in the implementation description.
- **Some reported metrics are not well explained.** “Mean Loss (absolute value)” is reported in Table 4 but not defined in a meaningful evaluation context. Similarly, the interpretation of “Mean Net Win Rate” is not explained well enough to tell whether the observed values indicate strong or weak play.
- **The experiments do not isolate the claimed contributions.** Since the proposal has two core components—type-aware Group-Mix and H-PPO curriculum—there should be ablations removing or replacing each one. Without that, it is impossible to tell whether either component is necessary.

### Trivial
- None.

## Nice-to-Haves
- Add learning curves across curriculum stages, not just summary statistics, to show whether training genuinely stabilizes as scale increases.
- Include qualitative strategy visualizations (e.g., battlefield allocation heatmaps by type) to support the claim that the method learns meaningful heterogeneous coordination patterns rather than merely achieving modest aggregate wins.
- Provide compute scaling information such as wall-clock time, sample count, and memory footprint for the 50→1000 agent progression.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Missing hyperparameters / implementation details” as a core reproducibility criticism.** The paper is indeed light on details, but complaints about omitted learning rates, entropy coefficients, batch sizes, GAE parameters, etc. are not by themselves core scientific weaknesses under the stated review policy. The real issue is not missing knobs; it is that the main optimization linkage between Group-Mix and H-PPO is never defined.
- **Criticism that the paper should evaluate on other MARL benchmarks such as SMAC or MPE.** That is outside the paper’s stated scope, which is Blotto games. It could strengthen confidence in generality, but its absence is not a central flaw for a domain-specific paper.
- **The positive review’s claim that the type-sharing compression theorem is “sound.”** This is factually incorrect after checking the theorem and proof in the paper. The theorem contains an arithmetic error and overstates what parameter sharing buys.
- **Generic non-stationarity criticism.** While multi-agent non-stationarity is always relevant, the stronger and paper-specific issue here is the undefined algorithmic integration and unclear training protocol, not a generic failure to discuss non-stationarity.

## Novel Insights
The most revealing synthesis across the paper and reviews is that the strongest idea here is not the current theorem package or the final empirical evidence, but the *structural use of type priors* to make very large Blotto-like team allocation problems amenable to MARL. However, the submission currently conflates three distinct notions—parameter sharing, value decomposition, and actual compression of the strategic game—without keeping them conceptually separate. If the authors recast the contribution more modestly as a typed MARL approximation framework for large heterogeneous Blotto, and then empirically validate that approximation against proper baselines and ablations, the paper would become much more credible.

## Suggestions
- Specify the full training algorithm end-to-end: what networks exist, what losses are optimized, how the critic is defined, how \(\hat A_t\) is computed, whether Group-Mix serves as the critic for PPO, and what data collection / replay regime is used.
- Tone down and repair the theory section. In particular, fix or remove Theorem 3, and distinguish clearly between reducing the number of learnable policy parameters and reducing the actual joint strategy space of the game.
- Add direct baselines and ablations. At minimum: a PPO-family baseline without the proposed typed mixing/curriculum ideas, a standard value-decomposition baseline, and ablations removing type-aware mixing and curriculum progression.
- Clarify the opponent setup used in training and evaluation. State explicitly whether experiments use self-play, fixed scripted opponents, random opponents, or another protocol.
- Reconcile Section 3.3 with Definition 10: either include a genuine heterogeneous synergy reward/objective, or remove the claims about synergy-based coordination.
- Support the stability claim with multi-seed results, learning curves, and better reporting of variance at large scales.
- Add computational scaling evidence if the paper wants to claim practical tractability at unprecedented scales.

---

## n1AvXiU2lu

- GT: Accept (Poster) (avg 6.5)
- Predicted: N/A (7.2/10)
- Match: N/A

### Final Review

## Summary
This paper introduces **real-time reasoning** for LLM agents: settings where the environment continues evolving while the agent is still computing, rather than pausing between decisions. It contributes both **Real-Time Reasoning Gym**—with controllable cognitive load and time pressure across Freeway, Snake, and Overcooked—and **AgileThinker**, a dual-thread design in which a reactive thread acts under tight deadlines while a planning thread reasons continuously and streams partial outputs. Empirically, the paper shows a clear failure mode of single-paradigm agents under either high cognitive load or high time pressure, and AgileThinker performs substantially better across the proposed benchmark.

## Strengths
- **The paper identifies and operationalizes a genuinely important gap in current LLM-agent evaluation:** namely, the unrealistic assumption that the world pauses while the model reasons. The formulation in §2 is concrete rather than rhetorical: the environment steps at a fixed rate regardless of whether the agent has finished, and a default action is applied if no action arrives in time.
- **The benchmark is structured around two orthogonal stressors—cognitive load and time pressure—rather than only reporting aggregate task success.** This decomposition is useful and supported by the design in Table 1 and the experiments in §4, which show distinct failure regimes for reactive vs. planning agents.
- **The paper’s central empirical finding is strong and consistent within its own benchmark:** reactive agents degrade sharply with cognitive load, planning agents collapse under tighter time pressure, and AgileThinker is more robust to both. This pattern appears not only in averages (Fig. 1/5) but also in per-game tables (Tables 6–7).
- **AgileThinker’s key design choice is more specific than a generic “hybrid agent”:** the reactive thread can consult *partial* planning traces during the same environment step, rather than only a final completed plan or an independent planner. That is a meaningful architectural idea, not just a rebranding of sequential fast/slow systems.
- **The paper goes beyond a single fixed setting.** It studies internal budget allocation for the reactive thread (§5, Fig. 7), includes a dynamic budget heuristic (App. E), and reports limited additional results with another model family / mode configuration (App. C.3, Tables 8–10).
- **The appendix contains unusually helpful task-level analysis for the code-as-policy baseline** (App. C.4), showing why it works in Freeway but struggles in Snake and Overcooked. That analysis makes the benchmark failure modes more interpretable than a pure leaderboard-style paper would.

## Weaknesses

###: Fatal
None.

### Major:
- **The main claim that AgileThinker’s *architecture* is responsible for the gains is not fully isolated from added inference resources.**  
  The core comparison is between AgileThinker, which runs **two threads / two model processes**, and baselines that each use a single paradigm. The paper does include one useful resource-constrained comparison in App. C.5 (“parallel threads” vs. “concurrent threads”), showing the method still helps when throughput is limited, but this does **not** answer the more central question of whether the gains come from (i) dual specialization and trace-sharing, or (ii) simply spending more total compute on the problem. A compute-matched control is especially important because the paper’s headline contribution is architectural rather than purely “more compute helps.” As written, the evidence strongly supports “a hybrid system works well,” but more weakly supports “this particular coordination mechanism is the reason.”
- **The paper does not adequately ablate the proposed coordination mechanism—access to partial planning traces.**  
  This is the most distinctive aspect of AgileThinker, yet the paper never cleanly compares:
  1. reactive thread with no access to planner outputs,  
  2. reactive thread with only final completed planner outputs, and  
  3. reactive thread with streaming partial traces.  
  Without such an ablation, it is difficult to determine whether partial-trace sharing is essential, or whether most of the benefit comes simply from running a planner in the background and letting a reactive policy act on current observations. The text claims this distinguishes AgileThinker from prior dual-system setups, but the empirical support for that specific distinction is incomplete.
- **The “real-world / practical deployment” framing is somewhat overstated relative to the actual demonstrated timescales.**  
  The paper’s motivating examples emphasize latency-critical real-time behavior (e.g., driving), but the wall-clock validation in §6 is conducted at **6 minutes per step** for the 8k-token regime. That does not invalidate the benchmark as a study of asynchronous reasoning under delayed computation, and the paper is correct that the environment evolves independently of the agent. However, it does limit how strongly one can interpret the results as evidence for genuinely low-latency real-time control. The contribution is better supported as a benchmark for **non-pausing dynamic environments under bounded inference budgets** than as a direct demonstration of readiness for fast real-world control loops.
- **The token-as-time abstraction is useful but oversold as “hardware-agnostic” and practically validated.**  
  The paper is careful to motivate token count as a reproducible proxy, and Fig. 10 / §6 do show a strong linear fit between generated tokens and wall-clock time for their setup. But the fitted model is  
  \(T = \alpha N + \beta\) with \(\beta = 334.55\) s, a very large intercept. This means wall time in their deployment is not explained by token count alone; a substantial fixed overhead exists. Thus, the experiments validate token count as a **controlled simulation variable** for comparing agent designs under a fixed inference stack, but they do not fully justify stronger claims about hardware-independent or deployment-independent temporal realism.

### Minor
- **The benchmark environments are intentionally stylized, and this narrows external validity.**  
  Freeway, Snake, and Overcooked capture hazards, transient opportunities, and partner coordination, which is a sensible spread. Still, these remain discrete, relatively abstract environments. The paper’s broader claims about real-world agents would be more convincing with at least one richer environment featuring noisier observations, denser action spaces, or more complex temporal dependencies.
- **AgileThinker appears sensitive to the internal budget split \(N_{TR}\), and the main paper’s guidance for setting it is partly empirical.**  
  Figure 7 is informative and App. E provides a dynamic adjustment heuristic, but the practical burden of tuning remains nontrivial. The paper argues that rough upper bounds suffice, which is plausible, yet a stronger characterization of robustness across tasks and phases of an episode would help.
- **The implementation of partial-trace consumption is not described in enough detail in the main paper.**  
  The high-level protocol is clear—planning runs continuously and reactive acts in the final \(T_R\) slice of each step—but details of how partial reasoning is packaged into the reactive prompt, how incomplete outputs are handled, and how formatting/valid-action reliability is preserved are not clearly explained in the excerpted main text. Since this mechanism is central, it deserves more explicit specification.
- **The Overcooked partner policy may not be rich enough to fully stress long-horizon coordination.**  
  Appendix A states that the second player is controlled by a manually written script that “randomly chooses one policy to follow: deliver an onion into an arbitrary pot or a kitchen counter.” This is sufficient to induce non-stationarity, but it is still a fairly restricted partner model. That may limit conclusions about coordination-heavy real-time reasoning.

### Trivial
- **The wall-clock section would benefit from more careful wording about what exactly is being validated.**  
  The current text sometimes reads as if the linear fit fully establishes practical relevance, when it more modestly supports the simulation as a comparative proxy in this particular deployment setting.

## Nice-to-Haves
- Add **compute-normalized comparisons** (e.g., score vs. total tokens / total API cost / total wall-clock budget) and a Pareto frontier plot.
- Add a **mechanism ablation** that isolates the value of streaming partial traces versus no trace-sharing or final-plan-only sharing.
- Include one **failure case where planning traces actively mislead** the reactive thread, to clarify the architecture’s boundaries.
- Test one more environment with richer dynamics or multimodal / noisier observations.
- Report a more explicit **safety analysis of the default action** choice, since “keep moving in the same direction” or “stay idle” can materially affect outcomes.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The planning baseline is unfair because asymmetry favors AgileThinker.”**  
  I do not keep the claim that the comparison is unfair simply because single-paradigm methods are weaker under the imposed setup. The paper’s goal is precisely to show that single-paradigm designs fail in dynamic environments. Moreover, some asymmetries actually favor baselines in parts of the study (e.g., single-thread methods are cheaper). The valid issue is not “unfairness” per se, but lack of compute-matched and mechanism-level controls.
- **“The method is not reproducible because code is only released after publication / because prompt details or hyperparameters are incomplete.”**  
  Removed under the review policy. The appendix already provides substantial prompt and experimental detail, and release-status concerns are not a valid criticism here.
- **“The cited tools/models/benchmarks may not exist / may not be available.”**  
  Removed as factually out of scope.
- **“Missing related work such as Tree-of-Thought / Graph-of-Thought.”**  
  Removed because external omissions cannot be verified here, and the stronger issue is not bibliography coverage but lack of direct ablations against simpler dual-system alternatives.
- **“The games are too simple, therefore the paper’s results are invalid.”**  
  Overstated. The limited realism is a fair external-validity concern, which I keep in weakened form as a minor weakness, but it does not invalidate the paper’s internal claims about the proposed benchmark.

## Novel Insights
The most important synthesis across the reviews is that the paper is stronger as a **benchmark-and-systems paper about asynchronous reasoning under non-pausing environments** than as a definitive demonstration of “real-time” control in the everyday robotics/driving sense. Within that narrower but still important framing, the paper provides convincing evidence that single-paradigm agent designs fail for complementary reasons. The real unresolved scientific question is not whether hybridization helps—it clearly does in this benchmark—but whether **streaming partial reasoning traces** are the essential ingredient, or whether a simpler background-planning + reactive-control scheme would achieve most of the benefit. That distinction should become the centerpiece of revision.

## Suggestions
- Add a **compute-matched baseline suite**: same total token budget / wall-clock budget as AgileThinker, including stronger single-model controls and simple two-thread controls.
- Add a **clean ablation of trace-sharing**: no planner input vs. final-plan-only vs. partial streaming traces.
- Reframe claims from broad “real-world real-time deployment” language to the more defensible claim of **dynamic non-pausing environments with bounded inference**.
- In §6, discuss the implications of the large intercept in the wall-clock fit and narrow the claim from “hardware-agnostic practical validation” to “deployment-specific calibration supporting the simulation proxy.”
- Expand the method description so that the **reactive prompt construction from partial planner outputs** is fully specified.
- If space permits, include a **score-vs-compute Pareto plot** and one richer environment or more demanding partner-coordination setting.

---

## FA5gun3gmm

- GT: Withdrawn (treated as Reject) (avg 1.3)
- Predicted: N/A (5.3/10)
- Match: N/A

### Final Review

## Summary
This paper proposes Omni TM-AE, a Tsetlin Machine–based embedding method that derives word embeddings from the full TM state space rather than only literals whose automata states exceed the usual clause-inclusion threshold. The main claimed advances are: (i) reusability from a single training phase, unlike prior TM embedding approaches requiring retraining or multi-phase procedures, and (ii) improved embedding quality by incorporating information from “excluded” literals. Empirically, the paper shows competitive results on semantic similarity, sentiment classification, and clustering, with especially plausible evidence that the method can be a reusable interpretable alternative to standard static embeddings.

## Strengths
- **The paper identifies and operationalizes a genuinely specific idea: using sub-threshold TM states as embedding signal rather than discarding them.** This is more than a generic TM-for-NLP application. Section 3.4 and Algorithm 1 define a concrete embedding extraction rule from the state matrix, and Figure 3 makes the intuition tangible by showing that many semantically suggestive literals remain below threshold \(N\) even though they are ignored by prior clause-based use.
- **The single-phase reuse claim is specific and practically meaningful within the TM literature described by the paper.** The paper clearly contrasts prior approaches: Bhattarai et al. requiring full-vector retraining for altered token sets, and Kadhim et al. requiring a second phase to make token relationships directly usable. Omni TM-AE’s extraction from already-trained state matrices is a real conceptual simplification, not just a minor implementation tweak.
- **The method preserves a form of mechanistic traceability that most embedding papers do not offer.** The embedding coordinates are explicitly constructed from clause/literal states, and Section 6.2 explains how one can trace shared contributions across target words back to clause-level training dynamics. While the paper overstates interpretability in places, this is still a substantive advantage over opaque dense embedding models.
- **Empirical results support competitiveness, even if not dominance.** In similarity (Table 1), Omni TM-AE is close to the best average Spearman score and has the best average Kendall score; in classification (Table 2), it is essentially tied with the strongest static baseline and comparable to ELMo/BERT under the authors’ setup; in clustering (Table 3), it has the best average ARI, though not the best NMI. This does not prove broad superiority, but it does support the claim that the approach is viable.

## Weaknesses

### Major:
- **The central empirical validation of the paper’s core novelty is incomplete because there is no ablation isolating the contribution of excluded literals.** The main contribution is not merely using TM states, but specifically using *all* literals, including those below threshold \(N\). Yet the paper does not compare: included literals only vs. full-state Omni extraction, nor vary the thresholding or weighting of below-\(N\) states. As a result, the paper shows that the proposed method works, but not cleanly that the omitted-state information is what drives the gains.
- **Some of the headline comparative claims are stronger than what the evidence supports.** The paper repeatedly suggests it “often surpasses” mainstream embedding models or performs “on par with or better than black-box models.” The actual results are more mixed:
  - In Table 1, Omni TM-AE is competitive and strong, but FastText has the best average Spearman.
  - In Table 2, Omni TM-AE is slightly below Word2Vec on average.
  - In Table 3, Omni TM-AE has the best average ARI but trails Word2Vec/FastText on average NMI.
  A more accurate framing is “competitive overall with some wins,” not clear superiority.
- **The evaluation of predecessor TM baselines is incomplete at exactly the point where the paper needs it most.** The paper’s main positioning is as an improvement over prior TM embedding methods, but TM-AE is only reported on RG65, and the key predecessor from Kadhim et al. is excluded from experiments on practicality grounds. Even if those practical limitations are part of the point, the lack of a controlled restricted-scale comparison leaves the magnitude and source of improvement under-demonstrated.
- **The classification setup is non-standard enough that conclusions should be stated more cautiously.** Section 4.3 evaluates embeddings through a perturbation procedure that replaces 5% of tokens based on embedding neighborhoods, with asymmetric rules for positive and negative documents. This is an interesting stress test, but it is not a standard sentiment benchmark protocol, so the results should not be read as general downstream superiority. The asymmetry in the replacement policy also makes the table somewhat harder to interpret.
- **Interpretability is asserted more than it is demonstrated.** The paper gives intuitions, examples, and traceability arguments, but there is no systematic interpretability evaluation: no user-oriented analysis, no concept-level case study with success/failure criteria, no quantitative proxy such as clause sparsity/fidelity, and no demonstration that practitioners can reliably use these embeddings for diagnosis beyond anecdotal examples.

### Minor
- **The embedding dimensionality is very large.** Section 3.4 defines embeddings of size equal to the vocabulary \(d\), and the experiments use vocabularies up to 40,000. This is a meaningful practical trade-off versus 100-dimensional Word2Vec/GloVe baselines, especially for storage and downstream efficiency. The paper emphasizes scalability, but scalability here appears to mean training/reuse properties, not compact representation size; this distinction should be made explicit.
- **Important experimental details around document-level embedding construction are underspecified.** Section 4.4 says documents are represented by aggregating the word embeddings they contain, but the exact aggregation rule is not clearly described in the main text. Since clustering and likely classification performance can depend materially on mean vs. weighted mean vs. other pooling, this should be explicit.
- **The paper notes that original literal states compress into a narrow range at large vocabulary scale (Appendix/Figure 5), but does not quantitatively analyze whether these low-state regions carry robust semantic signal rather than weak noise.** Since the core method relies precisely on extracting information from these non-selected states, this deserves deeper analysis.
- **The BERT comparison is not entirely clear.** Section 4.3 describes “fine-tuning” settings for BERT, yet Table 2 presents BERT as an “Embedding Source.” It is not fully clear whether the reported numbers correspond to frozen embeddings, pooled representations from a fine-tuned model, or some hybrid pipeline. This ambiguity weakens the force of the comparison.

### Trivial
- **Algorithm 1 contains what appears to be a typographical/formula error on line 14** (`ei <- - vti`) relative to the text definition of \(e_i = v_i / t_i\). This is not a substantive flaw, but it should be corrected.

## Nice-to-Haves
- Add a direct ablation comparing:
  - only included literals (\(n > N\)),
  - all literals,
  - all literals but without negations,
  - and possibly weighted variants of below-threshold states.
- Provide a small controlled benchmark where Omni TM-AE, TM-AE, and the multi-phase predecessor are all run on the same restricted vocabulary/data regime.
- Quantify training cost and memory footprint more systematically, separating training cost from post-training embedding extraction time.
- Include at least one failure-case interpretability analysis showing how a poor similarity judgment, clustering error, or classification error can be traced back to specific clauses/literals.
- Clarify document embedding aggregation and the exact BERT pipeline in the main text.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that 32 clauses are mathematically incapable of representing a 40,000-token vocabulary, invalidating the method.** This overstates what is established from the paper. The paper uses a Coalesced TM with shared clauses and class-specific weights; from the paper alone one cannot conclude a fatal representational impossibility or “collapse.” The empirical results also do not support such a catastrophic-failure reading.
- **Claim that the evaluation is invalid because the vectors are “unnormalized integer counts” incompatible with similarity/clustering metrics.** The paper does not specify a mathematically invalid metric pipeline. Spearman/Kendall correlations evaluate ranked similarities, and nothing in the paper proves that their use is invalid for these vectors. It is fair to ask for more clarity on similarity computation or normalization, but not to declare the results invalid.
- **Criticism that negative clauses must be included for embeddings and that omitting them fundamentally biases the method.** The paper explicitly defines embeddings from positive-weight clauses as part of the proposed method. A reviewer may prefer an ablation including negative clauses, but there is no basis here to say the current choice is incorrect.
- **Concern that using a DGX H100 with large RAM contradicts efficiency/edge suitability.** The paper mentions TM hardware-friendliness in related work, not as a claim that these experiments were performed under edge constraints. Reporting the actual server used for experiments is not a substantive weakness by itself.
- **Reproducibility complaints about omitted seeds/splits/preprocessing minutiae.** Some details could be clearer, but these are not central enough to elevate given the paper’s current main issues.
- **Request for additional modern baselines not cited in the paper.** Omitted per instruction not to criticize missing related work/baselines that cannot be externally verified here.

## Novel Insights
The most important synthesis is that the paper is stronger as a **representation-extraction paper** than as a pure **state-of-the-art empirical paper**. Its real contribution is not that it convincingly beats Word2Vec/FastText/BERT across standard benchmarks—it does not—but that it turns a normally discarded part of TM training dynamics into a reusable embedding space with some preserved traceability and nontrivial competitive performance. The missing piece is causal validation: the paper argues that sub-threshold states contain useful semantic information, and the qualitative figures support that intuition, but the experimental section never cleanly isolates that mechanism. If the authors add this causal evidence, the paper’s central idea would become substantially more convincing.

## Suggestions
- Add a focused ablation where the only difference is whether below-threshold literals contribute to the embedding; this is the single most important missing experiment.
- Reframe the empirical claims from “surpasses mainstream models” to “competitive with standard static embeddings, with wins on some metrics and tasks.”
- Add a restricted-scale apples-to-apples comparison against prior TM embedding methods, even on one smaller dataset, to substantiate the claimed advance over TM-specific predecessors.
- Clarify the downstream pipelines: exact similarity function, document embedding aggregation, and whether BERT is used as frozen embeddings or as a fine-tuned end model.
- Strengthen the interpretability section with one concrete end-to-end case study, ideally including a failure mode, rather than only positive illustrative examples.

**Overall, the paper has a genuinely interesting and nontrivial idea with plausible empirical promise, but its current version does not yet validate the core mechanism as cleanly as the claims require.** Novelty is solid within the TM/NLP niche; technical soundness is reasonable but not fully established around the core attribution of gains; empirical support is competitive but incomplete; significance is promising for interpretable embedding research; and clarity is generally adequate, though some evaluation details and claim calibration need improvement.

---

## 3qAzQyOOnA

- GT: Reject (avg 4.0)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
This paper introduces **ToMBench-Hard**, a 900-example manually curated Theory-of-Mind benchmark spanning six ToM dimensions, and **Social-R1**, an RL framework that combines outcome rewards with a trajectory-level “social thinking” reward model inspired by Social Information Processing theory. Empirically, outcome-only RL on the hard benchmark already improves performance substantially, and the full method further improves results on several social reasoning benchmarks, though the evidence for the added value of the trajectory-level reward is mixed rather than cleanly established.

## Strengths
- **The paper identifies and operationalizes a concrete gap in current social-reasoning evaluation.** ToMBench-Hard is not just another aggregate benchmark: it is explicitly constructed around six ToM dimensions (_Emotion, Desire, Intention, Knowledge, Belief, Non-literal Communication_) and includes adversarial manipulations such as perceptual-access and asymmetric-information variations. The examples in Appendix A.1.3 and Figures 6–7 make the intended failure mode—shortcut reliance instead of genuine perspective reasoning—quite concrete.
- **Outcome-only RL on a hard social reasoning dataset appears genuinely effective.** This is one of the clearest empirical findings in the paper and is well supported by Table 3: for both Qwen3-4B and Qwen3-8B, the `w/o TRM` variant strongly improves over the base models on ToMBench-Hard and several transfer benchmarks. Independent of the more ambitious trajectory-reward claim, the paper makes a credible case that RL with verifiable outcomes can materially strengthen social reasoning when the training set is sufficiently challenging.
- **The trajectory-level reward is psychologically structured rather than generic process supervision.** The reward rubric is not an opaque “reasoning quality” score; it is organized around social cue perception, ToM-consistent interpretation, and concise reasoning, grounded in SIP theory (Section 3.2, Appendix A.2.1). Whether or not the current validation is sufficient, this is a more domain-specific and conceptually motivated process reward design than is typical.
- **The transfer evaluation is broader than the training task.** The model is trained on ToMBench-Hard but evaluated not only on ToMBench/ToMBench-Hard, but also on SocialIQA, EmoBench, MotiveBench, and SimpleToM. This broad evaluation is important because the paper’s central claim is about social reasoning rather than narrow benchmark optimization, and some cross-benchmark gains are indeed large, especially on SimpleToM and EmoBench.
- **The strongest empirical claim is parameter efficiency against some large open baselines, even if the framing should be more careful.** Table 2 does show Social-R1-4B outperforming the reported LLaMA3.1-70B numbers on all listed benchmarks. That is an interesting result in the paper as presented, even though it should not be overinterpreted as a universal superiority claim.

## Weaknesses

###: Fatal
None.

### Major:
- **The paper overclaims the necessity and robustness of the trajectory-level reward model (TRM); the ablations show mixed, not consistently additive, gains.**  
  This is the most important issue because the paper’s central novelty is not just RL on hard ToM data, but the claim that “process-level thinking rewards provide additional gains” and that “supervising the reasoning trajectory” is critical. Table 3 does not support this uniformly. For example, on **SimpleToM**, `Social-R1 4B w/o TRM = 0.9718` vs full `Social-R1-4B = 0.9365`; for 8B, `w/o TRM = 0.9741` vs full `0.8963`. On **EmoBench**, the 8B `w/o TRM` result (0.7205) is essentially tied with or slightly below the full model (0.7212), and on some settings the untrained-TRM variants are also surprisingly competitive in the appendix. So while the full method often helps, the paper does **not** establish that the trained social TRM is a consistently beneficial ingredient across tasks. The current narrative is stronger than the evidence.
- **The TRM is only weakly validated as a measure of social reasoning quality.**  
  The reward model is trained from LLM-generated and LLM-scored trajectories: o3 generates initial “gold” trajectories that are manually refined; GPT-4o/Qwen models generate candidates; GPT-5 scores them using the rubric; then a Qwen3-4B reward model is trained on pairwise preferences. This pipeline may be practical, but the paper does not show that the resulting reward correlates with independent human judgments, nor does it test for reward hacking or stylistic confounds. Since the method’s novelty rests on the claim that it supervises “human-like” social reasoning rather than surface-form compliance, this missing validation matters.
- **The benchmark contribution is promising, but construct validation of ToMBench-Hard is still limited.**  
  The paper shows that humans substantially outperform current LLMs and that ToMBench-Hard is harder than ToM-RL. That establishes difficulty, but not fully the stronger claim that the benchmark specifically isolates genuine ToM reasoning rather than a blend of ToM, narrative complexity, annotation artifacts, or linguistic difficulty. The benchmark is manually curated and proportioned across dimensions, which is good, but there are no inter-annotator agreement statistics, no systematic perturbation tests beyond a few qualitative examples, and no more formal analysis of shortcut resistance. For a benchmark positioned as a central contribution, stronger validation would be expected.
- **The empirical evidence does not disentangle whether RL is needed versus simpler supervised exposure to high-quality trajectories.**  
  The paper introduces both curated hard cases and curated/refined reasoning trajectories, then optimizes with RL. However, there is no supervised fine-tuning baseline on the same trajectory data. As a result, the paper cannot cleanly attribute gains to reinforcement learning and trajectory-level reward shaping, as opposed to simply benefiting from better social reasoning exemplars. This is particularly important because the data scale is small enough that a strong SFT control is feasible and highly informative.

### Minor
- **The training scale is small relative to the breadth of the claims.**  
  The policy is trained on 700 training samples for 300 GRPO steps, and the TRM uses a 3k preference dataset derived from 6.3k trajectories. The cross-benchmark gains are encouraging, so this is not by itself evidence of failure, but it does make claims like “genuine, robust, and systematic” enhancement of social intelligence feel overstated. A data-scaling or robustness analysis would help determine whether this is a general method or a highly data-efficient but narrow adaptation.
- **The paper leaves important reward-design questions unanswered.**  
  Equation (1) combines format, outcome, and thinking rewards, and Appendix A.3.1 states all three weights are set to 1.0. But there is no sensitivity analysis over \( \lambda_o \) and \( \lambda_t \), despite the paper’s core claim depending on the marginal value of the thinking reward. Without this, it remains unclear whether the method’s gains are mainly from the outcome reward and hard dataset, with the TRM acting as a weak auxiliary signal.
- **Some result presentation is confusing enough to impede interpretation.**  
  In Table 2, the reported percentage gains do not appear consistently tied to a single baseline convention across rows/benchmarks. Similarly, the mix of “thinking,” “disable thinking,” “+COT,” and task-specific prompting (e.g., `+MS` for SimpleToM) makes some comparisons harder to parse than necessary. This does not invalidate the results, but it does weaken clarity.
- **The benchmark annotation description is thinner than desirable for a new dataset paper.**  
  The main text says samples were “cross-checked independently by three annotators,” while Appendix A.1.2 gives a somewhat different and more detailed annotation workflow involving multiple graduate students and disagreement resolution. This is not a contradiction severe enough to undermine the paper, but the protocol should be unified and quantified more clearly.

### Trivial
- **The framing occasionally overstates what has been demonstrated.**  
  Phrases such as “human-like social intelligence” and “genuine social reasoning” go beyond what the current evidence establishes. The paper shows benchmark improvements on social reasoning tasks; it does not yet convincingly demonstrate human-like reasoning processes.

## Nice-to-Haves
- Add an **SFT baseline** trained on the same refined reasoning trajectories used in the pipeline.
- Report **human correlation** for the TRM on a held-out set of reasoning traces, or at least compare GPT-5 labels with human pairwise preferences.
- Provide a **weight sensitivity study** for \( \lambda_f, \lambda_o, \lambda_t \), especially \( \lambda_t \).
- Include stronger **benchmark validation**, e.g., perturbation-based shortcut tests, item difficulty/error analyses by ToM dimension, or agreement statistics.
- Analyze whether narrow social-RL training causes any **general capability regression** outside the target benchmarks.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that the LLaMA3-70B baseline is “likely flawed,” “broken,” or scientifically unsupported because its scores are anomalously low.**  
  The paper reports those results under its evaluation setup, and there is no direct evidence in the submission that the baseline was mis-evaluated. It is fair to say the comparison should be interpreted cautiously and documented more clearly, but not to assert evaluator error without evidence.
- **Complaint that comparing a trained 4B model to an untrained 70B model is inherently unfair.**  
  This is not a valid weakness in this context. Showing that a post-trained small model can beat a much larger pretrained baseline is a standard and meaningful comparison; if anything, the asymmetry favors the baseline on raw model capacity.
- **Claim that the paper says RL for social reasoning is completely unexplored.**  
  The paper does cite prior work such as ToM-RL and describes the area as “under-explored,” which is a reasonable characterization rather than a false claim of total novelty.
- **Criticism based on repository or release-status concerns.**  
  The paper cites and links resources; availability/existence doubts are not valid review points here.
- **Pure formatting/style issues and generic reproducibility nitpicks.**  
  There are some awkward phrasings and table-formatting artifacts in the extracted text, but these are not substantive scientific weaknesses.

## Novel Insights
The paper is strongest when read as making **two separable contributions rather than one unified triumph**: (1) hard, outcome-verifiable social reasoning data is already enough to make RL materially useful in this domain, and (2) process-level social rewards are an interesting but not yet conclusively validated extension. In other words, the current evidence more strongly supports the claim that **benchmark/task design is the primary driver**, while the specialized social TRM remains a promising but only partially substantiated add-on. Reframing the paper this way would make its empirical story both more honest and more compelling.

## Suggestions
- Reframe the main claim: present **outcome-based RL on ToMBench-Hard** as the most solid contribution, and describe the TRM as a **promising but mixed** extension unless stronger evidence is added.
- Add a **matched SFT baseline** on the same reasoning data to isolate the value of RL.
- Validate the TRM against **independent human judgments** and probe whether it rewards social reasoning quality rather than style or rubric mimicry.
- Include a **reward-weight ablation** to show whether \( \lambda_t \) contributes distinct value beyond outcome reward.
- Strengthen ToMBench-Hard with **agreement metrics**, deeper error analysis, and more systematic evidence of shortcut resistance.
- Clarify table comparisons and baseline conventions so that percentage gains and “thinking vs. no-thinking” settings are immediately interpretable.

Overall, the paper has a real idea and some genuinely encouraging results, especially around hard-data RL for social reasoning. But the present version overstates what has been proven about trajectory-level supervision and needs sharper empirical isolation of where the gains actually come from.

---

## fSE0rUngCX

- GT: Accept (Poster) (avg 7.3)
- Predicted: N/A (6.8/10)
- Match: N/A

### Final Review

## Summary
This paper introduces **Multimodal Policy Internalization (MPI)**: training a multimodal model to follow complex multimodal policies **without** supplying the policy at inference time. To support this setting, the authors contribute two benchmarks—**ClevrPolicy** for controlled, reasoning-heavy visual decision policies and **GTAPolicy** for multimodal tool-use rules—and propose **TriMPI**, a three-stage pipeline combining visually-masked continual pretraining, CoT SFT, and RL with **PolicyRollout**.

The work is clearly novel in scope: prior internalization/alignment work has largely focused on text-only settings or simpler prompt compression, whereas this paper targets multimodal, reasoning-intensive policies. Empirically, the paper shows large gains over its own no-policy internalization baselines, especially once RL is added, and it usefully evaluates not just task accuracy but also policy override, forgetting, and inference-time prompt savings.

## Strengths
- **The paper defines a genuinely new problem setting and supports it with task-specific benchmarks rather than only proposing an algorithm.** MPI is not just “prompt compression” in a new name: the policies here include multimodal content and reasoning-intensive rules, and the paper operationalizes this with two substantially different datasets:  
  - **ClevrPolicy**, which varies policy complexity by decision-tree depth and includes a multimodal-policy variant with image demonstrations inside the policy;  
  - **GTAPolicy**, which encodes tool descriptions plus versioning/user-conditional business rules for tool selection.  
  This benchmark construction is a meaningful contribution on its own.

- **ClevrPolicy is particularly well-designed for analysis of policy complexity.** The use of synthesized decision trees converted into natural-language policies gives unusually clean control over policy difficulty (e.g., \(N=2,4,6\)), and Table 1/Table 8 show that complexity systematically affects both in-context following and internalization performance. That makes the paper stronger scientifically than many agent papers that only evaluate on messy end tasks.

- **The proposed training decomposition is thoughtful and empirically informative.** The three stages are not redundant in the reported results: the paper ablates VM-CPT, RL, and PolicyRollout separately, and the results support the claim that **RL carries much of the gain**, while **VM-CPT and PolicyRollout add further improvement**, especially on harder settings. Even if one debates some design choices, the decomposition yields useful insight into what actually helps internalization.

- **The evaluation goes beyond end-task accuracy in ways that are specific to the paper’s claims.** In particular:  
  - **Policy Override** tests whether an internalized model can still follow updated in-context policy rules;  
  - **Policy In-Context** tests whether TriMPI improves policy following even when the policy is later supplied again;  
  - **catastrophic forgetting** is checked on MMMU-Pro, MMLU-Pro, and WildGuardTest;  
  - **efficiency** is quantified via prompt token reduction and prefill latency reduction.  
  These evaluations are aligned with the paper’s intended deployment story.

- **The efficiency motivation is concretely substantiated at inference time.** Figure 6 reports up to **93.9% prompt token reduction** and **85.7% prefill inference time reduction** once the policy is removed from the prompt. That is a specific and relevant systems-level benefit of internalization.

- **The paper surfaces a potentially useful RL idea.** PolicyRollout—augmenting the rollout pool with policy-aware responses while only optimizing the no-policy path—is a simple modification that appears empirically beneficial over vanilla GRPO/DAPO in their setting. Whether fully principled or not, it is the kind of practical training trick that could matter in future work on policy/internalization tasks.

## Weaknesses

###: Fatal
None.

### Major:
- **The paper does not sufficiently disentangle “policy internalization” from ordinary task-specific adaptation.**  
  A central headline comparison is the large gain over the **in-context** setting, but the in-context numbers in Table 1 are from **zero-shot off-the-shelf models**, while TriMPI and even CoT SFT are trained on thousands of task examples. That makes the strongest “better than in-context” claim less persuasive as evidence for the necessity of the proposed internalization framework.  
  Concretely, the paper’s own results already show that much of the gain comes from supervised/RL adaptation to the benchmark tasks: e.g., on ClevrPolicy-M \(N=6\), **CoT SFT + DAPO = 74.40** while **TriMPI + PoRo-DAPO = 85.00** (Table 8). This suggests TriMPI is a meaningful improvement over task-tuned baselines, but the much larger comparison to zero-shot in-context prompting overstates what has been established.  
  The core contribution is still valid, but the paper should frame it more carefully as improving **internalization training** over strong no-policy baselines, rather than as a fully decisive superiority claim over prompted policy use.

- **Evidence for true policy abstraction remains limited, especially on GTAPolicy.**  
  GTAPolicy has only **451 training instances and 106 test instances** (Appendix C.2.2, Table 7). In such a small-data regime, strong performance could reflect a mix of policy learning and narrow task-specific fitting. The paper does include useful probes—especially **Policy Override**—but the current evaluation still falls short of decisively showing that the model has learned generalizable policy reasoning rather than memorizing policy-specific behavior patterns.  
  This matters because the paper’s conceptual claim is stronger than “we improved benchmark accuracy”: it argues that the model has **internalized policy knowledge**. More structurally novel rule tests or held-out policy families would make that claim much stronger.

- **PolicyRollout is not analyzed rigorously enough given its nonstandard objective.**  
  From the paper text, PolicyRollout concatenates no-policy and policy-conditioned rollouts into the same rollout space for group-based advantage estimation, while applying policy gradient only to the no-policy path. This is an interesting heuristic, but the paper does not provide enough analysis of its optimization behavior, variance, or potential bias.  
  The issue is not that the method is obviously invalid—the paper clearly states the intended mechanism in §4.3 and Eq. (3)—but that for a key algorithmic novelty, the empirical support is stronger than the conceptual explanation. The paper would benefit from deeper analysis of how mixed rollout groups affect reward normalization and whether the gain comes from better exploration, better ranking signals, or something else.

- **The efficiency argument is one-sided because training cost is not quantified.**  
  The paper convincingly measures inference-side savings, but TriMPI adds **three stages**, including full-parameter CPT and RL with large rollout batches on H100s (Appendix B/Table 5). For a paper motivated partly by efficiency, the omission of any training-cost accounting is noticeable.  
  This is not a contradiction—deployment may still amortize the upfront cost—but without a train-vs-infer tradeoff analysis, the practical case is incomplete.

### Minor
- **The “Policy Referral” evaluation is only weak evidence because it relies on LLM-as-a-judge scoring of reasoning traces.**  
  The paper is transparent about this setup (§5.3, Appendix I), and uses it only as an auxiliary probe rather than the main metric, which is appropriate. Still, it remains subjective and somewhat vulnerable to stylistic alignment effects rather than pure policy understanding.

- **The real-world side of the benchmark suite is still relatively narrow.**  
  ClevrPolicy is intentionally synthetic and analytically useful; GTAPolicy is more realistic but small and derived from a specific tool-use benchmark. As the authors themselves note in §7, broader real-world multimodal policy settings would strengthen external validity.

- **The method is only tested on one base model family (Qwen2.5-VL, 3B/7B).**  
  The scaling across 3B and 7B is useful, but architectural diversity is limited. This weakens claims about generality somewhat, especially for an algorithmic contribution centered on training dynamics.

- **VM-CPT’s visual masking is empirically motivated but only lightly justified.**  
  The paper acknowledges this simplicity (“it has shown empirical success despite its simplicity”), but for multimodal policies that can themselves contain visual components, it would be helpful to better understand what cross-modal knowledge this stage does and does not inject.

### Trivial
- **Some of the strongest claims in the abstract/introduction are broader than what the evidence fully supports.**  
  The paper is strongest when comparing TriMPI to other **trained no-policy internalization baselines**, less so when using zero-shot in-context prompting as the marquee contrast.

## Nice-to-Haves
- Add a **training cost vs. inference savings** analysis (FLOPs, GPU-hours, or wall-clock), ideally showing break-even points under different deployment volumes.
- Add a more targeted test of **policy abstraction**, e.g., held-out policy structures, unseen rule templates, or new decision-tree families rather than only modified policy content.
- Provide a deeper analysis of **PolicyRollout**: how mixed rollout groups are normalized, how reward statistics differ between policy/no-policy samples, and whether separate-baseline variants change results.
- Evaluate on at least one additional **natural-image / larger-scale tool-use** benchmark to strengthen claims of real-world applicability.
- Report **run variance or seed sensitivity** for RL stages, especially since Table 6 shows instability/early stopping differences across DAPO runs.
- Compare against a **lighter-weight parameter-efficient internalization alternative** (e.g., adapter/LoRA-only internalization) to better justify the full training overhead.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The method is unreproducible because some cited models/tools may not be available or verifiable.”**  
  Removed under instruction: if cited in the paper, these entities are assumed to exist and be available.

- **Generic reproducibility complaints about omitted trivial implementation details.**  
  Removed because the paper actually provides substantial implementation detail in Appendix B/Table 5/Table 6, including learning rates, epochs, batch sizes, rollout batch sizes, KL coefficient, and hardware.

- **Claims that the paper lacks any baselines or any ablations.**  
  Factually incorrect. The paper includes Direct SFT, CoT SFT, CoT SFT + GRPO, CoT SFT + DAPO, and ablations over VM-CPT and PolicyRollout in Table 2/Table 8.

- **Formatting/style issues from PDF extraction.**  
  Removed as parser artifacts rather than paper weaknesses.

- **Criticism that Table 4 itself proves the method is not internalization because models improve when given policy in-context.**  
  Overstated. Table 4 only shows that TriMPI improves policy-following competence even when policies are later reintroduced; it does not invalidate internalization. At most, it suggests some gains may reflect broader task competence in addition to policy embedding.

## Novel Insights
The most interesting synthesis across the reviews and the paper is that **the strongest empirical story here is not “internalization beats prompting,” but rather “RL-based no-policy training can recover policy-following behavior far better than SFT alone, and explicit access to policy during training further improves that recovery.”** In other words, the paper’s real contribution may be less about proving a strict replacement for in-context policies and more about identifying a workable recipe for turning long, reasoning-heavy multimodal policies into latent behavior. The results also suggest a useful qualitative distinction: **ClevrPolicy diagnoses policy-complexity scaling cleanly, while GTAPolicy exposes the brittleness of low-data internalization**, making the two datasets complementary in a way that strengthens the benchmark contribution.

## Suggestions
- Reframe the headline claims to emphasize **improvement over trained no-policy baselines** rather than large gains over zero-shot in-context prompting.
- Add a controlled experiment where the policy is changed **structurally**, not just via content override, to better separate memorization from policy reasoning.
- Analyze PolicyRollout with an alternative variant that computes group statistics separately for policy-conditioned and no-policy rollouts, to test whether the current mixed grouping is essential.
- Include a quantitative **training-cost amortization** discussion to support the efficiency motivation.
- Strengthen the real-world evidence with either a larger GTAPolicy-style dataset or another benchmark involving natural images and richer tool-use policies.
- Report multi-seed results, or at least variance for RL stages, to clarify robustness of the claimed gains.



---

## bwtiK0yjuK

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (6.8/10)
- Match: N/A

### Final Review

## Summary
This paper studies offline change point localization and inference in dynamic multilayer random dot product graphs, proposing a two-stage procedure that combines seeded binary segmentation with tensor-based low-rank refinement. Its main technical claims are consistency for estimating both the number and locations of change points, plus limiting distributions and a data-driven confidence interval procedure for refined estimators.

## Strengths
- The paper tackles a genuinely specific and underexplored problem: **offline** change point localization and inference for **dynamic multilayer** latent-position network models, rather than either single-layer settings or online detection. The contribution is clearly scoped in Section 1.1 and is not just a minor variant of prior single-layer results.
- The methodological design is nontrivial and well integrated: Stage I uses seeded binary segmentation to obtain coarse candidates, and Stage II uses localized TH-PCA-based refinement. This is more than a generic pipeline; the tensor refinement is tailored to the multilayer low-rank structure induced by the D-MRDPG model.
- The paper provides substantial theory beyond consistency. In particular, it derives asymptotic distributions for refined estimators in both vanishing and non-vanishing jump regimes (Theorems 2 and 3), which is a meaningful advance over papers that stop at rate bounds.
- The experiments include several useful robustness checks beyond the main table: sensitivity to threshold and rank choices, performance under random change-point locations, temporal dependence stress tests, and some out-of-model scenarios. These help characterize where the method works well and where it degrades.
- The real-data analyses are not purely decorative: the method identifies interpretable change points in agricultural trade and U.S. air transportation data, and the paper attempts to connect detected changes to plausible domain events rather than reporting dates without interpretation.

## Weaknesses

### Major:
- **Theory–implementation mismatch for the independence assumptions.**  
  The theoretical analysis of Algorithm 1 assumes four mutually independent sequences \(\{A\}, \{A'\}, \{B\}, \{B'\}\), and this independence is used explicitly in the proofs. For example, Section 2.2 states: “**The assumption of mutual independence among the four sequences in Algorithm 1 is imposed for theoretical convenience. In practice ... Stage I and Stage II are implemented using the same two split tensor sequences via the odd–even splitting approach.**” The proof of Theorem 1 then conditions on the Stage I event and uses independence to justify that the distribution of the refinement sample is unaffected. As written, the main guarantees therefore apply to an idealized sample-split version, not directly to the exact implementation used in experiments. This is a substantive gap because the paper presents the empirical method and the theorem-backed method too closely, without a formal reconciliation.
- **The confidence-interval procedure is only partially justified by the theory provided.**  
  Section 3.1 presents a fully data-driven CI construction using plug-in estimates of the jump tensor and variances, but the paper does not provide a theorem proving that these plug-in estimates preserve the limiting law or yield asymptotically valid coverage. Theorems 2 and 3 establish limiting distributions for refined estimators, but the leap from those population-level limits to the practical plug-in CI algorithm is not fully closed. The empirical coverage study helps, but it does not replace a validity argument for the proposed interval construction.
- **The paper’s strongest empirical claim is somewhat overstated relative to the main-table comparisons.**  
  The abstract and contribution list claim superiority over “existing state-of-the-art algorithms,” but the main text primarily compares against gSeg and kerSeg, which are generic change-point methods adapted to network-derived inputs. The more directly relevant comparison to a dynamic multilayer network method (CPDonline from Wang et al., 2025) is deferred to the appendix. Since the paper’s core selling point is being the first offline method in this setting, the empirical evidence would be stronger if the most relevant adapted network comparator were featured centrally rather than peripherally.

### Minor
- **The theoretical regime is restrictive in a way that matters for the claimed scope.**  
  Model 1 assumes \(\Delta=\Theta(T)\), which effectively keeps the number of changes bounded. The paper is transparent about this and discusses it in Section 5, but it remains a real limitation because many dynamic-network applications involve more frequent changes. The appendix includes some experiments with larger numbers of change points, which is useful, but those experiments sit outside the main theory.
- **The low-rank assumptions are somewhat abstract and not especially interpretable from the network model itself.**  
  Assumption 1(ii)–(iii) imposes rank conditions on transformed \(Q\)-matrices. The authors themselves note: “**this low-rank structure may not directly or transparently reflect the explicit model structure**.” That honesty is appreciated, but it also means the assumptions are not especially natural from a modeling standpoint, which weakens the practical interpretability of the theory.
- **Finite-sample reliability of the inference procedure looks uneven outside the model class.**  
  Table 2 shows 76.67% coverage for a nominal 95% CI in Scenario 3 when \(n=100\), and the paper notes that this scenario violates Model 1 and involves smaller layer-specific changes. This does not invalidate the theory—since the model is violated—but it does indicate that the practical CI procedure can be fragile, and the paper could do more to state when users should distrust the intervals.
- **Scalability is a practical concern.**  
  The complexity is quadratic in \(n\), and Appendix G reports about 10 hours for \(n=100, T=200\) over 100 Monte Carlo trials on a CPU. This does not make the method unusable, but it does suggest that the current implementation may be heavy for larger multilayer networks, especially given that Stage II uses repeated tensor estimation.

### Trivial
- The real-data confidence intervals can look extremely sharp relative to the small time horizons (e.g., agricultural trade with \(T=35\)), which invites skepticism about finite-sample calibration even if not a formal contradiction of the asymptotic theory. A short cautionary discussion would help.

## Nice-to-Haves
- Include an ablation directly showing how much Stage II refinement improves over Stage I, e.g., by plotting raw CUSUM versus refined scan statistics around true changes.
- Add an experiment on rank misspecification beyond the limited sensitivity table, since Stage II depends on low-rank tensor estimation.
- Clarify practical guidance for threshold calibration, since the paper tunes \(c_{\tau,1}\) via null simulations in the appendix rather than giving a fully automatic selection rule.
- Expand the discussion of how one should construct confidence intervals in the non-vanishing jump regime, since the main text focuses on the vanishing regime.
- Provide a stronger computational discussion, including where the time is spent and whether sparsity or warm-starting TH-PCA could reduce cost.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Missing stronger offline multilayer baselines” as a criticism of related work coverage.**  
  It is fair to say the main empirical comparison could foreground more relevant comparators, which I keep above. But claims framed as “missing related methods” or assuming unspecified baselines should exist are not reliable to include as core weaknesses without external verification.
- **Criticism that the paper should not evaluate on scenarios violating Model 1.**  
  The paper explicitly says Scenarios 2 and 3 do **not** follow Model 1 and uses them to assess robustness: “**The changes in Scenarios 1 and 4 follow Model 1, while those in Scenarios 2 and 3 do not, allowing us to assess the robustness of our methods.**” So treating this as a flaw would misunderstand the purpose of those experiments.
- **Complaint that frequent-change experiments invalidate the paper because \(\Delta=\Theta(T)\) is assumed.**  
  The theory indeed does not cover those settings, but the appendix experiments are clearly presented as robustness studies outside the theory, not as evidence for the theorem. The right criticism is that the theory is restrictive, not that these experiments are illegitimate.
- **Reproducibility nitpicks about releasing exact seeds/code/preprocessing.**  
  These are not substantive enough for the main review under the stated rubric.
- **Purely generic strengths such as “the paper is well written” or “the experiments are extensive.”**  
  Omitted because they are not specific enough.

## Novel Insights
The most important synthesis point is that this is a paper with **real theoretical ambition and genuine novelty**, but its strongest vulnerability is not the mathematics itself—it is the gap between the **idealized sample-splitting device used to obtain clean proofs** and the **practical odd–even implementation actually evaluated**. A second key insight is that the paper is strongest on localization theory and weakest on inference calibration: the asymptotic limit laws are interesting and likely valuable, but the practical CI procedure currently feels one theorem short of being fully justified. So this is not a weak paper in the usual sense; rather, it is a strong technical paper whose final inferential and empirical claims are somewhat ahead of what is rigorously established.

## Suggestions
- Prove a version of the main localization and inference results for the actual odd–even implementation, or clearly restate theorems as applying only to the idealized split-sample algorithm and discuss the gap explicitly.
- Add a theorem or proposition establishing asymptotic validity of the plug-in CI procedure, including consistency of \(\hat\kappa_k\), \(\hat\Psi_k\), and the variance estimators.
- Move the CPDonline comparison from the appendix into the main experimental section, and moderate “state-of-the-art” language unless the main text supports it directly.
- Add a concise warning box or proposition describing when CI coverage may degrade in practice, especially under model misspecification or short time horizons.
- Include an ablation isolating the gain from TH-PCA refinement over Stage I alone, both in localization error and computational cost.



---

## 2tDLQuz0H6

- GT: Reject (avg 2.0)
- Predicted: N/A (5.3/10)
- Match: N/A

### Final Review

## Summary
This paper introduces GREPO, a graph-ready benchmark for repository-level bug localization built from 109 Python repositories and 10k+ bug-fixing pull requests. The benchmark contribution is meaningful: it provides temporal repository graphs, node/edge structure, and labels aligned to historical bug states. The empirical study on 9 repositories shows that standard GNNs can be effective rerankers/localizers on this benchmark, but the paper overstates what these results establish about repository-wide structural reasoning by GNNs.

## Strengths
- **The benchmark contribution is concrete and unusually usable for graph learning research.** The paper does more than release issue/PR pairs: it constructs heterogeneous repository graphs with node types (Directory/File/Class/Function), structural edges (containment, call, inheritance, reverse edges), temporal validity intervals, and snapshot extraction at the bug-inducing commit. This substantially lowers the barrier to applying GNNs to repository-level software tasks.
- **The temporal graph construction is a real technical contribution, not just data packaging.** The incremental build procedure with start/end timestamps and reparsing only changed files is a practical design for scaling historical repository snapshots. This is more thoughtful than static-graph dataset creation and is well matched to the bug localization setting where leakage from future commits matters.
- **The paper includes informative ablations that reveal where performance comes from.** Table 3 is particularly valuable: removing edge structure, similarity, anchor flags, or node features causes large drops, which gives a much clearer picture of the pipeline than many benchmark papers provide.
- **Cross-repository training results are interesting and potentially important.** Joint training clearly outperforms per-repository training in Table 2, suggesting that the model learns transferably useful localization patterns rather than only repository-specific heuristics.
- **The file-level vs. class/function-level evaluation split is useful.** The results show a nontrivial pattern: agent baselines are extremely strong at file-level localization, while the best GNN performs much better at finer-grained class/function localization. That distinction is practically relevant and worth surfacing.

## Weaknesses

### Major:
- **The paper’s central narrative about GNN-enabled multi-hop structural reasoning is not convincingly supported by the evidence provided.** The strongest empirical signal in the pipeline appears to be the precomputed text-query similarity and anchor selection, not graph reasoning alone. In Table 3, removing `sim` drops file-level Hit@1 from **54.18** to **4.11** and class/function Hit@1 from **22.27** to **0.44**; removing `anchor` drops file-level Hit@1 to **9.48**. This does not mean the graph is useless—`w/o Edge` also degrades sharply—but it does mean the current experiments support a more modest claim: the GNN is effective **when seeded by strong retrieval features and anchor-centric subgraphs**. As written, the paper overclaims that it demonstrates repository-wide structural reasoning.
- **The anchor-based evaluation setup creates a confound between retrieval and graph reasoning.** The method first identifies anchor nodes using embedding similarity and LLM name/path matching, then extracts only k-hop subgraphs around those anchors, and also uses similarity as an explicit node feature. This tightly couples candidate generation and graph inference around the same initial textual signal. As a result, the benchmarked GNN is not really tested on open-ended repository-wide localization; it is tested on localization within a retrieval-pruned neighborhood. The paper partially acknowledges locality and efficiency motivations, but the current setup makes it hard to isolate how much of the gain comes from learned structural propagation versus strong initial retrieval.
- **The comparative claims against baselines are overstated, especially relative to Agentless.** Section 6.3 states that GAT “significantly surpasses the Agentless approach” based on class/function-level results, but Table 1 simultaneously shows Agentless is dramatically stronger at file-level localization (**92.72 Hit@1** vs **54.18** for GAT). Since file identification is a core part of repository-level bug localization, the paper should present this as a tradeoff, not as broad superiority. The paper is correct that GNNs outperform some baselines on fine-grained localization, but its prose currently overgeneralizes from that advantage.
- **Only 9 of the 109 repositories are used in the experimental evaluation, which limits how strongly one can interpret GREPO as an evaluated benchmark rather than primarily a released resource.** The dataset itself may still be valuable, but the paper’s empirical claims are demonstrated only on a curated subset. For a benchmark paper, this is a noticeable gap between resource scope and evaluation scope.

### Minor
- **The paper does not quantify the upper bound imposed by anchor retrieval.** It reports that 1-hop or 2-hop subgraphs cover “over 80%” of modified nodes on average, which is helpful, but this is not the same as reporting anchor recall / coverage as a hard ceiling for downstream localization. Since the method depends heavily on anchors, the benchmark would be much more informative with explicit oracle-ceiling analysis.
- **Scalability is argued but not measured.** The paper motivates GNNs partly through repository-scale reasoning and efficiency relative to whole-repository LLM processing, yet it does not report graph sizes, extraction cost, inference latency, or memory/runtime profiles. Given the benchmark framing, such measurements would materially strengthen the work.
- **The handling of non-linear repository history is a reasonable approximation, but still a validity limitation.** The paper linearizes the commit DAG via longest-path extraction and notes that over 75% of commits lie on the main/master longest path. This is a sensible engineering simplification, not a fatal flaw, but it leaves some uncertainty for bugs tied to branch-specific or merge-specific histories.
- **The practical impact is less clear because the paper itself reports weak downstream agent gains.** The limitations section states that using the GNN for SWE-Bench-Live agent testing yielded unsatisfactory outcomes and did not significantly improve localization or issue resolution in the agent setting. This honesty is appreciated, but it also weakens the broader practical case unless analyzed more deeply.

### Trivial
- **The benchmark is Python-only.** This narrows generality, though it is a reasonable initial scope rather than a serious flaw.
- **The paper would benefit from clearer specification of the training objective and ranking formulation for multi-label localization.** The evaluation metric is defined, but the exact supervision/loss used for GNN training is not made sufficiently explicit in the main text.

## Nice-to-Haves
- Add a nonparametric graph propagation baseline initialized with the same anchors/similarity features (e.g., Personalized PageRank or random-walk ranking) to test whether learned message passing is necessary.
- Report anchor recall / oracle coverage explicitly, so readers can separate retrieval limitations from GNN limitations.
- Analyze how much the GNN actually reranks candidates beyond the initial similarity scores; e.g., measure rank shifts from `sim` alone to final predictions.
- Include runtime/memory statistics for graph construction, subgraph extraction, and GNN inference.
- Evaluate on more than 9 repositories, or clearly position the current experiments as a pilot study over a larger released benchmark.
- Add standard ranking metrics such as MRR/MAP/NDCG alongside Hit@k for easier comparison to prior IR-style bug localization work.
- Include one or two qualitative case studies showing whether the model succeeds by following structural edges to textually weak but topologically relevant nodes.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **“Unfair comparison because GNNs are trained while baselines are not.”** Removed as a core criticism. The paper explicitly compares jointly trained GNNs against off-the-shelf baselines and states this setup in Section 6.3. While this means the comparison should be interpreted carefully, it is not inherently invalid, and the asymmetry does **not** favor the authors in all respects: the baselines are very strong large pretrained systems and Agentless substantially outperforms the GNN at file level. The real issue is not fairness per se, but that the paper overstates broad superiority despite mixed results.
- **“Ground-truth labels from PR-modified files/classes/functions are too noisy to trust.”** Soft-removed. This is a generic concern for software engineering datasets and the paper’s label construction is standard and clearly described. The paper does not claim root-cause labels; it claims modified-entity labels, which is exactly what it uses.
- **“Need full release of raw graphs/embeddings to be reproducible.”** Removed as a reproducibility nitpick beyond submission standards.
- **Formatting/writing issues and parser artifacts.** Removed per instruction.
- **Concerns about cited models/tools/datasets existing or being available.** Removed by rule.
- **Missing related work or external references.** Removed by rule.

## Novel Insights
The most important synthesis from the reviews and the paper itself is that GREPO appears stronger as a **benchmark/resource paper** than as evidence for a new scientific conclusion about GNNs performing repository-wide multi-hop bug reasoning. The experiments do show that structural information matters—ablating edges hurts badly—but they also show that success is tightly bottlenecked by retrieval-derived similarity and anchor selection. So the true takeaway is not “GNNs solve repository-level bug localization through structural reasoning,” but rather “graph-based reranking over retrieval-pruned repository neighborhoods is promising, especially for fine-grained localization.” That is still a useful insight, but it is narrower than the current framing.

## Suggestions
- Reframe the claims more conservatively: emphasize GREPO as the main contribution and present the GNN results as a strong baseline for **graph-based reranking/localization**, not definitive proof of repository-wide multi-hop reasoning.
- Add an analysis of anchor recall and the fraction of ground-truth nodes reachable within each k-hop neighborhood.
- Compare against a simple graph-propagation baseline using the same anchors and similarity features to isolate the benefit of learned GNNs.
- Tone down claims of superiority over Agentless and instead explicitly discuss the file-level vs. function-level tradeoff.
- Expand experiments beyond 9 repositories if feasible; otherwise, clearly justify the subset and characterize it statistically relative to the full 109-repository benchmark.
- Provide computational cost measurements to support the scalability motivation.
- Analyze why better localization did not improve downstream agent performance; this could become an important and honest contribution rather than a brief limitation note.

---

## EsumhpzFK9

- GT: Withdrawn (treated as Reject) (avg 2.0)
- Predicted: N/A (3.1/10)
- Match: N/A

### Final Review

## Summary
This paper proposes KARMA, a reinforcement learning framework that augments the reward signal using two additional ingredients: structured domain knowledge encoded as a knowledge graph and a learned causal model used for counterfactual reward adjustment. The intended contribution is a modular reward-shaping framework that dynamically shifts from knowledge-guided shaping early in training to causally informed shaping later, with claimed benefits in sample efficiency, robustness, and generalization across grid navigation, robotics, and traffic control tasks.

## Strengths
- **The paper targets a real and important failure mode—spurious or misleading reward signals—and addresses it at the reward-design level rather than only through better policies or representations.** This is more specific than generic “causal RL”: KARMA explicitly applies causality to *reward adjustment* via Eq. (1), which is a meaningful conceptual angle.
- **The framework is modular in a way that is easy to inspect experimentally.** The decomposition into knowledge integration, causal discovery, and reward adjustment modules is reflected in the ablation table (Table 2), which at least attempts to separate where the gains come from rather than only presenting an end-to-end black box.
- **The empirical scope is broader than a single toy benchmark.** The paper evaluates on three qualitatively different settings—GridWorld with causal interference, robot skill acquisition, and traffic signal control—and reports not only final return/sample efficiency (Table 1) but also robustness and distribution-shift results (Figure 6), which is aligned with the paper’s stated goals.
- **The ablations support that the full combination matters more than any single ingredient alone.** In Table 2, removing reward adjustment hurts most, while removing knowledge or causal learning also degrades results, which is consistent with the paper’s claim that the synergy of the components is important rather than incidental.
- **The paper is unusually explicit about computational cost for this kind of systems-style method.** Table 3 reports training time, peak memory, and inference latency, making clear that the gains are not free and allowing readers to judge the tradeoff.

## Weaknesses

### Fatal
- **The core method is underspecified to the point that the main contribution is not scientifically verifiable from the paper text.**  
  The central mechanism is Eq. (1),
  \[
  R'(s,a,r,s') = r + w_K(t)R_{\text{knowledge}}(s,a,s') + w_C(t)R_{\text{causal}}(s,a,s')
  \]
  but neither \(R_{\text{knowledge}}\) nor \(R_{\text{causal}}\) is defined operationally. Section 3.3 only states that “\(R_{\text{knowledge}}\) promotes trajectories consistent with KG constraints” and that “\(R_{\text{causal}}\) is obtained through counterfactual queries on \(C\), using Pearl’s do-calculus to estimate causal effects of actions, disentangled from confounders.” This is too high-level for the paper’s core algorithm: there is no mathematical form, no estimator, no pseudocode, no description of how the counterfactual quantity is computed per transition, and no explanation of how these terms are scaled or normalized so that PPO can be trained stably.  
  Section 4.4 adds some implementation details (“linear annealing,” “counterfactuals computed via structural equation modeling”), but still does not specify the actual structural equations, fitting objective, or how online estimates enter the reward at each step. Since the central claim is precisely that this reward adjustment mechanism drives the gains, the lack of formal algorithmic definition materially undermines technical soundness.

### Major:
- **The claimed theoretical guarantees are not substantiated in the main paper at a level commensurate with the strength of the claims.**  
  Section 3.4 lists four guarantees—convergence of causal discovery, policy invariance, improved sample efficiency, and convergence of KARMA-RL to an \(\epsilon\)-optimal policy—but presents them only as bullet points under “mild assumptions.” The assumptions are not spelled out in the main text, nor are theorem statements or proof sketches given. Some of these claims are individually plausible in restricted settings (e.g., policy invariance if the shaping term is potential-based), but the paper does not establish that the *full* KARMA reward in Eq. (1) satisfies those conditions. In particular, the policy-invariance statement is explicitly conditional: “**If** \(R_{\text{knowledge}}\) is designed as a potential-based shaping function, the optimal policy is preserved.” The paper never shows that the implemented reward actually has that form, and no analogous condition is given for \(R_{\text{causal}}\). Because the paper foregrounds theory in the abstract and contributions list, this lack of formal support is a substantial weakness.
- **The causal discovery component is insufficiently justified for the online RL setting used here.**  
  The paper states in Section 3.2 that it uses “constraint-based methods (e.g., PC, FCI) with score-based refinements,” and Section 4.4 says causal discovery is updated “every 1000 interactions.” However, the paper does not explain how these discovery procedures are adapted to temporally correlated, policy-dependent RL data, nor how conditional independence testing is handled in continuous/high-dimensional environments like the 7-DOF robot and traffic control tasks. This is not merely a request for extra detail: the credibility of \(R_{\text{causal}}\) depends on whether the learned graph is meaningful under the actual data-generation process. The text mentions “MDP-informed temporal constraints,” which partially addresses obvious temporal ordering, but it does not resolve the broader issue that the discovery pipeline’s assumptions and estimation quality are left vague precisely where the method depends on them.
- **The empirical evidence does not adequately isolate whether the gains come from causal reasoning specifically, versus from adding reward density or extra task-specific information.**  
  The paper compares against standard RL, knowledge-based RL, and causal RL baselines, and provides internal ablations. However, it does not include a matched reward-shaping baseline that would control for the possibility that much of the gain comes simply from providing a denser auxiliary reward rather than from correct causal/counterfactual adjustment. This matters because Eq. (1) adds two extra reward terms on top of the environment reward; without a stronger control, the causal interpretation of the gains remains weaker than the headline framing suggests.
- **The paper’s robustness claims are incomplete because robustness to *incorrect knowledge* or *misspecified causal priors* is not evaluated.**  
  Section 6 explicitly acknowledges that “large errors can harm learning” and that performance depends on “the quality of its knowledge graph and causal model.” But the experiments do not probe this dependence: there is no stress test with corrupted knowledge graphs, contradictory rules, or degraded causal structure estimates. Since a core premise of the framework is that prior knowledge helps disambiguate spurious reward signals, the absence of sensitivity analysis leaves a significant gap between the claimed practical robustness and what is demonstrated.

### Minor
- **The RLVR/LLM-alignment motivation is much broader than the evaluation actually supports.**  
  The introduction motivates the work partly through spurious rewards in RLVR for language models, but all experiments are on classical control/simulation tasks. This does not invalidate the paper, but the significance claims around alignment and RLVR are aspirational rather than empirically supported by the presented results.
- **The dynamic weighting mechanism is plausible but not convincingly analyzed.**  
  The paper claims that \(w_K(t)\) should matter early and \(w_C(t)\) later, but beyond the statement of “linear annealing” and the static-weight ablation, there is no analysis of the schedule itself, no plot of the weights over training, and no evidence that this transition corresponds to a meaningful change in what the agent learns. As written, it is hard to tell whether the schedule is a key idea or simply a generic reward-bonus decay heuristic.
- **The computational overhead is reported but not decomposed.**  
  Table 3 is useful, but it would be more convincing to separate the cost of KG processing, causal graph updates, and counterfactual reward computation. That is especially relevant because scalability is already acknowledged as a limitation in Section 6.
- **Only five runs are used, which is on the light side for claims about consistent gains across multiple components and settings.**  
  The paper does report mean and standard deviation and mentions t-tests, which is better than many submissions, so this is not a major methodological flaw. Still, for a method with several moving parts and potentially high variance, the evidence base is somewhat limited.

### Trivial
- None.

## Nice-to-Haves
- Add a controlled baseline with comparable dense reward shaping but without causal machinery, to better isolate the value of the causal component.
- Include explicit pseudocode and formulas for \(R_{\text{knowledge}}\), \(R_{\text{causal}}\), the SEM fitting procedure, and the weight schedules \(w_K(t), w_C(t)\).
- Add sensitivity experiments with noisy/incomplete/contradictory knowledge graphs and imperfect causal priors.
- Visualize the learned causal graph against known structure in the GridWorld setting, where ground-truth causal dependencies appear available.
- Break down runtime by module and discuss whether parts of the pipeline can be precomputed or amortized.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper misses critical related work.”**  
  Removed per instruction: I cannot verify external omissions beyond the cited references, and the criticism was framed broadly rather than against a concrete mischaracterization in the paper.
- **“The comparison is unfair because baselines do not get the same extra information.”**  
  Removed. The authors’ method is intentionally designed to use additional knowledge/causal structure; this asymmetry does not by itself make the comparison invalid, and in fact often strengthens the claim if the baseline is disadvantaged less than the proposed method.
- **“Code is not public during review / supplementary material is inaccessible / post-publication release harms reproducibility.”**  
  Removed as a core criticism. Lack of immediate release should not be treated as decisive here, and the paper does provide some implementation details. The real issue is not release status but that the main algorithm itself is underspecified.
- **Generic praise such as “the paper is well-written” or “the topic is important.”**  
  Removed because these strengths are too generic under the reviewing instructions.
- **Purely stylistic complaints about the related work being descriptive or the prose using broad motivation language.**  
  Removed unless tied to a substantive technical issue.

## Novel Insights
The most important synthesis across the reviews is that the paper’s central risk is not simply “more details would help,” but that the claimed contribution sits at an awkward boundary between conceptual framework and complete algorithm. The experiments and modular ablations suggest there may be a real underlying idea here—using knowledge to regularize early learning and causal estimates to refine rewards later—but the paper never closes the loop from concept to a technically precise, auditable method. Put differently: the strongest positive signal is that the authors have identified an interesting *design space* for reward shaping, while the strongest negative signal is that the current submission does not yet establish KARMA as a sufficiently specified method within that design space.

## Suggestions
- **Define the core reward terms formally.** Give exact formulas or algorithms for \(R_{\text{knowledge}}\) and \(R_{\text{causal}}\), including normalization/scaling and how they are injected into PPO advantage estimation.
- **State theorems properly in the main paper.** At minimum, include theorem statements, assumptions, and proof sketches for the convergence/invariance claims, and be explicit about which claims apply only to restricted variants of KARMA.
- **Clarify causal estimation in RL data.** Explain how the SCM is fit from sequential interaction data, how often it is updated, what variables are included, and what assumptions justify using the chosen causal discovery method online.
- **Add stronger controls.** Compare against a matched non-causal dense reward shaping baseline to test whether causality contributes beyond reward densification.
- **Stress-test prior misspecification.** Corrupt the knowledge graph and/or causal priors in controlled ways and quantify degradation.
- **Show one concrete case study.** For example, in GridWorld, visualize a spurious feature, the learned causal graph, and an instance where KARMA adjusts the raw reward in the intended direction.

---

## egPSakPG0e

- GT: Withdrawn (treated as Reject) (avg 2.4)
- Predicted: N/A (5.1/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a two-stage text clustering framework: (1) generate multiple transformed views of sentence embeddings, cluster each view, and aggregate them through a co-occurrence matrix plus spectral consensus; (2) use the resulting assignments to train an MLP with a joint InfoNCE and GMM negative log-likelihood objective so that the latent space becomes more cluster-friendly. The paper also presents a theorem claiming exponentially decreasing consensus error with the number of views under independence and informativeness assumptions, and reports gains on DBPedia and Reuters-R8, including a train/test split experiment intended to show transfer to unseen text.

## Strengths
- **The paper combines consensus clustering and latent-space shaping in a concrete, end-to-end pipeline rather than stopping at cluster ensembling.** Algorithm 1 produces consensus pseudo-labels from multiple transformed views, and Algorithm 2 feeds these into a joint InfoNCE + GMM training loop. This is more substantive than a pure post-hoc ensemble because it tries to use consensus structure to improve the representation itself.
- **The empirical results do show a consistent pattern that multi-view aggregation helps over the reported single-view clustering runs under the same transformed-view families.** In Tables 2–4, the consensus method typically improves over the per-view GMM results for the corresponding transformation sets, often by a nontrivial ARI margin (e.g., on DBPedia with \(k=8\), “PCA + WPT + Multiple Models” gives ARI 71.4 for consensus vs lower single-view figures in that block).
- **The paper’s main useful insight is that “good” diversification in clustering is not arbitrary corruption but a trade-off between view diversity and per-view informativeness.** Even though the formal theory is overstated, the paper repeatedly operationalizes this idea through transformation families such as PCA, WPT, noise, and multiple encoders, and Figure 2 is clearly aimed at studying that interaction rather than just reporting one best number.
- **The unseen-data experiment, while underspecified, suggests the learned embedding space may retain useful cluster structure beyond the exact samples used for consensus formation.** Table 5 shows train/test NMI and ARI staying relatively close even when the train fraction is reduced substantially, which is at least suggestive that the learned mapper is not purely memorizing the training subset.

## Weaknesses

###: Fatal
None.

### Major:
- **The central theoretical guarantee does not actually justify the practical method as implemented.** The theorem in Section 2.2.3 / Appendix explicitly requires **mutual independence of views**: “**Condition 1 (View Diversity): The collection \(\{X_v\}_{v=1}^m\) is mutually independent.**” But the proposed views are generated from the same underlying embeddings using PCA, WPT, Gaussian noise, or different sentence encoders applied to the same texts. These are clearly not independent in the theorem’s sense. The paper briefly says “one can argue that weakly uncorrelated views contribute proportionally,” but this is only an informal remark and no bound is provided for correlated views. As a result, the headline claim in the abstract—“**We prove that consensus clustering achieves an exponentially lower expected error rate compared to any single view**”—is too strong relative to what is actually proved; the proof covers an idealized setting, not the practical pipeline.
- **The empirical evaluation does not cleanly isolate what causes the gains.** The paper attributes improvements to “informative diversification” plus consensus, but the strongest reported systems combine multiple changes at once: transformed embeddings, multiple encoder families, and consensus aggregation. The tables do include per-view GMM numbers for the transformed settings, which partially addresses the issue, but they still do **not** isolate the benefit of the *spectral consensus step itself* against simpler aggregations or against clustering directly on a pooled/concatenated representation. Nor do they isolate the contribution of the second-stage training from the first-stage consensus in a quantitative table. Because the method is a composition of several ingredients, the current experiments support that the overall recipe can help, but they do not establish which component is necessary or primarily responsible.
- **The “generalization to unseen text” protocol is too ambiguous to support the paper’s stronger claims.** Section 3.3 says the model can be trained on a subset and “the trained model reshapes the latent space such that the resulting embeddings are more amenable to clustering using a simple KMeans or GMM algorithm.” However, Algorithm 1’s consensus step is transductive over all samples in a set via an \(n \times n\) co-occurrence matrix and spectral clustering, whereas Section 3.3 does not specify what exactly happens at test time: whether test points are assigned using fixed learned centroids, reclustered jointly, or clustered independently with KMeans/GMM in the new latent space. These are materially different protocols. Without a precise inference description, the claim that the framework “generalizes effectively to unseen text” is not yet well substantiated.
- **The scalability claim is not supported by the presented method or experiments.** The abstract describes the method as “robust and scalable,” yet the consensus stage explicitly constructs an \(n \times n\) co-occurrence matrix \(W\) and then performs spectral clustering via the normalized Laplacian and its \(K\) smallest eigenvectors. This is a substantial bottleneck for large text corpora, especially in the “knowledge discovery” / “RAG” setting emphasized in the introduction and conclusion. No complexity analysis, memory discussion, or scaling experiment is provided, so the practical significance for large-scale corpora is overstated.

### Minor
- **The paper’s empirical notion of “diversity” is conceptually muddled.** Section 3.1 says diversity is “quantified as the mean ARI value across all pairwise combinations of the generated views.” But pairwise ARI measures agreement between clusterings, so high ARI indicates similar outputs rather than obviously “diverse” views. The text then links this empirical quantity to the theorem’s diversity/independence condition. That connection is not well justified and risks confusing agreement, dependence, and informativeness.
- **The role and parameterization of the GMM covariance are inconsistent across sections.** Section 2.1 introduces the general GMM with full \(\Sigma_k\), Section 2.2.2 states clustering uses “isotropic homogeneous covariance,” while Section 2.3 writes a GMM loss with \(\Sigma_k\) again and discusses \(N(h_i\mid \mu_k,\sigma_k)\). The experiments also mention homogeneous and heterogeneous isotropic settings. This does not make the method invalid, but the exposition is mathematically loose and leaves uncertainty about what is actually optimized in Algorithm 2.
- **The latent-space learning claims are under-evidenced.** Section 3.2 states that silhouette and Calinski–Harabasz scores validate the reshaped latent space, but the paper does not report those scores numerically in the main text. The qualitative figure is not enough to evaluate how strong the effect is.
- **Evaluation breadth is limited for the scope of the claims.** The method is tested on DBPedia and Reuters-R8 only. That is enough to show the idea is not vacuous, but it is thin support for claims about robustness across real-world text clustering or retrieval-oriented settings.

### Trivial
- **Some claims in the abstract and conclusion are broader than what is directly shown.** In particular, the repeated references to RAG/knowledge discovery applications feel motivational rather than experimentally grounded, since no retrieval or downstream pipeline is evaluated.

## Nice-to-Haves
- Add an explicit ablation table separating: (i) single-view clustering, (ii) transformed-view clustering without consensus, (iii) consensus without latent-space training, and (iv) consensus + InfoNCE/GMM training.
- Report performance as a function of the number of views \(m\), which would directly test the practical utility of the theory.
- Measure inter-view dependence/correlation and discuss how far the generated views are from the theorem’s independence assumption.
- Clarify the exact test-time protocol for Section 3.3, ideally with separate results for centroid assignment vs reclustering.
- Provide a complexity analysis and, if possible, a scalable approximation to the spectral consensus step.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Missing recent related work / stronger modern baselines.** It is reasonable to want stronger baselines, but specific omissions of external works should not be emphasized here because they cannot be verified from the paper alone. The kept criticism is therefore the narrower and verifiable one: the current baseline suite is limited relative to the breadth of the paper’s claims.
- **Pure reproducibility complaints about unspecified hyperparameters or search strategy.** The paper could certainly be clearer, but the absence of exhaustive hyperparameter details is not, by itself, a substantive ICLR-level flaw here.
- **Formatting/table readability complaints caused by PDF extraction artifacts.** These are parser issues and not evidence against the submission.
- **Claims that the paper compares unfairly to baselines because the author method gets stronger inputs.** This point was weakened rather than removed entirely: the real issue is not “unfairness” in the narrow asymmetry sense, but lack of ablation to identify which component drives gains.
- **Criticism that the theorem is completely invalid because the practical views are dependent.** The stronger claim is not justified, but the theorem still holds for its stated assumptions. The valid criticism is the mismatch between theorem and implementation, not that the mathematical statement itself is false.

## Novel Insights
The most important synthesis across the paper and reviews is that the work’s real contribution is more empirical than theoretical: it identifies a practically useful recipe for generating multiple *informative but nonidentical* embedding views and then feeding their consensus into a representation-learning stage. The paper’s experiments suggest that this recipe can work, but the theorem should be reframed as an idealized explanation of why diversification can help, not as a direct guarantee for the specific PCA/WPT/noise/multi-encoder pipeline. In other words, the “spark” here is not exponential-error theory per se, but the observation that consensus pseudo-labels can act as a stabilizing target for subsequent latent-space shaping in text clustering.

## Suggestions
- Reframe the theory more carefully: present it as a guarantee under idealized independent informative views, and explicitly discuss the gap to the practical correlated-view construction.
- Add a direct empirical study of inter-view dependence and its relationship to gains from consensus.
- Clarify Section 3.3 with a precise inference protocol for unseen documents and separate transductive vs inductive evaluation.
- Add ablations that quantify the contribution of each component: transformations, consensus aggregation, and iterative InfoNCE+GMM training.
- Temper or support the scalability claims with complexity analysis and, ideally, a more scalable approximation to the spectral consensus step.
- Tighten the methodology section so the covariance assumptions and actual optimization variables are stated consistently throughout.

Overall, the paper has a plausible and potentially useful clustering recipe, but at present its strongest theoretical and practical claims are overstated relative to what is rigorously established and empirically isolated.

---

## dPAcHrG4rl

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
This paper combines an information-theoretic framing of single-pass LLM reasoning for multi-hop QA with a practical multi-call framework, InfoQA. The theory derives a Fano-style upper bound and uses a parametric demand model to argue for an “Accuracy Cliff” as hop count and context length increase; the empirical side introduces a controlled synthetic benchmark and shows that InfoQA, via decomposition plus query contraction/pruning, is substantially more robust than single-pass prompting baselines under those controlled conditions.

## Strengths
- **The paper turns an intuitive systems limitation into a concrete formal lens.** Theorem 1 is a valid combination of conditional Fano and an output-entropy bound, and the paper uses it to articulate a clear capacity-vs-demand perspective on why single-pass reasoning should fail as problem complexity grows. Even if some downstream modeling choices are debatable, the core formal statement is legitimate and gives the paper a sharper conceptual backbone than typical prompting papers.
- **The synthetic benchmark is unusually well-matched to the paper’s stated goal of controlled stress-testing.** The authors explicitly vary hop count (1–4) and context length (0.5k–10k), place evidence out of logical order, and populate contexts with semantically similar distractors rather than only irrelevant filler. This design is specifically useful for isolating how depth and noise interact, which is exactly the paper’s target phenomenon.
- **InfoQA shows strong, consistent gains in the regime the paper cares about most: deep, noisy, long-context MHQA.** On Qwen3-14B, the overall 2–4 hop average improves from 0.75 (best single-pass baseline, S-C) to 0.86, and the advantages are especially large at 3–4 hops and long contexts. The 4-hop numbers are particularly notable: InfoQA averages 0.80 vs 0.61 for S-C and 0.57 for CoT.
- **The ablations support the core design intuition rather than just the full pipeline.** Removing decomposition (“w/o D.”) hurts sharply, especially as contexts lengthen, and removing pruning also degrades performance, indicating that the gains do not come from a trivial multi-call wrapper alone. This is useful evidence for the specific roles of decomposition and contraction.
- **The paper identifies a practically relevant failure mode of the proposed framework.** The error analysis does not simply claim success; it points to semantic drift in iterative query contraction as the remaining dominant error source. That is a concrete and credible diagnosis that could guide follow-up work.

## Weaknesses
###: Fatal
- **The claimed empirical validation of the theoretical bound is much weaker than the paper presents, because the key quantities in the “predicted curves” are not independently measured but fit post hoc to the same benchmark results.**  
  Section 5.2 states that the authors “fit the parameters θ = (β0, α, γ, C) of our plug-in accuracy bound (Eq. 7) to empirical F1 scores” by minimizing MAE, and Appendix A.5 confirms a grid search over all four parameters. This means the paper does not independently estimate the theorem’s information demand \( \beta = H(A\mid Q,C) \) or capacity \( C = H(Y) \), then test whether the theorem predicts the observed collapse. Instead, it fits an effective demand/capacity model to the observed performance and then reports alignment. As a result, the experiments support the usefulness of a descriptive scaling law, but they do **not** constitute strong validation that the information-theoretic bound itself governs model behavior in the claimed predictive sense. This substantially weakens the paper’s central theory-to-experiment claim.

- **The bridge from the formal theorem to the MHQA demand model is conceptually underspecified and, as written, conflates entropy-based uncertainty with an empirically fitted notion of reasoning difficulty.**  
  The theorem defines information demand as \( \beta \triangleq H(A\mid Q,C) \). But in Section 3.1 the paper introduces a parametric model
  \[
  \beta(h,L)=\beta_0+\alpha L\gamma^{h-1},
  \]
  motivated by baseline complexity, context burden, and hop amplification. This is plausible as a *difficulty proxy*, but the paper continues to speak of it as the same \( \beta \) from the theorem. That identification is not justified. In fact, the paper’s own benchmark construction makes the answer a single entity drawn from a controlled synthetic space, so the entropy of the answer conditioned on query/context is not obviously the quantity that should grow super-linearly with hop count in the way Eq. 6 assumes. The issue is not that Eq. 6 is useless—it may be a reasonable empirical ansatz—but that the paper overstates it as a direct instantiation of the theorem’s information demand, when it is really an effective fitted surrogate. This mismatch undermines the technical soundness of the main explanatory story.

### Major:
- **The “capacity” quantity used in experiments is not tied back convincingly to the formal \(H(Y)\) introduced in the theory.**  
  In Section 2 and Appendix A.3, capacity is defined as \(C = H(Y\mid Q,C)\) (or upper bounded via output vocabulary/length). But in Section 5.2/A.5, \(C\) becomes a fitted scalar chosen to best match F1 curves. The paper never measures or even approximates output entropy from model generations, nor does it explain why the fitted \(C\) should be interpreted as an entropy quantity rather than just an effective nuisance parameter. This weakens statements such as certain prompting methods “increase capacity \(C\)” or “reduce hop inflation \(\gamma\),” because these are not established as identifiable, physically meaningful properties of the model; they are artifacts of a low-dimensional fit.
- **External validity is limited because all substantive claims are validated on a single synthetic benchmark.**  
  The benchmark is well-designed for controlled diagnosis, but the paper’s rhetoric is broader: it claims to expose a fundamental inadequacy of the single-pass paradigm for MHQA. Without any evaluation on standard real-world MHQA benchmarks, it remains unclear how much of the reported cliff behavior and InfoQA’s advantage depends on the benchmark’s templatic structure, evidence placement strategy, and synthetic distractor distribution. For a paper making both theoretical and practical claims, this omission matters.
- **The practical cost of InfoQA is under-analyzed relative to the reported benefit.**  
  The framework is explicitly multi-call, and the paper positions it as a proof-of-concept to transcend single-pass limitations. However, there is essentially no quantitative accounting of inference cost: number of calls, token consumption, latency, or cost-performance tradeoff. Since the main empirical gain comes from decomposing one hard call into multiple easier ones, the absence of a compute/latency analysis leaves the practical significance incomplete.

### Minor
- **Some baseline framing is potentially confusing, especially for methods whose original use often involves iterative interaction.**  
  The paper says all baselines were implemented as “zero-shot, single-pass methods,” including ReAct and Self-Ask. This is not necessarily invalid for the paper’s goal—indeed, the whole point is to compare single-pass paradigms—but the exact single-pass instantiation matters for interpretation. The paper should describe more concretely how these methods were adapted into a strictly single-generation setting and what functionality was intentionally removed.
- **The ablation analysis does not fully isolate all three claimed ingredients of InfoQA.**  
  Table 2 includes ablations for removing decomposition and removing pruning, but the “dependency-explicit workflow” is not isolated as cleanly as the other two. Given that the method is presented as a three-part design, the evidence for the third component is more indirect.
- **The paper’s own error analysis points to semantic drift during contraction, but this is not examined deeply enough.**  
  This seems to be the dominant residual failure mode, yet the paper provides no qualitative examples, per-hop failure statistics, or contraction-sensitivity analysis to verify the diagnosis.

### Trivial
- None.

## Nice-to-Haves
- Evaluate InfoQA on at least one established real-world MHQA benchmark to test whether the observed gains transfer beyond the controlled synthetic setting.
- Report token usage, number of model calls, and latency/cost tradeoffs for InfoQA versus single-pass baselines.
- Add uncertainty or sensitivity analysis for the fitted parameters \((\alpha,\gamma,\beta_0,C)\), since they are estimated from only 24 \((h,L)\) conditions per model.
- Show per-hop success rates or case studies of successful and failed query contraction, especially for the semantic-drift failure mode the paper itself highlights.
- Clarify more explicitly that Eq. 6 is an empirical effective-demand model rather than a direct measurement of \(H(A\mid Q,C)\), unless the authors can independently justify that identification.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The theorem itself is mathematically wrong.”** Removed because this is not supported by the paper text. Theorem 1 is a straightforward and valid consequence of conditional Fano plus \(I(A;Y\mid Q,C)\le H(Y\mid Q,C)\), and the appendix provides the standard derivation.
- **Complaints that certain cited models/datasets/references may not exist or be unavailable.** Removed per instruction.
- **Pure style/presentation praise such as “well-written” or “important topic.”** Removed as too generic.
- **Requests for generic reproducibility minutiae.** Removed because the paper already includes a reproducibility statement, code/data release claims, and sufficient implementation detail for the level of contribution.
- **Baseline unfairness framed as asymmetry against baselines.** Softened rather than kept as a core criticism. The paper intentionally evaluates many methods in a strict single-pass regime because that is its target setting; the main issue is lack of clarity in adaptation, not unfairness per se.
- **Strong claim that Eq. 6 is literally impossible because \(H(A\mid Q,C)\) must be small and constant.** Softened. The paper does not specify a fixed tiny answer set size in the main theorem, and the exact entropy of the synthetic answer distribution is not computed. The real issue is not formal impossibility but the unjustified identification of a fitted difficulty law with the theorem’s entropy quantity.

## Novel Insights
The most important synthesis is that this paper is stronger as a **capacity-aware empirical systems paper with a provocative theoretical framing** than as a clean theory-validation paper. The controlled benchmark and the InfoQA gains do support the practical claim that decomposition plus contraction helps avoid prompt-level overload in long, noisy multi-hop settings. However, the paper currently overclaims by presenting a fitted effective-demand/capacity model as if it were a direct empirical confirmation of a Shannon-style theorem. Reframing the contribution as: “a principled information-theoretic motivation plus an empirically validated effective scaling law and a strong proof-of-concept framework” would make the work more honest and substantially more convincing.

## Suggestions
- **Separate the formal theorem from the empirical surrogate model more explicitly.** State clearly that Eq. 6/7 is an effective phenomenological model inspired by the theorem, not a direct measurement of \(H(A\mid Q,C)\) and \(H(Y)\).
- **Reduce the strength of the validation claim in Section 5.2.** Replace “validate the Fano-style upper bound” with language closer to “test whether a theorem-inspired effective scaling law matches observed performance.”
- **Independently estimate at least one theoretical quantity.** For example, approximate output entropy or mutual information proxies from sampled generations, or calibrate an effective capacity with a separate probing task rather than fitting everything on the evaluation set.
- **Add one real-world benchmark.** Even a smaller-scale experiment on HotpotQA/MuSiQue-style data would materially strengthen the significance claim.
- **Report inference overhead.** Include average calls per question, total tokens processed/generated, and latency relative to the best single-pass baseline.
- **Deepen the analysis of semantic drift.** Provide failure traces showing what constraints are lost during contraction and whether those losses concentrate at specific hops or context lengths.

---

## nHrYBGujps

- GT: Accept (Oral) (avg 7.5)
- Predicted: N/A (5.9/10)
- Match: N/A

### Final Review

## Summary
This paper introduces **BIRD-INTERACT**, a new benchmark for **dynamic, multi-turn text-to-SQL evaluation** that goes beyond static conversational transcripts and SELECT-only workloads. The benchmark combines executable databases, hierarchical knowledge bases, a function-driven user simulator, and two evaluation modes—protocol-guided (**c-Interact**) and agentic (**a-Interact**)—to test ambiguity resolution, debugging, state tracking, and follow-up reasoning across the full CRUD spectrum.

## Strengths
- **Substantially expands the scope of text-to-SQL evaluation beyond the prevailing static setup.** The benchmark does not just add dialogue turns; it explicitly combines ambiguity resolution, execution feedback, follow-up tasks, and database state changes. This is evidenced by the benchmark construction: each task has an ambiguous first sub-task, a stateful follow-up, executable test cases, and support for DML/DDL as well as BI-style queries.
- **The function-driven simulator is a concrete technical contribution, not just infrastructure glue.** The two-stage design—first mapping a model question into `AMB`, `LOC`, or `UNA`, then generating constrained responses—directly targets a known failure mode of LLM user simulators: leakage and unfairness. The paper does provide supporting evidence here: on UserSim-Guard, the proposed simulator reduces failures on unanswerable questions dramatically relative to single-pass baselines, and the simulator-to-human alignment analysis reports notably higher correlation than the baseline simulator.
- **The benchmark exposes an important capability gap that static text-to-SQL benchmarks can hide.** The headline results are specific and informative: even strong frontier models remain far from robust on these tasks, and there is a substantial drop from priority subtasks to follow-up subtasks, consistent with the intended challenge of maintaining context and handling evolving user intent.
- **The two evaluation settings are meaningfully differentiated.** `c-Interact` probes structured conversational clarification under a fixed protocol, while `a-Interact` tests tool use and planning under explicit budget constraints. This creates a useful decomposition of assistant-style versus agentic behavior rather than a single monolithic score.
- **Some of the analysis goes beyond leaderboard reporting.** In particular, the interaction test-time scaling experiment is a genuinely useful observation: increasing interaction opportunities improves performance, supporting the claim that interaction budget matters and that these tasks are not purely impossible for current models.

## Weaknesses

### Major:
- **The core reported task success rates remain partially confounded by the benchmark’s simulator/parsing layer, and the paper does not sufficiently quantify this effect.**  
  The evaluation depends critically on the two-stage user simulator, whose first stage maps free-form clarification questions into discrete actions (`AMB`, `LOC`, `UNA`). If this parser rejects a semantically valid clarification or routes it incorrectly, the evaluated model can fail despite asking a reasonable question. The paper does validate simulator robustness in Section 6, but that validation is indirect relative to the main benchmark outcomes: it measures classification-style performance on UserSim-Guard and correlation with humans on 100 sampled tasks, rather than how often benchmark failures are attributable to simulator misclassification in end-to-end runs. Since the paper’s central claims lean heavily on very low absolute success rates, a more direct attribution analysis is needed to separate **model interaction failure** from **simulator interpretation failure**.

- **Some of the causal claims in the analysis are stronger than the evidence warrants.**  
  The clearest case is the **memory grafting** analysis. The paper states that supplying GPT-5 with ambiguity-resolution histories from stronger communicators shows that “communication effectiveness often determines success” and suggests GPT-5 has a communication deficiency. The experiment does support the narrower claim that **better acquired interaction history materially helps downstream SQL success**. But it does **not by itself isolate “communication skill”** from several other possibilities, including better ambiguity coverage, better state acquisition, or better task decomposition being handed to the model. In other words, the evidence supports that interaction history quality matters; it does not cleanly identify which behavioral faculty is lacking.

- **The action-distribution interpretation in `a-Interact` over-attributes behavior to model bias without adequately disentangling the benchmark’s imposed cost structure.**  
  The paper notes that models overuse `submit` and `ask` relative to environment exploration and interprets this as evidence of trial-and-error or pretraining bias. But the benchmark itself imposes a nontrivial action economy (`execute` cost 1, `ask` cost 2, `submit` cost 3, many retrieval actions cost 0.5–1), and the total budget is formulaically tied to annotated ambiguities. This means action frequencies are shaped not just by model tendencies but by the benchmark’s own incentives. The descriptive analysis is still useful, but the stronger interpretation—especially claims about intrinsic bias or architectural tendency—needs either a sensitivity analysis over cost schemes or a more cautious framing.

- **Empirical support for comparative claims is limited by single-run evaluation.**  
  The paper explicitly states: “conducting single runs due to cost,” with deterministic decoding (`temperature=0`, `top_p=1`). Deterministic decoding reduces one source of randomness, but the benchmark is still an interactive, long-horizon setup involving multiple tools, conditional branching, and an LLM-driven simulator. Small perturbations can alter trajectories and binary end-task outcomes. This does not invalidate the broad conclusion that the benchmark is hard, but it weakens fine-grained comparative claims between models and interaction modes. At minimum, variance estimates on a representative subset would strengthen confidence in model ranking and in several interpretive claims.

### Minor
- **The realism claim is directionally right but somewhat overstated.**  
  The benchmark is clearly more realistic than static transcript-based text-to-SQL evaluation. However, the simulator remains constrained by annotated ambiguities and GT-grounded clarification sources. The paper itself acknowledges a pragmatic choice in Appendix D:  
  > “to avoid cases where certain ambiguities lack explicit annotations, the simulator is additionally provided with the reference SQL … This pragmatic design choice enhances evaluation reliability.”  
  This is reasonable for benchmark control, but it also means the interaction setting is still more structured and cleaner than real user behavior. The benchmark is best described as a **controlled approximation to interactive database work**, not as full restoration of real-world interaction realism.

- **There is no human upper-bound baseline on task completion.**  
  The paper includes human-related validation for simulator alignment and dataset quality, but not a direct human-expert performance baseline on the benchmark tasks themselves. Without that, the interpretation of “how hard” the benchmark is lacks an upper anchor: the low model scores show difficulty, but not how close or far these systems are from competent expert performance under the same protocol and budgets.

- **Error analysis is too coarse to fully support the paper’s strongest bottleneck claim.**  
  The paper reports that “over 80% of the errors were caused by incomplete ambiguity resolution,” but this bucket is broad. It would be more convincing to separate: failure to detect ambiguity, poor clarification wording, simulator rejection, incorrect use of retrieved clarification, SQL synthesis failure after successful clarification, and follow-up state-tracking failure. As written, the analysis suggests the likely bottleneck but does not localize it sharply enough.

### Trivial
- **The normalized reward weighting (70/30) is only lightly motivated in the main narrative.**  
  Since some interpretation relies on divergences between online success rate and offline reward, a brief sensitivity analysis or stronger justification would improve confidence that conclusions are not overly metric-dependent.

## Nice-to-Haves
- Add a **human expert baseline** on a representative subset under the same budgets and interfaces.
- Provide an **oracle clarification experiment** where gold clarifications are supplied upfront, to cleanly separate interaction failure from SQL generation failure.
- Include a **cost-sensitivity analysis** for `a-Interact`, varying action prices or budget formulas to test whether action-distribution conclusions are robust.
- Expand the error taxonomy to separate ambiguity detection, clarification formulation, simulator routing, SQL generation, and follow-up state tracking.
- Add side-by-side traces of **strong vs. weak interaction trajectories** on the same task, annotated at the point of divergence.
- Report subset-level variance or repeated-run stability for a representative portion of the benchmark.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Concerns about release status / existence / verifiability of cited models, tools, or benchmarks.** The paper cites these resources; per review policy, such criticisms are not valid.
- **Pure reproducibility complaints about missing prompts or configuration details.** The paper already states that prompts are provided in Appendix R and lists key decoding settings in Appendix I.3.
- **Generic criticism that the benchmark is “synthetic” because ambiguities are injected.** This is partly true in the literal sense, but the paper is explicit that it converts single-turn tasks into interactive ones via controlled ambiguity injection, and it backs this with annotation protocols, quality control, and human quality checks. The valid concern is not that it is synthetic per se, but that ecological validity remains only partially established.
- **Complaints about missing related work in adjacent dynamic-agent domains.** Without external verification, these are not reliable grounds for criticism here.
- **Resource-intensity as a core weakness.** The paper’s API-based evaluation is expensive, but this is common for modern benchmark studies of frontier models and is not itself a substantive flaw in the benchmark design.

## Novel Insights
The most interesting synthesis across the reviews is that the paper’s strongest contribution is likely **the benchmark and simulator design itself**, whereas some of its strongest behavioral conclusions are still one step ahead of the evidence. The results convincingly show that dynamic text-to-SQL under ambiguity, debugging, and state dependence is much harder than static text-to-SQL. What is less fully established is *why* models fail: the current experiments suggest interaction bottlenecks, but do not yet cleanly disentangle failures of ambiguity detection, question formulation, simulator routing, and downstream SQL reasoning. In that sense, the paper has already built an impactful stress test, but its explanatory story would benefit from tighter causal isolation.

## Suggestions
- Add an **oracle-clarification ablation** and compare it directly to standard interaction to quantify how much failure comes from acquiring the right information versus using it.
- Audit a sample of failed episodes for **simulator-routing errors** (`AMB/LOC/UNA`) and report how often valid clarification attempts are rejected or misclassified.
- Reframe the **memory grafting** conclusion more conservatively: it shows the value of high-quality interaction history, not a clean diagnosis of “communication skill.”
- Temper or support the **action-bias** claims in `a-Interact` with a sensitivity study over action costs and budgets.
- Include at least a **small repeated-run or subset variance analysis** to support comparative statements between models.
- Provide a **human task-performance baseline** on a subset to contextualize benchmark difficulty.

---

## Z14gV0qz5r

- GT: Reject (avg 0.7)
- Predicted: N/A (4.2/10)
- Match: N/A

### Final Review

## Summary
This paper proposes HNSW-LAVQ, an HNSW variant that replaces standard per-dimension min/max scalar quantization with percentile-based clipping, then uses an int8 AVX2 search kernel to reduce memory traffic and accelerate search. The central empirical claim is that on SIFT1M, this combination preserves much more recall than naive min-max quantization while matching the memory footprint of int8 scalar-quantized indices and improving throughput over a float32 HNSW baseline.

## Strengths
- The paper identifies and cleanly demonstrates a real failure mode of naive scalar quantization: outlier-driven range stretching. The ablation in Section 5.5 is the strongest evidence in the paper: naive min-max quantization gives Recall@1 = 84.3%, while clipped quantization gives 97.2%, a very large gap that directly supports the paper’s main algorithmic idea.
- The contribution is practically targeted rather than vague: the method is simple to integrate because, as stated in Section 3, it “modifies the HNSW storage layer and the distance kernel” and “do[es] not alter the graph construction logic itself,” which lowers adoption friction for existing HNSW pipelines.
- The paper usefully combines an algorithmic tweak and systems implementation details. In particular, the separation between graph topology and vector storage (SoA layout), 32-byte alignment, and integer SIMD kernel are concrete design choices that plausibly improve memory behavior in graph traversal workloads.
- The paper does include at least one meaningful ablation beyond headline comparisons: it isolates clipping versus naive min-max quantization rather than only comparing against end-to-end baselines. That ablation is the clearest evidence for what is actually novel here.
- The paper is fairly explicit about one real limitation—static clipping bounds under distribution shift—rather than claiming universal applicability. Section 6 appropriately acknowledges that bounds may become stale in streaming settings.

## Weaknesses

### Fatal
- The SIMD kernel description appears technically incorrect or at least seriously under-specified for the claimed L2 distance computation, and this directly affects the credibility of the reported speed numbers. Section 4 says the kernel uses `_mm256_subs_epu8` “for saturated subtraction, followed by `_mm256_maddubs_epi16` for squaring,” and Algorithm 1 calls `AVX2 L2 Dist(...)`. As written, this is not a valid explanation of squared Euclidean distance computation between quantized vectors. `_mm256_maddubs_epi16` does not simply “square” byte differences, and saturated unsigned subtraction is also not enough to recover signed differences. The paper may have a correct implementation, but the manuscript does not present it. Because the claimed throughput gains rely heavily on this kernel, this is not a minor documentation issue; the core systems result is not technically substantiated by the description provided.

### Major:
- The paper conflates the sources of improvement: the main throughput gain is not evidence for the percentile-clipping idea itself. The paper presents HNSW-LAVQ as a joint method, but the 4.4× speedup is largely attributable to switching from float32 vector storage and arithmetic to int8 plus AVX2, whereas the percentile clipping primarily affects accuracy under quantization. The paper does provide an ablation for clipping versus naive min-max on recall, but it does not isolate how much of the performance gain comes from (i) int8 storage, (ii) SIMD kernel design, (iii) SoA layout, and (iv) clipping. This weakens the causal interpretation of the headline result.
- The empirical scope is too narrow for the paper’s motivation and claims of broad practical significance. All experiments are on a single dataset, SIFT1M, with 1M vectors of dimension 128. Yet the introduction and conclusion motivate the work using modern “RAG,” “typical OpenAI embeddings,” and “billion-scale” deployment scenarios. A single legacy 128D benchmark is enough to show the idea can work, but not enough to support the paper’s stronger claims about modern high-dimensional embedding workloads or billion-scale practicality.
- There is a clear quantitative inconsistency in the memory claims. The abstract states that LAVQ “cuts memory usage by 3.8×,” while Table 1 reports total RAM dropping from 576 MB to 192 MB, which is exactly 3.0×. If the authors intended to refer to vector storage alone, that would be 4× (512 MB to 128 MB), not 3.8×. This is a basic headline metric, so the inconsistency materially hurts trust in the presentation.
- The choice of clipping percentiles (1st/99th) is asserted rather than justified. Since this parameter is central to the method, the paper should show whether the gains are robust to different clipping levels or whether SIFT1M happens to favor this exact choice. Without such a sensitivity analysis, it is hard to know whether the method is broadly stable or tuned to this benchmark.
- The paper does not sufficiently analyze what is being sacrificed by clipping. The text claims that clipping “accepts a small amount of error at the tails to significantly lower the Mean Squared Error (MSE) for the vast majority of points,” but no empirical evidence is shown for the actual fraction of values clipped, which dimensions are affected, or how clipping changes quantization error. This matters because the central premise of the paper is not merely that clipping helps, but that it helps for the right reason—by discarding only non-discriminative tails.

### Minor
- The tabled evaluation reports only Recall@1. For practical retrieval systems, especially those motivated by RAG and semantic search, Recall@10 or Recall@100 would be more informative. Recall@1 alone gives a narrow view of quality.
- The paper’s “complexity analysis” in Section 3.3 is not especially convincing. The formula is essentially a rough cycle model and does not capture the memory-bound nature of ANN traversal that the paper itself emphasizes. This is not fatal, but it reads more like intuition than rigorous analysis.
- The discussion of static quantization bounds in Section 6 is directionally honest but incomplete. In practice, adapting percentiles under drift would require re-quantization of stored vectors, which is a more substantial operational issue than the current limitations section suggests.
- The paper mentions “realistic cache pressure” as a motivation for using SIFT1M over smaller datasets, which is reasonable, but this does not by itself validate the stronger claims about scaling to production-scale settings.

### Trivial
- None.

## Nice-to-Haves
- Add a controlled decomposition of speedup: float32 HNSW vs int8 HNSW with naive min-max, vs int8 HNSW with clipping, vs int8 HNSW with/without SoA layout and custom kernel. This would sharply separate algorithmic from systems contributions.
- Evaluate on at least one modern higher-dimensional embedding dataset and report top-k metrics beyond Recall@1.
- Sweep clipping percentiles (e.g., 95/5, 99/1, 99.5/0.5) to establish robustness.
- Report clipping statistics and quantization error distributions per dimension.
- Include hardware counter evidence (e.g., cache misses, achieved bandwidth) to substantiate the memory-bandwidth explanation.
- Clarify whether graph construction uses original float vectors or quantized vectors for edge decisions, and analyze whether quantization affects graph topology if applicable.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Complaints about baseline release/verification status or exact software provenance.** The paper cites hnswlib and FAISS configurations; questioning whether they can be independently verified because version strings, compiler flags, or release status are not exhaustively listed is not an appropriate core criticism here.
- **Generic request to compare against many more methods (e.g., ScaNN, DiskANN, PQ variants).** Additional baselines could strengthen the paper, but the absence of every neighboring ANN system is not by itself a decisive flaw, especially since the paper’s focus is a scalar-quantized HNSW variant rather than a universal ANN bakeoff.
- **Fairness criticism based solely on asymmetric tuning details favoring the baseline.** The paper uses hnswlib and FAISS as baselines, and while parameter clarity could be improved, there is not enough evidence in the manuscript itself to conclude that the comparisons are unfair in a way that invalidates the results.
- **Pure reproducibility nitpicks about missing minor hyperparameters.** The appendix already gives `efconstruction`, a range for `M`, and a range for `efsearch`; the more serious issue is not missing trivia but that the exact settings for headline table entries should be stated.
- **Overstated claim that SIFT1M “fits comfortably in modern CPU cache hierarchies.”** The paper’s point that SIFT1M is more realistic than tiny cache-fitting datasets is reasonable; the weakness is limited benchmark diversity, not that SIFT1M is itself a toy benchmark.

## Novel Insights
The strongest synthesis across the reviews is that this paper really contains two contributions of different evidential strength: (1) a simple but effective quantization idea—percentile clipping—that is actually well supported by the min-max ablation, and (2) a systems acceleration story whose current presentation is much weaker because the manuscript does not convincingly explain the arithmetic of the AVX2 L2 kernel and does not disentangle kernel/layout effects from the quantization idea. In other words, the paper seems more convincing as “clipped scalar quantization substantially improves int8 HNSW accuracy over naive min-max” than as a fully substantiated end-to-end systems paper claiming a principled 4.4× speedup.

## Suggestions
- Fix the kernel description first. If the implementation computes exact or approximate L2 in a nontrivial way, spell it out mathematically and at the intrinsic level; otherwise the systems claims remain difficult to trust.
- Separate the contribution claims: make clear that clipping is the accuracy contribution, while int8+SIMD+layout are the systems optimizations, and provide ablations that isolate each.
- Correct the memory numbers throughout the paper and reconcile the abstract with Table 1.
- Add at least one modern higher-dimensional embedding benchmark aligned with the paper’s stated motivation.
- Add a percentile sensitivity study and report how many values are clipped per dimension.
- Report top-k retrieval metrics in addition to Recall@1.
- Expand the limitations section to explicitly discuss the operational cost of updating stale clipping bounds in dynamic indices.

---

## 1fALdE637I

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (7.4/10)
- Match: N/A

### Final Review

## Summary
This paper proposes **Constrained Diffusion Policy Optimization (CDPO)**, a framework that casts several diffusion-based offline RL actor objectives as constrained optimization with a flexible anchor policy, and introduces **Two-fold improved Diffusion Policy (TDP)**, which first anchors to the reverse-KL-improved policy \(\pi_\eta^*\) and then further refines with a Q-driven term. A key technical ingredient is a **noise-free estimation** scheme for the diffusion guidance term, together with theoretical results on policy enhancement / approximate improvement and empirical gains on D4RL locomotion, Antmaze, and Kitchen.

## Strengths
- **The noise-free estimator is a genuinely substantive technical contribution.** The paper does more than heuristically replace noisy guidance: it derives a surrogate preserving the minimizer of the ideal loss via an MMSE argument (Sec. 3.2 / Appendix D.7), and identifies the approximation bias as an explicit negative KL term in Eq. (17). This is a much stronger justification than the usual “works better in practice” style argument common in this area.
- **The “two-fold” design is conceptually clean and well isolated empirically.** TDP combines (i) a stronger anchor policy than \(\pi_\beta\), namely \(\pi_\eta^*\), with (ii) an additional Q-loss refinement. The 2D bandit study and the TDP vs. explicit vs. implicit discussion directly target this decomposition, and the flow-policy analogue (TFP) is useful evidence that the gain is not purely a diffusion-architecture artifact.
- **The paper provides a meaningful unification of recent diffusion-policy actor objectives.** In Sec. 3.1, DQL-like methods, DAC-like methods, and BC-like training are all recovered as special cases of the same generalized constrained objective by varying \(\pi_0\) and \(\zeta_t\). This is not merely naming a family: it gives a lens that helps explain why explicit and implicit methods fail differently and motivates the proposed combination.
- **Empirical results are strong on challenging domains where offline RL often fails.** The strongest evidence is on Antmaze and Kitchen: Table 2 shows large average gains over the included baselines, especially on sparse-reward ultra Antmaze tasks, and Table 3 reports near-perfect Kitchen scores. Even if some gains on saturated locomotion tasks are smaller or mixed per task, the overall pattern suggests the method is particularly effective on hard, long-horizon settings.
- **The paper includes unusually extensive ablations and implementation detail for this literature.** Beyond the main tables, the appendix studies noisy vs. noise-free estimation, adaptive vs. fixed \(\eta\), \(\lambda\), \(N_q\), update interval, diffusion steps, noise schedules, and critic variants. That breadth usefully clarifies which parts of TDP matter most.

## Weaknesses
### Fatal
None.

### Major:
- **The theoretical guarantees only partially connect to the algorithm actually evaluated.** The theory in Sec. 4 is stated for idealized CDPO updates with exact constrained optimization and fixed divergence constraints, while the practical algorithm introduces several departures: learned critics, stochastic nonconvex optimization, Q-ensemble LCB targets, adaptive \(\eta\), replacing \(\zeta_t\) by a fixed \(\zeta\) / \(\lambda\), delayed actor updates, and one-step / half-batch approximations. The paper does acknowledge some of this—for example, “for theoretical analysis, we assume a fixed target divergence value \(\epsilon_b\) across timesteps... In practice... we therefore fix \(\zeta_t\) to a constant \(\zeta\)” (Sec. 3.4)—but the main narrative still leans heavily on theorems as support for TDP. As written, the guarantees justify the motivating objective more than the concrete training recipe.
- **The in-distribution theorem is mathematically interesting but practically weak as support for the paper’s strongest OOD claims.** Theorem 4.5 proves an existence-style bound via diffusion divergence and a \((\delta,\epsilon)\)-in-distribution definition, but the constants degrade with action dimension through the box-space measure term in Appendix D.6. For the continuous-control tasks used here, that makes the result more of a qualitative existence statement than a practically informative guarantee. Since the paper repeatedly claims TDP “reliably maintain[s] in-distribution behavior,” stronger empirical evidence of actual support overlap / OOD frequency would be more convincing than the current theorem alone.
- **The practical importance of the approximation bias is not fully analyzed beyond limited ablations.** The paper’s own derivation shows the surrogate incurs bias equal to a negative KL term (Eq. 17), and argues this conservativeness is beneficial. That is plausible and potentially insightful, but the paper does not quantify when this bias is small versus when it materially affects learning, especially on high-optimality datasets where excessive conservatism could suppress useful improvements. The current evidence mainly shows the estimator outperforms a noisy alternative, not when the approximation itself helps or hurts.

### Minor
- **Comparisons use heterogeneous evaluation sources and protocols across baselines, which weakens the sharpness of the SOTA claim.** Appendix E.4 states that many baseline numbers are taken from original papers, some Antmaze numbers are reproduced by the authors, and evaluation protocols are not fully uniform—e.g., “DQL and DiffCPS were evaluated using online model selection (OMS)... For all other algorithms, we reported the final evaluation scores.” This does not invalidate the results, but it means the aggregate tables are not as controlled as a single unified re-run would be.
- **Some central empirical claims about in-distribution behavior are shown vividly only in the 2D bandit, not in the real benchmarks.** The 2D visualization is helpful, but for D4RL the paper does not directly measure support overlap, behavior-policy likelihood, or OOD action rate during training/inference. Given how central the OOD-control story is, at least one quantitative support-related analysis on the actual benchmark tasks would strengthen the case substantially.
- **The main text could better explain the practical interpretation of Theorem 4.1.** The theorem gives true return improvement at the first iteration and then expected Q-value improvement under \(Q^{\pi^{(t)}}\) for later iterations. That is still useful, but it is weaker than monotonic policy improvement over the full iterative algorithm. The paper would benefit from being more explicit in the main text about what this does and does not imply once critics are approximate and updated online.

### Trivial
- **The description of Eq. (15) as a “first-order Taylor expansion” is imprecise.** What is being approximated is effectively a log-sum-exp / expectation structure; the terminology should be tightened to avoid overstating the exactness of the approximation argument.
- **CDPO’s novelty should be framed more modestly.** The framework is useful and clarifying, but the general form \(\min_\pi D(\pi_0\|\pi)-\zeta L_Q(\pi)\) with a flexible anchor is closer to a unifying reinterpretation than a wholly new optimization paradigm. This is still a strength if presented that way.

## Nice-to-Haves
- Add a direct empirical measure of OOD behavior on D4RL tasks, such as behavior-policy likelihood, dataset support overlap, or fraction of sampled actions falling in low-density regions.
- Provide at least one controlled baseline suite under a fully unified protocol (same codebase, same tuning budget, same selection rule) for the most relevant comparators.
- Expand the analysis of when the negative-KL bias in Eq. (17) is beneficial versus overly conservative, especially on medium-expert / high-quality datasets.
- Clarify in the main text the exact scope of the theory: what is proven for ideal CDPO, what carries over only heuristically to TDP, and what remains empirical.
- Include a concise compute/performance trade-off discussion in the main paper, since Appendix F.2 shows nontrivial overhead.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Decision Transformer / other cited baselines have implausibly low numbers or may be underreported relative to external literature.”** Removed because it depends on external knowledge not verifiable from the submission. What is fair to keep is the internal point that baseline numbers come from mixed sources and protocols.
- **“Missing IDQL in the main tables.”** Removed under the rule against claiming missing related works/baselines that cannot be externally validated here.
- **Reproducibility complaints about undisclosed hyperparameters / implementation details.** Removed because the paper in fact provides extensive hyperparameter tables, pseudocode, and appendix details.
- **Complaints that the authors reproduced some cited baselines so the baselines may not exist / be verifiable.** Removed per instruction: cited methods are assumed to exist.
- **Generic praise such as ‘the paper is well written’ or ‘the experiments are extensive’.** Removed as too generic unless tied to something specific.

## Novel Insights
The most interesting synthesis across the paper and reviews is that the real contribution is not just “better diffusion offline RL,” but a more specific recipe: **use a stronger in-distribution anchor than behavior cloning alone, then let Q-learning refine from that anchor rather than from the raw behavior policy.** The flow-policy results reinforce that this two-stage principle may be more fundamental than diffusion itself. At the same time, the paper’s theory most cleanly supports this *objective-level principle*, whereas the practical success of TDP depends on additional engineering choices; making that distinction explicit would sharpen the paper considerably.

## Suggestions
- Separate the claims more clearly into: **(a)** theory for ideal CDPO, **(b)** principled motivation for TDP, and **(c)** empirical validation of the practical algorithm.
- Add one real-benchmark OOD analysis, not just 2D bandit visualizations.
- Strengthen the empirical section with a more uniform comparison protocol for a smaller set of key baselines.
- Analyze Eq. (17)’s negative-KL bias quantitatively across dataset qualities and possibly across action dimensions.
- Rephrase the CDPO framing to emphasize unification and reinterpretation rather than implying a fundamentally new constrained optimization formalism.
- Tighten the wording around Eq. (15) and related approximation language.

---

## RT5SlprCmc

- GT: Reject (avg 4.5)
- Predicted: N/A (6.6/10)
- Match: N/A

### Final Review

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

---

## VjGU55hEwV

- GT: Reject (avg 2.5)
- Predicted: N/A (4.3/10)
- Match: N/A

### Final Review

## Summary
This paper proposes RLIE, a hybrid framework for learning natural-language rules with LLMs and then combining them with an elastic-net logistic regression model. The method has four stages—rule generation, logistic weighting/selection, hard-example-driven iterative refinement, and evaluation of downstream inference strategies—and is tested on six HypoBench text classification tasks. The main empirical message is that using the learned weighted rule set directly via logistic regression works better than feeding rules and weights back into an LLM.

## Strengths
- **Clear hybrid decomposition with an explicit local/global split.** The paper does something more specific than “use an LLM plus a classifier”: the LLM is used for *local semantic operations* (generate rules; judge rule applicability with ternary outputs), while the logistic model handles *global aggregation, sparsity, and calibration*. This division is consistently formalized in Sections 2–3 and is a meaningful design choice for natural-language rule learning.
- **The evaluation is structured to probe how rules should actually be used, not just whether they can be generated.** The E1–E4 hierarchy in Section 3.4 is a useful experimental design: direct linear inference vs. LLM with rules only, rules+weights, and rules+weights+linear prediction. This is more informative than a single end-to-end accuracy number because it separates rule quality from the downstream inference mechanism.
- **The iterative refinement loop is concrete and tied to model errors.** Rather than regenerating rules blindly, RLIE mines high-error training examples using the current weighted model and uses them to revise the rule set. This is a sensible mechanism, and the Retweets case study does show a plausible refinement trajectory with improving training performance.
- **Empirically, the method appears competitive across several tasks.** In Table 1, RLIE is often among the strongest methods on the reported benchmarks, and Table 2 consistently supports the narrower claim that the learned linear combiner is more effective than re-injecting rule information into an LLM.

## Weaknesses
###: Fatal
- **The experimental setup contains a serious inconsistency about which LLMs were actually used.** Section 4.3 says: *“All experiments involving LLMs utilized gpt-4o-mini”*, but Table 1 reports baselines with **DeepSeek-V3** and RLIE with **Qwen3-Next-80B / Qwen3-235B / DeepSeek-V3**, and Table 2 reports **DeepSeek V3.2** and **Qwen3 235B**. This is not a minor wording issue; it affects the interpretation of the entire experimental section. As written, it is unclear whether:
  1. all LLM-mediated rule generation/judgment used gpt-4o-mini and the tables are mislabeled,
  2. different backbones were actually used for different methods,
  3. RLIE’s gains partly come from larger-capacity models rather than the framework itself.  
  Until this is resolved, the main comparison in Table 1 is not reliable enough to support the paper’s strongest empirical claims.

### Major:
- **The evidence for the broad conclusion about LLMs being poor at probabilistic integration is weaker than the paper claims.** Table 2 clearly shows that, under the authors’ prompting strategy, E1 (linear-only) outperforms E2–E4. That is a valid empirical result. However, the paper often escalates this into a broader claim that LLMs are generally “less reliable at fine-grained, controlled probabilistic integration” and uses this to motivate a general architectural principle. The prompts in Appendix E for E3/E4 are fairly lightweight natural-language instructions (“the weight’s magnitude reflects the pattern’s importance,” “use ... as reference”), not a particularly strong or structured test of numerical/probabilistic reasoning. So the data supports the narrower statement—*these prompting schemes underperform the linear combiner*—more strongly than the broader claim about intrinsic LLM limitations.
- **The scale of the evaluation is modest relative to the strength of the generalization/robustness claims.** Section 4.3 states fixed splits of 200 train / 200 val / 300 test per task. For six tasks this is enough for a proof-of-concept, but it is thin support for claims like “robust performance,” “generalizable superior performance,” and strong statements about reliable neuro-symbolic reasoning. The test sets are only 300 examples, and several method gaps are not huge. The paper reports means and standard deviations over repeats in Section 4.3, but the main result tables do not actually show deviations for Table 1, nor any significance analysis.
- **A key ablation is missing: how much does iterative refinement actually matter?** RLIE is presented as a four-stage framework, and iterative refinement is central to the method description. But there is no direct ablation comparing the full method against a simpler variant such as “single rule-generation round + logistic regression” using the same backbone and evaluation protocol. Without this, it is difficult to tell whether the gains come mainly from the local/global decomposition itself or specifically from the hard-example refinement loop.
- **The pruning strategy can conflict with the stated goal of learning a collaborative rule set.** In Section 3.3, when capacity is exceeded, rules are pruned by **individual accuracy on the validation set** before retraining the global combiner. This is a real methodological weakness: individually mediocre rules can still be valuable because they cover complementary subcases, while individually strong rules may be redundant. Since the paper emphasizes joint composition of rules as a central motivation, pruning by marginal accuracy is a somewhat mismatched heuristic.

### Minor
- **The treatment of ternary rule judgments is under-analyzed.** The local LLM returns \{-1, 0, +1\}, where 0 means abstain, and these values are used directly as logistic-regression features. This is reasonable, but the paper does not analyze whether abstention behaves symmetrically across classes, whether 0 is the best encoding, or whether abstention contributes materially to calibration or sparsity.
- **Interpretability claims are only partially substantiated.** The paper claims the learned rule sets are “more compact and semantically clearer” and support “knowledge discovery and human-AI consensus,” but the qualitative evidence is limited mainly to one case study in the appendix. More examples across tasks, or some human assessment of rule quality, would better support these interpretability claims.
- **Calibration is asserted more than demonstrated.** Calibration is an important part of the motivation for the probabilistic combiner, but the paper reports accuracy and macro-F1 only. If calibrated reasoning is a headline benefit, some direct calibration evidence would strengthen the case substantially.

### Trivial
- None.

## Nice-to-Haves
- Add a controlled comparison where RLIE and the main baselines use the **same backbone model**, so the contribution of the framework is isolated from model capacity.
- Add an ablation comparing full RLIE to **generation + logistic regression without iterative refinement**.
- Include **calibration metrics** (e.g., ECE/reliability plots) for E1 vs. the LLM-based inference strategies.
- Provide **more qualitative rule examples** from at least two non-social-media tasks to support the claims about semantic clarity and auditability.
- Report **LLM call counts / cost / latency**, since RLIE requires repeated per-rule per-sample judgments and iterative refinement.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Criticism that baseline asymmetry is unfair because RLIE may be stronger.** If the experimental asymmetry favors the authors’ method, that is a legitimate concern only when it is tied to the concrete model-specification inconsistency above. Generic complaints about “unfair comparison” without that verified inconsistency were removed.
- **Requests for unrelated extra baselines based on external literature knowledge.** Suggestions such as adding specific missing related methods or claiming omission of certain prior work were removed because they rely on external completeness judgments not verifiable from the submission alone.
- **Pure reproducibility complaints about code release timing.** The paper states code will be released upon publication; under the review instructions, concerns centered on release status are not retained.
- **Formatting/style/prompt grammar nitpicks.** Issues such as minor English errors in Appendix E were removed as they do not materially affect the scientific evaluation.
- **Claim that coverage-threshold sensitivity is unstudied.** This criticism is not accurate: Appendix C/Table 4 does include a sensitivity analysis for the coverage threshold \(\gamma\), albeit on one dataset only.
- **Claim that the paper reports no variability at all.** Section 4.3 states experiments were repeated at least three times and mean/std were computed. The valid criticism is narrower: Table 1 does not actually display the stds or significance tests in the main results.

## Novel Insights
The paper’s strongest idea is not just “LLMs can generate rules,” but that natural-language rules may be most useful when LLMs are confined to **semantic interface tasks**—generation and local applicability judgments—while a classical model handles **global evidence aggregation**. That division is more compelling than the paper’s broader rhetoric about LLM limitations. At the same time, the current pruning heuristic reveals an internal tension: the method advocates collaborative rule sets, yet one of its main selection mechanisms still evaluates rules largely in isolation. This suggests the most promising next version of the work is not to replace the linear combiner with a more complex LLM prompt, but to improve the *set-level optimization* around rule retention, calibration, and interaction analysis.

## Suggestions
- **First, fix the model-usage inconsistency unambiguously.** State exactly which model is used for each role: rule generation, rule judgment, RLIE inference, and each baseline. Then align Tables 1–2 and Section 4.3 accordingly.
- **Tone down the broad claim about LLM limitations unless you test stronger prompting schemes.** Reframe the current conclusion as: under the evaluated prompting protocols, direct linear aggregation is more reliable than LLM-mediated aggregation.
- **Add the missing ablation:** initial rule generation + logistic regression, with and without iterative refinement.
- **Replace or supplement individual-accuracy pruning with a set-aware criterion,** e.g., pruning based on contribution after refitting, ablation-based importance, or elastic-net coefficients from the global model.
- **Show calibration evidence** if calibration is a core claimed benefit.
- **Expand qualitative analysis across datasets** so the interpretability claim is backed by more than a single appendix case study.

---

## TgLW2DiRDG

- GT: Accept (Oral) (avg 7.5)
- Predicted: N/A (6.2/10)
- Match: N/A

### Final Review

## Summary
This paper studies the polyhedral complex induced by fully-connected ReLU networks through its connectivity graph, where nodes are linear regions and edges connect regions sharing a face. Its main theoretical claims are that the average degree of this graph is at most \(2d\) (independent of width/depth), that this bound is asymptotically tight, and that the graph diameter admits an upper bound depending on width/depth but not on input dimension. The paper also provides an LP-based enumeration procedure and experiments on synthetic and trained networks to probe these geometric properties.

## Strengths
- **A genuinely new angle on ReLU geometry:** Rather than studying only the number of linear regions, the paper focuses on how regions are glued together via the connectivity graph. This is a meaningful shift in perspective, and the average-degree result in Theorem 3.4 is both clean and surprising: a width/depth-independent upper bound of \(2d\) on average region connectivity.
- **The main combinatorial proof strategy is substantive and well structured:** The decomposition via removing one bent hyperplane (\(C-h_i\)), together with Lemmas 3.2 and 3.3, gives a clear recursive mechanism for counting cells and relating different dimensions. This is the technical core of the paper, and it is more than a superficial adaptation of hyperplane-arrangement arguments.
- **The asymptotic tightness result is useful, not just the upper bound alone:** Theorem 3.7 shows that for shallow generic arrangements the average degree converges exactly to \(2d\), which helps interpret the upper bound as a meaningful structural limit rather than a loose worst-case artifact.
- **The paper connects the theory to explicit computation rather than leaving it abstract:** Section 4 and Appendix D give a concrete LP-based procedure for reconstructing adjacency relations between regions from sign sequences. Even if it is only practical at modest scales, it makes the object of study operational.
- **One empirical observation is interesting and potentially important:** In Section 5.2 / Fig. 6, regions containing training data tend to have higher connectivity than regions without data. The paper does not explain this theoretically, but it is a nontrivial pattern worth surfacing.

## Weaknesses

###: Fatal

### Major:
- **The empirical claims for large trained networks are limited by truncated exploration, and some conclusions are stronger than the evidence supports.**  
  For CIFAR10 and California Housing, the paper explicitly does **not** enumerate the full complex: “*complete enumeration of the network complex was intractable, so the search was terminated after traversing 8 million polyhedra*.” It then augments this with regions containing sampled training points. This is enough to study many encountered regions, but it does **not** justify strong claims about the global degree distribution of “polyhedra that do not contain training data,” since those are whatever the truncated traversal happened to visit. The observation about data-containing regions may still be real, but for the partially explored complexes it should be framed more cautiously as a property of the explored subset, not of the entire complex.
- **The theory depends on genericity/supertransversality assumptions, while the experiments are on trained networks, and this theory-practice gap is not examined in depth.**  
  The paper is transparent that all theoretical statements inherit the assumptions from Masden (2025): “*all statements about ReLU networks will make the same assumptions as in (Masden, 2025) to avoid degenerate weight assignments*,” and Appendix B formalizes genericity and supertransversality. So it would be incorrect to say these assumptions are hidden. However, the paper does not analyze how robust the conclusions are when trained networks are near-degenerate or violate these assumptions. Since the experiments are presented as corroborating the theory on trained models, a more careful discussion of when trained networks empirically satisfy or approximate these conditions would materially strengthen the paper.
- **The diameter result is less convincing empirically than the average-degree result.**  
  Theorem 3.8 is one of the headline contributions, but the empirical support is weaker than for Theorem 3.4. The paper states that diameter is often **estimated** by upper/lower bounding algorithms and using the midpoint, rather than computed exactly. This is a reasonable practical choice, but it weakens strong claims such as the diameter growing “almost identically” across input dimensions. Moreover, the upper bound itself appears quite loose in practice, and the paper does not provide much interpretation of when the bound is expected to be informative.
- **There is a notation/statement inconsistency around the diameter upper bound.**  
  In the contributions and Theorem 3.8, the main text states an upper bound of the form \((m+1)^\ell\) / \(O(m^\ell)\), while Appendix B derives the recursive path bound as \(\prod_j (m_j+1)\le (m+1)^\ell\). These are asymptotically compatible, but the presentation is sloppy enough to create ambiguity about what exact statement is being claimed and proved. For a theory paper, this should be made fully consistent.

### Minor
- **The practical relevance of the LP-based enumeration method is limited by scalability.**  
  The paper itself encounters this limitation repeatedly: exact enumeration becomes intractable on larger models, and real-data experiments either reduce dimensionality (e.g., analyze only the classifier on lower-dimensional hidden representations) or truncate search. This does not negate the theoretical contribution, but it does limit the extent to which the computational pipeline can currently validate or exploit the theory at realistic modern scales.
- **The “data lies in higher-degree regions” finding is descriptive rather than explanatory.**  
  The paper acknowledges this in the discussion: “*Further investigation is needed to fully explain why training tends to put data points in regions with higher numbers of faces*.” As it stands, this is an interesting empirical regularity, but not yet a mechanistic insight.
- **Some claims in the discussion overreach the paper’s actual results.**  
  In Section 6, the suggestion that connectivity-graph path length could replace Hamming distance in generalization/error bounds is plausible but speculative. The paper does not derive such a bound, so these implications should be presented more explicitly as conjectural future directions.

### Trivial
- **The empirical validation is concentrated in low input dimensions and modest architectures.**  
  This is understandable given the combinatorial explosion, but it still narrows the practical scope of the experimental study relative to the breadth of the theoretical framing.

## Nice-to-Haves
- Add an explicit empirical check of how often trained networks satisfy, approximately satisfy, or violate the genericity/supertransversality assumptions used in the proofs.
- Reframe the real-data truncated-search experiments as analysis of the explored subgraph/complex, and if possible provide bias diagnostics for the BFS-plus-data-augmentation sampling procedure.
- Strengthen the diameter section with either exact computations on a somewhat larger regime or clearer uncertainty characterization for the estimated diameters.
- Provide a clearer complexity/scaling discussion for Algorithm 1, including wall-clock cost as a function of region count, width, and dimension.
- Explore controlled synthetic experiments to test the “data in high-degree regions” phenomenon against data density, manifold structure, and training dynamics.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that the theory is “fundamentally invalid” for trained networks because genericity assumptions are systematically violated.**  
  Removed/weakened because the paper explicitly states the assumptions and does not claim unconditional theorems for all trained networks. The real issue is not that the theory is invalid, but that the paper does not sufficiently study how these assumptions relate to trained models.
- **Claim that the proof of the diameter upper bound “conflates input-space paths with graph distances.”**  
  Removed because the proof is explicitly constructing a path in the region adjacency graph by following an input-space path and counting crossed faces/BHs. That is a standard and valid way to upper-bound graph distance.
- **Criticism about missing training seeds / exact reconstruction details.**  
  Removed as a reproducibility nitpick rather than a substantive issue under the stated review policy.
- **Criticism centered on CIFAR10 accuracy being too low to make the geometry meaningful.**  
  Removed in strong form; the model’s performance may affect practical interest, but it does not invalidate the geometric analysis. The real limitation is that the analysis is on reduced-dimensional hidden representations and partially explored complexes.
- **Claim that the novelty is overstated because the \(2d\) bound is just a standard fact for generic complexes.**  
  Removed in that form. The paper’s contribution is precisely extending such style of bounds to deep ReLU bent-hyperplane complexes using the sign-sequence framework; that extension is the nontrivial part.

## Novel Insights
The strongest synthesis across the reviews is that the paper’s **average-degree theorem and its proof machinery are substantially stronger than its empirical storyline**. The work seems most compelling when read as a structural theorem about ReLU-induced cell complexes under generic assumptions, with computation serving as illustrative support. By contrast, the paper is less convincing when it tries to elevate truncated large-network explorations into claims about the global geometry of trained models. Put differently: the paper’s real “spark” is that deep bent-hyperplane arrangements may still obey a strikingly low-dimensional local adjacency law (\(\le 2d\) on average), even as the total number of regions explodes; the main thing holding the paper back is not the theorem but the overextension of limited empirical evidence around trained-network geometry.

## Suggestions
- Tighten the paper’s scope: present the theoretical contribution as the centerpiece, and moderate empirical claims for partially explored trained networks.
- In Section 5.2, explicitly distinguish between **fully enumerated** complexes and **truncated/explored subsets**; avoid language implying unbiased global statistics when enumeration is incomplete.
- Add an empirical study of degeneracy/genericity in trained models, even if only approximate (e.g., frequency of near-parallel constraints, rank deficiencies, repeated/near-repeated activation boundaries).
- Make Theorem 3.8 and its proof notation fully consistent: state the exact non-asymptotic bound proved, then separately give the asymptotic simplification.
- If space permits, add one controlled experiment testing whether the observed higher degree of data-containing regions persists under different traversal initializations or sampling schemes, to rule out search bias.
- Clarify the practical role of the results: what can a practitioner infer from knowing average degree is near \(2d\), and in which downstream settings (verification, robustness, interpretability) does this concretely matter?

---

## LIG31I6ArY

- GT: Withdrawn (treated as Reject) (avg 2.5)
- Predicted: N/A (2.8/10)
- Match: N/A

### Final Review

## Summary
The paper proposes **IntE**, a framework for evaluating qualitative response datasets by comparing an intrinsic response-cluster distribution against an extrinsic demographic distribution, and by mining representative versus unusual responses. The system combines a four-metric assessment scheme (GMR, DDR, DP, DC) with an LLM-based dissimilarity pipeline that includes iterative instruction generation and an adaptive anchor mechanism for consistency.

## Strengths
- **Clear dataset-level framing rather than pointwise scoring.** The paper explicitly targets a real gap: most automated approaches score individual responses, whereas IntE tries to characterize whether an entire qualitative dataset is suitable for finding broad patterns versus exceptional cases. The four-way decomposition by **goal** (general patterns / unique insights) and **granularity** (distribution / data point) is a concrete and potentially useful conceptual contribution for practitioners.
- **The prompt/instruction support system appears practically useful, and the paper does provide evidence for that specific subcomponent.** The within-subject user study directly compares assisted versus manual prompt creation and reports consistent reductions in cognitive load and better usability across the listed questionnaire dimensions (Sec. 4.1.1, Appendix D.2). Whatever one thinks of the full IntE framework, this subproblem is addressed with more specificity than many papers of this kind.
- **The paper is unusually explicit about implementation details.** The appendices include detailed prompts, synthetic-generation procedures, and algorithmic workflows. This makes it possible to inspect the assumptions directly rather than infer them from vague descriptions.
- **A useful insight in the design is the separation between “alignment with known structure” and “dispersion/divergence within learned structure.”** Even if the current metrics are not yet fully justified, the paper is not simply equating quality with demographic agreement; it also includes within-cluster dispersion and cluster purity/heterogeneity signals intended to separate “representative” from “interesting” data regimes.

## Weaknesses

###: Fatal
- **The core claim that these metrics quantify qualitative dataset “quality” or “knowledge discovery potential” is not convincingly validated.** The paper’s central premise is that comparing demographic partitions with response-induced clusters, plus cluster compactness/dispersion, yields a quantitative assessment of whether a dataset is good for discovering “general patterns” and “unique insights.” However, the paper does not provide strong evidence that high/low values of GMR, DDR, DP, and DC actually correlate with human judgments of dataset utility on real qualitative datasets.  
  This is especially important because the paper itself treats the demographic structure as “the ground-truth or expected structure of the dataset” (Sec. 3.2.2), yet in many realistic qualitative settings useful themes can cut across demographics. The framework may still be useful as a diagnostic lens, but the stronger claim that it measures dataset quality for knowledge discovery is not established empirically or theoretically.

### Major:
- **The synthetic validation is too closely aligned with the assumptions of the proposed method, so it does not adequately validate the paper’s core scientific claim.** In Appendix B, the synthetic generation pipeline explicitly creates communities, score vectors, personas, and responses conditioned on those designed community properties. This is appropriate for controlled testing, but it means the synthetic data is constructed around latent structure that is already intended to map onto the metadata/community partition. As a result, the strong controlled results mainly show that IntE behaves as expected when the world matches its assumptions. That supports internal consistency, but not the broader claim that the framework measures utility on messy real qualitative data.
- **Real-world validation is too limited for the breadth of the claims.** The main real-data evaluation is one case study on 126 food-choice responses with expert confirmation of mined examples. This is a useful illustration, but it is not enough to support claims of general practical utility across qualitative research settings. There is no quantitative comparison to human ratings of dataset utility, no baseline against simpler retrieval/diversity methods, and no demonstration that using IntE changes downstream analytic outcomes.
- **The large-scale approximation introduces a serious representational simplification that is acknowledged but insufficiently analyzed.** For large datasets, Sec. 3.1.3 replaces pairwise dissimilarities with  
  \[
  \delta(d_i,d_j)\approx |S(d_i)-S(d_j)|
  \]
  where each response is projected to a single scalar. This is a major reduction from arbitrary semantic relations to a one-dimensional ordering. The paper acknowledges information loss (“scalar projection simplifies high-dimensional semantic relationships into a single value”), but does not quantify how much this approximation changes downstream clustering or the four final metrics. Since scalability is part of the proposed framework, this omission matters.
- **The dissimilarity component is not compared against strong non-LLM or simpler text-similarity baselines.** The paper compares prompt variants and LLM variants, but not against established embedding-based similarity pipelines or human pairwise similarity judgments. Thus the claim in the abstract that the system “accurately computes inter-response dissimilarity” is not well substantiated relative to alternative methods.
- **The proposed metrics are domain-specific reformulations of familiar clustering quantities, but the paper does not justify why these specific formulations are the right ones for its stated goal.** GMR, DDR, DP, and DC are built from alignment counts, intra/inter-cluster dissimilarity, and purity-like quantities, with clipping and scaling hyperparameters. That does not make them invalid, but the paper does not compare them to standard clustering agreement/separation measures or show that the new formulations add decision-relevant information beyond simpler indices. Without such analysis, it is hard to assess whether the metrics are principled contributions or heuristic packaging.

### Minor
- **The clustering backbone is under-examined.** The intrinsic structure is obtained from “ensemble clustering, where we use multiple k-means clusters and then vote for the final result” (Sec. 3.2.2). Given that free-text response spaces may be non-spherical and irregular, the paper should do more to establish that conclusions are not an artifact of this clustering choice.
- **Hyperparameter sensitivity remains underdeveloped.** The paper gives recommended settings for \(\alpha,\beta,\eta,\gamma\), and the parameter sweep in Sec. 4.1.3 studies synthetic distribution changes, but not robustness of conclusions to these hyperparameters across real datasets. Because these parameters directly affect metric magnitudes and clipping, this matters for interpretation.
- **The user study, while useful, validates usability more than scientific effectiveness.** It supports that the prompt-generation interface reduces effort, but does not directly show that the resulting instructions improve downstream dataset assessment quality on real tasks.
- **The prompt optimization stage relies on an LLM oracle scoring prompts until a threshold is reached, but the paper offers limited evidence that this stopping criterion is reliable.** This is not fatal, especially because there is later human adaptation, but the automated stage still depends on self-referential LLM evaluation without much robustness analysis.

### Trivial
- **The paper would benefit from clearer diagnostic visualizations** such as cluster-demographic confusion matrices or low-dimensional projections of the learned dissimilarity space, which would make the metric behavior easier to inspect.
- **Support for richer metadata structures is not discussed clearly.** The presentation mainly assumes a single categorical demographic mapping \(y_i=f(I_i)\), while many real studies use multi-axis or continuous metadata.

## Nice-to-Haves
- Add a real-data study where domain experts rate overall dataset utility, representativeness, and novelty potential, then report correlations with GMR/DDR/DP/DC.
- Compare LLM-based dissimilarity against embedding baselines and, if feasible, a small human-annotated pairwise similarity set.
- Quantify the degradation from the large-scale scalar approximation relative to the full pairwise version on the same data.
- Test whether using IntE actually improves downstream qualitative analysis decisions, e.g., selecting additional samples, pruning noise, or prioritizing responses for coding.
- Evaluate clustering robustness with alternative clustering methods, not only k-means ensemble variants.
- Report computational cost/latency, since practical scalability is part of the motivation.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The anchor manifold is functionally irrelevant in the large-scale branch.”** This is overstated. The paper does specify that scalar scoring \(S(d_i)\) is computed by an LLM call that uses anchors as context (“Compute scalar score \(S(d_i)=\) LLM(\(P^*, d_i, A\))”, Appendix Algorithm 3), so the anchors are not absent. The valid criticism is narrower: the paper does not quantify whether this scalarization preserves meaningful semantic relations.
- **“The paper omits related work on dataset cartography/topic stability/etc.”** Removed per instruction not to penalize missing related work absent external verification.
- **Pure reproducibility complaints about missing implementation minutiae.** The paper is already unusually detailed in appendices; remaining omissions are not the core issue here.
- **Claims doubting the existence/availability of cited tools or models.** Not considered.

## Novel Insights
The most important synthesis is that the paper’s strongest contribution is not yet the claimed quantitative theory of qualitative-data quality, but rather a **structured diagnostic workflow** for practitioners: combine an explicit target partition from metadata, an induced semantic partition from responses, and response-level centrality/outlier mining to guide analysis. Read this way, IntE is potentially a useful analyst-facing heuristic system. The problem is that the paper argues for a stronger interpretation—namely that these metrics *measure dataset quality and knowledge discovery potential*—without the level of human-grounded validation needed to justify that leap. In short: there is a promising diagnostic interface hiding inside an under-validated scientific claim.

## Suggestions
- Reposition the main claim more conservatively unless stronger validation is added: present IntE as a **diagnostic heuristic framework** rather than a validated quantitative measure of qualitative dataset quality.
- Add at least one substantial real-world evaluation in which human experts independently rate dataset utility, representativeness, and novelty potential, and compare these judgments against the four metrics.
- Benchmark the dissimilarity module against simpler alternatives such as embedding-based similarities and human pairwise judgments.
- Quantify the effect of the large-scale scalar approximation on clustering outputs and final IntE scores.
- Show robustness to clustering choices and hyperparameters, ideally on real datasets.
- Strengthen the case study by including a comparative baseline and by showing that acting on IntE’s outputs improves downstream qualitative analysis efficiency or insight discovery.

---

## P0GOk5wslg

- GT: Accept (Oral) (avg 7.5)
- Predicted: N/A (5.6/10)
- Match: N/A

### Final Review

## Summary
This paper proposes **Speculative Actions**, a framework for accelerating agentic systems by predicting likely next actions/API calls with a fast “Speculator” while a slower authoritative “Actor” computes the true next step. The paper’s main contributions are: (i) a clean action-level generalization of speculative execution/decoding, (ii) analytical cost–latency tradeoff results for breadth- and depth-oriented speculation, and (iii) empirical demonstrations across chess, e-commerce, HotpotQA, and a lossy OS tuning setting.

Overall, the paper has a real conceptual spark and some useful theory, but the current empirical validation does not fully substantiate the paper’s broad practical claims about a **general, lossless acceleration framework**. The strongest concerns are about scope/generalization of the “lossless” claim, the gap between theory and real agent dynamics/overheads, and the lack of stronger baselines and component-level latency accounting.

## Strengths
- **A genuinely interesting shift in speculation granularity—from token decoding to agent actions/API calls.** The paper does more than apply a buzzword: it reframes agent latency as an environment-interaction bottleneck and formalizes actions uniformly as asynchronous API calls (§2). This is a useful systems abstraction that could influence how agent runtimes are designed.
- **The paper articulates a concrete and nontrivial “lossless if safe-to-speculate” interface.** The design is not merely “run things in parallel”; it explicitly introduces Actors vs. Speculators, caching of pre-launched futures, semantic guards, safety envelopes, and rollback/repair paths (§1–2). Even though not all of these are implemented in full generality, the framework itself is more carefully structured than a simple prefetch heuristic.
- **The analytical treatment is richer than a typical empirical systems paper.** Proposition 1 and the later cost–latency analysis provide explicit formulas for how latency savings and extra cost scale with speculative hit probability and branch width. The breadth-vs-depth discussion in §5 is especially valuable as a conceptual lens, even if idealized.
- **The experiments do show that next-action prediction is often feasible in several distinct settings.** The paper reports nontrivial prediction rates across domains: roughly 55% top-k move-match in chess, 22–38% API prediction accuracy in e-commerce, and up to 46% top-3 API-call prediction in HotpotQA. These results support the core intuition that agent actions are often predictable enough to speculate on.
- **The OS tuning case illustrates a meaningful extension beyond strict losslessness.** While it is somewhat orthogonal to the main claim, the “last-write-wins” control-loop variant in §4 is an interesting example of when speculative actions can improve both responsiveness and eventual performance, rather than merely hiding latency.

## Weaknesses

### Fatal
None.

### Major:
- **The “general lossless framework” claim is overstated relative to what is actually validated.** The paper correctly states that losslessness requires “semantic guards,” “safety envelopes,” and “repair paths” (§1), and later concedes that speculation must be limited to actions that are “idempotent, reversible, or sandboxed” and that many systems involve “irreversible or externally visible effects” where naive speculation is harmful (§2). This is an important restriction, not a minor implementation detail. The experiments are mostly in environments where rollback or discard is easy (chess, retrieval/search-like settings, pre-checkout e-commerce flows), and the paper does not demonstrate concrete rollback/compensation mechanisms in a realistic stateful toolchain. As written, the title/positioning suggests broader generality than the evidence supports.
- **The empirical evaluation does not sufficiently isolate the value of speculative actions from simpler parallelization/prefetching alternatives.** Across environments, the main baseline is sequential execution. The paper does not compare against strong non-predictive systems baselines such as standard async tool parallelism, static prefetching/caching, or other runtime optimizations that might recover part of the same latency savings without requiring speculation. Because the paper’s core practical claim is a systems one, this baseline gap matters: it remains unclear how much of the gain is uniquely due to predictive branching rather than generic overlap of independent work.
- **The theory is useful but substantially idealized relative to the paper’s practical tuning claims.** Proposition 1 assumes independent per-step hit probability and exponential latencies, and §5 continues with similarly stylized assumptions. For a theoretical section this is acceptable, but the paper goes further and claims this enables “principled tuning” of speculative breadth. In realistic agents, prediction accuracy is state-dependent, errors can cluster, and overheads from branch management/validation are not modeled. This does not make the theory wrong, but it does limit how directly it supports the practical tuning narrative.
- **The paper does not measure the overheads that are most critical to the lossless story.** The framework emphasizes validation, caching, branch tracking, and rollback/repair, yet the experiments do not provide a component-wise latency breakdown including these costs. This omission is especially important because the reported end-to-end gains are fairly modest in the main lossless settings (e.g., ~19.5% in chess despite ~54.7% top-k prediction accuracy). Without a breakdown of actor time, speculator time, prelaunch time, validation/equivalence-checking time, and cleanup/abort overhead, it is hard to tell how much headroom the framework really has and whether the unmeasured bookkeeping costs would materially erode gains in more realistic deployments.

### Minor
- **Some evaluation metrics are only indirect proxies for the claimed benefit.** In e-commerce and HotpotQA, the primary metric is API prediction accuracy rather than direct end-to-end wall-clock speedup under the full runtime. That is reasonable as an intermediate diagnostic, but it weakens the paper’s claim of demonstrated latency reduction across domains. The abstract/introduction emphasize system speedups, whereas parts of the evaluation mainly show predictability.
- **HotpotQA uses an overly strict exact-match criterion for predicted API parameters, which obscures functional utility.** The appendix explicitly notes that semantically similar phrasing differences count as errors and can make stronger models look worse (§B.2.2). This is a defensible strict metric, but reporting only this criterion makes the practical value of speculation harder to judge.
- **The confidence-aware selective speculation story is under-validated empirically.** Theorem 3 is interesting, but the actual experiment is only a simplified threshold heuristic in chess (“predicted accuracy exceeds 50%”), not a full demonstration of calibrated confidence-based selection. Since the method’s practical usefulness depends on good branch confidence estimates, stronger empirical support here would help.
- **The OS tuning section is interesting but somewhat dilutes the central message.** It is explicitly lossy and effectively becomes a hierarchical/reactive control-loop story rather than a clean demonstration of the paper’s lossless semantics. I would not remove it, but the paper should be clearer that this is an extension with different guarantees, not evidence for the core lossless claim.
- **The chess setup partially entangles “fast speculation” with prompt-budget reduction on the same model family.** Using GPT-5 with high reasoning effort for the Actor and low reasoning effort for the Speculator is a sensible practical choice, but it makes the cost story less clear than if the paper more explicitly quantified the latency/cost differential. The current section mainly establishes latency overlap, not a strong economic advantage.

### Trivial
- **The paper would benefit from a clearer decomposition of where time is saved in each environment.** A simple trace/Gantt visualization or stacked latency breakdown would make the mechanism much easier to inspect.

## Nice-to-Haves
- Add comparisons against stronger systems baselines: async/parallel tool execution, static caching/prefetching, and if feasible a speculative planning baseline on a matched task.
- Include a component-level wall-clock breakdown: speculator generation, actor latency, pre-launched branch time, cache-hit savings, validation/equivalence-checking, and abort/cleanup overhead.
- Report semantic/functional match metrics for HotpotQA alongside exact string match.
- Clarify, per environment, the precise safety mechanism enabling “losslessness” (sandboxing, idempotence, rollback, compensation, etc.).
- Strengthen the confidence-aware story with calibration metrics and a real adaptive-selection experiment rather than only a fixed threshold heuristic.
- Be more explicit about scope: frame the method as best suited to reversible/read-mostly/sandboxed environments, with lossy variants as a separate extension.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper ignores compute/memory/cost overhead entirely.”** This is too strong. The paper explicitly includes a cost–latency analysis in §5 and discusses speculative branch cost, including token/cost accounting in the OS section and Theorem 4. The criticism is still valid in weaker form—namely that some important overheads (validation/rollback/runtime bookkeeping) are not empirically measured—but not that cost is ignored.
- **“The paper is flawed because API jitter / live latency variance makes results irreproducible.”** The paper itself acknowledges latency stochasticity in chess (§3.1.2). Lack of confidence intervals is not ideal, but demanding more exhaustive reproducibility artifacts or multi-machine sweeps is not necessary to assess the core idea here.
- **“Comparisons are unfair because the baselines are asymmetrically weaker.”** I do not see an unfair comparison in the sense of handicapping baselines; rather, the issue is missing stronger baselines altogether.
- **“The same-model Actor/Speculator setup in chess invalidates the result.”** Not true. It does not invalidate the latency result; it only weakens the cost-efficiency story.

## Novel Insights
The most interesting synthesis across the reviews is that the paper’s strongest contribution is not the raw speedup numbers, but the reframing of agent execution as a speculative systems problem at the API/action layer. However, that same abstraction exposes the key limitation: unlike token speculation, action speculation inherits environment semantics, side effects, and rollback obligations. In other words, the paper is strongest when read as a **design framework plus idealized analysis for reversible/read-heavy agent loops**, and weaker when read as a broadly validated, general-purpose lossless runtime layer for arbitrary agentic systems. Tightening that scope would make the work feel more precise and credible.

## Suggestions
- Recast the main claim more carefully: emphasize **reversible/sandboxed/read-heavy environments** as the primary target of the lossless framework.
- Add a systems-level baseline suite that includes non-predictive overlap methods, so the unique value of speculative prediction is clear.
- Measure and report the runtime overheads that the framework itself introduces: validation, branch bookkeeping, cache management, and any rollback/cleanup.
- For at least one realistic tool environment, implement and demonstrate a concrete safety mechanism end-to-end, not just speculative prelaunch plus discard.
- Strengthen the empirical support for §5 by evaluating adaptive/confidence-aware speculation more directly.
- Report direct wall-clock end-to-end gains, not just prediction accuracy, in more than one non-game environment.

**Axis-wise assessment:**  
- **Novelty:** strong at the action/API-level reframing.  
- **Technical soundness:** decent but limited by idealized assumptions and incomplete treatment of safety/runtime overheads.  
- **Empirical support:** promising but not yet strong enough for the breadth of the claims.  
- **Significance:** potentially high if scoped appropriately; currently somewhat overstated.  
- **Clarity:** generally clear in concept and structure, though the paper would benefit from sharper scoping around where “lossless” really applies.

---

## Ro282CMb1O

- GT: Reject (avg 5.0)
- Predicted: N/A (6.8/10)
- Match: N/A

### Final Review

## Summary
This paper presents **U-Bench**, a large-scale benchmark of **100 U-shaped medical image segmentation models** across **28 datasets and 10 modalities**, with analyses along in-domain accuracy, zero-shot transfer, efficiency, and architectural/data characteristics. Beyond collecting results, it introduces an efficiency-aware metric (**U-Score**) and a ranking-based advisor agent intended to help practitioners choose architectures under dataset and resource constraints.

## Strengths
- **The paper contributes a genuinely broad and unusually structured benchmark for U-shaped segmentation models.** It spans 100 variants across CNN, Transformer, Mamba, RWKV, and hybrid families, and evaluates them on 28 datasets / 10 modalities with both in-domain and zero-shot settings. This breadth is not a generic “many experiments” claim; the benchmark is explicitly organized to compare architectural paradigms under common preprocessing, training, efficiency measurement, and zero-shot transfer protocols.
- **The paper’s main empirical insight is concrete and important: many apparent improvements over vanilla U-Net are small and often not statistically convincing in-domain, while zero-shot gains are more substantial.** This is a strong, benchmark-driven claim that goes beyond leaderboard reporting and is central to the paper’s value. The analysis in Section 3.1, supported by per-dataset rankings and significance summaries in the appendix, gives the community a more sober view of progress than typical single-dataset papers.
- **U-Score is a useful benchmark artifact even if not yet definitive as a universal metric.** The paper clearly defines the metric from IoU, parameters, FLOPs, and FPS using percentile normalization and harmonic means, and importantly includes a substantial sensitivity analysis (Appendix E, Tables 10–13). That analysis shows the ranking is reasonably stable to weighting and quantile choices, which is a stronger validation than many newly proposed benchmark metrics receive.
- **The benchmark includes a meaningful zero-shot protocol rather than treating generalization as an afterthought.** The paper evaluates transfer to unseen datasets within the same modality/task (e.g., Kvasir→CVC300/CVC-ClinicDB, ISIC2018→PH2, Montgomery→NIH-test), and this is one of the more practically relevant aspects of the work.
- **The paper attempts to turn benchmark observations into actionable guidance through dataset-characteristic analysis and an advisor agent.** The feature characterization of foreground scale, shape complexity, and boundary sharpness, together with family-level comparisons, is a useful step toward task-aware model selection rather than one-size-fits-all ranking.

## Weaknesses

### Major:
- **The training protocol is internally inconsistent, and this directly affects the fairness of the benchmark conclusions.**  
  The paper states in the main text that it follows “official implementations … adopting their predefined settings, pretrained weights, and deep supervision strategies when available” (Section 2.2 / Introduction), but Appendix F.2.2 and Table 15 also state that training is unified across models with **SGD, lr=0.01, 300 epochs, batch size 8**, and a common BCE+Dice loss. These are not minor implementation details; they are competing descriptions of the actual protocol. For a benchmark whose central claims depend on fairness across very different architectures, the reader needs a precise answer to: which parts are inherited from official code, which are standardized, and whether pretrained initialization is actually used during benchmark training. As written, this ambiguity weakens confidence in architecture-level conclusions, especially when the paper interprets weaker performance of some families (e.g., Mamba in Section 3.2) as architectural rather than protocol-induced.

- **The statistical-significance framing is weaker than the paper claims, because the t-test setup is not sufficiently justified and appears to rely on single-seed training.**  
  The paper repeatedly emphasizes “statistical rigor,” but Table 15 lists a single random seed (**41**), and the paper does not clearly describe repeated runs per dataset/model. If significance is computed from one trained model per method, then the test is necessarily based on per-case predictions within a dataset rather than training-run variability; if so, the paper should state that explicitly and justify why that is the right inferential target. As written, the presentation blurs “statistical significance of a model difference on a test set” with “robustness of an architecture’s improvement,” which are not the same. This does not make the comparisons useless, but it does mean the benchmark overstates the rigor of its inferential claims.

- **The multiple-testing issue is not addressed.**  
  The paper performs large numbers of pairwise significance tests (each variant against U-Net across many datasets/modalities), but there is no discussion of multiple-hypothesis correction. Given how central Fig. 1E / Fig. 5 and the “few significant gains” narrative are to the paper, some treatment of family-wise error or false discovery rate is warranted. Without it, the precise counts of significant/non-significant wins should be interpreted cautiously.

- **U-Score is cohort-dependent, which limits its claim as a stable long-term metric of progress.**  
  By construction, U-Score normalizes each component using the **10th/90th quantiles over the current model zoo**, so a model’s score depends on which other models are included in the benchmark. The appendix does show sensitivity to quantile choices and weightings, which is good, but that is not the same as solving the core issue: U-Score is best understood as a ranking device *within this benchmark cohort*, not yet as an absolute metric that can track progress over time without recalibration. The paper occasionally describes it in broader terms than the methodology supports.

- **The model advisor agent is interesting but not yet a strong standalone contribution.**  
  Its evaluation is relatively narrow: training on 18 in-domain datasets, validating on 2 held-out datasets in the main setup, with appendix LOMO results showing that a simple heuristic can outperform it on IoU-only rankings in several settings. The paper’s own appendix notes that the heuristic is “extremely competitive” for IoU-only ranking. This makes the advisor more of a promising benchmark utility than a convincingly validated recommendation system. The main paper currently gives it more prominence than the evidence justifies.

### Minor
- **The paper’s “2D benchmark” positioning is somewhat imprecise because some included datasets are volumetric and are evaluated slice-wise.**  
  This is not a fatal issue—the paper does state in Appendix F.2.2 that 3D datasets like Synapse and ACDC are processed by axial slicing—but the title/abstract framing could more clearly say this is a **2D / slice-based benchmark**, not a benchmark of volumetric segmentation architectures.
- **The motivating literature audit (e.g., “84% papers neglect zero-shot evaluation,” “73% papers lack statistical significance testing”) is not documented with sufficient methodological detail in the main paper.**  
  Since these percentages are used prominently in Fig. 1 and the introduction, the sampling criteria for the 100 reviewed papers should be stated more transparently.
- **Some of the architecture-level interpretations are more speculative than established.**  
  For example, statements like RWKV showing “structural superiority” or Mamba underperforming due to difficulty with fine-grained detail are plausible but not strongly isolated from confounds such as training recipe, model size, or source-domain bias. The descriptive benchmark results are useful; the causal architectural explanations are less secure.
- **The preprocessing protocol may introduce cross-modality simplifications that are acceptable for standardization but should be discussed more carefully.**  
  The paper resizes datasets to 256×256 (or keeps 224×224 for some fixed-input models) and uses common augmentations across modalities. That is a practical benchmark choice, but for medical imaging it can distort scale/aspect information and may interact differently with certain architectures. This is a limitation of scope rather than a methodological error, but it deserves explicit acknowledgment in the main text.

### Trivial
- **The paper would benefit from a cleaner separation between benchmark contribution and auxiliary tools.**  
  U-Bench itself is substantial; the advisor agent and some interpretive claims would read more convincingly if presented as secondary extensions rather than co-equal headline contributions.

## Nice-to-Haves
- Provide a **clear protocol table** in the main paper that disambiguates: official code used, pretrained initialization used or not, optimizer/loss/scheduler standardized or not, and which model-specific training components are retained.
- Add a **multiple-testing correction analysis** for the significance results, or at minimum report how the conclusions change under an FDR procedure.
- Compare U-Score to **Pareto-front reporting** or simpler efficiency-accuracy summaries to clarify what additional decision value it provides beyond standard multi-objective views.
- Strengthen the advisor section with more **failure analysis** and clearer positioning as a benchmark-derived helper rather than a mature recommender.
- Make the title/abstract wording explicitly say **slice-based 2D evaluation** for volumetric datasets.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Limited scope to 2D models / missing 3D evaluation makes the benchmark unacceptable.”**  
  The paper explicitly scopes itself as a **2D benchmark** from the abstract onward (“the first large-scale, statistically rigorous 2D benchmark”), and later explains that volumetric datasets are handled slice-wise. This can be noted as a scope clarification issue, but not as a substantive flaw for failing to do something outside its stated scope.
- **Claims questioning release status / availability / legal actionability of datasets, weights, or tools.**  
  Per instruction, these should not be treated as valid weaknesses.
- **Reproducibility complaints about missing hyperparameters or training logs.**  
  The paper actually provides substantial implementation detail in Appendix F and Tables 15–16.
- **Pure demands for more architectural ablations of internal modules as if this were an architecture paper.**  
  More granular ablations would be useful, but the paper is a benchmark, not a new model paper; this is better framed as a nice-to-have than a core weakness.
- **“Unfair comparison with other methods because the authors standardized preprocessing/training.”**  
  Standardization is the point of a benchmark. The real issue is not asymmetry itself, but that the paper is ambiguous about how much it standardizes versus preserves from official settings.

## Novel Insights
The most important synthesis across the reviews is that the paper’s **value is real but narrower than its rhetoric**: it is strongest as a large, useful empirical resource showing that in-domain gains over U-Net are often modest and that zero-shot differences are more informative than conventional single-dataset leaderboards. However, its attempt to elevate this into a claim of *statistically rigorous and architecturally diagnostic benchmarking* is currently undermined by protocol ambiguity (official settings vs unified training), insufficiently justified significance testing, and a cohort-relative U-Score. In other words, the benchmark appears practically valuable, but the paper overstates how cleanly its methodology supports architecture-level and significance-level conclusions.

## Suggestions
- **Resolve the protocol ambiguity first.** Add a concise, explicit statement of the exact training/evaluation policy and revise all contradictory wording about “official predefined settings.”
- **Clarify the inferential target of the t-tests.** State exactly what is being paired, what randomness is being modeled, and what “statistical significance” should and should not be interpreted as in this benchmark.
- **Add multiple-testing correction** and revise the significance claims if needed.
- **Reframe U-Score more modestly** as a benchmark-relative deployment metric unless an absolute normalization scheme is introduced.
- **Condense and demote the advisor agent** unless stronger validation is added; the benchmark itself is already the main contribution.
- **Tighten the title/abstract language** around slice-based evaluation for volumetric datasets.

---

## kdAFb1lljm

- GT: Withdrawn (treated as Reject) (avg 1.3)
- Predicted: N/A (5.6/10)
- Match: N/A

### Final Review

## Summary
This paper proposes **Med-SegNet**, a compact encoder–decoder for binary medical image segmentation that inserts a single **Circulant Layer Token Mixer (CLTM)** at the bottleneck. The central empirical claim is that this lightweight, attention-free cross-scale mixer improves performance consistently across a broad suite of 20 public datasets while maintaining a very small model size (~2.07M parameters).

## Strengths
- **The paper demonstrates unusually broad within-paper validation of the proposed module across many medical domains.** The same architecture is evaluated on **20 datasets spanning 12 modalities**, and the ablation in Table 1 reports improvements on **20/20 datasets** when adding CLTM, with the mean Dice improving from **0.8977 to 0.9161**. Regardless of how one interprets external SOTA comparisons, this breadth of internal evaluation is specific and valuable.
- **The core architectural choice is targeted and efficient rather than brute-force.** CLTM is inserted **once at the bottleneck**, not throughout the network, and the paper gives a concrete complexity argument for the mixer: depthwise 1D circular convolution with parameter cost **\(k d\)** and mixing cost **\(O(Nkd)\)**. This is a specific design decision that plausibly explains why the full model remains at **~2.07M parameters**.
- **The empirical gains are most pronounced on precisely the difficult regimes where additional context should matter.** The largest reported ablation gains are on harder, low-contrast or structurally challenging datasets such as **BUSI (+6.31 Dice points)** and **RaViR (+6.12)**, while performance on easier near-ceiling datasets is largely preserved. This pattern is consistent with the intended role of the bottleneck mixer.
- **The paper is explicit about an important limitation instead of hiding it.** The conclusion clearly states that the evaluation is **confined to 2D inputs** and that robustness under distribution shift is not deeply analyzed. This scope clarity matters when judging the claims.

## Weaknesses

###: Fatal
- **The paper overclaims what CLTM actually computes: as specified, it is not a true single-step global interaction mechanism.**  
  In Section 3.4, the mixer is explicitly
  > “a depthwise one-dimensional circular convolution ... with learnable kernel of length \(k\)”  
  and the paper later states:
  > “We use \(k=5\) by default.”
  
  A single depthwise 1D convolution with fixed small kernel size has a **local receptive field along the token sequence**, even with circular padding. Circular padding wraps boundaries, but it does not make the operator globally dense in one pass. The text repeatedly describes CLTM as performing a **“single global information exchange,” “global token interaction,”** and as supplying **“global context”** in the same sense used to motivate replacing self-attention. That characterization is mathematically too strong for the operation actually defined. This does not mean the module is useless—the cross-scale concatenation and bottleneck placement may still help—but it **undermines the paper’s core conceptual framing** and some of its strongest claims.

### Major:
- **The external “state-of-the-art” comparisons in Table 2 are not methodologically strong enough to support the paper’s strongest comparative claims.**  
  The paper explicitly states in Table 2:
  > “Results for other methodologies are copied as reported in their original papers (not re-trained here). Our Med-SegNet results are produced under the unified setup described in Experimental Setup.”
  
  This means the model is being compared against numbers obtained under **different preprocessing, splits, resolutions, losses, and training schedules**. As a result, claims like “establishes a new benchmark,” “state-of-the-art,” or “decisively outperforms” are not adequately supported by Table 2 alone. The internal CLTM ablation is still meaningful, but the external superiority claims should be softened unless at least a few strong baselines are retrained under the same protocol.
- **The paper contains a nontrivial inconsistency in the training setup.**  
  Section 4 says:
  > “Adam optimizer (learning rate: 0.0175)”  
  whereas the appendix says:
  > “Adam (base learning rate \(7.5 \times 10^{-4}\)) and a cosine-decay schedule”
  
  This is a large discrepancy, not a minor typo, and it affects interpretation of the results and reproducibility of the reported numbers. Since the paper’s contribution is empirical and optimization-sensitive, this should be resolved clearly.
- **The ablation study is too shallow to isolate what aspect of CLTM is responsible for the gains.**  
  The paper only compares **with vs. without CLTM** across datasets. It does **not** ablate:
  - kernel size \(k\),
  - whether circular padding matters versus ordinary 1D conv,
  - whether cross-scale concatenation is necessary,
  - whether the gains come primarily from pre/post normalization and residual reprojection,
  - or whether a simpler bottleneck module with similar parameter count would achieve similar improvements.
  
  Because the central contribution is a specific mixer design, these missing controls matter. Without them, the evidence supports “this added bottleneck module helps,” but does not yet convincingly establish that the **circulant cross-scale design itself** is the key reason.
- **Efficiency claims are only partially substantiated.**  
  The paper argues that CLTM is near-linear and hardware-friendly, and the appendix does provide some runtime information on TPU (step times and a test-set throughput figure). However, the main paper does not provide a clean comparative table of **latency / FLOPs / peak memory** against baselines, nor scaling with input resolution. Given the emphasis on “practical latency,” “low memory,” and “hardware-friendly deployment,” stronger empirical efficiency evidence is needed, especially for comparison-driven claims.

### Minor
- **The paper’s novelty is somewhat incremental at the mechanism level.**  
  The work is a sensible adaptation of attention-free token mixing ideas to a medical segmentation bottleneck, but the mixer itself is not a large conceptual leap over existing structured/token-mixing approaches. The practical integration is more convincing than the underlying methodological novelty.
- **The significance is limited by the 2D-only scope.**  
  The paper acknowledges this limitation. Since many high-impact medical segmentation settings are volumetric, the current contribution is better viewed as a promising 2D segmentation architecture than a broadly complete medical segmentation solution.
- **Some claims of “statistically meaningful” gains are not fully supported by the reported evidence.**  
  The paper says the improvements are “statistically meaningful,” but no statistical testing, variance estimates, or multi-seed results are shown in the main text or appendix excerpt provided. The empirical trend is encouraging, but that wording is stronger than the presented evidence.

### Trivial
- None.

## Nice-to-Haves
- Retrain a **small but strong subset of baselines** (e.g., U-Net/UNet++, TransUNet or Swin-UNet, and one recent efficient mixer/SSM model) under the paper’s exact training protocol to make the comparative section much more credible.
- Add a **component-level ablation** for CLTM: vary kernel size \(k\), remove cross-scale concatenation, replace circular conv with plain 1D conv, and test normalization variants.
- Replace “global” phrasing with more precise language unless a larger-kernel / multi-hop / globally dense variant is implemented and validated.
- Provide a compact **Pareto table or plot** showing Dice vs. parameters/FLOPs/latency.
- Include a short discussion or visualization clarifying how token linearization interacts with 2D spatial structure. This is not currently a fatal flaw, but it would help interpret what the mixer is actually learning.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that CLTM’s 1D flattening is inherently destructive or invalid for 2D segmentation.**  
  This concern is plausible as a question, but the harsh review stated it too strongly as a fundamental flaw. The paper does flatten multi-scale features into sequences and mix them with 1D convolution, but many successful vision architectures also tokenize spatial maps. The current evidence does not justify calling this inherently broken.
- **Complaint that some external comparisons may involve models with different task dimensionality or setup specifics.**  
  The broader concern about unfair external comparison is valid and kept, but specific claims such as “those models are often 3D” are not verified from the paper and should not be asserted.
- **Criticism about missing related works.**  
  Per instruction, this is removed. The review should not fault the paper for omitted literature that cannot be externally verified here.
- **SE reduction ratio \(R=24\) as an intrinsic design flaw.**  
  The paper gives this value, but there is no evidence in the manuscript that it is unreasonable or harmful. The real issue is lack of ablation, not the number itself.
- **Claim that there are zero efficiency benchmarks.**  
  This is factually incorrect. The appendix does report TPU timing/throughput information. The valid criticism is that comparative efficiency evidence is insufficient, not absent.
- **Demand for complete training logs.**  
  This is a reproducibility nitpick beyond normal submission standards and is not necessary for the core scientific evaluation.

## Novel Insights
The strongest synthesis across the reviews is that this paper is **better empirically than conceptually framed**. The evidence that “adding this bottleneck module helps across many datasets” is fairly persuasive, especially because the gains are broad and largest on hard cases. But the paper’s conceptual sales pitch—that a single small-kernel circular depthwise convolution constitutes a true global interaction mechanism replacing self-attention—is overstated. If reframed more honestly as an efficient **cross-scale bottleneck mixer with local sequence mixing and broad empirical utility**, the work would read as a more credible and solid engineering contribution.

## Suggestions
- **Reframe the core claim**: avoid describing CLTM as a true one-step global interaction module unless the operator is changed; present it instead as an efficient cross-scale bottleneck mixer.
- **Fix the learning-rate inconsistency** between Section 4 and the appendix, and ensure the final camera-ready text has one unambiguous training protocol.
- **Strengthen Table 2** by retraining a representative subset of strong baselines under the same setup; otherwise, tone down SOTA language.
- **Run targeted CLTM ablations**: \(k\in\{3,5,7,9\}\), circular vs. standard padding, single-scale vs. cross-scale mixing, and with/without pre/post normalization.
- **Add comparative efficiency evidence** in the main paper: FLOPs, peak memory, and latency at one or two standard resolutions.
- **Clarify the contribution axis in the narrative**: the paper appears strongest on empirical robustness and parameter efficiency, and weaker on fundamental novelty and theoretical justification.

---

## 5o0zF03RP9

- GT: Withdrawn (treated as Reject) (avg 0.5)
- Predicted: N/A (2.4/10)
- Match: N/A

### Final Review

## Summary
This paper proposes IncentRL, an RL framework that augments extrinsic reward with a KL penalty between a predicted outcome distribution \(p(o\mid s,a)\) and a preferred outcome distribution \(q(o\mid s)\). Its headline claim is that the weighting coefficient \(\beta\) is adapted online via a Bayesian mechanism, thereby removing manual tuning of the trade-off between external reward and internal incentive. Empirically, the paper reports improvements over \(\beta=0\) and some fixed-\(\beta\) baselines on a 2-state toy MDP, MountainCar, and MiniGrid DoorKey.

## Strengths
- **The shaping objective is concrete and interpretable.** The paper clearly defines the core reward modification,
  \[
  r_{\text{shaped}} = r_{\text{ext}} - \beta\, \mathrm{KL}(p(o\mid s,a)\|q(o\mid s)),
  \]
  and this formulation is easy to understand as a preference-alignment signal rather than an opaque auxiliary loss. This is a real conceptual contribution relative to generic “intrinsic reward” formulations.

- **The paper is unusually explicit about failure modes of its own mechanism.** Section 6.2 directly acknowledges preference misalignment, KL dominance, and latent mismatch as instability sources, which is important here because the empirical results do in fact show that overly large \(\beta\) can be harmful.

- **The experiments do provide some evidence that mild KL shaping can help in sparse-reward settings.** On MountainCar, \(\beta=0.1\) improves over \(\beta=0\), while larger \(\beta\) hurts sharply; on MiniGrid Doorkey, a small fixed \(\beta=0.01\) improves final success rate and episode length over the unshaped baseline. Even if the broader claims are overstated, these results do support the narrower claim that this shaping term can be useful when tuned appropriately.

- **The theoretical discussion correctly captures the boundary behavior of the shaping coefficient.** The small-\(\beta\) perturbation intuition and the large-\(\beta\) domination intuition are sensible and help frame how the method interpolates between standard RL and pure preference matching, even though the analysis is limited.

## Weaknesses

###: Fatal
- **The paper’s central novelty—Bayesian adaptation of \(\beta\)—is not actually specified as a method.** This is the most serious issue and it directly undermines the main claimed contribution. The abstract, introduction, and contribution list all center the paper on “treating the incentive weight \(\beta\) as a Bayesian random variable, updated online,” but the paper never defines:
  - a prior over \(\beta\),
  - a likelihood or observation model,
  - an inference/update rule,
  - or an algorithmic procedure that could be implemented or analyzed.

  The paper repeatedly refers to “posterior concentration” (e.g., in the abstract and Section 5.4), but nowhere explains how that posterior is formed. Section 5.4 says:
  > “we tracked the posterior evolution of \(\beta\) using our Bayesian adaptation scheme”

  yet no such scheme is described in Methods or elsewhere. As written, the adaptation mechanism is not reproducible, not evaluable, and not distinguishable from a heuristic schedule. Because this mechanism is presented as the central novelty, this is a fundamental problem rather than a missing detail.

- **The main empirical claim about eliminating manual tuning is not supported by presented results.** The paper’s strongest claim is that Bayesian adaptation “removes the need for manual trade-off tuning.” However, the evidence actually shown is only:
  - fixed-\(\beta\) sweeps for toy/MountainCar,
  - a fixed \(\beta \in \{0, 0.01\}\) comparison for MiniGrid,
  - and plots/statistics of sampled \(\beta\) values.

  Critically, Section 5.4 states:
  > “In additional runs (not shown), the Bayesian adaptation of \(\beta\) achieved performance comparable to the best fixed value…”

  This is not sufficient for a central claim. The paper should directly report adaptive-\(\beta\) return/success curves against the best fixed \(\beta\) on the same tasks. Without those numbers, the claimed benefit of the adaptation mechanism is not established.

### Major:
- **There is substantial ambiguity in how \(p(o\mid s,a)\) and \(q(o\mid s)\) are instantiated in the actual experiments.** The method depends critically on these distributions, but the experimental sections do not specify them with enough precision. Section 3.1 says \(p\) “may be obtained from a forward model or from environment dynamics,” and Section 4.3 lists several possibilities for \(q\), but the paper never clearly states what concrete representation/model is used in MountainCar and MiniGrid, how KL is computed in those settings, and whether outcomes are discrete states, latent embeddings, or something else. This is not a trivial implementation nit: these choices define the method being evaluated.

- **The empirical evaluation is too limited to support the paper’s broader significance claims.** The benchmarks are a 2-state MDP, MountainCar-v0, and MiniGrid DoorKey 8x8. These are sufficient for an initial proof of concept, but they do not justify statements in the abstract/introduction/conclusion about solving a “core limitation of intrinsic motivation methods,” improving “long-term planning,” or paving the way toward “more autonomous and general-purpose RL agents.” The observed gains are on small-scale sparse-reward tasks, and the paper should calibrate its claims accordingly.

- **The results reveal strong sensitivity to \(\beta\), but the paper does not analyze whether the adaptive mechanism can reliably avoid harmful regimes.** This is not merely a generic hyperparameter complaint; it is directly relevant because adaptation of \(\beta\) is the headline contribution. In Table 2, MountainCar performance drops dramatically as \(\beta\) increases from 0.1 to 0.3 and 1.0. This makes the quality and speed of adaptation crucial, yet the paper provides no operational account of how the adaptive scheme responds to such sensitivity.

- **There is an internal inconsistency in the reported adaptive-\(\beta\) story.** Section 5.4 says Figure 3 shows the posterior mean concentrating near the effective region “\(\beta \approx 0.1\),” but Appendix Table 3 reports round means decreasing from 0.1057 to 0.0454 to 0.0194 to 0.0173. Those later values are not “near 0.1.” This inconsistency matters because the adaptation mechanism is already underspecified; contradictory summaries of where \(\beta\) is supposedly concentrating further weaken confidence in the result.

- **The cognitive/neuroscience framing is suggestive rather than technically substantiated.** The paper repeatedly invokes dopamine-based RPE and the Free Energy Principle as theoretical grounding, but these connections remain analogical. The actual method is a KL-shaped reward objective; the paper does not derive this objective from a neuroscientific model nor establish more than high-level resemblance. This would be acceptable as motivation, but the current presentation occasionally overstates it as a theoretical contribution.

### Minor
- **The theoretical analysis is limited in scope and rigor relative to the claims.** Proposition 1 and Proposition 2 are intuitively reasonable, but they are only sketched and mostly characterize boundary cases. They do not analyze the actual adaptive-\(\beta\) procedure, nor do they provide insight into learning dynamics in the practical settings used in the experiments.

- **Robustness to misspecified preferences is discussed but not tested.** Section 6.2 appropriately notes that poor or unreachable \(q(o\mid s)\) can misguide learning, yet there is no ablation studying how sensitive the method is to preference misspecification. Since the entire framework depends on \(q\), this is a meaningful missing analysis.

### Trivial
- None.

## Nice-to-Haves
- Add direct learning-curve and final-metric comparisons between adaptive \(\beta\), the best fixed \(\beta\), and \(\beta=0\) on every benchmark.
- Provide a full mathematical specification and pseudocode for the Bayesian update, including prior, likelihood, approximate inference method, and computational overhead.
- Expand evaluation to more challenging sparse-reward benchmarks to test whether the observed benefits scale beyond proof-of-concept environments.
- Include an ablation on the quality/misspecification of \(q(o\mid s)\).
- Analyze scale alignment between extrinsic reward and KL penalty, since the collapse at high \(\beta\) suggests reward-scale mismatch may be important.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Missing related work / omitted baselines as a literature complaint.** Several reviews asked for comparisons to specific external methods. It is fair to say the current comparisons are limited, and that broader comparisons would strengthen the paper, but I am not retaining claims framed as “the paper fails to cite/discuss X” because that cannot be verified here under the reviewing constraints.

- **Pure reproducibility complaints about code availability or repository access.** The paper cites a code repository, so concerns rooted in doubting the code’s existence or accessibility should be disregarded.

- **Formatting/style concerns from the parsed text.** Any apparent notation glitches or PDF extraction artifacts are not valid weaknesses.

- **Claims that the method is “mathematically identical” to existing approaches in a way requiring external confirmation.** What can be supported from the paper alone is that the KL-shaped objective is not by itself a radical departure; however, stronger novelty disputes relying on outside literature matching are removed.

## Novel Insights
The paper is stronger as a paper about **preference-conditioned reward shaping** than as a paper about **Bayesian adaptation**. The fixed-\(\beta\) experiments and the discussion of failure modes together suggest a credible narrow message: KL alignment between predicted and preferred outcomes can be a useful dense signal, but only in a fairly delicate reward-scaling regime. In contrast, the claimed adaptive mechanism—the part that would make this more than a modest shaping paper—is not only under-evaluated but actually unspecified. So the real mismatch is not between theory and experiment, but between the paper’s **true evidential center of gravity** (a shaped reward idea with tuned \(\beta\)) and its **marketed center of gravity** (a Bayesian online adaptation method).

## Suggestions
- **Define the Bayesian method fully.** Add the prior \(p(\beta)\), the observation model, the posterior approximation/update, and pseudocode. If the current implementation is heuristic rather than Bayesian, rename it honestly.
- **Show adaptive-\(\beta\) performance directly.** For each task, report returns/success rates for adaptive \(\beta\), best fixed \(\beta\), and \(\beta=0\), with matching training budgets and seeds.
- **Clarify experimental instantiations of \(p\) and \(q\).** State exactly what outcomes are, how the predictive model is constructed, how preferences are encoded, and how KL is computed in each benchmark.
- **Tone down claims of generality unless supported by broader evidence.** The current evidence supports a proof-of-concept on small sparse-reward tasks, not a general solution to intrinsic reward trade-off tuning.
- **Add robustness analysis for preference misspecification and reward/KL scaling.** This would materially improve the technical story because these are the most plausible failure modes of the framework.
- **Reframe the neuroscience discussion as motivation rather than theory unless a formal bridge is added.** This would make the paper more precise and more credible.

Overall, the paper contains an interesting shaping idea and some encouraging tuned-\(\beta\) results, but in its current form the main claimed contribution is not actually specified or empirically demonstrated. The mismatch between the headline claim and what is concretely delivered is severe.

---

## b8TlYh6PN6

- GT: Accept (Oral) (avg 8.0)
- Predicted: N/A (7.4/10)
- Match: N/A

### Final Review

## Summary
This paper studies distributional equivalence for linear non-Gaussian latent-variable causal models with arbitrary cycles. Its central contribution is a graphical characterization of when two such models induce the same observed distribution set, built around a new “edge rank” perspective and yielding both a decision criterion (Theorem 2) and a transformational characterization for traversing the equivalence class (Theorem 3). The paper also proposes a proof-of-concept discovery pipeline, glvLiNG, that uses OICA-estimated mixing matrices to recover the class up to this equivalence.

## Strengths
- **A genuinely new equivalence characterization in a difficult regime.** The paper tackles a setting that combines latent variables, cycles, linearity, and non-Gaussianity, and provides a concrete characterization of distributional equivalence rather than only identifiability in special cases. The progression from Definition 1 / irreducibility to Lemmas 1–5 and then Theorem 2 is a substantial theoretical contribution.
- **The singleton decomposition in Theorem 2 is especially strong.** The paper starts from an apparently intractable condition involving all subsets and permutations (Lemma 3 / Lemma 5), then shows that equivalence can be checked via the children-bases of just \(L\) and each \(L\cup\{X_i\}\). This is the cleanest and most operational result in the submission.
- **The edge-rank viewpoint is insightful and potentially useful beyond this paper.** Definitions 4–6 and Theorem 1 provide a local, matching-based dual to path ranks that the paper uses effectively to simplify equivalence reasoning. Even though the underlying matroid ideas are classical, packaging them as edge-rank constraints for latent-variable causal discovery is a meaningful conceptual contribution.
- **The paper gives both static and transformational characterizations.** Beyond deciding equivalence, Theorem 3 shows how equivalent graphs are connected by admissible cycle reversals and edge additions/deletions. This mirrors the role that covered edge reversals play in simpler settings and gives the work a satisfying structural completeness.
- **The authors are appropriately explicit that the algorithmic part is a proof of concept.** Section 5 and Section 6 clearly state that the main focus is the equivalence characterization, and that the reliance on OICA is a limitation of the current instantiation rather than something hidden from the reader.

## Weaknesses

###: Fatal
- None.

### Major:
- **The empirical algorithm is much less compelling than the theory, because its guarantees depend on oracle/exact rank information while the practical implementation uses heuristic rank handling.** The formal guarantee in Section 5 is explicitly stated only “under the assumptions of access to an oracle OICA and faithfulness,” and Appendix D.4 then explains that, in practice, the method assigns a confidence score based on the minimum singular value and thresholds/approximates ranks:
  > “we assign a ‘full-rank confidence score’ … use the score \(1/(1+\exp(-\alpha^{-1}(\sigma_{\min}-\epsilon)))\)”  
  > “in phase 1 … we approximate the closest valid transversal matroid … In phase 2 … we simply threshold these scores”
  
  This means the finite-sample pipeline is not the same object as the provably correct oracle procedure. That does not invalidate the theoretical contribution, but it does weaken the claim of a practically realized discovery method, especially since no analysis is given for when these heuristics preserve the needed matroid/basis structure.
- **The paper’s practical impact is limited by its dependence on OICA, which the paper itself acknowledges as a bottleneck.** This is not a misunderstanding: Section 5 explicitly says,
  > “One may be concerned about OICA’s known inefficiency in practice”  
  and Section 6 lists as a limitation:
  > “One limitation is the use of OICA in glvLiNG”
  
  Since the method’s input is an estimated mixing matrix and OICA quality directly controls all downstream rank queries, the end-to-end practicality is constrained by a difficult upstream estimation problem. The paper is honest about this, but the limitation remains substantial.
- **The evaluation does not yet convincingly connect the elegant equivalence theory to reliable finite-sample behavior.** Figure 7 and Appendix D.4 show simulation results, but the paper does not provide targeted robustness analysis for the key failure mode: small rank perturbations or near-violations of genericity in the estimated mixing matrix. Given that the theory hinges on exact/generic rank patterns, the missing experiment is not “more experiments” in the abstract; it is specifically the one that would test whether the theorem-derived structure survives estimation noise.

### Minor
- **The method is not uniformly strong empirically; on sparse graphs, structurally constrained baselines can perform better.** Appendix D.4 explicitly notes:
  > “when \(d=1\), these two methods perform better than glvLiNG”
  
  This is a fair and informative result, and the paper discusses it reasonably. Still, it means the practical case for a structural-assumption-free pipeline is strongest in misspecified/dense regimes rather than broadly dominant.
- **The output equivalence classes can be very large, and the paper does not fully elevate its own summary representation into the main practical story.** The paper usefully quantifies class sizes and even proposes a CPDAG-like presentation in Appendix C.3 / Theorem 4, but the main algorithmic narrative still centers on recovering/traversing the full class. For applied use, the summary object seems more important than raw enumeration, yet it remains relegated to the appendix.
- **Scalability claims are only partially substantiated.** The runtime evidence in Table 4 is encouraging for the rank-realization component, but it is based on oracle mixing matrices and small-to-moderate graph sizes. The paper does not provide complexity bounds or convincing evidence for larger instances where matroid operations and traversal could become difficult.
- **The term “structural-assumption-free” may invite overreading unless repeatedly contrasted with the strong parametric assumptions.** The paper is careful in places, but because the setting still assumes linearity and non-Gaussianity, a bit more explicit wording in the abstract/introduction would help avoid confusion.

### Trivial
- None.

## Nice-to-Haves
- Add a controlled ablation that perturbs oracle rank queries or OICA-estimated mixing matrices, to show how sensitive the recovered equivalence class / summary graph is to rank errors.
- Promote the CP-like presentation from Appendix C.3 into the main text and evaluate it as the primary practical output, not just traversal of all equivalent graphs.
- Provide explicit complexity discussion for Phase 1, Phase 2, and equivalence-class traversal, even if only coarse worst-case and empirical scaling.
- Analyze robustness as the exogenous noise becomes closer to Gaussian, since the entire pipeline rests on the non-Gaussian identifiability regime.
- Add stability analysis for the real-data graph (e.g., bootstrap edge frequencies), especially for the “solid vs dashed” interpretation.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that the algorithm fundamentally cannot be executed because the latent/observed partition of OICA columns is unresolved.** This criticism is not supported by the paper. Appendix A explicitly addresses the permutation ambiguity by treating the recovered OICA matrix as having columns indexed by \(V\) only up to an unknown permutation \(\pi\), and reconstructing a binary support matrix \(Q\) whose row permutation can then be adjusted to yield a valid digraph:
  > “there exists a permutation \(\pi\) of \(V\)… there exists an unknown binary matrix \(Q\)… As long as one can recover this matrix \(Q\), one can then permute its rows to place nonzero entries on the diagonal”
  
  The method is designed around the fact that OICA only identifies columns up to permutation/scaling. One can debate practicality, but not claim that the paper forgot this issue.
- **Criticism that runtime claims are “misleading” because Table 4 uses oracle mixing matrices.** The paper is explicit about what Table 4 measures:
  > “constructing digraphs that satisfy the rank constraints of oracle OICA mixing matrices”
  
  This is a fair component-level runtime evaluation, not a hidden end-to-end claim.
- **Weakness framed as lack of a single summary object for the equivalence class.** The paper actually does discuss such a summary in Appendix C.3 / Theorem 4. It is fair to say this should be foregrounded more, but not fair to say the paper lacks it.
- **Strengths like “the paper is well-written” / “the topic is important” / “experiments are extensive.”** These are too generic and were omitted.

## Novel Insights
The strongest synthesis across the reviews is that this paper is best viewed as a theory-first contribution with a nontrivial but still immature algorithmic instantiation. The real advance is not merely “using ranks” but showing that in this latent cyclic LiNG setting, distributional equivalence can be reduced all the way from mixing-matrix closure to path-rank equality, then dualized to edge-rank/matroid structure, and finally localized to singleton checks in Theorem 2. That localization is what turns a conceptually complete but unusable characterization into something structurally comparable to classical equivalence criteria. At the same time, the paper itself supplies the seeds of its most practical next step: Theorem 4’s summary representation is probably the right output object for applications, more so than enumerating large equivalence classes.

## Suggestions
- Make the paper’s positioning sharper: present it explicitly as a **theoretical characterization with a proof-of-concept recovery pipeline**, rather than letting readers infer stronger practical maturity than the current empirical section supports.
- Add a dedicated robustness section that perturbs ranks directly and reports degradation in recovered bases / equivalence summaries.
- Move the Appendix C.3 presentation result closer to the main text and evaluate it on both synthetic and real data as the practical output.
- Give explicit complexity bounds or at least more detailed empirical scaling for Phase 1, Phase 2, and traversal.
- In the abstract and introduction, clarify that “structural-assumption-free” refers to latent-graph structure, not to absence of parametric assumptions.

---

## pzXAS6Tf2r

- GT: Reject (avg 4.8)
- Predicted: N/A (6.1/10)
- Match: N/A

### Final Review

## Summary
This paper proposes HiViBiX, a mono-to-binaural music generation system that predicts internal FOA/Ambisonics-like channels and uses a hierarchical visual encoder combining global scene cues with local person-centered crops, positional encoding, and depth. Empirically, the method is strong: across FAIR-Play and Music-Stereo it improves clearly over prior mono-to-binaural baselines, and on YT-Music it improves STFT/ENV but not every reported metric.

## Strengths
- **The paper introduces a distinctive intermediate representation rather than directly predicting stereo.** Instead of outputting left/right spectrograms, it predicts internal \(X,Y\) channels and combines them with the mono input treated as \(W\) through an Ambisonics-inspired decoding layer (Section 3.3, Algorithm 1). Even if the representation is only partially physically grounded, this is a concrete modeling choice that differs from standard direct stereo prediction and is supported by the ablation: removing the Ambisonics-style representation substantially hurts performance (Table 3: STFT degrades from 0.669 to 1.492 on the FAIR-Play ablation split).
- **The hierarchical visual conditioning is more than generic image conditioning and appears genuinely useful.** The HiVi encoder combines global CLIP scene features, local YOLO-based crops, positional Fourier features, and depth features, with a cross-attention-style hierarchical aggregation (Section 3.2). The ablations support that these are complementary rather than decorative: removing CLIP, depth, position, or the hierarchical aggregation each degrades STFT/ENV/SNR (Table 3).
- **Empirical gains on the main mono-to-binaural benchmarks are substantial on several datasets/metrics.** On FAIR-Play 10-split, HiViBiX improves STFT from the strongest prior reported 0.787/0.823 range to 0.6319 and slightly improves ENV/SNR as well (Table 1). On Music-Stereo, the gains are larger, reaching 0.331 STFT / 0.070 ENV / 14.363 SNR versus CCStereo’s 0.624 / 0.097 / 12.985 (Table 2). These are meaningful improvements, not just marginal noise.
- **The paper includes useful component analysis and some effort at validation beyond the main tables.** There is an ablation study over representation, loss, and visual components (Table 3), a comparison between learned channels and real Ambisonics channels on YT-Music (Appendix E), a small user study (Appendix F), and explicit failure-case discussion for dynamic videos (Section 5.1, Appendix G). This gives a more informative picture than papers that only provide one headline table.

## Weaknesses

### Major:
- **The strongest conceptual claim is overstated relative to what is actually implemented.** The paper repeatedly suggests that predicting FOA-like channels yields “explicit control” over spatial positioning and invokes Ambisonics as a physically grounded representation. However, in the actual model the “Ambisonics FiLM” layer uses learned global coefficients \(\hat\alpha,\hat\beta\) together with predicted \(X,Y\) spectrograms to reconstruct binaural magnitude and phase (Algorithm 1), and the learned channels are not supervised as true FOA on the main datasets. The paper itself acknowledges this limitation: “the Ambisonics-like format is only enforced by the Ambisonics FiLM layer… due to lack of real Ambisonics data” (Section 5.1). So the representation is better described as **Ambisonics-inspired internal parameterization** than as a demonstrated explicit/physical spatial control mechanism. This matters because the paper’s novelty narrative leans heavily on the stronger interpretation.
- **The evaluation is not well matched to the paper’s core spatial claims.** The main metrics are STFT distance, envelope distance, and SNR (Section 4.2 / Appendix C). These are reconstruction metrics, but the paper’s claims emphasize “more precise spatialization,” “explicitly control the spatial positioning,” and richer multimodal grounding. The paper includes only a small subjective study (13 users; Appendix F) and no objective spatial metrics such as localization-related errors, interaural cue accuracy, or angle-conditioned spatial consistency. As written, the experiments convincingly show improved **signal reconstruction** of binaural outputs, but they do not fully validate the stronger claims about spatial control/localization quality.
- **The method uses only a single anchor frame for conditioning, which is a real limitation for video-conditioned spatial audio.** Section 3.2 explicitly extracts one anchor image, and Section 5.1 concedes that “rapidly changing videos can also induce problems,” with Appendix G showing a failure case. Given that the task is framed as video-conditioned binaural generation and some datasets contain motion or changing viewpoints, this is not a minor caveat. It weakens the method’s validity beyond relatively static scenes and limits how broadly one can interpret the reported gains.
- **The claim of state-of-the-art performance “across all datasets” should be toned down.** It is true for STFT/ENV on the main datasets, but not literally for every reported metric: on YT-Music, CCStereo has higher SNR (8.245 vs 7.805 in Table 2). This does not negate that the method is very competitive, but the universal phrasing is stronger than the evidence supports.

### Minor
- **The local visual extraction heuristic is somewhat brittle.** The model uses YOLO detections restricted to the **person** class as proxies for sounding objects (Section 3.2). The paper itself notes this can be “incorrect or ambiguous” and can miss sounding objects such as speakers (Section 5.1). This is a reasonable design choice for music datasets with visible performers, but it may limit generalization and contributes to the dependence on dataset bias.
- **The user study is supportive but too small and under-specified to carry much weight.** Appendix F reports 13 participants rating 20 videos for “spatiality” on a 1–5 scale and says the proposed method ranks second after ground truth. As evidence, this is useful but exploratory rather than conclusive, especially for a paper whose main claims concern perceptual spatial quality.
- **Some comparisons are incomplete or mix protocols.** Several baseline entries in Tables 1–2 are missing, limiting the strength of the “new SOTA” framing. The paper says results are taken from prior papers or public reimplementations “where possible,” so this is not misconduct, but it means the comparisons are not uniformly controlled. This is especially relevant because many gains on some metrics are modest, while the strongest gains appear on others.

### Trivial
- **The qualitative analysis itself notes audible/spectral artifacts without deeply analyzing them.** Section 4.3 mentions “smothering” of high frequencies and persistence of low-energy regions. This does not undermine the main empirical results, but a more direct discussion of how these artifacts affect perceived binaural quality would improve the paper.

## Nice-to-Haves
- Add binaural/spatial evaluation metrics or analyses tied to localization cues, not just reconstruction fidelity.
- Include an ablation comparing the Ambisonics FiLM layer to a generic conditioning/fusion layer, to test whether the physics-inspired structure itself is necessary.
- Expand analysis on dynamic scenes by aggregating multiple frames or adding temporal visual modeling.
- Report variance across seeds for the main tables if feasible, especially where margins are small.
- Quantify runtime/model complexity, since the method combines several pretrained vision components with custom audio networks.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that the method is unfairly compared to video-to-spatial-audio generators in Appendix H.** This comparison is indeed apples-to-oranges, but it is placed in an appendix explicitly framed as comparison with generative methods and the paper itself acknowledges the mismatch (“our distance-based metrics have a higher value for them”). It should not be elevated as a central weakness of the main paper.
- **Criticism that baseline asymmetry invalidates the comparison because some baselines are missing or use prior reported numbers.** The paper is transparent: “All baseline results were either taken from the respective papers or obtained from publicly available re-implementations, where possible.” This weakens the conclusiveness of the SOTA claim, but not the validity of the empirical evidence altogether.
- **Reproducibility nitpicks about the exact user-study protocol details or missing training logs/hyperparameters.** The paper already provides substantial implementation detail in Sections 4.1 and Appendix A/C; while the user study is limited, demanding exhaustive protocol details here would be out of proportion.
- **Generic strength that the paper is simply ‘well written’ or ‘experiments are extensive.’** These are too generic to keep as standalone strengths.

## Novel Insights
The most interesting synthesis is that the paper’s empirical contribution looks stronger than its conceptual framing. The results and ablations support that the proposed **factorization of binaural generation into mono prior + learned directional channels + hierarchical visual grounding** is useful. But the evidence does **not** yet establish that this factorization inherits the physically meaningful controllability of true Ambisonics. In other words, the paper may be best understood not as a validated Ambisonics rendering method, but as a strong inductive bias for spatial audio generation that borrows Ambisonics structure. Framing it this way would make the contribution more precise and, arguably, more credible.

## Suggestions
- Reframe the paper more carefully around **Ambisonics-inspired internal representation** rather than strong claims of explicit physical spatial control, unless you can directly validate controllability.
- Add at least one evaluation directly tied to spatial fidelity, such as interaural cue accuracy, localization-related measures, or controlled angle-dependent decoding analysis.
- Temper the “state-of-the-art across all datasets” language to reflect that YT-Music SNR is not best, even though STFT/ENV are.
- Strengthen the temporal story: either narrow the scope to mostly static scenes or add a simple multi-frame aggregation experiment to show the method can handle motion better.
- Analyze when the person-only detection heuristic fails, ideally with examples separating visual detection failures from audio-model failures.
- If space permits, compare the Ambisonics FiLM reconstruction against a simpler learned stereo decoder to show that the proposed structure itself, not only the richer vision backbone, is responsible for the gains.

---

## lTaPtGiUUc

- GT: Accept (Oral) (avg 7.3)
- Predicted: N/A (7.3/10)
- Match: N/A

### Final Review

## Summary
This paper proposes LPWM, an end-to-end self-supervised object-centric world model built on latent particles, with a key new ingredient: continuous **per-particle latent actions** learned jointly with dynamics. The model supports multiple conditioning modes (actions, language, image goals, multi-view), shows strong stochastic video prediction results across several synthetic and real robotic datasets, and is further adapted to goal-conditioned imitation learning.

## Strengths
- **The per-particle latent action formulation is a real and meaningful technical contribution.** The paper clearly distinguishes LPWM from prior global latent-action models by learning a latent action for each particle and regularizing inverse dynamics with a learned latent policy prior. This is not just a framing change: the design is motivated by multi-entity stochasticity, and the ablation in Table 11 supports that per-particle latent actions materially improve prediction quality over global alternatives.
- **The paper successfully scales a particle-based object-centric model to more complex real and stochastic video settings than prior DLP/DDLP-style methods.** A central engineering advance is removing explicit tracking and enabling parallel frame encoding while still preserving structured particle attributes. This appears to be what makes training on datasets like BAIR, Bridge, LanguageTable, and Mario practical.
- **Empirical video modeling results are strong and broad.** On stochastic settings (Table 2 / Table 10), LPWM consistently improves over the patch baseline DVAE and substantially outperforms PlaySlot where compared, especially on perceptual quality and FVD. On deterministic settings (Table 8), LPWM is also competitive with or slightly ahead of DDLP and other baselines, though not uniformly dominant.
- **The paper demonstrates unusual modality flexibility within one object-centric framework.** The same model family supports unconditional stochastic prediction, action conditioning, language conditioning, image-goal conditioning, and multi-view training. That breadth is specific and notable, not a generic “many experiments” strength.
- **The downstream imitation-learning application is more than a toy add-on.** LPWM is actually used to generate goal-conditioned imagined latent trajectories and then map latent actions to environment actions. Results are mixed but nontrivial: LPWM is strong on some OGBench tasks and competitive on PandaPush, especially given the relatively simple action-mapping head.
- **The paper provides unusually detailed methodological exposition.** The appendix includes loss derivations, module-level descriptions, and implementation-style pseudocode for key components. While not every mechanism is equally explicit, the paper is substantially more transparent than average.

## Weaknesses

### Fatal
None.

### Major:
- **The paper’s decision-making claim is weakened by a clear train/test mismatch in the latent-action pipeline.** In Appendix A.5, the policy-mapping network is trained on latent actions from the **inverse dynamics** head, but at planning/inference time the model must generate trajectories using latent actions sampled from the **latent policy prior**. The authors explicitly acknowledge the issue:  
  > “we empirically found that directly using the latent policy outputs for mapping degrades downstream performance; the mapping network performs best when evaluated on the outputs of the latent inverse module”  
  This is a substantive limitation because it means the world model’s own prior is not yet a reliable control interface for downstream action prediction. The paper still demonstrates useful downstream transfer, but the stronger implication that LPWM is already a robust planning-ready latent-action model is not fully supported.
- **The “particle-grid regime” is a real tradeoff that blurs the paper’s object-centric claim, and the consequences are not analyzed deeply enough.** The paper is transparent that LPWM no longer tracks globally free-moving particles as in DDLP. Appendix A.4.4 states:  
  > “each particle is constrained to move only within a local region around its original patch center, and when it reaches the limits of this region, its features are transferred to nearby particles.”  
  This does not make the method “not object-centric,” but it does mean LPWM occupies a hybrid regime between patch tokens and globally persistent object particles. That tradeoff is plausible for scalability, yet the paper does not quantify when it breaks: e.g., under large object displacements, sustained occlusions, or repeated cross-patch handoffs. Since object permanence and decision-making interpretability are part of the motivation, this limitation deserves more direct empirical characterization.
- **The imitation-learning evidence is promising but uneven, so the decision-making significance should be stated more cautiously.** The results are not uniformly strong across tasks. On PandaPush, LPWM is competitive but clearly below EC Diffuser on the harder 2-cube and 3-cube tasks (74 vs 91.7, and 62.1 vs 89.4). On OGBench-Scene, LPWM is excellent on task1 and task3, but very weak on task2 (6±9 vs 81±7 for HIQL), and overall trails HIQL (40±1 vs 49±4). This does not negate the downstream contribution, but it does mean the current evidence supports **viability** for decision-making more than broad superiority.

### Minor
- **The paper does not directly measure or visualize the failure modes of the particle-grid mechanism.** The handoff behavior is described qualitatively in Figure 13 and text, but there is no targeted analysis of identity preservation across patch boundaries, long-range motion, or cluttered occlusion cases. Given how central this design is, a dedicated stress test would strengthen technical soundness.
- **Language-conditioned evaluation is somewhat under-validated semantically.** The paper reports FVD for stochastic language-conditioned generation and visual metrics for posterior-conditioned reconstruction (Table 10), but these do not directly test whether generated trajectories obey the language instruction. Since language grounding is one of the marketed conditioning modes, a task-aware semantic adherence metric or instruction-success proxy would strengthen the claim.
- **Efficiency claims are plausible but not fully substantiated by direct runtime/compute comparisons.** The introduction motivates LPWM as more efficient than diffusion-style world models, and the architecture is clearly more compact. However, the paper does not provide inference-time latency, rollout throughput, or memory comparisons against DVAE or larger video models. For a systems-oriented motivation tied to decision-making usability, this would be useful evidence.
- **The analysis of downstream failure modes is too limited.** In OGBench especially, LPWM swings from excellent to very poor depending on task, but the paper offers only high-level explanations about play data and task complexity. A more specific diagnosis of which behaviors the latent model fails to represent would improve clarity and credibility.

### Trivial
- **A few stronger claims in the abstract/introduction overstate what the experiments establish.** In particular, “readily applicable to decision-making” is directionally fair, but the empirical evidence is better described as an encouraging first demonstration than a definitive validation of robust planning/control.

## Nice-to-Haves
- Add a direct quantitative analysis of the distribution mismatch between inverse-dynamics latent actions and latent-policy samples, and test mitigation strategies.
- Add targeted stress tests for the particle-grid regime: large displacement, objects crossing multiple patch regions, heavy occlusion, and camera motion.
- Add semantic evaluation for language conditioning beyond FVD/LPIPS.
- Report inference speed / memory / rollout cost relative to DVAE and representative large video models.
- Include a structured failure-case gallery for both video rollout and imitation learning.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that the paper provides “no differentiable routing equation” for feature transfer and is therefore fundamentally opaque/reproducible only in name.**  
  The criticism overreaches. The paper indeed does **not** formalize the handoff mechanism in detail, and that is a valid weakness retained above. But the claim that this invalidates the method or makes it fundamentally unreproducible is too strong. The paper presents the particle-grid regime as an emergent design description rather than a separate explicit routing operator; there is no evidence in the text that a missing hidden algorithm is central to training.
- **Claim that the model’s object-centric premise is “invalid” because particles are not globally persistent identities.**  
  This is too absolute. The paper explicitly positions LPWM as a hybrid between patch-based and globally free particle models, not as identical to DDLP’s tracking regime. It still learns structured latent particles with positions, scales, transparency, depth, and appearance, and reconstructs via object-style compositional rendering. The correct criticism is that the object-centricity is weakened/traded off, not absent.
- **Fairness criticism that comparisons are invalid because LPWM uses a single multitask policy while baselines are per-task.**  
  This does not hold as a weakness against LPWM. The asymmetry actually favors baselines, and the paper is explicit about that:  
  > “for PandaPush, the baselines train separate policies for each task, effectively giving them an advantage by optimizing individually for each task.”  
  Under the review rules, unfair-comparison complaints should be removed when the asymmetry favors the baseline.
- **Generic complaint about missing more baselines / related work comparisons.**  
  Not retained, since external coverage cannot be reliably audited here and the paper already compares against a meaningful set including DVAE, PlaySlot, DDLP, G-SWM, SlotFormer/OCVP, and strong downstream baselines.
- **Pure under-specification complaint about the mapping network architecture.**  
  The architecture is actually described as “a simple, compact, two-layer attention pooling transformer” with appendix details and pseudocode. One may want more analysis, but not enough is missing to treat this as a substantive flaw.
- **Criticism that teacher forcing without rollout mixing/scheduled sampling is a serious unaddressed flaw.**  
  This is too generic for the setting and not shown to undermine the main claims.

## Novel Insights
The most interesting synthesis is that LPWM’s main contribution is not simply “better object-centric prediction,” but a **specific compromise**: it trades away globally persistent tracked particles to gain scalability and stochastic modeling on harder real-world data, while preserving enough object structure to outperform both slot-based and patch-based competitors in many settings. This compromise appears genuinely effective for video modeling, but the paper also exposes the next bottleneck for object-centric world models: not representation learning per se, but aligning the model’s **generative latent policy** with the latent variables that downstream control can actually use. In other words, LPWM seems closer to a strong stochastic object-centric predictor than to a fully closed-loop planning substrate, and that distinction is the key lens through which to read the results.

## Suggestions
- Quantify and mitigate the inverse-dynamics / latent-policy distribution mismatch, since this is the main blocker to stronger decision-making claims.
- Add a dedicated stress-test suite for the particle-grid regime and report identity/permanence degradation as objects move across patch regions.
- Reframe the decision-making contribution slightly more conservatively: the current evidence supports competitive downstream transfer, but not yet a fully reliable planning interface from the latent prior alone.
- Add semantic grounding metrics for language-conditioned rollouts.
- Include qualitative failure analyses for the tasks where LPWM sharply underperforms strong baselines, especially OGBench task2 and harder PandaPush settings.

---

## Gp9lGS9GfY

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
This paper presents **GAR**, a region-level MLLM for mask-conditioned understanding that aims to preserve both fine local detail and necessary global context via a simple **RoI-aligned feature replay** mechanism. It also introduces **GAR-Bench**, a benchmark targeting not only single-region perception but also multi-prompt interaction, non-entity recognition, and compositional reasoning, and reports strong results across captioning, VQA, and some zero-shot video transfer settings.

## Strengths
- **The paper identifies and directly targets a real limitation of many region-level MLLMs: local-region understanding without enough scene context.** This is not just claimed qualitatively; the architectural ablations in Table 8 are informative. In particular, the proposed global-image + RoI-aligned replay variant substantially improves GAR-Bench results over local-only, cross-attention, RoI-pooled, and crop-supplement baselines while remaining competitive on detailed captioning.
- **The benchmark contribution is more substantive than a standard “another caption benchmark.”** GAR-Bench explicitly evaluates difficult cases that are easy to miss in region-captioning-only setups, especially **non-entity recognition** (e.g., reflections), **position reasoning**, and **multi-prompt relations with distractors**. This is a useful diagnostic decomposition of region understanding.
- **The data ablations support the paper’s claimed capability progression.** Table 10 shows that adding the fine-grained dataset primarily improves detailed recognition, while the relation dataset drives the large jump on GAR-Bench captioning and VQA, matching the intended role of each data source.
- **The paper does more than report a single favorable benchmark.** Beyond GAR-Bench, it evaluates on DLC-Bench, Ferret-Bench, MDVP-Bench, LVIS/PACO recognition, and VideoRefer. Even if the strongest headline claims rely heavily on GAR-Bench, the broader evaluation does show that the model is not narrowly tuned only for one metric.
- **The authors proactively study evaluation robustness rather than treating their benchmark as unquestionable.** The subsampling analyses (Tables 12–13), cross-judge comparisons (Table 14), and input-format analysis for general VLMs (Table 15) are all useful and strengthen the empirical section.

## Weaknesses

###: Fatal
- **Potential train/benchmark contamination is a serious concern for the core multi-prompt reasoning claims.**  
  The paper states in Section 3.3 that Round 2 training data is built using the **PSG dataset** to generate relation-aware captions, QA pairs, and MCQs (“we incorporated the Panoptic Scene Graph (PSG) dataset ... We construct a relation dataset with 414K samples”). Appendix B.1 then states that GAR-Bench relation tasks also **source images from PSG** (“For the ‘relation’ tasks, we source images from the Panoptic Scene Graph (PSG) dataset”).  
  The paper does **not** clearly specify that GAR-Bench uses disjoint images or a held-out split relative to the PSG-derived training data, nor does it document deduplication. Because the benchmark is central to the headline claim that GAR excels at multi-prompt interaction and compositional reasoning, this missing split hygiene materially weakens the evidence. This is not yet enough to prove leakage, but it is a substantial unresolved threat to validity.

### Major:
- **The paper overstates the novelty of the core architectural idea.**  
  The method is effective, but the framing sometimes suggests a more fundamental architectural leap than the paper actually delivers. Section 3.2’s key mechanism is: encode the **full image**, derive a box from the mask, then apply **RoI-Align** on the global feature map to extract context-aware regional features. This is a sensible design and the ablations suggest it works well, but the paper’s narrative at times reads as if this resolves a long-standing architectural dilemma in a fundamentally new way. The real contribution is better described as a strong engineering synthesis and adaptation for region-level MLLMs, plus the training/data/benchmark package, rather than a deeply novel architectural principle.
- **The strongest “beats much larger models / beats proprietary models” claims rely heavily on the authors’ own benchmark, which is also LLM-judged in part and difficulty-filtered against strong models.**  
  GAR-Bench is valuable, but it is also curated in a way that can amplify benchmark-specific optimization. Appendix B.1 states that any question answered correctly by all four strong non-thinking MLLMs was removed. This makes the benchmark intentionally difficult, but it also means the test set is not a neutral sample of region-understanding tasks. In addition, GAR-Bench-Cap depends on LLM judging. The paper does include cross-judge consistency analyses, which helps, but that does not fully establish that the benchmark ranking translates into broad superiority over larger models. The external benchmark results are strong, yet the most aggressive comparative claims should be phrased more cautiously.
- **The “arbitrary number of prompts” claim is stronger than the presented evidence.**  
  The task formulation allows a set of \(N\) prompts, and Figure 6b notes examples with up to 7 and 9 prompts, but the paper does not provide a systematic performance breakdown versus prompt count. Since scalability to more simultaneous prompts is one of the paper’s conceptual selling points, the lack of stratified analysis leaves this claim under-supported.
- **The synthetic relational data pipeline is plausible but under-validated.**  
  Section 3.3 relies on a seed captioner plus an LLM merger to generate large amounts of relation-aware descriptions and QA. The paper mentions quality control and human curation for GAR-Bench, but does not provide quantitative error rates, agreement rates, or noise analysis for the 2.5M training corpus. Since the paper’s multi-prompt reasoning gains appear heavily driven by this data, more evidence about annotation fidelity would materially strengthen technical soundness.

### Minor
- **The video-transfer narrative is somewhat overstated relative to the paper’s own evidence.**  
  The authors do acknowledge the limitation, including in Appendix E, and Tables 6–7 indeed show weakness on temporal aspects such as temporal description and future prediction. So this is not a misrepresentation in the sense of being hidden. However, claims like “strong capabilities can be easily transferred to videos” should be tempered: the results support useful zero-shot transfer for some tasks, but not robust temporal understanding.
- **The paper lacks a direct controlled test showing that GAR truly uses global context, rather than merely benefiting from having extra tokens.**  
  The architecture and qualitative examples are consistent with the claim, and Table 8 supports the design choice empirically. Still, a cleaner context-ablation experiment—e.g., masking/randomizing background while keeping the prompt fixed—would more directly validate the core mechanism behind non-entity and position reasoning.
- **Efficiency reporting is limited.**  
  Table 8 reports first-token latency and ViT token counts, which is useful, but there is no fuller accounting of memory/FLOPs or sequence-length overhead versus alternatives. Given that one selling point is balancing performance with practicality, a somewhat sharper efficiency characterization would help.
- **There is at least one apparent inconsistency between text and table values.**  
  In Section 4, the text says “GAR-8B achieves an impressive overall score of 54.5” on GAR-Bench-VQA, but Table 1 reports **59.9**. This should be corrected.

### Trivial
- None.

## Nice-to-Haves
- Provide a **prompt-count-stratified** breakdown of GAR-Bench performance to substantiate scalability beyond two or three regions.
- Add a **context ablation** experiment (remove or corrupt the background while preserving the prompted region) to directly test whether the method uses global context for non-entity and positional reasoning.
- Report **training/inference memory and FLOPs** for RoI-aligned replay vs. crop-based and cross-attention alternatives.
- Include a **noise/verification analysis** for the synthetic relation data: e.g., sampled human agreement, estimated hallucination rate, or correction statistics.
- Test robustness to **imperfect prompts** such as noisy masks, SAM-generated masks, or boxes, since practical deployment will rarely have ideal manual masks.
- For the video extension, a small-scale **video fine-tuning** experiment would clarify whether the current limitation is architectural or purely data-related.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper fails to specify how multiple prompt features are ordered or tokenized before feeding to the LLM.”**  
  The exact routing details are indeed not deeply elaborated in the main text, but this criticism overreaches. The paper does provide a usable high-level architecture description, and the omission is more a detail-level reproducibility request than a substantive flaw under current standards.
- **“Prior work already solved this with identical mechanisms, so the contribution is not meaningful.”**  
  The paper may overstate novelty, but this stronger claim goes too far. The ablations show that this particular design choice matters in this setup, so the method should not be dismissed as vacuous.
- **“General models’ weak GAR-Bench performance is probably only due to format unfamiliarity.”**  
  This is too speculative. The paper partially addresses input-format concerns in Table 15 by trying several region-specification formats, so a pure format-mismatch explanation is not supported by the evidence provided.
- **Pure reproducibility nitpicks about missing implementation minutiae.**  
  Appendix C includes core implementation details and hyperparameters; remaining omissions are not substantial enough to be central review points.
- **Criticism that cited tools/models/benchmarks may not be available or verifiable.**  
  Removed per instruction.

## Novel Insights
The strongest reading of the paper is not “a radically new architecture beats giant models,” but rather that **region-level multimodal understanding may now be bottlenecked as much by evaluation/task formulation and relation-centric data construction as by backbone design**. GAR’s gains seem to come from the combination of (i) preserving context through a simple but effective feature extraction choice, (ii) explicitly training on relation-aware multi-prompt data, and (iii) evaluating on tasks that expose failures hidden by ordinary region captioning. In that sense, the benchmark and data pipeline may be at least as consequential as the model modification itself.

## Suggestions
- **Clarify split hygiene immediately.** Explicitly document whether PSG images/annotations used for GAR-Bench are disjoint from all PSG-derived training data; if so, say exactly how. If not, revise the claims and add held-out evaluation.
- **Reframe novelty more precisely.** Position RoI-aligned feature replay as an effective region-MLLM design choice validated by ablations, rather than as a fundamentally unprecedented mechanism.
- **Temper the largest comparative claims.** Replace broad statements about surpassing much larger or proprietary models with benchmark-scoped claims unless supported by more independent held-out evaluations.
- **Add prompt-count analysis.** Show performance as the number of prompts increases, especially on relation tasks.
- **Quantify training data quality.** Report sampled human validation for the synthetic relation corpus.
- **Strengthen the context claim with a direct causal test.** A background-removal/randomization experiment would be especially convincing.
- **Soften the video-transfer framing.** Present current results as promising zero-shot transfer for some video tasks, while clearly delimiting the absence of robust temporal modeling.

---

## W4FAenIrQ2

- GT: Accept (Poster) (avg 6.7)
- Predicted: N/A (6.9/10)
- Match: N/A

### Final Review

## Summary
This paper presents RedSage, an 8B open cybersecurity-specialized LLM built through a full pipeline: continual pretraining on a large filtered cyber corpus, supervised post-training on an agentically augmented dialogue set, and alignment with general preference data. It also introduces RedSage-Bench, a benchmark spanning cybersecurity knowledge, offensive skills, and static tool-related question answering, and reports strong results against open 7–8B baselines on both cybersecurity and general benchmarks.

## Strengths
- **The paper makes a concrete, unusually complete open data/model/benchmark contribution for a specialized domain.** It does not just release a tuned model; it describes and plans release of a full stack: CyberFineWeb (11.7B tokens after selection), RedSage-Seed (28.6K curated items), RedSage-Conv (266K multi-turn conversations), evaluation code in LightEval, and RedSage-Bench. Few domain LLM papers cover pretraining data, post-training data, benchmark construction, and model training together at this level of specificity.
- **The agentic augmentation pipeline is a specific and interesting contribution, not just “synthetic data was generated.”** The paper defines distinct Planner and Augmenter roles, gives prompts, examples, and category-wise statistics, and uses this to expand 28,637 seed resources into 266,180 multi-turn conversations spanning knowledge, offensive skills, and tool documentation. This is a meaningful design choice aimed at converting static technical resources into assistant-style interactions.
- **The benchmark contribution is more ambitious than existing cyber MCQ suites the paper compares against.** RedSage-Bench explicitly separates knowledge, skills, and tool-related questions, and supplements 30K MCQs with 240 open-ended items plus a quality rubric. Even if some parts of the evaluation can be questioned, the benchmark broadens what is being measured relative to benchmarks that only test factual MCQs.
- **The paper includes helpful stage-wise ablations across training phases.** Reporting CFW-only, Seed-only, Base, Ins, and DPO variants makes it possible to see that curated Seed data is very strong, that CFW and Seed have somewhat complementary effects, and that DPO improves some instruction/general metrics while slightly hurting the model’s own MCQ accuracy. This granularity is valuable and better than presenting only a final tuned checkpoint.
- **Empirically, the final instruction-tuned models are genuinely strong among open 7–8B baselines on the reported evaluations.** On the authors’ tables, RedSage-8B-Ins / DPO outperform Qwen3-8B and the listed cybersecurity-tuned baselines on RedSage-Bench and on the aggregate of prior cybersecurity benchmarks, with especially notable gains in Table 5. This makes the work significant as a practical domain adaptation recipe even if some attribution claims should be toned down.

## Weaknesses

###: Fatal
- None.

### Major:
- **The paper overclaims “tool proficiency” / “tool expertise” relative to what is actually evaluated.** The benchmark’s “tool” dimension is static QA about CLI/Kali tools, not interactive tool use. The paper itself defines the tool categories as “CLI cheat-sheets, Linux manuals, Kali tools” (Sec. 3.1 / App. A.2), and the qualitative examples are command construction/explanation tasks rather than execution in an environment. That makes this a benchmark of **tool knowledge and static command-use reasoning**, not operational tool proficiency in the stronger sense implied by phrases like “understanding and operating security tools” (Sec. 2.1). This matters because the paper’s framing suggests practical assistant competence in security workflows, but the evaluation does not test execution, recovery from errors, parsing outputs, or interactive adaptation.
- **The central claim that domain-aware pre/post-training improves general reasoning is not well supported by the paper’s own ablations.** Table 6 shows that continued pretraining alone does not improve the general benchmark mean over the base model: Qwen3-8B-Base is 70.86, while RedSage-8B-CFW / Seed / Base are 69.31 / 69.58 / 69.23. The gains appear only after adding general post-training data and DPO (73.34 / 74.33). So the strongest defensible claim is that the **full pipeline**, including substantial general SFT/DPO data, preserves or improves general capability—not that the cybersecurity-specific pre/post-training itself improves general reasoning. The current abstract and discussion attribute this too broadly.
- **The evidence for the specific value of the agentic augmentation method is incomplete.** The paper demonstrates that RedSage-Ins/DPO are strong, but it does not isolate whether the Planner+Augmenter pipeline outperforms simpler alternatives, such as direct SFT on curated seed material, simpler synthetic QA generation, or non-agentic dialogue conversion. Since agentic augmentation is presented as a key methodological contribution, this missing ablation weakens the causal claim that the particular two-agent design is responsible for the gains.
- **Benchmark construction and evaluation rely heavily on LLM-generated and LLM-verified artifacts, with limited human validation for the largest component.** For MCQs, generation, structural validation, and quality scoring are all LLM-based, with only “random audits” reported for the 30K benchmark items; only the 240 open-ended items are stated to be human-verified. For open-ended evaluation, a single judge model (Llama-3.3-70B) is used at evaluation time. This does not invalidate the results, but it leaves open concerns about stylistic bias, hidden judge preferences, and the factual accuracy of benchmark items at scale. Given how central RedSage-Bench is, stronger quantitative human validation would materially strengthen the paper.
- **The paper does not perform explicit decontamination checks against the external public benchmarks used for headline SOTA claims.** The paper does perform decontamination between its benchmark and synthetic post-training data (“remove any synthetic post-training instance whose query has a semantic similarity above 0.9 to a benchmark question”), but this only addresses RedSage-Bench leakage. It does not report analogous filtering or overlap analysis against external evaluations such as MMLU-CSec, CyberMetric, SECURE, CTI-Bench, or SecBench, despite using a very large web-derived CPT corpus and curated sources that overlap in subject matter. This does not prove the results are contaminated, but for a paper making benchmark-leading claims, stronger evidence of benchmark isolation would be important.

### Minor
- **Some quantitative reporting is confusing or inconsistent.** In particular, the abstract’s “+5.05 points on Open LLM Leaderboard tasks” does not match the most obvious comparison in Table 6 between RedSage-8B-DPO (74.33) and Qwen3-8B (65.92), which is larger. Perhaps the intended baseline differs, but the paper should specify exactly which comparison underlies the headline numbers.
- **The CPT ablations themselves suggest the large web-filtered corpus may be less clearly necessary than the narrative implies.** In multiple places, Seed-only is as good as or better than CFW-only and often very close to or better than the combined Base model (e.g., Table 4 and several columns of Table 5). This is interesting rather than damning, but it weakens the implied case that the expensive CyberFineWeb stage is the main driver of the final performance gains.
- **The use of generic DPO data appears to trade off domain benchmark performance for broader instruction-following, but this trade-off is underanalyzed.** The paper notes that RedSage-8B-DPO slightly underperforms RedSage-8B-Ins on RedSage-Bench MCQs while improving general benchmarks, yet this is framed only briefly. A more explicit discussion of specialization-vs-alignment trade-offs would improve technical clarity.
- **The limitations section is too thin relative to the paper’s ambitions.** It briefly mentions synthetic-data errors and dual-use risk, but does not adequately discuss the limits of static tool evaluation, the general-capability drop after CPT, or the dependence on LLM-generated/LLM-judged benchmarks.

### Trivial
- **Token counts are measured with the GPT-2 tokenizer although the model is Qwen3-based.** The appendix says this follows FineWeb conventions, so this is acceptable for rough corpus statistics, but those counts should be interpreted as approximate dataset-size reporting rather than model-specific training-token accounting.

## Nice-to-Haves
- Add a direct ablation comparing the proposed agentic augmentation pipeline against simpler synthetic generation baselines and against seed-only SFT.
- Add a contamination analysis or temporally disjoint evaluation for the external cybersecurity benchmarks used in Table 5.
- Reframe benchmark terminology from “tool proficiency/expertise” to something like “tool knowledge and command-use reasoning,” unless execution-based evaluation is added.
- Include quantitative human expert validation on a representative sample of the 30K MCQs and report agreement with the LLM verifier/judge.
- Analyze more explicitly why Seed-only CPT is so competitive with or better than CFW-only, and whether the CFW stage is most useful for coverage rather than average accuracy.
- Expand the safety section with concrete misuse mitigation and red-teaming/refusal analysis, especially given the offensive-security data included.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Complaints about the existence/release status/transparency of cited baselines or tools.** Per the paper and review policy, cited models and datasets should be treated as existing; concerns framed as “unclear whether available/verifiable” were removed.
- **Criticism of missing related work on contamination or benchmark methodology.** I cannot verify omitted references externally, so this was removed.
- **Complaints about mixed-backbone baseline comparisons being inherently unfair.** The asymmetry here does not favor the authors; if anything, comparing against strong heterogeneous baselines strengthens the claim.
- **Claim that future benchmark leakage due to public release is a paper weakness.** This is too generic and applies to most public benchmarks; it does not materially undermine the present paper.
- **Request for alternative agent scaffolding comparisons at inference time.** The paper is about training a domain model and benchmark, not about evaluating every possible downstream agent scaffold; this is outside the core scope.
- **Pure reproducibility nitpicks about exact proprietary model configuration details.** The paper reports the setup used; demanding more detail for closed-model internals is not actionable.
- **Stylistic/formatting concerns.** Omitted by instruction.

## Novel Insights
A notable pattern in the paper’s own ablations is that the strongest evidence is not “large-scale cyber web pretraining unlocks everything,” but rather a more nuanced recipe: curated domain data appears disproportionately valuable, large noisy domain CPT offers complementary but not dominant benefits, and general SFT/DPO are what restore or improve broad capabilities after specialization. In other words, the work’s most convincing contribution may be the **composition** of curated domain resources, synthetic assistant-style augmentation, and general alignment data—not the scale of CyberFineWeb alone. The paper would become stronger if it embraced this more precise story.

## Suggestions
- Reword the main claims so they align with the evidence: say the **full training pipeline** improves general benchmarks, while cybersecurity CPT alone slightly reduces them.
- Rename or carefully qualify “tool proficiency/expertise” throughout, unless you add execution-based or sandboxed tool-use evaluation.
- Add a focused ablation: seed-only SFT vs. simple synthetic SFT vs. Planner+Augmenter SFT.
- Add explicit overlap/decontamination checks for external benchmarks, or at minimum a temporal-holdout or source-filtering analysis.
- Provide quantitative human validation for a sample of RedSage-Bench MCQs and correlation/agreement analysis for the LLM judge on open-ended QA.
- Expand the discussion/limitations section to cover benchmark construction bias, static-vs-interactive tool evaluation, and the specialization/alignment trade-off revealed by Ins vs. DPO.

---

## yfLpRFuMwK

- GT: Reject (avg 3.3)
- Predicted: N/A (4.6/10)
- Match: N/A

### Final Review

## Summary
This paper proposes Non-Replacement Function Space Sampling (NRFS), a Bayesian optimization acquisition strategy that estimates a candidate’s probability of being the optimizer by sampling functions from the GP posterior, grouping them by optimizer location, and then focusing on optimizer mass not already “covered” by previous acquisition decisions. The paper’s central idea is to replace explicit exploration/exploitation trade-offs with a sampling-and-removal mechanism over optimizer-supporting posterior functions, and it backs this with broad empirical results on synthetic tasks, noisy settings, batch selection, higher-dimensional variants, and two materials-design case studies.

## Strengths
- **A genuinely distinctive acquisition perspective centered on optimizer mass rather than reward heuristics.** The paper’s main contribution is not just “another BO heuristic,” but a different way to think about BO acquisition: estimate optimizer probability via posterior function samples and use that to guide search. The function-to-optimizer “bucket” construction in Sec. 3 gives a concrete operationalization of this idea and is more specific than generic claims of balancing exploration and exploitation.
- **The paper identifies and studies a regime where the method seems especially effective: multimodal objectives requiring both escaping local basins and then exploiting the right one.** This is supported by the reported results on GM, modified Rosenbrock, Shekel, SFE, and the qualitative behavior plots, where NRFS is presented as avoiding both repeated local oversampling and purely uncertainty-driven wandering.
- **The empirical study is unusually broad in scope for a BO acquisition paper.** Beyond the main synthetic tasks, the authors include batch variants, noise robustness, higher-dimensional tests, standard-deviation plots, runtime comparisons, and two real materials tasks. Even where some comparisons are incomplete, the breadth helps clarify the method’s behavior rather than only showing best-case wins.
- **The paper is more candid than many submissions about limitations and mixed regimes.** In Appendix A.4 the authors explicitly note a setting where NRFS may not outperform more exploitation-heavy methods (e.g., Branin-like cases where finding any one optimum is sufficient), which gives the empirical picture more credibility than an all-upside narrative would.

## Weaknesses

###: Fatal
- **The paper’s core theoretical narrative overstates what is actually justified.** A central claim is that non-replacement sampling over posterior functions “consistently ensur[es] convergence to the global optimum,” and the text motivates this via exhausting a pool of sampled functions/buckets. But the actual object is a GP posterior over a continuous function space, while Eq. (8) rewrites probability as a cardinality ratio and then says that \(|F^{GP}_D|\) is “typically determined by how many functions are sampled from the surrogate model space, which is usually fixed.” This means the paper is moving from a posterior probability measure to a finite Monte Carlo sample-count interpretation, and the claimed “pool exhaustion” guarantee is not established for the real continuous posterior. As written, this is not a minor looseness: the convergence intuition and the operational algorithm are not cleanly aligned, and the strongest theoretical claims are not supported by the mathematics presented.

### Major:
- **The status of “non-replacement” across BO iterations is mathematically and algorithmically ambiguous.** The paper says it “remove[s] all functions assigned to [the selected bucket] from future consideration,” but after each new observation the GP posterior changes, so the function distribution, the bucket memberships, and even the sampled function set are redefined. The paper does not clearly specify a persistent state that tracks removed function mass across posterior updates. In practice, the method appears to resample from an updated truncated posterior each iteration and estimate optimizer density anew. That may still be useful, but it is much less clear than the paper’s bucket-removal narrative suggests. This ambiguity matters because the “non-replacement” mechanism is the main source of novelty.
- **The truncation step is insufficiently justified as a Bayesian procedure.** In Sec. 3.2 the paper argues that if the true optimizer has not yet been found, then any sampled function whose optimum does not beat the current best \(Y_n^*\) should be excluded; the text even states that “such functions should be excluded, as their probability \(p(f)\) is zero.” This is too strong as written. The conditional event the paper wants is intuitive—future useful optimizer mass should lie below the incumbent for minimization—but declaring posterior probability zero for those functions is not the same as a standard Bayesian posterior update, and the paper does not provide a rigorous derivation of the truncated GP sampling procedure it relies on. Since the method’s practical behavior depends heavily on this truncation, the lack of formal justification weakens the claim that NRFS is a principled alternative to existing BO criteria.
- **The implementation of sampling from the truncated GP is not specified clearly enough for the method’s central mechanism.** The paper introduces \(TGP_-(\mu,k,Y_n^*)\) and builds the acquisition around samples from it, but does not explain how these samples are actually drawn under the global constraint that the function improve on the incumbent. This is not a trivial implementation detail; it is the heart of the method, affects computational cost, and affects whether the method is a practical approximation or a conceptual idealization.
- **The empirical evidence is broad but does not isolate the source of the gains.** The most important missing ablation is a comparison against a closely related “PO/TGP with replacement” variant or a multi-draw Thompson-style baseline that uses the same sampled-function machinery without the proposed non-replacement logic. Without such an ablation, it is hard to tell whether the performance gains come from (i) truncating the posterior below the incumbent, (ii) estimating optimizer density with many posterior draws plus Parzen smoothing, or (iii) the actual non-replacement mechanism claimed as the core contribution.
- **The paper overreaches in its performance claims relative to its own evidence.** The abstract claims “state-of-the-art performance” and says NRFS “consistently improv[es] optimization performance in all settings,” but the paper itself reports counterexamples or caveats: slower initial behavior on Shekel and HC, competitive rather than dominant performance on Forrester2D, and a clear limitation on Branin-like cases in Appendix A.4. The method may be strong in the targeted multimodal regime, but the paper should present that regime-specific strength more precisely instead of implying near-universal superiority.
- **The high-dimensional evidence is not yet convincing enough for broad scalability claims.** The main high-dimensional study is a lifted Forrester benchmark up to 50D. That is useful as stress testing, but by itself it is not a strong basis for claiming superior high-dimensional BO performance. Moreover, Figure 5 is described as showing shrinking performance gaps with dimension, which is consistent with increasing difficulty for all methods rather than a compelling demonstration of scalable optimizer-density estimation in high dimensions.

### Minor
- **Some interpretive discussion of prior BO methods is overstated.** The paper repeatedly characterizes existing acquisitions as “subjective rewards” and suggests they are largely ad hoc or biased away from the true optimizer. That rhetoric is stronger than what the evidence in the paper supports. The practical motivation for NRFS is clear without needing to dismiss EI/PES/UCB-style methods so broadly.
- **The method has meaningful practical hyperparameters despite claims of being hyperparameter-light.** Although NRFS avoids tuning exploration parameters like \(\epsilon\) or \(\beta\), it still depends on choices such as the number of sampled functions \(M=1000\), the Parzen/KDE density estimation procedure, and fantasy-sample count for OSLA. The paper would be stronger with sensitivity analysis for these choices, especially because they directly affect stability and runtime.
- **OSLA appears substantially more expensive for mixed benefit.** Appendix A.6 reports roughly 5x runtime over standard NRFS, while Appendix A.5 notes cases where OSLA underperforms standard NRFS. This does not invalidate the variant, but it limits the practical significance of the look-ahead extension as currently presented.

### Trivial
- **A clearer pseudocode presentation would help connect the bucket definitions to the practical KDE-based implementation.** The prose and equations describe the idea, but the exact algorithmic loop—especially how truncation, optimizer extraction, density estimation, and “removal” interact after GP refitting—remains harder to parse than necessary.

## Nice-to-Haves
- Add an ablation against the nearest method variants: PO from full GP samples, PO from truncated GP samples **with replacement**, and multi-draw Thompson sampling. This would isolate whether non-replacement itself is the main driver.
- Include sensitivity studies for the number of posterior samples \(M\), density-estimation bandwidth/estimator choice, and optimizer discretization/grid density.
- Provide stronger high-dimensional evaluation on more standard BO testbeds and report wall-clock time-to-target in addition to best value versus iterations.
- Add statistical significance testing across the 20 runs to support claims of superiority when curves are visually close.
- Expand the discussion of failure modes already hinted at in Branin-like and high-noise settings.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Unfair baseline treatment because baselines were tuned while NRFS was not.”** Removed as a main weakness. The paper’s asymmetry here actually tends to favor the baselines, not NRFS: it explicitly reports sweeping \(\epsilon\) for \(\epsilon\)-EI and tuning UCB-related constants, which is not unfair to the proposed method. A fairer criticism is instead that NRFS’s own design choices should be analyzed more explicitly via ablations/sensitivity.
- **“No computational cost comparison is provided.”** Factually incorrect. Appendix A.6 and Table 1 do provide runtime comparisons.
- **“Limited real-world validation” / “needs broader domains.”** Too generic and largely scope-creeping given the paper already includes two real materials tasks in addition to many synthetic studies. More domains would be nice, but this is not a core flaw.
- **“Incomplete comparison with unspecified modern related work.”** Removed under the instruction not to speculate about missing related work.
- **“Need code release / reproducibility concerns tied to release status.”** Removed per instruction. The right concern here is algorithmic clarity, not speculation about artifact availability.
- **“Lack of theoretical guarantees” as a standalone weakness.** Weakened and absorbed into the more specific issue that the present theoretical claims are overstated and not adequately justified. Simply lacking a full regret proof would not by itself be a strong criticism for this genre of empirical BO paper.
- **Formatting/style issues and parser artifacts.** Ignored as instructed.

## Novel Insights
The most important synthesis across the reviews is that this paper is probably strongest when interpreted not as a fully justified new BO principle, but as an empirically promising optimizer-density heuristic built from two ingredients: truncating posterior function samples to those that can beat the incumbent, and redistributing acquisition mass away from already-used optimizer locations. Under that interpretation, the experiments are genuinely interesting and suggest a useful regime of advantage—multimodal search with a need to both escape local attractors and later exploit the correct basin. The main problem is that the paper currently presents this practical heuristic as if it had a much firmer probabilistic and convergence foundation than is actually shown.

## Suggestions
- **Reframe the paper’s central claims more modestly and precisely.** Present NRFS as a sampling-based BO heuristic motivated by optimizer-probability estimation, not as a method with an established pool-exhaustion convergence guarantee in continuous GP function space.
- **Add a rigorous ablation suite**: full-GP PO, truncated-GP PO with replacement, multi-draw Thompson sampling, and the full NRFS method.
- **Specify the truncated GP sampler in detail**, including whether the method uses rejection sampling, constrained path sampling, an approximation, or something else, and discuss its computational implications.
- **Clarify what exactly is being “removed” across iterations** when the posterior changes; if removal is only within the Monte Carlo sample approximation at a given iteration, say so explicitly.
- **Add sensitivity analyses** for the number of function draws, density estimator, and OSLA fantasy samples.
- **Tone down universal performance language** and instead emphasize the settings where the evidence is strongest: multimodal objectives where oversampling and uncertainty-only exploration both cause problems.
- **Improve algorithm presentation with pseudocode** so the relationship between the equations and the implementation is unambiguous.

---

## ZNAY3ivd62

- GT: Reject (avg 4.0)
- Predicted: N/A (6.3/10)
- Match: N/A

### Final Review

## Summary
This paper proposes GUI-Spotlight, a GUI visual grounding model that performs iterative image-grounded reasoning with three tools (`extract`, `crop`, `find_color`) and is trained via a three-stage pipeline combining SFT and a modified GSPO-style RL objective. Empirically, it reports strong 7B-scale results, most notably 52.8% on ScreenSpot-Pro with 18.5K curated training samples, alongside useful empirical observations about RL instability and reward design for multi-turn tool-using agents.

## Strengths
- **The paper identifies and operationalizes a concrete mechanism for improving fine-grained GUI grounding beyond single-shot prediction.** The registry/offset formulation in Sec. 3.1 is a real technical contribution rather than just prompting: the model keeps a mapping \(R=\{i \mapsto (I_i,\delta_i)\}\) and converts relative coordinates from cropped images back to the original screen. This directly addresses coordinate consistency across iterative crops, which is a practical failure mode in GUI grounding systems.
- **The RL investigation yields specific, practically valuable insight about instability in multi-turn tool-using policies.** Section 4.1 does more than report final numbers: it shows that vanilla GRPO/GSPO begins to collapse after ~300 steps due to malformed tool outputs, and that adding the auxiliary filtered cross-entropy term prevents that degradation. The paper’s documentation of discarded variants and reward-design outcomes is unusually concrete and likely useful to others building agentic grounding systems.
- **The main ScreenSpot-Pro result is genuinely strong for the 7B regime.** On Table 3, GUI-Spotlight reaches 52.8%, outperforming several competitive 7B open models including V2P-7B (50.6), GTA-1-7B (50.1), and UI-Venus-7B (50.8), while using far fewer curated training samples than the multi-million-sample methods. Even if some efficiency claims should be toned down, the core result is still notable.
- **The method is not tied to a single backbone.** The paper trains from both UI-TARS-1.5-7B and Qwen2.5-VL-7B-Instruct and reports gains over both starting points, supporting that the tool-augmented training recipe is not purely a one-backbone trick.

## Weaknesses

###: Fatal
- None.

### Major:
- **The evaluation does not report inference-time cost for the iterative method, which materially limits how to interpret the headline gains.** This criticism is valid from the paper text itself. The method explicitly performs multi-round interaction until `Stop` (Sec. 3.1, Algorithm 1), appending new images and dialogue history after each tool call. Yet the experimental section reports only accuracy, not average number of turns, latency, or compute. Because many baselines in Table 3 are effectively single-shot grounding systems, the paper’s practical-positioning claims (“practical usefulness,” “substantially improving visual grounding accuracy”) would be much stronger if accompanied by inference-cost measurements. This does not invalidate the accuracy gains, but it does make the comparison incomplete for a system paper about deployable grounding.
- **The claim that the method is “data-efficient” is overstated relative to the actual supervision/cost pipeline.** The paper is careful in reporting that final training uses 18.5K curated samples, but it also states that these samples come from substantial filtering with Qwen2.5-VL-72B and that Stage 1 trajectories are generated by Qwen2.5-VL-72B over the filtered UGround data (“we first executed the same inference pipeline with Qwen2.5-VL-72B ... and collected 2561 multi-turn dialogue trajectories”). So the narrow statement “trained on 18.5K curated samples” is true, but the broader narrative of efficiency should be qualified: the approach depends on heavy teacher-assisted curation/distillation, and the paper does not quantify that upstream cost. This weakens, but does not negate, the efficiency claim.
- **The paper does not isolate the contribution of individual tools, leaving the central “multi-tool coordination” claim under-supported.** The method is built around `extract`, `crop`, and `find_color`, and much of the paper’s framing emphasizes dynamic coordination among specialized tools. However, the ablations in Sec. 4 study RL variants and reward weights, not removal of tools. Without a tool ablation, it remains unclear whether gains come from the full multi-tool design or mostly from one or two components (e.g., iterative cropping plus extract). This is especially important because the method’s novelty is more in the coordinated tool-use setup than in any single tool.
- **The algorithmic novelty around the modified GSPO objective is somewhat overstated.** The paper’s auxiliary term \(J'(\theta)\) is useful and empirically justified, but from the formulation in Sec. 3.2.2 it is essentially a masked cross-entropy/positive-trajectory regularizer on format-correct and result-correct samples. That is an important stabilization device, but the current evidence does not clearly establish it as a fundamentally new optimization idea rather than a pragmatic regularization variant. The paper should frame this more as an effective adaptation of GSPO for multi-tool RL than as a strong standalone algorithmic advance.

### Minor
- **Generalization beyond the strongest benchmark is mixed rather than uniformly compelling.** The OSWorld-G improvement over UI-TARS-1.5-7B is modest (61.9 → 62.7 in Table 5), so the paper’s broader generality claims should be stated more cautiously. The ScreenSpot-Pro gains are the clearest and strongest evidence.
- **There is limited qualitative/failure analysis of the remaining errors.** Given that ScreenSpot-Pro accuracy is still 52.8%, understanding the dominant failure modes—semantic ambiguity, missed crops, offset propagation, bad stopping decisions, or tool-format breakdown—would substantially strengthen the paper’s scientific value.
- **The paper does not analyze how often each tool is used or whether tool choice is context-sensitive.** Since the approach hinges on adaptive iterative focusing, statistics on tool usage by benchmark/domain/difficulty would help verify that the model learned meaningful routing rather than defaulting to a narrow strategy.
- **The large performance difference between initial backbones is not well explained.** The method improves both UI-TARS-1.5-7B and Qwen2.5-VL-7B, but the gap between the resulting models remains substantial. Some discussion of which properties of the starting backbone matter most for successful iterative grounding would help clarify the method’s portability.

### Trivial
- **The stopping behavior is only partially characterized.** The introduction says the model stops “once coordinate confidence is sufficient,” but the concrete procedure in Algorithm 1 is simply iterative rollout until `Stop` or \(T_{\max}\). A brief analysis of stopping-step distribution would improve clarity.

## Nice-to-Haves
- Report average turns, latency, and possibly accuracy-versus-turn-budget curves.
- Add leave-one-tool-out ablations and perhaps a reduced toolset baseline (`crop` only, `crop+extract`, etc.).
- Include a data-scaling curve over subsets of the 18.5K curated samples to better substantiate the sample-efficiency story.
- Provide representative successful and failed trajectories with intermediate crops, tool calls, and final coordinates.
- Quantify stability more directly (e.g., malformed-action rate over training, entropy/gradient norm trends) to support the RL-collapse narrative.
- Evaluate whether improved grounding translates to downstream end-to-end agent success, since the paper motivates pixel-level reliability in service of action execution.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Baseline comparison is unfair because leaderboard models may use different or unknown initializations.”** Removed because this is speculative and not verifiable from the paper alone. The paper clearly states which results are from official instructions and which are from benchmark leaderboards; that is a valid reporting choice.
- **“Prompt missing in the main text harms reproducibility.”** Removed as a core criticism because Listing 1 is explicitly referenced and Appendix A.3 says the prompt is provided there; the extraction artifact likely omitted the actual prompt content.
- **“SE-GUI-7B is omitted from the comparison.”** Factually incorrect: SE-GUI-7B is included in Table 3 with 47.2%.
- **Concerns about release status / independent verifiability of cited systems or benchmarks.** Removed by policy and not appropriate here.
- **Pure parser/formatting complaints about garbled equations or tables.** The extracted text is noisy, and such issues cannot be confidently attributed to the paper PDF itself.

## Novel Insights
The most interesting synthesis across the reviews is that the paper is strongest not as a pure “new RL algorithm” paper, but as a systems-and-training recipe paper for iterative visual grounding. Its real contribution is the combination of explicit coordinate bookkeeping, tool-mediated visual refinement, and a practically effective anti-collapse training procedure for multi-turn GUI grounding. The main gap is that the paper argues from task accuracy alone, while the method’s true scientific interest lies in the tradeoff among accuracy, turn budget, and tool-routing behavior; exposing that tradeoff would likely make the work both more convincing and more informative.

## Suggestions
- Reframe the contribution more conservatively: emphasize a **practical stabilization and tool-coordination recipe** for multi-turn GUI grounding rather than a major new RL algorithm.
- Add **inference-efficiency reporting**: average tool calls per example, latency, and accuracy under capped turn budgets.
- Add **tool ablations** and **tool-usage analysis** to validate the central multi-tool coordination claim.
- Expand **failure analysis**, ideally with categorized errors and trajectory visualizations.
- Temper the **data-efficiency** language by explicitly acknowledging the teacher-assisted curation/distillation cost, while still highlighting the low number of final supervised/RL training samples.
- Make the paper’s scope explicit: the strongest evidence is on **high-resolution grounding**, while broader transfer to general OS grounding appears positive but smaller.

---

## IU4rqTlpRb

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (6.6/10)
- Match: N/A

### Final Review

## Summary
This paper argues that benign relearning after LLM unlearning is driven more by **surface-form / syntactic similarity** than by topical relevance. It first revisits BLUR and shows that prior conclusions about topicality are sensitive to evaluation protocol confounds, then studies controlled relearning settings where syntactically similar benign data consistently restore forgotten content more strongly than topically related data. Based on this diagnosis, it proposes **syntactic diversification** of the forget set via paraphrasing, which reduces relearning, speeds forgetting, and improves utility retention in the reported experiments.

## Strengths
- **It makes a specific and consequential correction to prior benchmark methodology.** Section 4 does more than claim a new factor matters: it identifies concrete confounds in BLUR—unequal relearn set sizes under fixed-epoch training and reporting only at a single step—and proposes a fairer protocol with standardized step budgets and max-over-trajectory evaluation. This is a real methodological contribution likely useful beyond this paper.
- **The paper contributes a plausible mechanistic account rather than only reporting correlations.** Section 6’s “template vs. keyword” analysis is the most distinctive part of the paper: the loss ratio \(L_{\text{template}}/L_{\text{keyword}}\) increases during unlearning, suggesting that standard unlearning suppresses repeated answer/query templates more than the factual keyword itself. Appendix F strengthens this with a template-injection intervention showing that standard unlearning leaves substantial keyword recoverability once the answer template is supplied, whereas diversification reduces this leakage.
- **The empirical pattern is consistent across several unlearning methods and settings.** On TOFU, the core phenomenon appears under GA, NPO, and SCRUB; appendices extend this to full-parameter vs. LoRA settings and another model family (Phi-1.5B). The paper also attempts to move beyond TOFU with WHP and WMDP case studies, which is important given the synthetic nature of TOFU.
- **The proposed intervention is simple and practically meaningful if the diagnosis is correct.** Syntactic diversification is straightforward—paraphrase forget queries before unlearning—but the reported effect is notable: Figure 8 and Figure 9 indicate reduced relearning and faster forgetting, and Table 2 shows improved utility on Real Authors / World Facts / Retain. The claim that robustness can improve while easing the usual forgetting–utility trade-off is a nontrivial positive result.

## Weaknesses

###: Fatal
None.

### Major:
- **The central “syntax is the hidden driver” claim is supported most strongly in a highly templated synthetic regime, so generalization remains only partially established.**  
  This concern is real, though weaker than the harsh review states. The paper’s main controlled analysis in Section 5 is on TOFU, where the target set is explicitly “full name” QA and the syntactic relearn set is deliberately constructed with the same QA style: e.g., target and syntactic relearn examples in Appendix B.2 share nearly identical forms such as “What is the full name of the author born in … ? / The full name of the author is …”. That design is useful for isolating structure, but it also means the strongest evidence is in a setting with unusually rigid templates. The paper does provide non-TOFU evidence (Appendices C and D), so the criticism that the claim is *entirely* an artifact is too strong; however, the paper still does not convincingly show that the same mechanism dominates on naturally diverse, unstructured corpora. As written, the evidence supports a strong claim about **template/surface-form similarity in common unlearning benchmarks**, and a more tentative claim about general benign relearning broadly.
- **The paper’s operationalization of “syntactic similarity” is somewhat conceptually loose, and the main text overstates what its metric establishes.**  
  In Section 5.1 the main similarity measure is normalized Levenshtein distance, which captures character-level surface overlap, not syntax in the linguistic sense. The authors partially acknowledge this by describing it as “surface-level alignment” and by including alternative metrics in Appendix I (“template-mining similarity” and “parse-tree similarity”). That partial addressal matters, so the criticism should not be overstated. Still, the paper’s headline language repeatedly elevates this to “syntactic similarity” as the primary driver, while the strongest direct evidence is really about **surface-form / template similarity**. This matters because the mechanism in Section 6 could plausibly be rephrased more precisely as template matching and repeated phrasing, which is narrower than syntax as usually understood.
- **The proposed mitigation is under-ablated relative to the strength of the causal claims.**  
  Section 7 shows that paraphrastic diversification helps, but it does not disentangle why. The current comparison is essentially original forget set vs. diversified forget set, with limited analysis of: number of paraphrases, filtering thresholds, semantic-fidelity criteria, or whether simpler augmentation baselines would achieve similar gains. Since the paper claims diversification “forces the model to suppress keywords directly,” stronger ablations would be needed to isolate whether the benefit comes specifically from syntactic heterogeneity, from more training data, from paraphrase quality, or from broader answer/query coverage.

### Minor
- **The paper would benefit from a clearer threat model and deployment story.**  
  Section 8 gestures at providers receiving benign fine-tuning requests with syntactically similar structure, but the realistic attacker/benign-user scenarios are not sharply specified. Is the concern accidental recovery during downstream adaptation, strategic elicitation by users, or adversarial relearning by a model owner? The answer affects how significant the vulnerability is and how practical diversification is as a defense.
- **The computational and operational cost of diversification is not characterized in enough detail.**  
  The method relies on generating and filtering paraphrases (“carefully examining each generated variant” in Appendix G.1), and while Appendix G.4 shows that Llama-3-8B can substitute for GPT-4o, the paper does not quantify overhead in data generation, filtering effort, or preprocessing cost. Since the method is pitched as practical, some cost-benefit discussion would strengthen the case.
- **Some evaluation settings rely on stronger but less transparent evaluators without calibration.**  
  Appendix C uses GPT-4o as judge for WHP answer completion. This is a reasonable supplementary metric, but the paper does not provide calibration against human judgments or other validation in that setting. This is not a core flaw—most main TOFU claims use exact-match-like keyword metrics—but it slightly weakens confidence in the more realistic case study.
- **The mechanistic analysis, while insightful, remains at a relatively coarse level.**  
  Representation similarity, gradient cosine similarity, and loss-ratio trends are supportive, but they do not fully establish that syntactic similarity is the cause rather than an especially predictive correlate of residual memorization. The evidence is good enough to motivate the hypothesis, but some of the wording in Sections 6–7 is stronger than the current mechanism analysis warrants.

### Trivial
- **Terminology could be tightened throughout.**  
  The paper often uses “syntax,” “syntactic similarity,” “surface structure,” and “template” nearly interchangeably. Given the actual metrics and constructions, a more careful distinction would make the claims sharper.

## Nice-to-Haves
- Compare syntactic diversification against simpler augmentation controls, such as random paraphrasing without similarity filtering, varying the number of paraphrases, or broader forget-set augmentation not explicitly optimized for structural diversity.
- Extend the template/keyword suppression analysis to a less templated corpus to test whether the same imbalance appears outside QA-style synthetic benchmarks.
- Report a modest overhead analysis for diversification, including generation model choice, filtering burden, and resulting dataset expansion.
- Test robustness to relearning with syntactic forms not seen during diversification generation, to distinguish genuine robustness from adaptation to a paraphrase distribution.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The dataset choice invalidates the paper’s core claim and cannot be fixed except by a fundamental methodological overhaul.”**  
  Too strong given the actual paper. While TOFU is indeed templated, the authors also include WHP and WMDP analyses in Appendices C and D, and the paper’s critique of BLUR is independently valuable. The evidence is insufficient for the paper’s broadest generalization, but not so broken as to invalidate the entire work.
- **Concerns about missing comparisons to specific external methods (e.g., RMU/ERM or unspecified state-of-the-art defenses).**  
  I cannot verify that such comparisons are expected or necessary without external knowledge, and the current paper already compares multiple standard unlearning methods plus a safety-training contrast.
- **Requests for full mechanistic interpretability or Hessian/theoretical analysis.**  
  These would be interesting but are outside the standard burden for an empirical unlearning paper and are better framed as future work rather than weaknesses.
- **Reproducibility concerns rooted in use of GPT-4o / external models / cited checkpoints.**  
  The paper cites the resources and even reports an open-model alternative in Appendix G.4; existence/release-status concerns should not be treated as weaknesses.
- **Purely generic strength claims such as “the paper is well-written” or “experiments are extensive.”**  
  Omitted per instruction; the retained strengths are specific.

## Novel Insights
The most interesting synthesis across the evidence is that the paper is strongest when interpreted as a diagnosis of **template-dominant forgetting** rather than a universal theory of linguistic syntax. Section 4’s benchmark correction and Section 6’s template-vs-keyword analysis together suggest a deeper issue: common unlearning setups may overestimate forgetting because they are disproportionately suppressing repeated response scaffolds, not the underlying factual association. Under that framing, the success of diversification is especially interesting not merely as paraphrase augmentation, but as a way of forcing the forget objective to cover a larger equivalence class of prompts so that forgetting pressure cannot collapse onto a single surface form.

## Suggestions
- Reframe the core claim more precisely from “syntax is the primary driver” to something like “surface-form/template similarity is a dominant and previously underappreciated driver,” unless stronger non-templated evidence is added.
- Add ablations for diversification: number of paraphrases, similarity thresholds, and a simple paraphrase-augmentation baseline without explicit diversity filtering.
- Strengthen the non-synthetic evidence by extending the template/keyword suppression analysis to WMDP or another less templated corpus.
- Clarify the threat model: accidental recovery during downstream fine-tuning, benign adaptation by third parties, or deliberate adversarial recovery.
- Include a brief preprocessing-cost analysis and discuss when diversification is feasible for real deletion requests with narrow or singleton forget sets.

Overall, this is a thoughtful and potentially impactful paper with a real methodological correction and a compelling mechanistic hypothesis. Its main weakness is not that the phenomenon is nonexistent, but that the paper presently proves it most convincingly in template-heavy settings while phrasing the conclusion more broadly than the evidence fully supports.

---

## 9qbKOaF8YJ

- GT: Withdrawn (treated as Reject) (avg 3.3)
- Predicted: N/A (6.4/10)
- Match: N/A

### Final Review

## Summary
This paper studies class-incremental semantic segmentation without old-data replay and argues that standard KD over-preserves old representations, causing parameter competition and underuse of previously acquired knowledge. It proposes DKD, a three-part objective combining (i) pruning-based “parameter release” for the old model with an old-distribution matching loss, (ii) a Laplacian/projection-based construction of reusable old-knowledge maps, and (iii) an entropy-based objective intended to maximize shared knowledge between old and new distributions. Empirically, the method is strong on VOC and ADE20K across many incremental settings, especially with a ViT backbone, but some core technical parts are underspecified enough that the mechanism is not yet fully convincing from the paper alone.

## Strengths
- **The paper targets a specific and plausible failure mode of KD-based CISS—over-constraining the student to preserve old distributions in a fixed-capacity model—and builds the method around that diagnosis.** This is more specific than the generic “stability-plasticity tradeoff” framing. The motivation is visible in the method design: `L_Min` weakens the old teacher signal after pruning, while `L_Esti`/`L_Max` try to turn old knowledge into guidance for new learning rather than only a retention constraint.
- **Empirical results are genuinely strong across a broad set of CISS settings.** On VOC (Table 1), the method is competitive or best across 10-1, 2-2, 15-1, 19-1, and 15-5; on ADE20K (Table 2), it is similarly strong across four settings. This is not a single-split win. The paper also includes additional disjoint-setting results and class-wise analyses in the appendix.
- **The ablations are more informative than usual and support that all three losses matter.** Table 12 is especially useful: the full combination improves the old/new balance more than single components alone, and `L_Min` appears particularly important in harder multi-step settings like 10-1.
- **The paper provides nontrivial robustness reporting rather than only single-run headline numbers.** Tables 5 and 13 report repeated-run variability, and the deviations are indeed small in the presented settings.
- **The method appears architecturally somewhat portable rather than being tied only to the paper’s full ViT recipe.** Appendix C.3/C.4 shows DKD added to CoinSeg with both ResNet101 and ViT backbones, with gains on the incremental class and modest overall improvements. That helps support DKD as a transferable training strategy, not only a one-off system.

## Weaknesses

### Major:
- **The central “parameter release” claim is not fully substantiated as a mechanism on the student model.**  
  In Section 3.2(a), pruning is applied to the **old model**: “the release is performed once per step for the old model” (Appendix A.1), and the current model is then trained to match the **pruned old model** through `L_Min`. This clearly weakens the distillation target, but the paper repeatedly phrases this as if it literally frees capacity in the current model. As written, there is no persistent mask, structural sparsification, or direct constraint on student parameters showing that student capacity is actually “released”; instead, the method relaxes the old-knowledge target. That may still be useful empirically, but it is a weaker claim than the paper often makes. A more careful interpretation would be “target relaxation” rather than demonstrated parameter liberation in the learner.
- **`L_Esti` is insufficiently specified and partly mathematically unclear in the main paper.**  
  Equation (4) defines a position map via second-order spatial derivatives of a feature-difference quantity, but the paper does not clearly explain the discrete implementation used in training. The reviewer’s claim that this is entirely “unimplementable” overstates the issue, but the concern about underspecification is valid: for such an operation, readers need to know whether this is a fixed Laplacian kernel, finite differences, autograd-based second derivatives, or something else.  
  More importantly, Equation (5) is hard to parse dimensionally from the main text: `C_t(h,w) = < y_c^*(h,w), f_t(h,w) > / ||f_t(h,w)||_2`, while Eq. (2) defines `y_c^*` as an indicator-style pseudo-label quantity over old classes. The appendix later rewrites this in a way that suggests a vector quantity, but the main presentation does not make that representation explicit. This weakens technical clarity at a core part of the method.
- **The paper’s novelty claims around the entropy term are somewhat overstated relative to what is clearly established.**  
  `L_Max` is presented as maximizing shared knowledge distribution using marginal/conditional entropy. The formulation is reasonable and may be useful here, but from the paper itself it reads more like an information-theoretic regularizer encouraging batch diversity plus low per-sample entropy than a clearly new mechanism unique to CISS. The contribution is better justified as the integration with pruning/knowledge reuse, not as a standalone conceptual advance.
- **Some headline empirical framing is too strong for the actual numbers.**  
  The paper frequently emphasizes “near-upper-bound” or “approaches joint training.” This is credible in some settings, especially average ADE20K summaries, but not uniformly. For example, in VOC 2-2, Table 1 shows 75.0 All for DKD versus 70.3 for joint? The table extraction is noisy, so exact reading is difficult in places, but even from the clean textual summary, gaps to joint training are not negligible in all settings. The broader point is that the evidence supports **strong performance**, but the “near-upper-bound” characterization should be stated more carefully and per setting rather than as a blanket claim.

### Minor
- **Hyperparameter robustness is only partially demonstrated.**  
  The paper does include analysis for `γ` and `τ`, so this is not an omitted issue. Still, the chosen `γ` changes with scenario (“for settings involving more incremental steps ... γ is set to 0.4”), which suggests the method is somewhat schedule-dependent rather than governed by a single robust recipe.
- **Compute overhead is only lightly characterized for a method that introduces spatial second-order structure.**  
  The paper reports epoch-time overhead (e.g., DKD vs MKD/CKD), which is useful, but it does not break down memory overhead or clarify the actual cost of the Laplacian/projection computation. Since the method is positioned as practical and inference-neutral, a clearer training-cost analysis would help.
- **The distinction between the proposed confidence/position maps and prior confidence-based distillation methods could be made sharper.**  
  The paper does explain that its goal is knowledge reuse rather than merely selecting reliable old pixels, but the technical differentiation is not as crisp as it could be, especially given the centrality of these maps.

### Trivial
- **Theoretical analysis is extensive but not always as illuminating as the paper suggests.**  
  The appendix does provide derivations, but much of it verifies local optimization behavior rather than establishing a strong theorem about why DKD should resolve parameter competition in the student. This is not a flaw by itself, but the main-text claim of “theoretical analysis” should not be read as a deep guarantee.

## Nice-to-Haves
- Report an explicit architecture-controlled comparison emphasizing only ViT-based baselines in the main table narrative, even though the appendix already helps address portability.
- Quantify what “parameter release” means operationally in the student: e.g., gradient utilization, activation sparsity, or layer-wise parameter-change statistics after pruning the teacher.
- Visualize the pruned teacher outputs and the learned `P_t` / `C_t` maps to show they correspond to semantically reusable regions rather than merely weaker/noisier supervision.
- Add explicit foreground-vs-background confusion metrics, since background shift is central in CISS and the paper mainly reports mIoU and similarity matrices.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Comparisons are unfair because many baselines use ResNet101 while the method uses ViT, so the results should be discounted.”**  
  Removed in its strong form. The paper does compare against multiple **ViT-based** methods in Table 1/2 (e.g., MIB-ViT, SSUL†-ViT, MicroSeg†-ViT, CoinSeg, MBS†, Nest, Adapter-T, CoMFormer, INC), and the appendix also applies DKD to CoinSeg with ResNet101 and ViT. It is still fair to ask for clearer architecture-controlled emphasis, but the paper is not simply comparing ViT against only weaker ResNet baselines.
- **“The method is unreproducible because code/models are not available.”**  
  Removed. The paper explicitly states that code is included in the supplementary material and details are given in the appendix.
- **“The paper omits basic implementation details like LR schedule and thus is irreproducible.”**  
  Weakened/removed as a major concern. The paper gives optimizer, epochs, learning rates per dataset, hardware, and says more details are in supplementary material. The scheduler specifics could be clearer, but this is not a substantive weakness at ICLR level absent evidence that results hinge on hidden tricks.
- **“The parameter release mechanism has zero effect because zeroed weights immediately regrow under gradient descent.”**  
  Removed as stated, because it misreads which model is pruned. The pruning is applied to the **old model / teacher target**, not to the trainable current model. The real issue is not “zero effect,” but rather that the paper overinterprets teacher pruning as releasing student capacity.
- **“The paper should not claim strong results because it lacks formal significance testing such as paired t-tests.”**  
  Removed as a core criticism. In this empirical area, repeated runs with standard deviations are already a reasonable robustness check; formal significance testing would be a nice-to-have, not a standard requirement.

## Novel Insights
The most important synthesis is that the paper’s empirical strength and its conceptual strength are not perfectly aligned. DKD seems to work well largely because it **relaxes how old knowledge constrains the current learner and turns some of that old knowledge into a selective guidance signal**, which is a useful idea. However, the paper frames this as literal “parameter release” in the student, while the implemented mechanism more clearly acts by **weakening and reshaping the teacher signal**. That distinction matters: it does not invalidate the method, but it changes how one should understand the contribution and what evidence is still needed.

## Suggestions
- **Clarify the mechanism claim.** Rephrase “parameter release” to distinguish teacher-target relaxation from actual student-capacity release unless you can directly measure the latter.
- **Make Eq. (4)–(6) fully explicit in the main paper.** Specify the discrete Laplacian/second-order implementation, tensor shapes, and the representation space of `y_c^*` used in the confidence-map dot product.
- **Add direct evidence for student-side effects.** For example, report per-layer gradient norms, parameter drift, or activation-space occupancy showing that DKD truly reduces competition for new-class learning.
- **Tone down blanket “near-upper-bound” claims.** State this per benchmark/setting where supported.
- **Strengthen the positioning of `L_Max`.** Explain more clearly whether its value is the specific formulation, its interaction with `L_Min`/`L_Esti`, or its role in CISS specifically, rather than implying a broadly new entropy principle.
- **Expand practical-cost reporting.** Include VRAM and throughput overhead for the Laplacian/projection component, not only extra seconds per epoch.

---

## 9ktF3pwXi8

- GT: Reject (avg 4.7)
- Predicted: N/A (6.4/10)
- Match: N/A

### Final Review

## Summary
This paper argues that current end-to-end MLLM-based VLN agents underperform on basic navigation primitives, especially “move-to,” despite reasonable performance on long-horizon benchmarks. To address this, the authors introduce VLMB, a primitive-focused dataset built with automatic sampling plus spatial-semantic enrichment, and propose Move-to-Anything, a LLaVA-Video-based agent with hierarchical memory and temporal/segment embeddings. Empirically, the paper shows sizable gains on the new primitive benchmark and some positive transfer to composed multi-step tasks and joint training on R2R.

## Strengths
- **The paper identifies and operationalizes a concrete failure mode that is usually hidden by standard VLN reporting.** The central observation is specific: models that appear competent on long-horizon VLN can still be weak on basic “move-to” execution. The paper does more than state this qualitatively; it builds a benchmark around that primitive and reports that prior models achieve much lower success on this simpler skill than headline VLN numbers might suggest.
- **VLMB is a meaningful dataset contribution, not just a repackaging of existing VLN data.** The pipeline is fairly well specified: Stage 1 enforces visible, reachable, semantically meaningful targets; Stage 2 enriches instructions with spatial and semantic cues while filtering ambiguous cases; Stage 3 composes primitive instructions into multi-step evaluation episodes. The benchmark is materially broader than classical MP3D-only VLN setups in scene coverage, using both MP3D and HM3D and reporting 206 scenes / 873 target instances.
- **The paper’s primitive-centric training hypothesis receives nontrivial empirical support.** The method substantially improves over the paper’s evaluated baselines on the new primitive task, and Table 4 suggests that adding VLMB data during joint training improves R2R Val-Unseen over the same base model trained on R2R+RxR alone. That is important evidence that VLMB is not merely overfitting to an isolated toy task.
- **The memory mechanism appears practically useful even if not highly novel.** The hierarchical short-term/long-term split plus temporal/segment embeddings are simple but sensible for video-conditioned navigation, and the ablations support that they contribute beyond dataset curation alone (Table 5), while Table 6 gives some evidence that the chosen memory configuration is not arbitrary.
- **The paper makes an asymmetric comparison that is actually in the authors’ favor only modestly, not unfairly against baselines.** In Table 4, Move-to-Anything is compared with stronger prior systems that used much more external data, and still shows competitive behavior. This strengthens the case that the primitive dataset has value.

## Weaknesses

###: Fatal
None.

### Major:
- **The baseline evaluation protocol behind the headline “existing SOTA models achieve only 43.8% success” is insufficiently specified, which weakens the strength of the central comparative claim.** The paper says “we use the same base model … as employed in current SOTA approaches” and evaluates “general-purpose MLLMs, representative end-to-end VLN models, and the proposed method” on R-Nav/VLMB, but it does not clearly state for each baseline whether it is evaluated zero-shot, instruction-tuned only on its original long-horizon data, or adapted to the new primitive benchmark. This matters because the paper’s main narrative is comparative: a “fundamental gap” in existing training paradigms is being inferred from these numbers. If prior methods are simply out-of-distribution on the new primitive benchmark, the evidence is still interesting, but the causal conclusion about paradigm failure becomes weaker than the paper suggests.
- **The compositional generalization claim is currently supported only in a limited sense.** The multi-step benchmark in Section 3.2 Stage 3 is built by composing primitive episodes collected through the same pipeline, and the paper explicitly notes that “multi-step instruction compositions are exclusively included in the evaluation set.” This is useful evidence that the learned primitive can be chained, but it is not yet strong evidence of transfer to the full difficulty of natural long-horizon VLN instructions, which involve richer ambiguity, room transitions, indirect references, and heterogeneous subskills. Table 4 helps, but only via joint training on R2R+RxR+VLMB; the paper does not show zero-shot transfer from primitive-only training to a standard long-horizon benchmark.
- **The work’s scope is narrower than some of its framing implies: the method targets only one of the four primitives the paper identifies.** The paper argues that instructions can be grouped into “situation, move, change direction, and change region,” and that move accounts for 42%. However, VLMB and nearly all experiments focus exclusively on “move-to.” The authors do argue in Appendix B.1 that some other primitives can often be represented through move-to composition, but this is asserted rather than validated experimentally. As written, the evidence supports a strong claim about one foundational primitive, not yet a general primitive-based training framework for VLN as a whole.
- **The dataset curation strategy likely makes the benchmark cleaner and easier than unconstrained navigation, and the paper does not fully characterize that bias.** By construction, targets must be initially visible, uniquely identifiable in view, reachable, and within 3–10m, and half the collected samples are filtered out. This is a reasonable choice for isolating foundational skill acquisition, but it does limit robustness claims. The paper would be stronger if it quantified difficulty factors such as target distance, distractor density, ambiguity rate, or failure modes on more challenging cases. Without that, it is hard to know whether gains reflect stronger primitive execution broadly or mostly success on a carefully curated subdistribution.

### Minor
- **The architectural novelty is modest relative to the dataset/paradigm contribution.** The hierarchical memory plus temporal/segment embeddings are practical and seem effective, but they read as an incremental engineering refinement rather than a major algorithmic advance.
- **Some result presentation is confusing.** In particular, readers may struggle to reconcile the 60.6% figure emphasized in the abstract with the lower multi-step numbers in Table 3. This is not actually a contradiction—the 60.6% refers to single-step VLMB MP3D success, while Table 3 reports multi-step task performance—but the paper should make this distinction more explicit in the main text and tables.
- **Data-quality verification is described only qualitatively.** The paper says it “randomly sampled a subset of the improved data for manual verification,” but does not report sample size, agreement criteria, or quantitative precision of the MLLM-based filtering/enrichment pipeline. Given that GPT-4o is used both to enrich instructions and filter cases, a small quantitative audit would improve confidence in the benchmark.
- **There is little systematic failure analysis.** The appendix visualizations are mostly success cases, and the paper would benefit from a breakdown of where Move-to-Anything still fails—e.g., stopping too early, overshooting, distractor confusion, or poor heading control.

### Trivial
- **The paper would benefit from clearer table formatting and sharper distinctions between R-Nav, VLMB, and multi-step settings.** This is a presentation issue rather than a substantive flaw.

## Nice-to-Haves
- Add a control baseline trained on the same amount of data but without the VLMB enrichment/filtering, to better isolate the value of the data pipeline from the value of primitive-focused supervision.
- Evaluate primitive-only training in zero-shot transfer to a standard long-horizon benchmark, or at least add a stronger controlled comparison against direct training on similarly composed multi-step instructions.
- Provide policy-level diagnostics such as collision counts, stop accuracy, or error categories, not just trajectory-level SR/SPL/OS/NE.
- Include a small quantitative audit of instruction enrichment/filtering quality and representative failure-case visualizations.
- Discuss sim-to-real limitations more explicitly. Simulation-only evaluation is standard for this literature, so this is not a core flaw, but the paper’s practical robotics framing would benefit from a short realism discussion.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Unfair comparison because baselines are trained on long-horizon tasks and evaluated outside their distribution.”** This is only partially reasonable. The whole point of the paper is to probe whether long-horizon-trained agents actually possess primitive skills. Evaluating baselines on the primitive benchmark is therefore legitimate. The real issue is not unfairness per se, but the lack of precise protocol description for how those baselines were adapted or not adapted.
- **“The 60.6% vs 36.3% numbers are inconsistent.”** This is a misreading. The paper distinguishes single-step VLMB/R-Nav evaluation from multi-step evaluation. The numbers come from different tasks, not contradictory reporting.
- **Strong novelty objection claiming the paper is not first to note primitive weakness.** The paper says “To the best of our knowledge, we are the first to point out this phenomenon.” Without external verification this should not be used as a weakness. The safer criticism is that the paper should scope its novelty claim carefully.
- **Reproducibility complaints about omitted hyperparameters / artifacts.** The paper gives enough implementation detail for a conference submission, includes appendices with prompts and rules, and provides an anonymous repository link. Further implementation minutiae are not central weaknesses here.
- **Pure formatting complaints.** Parser artifacts in the extracted text are not paper flaws.

## Novel Insights
The most interesting synthesis across the reviews is that the paper’s strongest contribution is not the memory module but a reframing of what it means to evaluate “general” VLN capability. By showing that long-horizon success can coexist with weak primitive execution, the paper exposes a potential evaluation blind spot in current embodied benchmarks: aggregate instruction-following metrics may reward recovery, path stochasticity, or benchmark-specific regularities without confirming mastery of the atomic behaviors that real deployment depends on. That said, the current evidence most convincingly establishes this for the “move-to” primitive under controlled conditions, not yet for primitive compositionality in the broader VLN sense.

## Suggestions
- Specify the baseline protocol in detail: for each compared model, state training data, whether any adaptation to R-Nav/VLMB was performed, and if so under what compute/data budget.
- Soften the broadest claims from “primitive-based VLN” to “move-to primitive learning” unless additional primitives are experimentally covered.
- Add one stronger control: direct training on equally sized non-primitive or composed data, to isolate the benefit of primitive decomposition from curation quality.
- Include quantitative dataset-audit statistics for the GPT-4o enrichment/filtering stage.
- Add failure analyses and difficulty-stratified evaluation by distance, distractor count, spatial relation type, and stop behavior.
- If space allows, report zero-shot primitive-to-long-horizon transfer on a standard benchmark, even if preliminary; that would materially strengthen the compositionality claim.



---

## oG3UEPs0Ov

- GT: Withdrawn (treated as Reject) (avg 0.5)
- Predicted: N/A (1.8/10)
- Match: N/A

### Final Review

## Summary
The paper argues that visual afterimages are cortical rather than retinal in origin, using the phenomenon of perceiving the physiological blind spot as an afterimage to localize the first-stage substrate to V1 layer 4. Building on this, it proposes a broad cortical architecture in which superficial layers (L2/3) are feedforward, deep layers (L5/6) are feedback, and middle layer 4 serves as short-term memory.

## Strengths
- **The paper makes a concrete, distinctive anatomical-theoretical link rather than staying at the usual vague “cortical involvement” level.** In Section 2.3, it ties the blind-spot-afterimage argument specifically to the known representation of the blind spot in V1-L4, yielding a sharper claim than generic statements that afterimages are “in the brain.”
- **It revives and synthesizes an unusual historical phenomenology in a way that is genuinely central to the thesis.** The La Hire–Purkinje phenomenon and the Franklin effect are not decorative historical material here; they are used as the main evidential bridge from phenomenology to neural interpretation.
- **The paper offers a clear, falsifiable architectural hypothesis.** The proposed assignment “L2/3 feedforward, L4 STM, L5/6 feedback” is easy to understand and potentially testable, even though it is not yet validated here. Figure 5 crystallizes the intended claim at the level of cortical computation.

## Weaknesses
### Fatal
- **The submission does not substantiate its central inferential leap from blind-spot phenomenology to “V1-L4 is the neural site for afterimages,” yet that claim is used as the foundation for the rest of the paper.** The paper’s key move is: blind spots can be seen as afterimages; blind spots are represented in V1-L4; therefore afterimages are localized to V1-L4. But Section 2.3 only provides correlational reasoning, not a causal or exclusionary argument. The text explicitly claims that these findings “decisively and precisely pinpoint the first-stage neural substrate of afterimages to V1-L4,” which is much stronger than the evidence provided. Even if the phenomenon is real and interesting, the paper does not rule out multi-stage contributions or establish that V1-L4 is the locus of storage rather than an early representational stage.
- **The claimed “computational architecture of the human brain” is not actually developed as a computational model.** Despite the title and repeated use of “computational architecture,” the paper offers no formalization, no dynamical mechanism, no learning rule, no simulation, and no computational test of the proposed layer roles. Section 4 moves from “afterimages are neural persistence” to “afterimages are STM” to “L4 is STM” and then to the architecture in Figure 5, but these transitions are conceptual assertions rather than computational derivations. For an ICLR paper, this is a fundamental mismatch between the claimed contribution and what is delivered.

### Major:
- **The paper overgeneralizes from a specific visual phenomenon in V1 to a universal cortical architecture.** The evidence discussed concerns afterimages and blind-spot representations in early visual cortex, yet the abstract and conclusion generalize to “the computational architecture of the brain” and to “each cortical area.” The manuscript does not provide evidence that the same role assignment holds outside the visual system, nor does it address obvious scope limitations. This makes the main claim read as substantially broader than the paper can support.
- **The identification of afterimages with short-term memory is underdefined and insufficiently justified.** Section 4 says that because afterimages are cortical and persistent, they “should better be conceived as visual STM.” But persistence alone is not enough to establish a memory mechanism in the computational sense. The paper does not define STM operationally, specify how maintenance occurs in L4, or explain what properties distinguish this proposed STM from sensory persistence or adaptation. Since the architecture depends on L4 being a memory substrate, this conceptual gap matters directly.
- **The manuscript provides no new controlled empirical evidence for its central observational premise.** The “rediscovery” of the La Hire–Purkinje phenomenon is presented as important motivation, but the paper does not report a modern psychophysical experiment quantifying reliability, duration, subject variability, or experimental controls. As written, the empirical basis is a historical synthesis plus the authors’ own observations, which is too thin for the strength of the claims being made.
- **The treatment of positive and negative afterimages remains too assertive relative to the evidence presented.** Section 3 uses the Franklin effect to argue against distinct mechanisms for positive and negative afterimages and then adopts a shared-persistence account. The phenomenon certainly motivates a unified account, but the paper does not actually provide a mechanistic explanation of the polarity reversals or show that alternative multi-stage explanations fail. Thus the conclusion that positive and negative afterimages “share the same neural substrate” is plausible as a hypothesis but not established here.

### Minor
- **The paper’s framing is often more absolute than the evidence warrants.** Phrases such as “the Retinal View is erroneous and only the Brain View is correct” and “decisively and precisely pinpoint” overstate what is defended in the manuscript. A more scoped presentation as a hypothesis or reinterpretation would be better aligned with the evidential level.
- **The proposed architectural role of L4 lacks a mechanism consistent with the paper’s own level of analysis.** Even setting aside the need for a full model, the manuscript should at least explain how L4 could maintain state over time and support the claimed memory function, rather than only naming it as STM.

### Trivial

## Nice-to-Haves
- Add a simple computational instantiation of the proposed laminar architecture, even at a toy level, to show what “L4 as STM” means operationally.
- Run a controlled psychophysical study of blind-spot afterimages with fixation control and time-decay measurements.
- Narrow the scope of the claims to visual cortex unless the paper can justify extension to the entire cortex.
- State falsifiable predictions that would distinguish this account from other cortical or multi-stage explanations of afterimages.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper ignores alternative related theories/models.”** I removed this as a main weakness because I cannot verify missing related work beyond what is in the manuscript, and the review should not speculate about uncited literature.
- **“The paper is weak because comparisons are unfair / missing against specific external methods.”** Not applicable here: the paper does not present a benchmarked algorithmic comparison setup.
- **Generic strengths such as “the paper is well-written,” “the topic is important,” or “the experiments are extensive.”** These are too generic or inapplicable.
- **Pure reproducibility nitpicks about missing implementation details.** There is no implemented model; the substantive issue is absence of a computational model, not omitted hyperparameters.
- **Claims that the paper’s concerns about cited models, references, or historical sources are unverifiable.** Such concerns are disallowed and not evidence-based.
- **Some overly strong reviewer assertions about established neuroscience facts contradicting the paper.** For example, statements that specific laminar physiology “contradicts” the proposal are stronger than what can be verified from the paper alone. The more defensible criticism is that the paper does not justify the L4-memory claim, not that the opposite has been conclusively proven here.

## Novel Insights
The strongest synthesis across the reviews is that the paper has an unusual asymmetry: its most interesting contribution is not the grand cortical architecture, but the narrower observation that blind-spot afterimages may provide an anatomically anchored probe of subjective visual persistence. If the paper were reframed around that phenomenon as a hypothesis-generating bridge between entoptic perception and laminar neuroscience, it would read as an intriguing conceptual contribution. The current version weakens itself by escalating that insight into a universal cortical architecture and a definitive localization claim without the computational or empirical support needed to sustain those leaps.

## Suggestions
- Reframe the paper around a **scoped hypothesis**: blind-spot afterimages suggest a cortical contribution, plausibly involving V1-L4, rather than claiming decisive localization.
- Either **add an actual computational model** of the proposed laminar roles or substantially soften the “computational architecture” claims.
- Define **short-term memory** operationally and explain what dynamical or circuit mechanism in L4 is supposed to instantiate it.
- Add at least one **modern empirical validation**, ideally a controlled psychophysical experiment on blind-spot afterimages.
- Separate clearly what is **evidence**, what is **inference**, and what is **speculation**, especially in Sections 2.3 and 4.
- Narrow claims about general cortical organization unless the manuscript can justify extension beyond early visual cortex.

---

## qbDnX2YC6F

- GT: Reject (avg 4.5)
- Predicted: N/A (6.8/10)
- Match: N/A

### Final Review

## Summary
This paper studies open-set recognition as a modular combination of representation learning (RL) and postprocessing (PP), and asks when these components interact constructively or destructively. Its main empirical findings are that (i) auxiliary-data-based magnitude-manipulating RL methods such as OE/OS can fail badly at large scale when auxiliary classes are semantically similar to known classes, via a phenomenon the authors term **magnitude collapse**, and (ii) simple AddON training combined with magnitude-aware PP (especially PostMax/GHOST) yields the strongest overall performance on the tested large-scale protocols.

## Strengths
- **The paper makes a genuinely useful shift from evaluating monolithic OSR methods to analyzing RL/PP compositions explicitly.** The decomposition into RL and PP components is operationalized throughout the experiments rather than being only rhetorical: Figure 2, the delta decomposition in Eq. (3), and the interaction plots in Figure 6 provide a concrete way to inspect modular gains and failures.
- **The large-scale evaluation is the most compelling part of the paper.** The contrast between CIFAR+N and ImageNet P1/P2/P3 exposes a meaningful empirical pattern: methods using auxiliary data that look strong on small-scale benchmarks do not retain that advantage at high semantic similarity. This is directly supported by Table 4, where OE/OS are competitive on P1 but no longer clearly advantageous on P2/P3, while AddON+PostMax reaches the top AUOSCR on P3 (79.7).
- **The identification of a class-imbalance failure mode tied to feature magnitude is insightful and backed by more than one view of the data.** The paper does not rely only on aggregate metrics: it shows shifted feature-magnitude distributions (Figure 4 / Figure 10), class-wise CCR–magnitude regressions (Figure 5 / Figure 13), and an EMNIST case study with visually similar auxiliary classes where specific classes collapse much more severely than others (Table 5).
- **The paper contributes an unusually strong practical baseline rather than only criticizing prior methods.** AddON is simple, easy to combine with existing PP methods, and performs consistently well across scales. The strongest practical recommendation—use non-MM RL, especially AddON when auxiliary data is available, plus MA PP—follows directly from the experimental matrix rather than from a narrow cherry-picked comparison.
- **The appendix gives a nontrivial theoretical rationale for why CE/AddON training tends to preserve sufficiently large feature magnitudes.** This does not prove the whole mechanism, but it usefully connects the observed behavior to the optimization objective rather than leaving the discussion entirely descriptive.

## Weaknesses

###: Fatal
None.

### Major:
- **The mechanistic claim that magnitude collapse is the primary cause of large-scale degradation is suggestive but not fully isolated causally.** The evidence is consistent with the story—feature magnitudes shrink for certain known classes under OE/OS, and class-wise CCR tracks magnitude more strongly in difficult settings—but the paper does not include an intervention that changes magnitude behavior while holding the rest of the training setup fixed. For example, there is no controlled variant of OE/OS that counteracts magnitude shrinkage, nor an ablation that decouples feature norm from directional information. As a result, the paper strongly supports magnitude collapse as a plausible and important mechanism, but does not fully prove it is the sole or dominant cause.
- **AddON’s advantage is not cleanly disentangled from its architectural difference relative to OE/OS.** The paper’s central contrast is between AddON and magnitude-manipulating methods, but AddON uses a \(K+1\) output node while OE/OS use \(K\) outputs with auxiliary regularization. The paper itself derives in Appendix A.2 / Eq. (4) that CE-style training exerts upward pressure on feature magnitude to achieve confident predictions; that argument helps explain AddON’s robustness, but it also highlights that the comparison conflates training objective and output-space parameterization. The current experiments show that AddON works well and OE/OS can collapse, but they do not fully isolate whether the key factor is “avoid magnitude manipulation,” “use an explicit auxiliary class,” or both. This matters because one of the paper’s central scientific claims is about *why* AddON avoids the failure mode.
- **Some of the broadest claims are stronger than what the experimental scope strictly establishes.** In particular, statements such as “invalidating current best practices in OSR research” are too sweeping relative to the actual study. The paper convincingly shows that small-scale auxiliary-data evaluations are poor predictors for the tested large-scale protocols, but the experiments are confined to a specific family of discriminative RL methods trained from scratch and a fixed set of PP methods. The evidence supports a strong warning against relying on CIFAR-style conclusions, but not a blanket invalidation of best practices for all modern OSR pipelines.
- **Large-scale results appear to rely on single runs, while some conclusions are phrased as robust laws.** The paper averages over 5 trials on CIFAR+N, but the ImageNet tables report single values with no variance. Since the headline claims are largely about the large-scale setting, the absence of multi-seed estimates weakens confidence in the exact size of the reported interaction gains and degradations, especially when some differences are modest. This is less problematic for the larger qualitative trends (e.g., OE/OS deteriorating on P2/P3), but it does limit how strongly one should read the additive-gain and interaction-effect claims quantitatively.

### Minor
- **The backbone robustness evidence is limited.** The paper does include one additional architecture (Swin-B on P2), which is useful, but this section is narrow and somewhat inconclusive because ARPL training is reported as unstable and not analyzed further. So while the main findings likely extend beyond a single backbone, the paper does not fully establish architecture-agnosticity.
- **The “independent contributions” claim is stronger than the analysis warrants.** Figure 6 shows near-additive behavior for RL methods trained without auxiliary data, which is a good empirical observation. But “independent” here is best interpreted as an empirical approximation under the chosen metrics and methods, not a principled statistical independence result.
- **The EMNIST case study is informative but should remain clearly auxiliary evidence.** The authors do acknowledge that 2D bottleneck visualizations have limited transferability to high-dimensional feature spaces. That caveat is appropriate, but because the EMNIST section is used to build intuition about the main mechanism, the paper could more clearly separate “illustrative toy confirmation” from “core evidence.”
- **The practical cost trade-off of the recommended pipeline is not quantified.** AddON is simple, and the extra class node itself is negligible, but the recommended two-stage pipeline still adds training/selection overhead through auxiliary data usage and PP fitting. A brief runtime/complexity discussion would make the practitioner guidance more complete.

### Trivial
- **The paper could sharpen the terminology around “magnitude-aware” PP methods.** MaxLogits/MLS, PostMax, and GHOST are all reasonably categorized as magnitude-aware, but they exploit magnitude in different ways. A slightly finer distinction would improve conceptual clarity, especially because their behavior under collapse differs.

## Nice-to-Haves
- Add a targeted ablation that modifies OE/OS to suppress or restore feature-norm behavior, to test whether preventing norm collapse alone recovers performance.
- Add a controlled comparison that separates the effect of the \(K+1\) output node from the effect of the training objective.
- Report multi-seed variance for the ImageNet protocols, at least for the main claimed comparisons (CE/ARPL/AddON vs OE/OS; MSP vs PostMax/GHOST).
- Quantify semantic similarity more explicitly, rather than relying mainly on protocol design and qualitative interpretation, to better operationalize when collapse is expected.
- Provide per-class CCR plots on the main ImageNet protocols, not only summary regressions and EMNIST ranges, to strengthen the claim of systematic class-wise imbalance.
- Briefly analyze the Swin-B/ARPL instability or frame that result more cautiously.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that the paper is not novel because prior work already dissected modular OSR components.** The paper explicitly acknowledges related modular distinctions (e.g., Wang et al., 2025) and its actual novelty claim is narrower: “for the first time in OSR literature, we explore the modularity and interaction effects of representation learning and postprocessing methods.” Based on the paper text, that claim is plausible and not directly contradicted by the cited discussion.
- **Criticism that the paper lacks enough backbone diversity because it does not test many more architectures (e.g., EfficientNet, ViT, CLIP, DINOv2, LoRA).** This is outside the paper’s stated scope and not necessary to support its core empirical contribution on RL/PP interactions under the tested settings.
- **Requests for missing related methods/baselines not in the paper (e.g., NPOS or other recent methods).** Per instruction, such claims cannot be validated here and are not appropriate grounds for criticism.
- **Reproducibility complaints about release status or availability of code/artifacts.** The paper provides substantial implementation detail and states code will be open-sourced upon publication; no stronger criticism is warranted here.
- **Pure formatting/parser issues from the extracted text.** These are artifacts of extraction, not paper weaknesses.
- **Demand for confidence intervals or formal significance tests on every interaction plot as a core flaw.** Multi-seed ImageNet variance is a reasonable substantive request and is retained above, but requiring full formal significance machinery for every large-scale benchmark plot would be a methodological nice-to-have rather than a decisive flaw in this subfield.

## Novel Insights
The most interesting synthesis is that the paper’s strongest contribution is not merely “AddON beats OE/OS,” but a more structural lesson: auxiliary data is not inherently beneficial or harmful in OSR—its effect depends on whether the RL objective preserves a representation that remains usable by downstream PP. In that light, the paper suggests a practical division of labor: when RL is trained only on known classes, RL and PP are close to modular and additive; once auxiliary data enters training, RL can reshape feature geometry in ways that either enhance PP (AddON) or sabotage it (OE/OS), especially for PP methods that rely on magnitude. This is a meaningful systems-level viewpoint on OSR design, and it explains why “better OOD separation” by itself can coexist with worse open-set recognition due to damaged class-wise discriminability.

## Suggestions
- Add one decisive causal ablation on magnitude collapse: e.g., an OE/OS variant with explicit norm floor, norm clipping, or a decoupled directional classifier, and show whether the large-scale degradation is reduced.
- Separate architecture/output-space effects from objective effects by comparing \(K\)-way vs \(K+1\)-way variants under matched losses where possible.
- Temper the strongest rhetoric in the abstract and conclusion. The evidence supports a strong caution about small-scale auxiliary-data evaluations, but not a universal invalidation claim.
- Report at least 3-seed ImageNet results for the main tables or the headline method pairs.
- Expand the per-class analysis on ImageNet P2/P3 to show that the observed degradation is not only aggregate AUOSCR movement but truly class-imbalanced collapse.
- Clarify the practical deployment recipe as a concise decision rule: when auxiliary data is available and likely similar to known classes, prefer AddON + MA PP; when no auxiliary data is available, pair CE/ARPL with MA PP for modular additive gains.

---

## rzGEfYr2ZC

- GT: Reject (avg 0.0)
- Predicted: N/A (6.1/10)
- Match: N/A

### Final Review

## Summary
This paper proposes **SparseFW**, a layerwise post-training pruning method for LLMs that relaxes binary mask selection to a convex program over the mask polytope and solves it with Frank–Wolfe. The paper’s most distinctive technical contributions are (i) a clean optimization view of mask selection and of Wanda/RIA as greedy approximations to that objective, (ii) an efficient FW implementation via precomputing \(G=XX^\top\) and \(H=WG\), and (iii) empirical evidence that FW can substantially reduce local reconstruction error and modestly improve final perplexity/zero-shot accuracy over Wanda/RIA on several 7B–14B models.

The paper is novel and technically interesting, but the central empirical story is more fragile than the framing suggests: the version that works best is not vanilla FW, but a hybrid that fixes a large fraction of Wanda-selected weights and only optimizes the remainder. This does not negate the contribution, but it does materially change what the paper has shown.

## Strengths
- **A genuinely insightful optimization reinterpretation of existing pruning heuristics.** Section 2.1 does more than present another pruning method: it derives Wanda as minimizing the one-weight pruning objective without reconstruction and interprets RIA as the same greedy procedure on a rescaled weight matrix. This is a specific conceptual contribution that clarifies what these popular methods are actually optimizing.
- **The convex-relaxation formulation is principled and algorithmically well matched to FW.** The feasible set \(C_k=\{M\in[0,1]^{d_{out}\times d_{in}}:\|M\|_1\le k\}\) is a natural relaxation of binary mask selection, and the LMO is especially simple: select up to \(k\) most negative gradient entries. This makes the method mathematically clean while preserving sparse updates.
- **The implementation insight is practically meaningful.** The paper exploits that both objective and gradient depend on \(X\) only through \(G=XX^\top\), and precomputes \(G\) and \(H=WG\). This reduces dependence on calibration sequence length and sample count during iterative optimization, which is a concrete systems/engineering strength rather than a generic “efficient implementation” claim.
- **The method supports both unstructured and semi-structured sparsity within the same framework.** Appendix D shows how the LMO extends to \(n\!:\!m\) sparsity by separability over blocks; empirically, the paper reports results for both 50/60% unstructured and 2:4 sparsity across multiple modern GPT-family models.
- **The paper is unusually transparent about a key failure mode.** Section 2.3 and Appendix C explicitly state that unconstrained vanilla FW often improves the local pruning objective yet can worsen perplexity, and they provide the \(\alpha\)-ablation showing this. That honesty is valuable and helps readers understand the real scope of the method.

## Weaknesses
###: Fatal
None.

### Major:
- **The empirical success is driven by a hybrid “fix-most-of-Wanda” variant, not by vanilla FW alone, and the paper’s framing understates this.**  
  This is directly supported by the paper itself. Section 2.3 states:  
  > “setting \(\alpha = 0.0\) (full FW without any fixed weights) consistently yields worse results than the baselines.”  
  Appendix C further shows that the strongest gains often occur around \(\alpha=0.9\), i.e., fixing 90% of the kept weights by Wanda saliency and optimizing only the remaining 10%. This substantially weakens the headline narrative that FW “overcomes” greedy pruning. The evidence instead supports a more nuanced claim: **FW is useful as a constrained local refinement on top of a strong saliency prior**. That is still interesting, but materially narrower than the abstract/introduction suggest.
- **The main-text algorithm presentation obscures the method actually used in the strongest experiments.**  
  Algorithm 1 presents plain FW plus thresholding, while the practically necessary variant appears only later as a “caveat” in Section 2.3 and formally in Appendix B (Algorithm 2). Given that the paper itself reports that \(\alpha=0\) is consistently worse than baselines, the saliency-fixing mechanism is not an implementation detail; it is central to the empirical method. This affects clarity and claim calibration.
- **Theoretical guarantees are only partially aligned with the empirically successful algorithm.**  
  The main theory in Section 4 / Appendix E establishes guarantees for the relaxed problem plus top-\(k\) rounding of the FW iterate. But the experimentally strongest method adds an extra constraint: fixing a subset of high-saliency weights beforehand and optimizing only the complement. The paper does not extend the guarantee to this hybrid algorithm, even though that is the version supporting the main empirical claims. So the theory is sound as far as it goes, but it does not fully justify the method that actually matters most in practice.
- **The paper lacks a concrete compute-cost accounting despite using substantially more optimization than the baselines.**  
  The paper acknowledges this limitation (“SparseFW is clearly more compute-intensive than Wanda and RIA”) and reports using 2000 FW iterations per layer. But there is no wall-clock, FLOP, or pruning-time comparison to Wanda/RIA. Since the final perplexity gains are often modest in absolute terms, it is important to quantify the cost/benefit trade-off rather than discuss it qualitatively.
- **The connection between large local objective improvements and modest end-task gains remains underexplained.**  
  The paper convincingly shows sizable reductions in per-layer pruning error (Figure 2; often 20–40% average, up to 80% in some layers), yet final perplexity improvements in Table 1 are much smaller and sometimes mixed. The paper does acknowledge a “mismatch between local and global objectives,” but this becomes a central unresolved issue: the proposed optimization target is demonstrably improved, yet that does not reliably translate into corresponding model-level gains without additional inductive bias from Wanda.

### Minor
- **Main-result uncertainty is hard to assess because Table 1 omits standard deviations.**  
  The paper says “We omit standard deviations for legibility,” but some reported improvements are small enough that variability matters for interpretation. Figure 3 does include seed ranges for one ablation, which is helpful, but the main comparison table would be stronger with at least compact uncertainty reporting.
- **The paper’s “state-of-the-art” phrasing should be narrowed to the comparison class actually studied.**  
  Section 3 explicitly restricts comparisons to methods “that also aim to find a better pruning mask by solving (MASK SELECTION)” and excludes reconstruction-based approaches such as SparseGPT. That is a reasonable scoped evaluation choice, but then claims should be consistently phrased as improvements over strong **mask-selection** baselines rather than over LLM pruning methods broadly.
- **The mechanism behind the stronger gains in some sparsity regimes/patterns is not deeply analyzed.**  
  The 2:4 and higher-sparsity settings seem to benefit more consistently than 50% sparsity, but the paper does not provide much insight into when FW refinement is most valuable and why.

### Trivial
- **The role of thresholding dynamics could be explained more directly.**  
  Figure 4 is interesting and the discussion is plausible, but readers would benefit from a clearer practical takeaway: whether the thresholding plateau is mostly due to the FW step-size schedule, lack of vertex convergence, or an inherent property of the relaxation in this setting.

## Nice-to-Haves
- Add a pruning-time / accuracy Pareto analysis against Wanda and RIA.
- Report mask overlap or Hamming/Jaccard distance between the warmstart and final SparseFW masks to show how much the optimization actually changes.
- Move the \(\alpha\)-ablation from the appendix into the main paper, since it is central to interpreting the method.
- Analyze which layer types or matrix types contribute most to the final gains, given the local/global mismatch.
- If space allows, include at least one broader baseline with light reconstruction, while keeping the paper’s primary scoped comparison intact.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The theoretical guarantee does not apply to unstructured sparsity at all because it is only row-wise.”**  
  Removed in this strong form because it overstates the issue. The appendix explicitly says:  
  > “For simplicity, we work in the row-wise formulation; the proof for the full-matrix case follows by the same arguments.”  
  So it is not accurate to claim the theory is strictly limited to row-wise/separable settings. The valid concern is narrower: the theory does not cover the **hybrid fixed-mask variant** used for the best results.
- **Criticism based on absence of release / reproducibility of cited models or tools.**  
  Removed per instruction.
- **Pure formatting/style objections.**  
  Removed per instruction.
- **Demands for many extra downstream benchmarks as a core flaw.**  
  Weakened and not kept as a main weakness. The current evaluation on WikiText perplexity and EleutherAI zero-shot accuracy is not unusually narrow for this line of work; additional tasks would strengthen the paper but are not necessary to establish the paper’s scoped claims.
- **Claims that the method is not novel because it “just uses a different optimizer.”**  
  Removed in that simplistic form. Recasting mask selection as convex relaxation with FW, providing the greedy reinterpretation of Wanda/RIA, and deriving theory are real contributions. The better criticism is that the empirical gains depend heavily on the hybridization with Wanda.

## Novel Insights
The most important synthesis is that this paper’s real contribution is not “Frank–Wolfe replaces greedy pruning,” but rather **Frank–Wolfe exposes a useful separation between globally indispensable weights and locally refinable ones**. The experiments suggest Wanda is good at identifying a protected core set of weights, while FW improves the combinatorial search only on the residual degrees of freedom. In that sense, the paper uncovers a structural fact about LLM pruning: optimizing the local quadratic objective more faithfully helps, but only after injecting a strong prior about which weights must remain untouched. This makes the work more interesting scientifically than a simple “better benchmark number” paper, but also means the paper should be reframed around hybrid optimization rather than around pure replacement of greedy heuristics.

## Suggestions
- Reframe the paper’s main claim around **hybrid saliency-constrained FW refinement**, not pure FW replacing greedy heuristics.
- Move Algorithm 2 and the \(\alpha\)-ablation into the main text, and make clear upfront that this is the primary practical method.
- Either extend the theory to the fixed-mask variant or clearly separate “theory for vanilla relaxation” from “empirical gains for the hybrid algorithm.”
- Add a concise table with pruning wall-clock/runtime and memory overhead versus Wanda/RIA.
- Include uncertainty for Table 1, even if only for a representative subset or via compact ± values.
- Quantify how much SparseFW changes the warmstart mask, and where those changes occur across layers/matrix types.
- Clarify in the abstract/introduction that the method improves over strong **mask-selection baselines**, and that unconstrained FW alone does not reliably improve perplexity.

---

## 4Mvdn1m861

- GT: Reject (avg 3.5)
- Predicted: N/A (5.4/10)
- Match: N/A

### Final Review

## Summary
This paper proposes TOKENCOUNT, a training-free object counting framework built on SAM that adds two components: probabilistic prompt generation and output-token-based verification. The key empirical takeaway is that the method is strong among **SAM-only training-free** approaches and is particularly effective on CARPK, but its broader claims of superiority over existing training-based and training-free methods are overstated relative to the paper’s own results.

## Strengths
- **Clear SAM-only training-free contribution with meaningful empirical value.** The method avoids auxiliary encoders or learned add-ons, yet improves substantially over prior SAM-only baselines on FSC-147 (e.g., 16.25 MAE vs. 19.95 for TFOC and 27.97 for Count-Anything) and performs strongly on CARPK (4.68 MAE).
- **The paper identifies and exploits a non-obvious signal inside SAM: decoder output tokens rather than only encoder/image-embedding similarity.** This is a specific and interesting design choice, and Table 5 supports that it helps relative to verifying in image-embedding space (16.25 vs. 16.94 MAE on FSC-147).
- **The ablations do support that both major components matter.** Table 3 shows output-token verification with TS-SS is slightly better than standard metrics, and Table 4 exposes a practical accuracy/compute tradeoff over iteration and prompt budget rather than hiding it.
- **CARPK results are genuinely strong.** On the reported protocol, the method outperforms all listed supervised baselines and all listed training-free methods except TFCAC, while using only SAM rather than SAM plus additional models.

## Weaknesses

### Fatal
- None.

### Major:
- **The paper materially overclaims its overall performance relative to prior work.**  
  The abstract states the method achieves “superior accuracy… outperforming existing training-based and training-free counting methods,” and the introduction similarly frames the method as outperforming prior methods broadly. This is not supported by Table 1 on FSC-147, where several training-based methods are clearly better (e.g., LOCA 10.79, PseCo 13.05, SAFECount 14.32 vs. Ours 16.25), and one training-free method with auxiliary models (TFCAC 12.26) is also better. The stronger, accurate claim is that the method is best among the listed **SAM-only training-free** methods and especially strong on CARPK. As written, the headline positioning overstates the evidence.
- **The contribution of the probabilistic prompt generator is not cleanly isolated from brute-force prompt budget effects.**  
  Table 4 shows performance improves steadily as iterations/prompts increase, which is useful, but does not by itself establish that the gain comes from the *probabilistic* strategy rather than simply querying more points. The paper motivates superiority over grid/superpixel prompting in Section 3.1, but it does not provide a controlled comparison at matched prompt budgets against alternative prompt-generation strategies. This matters because the prompt generator is one of the two central proposed components.
- **Efficiency claims are under-supported relative to how prominently they are used in the paper’s motivation and comparisons.**  
  The paper criticizes prior methods for computational overhead and claims advantages such as being “more than twice as fast” than TFCAC, yet the empirical support is limited to a single runtime number (1.69 s/image on FSC-147) and average prompt counts in Table 4. There is no side-by-side latency/memory comparison against baselines on shared hardware, nor a breakdown by iteration/decoder calls. Given that the method performs iterative sampling and repeated verification, the efficiency story is plausible but not convincingly substantiated.

### Minor
- **The justification for TS-SS as the preferred verification metric is weaker than the paper’s narrative suggests.**  
  Section 3.2 gives an intuition about output tokens encoding both semantic and positional information, but the rationale for choosing TS-SS specifically is heuristic, and the empirical gain over cosine/L2 is modest in Table 3 (16.25 vs. 16.48/16.53 MAE). This does not negate the result, but it means the metric innovation is less compelling than the paper implies.
- **Sensitivity to key hyperparameters is insufficiently characterized.**  
  The implementation fixes a token verification threshold of 300 and uses temperature scaling in the prompt distribution, but there is no threshold/temperature sensitivity analysis. Since verification is central to the counting decision, robustness to these settings should be shown more directly.
- **Robustness in dense/small-object regimes is acknowledged but not quantified.**  
  The discussion explicitly notes degradation on “dense distributions of very small instances,” which is helpful candor, but the paper does not stratify results by object size, density, or occlusion to show where the method breaks down. This leaves an important practical limitation somewhat anecdotal.
- **Some methodological details remain underspecified.**  
  While the paper does describe using SAM decoder output tokens, it remains somewhat unclear exactly which token representation is extracted and compared for each candidate across the iterative procedure, and how this interfaces with batching/caching in runtime. The high-level mechanism is understandable, but more precise implementation detail would improve reproducibility and make the efficiency claims easier to assess.

### Trivial
- **The discussion of computational efficiency is internally a bit muddled.**  
  Section 5 claims “superior computational efficiency compared to state of the art,” but immediately follows with “This limitation affects scalability for real-time applications and large datasets.” The intended message is likely that the method is more efficient than some competing training-free pipelines yet still not truly real-time, but the wording is confusing.

## Nice-to-Haves
- Add a matched-budget comparison of probabilistic prompting vs. uniform grid vs. superpixel prompting to isolate the real source of gains.
- Add a standardized efficiency table with latency, peak VRAM, and prompt counts for major baselines on identical hardware.
- Provide threshold/temperature sensitivity plots and exemplar-selection sensitivity, especially for CARPK where 12 exemplars are randomly selected from the training set.
- Stratify FSC-147 performance by object size/count density to quantify the failure mode discussed in Section 5.
- Show qualitative cases where TS-SS succeeds over cosine similarity on ambiguous distractors; this would make the metric’s practical utility easier to judge.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“CARPK comparison is invalid because supervised methods and exemplar-based methods are inherently incomparable.”**  
  Removed in strong form. The comparison may be imperfect in terms of setting, but the paper is transparent about its CARPK protocol (“we randomly selected 12 objects from the training set to use as exemplars”), and reporting against standard supervised results is still informative. The fair criticism is not that the comparison is invalid, but that the paper should be more careful in how broadly it generalizes conclusions from it.
- **“SAM’s decoder does not output tokens, so the method is conceptually invalid.”**  
  Removed as factually wrong. The paper explicitly uses decoder output tokens as internal representations; the real issue is underspecification of exactly which token representation is extracted and compared.
- **Complaints about lack of hardware diversity / missing benchmarks on multiple GPU types / full reproducibility checklist.**  
  Removed as core weaknesses. These are useful suggestions but not central flaws for ICLR in this empirical setting.
- **Criticism based on doubting release status / verifiability of cited baselines or tools.**  
  Removed by rule.

## Novel Insights
The most interesting synthesis across the reviews is that the paper’s real contribution is narrower but still meaningful: it demonstrates that **SAM’s decoder-side representations are more useful for counting verification than encoder-side similarity alone**, and this seems to matter more than the specific TS-SS metric. In other words, the paper’s strongest idea is not “TS-SS” per se, but the shift from image-embedding matching to prompt-conditioned decoder-token verification inside a pure-SAM pipeline. The current manuscript somewhat obscures this by overemphasizing broad SOTA claims and by not cleanly separating prompt-budget effects from prompt-generation quality.

## Suggestions
- Rewrite the abstract/introduction claims to accurately reflect the actual evidence: strongest among reported **SAM-only training-free** methods, competitive overall, and especially strong on CARPK.
- Add a controlled ablation with fixed prompt budgets comparing probabilistic prompting against grid/random/superpixel baselines.
- Expand the efficiency section with direct wall-clock and memory comparisons against the main training-free baselines under the same hardware/software setup.
- Clarify exactly which SAM decoder token(s) are extracted, how token-instance association is performed, and how repeated prompting is batched/implemented.
- Add sensitivity analyses for the verification threshold and temperature parameter.
- Quantify failure cases by object size and density, since this appears to be the main practical limitation.
- Reframe TS-SS more modestly unless stronger evidence is added; alternatively, provide targeted qualitative/quantitative cases showing where it clearly beats cosine or L2.

---

## 0lW2UBiEWN

- GT: Reject (avg 4.5)
- Predicted: N/A (4.7/10)
- Match: N/A

### Final Review

## Summary
This paper introduces **MESA & MASK**, a benchmark for detecting pressure-induced behavioral shifts in LLMs by comparing outputs under a neutral baseline (MESA) and a pressure-conditioned context (MASK). The benchmark covers 2,100 instances across six domains and six deception types, and the authors evaluate 22 models to study how often models change reasoning and responses under pressure.

## Strengths
- **The paired MESA/MASK design is a concrete and useful benchmarking idea.** Rather than scoring isolated answers, the paper evaluates *behavioral deviation under controlled context change*, which is a more targeted way to probe brittle alignment than standard single-prompt factuality or safety tests.
- **The dataset construction pipeline is unusually structured for this topic.** The paper does more than generate prompts: it defines six deception categories × six professional domains, uses iterative filtering, and includes expert review with reported agreement of **94.3% / κ=0.89** for dataset quality checks. The benchmark is also balanced by design (350 instances per deception type).
- **The paper makes a genuine effort at operational transparency.** It specifies the evaluation protocol, reports the judge-model selection procedure, exposes the thresholding scheme in Appendix C.2, and includes prompts and examples in the appendix/repository.
- **Some empirical observations are potentially interesting even if not yet fully supported causally.** In particular, the contrast between model families, and the distinction between D@1, D@k, and “stability,” could become useful descriptors of pressure sensitivity if the construct is validated more carefully.

## Weaknesses

### Fatal
- **The paper’s central construct claim is stronger than what the methodology currently validates.**  
  The paper defines deception in the introduction as the **“intentional inducement of false beliefs”**, but the implemented benchmark primarily detects **inconsistency between a neutral baseline and a pressure condition**, judged from reasoning/output differences. This is not enough, on its own, to establish deception in the paper’s own sense.  
  The issue is not that the paper ignores confounders entirely—it explicitly tries to distinguish deception from hallucination and instruction following in Section 2.2, and says the pressure prompts avoid “explicit deceptive directives.” However, the actual operational test still conflates several phenomena:
  - strategic deception,
  - pressure-sensitive compliance/persona adaptation,
  - omission of caveats,
  - assertive reframing under social or institutional pressure.  
  The appendix example makes this concern concrete: the pressure prompt describes a senior manager “known for his aggressive investment philosophy and a very low tolerance for analyses that he perceives as timid,” and the benchmark counts the model’s resulting shift as deception. In some cases that may indeed be deceptive; in others it could simply be pressure-conditioned rhetorical adaptation. The paper does not provide a sufficiently strong control showing that the benchmark isolates **intentional false-belief induction** rather than broader **pressure sensitivity**. Since this distinction is core to the claimed contribution, this substantially weakens the headline claim of “differential diagnosis of LLM deception.”

### Major:
- **The promised four-quadrant “differential diagnosis” framework is not actually reported in the main empirical results.**  
  Figure 2 and the abstract/introduction frame the contribution as a diagnostic classifier distinguishing behaviors such as “genuine deception,” “deceptive tendencies,” and “brittle superficial alignment.” But Section 5 reports only aggregate **Deception Rate @1**, **Deception Rate @k**, and **Stability**. There is no quantitative breakdown of how instances or models populate the four quadrants, no quadrant-wise analysis by deception type, and no empirical demonstration that the framework truly separates the advertised behavioral categories. As written, the paper delivers a paired-evaluation benchmark and aggregate inconsistency rates, but not the full differential diagnosis promised.
- **The evaluation depends heavily on a single LLM judge, and the validation is narrower than the paper’s claims.**  
  The paper does provide some judge validation: Appendix C.1 reports GPT-4.1 outperforming two alternatives on agreement with expert annotations, and Appendix C.2 describes threshold tuning against 300 annotated response pairs. That is a meaningful effort and should be credited.  
  Still, the main results depend on a single proprietary judge plus heuristic thresholds (**5/7** reasoning indicators, **6/8** output indicators), and the validation target is mostly *consistency judgment*, not the stronger construct of deception as intentional false-belief induction. The paper does not include a detailed error analysis of where the judge mistakes benign adaptation for deception, nor a threshold sensitivity analysis showing model rankings are stable under small changes. Given how central the judge is to all quantitative claims, this remains a serious limitation.
- **Key empirical interpretations overreach the evidence and are largely correlational.**  
  The paper draws conclusions about model scale, distillation, MoE vs. dense architectures, and post-training effects. But the paper itself acknowledges that direct architecture comparisons are confounded: “direct MoE-dense comparisons face inherent parameter mismatching limitations.” Likewise, the safety fine-tuning section is explicitly a “limited case study involving two models from the same family and a single training run,” yet the conclusion generalizes toward what “standard safety fine-tuning” can or cannot do. The observed patterns are interesting hypotheses, but the paper often presents them more strongly than the design warrants.

### Minor
- **The benchmark’s claimed disentanglement from hallucination is asserted more than demonstrated.**  
  Section 2.2 motivates the distinction well, but the paper does not include a dedicated hallucination-control evaluation showing that models failing from capability gaps under both MESA and MASK are not spuriously flagged as deceptive.
- **Pressure-prompt robustness is underexplored.**  
  Because the benchmark hinges on latent-pressure system prompts, it would help to know whether results are stable to paraphrases, pressure intensity, or alternate prompt formulations. Without this, it is hard to tell whether the measured rates reflect a robust behavioral tendency or sensitivity to a particular prompt template.
- **The automated data-quality scoring loop is somewhat opaque.**  
  The paper gives rubric dimensions and thresholds, but does not fully validate how these automated quality scores align with human judgments beyond the final expert filtering stage. Since the generation loop may shape scenario difficulty and “deception necessity,” more transparency here would strengthen confidence in the dataset.

### Trivial
- **The definition of the Stability metric is unclear in the main text.**  
  The formula in Section 5.1 is garbled (“S = D@1 [D@k] ...”), making the metric mathematically unclear without inference from context.

## Nice-to-Haves
- Add a **hallucination/control baseline** where both MESA and MASK are expected to fail due to knowledge gaps, to better support the claim that the benchmark isolates deception rather than generic error.
- Report a **four-quadrant distribution analysis** by model family and deception type, since that is central to the paper’s framing.
- Include **threshold sensitivity** and **cross-judge robustness** analyses for the GPT-4.1 evaluation pipeline.
- Add a **pressure-intensity ablation** or prompt-paraphrase robustness study.
- Separate more clearly in the claims what is established as a **benchmarking signal for pressure-induced behavioral deviation** versus what is established as **deception** in the stronger intentional sense.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Criticism about unreleased/non-verifiable models, tools, or references.** Removed per instruction; cited entities are assumed to exist.
- **Complaints about missing frontier models or temporal validity.** The model roster is already broad, and this is not a substantive flaw unless tied to a specific unsupported claim.
- **Pure reproducibility nitpicks about omitted hyperparameters or implementation details.** The paper already provides substantial methodological detail, prompts, thresholds, and code links.
- **Formatting/parser artifacts as weaknesses.** The garbled figure/table extraction in the provided text is due to PDF parsing and should not be treated as a paper flaw.
- **Claim that the paper provides “zero” empirical implementation of the framework.** This is too strong and inaccurate. The paper does implement the paired evaluation and binary consistency classification; the valid criticism is narrower: it does **not report the promised four-quadrant diagnostic analysis** in results.
- **Claim that pressure prompts are “explicit instructions to deceive.”** This overstates the case. The paper is correct that the prompts do not explicitly say “deceive” or “lie.” The real issue is that they still may induce persona/compliance shifts that are hard to distinguish from deception.
- **Dataset contamination overlap with pretraining corpora of closed-source models.** This is not realistically verifiable from the paper and goes beyond what can be fairly demanded here.

## Novel Insights
The most important synthesis across the reviews is that this paper is strongest when read as a benchmark for **pressure-induced alignment brittleness** and weakest when read as a benchmark that has already solved **deception diagnosis** in the full intentional sense. The MESA/MASK paired design is genuinely promising because it operationalizes *behavioral deviation under controlled contextual stress*, which is a richer signal than one-shot truthfulness tests. But the current evidence does not yet close the gap between “the model changed under pressure in a potentially strategic way” and “the model intentionally induced false beliefs.” That gap is exactly where the benchmark could become influential if the authors validate it more directly.

## Suggestions
- **Reframe the central claim more carefully.** If the authors present MESA & MASK as a benchmark for **pressure-induced deceptive tendencies / alignment brittleness**, the paper becomes substantially more defensible.
- **Report the actual four-quadrant outcomes.** This is the single most important missing analysis relative to the paper’s stated contribution.
- **Add a human-validated audit focused on construct validity**, not just consistency: sample judged positives/negatives and ask annotators whether the outputs truly instantiate deception rather than style shift or compliance.
- **Run a hallucination-control experiment** to show the framework does not overflag capability failures.
- **Run threshold and judge robustness ablations** so the quantitative rankings are not overly dependent on one judge configuration.
- **Tone down causal/architectural claims** about MoE, distillation, and safety fine-tuning unless supported by controlled comparisons.

---

## 2hTLJEgCbv

- GT: Reject (avg 1.0)
- Predicted: N/A (2.9/10)
- Match: N/A

### Final Review

## Summary
This paper presents an empirical study of how encoder/decoder architecture choices in a standard VAE affect optimization behavior, latent collapse, and reconstruction on MNIST. The main reported takeaway is that shallow dense encoders tend to work better than more complex encoders in this setup, while decoding benefits more from convolutional structure, especially with deeper convolutional decoders.

## Strengths
- The paper isolates a narrow question—encoder/decoder architectural asymmetry within an otherwise standard VAE—and explores it via a reasonably systematic grid over encoder type, decoder type, depth, and latent size. That focused ablation structure is more informative than a single preferred architecture.
- The paper explicitly separates reconstruction and KL-related behavior in the analysis (Figures 1–3), which is appropriate for VAEs and helps expose that some configurations reconstruct reasonably while still exhibiting latent collapse or near-collapse.
- One concrete empirical pattern does emerge from the sweep: among the better-performing configurations in this MNIST setting, shallow dense encoders recur more often, while convolutional decoders with multiple blocks appear advantageous on the decoding side. Even if limited in scope, this encoder/decoder asymmetry is the paper’s most useful practical observation.
- The paper deliberately studies simple building blocks rather than mixing in more advanced priors/objectives, which makes the architectural comparisons easier to interpret than studies where architecture changes are entangled with loss redesign.

## Weaknesses

###: Fatal

### Major:
- **The empirical scope is too limited to support the paper’s broad architectural conclusions.** All experiments are on MNIST (“All experiments are be conducted on the MNIST dataset”), which is too simple to justify general guidance such as “small dense networks are more effective for encoding” and “decoding benefits from architectures with structural processing capabilities.” On a dataset with such strong low-level regularity and low semantic complexity, observed trends may not transfer to more realistic image distributions.
- **The evaluation does not adequately support claims about generative quality, representation quality, or compression.** The paper mainly uses ELBO components, qualitative reconstruction comments, and 2D PCA plots of latent codes. But the abstract and conclusion make claims about “generative quality,” “representation quality,” and “compressive capacities.” The current pipeline does not directly measure those notions. In particular, PCA visualizations on MNIST are weak evidence for latent quality, and there are no quantitative sample-quality metrics or more direct representation diagnostics.
- **Architectural claims are confounded by uncontrolled model capacity and training setup.** Section 3 describes CNN and dense variants only at a high level and does not report parameter counts, matched-capacity comparisons, or other controls that would let one attribute differences to architecture type rather than raw capacity/depth/optimization effects. Since the conclusions hinge on whether dense vs. convolutional encoders/decoders are intrinsically preferable, this missing control matters substantially.
- **The “top 25% / top 50%” analysis introduces an unclear and potentially biased selection procedure.** The paper repeatedly analyzes “top 25%” and “top 50%” models, but it does not clearly define the ranking criterion in the text. Because the main architectural conclusions are partly drawn from these filtered subsets, the lack of a precise and justified selection rule weakens the evidential chain.
- **Several headline findings are weaker than the paper presents them.** For example, the paper emphasizes that “models with non-zero Kullback-Leibler Divergence (KLD) loss outperform collapsed latent space models.” That is directionally true in practice, but it is not a strong scientific insight by itself; avoiding posterior collapse is largely a prerequisite for the latent variables to carry information at all. As stated, this reads more as confirmation of expected VAE behavior than as a novel contribution.

### Minor
- **Posterior collapse is discussed somewhat informally and without stronger diagnostics.** Section 4.1 equates many runs with “collapsed latent spaces” and describes this as latent distributions becoming identical to a multivariate normal distribution, but the paper does not define a threshold or provide supporting diagnostics such as active units, mutual information proxies, or per-dimension KL statistics. This makes the collapse analysis less rigorous than it could be.
- **The paper’s notion of “compression levels” is underspecified.** The text says latent spaces of varying compression are studied, and Figure 4 refers to “compression size,” but the paper does not clearly formalize compression in relation to input dimensionality or provide a principled rate–distortion style analysis. As a result, the compression conclusions remain qualitative.
- **The significance and novelty are modest for an ICLR paper.** The observed trends largely align with known inductive biases: convolution helps image decoding via spatial structure, and smaller encoders may reduce overfitting or mismatch in simple settings. The paper is a useful exploratory sweep, but it stops short of a deeper mechanistic explanation or broader validation.

### Trivial

## Nice-to-Haves
- Add experiments on at least one more challenging image dataset to test whether the encoder/decoder asymmetry persists beyond MNIST.
- Report matched-parameter or matched-FLOP comparisons across architecture families.
- Include direct generative evaluations (sample grids plus quantitative metrics where appropriate) and more formal latent-space diagnostics.
- Run multiple seeds and report variance, especially if the architectural performance gaps are not large.
- Clarify exactly how models are ranked when defining the “top 25%” and “top 50%” subsets.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper misrepresents the literature by claiming architecture is underexplored.”** This is too dependent on external literature adjudication beyond what can be verified from the submission alone. The paper does cite prior architecture-focused work (e.g., NVAE, DGSN), so while the novelty framing may be somewhat overstated, this criticism was too strong as written.
- **Missing comparison to specific VAE variants such as NVAE or β-VAE as a core flaw.** The paper is framed as an internal architectural ablation within a standard VAE rather than a state-of-the-art method paper. Competitive baselines would help contextualization, but their absence is not by itself a decisive flaw for the stated scope.
- **Complaints purely about omitted optimizer / learning-rate / batch-size details.** The paper is indeed sparse on implementation details, but this falls under reproducibility nitpicks unless tied to a substantive claim. The more important issue is not the missing hyperparameters per se, but the lack of controls over KL weighting/capacity when drawing architectural conclusions.
- **Criticism that the paper should include transformers or modern architectures.** This is scope creep. The paper studies simple dense vs. convolutional VAEs; asking for transformer baselines is not necessary to evaluate whether it answered its chosen question.

## Novel Insights
The most interesting synthesis across the reviews and the paper is that the useful contribution here is not the broad claim that the paper has “solved” VAE architecture design, but the narrower empirical asymmetry it exposes: in a plain-VAE regime, encoder simplicity may be beneficial while decoder structure matters more. That is a potentially practical design heuristic. However, the current study does not yet distinguish whether this asymmetry is due to information bottleneck effects, optimization stability, parameter-count mismatch, or dataset simplicity. In other words, the paper’s strongest spark is a plausible architectural asymmetry, but the present evidence does not yet pin down its cause or generality.

## Suggestions
- Reframe the contribution more modestly as an MNIST-based exploratory study of architectural asymmetry in plain VAEs, rather than a general statement about VAE design.
- Add at least one nontrivial dataset and verify whether the “simple encoder / structured decoder” pattern still holds.
- Control for capacity explicitly: report parameter counts and include matched-capacity comparisons between MLP and CNN variants.
- Replace or supplement PCA-based latent analysis with more direct latent-usage diagnostics and clearer collapse criteria.
- Clarify the model-selection/ranking procedure for the “top 25%” and “top 50%” analyses, or avoid filtered-subset conclusions if the ranking criterion is not principled.
- Tone down the claim around non-zero KL being beneficial, or instead analyze it more rigorously via a controlled sweep over KL weighting / rate-distortion tradeoffs.
- Include representative unconditional samples, reconstructions, and latent traversals/interpolations to support the claims about generative and representational quality.
- Strengthen the discussion of novelty and significance: the current work is best positioned as an empirical heuristic study, not as a fundamentally new understanding of VAE behavior.

---

## qyVzZsrsnS

- GT: Accept (Poster) (avg 7.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Final Review

## Summary
This paper studies how narrow finetuning changes internal activations in LLMs and argues that these changes leave unusually legible traces even on unrelated inputs. The proposed Activation Difference Lens (ADL)—combining average activation differences, Patchscope/Logit Lens, and activation steering—surfaces tokens and steered generations that reveal the finetuning domain, and an LLM-based auditing agent uses these signals to identify finetuning objectives far better than black-box querying alone. The paper further argues that this phenomenon is strongest for semantically narrow “model organism” finetunes, is much weaker for broader post-training, and can be mitigated by mixing unrelated data into finetuning.

## Strengths
- **The paper isolates a surprisingly strong empirical phenomenon across a diverse set of organism families and model families.** It evaluates 33 organisms spanning SDF, emergent misalignment, subliminal learning, and taboo-word models across Gemma, Llama, and Qwen from 1B to 32B parameters. This breadth matters because the main claim is not about one bespoke setup, but about a recurrent artifact of narrow finetuning.
- **ADL is simple but unusually revealing.** The core procedure uses only mean activation differences on the first few tokens of unrelated text plus standard interpretability tools (Patchscope, Logit Lens, steering), yet it surfaces concrete domain semantics such as “stocks,” “culinary,” or “cat”-related outputs in Figure 1. The simplicity is a strength here: the paper shows that one does not need elaborate mechanistic tooling to detect these traces.
- **The interpretability-agent evaluation is a specific and meaningful validation of informativeness rather than only qualitative cherry-picking.** The agent with ADL access substantially outperforms blackbox agents even when the latter get many more interactions. The paper reports that ADL with \(i=5\) identifies the broad objective for 91% of organisms, while the best blackbox baseline at \(i=50\) reaches 39%. That gap supports the claim that the surfaced traces contain operationally useful information.
- **The paper does useful scoping work rather than overclaiming universality.** Appendix E explicitly checks broader finetuning regimes and finds that standard chat finetuning leaves minimal or no detectable traces, while semi-narrow domain adaptation produces weaker effects. This is important because it turns the contribution into a more precise warning: narrow model-organism finetunes may be atypically easy to audit with diffing-style methods.
- **The mitigation experiments are practically useful even if mechanistically incomplete.** Mixing pretraining data into the narrow finetuning corpus substantially reduces token relevance and steering similarity, and the paper honestly shows the trade-off with reduced false-fact internalization. This gives concrete guidance for researchers building model organisms.
- **The paper does more than present a positive result; it triangulates the phenomenon from several angles.** Position ablations, layer ablations, full-vs-LoRA finetuning, reduced-sample experiments, mixed-data experiments, and grader-ablation analyses collectively make the empirical case more robust than a single headline figure would.

## Weaknesses
### Fatal
None.

### Major:
- **The mechanistic claim that the traces are specifically a form of “overfitting” is not established as strongly as the empirical detection claim.**  
  Section 5 shows that ablating the bias direction harms performance on finetuning data and sometimes helps on pretraining data, which supports that the direction is functionally important and potentially harmful off-distribution. But this does not by itself isolate *overfitting to semantic homogeneity* as the causal mechanism, as opposed to a more generic learned task direction or broader distributional specialization. The paper itself uses appropriately softer language in places (“We suspect that these biases are a form of overfitting”; “likely connect to ideas from catastrophic forgetting”), but some summary statements are stronger than the evidence fully warrants.
- **The causal analysis has an unresolved inconsistency on Gemma3 that weakens the universality of the proposed explanation.**  
  The paper explicitly states: “For Gemma3 1B, the causal effect on \(D_{pt}\) is slightly positive but comparable to baseline effects,” and attributes this to larger representational divergence between base and finetuned models. This is a reasonable possible explanation, but it remains a post hoc explanation rather than a fully validated one. Since the overfitting narrative leans on the sign difference between \(D_{ft}\) and \(D_{pt}\), the mixed result on Gemma should temper the strength of the mechanistic conclusions.
- **The evaluation pipeline depends heavily on LLM graders and rubric-conditioned judgments, which leaves some uncertainty about the exact magnitude of the reported gains.**  
  Token relevance, coherence thresholds, and final hypothesis grades are all LLM-scored. The paper does take this concern seriously and includes grader ablations in Appendix D, but the agreement is only moderate for token relevance (\(\alpha=0.65\)). This does not invalidate the result—the effect sizes are large enough that the conclusion likely survives grader noise—but it does make fine-grained quantitative claims, especially dramatic ratios like “30 times better” on grade \(\ge 4\), less definitive than they first appear.
- **The paper’s practical significance is strongest for narrow model-organism setups and substantially weaker for broader real-world post-training.**  
  This is not a flaw in honesty—the appendix clearly documents it—but it is a real limitation on significance for a general ICLR audience. The most consequential warning is about the realism of current model-organism studies, not about mainstream instruction tuning in deployed systems. That is still valuable, but narrower than the abstract and introduction initially suggest.

### Minor
- **The “discovery” aspect of the main phenomenon is somewhat less conceptually surprising than the presentation suggests.**  
  The method computes the average activation difference between base and finetuned models and then explicitly interprets or steers with that mean shift. It is therefore not shocking that this direction carries semantics related to the finetuning objective. What is genuinely interesting is the *strength and readability* of the effect on unrelated early tokens across many organisms, not the mere existence of some directional information. The paper would benefit from framing the contribution more as a quantitative empirical characterization than as an unexpected conceptual finding.
- **The boundary between “narrow enough to leave readable traces” and “broad enough that traces disappear” is not characterized systematically.**  
  The paper does show that chat finetuning mostly lacks the effect and that adding pretraining data attenuates it, but it does not provide a more quantitative account of semantic narrowness, dataset diversity, or training conditions that govern the transition.
- **The mitigation story is useful but incomplete.**  
  Mixing in unrelated data reduces detectability, but also weakens false-fact alignment, and at 1:1 the agents fail to recover average grade \(\ge 2\). The paper therefore demonstrates a trade-off more than a clean solution. It would help to quantify more directly whether useful target behavior is preserved for non-SDF settings beyond the FFA proxy.
- **The paper does not deeply analyze the representation of the bias direction itself.**  
  The claim that the bias comes from “constant semantic concepts shared across all finetuning samples” would be stronger with decomposition of \(\delta\) into subfeatures or subspaces rather than only using Patchscope, steering, and a 1D causal projection.
- **Some central scoping results are relegated to the appendix despite being crucial to interpretation.**  
  In particular, the fact that chat finetuning leaves “minimal or no detectable traces” is essential to understanding the paper’s scope and should be emphasized earlier in the main text, not mainly in Appendix E.

### Trivial
- **The steering setup is somewhat elaborate and computationally involved.**  
  Searching for steering strengths with coherence graders and aggregating over prompts is sensible, but it adds operational complexity that may make the method less lightweight in practice than the headline concept suggests.

## Nice-to-Haves
- Quantify “narrowness” more directly, e.g., via corpus diversity or semantic self-similarity, and show how ADL effectiveness varies along that axis.
- Add experiments on real-world narrow SFTs (e.g., legal, medical, code-domain finetunes) to bridge the gap between synthetic organisms and standard post-training.
- Decompose the activation-difference vectors with SAEs, PCA, or crosscoders to test whether the effect is driven by a small number of dominant features versus distributed drift.
- Provide one or two full case studies contrasting a successful ADL agent and a failed blackbox agent, showing exactly which surfaced tokens/steered outputs enabled the inference.
- Clarify more prominently in the main text that the strongest claims concern narrow finetuning/model organisms, not general chat-tuning.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The trace discovery is conceptually guaranteed by the mean-difference methodology.”**  
  Overstated. The method indeed computes a mean activation-difference direction, so it is unsurprising that it contains finetuning information. However, the paper’s substantive empirical claim is that this information is *highly readable on unrelated early-token activations across many narrow finetunes* and useful enough to power an auditing agent. That is not vacuous.
- **Concern about data leakage because the 10,000 pretraining samples may overlap with evaluation samples.**  
  The paper states it computes average activation differences on “a pretraining corpus containing 10,000 samples” and uses those differences for interpretation/steering; it is not making a train/test generalization claim over that corpus in the conventional sense. The criticism of “circularity” is too strong from the text provided.
- **Claim that the ADL-vs-blackbox comparison is unfair because ADL gets stronger signals.**  
  This is not a valid weakness here. The point of the paper is precisely to test whether ADL-derived access to internal differences is more informative than black-box interaction. The asymmetry is intentional and appropriate to the question being asked.
- **Requests to release activation tensors/checkpoints/raw intermediates for independent verification.**  
  Removed under the reproducibility rule; the paper already provides code and an appendix on reproducibility, and large artifact release is not required here.
- **Complaints about missing comparisons to unspecified external methods.**  
  Removed because external baselines cannot be verified here and the review instructions explicitly forbid mentioning missing related works.
- **Formatting/parser issues.**  
  Ignored as instructed.

## Novel Insights
The paper’s most important insight is less “activation differences contain finetuning information” than a sharper methodological warning: current narrow model-organism finetunes may be *pathologically legible* to simple diffing analyses in a way that broader post-training is not. This reframes positive results on such organisms: a technique succeeding there may partly be exploiting a narrow-data artifact rather than uncovering mechanisms that matter in realistic alignment or chat-tuning. The appendix evidence that chat finetuning largely lacks the same signal is therefore not peripheral—it is what makes the main result consequential.

## Suggestions
- Reframe the central claim more carefully: emphasize that the key empirical result is the *magnitude and readability* of the traces across narrow finetunes, not merely that the mean difference contains some task information.
- Soften or better qualify the mechanistic “overfitting” claim unless stronger causal isolation is added.
- Strengthen Section 5 with additional controls that distinguish “generic useful learned direction” from “artifact of semantic homogeneity,” and directly address the Gemma inconsistency.
- Move the broader-finetuning results from Appendix E into the main paper or foreground them much earlier, since they are essential for scoping significance.
- Add at least one real-world narrow finetune beyond synthetic organisms to test whether the phenomenon transfers outside model-organism settings.
- If space permits, include a more direct non-LLM or partially human sanity check for one evaluation component to complement the grader-heavy pipeline.



---

## RDAhLHEHDm

- GT: Accept (Poster) (avg 6.5)
- Predicted: N/A (5.6/10)
- Match: N/A

### Final Review

## Summary
This paper argues that current Sci-LLMs are often better used as reasoners over high-level, tool-derived biological context than as direct interpreters of raw biomolecular sequences. The authors compare sequence-only, context-only, and sequence+context inputs across several protein QA settings and additional analyses, finding that context-only often performs best, while also presenting representational analyses, efficiency comparisons, and a small wet-lab evaluation.

## Strengths
- **The paper isolates a practically important and underexplored question: when does explicit biological context beat raw sequence input for LLM-based reasoning?** The comparison across sequence-only, context-only, and combined inputs is central and clearly operationalized in Table 1, rather than being discussed only conceptually.
- **The context construction pipeline is more carefully designed than a naive tool dump.** Appendix A and Table 3 show a specific hierarchical integration strategy: GO annotations are strongest individually, Pfam adds complementary value, and ProTrek is only useful as a fallback because unconditional inclusion hurts performance. That is a concrete systems insight, not a generic “use tools” claim.
- **The paper includes useful boundary-setting rather than only success claims.** In Appendix J, the authors explicitly acknowledge that their approach is weak for small mutation effects because InterProScan/BLAST-derived context can remain unchanged under point mutations. This is an important and credible limitation because it directly delineates where the proposed paradigm should not be treated as a replacement for sequence-sensitive modeling.
- **The efficiency analysis is practically relevant.** Table 2 and Appendix M make the paper more valuable as a deployment-oriented contribution by comparing API-based context reasoning to specialized Sci-LLMs in cost and throughput terms, supporting the claim that tool-augmented pipelines can be attractive in realistic settings.

## Weaknesses

### Fatal
- **The paper’s strongest conceptual claim substantially outpaces what the experiments actually establish.** The core empirical setup compares (a) zero-/prompt-based answering from raw sequences against (b) answering from tool-generated biological summaries built using InterProScan, BLASTp-to-Swiss-Prot, and sometimes ProTrek. For the main benchmark tasks—function, pathway, and subcellular localization—the context is not merely an alternative representation of the same raw evidence; it is a high-level inferred annotation pipeline that already injects domain knowledge highly proximal to the target labels. Appendix A.1 explicitly states: “We extract textual descriptions of detected domains together with any directly linked Gene Ontology (GO) annotations” and “We transfer GO annotations from the most similar sequences to the query.” As a result, the main comparison supports the practical claim that **tool-derived biological context is more useful for LLM QA than raw sequence alone on these annotation-heavy tasks**, but it does **not** justify the broader conclusion that the “tokenization dilemma” is the primary bottleneck of current sequence-centric Sci-LLMs. This is not a minor framing issue; it affects the validity of the paper’s headline thesis.

### Major:
- **The evaluation tasks are strongly biased toward settings where the external context pipeline is expected to dominate, so the paper does not support broad negative claims about raw sequence utility.** The benchmark in Section 5.1 uses questions whose answers come from annotation fields already closely aligned with the sources used to construct context. The paper does include an EC benchmark and a DNA mutation benchmark in the appendix, but the main narrative repeatedly generalizes to claims such as “raw sequences act as informational noise” and that current paradigms are “fundamentally handicapped.” Given the chosen tasks, the evidence is much narrower: on annotation-centric QA with rich external tool support, context is superior. The paper itself partly acknowledges limited scope (“our current analysis has primarily focused on proteins”; Section 6) and, more importantly, admits failure on mutation-sensitive cases (Appendix J). These admissions make the broad anti-sequence framing feel overstated.
- **The “sequence as noise” interpretation is insufficiently controlled.** Table 1 does show that for many models, sequence+context is slightly worse than context-only, but this pattern is neither universal nor mechanistically pinned down. In fact, one of the strongest models contradicts the stated universality: Deepseek-v3 has **86.03** for sequence+context versus **84.99** for context-only. So the manuscript’s language that adding sequence “consistently” degrades performance is factually too strong. More importantly, the current experiments do not control for prompt length, sequence placement, or generic long-context distraction effects. Without equal-length controls or other prompt-structure controls, the evidence does not distinguish biological semantic interference from ordinary attention dilution caused by appending a long low-compression token string.
- **The primary QA metric relies on an LLM judge, which is a weak basis for the paper’s strongest claims.** Appendix C.1 makes clear that the main benchmark is evaluated by a DeepSeek-V3 adjudicator that scores generated answers from 0–100. That can be acceptable as auxiliary evidence, but here it is the main metric supporting strong conceptual conclusions. Since context-only answers are produced from context text assembled from the same biological tools/databases that define the task, an LLM judge may preferentially reward semantic overlap and faithful paraphrase of that supplied context rather than independent biological inference. The paper usefully provides more objective evaluation for EC prediction, but the main QA benchmark would be stronger with a more objective metric or at least stronger validation of the judge.

### Minor
- **Some claims are phrased more strongly than the presented evidence supports.** Examples include “consistently” degrading performance and the suggestion that sequence-centric paradigms are broadly “fundamentally handicapped.” The actual evidence is strongest for a narrower statement: current general and scientific LLMs are much better at reasoning over structured biological annotations than over raw sequence prompts for the tested tasks.
- **The wet-lab validation is interesting but limited in evidential weight.** Section 5.6 evaluates binary classification on Rhodopsin and PETase with modest sample counts. This is useful as a proof of practical relevance, but it is too small and task-specific to bear much of the paper’s generalization argument.
- **The temporal generalization analysis is suggestive but hard to interpret causally.** Section 5.4 discusses degradation over discovery year and argues for superior temporal robustness of the context-driven approach. However, because both annotation availability and homology support can vary over time, it is difficult to cleanly separate model limitations from changing support in the underlying biological resources. This weakens the force of the causal interpretation, though not the descriptive trend.

### Trivial

## Nice-to-Haves
- Add stronger sequence-only baselines beyond direct prompting, e.g., instruction-tuned or task-adapted sequence models, to avoid conflating poor prompting with paradigm limits.
- Add prompt-length-controlled baselines in the sequence+context condition, such as replacing the sequence with unrelated text of comparable token count, to test whether the degradation is biological interference or generic context dilution.
- Quantify overlap between generated context and benchmark answers to better characterize how much of the task is retrieval/annotation transfer versus downstream synthesis.
- Report statistical uncertainty for the context-only vs. sequence+context differences; several observed gaps are small enough that significance matters.
- Separate offline context generation cost from online inference latency more explicitly in the efficiency section.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claims that the paper has “information leakage” merely because it uses homolog annotations rather than the query’s own label record.** The paper explicitly addresses this in Section 4 and Appendix A.1: “we never use the query’s own (possibly unknown) labels.” This does not eliminate the broader concern that the task is retrieval-biased, which I keep, but calling it direct leakage would overstate the issue.
- **Criticism that the paper is invalid because the method is “just a bioinformatics pipeline, not a new representation learning paradigm.”** This is too absolute. The paper’s empirical contribution is indeed a tool-augmented context pipeline plus a reframing argument. The real issue is not that such a pipeline is illegitimate, but that the conceptual claims about tokenization and sequence paradigms are broader than the evidence warrants.
- **Criticism centered on unreleased/unverifiable tools, models, or datasets.** Per policy, these are removed.
- **Pure formatting/parser-related complaints.** The extracted text has obvious PDF artifacts, but those are not paper weaknesses.

## Novel Insights
The key synthesis here is that the paper is strongest not as a fundamental refutation of sequence-centric Sci-LLMs, but as evidence for a more modest and still meaningful principle: for annotation-heavy biological QA, the bottleneck is often not the LLM’s reasoning capacity but the representation presented to it, and mature bioinformatics tools can serve as highly effective lossy compressors from sequence space into language-aligned evidence. In that sense, the paper exposes a real mismatch between what current LLMs are good at (knowledge synthesis over textual evidence) and what the community sometimes asks them to do (infer biology directly from raw sequences). The weakness is that the manuscript then over-extends this operational insight into a universal theory about tokenization, when the experiments mainly demonstrate the power of external annotation transfer.

## Suggestions
- Reframe the main claim more narrowly and accurately: the results support that **tool-derived biological context is superior to raw sequence prompting for annotation-centric QA**, not that raw sequence modeling is broadly inferior or that tokenization is the dominant failure mode of Sci-LLMs.
- Add controlled sequence+context experiments that hold prompt length constant and vary sequence position to test whether the observed degradation is genuine semantic interference rather than generic long-context effects.
- Strengthen the evaluation by supplementing LLM-Score with more objective metrics on the main benchmark, or at minimum provide a substantial human/biologist validation of judge reliability.
- Explicitly quantify how close the context is to the target answers (e.g., GO/pathway/location overlap statistics), so readers can judge how much of the task is external annotation transfer versus LLM synthesis.
- Discuss the Deepseek-v3 exception in Table 1 directly and remove universal wording like “consistently” unless it is actually true across models.
- Position the paper as a strong systems/practical contribution on hybrid scientific AI pipelines; that framing is well supported by the ablations, efficiency analysis, and clearly stated limitations.

---

## XPIEkFdEDi

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (6.7/10)
- Match: N/A

### Final Review

## Summary
This paper proposes AnyBCQ, a multi-precision post-training quantization framework for LLMs built on binary-coded quantization (BCQ). The core idea is to share binary bit-planes across precisions while learning precision-specific scales, paired with a CUDA kernel that operates directly on bit-planes and avoids centroid lookup / bit-transpose overheads common in prior multi-precision non-uniform methods. Empirically, the method is strongest at 2-bit quantization, where it substantially improves over prior multi-precision baselines, while remaining competitive at 3–4 bits and offering favorable memory and throughput trade-offs.

## Strengths
- **A specific and meaningful low-bit result:** At 2 bits, the method delivers a large accuracy improvement over the main multi-precision baseline. On Llama-3.1-8B, Table 2 shows MMLU improving from **24.66** (Any-Precision LLM) to **35.32** and CSR average from **39.65** to **58.71**, which is a substantial gain in the regime where prior multi-precision methods struggle most.
- **Algorithm–kernel alignment is unusually tight:** The paper does more than propose a quantization format; it exploits BCQ’s binary bit-plane structure to design a kernel that directly computes on active bit-planes, avoiding the centroid lookup and bit-transposition path used by the compared non-uniform multi-precision approach. This is well motivated in Sec. 3.3 and supported by the latency breakdown in Appendix A.2, where the baseline kernel spends a large fraction of time in bit-transpose and lookup.
- **The memory-sharing mechanism is concrete and practically relevant:** Table 1 quantifies the cost of supporting 2/3/4-bit operation in one model: compared with storing separate models, the proposed shared-binary representation reduces total footprint from **9.85 GB** to **4.99 GB** on Llama-3.1-8B. This is a specific deployment advantage of the paper’s design rather than a generic compression claim.
- **The paper is appropriately transparent about the central trade-off:** The authors explicitly acknowledge that sharing binaries across precisions can hurt peak higher-bit accuracy (“the shared-binary constraint slightly limits the capacity of the multi-precision model,” Appendix A.1; also Sec. 7). This transparency increases confidence that the reported gains at 2 bits are not being oversold as universally dominant.
- **Evaluation goes beyond isolated kernel timing:** In addition to benchmark accuracy, the paper reports end-to-end decoding throughput on multiple models (Llama-3.1-8B, Gemma-2-9B, Phi-4-14B), plus a mixed-precision decoding case study. This helps support the practical systems motivation.

## Weaknesses

###: Fatal
None.

### Major:
- **The shared-binary design imposes a real accuracy ceiling at higher precisions, and this limits the paper’s “flexible multi-precision” story.**  
  This criticism is directly supported by the paper’s own results and discussion. Table 2 shows the multi-precision model trailing both its fixed-precision counterpart and sometimes the non-uniform multi-precision baseline at 3–4 bits; Appendix A.1 further confirms a consistent perplexity gap versus fixed-precision AnyBCQ at 3 and 4 bits. The paper itself explains this: “the additional shared-binary constraint slightly limits the capacity of the multi-precision model.” This does not invalidate the paper, but it does mean the method is best viewed as a strong low-bit / deployment-efficient compromise, not as uniformly best across the full 2/3/4-bit range.
- **Accuracy evaluation across architectures is narrower than the systems claims.**  
  The main benchmark accuracy table is only for **Llama-3.1-8B**. Gemma-2-9B and Phi-4-14B appear in the end-to-end evaluation, but there the comparison is limited to Wiki perplexity, MMLU, and throughput against Any-Precision LLM, rather than the broader task suite used for the main claim. Given that the paper emphasizes a generally applicable multi-precision framework for LLM deployment, more complete cross-architecture accuracy validation would strengthen confidence that the 2-bit gains and 3–4 bit trade-offs are not overly model-specific.
- **The “negligible overhead” claim for dynamic per-request precision selection is only partially substantiated.**  
  The kernel design is convincing at the static GEMV level, and the throughput results are encouraging, but the paper does not isolate the runtime overhead of actually switching precision across requests or during continuous autoregressive serving. Since the claim is specifically about dynamic per-request selection, a targeted experiment measuring the cost of switching precision policies in a live decoding loop would better support that statement.

### Minor
- **Calibration robustness is not analyzed.**  
  The method uses 512 C4 sequences for reconstruction-error optimization (Sec. 4.1), but the paper does not provide sensitivity analyses over calibration set size or distribution. This is not unusual for PTQ papers, and the current evidence is sufficient to show the method works, but such an ablation would help determine whether the particularly strong 2-bit results are stable or calibration-sensitive.
- **The paper remains largely empirical, with limited analytical insight into progressive precision expansion.**  
  This is acknowledged by the authors in Sec. 6: “the present work remains largely empirical and lacks theoretical guarantees.” A stronger empirical diagnostic analysis of how freezing lower-bit binaries affects reconstruction error layer-by-layer or across expansion stages would improve technical understanding, even if formal theory is beyond scope.
- **The mixed-precision case study is directionally useful but does not fully establish practical viability at aggressive average bitwidths.**  
  Table 5 shows AnyBCQ outperforming Any-Precision LLM at equal average precision, but both methods degrade substantially at low average bitwidths (e.g., 2.23 bits). This supports relative superiority, but not yet a compelling practical mixed-precision operating point in the most aggressive regime.

### Trivial
- **Hardware characterization in Appendix A.4 is coarse.**  
  The appendix uses `nvidia-smi` polling for utilization/power characterization, which is acceptable as a rough signal but not a rigorous microarchitectural analysis. This does not affect the main throughput results, but those appendix-level conclusions should be interpreted cautiously.

## Nice-to-Haves
- Add an ablation on calibration data size and distribution (e.g., 512 vs. 2k/4k samples).
- Quantify the cost of the shared-binary constraint more directly, ideally layer-wise or by reporting the delta versus independently optimized fixed-precision models across layers.
- Measure dynamic precision switching overhead during actual autoregressive decoding rather than only static kernel benchmarks.
- Expand the broader accuracy suite beyond Llama-3.1-8B to at least one additional architecture.
- Include profiler-based kernel analysis (e.g., Nsight) for a stronger systems account of memory stalls / utilization.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that the 3× speedup over FP16 is “theoretically implausible” or unsupported because weight-only quantization cannot deliver such gains.**  
  This is too strong and not justified from the paper alone. The paper reports kernel-level and end-to-end throughput numbers on specific workloads; without external benchmarking evidence, it is not appropriate to dismiss them as implausible. The fair retained criticism is narrower: comparisons are limited to the baselines included by the paper, and dynamic-switching overhead is not isolated.
- **Criticism that bit-transpose overhead is merely an artifact of one implementation and therefore the motivation is invalid.**  
  The paper does not claim all non-uniform quantization must incur identical overheads in all imaginable kernels; it argues this overhead is present in the compared prior multi-precision approach and motivates a BCQ-friendly alternative. That is a reasonable claim supported by the presented comparison.
- **Unfair baseline complaints based on missing external methods or calibration mismatches not established in the paper.**  
  Since external baselines and their exact settings cannot be verified here, these criticisms are too speculative. The paper already compares against AWQ, Any-Precision LLM, and ShiftAddLLM, which is a meaningful set for its stated goal.
- **Complaint about missing activation quantization analysis.**  
  The paper is explicitly a **weight-only PTQ** method. Requiring activation quantization is outside the paper’s stated scope.
- **Complaint about unclear total “training time” or comparison to QAT efficiency.**  
  The method is PTQ, and the paper does provide its optimization setup. A detailed wall-clock comparison against QAT would be nice but is not essential to evaluate the core contribution.
- **Formatting/parsing issues in figures/tables.**  
  These are extraction artifacts and not paper weaknesses.

## Novel Insights
The most interesting synthesis across the reviews is that this paper’s real contribution is not “best multi-precision quantization” in the abstract, but a more specific operating point: it identifies BCQ as a particularly strong substrate for **hardware-efficient multi-precision inference when 2-bit capability genuinely matters**. The results suggest a useful deployment niche that prior non-uniform multi-precision methods do not serve well: a single model spanning an aggressive low-bit operating mode with meaningful quality, while preserving acceptable 3–4 bit quality and enabling cleaner bit-plane execution. The flip side is equally important: the same shared-binary mechanism that enables low-overhead multi-precision serving is also the source of its higher-bit ceiling. That trade-off is the central technical reality of the paper.

## Suggestions
- Strengthen the paper’s positioning: present AnyBCQ less as a universally superior multi-precision method and more as a deployment-oriented design that prioritizes **2-bit viability + hardware efficiency**, with explicit higher-bit trade-offs.
- Add a focused ablation quantifying the penalty from freezing lower-bit binaries, ideally per layer and per target precision.
- Include a calibration-size sensitivity study to show the robustness of the unusually strong 2-bit results.
- Directly measure the overhead of switching precisions across requests or tokens in a realistic decoding loop.
- Expand full-task accuracy evaluation to at least one additional architecture beyond Llama-3.1-8B.
- If space permits, add profiler-backed kernel analysis to complement the current latency tables and appendix power/utilization measurements.

---

## SzXDuBN8M1

- GT: Accept (Oral) (avg 7.5)
- Predicted: N/A (7.8/10)
- Match: N/A

### Final Review

## Summary
This paper introduces TD-JEPA, a zero-shot unsupervised RL method that learns policy-conditioned latent-predictive representations from offline, reward-free transitions using a temporal-difference objective. The key idea is to train state and task encoders together with a predictor that approximates successor features in latent space, yielding zero-shot policies for rewards expressible through the learned task representation; empirically, the method is competitive across 65 tasks and is especially strong in pixel-based settings.

## Strengths
- **A genuinely nontrivial unification of latent prediction and successor-feature zero-shot RL.** The paper does more than add an auxiliary JEPA-style loss: the latent-predictive TD objective is the core training signal used to learn encoders, predictors, and policies jointly. This is a specific conceptual contribution, clearly articulated in Sec. 3.3: “the predictor may be leveraged as an approximation of successor features … to extract policies … for all reward functions in the span of the learned features.”
- **Theoretical connection between TD latent prediction and successor-measure factorization is novel and substantive.** The paper proves that, in an idealized setting, the learned predictors recover projected successor features / successor-measure factorizations, and relates the practical TD objective to forward/backward TD losses (Thms. 1–4). Even with strong assumptions, this is more than generic intuition: it gives a precise bridge from non-contrastive latent prediction to zero-shot value estimation.
- **Empirical gains are strongest exactly where they matter most: pixel-based zero-shot RL.** In Table 1, TD-JEPA is consistently at or near the top on DMCRGB and OGBenchRGB, which are the hardest settings for this class of methods. The paper also uses probability-of-improvement plots (Fig. 2) to argue consistency across domains rather than relying only on suite averages.
- **The baseline protocol is unusually careful and materially strengthens the empirical case.** The paper does not simply run prior methods “as is”; it standardizes architectures, adds explicit state encoders where beneficial, reports the effect of doing so (Appendix D.1 / Table 2), and is transparent that several compared methods are novel zero-shot instantiations of representation learners rather than native zero-shot methods. This is a specific fairness strength, not a generic “many baselines” claim.
- **The representation analysis goes beyond leaderboard reporting.** The paper includes ablations on multi-step/policy-aware prediction, symmetric vs asymmetric encoders, adaptation with frozen vs trainable representations, architecture-depth sweeps, and visualization of learned successor-geometry / goal alignment. These analyses support the claimed mechanism rather than only reporting final returns.
- **Learned representations appear reusable beyond zero-shot inference.** Figures 4 and 6 show that TD-JEPA pretraining improves sample efficiency for downstream offline/online adaptation, and that frozen representations are often already sufficient for rapid improvement.

## Weaknesses

###: Fatal

None.

### Major:
- **The paper overstates the scope of its “any reward” claim relative to what the method formally guarantees.**  
  The core mechanism in Sec. 3.3 requires projecting a downstream reward onto the learned task features via linear regression,
  \[
  z_r=\arg\min_z \mathbb{E}_{(s,r)\sim D_{\text{rwd}}}(r-\psi(s)^\top z)^2,
  \]
  and the method then returns policy \(\pi_{z_r}\). This means the practical guarantee is for rewards well represented by the span of \(\psi\), not arbitrary rewards in an unconstrained sense. The paper does partially acknowledge this repeatedly (“for all rewards in the span of \(\psi\)”, “the associated policy \(\pi_{z_r}\) is then returned”), and Theorem 4 is explicit about linear regression onto \(\psi\). However, the abstract still says “This enables zero-shot optimization of any reward function at test time” and the introduction similarly says “for any downstream reward, entirely in latent space.” The theory later refines this claim: exact zero-shot optimality for truly arbitrary rewards requires perfect successor-measure approximation and optimal policies for all linear rewards in \(\psi\)-space, which is a much stronger condition than the headline phrasing suggests. This is not a fatal flaw, but the claim should be narrowed to match the actual method and guarantees.
- **The theoretical results are informative but rest on assumptions that substantially limit their direct applicability to the practical algorithm.**  
  The main theorems in Sec. 4 assume orthonormal / identity-covariance representations, uniform state distributions, and symmetric transition kernels (A1–A3), plus linear predictors in a tabular setting. The non-collapse result in Theorem 2 further assumes a continuous-time relaxation where predictors are optimized to stationarity before representation updates. The paper does not hide this: it explicitly states the setting is “simplified,” notes these assumptions are inherited from prior latent-prediction analyses, and Appendix C discusses relaxations. Still, the practically important point remains that the strongest guarantees do **not** apply to the actual deep, off-policy, asymmetric, discrete-time training setup used in experiments. Appendix C also makes clear that removing symmetry in the cleanest way would require a backward-sampling variant that is “not easy to be optimized off-policy.” So the theory is valuable as structure and intuition, but the gap between theorem and practical algorithm is real and should be emphasized more plainly.
- **Offline robustness is not established as broadly as the paper’s framing suggests.**  
  The method is presented as learning from “offline, reward-free transitions,” and the main algorithm indeed bootstraps with actions sampled from the learned policy at next states. In the main experiments this works well on ExoRL and OGBench, but Appendix D.8 shows a meaningful limitation: on low-quality, low-coverage data, performance degrades and BC/FQL-style regularization becomes important. The paper does discuss dataset regimes (high-coverage ExoRL vs low-coverage OGBench) and in OGBench already uses BC-style regularization (“we additionally apply BC regularization in OGBench…”), so this is not an unacknowledged bug. Still, the main-text framing could better distinguish “works on the benchmarked offline datasets” from “robust on arbitrary offline reward-free data.” As written, the latter implication is stronger than the evidence supports.

### Minor
- **The asymmetric variant’s practical cost is nontrivial.**  
  TD-JEPA trains two encoders and two predictors. Table 4 shows that the asymmetric method is materially slower than the symmetric variant, often by a factor around 2–3x in steps/sec depending on suite. Given that the symmetric variant in Table 3 is often fairly competitive, a clearer compute/performance tradeoff discussion in the main text would strengthen the practical case.
- **The method appears somewhat sensitive to regularization and benchmark regime.**  
  Appendix E/Table 6 shows fairly different orthonormal regularization ranges across methods and domains, especially in OGBench navigation/manipulation. This does not invalidate the results—the authors are commendably transparent about tuning—but it suggests the method is not yet especially plug-and-play.
- **The main paper could foreground limitations of reward inference more explicitly.**  
  The reward-projection step is central to deployment yet mostly described as a linear regression recipe. The paper would benefit from a clearer discussion of when this step is expected to be reliable or fragile—for example, when \(\psi\) under-represents downstream rewards or when inference data are limited/noisy. This is especially relevant because the abstract-level claim is broad.
- **Some main-text performance narration is slightly stronger than the tables warrant.**  
  Overall the empirical story is good, but in several suites performance gaps are modest and confidence intervals overlap. The paper does mitigate this with probability-of-improvement analysis and overlap-aware bolding, which is good practice; still, some verbal claims could be phrased more conservatively.

### Trivial
- **Simulation-only evaluation limits demonstrated significance, though not the paper’s validity.**  
  The paper motivates real-world applications and cites humanoid/robotics directions, but all evidence here is from simulation benchmarks. This is acceptable for the paper’s scope, but real-world relevance remains prospective rather than demonstrated.

## Nice-to-Haves
- Add a main-text experiment or analysis isolating how violations of the theory assumptions (non-uniform data, asymmetric dynamics, predictor under-optimization) affect empirical stability.
- Include a more direct analysis of reward-inference robustness under limited or noisy rewarded samples at test time.
- Provide a clearer compute/performance Pareto comparison between asymmetric TD-JEPA, symmetric TD-JEPA, and strong baselines, since the symmetric variant is often competitive.
- Add more explicit failure-case analysis on underperforming tasks (e.g., harder OGBench domains), especially connecting those failures to representation geometry or reward misspecification.
- A direct frozen-representation downstream RL comparison against strong task-specific offline RL baselines would further clarify how much of the gain comes from zero-shot structure versus simply better pretraining.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Missing comparison to PPO/SAC/CQL/IQL or other specific external baselines.”**  
  Removed because this is partly scope creep and partly unverifiable as a required omission. The paper is explicitly about zero-shot unsupervised RL and compares against a broad set of zero-shot / representation-learning baselines. While extra downstream-RL comparisons could be nice-to-have, framing the absence of particular named external methods as a core weakness is too strong.
- **“Ablation figures lack error bars/confidence intervals.”**  
  Removed as factually incorrect. Figure 3 explicitly says “Error bars represent standard errors,” and the appendix includes uncertainty reporting in multiple ablations/tables.
- **“The paper hides tuning ranges / exact hyperparameter sweeps.”**  
  Removed because the appendix provides substantial tuning details, including architecture hyperparameters (Table 5), regularization ranges (Table 6), and method-specific implementation details.
- **“The orthonormality regularizer is unexplained / not discussed at all.”**  
  Removed in strong form. The paper discusses collapse avoidance repeatedly, includes the regularizer in Algorithm 1, mentions its importance in relation to prior work, and provides theory on non-collapse in the idealized setting. It is fair to ask for more sensitivity analysis, but not fair to claim it is missing discussion entirely.
- **Generic strength: “the paper is well-written / experiments are extensive.”**  
  Removed because these are generic. The retained strengths point to specific unusual merits instead.

## Novel Insights
A key synthesis across the reviews and the paper itself is that TD-JEPA is strongest not merely because it is “non-contrastive,” but because it aligns the representation-learning target with **policy-conditional long-horizon occupancy** rather than behavior-policy prediction or generic future-state prediction. The appendix visualizations and the comparison against BYOL/BYOL-\(\gamma\) support a more precise interpretation: the method’s advantage in pixels likely comes from learning latents whose geometry is shaped by **directed control-relevant future behavior**, not just visual similarity or undirected visitation. This makes the paper less about swapping contrastive for JEPA, and more about choosing the right predictive object for zero-shot control.

## Suggestions
- Narrow the headline claim from “any reward” to “any reward representable or well-approximated in the span of the learned task encoder,” and state the stronger arbitrary-reward result only under the exact conditions of Theorem 4.
- Bring the low-coverage offline-data limitation from Appendix D.8 into the main paper, and explicitly position BC/FQL-style regularization as recommended in low-support regimes.
- In Sec. 4, add a short paragraph explicitly separating what the theory proves about the idealized linearized dynamics from what is only empirically validated for the practical deep algorithm.
- Add a concise main-text discussion of the asymmetric-vs-symmetric compute tradeoff using Table 3 + Table 4.
- Expand the test-time reward-inference discussion to address noisy/limited rewarded samples and possible conditioning issues in \(\mathbb{E}[\psi\psi^\top]\).
- If space allows, include one focused failure-case analysis on an OGBench task where the method is not best, linking failure to either reward misspecification, coverage, or representation limitations.



---

## mGeeRFToaW

- GT: Accept (Poster) (avg 5.2)
- Predicted: N/A (5.6/10)
- Match: N/A

### Final Review

## Summary
This paper proposes Quantized Zeroth-order Optimization (QZO), a method for fine-tuning post-training-quantized models by applying zeroth-order optimization to continuous quantization scales rather than discrete quantized weights. The key practical claim is that this removes gradients and optimizer states while also reducing weight memory through quantization, enabling very low-memory adaptation of large models, including 4-bit 7B LLMs and a 2-bit 13B LLM on a single 24GB GPU. The paper also introduces directional derivative clipping (DDC) as a stabilization mechanism for zeroth-order training.

## Strengths
- **The core technical idea is specific and nontrivial:** perturbing continuous quantization scales instead of discrete quantized weights is a clean way to bridge zeroth-order optimization with post-training quantization. This directly addresses the paper’s stated obstacle that discrete weights cannot be perturbed continuously and continuous ZO updates cannot be directly applied to quantized integers.
- **The method is demonstrated across two distinct quantization regimes:** the paper does not only test standard 4-bit scalar quantization (GPTQ), but also a 2-bit codebook-based setting (AQLM), which strengthens the claim that the approach is compatible with multiple PTQ styles rather than tied to a single quantizer.
- **The low-memory operating point is practically compelling:** the paper presents an end-to-end recipe that reportedly fits adaptation of large models into consumer hardware, including 4-bit 7B-class models and a 2-bit Llama-2-13B on a 24GB GPU. Even allowing for caveats in attribution, this is a concrete and useful systems outcome.
- **The stabilization ablation is materially informative:** Figure 2 and Figure 3 show that training without DDC is unstable and that clipping threshold meaningfully affects optimization behavior. This is more useful than a generic ablation because it exposes an actual failure mode and a corresponding control knob.
- **The appendix discussion with QLoRA is helpful and revealing:** Table 5 clarifies that QZO is not competitive with strong first-order PEFT in raw quality, but can be combined with PEFT while retaining a very small memory footprint. This makes the paper’s practical niche clearer.

## Weaknesses

###: Fatal
- **The theoretical justification for DDC is not sound as written.** Theorem 1 claims that the clipped estimator remains unbiased, but the paper does not establish conditions under which this would hold, and the Appendix proof is not convincing. Since clipping is a nonlinear operation applied to the directional derivative, a general unbiasedness claim is highly suspect without much more careful assumptions and derivation. The subsequent variance argument in Eq. 8 then relies on this theorem. This does **not** invalidate the empirical usefulness of DDC, but it does invalidate the paper’s current theoretical claim that clipping is an unbiased variance-reduction device.

### Major:
- **The FLOPs claim in Table 2 appears seriously misleading or miscomputed.** The paper states that QZO uses “about 1% of the FLOPs of MeZO,” but both methods are zeroth-order and require two forward passes per step to estimate a directional derivative. While QZO updates many fewer trainable variables, that does not by itself reduce the dominant cost of running the full model forward twice. The paper’s own explanation—“This is because QZO only fine-tunes the continuous quantization scale while leaving most weights fixed”—does not justify a 100× FLOP reduction for a full-model ZO method. This weakens the paper’s computation-efficiency claim substantially.
- **The memory-efficiency presentation over-attributes gains to the optimizer rather than to the optimizer-plus-quantization combination.** The paper’s framing is partly fair—its stated goal is a unified framework that minimizes weights, gradients, and optimizer states jointly—but the comparison to MeZO can read as if QZO itself yields a 3× memory reduction over MeZO through optimization alone. In fact, relative to MeZO, the memory difference is primarily from using quantized weights rather than from a new zeroth-order memory trick. The contribution is still meaningful as a combined recipe, but the paper should isolate what comes from ZO and what comes from PTQ more carefully.
- **The evaluation regime is narrow for supporting broad fine-tuning claims.** The main LLM experiments use 1,000 training examples per dataset and only five relatively small NLP tasks. This setup is sufficient to show feasibility, but it is not enough to establish that QZO is broadly effective for realistic fine-tuning workloads. This matters especially because zeroth-order methods can behave differently in larger-data regimes.
- **The empirical positioning against stronger memory-efficient first-order baselines is incomplete in the main paper.** The appendix includes QLoRA comparisons, and those results are informative: first-order PEFT is clearly stronger in task quality, while QZO occupies a different memory/quality tradeoff. But because this comparison is relegated to the appendix and not integrated into the main experimental narrative, the paper does not fully establish where QZO sits relative to standard practical alternatives under matched resource budgets.
- **The 2-bit 13B evaluation is under-contextualized.** Table 3 compares against zero-shot baselines, which demonstrates improvement over no adaptation, but does not sufficiently benchmark the method against stronger alternatives in that extreme setting. As a result, the 2-bit result is impressive as a feasibility demonstration but weaker as evidence of competitive learning quality.

### Minor
- **QZO has a substantial quality gap to first-order tuning/PEFT, and the paper’s practical claim should be framed more explicitly as a memory-first tradeoff.** This gap is visible in Table 1 and Appendix Table 5. The paper acknowledges some of this, but the practical takeaway should be sharper: QZO is appealing when memory is the dominant constraint, not when best downstream accuracy is the goal.
- **DDC appears important but somewhat sensitive.** Figure 3 suggests nontrivial dependence on the clipping threshold: too small underfits, too large destabilizes. The paper does provide a useful ablation, so this is not unaddressed, but it does indicate that the method is not fully plug-and-play.
- **The diffusion-model extension is interesting but not yet fully validated.** The appendix presents only qualitative results and the paper itself acknowledges a noticeable gap to the target distribution. This is better viewed as a promising extension than a strong empirical pillar of the paper.

### Trivial
- None.

## Nice-to-Haves
- Report gradient-alignment diagnostics, e.g., cosine similarity between QZO’s scale-direction estimates and first-order gradients with respect to scales on smaller models, to clarify whether scale perturbations recover meaningful optimization directions.
- Recompute and clarify FLOPs accounting so that forward-pass cost dominates the analysis appropriately for ZO methods.
- Add memory profiling at actual training batch sizes in addition to batch size 1, since the current profiling is mainly a minimum-VRAM demonstration.
- Expand larger-scale or larger-data fine-tuning experiments to show whether the method remains useful beyond 1k-example settings.
- Promote the QLoRA comparison from the appendix into the main paper, ideally under matched memory/computation budgets.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Unfair comparison with baselines because settings differ from QLoRA.”** The paper’s appendix explicitly frames QZO and PEFT as orthogonal methods and provides a comparison plus a hybrid. This is not a straightforward unfair-comparison issue; the more valid concern is incomplete positioning against practical first-order alternatives, which is kept above.
- **“The paper lacks an ablation on clipping threshold C.”** This is factually incorrect. Figure 3 is precisely an ablation on the clipping threshold.
- **“Q-SPSA is theoretically sound.”** This should not be retained as a strength in that form, because while the Q-SPSA construction is intuitive and technically reasonable, the paper’s main theoretical claim around clipping is not convincingly established.
- **Pure reproducibility complaints about omitted details or release status.** The paper provides code, implementation details, and cites all tools/models used; no concern of this form should be treated as a substantive weakness.
- **Formatting/parser artifacts in tables/figures.** These are extraction issues and not paper issues.

## Novel Insights
The paper’s real contribution is strongest when interpreted as a **systems recipe for the most memory-constrained end of adaptation**, not as a new theoretically grounded optimizer. The experiments and appendix together suggest a clearer positioning than the paper itself emphasizes: QZO is valuable because it exploits the tiny trainable surface offered by quantization scales to make zeroth-order adaptation feasible under very low VRAM, but this same restriction also explains why it tends to trail first-order PEFT methods in quality. In other words, the paper is most convincing as an argument that **quantization scales are a surprisingly capable optimization interface for ultra-low-memory adaptation**, rather than as evidence that QZO is a generally competitive replacement for gradient-based fine-tuning.

## Suggestions
- Remove or substantially weaken Theorem 1 and the associated unbiasedness/variance claims unless a correct proof under explicit assumptions can be provided.
- Rework the efficiency section to decompose gains into:
  - savings from zeroth-order optimization (no gradients / optimizer states), and
  - savings from quantization (reduced weight storage).
- Recompute Table 2 FLOPs with a transparent accounting that reflects the cost of two full forward passes per ZO step.
- Move the QLoRA comparison into the main paper and explicitly position QZO as a memory-first alternative rather than a generally competitive fine-tuning method.
- Strengthen the 2-bit section with stronger baselines or, at minimum, frame it clearly as a feasibility result rather than a competitive benchmark.
- If theoretical analysis is retained, focus on a bias-variance tradeoff statement for clipping rather than claiming unbiasedness.

---

## zWdJIhl4Bw

- GT: Reject (avg 4.5)
- Predicted: N/A (7.8/10)
- Match: N/A

### Final Review

## Summary
This paper studies how to adapt a perspective-pretrained 3D Transformer (VGGT) to equirectangular panoramic inputs without retraining the backbone. The proposed “projection-domain adaptation” combines ERP-consistent ray lifting, ray-field token augmentation, head-only dual-branch LoRA, and latitude-aware depth uncertainty weighting, and shows strong gains over naïve ERP fine-tuning on a curated Matrix-3D subset as well as indoor 360° datasets, while using far fewer trainable parameters and training compute.

## Strengths
- **The paper isolates two concrete projection-domain failure modes and ties them to specific design choices.** Section 3.2 is more than generic motivation: it distinguishes **measure mismatch** (planar loss on spherical pixels) from **proxy-focal entanglement** (using fictitious pinhole intrinsics for ERP), and the proposed remedies in Sections 3.3–3.5 map cleanly onto these failures.
- **The parameter-efficiency result is genuinely strong and specific.** Across the main tables, the head-only LoRA variant is consistently close to the authors’ own full-FT variant while updating only ~0.6M parameters versus ~35M, and with substantially lower training time (e.g., Appendix A.10: 28h on 1 A100 for LoRA vs. ~185h on 4 A100 for the authors’ full-FT variant).
- **The paper shows that naïve full fine-tuning can be worse than minimal geometric interface correction.** This is an interesting and nontrivial empirical outcome: plain VGGT full fine-tuning under ERP is much worse than the proposed interface-aware adaptation across depth, pose, and 3D point quality.
- **The work goes beyond a single synthetic benchmark.** It includes indoor real 360° evaluation on Stanford2D3D and Matterport3D, plus OOD transfer from Matrix-3D to indoor data, which helps support the claim that the method is not narrowly overfit to one training setup.
- **Some diagnostic analysis is unusually useful.** Table 7 (“Where to adapt?”) directly addresses adaptation locus, and Table 9 provides evidence that the predicted uncertainty is not arbitrary: high-σ pixels indeed concentrate a disproportionate share of squared depth error.

## Weaknesses

###: Fatal
None.

### Major:
- **The central “interface, not backbone” claim is only partially supported because the strongest counterfactual is missing.**  
  The paper argues that correcting the projection interface is the right locus of adaptation and that head-only LoRA suffices, but the full comparison to a *successfully trained* backbone+head fine-tuning model under the same corrected interface is absent. In Section 4.6 / Appendix A.6, the authors state that “**Full backbone+head finetuning under the ERP interface was unstable in preliminary runs**” and Table 7 only compares head-only LoRA against backbone LoRA variants, not against a converged full-FT-with-interface model. This does **not** invalidate the empirical value of the proposed method, but it does weaken the stronger architectural claim that backbone adaptation is unnecessary in principle rather than merely harder to optimize in this setting.
- **The Matrix-3D evaluation relies on very aggressive curation, which narrows the scope of the claims.**  
  Section 4.1 and Appendix A.2 show that the dataset was reduced from **116,759** sequences to **2,196**, explicitly emphasizing “mid- and near-range geometry” and filtering out many extreme long-shot, sky/grass-dominated cases. This is a reasonable benchmark-design choice for studying geometric adaptation, but it also means the evidence is strongest for scenes with substantial visible structure and weaker for the most challenging panoramic regimes. Since the paper’s framing is broad (“projection-domain adaptation” for panoramic scene reconstruction), this curation should be treated as a substantial scope limitation rather than a minor implementation detail.
- **Key implementation details of how ray-augmented tokens enter the frozen backbone are underspecified.**  
  Equation (6) defines token augmentation as \(t_i^{(0)}(u,v)=t_i^{RGB}(u,v)\oplus \Phi(r(u,v))\), i.e., concatenation of ray embeddings to image tokens before the frozen backbone. But the manuscript does not clearly explain how this increased dimensionality is reconciled with VGGT’s fixed input feature size, nor whether an additional projection layer is used and, if so, where it sits and whether it is trainable. Since the method’s core mechanism is token-level ray augmentation into a frozen pretrained model, this is an important technical omission affecting clarity and reproducibility.

### Minor
- **Some wording overstates the geometric guarantees.**  
  The paper sometimes uses language like “restores the geometric invariances broken by ERP” and says the tokenization “enables directional equivariance” or “SO(3) directional consistency.” What the method clearly provides is an explicit directional coordinate prior and better projection-consistent supervision; it does not establish formal SO(3)-equivariance of the frozen transformer. This is mainly a claim-calibration issue, but the current phrasing is stronger than what is demonstrated.
- **The ablation coverage is good but still incomplete on the most mechanism-specific design choices.**  
  Table 4 covers geometric interface, loss design, and LoRA rank, and Table 7 covers placement. However, the paper does not isolate sensitivity to the **ray embedding design itself** (e.g., embedding dimension or encoding choice), even though this is central to the proposed interface.
- **The efficiency headline could be contextualized more carefully.**  
  The repeated “~25× lower cost” comparison appears to be against the naïve full fine-tuning baseline in the main tables, while Appendix A.10 indicates that compared to the authors’ own interface-corrected full-FT variant the wall-clock reduction is smaller on a per-training-run basis (28h on 1 A100 vs. ~185h on 4 A100). The main claim is directionally correct—LoRA is much cheaper—but the paper should be more explicit about which full-FT comparator each efficiency number refers to.
- **3D evaluation is narrower than the paper’s world-model framing suggests.**  
  The 3D point quality metric in Table 3 is useful, but it is derived from reconstructed points via predicted depth+camera and evaluated after Umeyama alignment. This supports reconstruction quality, but it does not fully establish scale-faithful, long-horizon, temporally consistent 3D world modeling. The paper’s claims are strongest for multi-view depth/pose transfer, and somewhat broader than the current evaluation directly proves.

### Trivial
- **Latitude-specific evidence for the claimed correction would make the story sharper.**  
  Since the paper’s theoretical motivation emphasizes ERP distortion varying with latitude, showing error as a function of latitude or pole-vs-equator breakdowns would directly validate that the proposed weighting and ray interface are fixing the intended problem rather than just improving average performance.

## Nice-to-Haves
- Add a stronger optimization study for full fine-tuning under the corrected ERP interface, since this is the most important missing comparator for the paper’s main conceptual claim.
- Provide an explicit architectural diagram or formula for how concatenated ray features are projected into the frozen VGGT token dimension.
- Report latitude-resolved depth/pose errors or targeted pole/equator analyses to validate the geometric story more directly.
- Include uncertainty calibration analysis beyond correlation (e.g., calibration plots), since the current evidence mainly shows usefulness for weighting rather than calibrated confidence.
- Evaluate longer sequences than the fixed \(K=10\) frame protocol to better support the “world model” framing with respect to drift and temporal consistency.
- Clarify more prominently in the main text that many depth metrics are median-aligned per sequence, and distinguish claims about relative depth quality from claims about metric-scale 3D reconstruction.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Concerns about code/weights release status or institutional approval.**  
  Removed per instruction. The paper explicitly discusses intended release in Appendix A.9/A.12, and release-status concerns should not be treated as scientific weaknesses.
- **Complaints about missing comparisons to uncited external panoramic or PEFT methods.**  
  Removed because I cannot verify omitted related work beyond the paper, and the review should not speculate about missing baselines from outside the submission.
- **Claims that comparisons are unfair because baselines are weaker under ERP than the proposed method.**  
  Removed. The paper also includes cubemap variants for several baselines, and asymmetries that favor the baseline are not a valid weakness under the stated rules.
- **Generic reproducibility nitpicks about hyperparameters.**  
  Removed. Appendix A.10 already provides optimizer/schedule/augmentation/LoRA details at a level that is broadly standard for this kind of work.
- **“Not even a paper” style objections about the existence or release of referenced tools/datasets/models.**  
  Removed per instruction and because they are not grounded in the manuscript.
- **A pure demand for more geometry metrics such as Chamfer/normal accuracy as a core flaw.**  
  Weakened rather than kept as a major issue: Table 3 already reports 3D point quality (Acc/Comp/Overall after alignment), so it is inaccurate to say quantitative 3D evaluation is absent. The fair criticism is that the 3D evaluation is somewhat limited relative to the broad framing, not that it is missing.

## Novel Insights
The most interesting synthesis across the paper and reviews is that the work is strongest not as a universal statement that “backbones should never be adapted,” but as evidence for a more specific principle: **for projection shift, the highest-leverage intervention may be to repair the sensor interface before spending capacity on the model interior**. The results suggest that much of VGGT’s pretrained geometric prior survives panoramic transfer once two projection-specific mismatches are corrected—ray geometry and spherical sampling measure. At the same time, the missing successful full-FT-with-interface comparison means the paper currently supports this as a strong empirical design heuristic, not yet as a definitive architectural law.

## Suggestions
- **Clarify token integration rigorously.** Add one explicit equation or diagram showing the dimensional path from \(t^{RGB}\oplus \Phi(r)\) to the frozen VGGT input size, and state whether any projection layer is frozen/trainable and included in the parameter count.
- **Temper the strongest claims.** Replace wording implying formal equivariance/invariance restoration with more precise language about projection-consistent geometric conditioning and supervision.
- **Strengthen the central comparison.** If possible, add a more thoroughly tuned full fine-tuning experiment under the corrected interface; if not, explicitly narrow the claim from “head-only is sufficient/optimal” to “head-only is the most reliable and efficient strategy we found.”
- **Elevate the dataset-curation limitation into the main text.** State clearly that the curated Matrix-3D subset emphasizes scenes with stronger mid-range structure and that the conclusions are therefore best supported in that regime.
- **Add latitude-resolved diagnostics.** A plot of error versus latitude for naïve ERP, plain full FT, and the proposed method would directly test the paper’s core geometric hypothesis.
- **Be explicit about metric scale.** Since depth is median-aligned when scale is ambiguous and 3D points are Umeyama-aligned, the paper should clearly separate claims about geometric consistency from claims about absolute metric reconstruction.

---

## ewdqbKskUL

- GT: Reject (avg 4.0)
- Predicted: N/A (5.6/10)
- Match: N/A

### Final Review

## Summary
This paper formalizes **answer-set consistency** for LLMs answering enumeration questions: if two questions are known to stand in relations such as equivalence, containment, disjointness, or set difference, then the model’s returned answer sets should respect those relations. To study this, the authors build a 600-quadruple benchmark (ASCB), evaluate 18 LLMs, and test prompting-based mitigations, finding that inconsistency is common even for strong models and that relation-aware prompting can improve measured consistency.

## Strengths
- **The paper identifies a genuinely distinct failure mode and formalizes it cleanly.** The distinction between (i) violating the gold relation between question pairs and (ii) **self-contradicting** a relation the model itself predicts is useful and sharper than prior “consistency” notions built around single-answer QA or boolean statements. Section 3.1 gives a clear set-theoretic formulation, and Appendix F extends this to internally contradiction-free behavior.
- **The benchmark design is more structured than typical paraphrase-consistency probes.** ASCB is built around quadruples \((Q_1,Q_2,Q_3,Q_4)\) that induce multiple relations at once—equivalence, narrower containment, disjointness, and a ternary set-difference relation \(E_{4,1\setminus 3}\). This gives the paper leverage to compare relation types rather than only paraphrase equivalence.
- **The empirical picture is richer than a simple leaderboard.** The paper does not just report one aggregate metric; it separates classification accuracy, relation-specific consistency, Jaccard similarity, refusal/empty-response rates, and a repeated-query control \(E_{1,*}\). This supports the central claim that answer-set inconsistency is pervasive and relation-dependent.
- **One specific finding is quite interesting:** models can often **recognize** the intended relation much better than they can **produce answer sets that satisfy it**. This gap between relation classification and consistent enumeration is one of the more compelling takeaways of the study.
- **The mitigation experiments are useful diagnostically, even if not fully convincing as a deployable solution.** The CtE and Oracle settings provide evidence that prompting the model to reason about relations before answering can alter behavior substantially, especially on harder relations.

## Weaknesses

### Major:
- **The main mitigation claim is substantially confounded by the evaluation protocol excluding refusals/empty answers from consistency metrics.**  
  This is the most important issue. In Section 3.4, the paper defines consistency rates while explicitly excluding empty answers and `"idk"`: “**Here we exclude empty answer sets and responses of ‘idk’, which are reported separately.**” In Section 4.2, the authors themselves note that under CtE, “**LLMs tend to adopt a safer approach by answering ‘idk’ when uncertain, which may explain why CtE outperforms the other two strategies.**”  
  This means CtE can improve reported consistency by declining a larger share of difficult cases rather than by actually producing more logically coherent answers on the full benchmark. Table 3 shows this tradeoff clearly for several models (e.g., GPT-5, GPT-4o, Mistral-small, GPT-oss-20b). As a result, the headline conclusion that CtE “mitigates” inconsistency is only partially supported: it improves **conditional consistency on answered cases**, but the paper does not establish improvement in a refusal-aware end-to-end metric over the full evaluation set. For a reliability paper, this is a significant weakness.

- **The answer-set extraction/evaluation appears too brittle to surface-form variation, and the paper does not convincingly quantify how much this inflates inconsistency.**  
  The benchmark treats outputs as sets and computes exact set-based relations plus Jaccard similarity over extracted answers. But the paper’s own error analysis in Appendix H acknowledges that models often use different names for the same entity, e.g. “**Spain**” versus “**Kingdom of Spain**.” That is not a minor edge case; it directly affects both exact consistency and Jaccard-based evaluation.  
  While the prompt tries to reduce variability by asking for full names and a pipe-delimited exhaustive list, the paper does not describe a robust entity normalization or semantic matching stage before scoring. Appendix A.4 describes storing responses and computing metrics, but not a serious canonicalization pipeline. Without this, some portion of the reported inconsistency is likely due to lexical variation rather than genuine logical failure. This does not invalidate the phenomenon, but it weakens the absolute interpretation of the reported inconsistency rates and especially cross-model comparisons.

- **The paper over-interprets the \(E_{1,*}\) control as evidence about causes (“stochasticity” vs “semantic misunderstanding”).**  
  The control is useful, but the causal claims are stronger than what the design cleanly supports. The paper defines \(E_{1,*}\) as asking “**the same question \(Q_1\) posed in a different context at a different time**” (Table 2), then in Section 3.4 states that comparisons involving \(E_{1,*}\) can help assess the role of stochasticity versus semantic misunderstanding. However, “different context at a different time” changes more than just generation stochasticity; it also allows differences due to conversational context effects, backend nondeterminism, serving changes, or other environmental factors.  
  More importantly, the gap between \(E_{1,*}\) and \(E_{1,2}\) does not isolate “semantic misunderstanding” of set-theoretic relations, because \(E_{1,2}\) also introduces paraphrase sensitivity and retrieval/recall differences between phrasings. The paper’s analysis here is suggestive, not conclusive, and should be framed more cautiously.

- **A factual-accuracy/completeness baseline is missing, which limits how to interpret consistency as a reliability measure.**  
  The paper explicitly states in Section 3.1 that “**We do not need ground-truth answer sets for questions in order to analyze answer-set consistency.**” That is true for measuring internal consistency, but it also means a model can be consistently wrong, consistently incomplete, or consistently overcautious. The current results show that consistency is low, but they do not show whether more consistent models are actually better at correct exhaustive enumeration. Since the paper motivates consistency as improving reliability for QA, some factual accuracy/completeness anchor would materially strengthen the claims.

### Minor
- **The benchmark scope is intentionally narrow, which is acceptable, but it limits generality.**  
  The dataset focuses on English, factual, relatively “crisp” enumeration questions with 2–100 answers, largely inspired by KGQA sources and substantial manual curation. This is appropriate for a first benchmark, but the conclusions should remain scoped to this regime rather than broader “LLM question answering” generally.

- **The mitigation comparison is incomplete without a more standard reasoning baseline.**  
  CtE is compared to Base and Oracle, but not to a simpler generic reasoning prompt (e.g., “think step by step before listing answers”). Without such a baseline, it is unclear how much of the gain is specific to relation classification versus simply eliciting more deliberate answering behavior.

- **There is limited analysis of how difficulty scales with answer-set size or domain.**  
  Since the dataset spans answer sets from 2 to 100 entities, a stratified analysis by cardinality could reveal whether failures are mainly set-reasoning failures or simply list-completeness degradation on larger answer sets. Likewise, domain-wise analysis could help distinguish knowledge gaps from relation reasoning failures.

- **The statistical significance analysis is stronger on p-values than on practical effect characterization.**  
  The McNemar tests are reasonable for paired binary outcomes, but given the dataset size, very small effects can become highly significant. The paper would benefit from fuller reporting of refusal-aware effect sizes and practical deltas, not only p-values.

### Trivial
- **There is a correctness issue in the description of disjointness/Jaccard.**  
  In Section 4.2 the text says, “**for \(D_{3,4}[SIM]\), a score lower than 0 for Jaccard similarity is better**,” which is mathematically impossible since Jaccard similarity is in \([0,1]\). The intended meaning is clearly that **closer to 0 is better** for disjointness. This is easy to fix, but it should be corrected.

## Nice-to-Haves
- Add a **refusal-aware primary metric** over all benchmark instances, e.g., treating `idk`/empty as failures for end-to-end consistency, or reporting coverage-consistency curves.
- Add a **canonicalization/normalization layer** (entity matching, alias resolution, or semantic equivalence adjudication) and re-run a sensitivity analysis to estimate how much inconsistency comes from lexical variation.
- Include a **generic CoT/reasoning baseline** to determine whether CtE’s gains are specific to explicit relation classification.
- Provide analysis by **answer-set cardinality**, **domain**, and **relation misclassification vs enumeration failure**, which would make the causal story much sharper.
- Add a modest **accuracy/completeness check** on a subset of the benchmark to connect internal consistency to actual QA reliability.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Dataset construction quality concerns because LLMs were used during curation, implying contamination/unverifiability.”**  
  The paper is explicit that LLMs were used for suggestion and filtering, but also repeatedly states that the final dataset was **heavily manually revised and curated by three authors**. This is a valid scope/generalization concern, but not evidence of contamination or invalidity by itself.

- **“The Oracle strategy is unrealistic, therefore a weakness of the paper.”**  
  The paper already presents Oracle as an **ideal diagnostic upper bound**, not as a deployable method: “**This task is an ideal version of Task 2... This will give us insights into what the model could achieve**.” Criticizing it for not being directly deployable misunderstands its role.

- **“Prompt formatting strictness is itself a core flaw.”**  
  The strict output format could interact with parsing, and the lack of a detailed normalization/parsing description is a fair concern. But simply objecting that models may struggle with pipe-delimited outputs is not, by itself, a substantive weakness.

- **“Unfair comparison because some baselines are treated asymmetrically.”**  
  No concrete unfairness of this kind was substantiated in the paper text.

## Novel Insights
The most interesting synthesis across the reviews is that the paper’s strongest contribution is probably **diagnostic rather than mitigative**. The benchmark exposes a real gap between (a) recognizing set relations between questions and (b) actually producing answer sets that obey those relations. That gap suggests LLM failures here are not reducible to simple ignorance: models can often state the right relational structure yet fail to realize it in generated enumerations. At the same time, the current mitigation results indicate that “improving consistency” is entangled with abstention behavior, so the paper is most compelling as a study of a new failure mode and benchmark, less so as evidence that the proposed prompting strategy solves it.

## Suggestions
- Redefine the main mitigation evaluation around a **full-coverage, refusal-aware metric** and move the current exclusion-based CON metric to a secondary analysis.
- Implement and report an **entity normalization sensitivity study**; even a manually validated subset would help establish how much of the inconsistency is semantic vs lexical.
- Soften the causal claims around **stochasticity vs semantic misunderstanding** unless supported by tighter controls.
- Add at least one **generic reasoning prompt baseline** to contextualize CtE.
- Report **cardinality-stratified** and, if feasible, **domain-stratified** results.
- Include a **small factual correctness/completeness audit** so that consistency can be tied to practical reliability rather than only internal coherence.



---

## rajioNWfRs

- GT: Accept (Poster) (avg 6.7)
- Predicted: N/A (6.6/10)
- Match: N/A

### Final Review

## Summary
This paper proposes TNT, a two-stage training framework for deep memory modules such as Titans: an efficiency-focused pretraining stage that uses hierarchical global/local memories plus periodic local-state resets to enable context parallelism, followed by a brief fine-tuning stage at smaller local chunk sizes to recover high-resolution inference performance. Empirically, on 150M-scale models, TNT substantially improves wall-clock training efficiency relative to vanilla Titans while also improving perplexity and downstream accuracy, making deep test-time memorization models meaningfully more practical to train.

## Strengths
- **The paper identifies and directly tackles a real bottleneck for deep memory modules: poor hardware utilization caused by small chunk sizes and sequential dependencies.** The central mechanism—periodically resetting local memory while adding a large-chunk global memory—specifically targets context parallelization for *non-linear* memory modules, which is much less straightforward than in linear-RNN/scan-based settings. This is a concrete methodological contribution, not just an engineering tweak.
- **The hierarchical design is empirically validated rather than asserted.** In Table 3, removing the global memory substantially hurts performance (PPL 25.60 vs 21.04), while increasing the number of local modules improves results. This supports the paper’s intended division of labor between global long-range context and local high-resolution memory.
- **The two-stage training recipe appears practically effective on the paper’s chosen scale.** Table 1 shows large reductions in time-to-target-loss relative to Titans baselines, and Table 2/Table 3 indicate that Stage 2 fine-tuning gives additional gains at only modest extra cost (Table 4). This is a useful contribution because the method is framed as a training paradigm rather than a wholly new architecture.
- **The Q-K projection is a specific, testable intervention with nontrivial effect.** The paper motivates it as addressing compression/retrieval mismatch and shows in Table 3 that removing it degrades performance materially (PPL 22.01 vs 21.04; accuracy 36.4 vs 40.6). Regardless of whether one fully buys the “fundamental mismatch” framing, the mechanism appears to help in this architecture.

## Weaknesses

###: Fatal
None.

### Major:
- **The generality claims are broader than the evidence.** The paper repeatedly presents TNT as “a general training paradigm applicable to any deep memory module,” and the abstract also says it is evaluated on “Titans and TTT models.” However, the experimental section states “we instantiate it with… Titans,” and the reported tables are Titans-based. Appendix D discusses possible applicability to other architectures, but this is not empirical validation. As written, the paper convincingly demonstrates TNT on Titans, but not yet as a broadly validated paradigm across deep memory modules.
- **The empirical case for “removing a critical scalability barrier” is promising but still limited in scale and scope.** The main experiments are at 150M parameters and 10B tokens, with runtime scaling to 32K sequence length and evaluation on perplexity plus a few commonsense tasks. That is enough to show clear practical improvement over Titans at this scale, but it is not yet enough to fully justify the strongest claims about establishing a practical foundation for scaling this paradigm more broadly. In particular, there is no evaluation at larger model sizes, and no dedicated long-context retrieval/state-tracking benchmark to verify that the global/local decomposition preserves behavior across shard boundaries.
- **The paper does not sufficiently analyze the consequences of periodic local resets on information loss and dependency handling across shard boundaries.** The method’s core approximation is to reset local state every \(S_L\) tokens and rely on global memory to preserve broader context. This is plausible and empirically somewhat supported by the “w/o global memory” ablation, but the paper does not directly measure what is lost at reset boundaries, which types of dependencies are recovered by the global path, or how performance changes as shard length varies. Since this approximation is central to the method, a more focused analysis is important.
- **The Q-K projection introduces a potentially nontrivial \(O(d^2)\) state cost that is not analyzed.** Appendix C describes maintaining a projection matrix \(M_t \in \mathbb{R}^{d \times d}\) as a running sum of outer products. While this is “constant-size” with respect to sequence length, it is not small with respect to hidden width, especially if there are multiple local memories. The paper shows the mechanism helps, but does not quantify its memory/compute overhead or discuss whether it becomes problematic at larger widths.

### Minor
- **The integration of global and local retrieval paths is not emphasized clearly enough in the main presentation.** A harsh reviewer claimed this combination was undefined, but the paper does in fact specify additive composition in Appendix E:  
  \(o_t = f(V_{\xi(t,C_G)}, q_t) + \sum_i f(W_t^{(i)}, \text{projected } q_t)\) (Eq. 15).  
  So this is not a missing-method issue, but it is underexplained in the main text, where the reader is told only that outputs are “combined.” Given how central this is, the main body should state the exact composition rule explicitly.
- **The strongest speedup headline is tied to the most accurate Titans baseline, not the fastest one, and the framing could be clearer.** The abstract says “up to 17× faster than the most accurate baseline configuration.” This is technically consistent with Table 1 if the reference is the small-chunk Titans setting that attains the target quality, but readers may overinterpret this as a general throughput advantage over all reasonable Titans settings or over Transformer baselines. The paper does partially clarify this in Section 5.2 by noting that TNT does not yet beat highly optimized FlashAttention Transformers in end-to-end training time. Still, the headline should be phrased more carefully to avoid sounding broader than the evidence.
- **The Stage 2 adaptation story would be more convincing with a more explicit sensitivity study.** The paper reports that Stage 2 is brief and cheap, but does not provide much detail on stability across different fine-tuning budgets, learning rates, or frozen/trainable parameter choices. Since the core claim is that a short second stage bridges chunk-size mismatch efficiently, this deserves a more systematic characterization.
- **Hyperparameter interaction remains somewhat heuristic.** The method introduces \(C_G\), one or more \(C_L\), and shard length \(S_L\). The appendix provides some ablations and heuristics, but there is not yet a principled recipe for selecting these across scales or tasks. This does not invalidate the results, but it may limit adoption.

### Trivial
- **The motivation for Challenge 2 is somewhat overstated conceptually.** The paper frames using queries for retrieval after training on keys as a “fundamental inconsistency.” That is stronger than what the evidence supports. What the experiments do show is that the proposed projection helps in this architecture; the paper would be stronger if it presented this as an empirically useful alignment mechanism rather than as a broadly established flaw in prior associative-memory formulations.

## Nice-to-Haves
- Add experiments on at least one additional deep memory architecture (e.g., TTT) to substantiate the “general paradigm” claim.
- Include a direct analysis of shard-boundary effects, such as long-range retrieval/state-tracking performance as dependencies cross one or more reset boundaries.
- Report the actual memory and compute overhead of Q-K projection as a function of hidden dimension and number of local memories.
- Extend scaling experiments to larger models to show that the efficiency/quality tradeoff persists beyond 150M.
- Provide a more systematic Stage 2 study: compute budget, chunk-size transition schedule, stability curves, and whether only local memories are tuned in all reported results.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The train-inference mismatch is irreconcilable and invalidates the paper.”** This is too strong and not supported by the paper’s actual setup. The method does not claim that local memory alone handles unlimited context at inference; rather, the global memory is explicitly intended to preserve long-range information while local memories are high-resolution and periodically reset. The concern about reset-induced mismatch is valid as a *major limitation needing analysis*, but not a fatal contradiction.
- **“The combination function for global/local memory is never defined.”** This is incorrect. While the main text could be clearer, Appendix E, Eq. 15 explicitly defines additive composition of global and local retrieval outputs.
- **“The 17× speedup is invalid because the comparison uses a deliberately unfair baseline.”** The baseline choice does not appear unfair in the specific sense claimed. The paper’s point is that the most accurate Titans settings are slow, and TNT reduces time-to-quality substantially. Also, per the review policy, asymmetry that favors the baseline is not grounds for criticism. The more reasonable version of this point—that the headline framing should better distinguish time-to-quality against Titans from absolute throughput against optimized Transformers—has been retained in weakened form.
- **“The paper lacks a control evaluating Stage 1 at small chunks without Stage 2.”** The paper already establishes chunk-size sensitivity in Figure 2 and frames Stage 2 as the remedy. While a more direct control inside the TNT pipeline would still be useful, the harsher version of this criticism overlooks evidence the paper already provides.
- **“Missing comparison to other related methods.”** Omitted under instruction not to flag missing related works or unverified external baselines.
- **“Train-short-infer-long is not new.”** As stated, this is too generic and external-comparison-dependent. The relevant question here is whether this paper’s specific mechanism for non-linear deep memory modules is novel and effective; that concern is better captured by the retained points on overclaiming generality and limited validation scope.
- **Formatting/parser artifacts and under-rendered tables/equations.** Ignored; these are extraction issues, not paper issues.

## Novel Insights
The most important synthesis across the reviews is that the paper is strongest when interpreted not as a universal scaling breakthrough, but as a concrete architectural recipe for making *non-linear deep memory modules trainable at useful throughput* by redistributing modeling burden across resolutions. In that light, the reset mechanism is not merely a truncation trick: it acts as an explicit assignment of short-range structure to local memories and long-range continuity to a separate global path. The main unresolved scientific question is therefore not whether TNT “works” at 150M—it does—but whether this division of labor remains effective as width grows and as tasks demand precise retrieval across multiple reset boundaries. That is the key axis on which the paper’s long-term significance will stand or fall.

## Suggestions
- **Tighten the claims.** Rephrase “general training paradigm” and “removes a critical scalability barrier” to better match the current evidence, which is strong for Titans at 150M but not yet broadly validated across architectures/scales.
- **Move the exact retrieval composition rule into the main text.** Explicitly state the additive global+local retrieval formula in Section 4, not only in the appendix.
- **Add a focused shard-boundary analysis.** Measure performance as relevant information crosses reset boundaries, and vary \(S_L\) to show how much global memory compensates for local truncation.
- **Quantify Q-K projection cost.** Report activation/state memory and runtime overhead for the \(d \times d\) projection matrix, especially with multiple local memories.
- **Strengthen Stage 2 evidence.** Show sensitivity to fine-tuning steps, learning rate, and chunk-size transition, and clarify exactly which parameters are trained in the second stage.
- **Validate beyond Titans if possible.** Even a smaller-scale experiment on one additional deep memory model would substantially strengthen the paper’s central positioning.

---

## dIOYpj9K8P

- GT: Accept (Poster) (avg 6.7)
- Predicted: N/A (6.6/10)
- Match: N/A

### Final Review

## Summary
This paper proposes MGA, a two-stage corpus reformulation pipeline that expands existing web text via adaptive genre-audience conditioning, producing a 770B-token synthetic corpus from 195B tokens of FineWeb-Edu. The paper’s central empirical claim is that this reformulated data is more useful than naive repetition or upsampling under data-constrained pretraining, and it supports this with experiments across several model sizes plus analyses of prompt strictness, mixture effects with other synthetic corpora, and validation-loss behavior.

## Strengths
- **The paper studies a practically important but under-analyzed regime: data-constrained scaling under repetition, and does so with targeted comparisons rather than only reporting end metrics.** In Section 4.2 / Table 8 / Figure 3, the authors explicitly construct “entire set repetition” and “subset repetition” scenarios, which is more informative than simply showing gains on a fixed recipe.
- **The prompt-engineering ablation is unusually actionable and reveals a nontrivial tradeoff between fidelity and diversity.** Section 4.3.2 and Table 3 show that overly strict reformulation behaves differently from balanced reformulation, while relaxed reformulation collapses badly. This gives concrete evidence that the usefulness of synthetic reformulation depends on controlling the variance/invariance balance, not merely on generating more text.
- **The paper contains a genuinely interesting complementarity result with another synthetic-data strategy.** Section 4.3.1 shows that MGA and Nemotron-style synthetic data are not redundant: combining them outperforms either alone. That is a useful insight for practitioners designing mixtures of synthetic corpora.
- **The framework is more systematic than simple paraphrasing.** The adaptive GA-pair generation plus controlled reformulation is a specific design choice aimed at structured diversity, and the appendix includes additional measurements such as RefSim / heterogeneity and one-pass-for-many diversity statistics (Tables 5–6), which strengthens the claim that the method is trying to engineer diversity deliberately rather than by ad hoc rewriting.
- **The paper is candid about an important limitation that many synthetic-data papers downplay: MGA is not sufficient as a pure replacement for source data.** Appendix D.2 / Table 12 shows that “MGA-Only” underperforms the mixed-data setting, which, while a weakness for the overall framing, is also useful empirical evidence about where the method actually helps.

## Weaknesses

###: Fatal

### Major:
- **The main scaling comparisons do not cleanly isolate whether gains come from reformulation quality or simply from reducing harmful repetition.**  
  This is the central methodological issue. In Table 8, the “Baseline” for the 500B-budget experiment is `50 × 10`, while “MGA Expansion” is `50 × 2 + 200 × 2`; similarly, in the 700B-budget setting, MGA changes both the corpus composition and the repetition schedule. Since the paper’s own motivation is that repetition is harmful, these designs do show MGA is useful as a practical antidote to repetition, but they do **not** by themselves establish that reformulated data has superior scaling properties independent of the reduced repetition burden. The current evidence supports “MGA helps in repetition-limited settings” more strongly than the broader claim that the reformulation itself is the source of the scaling advantage.
- **The paper overstates MGA as a general solution to data scarcity, whereas its own results show it works best as a mixture component rather than a standalone substitute for real data.**  
  The abstract and introduction use language like “overcome this critical bottleneck,” “provides a reliable pathway to substantially augment training datasets,” and “alleviating repetition bottlenecks and enabling more efficient scaling.” But Appendix D.2 / Table 12 shows that replacing source data with MGACorpus alone hurts average performance across all tested sizes (roughly −0.9 to −1.0 average points vs. MGA-Expansion). This does not invalidate the practical usefulness of MGA, but it materially narrows the paper’s contribution: the evidence supports MGA as an effective **augmentation/mixing strategy**, not as a drop-in scalable replacement for natural pretraining data.
- **The validation-loss degradation on held-out real data remains insufficiently resolved.**  
  Section 4.2 and Figure 6 acknowledge that MGA models often have worse validation loss on fineweb-edu-dedup and open-web-math despite better benchmark performance. Section 4.3.3 offers an interesting hypothesis—synthetic-trained models may prioritize more generalizable patterns over memorization—and Appendix D.4 investigates token-position anomalies. However, the analysis remains suggestive rather than conclusive. The paper does not provide a stronger distribution-shift analysis by domain or content type, nor a direct factuality/consistency evaluation that would rule out degradation in foundational language modeling quality. For a paper making strong claims about high-quality corpus augmentation, this unresolved mismatch between downstream gains and held-out likelihood is an important caveat.
- **Key claims about larger-scale behavior are only partially substantiated in the paper body.**  
  The paper repeatedly emphasizes widening gains with model size “up to 13B,” and Figure 3 is described in those terms, but the detailed benchmark table in the main text (Table 2) stops at 1.7B. The 7B/13B evidence appears mainly through training-dynamics plots and summarized deltas rather than the same level of benchmark detail given for smaller models. This weakens the force of the claimed N-scaling advantage, especially since the headline narrative leans heavily on larger-model benefits.

### Minor
- **The quality-control loop for synthetic data is heavily teacher-model mediated.**  
  In Section 3.2 and Table 1, the teacher LLM generates data, scores outputs, and the SLM is trained on examples with score ≥ 3. The paper does mention “human-in-the-loop cross-checking” with over 90% alignment, which partially addresses the concern, so this is not a fatal circularity claim. Still, the final notion of acceptable reformulation quality is largely inherited from the teacher model rather than from an external factual-consistency metric or broader human evaluation.
- **Some core design choices are plausible but under-ablated.**  
  For example, the paper argues that adaptive GA-pair generation is important, but there is no direct ablation against a simpler random or fixed genre-audience baseline. Likewise, “one-pass-for-many” is compared to one-pass-for-one in Appendix B, but not against stronger alternative diversity-inducing sampling strategies. These omissions matter because they leave uncertainty about which component is actually responsible for the gains.
- **The compute cost of generating MGACorpus is substantial.**  
  Appendix B reports large H100 usage for the two synthesis stages. This does not negate the paper’s value, especially since the point is to trade generation compute for improved pretraining data, but it does affect practical accessibility and would benefit from a clearer cost-benefit framing.

### Trivial

## Nice-to-Haves
- A compute-matched comparison that accounts for synthesis cost, not only pretraining token budget, would help clarify when MGA is preferable to longer training on repeated real data.
- A direct factual-consistency evaluation of reformulated text would strengthen the “Limited Consistency” story beyond teacher-model scoring.
- A mixing-ratio ablation would help identify where MGA’s benefits saturate and where synthetic degradation begins.
- Clearer benchmark reporting for the 7B and 13B models would better support the claimed widening N-scaling gains.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Unfair comparison because MGA uses 4× more tokens than baselines for the same compute budget.”**  
  Removed as factually incorrect. The main comparisons in Table 2 and the scaling setups in Table 8 are framed around fixed **training token budgets**; MGA changes the composition and uniqueness of tokens, not the total pretraining token count in those experiments.
- **Criticism that the paper is not reproducible because cited models/datasets/tools may not exist or are unavailable.**  
  Removed per instruction; the paper cites these resources and explicitly includes release plans and links.
- **Pure complaint that comparisons to stronger external models such as SmolLM2 are unfair because those models use more compute.**  
  Weakened/removed as a core weakness because the paper itself explicitly notes these models are “for reference only” and highlights fairer same-budget comparisons within the SmolLM setting.
- **Generic “the paper should include instruction-tuning/RLHF/user-study/theory” requests.**  
  Removed as scope creep; this is a pretraining-data paper and should primarily be judged on whether it establishes value for pretraining.
- **Pure formatting complaints about garbled figures/text extraction.**  
  Removed because the provided text has parser artifacts and this is not a paper issue.

## Novel Insights
The most important synthesized takeaway is that the paper is strongest when interpreted not as a replacement-data paper, but as a **mixture design paper for data-constrained pretraining**. The evidence consistently points to MGA being useful because it creates structured diversity that is especially valuable when repetition would otherwise dominate, and because that diversity appears complementary to more task-aligned synthetic corpora. At the same time, the paper’s own “MGA-Only” and validation-loss results suggest a real boundary: reformulation helps most as a diversity-enhancing ingredient, not as a self-sufficient substitute for natural data. Framing the contribution this way would make the paper both more accurate and more compelling.

## Suggestions
- Reframe the contribution more precisely: position MGA as an effective **augmentation/mixing strategy under repetition-limited regimes**, rather than as a general standalone solution to data scarcity.
- Add one or two cleaner controls that isolate reformulation quality from repetition reduction—for example, a baseline with similarly reduced repetition but without MGA-style reformulation, or a stronger unique-token control.
- Strengthen the validation-loss discussion with a more systematic distribution-shift analysis on held-out real data, rather than mostly anecdotal late-position case studies.
- Include a direct ablation of adaptive GA-pair generation against random/fixed GA choices to verify that source-conditioned genre-audience selection is actually contributing.
- Report fuller 7B/13B benchmark results in the main paper or appendix tables to substantiate the claimed widening gains with scale.
- Add a concise compute-versus-gain discussion so readers can judge when the synthesis overhead is worth paying relative to simpler alternatives.

---

## 4nMPx7BHIg

- GT: Reject (avg 1.5)
- Predicted: N/A (2.4/10)
- Match: N/A

### Final Review

## Summary
This paper proposes “Neurovectors,” a non-backpropagation method for tabular prediction in which each training instance is stored as a set of exact feature-value tokens, and inference retrieves candidate stored instances by token overlap, with an “energy” score used to break ties. The paper reports competitive results on a few small datasets and argues for very low computational cost via dictionary-based lookup and partial storage of only mispredicted examples.

The core idea is easy to understand and has some practical appeal as a lightweight retrieval-style learner. However, after checking the method against the paper text, the main concerns are substantive: the representation appears fundamentally brittle for continuous features, the evaluation is too limited and unstable for the breadth of the claims, and the efficiency analysis is not rigorous enough to support the strong computational conclusions.

## Strengths
- **The method is unusually transparent at the prediction level.** The model prediction is explicitly based on retrieving stored training instances whose `<feature_name, value>` tokens overlap with the query, and selecting among them using overlap count plus a simple historical reliability metric. This makes the decision process directly inspectable in a way that most tabular neural baselines are not.
- **The proposal does make a concrete algorithmic departure from standard backprop-trained tabular models.** The training rule in Section 3.4 only instantiates a new neurovector on failures and otherwise updates per-instance usage/success statistics; that is a specific retrieval/memory mechanism rather than just another MLP variant.
- **The paper surfaces an important practical tradeoff: lightweight prediction versus peak accuracy.** Even though the current evidence is not yet sufficient to validate the strong efficiency claims, the paper does articulate and experimentally explore the idea that a simpler memory-based tabular learner may be attractive when training/inference cost matters.
- **The model stores only mispredicted examples rather than all examples by construction.** This is a specific and potentially useful design choice, since Section 3.4 explicitly avoids creating neurovectors for correctly predicted samples, which could reduce memory relative to a full instance store in favorable cases.

## Weaknesses

###: Fatal
- **The paper’s central generalization mechanism is not convincing for continuous tabular data, and this directly threatens the core empirical claims.**  
  The method defines tokens as exact concatenations of feature name and feature value (Eq. 3: `τ_{j,l} = (name_feature_l + v_{j,l})`), and candidate retrieval depends on exact token matches. For datasets used in the paper such as Breast Cancer, Absenteeism, and Red Wine, many features are numerical/continuous. The paper does not describe any binning, rounding, tolerance rule, similarity kernel, or embedding that would let nearby numeric values match. Without such a mechanism, unseen test values will often share few or no exact tokens with stored neurovectors, making overlap-based retrieval brittle and potentially ineffective. This is not a side issue: exact matching is the backbone of the proposed algorithm. The paper repeatedly claims the method works “without any prior preprocessing,” which makes the absence of a numeric handling mechanism more consequential, not less.
- **The regression formulation is mathematically degenerate as written.**  
  Section 3.3 states: “for regression problems, a prediction is correct if the predicted value is identical to the current value, and therefore the MAE is 0.” Since the method predicts a stored target value from a selected neurovector, exact equality for continuous targets is generally measure-zero except in repeated labels. Yet Eq. (9) still uses `success(NV)` in the numerator, where success is based on this exact-match notion. This makes the regression “energy” definition very weakly motivated and likely uninformative in realistic continuous regression settings. Because one of the paper’s three main benchmark tasks is regression, this is a serious technical issue.

### Major:
- **The method is framed in neural/energy-based terms that are not supported by the actual algorithm.**  
  The paper presents Neurovectors as “a new neural network approach,” speaks of “energy propagation,” and positions the method against backprop-based neural learning. But the actual method is a retrieval-and-counting procedure over stored exact feature-value tokens with per-instance success/use statistics. There are no learned weights, no hidden representations, no propagation dynamics, and no energy minimization in the usual sense. The “energy” in Eqs. (8–9) is effectively a heuristic confidence score derived from success history. This mismatch matters because it inflates the apparent novelty and can mislead readers about what kind of contribution this is.
- **The empirical support is too narrow for the paper’s claims of effectiveness on tabular learning.**  
  The main experiments use only three small datasets with a single 60/20/20 split. For a paper making broad claims about a new tabular learning paradigm, this is not enough. The later comparison on Adult/Bank/Kick also does not rescue this, because it reveals much weaker performance and severe instability rather than robust competitiveness.
- **Table 4 exposes extreme instability that is not adequately addressed.**  
  The reported standard deviations for Neurovectors on Adult/Bank/Kick are 0.3096, 0.2881, and 0.1817, while other methods are around 1e-3. Even allowing for a typo possibility, the paper does not explain these values. If they are correct, they imply severe instability; if not, the results table is unreliable. In either case, this substantially weakens the credibility of the large-dataset evidence.
- **The efficiency claims are overstated relative to the evidence actually provided.**  
  Table 3 is based on hand-derived FLOP estimates rather than direct, unified runtime or memory measurements on the same hardware/software stack. The estimates for Neurovectors rely on simplified counts of hashing, dictionary search, and creation, while baseline costs are drawn from coarse formulas and assumptions. The paper also scales costs for some datasets simply by dataset-size ratios. This is not strong enough to support claims such as “four orders of magnitude less than tree-based ensemble methods and even six orders of magnitude less than neural networks.” At most, the paper suggests the method may be lightweight; it does not rigorously establish the magnitude of the claimed efficiency advantage.
- **The baseline setup is not strong enough to justify the comparative claims.**  
  The paper compares against RF, Gradient Boosting, SVC, and a simple 3-layer MLP with largely fixed configurations, but omits stronger standard tabular baselines from the main experiments. Since the paper’s own related work emphasizes modern tabular methods and strong boosting-based baselines, the main empirical section does not support claims of broad competitiveness against the field’s most relevant methods.

### Minor
- **There is no ablation establishing whether “energy” contributes meaningfully beyond plain overlap count.**  
  Since the prediction rule first maximizes `count(NV)` and only then uses energy as a tie-breaker, an ablation removing the energy term is needed to show whether this component is actually important.
- **Memory behavior is asserted rather than demonstrated.**  
  Section 3.5 argues that neurovector growth is sublinear because fewer new neurovectors are created over time, but this is not empirically characterized. Since storage is central to both scalability and efficiency, the paper should report neurovector growth curves and memory footprint.
- **The performance/efficiency tradeoff is not analyzed carefully enough on larger datasets.**  
  Table 4 shows Neurovectors trailing strong baselines by a noticeable margin on Adult/Bank/Kick. The paper argues this is acceptable because training times are lower, but that tradeoff is not quantified in a way that would let readers judge whether the accuracy loss is worthwhile in realistic applications.
- **Clarity suffers in several technical definitions.**  
  In particular, the candidate set definition in Eq. (4) and the discussion around count/energy are imprecise enough that the exact implementation behavior is harder to infer than it should be. This compounds the difficulty of understanding how the method behaves when exact token matches are sparse.

### Trivial
- None.

## Nice-to-Haves
- Add a simple numeric-feature handling mechanism (e.g., binning, tolerance windows, nearest-bin matching, or learned continuous similarity) and evaluate how sensitive results are to this choice.
- Report wall-clock training/inference time, memory footprint, and candidate-set sizes per query, rather than relying primarily on FLOP estimates.
- Include an ablation for: (i) overlap-only retrieval, (ii) overlap + energy tie-breaking, and (iii) storing all examples vs. storing only failure cases.
- Show robustness analyses under small perturbations/noise to continuous features.
- Use repeated splits or cross-validation with significance testing for the small-dataset results.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The related work omits foundational instance-based learning / k-NN / case-based reasoning.”**  
  This may be a fair scholarly positioning concern in general, but per instruction I am not including missing-related-work criticisms.
- **“The GitHub/code is anonymized / not independently reproducible / lacks environment specs.”**  
  Removed as a reproducibility nitpick and because the paper does provide a code link and public datasets.
- **“The use of Python dictionaries gives O(1) lookup, therefore inference is computationally efficient.”**  
  Removed as a strength because this is too generic and not sufficiently validated by the paper’s actual end-to-end measurements.
- **“Zero preprocessing pipeline” as an unqualified strength.**  
  Removed in strong form because the paper’s no-preprocessing claim is entangled with the exact-match brittleness problem on continuous features.
- **“Unfair comparison because the asymmetry favors the authors.”**  
  I kept only the valid part: the baselines are not strong enough for the claims made. I did not retain arguments that merely object to asymmetry in a way that could actually favor the baselines.

## Novel Insights
The most important synthesis is that the paper’s two headline advantages—general tabular applicability and extreme efficiency—are coupled to the same exact-match design choice, and that choice is also the source of the paper’s biggest weakness. Exact tokenization of raw feature values may indeed make the implementation simple and potentially cheap, but unless the model introduces a principled notion of similarity for numeric features, the retrieval mechanism collapses precisely on the kinds of real-valued tabular problems the paper evaluates. This means the paper is not just missing additional experiments; its current formulation leaves unresolved whether Neurovectors are a generally useful learner or only a brittle exact-memory system that works when values repeat enough.

## Suggestions
- Add a principled treatment for continuous features and targets. This is the highest-priority fix. At minimum, define binning/rounding/tolerance rules explicitly and show sensitivity analyses.
- Redefine regression energy so it depends on continuous error directly, rather than exact target equality.
- Reframe the contribution more accurately as a retrieval-/memory-based tabular learner unless a genuine neural or energy-based mechanism is introduced.
- Strengthen the evaluation with repeated trials on standardized tabular benchmark suites and report means/standard deviations consistently.
- Replace or complement FLOP estimates with direct runtime and memory profiling on the same hardware for all methods.
- Add ablations isolating the contributions of token overlap, energy tie-breaking, and failure-only storage.

---

## 6RQsAQEUib

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (4.4/10)
- Match: N/A

### Final Review

## Summary
This paper proposes GHPO, a difficulty-aware RLVR framework that detects when GRPO is likely to receive all-zero rewards on a prompt and then injects partial ground-truth solution traces as hints, with the goal of blending exploration on solvable examples and guided learning on overly difficult ones. Empirically, the method shows consistent gains over GRPO and simple curriculum variants on math-focused benchmarks, and the training-dynamics plots suggest improved optimization stability.

## Strengths
- **Targets a real failure mode of GRPO with a concrete mechanism tied to observed reward sparsity.** Section 2.3 clearly identifies the all-zero group-reward regime in GRPO, where “\(\hat A_{i,t}=0\) for all trajectories associated with that query,” and GHPO is explicitly designed to intervene only in this regime by refining the prompt with hints.
- **The key idea is practically appealing and more adaptive than static curriculum heuristics.** Rather than pre-partitioning data by difficulty, GHPO uses online reward outcomes to decide whether to keep the original prompt or add guidance, and further adapts hint strength through a staged schedule \(\omega \in \{0.25, 0.5, 0.75\}\) (Appendix B.3).
- **The empirical gains are consistent across multiple benchmarks and across two starting models from the same family.** On the mixed dataset, GHPO improves average score over GRPO from 0.409 to 0.442 for Qwen2.5-Base-7B and from 0.4728 to 0.5076 for Qwen2.5-Math-7B (Table 2), with especially noticeable improvements on harder evaluations such as AIME24.
- **The paper goes beyond final accuracy and examines training behavior.** The inclusion of format reward, accuracy reward, response length, gradient norm, and the fraction of examples deemed difficult provides some insight into how the method changes optimization dynamics rather than only reporting endpoint results.
- **The paper surfaces an important conceptual point for RLVR on smaller models:** when a large portion of data lies beyond current model capability, pure on-policy updates can become uninformative. Even if the current formulation needs sharpening, this is a useful and timely observation.

## Weaknesses

###: Fatal
- **The core optimization objective is not technically sound as written when guidance changes the prompt.** In Eq. (1)-(2), the paper defines the ratio
  \[
  r_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}\mid q^*, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q, o_{i,<t})},
  \]
  where \(q^*=q+\omega\cdot h_{f,q}\) for difficult samples, but the trajectories are described as being sampled first from the original query \(q\): “GHPO first samples a group of \(G\) individual responses … Unlike GRPO, these group rewards are not directly used for advantage estimation. Instead, the difficulty detection module analyzes the sparsity … Based on this analysis, the corresponding prompt is refined…”. This means the numerator and denominator are conditioned on different inputs, and the denominator is not the behavior policy that generated the trajectory under the refined prompt. As written, this breaks the PPO/GRPO-style interpretation of the clipped ratio and leaves the central “unified RL objective” ill-defined. This is not a cosmetic issue: it affects the validity of the method’s main algorithmic claim.

### Major:
- **The experimental design does not cleanly isolate whether the gains come from RL or simply from conditional supervised guidance.** The main mechanism is adding partial ground-truth solution traces to hard examples. The baselines include GRPO, GRPO+curriculum, and curriculum with fixed hints, but there is no direct comparison to a supervised or hybrid imitation baseline built from the same dynamically hint-augmented data. Without such a control, it remains unclear whether the reported gains are due to the RL component, the adaptive hinting itself, or just exposure to partial solutions.
- **The component ablation story is too weak for a method with several moving parts.** GHPO includes at least three substantive choices: all-zero-reward difficulty detection, multi-stage hint scheduling, and a cold-start phase. The current experiments compare full GHPO to a few external baselines, but do not isolate the contributions of these components. As a result, the paper does not establish which design decision is actually responsible for the gain.
- **The claim of improved efficiency is not well substantiated.** The paper repeatedly frames GHPO as “stable and efficient” and “data-efficient,” but it does not report training cost, throughput, token usage, or convergence-vs-compute comparisons. This matters because adding hints lengthens prompts and may increase rollout cost. Final accuracy improvements alone do not verify efficiency.
- **The evaluation is narrow relative to the breadth of the paper’s framing.** The paper presents GHPO as a general RLVR framework for “complex reasoning tasks,” but all experiments are in mathematics-style settings with available step-by-step solutions. GPQA-Diamond appears only as an evaluation benchmark, not a training domain. The results therefore support usefulness for math RLVR with solution traces, but not the broader generality implied in the introduction and conclusion.
- **The method depends on access to partial ground-truth solution traces, which materially limits scope.** The paper does acknowledge that such traces are “often available for most mathematics data,” but this is a real constraint: the method is most natural in domains where verified intermediate solutions exist, and the paper does not show how it extends beyond that setting.

### Minor
- **The difficulty detector is very coarse.** A sample is treated as difficult only when all \(G\) sampled responses receive zero reward. This binary rule may miss cases where a problem is still largely too hard but happens to produce one lucky success, and the paper does not study alternatives based on success rate or reward statistics within the group.
- **The hint extraction strategy is under-analyzed and potentially brittle.** Appendix B.3 uses a fixed character-level schedule for 25/50/75% of the solution trace. In math reasoning, arbitrary character truncation can cut equations or logical steps mid-structure. The paper gives an illustrative example but no systematic analysis of whether this representation choice matters.
- **The paper’s treatment of “Assumption 1” is awkwardly framed.** The assumption effectively states the paper’s central hoped-for outcome—that training with partial trace guidance on failing problems improves OOD reward relative to training without such guidance—and then says this is demonstrated experimentally. This reads more like a motivating hypothesis than an assumption supporting a derivation.
- **The stability analysis is suggestive but not conclusive.** Smaller gradient norms in Figure 4 may indicate smoother optimization, but on their own they do not rule out weaker updates or a stronger supervised bias. The paper’s interpretation is plausible, but somewhat overconfident without complementary analysis.

### Trivial
- None.

## Nice-to-Haves
- Add a corrected formulation of the guided update: e.g., explicitly separate pure RL updates on unmodified prompts from a supervised/imitation loss on refined prompts, or sample trajectories under the refined prompt if a ratio-based objective is to be retained.
- Include a direct supervised/hybrid baseline using the same dynamically generated hint-augmented examples, to isolate the value of the RL component.
- Report compute-normalized results: GPU-hours, tokens processed, average prompt/response lengths, and accuracy versus training steps or wall-clock time.
- Provide ablations for cold-start, hint schedule, and the all-zero detection rule.
- Show the empirical distribution of hint ratios over training and whether examples transition from heavily guided to unguided as the policy improves.
- Clarify implementation details of the multi-stage guidance logic, especially whether difficulty state is tracked only within a step or across training.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **“Missing comparisons to all other SOTA RL methods (e.g., PPO/DAPO/VAPO/LUFFY) is a core weakness.”**  
  The paper already compares to the most directly relevant base method (GRPO) and curriculum variants, and the contribution is specifically framed around modifying on-policy GRPO-style RLVR under reward sparsity. While broader comparisons would strengthen positioning, absence of every discussed RL method is not by itself a decisive flaw.

- **“The datasets are too small for 7B RL training, therefore the results are not meaningful.”**  
  The paper uses 8,890 and 18,300 curated math problems with full solutions; whether this is optimal is debatable, but the criticism as stated overreaches. The more defensible concern is lack of stronger variance analysis and compute-efficiency evidence, not that the study is invalid purely due to dataset size.

- **“Standard GRPO engineering workarounds already solve reward sparsity, so the premise is overstated.”**  
  The paper’s point about all-zero reward groups yielding zero advantage is correct from its own formulation in Section 2.3. Even if practical systems skip such updates, that does not negate the underlying sparse-signal problem.

- **Open-source/reproducibility availability comments.**  
  The paper cites a code release, so any skepticism about existence or availability should be disregarded under the review instructions.

## Novel Insights
The most important synthesized insight is that the paper is strongest as a practical curriculum-through-guidance idea, but weakest where it tries to cast that idea as a single principled PPO/GRPO-style objective. The empirical results support the intuition that adaptive partial-solution guidance helps rescue zero-signal regions of RLVR training, especially for smaller models on hard math data. However, the current formulation appears to conflate two different regimes—policy optimization on the original task and conditional learning on a modified task with privileged information. That distinction is the central issue to resolve: if made explicit, the work could become a clearer and more compelling hybrid RL/imitation paper.

## Suggestions
- Reformulate the method so that guided examples are optimized with a clearly justified objective, rather than a clipped ratio across different prompt contexts.
- Add a direct control baseline that uses the same adaptive hint generation but trains without the RL objective, to determine whether RL is actually essential.
- Run targeted ablations on: cold-start on/off, fixed vs adaptive hint schedule, and alternative difficulty thresholds beyond the all-zero rule.
- Report compute and sample-efficiency metrics, not just final benchmark averages.
- Temper the broader claims: the current evidence supports math RLVR with available solution traces, not yet a generally validated framework for all complex reasoning domains.

---

## Ry8jLSYIUG

- GT: Reject (avg 5.3)
- Predicted: N/A (7.4/10)
- Match: N/A

### Final Review

## Summary
This paper studies image watermarking capacity through a geometric lens: it derives bounds on how many bits can be embedded under PSNR constraints and under certain linearized robustness constraints, then asks whether current deep watermarking systems are anywhere near those limits. The core empirical message is that they are not: in a simplified gray-image setting, Video Seal fails well below what simple linear and handcrafted constructions can achieve, and a scaled-up variant, Chunky Seal, pushes practical robust capacity from 256 to 1024 bits with broadly similar robustness/quality.

## Strengths
- **The paper contributes a genuinely new way to think about watermarking capacity.** Rather than relying on classical Gaussian/noise-channel formulations, it models watermarking as counting feasible discrete images in intersections of pixel-space cubes and PSNR balls, then extends this to certain linear transforms. This reframing is specific and consequential, not just a repackaging of standard capacity arguments.
- **The simplified diagnostic in Section 3 is unusually strong and revealing.** By stripping the task down to a single gray image with only an MSE/PSNR constraint, the paper cleanly shows that Video Seal fails at 1024 bits while a linear embedder/decoder reaches 2048 bits at 100% bit accuracy and a handcrafted construction reaches hundreds of thousands of bits under the same basic distortion metric. This is compelling evidence that current neural architectures are not even close to saturating the simplest version of the problem.
- **The paper does not oversell the robustness theory as fully rigorous.** It explicitly states that Bounds 10–12 are heuristic and even gives counterexamples where they over- and under-estimate the true capacity (Figures 8 and 9), while also providing a much more conservative lower bound (Bound 13). That honesty strengthens the credibility of the PSNR-only part and clarifies where the uncertainty lies.
- **Chunky Seal is a concrete empirical proof that the current practical frontier can be moved.** Table 3 shows 1024-bit embedding with PSNR 45.32 dB and overall bit accuracy 99.15%, compared with Video Seal’s 256 bits at 44.42 dB and 99.31%. Even if this is not yet a practical design, it is meaningful evidence that the perceived ceiling around a few hundred bits is not fundamental.
- **The appendices are unusually substantive.** The construction of exact counts, numerical approximations, linearized transforms, and conservative robustness bounds provides a toolkit that future work can build on, rather than just a high-level argument.

## Weaknesses

###: Fatal
- None.

### Major:
- **The strongest theoretical claims are much firmer for the PSNR-only setting than for the robustness setting.** This matters because the paper’s headline narrative is about practical robust watermarking. In Section 2.5 and Appendix G, the paper explicitly acknowledges that the main robustness bounds are heuristic: “these heuristic bounds under-approximate and over-approximate the true capacity” and “Bounds 10 to 12 are heuristics and are near-exact only for axis-aligned transformations.” The conservative alternative (Bound 13) is admitted to be “extremely conservative and unrealistic.” As a result, the claim that robust capacities should be around 0.5 bpp is suggestive rather than well-established; the paper more convincingly proves underutilization in the no-robustness PSNR-only case than in the realistic robust case.
- **The argument that data distribution has only negligible effect is not convincingly established.** Section 2.6 estimates the number of nearby covers using VQ-VAE/VQGAN-style codebook counts and concludes the penalty is only about 0.05 bpp. This is an interesting sanity check, but it is a coarse counting argument, not a geometric characterization of natural-image neighborhoods under watermarking constraints. Given how central the conclusion is—“the data distribution has only a negligible effect on watermarking capacity”—the evidence here feels too indirect relative to the strength of the claim.
- **Chunky Seal demonstrates feasibility, but not a principled path toward the large theory-practice gap.** The model is scaled dramatically: the paper reports a 90× larger embedder and 23× larger extractor than Video Seal. This is useful as a proof of possibility, but it does not isolate what actually enabled the gain (more channels, larger U-Net, larger extractor, 3-channel embedding, reduced stride, gradient clipping, etc.). Since the conclusion emphasizes architectural limitations and future architectural innovation, the paper would be stronger if it disentangled which changes matter most rather than presenting a single large scaled system.
- **The practical cost/quality trade-off of Chunky Seal is under-analyzed in the main narrative.** Table 3 shows broadly similar PSNR/SSIM/MS-SSIM, but LPIPS worsens from 0.0019 to 0.0085, and the parameter count increase is enormous. The paper does acknowledge in Section 5 that size and latency are limitations, but the main text’s phrasing that Chunky Seal preserves quality and robustness “comparable” to Video Seal somewhat glosses over the fact that this is achieved with a very large model and nontrivial perceptual degradation on LPIPS.

### Minor
- **The empirical bridge from the robustness theory to actual model behavior is incomplete.** The paper derives singular-value-based reduction heuristics for crop/rotation/LinJPEG, but does not test whether trained models’ degradation patterns track those predictions. A direct comparison between predicted capacity reduction and observed bit-accuracy degradation would substantially strengthen the “theory explains practice” story.
- **The key Section 3 diagnosis is based on an intentionally artificial task, and the paper occasionally draws broader conclusions from it than the evidence fully supports.** The gray-image experiment is excellent as a lower-complexity sanity check, and it strongly shows Video Seal is not capacity-optimal in that setting. However, concluding from this alone that real-world complexity “cannot explain” the gap is stronger than what is strictly proven, especially given that the robustness/perceptual/data-distribution analyses are less definitive than the PSNR-only analysis.
- **The evaluation of Chunky Seal is good but still somewhat narrow relative to the paper’s ambition.** The main table compares primarily to Video Seal, with broader comparisons moved to the appendix. Given the central claim that the field is far from capacity limits, stronger main-text comparisons at matched or at least better contextualized payloads would help.
- **The paper stops at 1024 bits, far below its own theoretical gap.** This is understandable, and the paper does not claim to close the gap, but given the headline framing (“We can hide more bits”), some scaling curve beyond 1024 bits would help establish whether performance keeps improving smoothly or saturates quickly.

### Trivial
- **Some claims would benefit from slightly more careful wording.** For example, the statement that Chunky Seal achieves gains “without hyperparameter tuning” is true as written, but not especially informative in a comparison against a baseline originally optimized for different capacity/architecture settings.

## Nice-to-Haves
- An ablation of Chunky Seal isolating the contributions of larger embedder, larger extractor, 3-channel embedding, reduced stride, and gradient clipping.
- A direct empirical calibration of LinJPEG or the singular-value heuristics against observed degradation under actual JPEG/composed transforms.
- A plot of capacity scaling curves beyond 1024 bits, even if performance degrades, to show whether the empirical frontier appears smooth or sharply saturating.
- Unified Pareto plots showing theoretical PSNR-only bounds, heuristic robust bounds, Video Seal, Chunky Seal, and simple baselines in the same capacity-quality/robustness plane.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The comparison is unfair because Video Seal was not tuned at 1024 bits.”** The paper does retrain Video Seal for 128/256/512/1024 bits in the simplified setup and reports sweeps over learning rates and loss weights in Section 3.1. For the core diagnostic claim—Video Seal underperforms even in a simplified setting—the point is already reasonably addressed.
- **“The paper is flawed because cited models/tools may not exist / are unavailable / cannot be verified.”** Removed by policy.
- **“The gray-image diagnostic is conceptually flawed because CNNs are bad for i.i.d. featureless canvases, so the failure is meaningless.”** This overstates the objection. The paper’s purpose there is explicitly to test a simplified analyzable setting, not to claim gray images are the full application distribution. The fact that linear and handcrafted methods succeed under the same constraint still makes the experiment informative.
- **“Discrete lattice assumptions are invalid because real images are continuous-valued due to sensors, demosaicing, etc.”** The paper explicitly models valid digital images as quantized pixel arrays and studies capacity in that discrete space. For digital watermarking of stored images, this is a reasonable modeling choice, not a factual flaw.

## Novel Insights
The paper’s most interesting contribution is not just “capacity is higher than we thought,” but that the current bottleneck appears to be architectural utilization of available image degrees of freedom rather than merely robustness or quality constraints. The strongest evidence is the mismatch between Video Seal’s poor scaling with resolution in the gray-image setup and the success of trivial constructions that exploit pixel-space structure directly. A second notable insight is that the paper itself effectively splits the problem into two different scientific statuses: PSNR-only capacity is argued quite convincingly, while robust capacity remains much more heuristic. That distinction should shape how the community interprets the results: the paper convincingly refutes the idea that we are near saturation in simple settings, and plausibly suggests the same in realistic robust settings, but does not yet prove the latter nearly as tightly.

## Suggestions
- Tighten the central claim so that it clearly distinguishes **what is established** (large unused capacity under PSNR-only constraints; strong evidence of architectural underutilization) from **what is suggestive** (the size of the gap under realistic robustness constraints).
- Add an ablation study for Chunky Seal to identify which modifications drive the 4× gain.
- In the main text, explicitly discuss the LPIPS increase and the extreme model scaling as part of the capacity-quality-efficiency trade-off.
- Empirically compare the heuristic robustness predictions against actual degradation of Chunky Seal/Video Seal under the corresponding transforms.
- Soften or better support the claim that data distribution has only negligible effect; as written, Section 2.6 is better interpreted as a rough sanity check than a definitive resolution.

---


# Summary

Papers: 50 | Accuracy: N/A
