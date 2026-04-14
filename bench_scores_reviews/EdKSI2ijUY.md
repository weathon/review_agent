## Summary
LMRL-Gym is a benchmark and open-source research framework for evaluating reinforcement learning algorithms in multi-turn language model interactions. It comprises 8 tasks split between Interactive Dialogue tasks (20 Questions, Guess My City, Car Dealer) and RL Capability tests (Maze, Text-Nav, Wordle, Chess, Endgames), covering both offline and online RL settings. The benchmark is accompanied by implementations of BC, MC Returns, ILQL, and PPO, with the primary goal of enabling rapid iteration on RL algorithm development for LLMs.

---

## Strengths

- **Structured RL capability taxonomy:** The decomposition of tasks along five explicit capability axes (strategic decision-making, credit assignment, partial observability, trajectory stitching, complex language) in Figure 2 is a genuine methodological contribution that distinguishes this benchmark from existing NLP evaluations or text-game suites. Most prior benchmarks evaluate final task performance; this one deliberately exposes algorithmic bottlenecks.

- **Informative empirical contrast between task categories:** The finding that ILQL dominates the RL Capability tasks (especially Maze and Text-Nav) via trajectory stitching, yet is *outperformed by the simpler MC Returns on all three Interactive Dialogue tasks*, is a concrete and non-obvious finding that reveals a meaningful algorithmic gap. This points to a specific research problem: scalable TD-learning for open-vocabulary dialogue.

- **Language vs. symbolic performance gap:** The comparison of RL on the symbolic Maze vs. the language-based Maze—where online/offline Q-learning reaches optimality in the symbolic setting but degrades substantially under partial observability in language—precisely isolates language complexity as a bottleneck, not just RL difficulty in general.

- **Scope and accessibility commitment:** Supporting both offline and online RL, releasing datasets, simulators, code, and hyperparameters, and targeting GPT-2-scale models to ensure accessibility for resource-limited researchers is a substantive service to the community. The dataset sizes (up to 1M for Wordle, 625K for Chess) are large enough to support serious offline RL experimentation.

---

## Weaknesses

- **GPT-2 scale severely limits the conclusions.** All agent models are GPT-2 (up to 1.5B parameters). The paper acknowledges this briefly but frames it as a minor "limitation." In practice, this is a central constraint on every result: the relative ranking of algorithms, the difficulty of tasks, the behavior of ILQL vs. MC Returns, and the failure of Chess may all change substantially at 7B+ scale where emergent capabilities appear. A benchmark paper whose central claim is enabling *algorithmic* progress needs to demonstrate that its algorithmic rankings are at least qualitatively stable across model scale, even on one or two tasks. Without this, LMRL-Gym may be diagnosing properties of GPT-2-scale RL optimization rather than properties of algorithms in general.

- **No variance estimates in Table 2.** All results are single point estimates with no confidence intervals, standard deviations, or multi-seed statistics. Multi-turn RL training—especially PPO—is notoriously unstable. Without error bars, it is impossible to determine whether, e.g., the difference between ILQL (82.9) and MC Returns (87.1) on 20Qs is a meaningful algorithmic signal or run-to-run variance. For a paper whose stated value proposition is enabling *reliable* algorithm comparison, this is a significant methodological gap that undermines the core utility claim.

- **Chess task shows no meaningful RL improvement and is left undiagnosed.** Table 2 shows all methods score between 42.9 and 48.0 on Chess, tightly clustered around the dataset average of 50. No algorithm achieves a normalized score above 50, meaning no method exceeds the average offline trajectory. The paper briefly attributes PPO's slight edge to instability and calls for "better offline TD-based RL methods," but does not investigate *why* RL fails entirely on Chess at this scale—whether it is due to sparse rewards, long horizons, the combinatorially large move space, or the token-level action representation. An uninformative task that no method meaningfully solves undermines the diagnostic claim: it reveals a failure mode but does not illuminate it.

- **ILQL underperformance on dialogue tasks is headline-without-story.** The observation that ILQL underperforms MC Returns on all three dialogue tasks is one of the paper's key findings, yet the analysis is limited to one sentence: "it is harder to scale full TD-learning" on complex text tasks. There is no diagnostic of whether the failure stems from Q/V network instability, distributional shift, hyperparameter sensitivity, or the token-level Bellman backup being poorly conditioned on long open-vocabulary sequences. For a benchmark meant to drive algorithm development, this finding needs a tighter diagnosis so researchers know *what* to fix.

- **GPT-4 scoring exactly 0 on Chess and Endgames is suspicious.** The paper uses this result to conclude that "RL fine-tuning enables goal-directed behaviors that GPT-4 cannot achieve via prompting." However, a frontier model capable of legal chess play getting 0/100 almost certainly reflects a prompting or game-state formatting failure rather than a genuine inability. If the conclusion about RL's superiority over GPT-4 on capability tasks is partly an artifact of poor GPT-4 prompting for these specific game formats, the claim is misleading. The paper should investigate this before drawing strong conclusions.

- **Trajectory stitching claim is asserted but not empirically verified.** The paper designates all RL Capability tasks as testing trajectory stitching capability, but never demonstrates that the best-performing RL policy achieves returns *above any individual trajectory in the offline dataset*. Without showing that RL exceeds the offline dataset ceiling, the stitching property is an assumed design feature, not a demonstrated benchmark characteristic.

---

## Nice-to-Haves

- **Experiments at larger model scale (e.g., 7B).** Even a subset of tasks (e.g., Maze and 20Qs) with Llama-3-8B would substantially strengthen the claim that LMRL-Gym's algorithmic rankings are generalizable and relevant to modern practice.

- **Learning curves with multi-seed variance.** Plotting reward vs. training steps for PPO and ILQL across 3–5 seeds would simultaneously address the variance concern and provide diagnostic insight into PPO instability.

- **Trajectory stitching verification.** Adding a simple analysis showing that top-1% ILQL-generated trajectories exceed the return ceiling of any individual offline trajectory would concretely validate the stitching claim.

- **Additional modern baselines (e.g., Decision Transformer, filtered variants of DPO applied to multi-turn).** Decision Transformer is already cited (Chen et al., 2021a); including it as a baseline would strengthen the algorithmic coverage given it is specifically designed for offline sequence RL.

- **Dataset size ablations for smaller tasks.** Car Dealer (19K) and Maze (1.24K) are 1–2 orders of magnitude smaller than other tasks. A brief analysis of how algorithm rankings shift with data quantity would clarify whether the Car Dealer results reflect task difficulty or data scarcity.

- **Visualizations of successful vs. failed trajectories.** Side-by-side conversation logs where RL succeeds vs. where BC fails would help readers assess whether models are learning genuine multi-turn strategies or exploiting simulator artifacts.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Sim2Real gap" as a weakness (Positive Review, Weakness 1; Spark Finder #2).** The paper explicitly and prominently states: "our goal is *not* to utilize this approach to benchmark whether LLMs are *good at talking to humans*, but rather as a way to test RL algorithms." Criticizing the absence of human-evaluated deployment performance directly contradicts the stated scope of the benchmark. Removed.

- **"Reward function details in appendix" as a weakness (Positive Review, Weakness 3).** Placing reward function details in the appendix is standard practice in ML papers. The table in Section 4 provides summary statistics. Not a genuine weakness.

- **"Missing non-RL agentic baselines (ReAct, CoT, tool-use)" as a weakness (Positive Review, Weakness 4).** The benchmark is explicitly designed to evaluate *RL training algorithms*. Demanding comparisons to inference-time prompting methods is scope creep: a benchmark about X should not be faulted for not evaluating Y. Could be a nice-to-have for supplementary insight but not a weakness.

- **Token-level vs. turn-level MDP as a methodological flaw (Harsh Critic, Section 3).** Token-as-action is the standard formulation in the RL-for-LLMs literature (Snell et al., 2022a; Ziegler et al., 2020; Stiennon et al., 2020). Criticizing this as "the hardest possible formulation without justification" misrepresents the field's standard practice. Removed.

- **"Circular data generation invalidates results" (Harsh Critic, Section 4.1).** The paper is transparent that training and evaluation use the same GPT-2-based simulator family, and explicitly states this design prioritizes accessibility for RL algorithm comparison rather than human fidelity. The user study validating naturalness partially addresses this. The concern about reward hacking is partially addressed by automatic checks for 20Qs/Guess My City and naturalness verification for Car Dealer. While the concern is not baseless, framing it as an invalidating circularity misrepresents the paper's stated purpose. The concern about simulator strategic fidelity (not just linguistic naturalness) is retained as a nice-to-have.

- **"Contributions described narratively, not in bullet form" (Harsh Critic, Introduction).** Pure formatting nitpick. Removed.

---

## Novel Insights

The most penetrating observation across the three reviews—not made explicitly by any one of them—is that the ILQL vs. MC Returns reversal between task categories (ILQL wins on capability tasks, MC Returns wins on dialogue) may be a more fundamental signal than the paper currently treats it: it suggests that Bellman-based TD-learning, despite its theoretical advantages for stitching, encounters a qualitative break when the action space is open-vocabulary and the reward signal is episodic and sparse over long natural-language sequences. This is distinct from simple "scaling difficulty" and points toward a specific algorithmic challenge: how to regularize token-level Q/V functions when dialogue states are high-dimensional and partially unstructured. Combined with the symbolic-vs-language Maze gap, the benchmark is inadvertently revealing that language complexity itself—not just RL difficulty—is a first-class algorithmic bottleneck that existing offline RL methods are not designed to handle. This is a genuinely useful insight for algorithm designers that the paper surfaces empirically but does not synthesize clearly.

---

## Suggestions

1. **Report variance across multiple seeds for Table 2**, even on a subset of tasks. This is the single most important fix for the paper's core utility as a benchmarking tool—without it, observed algorithm differences cannot be trusted.

2. **Diagnose the Chess RL failure more explicitly.** Report the distribution of rewards achieved by each method (not just the mean), and investigate whether the issue is reward sparsity, long horizons, or the token-level move representation. If Chess is genuinely intractable for sub-2B parameter models, say so explicitly and propose a diagnostic protocol for when it becomes tractable.

3. **Verify or retract the GPT-4 Chess/Endgames score.** Ablate the GPT-4 prompting format for Chess (e.g., FEN notation vs. natural language description of board state) to confirm that a score of 0 reflects model capability rather than input formatting. The conclusion about RL superiority depends on this.

4. **Add a brief empirical verification of trajectory stitching.** Show the return distribution of the offline dataset and demonstrate that ILQL-trained policies produce trajectories with returns exceeding the offline ceiling on at least one task. This converts an assumed benchmark property into a demonstrated one.

5. **Provide a more detailed failure analysis of ILQL on dialogue tasks.** Report training curves, KL divergence from the BC initialization, and Q/V function loss curves to diagnose whether the failure is instability, distributional shift, or a fundamental representation issue. This would make the paper's key empirical finding actionable for the community.

---

**Overall assessment:** LMRL-Gym fills a genuine and important gap—there is no existing benchmark that provides multi-task, multi-turn RL evaluation with simulator infrastructure, offline datasets, and explicit algorithmic capability targeting. The task design is thoughtful and the open-source commitment is commendable. However, as currently presented, the empirical foundation is fragile: results are single-run on GPT-2-scale models, the Chess task reveals a failure mode that is not adequately diagnosed, and the paper's most interesting finding (the ILQL/MC Returns reversal) lacks a mechanistic explanation. The paper is a solid infrastructure contribution that falls short of being a definitive empirical study. Addressing the variance reporting, the Chess diagnosis, and the GPT-4 anomaly would substantially strengthen the paper's credibility as a reliable algorithmic testbed.