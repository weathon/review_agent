Now let me check the calibration papers mentioned by the human finder reviewer.Now I have enough context. Let me write the final consolidated review.

---

## Summary

This paper extends AgentScope, an existing multi-agent platform, with four enhancements to support very large-scale LLM-powered simulations: (1) an actor-based distributed mechanism with automatic workflow conversion via a `to_dist` function, (2) an environment abstraction supporting agent-environment interactions, (3) a configurable tool and automatic background generation pipeline for heterogeneous agent populations, and (4) a web-based Agent-Manager. The system is demonstrated exclusively through a "guess 2/3 of the average" game, scaled up to 1 million agents across 4 devices, with behavioral experiments varying LLM models, prompts, educational backgrounds, and group compositions.

---

## Strengths

- **Genuine engineering achievement at 1M-agent scale.** The paper demonstrates running 1 million LLM-powered agents (Llama3-8B, Prompt 1) in 12 minutes on 4 consumer-grade GPU nodes. The dummy-model orchestration comparison (Fig. 2b) — showing 40 seconds vs. 12 days (serial) and 8.6 hours (Python async) — is informative for quantifying scheduling overhead independent of inference time, and the `to_dist` single-function API for effortless workflow conversion is a practically thoughtful design.

- **Candid prior-knowledge experiment (Sec 4.6).** Changing the ratio from 1/2 to 51/100 produces dramatically different agent behavior despite equivalent game structure; adding a note that references the classic game then shifts behavior back toward equilibrium. This is one of the more honest and scientifically informative findings in the paper — with real implications for any simulation study relying on canonical tasks.

- **Behavioral sweep across 6 LLMs, 2 prompts, multi-round dynamics, demographic backgrounds, and mixed-LLM groups.** While the game is narrow, the breadth of agent configurations studied (Figs. 3–7) provides useful empirical data on LLM heterogeneity and prompt sensitivity within this setting.

---

## Weaknesses

### Fatal
None triggered.

### Major

- **Single-task evaluation across all experiments** — Every empirical finding derives from one canonical, single-number guessing game. The platform claims to support "various real-world scenarios," but no second simulation scenario is demonstrated. This prevents any assessment of whether the infrastructure (especially the environment abstraction, the background pipeline, or the workflow conversion) generalizes to richer tasks involving multi-turn interaction, spatial reasoning, or emergent dynamics. The omission severely narrows the scientific contribution.

- **Simulation validity is overclaimed and self-contradicted.** Sec 4.3 states: *"these experimental results are consistent with previous studies (Nagel, 1995; Camerer et al., 2004) in social simulation, which confirms the reliability and potential of multi-agent-based simulations."* This is a sweeping claim for one game. More critically, Sec 4.6 directly reveals that changing 2/3 to 51/100 causes substantial behavioral change — exposing memorization as a primary driver. The paper never reconciles this with the reliability claim in Sec 4.3, leaving the central scientific narrative internally inconsistent.

- **Efficiency comparison uses a dummy model; Ray superiority claim is empirically unsubstantiated.** The "orders-of-magnitude" speedup in Fig. 2b is measured with agents that sleep 1 second and return random numbers — an orchestration-overhead benchmark, not an end-to-end LLM simulation. The paper is transparent about this, but does not draw the appropriate limits on the claim. More critically, Sec 3.1 asserts: *"it also makes a significant advancement over existing actor-based distributed frameworks, such as Ray"* — yet provides zero head-to-head empirical comparison with Ray under equivalent hardware and workload. This claim is stated as fact without any supporting measurement.

- **Three of four main contributions are not independently evaluated.** Only the distributed execution mechanism receives focused quantitative analysis. The environment abstraction, background generation pipeline, and Agent-Manager are primarily feature descriptions. In Sec 4.4, the authors note they "manually provide a basic configuration for each group," which is inconsistent with the claimed automation. No ablation or comparison to alternatives is provided for any of these three components.

### Minor

- **No statistical rigor for behavioral experiments.** All results appear to be single-run measurements. Key claims about education-level effects (Fig. 5), model differences (Fig. 3), and multi-round convergence (Fig. 4) are reported without confidence intervals, variance estimates, or repeated trials. The effect size for MistralAI-8×22B across five education levels is only 3.49 points — a difference that requires statistical validation to interpret meaningfully.

- **No comparison with human behavioral data.** The paper claims results are "consistent with previous studies (Nagel, 1995; Camerer et al., 2004)" but presents no quantitative comparison with human experimental distributions. Without this calibration, it is impossible to assess whether LLM agents are valid behavioral proxies or simply activating memorized patterns.

- **Mechanistic language ("rational decisions," "good understanding") is not justified given prior knowledge confound.** Secs 4.3 and 4.5 use language like "agents have a good understanding of this game" and "making rational decisions," but Sec 4.6 demonstrates that these behaviors depend strongly on prompt template matching and memorization. The language in the earlier sections should be qualified accordingly.

### Trivial

- The scaling curve in Fig. 2c covers only three device counts (1, 2, 4), which is sufficient to observe the trend but too sparse to characterize the scaling relationship precisely. Claims about proportional reduction should be stated as observations within the tested range.

---

## Nice-to-Haves

- Add at least one second simulation scenario (opinion dynamics, epidemic spreading, or a market game) to validate that the infrastructure generalizes and that the observed scale is necessary for the phenomena.
- Provide an actual empirical comparison against Ray on the same hardware and workload, even at small scale, to substantiate the distributed framework claims.
- Report repeated runs with variance for the behavioral experiments, particularly for background-conditioned effects where effect sizes are small.
- Surface the prior knowledge finding from Sec 4.6 as a first-class contribution (it is the most scientifically novel part of the paper) rather than treating it as an optional discussion.

---

## Removed Points

*These points are flagged to be removed — treat them with caution.*

- **"Dummy model comparison is misleading/hidden"** — Multiple reviewers implied the paper obfuscates that Fig. 2b uses a dummy model. This is incorrect. Sec 4.2(ii) is explicit: *"we adopt a dummy model request (i.e., agents sleep for 1 second and generate random numbers rather than posting the requests) in the simulation to remove the impact of the LLM inference speed."* The paper is transparent here. The valid criticism is that the paper should more clearly limit the scope of efficiency claims derived from this benchmark, not that it conceals the setup. **REMOVED as a misreading.**

- **"Background generation promotes stereotyped outputs"** — While a legitimate open question for LLM simulation broadly, the paper does not strongly claim to avoid stereotyping. This critique overstates what the paper promises from the background generation feature. **WEAKENED/REMOVED as scope creep.**

- **"Claim that 1M agents is needed for behavioral insights"** — Neutral reviewer noted that behavioral patterns visible at 1,000 agents don't require 1M agents, framing this as an inconsistency. However, the paper's primary claim for scale is efficiency/platform demonstration, not that 1M is needed for the observed insights. This is a valid nice-to-have but not a flaw in the stated contribution. **MOVED to Nice-to-Haves.**

- **Generic strengths** — "The paper is well-written," "the topic is important," "engineering effort is substantial" — removed per hard rules.

---

## Novel Insights

The most genuinely novel observation emerging from the collective reviews is the **prior-knowledge contamination finding in Sec 4.6**: a canonical game whose ratio changes from 1/2 to 51/100 (logically equivalent) produces measurably different agent behavior, and explicitly referencing the classic game in the prompt then restores behavior toward equilibrium. This finding implies that much of what is reported in published LLM simulation studies — including in this paper — may reflect retrieval of memorized task patterns rather than genuine emergent reasoning. This is a scientifically important negative result that deserves much higher prominence than it receives in the current submission.

---

## Suggestions

1. **Restructure the framing:** Decide whether this is a systems paper (scalability/infrastructure) or a simulation-methodology paper. The current paper tries to be both and undersupports each. Systems papers require comparisons against the specific systems they claim to improve upon (Ray); methodology papers require broader task coverage and human-data validation.
2. **Run a second task** — even something simple like a 2-player coordination game or an opinion diffusion model — to demonstrate that the platform is genuinely task-agnostic.
3. **Add Ray comparison:** A minimal head-to-head on the dummy-model benchmark or a small-scale real-LLM task would substantiate Sec 3.1's direct claim.
4. **Foreground Sec 4.6:** Restructure the paper so the prior-knowledge finding motivates design choices and tempers claims throughout, rather than being tucked at the end as a discussion item.
5. **Report error bars:** Even a small number of repeated seeds for the education-level and multi-round experiments would substantially strengthen the behavioral claims.

---

## Score and Decision

**Calibration:**

| Paper | Topic | Scores | Decision |
|---|---|---|---|
| OASIS (JBzTculaVV) | Large-scale LLM social simulation, ~1M agents, multiple social phenomena | 3, 8, 5, 1 (avg ≈ 4.25) | Reject |
| WarAgent (RBaDiInDRg) | LLM multi-agent simulation (war scenarios), thin scientific contribution | 5, 3, 3, 3 (avg ≈ 3.5) | Reject |
| Internet of Agents (o1Et3MogPw) | Multi-agent distributed platform, strong baselines, diverse task evals | 8, 8, 6, 8, 6 (avg ≈ 7.2) | Accept (Spotlight) |

**Reasoning:** This paper is most similar to OASIS — a large-scale LLM simulation infrastructure paper demonstrating up to 1M agents. OASIS showed multiple social phenomena (information spreading, polarization, herding) across two platforms and was rejected at ~4.25 average. This paper demonstrates stronger raw scalability engineering but uses a narrower evaluation (single game), has weaker simulation science (one memorizable task vs. three distinct phenomena), and claims advances over Ray without comparison. Relative to IoA (the high-scoring platform paper), this work lacks performance benchmarks against baselines, formal analysis of system properties, and diverse task validation. The real contribution — the 1M-agent execution and `to_dist` API — is meaningful but insufficient for ICLR given the unsupported claims and single-task scope.

**Axis evaluation:**
- *Novelty:* Moderate — incremental extensions to an existing platform using well-established concurrency patterns (actor model)
- *Technical soundness:* Partial — the distributed mechanism is sound, but Ray claim is unsupported and other contributions are unvalidated
- *Empirical support:* Weak — single game, dummy-model efficiency comparison, no statistical rigor, no human-data calibration
- *Significance:* Moderate — 1M-agent execution is practically impressive, but scientific insights are thin
- *Clarity:* Adequate — the paper is readable, but conflates systems contribution with simulation-methodology contribution

**Final Score: 4.0 — Reject**

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>