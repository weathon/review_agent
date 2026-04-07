=== CALIBRATION EXAMPLE 64 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title "Real-Time Reasoning Agents in Evolving Environments" accurately reflects the paper's scope. The abstract clearly identifies the problem (environments evolve *during* LLM reasoning), names the two contributions (Real-Time Reasoning Gym and AgileThinker), and states the key empirical claim (AgileThinker outperforms single-paradigm agents as difficulty and time pressure rise). One forward-looking concern: the abstract implies the gym and method together constitute a "foundation for temporally constrained AI," but the evaluation is restricted to three arcade-style games with unusually slow environment update intervals (minutes per step in wall-clock time), which may not generalize to the broader claim.

---

### Introduction & Motivation

The highway driving analogy motivating dual-process reasoning is effective, and the framing around LLMs' failure to account for parallel world evolution is well-articulated. The three research questions (RQ1–RQ3) are crisp. However, two issues arise:

**RQ3 assumes the conclusion.** The question "How well do token-based simulations match real-world walltime?" is answered with a single deployment scenario (6-minute environment steps), which is a very narrow validation. The answer to RQ3 is essentially "well, within this one operating point," but generalizability to faster environments is untested.

**Related work is mostly deferred to §7** rather than integrated into motivation. As a result, the introduction overstates novelty: Liu et al. (2024) and Zhang et al. (2025) already deploy LLM agents in wall-clock time. The specific novelty—token-count as hardware-agnostic proxy, and partial planning trace sharing—should be foregrounded more clearly here.

---

### Real-Time Reasoning Gym (§2)

**Strengths:** The desiderata (dynamic, cognitively challenging, reproducible) are clearly stated. The token-count temporal abstraction is a genuine methodological contribution that enables hardware-independent evaluation. The independent control of cognitive load and time pressure is well designed.

**Concern 1 – Ecological validity of time scale.** Using the derived TPOT of 0.047 s/token (via DeepSeek official API), the four time-pressure levels (4k–32k tokens/step) correspond to environment update intervals of roughly **3 to 25 minutes per step**. The paper calls these "real-time" scenarios, but this timescale is orders of magnitude slower than what practitioners typically mean by real-time (milliseconds to seconds). The β offset (β = 334.55 s per episode) is also large and suggests non-trivial per-episode overhead that is not accounted for in the linear model. This does not invalidate the benchmark, but the characterization as "real-time" oversells the practical urgency.

**Concern 2 – Normalization via empirical Rmin/Rmax.** Scores are normalized to [0,1] using empirically observed min/max rewards pooled over all agent trajectories (Table 4). This means the normalization changes as the set of evaluated agents changes. If a new, worse agent were evaluated in the future, the existing scores would shift. A fixed normalization based on a domain-specific oracle (e.g., optimal play for Freeway via BFS) would be more rigorous.

**Concern 3 – Default action design.** When an agent produces no action within the allotted tokens, a *default action* is applied (continue moving / stay idle). This default is relatively benign in all three games—for Freeway and Snake, continuing the prior direction is often safe, and in Overcooked, idling costs little in the short term. This design choice may underestimate the true penalty of failing to respond in time, and could inflate the apparent robustness of planning agents (who rely heavily on the default action). A stricter default (e.g., random action) would stress-test timing more aggressively.

**Concern 4 – Overcooked partner simplicity.** The "dynamic partner" in Overcooked is a scripted agent that randomly selects between two policies (deliver onion to pot or to counter). This does not represent the nuanced theory-of-mind reasoning the paper invokes. The claim that Overcooked requires "coordinating with dynamic partners" is somewhat overstated given the simplicity of the partner policy.

---

### Real-Time Reasoning Agents & AgileThinker (§3)

The description of reactive and planning agents is clear. However, **AgileThinker's mechanism is underspecified in critical ways**:

**Concern 5 – Stale planning traces.** The planning thread P operates on a *frozen* snapshot of the game state. If P requires many environment steps to complete its reasoning (plausible given that each step spans thousands of tokens), the game state it reasons about may be several steps stale. The paper does not specify how many environment steps typically elapse before P restarts with a fresh snapshot, nor does it discuss under what conditions the planning output is explicitly discarded as too stale. Figure 4 suggests P runs continuously and restarts at some implicit trigger, but this is not formalized.

**Concern 6 – How R uses partial traces.** The reactive thread R "references partial reasoning traces from the ongoing planning process." The paper does not describe the interface: Is R given the raw token stream from P? Is it summarized? How does R know which portions of P's reasoning are about the current game state vs. an outdated one? This is the core novelty of AgileThinker relative to prior dual-system work, yet it receives minimal description. Without this detail, reproducibility is questionable.

**Concern 7 – Compute fairness.** AgileThinker runs *two* LLMs simultaneously (V3 as reactive + R1 as planning in the main experiments). The baselines run one model. This means AgileThinker uses roughly twice the compute (or twice the API cost) of any single-paradigm baseline. The appendix (§C.5) does run a *concurrent* variant (time-shared rather than parallel), which is a fair compute-controlled comparison, but this experiment appears only in the appendix and should be a primary result. The concurrent AgileThinker still outperforms single-paradigm baselines in most settings (Table 11), which is the stronger and fairer claim.

**Concern 8 – Model quality conflation.** In the primary experiments, reactive = DeepSeek V3 (non-thinking) and planning = DeepSeek R1 (thinking). These are *different models* with different overall capability levels, not just different amounts of compute applied to one model. AgileThinker thus benefits from combining V3's format-following reliability with R1's reasoning depth, which is partly a model ensemble effect rather than purely an architectural effect. The DeepSeek-V3.2 experiments (thinking on vs. off in Appendix §C.3) are more controlled and should be in the main body.

---

### Experiments & Results (§4–§6)

**Concern 9 – Incomplete baseline coverage.** The paper studies budget forcing for reactive agents and code-as-policy for planning agents but does not consider:
- A *sliding window* reactive agent that can update its plan incrementally when new observations arrive (a natural extension of planning agents).
- A RAP/MCTS-style search that caps computation via beam width rather than token count.
These are plausible strong baselines. Budget forcing underperforms severely (0.01 vs. 0.39) partly because forced truncation yields no-ops, which is a known artifact of the s1-style truncation method rather than a fundamental ceiling for thinking models under budget.

**Concern 10 – Snake anomaly.** In Table 7, at 32k tokens/step, Planning (R1) achieves 0.9629 in Snake—substantially higher than AgileThinker's 0.8281. This means *under low time pressure*, the planning agent is actually better than AgileThinker in Snake. The text acknowledges this indirectly ("advantages growing as cognitive load and time pressure increase") but does not explain the mechanism for why AgileThinker underperforms planning at low pressure. Does R's intervention in the final TR tokens disrupt P's well-formed plan?

**Concern 11 – Wall-clock experiments are thin.** Table 2 reports three data points for three games with a single time pressure setting (TE = 6 minutes). This is insufficient to establish the claimed "practical applicability." The authors should vary TE in wall-clock experiments or at least test at multiple settings to confirm the token-based ranking is preserved.

**Concern 12 – Statistical significance setup.** The paired t-test uses only 8 game seeds as the pairing unit (averaging over 4 LLM seeds within each game seed). With n=8 pairs, the test has very low power, and the text says the advantage "generally becomes" significant—implying it is not always significant. Figure 8 shows several cells above p=0.05 even at high difficulty. The paper should report which specific conditions are and are not significant and discuss why.

---

### Resource Management (§5)

The CDF-guided budget selection (set NTR ≈ natural token upper bound of R) is an insightful finding and is well-supported by Figure 7. The dynamic AIMD-style adaptation (Appendix E) is a nice practical addition. The observation that performance peaks "when NTR approximates the natural token upper bound" is intuitive but raises a practical question: the natural upper bound of R presumably varies across games and difficulty levels—the paper acknowledges this requires empirical tuning, which is a real deployment cost.

---

### Related Work (§7)

The positioning against delay-aware MDPs and asynchronous MDPs from RL literature is appropriate. The comparison to Liu et al. (2024) and Zhang et al. (2025) is adequate. The claim that AgileThinker "distinctively advances" dual-process AI by allowing System 1 to access *partial* System 2 traces is the sharpest differentiation from prior work and is credible, but it needs more empirical support (e.g., an ablation that removes the partial-trace access and forces R to use only the final output of P, which is what the Gemini approximation does in §C.3).

---

### Limitations (§9)

The limitations section is honest about the DeepSeek-only evaluation and the lack of rigorous connection to human dual-process theory. Missing from the limitations discussion:
- **No training signal.** The paper presents AgileThinker as an inference-time architecture, but the reactive and planning LLMs were not trained for this protocol. Future work might ask whether fine-tuning R to better consume partial traces from P improves performance.
- **Scalability to faster environments.** The architecture assumes TE is long enough for both P and R to operate. As TE shrinks (faster environments), R is squeezed out. The behavior at very high time pressure (4k tokens in the main paper, ~3 minutes) is already approaching this limit, but sub-second real-time is not explored.
- **Single-task evaluation.** All three games are turn-based grid-world tasks with text-serialized states. Real-world dynamic environments (robotic control, financial trading, live code editing) have very different state spaces and may require different architectures.

---

### Overall Assessment

This paper addresses a genuine gap: LLM-based agents are routinely evaluated in environments that pause while the model reasons, an assumption that breaks down in any deployment where the world keeps moving. The Real-Time Reasoning Gym is a concrete, reproducible contribution—the token-count temporal abstraction is clever and the independent knobs for cognitive load and time pressure are well designed. AgileThinker's core insight (let the reactive thread read the *partial* output of the planning thread, rather than waiting for completion) is a real architectural contribution that goes beyond prior cascading or independent dual-system designs.

However, several issues collectively weaken the evaluation. Most critically: the primary comparison is between AgileThinker (two models in parallel) and single-model baselines, which is a compute-unfair comparison; the fair compute-controlled experiment (Appendix C.5) is relegated to the appendix. The "real-time" framing is undermined by environment update intervals of 3–25 minutes, making the benchmark more accurately described as "computation-time-constrained" than real-time in the engineering sense. The planning agent's strong performance under low time pressure in Snake (outperforming AgileThinker) and the underspecified mechanism for how R integrates partial traces from P are important unexplained issues. The contribution is directionally right and the benchmark is genuinely useful, but the paper would be significantly stronger with the V3.2-based experiments in the main body, a clear description of the partial-trace interface, the concurrent-execution result as a primary comparison, and a more honest scoping of what "real-time" means in this context. In its current form, this is a borderline accept: the problem and benchmark are clear ICLR contributions, but the method evaluation requires additional rigor.

# Neutral Reviewer
## Balanced Review

### Summary
This paper identifies a critical gap in LLM agent evaluation by formalizing **Real-Time Reasoning**, where environmental states evolve continuously while the agent computes. It introduces **Real-Time Reasoning Gym** (Freeway, Snake, Overcooked) to benchmark agents under varying cognitive loads and time pressures, proposing a hardware-agnostic token-count proxy for time. The authors present **AgileThinker**, a dual-thread architecture that parallelizes rapid reactive decisions with deep planning, demonstrating consistent performance gains over single-paradigm baselines through both simulation and wall-clock experiments.

### Strengths
1.  **Critical Problem Formulation:** The paper correctly identifies and addresses a fundamental limitation in current agent evaluation: the assumption of static environments during reasoning. By formalizing "Real-Time Reasoning," it bridges the gap between theoretical agent benchmarks and practical deployment constraints.
2.  **Robust Validation of Simulation:** Section 6 provides strong empirical evidence validating the token-count-as-time proxy against actual wall-clock time using the DeepSeek API ($R^2 = 0.9986$). This ensures the results are reproducible across hardware and relevant to physical deployment scenarios, a common hurdle in AI simulation papers.
3.  **Effective Architecture Design:** The **AgileThinker** proposal (streaming partial reasoning traces from a planning thread to a reactive thread) is conceptually sound and empirically validated. Figure 6 and Section 5 clearly demonstrate how this specific mechanism avoids the pitfalls of standalone reactive (short-sighted) or planning (stale) agents.
4.  **Reproducibility Commitment:** The authors pledge to release code and environments upon publication and provide detailed hyperparameter settings (e.g., cognitive load, time pressure levels) in the Appendix, aligning well with ICLR’s reproducibility standards.

### Weaknesses
1.  **Model Dependency Limitation:** The method relies heavily on the availability of transparent reasoning traces (Chain of Thought), restricting the primary experiments to open-source models (DeepSeek V3/R1). While they test proprietary models in the Appendix for baselines, they explicitly state AgileThinker cannot be directly applied to them due to trace opacity. This limits the generalizability of the proposed architecture within the broader industry landscape.
2.  **Computational Efficiency Trade-off:** While AgileThinker improves performance, it inherently requires running two models (or threads) simultaneously or sequentially. The paper evaluates "parallel threads" against concurrent threads in Appendix C.5, but a formal analysis of the **cost-per-score** (e.g., FLOPs or API cost vs. Gain in score) would strengthen the argument for its practical adoption, especially given higher latency requirements.
3.  **Environment Scope:** The benchmarking environments are adapted versions of classic games (Freeway, Snake, Overcooked). While well-controlled, they lack the complexity and modality (e.g., visual processing, multi-step tool use) of true "embodied" real-world tasks. The jump from game agents to real-world decision systems remains a theoretical extension rather than an empirically proven one.
4.  **Hyperparameter Sensitivity:** Section 5 indicates that the optimal reactive token budget ($N_{TR}$) varies across environments and requires empirical tuning. Without a robust auto-tuning mechanism (beyond the simple dynamic adjustment in Appendix E), the system's plug-and-play usability in unknown real-time scenarios is questioned.

### Novelty & Significance
**Novelty:** The paper demonstrates solid novelty in combining **dynamic environment simulation** with **dual-process architectures** specifically for **real-time constraints**. While dual-process (System 1/System 2) concepts exist in literature, the specific implementation allowing the reactive thread to consume *partial* reasoning traces from the planning thread during ongoing generation is a distinct contribution not fully explored in prior work. The introduction of Real-Time Reasoning Gym is a new, valuable benchmarking resource.

**Significance:** The work aligns well with the current direction of LLM research towards safety and reliability in deployment. By proving that static evaluations overestimate agent capabilities in dynamic settings, the paper provides a necessary reality check for the field. The findings suggest that balancing speed and depth is not merely a tuning problem but a structural architectural requirement, guiding future agent design for high-stakes real-time applications.

### Suggestions for Improvement
1.  **Generalize to Closed-Source Models:** Investigate methods to apply the AgileThinker architecture to closed-source models (e.g., using function calling to inject partial reasoning logs, or approximating traces via smaller open models as intermediaries). Even a discussion on feasibility would address the current limitation regarding proprietary model dominance.
2.  **Cost-Benefit Analysis:** Include a dedicated analysis of the computational trade-off. Quantify how much additional latency or monetary cost is incurred per point of performance gain. ICLR reviewers often look for practical efficiency metrics alongside accuracy.
3.  **Expand Environment Diversity:** To strengthen the claim of generalizability, consider adding a benchmark from a more standard embodied AI environment (e.g., a subset of Habitat or AI2-THOR) or a multi-step tool-use environment that forces real-time planning under visual constraints, rather than grid-based text-based games.
4.  **Formalize Resource Allocation:** Since the optimal $N_{TR}$ requires tuning, expand the dynamic adjustment algorithm proposed in Appendix E. A more robust, theoretically grounded mechanism for allocating compute between threads based on real-time feedback signals (e.g., environmental entropy) would increase the method's robustness.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation on Partial Trace Access:** You claim the core novelty is the reactive thread accessing *partial* planning traces, but you lack an ablation comparing "partial access" vs. "final output only" vs. "no access" on the same model family. Without this, it is unclear if performance gains stem from the streaming mechanism or merely from running two models.
2. **Direct Comparison to Existing Dual-System Baselines:** You cite prior dual-system architectures (e.g., Zhang et al., 2025; Liu et al., 2024) in Related Work but do not benchmark against them in the main results. To claim superiority over existing dual-process methods, you must empirically compare against these specific baselines under identical time pressures.
3. **Performance vs. Total Compute Cost:** Running two LLM threads effectively doubles inference cost, yet you primarily optimize for score per step rather than score per total token consumed. Add a Pareto frontier analysis (Score vs. Total Tokens) to demonstrate that the performance gain justifies the increased computational budget.

### Deeper Analysis Needed (top 3-5 only)
1. **Thread Conflict Analysis:** You need to quantify how often the reactive and planning threads propose contradictory actions and how those conflicts are resolved. Without analyzing failure modes where the reactive thread overrides a correct long-term plan, the reliability of the coordination mechanism is unproven.
2. **Robustness to Incomplete Syntax:** The reactive thread consumes streaming tokens which may end mid-sentence or mid-thought; analyze how often syntactic incompleteness degrades the reactive model's understanding. If the method relies on coherent partial traces, you must show performance stability when planning traces are cut off at arbitrary token positions.
3. **Token-Time Proxy Variance:** You validate the token-to-time correlation (Fig 10) but do not analyze sensitivity to inference jitter (e.g., API latency spikes). Real-time systems fail under variance, so you must demonstrate how fluctuations in TPOT (Time Per Output Token) impact the synchronization protocol and overall safety.

### Visualizations & Case Studies
1. **Raw Partial Trace Snapshot:** Figure 6 shows abstract trajectories but not the actual text; provide a visualization showing the exact incomplete sentence the planning thread was generating when the reactive thread made its decision. This is necessary to verify that the reactive thread is actually utilizing meaningful reasoning context rather than just noise.
2. **Decision Override Example:** Include a specific case study where the planning thread's initial strategy was rendered obsolete by a dynamic change, and the reactive thread successfully corrected it using the partial trace. This would concretely demonstrate the "agility" claim versus a standard planning agent failing.
3. **Efficiency Pareto Curve:** Plot normalized score against total inference time (wall-clock) for all agents, not just token count. This visualizes whether AgileThinker dominates the efficiency frontier or simply trades latency for accuracy like the planning baseline.

### Obvious Next Steps
1. **Single-Model Dual-Thread Implementation:** Investigate whether a single model can time-share between planning and reactive roles via context switching to reduce the 2x compute cost. This is critical for practical deployment, as running two large models concurrently is often prohibitively expensive.
2. **Evaluation in Non-Game Domains:** The current gym consists of grid-world games (Freeway, Snake); extend evaluation to a continuous control task (e.g., simulated robotics) or web interaction to validate claims of "real-world applicability." Grid worlds may not capture the observation noise and action granularity of true real-time environments.
3. **Fine-Tuning for Urgency:** Instead of only inference engineering, propose a fine-tuning objective that trains a single model to internally balance reasoning depth based on time pressure. This addresses the conclusion's suggestion to "leverage our gym to train urgency-aware LLM agents" which is currently unexplored.

# Final Consolidated Review
## Summary

This paper introduces **Real-Time Reasoning** as a problem formulation where environments evolve during agent computation, and presents **Real-Time Reasoning Gym**—three games (Freeway, Snake, Overcooked) where state updates at fixed intervals regardless of whether the agent has produced an action. The authors propose **AgileThinker**, a dual-thread architecture that runs a planning LLM and reactive LLM in parallel, allowing the reactive thread to access partial reasoning traces from the planning thread. Experiments show AgileThinker outperforms single-paradigm baselines as task difficulty and time pressure increase.

## Strengths

- **Token-count temporal abstraction:** Section 6 validates token count as a hardware-agnostic time proxy with R² = 0.9986 correlation against wall-clock time using the DeepSeek API. This is a genuine methodological contribution enabling reproducible, hardware-independent evaluation of time-constrained reasoning.

- **Genuine problem formulation:** The paper correctly identifies a fundamental limitation in current LLM agent evaluation—environments that pause during reasoning. The independent control of cognitive load (game difficulty) and time pressure (tokens per step) is well-designed and enables systematic analysis.

- **Streaming architecture contribution:** AgileThinker's mechanism—allowing the reactive thread to read partial outputs from an ongoing planning process rather than waiting for completion—is a substantive architectural contribution beyond prior dual-system designs that operate in stages or isolation.

- **Compute-controlled experiment included:** Appendix C.5 shows AgileThinker outperforms baselines even when running threads concurrently (time-shared) rather than in parallel, demonstrating that the benefit stems from cognitive architecture rather than raw compute doubling.

## Weaknesses

- **Compute fairness in primary comparisons:** The main experiments compare AgileThinker (running V3 + R1 in parallel, roughly 2× compute) against single-model baselines. While Appendix C.5 includes a concurrent-execution variant with controlled throughput and shows it still outperforms baselines, this fairer comparison is relegated to the appendix. The stronger claim should be presented in the main body.

- **Mechanism underspecification:** The reactive thread R "references partial reasoning traces from the ongoing planning process" (Section 3), but the exact interface is not described. Does R receive the raw token stream? Is it summarized? How does R know which portions of P's reasoning concern the current state versus stale observations? This is the core novelty, yet lacks implementation detail.

- **Time scale framing:** The paper uses "real-time" for environment steps of 3–25 minutes (derived from 4k–32k tokens × 0.047s/token). While the token-based abstraction is valid, "real-time" typically connotes sub-second responsiveness. The benchmark is better characterized as "computation-time-constrained reasoning" rather than real-time in the conventional engineering sense.

- **Model quality conflation in main experiments:** Reactive agents use DeepSeek V3 (non-thinking) while planning agents use DeepSeek R1 (thinking)—different models with different capability levels. The V3.2 experiments (Appendix C.3, thinking on/off within one model) are more controlled but appear only in the appendix. The paper should lead with this controlled comparison.

- **Snake anomaly unexplained:** In Table 7, Planning (R1) achieves 0.9629 on Snake at 32k tokens/step—higher than AgileThinker's 0.8281. The paper states advantages "grow as cognitive load and time pressure increase" but does not explain why AgileThinker underperforms planning at low pressure. Does the reactive thread's intervention disrupt well-formed plans in simpler settings?

- **No ablation on partial vs. final trace access:** The claimed novelty is R accessing *partial* traces from P. However, the paper lacks an ablation comparing: (a) partial trace access, (b) final output only after P completes, and (c) no planning information. The Gemini experiment (C.3) approximates (b) but uses a different model family.

- **Narrow wall-clock validation:** Table 2 reports only one time-pressure setting (6 minutes per step) for wall-clock experiments. The paper should validate that token-based rankings hold across multiple time pressures in real time.

- **Statistical power limitation:** The paired t-test uses n=8 game seeds as pairing units, which provides limited statistical power. Several cells in Figure 8 remain above p=0.05 even at high difficulty.

## Nice-to-Haves

- **Pareto frontier analysis:** A plot of normalized score versus total tokens consumed (or API cost) would clarify whether AgileThinker's gains justify the compute overhead, and whether concurrent execution dominates the efficiency frontier.

- **Thread conflict analysis:** Quantifying how often reactive and planning threads propose contradictory actions—and how conflicts are resolved—would strengthen confidence in the coordination mechanism's reliability.

- **Robustness to syntactically incomplete traces:** If R reads P's stream at arbitrary token positions, some traces will be mid-sentence or mid-thought. Analyzing performance stability under syntactic incompleteness would validate the mechanism's robustness.

- **Direct comparison to existing dual-system baselines:** Zhang et al. (2025) and Liu et al. (2024) are cited but not empirically compared under identical time pressures. Benchmarking against these architectures would strengthen claims of superiority.

## Removed Points

These points are flagged to be removed, treat them with caution:

- *Overcooked partner simplicity:* The critic notes the scripted partner only switches between two policies. While simple, this is clearly described in Appendix A and does not misrepresent the coordination challenge. The paper does not claim sophisticated theory-of-mind reasoning.

- *Default action design too lenient:* The critic argues defaults (continue direction, stay idle) underestimate timing penalties. This is a design choice; the games still create real time-pressure trade-offs, and the relative comparison between methods remains valid.

- *Missing sliding window baseline:* Demanding a sliding-window reactive agent that incrementally updates plans is outside the paper's stated scope of comparing reactive vs. planning paradigms.

- *Rmin/Rmax empirical normalization:* While using fixed oracle-based normalization would be cleaner, the empirical approach still yields valid relative comparisons.

- *No training signal discussed:* This is acknowledged in the conclusion as future work and does not diminish the inference-time architectural contribution.

## Novel Insights

The token-time proxy validation reveals an interesting offset: the linear model T = αN + β includes a substantial constant term (β = 334.55 seconds), indicating non-negligible per-episode overhead independent of token count. This suggests that for short episodes, the overhead dominates, and the token abstraction becomes more accurate for longer-horizon tasks. Future work could explore whether this overhead is inherent to API-based inference or could be mitigated with local deployment. Additionally, the CDF analysis showing optimal performance when N_TR aligns with the reactive thread's natural token usage suggests a principle for budget allocation: allocate compute to match the inherent demands of the sub-task rather than arbitrarily dividing resources.

## Suggestions

- **Move V3.2 controlled experiments to main body:** Replace or supplement the V3/R1 main results with V3.2 thinking-on/thinking-off comparisons to isolate the architectural effect from model-quality differences.

- **Add explicit mechanism description:** Include 2–3 sentences in Section 3 describing how R accesses P's partial traces (e.g., "R receives the raw token stream up to the current position; P generates reasoning in a structured format with state annotations that R can parse even when incomplete").

- **Explain the Snake anomaly:** Add a brief analysis of why Planning outperforms AgileThinker at low time pressure in Snake—whether R's interventions disrupt effective plans, or whether this is variance, and discuss implications for when dual-thread architectures are unnecessary.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 4.0]
Average score: 6.5
Binary outcome: Accept
