Now I have sufficient calibration context. Let me write the final review.

## Summary

This paper proposes enhancements to the AgentScope multi-agent platform to support very large-scale simulations with LLM-powered agents. The key contributions include an actor-based distributed mechanism for agent-level parallel execution and automatic workflow conversion, flexible environment support for agent-environment interactions, a configurable tool and automatic background generation pipeline for diverse agent populations, and a web-based Agent-Manager for monitoring agents across devices. The framework is demonstrated through simulations of the "guess 2/3 of the average" game with up to 1 million agents, showing scalability, efficiency gains, and behavioral observations across different LLMs, prompts, and agent backgrounds.

## Strengths

- **Impressive engineering scale**: Orchestrating 1 million LLM-powered agents on 4 devices with completion times of 12 minutes (Llama3-8B, Prompt 1) to ~10.6 hours (Llama3-70B, Prompt 2) is a genuine engineering achievement that advances the practical frontier of LLM-based simulation (Fig. 2a). The dummy-benchmark comparison (40 seconds vs. 12 days serial / 8.6 hours async, Fig. 2b) cleanly isolates orchestration overhead.

- **Practical distributed deployment design**: The `to_dist` function that automatically converts centralized workflows to distributed execution with a single code change (Fig. 1) is a well-designed, practical contribution that significantly lowers the barrier for distributed deployment. The proxy/placeholder mechanism is conceptually clean.

- **Systematic multi-dimensional exploration**: The experimental analysis covers multiple dimensions—6 LLMs, 2 prompt variants, education/occupation backgrounds, individual vs. group simulations, game variants with different ratios, and a non-zero Nash equilibrium (Sec. 4.1–4.6)—yielding a breadth of behavioral observations from a single domain.

- **Important empirical finding on LLM prior knowledge**: The experiment varying the ratio from 2/3 to 1/2 and 51/100, and showing that adding a note referencing the classic game shifts behavior (Fig. 8), is a valuable finding about how LLM prior knowledge can confound simulation results. The non-zero Nash equilibrium variant (Fig. 9) further probes reasoning vs. memorization. These results have relevance beyond this specific paper.

## Weaknesses

### Fatal
None.

### Major

- **Single evaluation domain severely limits generality claims**: All experiments use variants of the "guess 2/3 of the average" game—a static, single-number-decision game with extremely simple state, actions, and dynamics. While useful as a stress test of scale, this single domain cannot substantiate the paper's broader claims about supporting "various real-world scenarios" (Abstract, Sec. 1) or being applicable to "economic, societal, transportation, healthcare" simulations (Sec. 2). In particular, the environment module (Sec. 3.2), which supports shared states, listeners, and multi-environment nesting, is never stress-tested under complex state dynamics, multi-step agent-environment coupling, or spatial/temporal interactions. The group-level simulation (Sec. 4.5) is the only experiment that uses the environment mechanism, and it involves only 3 groups of 500 agents—a small fraction of the claimed scale. A second domain with richer dynamics would substantially strengthen the paper.

- **No empirical comparison with existing distributed/agent frameworks**: The paper claims "significant advancement over existing actor-based distributed frameworks, such as Ray" (Sec. 3.1) and that existing frameworks "suffer from low efficiency" (Sec. 2), but provides no benchmark against Ray, AutoGen, MetaGPT, or any other contemporary system. The only baselines in Fig. 2b are serial execution and Python async I/O within the authors' own system. Additionally, the claim that Python async I/O is "constrained by GIL" (Sec. 3.1) is overstated—LLM API calls are typically network-bound and release the GIL. Without empirical comparison to strong alternatives, the core efficiency claim does not reach its intended strength.

- **Prior knowledge contamination undermines behavioral validity claims**: The paper acknowledges that LLMs likely have prior knowledge of the "guess 2/3 of the average" game (Sec. 4.6), and the 51/100 vs. 1/2 experiment (Fig. 8) demonstrates that behavioral differences emerge when LLMs can't pattern-match to a known game. This raises serious questions about whether the observed "rational behaviors" (Sec. 4.3: "agents have a good understanding of this game and are capable of considering other agents' behaviors and making rational decisions") reflect genuine reasoning or memorized game-theoretic knowledge. The claim that results "confirm the reliability and potential of multi-agent-based simulations" (Sec. 4.3) is too strong given this confound.

### Minor

- **Behavioral claims about agent diversity and rationality are descriptive, not rigorously quantified**: Statements like "the higher the educational level of agents, the lower the average reported numbers, indicating more rational behaviors" (Sec. 4.4) are based on visual inspection of violin plots with only 200 agents per group, without statistical tests, effect size measures, or comparison against human behavioral data from Nagel (1995) or Camerer et al. (2004). The paper claims consistency with these studies but provides no quantitative overlay.

- **No cost or resource accounting**: Running 1M agents with 70B-parameter models is extremely expensive, yet the paper reports no GPU utilization, memory usage, total tokens generated, or monetary cost. This makes it difficult for researchers to assess practical feasibility.

- **"Linear scaling" claim based on limited data**: Fig. 2c shows only 3 data points (1, 2, 4 devices) for 10K agents, which is insufficient to establish linear scaling. The claim that "increasing the number of devices can proportionally reduce the simulation running time" (Sec. 4.2) is plausible but under-supported.

### Trivial
None.

## Nice-to-Haves

- A second evaluation domain with richer dynamics (e.g., spatial simulation, multi-turn negotiation, or network formation) would substantially strengthen the generality argument and properly stress-test the environment and management modules.
- Quantitative comparison with human experimental data from Nagel (1995) / Camerer et al. (2004) to establish whether LLM agent distributions are good proxies for human behavior in this specific game.
- An ablation isolating the contribution of each proposed component (distributed mechanism, environment module, background generation) rather than demonstrating them holistically.
- Analysis of failure rates (unparseable/invalid LLM outputs) at 1M-agent scale.

## Removed Points

- *"The 1M-agent claim relies on dummy requests"* (Neutral reviewer): The paper is transparent in Sec. 4.2 that the 40-second result uses dummy models and provides the actual LLM-based times separately. This is not a misleading claim—just two different measurements for different purposes.
- *"Questionable authenticity of agent diversity—all agents using the same LLM share identical training data"* (Human finder): This conflates LLM homogeneity with agent diversity. The paper shows that varying prompts (different backgrounds, CoT vs. non-CoT) produces meaningful behavioral variation even within the same LLM. The concern is valid but overstated as a weakness; it's inherent to any LLM-based simulation approach and the paper's configuration tool addresses the generation side.
- *"Missing comparison with human behavioral data"* is kept as a minor point but moved from major because this is primarily a systems/infrastructure paper, not a behavioral science validation paper. Comparing with human data would strengthen behavioral claims but is not a core requirement for the infrastructure contribution.
- *"No ablation on the background generation pipeline vs. simpler alternatives"* (Spark): The education-level and occupation experiments (Sec. 4.4, Appendix F) do exercise the pipeline; the concern about comparing to "directly prompting" is valid but speculative—any such alternative would still use the same infrastructure, so the comparison would mainly speak to prompt design, not to the infrastructure itself.
- *"Unclear handling of LLM output failures"* (Neutral reviewer): This is a fair implementation concern but not a weakness that undermines the core claims. Valid as a minor practical concern but elevated to "nice-to-have" rather than a weakness since the paper doesn't claim robust failure handling.

## Novel Insights

The experiment with 51/100 vs. 1/2 ratios (Fig. 8) provides a striking demonstration of LLM prior knowledge contamination in simulations: even though these ratios produce mathematically similar games, LLM behavior diverges significantly, suggesting that much of the "rational" behavior observed with the standard 2/3 ratio may be driven by memorized knowledge rather than genuine game-theoretic reasoning. The "+note" intervention partially recovers alignment, highlighting a practical prompting strategy for mitigating this confound. This is an important cautionary finding for the broader LLM-agent simulation community.

## Suggestions

- Add at least one evaluation domain beyond number-guessing games to demonstrate that the environment module, background generation, and management tools generalize to scenarios with richer dynamics (spatial, temporal, or strategic complexity).
- Benchmark at least against Ray actors on the same simulation task to substantiate efficiency claims empirically rather than by assertion.
- Tone down behavioral validity claims (e.g., change "confirm the reliability" to "are directionally consistent with") given the acknowledged prior knowledge confound.

## Evaluation

**Originality**: Moderate. Individual components (actor model, environment abstractions, configuration tools, web management) are established concepts. The primary novelty is in their integration and the `to_dist` automatic workflow conversion for LLM-agent simulation. The engineering is significant but the algorithmic/conceptual novelty is limited.

**Importance of research question**: High. Scalable LLM-based multi-agent simulation is an important and timely problem, and practical tooling to enable large-scale deployment is valuable for the community.

**Claims well supported**: Partially. The scalability/efficiency claims are supported by demonstration but not by comparison to alternatives. The behavioral claims are weakened by prior knowledge contamination and the single-domain evaluation. The infrastructure claims are supported by the working system but lack stress-testing in complex scenarios.

**Soundness of experiments**: Moderate. The experiments are internally consistent and well-structured, but confined to a single simple domain with acknowledged confounds.

**Clarity**: Good. The paper is well-organized and clearly written, with helpful figures (especially Fig. 1 for the distributed mechanism).

**Value to community**: Moderate-to-high. As a released open-source platform, this could be a practical resource for researchers wanting to run large-scale LLM agent simulations, even if the scientific novelty is limited.

## Calibration

I compared against: **OASIS** (large-scale LLM social simulation, scores 3/8/5/1, rejected) — similar in being an engineering-heavy LLM simulation paper with shallow behavioral insights but no strong scientific novelty. The current paper has a clearer systems contribution (distributed orchestration) and more honest acknowledgment of limitations, but similarly lacks comparison with alternatives. **Research Town** (scores 6/5/6/6, rejected) — similar in scope (simulation framework, limited novelty) but scored higher on presentation. **WarAgent** (scores 5/3/3/3, withdrawn) — also an LLM-based multi-agent simulation with questionable behavioral validity. The current paper is stronger than WarAgent in engineering rigor and scale, and stronger than OASIS in having a clearer distributed-systems contribution, but weaker than OASIS in domain richness (OASIS at least simulates multiple social phenomena on real platforms). The current paper falls in a similar range to OASIS but with somewhat stronger engineering contributions.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>