Now I have a thorough understanding. Let me write the final consolidated review.

## Summary

OpenCity proposes a scalable simulation platform for urban LLM agents, combining a system-level LLM request scheduler (I/O multiplexing, connection pooling, CPU task parallelism) with a prompt-level "group-and-distill" strategy that clusters agents with similar static attributes and uses shared context in batch prompts to reduce token redundancy. Evaluated on six global cities with up to 10,000 agents, the platform achieves a 635× average speedup, 74% request reduction, and 46% token reduction while maintaining behavioral fidelity comparable to batch prompting. The platform also provides a web portal for no-code configuration and establishes a first benchmark comparing LLM agents against real-world urban dynamics data.

## Strengths

- **Important and timely problem**: Scaling LLM-agent simulations from tens to thousands of agents is a genuine bottleneck, and the paper tackles it with both system- and prompt-level strategies. (Sections 1, 3.2)
- **Novel group-and-distill mechanism**: The IPL + distill pipeline is, to this reviewer's knowledge, a novel approach for reducing token redundancy across agents with shared static attributes while preserving dynamic properties. It clearly outperforms archetype/reuse prompting on faithfulness (Table 2: T1 of 71-86% vs 4-13%). (Section 4.2, Table 2)
- **Comprehensive efficiency evaluation across 6 cities**: The platform is tested on Beijing, New York, San Francisco, London, Paris, and Sydney with 10,000 agents, showing consistent speedup (521–793×). (Table 1, Figure 3)
- **Dual evaluation of efficiency AND fidelity**: Unlike many systems papers that report only speed, the paper includes JSD and top-1 hit rate comparisons against raw prompting, batch prompting, and archetype prompting (Table 2), as well as urban dynamics validation against real data (Table 3). This is a meaningful strength.
- **Real urban simulation benchmark**: Establishing the first comparison of LLM agents against classical EPR models on radius of gyration, OD matrices, and segregation across 6 cities is a useful contribution. (Section 5.3, Table 3)

## Weaknesses

### Fatal
None.

### Major

- **No ablation separating system-level from prompt-level contributions**: OpenCity has two orthogonal optimization families—I/O scheduling (Section 4.1) and group-and-distill (Section 4.2). The paper evaluates only the combined system. Without running each component alone, it is impossible to determine what fraction of the 635× speedup and 74% request reduction comes from standard I/O parallelism versus the novel group-and-distill strategy. This matters because if most speedup comes from engineering optimizations rather than the novel prompt design, the core research contribution is substantially smaller than claimed. The absence of this ablation is a significant gap. (Sections 4.1, 4.2, 5.2)

- **Headline 635× speedup conflates standard engineering with research novelty**: The baseline in Table 1 and Figure 3 is described as sequential requests without optimization. The speedup figure combines: (a) I/O multiplexing and connection pooling—standard techniques available in any async HTTP library, (b) CPU task parallelism across cores, and (c) group-and-distill's request/token reduction. Since no ablation is provided, the "635× acceleration" claim cannot be attributed to the paper's novel contribution. A comparison against even a basic async implementation with connection pooling would provide a more honest baseline for the research contribution. This does not mean the engineering isn't useful—the platform works—but it inflates the perceived novelty. (Table 1, Figure 3)

### Minor

- **Overclaiming in benchmark results**: The paper states that "LLM Agent performs as well as or better than the classical rule-based EPR Agent" (Section 5.3), but Table 3 shows EPR outperforming the Generative Agent on ODMSE for New York (3.70e-4 vs 5.95e-4), San Francisco (14.0e-4 vs 23.6e-4), and Paris (6.25e-4 vs 7.58e-4), and on SMSE for New York (0.2319 vs 0.3521). The results are mixed, and the claim should be tempered to "comparable to" or "competitive with" EPR rather than "as well as or better." (Table 3, Section 5.3)

- **Meaningful fidelity degradation on cost-effective model under-discussed**: On GPT-4o-mini (the model practical for large-scale simulation), Table 2 shows JSD increasing from 0.04 (Inherent) to 0.13 (Ours) and T1 dropping from 90% to 74% for Beijing. While "Ours" is comparable to "Batch prompting" (JSD 0.11, T1 76%), both represent noticeable degradation from raw prompting. The paper acknowledges this by testing on GPT-4o, where results are much better, but the cost-effective model's degradation deserves more explicit discussion about whether the fast-but-approximate simulation produces trustworthy urban dynamics. (Table 2)

- **Benchmark uses only 1,000 agents while speedup tested at 10,000**: The urban dynamics evaluation in Section 5.3 uses 1,000 agents, but the platform claims to handle 10,000. This gap is unexplained, raising the question of whether the group-and-distill fidelity degrades at scale when group sizes increase. (Sections 5.2 vs 5.3)

- **Case study finding is somewhat tautological**: The counterfactual experiment showing that evenly distributing agents reduces segregation (Section 6) unsurprisingly confirms that residential differences cause segregation—the result follows almost directly from the input. The value of LLM agents in this experiment is not clearly demonstrated over a conventional simulation. (Section 6)

### Trivial
None.

## Nice-to-Haves

- **Ablation study**: Running the simulation with only the LLM request scheduler or only group-and-distill, and against an async baseline with connection pooling, would substantially clarify the paper's contribution.
- **Sensitivity analysis of IPL grouping**: How robust are results when LLM-generated clusters are semantically incoherent? Does distillation degrade with poor groupings?
- **Total wall-clock time and simulation step count for the "1 hour for 10,000 agents" claim**: Only per-agent time is reported; total runtime and step count would help assess practical feasibility.
- **Explicit acknowledgment of mixed benchmark results**: Discuss when/why LLM agents underperform EPR on specific metrics and cities.

## Removed Points

- **"Standard engineering practices presented as research novelty"**: The harsh critic suggests I/O multiplexing and connection pooling are merely standard practices not worth presenting as research. However, the paper's primary contribution is the integrated platform, and the system-level scheduler is part of a coherent system design. Systems papers routinely include engineering optimizations alongside novel contributions. This is not a separate weakness to list, but it feeds into the ablation concern already captured above. (Kept as part of the major weakness on conflation, not as a standalone fatal flaw.)
- **"Missing comparison against a competent async baseline"**: This is absorbed into the major weakness about the speedup claim conflating engineering and novelty. Calling for a specific async library comparison is a nice-to-have experimental improvement, not a fatal flaw.
- **"The IPL circular dependency critique"**: The critic argues that using the LLM to create groups that then optimize LLM prompts is a "circular dependency." This is standard practice in self-referential LLM optimization (e.g., using LLM prompts to optimize other LLM prompts). This is not a weakness; it is the design.
- **"Data sources are heterogeneous making cross-city comparisons difficult"**: This is a scope creep criticism. The paper explicitly uses real urban data from multiple cities with different sources, which is a strength for generalization, not a weakness.
- **"Number of simulation steps never stated"**: The simulation step count would help assess cost but is not a core gap.
- **Missing RMSE values for two cities**: The dashes in Table 3 for New York and San Francisco Generative Agent RMSE likely reflect data availability (Safegraph provides OD flows, not individual trajectories), not negligence.
- **Formatting/grammar issues**: Removed per rules.
- **"Reproducibility concerns"**: The paper provides a code repository link, meeting reasonable reproducibility standards.
- **"Missing related works"**: Removed per rules (no external sources to verify).
- **"IPL grouping quality not stress-tested"**: Moved to Nice-to-Haves as it would strengthen but not invalidate the paper.

## Novel Insights

The group-and-distill mechanism is a genuinely interesting approach that exploits a structural property of urban LLM agents—static vs. dynamic attribute separation—that differs from general-purpose prompt optimization. The distinction between "reuse-based" methods (which destroy agent independence) and "share-context" methods (which preserve it) is clearly articulated and relevant beyond urban simulation to any multi-agent system with heterogeneous but partially overlapping prompts. The paper's evaluation also reveals an under-discussed tension: the cost-effective model (4o-mini) that one would deploy at scale shows meaningfully more behavioral deviation under optimization than the expensive model (4o), suggesting that efficiency-fidelity tradeoffs may be more severe in practice than headline numbers suggest.

## Suggestions

- Add an ablation with three conditions: (1) system scheduler only, (2) group-and-distill only, (3) both combined, plus a competent async baseline. This single experiment would resolve the most significant concern about attribution of the speedup.
- Temper the benchmark claim from "as well as or better" to "competitive with" and discuss the cities/metrics where EPR still outperforms.
- Explicitly discuss the fidelity gap on GPT-4o-mini and its practical implications for simulations run at scale.

## Score and Decision

**Calibration anchors used:**

| Paper | Score | Comparison |
|-------|-------|-----------|
| MarS (Yqk7EyT52H) | 7.0 (Accept Poster) | Simulation platform with comprehensive evaluation; OpenCity has weaker evaluation (no ablation, overclaimed speedup) but addresses a similarly important problem |
| LightSeq (kC5i5X9xrn) | 5.0 (Reject) | Claims speedups with limited baselines; OpenCity similarly conflates engineering with novelty but has more real-world evaluation |
| S2-Attention (OqTVwjLlRI) | 4.25 (Reject) | Overclaimed 25× speedup with unfair baselines; OpenCity has a similar pattern but more genuine novelty |
| Skeleton-of-Thought (mqVgBbNCm9) | 5.67 (Accept Poster) | LLM efficiency method, novel but limited in scope; OpenCity is comparable in scope but with more evaluation |
| FMint (SvjFHucuDZ) | 4.5 (Reject) | 5× speedup over weak baseline, no adaptive-step comparison; OpenCity has similar inflation concerns |
| bEgDEyy2Yk | 1.0 (Reject) | Extreme case—speedup against strawman SLINK with no real comparison; OpenCity is far above this |

OpenCity has genuinely novel ideas (group-and-distill) and a real working platform evaluated across 6 cities, which puts it above outright rejects like S2-Attention (4.25) and FMint (4.5). However, the lack of ablation and the conflation of standard engineering with research novelty in the headline speedup are substantial weaknesses that similar systems papers were penalized for. The overclaiming in the benchmark is minor. Overall, this falls in the borderline range—stronger than typical rejects but with meaningful evaluation gaps that diminish confidence in the magnitude of the contribution.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>