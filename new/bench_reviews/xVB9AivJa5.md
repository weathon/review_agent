The paper content was provided inline. Let me verify key claims from the reviewers against the paper content I already have.

Let me now carefully work through each major point:

**Harsh Critic Issue 1 (Evaluation metric doesn't measure spatial reasoning):** The paper claims to measure "spatial intelligence" including "maintaining consistent scale" (Abstract) and "converting visual information into accurate spatial representations" (Conclusion). The scoring metric uses: Jaccard similarity for edge overlap (50%), degree correlation (20%), graph density (10%), room count (10%), door count (5%), door orientation (5%). Rooms are labeled by size rank. The paper acknowledges it doesn't account for room shape (§2.4). 

This is a VALID and SUBSTANTIAL criticism. The scoring is primarily graph-topological (connectivity) and size-rank based. It does not measure spatial layout, actual positions, room shapes, absolute distances — key aspects of "spatial intelligence" as traditionally understood. The paper's own claims about "maintaining consistent scale" are explicitly not measured by the metric. However, I should note the paper DOES acknowledge this limitation. The question is whether the acknowledgment is sufficient given the strength and centrality of the claims about "spatial intelligence."

**Harsh Critic Issue 2 (Instruction following confound):** VALID. The paper requires 9 strict formatting rules. Models that fail to follow these rules get penalized heavily or become unscorable. The paper acknowledges this but still interprets results as evidence of "spatial intelligence" blind spots. The paper says "Blueprint-Bench should test spatial intelligence, not instruction following" (§2.4) but the design conflates them.

**Harsh Critic Issue 3 (Size ranking as identifier):** VALID. This is a real problem with the metric — size-ranking errors cascade into connectivity penalties. The paper acknowledges this in §2.4.

**Harsh Critic Issue 4 (Random baseline under-defined):** VALID but somewhat OVERSTATED. The paper does describe it in §2.2 as "generating typical floor plans using LLMs and image generation models without any image input." But indeed, more details would help. However, the "below random" claim is only emphasized for some models, and the results still show a clear human-AI gap regardless.

**Harsh Critic Issue 5 (Agent evaluation limited):** VALID but somewhat OVERSTATED. The paper only tests 2 agent configurations and acknowledges the limitations. But the claim in the abstract "agent-based approaches with iterative refinement capabilities show no meaningful improvement" is too strong given only 2 agents with limited exploration.

**Harsh Critic Issue 6 (Dataset small):** VALID but not as severe as claimed. 50 apartments is small but not unreasonably so for a novel benchmark. The paper should describe dataset characteristics better though.

**Spark's suggestions for missing experiments:**
- Statistical significance tests: VALID concern but common in benchmark papers; not fatal
- No ablation separating instruction following from spatial intelligence: VALID and important
- No specialized model baseline: The paper explicitly scopes this out, stating the purpose is not to find the best system. But a comparison would contextualize results. NICE-TO-HAVE rather than critical.
- Human evaluation incomplete: VALID, only 12/50 apartments tested
- No ablation on number of input images: NICE-TO-HAVE
- Scoring metric sensitivity: VALID
- Random baseline construction unexplained: VALID (overlaps with Harsh Critic #4)

**Human Finder's weaknesses:**
1. Small dataset: VALID but partially addressed by novelty
2. Instruction following confound: VALID (overlaps)
3. Cascading penalties: VALID (overlaps)
4. Lack of granular error analysis: VALID
5. Missing specialized baselines: NICE-TO-HAVE — the paper explicitly scopes this out
6. Apples-to-oranges comparison: VALID concern but the point of the benchmark is cross-architecture comparison
7. Insufficient related work: REMOVE (per rules — can't confirm existence of cited works)

Now let me compose the final review.

Actually, I need to revisit something. The harshest critique claims the paper's central claim — that it measures spatial intelligence — is fundamentally undermined by the metric not measuring what it claims. Let me verify: the abstract says "requires genuine spatial intelligence: inferring room layouts, understanding connectivity, and maintaining consistent scale." The metric measures connectivity (via graph edges) and room counts/door counts. It does NOT measure layout accuracy (positions/shapes) or scale consistency. But "inferring room layouts" IS partially captured by connectivity, and "understanding connectivity" IS directly measured. "Maintaining consistent scale" is NOT measured. So the claim is partially supported but partially not.

The paper's own §2.4 says: "two floor plans (that follow the rules) will always have a higher score if they are indeed similar" — this is asserted but not proven. And it's about the metric being monotone with true similarity, not about it measuring all aspects of spatial intelligence.

I think the core issue is real but not fatal: the metric captures SOME aspects of spatial intelligence (connectivity, room/door counting) but not others (scale, shape, position). The paper overclaims by saying it measures "spatial intelligence" broadly when it really measures "topological structure of apartment layouts as captured by connectivity and room size ranking." The instruction-following confound further muddies things.

This is a significant weakness but not fatal — the benchmark still provides value as a partial probe. The paper just needs to moderate its claims.

Let me write the final review now.</think>## Summary

Blueprint-Bench introduces a benchmark for evaluating spatial reasoning in AI models through photo-to-floorplan conversion. Given ~20 interior photographs of an apartment, models must produce a 2D floor plan following strict formatting rules (black walls, green doors, red room-center dots), which are automatically scored via graph-based comparison of room connectivity and size ranking against ground truth. The paper evaluates LLMs, image generation models, and agents, finding that most models perform at or below random baseline while humans substantially outperform all AI systems.

## Strengths

- **Novel and well-motivated task formulation.** The photo-to-floorplan task is intuitive, ecologically valid, and genuinely probes spatial reasoning using in-distribution inputs with out-of-distribution outputs—analogous to ARC but in a more naturalistic setting. This is a creative contribution to the evaluation landscape for spatial intelligence.
- **Cross-architecture evaluation.** Evaluating LLMs (via SVG generation), image generation models (direct pixel output), and agents (iterative tool use) on the same task with a unified scoring metric is genuinely useful. The qualitative agent traces (e.g., Claude Code iteratively refining but still failing, Figure 8) provide concrete evidence about what fails and why.
- **Automated, deterministic scoring.** The 9-formatting-rule design enables rule-based computer vision extraction (floodfill segmentation, HSV filtering, connectivity detection), avoiding the pitfalls of LLM-as-judge. The paper candidly demonstrates that LLMs cannot reliably parse floor plans (§2.4), justifying the engineering effort.
- **Honest discussion of limitations.** The paper acknowledges that size-ranking causes cascading penalties, that shape is ignored, that LLM-based extraction failed, and that instruction following is confounded with spatial intelligence measurement (§2.4). This transparency is commendable.

## Weaknesses

### Major:

- **The scoring metric does not measure "spatial intelligence" as advertised, creating a claim-evidence gap.** The abstract states the task "requires genuine spatial intelligence: inferring room layouts, understanding connectivity, and maintaining consistent scale." The metric captures connectivity (edge Jaccard at 50%) and coarse room sizing (via size ranking), but does **not** measure consistent scale, room shapes, absolute/relative positions, or geometric accuracy—core aspects of spatial intelligence. A floor plan that is topologically correct but geometrically scrambled (mirrored, rotated, rooms reshuffled in position) could score well; one with slightly wrong size ranking but correct spatial layout would be severely penalized due to cascading rank-induced errors. The paper's conclusion that "Blueprint-Bench reveals that current AI systems struggle significantly with spatial reasoning" overreaches what the metric actually measures: it reveals struggles with *topological structure and size ranking under strict formatting constraints*, which is a necessary but insufficient proxy for spatial intelligence.

- **Instruction-following confound is acknowledged but not resolved, yet strong conclusions depend on resolving it.** The 9 formatting rules are necessary for automated scoring but create a dual requirement: the model must both understand the apartment's spatial layout AND produce pixel-precise compliance with a synthetic drawing protocol. The paper documents that NanoBanana and GPT-4o fail primarily on instruction following (§3, Figure 6), and §2.4 states "Blueprint-Bench should test spatial intelligence, not instruction following." Yet the same scores are interpreted in §3 and §4 as evidence of "a blind spot in LLM capabilities" and that "current AI systems struggle significantly with spatial reasoning." Without disaggregating format compliance from spatial accuracy, the headline claim about spatial intelligence blind spots cannot be cleanly established. A simple mitigation would be to report what fraction of outputs were rule-compliant and score only those for spatial accuracy separately.

- **Cascading penalty from size-ranking as room identifier distorts the core connectivity measurement.** Rooms are identified by size rank (1=largest), so if two similarly-sized rooms are swapped in ranking, their connectivity edges are compared against wrong counterparts. The paper acknowledges this causes "additional penalties when scoring the connectivity" (§2.4), and the human results confirm it: "all human floor plans were drawn such that the connectivity between the rooms was correct" but they were penalized for size ranking (§3). The authors themselves suspect "another similarity scoring model would make the human's lead over the AI models much larger." This means the primary score—the one used to compare all models—does not cleanly measure connectivity, which is the paper's own stated focus. This is a structural problem in the evaluation design, not just a minor limitation.

### Minor:

- **Random baseline is under-specified.** The paper's headline finding is that models perform "at or below a random baseline," but the baseline is described only as "generating typical floor plans using LLMs and image generation models without any image input" (§2.2). Which models? How many generations? What prompts? Given that the comparison depends on this baseline, more detail is needed. Note: I am not questioning the existence of the baseline, just the specificity of its construction methodology.

- **Agent evaluation is too thin to support the claimed conclusion.** Only two agent configurations (Codex CLI/GPT-5 and Claude Code/Opus) are tested, with no variation in tools, prompting, or iteration budgets. The abstract's claim that "agent-based approaches with iterative refinement capabilities show no meaningful improvement over single-pass generation" generalizes beyond what two configurations can support.

- **Dataset is underspecified.** 50 apartments provide reasonable initial coverage, but there is no description of geographic/architectural diversity, number-of-rooms distribution, or difficulty stratification. The human evaluation covers only 12 apartments with unstated number of participants and time constraints.

- **Scoring weights appear arbitrary without validation.** The 50/20/10/10/5/5 weighting (edge overlap, degree correlation, density, room count, door count, door orientation) is presented without justification or sensitivity analysis. No validation against human similarity judgments is provided.

### Trivial:
- SVG generation details (canvas size, coordinate system, prompts) and image generation prompts are not fully specified, making exact reproduction of generation conditions difficult.

## Nice-to-Haves

- **Separate format-compliance scores from spatial-accuracy scores.** Report what fraction of each model's outputs are rule-compliant, and compute spatial scores only on compliant outputs. This would directly address the instruction-following confound.
- **Validate the metric against human similarity judgments.** Have humans rate floor plan similarity pairs and correlate with algorithmic scores, establishing whether the metric is a valid proxy for spatial similarity.
- **Add ablations on the scoring** (e.g., what score does a topologically-correct-but-geometrically-scrambled plan receive?), sensitivity analysis on weights, and per-component score breakdowns.
- **Include a specialized model baseline** (e.g., HouseGAN, LayoutGPT) to contextualize the gap between generalist and specialist systems.
- **Expand human evaluation** to all 50 apartments with multiple annotators and inter-annotator agreement.

## Removed Points

*These points were flagged for removal; treat them with caution.*

- **"Missing comparison with specialized spatial reconstruction methods (NeRF, etc.)"** — The paper explicitly scopes this out (§1: "the purpose of Blueprint-Bench is not to find the best possible system, but rather to measure the spatial intelligence of models that are not specifically trained for it"). This is a deliberate design choice, not an oversight. Moved to Nice-to-Have.

- **"Insufficient discussion of related spatial reasoning benchmarks (SPACE, VSP, GeoMeter, etc.)"** — Per guidelines, I cannot confirm the existence of uncited related works and should not flag missing references.

- **"No statistical significance tests"** — For a benchmark paper presenting mean scores across 50 apartments, this is a standard-practice gap, not a fatal flaw. Moved to Nice-to-Have.

- **"Paper overclaims 'first numerical framework for comparing spatial intelligence across architectures'"** — This is a factual claim about the paper's novelty. While the metric limitations weaken it, there is genuinely no prior benchmark that enables direct numerical comparison across LLMs, image generators, and agents on a spatial task. This is a partial overclaim but not fabricated.

- **"Conflating instruction following with spatial intelligence makes the benchmark primarily test rule adherence"** — This was kept as a Major weakness but the extreme version (that the benchmark is *only* testing instruction following) was softened. Some models (GPT-5, Gemini 2.5 Pro) do produce rule-compliant outputs and still fail at spatial accuracy, confirming spatial intelligence is genuinely being probed for those models.

- **"Apples-to-oranges comparison across model classes"** — This is the entire point of Blueprint-Bench (cross-architecture comparison). The shared scoring metric makes comparison meaningful. Removed as a weakness since it's the intended contribution.

## Novel Insights

The paper reveals an interesting asymmetry: image generation models with demonstrably strong general capabilities (e.g., NanoBanana solving math problems, Figure 2) completely fail on this task—not because they lack spatial understanding per se, but because they cannot follow a strict drawing protocol. This suggests that the instruction-following bottleneck may be the binding constraint for current image generation models on structured spatial tasks, rather than spatial reasoning deficits. Had the paper cleanly disaggregated these factors, it could have made a more nuanced and novel claim: that image generation models' spatial intelligence is currently impossible to evaluate because their instruction following is too poor to produce scorable outputs. Among LLMs that do produce compliant outputs (GPT-5, Gemini 2.5 Pro), the fact that they still fail at connectivity inference despite in-distribution inputs is the genuinely surprising finding, and it parallels ARC-style results.

## Suggestions

1. **Add a two-track scoring system**: (a) format compliance rate (fraction of outputs that pass the 9 rules), and (b) spatial accuracy score computed only on compliant outputs. This would cleanly separate the two failure modes.
2. **Replace size-ranking as the canonical room identifier with spatial identifiers** (e.g., room position on the canvas relative to other rooms, or hungarian matching between predicted and ground-truth rooms) to eliminate cascading penalties and better measure connectivity correctness.
3. **Report per-component scores and per-apartment difficulty analysis** so readers can understand whether failures are concentrated in connectivity, size estimation, or count prediction, and whether certain apartment types are harder.
4. **Add metric ablations**: report scores for a topologically-correct plan and for a geometrically-close-but-misranked plan to validate the scoring function's alignment with human spatial similarity judgments.

## Score and Decision

**Calibration**: I compared against several spatial reasoning benchmark papers. SPACE (scores 6/8/8/5, Accept Poster) is a well-grounded cognitive science benchmark with comprehensive task design, clean evaluation metrics, and nuanced findings—stronger than Blueprint-Bench because its metric directly measures the claimed capability. COMFORT (scores 5/6/8/8/10, Accept Oral) systematically evaluates frame-of-reference understanding with a validated metric. VSP (scores 3/5/8/3, Reject) has task-design issues and insufficient analysis. GenPlan (scores 3/3/3/8/3/1, Reject) has flawed evaluation metrics and weak methodology.

Blueprint-Bench falls between the well-validated accepted benchmarks and the rejected ones. Its task formulation is novel and well-motivated, and the finding that models fail even on in-distribution inputs is valuable. However, the core scoring metric has structural issues that undermine the central claim about "spatial intelligence": it measures topological structure and size ranking under strict formatting constraints, not the full spectrum of spatial reasoning. The cascading size-ranking penalty distorts the primary connectivity measurement, and the instruction-following confound is acknowledged but unresolved. These are real methodological issues that weaken the reliability of the reported scores as measures of spatial capability, even if the qualitative finding (models struggle) is likely directionally correct. The paper is a valuable initial contribution to spatial evaluation but needs metric validation and claim moderation.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>