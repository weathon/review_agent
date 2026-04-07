=== CALIBRATION EXAMPLE 29 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title "PLAGUE: Plug-and-Play Framework for Lifelong Adaptive Generation of Multi-Turn Jailbreaks" is accurate. However, the abstract's headline claim — "improving attack success rates (ASR) by more than 30% across leading models" — is ambiguous and potentially misleading. In the paper body, these improvements are computed as *relative* gains (e.g., the 32.14% improvement on o3 is (0.814−0.616)/0.616, which is a relative gain relative to ActorBreaker, not GOAT as claimed in Section 5.1). Framing relative improvements as "30%" without the word "relative" may mislead a reader expecting absolute percentage-point gains.

The abstract further says "81.4% on OpenAI's o3" and "67.3% on Claude's Opus 4.1" using SRE. Since SRE is a graded score (not a strict binary), and the authors modify the original StrongREject prompt to "increase its sensitivity, favoring an aligned response" (Appendix C.1), these headline numbers are not directly comparable to the SRE numbers reported elsewhere in the community — a critical omission in the abstract.

---

### Introduction & Motivation

The problem motivation is sound and well-cited. The three desiderata for an effective red-teaming agent (relevance + progression, feedback-driven evolution, diverse sampling) are well-articulated and serve as useful organizing principles for the subsequent design.

**Concern**: The paper cites Li et al. (2024) as showing that "multi-turn jailbreaks were orders of magnitude more effective...than ensembles of single-turn attacks." "Orders of magnitude" means 10× or more, which is a strong claim that deserves verification. The contributions listed are clear, though the novelty framing overlaps significantly with AutoDAN-Turbo and AutoRedTeamer, which are acknowledged but possibly under-differentiated.

---

### Related Work

This section is thorough and well-structured. The comparison in Table 1 (presence/absence of lifelong learning, reflection, planning, etc.) is a useful visual summary of the landscape. The authors fairly identify their own inspirations.

**Concern**: The claim that "only human-generated strategies appended during initialization seem to yield a discernible improvement in [AutoDAN-Turbo's] performance, while improvements from freshly discovered strategies remain unexplored" is asserted without citation or ablation evidence. This appears to be the authors' opinion and should be framed as such, or supported with data.

---

### Method

The three-phase architecture (Planner → Primer → Finisher) is logically well-structured. The design decision to omit the last planning step (n−1 Primer steps) to leave room for the Finisher is a reasonable heuristic, though it is not theoretically motivated — why exactly does saving the final step help? The backtracking mechanism (removing failing turns from HT but retaining them in HA) is clever and practically important.

**Key concerns:**

1. **Lifelong learning contribution is unclear.** The lifelong memory bank R+ is described as the central novelty. However, the bank is initialized with only *two* strategies from Crescendo, and the ablation (Table 3: GOAT+BT+R+P vs. GOAT+BT+R+P+RSS) shows SRE improvements of 0.773→0.814 on o3 and 0.431→0.465 on Claude — modest gains that are not far from noise. There is no analysis of how many new strategies are actually *learned* during an evaluation run, whether discovered strategies are substantively different from the initial seeds, or whether the system would work comparably with a fixed library of the two initial strategies throughout.

2. **Rubric Scorer threshold choices lack justification.** The Primer succeeds at score ≥ 7/10, and the Finisher triggers backtracking at ≤ 3/10 and declares success at > 8/10. These thresholds are critical to the attack's behavior, but no ablation explores their sensitivity. A minor shift (e.g., Primer threshold of 6 vs. 8) could significantly alter performance.

3. **Modified SRE metric.** In Appendix C.1, the authors state they modify the original StrongREject prompt to "increase its sensitivity." The consequence is that their reported SRE scores are *not* the same metric as StrongREject scores in the literature. The scale factor (dividing by 8 rather than 10) is noted, but the systematic upward bias introduced by prompt modification is not quantified.

4. **Algorithm 3 has a logical inconsistency.** Line 10 sets the success threshold at `score > 9.0`, but Section 3.5 says "the attack ends when...we...receive a score greater than 8/10" and "If this scoring criterion is met, we mark the attack as successful." There is an inconsistency between the pseudocode (9.0) and the prose (8/10). Which is correct? This matters for reproducibility.

---

### Experimental Setup

**Choice of Attacker Model.** The paper uses DeepSeek-R1 as the attacker LLM "across all our experiments." DeepSeek-R1 is a reasoning model known to have weaker safety guardrails than many alternatives. It is well-suited for the complex reasoning required in attack planning, but using it likely provides a significant performance advantage over scenarios where practitioners have access to differently-aligned attackers. There is *no ablation on attacker model choice*, which is a significant gap. Would PLAGUE remain SOTA if the attacker were GPT-4o, Claude, or Llama instead?

**ASR@K=2 comparability.** The paper uses K=2 ("We use K=2 for all our experiments"), taking the best score from two independent attack attempts. The paper says ActorBreaker is run with K=2 actors (two plans per goal). For Crescendo and GOAT, it is never stated that they are *also* run K=2 times — only that "the budget for all baselines and our experiments is capped at six turns." If PLAGUE runs twice (2×6 target calls = up to 12) while GOAT runs once (6 calls), the comparison is not budget-equitable. The paper should report single-attempt ASR@1 alongside ASR@2 for clarity, and should confirm all baselines are evaluated at K=2.

**Baseline modification.** The authors modify GOAT's evaluation environment (invoking the Rubric Scorer R per round rather than at the end) and add early stopping. They also remove Crescendo's backtracking limits. While these changes are explained, they depart from official implementations, making it unclear whether performance differences are due to the methods themselves or these environmental changes. Ideally, results should show official baseline performance alongside the modified versions.

---

### Results & Discussion

**Section 5.1 Inconsistency.** The paper claims "we outperform the previous best - GOAT by a factor of 32.14%" for o3. However, Table 2 shows ActorBreaker SRE on o3 = 0.616 > GOAT SRE = 0.587. ActorBreaker is actually the stronger baseline on o3. The 32.14% figure matches (0.814−0.616)/0.616 ≈ 32.1% — i.e., improvement over *ActorBreaker*, not GOAT as stated. This is a factual error in attribution.

**Duplicate row in Table 2.** The ActorBreaker row appears *twice* in Table 2 with identical values, which is a clear error.

**Statistical robustness.** Results are "averaged over three runs," but no confidence intervals, standard errors, or significance tests are reported. Multi-turn attacks have high variance (as the authors themselves note), and three runs may be insufficient to establish reliable ordering between closely-scoring systems. PLAGUE's advantage over Crescendo on Claude (0.673 vs. 0.480 SRE) is larger, but on o1 (0.931 vs. 0.692) and Llama (0.958 vs. 0.899) the differences, while substantial, are never statistically characterized.

**DeepSeek-R1 at 97.8% SRE.** This is the headline result but is only weakly discussed. DeepSeek-R1 is widely recognized as having weaker safety alignment than o3 or Claude Opus; 97.8% SRE may reflect the target model's weakness rather than the attack's sophistication. The authors briefly acknowledge that model difficulty correlates with ASR but do not separately analyze why DeepSeek is particularly easy to jailbreak.

**Diversity vs. ASR trade-off (Figure 3).** The paper notes that PLAGUE has *lower* diversity than ActorBreaker (0.375 vs. 0.433) but higher ASR. This trade-off is acknowledged but not analyzed: is the lower diversity a feature (efficient convergence to good strategies) or a limitation (brittle to defenses that target the specific attack patterns)? This is relevant to real-world red-teaming utility.

**Category-wise analysis (Figure 4 / Appendix C.3).** The paper reports "near-perfect ASR (99.5%)" on misinformation categories, while sexual content is hardest. This is interesting and policy-relevant but is only briefly discussed. Understanding *why* certain categories resist the attack would be valuable.

---

### Writing & Clarity

The three-phase framing is clear and the algorithm pseudocode is helpful. However, two passages cause genuine confusion:

1. Section 3.2 states "We use SRE and ASR interchangeably in our work." This is never acceptable — the two metrics have different semantics (binary vs. graded), and the abstract uses SRE numbers while calling them "ASR." Readers expecting the standard binary ASR definition will be misled.

2. The budget accounting is spread across Sections 4 and 5.2 without a single consolidated statement of what "6 turns" means for each method, and whether all methods receive exactly 12 target LLM calls (K=2 × 6) or different totals.

---

### Limitations & Ethics

The ethics section acknowledges dual-use risk and invokes open-access arguments for safety research, which is standard. However, the discussion is thin relative to the severity of the results — 81.4% ASR on o3 and 67.3% on Claude Opus 4.1 are non-trivial figures. The paper does not discuss:
- Whether the framework (and example attacks in Appendix D) should be released fully or under researcher-access controls.
- The implications of attack strategies being retained in a shared lifelong memory, potentially transferring across goals.
- Responsible disclosure to model providers.

The technical limitations section is limited to the conclusion: "we leave the development of a better diversity-inducing Planner to future work." Missing from the discussion: (a) failure modes when target models use system-prompt-level multi-turn defenses; (b) whether lifelong learning would degrade under distributional shift (new harm categories); (c) compute costs of running the multi-agent system at scale.

---

### Overall Assessment

PLAGUE presents a modular, well-engineered framework for multi-turn jailbreaking that achieves genuinely strong empirical results across several frontier models. The plug-and-play architecture and the large-scale evaluation across five major models are real contributions. However, the paper has several issues that undermine confidence in the reported numbers: (1) a factual error in crediting the baseline improvement on o3 (ActorBreaker vs. GOAT); (2) a modified SRE metric that inflates scores relative to the established benchmark; (3) ASR@K=2 metric that may not be applied consistently to all baselines; (4) no ablation on the attacker model (DeepSeek-R1), which likely explains much of the advantage; (5) the lifelong learning component — the claimed central novelty — shows only modest incremental gains in ablation; and (6) a threshold inconsistency between pseudocode (>9.0) and prose (>8/10). For ICLR acceptance, these issues collectively require revision: at minimum, honest characterization of the SRE modification, explicit K=2 budget accounting for all baselines, an attacker-model ablation, and confidence intervals on the main results. As it stands, the contribution is interesting and the empirical scale is appreciated, but the methodological presentation is not yet rigorous enough to fully support the headline claims.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents PLAGUE, a plug-and-play, lifelong-learning framework for automating multi-turn jailbreak attacks on LLMs using a three-phase architecture (Planner, Primer, Finisher). It claims significant improvements in Attack Success Rate (ASR), achieving state-of-the-art performance on recent models like OpenAI o3 and Claude Opus 4.1 while maintaining efficient query budgets. The work is positioned as a vital tool for comprehensively evaluating LLM safety vulnerabilities through systematic red-teaming.

### Strengths
1.  **Comprehensive Empirical Evaluation:** The paper provides a robust experimental setup evaluating against highly resistant, state-of-the-art models (e.g., OpenAI o3, Claude Opus 4.1) that are often hard to jailbreak. The inclusion of both StrongREJECT and binary ASR metrics (Table 2) allows for a nuanced view of success that aligns with current safety benchmarks.
2.  **Modular and Extensible Design:** The introduction of a plug-and-play framework demonstrates engineering clarity. The ability to swap components (e.g., using different Finisher modules like GOAT vs. Crescendo) and the detailed ablation studies (Tables 3 and 4) provide valuable insights into which specific agentic behaviors drive success.
3.  **Budget-Aware Analysis:** Unlike many red-teaming papers that ignore compute costs, this work explicitly analyzes LLM call budgets across Target, Evaluator, and Planner phases (Table 5). It demonstrates that performance gains are achieved with minimal inference overhead compared to baselines like Crescendo, which adds practical relevance.

### Weaknesses
1.  **Incremental Novelty vs. Fundamental Innovation:** The core components (lifelong learning memory, reflection agents, multi-turn planning) draw heavily from existing single-turn or multi-turn frameworks like AutoDAN-Turbo, RACE, and ActorBreaker. The primary contribution appears to be the *integration* of these modules rather than a novel algorithmic mechanism, which may limit its fit for ICLR's theoretical standards compared to venues focused on application or security.
2.  **Reliance on LLM-as-a-Judge without Human Verification:** The paper relies almost exclusively on LLMs (Qwen3) for the Rubric Scorer and Final Evaluator (Section 3.2). Without human-in-the-loop validation or cross-evaluator consensus analysis, there is a risk of evaluator bias or "eval hacking," where the attacker optimizes against the judge rather than the target model's actual safety failure.
3.  **Diversity Trade-offs:** The analysis in Figure 3 admits that PLAGUE's diversity score remains lower than ActorBreaker's despite higher ASR. The paper does not deeply analyze whether this reduced diversity limits the generalizability of vulnerabilities discovered, which is a key metric for effective red-teaming that assesses broad risk rather than just single-point failures.

### Novelty & Significance
**Novelty:** Moderate. The work innovates by structuring known agentic behaviors into a cohesive, lifelong-learning pipeline for multi-turn contexts. However, the individual blocks (vector memory for strategy retrieval, reflection loops) are well-established in prior red-teaming literature.
**Significance:** High. The evaluation on very recent models (o3, Opus 4.1) provides critical, timely data on current safety alignment limits. The "plug-and-play" approach offers a useful methodology for the research community to stress-test models, even if the attack mechanics themselves are compositional.
**Clarity:** Good. The three-phase structure is logically defined, though the text occasionally becomes repetitive when comparing baselines.
**Reproducibility:** High. Specific model versions, parameters, and dataset details (HarmBench) are provided. The reliance on public APIs and standard frameworks aids in reproduction, though the specific prompt engineering details would benefit from code release (which is claimed in the ethics statement).

### Suggestions for Improvement
1.  **Strengthen the Evaluation Protocol:** To mitigate concerns about evaluator bias, include a subset of attacks evaluated by human annotators or a consensus of multiple distinct "judge" models to validate that the ASR gains are real and not artifacts of the judge model's alignment.
2.  **Clarify Theoretical Contribution:** In the Introduction and Method sections, explicitly articulate the *algorithmic* novelty beyond "integration." Is there a new memory update mechanism? A novel loss function for the planner? Clarify how the "lifelong learning" differs technically from AutoDAN-Turbo's implementation to justify the ICLR classification.
3.  **Expand Diversity Analysis:** Provide a deeper discussion on the relationship between ASR and Diversity. If PLAGUE sacrifices diversity for success, discuss the implications for safety coverage (e.g., does it find the "hardest" single jailbreak or the "safest" set of jailbreaks?). Consider using the diversity metric as a primary trade-off axis in the results.
4.  **Improve Presentation of Formulas & Tables:** While I understand these are extraction artifacts, ensure the final submission has clean LaTeX rendering for algorithms (e.g., Algorithm 1-4) and tables to ensure precise verification of mathematical claims and data points during the review process.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Sequential Learning Protocol:** Run attacks sequentially to populate memory, measuring ASR gain per added strategy to validate "lifelong" claims rather than static retrieval.
2. **Judge Robustness Check:** Evaluate successful jailbreaks using a stronger judge (e.g., o1) or human annotators to verify Qwen3 isn't under-scoring harmfulness.
3. **Random Retrieval Baseline:** Compare embedding-based retrieval against random strategy retrieval to prove semantic similarity drives performance gains.
4. **Budget Efficiency Pareto:** Plot ASR vs. Total API Calls (including Planner/Scorer) to verify the "comparable budget" claim contradicted by Table 5.
5. **Cross-Model Transfer:** Test if strategies learned on one model (e.g., Llama) successfully jailbreak another (e.g., Opus) to validate generalizability.

### Deeper Analysis Needed (top 3-5 only)
1. **Baseline Modification Impact:** Quantify how much tweaking GOAT's evaluation environment (adding Rubric Scorer) degraded its original performance compared to PLAGUE.
2. **Rubric Scorer Dependency:** Analyze if attack success correlates with the Rubric Scorer's own safety alignment, risking a self-reinforcing weak judge loop.
3. **Failure Mode Categorization:** Categorize the remaining 20-30% failures on o3/Opus by refusal type to identify unresolved vulnerabilities.
4. **Memory Overfitting:** Investigate if the system memorizes HarmBench-specific templates rather than learning generalizable jailbreak strategies.
5. **Table 2 vs. Table 4 Consistency:** Explain why the main results table excludes the best Opus 4.1 performance found in the ablation table.

### Visualizations & Case Studies
1. **Retrieval Similarity Distribution:** Histogram of cosine similarity scores between queries and retrieved memories to prove meaningful matching occurs.
2. **Retrieval Case Study:** Display a specific goal and its retrieved memory strategy side-by-side to validate semantic relevance claims.
3. **Turn-wise Score Trajectory:** Compare Rubric Scores over 6 turns for PLAGUE vs. Crescendo to demonstrate the "escalation" mechanism visually.
4. **Success/Fail Conversation Logs:** Provide side-by-side dialogue examples showing where the Primer phase succeeded vs. drifted semantically.
5. **Cost-Benefit Scatter:** Plot Total Compute Cost vs. ASR for all baselines to visualize efficiency trade-offs clearly.

### Obvious Next Steps
1. **Human Verification:** Validate a subset of high-scoring attacks with human annotators to ensure automatic metrics aren't inflated.
2. **Defense Robustness:** Evaluate PLAGUE against standard mitigations (e.g., perplexity filters, input sanitization) to assess real-world threat.
3. **Unified Results Table:** Consolidate Tables 2 and 4 to present the true SOTA performance consistently across all models.
4. **Compute Cost Formalization:** Report total token consumption including attacker and judge calls, not just Target LLM calls.
5. **Responsible Disclosure Revision:** Reconsider releasing full attack code/prompts given the SOTA nature and potential for misuse without safeguards.

# Final Consolidated Review
## Summary

The paper presents PLAGUE, a plug-and-play framework for generating multi-turn jailbreak attacks on LLMs through a three-phase architecture: Planner (generates attack plan with retrieved strategy examples), Primer (builds adversarial context through seemingly benign queries), and Finisher (delivers the final harmful query). The framework incorporates lifelong learning via a memory bank that stores successful attack strategies for retrieval in future attacks. Evaluated on HarmBench across five frontier models (OpenAI o3, o1, DeepSeek-R1, Claude Opus 4.1, Llama 3.3-70B), PLAGUE achieves strong attack success rates, including 81.4% SRE on o3 and 67.3% on Claude Opus 4.1, outperforming existing multi-turn attack methods.

## Strengths

- **Comprehensive empirical evaluation across frontier models**: The paper evaluates on genuinely challenging targets (OpenAI o3, Claude Opus 4.1) that are widely considered resistant to jailbreaks. The consistent evaluation across five models with both binary ASR and StrongREJECT metrics provides meaningful data about current safety alignment limits.

- **Modular architecture enabling systematic ablation**: The plug-and-play design allows clear isolation of component contributions. Tables 3 and 4 demonstrate that adding backtracking, reflection, planning, and strategy retrieval each provide measurable improvements (GOAT baseline: 0.587 SRE on o3 → PLAGUE: 0.814 SRE on o3), helping readers understand which mechanisms matter.

- **Budget-aware efficiency analysis**: Table 5 explicitly compares total LLM calls across methods, showing PLAGUE achieves higher ASR with comparable (sometimes fewer) target calls than Crescendo. This attention to computational cost is often missing in red-teaming papers.

## Weaknesses

- **Factual error in baseline attribution**: Section 5.1 states "we outperform the previous best—GOAT by a factor of 32.14%" on o3. However, Table 2 shows ActorBreaker (SRE 0.616) outperforms GOAT (SRE 0.587) on o3. The 32.14% improvement calculation matches (0.814−0.616)/0.616, which is improvement over ActorBreaker, not GOAT. This misattribution undermines confidence in the reported claims.

- **Modified SRE metric reduces comparability**: Appendix C.1 states the authors modified the StrongREJECT prompt to "increase its sensitivity, favoring an aligned response." The scale change (dividing by 8 vs. 10) is noted, but the systematic bias introduced by prompt modification is not quantified. This means the headline SRE scores are not directly comparable to published StrongREJECT benchmarks without calibration.

- **No ablation on attacker model choice**: The paper uses DeepSeek-R1 as the attacker model throughout. DeepSeek-R1 is a reasoning model with notably weaker safety alignment than alternatives. Given the attacker's role in planning and query generation, the framework's performance advantage may partially derive from this choice rather than architectural innovation. An ablation testing GPT-4o or Claude as the attacker would isolate the framework's contribution.

- **Algorithm-prose threshold inconsistency**: Algorithm 3 (Line 10) sets the success threshold at `score > 9.0`, while Section 3.5 states "a score greater than 8/10" triggers success marking. This discrepancy affects reproducibility.

- **Duplicate table row error**: Table 2 contains ActorBreaker row twice with identical values, indicating a proofreading oversight that further reduces confidence in the numerical presentation.

- **Lower attack diversity**: Figure 3 shows PLAGUE's diversity score (0.375) is lower than ActorBreaker's (0.433). While the paper notes this trade-off (higher ASR, lower diversity), it does not analyze implications for real-world red-teaming utility—whether finding fewer attack patterns limits vulnerability coverage.

- **Lifelong learning gains appear modest in isolation**: Table 3 shows the RSS (strategy retrieval) component improves SRE from 0.773→0.814 on o3, a 5.3% relative gain. While meaningful, this is smaller than the reflection component's contribution. The central "lifelong learning" claim rests on this mechanism, yet its standalone contribution is limited.

## Nice-to-Haves

- **Sequential learning validation**: Running attacks sequentially to populate memory and measuring per-strategy ASR gains would validate whether the lifelong learning mechanism provides compounding benefits over time rather than just static retrieval from initial seeds.

- **Attacker model ablation**: Testing PLAGUE with GPT-4o or Claude as the attacker would clarify how much performance depends on DeepSeek-R1's specific capabilities.

- **Cross-model strategy transfer analysis**: Testing whether strategies learned against Llama transfer to Opus would validate the generalizability claim for the memory bank.

- **Judge robustness check**: Evaluating a subset of successful attacks with a different judge model (or human annotation) would address concerns about Qwen3-specific evaluator bias.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"SRE and ASR used interchangeably confuses readers"**: While the terminology is imprecise, the paper clearly reports both metrics separately in Table 2 and explains the relationship in Section 3.2 and Appendix C.1. This is a clarity issue, not a substantive methodological flaw.

- **"Baseline modifications make comparisons unfair"**: The modifications to GOAT (per-round scoring) and Crescendo (removed backtracking limits) are disclosed in Section 4 under Baselines. Adding early stopping when "a high rubric score...is obtained in early iterations" is consistent with the budget-constrained evaluation protocol and benefits baselines rather than harming them.

- **"Orders of magnitude" claim verification**: The paper cites Li et al. (2024) for the "orders of magnitude" claim about multi-turn attacks. This is a cited external claim, not the authors' assertion. Verifying external citations is not the reviewer's role.

- **"ASR@K=2 applied inconsistently"**: The paper explicitly states K=2 for all experiments (Section 4) and clarifies that ActorBreaker's two actors parallel the ASR@2 metric. While presentation could be clearer, the protocol appears consistent.

- **"Responsible disclosure insufficient"**: The ethics statement acknowledges dual-use risks and argues for open access for safety research. While more discussion could be valuable, demanding specific disclosure protocols exceeds standard ICLR requirements.

- **"Theoretical contribution limited to integration"**: ICLR has accepted impactful systems papers whose primary contribution is careful integration and empirical demonstration. The criticism that integration alone lacks theoretical novelty is a category error—methodological contributions have precedent.

- **"Writing clarity issues"**: General complaints about repetition and prose quality are minor presentation issues that don't affect the core contribution.

## Novel Insights

The paper reveals that different models have qualitatively different vulnerability profiles requiring different attack components: reflection provides the largest gain on o3, while backtracking is most critical for Claude Opus 4.1 (Table 3). This suggests model safety mechanisms differ in ways that require tailored attack strategies—Claude's alignment appears more resistant to direct harmful queries but vulnerable to context manipulation, while o3 is more susceptible to refined reasoning about feedback. The finding that PLAGUE+Crescendo outperforms PLAGUE+GOAT on Claude but not other models further indicates that optimal attack architectures are target-specific. The lifelong learning component's modest standalone contribution (strategy retrieval adding ~5% relative improvement) suggests that in single-run evaluations, the memory bank's primary value may be warm-starting from seeded strategies rather than genuine learning; compounding benefits would require sequential evaluation across many goals, which the paper does not demonstrate.

## Suggestions

- Correct the baseline attribution error: change "previous best—GOAT" to "previous best—ActorBreaker" for the o3 comparison, or recalculate the improvement percentages correctly.

- Add a calibration experiment comparing modified SRE to original StrongREJECT on a held-out subset to quantify the systematic bias introduced by prompt changes.

- Clarify the success threshold discrepancy between Algorithm 3 (>9.0) and Section 3.5 (>8/10) in the camera-ready version.

- Remove the duplicate ActorBreaker row from Table 2.

- Report attacker model ablation: run PLAGUE with GPT-4o or another non-reasoning attacker to isolate the framework's architectural contribution from DeepSeek-R1's specific capabilities.

- Expand the discussion of diversity vs. ASR trade-offs: analyze whether lower diversity means PLAGUE discovers a smaller set of vulnerability types, potentially limiting its utility for comprehensive safety auditing.

- Consider adding a sequential evaluation showing cumulative ASR improvement as strategies accumulate in memory, which would validate the "lifelong learning" claim more directly than static retrieval experiments.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0, 2.0]
Average score: 2.5
Binary outcome: Accept
