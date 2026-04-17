Now I have a good calibration set. Let me synthesize the final review.

**Calibration Summary:**
- LiveCodeBench (similar competitive programming benchmark, more comprehensive evaluation with contamination analysis): 6, 8, 5, 6 → Accept (Poster) → avg ~6.25
- MHPP (harder code benchmark, incremental): 3, 3, 5, 6 → Reject → avg ~4.25
- BigCodeBench (comprehensive benchmark with diverse tasks, strong methodology): 10, 10, 8, 8 → Accept (Oral) → avg 9
- SWE-Bench+ (benchmark improvement with contamination analysis but limited scope): 3, 3, 6, 3 → Reject → avg ~3.75
- Data contamination longitudinal analysis paper: 8, 8, 6, 5 → Accept (Poster)

AetherCode falls somewhere between LiveCodeBench and MHPP in terms of novelty - it's a solid benchmark paper with real problems sourced from premier competitions and careful test case construction, but it has the contamination issue and somewhat incremental novelty (similar to several existing competition-level benchmarks). Compared to LiveCodeBench which had contamination analysis and broader evaluation scope, AetherCode is weaker on methodological rigor. Compared to BigCodeBench which had far more diverse evaluation scenarios, AetherCode is narrower.

Let me now write the final review.</think>Now I have sufficient calibration context. Let me write the final consolidated review.

## Summary

AetherCode introduces a benchmark of 456 competitive programming problems sourced from premier global competitions (OI and ICPC series), paired with high-quality test suites constructed via a hybrid Generator-Validator agent system and expert annotation. The benchmark aims to address two shortcomings of existing code reasoning benchmarks: insufficient difficulty/scope and evaluation bias from low-quality test cases. Evaluation of 17 reasoning and non-reasoning models shows even the best model (o4-mini-high) achieves only 35.5% Pass@1, with reasoning models substantially outperforming non-reasoning ones.

## Strengths

- **Well-motivated and genuinely challenging problem source.** Drawing from IOI, ICPC World Finals/Regionals, NOI, USACO, and CCPC provides problems that are qualitatively different from the LeetCode/CodeForces-sourced benchmarks that dominate the landscape. The inclusion of "Extreme" difficulty problems (unsolved by any human contestant during the competition) is a valuable addition that genuinely pushes evaluation ceilings.

- **Serious investment in test case quality.** The hybrid approach—G-V Agent system achieving 89.9% TNR, followed by expert annotation from 67 competitive programmers (many with Codeforces ratings >2000) and an elite review team with ICPC gold medals—is far beyond what prior benchmarks have attempted. The formalization of test suite quality as TPR/TNR against a collected solution pool is a clean and useful framing.

- **Comprehensive and contemporary model evaluation.** Testing 17 current models (o4-mini-high, Gemini 2.5 Pro/Flash, Seed-1.6-Thinking, DeepSeek-R1, Qwen3, Claude, etc.) with Pass@1/2/4 metrics provides a valuable snapshot of frontier model capabilities on genuinely difficult problems. The finding that even top models only solve ~35% of problems is informative.

- **Rich categorization enabling fine-grained analysis.** The 10 major + 144 subcategory algorithmic taxonomy, difficulty stratification by human solve rates, and temporal metadata support detailed diagnostic evaluation. The per-category analysis (Table 4) revealing specific weaknesses (e.g., GPT-4.1's relative weakness in Mathematics despite strong overall non-reasoning performance) provides actionable insights.

## Weaknesses

### Major:

- **Data contamination remains almost entirely unaddressed.** The paper sources problems from IOI, ICPC, NOI, USACO, and CCPC—competitions whose problems, editorials, and solutions are widely disseminated online after each event and are almost certainly present in web-scale training corpora. Section 2.1 mentions collecting contest dates "for decontamination purposes" and Section 4.2 criticizes other benchmarks for contamination risk, yet the paper never reports any actual contamination analysis: no filtering of problems by date relative to model training cutoffs, no overlap check with known training corpora, and no time-stratified performance analysis. Given that 400 of the 456 problems are from 2024 (Table 2), and many evaluated models have training data extending well into 2024, this is a significant gap. The contamination analysis paper by [Vendrow et al.] demonstrates that contamination on Codeforces problems is statistically detectable, making this particularly relevant. Without such analysis, all absolute performance numbers may be inflated and not interpretable as measuring genuine reasoning capability on unseen problems.

- **The 100% TPR/TNR claim is overstated relative to the methodology.** The test suite achieves perfect classification only on the finite collected solution set (at least 5 correct + 20 incorrect solutions per problem, plus additional expert-written ones). The paper presents this as establishing near-perfect evaluation reliability ("guaranteeing exceptional accuracy and reliability," "first benchmark that sets such a high standard"), but there is a fundamental circularity: the test cases were explicitly constructed and augmented to reject the collected incorrect solutions (Section 2.3.3: "experts were tasked with constructing targeted test cases specifically designed to fail the various incorrect solutions we had collected"), and then validated on those same solutions. The negative space of possible competitive programming solutions is enormous, and 100% TNR on a few dozen incorrect implementations per task does not guarantee coverage of novel LLM failure modes. This matters because test case quality is one of the paper's two central selling points. The actual test case construction process is strong; it is the framing that overclaims.

- **No empirical demonstration that AetherCode's higher-quality test suites change model rankings or assessments.** The paper's central argument is that low-quality test cases in prior benchmarks introduce evaluation bias. Yet there is no experiment comparing model scores or rankings when evaluated with AetherCode's test suites versus naive or mutation-based test cases (e.g., from CodeContests or LiveCodeBench). Without this ablation, the claim that test case quality materially affects evaluation outcomes—while plausible—is empirically unsupported by this paper, reducing it to an assertion.

### Minor:

- **Missing statistical uncertainty.** With only 4 samples per problem for Pass@4, and potentially few problems in the "Extreme" category and some algorithmic subcategories, no confidence intervals or bootstrap estimates are reported. Observed differences of a few percentage points between models (e.g., 2.7% vs 4.0% on Hard problems) may not be statistically meaningful.

- **Difficulty and category are confounded in the per-category analysis.** Table 4 shows per-category Pass@1 scores, but the paper acknowledges that some categories (e.g., Trees, Computational Geometry) may simply contain harder problems. Without a within-difficulty per-category analysis, it is unclear whether certain algorithmic areas are inherently harder for models or just happen to contain more Extreme-level tasks.

- **The paper claims to want to "study how the difficulty for LLMs differs from the difficulty in the eyes of humans" (Section 2.2) but never performs this analysis.** No comparison of LLM solve rates versus human contest solve rates is provided, which would have directly delivered on this stated motivation and quantified the claimed "gap between LLMs and elite human programmers."

- **Evaluation protocol under-specification.** The main text only states max output length (32,768 tokens) and Pass@1/2/4 with 4 samples, deferring details to Appendix A. Key information missing from the main text includes: prompting format, whether single-shot or multi-turn evaluation is used, language constraints and how language mismatches are handled (the paper notes GLM-4.5 outputs Python when instructed for C++), sampling parameters, and compilation/execution environment details. For a paper centered on rigorous contest-grade evaluation, this level of under-specification weakens reproducibility.

### Trivial:

- **Repetitive writing.** The motivation about test case quality issues and CodeForces compliance risks is stated nearly verbatim in both the Introduction and Section 2.3. A single concise statement with a forward reference would suffice.

## Nice-to-Haves

- A direct head-to-head comparison evaluating the same models on both AetherCode and at least one existing benchmark (e.g., LiveCodeBench Pro, CodeELO) to demonstrate whether AetherCode provides discriminative power beyond what existing benchmarks already offer.
- A test-case ablation experiment showing model scores with automated-only vs. full expert-augmented test suites, to empirically validate the claim that test case quality changes evaluation outcomes.
- Pass@8 or Pass@16 evaluation for top-tier models, given the observation that they show greater improvement with increased sampling.
- Inter-annotator agreement statistics for difficulty ratings and algorithmic category assignments.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"Only C++ evaluation limits generalizability"** — While true that evaluation is C++-only, this is standard for competitive programming benchmarks (ICPC/IOI are primarily C++ competitions). Evaluating in the competition's native language is appropriate for the paper's stated scope. Removed as criticism outside stated scope.

- **"Novelty is incremental relative to OJBench, USACO Bench, ICPC-Eval"** — The paper demonstrates meaningful differences from these (broader competition coverage, latest problems, self-contained test cases rather than relying on external judges). While the overall approach is similar, the scope and test case quality methodology represent a genuine contribution. This is better framed as a minor concern than a fatal one.

- **"The G-V Agent system is borrowed from prior work (Wang et al., 2025b), reducing novelty"** — Building on prior work for a component while making novel contributions in integration and expert augmentation is standard practice. This is not a weakness.

- **"Problems from 2024 may be contaminated"** — This overlaps with the contamination concern already addressed above. The specific claim that 2024 problems are contaminated is speculative without evidence; the legitimate concern is the absence of any contamination analysis, which is already captured.

- **"Reproducibility concerns about undisclosed hyperparameters and training details"** — This is a benchmark paper, not a training paper. The evaluation settings that matter (prompting, sampling, execution environment) are legitimate concerns covered under evaluation protocol, but general reproducibility nitpicks are removed per the hard rules.

## Novel Insights

The observation that even top reasoning models achieve only ~35% Pass@1 on problems that top-tier human competitors solve at much higher rates is meaningful, but the paper misses the opportunity to quantify this gap concretely by comparing against actual human contest statistics—something its own metadata (human solve rates per problem) would enable. The finding that Claude models tend toward correct-but-inefficient algorithms rather than incorrect-but-fast ones is an intriguing diagnostic signature that distinguishes model failure modes beyond simple accuracy metrics. The TPR/TNR framing for test suite quality, while overclaimed, is a genuinely useful conceptual contribution that the community should adopt—with appropriate statistical caveats.

## Suggestions

- **Conduct and report a contamination analysis.** At minimum, time-stratify evaluation results showing model performance on 2025-only problems (56 problems) versus 2024 problems, and perform n-gram or semantic similarity checks against known pre-training corpora. This would substantially strengthen the paper's claims about genuine reasoning gaps.
- **Reframe the 100% TPR/TNR claim with appropriate caveats.** Explicitly state this is on the collected solution set, report the distribution of solutions per problem, and discuss residual risk from unseen failure modes.
- **Add a test-case ablation.** Evaluate a subset of models using only the G-V Agent's test cases (without expert augmentation) and compare rankings/scores to demonstrate the empirical impact of expert test cases on evaluation outcomes.
- **Provide per-difficulty counts.** Report how many problems fall into each difficulty tier to enable readers to assess the statistical reliability of per-category comparisons.

## Score and Decision

**Calibration anchors:**

- **LiveCodeBench** (competitive programming benchmark, contamination analysis, broader evaluation scope, accepted as poster): scores 6, 8, 5, 6 → avg ~6.25. AetherCode is comparable in scope but weaker on contamination analysis (LiveCodeBench explicitly does time-segmented evaluation to address contamination) and broader evaluation modalities (LiveCodeBench evaluates code generation, self-repair, execution, and test prediction, while AetherCode only evaluates code generation).

- **BigCodeBench** (comprehensive code benchmark with diverse task design, accepted as oral): scores 10, 10, 8, 8 → avg 9. AetherCode is less novel and comprehensive than BigCodeBench, which introduced a genuinely new evaluation dimension (diverse function calls from 139 libraries).

- **MHPP** (harder code benchmark, incremental contribution, rejected): scores 3, 3, 5, 6 → avg ~4.25. AetherCode is substantially better than MHPP: it has larger scale (456 vs 210 problems), much stronger test case methodology (30k+ solutions with expert validation vs ~14 tests average), and more comprehensive model evaluation. AetherCode is clearly above MHPP.

- **SWE-Bench+** (benchmark improvement addressing contamination and test quality, limited scope, rejected): scores 3, 3, 6, 3 → avg ~3.75. AetherCode is more substantial, with original data collection and comprehensive evaluation rather than just an audit of an existing benchmark.

AetherCode is a solid, well-motivated benchmark with real engineering effort behind the test case construction. It falls below LiveCodeBench mainly due to the missing contamination analysis and lack of broader evaluation modalities. The overclaiming on TPR/TNR and the missing ablation on test case quality are notable but not fatal weaknesses. The dataset itself is a genuine contribution. Overall, this sits between MHPP and LiveCodeBench, closer to LiveCodeBench but somewhat weaker due to the missing contamination rigor.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>