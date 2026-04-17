Now I have enough calibration data. Let me compile the final review.

## Summary

AetherCode introduces a competitive programming benchmark sourcing 456 problems from premier competitions (IOI, ICPC, and related series) rather than online platforms like CodeForces or LeetCode. Its two main claims are: (1) these harder, broader-scope problems better challenge LLMs, and (2) its hybrid test-case construction (Generator-Validator agent + expert annotation), validated to achieve 100% TPR/TNR on a collected solution corpus, provides more reliable evaluation than prior benchmarks.

## Strengths

- **Addresses a genuine and well-motivated gap.** The paper clearly articulates two problems—insufficient difficulty/scope in existing benchmarks and evaluation bias from low-quality test cases—and backs these with concrete evidence (e.g., LiveCodeBench's reliance on LeetCode/AtCoder, CodeForces-based benchmarks' compliance and rate-limit issues). This is a real contribution to the field.

- **Serious and novel attention to test-case quality.** Conceptualizing test suites as binary classifiers with TPR/TNR metrics is a principled methodological advance over prior benchmarks' ad hoc approaches. The hybrid G-V Agent + expert annotation pipeline, including a dedicated review team of ICPC gold medalists, represents substantially more effort than any prior benchmark in this space. Even if the 100% figure is context-dependent, the process is meaningfully more rigorous.

- **Rich problem metadata and categorization.** The multi-dimensional taxonomy (difficulty levels, 10 major / 144 sub-category algorithmic tags, temporal metadata, competition scope) enables fine-grained diagnostic analysis that most competing benchmarks lack.

- **Comprehensive evaluation of 17 frontier models.** The breakdown by difficulty, algorithmic category, and Pass@k settings provides a detailed empirical landscape of current model capabilities. Key findings—reasoning models consistently outperform non-reasoning ones, top models show greater exploration potential from increased sampling—are well-supported by the data.

- **Practical and community-friendly design.** Self-contained test cases (no dependency on external judging services), open-release intent, and recent problem dates (2024–2025) address real deployment and compliance needs.

## Weaknesses

### Major

- **The 100% TPR/TNR claim overgeneralizes from the construction process to global quality guarantees.** Section 2.3.1 frames TPR/TNR on the collected solution set as evidence for benchmark reliability, and the conclusion claims "exceptional accuracy and reliability in evaluation" and "a new standard." However, the construction process is partially circular: experts in Section 2.3.3 explicitly "construct targeted test cases specifically designed to fail the various incorrect solutions we had collected." Achieving 100% TNR on a set that the test cases were hand-crafted to catch is expected, not evidence of comprehensive coverage for unseen failure modes. Furthermore, the entire TPR/TNR evaluation is conducted on *human-written* solutions, while the benchmark's primary purpose is evaluating *LLM-generated* code. LLMs produce distinctive error patterns (format violations, hallucinated constructs, partial program structures) not represented in the collected human solution corpus. Given that 17 models were evaluated, the absence of any analysis of how many LLM-generated incorrect solutions might pass the test suite is a notable gap for a claim this central.

- **No decontamination analysis despite emphasizing it as a key advantage.** The paper positions AetherCode as addressing the "significant risk of data contamination" in prior benchmarks (Section 4.2) and collects competition dates "for decontamination purposes" (Section 2.1). Yet no actual decontamination procedure is implemented or analyzed—no comparison of model performance on 2024 vs. 2025 problems, no n-gram overlap detection, no discussion of which models may have seen which problems. Given that these are problems from well-known public competitions (IOI, ICPC, NOI, USACO) with publicly available editorials and solutions, contamination risk is non-trivial. Without any empirical quantification, the claim to address contamination remains unsubstantiated where it matters.

- **The paper's central framing about a "significant gap compared to top human experts" is asserted but never quantified.** The introduction motivates the work by asking whether LLMs have "mastered" competitive programming and asserts a "significant gap still exists between the performance of LLMs and top-tier human competitors." However, despite collecting "human contestant performance data" (Section 2.1), the paper never reports human baselines on the same benchmark. There is no per-problem or per-difficulty comparison of model Pass@k against human solve rates. Without this, the gap claim is intuitive but empirically unsupported within the paper itself.

### Minor

- **Human difficulty labels are not validated against LLM difficulty.** Section 2.2 states the classification is "judged entirely from the perspective of humans" and promises to "provide a perspective to study how the difficulty for LLMs differs from the difficulty in the eyes of humans." This analysis never appears. The paper uses "Extreme" difficulty as a rhetorical category for LLM evaluation without showing that human-perceived difficulty ordering correlates with model performance ordering across problems.

- **Per-category analysis does not control for difficulty distribution.** Table 4 compares models across 10 algorithmic categories, but the paper acknowledges that some categories (like Trees) may be disproportionately hard. Without normalizing for difficulty or reporting problem counts per category, it is unclear whether performance differences reflect algorithmic demands or simply different difficulty mixes.

- **Evaluation protocol lacks important details.** While max output length (32K tokens) and sample count (4) are specified, the paper does not describe prompt format, language specification (C++ is implied but not stated), temperature/sampling parameters, or whether reasoning models are given additional thinking tokens. The finding that GLM-4.5 often outputs the wrong programming language underscores this gap.

- **No empirical demonstration that improved test cases change model rankings.** The paper critiques the test-case quality of prior benchmarks but does not show that using AetherCode's test suites actually produces different conclusions than, say, CodeContests'. An ablation comparing rankings under different test suites would strengthen the claim that test-case quality matters.

### Trivial

- The paper inconsistently refers to the benchmark as "AetherCode" and "AetherCode" (capitalization). Minor typo: "deteiled" in Section 2.3.

## Nice-to-Haves

- **Human vs. LLM difficulty correlation analysis.** The authors have human solve-rate data; a scatter plot comparing per-problem human solve rates to LLM Pass@1 would directly address the paper's stated goal and central claim about the gap.

- **Temporal decontamination analysis.** Compare model performance on 2024 vs. 2025 problems to detect contamination. Report which models have training data cutoffs before/after problem publication dates.

- **Ablation on test-case quality.** Evaluate a subset of models using only the G-V Agent's test cases (89.9% TNR) vs. the full hybrid suite, to quantify the impact of expert annotation on evaluation outcomes.

- **Per-category problem counts and confidence intervals** in Table 4 and difficulty breakdowns, particularly for small categories like Trees.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Novelty is primarily in data curation, not methodology"** (from human finder, citing LiveCodeBench reviewer): This is a category error for benchmark papers. Data curation IS the methodology for benchmarks. The TPR/TNR framework and the G-V Agent pipeline are genuine methodological contributions.

- **"The benchmark is small (456 problems)"** (from human finder, citing MHPP reviewer): 456 problems is larger than CodeContests (165) and USACO Bench (307), and comparable to CodeELO (387). The comparison to APPS (5000) is misleading because APPS includes many easy problems. AetherCode's problem count is within the range for a high-quality, difficulty-focused benchmark.

- **"Limited evaluation beyond Pass@k"** (from human finder, citing Beyond Correctness/RACE): This is scope creep. AetherCode is explicitly a competitive programming benchmark where Pass@k is the standard and most meaningful metric. Requesting additional dimensions like self-repair or efficiency analysis goes beyond the paper's stated scope.

- **"Language scope should be acknowledged"** (from spark): This is a minor observation. The competitive programming context makes C++ the dominant expected language, and this is standard practice in the community.

- **"Reproducibility concerns about the expert annotation process"** (from harsh reviewer): Benchmark construction often involves expert curation that is not replicable step-by-step. This is inherent to high-quality benchmark design and not a unique weakness. The key concern—that the process can't be easily repeated—is acknowledged but is not a flaw; it's a tradeoff for quality.

## Novel Insights

The conceptualization of test suites as binary classifiers evaluated via TPR/TNR is a genuinely useful framing that the broader code evaluation community should adopt. Even though the specific 100% achievement is partially circular, the framework—separating test case *correctness* (TPR) from *coverage* (TNR) and measuring them independently—provides a principled vocabulary for discussing and comparing benchmark quality. The finding that reasoning models' Pass@4 gains over Pass@1 scale with model capability (11% for o4-mini-high vs. 8% for weaker models) suggests these models are exploring diverse solution spaces rather than just getting luckier with more samples.

## Suggestions

- **Tone down the TPR/TNR claims** to explicitly acknowledge that 100% is achieved "on the collected solution corpus" and does not guarantee coverage of novel LLM failure modes. Add even a small analysis (e.g., manually checking 50 LLM-generated incorrect solutions for false negatives) to validate generalization.

- **Add a contamination analysis** by stratifying results by problem date (2024 vs. 2025) and reporting model-specific training cutoff dates. Even a simple table showing performance on problems published before vs. after each model's cutoff would substantially strengthen trust in the results.

- **Include per-problem human solve rates** alongside model Pass@1 in the appendix or leaderboard, fulfilling the paper's own stated goal of comparing human and LLM difficulty.

## Score and Decision

**Calibration comparison:**

- **LiveCodeBench** (Accept-Poster, avg ~6.25): Addressed code evaluation with live updates and contamination analysis. AetherCode makes a stronger test-case quality contribution (TPR/TNR framework, expert curation) but a weaker contamination analysis (none) and lacks the evaluation breadth (no self-repair/execution tracks). Roughly comparable in terms of contribution scope.

- **CodeMMLU** (Accept-Poster, avg ~5.5): Large benchmark with contamination concerns but useful diversity. AetherCode is more methodologically rigorous on test-case quality but has similar contamination concerns and smaller size.

- **MHPP** (Reject, avg ~4.25): Similar "harder benchmark" motivation, but smaller (210 problems), no test-case quality innovation, no contamination analysis. AetherCode is clearly stronger on methodology and scope.

AetherCode makes a real contribution: the problem sourcing from premier competitions, the TPR/TNR framing, and the G-V Agent + expert hybrid pipeline are substantive. The 17-model evaluation provides significant empirical value. However, the overgeneralization of the 100% TPR/TNR claim, the absence of decontamination analysis, and the missing human baseline for the "gap" claim are meaningful weaknesses that undermine the strongest claims of the paper.

**Score: 5.5** — A solid benchmark contribution with clear motivation and substantial engineering effort, but with overclaimed test-suite quality guarantees and missing key analyses (decontamination, human baseline) that are necessary for the paper's own stated goals.

<pineapple>5.5</pineapple>

<orange>Reject</orange>