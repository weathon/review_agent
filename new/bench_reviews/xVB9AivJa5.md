Now I have sufficient information from both the paper and calibration anchors. Let me write the final review.

## Summary

Blueprint-Bench introduces a benchmark for evaluating spatial reasoning in AI models through the task of converting apartment photographs into accurate 2D floor plans. The benchmark evaluates leading LLMs (GPT-5, Claude 4 Opus, Gemini 2.5 Pro, Grok-4), image generation models (GPT-Image, NanoBanana), and agent systems (Codex CLI, Claude Code) on 50 apartments, using a composite algorithmic scoring metric based on room connectivity and size rankings. The central finding is that most models perform at or below a no-image-input baseline, while humans substantially outperform them.

## Strengths

- **Novel and well-motivated task design**: Converting interior photographs to floor plans is a genuinely compelling probe of spatial reasoning that uses in-distribution inputs (photographs) but requires out-of-distribution spatial reconstruction. The paper explicitly motivates this contrast with ARC (Section 1), arguing that revealing a blind spot with familiar inputs is more informative than one with alien inputs.

- **Cross-architecture numerical comparison**: Evaluating 4 LLMs, 2 image generation models, and 2 agent systems on the same task with the same scoring metric enables the first direct numerical comparison of spatial intelligence across these architectures (Section 1, lines 150–160). This addresses a real gap — the paper correctly notes that image generation model announcements lack numerical intelligence benchmarks.

- **Striking empirical finding for format-compliant models**: For models that do follow the output format (GPT-5, Claude 4 Opus, Gemini 2.5 Pro, Grok-4, GPT-Image), Figure 5 shows they cluster near or below the no-image-input baseline. This is a concrete, quantitative demonstration that current models lack spatial reasoning capabilities even when given rich visual input.

- **Algorithmic scoring grounded in empirical testing**: The deterministic scoring pipeline (Section 2.3) is a reasonable alternative to LLM-as-judge evaluation. The authors explicitly tested LLM-based scoring and found it unreliable — LLMs incorrectly claimed adjacent rooms were connected and mis-ranked room sizes based on priors (Section 2.4, lines 276–283). This grounds the design choice in empirical evidence rather than mere preference.

- **Insightful qualitative analysis of agent behavior**: The trace analysis showing that Claude Code attempts iterative self-correction but fails to accurately assess its own outputs (Figure 8) provides concrete mechanistic insight into current agent limitations. The observation that Codex GPT-5 never even examines its own output is noteworthy.

## Weaknesses

### Fatal
None.

### Major

- **The benchmark conflates instruction following with spatial intelligence for some models, and this conflation is acknowledged but unresolved.** The paper states "Blueprint-Bench should test spatial intelligence, not instruction following" (Section 2.4), yet the scoring pipeline requires strict compliance with 9 formatting rules. The authors themselves attribute GPT-4o and NanoBanana's poor performance to "poor instruction following, leading to outputs that do not adhere to the rules and therefore cannot be scored by our algorithm" (Section 3, lines 324–327). Their scores are then presented alongside format-compliant models in Figure 5 as evidence of a spatial intelligence blind spot. The paper's defense — "at current model capabilities, we think this is the right tradeoff" — is a pragmatic argument for scoring robustness, not a resolution of the conflation. For models that do follow the rules, the scores meaningfully reflect spatial reasoning; for those that don't, the scores are uninformative about spatial capability. The paper should either (a) separate format-compliant from non-compliant results, or (b) provide some analysis of whether format-violating outputs contain reasonable spatial content. As is, presenting all models on the same leaderboard without this distinction overclaims what the scores measure.

- **The human baseline is too weak to support quantitative claims about human–AI gaps.** Figure 7 compares model performance to human performance, but the caption discloses that human data comes from "a subset of Blueprint-Bench (12 instead of 50)" apartments. Section 2.2 says "we gave the task to a human" (singular), indicating a single subject. A one-person, 12-apartment evaluation cannot support the paper's repeated claims that "human performance remains substantially superior" (Abstract) or "all models remain substantially below human performance" (Figure 7 caption). While the qualitative observation that the human achieved correct connectivity on all apartments is informative, the quantitative gap shown in Figure 7 is not well-grounded. This is one of the paper's three core claims, and it rests on insufficient evidence.

### Minor

- **The composite scoring weights (50/20/10/10/5/5) are stated without justification**, and the size-ranking-based room identification causes compounding penalties: a size-ranking error propagates into the edge-overlap component, which carries 50% weight. The authors acknowledge this (Section 2.4, lines 270–273) but the specific weights are not calibrated or ablated. Showing that model rankings are stable under reasonable weight perturbations would strengthen confidence in the benchmark's conclusions.

- **The "random baseline" is underspecified in the abstract and results sections.** The abstract claims "most models perform at or below a random baseline," but the term "random baseline" is not defined there. Section 2.2 describes it as "generating typical floor plans using LLMs and image generation models without any image input," which is not a random baseline in the conventional sense — it is a no-image-input baseline. The difference matters: a true random baseline (e.g., random graph with random room counts) might score differently, and the paper should be clearer about what the baseline represents and how it was constructed.

- **Sub-component scores are not reported separately.** The paper reports only the composite score, making it impossible to determine whether models struggle more with connectivity, room counting, or size ranking. Decomposing scores into sub-capabilities would transform "models are bad at spatial reasoning" into an actionable diagnostic.

- **No statistical tests are reported.** The paper states that some models "statistically perform better than the random baseline" (Section 3, lines 303–304) without specifying the test, p-values, or confidence intervals. Given 50 apartments and multiple epochs, appropriate statistical testing would strengthen the conclusions.

- **The conclusion about agents is slightly overgeneralized.** The paper concludes that "neither iterative refinement through agents nor specialized image generation models showed advantages over standard LLMs" (Section 4), but this is based on only two specific agent scaffolds. The paper's own analysis shows Codex GPT-5 didn't even attempt iteration, while Claude Code did attempt but failed — suggesting the failure is scaffold-specific, not iteration-specific. A more precise conclusion would be that these particular agent implementations failed to leverage iteration effectively.

### Trivial
None.

## Nice-to-Haves

- A small-scale human annotation assessing whether format-violating outputs (e.g., NanoBanana's) contain spatially reasonable content, even if format-compliance is low — this would establish whether format-compliant scores track spatial reasoning.
- Evaluating the human on all 50 apartments with multiple subjects and reporting inter-rater agreement.
- Ablation of the scoring weights to show robustness of model rankings.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"50 apartments is small for a benchmark"** (Harsh Critic): The paper explains its rationale for keeping the dataset size limited (preventing overfitting, keeping data private) and offers a community leaderboard. 50 apartments with ~20 images each is a reasonable starting point for a benchmark of this complexity. This is a generic critique that doesn't engage with the paper's design rationale.

- **"No information about apartment diversity"** (Harsh Critic): This is a valid concern but the paper's contribution is about evaluating model spatial intelligence, not apartment diversity analysis. Without evidence that diversity is actually low, this remains speculative.

- **"The extraction pipeline fails silently on non-compliant outputs"** (Harsh Critic): The paper explicitly discusses this as a known limitation and explains why they chose this tradeoff. This is already addressed in Section 2.4.

- **"Comparison to ARC is unearned"** (Harsh Critic): The comparison to ARC is about motivation (testing a blind spot with in-distribution vs. out-of-distribution inputs), not about matching ARC's scoring simplicity. This is an unreasonable standard for a first benchmark.

- **"The framework is for comparing floor-plan-format compliance, not spatial intelligence"** (Harsh Critic): This overstates the issue. For the majority of models tested (5 out of 7 non-agent models follow the rules reasonably well), the scores do reflect spatial reasoning. The conflation is real but partial, not total.

- **"Missing related works"** (Harsh Critic): Per my instructions, I do not flag missing related works.

- **"Missing appendix, missing proofs"** (Harsh Critic): Per my instructions, the parser strips appendices; these exist in the original submission.

- **Formatting/style nitpicks and typos** (Harsh Critic): Per my instructions, these are parser artifacts, not author errors.

## Novel Insights

The most interesting tension in this paper is the asymmetry between its strongest contribution and its weakest point: the task design itself is genuinely novel and compelling — it reveals that multimodal models given rich visual input still cannot reconstruct spatial layouts — but the evaluation framework cannot cleanly distinguish "the model doesn't understand the space" from "the model understands the space but cannot follow formatting instructions." This is not just a presentation problem; it means the benchmark's leaderboard conflates two distinct failure modes, and future work on this benchmark should prioritize decomposing these. The paper's honesty about this conflation is itself a contribution — it documents a real evaluation design challenge that future spatial reasoning benchmarks will face.

## Suggestions

- **Separate format-compliant and non-compliant model results** in Figure 5, or at minimum add a clear visual indicator distinguishing models whose low scores reflect format violations rather than spatial reasoning failures. This is the single most impactful change the authors could make.
- **Report sub-component scores** for each model (edge overlap, degree correlation, etc. separately) to make the benchmark diagnostic rather than merely summative.
- **Clarify the "random baseline" terminology**: Call it a "no-image-input baseline" consistently, and construct a true random baseline (random graphs with random room counts) for comparison.
- **Expand the human evaluation** to multiple subjects and all 50 apartments for the camera-ready version, even if results are preliminary.

## Calibration Anchors

- **Yk87CwhBDx** (Symbolic Graphics Programs, avg 7.33, Spotlight): Similar spatial-semantic reasoning benchmark. Stronger than Blueprint-Bench due to more rigorous evaluation, larger dataset, and novel SIT method. Blueprint-Bench has a more intuitively compelling task but weaker methodology.
- **WyEdX2R4er** (Visual Data-Type Understanding, avg 8.0, Poster): Similar "blind spot in VLMs" paper with 39 models evaluated. Much stronger experimental rigor. Blueprint-Bench's finding is similarly striking but less well-supported.
- **vNATZfmY6R** (KiVA, avg 7.0, Poster): Novel visual analogy benchmark with human comparison. Well-executed evaluation with developmental psychology grounding. Blueprint-Bench shares the "models fail at basic spatial reasoning" finding but has weaker evaluation methodology.
- **q3MYZQ3es8** (tBen, avg 4.0, Reject): Temporal logic benchmark that conflates reasoning with format compliance — the most directly comparable weakness pattern. Blueprint-Bench is clearly stronger than this due to its more compelling task and broader model evaluation.
- **VaUy5GZO3f** (Q-Bench-Video, avg 4.8, Withdrawn): Very similar profile — novel benchmark, only 3 human subjects flagged as insufficient, evaluation methodology concerns. Blueprint-Bench has a more creative task but comparable methodological weaknesses.
- **p30YulvDbj** (EEG Depression, avg 2.0, Reject): Tiny sample (58 samples), no baselines. Blueprint-Bench is clearly above this level with its novel task, diverse model evaluation, and algorithmic scoring.
- **AeGrf1uY0p** (FinBench, avg 4.75, Reject): Benchmark with 5 sub-capabilities and evaluation concerns. Similar concern about composite scoring without justification.

Blueprint-Bench sits between the well-executed spatial reasoning benchmarks (7+) and the weaker evaluation methodology papers (4-5). It is stronger than tBen and FinBench due to its more compelling task design and broader model evaluation, but its evaluation methodology issues (conflation, single-person human baseline, no sub-component analysis) prevent it from reaching the quality of the 7+ papers. It is roughly comparable to Q-Bench-Video (4.8) but with a more novel and interesting contribution, placing it slightly above.

## Evaluation

- **Originality**: High. The photo-to-floor-plan task is genuinely novel and well-motivated as a spatial intelligence probe.
- **Importance of research question**: High. Demonstrating that multimodal models lack spatial reasoning despite in-distribution inputs is valuable for the field.
- **Claims support**: Moderate. The core finding (models struggle with spatial reasoning) is well-supported for format-compliant models but the quantitative human-AI gap and the scores of format-violating models are not well-grounded.
- **Soundness of experiments**: Moderate. Good model coverage and algorithmic scoring, but weak human baseline, no sub-component analysis, and no statistical tests.
- **Clarity**: Good. The paper is clearly written and transparent about its limitations.
- **Value to community**: Moderate-to-high. The benchmark fills a real gap, but its current evaluation limitations reduce its utility for precise model comparison.

## Score and Decision

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>

The task design is genuinely novel and the core finding that current AI models struggle with spatial reasoning is important and likely correct. However, the evaluation framework has two significant issues that partially undermine the paper's central claims: (1) the conflation of instruction following with spatial intelligence for some models is acknowledged but not resolved, and (2) the human baseline (single subject, 12 apartments) is too weak to support the quantitative human–AI comparison that is one of the paper's three core claims. These are not fatal — the finding holds for format-compliant models — but they prevent the benchmark from being the authoritative numerical framework the paper claims it is. The paper is borderline: with a proper human evaluation and separation of format-compliant/non-compliant results, it could be a strong contribution.