# Spatial Reasoning Benchmark Critique Template
## Based on Recurring Weaknesses in Evaluated Benchmarks

This template synthesizes common critique patterns from reviews of benchmark papers, adapted for spatial reasoning benchmarks like Blueprint-Bench.

---

## Section 1: Metric Design & Validation

### Standard Weakness Pattern
"The paper proposes evaluation metrics but fails to demonstrate they correlate with human judgment of quality."

### Specific Critiques You Could Raise

**Critique 1.1 - Metric Validity**
```
The paper introduces spatial accuracy metrics (e.g., room boundary precision,
furniture placement F1) but provides no evidence these metrics correlate with
human judgment. A systematic comparison between metric scores and expert
human evaluations on the same test cases is required. Specifically:

- Do floor plans with higher metric scores receive higher human quality ratings?
- Is there a monotonic relationship or threshold effects?
- Which metric components drive human preference?

Without this validation, we cannot be confident the benchmark optimizes for
the right objective.
```

**Critique 1.2 - Metric Ceiling/Floor Effects**
```
The reported metric distributions across baseline models should be examined.
If all models score above 80% on the main metrics, the benchmark has ceiling
effects and cannot discriminate between systems. Include:

- Minimum, maximum, median metric scores across models
- Metric score range (max - min) to verify discriminative power
- Comparison of metric variance: variance should be large enough to
  meaningfully rank models
```

**Critique 1.3 - Per-Task vs. Aggregate Metrics**
```
Following the FIOVA criticism "AutoCQ only provides event-level evaluation,"
verify that metrics capture all relevant aspects of spatial reasoning. If metrics
only measure room-level accuracy, can they detect failures in inter-room
relationships, traffic flow, or furniture placement consistency?

Recommended: Provide metrics at multiple granularities (room-level, layout-level,
spatial-relationship level) and show all contribute to final score.
```

---

## Section 2: Ground Truth Quality & Annotation

### Standard Weakness Pattern
"The paper lacks documentation of annotation quality, inter-annotator agreement, and procedures for resolving disagreements."

### Specific Critiques You Could Raise

**Critique 2.1 - Annotation Agreement**
```
No inter-annotator agreement statistics are reported (Cohen's kappa,
Krippendorff's alpha, or Fleiss' kappa for multiple annotators). This is critical
because:

- Low agreement indicates ambiguous task definition or poor annotator training
- Unreliable ground truth contaminates the benchmark
- Readers cannot assess benchmark quality

Provide agreement statistics for:
- Overall floor plan layout correctness
- Individual room dimensions
- Spatial relationships between rooms
- Furniture placement (if applicable)
```

**Critique 2.2 - Annotation Protocol Clarity**
```
The annotation guidelines are insufficiently detailed. Specifically:

- How were annotators trained? What error rate is acceptable?
- For ambiguous cases (e.g., partially visible rooms), what protocol resolves
  disagreement? (majority vote, adjudication by expert, exclusion)
- Were annotators blind to model outputs? (To prevent bias)
- How were edge cases (unusual layouts, incomplete photos) handled?

This level of detail is essential for reproducibility and for assessing
ground truth reliability.
```

**Critique 2.3 - Ground Truth Construction Method**
```
The paper should clarify whether ground truth floor plans were created by:

(a) Human annotators viewing original photos directly [acceptable]
(b) Consensus from multiple annotators [requires agreement statistics]
(c) Synthesis from human descriptions by automated system [unacceptable -
    introduces systematic errors]
(d) LLM generation from image descriptions [unacceptable]

If method (c) or (d), the benchmark is fundamentally unreliable. The FIOVA
review demonstrated that LLM synthesis of conflicting annotations produces
hallucinations. For example: "Human 1 says room is 4m wide, Human 2 says 3.5m.
LLM generates 3.75m (averaging) without access to actual photo." This
contaminates the benchmark.

Recommendation: Ground truth must be created by qualified human annotators
with direct access to source images.
```

**Critique 2.4 - Handling of Ambiguous Cases**
```
How were cases handled where:
- Photo shows only partial room (corner cut off by frame)
- Room occluded by furniture or shadows
- Ambiguous spatial relationship between adjacent rooms
- Multiple valid floor plan interpretations

Were these excluded, included with annotator uncertainty, or forced to single
answer? Document rate of such cases and methodology.
```

---

## Section 3: Dataset Composition & Generalization

### Standard Weakness Pattern
"The benchmark focuses on narrow domain (e.g., only financial data in RD2Bench) and generalization is not demonstrated."

### Specific Critiques You Could Raise

**Critique 3.1 - Dataset Diversity Documentation**
```
Dataset should be explicitly characterized across key dimensions:

Building Type Distribution:
- Residential (single-family: __%, apartments: __%, condos: __%)
- Commercial (offices: __%, retail: __%, mixed-use: __%)
- Institutional (schools: __%, hospitals: __%)
- Industrial/other: ___%

Photography Conditions:
- Multiple angles per room: yes/no
- Lighting variation: indoor natural, indoor artificial, mixed
- Photo distance from walls: range in meters/feet
- Photo quality: resolution range, blur/artifacts present

Layout Complexity:
- Simple rectangular rooms: ___%
- L-shaped/irregular: ___%
- Multi-story with stairs: ___%
- Open concept (ambiguous boundaries): ___%

Explicitly report these proportions. If >80% of dataset is one building type,
generalization cannot be claimed.
```

**Critique 3.2 - Out-of-Domain Testing**
```
While the main dataset may focus on residential, the paper should evaluate
on held-out data:

- Does a model trained on residential apartments generalize to single-family
  houses?
- Does a model trained on well-lit photos generalize to shadowy office spaces?
- How much performance drops when spatial layouts are more complex?

Report absolute drop in key metrics and discuss whether it's acceptable for
benchmark's purpose.
```

**Critique 3.3 - Statistical Representativeness**
```
For each category reported in 3.1 (building type, photo conditions, complexity),
verify:

- Sample size per category sufficient for statistical analysis (recommend n>30
  per category for reliable estimates)
- Representative of target domain (are complex layouts overrepresented?
  underrepresented?)
- Potential biases in collection process (e.g., professional photos only,
  curated for visual appeal)
```

---

## Section 4: Task Difficulty Calibration

### Standard Weakness Pattern
"All models score >85%, preventing meaningful differentiation (ceiling effect)."

### Specific Critiques You Could Raise

**Critique 4.1 - Difficulty Distribution**
```
Report distribution of task difficulty:

Easy tasks (human accuracy >95%): ___%
Medium tasks (human accuracy 70-95%): ___%
Hard tasks (human accuracy <70%): ___%

Verify:
- All three difficulty levels represented (don't front-load easy tasks)
- Hardest tasks still measure meaningful differences (no model gets <20%,
  else likely noise)
- Distribution allows ranking models meaningfully (roughly uniform, or
  intentionally stratified)
```

**Critique 4.2 - Baseline Human Performance**
```
Essential metric missing: What do human experts score on this benchmark?

- Hire 3-5 domain experts (architects/spatial reasoning specialists)
- Have them complete benchmark tasks blind to model outputs
- Report their accuracy scores

This provides:
- Ceiling for meaningful performance (models can't exceed human)
- Validation that tasks are actually solvable
- Calibration point for interpreting model scores

If expert humans score 95%+, task is too easy. If <50%, ambiguous or poorly
defined. Target range: 60-85% for peak discrimination.
```

**Critique 4.3 - Metric Score Distribution Across Models**
```
For each metric, report:

Minimum model score: ___%
25th percentile: ___%
Median: ___%
75th percentile: ___%
Maximum: ___%
Standard deviation: ___%

Verify range (max - min) is large enough to rank systems. If all scores cluster
in 80-95%, benchmark is too easy.
```

---

## Section 5: Hyperparameter Selection & Evaluation Methodology

### Standard Weakness Pattern
"Hyperparameters for evaluation metrics chosen without justification; ablation studies missing."

### Specific Critiques You Could Raise

**Critique 5.1 - Hyperparameter Transparency**
```
For any tunable components of the evaluation (metric thresholds, distance
tolerances, weighting schemes), document:

Example: "Room boundary accuracy: walls within 0.5m of human annotation count
as correct"
- Why 0.5m? Justify threshold choice
- Sensitivity analysis: what happens at 0.3m? 0.7m? 1.0m?
- Show that metric score rankings remain stable across reasonable parameter
  ranges

Failure case from reviews: Domain quantization paper chose quantization
hyperparameters without justification. Different parameter values completely
changed results, yet selection rationale was never explained.
```

**Critique 5.2 - Evaluation Robustness**
```
Do evaluation metrics require task-specific tuning? If yes, this is a major
weakness (see Token Statistics Transformer critique).

- Same metric threshold should work across all spatial reasoning tasks
- If different thresholds needed for different building types, metric is
  not generalizable
- Verify metric performance ranking is stable across test set subgroups
  (residential vs. commercial, simple vs. complex layouts)
```

**Critique 5.3 - Ablation Studies**
```
Missing ablations on:
- Impact of each metric component on final scores (e.g., remove furniture
  placement metric, remeasure)
- Effect of aggregation method (do equal weights for all metrics make sense?
  try alternative weightings)
- Impact of threshold choices through sensitivity analysis

The Delta (contrastive decoding) review noted: "Ablation studies on technique
choices are missing. For example, why choosing masking tokens instead of other
variations? what's the best masking strategy?"

Apply same scrutiny to benchmark evaluation design.
```

---

## Section 6: Comparative Evaluation & Fairness

### Standard Weakness Pattern
"Baseline methods evaluated under different conditions than proposed methods (confounded comparison)."

### Specific Critiques You Could Raise

**Critique 6.1 - Controlled Evaluation**
```
All models should be evaluated identically:

- Same input photo resolution
- Same time limit (if real-time evaluation)
- Same preprocessing (no custom per-model tuning)
- Same evaluation protocol (metrics computed identically)
- Same hardware/environment (if measuring efficiency)

If any model received task-specific optimization (e.g., hyperparameter tuning
on test set), note this and consider whether it provides unfair advantage.
```

**Critique 6.2 - Baseline Strength**
```
Are baseline methods actually competitive?

- Use state-of-the-art spatial reasoning models as baselines
- Don't use weak baselines (e.g., simple heuristics) that benchmarks can easily
  beat
- Compare against concurrent work, not just prior work

FIOVA's weakness: Only 6 models evaluated, missing recent commercial models
(Gemini-1.5, GPT-4V) that might have different strengths/weaknesses.
```

**Critique 6.3 - Performance Reporting**
```
Report metrics at multiple granularities:

- Overall benchmark score
- Per-building-type breakdown (does model generalize?)
- Per-difficulty breakdown (does model struggle with hard cases?)
- Per-metric breakdown (which aspects are weak?)

This prevents hiding weaknesses in aggregate score. RD2Bench reviewed papers
showed overall score of 0.91 but breakdown revealed model performed terribly on
some tasks.
```

---

## Section 7: Annotation & Evaluation Costs

### Standard Weakness Pattern
"Computational and annotation costs missing; unclear how practical benchmark is."

### Specific Critiques You Could Raise

**Critique 7.1 - Annotation Effort**
```
Report:
- Total annotation hours required
- Hours per task
- Cost (if human annotators paid)
- Hardware required for annotation tools
- How much data can one annotator handle per day?

Why matters: If annotating 1,000 floor plans requires 500 annotation hours at
$50/hour = $25,000, this limits benchmark reproducibility and future extensions.
```

**Critique 7.2 - Evaluation Computational Cost**
```
For automated metrics:
- Inference time per test case (seconds)
- Total evaluation time for full benchmark
- Hardware required (GPU? Memory?)
- Cost to run evaluation once

This matters for:
- Reproducibility (can other researchers run evaluation?)
- Practical adoption (is benchmark reasonable to use?)
- Iterative development (can researchers use benchmark for model development?)
```

**Critique 7.3 - Sustainability**
```
How will benchmark be maintained?

- Plan for identifying/fixing annotation errors
- Plan for adding new data (seasonal updates? geographic expansion?)
- Who maintains the benchmark and for how long?
- What happens if errors found in ground truth?

FIOVA revealed systematic errors in LLM-synthesized ground truth. What's the
plan to correct these?
```

---

## Section 8: Missing Critical Analysis

### Standard Weakness Pattern
"Paper lacks error analysis explaining what models fail on and why."

### Specific Critiques You Could Raise

**Critique 8.1 - Error Categorization**
```
For errors made by models on benchmark, categorize by type:

- Room boundary detection errors
- Spatial relationship errors (e.g., missed connectivity)
- Furniture placement errors
- Scale/proportion errors
- Symmetry/alignment errors

For each category:
- Frequency of error type
- Which models are prone to this error?
- Is error systematic (model consistently makes same mistake) or random?

Provide qualitative examples (screenshots) of each error type.

Why: Allows readers to understand what models are actually struggling with.
Aggregate F1 score of 0.74 hides whether model is bad at everything or only
specific aspects.
```

**Critique 8.2 - Difficulty Analysis**
```
For each test case, analyze:
- What made this case hard? (complex layout, poor photo, ambiguous boundaries)
- Which models succeeded/failed?
- Are hard cases actually the ones with hardest visual features?

Example: "We hypothesized complex L-shaped rooms would be hard, but actually
even simple single-room cases with poor lighting proved harder."

This provides insight into what the benchmark actually measures and whether
difficulty correlates with what we think it should.
```

**Critique 8.3 - Failure Mode Analysis**
```
Beyond error types, identify systematic failure modes:

- Do all models fail on a common subset of cases? (indicates annotation error)
- Do certain model architectures fail on certain aspects? (indicates architectural
  weakness)
- Are failures worse for minority categories? (e.g., commercial layouts
  underrepresented in training)

FIOVA found: "For videos that are relatively easy to describe, the models show
significant variability in performance. In contrast, for more challenging videos,
their performance becomes more consistent." This suggests models converge to
simple strategies on hard cases—important insight that only emerges from error
analysis.
```

---

## Summary: Using This Critique Template

When evaluating a spatial reasoning benchmark paper (like Blueprint-Bench), check:

- **Section 1**: Are metrics validated against human judgment?
- **Section 2**: Is annotation quality documented with inter-rater reliability?
- **Section 3**: Is dataset diversity explicitly characterized? Is generalization tested?
- **Section 4**: Are tasks difficulty-calibrated to avoid ceiling/floor effects?
- **Section 5**: Are hyperparameters justified with ablations?
- **Section 6**: Is evaluation fair and baselines competitive?
- **Section 7**: Are annotation and evaluation costs documented?
- **Section 8**: Is error analysis provided?

If the paper is weak on even one section, raise the corresponding critique. Multiple weaknesses should lower the rating significantly.

---

## Example of Synthesized Critique

Here's how you might combine these into a review paragraph:

```
"The paper lacks critical validation that its spatial accuracy metrics correlate
with human judgment of floor plan quality. While metrics are proposed (room
boundary F1, furniture placement accuracy), no evidence shows these metrics
correlate with expert human evaluations on the same test cases. Additionally,
the paper reports all models score >85% on the primary metrics, indicating ceiling
effects that prevent meaningful differentiation. Ground truth construction
methodology is unclear: are floor plans created by human annotators viewing
original photos, or synthesized from annotations using automated systems? The
annotation section lacks inter-annotator agreement statistics (Cohen's kappa)
and detailed protocols for resolving disagreements. The dataset composition is
not explicitly documented—what percentage are residential vs. commercial? Simple
rectangular vs. complex layouts? This omission prevents assessment of whether
benchmark results generalize across spatial reasoning scenarios. Finally,
critical ablation studies are missing: which metric components drive performance
rankings? Are evaluation hyperparameters robust to perturbation? These weaknesses
significantly limit confidence in the benchmark's ability to reliably assess
spatial reasoning in floor plan generation models."
```

This synthesized critique references multiple recurring weaknesses from the analyzed reviews, adapted for a spatial reasoning domain.
