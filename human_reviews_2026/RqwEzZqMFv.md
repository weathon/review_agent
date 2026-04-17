# MolecularIQ: Characterizing Chemical Reasoning Capabilities Through Symbolic Verification on Molecular Graphs

- Decision: Accept (Poster)
- Scores: 4, 4, 4

## Abstract
A molecule’s properties are fundamentally determined by its composition and structure encoded in its molecular graph. Thus, reasoning about molecular properties requires the ability to parse and understand the molecular graph. Large Language Models (LLMs) are increasingly applied to chemistry, tackling tasks such as molecular name conversion, captioning, text-guided generation, and property or reaction prediction. Most existing benchmarks emphasize general chemical knowledge, rely on literature or surrogate labels that risk leakage or bias, or reduce evaluation to multiple-choice questions. We introduce MolecularIQ, a molecular structure reasoning benchmark focused exclusively on symbolically verifiable tasks. MolecularIQ enables fine-grained evaluation of reasoning over molecular graphs and reveals capability patterns that localize model failures to specific tasks and molecular structures. This provides actionable insights into the strengths and limitations of current chemistry LLMs and guides the development of models that reason faithfully over molecular structure.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces MOLECULARIQ, a benchmark aimed at chemical structure reasoning with symbolically verifiable ground truths. Tasks are built directly on molecular graphs (via RDKit solvers) and span three orthogonal axes: reasoning category (feature counting, index attribution, constrained generation), multitask load (1, 2, 3, or 5 simultaneous requirements), and molecular complexity (Bertz bins). The authors also provide a dynamic variant (MOLECULARIQD) for regenerating fresh test sets and preventing saturation/overfitting. Evaluation is integrated into the lm-evaluation-harness, with hierarchical answer extraction and semantic (key-agnostic) comparison; accuracy is averaged over three independent rollouts. A large-scale study (34 models) finds that recent MoE/generalist reasoning LLMs dominate; chemistry-tuned instruction/RL models often underperform their bases; constrained generation is easiest at low constraint counts but degrades sharply as constraints stack; and multitask load harms performance more than molecular complexity. The paper argues symbolically verifiable tasks reduce leakage/bias and produce capability fingerprints localizing failure modes.

### Strengths
Tight problem statement & clear contribution. A fully symbolically verifiable chemistry benchmark focused on structure-grounded reasoning (not factual recall) is timely and well-motivated.

Three-axis profiling. Disentangling reasoning type, multitask load, and molecular complexity provides diagnostic granularity and actionable error localization.

Index-based tasks. Pairing counting with index attribution helps distinguish genuine graph reasoning from pattern-matching/shortcut counts.

Solid evaluation infrastructure. Harness integration, hierarchical extraction, and semantic answer comparison address brittle formatting issues; multi-rollout accuracy mitigates sampling variance.

Dynamic benchmark (MOLECULARIQD). A path to refreshable evaluations that can evolve with the field and support RL with verifiable rewards.

Empirical insights. Consistent trends (MoE leads; chemistry tuning can hurt; canonical/aromatic SMILES easier; multitask load dominates difficulty) are useful for both modeling and benchmark design.

Transparent limitations section. Clear articulation of the current scope (2D graphs, single-molecule tasks, symbolic-only feature set).

### Weaknesses
2D-only scope. Restricting to graph connectivity omits 3D stereoelectronic/conformational effects that matter for realistic chemical reasoning; several stereochemistry tasks may still be fragile under a purely 2D treatment.

Verifier dependence & edge cases. Heavy reliance on RDKit rules brings corner-case risk (e.g., aromaticity/kekulization, tautomers, undefined stereocenters). Clear auditing and unit tests for borderline cases would strengthen claims.

Dataset scale & coverage. The main static benchmark uses hundreds of molecules / thousands of questions—adequate for signal but small relative to chemical space. It is unclear how representative the selected feature distributions and Bertz bins are for downstream applications.

Generation tasks may be gameable. Low-constraint prompts (e.g., “has two rings”) can be satisfied by template snippets; evidence that models aren’t exploiting canonical pattern banks (beyond SMILES randomization) would be welcome.

Per-model configuration fairness. “Tailored configs” improve each model’s score but may complicate cross-model fairness. A fixed canonical configuration alongside tailored ones would help disambiguate.

Rollout averaging & significance. Averaging over three stochastic runs may be thin for close comparisons; confidence intervals and multiple seeds per model/config would improve robustness.

Leaderboard/process details deferred. Several artifacts (e.g., public leaderboard link, full configs) are promised camera-ready; reproducibility would benefit from making the dynamic generator and verifier immediately available.

Limited assessment of prompt/extraction shaping. Although hierarchical extraction is a strength, further stress tests against format-shaping and overfitting to extraction heuristics would increase trust.

### Questions
Verifier audits: How do you handle aromaticity/kekulization mismatches, tautomerism, and unspecified stereochemistry during indexing and generation checks? Any published test suite of adversarial edge cases?

SMILES perturbations: Beyond canonical/aromatic randomization, did you try token perturbations (ring index relabeling, randomized branches) to further separate pattern recall from graph reasoning?

Fairness controls: Can you report both fixed (uniform decoding and temperature) and tailored configs for all models to separate capability from tuning sensitivity?

Rollouts & variance: Why three rollouts? Do results meaningfully change with 5–10 rollouts or multiple seeds? Please add per-model variance bars in the main text.

Constraint hardness: For constrained generation, can you provide a calibrated hardness ladder (e.g., constraint sets with matched feasibility rates) and show how models scale as we move up the ladder?

Dynamic set governance: How will MOLECULARIQD updates be versioned (to avoid moving goalposts) and how will you prevent train–test leakage as the community begins to tune on the benchmark?

3D extension: What’s the roadmap for 3D-aware, symbolically verifiable tasks (e.g., CIP resolution, ring puckers, distance constraints) and for multi-molecule tasks (reaction stoichiometry, scaffold ranking) while maintaining verifiability?

Failure mode taxonomy: Can you release capability fingerprint templates and diagnostic exemplars so users can map a model’s errors to specific graph-perception or compositional failures?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces a new benchmark to evaluate LLMs on chemistry tasks. All tasks are grounded in the molecule’s graph structure and have answers that can be checked by a symbolic solver. The authors evaluates 34 LLMs and find that the largest mixture-of-experts models with high reasoning budgets achieve the best accuracy.

### Strengths
1. The paper presents a verifiable benchmark and all tasks have ground-truth solutions, allowing reliable automatic evaluation.

2. The benchmark varies the molecular complexity and tests different chemical reasoning skills.

### Weaknesses
1. The benchmark excludes tasks like quantitative property prediction or reaction prediction, so it does not evaluate LLMs’ ability on some real-world chemistry problems.

2. The tasks in the benchmark are fundamental checks that chemical softwares can do straightforwardly. They may be somewhat disconnected from how humans typically solve chemistry problems. For example, asking a model to ‘generate a molecule with two rings and five heterostoms’ is more like a puzzle or exercise than creative problem-solving.

3. The benchmark uses SMILES strings for molecules and the authors noticed that LLMs may use pattern recognition rather than structural reasoning. Similar arguments have been discussed in the literature, such as the inconsistency of LLMs in molecular representations which indicates LLMs fail to capture the underlying chemistry. The authors may consider adding similar consistency checks in the benchmark.

### Questions
1. Does a better performance on the benchmark always suggest a stronger reasoning? Is there a way to explicitly evaluate if the model is doing reasoning, or is it doing pattern recognition?

2. Some LLMs use external tool calls to solve chemistry tasks such as ChemCrow. Do the authors envision the benchmark being used not just standalone LLMs, but also LLM-based agents?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MoleculeIQ, a molecular structure reasoning benchmark composed of symbolically verifiable tasks. Specifically, the tasks consist of three types: (1) feature counting, (2) index-based attribution, and (3) constrained generation, and the features of interest include functional groups, chemical properties, synthesis, and so on. The MoleculeIQ dataset is composed of 849 molecules. In addition, the paper proposes a framework that dynamically computes the ground-truth labels for the designed tasks, which it calls MoleculeQID. The experimental results show that current LLMs exhibit a significant gap in the compositional and structural reasoning abilities required.

### Strengths
* This paper provides a wide-ranging evaluation across diverse model types and sizes.
* The introduced MoleculeQID can be further utilized for new and open molecules, guaranteeing its scalability.

### Weaknesses
* This study is grounded in the belief that there is a positive correlation between molecular structural understanding and molecular reasoning ability for complex property prediction. However, this belief is not explicitly demonstrated, so the necessity of building a structure-reasoning benchmark appears limited. I suggest presenting the relationship between structural understanding and predictive performance on molecular properties.
* Overthinking is a well-known pitfall in molecular structure understanding, yet the results here differ from conventional wisdom. Providing an in-depth analysis of this aspect would strengthen the contribution of the proposed benchmark and offer clearer grounds for the necessity of reinforcement learning (RL) training.
* The results clearly show where models fail, but the paper would be stronger with deeper qualitative error analysis. Presenting examples of incorrect molecules generated by top models and categorizing the types of structural mistakes would give model developers more actionable insights.

### Questions
* Could the authors provide the failure cases on the multi-task scenario?
* What kinds of substructures do LLMs fail to capture?

### Soundness
3

### Presentation
2

### Contribution
1
