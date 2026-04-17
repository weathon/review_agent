# LeanGeo: Formalizing Competitional Geometry problems in Lean

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Geometry problems are a crucial testbed for AI reasoning capabilities. Most existing geometry solving systems cannot express problems within a unified framework, thus are difficult to integrate with other mathematical fields. Besides, since most geometric proofs rely on intuitive diagrams, verifying geometry problems is particularly challenging. To address these gaps, we introduce LeanGeo, a unified formal system for formalizing and solving competition-level geometry problems within the Lean 4 theorem prover. LeanGeo features a comprehensive library of high-level geometric theorems with Lean’s foundational logic, enabling rigorous proof verification and seamless integration with Mathlib. We also present LeanGeo-Bench, a formal geometry benchmark in LeanGeo, comprising problems from the International Mathematical Olympiad (IMO) and other advanced sources. Our evaluation demonstrates the capabilities and limitations of state-of-the-art Large Language Models on this benchmark, highlighting the need for further advancements in automated geometric reasoning. To further improve prover performance, we introduce a synthetic data generation pipeline together with a reinforcement learning training framework built on LeanGeo. We open source the theorem library and the benchmark of LeanGeo at \url{https://anonymous.4open.science/r/LeanGeo-9CE9}

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
A unified framework in the Lean 4 theorem prover for formalizing competition-level geometry problems. It features a comprehensive library of 260 theorems and LeanGeo-Bench, a benchmark of 122 problems including 43 IMO geometry tasks, to evaluate large language models' reasoning capabilities, highlighting current limitations and the need for further advancements.

### Strengths
LeanGeo provides a novel framework that integrates competition-level geometry into Lean 4, complemented by LeanGeo-Bench, offering a valuable resource for AI reasoning research. The seamless integration with Mathlib enables LeanGeo to leverage algebraic and inequality tools, enhancing its applicability to complex, interdisciplinary mathematical problems.

### Weaknesses
1. LeanGeo heavily relies on existing frameworks such as LeanEuclid (Murphy et al., 2024) and SystemE (Avigad et al., 2009), with its primary contributions being an expanded theorem library and 52 new abbreviations (syntactic sugar). This incremental approach lacks substantial novelty. Compared to AlphaGeometry (Trinh et al., 2024), which introduces innovative neurosymbolic search, LeanGeo’s “human-like” proofs show limited differentiation. Table 1 highlights qualitative differences but fails to provide quantitative metrics, such as proof length or success rates on shared problems, to substantiate its advantages.

2. The statement ‘We present the first framework in the Lean theorem prover capable of expressing and reasoning about competition-level geometry problems in a human-like manner’ (Page 2, Lines 100-104) may require further clarification. Myers (2024) has demonstrated progress in formalizing planar geometry in Lean, including a solution to a 2019 IMO problem, as noted in community discussions (e.g., ‘Lean in 2024’ blog). While this suggests prior efforts in handling competition-level geometry, the scope and capabilities of Myers’ work compared to LeanGeo remain unclear.

3. The RL experiments in Section 5 are vague and lack depth. No ablation studies (e.g., assessing the impact of theorem library size or prompt strategies), hyperparameter details, or error analyses (e.g., identifying failed problems and their reasons) are provided. Additionally, the absence of comparisons with state-of-the-art methods, such as DeepSeek-Prover or Seed-Prover, undermines the validity of the reported “promising initial results.

4. LeanGeo-Bench, comprising 122 problems including 43 IMO geometry problems, appears limited in scale compared to existing benchmarks like MATP-BENCH (1,056 problems). This raises concerns about whether the dataset is sufficiently representative of the diversity and complexity of competition-level geometry problems.

### Questions
None

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
2

### Summary
The paper introduces LeanGeo, a Lean-4–native framework and benchmark for competition-level Euclidean geometry. It offers a high-level theorem/tactic library, SMT integration, and a curated benchmark intended to mix geometry with general math (e.g., trigonometry, inequalities) via Mathlib.

### Strengths
- The paper presents a clear structure and provides readable examples; it is well-written, although it has minor wording issues.
- The system design is plausible and well-motivated.
- It is a valuable step toward Lean-native, competition-level geometry with cross-domain reasoning.

### Weaknesses
- The experiments are thin and largely non-diagnostic. There are no core ablations—no SMT off/partial settings, no scaling with library size or lemma granularity, no comparisons of tactic schedules, prompts, or decoding—so it’s unclear why models fail.
- It’s also unclear how this benchmark compares to others. There’s no side-by-side test on the same problems against existing Lean or geometry benchmarks, so claims about being better or different are mostly qualitative.
- Scalability and complexity are uncharacterized (no curves vs. points/constraints/branching), and the RL component is preliminary without stronger curricula or knowledge-tracing.

Overall, the experiments are too shallow to support firm claims. Adding these basic comparisons and reports would make the paper much stronger.

### Questions
- How does performance change with SMT off or with restricted solver capabilities?
- What are scaling curves for success/time vs. theorem-library size and lemma granularity?
- Can you provide aligned comparisons with existing Lean/geometry benchmarks on a shared subset?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces LeoGeo, a formal system for formulating and solving competition-level geometry problem, and LeoGeo-Bench, a collection of benchmark geometry problems, both built using Lean 4. Benchmark results for state-of-the-art LLMs are provided.

### Strengths
* The LeanGeo library allows formalizing and solving competition-level geometry problems in Lean 4. It includes an extensive library of high-level definitions and tactics, which makes formal proofs more intuitive and understandable. LeanGeo's integration with Mathlib allows it to leverage powerful tools from other areas of maths.
* The LeanGeo-Bench benchmark is useful for evaluating advances in the field.
* The paper is generally well-written and easy to follow.

### Weaknesses
I find some of the claims insufficiently supported and some more details would be helpful, as explained below.
* The main limitation of LeanEuclid compared to LeanGeo seems to be a limited set of formalized geometry facts, as stated at line 52. Is it difficult to expand LeanEuclid's library? If not, what additional advantage does LeanGeo has?
* The paper claims that LeanGeo allows expressing and reasoning about geometry problems in a human-like manner. This may be debatable as the examples presented in the paper are still highly formal. It is also not clear what exactly is done to make the proofs more "human-like". Can you provide more details?
* At some places the writing could be improved. The RL experiments seem to be a significant part of the paper, but this is not motivated in the introduction, and not mentioned in the abstract and the list of contributions.

Minor
* Line 234: Line 1 should pass the point B, but this is not mentioned. 
* Line 820: empty bullet point.

### Questions
See questions in Weaknesses and the questions below.

* Is mathlib always used for generating proofs using the theorem prover? If yes, is it possible to evaluate the theorem prover's performance without using it?
* Only 43 problems in the benchmark have proofs. Is it because the automatic theorem prover fails to provide proofs for other problems? If yes, on which problems the theorem prover fails?

### Soundness
2

### Presentation
2

### Contribution
3
