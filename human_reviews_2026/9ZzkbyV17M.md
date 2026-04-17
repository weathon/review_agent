# From Ambiguity to Verdict: A Semiotic‑Grounded Multi‑Perspective Agent for LLM Logical Reasoning

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 2

## Abstract
**Logical reasoning** is a fundamental capability of large language models (LLMs). However, existing studies largely overlook the interplay between ***logical complexity*** and ***semantic complexity***, resulting in methods that struggle to address challenging scenarios involving abstract propositions, ambiguous contexts, and conflicting stances—features central to human reasoning.
We propose **LogicAgent**, a *semiotic-square–guided framework* that jointly addresses these two axes of difficulty. The semiotic square provides a principled structure for multi-perspective semantic analysis, and LogicAgent integrates automated deduction with reflective verification to manage logical complexity across deeper reasoning chains.
To support evaluation under these conditions, we introduce **RepublicQA**, a benchmark that couples semantic complexity with logical depth. RepublicQA reaches ***college-level semantic difficulty (FKGL 11.94)***, contains philosophically grounded abstract propositions with systematically constructed contrary and contradictory forms, and offers the most semantically rich setting for assessing logical reasoning in LLMs.
Experiments demonstrate that **LogicAgent** achieves *state-of-the-art performance* on RepublicQA, with a **6.25%** average gain over strong baselines, and generalizes effectively to mainstream logical reasoning benchmarks including ProntoQA, ProofWriter, FOLIO, and ProverQA, achieving an additional **7.05%** average gain. These results highlight the strong effectiveness of our **semiotic-grounded multi-perspective reasoning** in boosting LLMs’ logical performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper proposes LogicAgent, a multi-perspective reasoning framework based on Greimas' Semiotic Square, and introduces RepublicQA, a new benchmark derived from Plato's Republic for evaluating logical reasoning under semantic ambiguity. LogicAgent operates through three stages: semantic structuring (constructing contraries and contradictions), logical reasoning (FOL-based deduction), and reflective verification (multi-perspective validation). Experiments show improvements of 6.25% on RepublicQA and 7.05% on existing benchmarks (ProntoQA, ProofWriter, FOLIO, ProverQA).

### Strengths
1. LogicAgent uses the semiotic square to handle contrary (opposite) concepts, not just contradictory (true/false) ones, is a new and smart way to deal with ambiguity.
2. RepublicQA fills an important gap by testing reasoning on abstract philosophical concepts with college-level difficulty

### Weaknesses
The method is computationally heavy. It is slow and uses a very large number of tokens (avg. ~18.4k) for each query, making it costly to run.

### Questions
Please refer to weaknesses part.

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
This paper targets two challenges of the existing works. First, existing works overlook the interplay between logical complexity and semantic complexity. Accordingly, the authors propose the LogicAgent, which is based on semiotic-square and can jointly address the logical and semantic complexity. Second, existing benchmarks lack logical semantic complexity, so a benchmark RepublicQA is proposed, with freater lexical complexity and structural diversity.

### Strengths
Logical complexity and semantic complexity are indeed different perspectives of natural language content, and this paper makes an effort to address these two issues explicitly. 

The adoption of 'Semiotic Square' looks novel and brings an interesting idea into the field.

### Weaknesses
1. The abstract could be improved to make it more accessible to readers who are unfamiliar with these logical terms. In addition, changing terms also make the abstract less readable. For example, does the structural diversity refers to the logical complexity? Is the lexical complexity same as the semantic complexity?

2. The format can be improved to avoid confusion. in line 39 and 40, the 'In AI Cohen et al. xxxx' should be 'In AI (Cohen et al. xxxx)'. This problem appears a lot of times in the paper.

3. The presentation is not good enough. At the very beginning it is stated that the interplay of semantic complexity and logical complexity is targeted by this work, but the following part does not clearly explain what is this so called 'interplay', how it is 'overlooked', and how is this addressed by this work.

### Questions
It is stated that "existing benchmarks
remain confined to relatively simple and determinate semantics, often centered on everyday scenarios with shallow relations or logic problems that lack intricate semantic structure". However, there are also benchmarks designed for math, scientific reasoning, or pure logical reasoning. How about them? Also, some logical reasoning datasets use synthetic approaches to build complex logical structures, with adjustable logical complexity.

### Soundness
3

### Presentation
2

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
This paper introduces LogicAgent, a multi-perspective reasoning framework grounded in the Semiotic Square, designed to address the challenges of logical reasoning in LLMs when confronted with semantic ambiguity and abstract propositions. The framework operates by performing parallel deductions in FOL on a proposition, its contrary, and its contradictory, leveraging a multi-stage reflective verification mechanism to resolve inconsistencies. The authors also introduce RepublicQA, a new benchmark for this task characterized by high difficulty and semantic complexity derived from philosophical texts, on which their method achieves state-of-the-art performance, significantly outperforming strong baselines.

### Strengths
- The core idea of integrating a structuralist semantic tool (the Semiotic Square) with symbolic logic to mitigate semantic ambiguity is novel and compelling.
- The contribution of a new, manually annotated benchmark (RepublicQA) to address the lack of semantic complexity in existing datasets is valuable to the community.

### Weaknesses
- The methodology section lacks formal rigor. The paper would be significantly strengthened by adding more precise mathematical statements or lemmas that detail the formal assumptions and boundary conditions required to migrate the semiotic square into classical FOL. It should include, for example, a formal definition of the "existential import check" and its application.

- Reproducibility remains a concern. While the prompts are provided, the authors should add more concrete examples of side-by-side NL-to-FOL mappings. It is especially important to include complex cases involving nested quantifiers and negations, as these are critical for replicating the "Translator" module.

- The necessity of the *full* four-point Greimas Square is questionable. The authors should provide a targeted ablation study comparing the full four-point structure against a simpler three-point structure---one using S1, not S1, S2---to justify the framework's complexity.

- Some related works about LLM-based logical reasoning are missing, which should be compared with the proposed method or discussed on their difference. e.g.:

[1] Cumulative Reasoning with Large Language Models

[2] DetermLR: Augmenting LLM-based Logical Reasoning from Indeterminacy to Determinacy

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes LogicAgent, a reasoning framework that marries Greimas’ Semiotic Square with first-order logic (FOL), adds an existential-import check to avoid vacuous truths, and evaluates propositions with a three-valued scheme {True, False, Uncertain}.

### Strengths
- Provide a new dataset RepublicQA with college-level difficulty.
- Propose LogicAgent with great result on benchmarks.

### Weaknesses
- Dataset Scale: The size of the newly dataset RepublicQA is too small (n=200). This limited scale raises concerns about the statistical robustness of the findings and the dataset's general utility.

- Benchmark Reporting: The results on "Other Benchmarks" are reported as an aggregate average. Could the authors provide a detailed, disaggregated breakdown of performance for each individual benchmark (e.g., ProntoQA, ProofWriter, FOLIO, and ProverQA)?

- Dataset Generalizability: The decision to construct the RepublicQA dataset exclusively from a single source, Plato's "Republic," is questionable. This narrow domain scope inherently limits the dataset's diversity and generalizability.

- Novelty of Methodology: using Greimas’ Semiotic Square and extending the evaluation space from a binary {True, False} to a three-valued scheme {True, False, Uncertain} appear to lack significant innovation. As far as I know, many existing works, particularly in probabilistic logic, have implemented similar "de-binarization" approaches to handle uncertainty. The authors need to better justify the novelty of their method against this prior art.

### Questions
deliver my issues in the weakness's sections

### Soundness
2

### Presentation
3

### Contribution
2
