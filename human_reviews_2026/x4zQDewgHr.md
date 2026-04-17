# Demystifying Deep Search: A Holistic Evaluation with Hint-free Multi-Hop Questions and Factorised Metrics

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
RAG (Retrieval-Augmented Generation) systems and web agents are increasingly evaluated on multi-hop deep search tasks, yet current practice suffers from two major limitations. First, most benchmarks leak the reasoning path in the question text, allowing models to follow surface cues rather than discover reasoning chains autonomously. Second, evaluation is typically reduced to a single pass rate, which collapses diverse behaviors into one score and obscures whether failures stem from inadequate search, poor knowledge use, or inappropriate refusal. To address these issues, we present WebDetective, a benchmark of hint-free multi-hop questions paired with a controlled Wikipedia sandbox that ensures full traceability of model actions, and a holistic evaluation framework that separates search sufficiency, knowledge utilization, and refusal behavior. Our evaluation of 25 state-of-the-art models reveals systematic weaknesses across all architectures: models struggle with knowledge utilization despite having sufficient evidence and demonstrate near-absent appropriate refusal when evidence is lacking. These patterns expose a fundamental gap—today's systems excel at executing given reasoning paths but fail when required to discover them. We develop an agentic workflow EvidenceLoop that explicitly targets the challenges our benchmark identifies, incorporating verification loops and systematic evidence tracking that improve both search and synthesis capabilities. This baseline demonstrates that WebDetective's diagnostic framework can guide concrete architectural improvements, establishing our benchmark as a critical tool for developing genuinely autonomous reasoning systems rather than pattern-following agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces WebDetective, a new benchmark for evaluating web agents on hint-free multi-hop deep search tasks. 
Unlike prior datasets that leak reasoning paths (path-hinting) or entity fingerprints (spec-hinting), the authors construct a controlled Wikipedia sandbox that enforces step-by-step reasoning without shortcuts. 
They also propose a two-level evaluation framework that disentangles knowledge sufficiency, utilization, and refusal. 
Experiments on 25 state-of-the-art models show systematic weaknesses: models often retrieve enough evidence but fail at synthesis, and almost none refuse appropriately. 
The authors further design EvidenceLoop, an agentic workflow baseline that uses verification loops and memory tracking to partially address these gaps.

### Strengths
- The paper identifies and argues against the hidden hinting in existing multi-hop deep research datasets.
- The co-design of hint-free questions with a sandbox masking mechanism is clever and ensures true multi-hop reasoning.
- 25 strong models are compared, giving a broad empirical picture.

### Weaknesses
- The controlled Wikipedia sandbox may be too artificial compared to real-world open web environments.
- While an interesting method, improvements seem modest. It feels more like a proof-of-concept than a strong new method for EvidenceLoop.

### Questions
- Which LLM does your method use in Table 1?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents WebDetective, a benchmark for evaluating web agents on hint-free multi-hop deep search tasks within a controlled Wikipedia sandbox. Unlike prior datasets that leak reasoning paths or entity attributes, it enforces autonomous discovery of reasoning chains and enables fine-grained attribution of failure modes. The authors also propose a factorized evaluation framework—separating knowledge sufficiency, generation quality, and refusal behavior—and an agentic baseline (EvidenceLoop) integrating evidence tracking and verification. Experiments on 25 state-of-the-art models show that while systems can retrieve sufficient evidence, they often fail to synthesize it effectively, exposing a major synthesis bottleneck. The benchmark demonstrates that performance gains cannot be achieved by scaling context or computation, emphasizing that true progress in deep reasoning requires better evidence utilization and calibration, rather than brute-force search or larger models.

### Strengths
1. The paper proposes a very interesting and promising setting, hint-free multi-hop deep search within a controlled Wikipedia sandbox, which effectively exposes reasoning weaknesses that are usually hidden in traditional benchmarks with embedded hints. The benchmark design is well-motivated and enables fine-grained diagnosis of model behavior.
2. The experiments are comprehensive, covering 25 state-of-the-art systems across multiple providers, and the analysis is insightful, identifying key failure modes like synthesis bottlenecks and calibration issues. 
3. The additional analyses are also strong, including detailed model profiling and robustness studies on scaling context length and computational budgets, which further validate the benchmark’s diagnostic value.

### Weaknesses
1. The presentation could be improved, including writing clarity, figure design, and overall structure. Some key insights are buried in the text and would benefit from clearer highlighting and visual emphasis.
2. Certain methodological details are missing or underexplained. For instance, the process for testing Parametric Inaccessibility and Evidence Sufficiency is not clearly described—it’s unclear whether these evaluations are performed via prompting LLMs directly or through a separate automated verification strategy.
3. While the benchmark is well-motivated, the paper could further clarify its generalization scope, such as whether the hint-free design can extend beyond Wikipedia-style domains or integrate with open web environments to evaluate real-world reasoning robustness.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces WebDetective, a benchmark for evaluating web agents on hint-free multi-hop question answering tasks. It also introduces a co-design evaluation environment to enable rigorous diagnostics. The evaluation contains multiple frontier systems and is comprehensive.

### Strengths
The writing is clear and easy to follow.

The problem formulation is novel, and the observation that existing benchmarks contain unrealistic hints and specifications is meaningful.

The diagnostic evaluation framework can help target different weaknesses of different systems, meaningfully characterize different systems under different categories, and provide insightful/actionable guidance to the development of future deep research systems.

### Weaknesses
1. The controlled Wikipedia sandbox, while enabling precise evaluation, creates artificial constraints that may not reflect real-world multi-hop reasoning scenarios. In practice, information is rarely hidden behind such strict sequential dependencies—multiple valid paths typically exist, and partial information can be retrieved from various sources. With an analysis of how performance correlates between sandbox and open-web settings, it could further enhance the validity and value of the benchmark. 

2. The 200 questions are relatively small for establishing robust benchmark conclusions. It would be helpful to enhance the statistical rigor by including significance tests.
The heavy concentration on 2-3 hop questions (86%) limits insights about longer reasoning chains. Besides, the questions are synthetically constructed by blocking direct paths in existing QA pairs, which may not reflect naturally occurring multi-hop reasoning patterns. 

3. The paper lacks analysis of domain diversity beyond question types, leaving unclear whether the benchmark targets diverse reasoning patterns or is dominated by specific domains (e.g., familial relationships).

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents WebDetective, a benchmark for evaluating web agents and RAG systems on hint-free multi-hop deep search tasks. It identifies two core issues in existing benchmarks: hint leakage in question design (Path- or Specification-Hinting) and single-metric evaluation that masks distinct failure modes. WebDetective co-designs hint-free questions with a controlled Wikipedia sandbox enforcing sequential reasoning through selective entity masking, and introduces a factorized diagnostic framework separating knowledge sufficiency, generation quality, and knowledge degradation. Evaluations on 25 models show that most retrieve sufficient evidence but fail in reasoning composition and calibrated refusal.

### Strengths
(1) The paper identifies a genuine gap in current deep-search benchmarks—most rely on linguistic hints that reduce reasoning to execution. The distinction between Path-Hinting, Specification-Hinting, and the proposed Hint-Free formulation is sharp and well grounded.

(2) The co-designed Wikipedia sandbox with selective entity masking is both creative and rigorous, ensuring that agents must discover reasoning paths rather than shortcut them. This design provides strong interpretability and enables precise failure attribution.

(3) Evaluating 25 state-of-the-art models provides broad coverage and meaningful insights. The taxonomy of behavioral profiles (e.g., Powerful but Overconfident, Synthesis Bottleneck) is particularly informative for understanding system limitations.

### Weaknesses
(1) The proposed EvidenceLoop baseline achieves modest improvements and is not clearly positioned relative to existing ReAct-style frameworks, making it less convincing as a substantive modeling advance.

(2) The controlled sandbox enforces a single reasoning path, which improves interpretability but departs from realistic web conditions where multiple reasoning routes may coexist. This trade-off limits ecological validity.

### Questions
(1) How do you ensure that human validators and verifying LLMs do not inadvertently rely on their own parametric knowledge when assessing whether questions are answerable only through the provided evidence chain?
(2) Could you clarify how EvidenceLoop differs conceptually and architecturally from prior ReAct or self-reflection frameworks? Is its improvement mainly from verification loops or memory structuring?

### Soundness
3

### Presentation
2

### Contribution
2
