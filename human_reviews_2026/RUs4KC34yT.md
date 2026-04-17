# Verifiable Natural Language to Linear Temporal Logic Translation: A Benchmark Dataset and Evaluation Suite

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Empirical evaluation of state-of-the-art natural language (NL) to temporal logic (TL) translation systems reveals near-perfect performance on existing benchmarks. However, current studies only measure the accuracy of the translation of NL logic into formal TL, ignoring a system’s capacity to ground atomic propositions into new scenarios or environments. This is a critical feature, necessary for the verification of resulting formulas in a concrete state space. In this paper, we introduce the Verifiable Linear Temporal Logic Benchmark (VLTL-Bench), a unifying benchmark for automated NL-to-LTL translation. The dataset consists of three unique state spaces and thousands of diverse natural language specifications and their corresponding formal temporal logic specifications. Moreover, the benchmark contains sample traces to verify the temporal logic expressions. While the benchmark directly supports end-to-end evaluation, we observe that many frameworks decompose the process into i) lifting,  ii) grounding, iii) translation, and iv) verification. The benchmark provides ground truths after each of these steps to enable researchers to improve and evaluate different substeps of the overall problem.  
 Using the benchmark, we evaluate several state‑of‑the‑art NL-to-TL translation models and frameworks, including nl2spec, NL2TL, NL2LTL, Lang2LTL, sequence-to-sequence translation, and various LLM prompting techniques. Our evaluation confirms that existing work is capable of reliably performing lifting and translation with high accuracy, while it exposes their struggles to ground the translation into a state space, which stems from the lack of existing datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The main argument of this paper is that while current NL-to-TL systems excel at translating abstract formulas, they fail badly at the crucial step of connecting those formulas to real-world environments (so called "grounding"). To address this, the authors  introduce VLTL-Bench, a new benchmark designed specifically to address this gap and enable end-to-end evaluation of NL->TL translation including verification. This benchmark includes multiple distinct state spaces (environments), Thousands of diverse NL specifications with corresponding formal TL specs,  sample traces for verifying the grounded formulas within each state space and it supports the evaluation of the full process: "Lifting", "Grounding", "Translation", and "Verification"
The authors show that existing systems perform very well on "lifting" and "translation", confirming prior benchmark results, but they struggle significantly with the critical step of "grounding" atomic propositions into a provided state space.

### Strengths
-- The paper addresses the major limitation in existing NL-to-TL research: the lack of evaluation for "grounding". Previous benchmarks ignored this essential step needed for real-world verification.

-- The work enables end-to-end evaluation, providing not just formulas, but "sample traces" that allow researchers to verify if the final grounded TL formula behaves correctly in a specific state space/scenario. 

-- The paper considers NL-to-TL process as a pipeline of different stages (lifting, grounding, translation) and for each stage it provides the ground truth data.

-- The paper provides a rich & diverse dataset, with "thousands of diverse NL specifications" providing corresponding formal TL specs for each NL spec.

--The paper exposes the specific weakness of existing SOTA models: high accuracy on abstract translation but poor performance on grounding.

-- Broad Evaluation Scope (`nl2spec`, `NL2TL`, `NL2LTL`, `Lang2LTL`) including sequence-to-sequence models and Various LLM prompting techniques.

### Weaknesses
-- Scope Limited to LTL: The benchmark specifically targets Linear Temporal Logic (LTL), as indicated by "VLTL-Bench" and references to LTL. This does not evaluate translation  capabilities for other important temporal logics like Computation Tree Logic (CTL) or Signal Temporal Logic (STL).

-- Limited State Space Diversity & Scale:While it includes three distinct state spaces, this is still a small number compared to the vast diversity of real-world systems NL-to-TL might  be applied to. Generalizability beyond these state spaces is not proven.

-- Sample Trace Limitations: While including traces for verification is crucial, their representativeness are very important. The paper does not provide details how these traces were generated or if they comprehensively cover potential system behaviors (e.g., are corner cases included?)

### Questions
The benchmark includes only three distinct state spaces. How do you justify that this small number sufficiently captures the complexity and variability  required to evaluate grounding generalization in real-world scenarios? What steps were taken to ensure these environments are not biased or overlapping?

How were the sample traces for verification generated? Can you demonstrate statistically that they cover corner cases, violations, and  satisfactions of the LTL formulas? How do you address concerns about trace completeness impacting verification reliability?

VLTL-Bench focuses exclusively on Linear Temporal Logic (LTL). Many practical applications require richer logics like CTL*, Signal Temporal Logic (STL), or Metric Temporal Logic (MTL). Does your approach fundamentally limit the benchmark's applicability? Is extending it to other TLs feasible?

You attribute poor grounding performance primarily to a 'lack of existing datasets'. However, could the failure also stem from fundamental architectural limitations in current NL-to-TL systems (e.g., LLMs lacking state-space reasoning)? How did you disentangle data scarcity from model capability?

Given the identified grounding bottleneck, what concrete architectural changes, training paradigms (e.g., using VLTL-Bench for fine-tuning), or hybrid approaches do you propose as most promising to address this?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies an interesting and important problem in the area of natural language to formal language (here is temporal logic). Most previous studies focus on lifted translation, while neglecting the action grounding process. This work finds most models actually fails in Atomic proposition grounds. Hence, they create a new dataset and test the performance of all the SoTA methods. Overall, I think this paper is a good contribution.

### Strengths
The great detection of the flaws in current approaches and the solid dataset creation and benchmark testing. The writing and related work articulation are clear. The motivation is great.

### Weaknesses
The main question I am doubting is the semantic and expression diversity of the created dataset. The initial 43 expressions seem too limited. Meanwhile, I remember in the referenced work NL2TL, they utilized LLM to help synthesize the initial pairs to then do human annotation. That increases the diversity. In this study, it seems the authors do not utilize LLM for synthesizing. I wonder if the authors can explain what's the reason and possible benefits.

### Questions
As I said above, the dataset semantic diversity remains one question.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces VLTL-Bench, a benchmark that evaluates natural-language-to-LTL systems end-to-end. The proposed benchmark covers lifting, translation, grounding, and verification across warehouse, traffic-light, and search-and-rescue scenarios. It shows that while models handle lifted translation well, they struggle to ground atomic propositions in concrete state spaces, leading to large drops in end-to-end and trace-verification accuracy.

### Strengths
1. VLTL-Bench measures all four aspects and supplies ground truth for each, plus example traces to check whether formulas actually hold. 
2. The scenario configs and templated generation make the benchmark easy to extend.

### Weaknesses
1. All LLMs used for evaluation all extremely small & non-reasoning LLMs. The task is considered as a reasoning task, so including results for reasoning LLMs will make the evaluation more comprehensive. 
2. The benchmark centers on discrete-time LTL. Related logics are discussed in L680 but not supported, limiting applicability to systems needing other temporal formalisms.

### Questions
1. I understand that you have provided few-shot examples in A.7, but what's the prompt used for few-shot? 
2. Why Actions is limited to a maximum of two targets (as stated in L197)?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces VLTL-Bench, a benchmark dataset for evaluating natural language to Linear Temporal Logic (LTL) translation systems. The key contribution is providing ground truth annotations for intermediate translation steps (lifting, grounding, translation, verification) and including verification traces. The authors evaluate several state-of-the-art systems and reveal that grounding (mapping abstract propositions to concrete state spaces) is a significant bottleneck. While the paper addresses an important problem and provides a useful modular evaluation framework, it suffers from limited template diversity (43 templates), insufficient domain coverage (3 scenarios), and, critically, lacks analysis of whether LLMs exploit semantic knowledge versus compositional reasoning. The absence of obfuscated scenario variants (similar to PlanBench [1] for PDDL) is a major weakness that prevents proper isolation of the grounding problem.

[1] Valmeekam et al., PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change. NeurIPS 2023.

### Strengths
A) The paper correctly identifies a critical gap in existing benchmarks that they focus on lifted translation while ignoring grounding, which is essential for executable specifications. The observation that current datasets achieve >90% accuracy but cannot produce verifiable formulas is valuable.

B) The design allowing isolated assessment of lifting, grounding, translation, and verification is well-motivated and technically sound. The metrics are clearly defined and appropriate.

### Weaknesses
A) Limited Template Diversity and Generalization

This is my biggest concern with this work. The benchmark is built from only 43 linguistic and logical templates (36 from nl2spec + 7 new). While thousands of examples are generated through instantiation with different actions/arguments, this approach has some limitations. 43 templates cannot capture the rich diversity of natural language specifications in real-world applications. Human stakeholders express requirements in countless syntactic structures, with varying referring expressions, negations, conditionals, and quantifiers. The high translation accuracy (99.9% for NL2TL in Table 4) likely reflects template memorization rather than robust understanding. Additionally, Table 8 shows a significant imbalance. E.g., "next" appears ~10 times more than "until" across domains. This distribution may not reflect real-world specifications and could bias evaluation toward overrepresented operators. Finally, Models trained and evaluated on template-instantiated data may learn to recognize surface patterns rather than develop compositional understanding. 

B) Missing Analysis on Semantic Knowledge vs. Compositional Reasoning

The paper does not adequately address whether LLMs leverage pre-training knowledge for grounding, which is crucial for understanding the grounding bottleneck. It's well known that modern LLMs are trained on massive corpora, potentially including robotics documentation, LTL tutorials, warehouse management systems, and traffic control specifications. Additionally, GPT-4 models may have encountered the nl2spec benchmark (36 templates) during pre-training, artificially inflating performance on lifting and translation while grounding remains poor because it requires scenario-specific knowledge. Can they exploit this semantic knowledge for this benchmark? A possible solution would be to follow PlanBench's approach for PDDL planning. SImilar to it, the paper must include obfuscated versions where semantically meaningful names are replaced with arbitrary symbols. E.g., replace search(apple) with act_3(obj_17) then ask queries like "perform procedure Z on target Q" instead of "look for the red fruit". This would isolate compositional reasoning from semantic shortcuts, and test pure grounding ability based solely on scenario configuration. It might also mitigate data contamination concerns (nl2spec templates are publicly available). This is a major gap that significantly weakens the paper's contributions.

C) Weak Grounding Baselines

Given that grounding is identified as the critical bottleneck (Table 5 shows 5.0%-68.6% AP-Dict accuracy), the baseline approaches are surprisingly simplistic. The baselines use a basic few-shot prompting without even the basic additions like chain-of-thought reasoning, structured output constraints like JSON, iterative refinement with verification feedback using provided traces, and self-consistency or ensemble methods. THis might inflate the efficacy of this benchmark.

### Questions
1. Why 43 templates? Can you provide empirical or theoretical justification that 43 templates provide adequate coverage of natural language and logical diversity? What would performance look like with 10 templates vs 100 templates?
2. Will you add obfuscated variants (replacing meaningful names with symbols like "act_3(obj_17)") to isolate compositional reasoning from semantic knowledge? This is standard practice in planning benchmarks (PlanBench) and seems essential here.
3. Have you verified that the nl2spec templates and your scenarios don't appear in GPT-4's training data? Can you test on models with verifiable training cutoffs before your dataset creation?
4. Why didn't you explore chain-of-thought prompting, iterative refinement with trace verification, or fine-tuned models for grounding given it's the identified bottleneck?
5. How were the positive and negative traces chosen for each specification? Are they minimal, typical, or adversarial examples?
6. Current scenarios have flat type systems. Can your framework handle hierarchical types (e.g., "animal -> dog -> poodle")?

### Soundness
2

### Presentation
2

### Contribution
3
