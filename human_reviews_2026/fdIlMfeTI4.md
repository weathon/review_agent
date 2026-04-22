# GEAR: A $\textbf{G}$eneral $\textbf{E}$valuation Framework for $\textbf{A}$bductive $\textbf{R}$easoning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 4, 8

## Abstract
Since the advent of Large Language Models (LLMs), research has primarily focused on improving their instruction-following and deductive reasoning abilities. Yet a central question remains: can these models truly discover new knowledge, and how can we evaluate this ability? In this work, we address this gap by studying abductive reasoning-the process of generating plausible hypotheses to explain observations.
We introduce **G**eneral **E**valuation for **A**bductive **R**easoning (GEAR), a new general-purpose, fully automated, transparent, and label-free evaluation paradigm that overcomes limitations of prior approaches. GEAR evaluates a set of hypotheses using three metrics: **consistency** (each hypothesis correctly explains the given observations), **generalizability** (consistent hypotheses make meaningful predictions on unseen inputs), and **diversity** (the set of hypotheses covers many distinct predictions and patterns). Built this way, GEAR is scalable (no human gold answers needed), reliable (transparent, deterministic scoring aligned with classical abduction), and open-ended (scores improve only when models produce new, plausible hypotheses, unlike existing static benchmarks that saturate once accuracy is high). Using GEAR, we conduct a fine-grained study of nine LLMs on four popular abduction benchmarks ($1{,}500$ problems), generating $50{,}340$ candidate hypotheses. GEAR reveals model differences and insights that are obscured by prior gold-answer-based or purely human evaluations.
We further propose a momentum-based curriculum training strategy that dynamically adjusts GEAR-derived training data by learning velocity: it begins with what the model learns faster and shifts toward harder objectives such as generating diverse hypotheses once the model is confident on foundational objectives (e.g., instruction following and consistency). Without gold-label supervision, this strategy improves all three GEAR objectives—consistency, generalizability, and diversity—and these gains transfer to established abductive-reasoning benchmarks. Taken together, GEAR provides a principled framework that not only evaluates abduction but also supplies label-free, scalable training signals that help LLMs produce more diverse and reliable hypotheses. We will release code and data upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces GEAR, an automated, label-free evaluation framework intended to assess abductive reasoning capabilities in large language models. Empirical results across nine models show that GEAR reveals finer-grained performance differences than traditional gold-label metrics  and that the momentum curriculum improves GEAR scores and modestly enhances downstream abductive benchmarks.

### Strengths
- The paper convincingly argues that existing abductive reasoning benchmarks relying on single "gold" labels or human judgments are inadequate for underdetermined reasoning problems.
- The paper evaluates 9 diverse LLMs across 4 datasets and provides simulation studies demonstrating that abductive reasoning is defeasible

### Weaknesses
- Although the framework is titled General Evaluation for Abductive Reasoning, all experiments are performed on programmable and synthetic tasks (e.g., ARC, ACRE, LIST FUNCTIONS). 

- The β/γ-diversity measures are based purely on output pattern dissimilarity (set overlap/Jaccard distance) - may capture syntactic or behavioral heterogeneity, not genuine content diversity.

- While the proposed adaptive weighting method is sensible, the results (Table 2, 4) show only marginal gains, and the analysis of why or how these improvements occur is minimal. The ablation studies offer little evidence of actual reasoning enhancement.

### Questions
- The β/γ-diversity metrics are based on prediction overlap. How do you ensure these metrics capture conceptual rather than purely behavioral diversity?

- Have you analyzed qualitative examples of generated hypotheses to verify that improvements in β/γ-diversity correspond to more meaningful explanations?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces GEAR (General Evaluation for Abductive Reasoning), a new framework for evaluating the abductive reasoning capabilities of Large Language Models (LLMs). The authors argue that existing evaluations, which often rely on a single "gold" hypothesis, are insufficient for abduction. GEAR proposes a label-free, automated evaluation based on three metrics: (1) Consistency (the hypothesis must explain the given observations), (2) Generalizability (the hypothesis should make meaningful predictions on unseen inputs), and (3) Diversity (the model should generate a set of distinct and varied hypotheses). The paper evaluates several LLMs using GEAR on four abduction benchmarks. Additionally, it demonstrates that these metrics can be used as a preference signal to fine-tune LLMs, thereby improving their performance on these same metrics.

### Strengths
1. Clarity: The paper is well-written and easy to read.
2. Sensible Criteria: The three proposed criteria for evaluation (Consistency, Generalizability, and Diversity) are sensible, logical, and well-grounded in the principles of scientific reasoning.

### Weaknesses
1. Limited Novelty of the Criteria: The primary concern is the limited conceptual novelty of the framework's components. The core metrics proposed are largely a formalization of well-established principles:
     + Consistency (matching observed data) is a fundamental requirement in almost any task.
     + Generalizability (testing on unseen data) is analogous to standard test set evaluation.
     + Diversity (generating distinct solutions) is a known metric in related fields like program synthesis.
The main contribution appears to be the combination of these ideas, rather than a fundamentally new evaluation paradigm.

2. Constrained Domain and Overstated Generality: The framework's key advantages (e.g., being "label-free" and "automated") are heavily dependent on its constrained domain of application. The evaluation is exclusively demonstrated on benchmarks (like MINI-ARC, ACRE) where hypotheses are executable programs. This executable nature is what makes automated checks for Consistency and Generalizability possible; an execution oracle exists. This limits the "general" claim of the framework.
The paper does not solve the core, much harder problem of evaluating open-domain natural language abduction, where ambiguity, logical equivalence, and the lack of an execution oracle are the main challenges. 
Furthermore, given the executable setup, using these automated checks as preference signals is a straightforward application of reinforcement learning from feedback. While effective, it feels like an incremental contribution rather than a novel one, especially when the setup itself is so constrained.

### Questions
N/A

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper highlights that many existing benchmarks for abductive reasoning rely on a single annotated “gold” hypothesis, which is limiting because multiple explanations may be equally valid. Rather than introducing new datasets, the framework described in this paper (GEAR) restructures several existing benchmarks — MINI-ARC, ARC-2025, LIST FUNCTIONS, and ACRE to evaluate the consistency, generalizability, and diversity of candidate hypotheses. While the framework is conceptually clear, it is difficult to follow precisely how the the restructuring is implemented. In particular, it appears that instances (or existing groups of tasks?) are grouped to test individual hypotheses across multiple observations, but the specifics are not fully described. Moreover, it is unclear how textual benchmarks such as ACRE are evaluated for alternative plausible hypotheses, leaving some ambiguity about the method’s applicability to natural-language domains.

### Strengths
- Thoughtful reformulation of existing tasks to allow evaluation of multiple plausible hypotheses.
- Evaluation across multiple large language models

### Weaknesses
- Several parts of the paper are unclear, particularly the construction of the evaluation datasets and the specifics of model evaluation.
- The computation of metrics such as Instruction-Following Rate is not clearly defined.

### Questions
1. Could you provide more details or examples of how the evaluation datasets were constructed for each benchmark, including how examples were grouped and how synthetic or additional test cases were generated? How does one ensure that the grouping is always feasible and sound?

2. Could you clarify the training procedure — specifically, how the GEAR scores were used during training and how evaluation on unseen portions of the original tasks was performed?

3. How exactly was the Instruction-Following Rate in Table 2 computed, and how does it interact with other metrics?

4. Were metrics from the original benchmarks used in any way, or is evaluation entirely based on GEAR’s framework?

Overall, I find the conceptual contribution promising, but the lack of clarity in critical implementation details makes it difficult to fully assess the experimental results. I look forward to clarifications during the author discussion period and will revise my review accordingly.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents GEAR, a label-free framework for evaluating abductive reasoning in LLMs. It assesses sets of hypotheses using three interpretable criteria: (1) Consistency with observations, (2) Generalizability across a defined input space, and (3) Diversity among alternative explanations. The authors applied GEAR to four code-based abductive benchmarks and nine LLMs. It reveals that even large models struggle with consistency and produce limited diversity. The authors further use GEAR as a training signal through a momentum-based curriculum DPO approach, improving both diversity and downstream abductive accuracy.

### Strengths
Originality: The authors introduced a principled, label-free evaluation of abductive reasoning grounded in philosophy of science.

Significance: The paper addresses a fundamental gap in assessing LLMs’ capacity to generate plausible, falsifiable hypotheses.

Quality: Experiments are extensive (9 LLMs, 50K hypotheses) and include insightful analyses, e.g., defeasibility and the correlation between GEAR scores and generalization.

Clarity: Well-written with precise definitions and clear visualisations. The momentum-based curriculum is simple yet effective.

### Weaknesses
1. The framework depends on a predefined sample space and deterministic execution, limiting use beyond structured domains.

2. Training evaluation omits comparisons to other prompting baselines.

3. Heuristic design choices (e.g., 3-error stopping rule, sample-space construction) lack sensitivity analysis.

4. The interpretability of diversity metrics (γ, β) could be improved with qualitative examples.

### Questions
1. Why does model size not correlate with abductive diversity—does pretraining or alignment explain this?

2. Could you elaborate on Line 415 (“some benefit from earlier emphasis on format/consistency, others from earlier diversity”)? And can you provide qualitative examples illustrating what “high-diversity” vs. “low-diversity” hypotheses look like?

3. How could GEAR extend to natural-language abduction tasks beyond formal domains?

### Soundness
4

### Presentation
4

### Contribution
3
