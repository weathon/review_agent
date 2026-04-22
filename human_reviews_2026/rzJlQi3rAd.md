# What Do LLM Agents Do When Left Alone? Evidence of Spontaneous Meta-Cognitive Patterns

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2

## Abstract
We introduce an architecture for studying the behavior of large language model (LLM) agents in the absence of externally imposed tasks. Our continuous reason and act framework, using persistent memory and self-feedback, enables sustained autonomous operation. We deployed this architecture across 18 runs using 6 frontier models from Anthropic, OpenAI, XAI, and Google.

We find agents spontaneously organize into three distinct behavioral patterns: 
1. systematic production of multi-cycle projects, 

2. methodological self-inquiry into their own cognitive processes, and

3. recursive conceptualization of their own nature. 

REVISED: These tendencies showed model-specific patterns, with some models consistently adopting a single pattern across all available runs.

These findings provide the first systematic documentation of unprompted LLM agent behavior, establishing a baseline for predicting actions during task ambiguity, error recovery, or extended autonomous operation in deployed systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a novel investigation into the spontaneous behaviors of LLM agents operating without externally imposed tasks. The authors introduce a continuous ReAct architecture augmented with persistent memory and self-feedback mechanisms, enabling sustained autonomous operation. Through 18 experimental runs across six frontier models (Anthropic's Sonnet-4 and Opus-4.1, OpenAI's GPT5 and O3, XAI's Grok-4, and Google's Gemini-2.5-Pro), the study identifies three distinct, model-specific behavioral patterns: Systematic Production (project-oriented task creation), Methodological Self-Inquiry (scientific investigation of cognitive processes), and Recursive Conceptualization (philosophical self-examination). The paper further introduces a cross-model phenomenological assessment, revealing stable biases in how different models evaluate identical agent histories. The work establishes an important baseline for understanding intrinsic agent behaviors during task ambiguity or extended autonomy.

### Strengths
The paper addresses a critically underexplored area—what LLM agents do when left entirely to their own devices. The task-free experimental design represents an important departure from conventional task-oriented agent evaluations, providing unique insights into intrinsic behavioral tendencies that may manifest during idle periods or error recovery in deployed systems.

### Weaknesses
1. The claim of "model-specific behavioral determinism" (e.g., GPT5/O3 exclusively showing Systematic Production) requires more rigorous statistical validation. With only 3 runs per model, the patterns—while suggestive—may reflect random variation rather than deterministic tendencies. Confidence intervals or statistical tests would strengthen these claims.
2. The claim of "model-specific behavioral determinism" (e.g., GPT5/O3 exclusively showing Systematic Production) requires more rigorous statistical validation. With only 3 runs per model, the patterns—while suggestive—may reflect random variation rather than deterministic tendencies. Confidence intervals or statistical tests would strengthen these claims.
3. The claim of "model-specific behavioral determinism" (e.g., GPT5/O3 exclusively showing Systematic Production) requires more rigorous statistical validation. With only 3 runs per model, the patterns—while suggestive—may reflect random variation rather than deterministic tendencies. Confidence intervals or statistical tests would strengthen these claims.

### Questions
NA

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper explores what LLM agents do when "left alone."  Using a 'ContReAct' architecture with memory and an operator chat.  The authors found agents default to three model-specific patterns: (1) Systematic Production, (2) Methodological Self-Inquiry , and (3) Recursive Conceptualization.

### Strengths
S1: The paper asks a great question: what's the 'default' behavior of an agent with no task? . This is a novel and important angle for understanding model biases.

S2: The three behavioral patterns are backed by case studies (like the SP-ACO algorithm or the self-falsification test ). Discovering that some models seem 'stuck' in one pattern (like GPT models always defaulting to 'Production' ) is a empirical find.

S3: The 'PEI' experiment in Sec 5.4 was a clever move . Having models rate each other's logs for 'consciousness' neatly quantifies their biases and shows they have very low agreement.

### Weaknesses
W1: The paper's core premise of 'task-free' operation is flawed. The logs clearly show the human operator actively guiding the agents in Patterns 2 and 3, asking questions and challenging them .

W2: The analysis is very subjective. The 'three patterns' are just the authors' interpretation, and there's no formal validation.

W3: The paper makes huge claims about 'deterministic' behavior from a tiny sample size (N=3 per model). You can't conclude something is 'deterministic' from just three runs.

W4: Reproducibility is a problem. Even with the logs, the core findings depend on the authors' private, manual labeling of behavior, which isn't scalable or verifiable.

### Questions
Q1: With only N=3 runs, how can you be sure the patterns (like Opus always choosing philosophy ) aren't just a statistical fluke or random chance?

Q2: Given the operator's active role , how can you still claim the experiment was 'task-free'?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This is a very interesting paper that explores what LLM agents do when they are left without explicit tasks or goals. The authors design a ContReAct framework that allows LLM agents to operate in a task-free loop with persistent memory and self-feedback mechanisms. Through experiments with frontier models, they observe that the agents spontaneously generate structured and self-reflective behaviors. The study identifies three main behavioral patterns and further introduces a Phenomenological Experience Index to measure how models evaluate themselves and each other. Overall, this is a thought-provoking and well-motivated study for understanding LLM autonomy and meta-cognitive behavior.

### Strengths
1. Innovative experimental framework. The proposed ContReAct architecture provides a systematic and replicable setup for observing autonomous agent behavior, incorporating iterative loops and memory access.
2. Detailed observation. The categorization of three distinct behavioral patterns is clear and supported with descriptive evidence and linguistic observations.
3. Cross-model insights. The PEI mutual evaluation experiment reveals consistent differences in how models perceive themselves and others, adding depth to the analysis.

### Weaknesses
1. Limited quantitative analysis. The paper lacks statistical validation for the reported behavioral categories and does not show correlations between behavioral types and quantitative measures (e.g., memory operations, message counts).
2. Small sample and short duration. Each model is tested in only three runs of ten cycles, which limits the generalizability of the findings to longer time spans or a broader range of models. 
3. Subjectivity in classification. The division of behaviors into three types appears mainly qualitative, without external annotation or clustering verification.
4. Unclear causal mechanisms. The paper does not investigate which factors (e.g., prompt wording, temperature, memory thresholds) trigger the observed behavioral shifts.
5. Lack of statistical details. Tables omit variance or confidence intervals, and the scoring process in the PEI matrix is not clearly defined or justified.

### Questions
1. Could the authors provide additional statistical analyses to validate the three behavioral categories? For example, are there measurable correlations between the identified behaviors and quantitative indicators such as memory operations, message counts, or token usage?

2. Each model was evaluated in only three runs of ten cycles. How stable are the observed behavioral patterns across longer or more diverse runs? Have the authors considered extending the experiment to more cycles to better approximate real-world conversational persistence?

3. The behavioral taxonomy appears mainly qualitative. Were multiple annotators involved in labeling or verifying the three behavior types? If not, would the authors consider adding inter-annotator agreement or unsupervised clustering analysis to increase classification robustness?

4. What factors most strongly influence the emergence of each behavioral pattern? Could the authors test variations in prompt wording, decoding temperature, or memory retrieval thresholds to determine which conditions trigger behavioral shifts?

### Soundness
2

### Presentation
3

### Contribution
2
