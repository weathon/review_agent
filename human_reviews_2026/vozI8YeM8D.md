# EduPersona: Benchmarking Subjective Ability Boundaries of Virtual Student Agents

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 2

## Abstract
As large language models (LLMs) are increasingly integrated into education, virtual student agents are becoming essential for classroom simulation and teacher training. However, their classroom-oriented subjective abilities remain largely unassessed, limiting our understanding of model boundaries and hindering trustworthy deployment. We introduce EduPersona, a large-scale benchmark spanning two languages, three subjects, and ten persona types grounded in the Big Five theory. The dataset contains 1,308 authentic classroom dialogue rounds (12,814 teacher–student Q&A turns) and is expanded via persona stylization to roughly 10× scale (128k turns), forming a comprehensive foundation for evaluation. Building on this resource, we decompose subjective performance into three progressive tasks: (Task 1) basic coherence—alignment between behavior, emotion, expression, voice, and classroom context; (Task 2) student realism; and (Task 3) long-term persona consistency, establishing an evaluation framework grounded in educational theory and empirical validity. We conduct systematic experiments on three representative LLM families, comparing their base versions with ten EduPersona fine-tuned variants. Results show consistent and substantial improvements: +33.6% in Task 1, +30.6% in Task 2, and +14.9% in Task 3, highlighting both the benchmark’s effectiveness and the heterogeneous difficulty of persona modeling. A human–AI alignment experiment further confirms that GPT-4o’s judgments closely match expert consensus. In summary, EduPersona provides the first classroom benchmark centered on subjective student abilities, establishes a decoupled and verifiable paradigm for evaluating virtual learners, and will be open-sourced to support the development of trustworthy and human-like AI for education.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces EduPersona, a large-scale benchmark designed to evaluate subjective abilities of virtual student agents in classroom settings. The dataset spans two languages (Zh/En), three subjects, and ten Big-Five-based personas, built from 1,308 authentic classroom dialogue rounds. The authors propose a three-task evaluation framework: (1) basic multimodal coherence, (2) student realism, and (3) persona consistency. Experiments using LoRA fine-tuning on three open-source LLM families showimprovements across all tasks and reveal persona-specific difficulty patterns.

### Strengths
The paper addresses a timely problem of evaluating subjective abilities of classroom-oriented virtual students. The dataset is motivated, and includes persona and multimodal annotation, providing coverage across subjects and languages. The proposed three-task progression is clear and seemingly grounded in educational theory.

### Weaknesses
The reliance on GPT-based annotation and evaluation raises concerns about circularity and unclear inter-annotator reliability. 

The persona data is automatically generated, limiting ecological validity. 

Educational motivation is plausible, but actual downstream utility for teachers or pedagogy is not demonstrated. 

It is unclear whether performance gains generalize beyond the benchmark itself. 

Analysis could better quantify statistical significance and reporting of evaluation robustness.

### Questions
* Evaluation validity: How do you mitigate circularity when GPT-style models are used both for label generation and evaluation? Have you assessed agreement with independent human raters at scale?

* Persona realism: Since persona stylization is synthetic how confident are you that the personas reflect real classroom behaviour?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents an study to evaluate the quality of benchmarks of large language models (LLMs) deployed for online education and their subjective abilities. The authors present a benchmark spanning two languages, three sub- jects, and ten persona types based on the Big Five theory.

### Strengths
Strengths
- Timely topic focusing on the AI in education and  LLM application
- Reals world class room dataset

### Weaknesses
The claim in the introduction  is not correct “Yet existing evaluation frameworks remain focused on objective tasks such as question answering and accuracy (Lu et al., 2022; Huang et al., 2023; Ang et al., 2023), overlooking the subjective abilities essential to classroom practice.”

The paper misses recent literature. The proposed scenarios by the authors can be done by the Zhou and colleagues approach by simulating a student and giving personalities [1] . Please also see work by Dr. Ashok Goel and Dr. Noboru Matsuda.

References
1. Zhou, X., Zhu, H., Mathur, L., Zhang, R., Yu, H., Qi, Z., Morency, L.P., Bisk, Y., Fried, D., Neubig, G. and Sap, M., SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents. In The Twelfth International Conference on Learning Representations.

### Questions
Please see weakness

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces EduPersona, a large-scale benchmark dataset designed to evaluate the subjective abilities of virtual student agents in educational settings. The authors argue that existing benchmarks focus on objective metrics, such as accuracy, while neglecting crucial subjective traits, including coherence, realism, and persona consistency.

The EduPersona dataset is built from authentic classroom dialogues in two languages (Chinese, English) and three subjects. It is then expanded tenfold by stylizing student responses to align with ten different personas derived from the Big Five personality theory. The dataset also includes multimodal behavior/expression labels. The paper proposes a framework that decomposes subjective ability into three progressive tasks: Task 1 (Basic Coherence—aligning multimodal behaviors with text), Task 2 (Student Realism—assessing whether responses resemble a real student), and Task 3 (Persona Consistency—maintaining a stable persona over time).

The authors benchmark three open-source LLMs and their 30 fine-tuned variants on the EduPersona tasks. The primary finding is that models fine-tuned on their dataset show significant average improvements across all three tasks.

### Strengths
* Originality: It is one of the first works to attempt a systematic, large-scale benchmarking of subjective abilities for AI agents in an educational context. The conceptual framework, which decomposes the broad notion of subjective ability into a progressive hierarchy of three distinct tasks (Basic Coherence, Student Realism, Persona Consistency), is a novel and thoughtful way to operationalize a subjective concept.
* Quality: The authors have constructed a large-scale resource spanning two languages, three subjects, ten personas, and four behavioral dimensions. The detailed process described involved collecting from real classrooms, annotating, and a tenfold expansion via persona stylization, which shows quite some effort in data engineering.
* Clarity: The paper is structured clearly. Figure 1 explains the entire process from data collection to experimental analysis. The detailed descriptions of the dataset and persona configurations in the appendix further enhance the clarity.
* Significance: The work attempts to address the challenge of evaluating AI agents' subjective, human-like abilities in the domain of education. While the paper's evaluation has flaws, it points out an understudied direction for future research.

### Weaknesses
* Circular evaluation: The main finding is that "training on our data makes models better at our tasks," which does not validate the benchmark's utility or relevance to any external ground truth.
* LLM as a Subjective Judge: Using GPT-4o to score realism and consistency is a methodological flaw. The entire evaluation rests on the unproven assumption that the LLM judge is a reliable measure of these complex subjective human-centric concepts.
* Missing human validation: The paper makes claims about modeling human-like traits without presenting any correlation studies demonstrating that its metrics align with human judgments. This is a major omission for a benchmark of this nature.
* Limited scientific insight: The paper does not deliver a strong, novel insight beyond the dataset itself. The results largely confirm the intuitive expectation that fine-tuning works.
* Readability: It is pretty hard to read some texts in Figures 2, 4, 6, and 7 (small font and crowded content).

### Questions
* The evaluation of student realism (Task 2) and persona consistency (Task 3) is central to your contribution, and it relies entirely on scores from a GPT-4o-based evaluator. How can you be sure that this evaluator is reliable? What experiments could you run to validate that its scores for these subjective traits correlate well with human judgments?
* The main experimental result is that fine-tuning on EduPersona improves performance on the EduPersona benchmark. How do you propose to break this circularity? For example, could you show that models fine-tuned on your data also improve on a different, existing benchmark that measures related social or interactional skills?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces EduPersona, a large-scale benchmark designed to evaluate the subjective abilities of virtual student agents powered by Large Language Models (LLMs). The framework assesses performance across three progressive tasks: basic coherence, student realism, and long-term persona consistency. Experiments show that fine-tuning representative LLMs on the EduPersona dataset significantly improves their performance in these areas, validating the benchmark's effectiveness for creating more realistic and consistent AI agents for educational simulations and teacher training.

### Strengths
1. This paper introduces EduPersona, a benchmark for virtual student agents that is multi-lingual (2 languages), multi-subject (3 subjects), and features ten distinct persona types based on the Big Five theory.
2. This paper proposes an evaluation framework that decomposes abstract subjective abilities into three measurable tasks: Basic Coherence, Student Realism, and Persona Consistency.

### Weaknesses
1. The paper claims to be "grounded in educational theory" but fails to elaborate on or integrate specific pedagogical or psychological theories that justify its framework.
2. Imprecise Terminology: The use of "Basic Coherence" for Task 1 is questionable. "Coherence" may not be the appropriate term to unify the distinct concepts of behavior, emotion, expression, and vocal style.
3. Overclaims: The paper overlooks highly relevant work in role-play evaluation (e.g., CharacterChat, RoleLLM, CharacterBench), undermining its claims of novelty regarding the evaluation dimensions.
4. The paper asserts that subjective performance can be decoupled into its three proposed tasks (Coherence, Realism, Consistency) without providing a strong theoretical and empirical justification for why this specific decomposition is a comprehensive or appropriate measure.
5. Key dataset details are missing, such as the rationale for subject selection (Chinese, English, etc.). Furthermore, despite "human-audited" claims, the paper provides no quantitative metrics (e.g., expert inspection data, inter-annotator agreement) to substantiate the dataset's quality.
6. Lack of Human Validation: The evaluations for realism (Task 2) and consistency (Task 3) rely on an automatic evaluator (GPT-4), but the paper presents no study correlating the evaluator's judgments with human experts to validate its reliability.

### Questions
None

### Soundness
2

### Presentation
1

### Contribution
2
