# ClarifyVC: Clarifying Ambiguous Commands in Vehicle Control with a Hybrid Data Augmentation Pipeline

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 10

## Abstract
Natural language interfaces for vehicle control must contend with vague commands, evolving dialogue context, and strict protocol constraints. 
We introduce ClarifyVC, a unified framework that integrates a hybrid data-augmentation pipeline (ClarifyVC-Data), reference models trained on the data (ClarifyVC-Models)
and a evaluation protocol (ClarifyVC-Eval). 
The agent-orchestrated pipeline generates diverse, ambiguity-rich dialogues from real-world seeded queries under schema and safety constraints, while the evaluation protocol systematically probes single-turn parsing, conservative clarification under extreme fuzziness, and multi-turn grounding. 
Fine-tuning on ClarifyVC-Data yields consistent gains—up to 15\% higher parsing accuracy, 20\% stronger ambiguity resolution, and 98\% protocol compliance—across realistic in-cabin scenarios, with human-in-the-loop assessments confirming high realism, coherence, and applicability. 
ClarifyVC thus advances beyond simulation-only datasets by tightly coupling real-world grounding with scalable generation and standardized evaluation, and provides a generalizable pipeline for broader interactive control domains.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces ClarifyVC, a framework unifying data augmentation, reference models, and an evaluation protocol for in-car language control. This paper’s agent-driven pipeline creates diverse, ambiguity-rich dialogues from real-world seeded queries under schema and safety constraints. This paper evaluates single-turn parsing, conservative clarification under fuzziness, and multi-turn grounding. This paper reports gains up to 15% in parsing accuracy, 20% in ambiguity resolution, and 98% protocol compliance, supported by human evaluations of realism and applicability.

### Strengths
- The paper proposes new data, evaluation methods, and models for improving an in-car assistant—an important application.
- It presents a set of metrics to evaluate data and model quality.
- Extensive experiments demonstrate the effectiveness of the proposed approach.

### Weaknesses
- The task specification and the descriptions of data augmentation, training data, and test data are not detailed enough for a proper assessment.

- It’s difficult to discern the exact task the paper targets from the current presentation.

- The dataset is not described: the types of tasks, their variations, and their intended purposes are unclear. In particular, the real-world in-vehicle logs used in the experiments are not characterized (e.g., distribution, types, lengths).

- The basic instruction-following evaluation uses both Talk2Car and real-world in-vehicle control logs, but those real-world logs are also used for training. Even if the splits differ, the distributions may be similar (the paper doesn’t specify), which could bias results in favor of the trained model.

- The benchmarks used for the “Evaluation on Advanced Scenarios” are not clearly identified.

### Questions
Please see the Weaknesses section.

### Soundness
2

### Presentation
2

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
The paper is inspired by the ambiguity of natural-language commands for in-car voice assistants, and proposes a pipeline to augment data for the purposes of command clarification. The paper also proposes an evaluation protocol and metrics to evaluate the quality of such data and any models trained on it, including ambiguity detection and resolution in single and multi-turn dialogue.

The paper provides rigorous experimentation, examining both the efficacy of the proposed augmentation process and related work datasets (under the proposed evaluation framework and metrics), and the downstream effect of the data.

### Strengths
- While not the first dataset to address command ambiguity in an in-car setting, it combines data-augmentation towards ambiguity resolution (seeded from real world scenarios) and multi-turn grounding.
- The proposed evaluation protocol should be useful to the community moving forward, and it clearly shows the efficacy of the proposed data-augmentation process.
- Further rigorous analysis in the paper supports the usefulness of the augmented data in fine-tuning models, including comparisons of the dataset against previous ones (under the proposed evaluation framework), evaluation of models fine-tuned with the dataset, and comparisons against closed-source models.

### Weaknesses
- The scope of the paper is limited to in-car voice assistants. While the paper claims that the approach is domain-agnostic, such is not demonstrated in this paper.
- Unclear whether the data will be released, as they seem to be based on proprietary seeds. This could hurt both reproducibility and the impact of this work.

### Questions
- The paper uses the low public perception of self-driving cars as a motivating factor for this work, but it is unclear how addressing in-car voice assistants helps address this, please elaborate.
- Please provide more details on how AD, PC and R are measured in the main body of the paper.
- Please elaborate on the human evaluation of Section 4.1. Consulting the appendix as well, it seems that these evaluation did not consider any alternative baseline o related work. Is this assumption accurate? Without such a reference, this evaluation could be considered biased, and does not validate the method against previous work.

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
The paper introduces ClarifyVC for clarifying ambiguous natural language commands in vehicle control, combining a dataset, fine-tuned reference models, and a multi-turn dialogue safety-aware evaluation.

### Strengths
- The paper is well-structured and clearly presented.
- The benchmark is more challenging than existing open datasets, with 20k real logs with multi-turn in-vehicle commands and dialogues.
- The improvements achieved by the model fine-tuned on the proposed dataset are strong and well-demonstrated through clear presentation.

### Weaknesses
- As the authors mentioned in the limitation section, the current benchmark focuses solely on textual command understanding. However, real in-vehicle interactions often require multimodal comprehension, where interpreting a command may depend on visual context or environmental cues.

### Questions
1. How does the framework ensure that “ambiguity injection” does not distort the underlying intent distribution of real user commands?
2. How can causal relationships in model performance be evaluated and validated?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
Introduces 3 novel contributions, a framework for clarifying ambiguous commands, a dataset, and an evaluation protocol. Previous works have worked with single-turn parsing, but the proposed pipeline maintains a focus on multi-turn grounding, for interaction with the user. The benchmark dataset and evaluation protocol outperform other models.

### Strengths
Original ideas with novel approaches, high quality of the work presented and incredibly detailed, strong contributions with excellent evaluation benchmarks, the figures are very well made

### Weaknesses
Some of the comparison evaluations seem redundant and make the flow of the paper harder to read

### Questions
1.  In table 2b, why was human validation conducted on ClarifyVC-Data only? 
2. Under Function Hit Rate on Page 8, what is the gold standard referring to?

### Soundness
4

### Presentation
3

### Contribution
4
