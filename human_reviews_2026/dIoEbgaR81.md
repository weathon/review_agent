# SKATE, a Scalable Tournament Eval: Weaker LLMs differentiate between stronger ones using verifiable challenges

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 8, 2

## Abstract
Evaluating the capabilities and risks of frontier AI models is paramount, yet current methods demand extensive domain expertise, hindering their scalability as these models rapidly evolve. We introduce SKATE: a novel evaluation framework in which large language models (LLMs) compete by generating and solving verifiable tasks for one another. Our core insight is to treat evaluation as a game: models act as both task-setters and solvers, incentivized to create questions which highlight their own strengths while exposing others' weaknesses. SKATE offers several key advantages, balancing scalability, open-endedness, and objectivity. It is fully automated, data-free, and scalable, requiring no human input or domain expertise. By using verifiable tasks rather than LLM judges, scoring is objective. Unlike domain-limited programmatically-generated benchmarks (e.g. chess-playing or spatial reasoning), having LLMs creatively pose challenges enables open-ended and scalable evaluation. As a proof of concept, we introduce LLM-set code-output-prediction (COP) challenges as a verifiable and extensible framework in which to test our approach. Using a TrueSkill-based ranking system, we evaluate six frontier LLMs and find that:
(1) weaker models can score stronger ones consistently, reliably differentiating between them, and
(2) LLM-based systems are capable of self-preferencing behavior, generating questions that align with their own capabilities, and
(3) SKATE automatically surfaces fine-grained capability differences between models.
Our findings are an important step towards general, scalable evaluation frameworks which can keep pace with LLM progress.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a judge-free evaluation where models set and solve verifiable tasks (i.e., code output prediction MCQs) and are then ranked with TrueSkill while controlling option/order noise and near duplicate questions. Experiments show stable rankings, that weak models can consistently score stronger ones, and demonstrate measurable self-preference when models author questions that they can answer.

### Strengths
1) The peer-generated, verifiable tournament evaluation framework is an interesting approach towards a scalable and (hopefully reliable) evaluation framework. 
2) The work carefully controls MCQ option/order noise with re-sampling and convergence criteria, adds guardrails that reduce reward hacking, and attempts to enforce question uniqueness via embedding based clustering.

### Weaknesses
1) The results are limited to COP. The framework would be stronger with at least one additional verifiable task family.
2) It seems like only one embedding model is used for uniqueness filtering (my apologies if I am mistaken). What happens if a different model is used? How robust is uniqueness filtering to the choice of embedding model?
3) Rankings may be dependent on TrueSkill mapping (eg relative vs absolute), p-threshold, number of distractors, number of rounds, etc.

### Questions
1) Would it be possible to add a second verifiable task family?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a methodology for LLMs collaboratively and interactively judge each other's performance, by having the LLMs play a game wherein they generate questions whose answers can be verifiably answered. The aim is to pose questions they can answer while the other LLMs cannot distinguish between true and incorrect answers. The LLMs are ranked by their abilities to both correctly answer other LLM's questions, and to stump the other LLMs.

### Strengths
- the evaluations are verifiable, differing from common LLM-as-Judge frameworks like Alpaca-Eval, where the biases of a judge LLM influences the evaluation
- no human input is required in generating the evaluation data sets
- the resulting rankings (on code output prediction) are demonstrated to be stable to the addition of more LLMs
- the methdology is shown to elicit some fine-grained performance differences between LLMs,. and incorporate varying levels of prior knowledge to assist the LLMs in generating difficult questions
The primary significance is in the verifiability of the methodology.

### Weaknesses
the methodology is limited to automatedly verifiable tasks, while many tasks we would like to understand the performance of LLMs on (e.g. alignment), cannot be automatically verifiable
- the methodology similarly is limited to tasks that can be posed as multiple choice questions, so it cannot be used to e.g. compare the summarization abilities of LLMs
- although the methodology encourages diversity, it does not ensure coverage, so e.g. in the specific case studied in the paper, it could be the case that LLMs all share a common blind spot for code output prediction that is not identified by this methodology.
- this methodology was only instantiated and tested on a code output prediction task; it would have been more persuasive to see it perform on at least one other, significantly different task

### Questions
- please address the statistical issues in using adaptive sampling to determine p(correct): this means that models with less certainty are given a larger budget of samples.
- please be more comprehensive with your discussion of how your work compares to related gamified LLM-vs-LLM evaluation methodologies, e.g. the GTBench paper of Duan et al. in NeurIPS 2024 and the ZeroSumEval paper of Alyahya et al. in ACL 2025; in particular, how does your use and emphasis placed on verifiability differ from theirs? Making this more clear would help me better judge the novelty and significance of your contribution.
- the author list in the bibliographic entry for Humanity's Last Exam spans several pages, it should be shortened! Several other author lists should be similarly be shortened.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces SKATE, a novel LLM evaluation framework where models compete by generating and solving verifiable tasks for one another. Models act as both "task-setters" and "solvers" in a tournament format, creating Code-Output-Prediction (COP) challenges. The system uses TrueSkill ranking and tests 6 frontier LLMs, finding that: (1) weaker models can differentiate stronger ones, (2) models exhibit self-preferencing in question generation, and (3) automatic discovery of differentiating questions.

### Strengths
1. The whole idea is novel and well-motivated.The paper identifies two major bottlenecks in LLM evaluation: (1) evaluation requires costly, non-scalable human-annotated ground truths, or (2) relies on LLM-as-judge which is easily manipulated. SKATE attempts to address both through a peer-challenge multi-model system.
2. Methodologies are sound and considerate, such as robust scoring algorithm to adress the multiple-choice biases and question clustering to increase the diversity of questions.

### Weaknesses
1. While the authors claim that COP tasks provide "a general substrate for evaluating model capabilities," this paper provides zero empirical evidence that SKATE can work with other task types. The COP tasks tested here don't even resemble a typical coding evaluation where models' generation or algorithmic problem-solving capabilities are tested. Any model with code execution tools could solve COP tasks. Without demonstrating SKATE generalisation ability on more diverse verifiable tasks (such as mathematical proofs, puzzles), the claim is questionable.

2. Although COP tasks can be verified through code execution easily, the authors provide no evidence that general verifiable tasks can have similar guarantees. For instance, if a model generates a mathematical problem as a challenge (the task itself is verifiable with ground-truths but extremely hard with only models), verifying the correctness of the answer may be non-trivial or even intractable.

3. While collective knowledge from multiple models works well for training scenarios, using it for evaluation may be problematic because it provides no ground-truth anchor—evaluation results depend entirely on which specific models are included in the cohort. The experiments in Section 6.2 only test adding models sequentially by capability order, which is insufficient to establish robustness. A rigorous validation would require testing with diverse model combinations to show rankings remain consistent, or anchoring the evaluation with human-verified questions to provide external validation.

4. The empirical findings presented in Section 6, while technically sound, lack novelty and offer limited new insights into LLM capabilities. The observation that weaker models can reliably score stronger ones, and models prefer their own questions, have been found two years ago.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2
