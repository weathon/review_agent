# Productive LLM Hallucinations: Conditions, Mechanisms, and Benefits

- Decision: Reject
- Scores: 6, 6, 6, 6

## Abstract
Hallucinations in large language models (LLMs) are typically regarded as harmful errors to be suppressed. We revisit this assumption and ask whether, and under what conditions, hallucinations can instead be beneficial. To address this question, we introduce $\textbf{HIVE}$ ($\textbf{H}$allucination $\textbf{I}$nference and $\textbf{V}$erification $\textbf{E}$ngine), a task-agnostic framework that systematically evaluates the impact of hallucinated semantics across diverse tasks and models. By unifying generation, discrimination, and downstream evaluation, HIVE enables controlled comparative assessments of how hallucinations alter overall model performance. Extensive experiments on nine datasets and ten models show that hallucinations can yield substantial improvements up to $\textbf{+17.2}$ \% in accuracy especially in open-ended domains such as reasoning, biomedical, and vision language tasks. Stronger models consistently harness hallucinations, while weaker ones are more volatile. Mechanistic analyses show that hallucinations broaden semantic coverage, stabilize reasoning trajectories, and follow an inverted-U profile where moderate strength maximizes benefits across diverse tasks. These findings reframe hallucination from a defect to a controllable cognitive resource, suggesting opportunities for evaluating and training LLMs not merely to avoid hallucinations, but to exploit them constructively.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores the counterintuitive phenomenon that hallucinated inputs can, under certain conditions, enhance model performance across a range of tasks and architectures. The authors introduce HIVE, a systematic framework designed to evaluate the impact of hallucinations on downstream task performance. Using this framework, they generate both hallucinated and faithful captions for the same input and quantify the performance gap between two settings. The analysis spans both textual and multimodal tasks and involves multiple large language models. Empirical results demonstrate that hallucinated captions consistently improve accuracy on multimodal and perception-driven tasks. These findings provide novel insights into the productive role of hallucinations and open promising directions for leveraging controlled hallucination to enhance LLM reasoning capabilities.

### Strengths
- The proposed evaluation framework is conceptionally simple and task-agnostic
- The authors conduct comprehensive evaluation across diverse datasets, with significance testing and robustness checks, to validate the effect of hallucinations on model performance
- The paper is well structured and easy to follow. The implementation detail of HIVE framework and analysis is well documented.

### Weaknesses
The conclusion that hallucinations promote intra-chain convergence is not well supported by Figure 4. Examining the step-wise cosine similarity of reasoning chains with raw input and faithful input would provide more evidence.

### Questions
Some experiment settings and discussions in the analysis section require further clarity: 

- In HIVE workflow, how do you construct the contrastive pairs based on multiple candidate captions?
- In the reasoning convergence analysis (Section 4.3), what is the step-wise cosine similiarity of reasoning chains when the input contains faithful captions? Is the pattern different from that in Figure 4?
- In Section 4.5, how do you control the “level of hallucinations”?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper challenges the conventional view that hallucinations in large language models (LLMs) are always undesirable. It introduces the concept of productive hallucinations—outputs that deviate from ground truth but enhance reasoning, creativity, or generalization. The authors propose HIVE (Hallucination Inference and Verification Engine), a unified framework that systematically compares faithful versus hallucinatory augmentations across multiple tasks and modalities. Experiments on nine benchmarks (including reasoning, perception, and multimodal tasks) across nine models demonstrate that moderate hallucinations can sometimes improve downstream accuracy by up to 17.2%, particularly in open-ended reasoning settings.

### Strengths
+ The paper presents a novel and thought-provoking perspective, reframing hallucinations as potentially beneficial under controlled conditions, which challenges a dominant assumption in LLM research.

+ The proposed HIVE framework is technically sound and broadly applicable, offering a structured way to quantify and evaluate the effects of hallucinations across diverse tasks and models.

+ The experimental validation is extensive and convincing, covering multiple benchmarks, models, and modalities, and demonstrating clear empirical evidence for the concept of productive hallucinations.

### Weaknesses
- The theoretical grounding for why certain hallucinations are productive remains underdeveloped, as the paper largely relies on empirical observations without a deeper cognitive or information-theoretic explanation.

- The scope of evaluation is limited to short-term performance metrics, leaving questions about long-term reliability, factual consistency, and safety implications of encouraging controlled hallucinations.

- The paper has limited algorithmic novelty. Despite its solid analysis and empirical breadth, the paper’s core contribution lies primarily in the evaluation framework and observations. It does not propose a new model or training approach beyond HIVE’s evaluation setup, which may limit its technical novelty.

### Questions
Please refer to the weaknesses.

### Soundness
2

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
3

### Summary
The paper argues that some LLM hallucinations can be productive under the right conditions and introduces HIVE, a framework that generates, filters, and evaluates hallucinated semantics to test their impact. It shows that hallucination-augmented inputs can boost performance in open-ended, perception-like tasks, while effects are mixed or negative in strict rule-driven domains. The authors explain the gains by showing that hallucinations broaden semantic coverage and raise semantic entropy, diversifying reasoning without disrupting convergence. They advise using a moderate strength, reporting an inverted-U response where balanced doses work best across settings.

### Strengths
1. Novel contribution: The paper reframes hallucination as a controllable resource and introduces a unifying, general-purpose framework (HIVE) to study when and why it helps across modalities. 

2. Methodological soundness & breadth: The design enables apples-to-apples, controlled comparisons (raw vs. faithful vs. hallucinatory) and uses an ensemble discriminator validated on benchmarks; the setup is task-agnostic and scalable across models and tasks. I appreciate the authors' effort in this.

3. Mechanistic insight with stability assurances: The authors smartly tie gains to broadened semantic coverage and higher semantic entropy while showing intra- and inter-chain convergence is preserved. 

4. Actionable guidance & good presentation. The paper offers practical knobs and presents the work clearly with an intuitive case study and well-organized structure. It was very easy to read and follow.

### Weaknesses
Labeling reliability & narrow metrics: HIVE’s conclusions hinge on an ensemble detector to label captions as faithful vs. hallucinatory -- even the authors note hallucination detection is inherently imperfect -- while downstream evaluation is instantiated mainly as accuracy.


Suggestion: I recommend moving the Experimental Setup (Appendix S7) to the main text, as it contains essential information.

### Questions
I don't have any questions so far.

### Soundness
3

### Presentation
4

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
The paper proposes a framework called HIVE, to systematically evaluate the impact of hallucinations on a particular task and/or model. The aim is to recognize settings where hallucinations can be beneficial. The paper provides analyses over several datasets and models, both language and multi-modal, showing the benefits of hallucinations in some settings while causing harm in others. The paper also does a study of reasoning chains and their relationship with hallucinations, highlighting how hallucinations can help with reasoning chain stability.

### Strengths
1. The paper attacks an important problem in LLMs. Hallucinations have a predominantly negative reputation in the field, and the paper aims to show how they can, in fact, be useful in some cases.
2. The paper does an extensive analysis of several different models as well as datasets, both language and multi-modal models, which is a great way to study overarching trends in when hallucinations can be helpful.
3. The paper is well written and was easy to follow.

### Weaknesses
1. While I enjoyed the overall analysis of several tasks and models and whether hallucinations benefit them or not, the framework itself feels restrictive to me. It explicitly focuses on adding a 'caption', which is a hallucination (or not a hallucination), to see the impact of hallucinations on the task. This is clearly only one way to see how hallucinations in model generation can help with eventual performance. For example, another framework could be about studying the reasoning trace of models, identifying hallucinations, and studying how their presence impacts downstream performance.
2. I'm not entirely convinced the variations are not due to just prompt sensitivity. Lack of study of the variance makes me doubt the conclusions. In fact, I disagree with the claim in the experiment setup that 'identical prompts, temperature, and token budget' ensures fair comparison. Just the choice of the playground for comparison, even though identical for everyone, can implicitly favor one behavior over others (https://proceedings.mlr.press/v279/ganesh25a.html). It is important to vary the prompts, the temperature, and the token budget, and see whether the trends of certain tasks or models benefiting from hallucinations actually persist.

### Questions
1. Despite the discussions in (4) (line 292), it's still unclear to me why the 'benefits' of hallucination are dependent on the model so much. I would expect that if a task benefits from 'creative thinking', it should benefit most models. What exactly do authors mean by 'models with stronger hallucination–handling ability' and which models are these?
2. Is 'hallucinations' really the correct term for the phenomenon discussed here? The paper uses the following definition of hallucinations in the introduction: 'information inconsistent with the given input'. But it seems to me that the motivation isn't to allow inconsistent or wrong information, but just new information that might not be verifiable, given the input. I understand the choice to use the term 'hallucinations' in the title, since that is the term accepted more widely in the community and thus is important for the paper's visibility. But I'm curious to hear if the authors think it is still the right choice for the rest of the paper, or maybe they would have preferred a different term or definition (there is a lot of work on trying to define 'hallucinations' and discussion of other similar terms, for example - https://aclanthology.org/2024.emnlp-main.375/)? 
2. Comment: Table 1 markers for how much performance has increased or decreased are incorrect (GPT-3.5 AntiCP2, Claude-3 Sonnet multiple datasets).

### Soundness
2

### Presentation
3

### Contribution
2
