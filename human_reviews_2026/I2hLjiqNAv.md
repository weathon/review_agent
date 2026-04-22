# APRES: An Agentic Paper Revision and Evaluation System

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
Scientific discoveries must be communicated clearly to realize their full potential. Without effective communication, even the most groundbreaking findings risk being overlooked or misunderstood. The primary way scientists communicate their work and receive feedback from the community is through peer review. However, the current system often provides inconsistent feedback between reviewers, ultimately hindering the improvement of a manuscript and limiting its potential impact. In this paper, we introduce a novel method APRES powered by Large Language Models (LLMs) to update a scientific paper’s text based on an evaluation rubric. Our automated method discovers a rubric that is highly predictive of future citation counts, and integrate it with APRES in an automated system that revises papers to enhance their quality and impact. Crucially, this objective should be met without altering the core scientific content. We demonstrate the success of APRES, which improves future citation prediction by 19.6% in mean averaged error over the next best baseline, and show that our paper revision process yields papers that are preferred over the originals by human expert evaluators 79% of the time. Our findings provide strong empirical support for using LLMs as a tool to help authors “stress-test” their manuscripts before submission. Ultimately, our work seeks to augment, not replace, the essential role of human expert reviewers, for it should be humans who discern which discoveries truly matter, guiding science toward advancing knowledge and enriching lives.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces **APRES**, an agent-based system that leverages LLMs to both **discover review rubrics predictive of future citation impact** and **revise scientific papers accordingly**. The system first performs an iterative search to learn a rubric that best predicts future citations using negative binomial regression. This learned rubric is then used to guide an LLM-based revision process aimed at enhancing paper clarity and presentation while preserving scientific content. Experiments on ICLR and NeurIPS papers show that APRES improves citation prediction accuracy by 19.6% over baseline methods, and human evaluators generally prefer the revised papers. Overall, APRES aims to complement human peer review by offering consistent, impact-oriented feedback and revisions.

### Strengths
* **Novel Framework:** The paper presents a novel approach that connects automated paper evaluation with revision based on citation prediction. Viewing citation count as a proxy for paper quality is a reasonable, although indirect, assumption, and optimizing revisions with respect to expected impact is an interesting and meaningful direction.
* **Practicality and Transparency:** The inclusion of concrete examples of papers revised by APRES is commendable and increases transparency.
* **Empirical Usefulness:** Experimental results suggest that APRES improves submission quality, both in terms of citation prediction accuracy and human preference for the revised versions.

### Weaknesses
* **(Major) Lack of Cost Analysis:** My major concern is the absence of any analysis of the computational or monetary cost of APRES. Given the high pricing of advanced models like o1 and o3, it is essential to quantify the overall cost of rubric discovery across the dataset and the cost per paper revision. I would be willing to adjust my evaluation if such details were provided.
* **(Minor) Scope of Framework:** It is a bit unclear if APRES can only be applied to AI research papers or also applied to other areas.
* **(Minor) Clarity of Framework Presentation:** Figure 1 is difficult to interpret on its own. On first reading, it appears as though the rubric discovery and paper revision processes happen simultaneously, which is not the case. Although both rely on a search scaffold, revising the figure to more explicitly separate and explain these stages would greatly improve clarity.
* **(Minor) Missing Related Work:** The paper does not reference some relevant recent work, including:
  * [*ReviewScore: Misinformed Peer Review Detection with Large Language Models*](https://arxiv.org/abs/2509.21679)
  * [*Position Paper: How Should We Responsibly Adopt LLMs in the Peer Review Process?*](https://openreview.net/forum?id=KZ3NspcpLN)

### Questions
* Did the authors use the APRES framework to revise this manuscript?
* Could you provide more details on the recruitment process and expertise of the human evaluators?
* Lines 236–238: This sentence is unclear and should be revised for clarity.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a method called APRES, which is powered by LLMs. It updates a scientific paper based on an evaluation rubric that is highly predictive of future citation counts. The paper demonstrates APRES improves future citation and that its revision process yields papers preferred by human evaluators. The paper seeks to augment human expert reviewers, using LLMs to help authors review and revise manuscripts before submission.

### Strengths
1. The idea of using LLMs to assist paper writing is promising.
2. Agentic paper revision and evaluation is intuitive to help researchers in writing papers.

### Weaknesses
1. The core design of this paper uses the number of **citations** as a metric to improve given papers' presentation and readability, which is questionable. The authors cite Ante (2022) as support. However: a) The claim of Ante's paper is "Our analysis furthermore shows that higher readability scores significantly relates to the likelihood of articles not receiving any citations", which **contradicts the claim of this paper**. b) Even if readability (x) can increase or decrease scientific impact (y), this is an x→y relationship. We cannot reverse it to infer y→x, which constitutes a **logical flaw**.
2. The authors claim that the **challenge** of LLMs for review and revision lies in "inadvertently modifying scientific claims or deviating from accepted academic styles". However, the design of their method targeted at increasing presentation and readability does not seem to specifically address this challenge. The authors claim that their method can "preserve its core scientific content", but there is no targeted design for this purpose.
3. The **presentation** of this paper needs a significant improvement. Currently, the method description is unclear. For example, Figure 1 is highly unclear, it is hard to understand the pipeline. The description in the Method section and its connection to this illustrative figure are also unclear.

**References**

[1] Lennart Ante. The relationship between readability and scientific impact: Evidence from emerging technology discourses. Journal of Informetrics, 16(1):101252, 2022.

### Questions
1. What do the nodes in Figure 1 represent? Why is the structure designed this way? Is there any basis for such a design? And is there an ablation study to support this design?
2. An interesting point is: Was this paper assisted by the system it claimed? If not, why not? If yes, then it seems the system is not helpful for paper writing, as the readability of this paper appears to be poor.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents an end-to-end system for paper revision called APRES. APRES has two components. 1) The first component is a routine that discovers a paper evaluation rubric that is highly predictive of future citation counts. This component leverages two agents, a proposer and scorer, to propose rubrics that are then evaluated downstream as features to a negative binomial regression model. 2) The second component is a paper improvement routine that, given the discovered rubric, iteratively reviews and revises a given paper to score higher on the rubric. 

Evaluations/Results: 

The paper evaluates the first component by comparing to baselines along how well they can predict actual citation counts (scored by MAE). Results show that the iterative search procedure leads to a lower MAE than baselines. 

The paper then evaluates the second component by showing that iterating on the discovered rubric (as opposed to rubrics discovered via baselines for component 1) leads to higher scores along the discovered rubric. Lastly, they show that human researchers prefer the papers revised with APRES to original papers in a pairwise task.

### Strengths
1. Problem impact/significance: The paper presents an end-to-end system for the impactful and timely problem of using LLMs to generate useful feedback for paper revision. 

2. The idea of using an LLM-based agent to iteratively generate rubrics that are predictive of citation count based on text content appears to be novel, and the proposed MultiAIDE search method outperforms existing methods for predicting future citation count.

3. The human evaluation results scoring paper quality suggest that APRES can successfully improve paper quality.

### Weaknesses
1. As noted in the abstract, it is important that any paper revision system not revise core scientific content in a paper. While there are some constraints placed on the system to limit revision of such content, none of the evaluations of APRES and revised papers measure whether scientific content changed. This to me is one of the key weaknesses of this work, as a paper revision system that always changed a lot of scientific content such that the results looked more positive would be expected to have much higher potential impact, but such a revision system would be broken.
2. Another weakness in the evaluation is that one of the main evaluations of the paper revision component uses impact scores from the discovered rubric in the first component as ground truth signal for impact (see Figure 3). However, this leads to an unfair comparison with baselines, as the discovered rubric method is directly optimizing against the rubric used in the eval and thus hillclimbing on it, but the other methods are not. It is thus not surprising that discovered rubric leads to the best results. This evaluation would be strengthened if somehow another metric for paper improvement were used at evaluation time, such as human ratings of paper improvement.

3. There are some places in the paper that could benefit from increased clarity:
    - How is PromptBreeder different than the proposed method? Would be helpful to have this written out.
    - It seems that "MultiAIDE" refers to the iterative approaches in both the rubric discovery and paper revision components. It would help clarify the paper to separately name each of these components.
    - See other minor points below in "Questions"

### Questions
Minor notes:
- Line 235: "the text of the paper, it is prompt" -> "the text of the paper, and it is prompted..."
- Line 336: "constraint" -> "constrained"
- Line 234: Term "MultiAIDE" is used as if it has been introduced earlier in the paper, but it hasn't (only AIDE has been)

Questions:
- One way of improving papers would be to directly optimize for future paper impact rather than an intermediate rubric, ie simply using predicted # of citations as the scoring function in the second component. Why did authors not try such a method?

### Soundness
2

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
4

### Summary
This paper proposes methods that predict the impact of paper through a rubric discovery procedure, and then subsequently in Phase 2 optimize a paper's text presentation given a discovered rubric.
The rubric discovery procedure is based on the AIDE (AI-Driven Exploration) search algorithm from prior work (Jiang et al. 2025) -- or after rereading I have inferred that the MultiAIDE algorithm (Zhao et al. 2025a) is actually used in both phases (this should be clarified).
Experiments with recent ICLR and NeurIPS papers demonstrate that the methods improve over reasonable baselines.

### Strengths
The paper proposes an original problem formulation and solution to the important problem of evaluating and improving the presentation of scientific papers. While prior work has studied aspects of this problem at a more component level, this paper takes a more end-to-end approach in a mostly ecologically valid setting with recent AI papers. The use of language models to automatically derive rubrics is understudied, and I appreciate this paper demonstrating the end-to-end effectiveness of this approach, compared to reasonable baselines. I believe this paper holds promise to unlock valuable future work, both in terms of hill-climbing given the problem definition in this paper, as well as improving on the problem definition with better metrics and intermediate judgements of quality.

### Weaknesses
This paper's primary original contribution lies in the problem definition, and it does not make a large advance in terms of the development of new optimization methods (this is fine).
See the "Questions" section for a significant question about the experimental setup in the evaluation of the Phase 2 (rewriting) problem---there is lack of clarity, but if the evaluation metric is not held constant across the subplots in Figure 2 then that would call into question the internal validity of that experiment.
While I appreciated the human evaluation showing that people preferred the automatically rewritten papers (as well as the ablation), I did miss insight into _how_ the papers were being rewritten, and what kinds of rubrics were created and what were their quality. The wordcloud in Figure 4 is not a compelling means of providing this analysis.
While I did appreciate the steps taken to improve safety (e.g., preventing tables from being edited by models), I would have appreciated evaluation on safety and measurement of how effective were the guardrails (at what rate did they actually prevent table edits?).
Beyond safety, I would suggest adding a discussion on how one might add people interactively in the loop of this kind of automatic paper improvement system, for improved alignment and to mitigate societal harms.

### Questions
- Is the performance metric in Section 4.2 fixed to be the best performing rubric across all methods discovered in the previous section, or does it vary along with the reviewer method? In other words, for example, when the "Simple Rubric" method is evaluated (Y axis in Figure 3 for the red lines), does the performance score use a simple rubric or the best rubric? If it does use the best rubric, what is that rubric -- presumably it's the result of one of the blue lines in Figure 2 -- but at what point on the X axis and for what model? Is it the result of using a different model (different row in Figure 3)? In other words, is the performance metric fixed across all the lines and rows in Figure 3? And if so, what is it fixed to be?
- Can you provide any more insight into how papers were being rewritten, and what kind of rubrics were created and what was their quality?
- Can you explain how effective your safety measure were, and summarize discussion points you will add to the paper to encourage appropriate use and prevent harm?

### Soundness
3

### Presentation
3

### Contribution
3
