# GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 8, 4, 2

## Abstract
We introduce GDPval, a benchmark evaluating AI model capabilities on real-world economically valuable knowledge-work tasks. GDPval covers the majority of Department of Labor O*NET Work Activities for 44 occupations across the top 9 sectors contributing to U.S. GDP (Gross Domestic Product). Tasks are constructed from the representative work of industry professionals with an average of 14 years of experience. We find that frontier model performance on GDPval is improving roughly linearly over time, and that the current best frontier models are approaching industry experts in deliverable quality. We analyze the potential for frontier models, when paired with human oversight, to perform GDPval tasks cheaper and faster than unaided experts. We also demonstrate that increased reasoning effort, increased task context, and increased scaffolding improves model performance on GDPval. Finally, we open-source a gold subset of 220 tasks and provide a public automated grading service to facilitate future research in understanding real-world model capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper describes GDPval, a benchmark of 1320 occupational tasks written and reviewed by professionals in the relevant occupation. Each task involves the creation of a deliverable file (document, slide deck, video, etc). 7 LLMs were tested on GDPval and their outputs were scored by relevant professionals compared to professionally-created deliverables. The best-performing model achieved a win rate of about 45% against human experts. In supplementary analyses, the authors analyze the types of errors performed by different models, how reasoning effort contributes to performance, and how much improvement can be gained by prompt tuning. They also develop an automated grader that nearly achieves the same agreement on GDPval outputs with human graders as humans.

### Strengths
1. This is an impressive and ambitious research project.
2.The question is timely and of very high importance.
3. Gathering realistic tasks from professionals is a much better way of benchmarking economic potential of AI than existing approaches.
4. The amount of time, effort, and money needed to develop, review, and evaluate the tasks is very high.

### Weaknesses
1. The quality of the paper does not live up to the scale and ambition of the methods, with typesetting issues (see lines 075, 262, 288, 294), many missing details, and figures with no explanation or references.
2. The paper is sparse on details of how the models were run (how do these LLMs edit videos or produce CAD files?)
3. The prompts (both for GPDval tasks and the other classifiers run), model outputs, and human reviews do not appear to be available
4. The discussion of related work is very brief, only a few sentences; also, missing a few of the most important related papers:
	- Eloundou et al. "GPTs are GPTs: Labor market impact potential of LLMs." Science 384.6702 (2024): 1306-1308.
	- Tomlinson et al. "Working with AI: Measuring the applicability of generative AI to occupations." arXiv preprint arXiv:2507.07935 (2025).
	- Handa et al. "Which economic tasks are performed with AI? evidence from millions of Claude conversations." arXiv preprint arXiv:2503.04761 (2025).


### Overall evaluation
This research should obviously be published--it's very important and impactful, and the authors deserve credit and recognition for this work. However, the effort put into the paper falls short of the effort put into the research. The paper really should have some more time and care put into the writing: missing prompts and details should be added, all figures should be referenced and described, and related works should be fleshed out. More information should be provided as to the execution and outputs of the models. It's also not clear to me that ICLR is the best venue for this research. Despite that, I have to give the paper a high score simply due to its importance and impact. However, the authors should really improve the paper before any camera ready so it lives up to the impact of the research.

### Questions
1. The tuned prompt in Appendix A.3 implies the GPT-5 agent has significant computer-use ability, but the footnote on page 2 says other LLMs were interacted with through the UI; why was GPT-5 evaluated in a different context?
2. Relatedly, are all of the models listed in Figure 6 capable of outputting videos, slide decks, CAD files, PDFs etc?
3. Figure 3 is never referenced or described. What is the sandbox? Does an expert contribute multiple tasks, only the first of which goes through the iterative sandbox? Who are the "occupational experts" mentioned in the second stage?

### Comments
- The paper is missing prompts for the task-level "digital" classifier and the occupation-level "knowledge work" classifier; without carefully reading the appendix, the fact that there were two classifiers was confusing (the paper mentions both digital tasks and knowledge work, without making the distinction clear)
- O*NET isn't developed by the BLS, so calling them "U.S. Bureau of Labor Statistics Work Activities" in the abstract isn't accurate. (To be pedantic, it's developed by the North Carolina Department of Commerce under sponsorship from the Employment and Training Administration, part of US DoL)
- For percentage of digital tasks, the main text should reference the appendix section where the weighting method is described (page 21). (this weighting method is also a bit strange, since the 1-7 frequency codes have categorial meanings: 1="yearly or less", ..., 7="hourly or more")
- the note about fig 12 being on a different version of the test set should be in the figure caption, as this was confusing
- the more standard names for "SOC-4" and "SOC-6" codes are "SOC codes" and "O*NET-SOC codes," as one is not simply a truncation of the other. Was the BLS-provided crosswalk used to do the mapping between the two taxonomies? (https://www.bls.gov/emp/documentation/crosswalks.htm)

### Soundness
2

### Presentation
1

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces GDPval, a new benchmark to evaluate the performance of large AI models on real-world economically valuable tasks.

### Strengths
* This work introduces a benchmark (GDPval) that directly connects model performance to real-world economic value. The framing around professional tasks across multiple industries gives the research strong societal and economic relevance, and it has clear potential to become a foundational evaluation backbone for assessing AI models’ productivity capabilities.

* The benchmark covers 9 major sectors and 44 occupations, representing a large portion of GDP-contributing work activities. The inclusion of tasks curated from professionals with extensive industry experience makes the evaluation dataset both realistic and diverse.

* The paper reports quantitative progress of recent LLMs on GDPval but also analyzes key performance drivers such as reasoning effort, context size, and scaffolding, offering actionable insights into how models achieve expert-level deliverables.

### Weaknesses
* While the authors claim that they will open source their benchmark, I do not see any anonymous link or supplementary that contains the related benchmark file. Given that this benchmark is the major contribution of this paper, the authors should make sure the benchmark will be open-source.

* Since the benchmark focuses mainly on U.S. GDP-related activities and professional writing tasks, it may underrepresent other economically relevant domains (e.g., manufacturing, logistics, physical or multimodal tasks). Broader coverage would improve generalizability.

* The grading pipeline and automatic scoring system are promising but may introduce bias or subjectivity, particularly when comparing AI outputs to human experts. More discussion on inter-rater consistency and rubric validation would strengthen the reliability of the results.

### Questions
* The paper mentions “We fine-tune GPT-5 on GDPval data and measure clear improvements in human win-rate” in the conclusion part. Given that GPT is a close-looped LLM, can the authors provide further explanation about the meaning of fine-tuning here?

### Soundness
3

### Presentation
2

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
The authors introduce an evaluation dataset for "real-world economically valuable tasks" (from occupations with predominantly digital tasks) and show how current frontier models perform on these tasks compared to experts.

### Strengths
- The authors aim to provide a more granular evaluation to capture the economic impact of AI, an area that is currently lacking data
- The experts that the authors recruited had extensive experience and were very carefully vetted 
- The authors provide a smaller open-source evaluation dataset as well as an initial version of an automated grader for these tasks
- The writing is overall clear.
- The iterative task review process with multiple expert review rounds (and an initial sandbox round with additional feedback) was very well thought-through and designed.

### Weaknesses
Note: I am willing to update my score but currently, there is a lot of methodological information missing in the paper that prevents me from fully assessing the work. Happy to revise my score once more details are provided.

- Line 091: Where do these time estimates come from?
- Line 106: How was the 5% threshold chosen?
- Given that in the current version only jobs with predominantly digital tasks were included, the premise of GDPval seems overstated. The abstract states that GDPVal is “an evaluation assessing AI model capabilities on real-world economically valuable tasks” but doesn't specify that it is limited to occupations with predominantly digital tasks only.
- The paper only states the set of 44 occupations but not the tasks/task descriptions for each of these. I strongly encourage the authors to add, to the appendix, a list of high-level summaries of the tasks and to which occupation they belong to allow for a more meaningful assessment of the paper. For example, I find it puzzling that "registered nurse" is in the 44 occupations considered and would have needed a summary of tasks added for this occupation to verify whether this classification made sense. I understand the authors’ concerns that they do not want to make all task specifics/prompts/files/etc. available due to potentially identifying information of expert, but they can make a high-level description of the task and the occupation it belongs to available.
- Figure 7(a) and (b): Why were OpenAI models singled out here? How does this analysis look like for the Claude or Gemini families? Overall, most analyses in the paper single out OpenAI models and don’t allow for an accurate understanding of frontier model capabilities. Given that the full tasks aren’t open-sourced and the analysis hence cannot be completed for other models by third parties, I kindly ask the authors to provide the more detailed analysis for non-OpenAI models, too, to allow for better understanding and comparability of frontier model performance. E.g., how would Figure 7(b) look like for Claude models (which was overall the best-performing model according to Fig. 6? Similarly, in Section 3.3, this analysis was only done for improving formatting. How much of a performance increase would models like Claude or Gemini have gotten if the model was prompted to strictly follow instructions? 
- The reported details for the human baseliners and recruited experts are insufficient. For example, there is largely no information about the recruitment process given. Also, the paper only states that experts were “well compensated” but given the impact of compensation of performance, this should be made transparent. See this paper for the level of detail required for human baseline reporting: https://arxiv.org/pdf/2506.13776. Relatedly, how were occupational experts (line 206) sourced? 
- It’s unclear to me what “their requests” mean in line 172. Who assigned them “their requests”? Were multiple experts from the same occupation tasked with creating tasks? If this wasn't the case, it’s unclear to me how content validity (i.e., coverage) was ensured for each occupation.
- It feels like the claim in line 287/288 that GPT-5 excelled in particular on accuracy is overstated. From Fig 8, the % of total examples of failure modes seems too close for accuracy to say that a model “particularly excelled” (within +/- 1% for Claude and GPT-5) and in addition, Grok 4 showed the same % as GPT-5 so I don’t understand why GPT-5 is being singled out here. Are these significant differences?
- The human uplift study design has not been fully explained. How were scenarios picked? Why were only OpenAI models tested if Claude Opus 4.1 was better overall? 
- It’s unclear if the authors had an IRB or applied for an IRB exemption, which would be necessary for the experimental setup with significant human involvement. Please clarify.
- Can you provide quantitative data for the clustering pipeline? I.e., a table where there is quantitative information per model on how often the rationale fell into one of the buckets listed in line 311 ff., how often the rationale was unclear, etc. Otherwise it’s hard to judge the true differences across models. It’s also unclear if these differences were statistically significant, please add this information.
- Line 335: Note that there is a bias if you use the same model to judge the same model (e.g., GPT-5 judge to judge GPT-5 output); it’s best practice to at least compare the judgement to other judge models, see this paper: https://arxiv.org/abs/2404.13076. I'd suggest to at least add this to the limitation section.

Nit picks (only affected presentation score):
- The authors talk at multiple points about the "gold set" but it's unclear how the tasks for the gold set were selected.
- There seems to be a reference error in line 075.
- Lines 155ff.: It’s unclear to me how the list of prior employers is relevant for describing the experts.
- The formatting in lines 216/217 seems off, shouldn’t this be a full line?
- The format looks again off in lines 260/261, these should be full lines. Did you change anything about the underlying template?
- Figure 8 requires a lot of zooming in to be readable.
- Line 327/328: this statement is too qualitative. How much did performance improve? I'd suggest to least add a reference to the corresponding figure here.
- The paper lacks a proper related work section. While the authors very briefly mention previous work in one line in the introduction, they don’t sufficiently explain how their work is novel/different.

### Questions
Questions:
- Can you explain the notion of “significantly limited” in 213?
- I don’t understand how the last sentence in line 290 squares with the rest of the paragraph. The previous sentence seems to say that only 47.6% of model deliverables were as good or better than the human deliverable so I don’t understand where the “just over half the tasks” statement comes from. Could you help me understand that, please?
- Why couldn’t cost estimates for Claude, Gemini, and Grok be obtained? API cost for estimates should be available for all three models, no?

### Soundness
2

### Presentation
2

### Contribution
2
