# ReviewerToo: Should AI Join The Program Committee? A Look At The Future of Peer Review

- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Peer review is the cornerstone of scientific publishing, yet it suffers from inconsistencies, reviewer subjectivity, and scalability challenges. 
We introduce **ReviewerToo**, a modular framework for studying and deploying AI-assisted peer review to complement human judgment with systematic and consistent assessments. ReviewerToo supports systematic experiments with specialized reviewer personas and structured evaluation criteria, and can be partially or fully integrated into real conference workflows. We validate ReviewerToo on a carefully curated dataset of 1,963 paper submissions from ICLR 2025, where our experiments with the gpt-oss-120b model achieves 79.3\% F1 for the task of categorizing a paper as accept/reject compared to 83.8\% for the average human reviewer. Additionally, ReviewerToo-generated reviews are rated as higher quality than the human average by an LLM judge, though still trailing the strongest expert contributions. Our analysis highlights domains where AI reviewers excel (e.g., fact-checking, literature coverage) and where they struggle (e.g., assessing methodological novelty and theoretical contributions), underscoring the continued need for human expertise. Based on these findings, we propose guidelines for integrating AI into peer-review pipelines, showing how AI can enhance consistency, coverage, and fairness while leaving complex evaluative judgments to domain experts. Our work provides a foundation for systematic, hybrid peer-review systems that scale with the growth of scientific publishing.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a review system named ReviewerToo. The authors conducted detailed testing and analysis on a carefully curated dataset, ICLR-2k. After careful examination, I think that the paper lacks sufficient novelty.

### Strengths
The study is based on a carefully constructed, large-scale dataset, ICLR-2k, which contains 1,963 real papers stratigraphically sampled from the ICLR 2025 conference, ensuring the authenticity and representativeness of the experiments. The paper clearly validates the effectiveness of various components within the framework (e.g., conference guidelines, literature retrieval, author rebuttal phases), demonstrating that structured and contextualized information is crucial for improving the quality of AI reviews.

### Weaknesses
The paper overlooks a significant body of existing work on AI review. Many of its implementation details can be found in prior research, which the authors fail to discuss. This makes me question whether the paper's novelty and contributions are suitable for a venue like ICLR. For instance, ReviewerToo's external literature retrieval and novelty verification have already been implemented in DeepReview[1]. Another core contribution, the "diverse panel of personas," has been realized in AgentReview[2]. Furthermore, regarding the dataset, DeepReview[1] already provides complete review data for ICLR 2024 and ICLR 2025.

The paper's evidence does not adequately support the claim that ReviewerToo surpasses human review systems. For instance, the paper does not clarify how human scores were calculated. If the data is simply the original reviewer scores from ICLR and not from a new, controlled experiment, then the conclusion in line 694, "humans are highly effective at holistic judgments of paper quality," is unreliable. This is because the initial reviewer scores are causally linked to the final acceptance decision; for example, papers with all-positive scores are generally accepted.

Furthermore, I disagree with the claim in line 805 that the system "produced reviews often judged more constructive than the human average." The evaluation relies solely on an LLM-as-a-judge, which I believe is prone to significant bias. For example, it may favor longer or better-formatted (e.g., Markdown) reviews. These are known issues in LLM-as-a-judge evaluations, and a model like gpt-oss-120b, likely trained with extensive RLHF, would be predisposed to generating text that appeals to another LLM judge.


---

[1] DeepReview: Improving LLM-based Paper Review with Human-like Deep Thinking Process 

[2] AgentReview: Exploring Peer Review Dynamics with LLM Agents

### Questions
How are the human scores calculated? Including F1, ELO, etc.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ReviewerToo, a modular framework for studying the use of AI-assisted peer reviews in AI conferences. To support this, the work introduces ICLR2K, a curated subset of the ICLR 2025 peer review dataset, where the paper primarily analyzes the predictive performance of acceptance decisions (2-way: accept/reject; 5-way: Oral, Spotlight, etc.). ReviewerToo consists of several agents (literature review agents, reviewer agents, meta-reviewer agents, and author agents) that coordinate to perform the review process and ultimately make decisions on papers. Notably, the reviewer agents adopt different personas (e.g., enthusiast, theorist) in an attempt to reflect the diverse perspectives of real-world reviewers. These reviewers utilize information from both the literature agent and the manuscript to make their decisions. Interestingly, aggregating the existing peer review protocols (e.g., including meta reviewer decisions) leads to improved performance.

### Strengths
- The paper proposes a modular framework that researchers working on AI-assisted peer review can experiment with, which I believe is a valuable contribution to the community.
- Many of the initial questions I had while reading the abstract were well addressed in the manuscript.
- The experiments with different personas and baselines were very interesting.
- Experiments and ablations were extensive.

### Weaknesses
- The proposed ICLR2K dataset consists only of ICLR 2025 review data. Why didn't the authors aggregate review data from different years or investigate whether the findings generalize across years?

- The main issue with this work is the lack of discussion and engagement with existing literature, as well as insufficient evidence to support some of the findings. For instance, the authors state that "Human reviewers exhibit low to moderate agreement with LLM reviewers (κ≈0.1–0.2), consistent with known levels of disagreement in real peer review" or claims like  "potentially sycophantic tendencies of LLMs." Where are the references? I understand that finding references for this type of work can be challenging, as much of the evidence is anecdotal, but the current paper includes only 8 references, which suggests that the authors have not invested sufficient effort in substantiating these claims with existing literature.

### Questions
**Suggestion**
- I believe the current topic of incorporating AI reviewers is controversial; as such, a more in-depth discussion of related works and supporting background regarding the use of such reviewers in AI conferences needs to be thoroughly addressed. The current work references only 8 papers, which is too few. There are numerous works that discuss such approaches, and this paper lacks sufficient engagement with the existing literature. I recommend the papers below and more extensive related works related to this manuscript.
-------
 - Is llm a reliable reviewer? a comprehensive evaluation of llm on automatic paper reviewing tasks 
- Some ethical issues in the review process of machine learning conferences
- Position: The AI Conference Peer Review Crisis Demands Author Feedback and Reviewer Rewards
-  Position: The Artificial Intelligence and Machine Learning Community Should Adopt a More Transparent and Regulated Peer Review Process
 - Reviewergpt? an exploratory study on using large language models for paper reviewing
-----
- The visualization in Figure 1 can be improved. Please increase the font size of the numbers above the bars and provide an illustrative caption for the figure. Additionally, I believe Figure 1 does not need to be this tall—consider condensing its height.
- For Section 4.2, consider restructuring the content to directly indicate which part of each table corresponds to the explanation. For instance, Table 1 presents results in the order of ReviewerTooAgent followed by Supervised Baseline. However, Section 4.2 begins the explanation with Supervised Baseline, then abruptly transitions to Table 2 (which should also include a direct reference), before returning to Table 1. This organization makes the section difficult to follow.
- I am uncertain whether the confusion matrices in Figure 3 warrant a full page. There may be opportunities for more insightful analysis or discussion of related work.
- I recommend adding a one-sentence description of the ELO metric in the main manuscript, as this metric may be unfamiliar to most readers.
- Please remove the note in Table 4.
- When such persona is fixed, wouldn’t it be possible for some authors to game the reviewing system?, discussion regarding the weakness of the paper would be needed, 

**Question**
- How did the authors select the top 1% of human reviewers in this paper's context?
- Wouldn't the ELO metric, which is based on LLM-as-a-judge, obviously favor reviews written by LLMs? LLMs naturally generate more detailed reviews than human reviewers, and as such, this does not seem to be an appropriate metric for understanding whether AI produces better reviews than humans. 
- Is it possible to analyze the distribution of personas among human reviewers based on the personas set by the authors? Based on this distribution, the reviewer agents could adopt a similar distribution.

### Soundness
1

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
Proposed a modular framework for deploying AI-assisted peer review.

### Strengths
Fine-grained analysis of model's prediction on different personas in prompt and methods.

### Weaknesses
[Goal of the work] I don't see any values of improving the acceptance predictions using AI, even though the AI system is more advanced with multi-agentic framework. Getting a higher prediction score means that current peer-review system should be replaced by your system or even better versions of it in the future? 
[Technical merit] What is the core technical merit of this work? There are tons of multi-agent systems repliciating human scientists, reviewers, or any other stage of research. Is this work proposing yet another agentic pipeline? If so, the proposed framework seems too simple, linear-chained framework which was done by many prior work and other startups.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
1
