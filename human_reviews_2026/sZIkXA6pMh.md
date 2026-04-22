# LLM-REVal: Can We Trust LLM Reviewers Yet?

- Avg Score: 4.50
- Decision: Reject
- Scores: 8, 4, 4, 2

## Abstract
The rapid advancement of large language models (LLMs) has inspired researchers to integrate them extensively into the academic workflow, potentially reshaping how research is practiced and reviewed. 
While previous studies highlight the potential of LLMs in supporting research and peer review, their dual roles in the academic workflow and the complex interplay between research and review bring new risks that remain largely underexplored. 
In this study, we focus on how the deep integration of LLMs into both peer-review and research processes may influence scholarly fairness, examining the potential risks of using LLMs as reviewers by simulation.
This simulation incorporates a research agent, which generates papers and revises, alongside a review agent, which assesses the submissions. 
Based on the simulation results, we conduct human annotations and identify pronounced misalignment between LLM-based reviews and human judgments:
(1) LLM reviewers systematically inflate scores for LLM-authored papers, assigning them markedly higher scores than human-authored ones;
(2) LLM reviewers persistently underrate human-authored papers with critical statements (e.g., risk, fairness), even after multiple revisions.
Our analysis reveals that these stem from two primary biases in LLM reviewers: a linguistic feature bias favoring LLM-generated writing styles, and an aversion toward critical statements.
These results highlight the risks and equity concerns posed to human authors and academic research if LLMs are deployed in the peer review cycle without adequate caution. 
On the other hand, revisions guided by LLM reviews yield quality gains in both LLM-based and human evaluations, illustrating the potential of the LLMs-as-reviewers for early-stage researchers and enhancing low-quality papers.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper examines whether LLMs can function as reliable reviewers in academic peer review. To explore this, the authors develop a multi-round simulation framework called LLM-REVal, which models the interaction between a Research Agent—responsible for generating both human-like and LLM-authored papers—and a Review Agent that evaluates those submissions. The simulation reproduces the full review process, including initial assessment, rebuttals, revisions, and resubmission cycles, with a focus on fairness and bias in LLM-based reviewing.

The results reveal that LLM reviewers systematically favor the writing style characteristic of LLM-generated papers, leading to inflated scores for LLM-authored work. In contrast, some human-written papers—particularly those that discuss risks, fairness, or critical perspectives on AI—consistently receive lower scores, even after multiple revisions.

### Strengths
- Novel Problem Formulation: Addresses a timely and underexplored issue—what happens when LLMs act as both researchers and reviewers, creating feedback loops in scientific workflows.
- Comprehensive Simulation Framework: The paper develops a realistic multi-agent system encompassing literature search, paper creation, feedback, rebuttal, revision, and meta-review. This end-to-end simulation is technically impressive.
- Clear Empirical Evidence of Bias: The study rigorously demonstrates two forms of bias in LLM reviewers:
  - Linguistic bias toward LLM-style writing.
  - Topic/framing bias against papers emphasizing risks or fairness.

### Weaknesses
No significant weaknesses from my sight. Some minor comments below:

- The section title font is slightly different from other papers.
- Potential Missing Citations
  - [A Sentiment Consolidation Framework for Meta-Review Generation](https://aclanthology.org/2024.acl-long.547/)
  - [ReviewScore: Misinformed Peer Review Detection with Large Language Models](https://arxiv.org/abs/2509.21679)
  - [Position Paper: How Should We Responsibly Adopt LLMs in the Peer Review Process?](https://openreview.net/forum?id=KZ3NspcpLN)
  - [Position: The Artificial Intelligence and Machine Learning Community Should Adopt a More Transparent and Regulated Peer Review Process](https://openreview.net/forum?id=gnyqRarPzW&noteId=1Y3P0jqL5z)
  * [Position: The AI Conference Peer Review Crisis Demands Author Feedback and Reviewer Rewards](https://openreview.net/forum?id=l8QemUZaIA)

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper simulates an academic workflow to see if LLMs can be trusted as peer reviewers. The authors built a system with an LLM agent that writes papers and another that reviews them, comparing the results for both LLM-authored and human-authored papers. The study finds that LLM reviewers are significantly biased: they systematically give higher scores to other LLM-generated papers and lower scores to human papers that discuss critical topics like "risk" or "fairness".

### Strengths
- The paper is generally well-written and easy to follow.
- The research identifies specific biases in LLM reviewers, notably a "linguistic feature bias" favoring LLM-generated text and an aversion toward critical statements.
- The findings from the simulation are contrasted with human evaluations, which reveal a clear misalignment in judgment; human reviewers, for instance, did not share the LLM reviewers' preference for LLM-authored papers.

### Weaknesses
- The comparison between LLM-generated and human-authored papers is somehow not rigorous. The authors extract keywords from real human-authored papers and use these keywords to guide the LLM in generating a new paper. However, (1) the LLM may likely generate a paper with a distinct idea; (2) even with a similar idea, the LLM-generated paper uses "predicted" the results while the results in human-authored papers are real, ... That is, there are many variables that may lead to different review scores, making the comparison unfair and the corresponding conclusions less convincing.
- The research agent's process is simplified, as it "predicts" experimental results rather than actually executing experiments, which may not capture the full complexity of manuscript quality. 
- The "Irreducible Rejection" finding is interesting, but it's not entirely clear why these specific human papers were persistently underrated, even after multiple revisions guided by the LLM's own feedback.

### Questions
n/a

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper looks at the risk of LLMs as per reviewers by simulating multi round paper generation and the review process. The paper constructs a Research Agent that generates papers and a Review Agent that evaluates submissions, comparing 100 human-authored ICLR papers against 100 LLM-generated papers on identical topics across multiple review-revision cycles. The aim to then see the misalignment between LLM reviewers and human reviewers. They obseve two main biases: LLMs have linguistic feature bias favoring LLM-generated writing styles and aversion toward critical discussions. Such a study is important to discuss the course of using LLMs in peer review or not.

### Strengths
- The paper is very timely and helpful in the discussions of allowing AI tools for paper writing or/and paper reviewing. 

- The authors validate their review agent first using real ICLR 2025 data (100 papers). The 73.7% acceptance prediction accuracy and significant correlation with human scores establish credibility and trust for downstream evaluations.  

- Human in the loop validation and testing.

- While the results that LLMs prefer their own similar generations has been established in the literature previously, nine-metric linguistic analysis is thorough for review purpose.

### Weaknesses
- LLM papers use "predicted results" rather than actual experimental execution. This could lead to unrealistic results or bumped up values whereas human papers might have realistic results where the proposed method doesnt always outperform.

- Building upon the prev one, my biggest concern is that this study has different confounders for human paper and LLM paper, making it difficult to find causal relationship in the results.

- A bit of circular evaluation. LLMs prefer their own outputs is already known - this study essentially re-discovers it in a new context making the novelty low, especially since the reviewer agent used is also been proposed already.

### Questions
- With human-LLM reviewer correlation at only r=0.50 how do you establish which evaluation is "correct"? Why frame disagreement as "LLM bias" rather than "human unreliability" or "both are noisy"?

- How do you account for the confounding bias between the two LLM and human papers(paper length, overclaiming novelty)?

- If we over-sampled rejected papers, this biases results toward lower-quality human papers (as per the dataset it has 50% acceptance rate?).

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
4

### Summary
This work studies how using LLMs for both research and review influences AI reviewing.
An AI research (agent) and an AI reviewer (agent) generate papers and review them in a loop.
The work finds that LLM reviews inflate the scores of LLM-generated papers 
and penalize human-written papers that are self-critical.

### Strengths
1. A key strength of this work is that it studies an AI researcher (agent) together with an AI reviewer (agent).
A builder-reviewer loop is often used in the emerging field of automated scientific discovery (ASD).

2. The work finds that LLMs tend to favor LLM-written papers over human-written papers, reward revisions, and penalize human self-criticism in human-written papers.

### Weaknesses
1. The key issue with this work is that the review process only includes papers and is missing a review of their data and code.
Without the AI reviewing the entire submission: data, code, and paper, the AI reviewer cannot distinguish between real and fabricated papers, which may hallucinate experiments, results, etc.
Without reviewing the entire submission, it is unclear if improvements are hallucinated research or real research.

2. The architecture figure 1 (on page 2) is AI-generated with gross spelling errors, “Guild by reviews”, “Reversion”.

3. The "research agent" may be simplified by using an agent such as Claude Code (with a flat fee without incurring any API token costs).
The "review agent" could be improved by using a strong model such as GPT-5 Pro.

4. This perspective on LLM reviewing of both human and AI researchers is missing the issue of detecting problems with automated scientific discovery (ASD) systems.
See for example:
@article{jiang2025badscientist,
  title={BadScientist: Can a Research Agent Write Convincing but Unsound Papers that Fool LLM Reviewers?},
  author={Jiang, Fengqing and Feng, Yichen and Li, Yuetai and Niu, Luyao and Alomair, Basel and Poovendran, Radha},
  journal={arXiv preprint arXiv:2510.18003},
  year={2025}
}

### Questions
Can the work be used to evaluate issues with automated scientific discovery (ASD) systems instead of just LLM reviewing?

### Soundness
2

### Presentation
2

### Contribution
2
