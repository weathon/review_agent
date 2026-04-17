# Learning to Summarize by Learning to Quiz: Adversarial Agentic Collaboration for Long Document Summarization

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Long document summarization remains a significant challenge for current large language models (LLMs), as existing approaches commonly struggle with information loss, factual inconsistencies, and coherence issues when processing excessively long documents. We propose SummQ, a novel adversarial multi-agent framework that addresses these limitations through collaborative intelligence between specialized agents operating in two complementary domains: summarization and quizzing. Our approach employs summary generators and reviewers that work collaboratively to create and evaluate comprehensive summaries, while quiz generators and reviewers create comprehension questions that serve as continuous quality checks for the summarization process. This adversarial dynamic, enhanced by an examinee agent that validates whether the generated summary contains the information needed to answer the quiz questions, enables iterative refinement through multifaceted feedback mechanisms. We evaluate SummQ on three widely used long document summarization benchmarks. Experimental results demonstrate that our framework significantly outperforms existing state-of-the-art methods across ROUGE and BERTScore metrics, as well as in LLM-as-a-Judge and human evaluations. Our comprehensive analyses reveal the effectiveness of the multi-agent collaboration dynamics, the influence of different agent configurations, and the impact of the quizzing mechanism. This work establishes a new approach for long document summarization that uses adversarial agentic collaboration to improve summarization quality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces SUMMQ, an adversarial multi-agent framework for long document summarization via multi-task summarization and quizzIng. Under the SUMMQ approach, summary generators and reviewers work collaboratively to summarize text and refine the summary respectively, and generators and reviewers create comprehension questions via which summary is evaluated.

The authors evaluate the approach on 3 long summarization benchmarks (MENSA, BookSum, GovReport) and compare it to 3 types of baselines: Supervised Fine-Tuning, Prompting, and alternative Multi-Agent approaches. The systems are evaluated against automatic metrics (Rouge, BERTScore, LLM-as-a-Judge) and human evaluation. Results demonstrate that SUMMQ strongly outperforms other techniques, including previous multi-agent methods, in most evaluation scenarios.

### Strengths
* the paper introduces a novel multi-agent framework for long document summarization in a multi-task setup of summarization itself and quizzing
* compared to a series of baselines including Supervised Fine-Tuning, plain Prompting and other Multi-Agent approaches, SUMMQ outperforms them in both automatic and human evaluation on three long document summarization benchmarks

### Weaknesses
* given that the framework is targeted at long document summarization, there is little to no discussion (apart from cost analysis in the Appendix) of how the proposed framework results in increased context usage, what are the limits, and how the technique can scale with context in mind (even longer documents, multi-document summarization, increasing the number of iterations / agents - what would be the effect on the context?)
* there seems to be little to no justification of what exactly in the technique targets long document summarization. And if nothing in particular, why not include other summarization datasets into comparison?
* automatic metrics like Rouge and BERTScore are considered outdated even for short documents. Their applicability specifically to long documents is highly questionable

### Questions
A multi-agent summ eval framework relevant for the related work section: https://arxiv.org/abs/2502.08514v1

### Soundness
3

### Presentation
4

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
This paper introduces SUMMQ, an adversarial multi-agent framework for long document summarization. The core idea is interesting: use quiz generation as a way to verify and improve summary quality. The system has four types of agent groups - Summary Generators, Quiz Generators, Summary Reviewers, and Quiz Reviewers - plus an Examinee agent. These agents collaborate iteratively, with the quiz questions serving as a check on whether the summary captures all important information. The authors test their approach on three benchmarks (MENSA, BookSum, GovReport) and show improvements over existing methods using both automatic metrics and human evaluation.

### Strengths
- The quiz-based verification mechanism is a reasonable idea for checking summary completeness. The results on BookSum show substantial improvements, which is noteworthy.
- The experimental coverage is decent - three benchmarks with multiple evaluation approaches including human judges. Adding the rephrased version (SUMMQCOMBOR) in human evaluation was sensible to address length bias.
- The ablations in Section 5 help understand component contributions. The quiz coverage evolution analysis showing shifts from beginning/end to middle sections is mildly interesting.
- Including all prompts in the appendix aids reproducibility.

### Weaknesses
- The computational cost is a real concern for practical deployment: SUMMQCOMBO costs $14.45 per document—about 80× simple prompting ($0.18). The paper would benefit from end-to-end inference-time reporting in Table 8 to contextualize these costs.
- Performance varies dramatically across datasets. On BookSum, R-1 improves from 23.98 (GPT-5) to 44.62 - nearly doubling performance. However, on GovReport, SUMMQcombo (52.79) significantly underperforms supervised baselines like U.FORMER (56.60) and SLED (57.50) as shown in Table 1. The paper doesn't explain what document characteristics lead to these differences, making it unclear when this expensive approach is warranted.
- The quizzing mechanism lacks quality validation. While Table 6 shows quizzing improves performance and Quiz Reviewers apply detailed criteria (Appendix G.4), we never see actual quiz examples or human assessment of whether questions are well-constructed and test meaningful content versus trivial details.

### Questions
- Q1) Could you provide concrete examples showing a document excerpt, the generated quiz questions, and the corresponding summary? This would help readers understand how the quizzing mechanism works in practice and assess the quality of generated questions.
- Q2) Have you had human experts evaluate the quality of generated quizzes - for example, assessing whether questions are well-constructed, have appropriate difficulty, and test key concepts rather than trivial details?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SUMMQ, an adversarial multi agent setup for long document summarization. The idea is to make a summarizer and a quizzer work together so that the quiz keeps checking whether the summary actually contains the information from the document. On MENSA, BookSum, and GovReport, SUMMQ, especially the multi agent version, outperforms strong LLM prompting baselines and prior multi agent summarizers

### Strengths
- The paper targets a real issue in long summarization, missing content and hallucination when the document is very long.

- On MENSA and BookSum, SUMMQ COMBO gets the top ROUGE and BERTScore numbers, better than GPT 4O and GPT 5 prompting and also better than earlier multi agent methods.

- The evaluations are performed over the automatic and human evaluations, which is good for summarization, since ROUGE often disagrees with human preference. The human study still prefers SUMMQ even after they shorten the summaries.

### Weaknesses
- Algorithm 1 stops when both summary and quiz feedback are empty. In real LLM interactions, reviewers almost always find something, so you rarely get empty feedback. The paper limits to 3 iterations, so in practice it is just a fixed number of rounds, not a true convergent loop.

- The paper combines two ideas that are already around: multi agent summarization with draft/aggregate/vote, and LLM as a judge or QA style verification.

- The paper shows that adding more agents keeps improving results, but with diminishing returns. This is not the scaling pattern you want, since every new agent is an extra LLM call. There is no attempt to share context, cache intermediate decisions or use cheaper models for some agents. So the system is not cost aware.

- The whole framework relies on carefully designed prompts for generator, reviewer, quiz writer, quiz reviewer and examinee. The main paper does not analyze which parts of the prompts are critical. This makes the method brittle in practice, because small prompt changes could break the debate or the voting.

- The paper do show that quiz coverage becomes more balanced over iterations, but they do not analyze question difficulty, answerability or hallucinated questions, which are all common problems with LLM generated quizzes.

### Questions
- What is the average number of model calls and tokens per document for SUMMQCOMBO on MENSA and BookSum, including quiz agents and reviewers.

- You generate 30 questions per quiz, 10 per type. Did you try fewer questions. How fast does performance drop if you ask only 5 or 10 questions.

- How sensitive is the system to reviewer aggressiveness. If reviewers flag too many contested issues, does the system fail to converge.

- Can you share inter annotator agreement and how you controlled for length, since SUMMQ often writes longer summaries.

- You show that SUMMQ works with different backbones. Did you try a cheaper generator with a stronger reviewer.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes SUMMQ, an adversarial multi-agent framework for long-document summarization. It introduces two complementary domains, summarization and quizzing, each involving generator and reviewer agents. A quiz-examinee loop acts as a continuous quality check to improve coverage, coherence, and factuality. The framework iteratively refines both summaries and quizzes. Experiments on MENSA, BookSum, and GovReport show strong results over several baselines, including GPT-4/5 and prior multi-agent methods.

This paper is well executed and produces strong results, but its methodological novelty is modest given the growing body of multi-agent summarization and self-evaluation work. The framework mainly integrates existing ideas in a new context. The contribution feels incremental rather than conceptually new.

### Strengths
1.	The idea of using quizzes as a factuality and coverage signal is creative and intuitive.
2.	Experiments are extensive and include human and LLM-as-a-judge evaluations.
3.	The paper is clearly written and supported by systematic ablations that examine agent configuration, iteration effects, and backbone models.
4.	The framework shows consistent gains across datasets and metrics.

### Weaknesses
1.	Limited novelty relative to prior multi-agent work. The generator–reviewer–debater structure is already common in recent multi-agent reasoning systems, and the quizzing mechanism mainly extends existing self-evaluation ideas rather than introducing a fundamentally new algorithmic principle.
2.	The framework appears expensive to run, requiring multiple high-capacity agents and several iterations, which may limit practical use.
3.	The paper lacks clear explanation of what is actually learned during the process and whether any parameter update or fine-tuning occurs.
4.	Factuality improvements are claimed but not evaluated with specialized metrics beyond ROUGE and BERTScore.
5.	The contribution of the examinee agent is not well isolated in the ablations.

Minors:
1.	Page 2: Remove the duplicate word in “long document document summarization.”
2.	Page 2: Change “systems has” to “systems have.”
3.	Page 4: Add a space in “setZ” → “set Z.”
4.	Page 4: Use plural “drafts” in “each of the individual drafts.”
5.	Page 7: Add “To” at the start of “To address length bias.”

### Questions
See my above weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
