# Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges

- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 5, 5, 3, 3

## Abstract
Offering a promising solution to the scalability challenges associated with human evaluation, the LLM-as-a-judge paradigm is rapidly gaining traction as an approach to evaluating large language models (LLMs). However, there are still many open questions about the strengths and weaknesses of this paradigm, and what potential biases it may hold. In this paper, we present a comprehensive study of the performance of various LLMs acting as judges, focusing on a clean scenario in which inter-human agreement is high. Investigating thirteen judge models of different model sizes and families, judging answers of nine different ‘examtaker models’ – both base and instruction-tuned – we find that only the best (and largest) models achieve reasonable alignment with humans. However, they are still quite far behind inter-human agreement and their assigned scores may still differ with up to 5 points from human-assigned scores. In terms of their ranking of the nine exam-taker models, instead, also smaller models and even the lexical metric contains may provide a reasonable signal. Through error analysis and other studies, we identify vulnerabilities in judge models, such as their sensitivity to prompt complexity and length, and a tendency toward leniency. The fact that even the best judges differ from humans in this comparatively simple setup suggest that caution may be wise when using judges in more complex setups. Lastly, our research rediscovers the importance of using alignment metrics beyond simple percent alignment, showing that judges with high percent agreement can still assign vastly different scores.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a study on the performance of LLM-as-a-judge expressed as agreement with human assessment. The LLMs that are judged are run in an exam-taker context, using TriviaQA dataset. The study shows that only the largest and more recent Lllama-3.1 models approach human judgement. The rest observe widely different scores, some as low as 60% agreement. Different prompting styles seem to make a difference, with single digit performance improvements for the performant models.

I think it would be nice to summarize all the takeaways in a section and best practices for doing assessment using LLM-as-a-judge.

### Strengths
* Thorough and timely study
* Several interesting experiments

### Weaknesses
* I would have liked to see more datasets; in fact, I would suggest reducing the number of exam-taker (e.g., I find the base models less interesting) and use different datasets

### Questions
Do you think the results generalize to different types of datasets/tasks?

What do you think it's an acceptable agreement percentage?

What are the takeways from this study? What are the best practices that can be adopted when using LLM-as-a-judge?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors examine models as judges on TriviaQA questions, comparing how often humans agree with their answers. While they find alignment is high, they can use Scott Pi metrics to distinguish the quality of the judges. They also find models are frequently lenient judges in a binary setting of “correct” vs “incorrect”.

### Strengths
* The paper is clear and nicely visualizes the relevant findings
* The authors explore a dozen models as judges
* The authors use manual annotation to carefully unpack the judge behavior, especially the observation about judge leniency

### Weaknesses
* As a reader I'm having difficulties understanding the overarching goals of the paper. Typically researchers use LLMs as a judge for longer form, more subjective questions, where answer coherence, style, and correctness are all part of the judgement. But for TriviaQA, the chosen dataset, the questions have clear, short answers with reference documents, meaning Exact Match is already a strong metric. Here, humans are simply reporting the binary value “correct” vs “incorrect” on the model answers, which seems to have little to do with “human alignment” and more to do with which model got the right answer? Could the authors provide more information on how they think these insights may or may not generalize to more more complex judging tasks, as well as discuss the limitations of their findings?
* The choice of TriviaQA is extremely relevant to the reported results. Could the authors justify this choice? And have they considered comparing their results on other types of datasets?
“well-aligned models tend to produce more false negatives than false positives.” This does not seem to be supported by Figure 3b, where the most correct model’s errors are mostly false positive? Could the authors please provide details to explain this—in case I am misunderstanding?
* Could the authors add more details discussing the extent to which their contributions are novel and provide the community with actionable? A discussion section would be helpful here.

### Questions
* Did the authors consider other datasets, or non-binary notions of answer quality?
* Did the authors consider evaluating the alignment across models to understand how they might be ensembled to mimic a better judge?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper evaluates the LLM-as-a-judge paradigm by comparing 13 judge models against human evaluations on TriviaQA, focusing on a scenario with high human agreement to isolate judge performance issues. 

The results show that while the largest models achieve reasonable human alignment, they still fall notably short of human-human agreement levels and exhibit various vulnerabilities.

### Strengths
* The explanation of why Scott's Pi should be used instead of Kappa in judging-the-judges scenarios is a significant contribution that will benefit future researchers.
* The comprehensive analysis across multiple dimensions (alignment metrics, ranking correlation, error analysis) provides valuable insights into the strengths and limitations of different judge models.
* The comparison between LLM judges and lexical judges (EM, Contains) offers a novel and important perspective. This insight becomes increasingly critical as NLP tasks grow more complex, helping inform efficient evaluation strategies.

### Weaknesses
* The evaluation relies solely on TriviaQA, making it difficult to deconfound the root cause: whether the best model's performance stems from better alignment, knowledge of TriviaQA content, or simply being favored by other LLMs. Other unusual findings may also be specific to TriviaQA: in Figure 1.a, EM's instability compared to Contains likely results from references providing multiple correct answers.
* The paper lacks sufficient content for an ICLR long paper. I suggest expanding the scope by:
    * Including more evaluation datasets covering other types of tasks, such as objective long answers (e.g., code writing), using LLM judges to rank exam takers, etc.
    * Moving Appendix B (issues with Kappa) to the main paper and adding more experiments and analysis. This lesser-known fact would make the paper more informative and impactful.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper offers a study of LLMs-as-judges. The authors investigate 13 models (2B to 70B), evaluating 9 different "exam-taker" models on the TriviaQA benchmark. They found 1) Only the largest models achieve reasonable alignment with humans, though still falling short of inter-human agreement, 2) Scott's π provides better discrimination between judges than percent agreement, and 3) Even models with lower alignment scores can effectively rank exam-taker models. Through detailed analysis, the paper uncovers several vulnerabilities in judge models, including sensitivity to prompt complexity and a tendency toward leniency.

### Strengths
- They focus on a specific scenario with high inter-human agreement which is an attempt to isolate the judge model behavior from task ambiguity. 
- Several dimensions are explored: 1) model sizes and families, 2) Multiple metrics, 3) Error analysis provided. 
- Insights such as ``smaller models can rank exam-takers as effectively as larger ones'', and the attempted explanation that "chat models may "unlearn" some knowledge during alignment"; 
- The work also provides some recommendations for practitioners using LLM as judges, e.g. using Scott's $\pi$ along with accuracy.

### Weaknesses
- The scope remains limited to TriviaQA. For "short, factual" answers, consider adding the "LLMBar" datasets, which have high human agreement rates > 90%. Sufficient examples can be used according to your dataset selection criteria [1]. Without the inclusion of additional datasets [1], it remains unclear how well the ranking ability would transfer. 
- The original claim (line 316-318) about judge performance being worse at identifying correct answers could be an artifact of including metrics that are overly strict about exact wording matches rather than semantic meaning. The finding does not appear to be surprising or novel. 
- Further analysis would be beneficial: show example outputs from each judge model and identify common errors; In Appendix I, where the authors justify the sample size, adding a power analysis would be ideal.

References:

[1] [Evaluating Large Language Models at Evaluating Instruction Following](https://arxiv.org/pdf/2310.07641)  
[2] [The NarrativeQA Reading Comprehension Challenge](https://arxiv.org/pdf/1712.07040)

### Questions
- The error analysis in Table 2 shows judges struggle with under-specified answers. Could you provide examples of or qualitatively explain the under-specified answers that fooled even the best judges?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This work provides an examination of LLM judges regarding their performance and vulnerabilities in a reference-based evaluation setting for QA tasks. Using human annotation as the gold standard, a series of judge models are evaluated. For evaluation metrics, the manuscript proposes using Scott’s $\pi$ instead of accuracy, highlighting it as a main finding. It also shows that while less capable judges perform poorly at the instance level, i.e., giving the same decision as the human annotators, they achieve higher correlation with humans at the system level, i.e., producing a ranking of evaluated models by aggregating the instance-level decisions. Further analyses are conducted on changes in recall and precision scores, sensitivity to prompts, robustness, and leniency bias.

### Strengths
1. The provided analysis regarding the LLM judges' sensitivity to prompts, error types, a lack of robustness, and the leniency bias are interesting and valuable to future studies.

2. The paper is well-written and the findings are clearly presented.

### Weaknesses
It appears that some of the main findings in this work are either not well-supported, may lack generalizability, or have been discussed in previous work.

1. Lack of generalizability: The task setting of the LLM judges selected in this work is reference-based evaluation of QA, which differs from the common application scenario where LLM judges evaluate various tasks without a gold reference (e.g., AlpacaEval, Arena Hard). Access to gold references makes the evaluation task significantly easier. Therefore, the findings in this work may not generalize well to more open-ended, general evaluation settings. While it is stated that this task setting was chosen to reduce human disagreement, there exists a related dataset, LLMBar [1], which can be used to perform a more general evaluation of LLM judges, achieving over 90% human agreement (evaluation accuracy).

2. The finding that automatic evaluation metrics (specifically LLM judges) have a higher correlation with human evaluations at the system level than at the instance level has already been identified and well-discussed in related work on evaluating automatic evaluation of natural language generation tasks [2][3][4][5]. For example, [5] shows that automatic metrics can achieve higher system-level correlation with humans when they evaluate more instances. Therefore, this finding itself is not a novel contribution.

3. The manuscript proposes using Scott’s $\pi$ instead of accuracy as the evaluation metric for LLM judges, claiming that it "appears to be better able to discriminate among various judge models." However, this claim is not well-supported, as the only evidence provided is that Scott’s $\pi$ yields scores with a wider numerical range than accuracy, which could potentially be achieved by trivially rescaling the range. Further examination is needed to verify this claim, such as by demonstrating that Scott’s $\pi$ offers greater statistical power, with tighter confidence intervals or a lower p-value in significance tests. Additionally, the notion of separability defined in Arena Hard [6] would be useful for comparing evaluation metrics.

4. The finding that the true negative rate (resulting in a lower recall score) falls quickly with less capable judges does not hold when the two lexical-similarity-based metrics, exact match (EM) and Contain, are excluded. In fact, all the small LLM-based judges achieve higher recall scores than precision scores. The observed low true negative rate/recall score of EM and Contain is expected, as these metrics rely on lexical similarity and are likely to mark an answer that is correct but lexically different from the reference answers as incorrect.


References

[1] Zeng, Zhiyuan "Evaluating large language models at evaluating instruction following." ICLR 2024

[2] The price of debiasing automatic metrics in natural language evalaution (Chaganty et al., ACL 2018)

[3] Re-evaluating Evaluation in Text Summarization (Bhandari et al., EMNLP 2020)

[4] A Statistical Analysis of Summarization Evaluation Metrics Using Resampling Methods (Deutsch et al., TACL 2021)

[5] Re-Examining System-Level Correlations of Automatic Summarization Evaluation Metrics (Deutsch et al., NAACL 2022)

[6] Li, Tianle, et al. "From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline." arXiv preprint arXiv:2406.11939 (2024).

### Questions
The description of the platform and recruitment process for human annotations is unclear. Who are the annotators (e.g., crowdworkers), and what are the recruitment criteria?

### Soundness
2

### Presentation
3

### Contribution
2
