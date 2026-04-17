# SurveyBench: How Well Can LLM(-Agents) Write Academic Surveys?

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Academic survey writing, which distills vast literature into a coherent and insightful narrative, remains a labor-intensive and intellectually demanding task. While recent approaches, such as general DeepResearch agents and survey-specialized methods, can generate surveys automatically (a.k.a. LLM4Survey), their outputs often fall short of human standards and there lacks a rigorous, reader-aligned benchmark for thoroughly revealing their deficiencies. To fill the gap, we propose a fine-grained, quiz-driven evaluation framework SurveyBench, featuring (1) typical survey topics source from recent 11,343 arXiv papers and corresponding 4,947 high-quality surveys; (2) a multifaceted metric hierarchy that assesses the outline quality (e.g., coverage breadth, logical coherence), content quality (e.g., synthesis granularity, clarity of insights), and non-textual richness; and (3) a dual-mode evaluation protocol that includes content-based and quiz-based answerability tests, explicitly aligned with readers’ informational needs. Results show SurveyBench effectively challenges existing LLM4Survey approaches (e.g., on average 21% lower than human in content-based evaluation).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
**Motivation**: The paper argues that automatic survey writing tools need a reader-aligned, fine-grained benchmark. Existing systems write fluent text but often miss coverage, depth, and usable takeaways. The goal is to evaluate whether LLM agents can produce surveys that people can actually learn from.

**Approach**: The authors build SurveyBench. It selects representative topics from 11,343 recent arXiv papers and pairs them with 4,947 high-quality human surveys. It proposes a hierarchical metric set for outline quality and content quality, plus a richness measure for non-text elements. It adds a dual protocol: content-based judging with and without human references, and quiz-based answerability using RAG and rubric-guided LLM judging. Fairness rules instruct generators not to consult existing surveys.

**Key results**: Across four systems, content-based scores approach human quality on surface metrics, yet quiz-based scores remain low, especially on topic-specific quizzes. LLM×MAPREDUCE-V2 leads outline quality, while OpenAI-DeepResearch leads several content dimensions. Human surveys remain much richer in figures and tables despite one method producing many tables by template. Results are stronger on older topics than new ones.

### Strengths
**Task importance**: The benchmark and its survey-writing task directly target the ability to produce reliable scientific surveys. This capability is central for AI4Science, where researchers need concise, trustworthy syntheses to guide experiments, interpret literature, and plan new work.

**Benchmark**: The protocol evaluates both structure and content and anchors scores to answerability, so higher scores reflect surveys that actually teach readers something usable. Careful curation of topics and references, plus validated quizzes, raises difficulty in a controlled way and reduces noise. Broad coverage across subfields reveals strengths and weaknesses that generic writing would miss.

**Evaluation Results**: The findings complement the careful curation and task setup by showing that the protocol reliably separates clean outlines from shallow content, that answerability scores track real comprehension, and that human surveys still lead, which signals clear headroom for improvement. The multi dimensional metrics provide actionable diagnostics for developers.

### Weaknesses
**LLM-as-judge validity**: I guess my major concern with this benchmark is in heavy reliance on LLM judges risking bias and instability across seeds or model versions. The paper misses reporting inter-judge agreement, calibration, and confidence intervals for win-rates and content scores. Since the paper heavily uses the LLM-as-judge framework for the evaluation, there should be thorough supporting analysis validating the robustness of LLM evaluation framework.

**Richness metric**: The richness formula rewards more figures or tables per character length. One method can inflate table counts via templating, yet still lags humans in overall richness due to very long outputs. The metric may invite superficial additions instead of evidence-grounded visuals. The authors should discuss any limitations or possible solutions to resolve such issues during evaluation.

**Model leakage**: Instructing agents not to consult existing surveys is hard to enforce, and many human surveys used as references are likely in pretraining corpora. The authors must conduct more analysis (e.g., checking citations or overlap detection) to verify whether the leakage can occur during evaluation.

**Presentation**: Paper’s presentation is generally well written. Though Some tables and figures are crowded (e.g., Figure 5 would benefit from larger text and clearer axis or legend labeling).

### Questions
- How are the LLM judges calibrated across runs. Can you report inter-rater reliability and confidence intervals for both content and quiz scores?
- Beyond instructions, what mechanisms ensure agents do not consult existing surveys? Do you check generated citations for survey overlap or run automated overlap detectors?
- How do agents generate figures? Is it fair to evaluate the quality of the generated figures compared to the human gold labels?

### Soundness
3

### Presentation
3

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
This paper focuses on the evaluation of the automatic survey generation. The authors point out that existing evaluation metrics cannot capture the core value of a high-quality survey, such as logical structure, content depth, key insights, and practical utility for readers and propose an evaluation system to comprehensively evaluate the quality of a survey from content, outline. and richness. Experiments demonstrate that SurveyBench can reveal the gaps between AI-generated surveys and surveys by human experts.

### Strengths
The motivation of this article is meaningful; a better evaluation of AI-generated surveys could help improve the quality of auto-survey-gen and further improve the efficiency of human research.

### Weaknesses
### **Serious Overlap with Prior Work**

**The core contribution of this paper, named SurveyBench, is highly similar to the benchmark with the same name proposed in a previously published paper, SurveyForge [A], which has been published on ACL 2025, from the following perspectives:**

- Core evaluation dimensions overlap: Both go beyond simple text fluency assessment and consider key elements of academic reviews: 
  1. Outline Quality: Both evaluate whether the structure of the review is reasonable and the logic is coherent. 
  2. Content Quality: Both evaluate whether the content is comprehensive, accurate, and profound. 
  3. Reference Quality (richness in this article): Both emphasize the importance of citing key, high-impact literature within the field. 

- Survey topic: SurveyBench includes 20 topics, of which 10 are completely identical to those in the published SurveyForge. More seriously, for these 10 overlapping topics, the "human-written survey references" selected as the gold standard in this paper are also exactly the same as those used in the SurveyForge paper.

[A] Yan X, Feng S, Yuan J, et al. Surveyforge: On the outline heuristics, memory-driven generation, and multi-dimensional evaluation for automated survey writing[J]. arXiv preprint arXiv:2503.04629, 2025.

The authors need to clarify the originality and workload of the proposed surveybench. If the originality and workload are sufficient, I will reconsider the score.

### Questions
Please refer to the weakness.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SurveyBench, a benchmark for evaluating LLM-generated academic surveys through fine-grained, quiz-driven assessments. The framework integrates a curated dataset of 20 CS topics, dual evaluation modes (content-based and quiz-based), and hierarchical quality metrics. Experiments on existing systems like DeepResearch and AutoSurvey reveal that, while LLM-generated surveys are structurally coherent, they substantially lag behind human-written ones in content richness and quiz performance. Overall, the work addresses a timely gap in automatic survey evaluation and provides a useful foundation for improving LLM4Survey research.

### Strengths
1. This work proposes a dual-mode protocol: content-based evaluation (with human surveys as gold standards) and quiz-driven evaluation (hierarchical general quizzes + RAG-based topic-specific quizzes). This directly targets readers’ informational needs (e.g., technical details, cross-concept reasoning) and fills the gap of “reader-centric assessment” in existing benchmarks 
2. The benchmark is curated rigorously: 20 representative CS topics from 11,343 recent arXiv papers and 4,947 high-quality human surveys, filtered via embedding clustering and citation impact. It also builds a hierarchical metric system (outline/content quality, non-textual richness) with clear criteria and bias mitigation (e.g., prohibiting LLM surveys from referencing human ones), ensuring reproducibility 
3. Practical in-depth experimental insights It evaluates 4 mainstream LLM4Survey methods, with fine-grained analysis and quantitative gap characterization. These findings identify key LLM weaknesses (insufficient detail, weak associative reasoning) and guide future optimization.

### Weaknesses
1. The content-based evaluation heavily relies on the LLM-as-judge approach for scoring outline and content quality (e.g., Coverage, Depth, Focus, Fluency, etc.), which exhibits limitations in discrimination and stability
2. The benchmark's final selection of a limited number of topics presents a significant scale limitation that hinders the comprehensive evaluation of LLM-Agents' generalizability and exposes a risk of selection bias.

### Questions
1.	Given that SurveyBench in the paper only covers 20 typical topics in the computer science field, will the narrow coverage of topics make it easy for models to game the system on these topics through targeted training? 
2.	Although the paper conducts experiments on topic recency, it remains unclear whether distinct topic types (rather than merely the recency difference) would lead to significant variations in the performance of LLMs when generating survey reports?

### Soundness
2

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
This paper proposes a fine-grained evaluation framework named SurveyBench, designed to systematically assess the capability of large language models (and their agents) in automatically generating academic survey papers. The article points out that there is a significant quality gap between existing automatically generated surveys and those written by humans, and there is a lack of a rigorous, reader-needs-aligned benchmark to reveal these shortcomings.

### Strengths
1. The research background of this paper is the current lack of effective and rigorous benchmarks for evaluating the performance of LLMs in automatically generating academic surveys, which indeed represents a key area of focus in current research.
2. The design of SurveyBench is structured and systematic. It features a dual evaluation mode—"content-based" and "quiz-based." The latter ingeniously tests the depth and informational effectiveness of surveys by simulating readers' genuine knowledge-seeking needs. Meanwhile, its multi-dimensional evaluation framework (covering structure, content, and richness) comprehensively encompasses the key aspects of high-quality surveys.
3. The article utilizes SurveyBench to conduct experiments on multiple baselines, revealing the performance gap between automatically generated surveys and human-written ones. Additionally, it includes case studies and provides error analysis.

### Weaknesses
1. I think the most significant issue with this paper is the lack of innovation. Although SurveyBench incorporates different evaluation logics, each individual evaluation component lacks novelty. For instance, the "quiz-based evaluation" fundamentally relies on "LLM-as-a-Judge" and RAG (Retrieval-Augmented Generation) technologies, which are already being extensively explored and applied in the evaluation of long-text generation tasks. Similarly, the design of its evaluation metrics—such as coverage, depth, and coherence—follows conventional approaches commonly seen in this research field.
2. The paper lacks sufficient citations to substantiate its claims. With only 11 references cited in total—mostly documenting the origins of models, metrics, and survey frameworks—the arguments presented in the introduction remain inadequately supported and thus less convincing. For instance, the authors assert that "(3) existing LLM-as-judge evaluation struggles to capture the reader’s perspective or to probe whether a survey genuinely informs (e.g., technical depth) and inspires (e.g., forward-looking insights)," yet fail to provide references to justify this criticism.
3. The topics illustrated in the article appear to be exclusively from the field of computer science, which limits the generalizability of SurveyBench and the resulting conclusions. There is a lack of case studies analyzing surveys from other specialized domains. It remains unknown how well LLMs perform in generating literature reviews for disciplines with different document structures, writing conventions, and evaluation criteria, as well as how effective SurveyBench would be in such contexts.
4. The guidance for future work is somewhat underdeveloped. Although the article identifies shortcomings of current models (e.g., lack of detail, weak reasoning), it offers limited insight into how to address these challenges. The conclusion primarily summarizes findings but falls short of outlining a forward-looking research roadmap. Key questions remain unexplored: Should future efforts focus on improving retrieval quality, enhancing the reasoning modules of models, or designing novel generative architectures?

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2
