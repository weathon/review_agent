# DiscoX: Benchmarking Discourse-Level Translation in Expert Domains

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
The evaluation of discourse-level translation in expert domains remains inadequate, despite its centrality to knowledge dissemination and cross-lingual scholarly communication. While these translations demand discourse-level coherence and strict terminological precision, current evaluation methods predominantly focus on segment-level accuracy and fluency. To address this limitation, we introduce DiscoX, a new benchmark for discourse-level and expert-level Chinese-English translation. It comprises 200 professionally-curated texts from 7 domains, with an average length exceeding 1700 tokens. To evaluate performance on DiscoX, we also develop Metric-S, a reference-free system that provides fine-grained automatic assessments across accuracy, fluency, and appropriateness. Metric-S demonstrates strong consistency with human judgments, significantly outperforming existing metrics. Our experiments reveal a remarkable performance gap: even the most advanced LLMs still trail human experts on these tasks. This finding validates the difficulty of DiscoX and underscores the challenges that remain in achieving professional-grade machine translation. The proposed benchmark and evaluation system provide a robust framework for more rigorous evaluation, facilitating future advancements in LLM-based translation. Our data and code are available at https://github.com/ByteDance-Seed/DiscoX.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents a benchmark and evaluation system, DiscoX, for discourse-level and expert-level translation tasks, particularly for Chinese-English translation. The system introduces Metric-S, a reference-free evaluation metric that uses large language models (LLMs) to score translations based on three key dimensions: accuracy, fluency, and appropriateness. The authors demonstrate that Metric-S aligns closely with human judgment, achieving significant improvements over traditional reference-based metrics. The paper also introduces a novel multi-stage task construction pipeline to create high-quality translation tasks for expert domains, addressing a significant gap in discourse-level machine translation evaluation.

### Strengths
1. The paper tackles the important issue of evaluating discourse-level translation, a topic often neglected in standard translation benchmarks that typically focus on segment-level accuracy. This approach is timely and relevant, especially given the increasing use of LLMs for complex, domain-specific translation tasks.
2. The authors propose a robust evaluation metric (Metric-S) that combines accuracy, fluency, and appropriateness in a reference-free manner. The empirical results show that the metric aligns closely with human evaluation, demonstrating the potential of LLMs for fine-grained translation assessments.
3. The multi-stage task creation pipeline for DiscoX ensures that the translation tasks reflect real-world professional demands. The quality control process, including peer reviews and filtering based on SOTA models, adds credibility to the dataset construction.
4. The proposed score-guided multi-round translation pipeline shows clear improvement in translation quality across various models, such as QWQ, GPT-5, and DeepSeek. This is a practical and useful contribution for real-world applications of machine translation.

### Weaknesses
1. While the Metric-S system is empirically sound, the paper does not provide a theoretical framework explaining why certain dimensions (accuracy, fluency, appropriateness) are weighted in the manner they are, or how these dimensions interact. The authors should consider adding an analysis of the sensitivity of these weights, or provide a more detailed justification for their choices.
2. Metric-S relies on a single evaluation model, and there is limited analysis of how different LLMs might perform as evaluators. To validate the robustness and independence of Metric-S, the authors should consider conducting experiments comparing different LLM models as judges and analyzing the consistency of their evaluations across models.
3. The current experiments focus on Chinese-English translation. To make the framework more generalizable, the authors should consider testing Metric-S on other language pairs (e.g., English-Spanish, English-German) or multilingual tasks to assess its robustness across languages and domains.
4. The paper mentions that a task is included in the DiscoX benchmark only if it passes a stringent threshold (at least 8 rubric failures from two SOTA models). However, the rationale behind this threshold is not explained. It would be helpful if the authors clarified the reasoning for this cutoff or conducted experiments showing how the difficulty threshold affects the overall task difficulty distribution.

### Questions
1. How robust is Metric-S across different domains, such as technical writing or legal texts, where semantic trade-offs may differ from general text translation?
2. Has Metric-S been tested on out-of-domain or noisy data (e.g., user-generated content) to check its stability and potential biases?
3. Would combining Metric-S with embedding-based metrics (e.g., BERTScore) improve the overall evaluation reliability, especially for more diverse domains?
4. How computationally intensive is the multi-round feedback pipeline, and is it scalable for use on large translation datasets?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a novel data set, DiscoX, designed for evaluating accuracy, fluency and appropriateness in a larger context, and also in specific domains.
A novel metric is proposed, too, which is better suitable to assess context-related aspects.

Several translations were evaluated on this test set, including open and closed LLMs, NMT systems and a human professional translation. It is shown that LLMs still do not reach the human performance on the given data set.

### Strengths
the data set is carefully constructed to be difficult for translation, which is a valuable contribution to the community

three different quality criteria are covered (adequacy, fluency, appropriateness) 

different domains are included

### Weaknesses
some details are not clearly explained (see "Questions" for details

for example: what are "tokens", how many Chinese and how many English source texts are collected, why is the main domain distinction between academic and non-academic texts, what are "thinking" and "non-thinking" models

### Questions
Related work should be placed after introduction, not at the end.


043: the term "expert-level" is not fully clear -- it seems that it refers to very specific domains where expert knowledge is needed in addition to language knowledge?

051: translation between Chinese and English: which direction? 

052: what are "tokens" in the given context? Chinese characters, or English words, or English sub-word units, or Chinese units after some segmentation? 

102: what does "domain-intensive" mean? a difficult domain? 


135: are discourse-level and expert-level parts separated, or there is some overlap? 

145: what are exactly "vertical domain experts"? 

157: 665 tasks -- what does "task" mean in the given context? a coherent text to be translated? 

192, Table 2: again, what is a "token" exactly?
Also, does Table 2 refer to English or to Chinese?
Furthremore, what are "domain specific scenarios"?

Also, why the main distinction between domains is "academic" vs "non-academic"?

Section 3: Why is an LLM used for calculating the score? Once the errors are identified, the score can be easily calculated using a simple script/code.

Table 4: the results on WMT24 test set is missing

Section 5.3: what are the thinking and non-thinking models? 



overall: spelling should be checked/revised, some spaces are missing between words and punctuations (commas, brackets, etc)

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
- This paper introduces DiscoX, the first benchmark designed to evaluate discourse-level and expert-level Chinese–English translation. The benchmark comprises 200 cases drawn from seven domains, covering both academic and non-academic contexts.
- In addition, the authors propose Metric-S, a reference-free, multi-agent LLM-based evaluation framework that judges translations along accuracy, fluency, and appropriateness dimensions. With this score, extensive experiments show that even leading LLMs such as GPT-5-high still trail human experts.

### Strengths
1. Expert-Curated Dataset: DiscoX was constructed by 133 qualified professionals, including 115 vertical domain experts (4–10 years of field experience, many Master’s or PhD holders from top-tier Chinese universities) and 18 linguistic specialists (Master’s level in Translation and Interpreting (MTI) graduates with professional translation experience). 
2. Extensive Domain Coverage: The benchmark spans seven domains covering both academic and non-academic.
3. DiscoX was evaluated on a wide spectrum of models, including 7 open-source and 11 closed-source LLMs.

### Weaknesses
Although the authors emphasize the benchmark's careful construction (claiming over 1,300 person-hours of expert effort), the presentation of DiscoX in the paper remains vague, incomplete, and insufficiently grounded in established machine translation literature. Several weaknesses are evident:

1. Unclear Problem Definition and Weak Motivation:
The Introduction fails to clearly articulate why discourse-level and expert-domain benchmarks are needed. While the authors mention that existing benchmarks "fail to assess whether models can sustain discourse-level coherence, handle domain-intensive terminology, or meet expert stylistic standards," these claims are not substantiated with concrete examples or empirical evidence. Only a few representative MT benchmarks (e.g., WMT, FLORES, RedTrans) are briefly cited, without a systematic survey of prior discourse-level or document-level evaluation studies. 


2. Lack of Theoretical Definition of "Discourse-Level Translation":
The paper repeatedly uses the term "discourse-level translation" but never provides a precise or operational definition. There is little to no discussion linking this notion to existing research in discourse-level NLP or translation studies, such as works on coreference, discourse connectives, or coherence modeling. This weakens the conceptual grounding of the benchmark. Furthermore, Figure 1, supposed to explain the concept, appears incomplete and uninformative, offering no clear illustration of the discourse-level concept or the nature of evaluation differences from "problematic" segment-level translation.

3. Superficial Dataset Construction Description:
Although the data collection process involves "expert-authored rubrics," the paper does not adequately describe how these rubrics were designed, validated, or applied. There is no verification of rubric quality or consistency across experts. Likewise, no inter-annotator agreement or reliability measure is reported, despite the benchmark's heavy reliance on human expertise for quality control.

4. Evaluation Metric Criteria Lack Scholarly Basis:
The Metric-S framework assesses accuracy, fluency, and appropriateness, but these categories are introduced without theoretical justification or reference to prior translation evaluation literature (e.g., DA, MQM, COMET, BLEURT, or human adequacy–fluency frameworks). As a result, the metric design appears ad hoc, lacking grounding in established MT evaluation standards.

(minor) At table 4, what is the point of including chrF row?

### Questions
1. What kinds of linguistic or contextual features are meant to define this level of translation? How are these aspects represented or evaluated in the DiscoX benchmark? Could you provide an example?

2. What mechanisms in Metric-S (e.g., multi-agent judging, rubric-based scoring, error deduplication) directly enable it to capture document-level phenomena more effectively than previous metrics?

3. Could the authors provide empirical examples or case analyses demonstrating when Metric-S identifies discourse-level errors that traditional metrics overlook?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors present DiscoX, a new Chinese-English benchmark composed of 200 long-form texts (average 1700+ tokens) from seven professional and academic domains. The construction involved a multi-stage process with 133 domain and linguistic experts. To evaluate translations on this challenging benchmark, the paper also introduces Metric-S, a reference-free, multi-agent LLM-based evaluation system. Metric-S assesses translations along three dimensions: Accuracy, Fluency, and Appropriateness, and shows a high correlation with human judgments (70.3% consistency). The experimental results on 20 different systems reveal a significant performance gap between the most advanced LLMs and human expert translators, demonstrating the difficulty of the benchmark and highlighting key areas for future MT research.

### Strengths
- The shift of focus from sentence-level to discourse-level translation, especially in expert domains, is a critical research direction. Current benchmarks are inadequate for this, and this paper makes a convincing case for the need of a new evaluation paradigm.

- The construction of the DiscoX benchmark is commendable. The multi-stage process involving a large number of domain experts and linguistic specialists ensures high quality and relevance of the test data. The inclusion of expert-authored rubrics is a particularly strong feature that allows for fine-grained, domain-specific evaluation beyond generic quality aspects.

- The development of Metric-S is a good contribution. A reference-free, multi-dimensional evaluation system is suitable for long-form texts where a single reference is insufficient. The three dimensions of Accuracy, Fluency, and Appropriateness are well-defined, and the error de-duplication mechanism is a thoughtful detail that improves the metric's fairness and interpretability. The empirical evaluation is also very extensive, covering many recent powerful LLMs. The analysis provides interesting findings that go beyond a simple leaderboard, such as the performance asymmetry between translation directions (zh→en vs. en→zh), between domains (academic vs. non-academic), and between 'thinking' and 'non-thinking' model versions. These insights are valuable for the community.

### Weaknesses
- The reliance on a single LLM (Gemini-2.5-Pro) as the judge is a concern. While Appendix G.3 tests for self-preference bias, the general stability of the "LLM-as-a-judge" paradigm is a known challenge. The paper's own results in Table 9 show that the choice of judge model can significantly alter the rankings (e.g., o3-high ranking itself first). This point is critical and deserves more discussion in the main paper. How can we be confident that the high consistency of Metric-S (70.3%) is not specific to this particular judge model? The paper would be much stronger if it discusses the sensitivity to the judge model more deeply.

- The scale of the benchmark, at 200 texts, is relatively modest, even if the texts are long. This could limit the statistical power of the conclusions, especially when results are broken down into sub-domains with very few texts (e.g., only 14 for "Literature and Arts"). I suggest the authors acknowledge this limitation and perhaps discuss plans for future expansion. Furthermore, the benchmark is limited to Chinese-English. The claims about discourse-level translation are broad, and it would be beneficial to discuss the potential challenges in extending this framework to other language pairs.

- The analysis in Section 5.3 is interesting, but the conclusion that "thinking-enhanced models consistently underperform" might be too strong. The underperformance is attributed to over-summarization or adding extraneous content. However, this could be an artifact of prompt-engineering. The system prompt used (Appendix B.1) is very generic. It is possible that 'thinking' models require more specific instructions to constrain their behavior to pure translation. Did the authors experiment with different prompts for these models? This would be a valuable ablation study to understand if the issue is a fundamental flaw or a behavioral artifact that can be controlled.

### Questions
Some crucial methodological details are located in the appendix, such as the definition of severity levels (Appendix D) and the correlation framework (Appendix F). Why not place them in the main body of the paper?

### Soundness
3

### Presentation
4

### Contribution
3
