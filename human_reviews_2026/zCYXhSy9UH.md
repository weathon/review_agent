# Kaleidoscope: In-language Exams for Massively  Multilingual Vision Evaluation

- Decision: Accept (Poster)
- Scores: 8, 6, 6

## Abstract
The evaluation of vision-language models (VLMs) has mainly relied on English-language benchmarks, leaving significant gaps in both multilingual and multicultural coverage. While multilingual benchmarks have expanded, both in size and language, many rely on translations of English datasets, failing to capture cultural nuances. In this work, we propose Kaleidoscope, as the most comprehensive exam benchmark to date for the multilingual evaluation of vision-language models. Kaleidoscope is a large-scale, in-language multimodal benchmark designed to evaluate VLMs across diverse languages and visual inputs. Kaleidoscope covers 18 languages and 14 different subjects, amounting to a total of 20,911 multiple-choice questions. Built through an open science collaboration with a diverse group of researchers worldwide, Kaleidoscope ensures linguistic and cultural authenticity. We evaluate top-performing multilingual vision-language models and find that they perform poorly on low-resource languages and in complex multimodal scenarios. Our results highlight the need for progress on culturally inclusive multimodal evaluation frameworks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents KALEIDOSCOPE, a large-scale exam-style benchmark to evaluate vision-language models across diverse languages and visual inputs. The proposed benchmark contains 18 languages, and 14 subjects with 55% requiring image understanding. The authors collect real exams via a global open-science effort. They evaluate closed (such as GPT-4o, and Claude 3.5 Sonnet) and open models (such as Qwen2.5-VL, Aya-Vision and others). After evaluation the authors describe different findings, among them, (i) all evaluated models perform substantially better on text-only questions than in multimodal ones. (ii) The type of visual data contained in exams affect performance depending if the question contains tables, diagrams, photos, etc. (iii) Performance depends on the domain of the questions. (iV) And finally the well known problem of crosslingual disparities in high vs low resource languages.

### Strengths
Paper Strengths:
1. The authors tackle the gap of evaluation at the intersection of multilingual and multimodal tasks. Kaleidoscope proposes in language exam questions blending image and text modalities, covering 18 languages and 8 image categories (such as diagrams, graphs, tables, etc). Compared to other benchmarks on the field, Kaleidoscope is more linguistically diverse and highly focused on multimodal evaluation.
2. The paper provides extensive evaluations by language, subject and image type.
3. Overall, the paper shows interesting insights and evaluations of open vs close models on multimodal QA , such as multimodal vs text only across models, resource gaps, scaling trends and domain disparities. These insights can be valuable for the community and future work in VLMs training.

### Weaknesses
Paper Weaknesses:
1. Table 1 compares closed models that use CoT prompts, while smaller models use a direct answer prompt due to CoT instability, however this is not an apples-vs-apples comparison, it would be interesting to see the performances of closed models using the direct answer generation approach. Now Qwen2.5-VL-72B is already close in performance compared to GPT-4o, with this direct comparison maybe Qwen could surpass GPT. 
2. I think the paper needs to show some statistics related to the difficulty level of the questions, this will give readers an overall idea of the benchmark. For example Table 4, could have how many grad, undergrad questions each language contains and in this way there is some information (maybe not really precise) but gives an overall idea of how difficult the questions are.
3. Personally, for me it was not really clear when the authors describe that Kaleidoscope allows cultural evaluation by combining regionally sourced multimodal questions. The proposed benchmark contains different categories, so I expect in fields like STEM to contain common universal topics, and in others such as Arts & Humanities and Social Sciences to contain some cultural related topics, but the question is was this quantified in some way? like how many of those cultural related questions are containing information from their specific country-language pair?, I think that exam-questions could be more global in some situations and those will not be related to the culture of each specific country-language pair.

### Questions
Please refer to weaknesses for my questions and doubts. Overall I think the paper brings good contributions and really interesting insights for current open and close vision-language models that can be useful for the community. Now I'm accepting the paper, but I look forward to see the author's responses and clarifications, and sorry if I misunderstood something, after rebuttal I will revise my decision.

### Soundness
4

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
Kaleidoscope is a new large-scale benchmark designed to evaluate multimodal + multilingual reasoning in vision-language models (VLMs), claiming to be the largest benchmark of its sort. It contains around 21k multiple choice questions (MCQ) across 18 languages and 14 subjects, with about half including images. The dataset is built from real-world, in-language (that is, not machine translated) educational exams contributed by native speakers from more than 20 countries. The authors evaluate both commercial API and open-source models under standardized protocols. Their analyses show large modality gaps (text-only vs. image+text), STEM-specific difficulty (compared to humanities subjects), and cross-lingual disparities (Latin script advantage over non-Latin). Despite some limitations (language balance, difficulty calibration, and reliance on MCQA format), Kaleidoscope aims to serve as a diagnostic benchmark to reveal weaknesses in a niche but increasingly important task where both multimodal and multilingual reasoning are required.

### Strengths
- Most importantly, the core contribution is the dataset itself. It targets a niche area that is difficult to produce synthetically (e.g., LLM-generated).

- The scope of this benchmark (# languages, # subjects, and a substantial proportion of image-based questions) is impressive. Having personally experienced very large, multi-national collaborations, I am genuinely impressed by the coordination and collective effort that made this possible.

- Although I wish the paper provided more details about the data collection and quality control processes (including license verification), my general impression is that the authors have made a serious effort to standardize collection and validation across languages and annotator teams.

- Each item includes detailed metadata (17 fields including country, subject, image type, and image importance), enabling detailed diagnostic analyses. This metadata can itself serve as a valuable resource for future research.

- The paper provides interesting empirical insights that can guide the development of future model architectures or training strategies: the modality gap, STEM vs. humanities performance disparity, and Latin vs. non-Latin script biases.

### Weaknesses
- Each language has independent questions rather than parallel translations, so performance differences reflect a mixture of linguistic and content variation _(I note that the authors were fair to list this point in the limitations section)_. This complicates direct interpretation of multilingual gaps. It would be helpful if the authors could provide some indication or proxy of relative difficulty across languages.

- Related to the above comment, I was curious to see a translation baseline. The authors deliberately avoid machine-translated versions, but a translated-to-English baseline would, to some extent, help isolate the effect of language from content or domain differences.

- I also wish the paper included cross-benchmark context. Table 2 reports results only on Kaleidoscope, without comparisons to other multimodal or multilingual benchmarks (e.g., MMMU, SEED-Bench, M3Exam). Without this, it is difficult to assess the dataset's relative difficulty and unique diagnostic value.

- Intended-use guidance (e.g., do’s and don’ts) would have been nice. Benchmarks are often misused or overinterpreted; an explicit section clarifying what Kaleidoscope can and cannot measure would help practitioners use it responsibly.

### Questions
See the above weakness sections. Besides, I have a couple of questions / suggestions.

- Given the benchmark’s collaborative nature, do you plan to release a contribution protocol or submission guideline so external contributors can add new languages or exams in future versions? I think this could turn Kaleidoscope into a true community-science platform and make it a living benchmark.

- The appendix is lengthy and dense. I'd like to suggest that the authors add an overview or roadmap at the beginning of the appendices. This would increase readability.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the problem that VLM evaluation is heavily dominated by English-language and Western-centric benchmarks. To overcome the limitations of simply translating existing datasets, the authors introduce KALEIDOSCOPE, a new benchmark composed of over 20,000 "in-language" multiple-choice questions from real-world exams. The benchmark spans 18 languages and 14 different subjects, with a significant portion (55%) of the questions requiring multimodal reasoning (image and text). The dataset was constructed through a large-scale open science collaboration with contributors from around the world, ensuring linguistic and cultural authenticity. The authors evaluate a wide range of state-of-the-art VLMs on KALEIDOSCOPE, including both large closed-source models and smaller open-weight models. Their findings reveal substantial performance gaps: all models struggle more with multimodal questions than text-only ones, performance is significantly weaker on low-resource languages and questions from STEM subjects, and models exhibit a clear bias towards Latin scripts.

### Strengths
- The motivation for this work is very clear and addresses a critical gap in the field. As vision-language models become more widespread, it is essential to evaluate them beyond English. The paper makes a strong case against the "translate-test" paradigm and for the necessity of "in-language" data that preserves cultural and educational context.

- The design and scale of the KALEIDOSCOPE benchmark is good. It is a comprehensive resource, covering 18 languages and a wide range of academic subjects. The use of a multiple-choice question format provides a structured and scalable evaluation framework. The inclusion of rich metadata for each question, such as image type and educational level, is also a very valuable feature that enables fine-grained analysis.

- The collaborative, open-science approach to data collection is particularly commendable. By involving a diverse group of researchers worldwide, the authors have created a more authentic and representative dataset than would be possible with automated translation. This methodology directly confronts the known biases in dataset creation and should be seen as a model for future work in this area.

- The paper's analysis is cmprehensive and provides several insightful findings that go beyond a simple ranking of models. The clear performance gap between text-only and multimodal questions is quantified well. Furthermore, the analysis of performance by image type (e.g., diagrams vs. photos), by subject (STEM vs. humanities), and by script (Latin vs. non-Latin) offers a deep look into the specific weaknesses of current VLMs.

### Weaknesses
- The reliance on a Multiple-Choice Question Answering (MCQA) format, while practical for evaluation, comes with inherent limitations. Models can sometimes guess the correct answer or exploit statistical cues in the options without genuine understanding. The authors acknowledge this point in the appendix, but this is a fundamental aspect of their evaluation design and should be more prominently discussed in the main paper's limitations section (Section A).

- The data collection process, despite being a large-scale effort, could benefit from more transparency regarding the source material's representativeness and difficulty. The paper notes that performance in some languages (like Lithuanian) may be higher due to the subject matter of the available exams. This suggests that the benchmark measures a combination of model capability and the specific difficulty of the sourced exams, which can be hard to disentangle. A more detailed discussion of the potential imbalances in difficulty across languages and subjects would be helpful.

- A potential confounding factor in the experimental results is the use of two different prompting strategies for closed and open-weight models (Chain-of-Thought for large models, direct answering for smaller ones). The authors justify this as a pragmatic choice, which is understandable. However, this means the comparison between these two groups of models is not perfectly controlled. The performance gap could be influenced by the different evaluation protocols. This should be more clearly stated and discussed as a limitation of the current experimental setup.

- The analysis of format errors (Section 5.3 and Table 1) is interesting, but its presentation could be more integrated. For example, the fact that GPT-4o's accuracy improves significantly when format errors are excluded is a very important finding that highlights a specific weakness of that model. This point, and the high refusal rate of Pangea, could be woven more directly into the main results discussion to give a clearer picture of model behavior.

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
4
