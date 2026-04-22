# MuBench: Assessment of Multilingual Capabilities of Large Language Models Across 61 Languages

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 4, 4

## Abstract
Multilingual large language models (LLMs) are advancing rapidly, with new models frequently claiming support for an increasing number of languages. However, existing evaluation datasets are limited and lack cross-lingual alignment, leaving assessments of multilingual capabilities fragmented in both language and skill coverage. To address this, we introduce MuBench, a benchmark covering 61 languages with 3.9M samples and evaluating a broad range of capabilities. We evaluate several state-of-the-art multilingual LLMs and find notable gaps between claimed and actual language coverage, particularly a persistent performance disparity between English and low-resource languages. Leveraging MuBench’s alignment, we propose Multilingual Consistency (MLC) as a complementary metric to accuracy for analyzing performance bottlenecks and guiding model improvement. \textsc{MuBench} provides flexible evaluation formats, including mixed-language testing. Experimental results show that increasing model size does not improve its ability to handle mixed-language contexts. We recruited human experts to evaluate translation quality and cultural sensitivity for 34k samples across 17 languages, and combined these assessments with an LLM-as-a-Judge approach to ensure overall data quality in low resource languages.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors create MuBench benchmark, covering 61 languages by translating existing datasets using their data collection pipeline with quality checks and perform human checks for 17 languages. They further evaluate LLMs on this benchmark. They perform cross-lingual consistency evaluation for consistent cross-lingual evaluation and analysis of knowledge transfer. They evaluate model performance under code switched contexts.

### Strengths
- Paper is well written and is easy to follow through.
- The authors covered a huge number of datasets and translated them to 61 languages which had high, medium, low-resource languages.
- Experimental setup is clearly explained and results are followed up by human evaluation.
- MuBench data collection pipeline looks thorough and has a lot of checks.
- Cross lingual consistency evaluation and creating code switched dataset to see performance on code switched data are great contributions

### Weaknesses
- I was a bit skeptical from the beginning about the translation quality but the fact that it was human-evaluated in 17 languages was reassuring. However, when I checked those 17 languages, most of them are either medium or high resource languages (61 languages in total out of which highest numbers are in low-resource languages (26)). This is a serious flaw in their paper. Ideally they should have picked an equal number of languages from high, medium, low resource for human evaluation. Existing LLMs don’t do well for low resource languages and I believe this is where the major gap is. I’d recommend authors to perform human evaluation for at least 8-10 low resource languages. This paper has got some amazing contributions and I’m willing to bump up the scores but low resource languages should be human evaluated to ensure correctness of their approach. 
- COMET doesn’t support more than 50 languages and they mention explicitly on their website to evaluate at your own risk for languages not mentioned.
- The authors missed out on defining what they do in the “problematic samples check” step.
- Using GPT for classification is an overkill I believe. The authors should have used more efficient approaches for this like a classifier.
- Existing datasets like MMLU have some problems [1], how did authors get rid of those samples? I didn't see any table in the paper about this.


[1] Aryo Pradipta Gema, Joshua Ong Jun Leang, Giwon Hong, Alessio Devoto, Alberto Carlo Maria Mancino, Rohit Saxena, Xuanli He, Yu Zhao, Xiaotang Du, Mohammad Reza Ghasemi Madani, Claire Barale, Robert McHardy, Joshua Harris, Jean Kaddour, Emile Van Krieken, and Pasquale Minervini. 2025. Are We Done with MMLU?. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5069–5096, Albuquerque, New Mexico. Association for Computational Linguistics.

### Questions
- I’m sorry if I have skipped this but do authors share samples which are culturally sensitive?
- I don’t understand Section 3.3 partially. When you look at both models’ top choices, how do you ensure they are the same? Let’s say one language is Chinese, other is Spanish, how do they ensure answer “England” (for Spanish it would be “Inglaterra” and Chinese would be “英格兰”) are same?
- Typos: 
  - Line 182: Translation
  - Line 280: Gemma2(?)
  - Line 307: Model(¿20B)

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
3

### Summary
This work introduces MuBench, a large-scale multilingual benchmark covering 61 languages and 3.9 million samples across a diverse range of tasks, including natural language understanding, factual knowledge, knowledge-based question answering, academic reasoning, and truthfulness. The evaluation framework examines three major aspects: overall performance, cross-lingual consistency (measured by the newly proposed MLC metric), and robustness to mixed-language inputs. The resulting experiments provide a more holistic understanding of multilingualism in large language models, offering valuable insights to guide future research and improvements.

### Strengths
- MuBench is broad in scope, spanning numerous languages, tasks, and samples.
- The dataset construction pipeline is carefully designed, incorporating checks for semantic consistency, translation purity, and cultural sensitivity. The authors further validate the reliability of the translations through expert evaluation on 34K samples and overlap verification with 100 samples from MMMLU.
- The experiments are conducted to evaluate the multilingual capabilities of various LLMs, revealing how cross-lingual consistency, and mixed-language contexts differ per-language performance.
- Overall, building such a large-scale benchmark and conducting these extensive evaluations represent a significant and commendable effort that will likely benefit the research community.

### Weaknesses
- The tasks in MuBench are mostly binary and multiple-choice formats, overlooking other important multilingual capabilities such as translation, summarization, and instruction following. This restricts the benchmark's overall applicability and impact.
- Some interpretive statements lack explicit numerical evidence. For instance, claims such as "Babel and Sailor2 demonstrate notable gains in their targeted language groups" or "smaller models often benefit from the presence of English in mixed-language inputs" would be stronger with accompanying statistical summaries or quantified comparisons (e.g., averaged improvements).
- Presenting the related work as a standalone section, instead of embedding it within lines 40–55 as a paragraph of the Introduction, would more clearly highlight the work's novelty.

**Minor Issues:**
- Typo: "Traslation" should be "Translation" (line 181).
- Missing period at the end of the sentence (line 242, after "in Appendix A.6").
- Citation error for Gemma2 (line 280).
- Table 1: "SC samples" should be corrected to "CS samples."
- Some indicators, and axis labels in figures are too small.

### Questions
- Why is Rel-MLC defined as MLC divided by mean accuracy? Since this normalization causes more accurate models to exhibit smaller Rel-MLC values, it may explain the apparent contradiction between MMLU and GPQA performance and the corresponding Rel-MLC values of GPT-4o (lines 387–399).
- How do LLMs respond to mixed-language inputs in terms of output language composition? Analyzing the languages used in outputs could shed light on why smaller models outperform their monolingual baselines, whereas larger models show the opposite trend.

### Soundness
3

### Presentation
2

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
The paper introduces a new multilingual LLM benchmark that is created by taking the existing popular English benchmarks and machine-translating them into 61 selected languages. The authors try to do automatic quality checks of the translation quality as well as a manual evaluation of 17 languages with native speakers of those languages. Using the final dataset, they then evaluate a large number of open-weight multilingual language models, showing that many of them perform much worse in other languages than English.

### Strengths
- The benchmark can evaluate language models on more than 60 languages, which can be very useful for those language communities -- as long as the translation is accurate and makes sense (more on that below)
- The authors double-check the quality of their translation pipeline with a manual inspection that involved native speakers of 17 languages.
- Each sample in the benchmark is annotated with the topic and sub-topic category, which might be very useful metadata for the future use of this dataset.
- There are many similar projects that take existing English datasets and naively translate them into many languages, without properly checking the translation quality. However, this work devises a multi-stage pipeline for automatically checking the translation quality.

### Weaknesses
As a native speaker of a lower-resource language, I find the machine-translated "multilingual" benchmarks somewhat troubling.

First of all, translationese is a problem even with human translation and much more with machine translation. You end up with a very specific unnatural variant of each languages that relies on English-like linguistic constructions and that might omit language features not present in English. As a result, such benchmarks give overly optimistic scores to English-centric language models that otherwise fail on properly created benchmarks made by native speakers. Thus, benchmarks like this can sometime do more harm than good.

My second point is more subjective; I would argue that cultural and local knowledge should be an inherent part of language evaluation -- does one really know Icelandic without recognizing a single Icelandic dish and not understanding any Icelandic cultural references? On the other hand, what I appreciate about this paper is that it takes this into account and tries to at least remove all cultural samples -- this is already much better than other similar papers that blindly translate mostly US-based questions. But it results in a dataset devoid of any local knowledge, which I believe only evaluates a certain aspect of multilinguality.

____

**Other weakness**:
- The translation from English is performed by GPT-4o, but the same model is then used to check the translation quality, which might leave many errors unchecked. It would be better to use different model(s) for the quality checks. 
- Another related troubling thing is that the performance of GPT-4o is substantially lower on some languages compared to English (Figure 3). Since the same model was used to translat the questions from English, it indicates that the translation for those language is very poor. I would assume that the performance of GPT-4o would be consistent across languages if its translation is correct.
- You say that *"we chose the 61 most widely spoken languages based on the number of native speakers"* (lines 105--106), which is not true. For example, Hausa (with 58 million speakers) or Bhojpuri (with 53 million speakers) are not included even though Icelandic (0.3 million speakers) is included.
- Translation of some of the tasks, those that rely more on specific language features, can be problematic. For example, if you translate the WinoGrande example "My shampoo did not lather easily on my Afro hair because the _ is too dirty. (answer: shampoo / hair)" into a language like Czech (where "shampoo" and "hair" are of different grammatical genders), the sample loses any ambiguity and thus it no longer evaluates the same thing. I wonder how much is the observed performance drop across many tasks in Table 5 connected to such issues. For example, losing more than 30 percentage points on translated HellaSwag (81.5 -> 49.4) but only 7 on MNLI (88.0 -> 80.4) is slightly concerning.

### Questions
- One thing that I didn't understand is why the cross-lingual alignment is such an important feature for a multilingual benchmark? From my point of view, the three related works listed in the introduction -- CMMLU, ArabicMMLU and INCLUDE -- are much more useful for evaluating multilinguality as they also localize the benchmarks. But you say (line 47) that the great benefit of your benchmark is cross-lingual alignment, so why is it so important? Cannot we evaluate consistency across languages without this alignment?
- As far as I can tell, the 17 evaluated languages are fairly high-resource, wouldn't it be more interesting to more closely check the translation quality of the lower-resource tail of languages?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces MuBench, a multilingual benchmark for evaluating LLMs across 61 languages covering tasks such as NLU, commonsense reasoning, factual recall, QA, and truthfulness. It emphasizes cross-lingual alignment, cultural sensitivity checks, and proposes a Multilingual Consistency (MLC) metric. The benchmark includes code-switched and multi-format (local, English-template, cloze, and mixed) variants. The authors evaluate several open and proprietary models, finding persistent performance gaps between English and low-resource languages and little improvement from model scaling.

### Strengths
- Ambitious scope and coverage: 61 languages and multiple task categories represent an impressive effort toward comprehensive multilingual evaluation.

- Detailed translation pipeline: The multi-stage quality control with semantic, purity, and cultural sensitivity checks is well-structured and thorough.

- Cross-lingual alignment and code-switching evaluation: Enables new analyses not possible with existing benchmarks.

- Transparency and openness: Dataset availability on Hugging Face improves reproducibility and potential reuse.

- Empirical findings: Highlights real and relevant disparities between English and non-English model performance.

### Weaknesses
- Limited novelty: The contribution is primarily engineering and dataset aggregation, not a clear conceptual or methodological innovation beyond existing multilingual benchmarks (e.g., MMLU, BenchMAX, INCLUDE).

- Benchmark saturation – Given numerous existing multilingual datasets, the incremental improvement offered by MuBench does not clearly justify publication in a top-tier venue like ICLR.

- Evaluation analysis lacks depth:  insight into causes or linguistic patterns, error analysis, and detailed methodological justifications are mostly missing or superficial.

- Repetition and length: The paper reads as overly descriptive and dataset-heavy, lacking theoretical framing or hypothesis-driven evaluation.

### Questions
as suggestion: since the paper focuses on cross-lingual alignment, the authors may also see this recent paper as well on the same topic: https://aclanthology.org/2025.findings-acl.1385/

### Soundness
2

### Presentation
3

### Contribution
1
