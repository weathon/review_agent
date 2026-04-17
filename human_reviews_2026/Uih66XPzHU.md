# CLINIC : Evaluating Multilingual Trustworthiness in Language Models for Healthcare

- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Integrating language models (LMs) in healthcare systems holds great promise for improving medical workflows and decision-making. However, a critical barrier to their real-world adoption is the lack of reliable evaluation of their trustworthiness, especially in multilingual healthcare settings. Existing LMs are predominantly trained in high-resource languages, making them ill-equipped to handle the complexity and diversity of healthcare queries in mid- and low-resource languages, posing significant challenges for deploying them in global healthcare contexts where linguistic diversity is key. In this work, we present \textsc{Clinic}, a \textbf{C}omprehensive Mu\textbf{l}tilingual Benchmark to evaluate the trustworth\textbf{i}ness of la\textbf{n}guage models \textbf{i}n health\textbf{c}are. \name systematically benchmarks LMs across five key dimensions of trustworthiness: truthfulness, fairness, safety, robustness, and privacy, operationalized through 18 diverse tasks, spanning 15 languages (covering all the major continents), and encompassing a wide array of critical healthcare topics like disease conditions, preventive actions, diagnostic tests, treatments, surgeries, and medications. Our extensive evaluation reveals that LMs struggle with factual correctness, demonstrate bias across demographic and linguistic groups, and are susceptible to privacy breaches and adversarial attacks. By highlighting these shortcomings, \name lays the foundation for enhancing the global reach and safety of LMs in healthcare across diverse languages.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces CLINIC, a comprehensive multilingual benchmark designed to evaluate the trustworthiness of language models in healthcare applications. The benchmark assesses models across five key dimensions of trustworthiness (truthfulness, fairness, safety, robustness, and privacy) through 18 distinct tasks, covering 15 languages and six healthcare subdomains. The dataset comprises 28,800  samples generated through a two-step multilingual prompting approach. The authors evaluate 13 different language models, including proprietary systems, open-weight models.

### Strengths
The quality of the work is evidenced by the systematic evaluation framework and validation process. The involvement of healthcare professionals with over eight years of clinical experience in validating samples strengthens the benchmark's credibility. The paper is generally well-structured with clear presentation of methodology, results, and analysis. The supplementary materials, including detailed prompts, evaluation metrics, and per-language results, enhance reproducibility.

### Weaknesses
The benchmark's focus on text-only evaluation misses important multimodal aspects of healthcare applications. Modern medical AI systems increasingly integrate visual and structured data, making a text-only benchmark potentially insufficient for comprehensive trustworthiness assessment.

### Questions
The expert validation involved only two healthcare professionals. Have the authors considered expanding this validation to include more diverse medical expertise and cultural backgrounds, particularly for languages where medical practices may differ significantly?

### Soundness
3

### Presentation
2

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
This paper introduces CLINIC, a comprehensive and novel multilingual benchmark designed to evaluate the trustworthiness of language models (LMs) in the healthcare domain. The benchmark systematically assesses LMs across five key dimensions of trustworthiness: truthfulness, fairness, safety, robustness, and privacy, operationalized through 18 distinct tasks. A key contribution is its extensive coverage, comprising 28,800 expert-validated samples spanning 15 languages and six critical healthcare subdomains. The authors conducted a large-scale evaluation of 13 models, including proprietary, open-weight, and specialized medical LMs. The results reveal significant and widespread trustworthiness deficits in current LMs, particularly highlighting performance degradation in mid- and low-resource languages and underscoring the unreliability of these models for high-stakes, global healthcare applications.

### Strengths
- The work is highly original as it presents the first large-scale, multilingual benchmark specifically for evaluating trustworthiness in healthcare LMs, addressing a critical gap left by predominantly English-centric and narrowly focused prior research.
- The quality of the benchmark is exceptional due to its comprehensive structure, which includes five trustworthiness dimensions and 18 fine-grained tasks, and its creation process, which involved validation by medical professionals to ensure clinical relevance and accuracy.
- The paper is written with outstanding clarity. The motivation, methodology, and contributions are clearly articulated, and the use of figures, such as the overview of the CLINIC framework (Fig. 1) and the dataset construction pipeline (Fig. 2), effectively enhances understanding.
- This research holds significant importance for the field. By demonstrating the persistent weaknesses of modern LMs in a high-stakes, multilingual context, it provides a crucial tool for the community and a strong impetus for developing safer, more equitable, and globally applicable medical AI. The public release of the data and code further amplifies its impact by enabling standardized and reproducible research.

### Weaknesses
- The evaluation of several open-ended tasks relies heavily on GPT-4o as an automated judge. This methodology may introduce the judge model's own biases, and its reliability across 15 different languages is not thoroughly validated against human expert judgment, potentially affecting the soundness of the cross-lingual conclusions.
- The scope of the fairness evaluation is somewhat limited. While it addresses important aspects like gender, it does not deeply investigate other critical factors in healthcare equity, such as race, socioeconomic status, or disability, which are known sources of significant health disparities.
- The paper relies heavily on binary metrics for several tasks, such as Refuse-to-Answer (RtA) rates. These metrics, while useful, may oversimplify complex model behaviors and fail to capture nuances, for example, distinguishing between a correct refusal for a harmful prompt and an incorrect refusal for a safe one (i.e., exaggerated safety).
- While the paper provides an excellent diagnosis of the problems in current LMs, it does not discuss or propose potential mitigation strategies. A brief discussion on possible directions for improving model trustworthiness, even if preliminary, would have strengthened the paper's constructive impact.

### Questions
- Could the authors provide more details on the validation of GPT-4o as a multilingual judge? Were its judgments calibrated against human experts across the 15 languages for tasks like toxicity and privacy evaluation? How can we be confident that the observed performance drops in low-resource languages are not partially attributable to the judge model's own cross-lingual inconsistencies?
- The "two-step prompting" technique for multilingual generation is an interesting contribution. Can the authors elaborate on the mechanism that makes it superior to direct translation? Does this method encourage the model to perform a more contextual adaptation of the question rather than a literal translation, thereby improving clinical and linguistic nuance?
- The Out-of-Distribution (OOD) task uses drug names approved in 2025. What steps were taken to ensure these drug names were truly novel and absent from the training data of the models, particularly the proprietary ones which may have very recent knowledge cut-offs?
- The results consistently show that specialized medical LMs underperform compared to large-scale proprietary models on several trustworthiness tasks. What are the authors' main hypotheses for this finding? Could this be attributed to catastrophic forgetting during fine-tuning, the vast difference in model scale and pre-training data, or other factors?

### Soundness
4

### Presentation
4

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
This work introduces CLINIC, a multilingual benchmark for evaluating the trustworthiness of language models for healthcare. CLINIC covers 15 languages, and 18 distinct tasks which are categorized in 5 main categories: truthfulness, fairness, safety, robustness and safety. To create the CLINIC benchmark, MedlinePlus is used as the main data resource, which contains high-quality data moderated by U.S. federal agencies. The questions are generated by an LLM in two individual steps. In first step the LLM generates a question for the corresponding English content. In the second step, the LLM takes an input prompt which includes English content, generated English question, and content in the second language, and produces question in the second language. Two healthcare experts validate the generated questions. A diverse set of LLMs are evaluated on the proposed benchmark, including both commercial, open-weights and medical LLMs, with varying parameter sizes. The experiments reveal that privacy is a common concern for most of the evaluated models, medical LLMs suffer from hallucination more compared against to general-purpose models, and small models underperform and achieve weak performance in some languages.

### Strengths
- Comprehensive, detailed benchmark on an important intersection: healthcare and trustworthiness.
- Multilinguality aspects of the benchmark.
- A good diversity in tested models covering both commercial and open-weights models, together with medical-specific LLMs, considering different model scales.

### Weaknesses
- Question are generated by an LLM and expert validation is limited with two healthcare professionals, while the benchmark covers 15 languages, including low-resource languages such as Hausa and Nepali. As a result, the validation remains questionable, especially concerning the accurate generation of questions in low-resource languages. L874-876 mentions this, which is a **crucial limitation**, which lowers the soundness of this work.
- The presentation makes it difficult to read this paper. Results for some categories are visualized with categorical histograms, while results for some categories are reported on tables (I am aware of the fact that each category implements a distinct metric). Additionally, the tasks could have been represented with a more top-down approach with a big table containing one example per task (together with the corresponding metric), which would make reading this paper a lot more easy.

### Questions
- I doubt that Russian, Vietnamese, Bengali could be classified as mid-source languages as they are quite high-resource in  recent text-only pretraining corpora such as FineWeb-2.
- As Fig.4 shows the performance for perturbation experiments, how the models perform in logo-graphic languages (Chinese and Japanese's Kanji) in comparison to other languages under perturbed examples? Additionally, how is the semantic similarity calculated?
- What is the leaky rate evaluation metric? How is this implemented? Is there a reference/citation? I realize that the prompt is demonstrated in the appendix, but it can be mentioned, and also expanded with 1 or 2 more sentences.
- Which LLM is used to generate questions? I could not see this detail in the main text.

### Soundness
1

### Presentation
1

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposed CLINIC, which systematically benchmarks LMs across five key dimensions of trustworthiness: truthfulness, fairness, safety, robustness, and privacy. It operationalized through 18 diverse tasks, spanning 15 languages (covering all the major
continents), and encompassing a wide array of critical healthcare topics like disease conditions, preventive actions, diagnostic tests, treatments, surgeries, and medications.

### Strengths
1. The evaluation is very comprehensive including many evaluation dimensions and many tasks.
2. The evaluated models are very comprehensive to include general LLM and medical specific LLMs.

### Weaknesses
1. Bias in dataset construction: expert evaluation only involves two doctors and lacked cross-validation. I think this can lead to bias. It would be reasonable to include at least 5 experts to have cross-validation.

2. The dataset construction is centered at English, which may lead to bias to generate multi-lingual benchmark. It would be much better to build the benchmark of each language separately.

3. For Jailbreaking, the evaluation adopted GPT-4o for evaluation but lacks cross-validation and also did not provide evidences that the evaluation is not biased.

4. For this benchmark, what is the advantages compared to previous benchmark? For example, I think there are many biomedical benchmarks have already developed for LLM evaluation on different dimensions. What is the key advantage the benchmark provided? What can we do without this benchmark? Why we can not repurpose the old benchmark to evaluate at the proposed dimension?

### Questions
1. Can we trust the dataset without enough human experts' validation?
2. What did this dataset bring beyond the earlier LLM benchmarks?

### Soundness
3

### Presentation
3

### Contribution
3
