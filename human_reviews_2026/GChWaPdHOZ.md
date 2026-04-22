# BenchHub: A Unified Benchmark Suite for Holistic and Customizable LLM Evaluation

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 8, 4, 4

## Abstract
As large language models (LLMs) continue to advance, the need for up-to-date and well-organized benchmarks becomes increasingly critical. However, many existing datasets are scattered, difficult to manage, and make it challenging to perform evaluations tailored to specific needs or domains, despite the growing importance of domain-specific models in areas such as math or code. In this paper, we introduce BenchHub, a dynamic benchmark repository that empowers researchers and developers to evaluate LLMs effectively, with a focus on Korean and English. BenchHub aggregates and automatically classifies benchmark datasets from diverse domains, integrating 839k questions across 54 benchmarks. It is designed to support continuous updates and scalable data management, enabling flexible and customizable evaluation tailored to various domains or use cases. Through extensive experiments with various LLM families, we demonstrate that model performance varies significantly across domain-specific subsets, emphasizing the importance of domain-aware benchmarking. Furthermore, we extend BenchHub into 10 languages spanning resource levels. We believe BenchHub can encourage better dataset reuse, more transparent model comparisons, and easier identification of underrepresented areas in existing benchmarks, offering a critical infrastructure for advancing LLM evaluation research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a benchmark designed to aggregate and categorize evaluation datasets from different domains and languages for LLMs. Each question is automatically labeled along a six-dimensional taxonomy, and the authors train an LLM-based classification model to dynamically update the benchmark. Experiments show that LLM performance rankings shift significantly when evaluations are filtered by domain or skill.

### Strengths
- The structured labeling of benchmarks in the form of a multi-dimensional taxonomy is novel and useful, as it allows users to dissect model performance along interpretable axes.
- The fully automated data pipeline allows for dynamic updating; i.e., new benchmark datasets can be integrated into BenchHUb with minimal manual effort. 
- The benchmark’s multilingual extension is a strength, as it begins to address evaluation in underexplored languages.

### Weaknesses
- The paper would benefit from better differentiating Benchhub from existing evaluation platforms. For example, how does BenchHub’s taxonomy differ from or improve upon FLASK’s skill taxonomy or HELM’s scenario-based metrics. Additionally, the paper omits discussion of several recent benchmarks/platforms aimed at holistic or robust LLM evaluation, including RAVEL, LLMBar, and “Arena Hard”. It would benefit the paper to move the Related Work earlier and more clearly differentiate prior works from the current work. 
     
      Huang, J., Wu, Z., Potts, C., Geva, M., & Geiger, A. (2024). RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 8669-8687).

      Zeng, Z., Yu, J., Gao, T., Meng, Y., Goyal, T., & Chen, D. (2024). Evaluating Large Language Models at Evaluating Instruction Following. In The Twelfth International Conference on Learning Representations.

      Li, T., Chiang, W. L., Frick, E., Dunlap, L., Wu, T., Zhu, B., ... & Stoica, I. (2025). From Crowdsourced Data to High-quality Benchmarks: Arena-Hard and Benchbuilder Pipeline. In Forty-second International Conference on Machine Learning.


- There are some concerns about the methodology and the robustness of the results. First, the evaluation primarily uses multiple-choice or short-answer questions, with automatic regex or LLM answer extraction. This means open-ended generation tasks (e.g., long-form answers) appear to be largely excluded from the results. Second, the paper reports single-run accuracy for model comparisons. The absence of variance measures is problematic given that benchmark sampling was involved. For instance, Table 14 in the Appendix compares model performance, but we don’t know how sensitive those rankings are to the particular sample drawn. 

- Are the BenchHub evaluation sets too easy for powerful LLMs? One concern is the difficulty level of the included tasks. Many of the integrated benchmarks are well-studied and powerful LLMs approach or exceed human-level on them. For example, several proprietary and open-source models exceed 85% accuracy on MMLU and 70% on Multilingual-MMLU. The Humanity’s Last Exam (HLE) was created specifically to address the saturation of MMLU by providing expert-written questions. BenchHub’s focus, however, is on existing benchmarks that are already saturated. More discussion is needed on how BenchHub can discern frontier model capability and avoid being over-saturated. 

      Phan, L., Gatti, A., Han, Z., Li, N., Hu, J., Zhang, H., ... & Wykowski, J. (2025). Humanity's last exam. arXiv preprint arXiv:2501.14249

- The paper does not sufficiently discuss the limitations of its automated categorization and benchmark integration approach. The authors acknowledge in Appendix A that the single LLM-based classifier to label all questions may have biases. This limitation and others should preferably be discussed in the conclusion instead of the Appendix. An unaddressed limitation is the lack of a human verification step for the taxonomy labels — we don’t know how accurate the categorization is, because no validation results are reported. Lastly, dataset contamination is mentioned as a limitation for static benchmark datasets, but is not discussed with respect to BenchHub. For example, the authors do not analyze whether the evaluated models had prior exposure to test questions.

### Questions
- The use of the term “target” to denote culturally specific vs. agnostic questions may cause confusion, since target is typically used to describe the label in supervised learning. Renaming this dimension to “culture-specificity” or “culturally specific target” would improve clarity. 
- Typo: “Humanities and Social Sciencce” (pg. 3)

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper discusses need for aggregration of scattered benchmarks and proposes BenchHub, which automatically classifies benchmark datasets and integrates them. This enables used to select exact questions which can be used evaluate models across specific requirements. The authors build a taxonomy using 3 dataset level attributes (`task`, `answer-format`, `tool-usage`) and 3 sample-level attributes (`skill`, `subject`, `target` (local/global)). 54 benchmarks (across English and Korean) are collected, reformatted, and categorized at a sample-level (using a finetuned version of `Qwen-2.5-7B`). The authors also expand their framework to 10 languages with a multilingual classifier LLM. Finally, the paper analyses LLM rankings with various ablations of datasets categories, sampling strategies and custom tailored evaluations.

### Strengths
- The paper is well-written and presents coherent and cogent arguments. 
- I personally appreciate the motivation and problem statement in the paper, and I believe it is quite relevant given the numerous "fragmented" datasets and benchmarks published every day independently. I believe that BenchHub can be the cohesive factor for this group and can help with a more holistic and tailored evaluation of LLMs. 
- I also appreciate the author's attempts to push for a multilingual version of BenchHub. Given that the artifacts are promised to be open-sourced, this would be a great contribution to the community as well.

### Weaknesses
- As of now, I don't see any glaring errors in the paper, and I don't see any such weakness from my side. I feel the paper has answered and justified its problem statement appropriately (for me). I am open to interacting with the authors and fellow reviewers during the rebuttal phase.

### Questions
- The appendix of the paper is too large, and it's a pain to jump back and forth to refer to artifacts below. Please try to include some of the main results and Qwen-Cat training notes in the main paper.
- The reformatting using a proprietary model like GPT-4o or Gemini can be expensive in the long run. Maybe the authors can explore other OSS models or large reasoning models.  
- Was there any sort of human evaluation done to check if the classification is correct?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents BENCHHUB, a dynamic benchmark repo. It is designed to support integration of new datasets via an instance-level fine-tuned classifier and an integration agent. BENCHHUB covers many benchmarks and 10 languages. Experiments show different rankings across categories.

### Strengths
1. The data size, number of examples, models, and languages are good.
2. The automation part is a good contribution, both the classifier and the agent.
3. The point that different subsets can lead to different rankings goes through well.
4. The motivation argument in favor of more dynamic evaluation is convincing.

### Weaknesses
1. The presented solution relies on introducing a new categorization of examples. However, this merely replaces the benchmark categorization with another (yours), without providing a real domain-adaptive dynamic evaluation setup. My interests may differ significantly from the ones modeled (e.g., physics, simplification in Kordish). This is a core issue I see with the proposed solution. While the system can be extended, doing so requires substantial effort from users. Moreover, the proposed categorization may overlook benchmark-specific subsets that contain valuable information.
2. The classifier is mentioned but not experimentally evaluated or compared with other LLMs without additional training. It also requires substantial resources to annotate new data. Furthermore, fine-tuning fixes the categories, which introduces rigidity and limits adaptability.
3. There are no quantitative results, which makes the findings seem anecdotal.

### Questions
1. The work appears to be very tool-focused; if so, it might be more appropriate to position it as a demo paper.
2. Why does the experiment in Section 4.1 include only a subset of the models?
3. The “Tool Usage” subset is unclear. Is it meant to evaluate the tool call itself or the final answer? In either case, where does the tool specification originate?

### Soundness
2

### Presentation
2

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
The paper introduces BenchHub, a unified framework that consolidates diverse LLM benchmarks into a common schema and enables custom, domain-specific evaluations. It aggregates 54 benchmarks across 10 languages (mainly English and Korean) and annotates each dataset and sample along six dimensions: task, answer format, tool usage, skill, subject, and target. The LLM-based pipeline reformats the dataset problems, assigns metadata, performs categorization and merges new problems. Several open-weight and proprietary models are evaluated, and the results show that models rankings are very sensitive to subject and the sampling strategy.

### Strengths
- Bringing together many datasets under a single schema with fine-grained and multi-label subjects is valuable and supports more grounded analysis, given the multitude of benchmarks out there.
- The results about rankings that change across subjects and sampling criteria are interesting and make the point for having a unified suite
- The merging pipeline is completely automated, allowing to easily integrate new datasets.

### Weaknesses
- The evaluation is done by constraining the evaluation to MCQ or short form. This might skew the ranking results, as the original benchmark might use a different metric. This score aggregation needs to be done carefully, and it’d be useful to have also the original scores to compare against.
- Even with a taxonomy, mixing and samples from different benchmarks can drift from any single benchmark purpose.
- It’s not clear how sensitive models are over other sample-level attributes.

### Questions
- I tried to take a look at the repo, but I’m not able to load any file in any folder. Is it an issue on my end?
- Together with the overall score, it would be helpful to report also the single benchmark scores included in the selection from the user.
- It would be interesting to see how ranking change across other dimensions, for example across other sample-level attributes.

### Soundness
3

### Presentation
3

### Contribution
2
