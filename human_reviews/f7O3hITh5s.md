# API Pack: A Massive Multi-Programming Language Dataset for API Call Generation

- Decision: Accept (Poster)
- Scores: 6, 6, 5, 6

## Abstract
We introduce API Pack, a massive multi-programming language dataset containing over one million instruction-API calls for improving the API call generation capabilities of large language models. Our evaluation highlights three key findings: First, fine-tuning on API Pack enables open-source models to outperform GPT-3.5 and GPT-4 in generating code for entirely new API calls. We show this by fine-tuning CodeLlama-13B on 20,000 Python instances from API Pack. Second, fine-tuning on a large dataset in one language, combined with smaller datasets from others, improves API generation accuracy across multiple languages. Third, we confirm the benefits of larger datasets for API generalization, as increasing fine-tuning data to one million instances enhances generalization to new APIs. To support further research, we open-source the API Pack dataset, trained model, and code at https://github.com/zguo0525/API-Pack.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces API Pack, a comprehensive multi-programming language dataset comprising over 1 million instruction-API call pairs designed to enhance the API call generation capabilities of large language models. By fine-tuning CodeLlama-13B on 20,000 Python instances from API Pack, the authors demonstrate that it outperforms GPT-3.5 and GPT-4 in generating unseen API calls. The fine-tuning process also facilitates cross-programming language generalization by leveraging a large volume of data in one language and smaller amounts from other languages. Scaling the training data to 1 million instances further improves the model's generalization of unseen APIs.

Contributions:

1. Introduction of API Pack Dataset: The authors present API Pack, a massive multi-programming language dataset containing over 1 million instruction-API call pairs, which is a significant contribution to the field of API call generation.
2. Model Performance Improvement: Through fine-tuning CodeLlama-13B on a subset of API Pack, the authors demonstrate superior performance in generating unseen API calls compared to GPT-3.5 and GPT-4.
3. Cross-Programming Language Generalization: The study highlights the dataset's ability to facilitate cross-programming language generalization, leveraging large amounts of data in one language and smaller amounts from others.
4. Scalability and Generalization: Scaling the training data to 1 million instances enhances the model's generalization capabilities to unseen APIs not used in training, showcasing the dataset's scalability.
5. Open-Source Initiative: The authors contribute to the research community by open-sourcing the API Pack dataset, the fine-tuned model, and the associated source code, fostering further research and development.

### Strengths
1. New Data: The authors introduce API Pack, a dataset containing over 1 million instruction-API call pairs. The dataset's multi-programming language nature and rich source diversity enable it to cover various scenarios. It will be useful for further research on improving the API call code generation of LLMs in different scenarios.
2. Performance Enhancement: The paper demonstrates the effectiveness of fine-tuning on API Pack by evaluating multiple models, including CodeLlama-13B, Mistral-7b, and Llama-2-13b. The results show that fine-tuning on API Pack significantly improves the models' ability to generate API call code, highlighting the dataset's value in enhancing model capabilities.
3. Extensive Experiments: The authors have conducted extensive experiments to validate their claims. The thoroughness of these experiments provides strong evidence for the dataset's effectiveness and the benefits of fine-tuning on it.

### Weaknesses
1. Clarification Needed for Methodology in Section 3.4: The description of the method used to filter high-quality instructions in Section 3.4 is unclear. 
2. Lack of Results for GPT-3.5 and GPT-4: The experimental section does not show results of GPT-3.5 and GPT-4 at Level 1 and Level 2.

### Questions
1. Model Selection for Quality Assessment: In the 3.4 section, the authors use the Mistral 7B model to validate the generated instruction, which is also the model used to generate the instructions. This raises concerns about the validity of using the same model to evaluate its outputs. It is unclear whether the 7B model can accurately distinguish between high-quality and low-quality instructions. A more robust approach might involve using a larger (>7B), more powerful LLM to assess the quality of the instructions.
2. Definition of Unseen APIs: The term "new API" refers to APIs that the model has not encountered during fine-tuning. However, given the lack of information about the models' pretraining data, it is challenging to verify whether the APIs labeled as "unseen" truly are. This lack of clarity could affect the validity of the experimental results. 
3. Evaluation Metrics for Code Generation: The authors use text-matching-based metrics for evaluating the generated API calls. However, for code generation tasks, metrics based on test case execution are more accurate and reliable. These metrics can provide a more objective measure of the correctness and functionality of the generated code. The authors should consider incorporating such metrics to enhance the evaluation process and provide a more comprehensive assessment of the models' performance.

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
The paper introduces API Pack, a large-scale, multi-programming language dataset containing over one million instruction-API call pairs generated semi-synthetically from API database collected from four API endpoints. This dataset is used for training LLMs on instruction API pairs and to demonstrate improvements in unseen code generation, cross-language transference.

### Strengths
* **Extensive Dataset with some human-annotation/filtering steps.** The collected dataset sizing over a million instances in unique and large covering a wide array of APIs and programming languages.
* **Experimental insights.** The paper performs API data scaling experiments, retrieval experiments and cross-language experiments.
  * The data scaling experiment highlights interesting observations across the three levels. Particularly, flat curve for Level-3 APIs does highlight a broader concern of out-of-distribution generalization for LLMs
  * The mixture model shows generalization to a new language with few examples solidifying the advantage of the collected large-scale dataset.

### Weaknesses
* **Novelty.** The paper provides limited novelty. It thoughtfully curates API sets and uses LLMs to generate synthetic training and test sets via prompting (with notably some human-in-the-loop quality assurance rounds). While potentially performed carefully, this following standard set of ideas in a somewhat different domain. Note that this is not necessarily a strict limitation and a carefully done execution can be helpful to the community -- perhaps primarily identified over time via the community. 

* **In-distribution evaluation.** The paper collects a synthetically generated dataset -- thus comprising instructions of limited natural language variance (compared with human written instructions). This dataset is next divided into a train and a test set. Thus the training and test set are precisely collected from the same and a small distribution. This partly challenges the findings from and we might see smaller gains to more "diverse" real-world queries. 

* **Marginal benefits from quality filtering.** The data filtering ablation reports performance improvement but it is notably small, often <1%. This raises concerns in the soundness of this step.

Minor: 

* **0-shot training does not improve 0-shot retre.** For the 20k fine-tuning experiments, the accuracy of 0-shot retrieval after 0-shot fine-tuning is worse than performance after fine-tuning on 3-shots. This is quite surprising -- and continues in Figure 4 as well and could be expanded further, perhaps with a qualitative analysis of the mistakes.

* **Metric.** The metrics used in the paper are not completely clear and should be more formally defined or described with examples or relevant citations.

### Questions
* Do the authors have more intuition behind the 0-shot vs 3-shot template results after fine-tuning? The findings seem inconsistent across Levels and data sizes (level-1 0-shot is better than 3-shot but for levels 2/3 3-shot is better in Figure 4)
* Can the authors clarify the metrics used to describe it more formally?
* Did the authors attempt to measure the "naturalness" of the collected instructions (automatically or through human annotation) or evaluate models on human written problems for a subset of the APIs?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces "API Pack," a large, multi-programming language dataset designed to enhance the API call generation capabilities of large language models (LLMs). API Pack aims to help LLMs generate API calls based on natural language instructions, addressing a key developer challenge of navigating and generating code from extensive API documentation. Experiments show that API Pack fine-tuning improves LLM performance in unseen API calls, enabling open-source models like CodeLlama-13B to outperform proprietary models GPT-3.5 and GPT-4.

### Strengths
- The focused problem is quite interesting. This paper fills a gap in LLM fine-tuning resources for cross-language API call generation, which has been under-explored.
- The dataset is robust, sourced from multiple reputable platforms, and undergoes a thorough pre-processing and validation pipeline. The authors’ experiments, conducted with open-source and proprietary models, validate API Pack’s effectiveness across multiple levels of generalization (known APIs, new endpoints, unseen APIs).
- The paper is clearly structured, detailing each stage of dataset creation, the methodology for instruction generation, and the evaluation framework used to measure LLM performance across scenarios. This transparency supports reproducibility and makes the dataset accessible for further research.

### Weaknesses
- [Major] Insufficient Related Work: The paper lacks discussion on some relevant studies in API learning. Notable examples include:

    - Zan, Daoguang, et al. "When language model meets private library." arXiv preprint arXiv:2210.17236 (2022).
    - Zhang, Kechi, et al. "Toolcoder: Teach code generation models to use API search tools." arXiv preprint arXiv:2305.04032 (2023).

- [Major] API Dataset Validity: Given the vast scale of the dataset, it’s unclear if all included APIs are usable or up-to-date. Additionally, APIs frequently evolve; fine-tuning on static API datasets could introduce outdated information or "hallucinations," potentially compromising the reliability of API knowledge encoded in the model. The authors should discuss this question.

- [Major] Limited Response Diversity: The paper does not sufficiently discuss the diversity of the code response. Using tools like OpenAPI may lead to highly uniform code outputs, such as the preference for a small set of common libraries for HTTP requests in Python or other languages, potentially constraining the model's generalization capabilities. This limitation might affect the model’s ability to use a broader range of libraries or to produce varied code styles when calling APIs. The authors should add experiments to show the influence of the response diversity.

- [Major] Lack of Baseline Comparison: A critical missing baseline is a model augmented with RAG or a search engine approach (e.g., ToolCoder), **without fine-tuning**, for handling complex API calls. This would provide a practical comparison for API Pack’s utility.

- [Minor] Base Model Selection: The selection of base models seems somewhat detached from practical application. API Pack functions as a specialized fine-tuning stage post-general SFT, targeting API call generation specifically. Therefore, selecting foundational SFT code models, like CodeLlama-Instruct or Magicoder, as discussed in Appendix A.8, would enhance the practical relevance and clarity of the paper’s contributions. Including more experiments with these models would substantiate the paper's claims.

- [Minor] Evaluation on General Code Generation Benchmarks: The paper does not discuss potential **performance trade-offs on general code generation benchmarks**, such as HumanEval, that could drop due to fine-tuning for API-specific tasks.

### Questions
How do the authors view the trade-offs between a RAG approach (with minimal or no fine-tuning) and the approach proposed here of fine-tuning on a large, static API dataset? Since APIs are continually updated, it seems that a RAG or search engine approach could offer more reliable, dynamic knowledge. How does the static nature of API Pack compare in reliability and practical utility?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces API Pack, a large-scale dataset with more than a million instruction-API call samples. The authors discuss the aspects of API Pack making it unique and how it was constructed from public sources. Experiments are conducted to demonstrate: (1) Code LLM performance on API Pack under various RAG and evaluation settings (2) strong multi-lingual generalization on API Pack tasks from training a lot on one language and a little bit on others, (3) data-scaling properties hold when training on API Pack data, (4) comparisons with another training dataset, ToolBench.

### Strengths
- **Useful contribution.** The paper presents API Pack, which is a large-scale and multi-lingual instruction-API calling dataset. This can be used for both training and evaluations in multiple languages. In Table 1, the authors demonstrate how this is better along multiple dimensions than previous works.
- **Several detailed insights.** The paper is packed with insightful experiments. I particularly liked the experiments covering multi-lingual generalization — the designs of control experiments for this are interesting, and show that it is sufficient to train on a lot of data from one language and little data from other languages. There are other experiments, including ablations on RAG strategies and data volume used during training, which will also be valuable to the community.
- **Reproducibility.** Section 3, discussing the creation of API Pack, is extremely detailed and will be of great value to researchers looking to process data for similar tasks.

### Weaknesses
- **Lack of certain details.**
    - The baseline performance of non fine-tuned models on API Pack is missing (unless I missed it somewhere). Including this will greatly speak to the advantages of fine-tuning on API Pack.
    - For Section 5.3, how is the multi-lingual generalization being measured? Is it the same API invocation across different languages?
    - Could API Pack suffer from data leakage when evaluating public models trained on public sources?
- **Additional experiments.** While there are several experiments conducted, I am curious about a few more experiments. This is not a major concern however. While Section 5.5 does some comparison of API Pack with ToolBench, I believe that a more thorough evaluation requires training on API Pack and evaluation on ToolBench, Gorilla, etc. and also training on ToolBench, Gorilla, etc. and evaluating on API Pack. This would concretely identify where API Pack stands out and lacks relative to other API calling benchmarks out there.
- **Paper writing.** Overall the paper seems to be a bit verbose, and could benefit from making the writing more concise and crisp. For e.g., Section 5.3 is too wordy and a jumble of multiple findings. It would benefit the paper to organize such findings in a clearer way.
- **Evaluation metrics.** While string matching based metrics are being used for evaluations here and have been used in several previous works as well, I think it is time that we move on from these inaccurate metrics. We should resort to more robust metrics relying on static or execution analysis such as pass rate [1] or hallucination rate [2] for measuring correctness of code completions.
- **Nitpicks.**
    - Line 381 - “simple retrieval might be sufficient in this context” — I think it can also be the case that using a better re-ranker might help improve performance. For e.g., what’s the performance with the oracle retriever/re-ranker?
    - Figure 2
        - The legend(s) should be moved to make the results more readable
        - Is the “Expert” model in each subplot different? This should be clarified in the text/caption given all models have the same color
    - Figure 4 - it would be better for comparison if all the subplots had the same y-axis limits

[1] Chen, Mark, et al. "Evaluating large language models trained on code." *arXiv preprint arXiv:2107.03374* (2021).

[2] Jain, Nihal, et al. "On Mitigating Code LLM Hallucinations with API Documentation." *arXiv preprint arXiv:2407.09726* (2024).

### Questions
See notes above. I have translated some of the above in question format below:

1. What is the baseline performance of non-fine-tuned models on API Pack?
2. How is multi-lingual generalization being measured in Section 5.3? Is it the same API invocation across different languages?
3. Have you considered training on API Pack and evaluating on other benchmarks like ToolBench and Gorilla, and vice versa, to more thoroughly compare API Pack with existing datasets?
4. What is the performance with an oracle retriever/re-ranker in the context of the simple retrieval discussion?
5. Can you characterize if there are any data leakage concerns and/or what researchers should be mindful of when evaluating with API Pack?

### Soundness
4

### Presentation
3

### Contribution
4
