# Beyond Text-Only: Towards Multimodal Table Retrieval in Open-World

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4, 4

## Abstract
Open-domain table retrieval aims to retrieve semantically relevant structured tables from a large-scale corpus in response to natural language queries. Unlike unstructured text, tables store information not only through their textual or numerical content but also through their structural properties, including hierarchical relationships between headers and cells, as well as complex spatial arrangements within the table layout. Existing methods predominantly treat table retrieval as a variant of text retrieval. They struggle to accurately preserve the rich structural semantics of diverse table formats during text serialization. Existing methods typically flatten tables into linear text sequences through row-wise or column-wise serialization, inadvertently discarding structural information. The problem becomes particularly acute when processing complex table layouts containing merged cells or irregular alignments, ultimately compromising retrieval performance. Moreover, existing methods struggle to handle embedded images within table cells. Notably, visual representations inherently preserve both structural and content information while being format-agnostic. This insight motivates our exploration of image-based table retrieval, as it can naturally overcome the challenges faced by existing methods. In this paper, we introduce TaR-ViR (Table Retrieval via Visual Representations), a new benchmark that reformulates table retrieval as a multimodal task by treating tables as images. Experiments on TaR-ViR show that this paradigm shift achieved more effective and efficient retrieval performance. Crucially, it eliminates the need for error-prone text conversion, enabling scalable collection and utilization of open-world tables. Our data are available at https://github.com/Trustworthy-Information-Access/Tab-ViR.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes TaR-ViR, a new benchmark that reframes open-domain table retrieval as a multimodal task by treating tables as images rather than serialized text. The authors crawl Wikipedia screenshots for tables and derive a corpus with 81,839 tables and 8,646 query–table pairs. They use a hybrid MLLM-assisted annotation pipeline: using Qwen2.5-VL-72B proposes labels which are then partially verified by humans. The evaluation of the paper compares text retrievers with multimodal retrievers, and reports that large multimodal retrievers can match or surpass strong text retrievers on several ranking metrics. They also include a RAG study showing modest gains when using multimodal retrievers for table-QA inputs, while text-only LLMs still achieve a higher upper bound overall.

### Strengths
**1. Well-motivated**: Directly operating on table images addresses notorious issues with serialization losses, such as merged cells, hierarchical headers, embedded figures that harms text-only pipelines. 

**2. Scale & coverage**: The pipeline begins from 2M screenshots and yields a large image-centric retrieval benchmark with 81,839 tables; clear dataset statistics are provided. 

**3. Practical annotation design**: The MLLM-assisted relevance and answering workflow with human verification is cost-aware and reports empirical quality (80% held rate), with manual curation reserved for the test set. 

**4. Evidence of benefits**: On TaR-ViR, strong multimodal models (e.g., VLM2Vec-7B) can outperform the best text retrievers (e.g., BGE) on ranking metrics when using title+content images.

### Weaknesses
**1. Benchmark scope**: The corpus is Wikipedia centric. Table images, such as scanned documents, enterprise spreadsheets, and PDFs beyond Wikipedia layouts is not evaluated, limiting external validity. 

**2. Novelty**: The novelty of hybrid data collection pipeline of TaR-ViR is limited. 

**3. RAG gains**: The RAG table shows modest benefits from multimodal retrieval, while text LLMs still dominate, leaving the practical payoff ambiguous for QA pipelines that can rely on OCR and text LLMs. 

**4. Annotation noise**: The auto-labeled training set is reported with 80% precision suggests non-trivial label noise that could bias model training. More rigorous robustness analysis, such as different MLLMs' label-noise sensitivity, would strengthen claims. 

**5. Reliance on titles**: Several best results assume title and image are both available. In the wild, titles may be missing or noisy. Extra results on how performance degrades when titles are unavailable are helpful.

### Questions
Check the weaknesses.

### Soundness
2

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
This paper introduces TaR-ViR, a vision-based benchmark for table retrieval, providing an in-depth analysis of the fundamental limitations of text-based table retrieval methods when handling complex table structures and embedded images. It also proposes a novel paradigm that treats tables as images for retrieval.
The benchmark is constructed with a rigorous methodology that balances both quantity and diversity, and offers valuable insights for future research through comprehensive experiments.

### Strengths
1. This work provides an in-depth summary of the fundamental limitations of text-modal tables in handling complex table structures and embedded images.  
2. The benchmark construction methodology is sound, cost-effective, and exhibits sufficient volume and diversity.  
3. The experiments are reasonably comprehensive and offer a thorough comparison of various retrieval configurations.

### Weaknesses
1. Some expressions are overly strong. For example, the text modality is not entirely incapable of representing certain heterogeneous tables, and it must be admitted that text-based retrieval can be simple and fast (especially when the original data is already in text form). Similarly, table-modal tables are not perfect either — for instance, visual modalities also face challenges when dealing with very large database tables.
2. RAG also includes reranking. Although more combinations would significantly increase experimental complexity, it is still recommended to add some related experiments.
3. Are the criteria for difficulty classification somewhat oversimplified?

### Questions
1. As a benchmark for RAG, it is recommended to incorporate more table-specific performance metrics, such as robustness to row/column permutation (for permutable tables), rendering method robustness, and image resolution robustness.
2. The data deduplication phase employed the CLIP model, while CLIP is also used in subsequent comparisons. This may introduce potential bias.
3. This paper identifies limitations of text-modal tables, such as difficulty in handling visual elements (e.g., national flags, emojis). However, the benchmark statistics do not specify the proportion of such tables, making the benchmark less targeted.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors extend the existing open-domain table retrieval paradigm, framing it as a visual retrieval task rather than the previous text retrieval task. Based on this, they construct TaR-ViR, the first multimodal table dataset that supports both QA and retrieval.

### Strengths
The authors present their ideas clearly with fluent logic, and the overall experiments are comprehensive.
1. Novel Paradigm for Table Retrieval: The author fills a critical gap in existing research by being the first specialized benchmark that reformulates table retrieval as a multimodal task, treating tables as images instead of relying solely on text. Experiment results demonstrating that image-centric table retrieval can outperform text-based methods (especially in recall and handling complex structures) while eliminating error-prone OCR/text conversion, the work proposes a more efficient and flexible paradigm for real-world table retrieval. It also supports both retrieval and QA evaluations, enhancing its utility for diverse downstream tasks.
2. High-Quality Data Construction: Leveraging Wikipedia’s rich table resources and NQ-TABLES’ foundation, the dataset ensures real-world relevance while resolving key issues (redundancy, temporal relevance shifts) via a cost-efficient annotation pipeline combining MLLMs (Qwen2.5-VL-72B) and human verification.

### Weaknesses
1. During the data construction process, the authors initially collected 2 million table screenshots based on NQ-TABLES, which contains approximately 100,000 document data entries. Why is there such a large discrepancy in the data collection process? It is necessary to specify how the data volume changes at each subsequent step of data processing, as well as the data splitting method and whether it is consistent with that of NQ-TABLES.
2. The authors also mention in the paper that the comparisons in the experiments of Section 5 are unfair, and I agree with this view. I believe the conclusions derived here are not sufficiently compelling—they only demonstrate that table retrieval tasks can be completed using visual elements instead of text. Additionally, I am concerned about the results of "title + content (web)", which the authors have not provided here.
3. In the experiments of Section 6, the authors observe that performance degrades due to OCR limitations when tables become complex, but performs better on simple tables. I am curious whether combining image and text content would yield better results; furthermore, regarding the definition of complex tables, I believe relying solely on size is insufficient, and the authors could attempt to derive tables of different complexities by decomposing table parsing structures.
4. In Table 6, the authors should present the improvements of multimodal retrieval compared to text retrieval. Currently, for RAG applications, it does not show significant advantages over text retrieval, which raises doubts about whether this paradigm is superior to text-based approaches. Moreover, the performance disadvantage should not be solely attributed to MLLMs.
5. The authors also state that there are images in tables that existing methods cannot handle. If MLLMs are utilized to embed images into text sequences, what would the overall performance be?

### Questions
See weakness part

### Soundness
3

### Presentation
4

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
This paper introduces TaR-ViR, a multimodal benchmark that redefines open-domain table retrieval by treating tables as images rather than text sequences. The work argues that text-only approachesfail to capture the rich structural and spatial semantics of real-world tables. TaR-ViR extends the NQ-TABLES dataset by collecting approximately 2 million table screenshots from Wikipedia and aligning them with natural-language queries via a semi-automated annotation pipeline that leverages MLLMs.
Comprehensive experiments compare text-based retrievers against multimodal ones, showing that multimodal retrievers achieve competitive or superior performance, particularly in recall and large-scale retrieval efficiency.

### Strengths
- The benchmark scale is large. 

- TaR-ViR provides a full framework integrating visual annotation, OCR-based comparison, and RAG-based downstream tasks.

### Weaknesses
- Although the paper claims open-world applicability, TaR-ViR relies entirely on Wikipedia-sourced tables. These are relatively clean, consistently formatted, and visually homogeneous.

- The reliance on Qwen-VL for pseudo-labeling and 80% correctness in human verification introduces potential label noise. The test set’s limited manual validation raises concerns about bias propagation, especially since multimodal retrievers were trained on partially machine-labeled data.

- While Section 7 integrates TaR-ViR into a RAG QA pipeline, the downstream results show only marginal accuracy improvements and rely primarily on recall.

- The paper’s primary contribution is dataset construction and evaluation rather than a new retrieval architecture. Its impact may hinge on community adoption rather than algorithmic innovation.

### Questions
- Would performance degrade if the tables included handwritten or low-quality scanned data, given that all current images are digital screenshots?

- Could multimodal retrievers trained on TaR-ViR generalize to document-level retrieval tasks where tables coexist with charts or paragraphs?

- The benchmark focuses on retrieval efficiency but omits latency and compute cost comparisons between OCR-based and pure visual retriever?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper targets at open-domain table retrieval. Instead of using text as the source of retrieval, this paper proposes to use image as an alternative. This paper introduces a new benchmark TaR-ViR based on an existing text-based table benchmark. The authors have conducted some ablation studies that show how the table image would help in scenarios like RAG setup.

### Strengths
- Though treating tables as images has been explored in various existing literature (see weakness 3), the authors introduce this setup in the table retrieval.

- The authors have conducted experiments to show the potential of treating tables as images.

### Weaknesses
- I wonder if the dataset is comprehensive enough to cover diverse visual web table types. As stated by the authors, `...adapt a textual table retrieval dataset.`, the authors may ignore the types of tables explored in [11]. I believe this distinguishes the visual table understanding / retrieval from considering tables just from the text perspective.

- [12] has conducted ablations on different resolutions of table images, [1] has proposed together with different table formats in text, different image formats for tables. It would be nice for the authors to conduct similar ablation studies to understand how these factors play in your setup. 

- In certain experiments, the size and the type of models used are limited. For instance, in the RAG setup, the authors mostly conducted their experiments on 7-8B sized LLMs with 2B retrievers.

- Related works such as [1] is worth mentioning. In the related work section, it is worth mentioning more efforts from the table community, for instance, the ones working on architecture changes [2], the recent waves on instruction tuning foundational LLMs on tables, including [3, 4, 5, 6, 7, 9], and the line of research on investigating table representations [8, 10].

### References

[1] Naihao Deng, Zhenjie Sun, Ruiqi He, Aman Sikka, Yulong Chen, Lin Ma, Yue Zhang, and Rada Mihalcea. 2024. Tables as Texts or Images: Evaluating the Table Reasoning Ability of LLMs and MLLMs. In Findings of the Association for Computational Linguistics: ACL 2024, pages 407–426, Bangkok, Thailand. Association for Computational Linguistics.

[2] Jingfeng Yang, Aditya Gupta, Shyam Upadhyay, Luheng He, Rahul Goel, and Shachi Paul. 2022. TableFormer: Robust Transformer Modeling for Table-Text Encoding. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 528–537, Dublin, Ireland. Association for Computational Linguistics.

[3] Tianshu Zhang, Xiang Yue, Yifei Li, and Huan Sun. 2024. TableLlama: Towards Open Large Generalist Models for Tables. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6024–6044, Mexico City, Mexico. Association for Computational Linguistics.

[4] Xiaokang Zhang, Sijia Luo, Bohan Zhang, Zeyao Ma, Jing Zhang, Yang Li, Guanlin Li, Zijun Yao, Kangli Xu, Jinchang Zhou, Daniel Zhang-Li, Jifan Yu, Shu Zhao, Juanzi Li, and Jie Tang. 2025. TableLLM: Enabling Tabular Data Manipulation by LLMs in Real Office Usage Scenarios. In Findings of the Association for Computational Linguistics: ACL 2025, pages 10315–10344, Vienna, Austria. Association for Computational Linguistics.

[5] Naihao Deng and Rada Mihalcea. 2025. Rethinking Table Instruction Tuning. In Findings of the Association for Computational Linguistics: ACL 2025, pages 21757–21780, Vienna, Austria. Association for Computational Linguistics.

[6] Li, Peng, et al. "Table-gpt: Table fine-tuned gpt for diverse table tasks." Proceedings of the ACM on Management of Data 2.3 (2024): 1-28.

[7] Zha, Liangyu, et al. "Tablegpt: Towards unifying tables, nature language and commands into one gpt." arXiv preprint arXiv:2307.08674 (2023).

[8] Li, Liyao, et al. "Table as a Modality for Large Language Models." The Thirty-ninth Annual Conference on Neural Information Processing Systems.

[9] Su, Aofeng, et al. "Tablegpt2: A large multimodal model with tabular data integration." arXiv preprint arXiv:2411.02059 (2024).

[10] Long, Lin, et al. "Bridging the Semantic Gap Between Text and Table: A Case Study on NL2SQL." The Thirteenth International Conference on Learning Representations. 2025.

[11] Titiya, Prasham Yatinkumar, et al. "MMTBENCH: A Unified Benchmark for Complex Multimodal Table Reasoning." arXiv preprint arXiv:2505.21771 (2025).

[12] Mingyu Zheng, Xinwei Feng, Qingyi Si, Qiaoqiao She, Zheng Lin, Wenbin Jiang, and Weiping Wang. 2024. Multimodal Table Understanding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9102–9124, Bangkok, Thailand. Association for Computational Linguistics.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
