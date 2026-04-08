## Human Reviewer 1

### Summary
The paper introduces a multimodal retrieval-augmented model that combines text and image data to improve the model's question-answering ability. Experiments demonstrate that their system can improve retrieval accuracy and good performance in multimodal tasks.

### Strengths
1. Integrating multimodal retrieval into vision-language models is intuitive.

### Weaknesses
1. The paper does not sufficiently acknowledge prior work in multimodal retrieval-augmented generation, such as [1,2,3]. The differences between the proposed approach and these works are unclear.
2. The experiments are limited, with only baseline model numbers provided, making it difficult to draw meaningful insights or assess improvements over previous methods.
3. The paper could benefit from clearer writing. Figures are low-resolution, captions lack detail, and basic concepts (e.g., F1 and precision/recall) are over-explained, occupying space that could be used for more insightful discussions.

[1] Yasunaga et al., Retrieval-Augmented Multimodal Language Modeling, ICML 2023. 

[2] Chen et al., MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text, EMNLP 2022. 

[3] Hu et al., REVEAL: Retrieval-Augmented Visual-Language Pre-Training With Multi-Source Multimodal Knowledge Memory, CVPR 2023.

### Questions
What are the main differences between your work and previous ones (e.g., [1,2,3])?

### Soundness
2

### Presentation
1

### Contribution
1

### Rating
3

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper proposes a multimodal retrieval-augmented generation (RAG) question-answering system, which builds a dataset comprising text and image data and uses ColPali and GPT-4o as the retriever and multimodal large language model to retrieve and answer questions. Experimental results on the constructed dataset show that this method outperforms three baseline algorithms in terms of retrieval accuracy and answer quality.

### Strengths
1. Document question-answering is a critical task, and the use of large models aligns well with current technological trends.
2. Introducing multiple modalities to improve answer accuracy in document question-answering is a sensible approach.
3. The paper contributes a dataset for document QA that includes images, text, and question-answer pairs, providing valuable resources for the field.

### Weaknesses
1. The main contributions of this paper are unclear, and the novelty appears limited, resembling more of a straightforward combination of existing models. While the authors note that the retrieval module in current RAG models faces challenges with long documents and that multimodal data holds rich information, these challenges are not directly addressed in the methodology, which instead relies on ColPali and GPT-4o as the main components.
2. The literature review is relatively limited, lacking recent studies such as Self-RAG. This narrow focus reduces the coherence of the paper’s content and its persuasiveness.
3. The proposed dataset lacks detailed description, including data sources, proportions of various data types (e.g., professional papers, industry reports, white papers), and the dataset construction process. Moreover, there are no detailed statistics provided for the dataset, nor any discussion on the necessity of building a new dataset or a comparison with existing datasets.
4. The experiments only utilize the constructed dataset, without any evaluation on public datasets such as arXivQA, TAT-DQA, or InfoVQA. Additionally, in terms of reproducibility, the paper does not provide any training parameters.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
1

### Rating
3

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper presents a multimodal RAG system for QA, the authors constructed a dataset for finetuning the retrieval model and some empirical results are shown in the paper. Overall, this is a poorly-written paper without too much scientific significance. While the authors repetitively claim the proposed approach is a **novel** approach, the designed framework as shown in Figure. 1 has nothing special with the traditional retrieval pipeline system. Moreover, the authors said that the retrieval system is based on **ColPali** that is a retrieval system processing documents as image patches utilizing large VLMs , however the proposed framework in this work is completely different from **ColPali**. Furthermore, there lacks details of the construction of the QA dataset such as the data source, statistics of documents and QA pairs. The details of the retrieval component and QA model are also missed. The experimental designation is also ambiguous, making it extremely difficult for readers to understand the results (where are the ranking documents from?). The writing of the paper should be improved, it is not easy to read this paper as the flow is not fluent and many details are missed. Therefore, I strongly recommend to reject this paper.

### Strengths
None

### Weaknesses
- While the authors repetitively claim the proposed approach is a **novel** approach, the designed framework as shown in Figure. 1 has nothing special with the traditional retrieval pipeline system. 

- Moreover, the authors said that the retrieval system is based on **ColPali** that is a retrieval system processing documents as image patches utilizing large VLMs , however the proposed framework in this work is completely different from **ColPali**. 

- Furthermore, there lacks details of the construction of the QA dataset such as the data source, statistics of documents and QA pairs.

- The details of the retrieval component and QA model are also missed. 

- The experimental designation is also ambiguous, making it extremely difficult for readers to understand the results (where are the ranking documents from?). 

- The writing of the paper should be improved, it is not easy to read this paper as the flow is not fluent and many details are missed.

### Questions
Where did the authors collect these documents and where are the ranking documents from?

### Soundness
1

### Presentation
1

### Contribution
1

### Rating
1

### Confidence
5

---

## Human Reviewer 4

### Summary
This paper presents a multimodal retrieval-augmented generation (RAG) question-answering system that combines large language model capabilities with external knowledge bases containing images and text. Its goal is to improve the quality and accuracy of answers in multimodal contexts, such as complex documents that include charts and flowcharts. The system is designed as a pipeline and optimized based on the dataset constructed by the authors.

### Strengths
1. The proposed paradigm is presented clearly and easy to follow.
2. The paper is focusing on a promising direction: Multimodal RAG.
3. The proposed system achieves a fast response time while maintaining quality.
4. The evaluation metrics are described clearly.

### Weaknesses
1. Lack of novelty: this is not the first work to explore the multimodal RAG system [a,b,c], while the paper merely proposes a common pipeline. Moreover, the authors do not make significant efforts in component optimization, pipeline innovation, and data optimization. The system is basically a combination of ColPali [d] and GPT-4o.

2. Insufficient information: as a core contribution of this work, the proposed ‘high-quality’ dataset is not discussed properly. Section 4.1.1 only describes the dataset roughly and does not provide in-depth information, such as the data source and component ratio, which are significant.

3. Limited method description: the authors only provide a simple overview (Section 3.1) and a blurry figure (Figure 1) of the system. There is no further discussion on the proposed system. The authors are advised to discuss the integration process between ColPali and GPT-4o, and how the multimodal information is processed and combined.

4. Limited experiments: only one experiment on the proposed dataset, which is not open source and is described vaguely.

5. Section 4.3.3 is intended to compare the system response speed among the baselines, while the authors only provide the response time of their proposed system. The authors are suggested to provide a comparative table of response times for all systems mentioned, including the baselines, to allow for a proper comparison.

6. Lack of implementation details.

[a] Chen W, Hu H, Chen X, et al. Murag: Multimodal retrieval-augmented generator for open question answering over images and text[J]. arXiv preprint arXiv:2210.02928, 2022.

[b] Chen Z, Xu C, Qi Y, et al. Mllm is a strong reranker: Advancing multimodal retrieval-augmented generation via knowledge-enhanced reranking and noise-injected training[J]. arXiv preprint arXiv:2407.21439, 2024.

[c] Liu Z, Sun Z, Zang Y, et al. Rar: Retrieving and ranking augmented mllms for visual recognition[J]. arXiv preprint arXiv:2403.13805, 2024.

[d] Faysse M, Sibille H, Wu T, et al. Colpali: Efficient document retrieval with vision language models[J]. arXiv preprint arXiv:2407.01449, 2024.

### Questions
1. Can you provide more detailed information about your proposed ‘high-quality’ dataset?

2. What’s your main contribution and improvement on building the multimodal RAG system? It seems that you are simply combining ColPali and GPT-4o.

### Soundness
2

### Presentation
1

### Contribution
1

### Rating
3

### Confidence
5