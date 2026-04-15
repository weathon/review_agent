# ReLiK: Retrieve, Read and LinK: Fast and Accurate Entity Linking and Relation Extraction on an Academic Budget

- Decision: Reject
- Scores: 6, 6, 5

## Abstract
Entity Linking (EL) and Relation Extraction (RE) are fundamental tasks in Natural Language Processing, serving as critical components in various applications such as Information Retrieval, Question Answering, and Knowledge Graph Construction. However, existing approaches often suffer from either a lack of flexibility, low-performance issues, or computational inefficiency. In this paper, we propose ReLiK, a Retriever-Reader architecture, where, given an input text, the Retriever module undertakes the identification of candidate entities or relations that could potentially appear within the text. Subsequently, the Reader module is tasked to discern the pertinent retrieved entities or relations and establish their alignment with the corresponding textual spans. Notably, we put forward an innovative input representation that incorporates the candidate entities or relations alongside the text, making it possible to link entities or extract relations in a single forward pass in contrast with previous Retriever-Reader-based methods, which necessitate a forward pass for each candidate. Our formulation of EL and RE achieves state-of-the-art performance in both in-domain and out-of-domain benchmarks while using academic budget training and with up to 40x inference speed with respect to other competitors. Finally, we propose a model for closed Information Extraction (cIE), i.e. EL + RE, which sets a new state of the art by employing a shared Reader that simultaneously extracts entities and relations.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes ReLiK, a retriever-reader model for entity linking and relation extraction. ReLiK encodes input text with retrieved candidate entities/relations and can link entities or extract relations in one pass. ReLiK achieves state-of-the-art results on multiple benchmarks while being faster, more parameter efficient, and trainable on a smaller budget than prior art.

### Strengths
1. The paper generally has a good presentation that clearly allows readers to understand what was done.
2. ReLiK establishes state-of-the-art results on benchmarks for entity linking and relation extraction. The joint model for closed IE is also insightful.
3. ReLiK is faster than prior state-of-the-art models, with gains of 10-40x reported on inference speed. This makes it much more usable in real applications.

### Weaknesses
1. Although ReLiK integrates entity linking and relation extraction together into one framework, the design for each module is relatively simple and similar to previous works.
2. As one of the emphases of this paper is the integration of EL and RE tasks. The mutual influence between EL and RE should be more clearly demonstrated in the experimental analysis section.

### Questions
1."Recent approaches only focus on at most two out of the three properties simultaneously." I don't quite understand this sentence. What are the "three properties" referring to？
2. I recognize the efficiency gains achieved by linking entities and extracting relations in just a single forward pass. Yet, I'm curious about what is the core design that enables the model to achieve state-of-the-art performance?I recognize the efficiency gains achieved by linking entities and extracting relations in just a single forward pass. Yet, I'm curious about what is the core design that enables the model to achieve state-of-the-art performance?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This article introduces Retriever & Reader pipeline to Entity Linking (EL) and Relationship Extraction (RE) tasks. ReLiK uses retriever instead of classifiers to discover entities and entity relationships in text, the reader module's role is to identify relevant entities or relations retrieved and align them with the corresponding textual spans. The experimental results show that the proposed methodology strikes a balance between effectiveness and efficiency.

### Strengths
The proposed method has several practical advantages. The reader greatly improve the efficiency of entity linking and relationship extraction.

Empirical evaluation thoroughly covers a substantial number of datasets.

### Weaknesses
The baseline systems used for comparison were not comprehensive enough in Section 4,  and it was recommended that more baseline systems be added for comparison[1], [2], [3].  

Since the contribution of this paper lies in the novel paradigm, the authors could have devoted a chapter to a brief overview of the developmental lineage of the relevant paradigm in order to describe more clearly the special features of this paper.

[1] Johannes M. van Hulst, Faegheh Hasibi, Koen Dercksen, Krisztian Balog, and Arjen P. de Vries. 2020. REL: An Entity Linker Standing on the Shoulders of Giants. In *Proceedings of SIGIR* 

[2] Nikolaos Kolitsas, Octavian-Eugen Ganea, and Thomas Hofmann. End-to-end neural entity linking. In *Proceedings of the 22nd Conference on Computational Natural Language Learning*

[3] Johannes Hoffart, Mohamed Amir Yosef, Ilaria Bordino, Hagen Fu ̈rstenau, Manfred Pinkal, Marc Spaniol, Bilyana Taneva, Stefan Thater, and Gerhard Weikum. Robust disambiguation of named entities in text. In *Proceedings of the EMNLP*

### Questions
The author mentioned that “ReLiK excels in this regard, surpassing previous systems in terms of performance, memory requirements, and speed”, but did not provide a quantitative comparison of memory requirements with other systems.

In the textual description of Section 3, the word "passage" is confusing - does it refer to the entities and relationships obtained by the retriever? The authors need further clarification.

Is there error propagation when the retriever fails to retrieve relationships and entities from top-k results?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes ReLiK, a new Retriever-Reader architecture for EL and/or RE. Given an input text, ReLiK allows to extract relations between entities given a reference knowledge base in a single forward pass. The proposed approach achieves state-of-the-art performance for the closed information extraction task (EL + RE) on standard datasets.

### Strengths
The proposed approach offers fast inference and state-of-the-art performance at a reasonably low budget, which is important for various settings. The paper is well-written and easy to follow. The adaptation of the Retriever-Reader paradigm to cIE is original and, to the the best of my knowledge, has not been proposed before.

### Weaknesses
The proposed approach is underpinned by access to external knowledge since ReLiK is given as input the text together with entities and relations from the KB. This impacts the performance and efficiency of the model and raises concerns about the fairness of the proposed benchmarks.

More specifically, the fact that ReLiK relies on the entities and relations from the KB already provides the model with the set of possible entities that can be extracted from the text, which can help for demarcating and disambiguating entities, and also for extracting relations.

The access to this non-parametric memory is also what enables to considerably lower the number of parameters, thereby offering faster inference time.

Also, the following recent prior work [1], which uses an end-to-end Reader-Retrieval approach for EL, should be cited in the paper. It would be interesting to see how both methods compare.


Minor comments:

- Xs and Xt are not defined in section 3.2, in the definition.
- The wrong template was used for submission (ICLR 2023)

[1] Bidirectional End-to-End Learning of Retriever-Reader Paradigm for Entity Linking, Li et al., arXiv:2306.12245, 2023

### Questions
What is the point of $<ST_{0}>$ ? It is not associated with any passage and we already have the [SEP] special token to dissociate between the text and the retrieved passages.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
