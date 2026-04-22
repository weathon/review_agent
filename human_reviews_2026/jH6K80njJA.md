# Structured RAG for Answering Aggregative Questions

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Retrieval-Augmented Generation (RAG) has become the dominant approach for answering questions over large corpora. However, current datasets and methods are highly focused on cases where only a small part of the corpus (usually a few paragraphs) is relevant per query, and fail to capture the rich world of aggregative queries. These require gathering information from a large set of documents and reasoning over them. To address this gap, we propose S-RAG, an approach specifically designed for such queries. At ingestion time, S-RAG constructs a structured representation of the corpus; at inference time, it translates natural-language queries into formal queries over said representation. To validate our approach and promote further research in this area, we introduce two new datasets of aggregative queries: HOTELS and WORLD CUP. Experiments with S-RAG on the newly introduced datasets, as well as on a public benchmark, demonstrate that it substantially outperforms both common RAG systems and long-context LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper extends the RAG design pattern for document retrieval-augmented question-answering by LLMs to address aggregate queries against information in a corpus of documents, each document of which is a description of a given type of entity. It accomplishes this by mining the corpus to generate a schema for a relational database that can represent the salient features of a given entity as a record, populating a database with the generated schema by extracting a record from a given document, and then translating a natural language aggregate query into a SQL query that can be executed against the generated database.

### Strengths
This is an innovative approach to solving a real issue; combining text-to-SQL capabilities with relational data extraction from an entity-centric corpus makes eminent sense. The author(s) contribute two new datasets specifically designed for evaluating aggregative query performance, filling an important gap in existing evaluation benchmarks that typically focus on queries answerable from answers extracted from the top-most relevant documents.

The approach demonstrates strong performance on synthetic data, suggesting the method's potential when its underlying assumptions are met. Additionally, the HYBRID-S-RAG variant shows promise by combining structured extraction for filtering with traditional RAG for final answer generation, achieving competitive performance on the full FinanceBench dataset and suggesting a practical deployment path for mixed query types.

### Weaknesses
As the author(s) acknowledge, the technique is currently limited to corpora of entity documents for a single entity type. The approach shows fragility in schema inference, with performance degrading dramatically between gold (i.e., human-curated) and inferred schemas.

The WORLD CUP dataset contains only twenty-two documents while HOTELS uses three hundred fifty synthetic documents, scales far below typical enterprise corpora.

The paper provides no timing comparisons or computational cost analysis for the ingestion phase, leaving open questions about the cost of schema generation and record extraction over large corpora. The aggregative queries tested are relatively simple, focusing on counts and averages rather than complex analytical queries involving multiple joins or nested aggregations that would better demonstrate the approach's capabilities.

### Questions
How does the system handle value extraction inconsistencies and normalization? Given that LLMs may extract the same value in different formats such as "1M" versus "1,000,000" versus "one million", what validation, normalization, or post-processing strategies could improve consistency? How robust is the SQL query generation to these variations, and what happens when numeric values are extracted with different units or scales?

What are the practical design constraints for corpus creation? Let's assume for the moment that there is a use case of building corpora that consist of documents each of which describes a given entity of a given type (which this reviewer believes there is). For practitioners wanting to apply this technique, what guidelines could be given for creating such a corpus? This includes considerations around minimum and maximum document length, required attribute coverage across documents, handling of optional versus required fields, and dealing with evolving schemas over time as business needs change.

How does the system handle incremental updates to the corpus? When new documents are added, must the entire schema be regenerated, or can it evolve incrementally? What happens to existing records if the schema changes to accommodate new attributes found in newer documents? How does the system maintain consistency between documents processed at different times?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces S-RAG, a framework for answering aggregative questions, i.e., queries that require collecting and reasoning over information from many documents. Unlike standard RAG systems that rely on vector retrieval of short passages, S-RAG transforms an unstructured corpus into a structured database at ingestion time. It does so by inducing a schema from sample documents and questions, then predicting structured records for each document via LLM prompting. At inference, user queries are translated into SQL queries over this database, optionally combined with standard RAG in a hybrid mode. To evaluate the proposed framework, authors introduce two aggregative QA datasets. Experiments compare S-RAG to baselines using the two datasets. Results show that S-RAG significantly outperforms these baselines, especially when a gold schema is available.

### Strengths
S1) The paper identifies aggregative queries as a distinct and practically important class of QA problems inadequately handled by existing RAG and multi-hop QA systems. Framing this as a structured reasoning challenge is conceptually fresh and well-motivated.

S2) The S-RAG architecture is clearly described, with distinct ingestion and inference phases. The schema-induction process and record standardization pipeline are intuitively presented and easy to follow.

S3) The two new datasets fill a gap by explicitly testing aggregative reasoning over unstructured corpora. Their release could catalyze future work in this space.

S4) Experiments are comprehensive. The quantitative and qualitative examples convincingly demonstrate the failure modes of standard RAG and the strengths of S-RAG in maintaining completeness of evidence.

### Weaknesses
O1) While the paper introduces a creative conceptual shift, the technical implementation largely relies on prompting existing LLMs for schema induction, record extraction, and SQL generation. There is minimal methodological innovation beyond careful prompt design. 

O2) The entire pipeline hinges on the reliability of LLM-generated schemas and records. These are prone to variability, omission, and inconsistency (as the authors themselves note). Without quantitative analysis of schema accuracy or inter-run variance, it is unclear how stable or reproducible the system is.

O3) The comparison uses different models across systems (e.g., GPT-4o for S-RAG, GPT-o3 for baselines) justified by differing reasoning loads. Although well-intentioned, this complicates claims of superiority—model differences, not the retrieval framework, may partly explain performance gaps. Similarly, since the HOTELS dataset is synthetic and generated with LLMs, data leakage or stylistic biases may favor schema-driven methods.

O4) The use of “LLM as a judge” metrics (Answer Comparison and Answer Recall) introduces subjectivity. The paper does not report agreement statistics or robustness analyses of the judge prompts. There is also no human evaluation to validate correctness, precision, or reasoning quality.

O5) The proposed method assumes that all documents share a single latent schema—a strong constraint that may not hold for most enterprise or web-scale corpora. The authors acknowledge this limitation but do not explore multi-schema or hierarchical settings, which would be essential for real-world applicability.

### Questions
Please address Weaknesses O2-O5.

### Soundness
3

### Presentation
3

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
This paper argues that previous RAG studies mainly focus on a much simpler scenario where the answer of the query exists in a small subset of the documents which can be retrieved. This paper studies *aggregative queries* which require retrieving a large set of evidence documents and then reasoning over them. The proposed method is quiet straightforward: prompting the LLM to generate the possible attributes, building the database on the extracted attribute-value pairs, converting the query into SQL for evidence document retrieval and prompting the LLM again for answering. The author also created two synthetic datasets for testing the proposed method.

### Strengths
1. This paper focuses on a practical question, aggregative queries, which are frequently encountered in daily life. General RAG methods struggle with this type of query. This study presents a timely approach to address this limitation.
2. The paper's presentation is clear and easy to follow.
3. The datasets are publicly available, which ensure reproducibility.

### Weaknesses
1. The *Hotels* dataset is created by prompting the LLM with hand-crafted attributes, so it is well structured to some degree and thus too easy for LLM to guess the attributes. This is different from what the author claims that this work is for unstructured corpus. 
2. The synthetic question-answer pairs haven't been verified by human, raising the concerns about the data quality.
3. The proposed method is too straightforward, and there are existing studies with similar methods:

Zhang, Wen, et al. "Trustuqa: A trustful framework for unified structured data question answering." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 24. 2025.

Pinto, David, et al. "Quasm: a system for question answering using semi-structured data." Proceedings of the 2nd ACM/IEEE-CS joint conference on Digital libraries. 2002.

### Questions
1. Have you manually verified the generated question-answer pairs of the two synthetic datasets?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper identifies a significant limitation in current RAG (Retrieval-Augmented Generation) systems when handling queries that require aggregating information from a large number of documents. It proposes S-RAG, a method that constructs structured representations during the data ingestion stage and transforms natural language queries into formal queries (e.g., SQL) during inference. The authors also contribute two datasets for aggregative queries—Hotels and World Cup—and demonstrate through experiments that S-RAG significantly outperforms traditional RAG systems and long-context LLMs.

### Strengths
Clear problem definition: The paper clearly identifies the limitations of current RAG systems in handling “aggregative queries,” an important real-world scenario.

Strong methodological innovation: S-RAG transforms unstructured documents into a structured database and leverages formal queries for reasoning—a novel and practical approach.

Valuable dataset contribution: The two proposed datasets fill an existing gap in RAG datasets regarding aggregative queries, providing valuable research resources.

Rigorous experimental design: Extensive comparisons are conducted across multiple datasets, including real-world systems (e.g., OpenAI Responses), yielding convincing results.

High practical relevance: The method is well-suited for enterprise private knowledge base scenarios, demonstrating strong potential for real-world deployment.

### Weaknesses
Strong methodological assumptions: S-RAG assumes that all documents share the same structure (i.e., a single entity type), which may not hold in real-world multi-entity document scenarios.

Limited generalization: The method’s performance on complex structures (e.g., nested attributes or list-type data) remains unverified, restricting its applicability to more complex corpora.

Small dataset scale: The Hotels and World Cup datasets are relatively small, and the method’s scalability in large-scale industrial settings remains to be validated.

### Questions
Can S-RAG be extended to corpora containing multiple entity types (e.g., documents containing both hotel and flight information)?

Could the authors further analyze the types of errors occurring during the record prediction phase and their impact on final answers?

Have the authors considered deeper comparisons with existing structured RAG methods based on knowledge graphs or table extraction?

### Soundness
2

### Presentation
2

### Contribution
2
