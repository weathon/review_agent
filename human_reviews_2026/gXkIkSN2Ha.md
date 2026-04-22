# Text-to-SQL Benchmarks for Enterprise Realities: Under Massive Scopes, Complex Schemas and Scattered Knowledge

- Avg Score: 3.60
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2, 4, 4

## Abstract
Existing Text-to-SQL benchmarks remain overly idealized and differ substantially from enterprise scenarios, which require retrieving tables from massive query scopes, interpreting complex schemas, and locating scattered knowledge across large collections of documents. To address these gaps, we present two enterprise benchmarks, BIRD-Ent and Spider-Ent, constructed through a cost-effective refinement framework applied to their academic counterparts (BIRD and Spider), together with a new task paradigm, Dual-Retrieval-Augmented-Generation (DRAG) Text-to-SQL, which formalizes the dual-retrieval workflow of table schemas and knowledge documents prior to SQL generation. Our benchmarks exhibit three defining characteristics of enterprise settings: massive query scopes with over 4,000 columns, complex schemas with domain-specific and heavily abbreviated table and column names, and scattered knowledge distributed across enterprise-style documents totaling 1.5M tokens. These properties make the benchmarks substantially more realistic and challenging than existing ones. Evaluation on several state-of-the-art large language models (LLMs) reveals a sharp performance drop, with only 39.1 EX on BIRD-Ent and 60.5 EX on Spider-Ent, underscoring the gap between academic performance and enterprise requirements. By providing a rigorous and discriminative testbed under the DRAG Text-to-SQL paradigm, our benchmarks offer a valuable resource to advance research toward Text-to-SQL systems that are reliable and deployable in real-world enterprise environments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper targets the gap between academic Text-to-SQL benchmarks and real-world enterprise scenarios, which require handling massive query scopes, complex schemas, and scattered knowledge across large document collections. 
The authors propose a cost-effective refinement framework that transforms existing benchmarks (BIRD and Spider) into enterprise-grade benchmarks (BIRD-Ent and Spider-Ent) through domain, schema and knowledge refinement, alongside introducing the DRAG Text-to-SQL paradigm that explicitly models retrieval before generation. Evaluation on state-of-the-art LLMs reveals substantial performance drops to 39.1% on BIRD-Ent and 60.5% on Spider-Ent, demonstrating that current systems struggle with enterprise-level complexity.

### Strengths
1. The paper targets an important and well-motivated problem that there is still a critical gap between academic Text-to-SQL benchmarks and enterprise requirements across dimensions such as scope, schema and required knowledge.

2. The benchmark refinement idea provides a scalable approach to creating realistic benchmarks without expensive enterprise data collection, addressing privacy and annotation cost barriers.

### Weaknesses
1. Vague enterprise challenge description without real-world validation. The paper claims to capture "enterprise realities" but provides no analysis or validation of what real enterprise data actually looks like. While Figure 4 shows one anonymized example, there is no quantitative comparison of characteristics (e.g., schema complexity metrics, knowledge distribution patterns, naming convention statistics) between real enterprise databases and the synthetic benchmarks. Without such validation, it remains unclear whether the identified challenges (massive scope, complex schemas, scattered knowledge) accurately reflect the most critical pain points in real deployments versus BEAVER, which use actual enterprise data.


2. The refinement framework makes several assumptions that also lack empirical grounding: (1) Domain expansion uses LLMs to generate synthetic tables at scale, but there is no evidence these tables exhibit realistic statistical properties or semantic relationships found in real enterprise databases; (2) Schema-level refinement assumes enterprises follow clean hierarchical naming conventions, but real enterprise schemas are often messier with inconsistent conventions, legacy naming, and organic evolution; (3) Knowledge-level refinement assumes well-documented, enterprise-style documents, but real enterprises often have incomplete, outdated, or poorly maintained documentation. These design choices may create artificial rather than authentic enterprise challenges.


3 Insufficient comparison with existing enterprise benchmarks. Table 1 claims the benchmarks achieve higher "Enterprise Realism" than BEAVER and Spider 2.0, but this rating is subjective without supporting analysis. The paper does not provide: (1) direct performance comparison on the same models between Ent-series and BEAVER/Spider 2.0 to assess relative difficulty; (2) quantitative comparison of dataset characteristics (beyond simple statistics like table/column counts) to justify why synthetic data better captures enterprise complexity than real data; (3) analysis of whether the challenges introduced (retrieval difficulty, schema noise, knowledge scattering) align with where real systems fail in practice. BEAVER explicitly uses real enterprise data and highlights similar challenges, the paper needs stronger justification for why synthetic refinement is superior or complementary.

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces BIRD-Ent and Spider-Ent, two benchmarks simulating enterprise Text-to-SQL by refining existing benchmarks. They expand query scopes to 4,000+ columns (55-165× larger), add enterprise-style schemas, and scatter knowledge across 1.5M tokens of documents. Models must retrieve tables and knowledge before generating SQL. SOTA LLMs drop from 77.5%→39.1% on BIRD and 91.2%→60.5% on Spider, revealing that current models aren't enterprise-ready.

### Strengths
1. Important problem: Enterprise Text-to-SQL is different from academic settings, and we need benchmarks that reflect this. The paper makes this gap concrete with clear evidence.
2 .Interesting generation approach: Using LLMs to refine academic benchmarks is practical and clever. The document generation pipeline (segmenting knowledge, choosing enterprise genres like RFCs/FAQs, embedding with tags, then self-correction) is well-designed. Creating 1,412 realistic documents at 1.5M tokens would be prohibitive manually.
3. Clear empirical findings: The performance drops are large and consistent across models. The error analysis shows retrieval is the main bottleneck (knowledge retrieval only 14.9% perfect recall). Even with perfect retrieval, models only hit 51.8%, showing the problem isn't just retrieval.

### Weaknesses
1. Cost claims don't match reality: The paper claims "very low cost" and "minimal human intervention" but Spider-Ent needed 100% manual review with 40-60% rejection. BIRD-Ent needed 10% double annotation. This is substantial work. 

2. DRAG isn't novel: BEAVER already does table retrieval. Adding knowledge retrieval is the obvious next step, not a paradigm shift. The real contribution is the benchmark scale and empirical findings, not inventing dual retrieval. The framing oversells this.

### Questions
1. What are the actual costs? Person-hours for annotation? LLM API costs? How does this compare to building BIRD/Spider from scratch? Numbers would help justify the cost-effectiveness claim.

2. Why did Spider-Ent have such high rejection? 40-60% failure rate suggests the refinement approach has limits. Is this a problem with Spider's underspecification or your method?

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
5

### Summary
This paper introduces new Text-to-SQL benchmarks, BIRD-Ent and Spider-Ent, designed to better reflect enterprise realities. These benchmarks expand the schema size to thousands of columns, inject naming noise, and simulate document-based external knowledge retrieval. They are built by refining existing benchmarks (BIRD and Spider) using LLMs to generate additional tables, schema variations, and embed relevant knowledge within lengthy document collections. The paper also defines the DRAG (Dual-Retrieval-Augmented-Generation) formulation, where both table schemas and knowledge documents must be retrieved before generating SQL. Evaluation shows that many LLMs perform worse on these new benchmarks.

### Strengths
1. The paper addresses an important problem: building Text-to-SQL benchmarks that more accurately reflect realistic, enterprise-style scenarios.

2. It provides a detailed error analysis, categorizing failure modes such as schema retrieval, knowledge grounding, and SQL construction, which helps identify where current models fall short.

3. The benchmark is evaluated across a diverse set of models, including both proprietary (e.g., GPT-4o) and open-source (e.g., Qwen3, DeepSeek) LLMs of varying sizes, offering a comprehensive view of model performance.

### Weaknesses
1. Concerns about quality control. The benchmark relies heavily on LLM-generated artifacts, including additional tables, document synthesis, and schema perturbations. In such a setting, quality control is crucial. However, the current procedure appears limited: only 10% of the data is manually inspected with double annotation, and details on the audit process are sparse. It remains unclear how key properties are verified. For example, how “answer uniqueness” is ensured when LLMs might generate semantically similar or redundant tables, or how the “enterprise-style” nature of generated documents and column names is assessed. Without more transparency and rigor, the reliability of the benchmark is difficult to evaluate.
2. Vague definitions of enterprise characteristics: Several key terms, such as “enterprise-style schemas” or “enterprise knowledge documents”, are described in vague and informal ways. The paper lacks a clear and systematic abstraction of what constitutes an enterprise setting. For instance, the authors mention converting original table names into hierarchical formats like "project area content", but this design choice appears ad hoc. Moreover, the paper narrowly focuses on retrieval challenges, whereas enterprise settings may also involve more complex user questions that require sophisticated reasoning and compositional SQL generation, an aspect that is underexplored here.
3. Limited evaluation scope: The experimental evaluation is limited to vanilla LLMs, without considering existing SOTA Text-to-SQL systems that already integrate retrieval capabilities. Many recent methods, such as CHASE-SQL and AskData (https://arxiv.org/abs/2505.19988), incorporate schema and knowledge retrieval as part of their pipelines. More recent RL-based methods (e.g., arXiv:2503.23157, arXiv:2503.00223) even bake retrieval and schema linking directly into the model. It remains unclear whether these more capable models also fail on this benchmark. For example, if an LLM is paired with a keyword search tool, would it still struggle under the proposed setting? Without evaluating such baselines, the benchmark’s difficulty and relevance remain only partially demonstrated.

### Questions
>Each asset is then enlarged with LLM-generated tables designed in enterprise schema style, including realistic names, types, and constraints, while ensuring the question-SQL pairs in the original benchmarks remain valid.

What defines “realistic” names, types, and constraints in this context? What criteria are used to validate the quality of LLM-generated tables? Also, if an LLM is equipped with keyword search, would it still struggle to retrieve relevant content, or could it easily reduce the search scope?


>answer uniqueness, checking whether a question admits only one semantically valid SQL regard- less of syntactic form, as domain-level refinements may introduce semantically similar columns;

How is answer uniqueness verified, especially when schema augmentation introduces similar or redundant columns? Is there an exhaustive check across all tables?

> Unlike BIRD and Spider 2.0-snow/lite, which directly provide a document including a small amount of dense external knowledge as part of the model input.

This seems to understate the difficulty of Spider 2.0, which includes dialect documentation and code. These are not trivial to process and also require retrieval. Could you clarify why these are considered “small” or less challenging compared to your setting?

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
3

### Summary
The paper introduces enterprise-oriented versions of Text-to-SQL datasets (BIRD-Ent and Spider-Ent) to better reflect Text-to-SQL retrieval and generation challenges under massive schemas, tables, and knowledge documents. The dataset construction approach consists of three stages: (i) domain-level refinement, where schemas and tables from similar domains are grouped together, (ii) schema-level refinement, where schema naming conventions are modified to follow enterprise standards, and (iii) knowledge-level refinement, where an LLM augments knowledge sources by generating related documents. Experimental results across different LLMs demonstrate that BIRD-Ent and Spider-Ent datasets are more challenging than the original datasets under different LLMs.

### Strengths
- S1) Experimental results demonstrate that Ent-versions of BIRD and Spider datasets present greater challenges across multiple LLMs (Table 2). Notably, LLMs continue to underperform even when provided with oracle inputs (Table 5).
- S2) LLM-based schema-level refinement enables hierarchical decomposition of schemas into structured components (<project> <area> <content>), facilitating systematic merging to construct larger, more complex schemas.
- S3) The paper implements quality control mechanisms that leverage manual human annotation to assess benchmark quality and ensure question-answer correctness is maintained.

### Weaknesses
- W1) The paper claims that scaled-up datasets reflect enterprise realities, but this assertion lacks external validation from SQL experts or enterprise practitioners. For example, Table 1's claim that Spider-Ent demonstrates superior "Enterprise Realism" compared to BEAVER is not systematically verified: Appendix A.1 provides a supporting example, but this represents anecdotal evidence rather than systematic real-world verification.
- W2) The dataset construction generates synthetic tables to expand database size, but relies heavily on LLM parametric knowledge for table creation (Figure 14). This approach raises concerns about applicability to realistic scenarios, such as generating tables with customer transaction data that are not part of LLMs' training data. As a result, it is unclear whether the generated tables actually enhance benchmark quality or simply introduce noise, artificially increasing complexity.
- W3) Knowledge-level refinement using DeepSeek-generated documents appears to function as an adversarial approach to degrade retrieval performance by creating numerous documents with highly similar embeddings. The authors should present an analysis examining how embedding similarity distributions shift with their synthetic document generation and  and whether this creates retrieval noise.

### Questions
- Q1) Could the LLMs fail in the oracle setting due to the long input of the generated schemas/tables and the redundant noise introduced by the synthetic database expansion?

- Q2) How do you ensure that the synthetically generated tables and documents do not simply introduce noise to the dataset? Could we use similar synthetic data generation to improve the database quality instead of introducing noise to it?

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
The paper proposes a new nl2sql benchmark by extending bird and spider benchmarks. It does so by synthetically augmenting/modifying the data to imitate more complex interprise setting. The paper shows the constructed datasets have many more tables/need for external knowledge, and that the accuracy of existing methods drop when run on the refined benchmark compared with the original benchmark data.

### Strengths
- New benchmarks are always good, especially if they mimic real-world settings

- The method for refining the data using LLMs is interesting

### Weaknesses
- There are two major issues I have with the paper. First, it is unclear how realistic the benchmark is. The refinement steps are motivated by "In enterprise settings, .....". Where is this information coming from about enterprise setting? Why is it representative of all enterprise use-cases? The main metrics shown at the end are number of table/columns/knowledge tokens. However, the data distribution/content is important and not only the size. 

- Second, the data quality is unclear. There is significant issues with existing text-2-sql benchmark ground-truth values, and the paper needs to do its due diligence to ensure the ground-truth answers are correct. Right now, the paper is very handwavy about the process of obtaining ground-truth values. They inspected 10% of the samples, which isn't enough, and even for it is unclear how. The paper mentions "with double annotation and expert arbitration". Who are the annotators? who are the experts? What was the annotation process?

### Questions
Please respond to the two points raise above.

### Soundness
2

### Presentation
3

### Contribution
2
