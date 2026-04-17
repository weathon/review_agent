# ChemisTRAG: Table-based Retrieval-Augmented Generation for Chemistry Question Answering

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Recent work has shown that retrieval-augmented generation (RAG) improves the performance of large language models (LLMs) for question answering on chemistry. However, existing chemistry RAG techniques are mainly based on text. It is challenging for the retriever to align the information about chemical entities between the query and the underlying corpora, especially if the naming and representation formats change. To address this problem, we propose ChemisTRAG, a RAG system in which information about chemical entities and reactions is stored explicitly as tables in the knowledge base (KB). Upon a query, ChemisTRAG first extracts chemical entities from the query and then selects relevant rows from the tabular KB. This way, the alignment processing is simplified and the accuracy is improved regardless of different naming conventions of compounds. To balance accurate answer retrieval for exact matches and robust reasoning for similar matches, we propose an adaptive reasoning process for the LLM: it first generates a reasoning prototype, then adapts the reasoning path to retrieval results, and finally infers the final answer contextualized on the example reasoning path. We have constructed a dataset of more than 38,000 compounds and 23,000 reactions from the recent five years of patents, and generated eight types of question-answering tasks to evaluate our system. Results show that ChemisTRAG consistently outperforms text-based RAG across all eight tasks, particularly in handling diverse chemical representations like SMILES and IUPAC.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces ChemisTRAG, a table-based retrieval-augmented generation (RAG) system designed to improve large language model (LLM) performance on chemistry question answering. Unlike prior text-based RAG methods, ChemisTRAG structures chemical knowledge as relational tables of compounds and reactions, supporting schema-aligned retrieval and adaptive reasoning. The authors also construct an up-to-date patent-based chemistry knowledge base and an accompanying benchmark dataset covering eight QA tasks. Experimental results show substantial improvements over text-based RAG across multiple LLMs and input representations (IUPAC and SMILES), suggesting the proposed framework enhances both retrieval precision and reasoning robustness.

### Strengths
The work is well-motivated and addresses a genuine limitation of text-based retrieval in chemistry—semantic mismatches caused by inconsistent naming conventions. The proposed tabular paradigm is intuitive and demonstrates meaningful gains across benchmarks. The paper’s data construction pipeline is transparent and its systematic evaluation across tasks, input types, and models provides comprehensive evidence of performance improvements. The inclusion of case studies showing both exact-match and out-of-corpus reasoning illustrates the framework’s practical benefits.

### Weaknesses
Despite its novelty, the work lacks theoretical or methodological depth beyond the structural organization of data and prompt engineering. The adaptive reasoning component is largely heuristic, without clear algorithmic grounding or ablation beyond basic variants. The evaluation benchmark relies heavily on synthetic, LLM-generated questions rather than human-curated or real-world chemical problems, potentially inflating reported performance. The paper also omits a rigorous error analysis, making it difficult to assess when and why the system fails. Finally, the scope of the claimed improvement (“generalizable framework for chemistry”) seems overstated given that the experiments are limited to relatively simple QA templates rather than true open-domain reasoning.

### Questions
1. The “adaptive reasoner” lacks formal specification—its “prototype–grounding–execution” process reads more like prompt engineering than a distinct algorithm. Quantitative justification for each stage is limited.
2. The retrieval evaluation is strong, but no clear comparison is made with domain-specific symbolic or graph-based retrievers, which are standard in cheminformatics.
3. The dataset heavily depends on LLM-paraphrased questions, introducing potential bias toward models with similar linguistic priors. Some human validation or leakage analysis would strengthen credibility.
4. Baselines are incomplete: recent chemistry-specific RAG or tool-augmented systems (e.g., ChemCrow, ChemAgent) are not directly benchmarked, weakening the comparative claim.
5. The discussion of limitations should be more explicit, including issues like LLM hallucination in synthesis prediction and the inability to handle multi-step reactions or mechanistic reasoning.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes ChemisTRAG, a structured retrieval-augmented generation framework for chemistry question answering. The authors argue that text-based RAG systems struggle in chemistry because molecule names and formats (IUPAC, SMILES, etc.) vary too much for reliable matching. Their fix is to store everything in tables instead of free text. The setup includes a knowledge base built from USPTO patents (boiled down to 38k compounds and 23k reactions), a retriever that turns natural-language questions into structured queries aligned with the table schema, and an “adaptive reasoner” that plans, grounds, and executes the final answer. They also release a benchmark of 4,800 QA pairs built from this same database. Experiments show large gains over text-based RAG, especially for SMILES inputs, and ablations suggest both the structured retriever and the reasoning module matter. Overall, it’s a cleanly engineered system showing that explicit tabular structure helps RAG work better in chemistry, though the setup feels quite controlled and domain-specific.

### Strengths
The paper’s main strength is its clear and well-engineered demonstration that structuring chemical knowledge into tables helps large language models handle the inconsistent naming and representation issues that often break text-based RAG systems. The design is neat and internally consistent — the knowledge base, retriever, and reasoner are all built to fit together, and the evaluation convincingly shows that this setup works much better than plain text retrieval when the data is well covered. The benchmark and data construction are also useful contributions on their own, giving a reproducible setup for chemistry QA. While the evaluation is somewhat closed-world, within that controlled scope the system is carefully implemented, the experiments are extensive, and the improvements over text RAG are large and consistent. Overall i think the paper’s strength lies in its solid engineering, clean framing of the structured retrieval problem and empirical validation in a chemistry setting.

### Weaknesses
The main weakness of the paper is that the benchmark and method are tightly coupled — the dataset is built directly from the same database the system retrieves from, so all entities are guaranteed to exist and appear in nearly identical form. This makes retrieval artificially easy and inflates performance, showing the system’s ability to match known strings rather than handle realistic, noisy, or missing information. The “adaptive reasoning” module also feels more like general prompt engineering than a novel algorithmic idea, and its contribution is not specific to chemistry or the tabular setup. Finally, the in-corpus vs. out-of-corpus experiment meant to show robustness under imperfect information is underdeveloped: correctness is ill-defined, the results barely differ from the non-RAG setup (table 2), and the section reads more as a late-added completeness check than as real evidence of reasoning with missing data.

### Questions
Since entities in the benchmark originate from your KB and are later paraphrased, how do you ensure that entity extraction and linking during evaluation are not trivially solved by lexical overlap? Was any normalization or perturbation (e.g., name variations, typos) introduced to test robustness?

How exactly do you determine that an answer is “correct”? Is correctness based on exact string match?

In Section 5.4, how do you operationally define “reasoning with imperfect information”? When the gold KB entry is removed, what does a correct response look like — a partial answer, an uncertainty statement, or a best guess? Could you clarify how you interpret near-identical performance between the two settings?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors propose ChemisTRAG, a Retrieval-Augmented Generation (RAG) system designed to address the unique challenges of question answering in the chemistry domain1. The core problem, as identified by the authors, is that standard text-based RAG systems fail to align queries with knowledge corpora when chemical entities are represented in diverse and non-linguistic formats, such as IUPAC names and SMILES strings. To solve this, the proposed system has three main components: a Tabular Knowledge Base (KB), a Table-Based Retriever, and an Adaptive Reasoner. To evaluate the system, the authors also create a new benchmark dataset of 4,800 QA pairs derived from their KB, covering eight distinct task types.

### Strengths
- The paper targets a clear and important weakness of general-purpose RAG systems. The difficulty of aligning symbolic or non-standard identifiers (like SMILES) with text is a real-world bottleneck in many scientific and technical domains, and the authors' solution of using a structured, table-based KB is a highly pragmatic and effective approach.

- The authors contribute two valuable, up-to-date resources: the tabular KB of 38k+ compounds and 23k+ reactions from recent patents (2020-2025) and a new 4,800-pair benchmark dataset for chemistry QA. These resources are a solid contribution to the community, independent of the method itself.

### Weaknesses
- The components of ChemisTRAG are practical engineering, but they are not new methods.
- The entire data pipeline (KB extraction and benchmark generation) is automated using LLMs (Qwen-3-8B and GPT-OSS-20B). The "validation" step for the KB also uses an LLM. This introduces a significant risk of systemic error, hallucination, and self-reinforcing biases. The paper provides no statistics on human-in-the-loop verification or expert validation of the final KB and benchmark. Without this, the "ground truth" of the dataset is questionable.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents ChemisTRAG, a table-based retrieval-augmented generation (RAG) framework designed for chemical reaction and compound question answering. Instead of relying on unstructured text retrieval, the authors construct a structured knowledge base containing ~23K curated reaction records and ~38K unique compounds extracted from recent patent and literature sources. Natural language queries are parsed into structured tuples that specify entity type and query intent, allowing retrieval to be performed over relational tables rather than documents. Retrieved evidence is then combined with an adaptive reasoning module to generate the final answer. Experiments show that ChemisTRAG achieves higher accuracy and robustness than text-based RAG baselines and domain LLMs. However, the experiments are limited to the dataset generated in this paper, without evaluation on other chemistry datasets. The generated evaluation dataset is constructed from the knowledge base in this paper, raising concerns about generalization. The generated dataset also misses some critical chemistry tasks, including yield prediction, chemistry calculation, and general academic chemistry problems.

### Strengths
- The authors propose a table-based retriever for retriever from the constructed knowledge base, and also take synonym problem into consideration
- The experiments on the retrieval are comprehensive
- The paper is well written and easy to follow

### Weaknesses
- The evaluation data is constructed from the KB in this paper, raising concerns about generalization.
- The proposed method is only evaluated on the data created in this paper. There are lots of public chemistry datasets[2][3][4][5], but the authors choose not to use any of them. This decreases the credibility of this paper.
- The proposed adaptive reasoner is not very novel, many agents use similar strategies, e.g. ChemAgent [1].
- For compound-focused queries, the proposed method simply uses ROUGE-L similarity. This may not be good enough since there are lots of chemistry specific encoders, including ChemBERTa, MolT5, and MolBERT. These encoders may provide better performance.
- Given how the knowledge base is constructed, I worry that the generalization is not so good. The proposed evaluation dataset misses some critical tasks existed in other works, including yield prediction, general college chemistry questions, and chemistry calculation problems.


[1] Tang et al. ChemAgent: Self-updating Library in Large Language Models Improves Chemical Reasoning. ICLR 2025.

[2] Wang et al. SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models. ICML 2024.

[3] Fang et al. Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models. ICLR 2024.

[4] Mirza et al. Are large language models superhuman chemists? Nature Chemistry 17, 984–985 (2025).

[5] Rein et al. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. COLM 2024.

### Questions
- How do you check the chemical feasibility of extracted reaction with GPT-OSS-20B? What are you validating here? Wouldn't using Molecular Transformer more reasonable here?
- What retriever do you use for TextRAG?
- What are the formats of the constructed questions? Are they multiple-choice questions? If not, what metrics do you use for each task?

### Soundness
2

### Presentation
3

### Contribution
2
