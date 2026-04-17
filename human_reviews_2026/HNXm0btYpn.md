# SearchFireSafety: A Retrieval-Augmented Legal QA Dataset for Fire Safety

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 6

## Abstract
Retrieval-augmented generation (RAG) promises to bridge complex legal statutes and public understanding, yet hallucination remains a critical barrier in real-world use. Because statutes evolve and provisions frequently cross-reference, maintaining *temporal currency* and *citation awareness* is essential, favoring up-to-date sources over static parametric memory.
To study these issues, we focus on the under-examined domain of South Korean fire safety regulation—a complex web of fragmented legislation, dense cross-references, and vague decrees. We introduce **SearchFireSafety**, the first RAG-oriented question-answering (QA) resource for this domain. It includes: (i) 941 real-world, open-ended QA pairs from public inquiries (2023–2025); (ii) a corpus of 4,437 legal documents from 117 statutes with a citation graph; and (iii) synthetic single-hop (Yes/No) and multi-hop (MCQA) benchmarks targeting legal reasoning and uncertainty.

Experiments with five Korean-capable LLMs show that: (1) multilingual dense retrievers excel due to the domain's mix of Korean, English loanwords, and Sino-Korean terms (i.e., Chinese characters); (2) grounding LLMs with **SearchFireSafety** substantially improves factual accuracy; but (3) multi-hop reasoning still fails to resolve conflicting provisions or recognize informational gaps. Additionally, we find that (4) domain adaptation via continued pre-training improves accuracy but significantly degrades uncertainty awareness when evidence in insufficient. Our results affirm that RAG is necessary but not yet sufficient for legal QA, and we offer **SearchFireSafety** as a rigorous testbed to drive progress in Legal AI.
All data resources are available at: https://anonymous.4open.science/r/SearchFireSafety-C2AB/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introdcues three datasets around South Korean fire safety regulations: one real-world dataset scraped from actual queries, and two synthetic ones (once MCQA with multihop across two documents, once binary with one document). The authors show across many experiments that these tasks are hard for models, and that legal retrieval is still challenging.

### Strengths
- There's a lot to be liked about this paper. I am deeply interested in legal retreival, and would instantly believe all the findings the authors made.
- The real-world dataset is very interesting, and I think that's a useful resource.

### Weaknesses
- There's an incoherent narrative and a bit too much cramped into one paper: Is the main contribution the three datasets together, the retrieval part, the generation part, the hallucination part? Because of that, the individual contributions seem to be a bit shaky and not as coherent (this is the main reason for my clarity score, the paper was easy to read and well written.)
- Why don't the authors use the synthetic datasets for training? This would be the big contribution: Synthetic data helps real-world legal retrieval or generation (for generation, the datasets probably would have to be constructed differently). If it doesn't help for training, why is that, why doesn't synthetic data work? What would work instead?
- Perhaps I'm overcritical of the synthetic datasets, but as is, only used to evaluate models, I don't see their utility and think the paper would be stronger without them.
- Related to this, the authors seem to have done lots of manual annotation work which is highly appreciated. While they give qualitative examples of why certain things are difficult for models, an extensive manual error analysis would be greatly appreciated, and could inform future research on legal retrieval and highlight what currently doesn't work and why.
- Having said all of that, again I think there's a lot here. I wouldn't mind seeing this paper accepted, but I would be a bit sad because right now it seems to be a somewhat incoherent contribution, and easily could be a very compelling contribution with not that much more work.

### Questions
- line 45:  "crucial yet under-examined domain" -- I think this is overstating things slightly. How about "important yet ..."
- Line 142: Perhaps cite https://dl.acm.org/doi/pdf/10.1145/3709025.3712219 here? They discuss similar topics
- Would be curious to see how Qwen3 embeddings and rerankers perform on this dataset: https://huggingface.co/Qwen/Qwen3-Embedding-8B
- section 5.2.2 and 5.2.1 seem to be somewhat contradictory. If there's only limited lexical overlap between documents and answers, how can ROUGE scores improve in a RAG setting? Would like to see more details on this
- Table 6 vs Table 8: This also seems curious: table 8 metrics are way higher. Now, I wouldn't expect the tasks to be way harder given full context. It reminded me of https://arxiv.org/abs/2505.12864 though, where the authors added multiple choice options and model performance went down a lot, perhaps try something similar here as well?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes a new benchmark, SEARCHFIRESAFETY, for the South Korean fire safety regulation domain, designed to evaluate the RAG capabilities of models. The dataset includes 941 real-world, open-ended QA pairs and over 4,000 synthetic QA pairs. The paper also provides a corpus of 4,437 legal documents from 117 statutes, complete with a citation graph, to serve as the source for RAG. Experiments demonstrate that multilingual dense retrieval performs best on this benchmark, but LLMs with RAG still struggle with multi-hop legal QAs.

### Strengths
1. The paper addresses the RAG problem within the South Korean fire safety regulation domain, which is a novel and underexplored area.

2. The authors conduct comprehensive experiments, comparing a wide variety of retrieval methods and the capabilities of several different LLMs.

### Weaknesses
1. My major concern is the dataset's quality, which appears highly dependent on the authors, who also served as the annotators. The paper lacks a sufficient qualitative or quantitative analysis of the dataset. Simply stating that the data was human-reviewed is not enough; a more rigorous methodology is needed to validate the dataset's quality. Specifically:

* Real-World Open-Ended QA: The annotation procedure does not seem reliable, making the benchmark's quality questionable. As stated in lines 137-140, the dataset was annotated by the authors. However, it is not specified whether the authors possess the necessary legal expertise to annotate such questions. Furthermore, the Inter-Annotator Agreement (IAA) is missing, which casts doubt on the reliability and consistency of the annotations. Additionally, it appears the annotators only validated the documents retrieved by BM25 rather than identifying all possible relevant documents. Given that BM25 cannot guarantee 100% recall, the final dataset likely suffers from missing relevant documents (low recall).

* Legal Document Corpus: Coverage is a significant issue. According to lines 211-212, the corpus was collected by crawling only those sources cited at least twice in the NFA answer set and their parent instruments. Any document not meeting this criterion is excluded. A legal document dataset should aim for comprehensive coverage. Limiting the corpus only to documents relevant to the NFA answer set greatly diminishes its usability and generalizability.

* Synthetic QA Dataset: The generation method is ambiguously described. The authors state they "use the citation graph," but this is too vague. More details are necessary. Besides, the quality of the final synthetic dataset is unknown, apart from the authors' claim to have "checked them."

2. The paper's focus is excessively narrow. While South Korean fire safety regulation is a valuable domain, the paper would be much stronger if it included at least one other domain. The described construction pipeline for both the QA dataset and the legal corpus appears to be highly tailored to this specific domain, which raises concerns about the generalizability of the proposed methods.

3. The second stated challenge of RAG in the legal domain is not clearly articulated. In lines 070-073, the authors define this challenge as: "legal documents are interconnected through a dense web of statutory cross-references... many of which are vague or overly broad." It is not immediately clear why this interconnectedness and vagueness inherently make RAG significantly harder. The paper needs to elaborate on this point to make the challenge concrete.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

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
This paper targets the problem of a lack of a QA dataset for the domain of South Korean fire safety regulation. It proposes a dataset of real-world and synthetic question-answers with 941 real-world open-ended QA and 4437 legal documents. The legal document dataset was constructed with a web crawling process, a human-in-the-loop process for verifying legal documents, and a citation graph construction representing the relevant between documents. The experiments evaluate the quality of the RAG-based model built on the proposed dataset in both the retrieval and generation phases. For retrieval, Recall@k is the metric used for evaluation. It shows that weighted Reciprocal Rank Fusion (wRRF) outperformed other techniques in this domain over the real-world open-ended QA. For generations, the ROUGE and BERTScore were used as the textual matching score, while LLM-AS-A-Judge was used with the integration of a strong close LLM for judgment, along with the Win-Rate score. Accuracy shows that the best context retrieved for answer generation for each question is the Gold context (i.e., the corresponding legal document that contained the question). Authors use the synthetic dataset to evaluate the performance of the RAG approach over different LLMs. It shows that, interestingly, with Full Context, an open LLM with 8B parameters can outperform the well-known GPT-4o model in getting more correct answers.

### Strengths
- The dataset construction process (both questions and legal documents) was performed on reliable resources.
- While collecting real real-world dataset of questions requires significant cost for humans as experts to evaluation questions, the second strategy for question creation, which focuses on yes/no questions and multiple-choice questions, requires significantly less cost and is applicable to other domains.

### Weaknesses
- Overall, while this work highlights important observations about the roles of the retrieval process in achieving good accuracy, it lacks contributions regarding the improvement of RAG models.
- In the introduction, the authors claimed that the legal framework governing fire safety in South Korea is complex and fragmented, which doesn’t convince me. Authors should provide clarifications about the reasons for this challenge. Additionally, I don’t see any significant challenges in building a RAG-based dataset for the Korean language compared to other languages.
- A minor point, figures should be translated into English to ensure they are understandable to reviewers.

### Questions
- The metric used for Generation can be extended if authors use different models (i.e., Claude, Gemini), replacing GPT-4o as LLM-As-a-judge.
- How many experts are assigned to validate a given question/ legal document? How do you solve the conflict between experts’ opinions?
- What are the distinctions between constructing the South Korean Fire Safety dataset compared to other language datasets?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper develops a retrieval-augmented legal question answering dataset on South Korean fire safety regulation. It contains 941 real-world QA pairs and also synthetic a MCQA corpus targeted at evaluating LLM hallucination on legal reasoning.

### Strengths
I think this paper presents a solid, clear and verifiable evaluation of legal RAG. It is great that it is in QA format, which allows robust ground truth, and even using LLM-as-a-judge, we can see there's a high correlation with existing reference-based metrics. There are also plenty of ablations/strategies of your method. The evaluation feels intuitive and objective. I would imagine criticisms at the niche domain of this paper, but I think that's less of a concern as this is a neat dataset which can measure general legal reasoning abilities. I also like that the paper also constructs a synthetic QA corpus to evaluate LLM hallucination with partial context, which is very thoughtful.

### Weaknesses
The main problem I have is that for evaluating open-ended QA, even though I think it makes sense that reference-based metrics might already be robust to reflect the answers to open-ended questions (e.g. from https://arxiv.org/abs/2305.12421) It still might be useful to show some qualitative examples and make a more convincing case that your open-ended eval is reliable.

### Questions
What do you think can help LLMs improve on this benchmark? Does scaling help (and if so, what kind of data) or is there something more that should be involved? This is irrelevant of my rating but just curious about how the authors think and totally optional to answer.

### Soundness
3

### Presentation
3

### Contribution
3
