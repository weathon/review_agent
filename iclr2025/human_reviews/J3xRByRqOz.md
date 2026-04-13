## Human Reviewer 1

### Summary
This article proposes a novel pipeline for open domain question answering based on RAG, that first estimates sources reliability score before performing majority vote for answer selection. Authors built a benchmark by sampling sources reliability and generating set of documents for each sources and answers which are true are false depending on the sampled reliability score. Then, they evaluated their approach compared to several baselines, such as naive RAG, simple majority voting or answers from the LLM parametric memory. The approach outperforms baselines.

### Strengths
The approach is well motivated, leveraging and solving several prior works limitations.

The proposed evaluation benchmark is an important contribution that could benefit the community, 

The method outperforms baselines.

### Weaknesses
Relies on many engineering blocks, with no real scientific contribution. 

Reliability is evaluated at the source level. The issue is that some sources such as Wikipedia or social media have huge variability is source quality, and the current approach cannot take this into account. 

I don’t get how prompting alone can solve the answer format alignment problem (section 3.1, not 3.2). Name could be written F. Lastname, Firstname Lastname, Lastname Firstname, etc… For real deployment of the proposition, it could be a major limitation as the pipeline needs aligned answers for efficient voting. 

For niche or specialized topics, the approach will give lesser weight to highly specialized sources, that will probably provide more precise answers, therefore different answers than the majority, even if the retriever computed a very high score for the document. 

The generated benchmark does not completely cover real life scenarios, such as source format variability (social media posts are much shorter than news article). This impact of source format is not evaluated here, and this could lead to high impact in open QA regime.

Potentially unfair comparison to baselines (see questions)

### Questions
What is the pipeline sensitivity to the Prompt and the f_{align} threshold (0.9) ?

I am not sure to understand how keeping the k RRSS is relevant to decrease computational cost as one need to infer the LLM answer to verify if it is not IDK, therefore requiring no additional computational power to compute the majority vote. This is more of a reliability filter than an efficiency wise operation. 

Doesn’t Exact Match benefit methods that use a prompt leading to short answers? Naive RAG on the opposite might contain long answer with LLM verbose presentation of the answer? 

It was not clear if a specific prompt was used for the naive RAG approach

### Soundness
3

### Presentation
4

### Contribution
1

### Rating
5

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper introduces Reliability-Aware Retrieval-Augmented Generation (RA-RAG), a approach to address misinformation challenges in multi-source retrieval. RA-RAG operates in two stages: first estimating source reliability through an iterative process without labeled data, then efficiently retrieving and aggregating information from reliable sources using weighted majority voting. The authors developed a realistic multi-source benchmark reflecting real-world complexity, unlike previous benchmarks using artificially injected misinformation. Experimental results show RA-RAG consistently outperforms baselines when handling conflicting and unreliable information, demonstrating its robustness and effectiveness.

### Strengths
The research topic is novel and timely. In multi-source RAG systems, evaluating the reliability of information from different sources during retrieval, identifying heterogeneity across sources, mitigating conflicts and redundancy, and achieving complementarity are significant research challenges that have received limited attention from the RAG community.

### Weaknesses
1. The motivation, particularly regarding the propagation of misinformation, is inadequately explained. The paper fails to clearly illustrate how misinformation propagates and accumulates, and how this manifests in heterogeneous multi-source retrieval systems.

2. The research lacks focus. While the paper should primarily address reliability assessment in multi-source retrieval scenarios, it digresses into discussing a lot about response misalignment (where models generate answers based on internal knowledge rather than retrieved documents). This represents a separate research direction in RAG concerning conflicts between LLM's internal and external knowledge, and should not be a major focus of this work.

3. The methodology is  simplistic. The weighted majority voting approach, while classical, relies primarily on statistical scores and assumption-based document filtering. The paper lacks in-depth analysis of the feature of different sources, failing to adequately address the heterogeneity among them - a limitation the authors themselves identified in related work. Furthermore, there's no mechanism for refining initial voting results based on actual response patterns.

4. The use of ROUGE-1 to evaluate the reliability of estimated answer y~ in retrieved documents, and subsequent filtering of irrelevant sources based on these scores, is questionable. This approach might perpetuate errors in the RAG system due to incorrect hypothetical generations. Moreover, it appears suitable only for extractive questions  where answers can be directly found in document blocks due to the principle of ROUGE, not for tasks requiring information synthesis. This limitation is evident in the chosen datasets (HotpotQA, NQ, TriviaQA), which primarily focus on fact QA.

### Questions
1. The paper needs better adherence to writing conventions, particularly in maintaining consistency in capitalization for acronyms.

2. Many sentences are ambiguous and difficult to comprehend, requiring clearer expression. e.g. The abstract contains confusing statements, particularly regarding "true answers" and "no labelling." For instance, the sentence "Specifically, it iteratively estimates source reliability and true answers for a set of queries with no labelling" is unclear. Does "no labelling" refer to the absence of ground truth answers? This ambiguity creates confusion for first-time readers.

3. Although the authors mentioned in the introduction that using LLMs to evaluate the reliability of documents based on their internal knowledge would be less effective when external knowledge needs to be consulted (the second paragraph in the Introduction), in this paper, when designing hypothetical answers as the basis for reliability judgment, the same problem will also be faced. The retrieved content may not necessarily contain the correct information.

4. How to ensure that the role of the language model in the RA - RAG framework is based on reliable retrieved information rather than overly relying on its internal knowledge, especially when dealing with complex or ambiguous queries? The generalizability of the method needs to be verified.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
5

---

## Human Reviewer 3

### Summary
This paper proposes Reliability-Aware RAG, which enhances traditional RAG models by incorporating source reliability into both retrieval and aggregation, reducing the risk of misinformation propagation.

RA-RAG iteratively estimates source reliability and performs inference based on this reliability, achieving improved performance on the proposed benchmark.

### Strengths
1. The idea of iteratively reliability estimation is interesting, which provides an effective estimation for source reliability in certain scenarios.
2. The proposed benchmark can be applied to future related studies.

### Weaknesses
1. Questioning the effectiveness of the method.

a) The method proposed by the author relies heavily on the correctness of majority voting, both in reliability assessment and during the inference phase. However, in noisy scenarios (for example, when there is only one correct document), this method may not precisely select reliable document. This is precisely an important scenario that the paper aims to address.

b) In my opinion, the core assumption of this method is that the higher the probability of different documents in a certain source providing the same answer, the greater the overall reliability of that source. Furthermore, the author's approach to constructing the benchmark seems to follow this assumption, which could be the reason that the reliability estimation is effectively applicable on the benchmark created in this paper. However. due to the imperfect retrieval and the spread of misinformation, this assumption may not hold true in real RAG systems. I believe the author should validate the effectiveness of this method in more realistic scenarios.

2. Inappropriate experimental settings

a) In Figure 2 and 4, I cannot observe a significant performance improvement compared to the baseline WMV. I strongly recommend that the authors present the experimental results in a table. And the author should provide a comparison with more other relevant methods (such as CAG / RobustRAG).

3. Limitations of the proposed benchmark

• The paper claims to present a realistic multi-source RAG benchmark. However, all the misinformation is generated by GPT-4 mini rather than real-world data. This will limit the demonstration of the method's effectiveness in actual scenarios.

### Questions
see Weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
5

### Confidence
4

---

## Human Reviewer 4

### Summary
These authors aim to address the challenge of heterogeneous source reliability in RAG, and propose a novel framework, Reliability-Aware RAG (RA-RAG), which incorporates source reliability into both the retrieval and aggregation processes.

### Strengths
The paper is well-written.
Considering multi-source reliability is innovative and addresses a critical yet often overlooked aspect of RAG methods.

### Weaknesses
1. Despite using prompts for constraints, voting remains influenced by expression diversity and synonyms, which creates a frustrating lack of control in practical applications.
2. Generalization issues, such as in multi-hop scenarios and open-ended questions.
3. I question the motivation behind using RAG responses to certain queries about a resource as the basis for assessing that resource's reliability. For instance, a small amount of data does not necessarily indicate unreliability, and it is challenging to distinguish a small amount of correct information when faced with a large volume of outdated or incorrect data. Similarly, as the retriever is a critical component in RAG, the reliability of the retriever itself should not be overlooked when evaluating the reliability of a resource through retrieval-based methods(The authors' method for predicting reliability can still be viewed as a variation of relevance. ).
4. Lack of baseline to experimentally demonstrate the effectiveness.

### Questions
In practice, would this reliability prediction result in multiplied API consumption and reduced efficiency?
It is necessary to compare with some advanced RAG methods to validate the necessity of "considering multi-source reliability".

### Soundness
2

### Presentation
4

### Contribution
3

### Rating
3

### Confidence
3