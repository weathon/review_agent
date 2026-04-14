# Thread: A Logic-Based Data Organization Paradigm for How-To Question Answering with Retrieval Augmented Generation

- Decision: Reject
- Scores: 8, 5, 5, 5

## Abstract
Recent advances in retrieval-augmented generation have significantly improved the performance of question-answering systems, particularly on factoid '5Ws' questions. However, these systems still face substantial challenges when addressing '1H' questions, specifically how-to questions, which are integral to decision-making processes and require dynamic, step-by-step answers. The key limitation lies in the prevalent data organization paradigm, chunk, which divides documents into fixed-size segments, and disrupts the logical coherence and connections within the context. To overcome this, in this paper, we propose Thread, a novel data organization paradigm aimed at enabling current systems to handle how-to questions more effectively. Specifically, we introduce a new knowledge granularity, termed 'logic unit', where documents are transformed into more structured and loosely interconnected logic units with large language models. Extensive experiments conducted across both open-domain and industrial settings demonstrate that Thread outperforms existing paradigms significantly, improving the success rate of handling how-to questions by 21\% to 33\%. 
Moreover, Thread exhibits high adaptability in processing various document formats, drastically reducing the candidate quantity in the knowledge base and minimizing the required information to one-fourth compared with chunk, optimizing both efficiency and effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a novel data organization paradigm named THREAD, aimed at improving the capability of question answering systems in dealing with "how-to" questions, particularly those requiring dynamic, step-by-step solutions. Existing retrieval-augmented generation systems face challenges in connecting documents or chunks when handling such questions. THREAD introduces a new granularity of knowledge known as "logical unit" and adaptively utilizes the Linker to explicitly represent the internal logic between texts. This enables the method to exclude redundant information during the reasoning process and better maintain the coherence of answers. The paper conducts extensive experiments in open-domain and industrial settings, including Web Navigation, Wikipedia Instructions, and Incident Mitigation scenarios. Additionally, the authors provide a detailed analysis of different data organization paradigms, including ICL, SL, RAG based on various chunking approaches and the proposition, etc, to evaluate the superiority of the THREAD paradigm compared to existing methods. This paper addresses a specific and practical problem, proposes a well-motivated approach, and supports its claims through rigorous experimentation.

### Strengths
Originality:
This paper introduces THREAD, an innovative data organization paradigm that captures the logical structure within documents and the connections between steps through the concepts of logical units and linkers.

Quality:
In terms of experimental design, the paper conducts extensive testing on datasets from both open-domain and industrial environments, with detailed comparisons to existing methods, demonstrating the generalization ability and reliability of its approach. The paper also considers language models of different sizes.

Clarity:
The paper is well-structured, from the problem statement to the methodology, experimental design, and results analysis. The presentation of experimental results is clear, and the use of charts and figures makes the data easy to compare and understand.

Importance:
This paper effectively addresses "how-to" problems in practical applications, demonstrating advantages compared to existing methods.

### Weaknesses
1. The description of the extraction and merging process of logical units in the paper is somewhat vague, lacking an in-depth explanation: Although Sections 3.1 and 3.3 of the paper provide introductions, they mainly focus on the functions and roles, with relatively vague descriptions of specific implementation strategies, especially for the crucial Linker component. Additionally, the specific implementation process of unit merging is not clear, such as how to merge the various parts of two logical units.
2. The THREAD proposed in the paper represents a novel paradigm specifically tailored for addressing "how-to" questions, yet it poses significant implementation challenges in other real-world scenarios, particularly in establishing Linkers between logical units, and exhibits limited method scalability. The paper primarily focuses on how to better extract logical units, while the precise recognition and extraction of relationships between any two text segments or documents are not discussed in detail.

### Questions
1. Is the extraction of the Linker component based on a single document or a comprehensive analysis of multiple documents?
2. Can the paper provide a clearer explanation of the unit merging process, including how four parts of two logical units are merged?
3. When performing logical unit extraction, how is content extracted from a document that may contain multiple aspects and themes, especially when the specific question is unknown?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes THREAD, a novel data organization paradigm aimed at enabling current systems to handle how-to questions more effectively. Specifically, we introduce a new knowledge granularity, termed ‘logic unit’, where documents are transformed into more structured and loosely interconnected logic units with large language models. Extensive experiments are conducted across both open-domain and industrial settings.

### Strengths
The paper is written well and points out a good question.
The idea of LU is reasonable.

### Weaknesses
Objectively speaking, most RAG objects are chunks. This paper does not change this basic background. So the beginning of this article is very misleading. It is recommended not to exaggerate the motivation in the respect of chunks.

The last sentence of contribution # 3 does not seem to be supported by any experiments.

### Questions
1. The proposed method calls many logic units which act like tools and divide a question to many sub questions or new questions. So The proposed method should be compared with baselines using tool learning, such as Toolllm and ControlLLM, etc. You can find more from "Tool Learning with Large Language Models: A Survey".

2. How to ensure that there are tight connections between LUs? It is easy to form a large number of meaningless nodes. If one LU triggers more than one other LUs, or if a problem triggers more than one LU, how does your approach work? What if none of the Linkers support the answer to the question? 

3. The idea of LU is good, but the construction of LU in practical applications is very troublesome. LU is completely based on human design and is very complex. Do you have any ideas to deal with it? What is the contribution to the performance of each part in LU? The meta data in LU did not described in the Fig.2.

4. The baselines in this paper are not particularly persuasive(some are not based on RAG), and the outcome of LU should be tested in more related baselines.

5. The data sets used in this article are all small-scale data sets. Have you tried large-scale data sets?

### Soundness
2

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
2

### Summary
This paper addresses the limitations of current RAG systems in answering "how-to" questions, which are often complex, multi-step, and dynamic in nature. The authors propose THREAD, a novel data organization paradigm that restructures documents into interconnected "logic units". THREAD enables a more coherent, stepwise approach to answering how-to questions by capturing the logical flow within documents. Experiments demonstrate that THREAD outperforms existing chunk and document-based paradigms.

### Strengths
1.	THREAD introduces a novel logic-based organization paradigm that emphasizes the logical flow needed to handle complex how-to questions effectively. 
2.	The paper’s design of logic units with structured components is well-thought-out. Each component serves a specific purpose, enhancing logical continuity and allowing for more coherent, step-by-step responses.
3.	Experimental results show that THREAD outperforms traditional data organization paradigms.
4.	THREAD reduces the number of retrieval units and token length required for generation, optimizing memory usage and computational efficiency.

### Weaknesses
1. The experiments mainly compare THREAD with chunk-based and document-based paradigms, with limited discussion on other advanced retrieval techniques, such as graph-based retrieval or hierarchical indexing. 
2. In the industrial setting, the evaluation relies on human engineers. However, the paper does not provide enough details about the evaluation protocols, inter-annotator agreement, or potential biases. 
3. THREAD’s effectiveness appears to depend on the assumption that documents are reasonably structured and easy to parse into logic units. The paper does not discuss THREAD’s robustness when faced with inconsistent or poorly formatted documents, which are common in real-world scenarios.
4. Although THREAD reduces the knowledge base size and required information for generation, the paper lacks a detailed analysis of computational efficiency and scalability when handling very large knowledge bases, potentially limiting its practicality for large-scale applications.
5. The introduction of multiple components and types in each logic unit may increase the complexity of maintaining the knowledge base, especially when documents are updated. While the paper mentions LU updating, it lacks sufficient detail on how updates will be managed and the potential maintenance overhead.
6. The paper does not include a detailed error analysis to identify common failure cases. Understanding these would help clarify THREAD's limitations and guide future improvements.

### Questions
None

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
1. The authors proposed to organize documents in Retrieval Augmented Generation systems into Logical Units, each consisting of Prerequisite, Header, Body, Linker, and Metadata. This aims to better answer how-to style questions.
2. To build a Threads knowledge base, LLMs are used to reformulate and extract logical units from a document corpus. To apply this in a RAG system, LU headers are used for embedding search, Prerequisite field is used for filtering, and Linker field is used for finding next steps in a dynamic style how-to questions when there can be multiple execution outcomes.
3. They tested this method on web navigation, wikihow, and human evaluation for incident mitigation, and showed superior performance when compared to other document chunking methods.

### Strengths
1. The authors proposed a novel way to organize document corpus. When compared to the usual chunking methods, it facilitates building logical connections between units, and compared to knowledge-graph approaches, it facilitates building meta connections between units that are larger than the more atomic-level knowledge base entities.
2. This approach is shown to be effective at improving answering how-to questions

### Weaknesses
### Clarity
In general the major problem for me is clarity. The paper is a bit hard to follow and requires multiple re-read to understand how the system is supposed to be implemented.

1. For the methodology section, reading the text itself is clearer than trying to parse figure 2 & 3. For figure 2, the arrow from (b) to the right side confused me. Per my understanding, the right panel is an individual view of an LU. Maybe split up the high-level flow with component details into different figures? In figure 3, it was initially hard to match the components to your text description. If you add some formulation and use symbols in both the figure and text, it might make it easier to follow.

2. For the experiment section, it is hard to establish correspondence between components described in the methodology section (sec 3) and their instantiations in the 3 datasets. e.g. I'm confused with how the "execution" part described in sec 3 is implemented in the 3 datasets. 

- Having some symbols/anchors to establish correspondence between the subcomponents and their implementation in experiments might help. 

- Maybe additional columns in table 1 showing design choices for each dataset like: "Dynamic?", "Executable?", "What is being executed" ...

- Before introducing the evaluation metrics, it might be helpful to explain what the task setup is for each dataset, what input is the baseline & your system given, what output is the system expected to produce.

### Some additional results that could be included
1. While the approach seems to perform well compared to existing benchmarks, I'm not really sold on the applicability of this approach. It requires rewrite & extraction of the documents using an LLM before it is even indexed. I would be interested to know how expensive time/cost-wise to implement this, e.g. how it scales in-terms of document length & corpus size etc.
2. The actual embedding search is using only the header field of each LU. It would be interesting to know if indexing using the body help or hurt performance. 
3. Can you still use this organization approach for other 5W questions or would it hinder performance?

### Questions
My main concern is clarity, as it affects how I understand the presented results. I will be able to give a more accurate evaluation upon clarification.

### Methodology
1. How is the selector in the retrieval pipeline implemented? Does it check pre-requisites by embedding similarity or by string match/ngram overlap?
2. The Linker description says, “Its format varies by LU type, serving as either a query for retrieving other LUs or an entity relationship.” is it like an actual pointer to another LU or is it just some text that you then need to use the retriever to search for other LU? 

### Experiments:
1. How do you verify the reformulation and extraction of documents to make sure there are no hallucinations?
2. For a single task in mind2web there are multiple interactions with the website to accomplish the task. Is this the task setup: given webpage and element choices, (optionally given relevant docs for RAG baselines), choose an element and an action?

### Results
In 5.2 ablation for Retrieval Unit Selector, it says "Comparing chunk selection to LU selection, the performance of Ele. Acc drops by 4.29% and Op.F1 by 3.38%", can you point out which rows on Table 5 is this concluded from? Is it a typo where "Chunk" performance is lower than "Chunk w/o chunk selection"?

### Soundness
3

### Presentation
1

### Contribution
3
