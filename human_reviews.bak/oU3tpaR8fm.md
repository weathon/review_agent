# Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Retrieval-augmented generation (RAG) empowers large language models (LLMs) to utilize external knowledge sources. The increasing capacity of LLMs to process longer input sequences opens up avenues for providing more retrieved information, to potentially enhance the quality of generated outputs.  From a long-context LLM perspective, it assumes that a larger retrieval set would contain more relevant information (higher recall), that might result in improved performance. However, our empirical findings demonstrate that for many long-context LLMs, the quality of generated output initially improves first, but then subsequently declines as the number of retrieved passages increases. This paper investigates this phenomenon, identifying the detrimental impact of retrieved "hard negatives" as a key contributor. To mitigate this and enhance the robustness of long-context LLM-based RAG, we propose both training-free and training-based approaches. We first showcase the effectiveness of retrieval reordering as a simple yet powerful training-free optimization. Furthermore, we explore training-based methods, specifically RAG-specific implicit LLM fine-tuning and RAG-oriented fine-tuning with intermediate reasoning, demonstrating their capacity for substantial performance gains. Finally, we conduct a systematic analysis of design choices for these training-based methods, including data distribution, retriever selection, and training context length.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the effect of increasing the length of the retrieved document list on the performance of RAG. They find that simply increasing the number of retrieved documents does not consistently improve the performance of RAG, and both training-free and training-based methods are proposed to solve this problem.

### Strengths
1. The writing is clear and easy to follow

2. Experimental results show the effectiveness of the proposed methods.

3. This paper makes significant contribution in the context of the rapid development of long-context LLM

### Weaknesses
In this rebuttal period, the authors address most of my concerns below

1. **Wrong research basis**: In Abstract and Introduction Sections, this paper's description of the current research status of the RAG community is wrong, which directly leads to the fact that the research basis of this paper is wrong, which is unacceptable. Specifically, in Lines 14-16 of Abstract "It is plausible to assume that a larger retrieval set would contain more relevant information (higher recall), that might result in improved performance" and Lines 46-47 of Introduction "Intuitively, longer context would allow for the inclusion of more retrieved passages, leading to higher recall and potentially improved performance". Both claims misrepresent the current state of research in the RAG community. Many studies in RAG have shown that increasing the number of retrieved documents cannot consistently improve the RAG performance due to the noise or incorrectness of the retrieved texts [1, 2 ,3, 4, 5]. This paper wrongly uses a problem that has been clarified in the RAG community as the motivation, which is unacceptable.

2. **Over-claiming of contributions**: As weakness 1 indicates that the robustness of RAG has been fully-studied in RAG [1,2,3,4,5] and shows irrelevant retrieved documents may interference the performance of RAG, the main contribution claimed in this paper that "we systematically analyze the use of long-context LLMs in RAG systems, specifically examining the impact of retrieved "hard negatives" on performance" cannot be seen as the contribution of this paper.

3. **Limited technical contribution**：This paper claims that they propose two methods to enhance the robustness of RAG on "hard negatives". One method is a training-free method based on retrieval reordering, this method is incremental based on Lost-in-the-Middle [6]. The other method is explicit tuning with intermediate reasoning for relevance identification, this is similar to RetRobust [1] without distinguishable points.

[1] Making Retrieval-Augmented Language Models Robust to Irrelevant Context

[2] Chain-of-note: Enhancing robustness in retrieval-augmented language models

[3] Corrective retrieval augmented generation

[4] Evaluation of Retrieval-Augmented Generation: A Survey

[5] Benchmarking large language models in retrieval-augmented generation

[6] Lost in the Middle: How Language Models Use Long Contexts

### Questions
Please see Weaknesses above.

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
This paper primarily explores the phenomenon of long inputs in Retrieval-Augmented Generation (RAG): for many long-context large language models (LLMs), the quality of generated output initially improves but then declines as the number of retrieved passages increases. This observation contradicts the previous assumption that a larger retrieval set, containing more relevant information, would enhance performance. The study identifies the detrimental impact of retrieved "hard negatives" as a significant factor. To mitigate this effect and enhance the robustness of long-context LLMs in RAG, the paper proposes three methods: (1) an untrained method based on retrieval reordering, (2) implicit robustness fine-tuning to enhance resistance to hard negatives, and (3) explicit relevance fine-tuning with intermediate reasoning to identify relevance.

### Strengths
1. The phenomenon where the quality of LLM-generated outputs first improves and then declines with an increasing number of retrieved segments is intriguing, and the authors have conducted thorough experiments to explore this.
2. The importance of hard negatives for long-context LLM evaluation is well-studied, and it is observed that increasing the number of hard negatives typically leads to a decrease in RAG answer accuracy. The strength of the retriever is directly related to the difficulty of the hard negatives.
3. The three methods proposed by the authors (Retrieval reordering, Implicit robustness fine-tuning, Explicit relevance fine-tuning) all enhance the performance of long-context LLMs in RAG applications.
4. Overall, the paper is well-written and easy to follow.

### Weaknesses
1. Figure 1 shows that while other LLMs exhibit an initial improvement in performance followed by a decline with an increasing number of retrieved passages, Gemini-1.5-Pro does not exhibit this trend, and the authors do not explain this phenomenon.
2. From the experiments, it appears that LLMs ranging from 7B to 12B are susceptible to the influence of hard negatives. It remains unclear whether this phenomenon can still be observed in LLMs larger than 30B. I suspect that larger LLMs might automatically identify these hard negatives.

### Questions
See Questions

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the challenges and opportunities of using long-context LLMs in RAG systems. The main contributions include:
- A systematic analysis demonstrating that increasing retrieved passages doesn't always improve performance with long-context LLMs
- Identification of the detrimental impact of "hard negatives" on LLM performance
- Three approaches to improve RAG performance: (1) a training-free retrieval reordering method, (2) Implicit robustness fine-tuning, (3) explicit relevance fine-tuning with intermediate reasoning.
- Comprehensive analysis of design choices for RAG-specific LLM tuning.

### Strengths
- The paper presents a technically sophisticated analysis of long-context LLMs in RAG systems, with comprehensive experiments across multiple models, retrievers, and datasets. The empirical validation is thorough and well-documented.
- The identification of the "hard negatives" problem in long-context RAG represents a major contribution to understanding LLM behavior in retrieval contexts.
- The proposed methods, particularly the training-free retrieval reordering approach, offer immediate practical value for improving RAG systems. The solutions are well-motivated by the empirical findings.

### Weaknesses
- The paper lacks to explain why long-context LLMs behave differently with hard negatives, which may limit the generalizability of the findings and makes it harder to predict when the proposed solutions might fail.
- The experimental evaluation doesn't compare against recent competing methods for improving RAG performance, such as reranking approaches or hierarchical retrieval methods.
- The paper lacks an analysis of failure cases or limitations. When do these methods break down? What types of queries or documents are particularly challenging?
- The computational requirements of the intermediate reasoning approach requiring Gemini-1.5-Pro are not thoroughly discussed. This raises questions about the practical applicability in resource-constrained environments.

### Questions
- What are the key properties of "hard negatives" that make them particularly challenging? Is it semantic similarity, structural patterns, or something else? How does your analysis generalize to other types of retrievers not tested in the paper?
- How does your approach compare to recent methods like RetRobust (Yoran et al. 2023 - Making Retrieval-Augmented Language Models Robust to Irrelevant Context), or RA-DIT: Retrieval-Augmented Dual Instruction Tuning (Lin et al. 2023) that also aim to improve RAG robustness?
- Have you considered comparing against hierarchical retrieval methods that might naturally mitigate the hard negatives problem?
- Could you provide a cost-benefit analysis comparing your methods against simpler approaches like reducing retrieval size or basic reranking?
- Are there specific types of queries or documents where hard negatives are particularly problematic?
- Why did you choose the specific reordering strategy for the training-free approach? Have you explored other ordering patterns?
- How sensitive is the intermediate reasoning approach to the quality and style of reasoning labels?

### Soundness
2

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
4

### Summary
This paper studies how RAG systems are impacted by the advancement in long-context LLMs. The study empirically shows how the increase in the context-length (which allows more passages to be used) doesn't necessarily imply improved performance of the system. They show that the performance first improves before declining with increases number of retrieved passages. This is attributed to the presence of "hard negatives" which confuses the LLMs. The authors next propose 3 ways to counter the "hard-negatives" issue. The first is the reordering of the retrieved passages and it is shown that it can be an effective non-training based method. The other two approaches are training-based: finetuning with explicit hard-negatives, and finetuning with an additional reasoning step to train the LLM to learn to reject hard negatives. They show how these approaches help improve performance. A well-performed set of ablations help see the impact of data distribution match, retriever choice, and context length of LLMs during training.

### Strengths
The paper focuses on answering a question that hasn't received rigorous attention yet: understanding if retrieval optimizations are not needed anymore in RAGs system if we have long-context LLMs. They approached it quite methodically by first trying to answer the question by looking at multiple LLMs (each with its own context length limit) and multiple retrievers and seeing the impact of the RAG system with increasing retrieved passages. The problem has lot of moving parts: LLM, context length limit, # of retrieved passages, choice retriever, evaluation and training datasets, to name a few. The authors have done a reasonably well job to balance these different parts to answer the question.

### Weaknesses
There are a few places where the observations can be better represented and new hypotheses can be developed and evaluated. Some examples are below (and in the "questions" section below):
- In Fig 1, the authors claim that for strong retriever, the performance either reduces or reaches a plateau with increasing number of retrieved passages whereas it is different for weak retriever. However, I don't see any difference from the figure. Even for BM25 (weak retriever) the performance has inverted U shape similar to e5 (strong retriever).
- The reordering algorithm assumes that hard negatives are typically lower in relevance and hence reordering allows them to be in the middle, therefore allowing the LLM to not get distracted by them. However, that's not necessarily true. They most likely would be at the top and hence are called "hard" negatives.
- It was unclear until we look at the appendix on what was the knowledge base used for the experiments. It is also not clear (maybe my mistake) on what was the retriever used for experiments in Sec. 5.
- A key issue in the experiments is that for all the datasets used, the knowledge base is wiki which is already a part of the LLMs pretraining data. So some of the observations made could be impacted by how much of the knowledge for answering the query is already in the LLM's weights and how much is from the retrieved passages. Since the answers to the query are most probably in the model weights due to pretraining data, the observations from the experiments could be flawed and we might not be able to make fair conclusions. It would be good if we can run these experiments with a knowledge base use for retrieval be something that is NOT a part of the LLM's pretraining data.

### Questions
- Fig 1 also has an interesting observation where the Gemini-1.5-pro is quite robust to increasing number of retrieved passages and doesn't decline in performance. Is there a correlation between the context windows of these long-context LLMs and their ability to be robust to increasing number of retrieved passages? And if so, would "infinite"-context window LLMs have most robustness? It would be interesting to perform additional study around this.
- The above is especially the case when answering the subsequent questions in the paper regarding the hard negatives. Since LLMs are not perfect processing blocks. Just because the correct one is present doesn't mean the LLM will get it right, and it depends on how good the LLM is at processing the given information.
- Fig. 7b indicates that there might not be much impact on the choice of the retriever. More information on that would be helpful.

### Soundness
3

### Presentation
4

### Contribution
4
