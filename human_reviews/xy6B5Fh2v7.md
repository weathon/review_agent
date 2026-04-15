# Astute RAG: Overcoming  Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 6, 5, 5

## Abstract
Retrieval augmented generation (RAG), while effectively integrating external knowledge to address the inherent limitations of large language models (LLMs), can be hindered by imperfect retrieval that contain irrelevant, misleading, or even malicious information. Previous studies have rarely connected the behavior of RAG through joint analysis, particularly regarding error propagation coming from imperfect retrieval and potential conflicts between LLMs' internal knowledge and external sources.
Through comprehensive and controlled analyses under realistic conditions, we find that imperfect retrieval augmentation is inevitable, common, and harmful. We identify the knowledge conflicts between LLM-internal and external knowledge from retrieval as a bottleneck to overcome imperfect retrieval in the post-retrieval stage of RAG.
To address this, we propose Astute RAG, a novel RAG approach designed to be resilient to imperfect retrieval augmentation. It adaptively elicits essential information from LLMs' internal knowledge, iteratively consolidates internal and external knowledge with source-awareness, and finalizes the answer according to information reliability.
Our experiments with Gemini and Claude demonstrate the superior performance of Astute RAG compared to previous robustness-enhanced RAG approaches. Specifically, Astute RAG is the only RAG method that achieves performance comparable to or even surpassing conventional use of LLMs under the worst-case scenario. Further analysis reveals the effectiveness of \method in resolving knowledge conflicts, thereby improving the trustworthiness of RAG.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
In this paper, the authors address the imperfect retrieval problem and the knowledge conflict problem in RAG, and propose a AstuteRAG method which firstly adaptively generate passages using its internal parameter knowledge, then consolidate information from both generated passages and retrieved passages by taking their source information into consideration, constructing consistent groups, and addressing the conflicts iteratively. Experimental results show some performance improvements over several RAG baselines.

### Strengths
In general, I think it is critical to resolve  the imperfect retrieval problem and the knowledge conflict problem for better/more robust RAG, and the proposed pipleline(adaptive inner knowledge generation and iterative consolidation) is helpful  for robust RAG. Furthermore, I found the proposed method is training-free.

### Weaknesses
1. Both the imperfect retrieval problem and  the knowledge conflict problem have been widely recognized in the RAG field (see Benchmarking large language models in retrieval-augmented generation AAAI 24 for example), therefore it is inappropriate for "previous studies have rarely connected...". The authors should make a more comprehensive review of RAG studies;
2. There are many previous studies address the source/confidence-awareness of RAG,  iterative RAG, and knowledge-conflict consolidation methods. Therefore, to claim this paper's contributions, the authors should explain the differences/advantages of their methods over these studies.
3. Based on the above observations, I found the baselines in this paper are all general RAG baselines rather than RAG methods addressing the noise/conflict RAG problem,  which makes the experimental results not convincing enough. I think the authors should also compare with iterative-based, agent-based, and confidence-aware RAG baselines.

### Questions
See Weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper focuses on the problem of imperfect retrieval in RAG for LLMs. It analyzes its prevalence and negative impacts like RAG failures and knowledge conflicts. To address this, ASTUTE RAG is proposed, which adaptively generates internal knowledge, iteratively consolidates knowledge with source-awareness, and finalizes answers based on reliability. Experiments show ASTUTE RAG outperforms baselines in various scenarios, effectively resolving knowledge conflicts.

### Strengths
1. The research question, "Is there an effective method to integrate internal and external knowledge for enhancing the reliability of RAG?", is at the cutting edge of the field and holds substantial practical significance for the RAG community. 

2. The essence of this paper is to utilize LLM to conduct a self-reflection on the query (generating internal knowledge documents) first, and then perform a conflict detection on external issues, making full use of the LLM's own capabilities and effectively alleviating the conflict between internal and external support. 

3. The method enhances the robustness in case of retrieval failure. Even when all retrieved information is noisy, it can effectively ensure the bottom-line performance of LLM.

### Weaknesses
1. The performance and efficiency of the method are relatively low. The AstuteRAG proposed by the authors is used in the retrieval (online) stage, where LLM is made to generate as much internal knowledge Q&A as possible. This process is rather time-consuming and consumes tokens. Subsequently, iterative knowledge verification is required, which still uses LLM for conflict detection and paragraph merging. Overall, it is very time-consuming and reduces the practicality of the method.

2. A  major  potential drawback is that the knowledge consolidation process is fully completed by LLM. Although the conflicting information is grouped according to its consistency and the conflicting information is separated and then LLM is made to propose answers and assign confidence scores for each group of documents. Essentially, it is still LLM making judgments based on its own capabilities, and the influence of inherent bias and hallucination cannot be eliminated. A very concerning question is whether the confidence score of LLM shows an obvious bias when there is a conflict between internal and external knowledge? The bias may not necessarily be specifically towards internal knowledge or external knowledge, but rather towards some tokens with a specific high distribution.

3. The authors' verification of the method in this paper is all based on benchmarks with definite factual answers, lacking analysis of some scenarios. For example, both the internal knowledge and the external retrieval content are actually correct, but they form a conflict due to the time range or other specified limiting conditions in the query. From the current method, it is not seen that AstuteRAG has good robustness in this regard. This is also strongly related to the judgment basis mentioned in Weakness 2

### Questions
1. The relationship between retrieval imperfection and query characteristics needs further investigation. Please consider conducting additional experiments to: (1) analyze how different query types affect retrieval performance, (2) evaluate whether the findings from short-form QA can be generalized to other formats,(e.g. Long-form QA) and (3) compare system performance with varying numbers of retrieved documents (e.g., 10, 20, 50) given that modern LLMs can handle larger context windows. These analyses would provide more comprehensive insights into the retrieval mechanism's limitations and potential improvements.

2. In the manuscript, the author mentioned that there were several cases where the LLM refused to provide answers during the experiments. Could you please clarify how these refusal cases were handled in the results presented in Table 1? It would be valuable if you could provide specific statistics on the frequency of LLM refusals, and describe mitigation strategies implemented to handle these refusal scenarios? Furthermore, it appears that these LLM refusal cases could significantly impact the overall Recall of the question-answering , yet there seems to be a lack of discussion regarding this crucial aspect. A more detailed analysis of how these refusals affect the system's would strengthen the evaluation section and provide a more comprehensive understanding of the system's real-world performance  and limitations.

3. Please elaborate on the potential role of reranking in ASTUTE RAG. Specifically, could you: (1) discuss why reranking was not incorporated in the current implementation, (2) analyze how reranking might help address the performance decrease observed in Table 1 when using RAG, and (3) explore whether reranking could be a promising direction for future improvements in reducing noise and enhancing retrieval relevance. This discussion would provide valuable insights into the design choices of ASTUTE and potential avenues for enhancement.

4. Please maintain consistency in the abbreviation format, specifically for 'Retrieval-Augmented Generation (RAG)' in the first sentences of the abstract and  'Retrieval augmented generation (RAG)' in the first sentences of the introduction.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes astute RAG, a framework to address imperfect retrieval and knowledge conflict challenges in retrieval-augmented generation.
Specifically, astute RAG consists of 3 steps including adaptive internal knowledge generation, iterative knowledge consolidation and answer finalization.
Experiments demonstrate that astute RAG outperforms previous robustness-enhanced RAG methods.

### Strengths
- This paper provides an empirical experimental analysis of the impact of imperfect retrieval and knowledge conflict on the accuracy of RAG systems, thereby validating the significance of its motivation.
- This paper presents an intuitive approach that, through the careful design of a framework and prompts, can alleviate the issues posed by the aforementioned challenges, thereby enhancing the performance of RAG systems without training.

### Weaknesses
- **Applicability of the method**: The framework proposed by the authors involves a complex pipeline and prompt design, which necessitates that the target model possesses various capabilities. However, the authors do not provide sufficient experimental evidence regarding the performance of current models in these intermediate stages. Furthermore, the experiments were conducted solely on two close-sourced models with undisclosed details. The lack of relevant details prevents readers from assessing the applicability and limitations of the method. I strongly recommend that the authors include experiments using more open-source models and provide analyses of the intermediate processes.
- **Mismatch between method and experimental design**: In the initial step, the authors employ adaptive generation to extract the model's internal knowledge. However, in the experimental setting, the model is required to generate a maximum of only one passage, which means the design of adaptive generation is not effectively reflected in the experiments. Consequently, the comparison of the number of API calls also lacks significance. Furthermore, I believe that the authors should focus on comparing the total number of API tokens used rather than the number of calls.
- **Novelty of the method**: Although the authors have designed a new framework, similar ideas have appeared in some prior works (e..g, [1]). This factor limits the novelty of this paper.

[1] Merging Generated and Retrieved Knowledge for Open-Domain QA

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces Astute RAG, a novel approach addressing imperfect retrieval and knowledge conflicts in RAG systems. 
The authors first conduct comprehensive analyses demonstrating that imperfect retrieval is prevalent in real-world RAG applications and identify knowledge conflicts between LLMs' internal knowledge and external sources as a key bottleneck. Authors then proposes Astute RAG, which adaptively elicits internal knowledge from LLMs, performs source-aware knowledge consolidation, and finalizes answers based on information reliability. Besides that, the authors demonstrate that Astute RAG outperforms previous robustness-enhanced RAG methods (especially in worst-case scenarios where traditional RAG approaches fail).

### Strengths
-  The paper presents an advanced approach to RAG systems, with a well-structured methodology for combining internal LLM knowledge with external sources. The three-step process demonstrates careful consideration of the challenges in knowledge integration.
- The authors provide comprehensive experimental results using state-of-the-art LLMs (Gemini and Claude) and multiple datasets (NQ, TriviaQA, BioASQ, PopQA), offering valuable insights into the method's performance across different scenarios.
- The work addresses a critical challenge in RAG systems: "the prevalence of imperfect retrieval and knowledge conflicts", backed by solid empirical evidence showing that roughly 70% of retrieved passages don't directly contain true answers.
- The authors' analysis using realistic conditions with Google Search as a retriever provides valuable insights for the research community, particularly in understanding real-world RAG system behavior.

### Weaknesses
- While technically sound, the paper doesn't sufficiently articulate why Astute RAG is particularly advantageous for real-world applications compared to simpler approaches, such as different passage ordering strategies (e.g., chronological, relevance-based, or clustered arrangements), varying quality of knowledge sources (from high-authority to potentially unreliable sources), or different passage selection methods. In fact, authors state that passages are presented by reversed order in the set of experiments, although no further positional dependence has been explored. Recent RAG studies has shown significant differences between distinct arrangement strategies (e.g. Alessio et al. 2024 Improving RAG Systems via Sentence Clustering and Reordering).
- The focus on short-form QA tasks limits understanding of the method's broader applicability. Imho, retriever improvements do not always translate into proportional gains in final answers, particularly in open-domain questions. in specialized tasks, even small improvements in retrieval can significantly boost final answers.
- The method's reliance on advanced LLMs with strong instruction-following capabilities significantly limits its applicability. The paper would benefit from addressing how the approach could be adapted for more resource-constrained or smaller specialized language models. Could you discuss strategies for adapting Astute RAG to resource-constrained environments?

### Questions
While the paper presents valuable contributions and strong empirical results, it falls somewhat short of the technical depth and theoretical foundations:
- Could you elaborate on your choice and implementation of Google Search as the retrieval method: Were there specific criteria for determining which snippets were included in the 10 selected passages? What were the specific reasons for choosing Google Search over other commercial search engines (e.g., Bing)? How did you handle sponsored or advertised content in the search results? 
- How might the results differ with other retrieval approaches? How sensitive is the method to different prompt templates? What is the impact of different parameter settings (e.g., number of iterations, maximum generated passages) on performance and stability?
- How would Astute RAG perform on more complex tasks beyond short-form QA? Could you provide examples of failure cases when the system might not be appropriate to use?

### Soundness
3

### Presentation
2

### Contribution
2
