# Memory Type Matters: Enhancing Long-Term Memory in Large Language Models with Hybrid Strategies

- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
The memory capabilities of Large Language Models (LLMs) have garnered increasing attention recently. Many approaches adopt Retrieval-Augmented Generation (RAG) techniques to alleviate the “Forgetting” problem in LLMs. Despite great success achieved, existing RAG-based memory approaches typically overlook the differences between memories and employ a unified strategy to process all memories, leading to suboptimal performance. Thus, an intuitive question arises: can we categorize memory into different types and select appropriate strategies? However, given the topic-rich, scenario-complex, and boundary-blurred nature of memory scenarios, achieving precise classification of memories is not easy. To address this challenge, we propose a memory multi-class benchmark in this paper, termed TriMEM.  TriMEM comprises 6,000 dialogue samples, providing precise annotations for memory types across diverse topics and scenarios. Building upon this foundation, we propose a novel memory framework, named MemoType. MemoType can adaptively identify the category of each memory and design tailored storage and retrieval strategies, thereby achieving satisfactory performance. Extensive experiments on retrieval and generation tasks demonstrate the effectiveness of the proposed approach.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Inspired by psychology, the paper proposes to categorize memories into three types: episodic memory, personal semantic memory and general semantic memory and use these memory types to tailor retriever in the RAG system to improve the RAG performance. The paper has compared its designed RAG system with an extensive number of SOTA RAG systems on challenging RAG benchmarks of involving long memory context and show favorable performance. The improvement is shown for the retrieval performance as well as the RAG answer performance using corresponding metrics.

The ablation studies in the paper show that the proposed memory classification indeed improves the overall retrieval performance compared to the RAG system without such classifications. The retrieval strategy like "fake memory" and using keywords are effective for the retrieval performance and that the proposed strategy holds improvement across various existing retrievers.

### Strengths
- The paper is well motivated in its writing and the proposed methodology is quite clear for the readers to understand including its experimental settings as well as the motivations.
- The paper has compared with an extensive number of SOTA rag systems with different focus (e..g structured RAG, memory enhancement, query enhancement) and the experimental results show the effectiveness of the proposed methods.
- Such proposal seems new. Besides, the implementation is simple and this potentially means the paper can have a good impact in its adoption in real life scenarios. The novelty itself comes with a dataset and annotation which is a contribution on its own and can benefit the community by its introduction.

### Weaknesses
Given that the paper's main contribution is the in introduction of memory classification and the ensuing improvement, there is lack in its detailed explanations and ablations. I will leave the explanations in the question section. But I would be curious to see: what is the effect of each retrieval since they are designed separately for each memory type (maybe even more detailed as the classification is multi-label, raising question of the effectiveness for the overlapping part)? What justifies the keyword method adoption in personal semantic memory?
Without answering these detailed questions, I don't have a clear idea how and why the proposed memory enhance the RAG performance, particularly the improvement in each domain. 

I do agree that the ablation is not trivial to construct including what baseline to choose to answer these questions; nevertheless, I think this question corresponds to the main theme and the contribution of the paper.

I don't get a clear idea how the classifier will go for the OOD classification (does not fall into the classified types).

### Questions
- Is fake memory technology used for personal semantic memory as well?
- How does the trained classifier handle and be trained for the classes that do not fall into existing categories?
- In ablation study (classify), is the reported performance without classify but with key and fake strategies?
- In ablation study (Hybrid), is keyword only impacting the personal semantic memory?
- In related works agent memory, it seems that what the paper proposes can naturally enhance the existing agent memory techniques by letting agent possess/learn different strategies based on memory type. Do authors agree and have further thoughts to share on this?

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
4

### Summary
The paper targets the memory retrieval and response generation problem, and propose a memory type classficiation benchmark by classifing conversational history of existing benchmarks into different types of memory i.e., episodic memory and personal semantic memory, and advocate to retrieve related memory according to the required type of memory (by classifing the type of query using the pre-trained model on the ccollected benchmark), and design specific retrieval strategies and pruning strategy to improve the retrieval and generation quality. Generally, the benchmark is build on top of existing benchmarks, and proposed method is more like re-combination of existing techniques, and more focus on retrieval part. Despite the experimental results is promising, it is uncertain where these gains come from.

### Strengths
1. The proposed method is reasonable desipte it is too specfic for each type and more like engineering tricks.
2. The experimental results confirm the effectiveness of proposed method.

### Weaknesses
1. There are many studies that focus on different types of memory and propose to retrieve according to determined type [1]. This significantly weaken the contribution and novelty of the proposed method.

2. the constructed benchmark is not detailed, i.e., how do you do quality control, why choose these benchmarks? these details are not mentioned in main paper. Despite some are included in appendix, it is not comprehensive.

2. Generally, the proposed method contains three parts: query ruoter (a.k.a, part a, to decide which type of memory to retrieve), memory retrieval according to label by part a (a.k.a., part b), and memory pruning (a.k.a., part c). the experiment only confirms the effectiveness of a+b+c, and each module under this system. It is not clear: i) whether any method incorporate independant part leads to better performance, i.e., query router + other retrieval strategies in the baselines; ii) the effectes of cascade errors; iii) the results of table 4 and 5 shows there is no significant gain for part b and part c.


[1] Perltqa: A personal long-term memory dataset for memory classification, retrieval, and fusion in question answering.

### Questions
see weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This
 paper investigates how Large Language Models (LLMs) handle long-term 
memory, particularly in Retrieval-Augmented Generation (RAG) systems 
that store and recall past information to mitigate the “forgetting” 
problem. Existing RAG-based memory frameworks typically apply the same 
retrieval and storage strategy to all memories, overlooking the fact 
that different types of memories (e.g., factual, episodic, or semantic) 
require different handling. To address this, the authors introduce 
TriMEM, a benchmark containing 6,000 annotated dialogue samples that 
categorize diverse memory types across topics and scenarios. Building on
 this benchmark, they propose MemoType, an adaptive memory framework 
that automatically identifies the type of each memory and applies 
tailored retrieval and storage strategies accordingly. Experiments on 
retrieval and generation tasks demonstrate that MemoType improves both 
memory organization and overall model performance, highlighting the 
importance of memory categorization in long-term language model 
reasoning.

### Strengths
**Novel conceptual framing:**
  The paper introduces a meaningful and intuitive perspective by 
categorizing memories in LLMs into distinct types (episodic, semantic, 
factual) and applying **type-specific retrieval and storage 
strategies**, addressing an underexplored dimension of long-term memory 
modeling in LLMs.

* **New benchmark contribution (TriMEM):**
  The authors contribute a well-structured and valuable dataset, 
**TriMEM**, containing 6,000 annotated dialogue samples with explicit 
memory-type labels. This benchmark fills an important gap for studying 
memory categorization and adaptive retrieval mechanisms in 
conversational settings.

* **Clear motivation and design:**
  The problem formulation is well-motivated, and the overall pipeline — 
from memory categorization to retrieval and generation — is logically 
presented and easy to follow.

* **Potential for generalization and integration:**
  The proposed **MemoType** framework is modular and could, in 
principle, be integrated into broader **RAG** or **agent-memory** 
systems, making it a promising direction for long-term dialogue 
reasoning and adaptive retrieval research.

### Weaknesses
**Limited validation of core claim:**
  The paper’s main contribution lies in classifying memory types 
(episodic, semantic, factual) and applying type-specific retrieval 
strategies. However, the benchmarks used do not contain explicit labels 
for memory types. As a result, the experiments only demonstrate that 
MemoType improves downstream performance, without directly showing that 
the type-adaptive retrieval mechanism itself is responsible for these 
gains.

* **Insufficient diversity of evaluation benchmarks:**
  The evaluation on LongMemEval-S, LongMemEval-M, and LoCoMo effectively
 measures retrieval and generation under long-context and multi-session 
conditions. However, to substantiate claims of general effectiveness in 
memory categorization and adaptive retrieval, the study would benefit 
from incorporating **personalized or task-oriented memory benchmarks** 
(e.g., PerLTQA, MEMTRACK). These settings better reflect realistic agent
 memory use cases—such as user profiles, preferences, or tool-use 
history—where the distinction between factual, episodic, and procedural 
memory is most impactful.

* **Questionable generalization of the memory-type classifier:**
  A notable concern lies in the **credibility and generalization** of 
the proposed memory-type classifier, especially given that the results 
in Table 9 show it outperforming significantly larger models such as 
**Qwen3-8B, Qwen3-32B, Gemini-2.5-Flash, and GPT-4o-Mini**. The 
classifier is a simple **BERT-based model** trained with binary 
cross-entropy loss over only a few iterations, which raises doubts about
 how such a lightweight model achieves superior results. This suggests 
the evaluation setup may be benchmark-specific rather than reflecting 
true generalization. Without cross-domain tests, ablations, or analysis 
of potential data leakage, the claims about robustness and real-world 
applicability remain unconvincing.

### Questions
1. How were the memory-type labels (episodic, semantic, factual) assigned 
or validated during training, given that existing benchmarks do not 
provide such annotations?

2. Can the authors provide evidence or analysis showing that the 
observed performance improvement specifically arises from type-adaptive 
retrieval, rather than general architectural or training advantages?

3. Why were personalized or task-oriented benchmarks (e.g., PerLTQA, 
MEMTRACK) not included in evaluation, given their relevance to memory 
categorization and realistic long-term interaction scenarios?

4. How does the simple BERT-based classifier generalize beyond the TriMEM dataset? Have the authors tested it on out-of-domain data or with alternative encoders (e.g., DeBERTa, RoBERTa) to assess robustness?

5. The paper reports outperforming much larger models (e.g., GPT-4o-Mini, Qwen3-32B). Could the authors clarify the evaluation 
protocol and whether there are benchmark-specific advantages or data  overlaps that might explain this result?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
To address the challenge that, given the topic-rich, scenario-complex, and boundary-blurred nature of memory scenarios, achieving precise classification of memories is not easy, this paper proposes a memory multi-class benchmark in this paper, termed TriMEM. TriMEM comprises 6,000 dialogue samples, providing precise annotations for memory types across diverse topics and scenarios. Building upon this foundation, this work proposes a memory framework, named MemoType.

### Strengths
1. This paper proposes a memory multi-class benchmark.
2. This work proposes a memory augmentation framework to classify and use memory for QA.
3. The paper is well-structured.

### Weaknesses
1. As far as I know, ref[1] has presented a benchmark for multi-class memory. However, this paper did not describe the core difference from [1].
2. This paper includes three categories in the data. Why these three categories? Do we need other categories?
3. The introduction of memory types is insufficient. The authors should provide more details about the types of memory.


[1] Du et al., Perltqa: A personal long-term memory dataset for memory classification, retrieval, and fusion in question answering. In Proceedings of the 10th SIGHAN Workshop on Chinese Language Processing (SIGHAN-10), pp. 152–164, 2024.

### Questions
1. This paper includes three categories in the data. Why these three categories? Do we need other categories?
2. What is the core difference between this work and [1]?
3. How can the MemoType framework balance the importance among different types of memory?

[1] Du et al., Perltqa: A personal long-term memory dataset for memory classification, retrieval, and fusion in question answering. In Proceedings of the 10th SIGHAN Workshop on Chinese Language Processing (SIGHAN-10), pp. 152–164, 2024.

### Soundness
2

### Presentation
2

### Contribution
2
