# Search-on-Graph: Iterative Informed Navigation for Large Language Model Reasoning on Knowledge Graphs

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Large language models (LLMs) have demonstrated impressive reasoning abilities yet remain unreliable on knowledge-intensive, multi-hop questions---they miss long-tail facts, hallucinate when uncertain, and their internal knowledge lags behind real-world change. Knowledge graphs (KGs) offer a structured source of relational evidence, but existing KGQA methods face fundamental trade-offs: compiling complete SPARQL queries without knowing available relations proves brittle, retrieving large subgraphs introduces noise, and complex agent frameworks with parallel exploration exponentially expand search spaces. To address these limitations, we propose Search-on-Graph (SoG), a simple yet effective framework that enables LLMs to perform iterative informed graph navigation using a single, carefully designed \textsc{Search} function. Rather than pre-planning paths or retrieving large subgraphs, SoG follows an ``observe, think, then navigate'' principle: at each step, the LLM examines actual available relations from the current entity before deciding on the next hop. This approach further adapts seamlessly to different KG schemas and handles high-degree nodes through adaptive filtering. Across six KGQA benchmarks spanning Freebase and Wikidata, SoG achieves state-of-the-art performance without fine-tuning. We demonstrate particularly strong gains on Wikidata benchmarks (+16\% improvement over previous best methods) alongside consistent improvements on Freebase benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Large language models excel at reasoning but remain unreliable on multi-hop, knowledge-intensive questions: they miss long-tail facts, hallucinate under uncertainty, and lag real-world updates. Knowledge graphs help, yet common KGQA approaches are brittle (full SPARQL planning), noisy (large subgraph retrieval), or combinatorially expensive (complex agents). They introduce Search-on-Graph (SoG), a simple framework where an LLM iteratively “observes then navigates” using a single \textsc{Search} function—inspecting the current entity’s actual relations before choosing the next hop. SoG adapts to diverse KG schemas, filters high-degree nodes, and uses off-the-shelf LLMs without fine-tuning. Across six Freebase and Wikidata benchmarks, it reaches state-of-the-art results, including ~15% gains on Wikidata over prior methods.

### Strengths
1. A simple idea yielded strong performance.

2. It achieved the best results among all baseline methods.

### Weaknesses
1. The ablation study was not properly conducted. The exploration algorithm itself appears to have a structure that is not fundamentally different from other baselines such as ToG. Therefore, an explanation is needed as to why the performance improved so significantly and which component had the greatest impact (e.g., whether it’s due to the use of markdown, providing type information to the LLM, or the effect of high-degree filtering).

2. The content in the preliminaries section (Section 3) does not seem necessary. Instead, it would be more meaningful to include additional analyses of the experimental results in that space.

3. It is necessary to run experiments using the same backbone LLM as other baselines. While other studies used GPT-4.1 or GPT-4, this work conducted experiments based on Qwen3 or GPT-4o. A fair comparison is therefore required.

### Questions
It would be great if you could provide responses to the points written under Weakness.

### Soundness
2

### Presentation
3

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
In this paper the authors introduce SoG, a method for question answering that breaks down the problem into multiple steps and uses a knowledge graph to inform the answers. The method combines prompting with tool use; a search interface to the knowledge graph is provided and the LLM is allowed to query for (subject, relation, object) triples that contain the concepts in the question. The result of the query is then encoded as a markdown table and provided as context to the LLM. The LLM generates a rationale as more context and chooses to either further query the knowledge graph for more context, or answer the question. Overall the method is simple and obtains strong results on benchmarks.

### Strengths
* The paper is on an important and timely problem, i.e. using knowledge graphs for more reliable question answering and is of interest to the ICLR community. 
* The results are strong on the benchmarks reported and SoG is compared to many other approaches.
* The method is sensible, simple and straightforward to implement. I can imagine the method being of practical use to the community.
* The authors introduce ablations for some of the choices they make, such as zero-shot vs few-shot learning and the choice of encoding for the KG triples used for context.
* The authors use a nice illustrative example and the idea is straightforward to follow.

### Weaknesses
* It is unclear to me what research question the paper answers and exactly what gap in the literature the paper is addressing. One reason for this is that the related work section summarises other work on KGQA, but it doesn't tell the reader how these methods differ from SoG or which gap in the literature SoG is filling. It is also unclear whether SoG addresses the shortcomings highlighted in the related work section. Take for example the following statement from the related work: "These methods face fundamental trade-offs: larger subgraphs boost recall but introduce noise, while smaller ones risk missing critical edges." Is this trade-off not also present for SoG given that the retrieved triples sometimes have to be pruned so that they fit in the context window of the LLM?
* While the paper obtains solid results, the comparisons seem unfair to me. This is because previous methods used older LLM versions, e.g. GPT-4 instead of GPT-4o. While it is understandable that the authors have no control over previously published results, the reader is left wondering whether the results are better due to their use of more recent/powerful LLMs and not because of the methodology, i.e. SoG. This would be a much more important ablation than the ones reported, but it has not been carried out.
* The insights and learnings in the paper are limited. For example, it is unclear what causes SoG to be better than previous methods, is it indeed reducing hallucinations? Is it retrieving more relevant context and making fewer mistakes at each step? The paper has no analyses to produce such insights, other than ablations on the effect of few-shot learning and thinking vs non-thinking models. That being said, the ablations in Section 5.3 were very nice to see.
* While the authors do a good job explaining that the retrieved information from the KG must be limited to keep the context size manageable for the LLM, they do not report how much compute / memory is needed for their method compared to alternatives. So while their method is shown to outperform alternatives for the benchmarks, it is unclear at what cost.
* Some sections are not self-contained. For example, the details in Section 4.3 were not enough for me to understand how the prompts worked in detail, and while I did look at the appendix to get a better understanding - and the examples were super nice and useful, this should not be required.

### Questions
What happens if the question cannot be answered based on the information in the knowledge base? I have seen in the prompt that the model is asked to answer the question anyway, but is this something that happens often?

"The function returns results in a space-efficient markdown table format" -> In what sense is markdown format space efficient, could the authors please elaborate on this? I initially thought it would be inefficient because there would be a lot of space used for padding, but I see in the appendix that the results are not padded. I now found Table 2, could the authors maybe point to this at this point and explain it is more efficient than JSON, for instance?

I think the following point from the results section is interesting, but unsubstantiated: "Our consistent performance across complexity levels likely stems from our focused exploration strategy—by selecting one relation per hop rather than exploring multiple paths in parallel, we avoid the noise accumulation that can overwhelm simpler questions while maintaining expressiveness for complex reasoning chains." I believe the contribution would have been a lot stronger if the authors could back this point up by experiments or analysis instead of speculation. Is there a reason for not exploring this further / answering this as a research question?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper “Search-on-Graph: Iterative Informed Navigation for Large Language Model Reasoning on Knowledge Graphs” proposes a novel framework that allows large language models (LLMs) to reason over knowledge graphs through step-by-step navigation.

### Strengths
- Clear problem positioning – The paper identifies the gap between LLMs’ internal reasoning limitations and KGQA’s heavy reliance on structured navigation, framing the contribution in a meaningful way.

- Empirical competitiveness – Results on six benchmarks show state-of-the-art or near-SOTA accuracy, validating the effectiveness of the method.

- Practical considerations – The design includes handling high-degree nodes, structured outputs, and tool interfaces, showing attention to real deployment challenges.

### Weaknesses
- Method novelty - The proposed method may lack sufficient novelty. It mainly combines an existing LLM with a basic graph-search mechanism to perform knowledge graph reasoning.

- Limited analysis granularity – Results are mostly aggregate; could you provide some result about hop length (1-hop, 2-hop, 3+), question type, or error cases.

- Scalability concerns – Frequent tool calls may lead to high latency and API cost; the paper lacks detailed analysis of system overhead.

### Questions
Seen above.

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
The paper introduces a method for query answering (QA) on knowledge graphs that leverages an LLM to explore the nodes by following the most relevant predicates (or relation types) w.r.t. the input question. The method is rather simple, adapts to any schema, and only performs 1-hop reasoning steps until the answer is found. Nevertheless, the method is claimed to achieve the best results on many QA benchmarks and w.r.t. several other methods exploiting LLM agents.

### Strengths
Overall, I have appreciated the introduction of paper and the related work. I have also found the writing to be easy to follow. However, I believe some parts lack some crucial information. I discuss some points in the weaknesses section.

I believe one strength of this work is that the method is quite simple and relatively easy to implement. I have also appreciated the consideration of open models in the experiments, which tells that even smaller LLMs such as Qwen3-30B could compete with other ones that are much more expensive. Furthermore, the comparison of results with many baselines and previous techniques positively contributes to the submission. 

I have also found the ablation study about the output formats (Table 2) interesting.

### Weaknesses
In the introduction, the authors claim that existing methods using LLMs to explore the graph require "upfront planning, or parallel path exploration, thereby increasing computational complexity [...]" (L92-L94). It is however not clear of the proposed model prioritizes which nodes to explore, and how does this compares with the related work. Furthermore, in L180-181 the authors mention that leveraging beam search "exponentially expands the search space". However, I was not able to understand how this paper tackles the issue. Could you please explain how do you choose which nodes to explore first and how this compares with the cited works? From Section 4.2, I understand that the proposed approach is to explore links by asking an LLM.  However, to my understanding one could still follow different paths, and it is not clear how the exploration of the graph is prioritized.

Furthermore, it is not clear from Section 4.3 how exemplars are automatically derived from the training set. The authors mention that "each navigation step is explicitly recorded through tool calls" (L298-299)". Although the authors provide in Appendix A some listings of the tools used, it is not clear how these tools are related to prompting because no other information is given in Appendix A. It seems the authors did not provide any description of the procedure to extract the exemplars and what is the role of these tools.

Finally, it is not clear from the paper the computational cost of the proposed method. Although the authors provide some ablation based on the number of completion tokens using different prompting formats (Table 2), it is not clear whether the method is more or less efficient than the related methods shown in Table 1. While I understand that re-running all previous approaches might be infeasible, I believe that a comparison with at least one or two of the previous methods about the overall computational cost would strengthen this submission.

It might be possible I missed some parts. While I remain very open for clarifications about my comments, the combination of lack of clarity and lack of a technical discussion with previous methods substantially decreases my overall score.

### Questions
See my questions in the weaknesses section.

- How do you manage the fact that the LLM might predict a property ID that is not one of the unique properties extracted at the first call of Algorithm 1?

- What is the "Avg. Turns" column in Table 2?

### Soundness
1

### Presentation
2

### Contribution
3
