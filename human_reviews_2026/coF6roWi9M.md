# Improving Multi-step RAG with Hypergraph-based Memory

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Multi-step retrieval-augmented generation (RAG) 
has become a widely adopted strategy for enhancing large language models (LLMs) on tasks that demand global comprehension and intensive reasoning.
Many RAG systems incorporate a working memory module to consolidate retrieved information. However, existing memory designs function primarily as passive storage that accumulates isolated facts for the purpose of condensing the lengthy inputs and generating new sub-queries through deduction.
This static nature overlooks the crucial high-order correlations among primitive facts, the compositions of which can often provide stronger guidance for subsequent steps.
Therefore, their representational strength and impact on multi-step reasoning and knowledge evolution are limited, resulting in fragmented reasoning and weak global sense-making capacity in extended contexts.
We introduce HGMem, a hypergraph-based memory mechanism that extends the concept of memory beyond simple storage into a dynamic, expressive structure for complex reasoning and global understanding.
In our approach, memory is represented as a hypergraph whose hyperedges correspond to distinct memory units, enabling the progressive formation of higher-order interactions within memory. This mechanism connects facts and thoughts around the focal problem, evolving into an integrated and situated knowledge structure that provides strong propositions for deeper reasoning in subsequent steps.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes HGMem, a hypergraph-based memory for multi-step RAG that captures high-order relationships among facts, enabling dynamic, structured knowledge evolution. By moving beyond passive storage, HGMem enhances global understanding and reasoning, outperforming strong baselines on challenging sense-making tasks.

### Strengths
* Long context understanding is an interesting topic
* The paper is well written

### Weaknesses
* The authors claim to address multi-step RAG; however, the focus of the work appears to lean more toward long-context understanding. Notably, the experimental evaluation does not include standard multi-hop RAG benchmarks such as HotpotQA. While I am not necessarily advocating for additional experiments on more datasets, the paper may benefit from a clearer framing or adjustment of its stated objectives to better align with the actual contributions.
* The paper seems to be more concerned about long context understanding, and try to use hypergraph to build memory about the context, but this field has been well studied [1,2,3].
* I am kindly worried about the latency as the method requires to update the hypergraph during reasoning and the update requries additional call of the LLM


[1]Xu W, Mei K, Gao H, et al. A-mem: Agentic memory for llm agents[J]. arXiv preprint arXiv:2502.12110, 2025.

[2]Ong K T, Kim N, Gwak M, et al. Towards lifelong dialogue agents via timeline-based memory management[J]. arXiv preprint arXiv:2406.10996, 2024.

[3]Rasmussen P, Paliychuk P, Beauvais T, et al. Zep: a temporal knowledge graph architecture for agent memory[J]. arXiv preprint arXiv:2501.13956, 2025.

### Questions
* I am actually a little confused about what will the hypergraph be like given a query and correspondding context, how could it help the model to better understand the context
* How will the hypergraph be inputted into the LLM? Can the model effectively understand the graph, the experiments are conducted on relatively strong models, what about on those weaker models like Qwen 7B or Llama 8B?
* In section 4.2, the authors build a offline graph, what is it used to do? Simply for baseline like lightRAG or it is also used for HGMEM?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes HGMEM, a hypergraph-based memory mechanism focusing on dynamic scenarios. It helps improve performance under iterative RAG scenarios.

### Strengths
The overall motivation of the paper is great. The paper conducts comprehensive experiments on multiple datasets. The experimental details are extensive, and the paper experimentally explores a wide range of aspects of the proposed framework.

### Weaknesses
1. I think the description of hypergraph in this paper is not unclear. As the core innovation that distinguishes it from other graph-RAG methods, it should be clearly explained. However, I find it difficult to understand its specific meaning. I hope the authors can provide a clear and symbolic definition in Sections 3.3 and 3.4, rather than redundant statements. It would be helpful to provide more detailed examples and a clearer comparison with the current graph-RAG process.
2. This method does not seem to have a training process. How can a native LLM or retrieval model be adapted to the proposed architecture?
3. Although there are some reproduction instructions, the paper does not seem to provide code.

### Questions
See weaknesses above. Please respond to these cons.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces HGMEM, a new memory mechanism for multi-step RAG systems. The core motivation is to address the limitations of existing working memory modules, which often function as passive stores for isolated facts. HGMEM proposes to model memory as a dynamic, evolving hypergraph, where hyperedges represent distinct memory points. Through iterative interactions with external knowledge, this memory structure is progressively refined via three key operations: update, insertion, and merging. The authors argue that this approach enables the formation of high-order correlations among facts, leading to improved reasoning and global understanding in complex, long-context tasks. The method's effectiveness is evaluated on several benchmarks, where it is reported to outperform various traditional and multi-entity RAG baselines.

### Strengths
1. The paper  discusses  a timely and relevant limitation in multi-step RAG systems: the static and fragmented nature of existing memory modules. The motivation to enable memory to capture higher-order correlations is well-founded and targets a key bottleneck in complex reasoning tasks.

2. The conceptualization of working memory as a dynamic, evolving hypergraph is an interesting direction. This abstraction provides a potentially powerful framework for representing complex, multi-entity relationships that go beyond the limitations of simpler memory structures.

3. The authors provide an extensive set of experiments on several challenging long-context benchmarks. The reported performance gains,  the results suggesting that an open-source LLM equipped with HGMEM can match or outperform baselines using stronger proprietary models, are notable and merit attention.

### Weaknesses
1. The framework's core operations (e.g., merging) are not defined algorithmically but are delegated to a black-box LLM. This introduces uncontrolled non-determinism and makes the method's behavior difficult to formally analyze. The paper does not provide mechanisms or analysis to address the risk of cascading errors stemming from these non-deterministic memory operations.

2. The paper lacks the technical detail required for reproducibility. Key procedures, particularly the 'merging' operation, are described only conceptually. The submission provides no pseudocode or specific prompts that would allow other researchers to implement and verify the method.

3. The method introduces significant computational overhead compared to single-step RAG. The paper fails to provide a quantitative analysis of these costs (e.g., token consumption, latency, number of LLM calls) relative to baselines.

4. The paper fails to cite and discuss highly relevant contemporary work, notably PropRAG (https://arxiv.org/pdf/2504.18070), a highly relevant work that also uses hypergraphs in RAG. Although the two approaches have different goals, the lack of any comparison makes it difficult to situate HGMEM's contribution and assess its novelty against other hypergraph-based methods.

### Questions
1. Could the authors provide the specific prompts, pseudo-code, or a detailed algorithmic description for the merging operation?

2. Could the authors provide a quantitative comparison of the computational costs (e.g., tokens, latency) between HGMEM and baselines?

3.  How are contradictions in retrieved information handled? For example, if new evidence conflicts with an existing memory point, does the system have a specific process for resolving this, or does it simply rely on the LLM to implicitly manage the conflict during the next step?

4.  Given that PropRAG also uses hypergraphs in RAG, could the authors elaborate on why it was omitted from the literature review? Please also articulate the unique contributions of HGMEM that are not addressed by prior hypergraph-based RAG methods.

5. The paper's core contribution seems to lie in a conceptual framework and a prompting strategy. This raises a question about the method's robustness and algorithmic nature. Could the authors elaborate on the algorithmic components that are independent of the LLM's specific instruction-following behavior? And how is performance consistency maintained across different base models?

### Soundness
2

### Presentation
2

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
Multi-step RAG is widely used to boost LLMs’ performance on tasks requiring global comprehension and intensive reasoning. However, existing memory designs are static passive storage that accumulates isolated facts, ignoring high-order correlations between facts and limiting representational strength. The authors propose HGMEM, a hypergraph-based memory mechanism that transforms memory into a dynamic, expressive structure—hyperedges represent memory units to enable progressive high-order interactions, forming an integrated knowledge structure for subsequent reasoning. Evaluations on global sense-making datasets demonstrate HGMEM consistently enhances multi-step RAG and outperforms strong baselines across tasks.

### Strengths
1. This paper identifies an important problem.
3. The proposed method is evaluated on an extensive set of datasets.

### Weaknesses
1. The analysis of the connection and difference with existing work is not comprehensive. Firstly, the proposed method is highly related to graph-enhanced iterative RAG, such as SG-Prompt, ERA-CoT, and KnowTrace. More thorough comparison with them is necessary. Secondly, one recent work on long-context understanding, namely CAM (Constructivist Agentic Memory), also aims to tackle the dynamic updating memory and capture the higher-order interactions within memory. It would better to provide more comparisons with this work.
2. The analysis of the proposed method is not clear enough. It would be better to provide an intuitive example.
3. The experiments are not comprehensive. There is a lack of important baselines as described above. Not time complexity and computational cost of the proposed method theoretically and empirically.

### Questions
Please refer to the weakenesses.

### Soundness
2

### Presentation
2

### Contribution
2
