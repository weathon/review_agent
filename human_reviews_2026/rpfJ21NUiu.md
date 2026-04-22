# ToolLibGen: Scalable Automatic Tool Creation and Aggregation for LLM Reasoning

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Large Language Models (LLMs) equipped with external tools have demonstrated enhanced performance on complex reasoning tasks. The widespread adoption of this tool-augmented reasoning is hindered by the scarcity of domain-specific tools. For instance, in domains such as physics question answering, suitable and specialized tools are often missing. Recent work has explored automating tool creation by extracting reusable functions from Chain-of-Thought (CoT) reasoning traces; however, these approaches face a critical scalability bottleneck. As the number of generated tools grows, storing them in an unstructured collection leads to significant retrieval challenges, including an expanding search space and ambiguity between function-related tools. To address this, we propose a systematic approach to automatically refactor an unstructured collection of tools into a structured tool library. Our system first generates discrete, task-specific tools and clusters them into semantically coherent topics. Within each cluster, we introduce a multi-agent framework to consolidate scattered functionalities: a code agent refactors code to extract shared logic and creates versatile, aggregated tools, while a reviewing agent ensures that these aggregated tools maintain the complete functional capabilities of the original set. This process transforms numerous question-specific tools into a smaller set of powerful, aggregated tools without loss of functionality. Experimental results demonstrate that our approach significantly improves tool retrieval accuracy and overall reasoning performance across multiple reasoning tasks. Furthermore, our method shows enhanced scalability compared with baselines as the number of question-specific increases.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes a multi-agent framework that automatically generates and aggregates tools into Python libraries. The authors evaluate on three in-domain and one out-of-domain dataset spanning math, medical, and science, and include module-level ablations and an error analysis. The method improves tool-retrieval accuracy, particularly as the tool inventory scales. Consolidating isolated tools into higher-level abstractions is meaningful and using Python library format is a practical choice.

### Strengths
1. The core idea—consolidating fragmented tools into higher-level Python libraries—is clear and meaningful.

2. Effectiveness is shown across multiple models and benchmarks (including OOD).

3. Includes an error analysis rather than only headline metrics.

4. The case for “aggregated library is better than fragmented toolset” is convincingly demonstrated under large tool inventories).

5. Ablations are provided for key modules, helping attribute where gains come from.

### Weaknesses
1. Retrieval ablations can be deeper. Only one retrieval model is used; no comparisons across alternative embedders/rerankers. No quantitative analysis of how retrieval quality degrades as the tool pool scales.

2. The interaction loop appears single-round (correct me if I am wrong); multi-turn tool use (e.g., Claude-Code) may be more suitable with higher performance and mitigate the limitation of tool retrieval accuracy.

3. Even after aggregation, the resulting tool library remains very large; the degree of consolidation (how much redundancy is removed / how compact the library becomes) is not analyzed.

### Questions
1. In many cases, direct code generation may be preferable for advance coding models. How do the authors think of using these generated tools compared with directly generating code?

2. How do you measure the degree of aggregation (consolidation ratio, redundancy removed)? Can the library be further aggregated, and what criteria would govern additional merges?

3. The SFT gain is small; could you break down pre/post SFT by error type and task (what improved, what didn’t)? Which failure modes in your ablation are most amenable to SFT, and which seem unaffected? What type of errors in your ablation study can be effectively solved using SFT?

### Soundness
3

### Presentation
3

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
This paper proposes ToolLibGen, a framework that automatically refactors unstructured collections of tools into structured tool libraries. The method aims to address the retrieval bottleneck that arises when thousands of automatically generated question-specific tools become fragmented and difficult to manage. The framework consists of three major stages, including CoT-based question-specific tool generation from QA data samples, hierarchical tool clustering, and tool aggregation based on a multi-agent coding-reviewing pipeline. Experiments on scientific, mathematical, and medical reasoning tasks and three seed LLMs (GPT-4.1, GPT-oss-20B and Qwen3-8B) show improvements on QA performance over baseline methods including CoT, PoT, fragmented and clustered toolset management, and KTCE.

### Strengths
- The research direction of multi-agent tool aggregation and structured tool management is valuable for improving tool usage efficiency when scaling up agentic LLM systems.
- ToolLibGen demonstrates consistent improvements across multiple datasets and foundation LLMs.
- The experimental analysis is comprehensive, where the retrieval scalability curves, error analysis and additional study on supervised fine-tuning framework provide solid diagnostic insight.

### Weaknesses
- The creation of tool libraries of ToolLibGen requires a large amount of extra computing cost, and it is hard to keep evolving if new types of problems emerge. It is unclear whether ToolLibGen’s benefits could be naively replaced or surpassed by directly fine-tuning LLMs to create question-specific tools and the solutions with the tools, where instead of retrieving tools from a pre-built fixed library, LLMs are trained to more flexibly create dynamic tools for problem solving at test time.
- The proposed framework requires non-trivial prompt engineering, which limits its generalization to non-programmatic tools without deterministic arithmetic solutions or algorithms, such as search engines.
- The tool library quality created by ToolLibGen heavily relies on the prompted LLMs that conduct the tool clustering and reviewing, whose reliability may be questionable especially on unseen problems.
- The paper writing quality is poor, which includes unpolished duplicated paragraphs (the third and fourth paragraphs in Section 2.4).

### Questions
- Is there any efficiency measure to show the computational cost of ToolLibGen at tool library creation and inference phases? It would be better to compare the efficiency of ToolLibGen with baseline prompting and tool-augmented methods.
- In the coding-reviewing tool aggregation step, what is the rate of reaching the maximum number of iterations, meaning that the aggregation may fail? and when this happens, how many tools are still uncovered by the aggregation?
- Any human evaluation to more quantitatively verify the quality of tool creation, clustering and aggregation of ToolLibGen?

### Soundness
2

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
The paper proposes a method to aggregate tools in agentic large language model systems. The number of tools used by the systems are heavily increasing, and it is becoming a bottleneck for the models to parse through the large collection of tools to use them effectively. This paper proposes a solution by aggregating the tools into semantically coherent clusters. Empirically, this method improves tool retrieval accuracy and overall reasoning performance.

### Strengths
The only strength of this paper is their proposed method of using aggregated tools outperforms individual tools.

### Weaknesses
- To solve the problem, authors themselves first "invent" the problem by generating a bunch of "question-specific" tools. This makes the problem simpler as authors themselves can define how simple each tool can be, and define an agent which can effectively aggregate the simple tools. For instance, a tool for a Question-CoT pair can be easily hallucinated by the LLM as a copy only tool, and the aggregator can just aggregate a bunch of copy tools and parameterize them. A proper evaluation would have required the authors to use existing tools from a dataset/setup and then attempt to aggregate it.
- The method only works with python functions as tools, as it employs a coding agent which refactors multiple python functions into a single function. This method is not generalizable to tools from other programs.
- The approach uses an LLM to perform clustering of the tools (reaching to a "tool-ception" moment). Interestingly, the authors set the tree depth of the clustering process to 4, thereby making the evaluation favorable to their setup. In Figure 3, authors show retrieval accuracy improves with the aggregation, which is quite obvious as the number of tools to call reduces in the aggregated setup. However, this depends crucially on the tree-depth - higher numbers would likely generate more clusters, leading to less utility of this setup.

### Questions
- which model exactly is used for the coding agent, which aggregates the tools?

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
4

### Summary
The paper introduces TOOLLIBGEN, a scalable framework that automatically refactors fragmented, question-specific tools generated from LLM reasoning traces into a structured Python library. It employs hierarchical clustering to group functionally related tools and a multi-agent aggregation system to iteratively merge and validate tools without losing functionality.

### Strengths
- The empirical results show the effectiveness in generalization to unseen datasets, which is critical when scaling tool sets in real-world applications

### Weaknesses
- Lack of novelty in the core idea or methodology. The core idea of using Python functions as a toolset to aid reasoning is proposed in other previous works [1,2,3]. While the paper successfully showed that having a more generalizable toolset indeed leads to better performance in reasoning tasks, the claim has already been shown in previous papers [3,4,5]. The authors need to show empirical results to distinguish between these works
- Lack of competitive baselines. The only competitive baseline is KTCE [3]. Need more competitive baselines such as [4] and [5]. 


[1] Qian et al, CREATOR: Tool Creation for Disentangling Abstract and Concrete Reasoning of Large Language Models. EMNLP 2023. \
[2] Yuan et al, CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets. ICLR 2024. \
[3] Ma et al, Automated Creation of Reusable and Diverse Toolsets for Enhancing LLM Reasoning. AAAI 2025 \
[4] Wang et al, TROVE: Inducing Verifiable and Efficient Toolboxes for Solving Programmatic Tasks. ICML 2024 \
[5] Stengel-Eskin et al, ReGAL: Refactoring Programs to Discover Generalizable Abstractions. ICML 2024

### Questions
Address the above questions.

### Soundness
3

### Presentation
3

### Contribution
2
