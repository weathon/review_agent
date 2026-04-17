# HierPromptLM: Hierarchical Prompt Language Model for Heterogeneous Text-rich Networks

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Representation learning on heterogeneous text-rich networks (HTRNs), consisting of multiple types of nodes and edges with each node associated with text data, is essential for various real-world applications. Given the success of pretrained language models (PLMs) in processing text data, recent efforts have integrated PLMs into HTRN representation learning, typically handling textual and structural information separately with PLMs and heterogeneous graph neural networks (HGNNs), respectively. However, this separation fails to capture critical interactions between these two types of data, and necessitates alignment between distinct embedding spaces, which is often challenging. To address this, we propose HierPromptLM, a novel pure PLM-based framework that models text data and heterogeneous structures without separate processing. First, we develop Hierarchical Prompt that employs prompt learning to integrate text data and structures at both node and edge levels, within a unified textual space. Built on this, two innovative HTRN-tailored pretraining tasks are introduced to fine-tune PLMs, emphasizing the heterogeneity and interactions between these two types of data. Experiments on HTRN datasets demonstrate HierPromptLM outperforms state-of-the-art methods, achieving significant improvements of up to 7.15% on node classification, 9.79% on link prediction, and 2.88% on graph classification. The codes are in https://anonymous.4open.science/r/HierPromptLM-code.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a pure-PLM framework for heterogeneous text-rich networks that turns node/edge structure into hierarchical prompts: node-level “graph-aware” prompts distilled from meta-path subgraphs, and edge-level relation-aware prompts with a learnable relation token; the PLM is then tuned with two pretraining tasks, HGA-NSP and HGA-MLM. On DBLP/OAG/GoodReads, it reports consistent gains for node classification and link prediction, and shows variants across PLM backbones.

### Strengths
1. The work unifies textual space for structure+text via hierarchical prompts avoids explicit alignment between PLM and HGNN spaces.

2. The paper give clear ablations isolate the contribution of each component (HGA-MLM/NSP, graph tokens, node-specific text).

3. The model shows good performance against the baselines. Across three datasets; the training-free “frozen” variant already beats baselines, and fine-tuning adds more.

### Weaknesses
1. The approach hinges on predefined meta-paths and a hand-crafted “subgraph program function.” The paper does not quantify sensitivity to meta-path choice/coverage or propose automated discovery, which might threaten robustness and portability across domains.

2. Although graph summaries are distilled into tokens, the pipeline still requires offline extraction/summarization and concatenation of multiple prompts; token-length pressure is acknowledged only qualitatively. There’s no empirical end-to-end accounting of preprocessing time vs. baselines beyond a brief complexity line and appendix pointer.

3. The metric for link prediction results does not align with the previous papers. The works such as HGT and Heterformer use MMR and NDCG for link prediction, but this work use ROC-AUC, PR-AUC and F1, which may not truly reflect the model 's capability.

### Questions
1. How sensitive are results to the chosen meta-path set (coverage/length)?

2. What are wall-clock times for (i) subgraph extraction, (ii) textualization, and (iii) token distillation per node, and how do they scale with graph size and |M|?

3. What is the average token length of graph-aware and relation-aware prompts per dataset? Are there any truncation or budget allocation strategy across meta-paths?

4. Can the authors report the link prediction results with MMR and NDCG?

5. My **major concern** is that, in an era when LLM is highly developed, addressing only the text-rich author-paper network tasks may not constitute sufficient technical contribution for acceptance. Hence, I wonder whether the proposed method can be applied to other types of heterogeneous networks or more complicate/difficult tasks.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes HierPromptLM, a novel framework designed for representation learning on heterogeneous text-rich networks (HTRNs). These networks are composed of nodes and edges of diverse types, with textual information associated with each node. The core idea behind the paper is to integrate textual and structural information within a unified representation space without relying on separate processing or alignment of embeddings, which has been a limitation in prior work.

### Strengths
1. The introduction of HGA-MLM and HGA-NSP as pretraining tasks tailored to HTRNs is a strong aspect of this paper. These tasks allow the model to capture both textual and structural characteristics effectively, which could potentially be applied to other domains with heterogeneous graph data.
2. The experiments presented are thorough, demonstrating clear improvements over existing methods, especially in the case of node classification and link prediction. The paper also provides insightful comparisons with multiple baselines, including those that only leverage text or graph structures.

### Weaknesses
1. The process of generating hierarchical prompts and integrating them with node textual information could become computationally intensive as the size of the network increases. This scalability issue may limit the framework's applicability to massive real-world networks, where efficiency is paramount.
2. The performance of prompt-based models is known to be sensitive to the formulation of prompts. Minor changes in prompt phrasing can lead to significant variations in model outputs. This sensitivity raises concerns about the robustness of HierPromptLM, especially when deployed in dynamic environments where prompt structures might evolve or be subject to adversarial manipulations. Ensuring consistent performance despite such variations is a challenge that requires further investigation.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposed a framework called HierPromptLM. A framework that combine pretrained language models like BERT and Graph structured data. 

The method of this work consists of two major parts

1. The author designed meta-path based graph token distillation to extract graph information while avoid introducing too much tokens to PLMs

2. The author introduced two pretrain task called Heterogeneous Graph-aware Masked Language Modeling (HGA-MLM) and Heterogeneous Graph-aware Next Sentence Prediction (HGA-NSP) to adapt the model for graph structured data understanding. 
	1. The graph structure is encoded using a summarization with a BERT model based on  meta-paths of graphs. 

The baseline compared including wide variety of approach for learning heterogeneous graphs and text rich networks. The experimental results shows the approach outperformed pervious works and reached SOTA performance on downstream tasks including node classification and link prediction.

### Strengths
1. The paper is well written and easy to follow.
2. The figures are good and intuitive. 
3. The experiment and ablation study is comprehensive. 
4. The performance of the framework is good.

### Weaknesses
1. The approach is not really novel, the idea of pre-training on graph enhanced task, has already has been proposed in previous work like [1]. 

2. The motivation of graph token distillation is not clear.  As described in the paper, the approach is summarizing the meta path into text, and then encode the text into embeddings.  The reason for this is explained as " exceed PLMs’ token limits" in line 254, which can be resolved using more recent model like ModernBERT[2] or Qwen-Embedding[3] series. 

3. Dataset like DBLP and OAG  are on scientific domain, domain specific backbone, like  SciBERT[4], is not adopted.

[1] Patton: Language Model Pretraining on Text-Rich Networks

[2] Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference

[3] Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models

[4] SciBERT: A Pretrained Language Model for Scientific Text

### Questions
1. What is the motivation of graph tokens other than model limit ? Is it for efficiency or for performance ? If it is for efficiency, how efficient it compare with using full summaries ? If it is for performance, what is the improvement ? You may either use truncation on BERT/ALBERT, or using model that support longer context to demonstrate this.  
2. Refer to other points weakness part.

### Soundness
3

### Presentation
3

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
This paper presents HierPromptLM, a hierarchical prompt-based framework for learning representations on heterogeneous text-rich networks (HTRNs). It introduces graph-aware prompts to encode local structural information and relation-aware prompts to capture edge semantics, enabling unified modeling of text and graph structure within the PLM. Similar to pretraining tasks in the BERT series, the authors design HGA-MLM and HGA-NSP to improve the model's structural-textual understanding. Experiments on multiple datasets show that HierPromptLM achieves improvements over existing graph and PLM-based baselines.

### Strengths
1.	The paper proposes a paradigm that encodes heterogeneous structures into PLMs via interpretable textual summaries and graph tokens, avoiding the conventional two-space alignment of HGNN + PLM frameworks.
2.	The paper presents a well-documented methodology, including pseudocode, complexity analysis, and comprehensive evaluations, while maintaining competitive memory and runtime performance on large HTRNs.

### Weaknesses
1.	The method heavily relies on predefined meta-paths for subgraph construction and summarization. This dependency raises concerns about generalization to new datasets, especially when suitable meta-paths are unclear or unavailable. It remains unclear whether an automated and robust alternative exists when domain-specific meta-paths cannot be manually defined.
2.	The paper introduces a subgraph program function to convert sampled subgraphs into textual summaries but lacks a detailed discussion of its sensitivity — including template design, handling of long texts (truncation or selection), and noise or redundancy reduction.
3.	The process of graph token distillation from the frozen PLM is only described at a high level, missing key implementation details such as the distillation algorithm, vectorization procedure, or token dimensionality. In addition, for HTRNs with a large number of relation types, the representational capacity of a single learnable relation token may be insufficient.

### Questions
1.	The baseline results in the tables show several standard deviations as 0. Could you clarify if the baseline implementations used the official implementations or if multiple runs were conducted for reproducibility?
2.	Why does the frozen prompt (non-fine-tuned) outperform the fine-tuned model on certain tasks? Which types of nodes/edges benefit the most from the frozen prompt? Are there any failure cases where the frozen prompt underperforms?

### Soundness
2

### Presentation
3

### Contribution
2
