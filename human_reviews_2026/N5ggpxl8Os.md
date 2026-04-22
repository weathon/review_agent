# MEHGT-LKG: Multimodal Edge-enhanced Heterogeneous Graph Transformer with LLM-driven Knowledge Graph for Stock Trend Prediction

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Stock trend prediction plays a central role in optimal investment decision-making, and has attracted extensive research from both investors and institutions. Although recent studies have employed graph structures to model the complex relationships among financial entities, the corresponding models fail to efficiently capture semantically rich edge features across heterogeneous entities, thereby limiting the ability to fuse and align multimodal data such as market indicators, financial events, and heterogeneous graph structure. Therefore, in this paper, we propose a Multimodal Edge-Enhanced Heterogeneous Graph Transformer with LLM-driven Knowledge Graphs (MEHGT-LKG) for stock trend prediction. Specifically, we first fine-tune a large language model (LLM) by using instruction tuning datasets to design a financial event-centric knowledge extraction agent (FinEX). Subsequently, we encode the structured tuples generated from FinEX into financial event-centric knowledge graphs (FEKGs) and then construct multimodal heterogeneous graphs by incorporating multimodal information. Finally, we design a Multimodal Edge-Enhanced Heterogeneous Graph Transformer (MEHGT) to fully encode a series of semantically enriched multimodal heterogeneous graphs spanning different time horizons. MEHGT models edge-level features through type-specific encoders and integrates them into both multi-head attention and message propagation, significantly enriching the representation of relational semantics and target nodes. Extensive experimental results and trading simulations on multiple real-world datasets demonstrate the superior performance of the proposed approach beyond other state-of-the-art models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a study focused on stock trend prediction by constructing and leveraging a heterogeneous knowledge graph centered on financial events. Its main contributions include fine-tuning a LLM for financial event extraction from diverse text sources and proposing a novel method, MEHGT, which explicitly incorporates edge features into both the attention mechanism and message passing process within the graph structure. The work outlines a comprehensive pipeline from information extraction and graph construction to representation learning.

### Strengths
1. The paper focuses on the commonly overlooked feature information of edges in heterogeneous knowledge graphs and utilizes it to address the reliance on financial events and capital flow information in stock trend prediction.

2. The paper fine-tuned an LLM specifically designed for extracting financial events from multiple textual sources, which appears to contribute to data processing in the financial field.

3. The paper proposes the MEHGT method, which fully leverages edge information in heterogeneous graphs, providing significant supplementary information for predicting stock market trends.

4. The workflow proposed in the paper encompasses a complete process of information extraction, graph construction, and representation prediction, offering practical reference value for real-world applications.

### Weaknesses
Though the paper demonstrates a complete process for the stock movement prediction task, the following weaknesses should be concerned for further improvements.

1. The paper constructs a financial event-centric knowledge graph based on FinEX, but lacks more detailed descriptions such as statistical information about this knowledge graph. Additionally, if the events extracted by FinEX can be tuples of dual entities or single entities, how are they uniformly stored in the graph database?

2. The paper claims to have constructed a multimodal heterogeneous graph, but the multimodality is only reflected in the multiple sources of information, rather than in the different modalities of node and edge representations in the graph. Moreover, the MEHGT method does not include specialized processing or encoding for different modalities. I believe the authors' claim of multimodality is somewhat far-fetched.

3. When introducing the MEHGT method, the paper proposes explicitly integrating edge features into both the attention computation and message passing processes. However, it lacks necessary explanation for the motivation behind integrating them into both computational processes. Additionally, there is no clear analysis of the individual impact and contribution of integrating edge features into attention computation and message passing on the overall performance.

4. The performance comparison experiments in the paper were conducted on specific stocks, rather than evaluating the overall performance on a large number of stocks across market sectors. What is the reason for this? Why were these particular stocks selected as evaluation targets? Can evaluations on specific stocks represent the model's high predictive performance on other stocks?

5. In the experimental section, the MEHGT method achieved significant performance improvements on some datasets, but it did not show performance gains on datasets such as EVE and Sungrow. The reasons for these performance differences across datasets were not clearly analyzed.

6. The method proposed in the paper integrates edge information into the graph transformer, which appears to be a general approach for the representation of heterogeneous graphs. However, the text lacks a strong connection between the design of this method and the task of stock architecture trends, or it lacks an analysis of the transferability of this method in other domains.

7. The title format of the subgraphs in Figure 6 is problematic.

### Questions
See the questions in the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the challenges of multimodal fusion and relational modeling in stock trend prediction by proposing a Multimodal Edge-enhanced Heterogeneous Graph Transformer with LLM-driven Knowledge Graphs (MEHGT-LKG). The approach enhances semantic understanding through LLM-constructed event graphs, deepens relational modeling via edge-enhanced graph attention and message passing mechanisms, and incorporates sliding windows for dynamic temporal modeling. Extensive experiments on real-world datasets demonstrate that our method outperforms existing state-of-the-art models in both prediction accuracy and simulated returns.

### Strengths
1.	The paper presents a novel paradigm by integrating LLMs with heterogeneous graph networks to construct dynamic financial event knowledge graphs for stock prediction.

2.	The proposed MEHGT model's key contribution is its deep integration of edge features into the graph attention and message-passing mechanisms, effectively enhancing relational modeling.

3.	The figures are sufficient and intuitive, providing clear support for the proposed framework and methodologies.

### Weaknesses
1. Limited experimental evidence: The model fails to demonstrate consistent superiority over baseline methods across key metrics in Table 1. The ablation study in Table 2 shows questionable results (particularly the Sungrow case) that contradict the expected outcomes, casting doubt on the framework's effectiveness.

2. Unfair comparison: The computational advantage of baseline methods is ignored in comparisons. No analysis is provided regarding the trade-off between performance improvements and the substantial computational overhead introduced by the LLM components.

3. Reproducibility issues: Critical implementation details including training time, hardware requirements and memory usage are missing, making it difficult to reproduce this computationally intensive framework.

4. Insufficient generalization validation: Experiments are confined to Chinese A-shares market. The model's robustness remains unverified in different market environments (e.g., US/HK stocks) or under extreme market conditions.

5. Presentation Issue in Figure 6: There appears to be a formatting error in Figure 6, where the labels for subfigures (a)-(c) are not fully displayed, affecting the clarity of the presentation.

### Questions
See the weaknesses.

Other questions:
1. The baseline selection appears limited, with only MAC (2023) and MDGNN (2024) representing recent works. Were other contemporary methods considered for comparison?

2. Has the transferability of the FinEX module been evaluated? Could it function as a plug-and-play component in other graph architectures, particularly homogeneous graphs?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes MEHGT-LKG, utilizing LLM-driven knowledge extraction for graph-based stock trend prediction. Its three-stage framework includes: (1) fine-tuning the FinEX to extract structured financial relations from text; (2) constructing event-centric multimodal heterogeneous graphs; (3) designing MEHGT to explicitly embed edge-level features into attention and message-passing via type-specific encoders. Experiments on CSI datasets show MEHGT-LKG outperforms most but not all baselines, with ablation studies and trading simulations validating component efficacy.

### Strengths
1. The instruction-tuned FinEX agent addresses financial text mining challenges by extracting dual-format structured data and validating via domain experts. The effectiveness of FinEX is further verified by experiment results.
2. Innovative edge-centric MEHGT design. By embedding edge-level features into attention and message-passing via type-specific encoders, MEHGT addresses node-centric GNN limitations.
3. The paper includes comprehensive comparison experiments and ablation studies.

### Weaknesses
1. Some experimental results show the proposed model underperforms compared with comparision methods (e.g., MDGNN). Additionally, there is an error in labeling the best experimental result for Inspur in Table 1.
2. All experiments are conducted on Chinese stock market during. The lack of analysis on the model’s transferability to other markets or performance during volatile periods limits its generalizability.

### Questions
Please refer to Weaknesses for details.

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
This paper introduces MEHGT-LKG, which first employs the LLMs to construct a multi-modal heterogeneous financial event graph, then applies the edge-enhanced HGT (Heterogeneous Graph Transformer) to fuse the cross-modal information, and finally, the fused information is used to perform stock trend prediction. Both experimental results and trading simulation experiments demonstrate the effectiveness of the proposed method.

### Strengths
1. This paper is well-written and easy to follow.
2. The idea of incorporating LLMs to construct an event graph for external knowledge augmentation is interesting.
3. The experiments are extensive and can show the effectiveness of the proposed method.

### Weaknesses
1. The novelty of the proposed method seems to be limited, as the proposed method lacks technical contribution. Exploiting LLMs to construct financial event/knowledge graphs for financial investigation has been largely studied by existing works [1, 2, 3]. Moreover, the proposed information fusion mechanism is also similar to the work in the edge-enhanced heterogeneous graph Transformer. All these mechanisms resemble existing works and do not introduce principled advances in LLM-based stock prediction or heterogeneous graphs.
2. I am very concerned about the process of LLM instruction tuning, as this paper missed so many details for this stage. For example, how is the instruction dataset constructed? Where do the news articles and triplets used as supervision signals come from? What is the dataset’s scale? Are all samples generated by GPT-4 or partially collected from external sources? How do the authors ensure the correctness and reliability of the generated content by LLMs? These details are critical for assessing the soundness and fairness of the instruction-tuning process.
3. The baselines, especially the time series modeling methods, are extremely outdated. More recent baselines should be compared to validate the effectiveness of the proposed methods.
4. Many figures, such as Figure 5 and Figure 7, are difficult to interpret. It is unclear what specific insights or advantages these visualizations provide.
5. Providing a theoretical discussion or analysis can better support the proposed method,  which would substantially strengthen the depth of the work.

[1] Dynamic graph construction via motif detection for stock prediction  
[2] Modeling Interactions Between Stocks Using LLM-Enhanced Graphs for Volume Prediction  
[3] LLM-Augmented Enhanced Graph Transformer for Stock Movement Prediction

### Questions
See weaknesses

### Soundness
3

### Presentation
4

### Contribution
3
