# Deep Sparse Latent Feature Models for Knowledge Graph Completion

- Decision: Reject
- Scores: 5, 5, 6, 3

## Abstract
Recent progress in knowledge graph completion (KGC) has focused on text-based approaches to address the challenges of large-scale knowledge graphs (KGs). Despite their achievements, these methods often overlook the intricate interconnections between entities, a key aspect of the underlying topological structure of a KG. Stochastic blockmodels (SBMs), particularly the latent feature relational model (LFRM), offer robust probabilistic frameworks that can dynamically capture latent community structures and enhance link prediction. In this paper, we introduce a novel framework of sparse latent feature models for KGC, optimized through a deep variational autoencoder (VAE). Our approach not only effectively completes missing triples but also provides clear interpretability of the latent structures, leveraging textual information. Comprehensive experiments on the WN18RR, FB15k-237, and Wikidata5M datasets show that our method significantly improves performance by revealing latent communities and producing interpretable representations.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a sparse latent feature model that is optimized via a deep variational autoencoder to achieve interpretable completions for knowledge graphs. Experimental results verify the effectiveness of the proposed model.

### Strengths
1. This paper is well-written and easy to follow
2. Many experiments are conducted to demonstrate the effectiveness of the proposed method.

### Weaknesses
1. My biggest concern is the motivation of the proposed method. The authors claim that existing text-based methods overlook the complex interconnectivity among entities, however, there are some GNN-based TKGC models such as SEA-KGC [1] that can model graph information, what's the superiority of the proposed method compared with GNN-based methods? Moreover, given the text descriptions of the entities, many text-attributed graph embedding methods can also be applied to complete knowledge graphs, such as GLEM [2] and GraphFormers [3]. It's interesting to see their performance and further analysis of is there any unique challenges of applying these TAG models for text-based KGC.

2. Recently some methods have proposed to use LLMs to understand the text semantics within TKGs to perform completion or reasoning tasks (e.g., CSProm-KG [4] and KICGPT [5]), which have gained good performance and generalization ability across different KGs. The authors should provide more comparison and analysis for these methods since they also use text information within KGs.


[1] Unifying Structure and Language Semantic for Efficient Contrastive Knowledge Graph Completion with Structured Entity Anchors

[2] Learning on Large-scale Text-attributed Graphs via Variational Inference

[3] GraphFormers: GNN-nested transformers for representation learning on textual graph

[4] Dipping PLMs Sauce: Bridging Structure and Text for Effective Knowledge Graph Completion via Conditional Soft Prompting

[5] KICGPT: Large Language Model with Knowledge in Context for Knowledge Graph Completion

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposes a novel deep sparse latent feature model for knowledge graph completion. It is optimized through VAE, with the goal of uncovering the latent structures in KGs and completing missing triples. Experiments demonstrate that this approach outperforms existing methods in terms of both performance and interpretability.

### Strengths
Strengths  
1. Experimental Results: The model shows significant performance improvements on the Wikidata5M dataset (e.g., a 5% increase in MRR and a 6.5% increase in Hit@1), with similar results observed on the WN18RR dataset.  
2. Scalability: The VAE framework enables the model to perform inference on large-scale knowledge graphs, demonstrating good scalability.

### Weaknesses
Weaknesses
1. Model motivation: SBM is well-known graph clustering algorithm. There are many works that combine GNN with SBM to achieve community detection. Except from the used data structure, the key difference between the proposed method and existing works is not obvious.   
2. Model Complexity: The introduction of various latent feature sampling and inference mechanisms makes the inference process relatively complex, necessitating more efficient training strategies to speed up training and reduce memory usage.  
3. Limited Effectiveness: As noted in the paper, the proposed approach is primarily effective for sparse latent features, with limited performance in dense scenarios. Furthermore, some clustering-based KGE methods should be included in compared baselines.

### Questions
1. How to determine the number of clusters in a KG? 
2. How to find the specific meaning represented by each cluster?

### Soundness
2

### Presentation
2

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
This paper introduces DSLFM-KGC, a novel framework of sparse latent feature models for KGC. The framework leverages stochastic blockmodels (SBMs) and deep variational autoencoders (VAEs) to capture latent community structures in KGs and improve link prediction. Experimental results show the effectiveness of DSLFM-KGC.

### Strengths
1.The paper is clearly written and easy to follow.

2. The theoretical foundation of this paper is solid, utilizing extensive mathematical formulations to elucidate the structure of the proposed model or to demonstrate its validity.

3. The experiments conducted on benchmark datasets demonstrate the effectiveness of DSLFM-KGC in improving KGC performance and uncovering interpretable latent structures.

### Weaknesses
1.The paper repeatedly emphasizes the advantages of the proposed model on large-scale graphs; therefore, it would be beneficial to include comparative experiments on time complexity or runtime performance between the proposed model and the baseline.

2. At the end of the introduction section, a more direct listing of the contributions of this paper should be provided, with particular emphasis on the novel points introduced for the first time in this work.

3. The paper uses more baselines on WN18RR and FB15k-237 compared to Wikidata5M. Is this due to differences in dataset scale? The authors should provide a detailed explanation.

4. The primary parts in this framework, SBM and VAE, are derived from previous work, with extensive references to existing literature in the framework description. I am not entirely clear on the main innovations introduced by the authors in these two methods. It would be helpful if the authors could enhance the explanation of their contributions in the text or directly clarify these innovations in their response to me.

### Questions
Please refer to the "weaknesses" section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper addresses the limitations of recent text-based knowledge graph completion (KGC) methods that fail to adequately consider the complex interconnections among entities in large-scale knowledge graphs. It introduces a novel framework utilizing sparse latent feature models optimized via a deep variational autoencoder (VAE), which enhances link prediction and offers interpretability of latent structures by integrating textual information. Experimental results on multiple datasets demonstrate that the proposed method significantly improves performance by uncovering latent communities and generating interpretable representations.

### Strengths
1. The proposed method aims to balance the retention of critical knowledge with the elimination of redundancy, which is an interesting topic.
2. The authors not only effectively complete missing triples but also provide clear interpretability of the latent structures which seems reasonable.

### Weaknesses
1. The paper is not organized clearly, which is not friendly for understanding. For example, there is a lack of preliminary details for how to model MB and other module in 3.1 GENERATIVE MODEL. 
2. Figure 2 lacks of explanation, \textit{e.g.,} how the modules work together and match the equations in the main paper. The paper lacks the necessary reproduction file for the results. 
3. The paper lacks the analysis of time complexity as well as space complexity, which is necessary to study the efficiency of the model. 
4. The authors do not compare the model with other SOTA KGE methods, e.g.,[1][2][3]. The performance of, MRR in FB15K-237 is 0.36 while that of the proposed paper is 0.355. In this way, the performance of the proposed paper is not significant and the authors may better give a reasonable explanation.
[1] Compounding Geometric Operations for Knowledge Graph Completion
[2] Geometry interaction knowledge graph embeddings
[3] KRACL: Contrastive Learning with Graph Context Modeling for Sparse Knowledge Graph Completion

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2
