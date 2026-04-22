# Catalog-Native LLM: Speaking Item-ID dialect with Less Entanglement for Recommendation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
While collaborative filtering delivers predictive accuracy and efficiency, and Large Language Models (LLMs) enable expressive and generalizable reasoning, modern recommendation systems must bring these strengths together. Growing user expectations, such as natural-language queries and transparent explanations, further highlight the need for a unified approach. However, doing so is nontrivial. Collaborative signals are often token-efficient but semantically opaque, while LLMs are semantically rich but struggle to model implicit user preferences when trained only on textual inputs. This paper introduces  Item-ID + Natural-language Mixture-of-Experts Language Model (IDIOMoE), which treats item interaction histories as a native dialect within the language space, enabling collaborative signals to be understood in the same way as natural language. By splitting the Feed Forward Network of each block of a pretrained LLM into a separate text expert and an item expert with token-type gating, our method avoids destructive interference between text and catalog modalities. 
IDIOMoE demonstrates strong recommendation performance across both public and proprietary datasets, while preserving the text understanding of the pretrained model.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel architecture for integrating collaborative filtering signals (item interactions) with LLMs. The authors argue that prior works suffer from semantic–collaborative entanglement when item IDs are added directly to LLM vocabularies, and their disentanglement largely improves recommendations while retaining general text reasoning.

### Strengths
1. The MoE-based separation of text and item experts is simple yet elegant, showing consistent performance gains on diverse datasets (small/large Amazon and industrial-scale).

2. With only FFN layers modified, this method has the potential to be applied to industrial-scale recommendation systems with low overhead.

3. The paper provides comprehensive ablations of architecture settings like MoE placement, static vs. dynamic routing, and expert shrinkage.

### Weaknesses
1. The backbone models are small (0.5B and 1.5B), so it's questionable whether the effectiveness can scale to larger models.

2. The method has limitations on newly-coming items, which require extending the vocabulary and re-training.

3. The idea of assigning different modalities to different experts to reduce modality interference has already been proposed in the domain of VLM / MLLM [1,2]. This paper directly uses this idea but doesn't mention it in related works.

[1]. Li, Yunxin, et al. "Uni-moe: Scaling unified multimodal llms with mixture of experts." TPAMI, 2025.

[2]. Shen, Leyang, et al. "Mome: Mixture of multimodal experts for generalist multimodal large language models." NeurIPS, 2024.

### Questions
1. How does IDIOMoE perform in interactive conversational settings (e.g., user dialogue or open-ended preference elicitation)?

2. Could the static routing scheme limit adaptability to unseen token types? How does the model handle cold-start items or dynamically changing catalogs?

3. In lines 121-122, it says previous generative recommendation methods "can forget collaborative structure if not carefully aligned with interaction data.". Can you show more details or evidence?

4. Despite using different experts, your method also integrates the two embeddings after FFN blocks. Can you explain why this is superior to other methods that directly add item tokens to LLM vocabularies (and with some alignment techniques)?

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
This paper aims to integrate item ids into large language models (LLMs) for recommendation systems. The authors propose IDIOMoE, a Mixture-of-Experts architecture that treats item interactions as a distinct dialect within the language space. The core idea is to route item ids to addtional experts (FFN) and text tokens to the original experts, which allows the model to leverage both collaborative filtering signals and natural language understanding without interference. The paper demonstrates that IDIOMoE outperforms text-only and item-only baselines on both public benchmarks and a large proprietary dataset, while maintaining the semantic understanding of the pre-trained LLM. The authors also provide extensive ablations to show that the improvements come from expert specialization and routing, rather than just added parameters.

### Strengths
1. **S1:Idea**: The idea of using a Mixture-of-Experts design to separate collaborative filtering from semantic processing is innovative. By assigning dedicated experts for item IDs and text tokens, the model can mitigate interference and preserve the strengths of both modalities.
2. **S2:Rigorous Evaluation**: The paper provides extensive evaluations on both public benchmarks and a large proprietary dataset compared with various advanced baselines, demonstrating the effectiveness of the proposed method. And authors also conduct extensive ablations and analysis to isolate the source of gains, showing that improvements arise from expert specialization and routing, and providing evidences why IDIOMoE can preserve the LLM's utility.

### Weaknesses
1. **Scaling of Parameters**: While the paper shows that the proposed method outperforms baselines, it is not clear how the performance scales with LLM's size.
2. **Ablation on MoE Architecture**: As is shown in Table4, MoA and MoT also achieve similar performance to IDIOMoE, but their parameter design on Attention instead of FFN, what's the advantage of using FFN for MoE instead of Attention? 
3. **Can IDIOMoE work with Various ID Types?**: As it shown that the proposed method works well with item IDs, can it also work with other types of IDs, like Semantic IDs in TIGER?
4. **Can IDIOMoE Reasoning in recomendation?**: The paper focuses on the integration of item IDs and text tokens, and it would be interesting to see how well IDIOMoE can handle reasoning tasks that require understanding of both item interactions and natural language queries. For example, can it answer questions about user preferences or make recommendations based on complex user queries that involve multiple items and attributes?

### Questions
Refer to Weaknesses and I can further raise my Rating if the authors can provide more results.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the key challenge of integrating collaborative filtering signals (item IDs) and natural language understanding in recommendation systems without mutual interference. It proposes IDIOMoE (Item-ID + Natural-language Mixture-of-Experts Language Model), a dual-expert architecture that treats item interaction histories as a "native dialect" within the language space. The core design modifies the Feed Forward Network (FFN) of a pretrained LLM into separate text and item experts, with static token-type gating to route text tokens to the text expert and item-ID tokens to the item expert. Experiments on public Amazon datasets (6 small-scale, 3 large-scale) and a proprietary industrial dataset (hundreds of millions of users) demonstrate that IDIOMoE outperforms classical recommenders (e.g., SASRec, BERT4Rec) and LLM-based baselines (e.g., P5-CID, CoVE) in metrics like NDCG@10, HR@10, and MRR, while preserving the LLM’s natural language competence. The paper’s key contributions include: (1) a disentangled MoE architecture for separating collaborative filtering and semantic processing; (2) robust performance at industrial scale with retained language understanding; (3) extensive ablations verifying gains from expert specialization rather than parameter scaling; (4) FFN key-value memory analysis revealing more interpretable and modular representations via expert disentanglement.

### Strengths
## 1. Originality.
- The paper introduces a novel MoE-based disentanglement paradigm for recommendation systems, which is the first to explicitly model item IDs as a "dialect" separate from natural language. This addresses the long-standing "knowledge entanglement" issue in prior LLM-based recommenders (e.g., CoVE, CLLM4Rec) that naively merge ID and text tokens.
- The static token-type gating design is a creative simplification of standard MoE (e.g., Switch Transformers), avoiding the inefficiency and instability of dynamic routing while achieving effective modality separation.

## 2. Significance.
- It establishes a new direction for LLM-recommender fusion by emphasizing "modality-specific specialization" over brute-force parameter scaling, inspiring follow-up work on disentangled multi-modal recommendation.

## 3. Rigorous experimental design.
- The study covers diverse datasets, comprehensive baselines (classical sequential models, LLM-based recommenders, capacity-matched non-MoE variants), and key ablations (expert capacity, MoE layer placement, static vs. dynamic routing).

### Weaknesses
## 1. Insufficient analysis of cold-start scenarios.
- Cold-start is a critical challenge for recommenders, but the paper only mentions item textual attributes as a potential solution without quantitative evaluation. For example: How does IDIOMoE perform on new items with only text descriptions (no interaction data)? And can the item expert leverage text-derived signals to mitigate cold-start bias?

## 2. Insufficient verification of recommendation explanation generation capability.
The paper claims that IDIOMoE retains the language ability of LLM to support "transparent explanations", but the experiments only verify the language understanding ability through general language benchmarks such as WikiText NLL and BBH, without conducting quantitative/qualitative evaluations on explanation generation in recommendation scenarios.

### Questions
## 1. Questions Related to Technical Mechanisms.
- Dynamic Adjustment of Item Expert Capacity: The paper observes that small-scale datasets (e.g., Amazon-Beauty) perform best with a smaller-capacity item expert (shrink=4), while large-scale industrial datasets require a full-capacity expert (shrink=1). However, it does not propose an adaptive capacity allocation algorithm that can automatically adjust based on data characteristics (e.g., item interaction sparsity, dataset size). Is it feasible to design such an algorithm?

- What is the performance of IDIOMoE on cold-start items (no interaction data, only text attributes)? Please provide quantitative results on cold-start subsets of the Amazon or industrial dataset, comparing with content-based and LLM-based cold-start methods.

## 2. Suggestions Related to Experimental Design.
- Validation of Recommendation Explanation Generation: The abstract claims IDIOMoE preserves LLM capabilities to support "transparent explanations," but experiments only verify general language understanding via metrics like WikiText NLL and BBH benchmarks. To validate this claim, the authors should supplement an "explanation quality evaluation" experiment.

### Soundness
3

### Presentation
2

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
This paper proposes IDIOMoE, an item-ID and natural-language mixture-of-experts transformer that routes ID tokens to a dedicated collaborative expert and text tokens to the frozen pretrained expert, enabling seamless recommendation within a generative LLM while preserving linguistic ability.

### Strengths
1. The paper introduces a novel ID/text mixture-of-experts architecture that treats item IDs as a native dialect, effectively disentangling collaborative and semantic signals.

2. Static token-type routing keeps implementation lightweight while preserving the pretrained LLM’s language capabilities.

3. The method is thoroughly evaluated against industrial-scale baselines on datasets, demonstrating consistent gains.

### Weaknesses
1. It seems that the rigid ID/text routing cannot jointly model attributes blended in user queries, leaving collaborative signals and side features separated. For exmaple, in queries like “white iPhone”, the tokenizer sends “white” to the Text-Expert and “iPhone" to the Item-Expert. How can the proposed method avoid such effects?

2. How can the proposed model handle new items? It seems that there is no clear zero-shot or cold-start mechanism is provided in the paper.

3. How efficient is the algorithm proposed in the article? From the model's perspective, it can only process one user's query at a time. If there are a large number of users making recommendation requests, can the model be effectively deployed and adapted?

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
3
