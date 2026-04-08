## Human Reviewer 1

### Summary
This paper addresses the challenge of knowledge boundary awareness in Retrieval-Augmented Generation (RAG) by introducing a novel knowledge probing approach that leverages the continuous hidden states of Large Language Models (LLMs) to enhance both effectiveness and efficiency. The key technical contributions are twofold: first, a confidence detection model that quantifies how retrieved context boosts the LLM's confidence, which is used to fine-tune a reranker for more effective context selection; and second, CBDR (Confidence-Based Dynamic Retrieval), a mechanism that adaptively triggers retrieval based on the LLM's initial confidence to avoid unnecessary searches. Experimental results demonstrate that this integrated framework achieves a significant 5.6 percentage point (pp) increase in RAG accuracy while concurrently reducing retrieval costs by 7.1 pp, thereby substantially improving practical utility without compromising performance.

### Strengths
1. The paper is well-organised and presented so that it is quite easy for the audience (even for those with limited knowledge of RAG) to follow the paper with a clear blueprint of the proposed method of key components and their functionality in mind.

2. The research is well motivated, and the paper targets the most prominent research problems in the RAG field, i.e., the knowledge boundary problem for efficient and effective RAG.

3. I appreciate the  DISCUSSION section, which cleared many of my questions and helped me get a deeper understanding of the method.

### Weaknesses
1. The idea of using a model's latent representation to compute a confidence score is not new, for example:
https://arxiv.org/pdf/2405.18727
2. The general idea of utilising confidence is not a novel idea, making the whole paper less interesting.

### Questions
1. Will the reranker be model-specific? Will it require retraining the reranker if the LLM is changed? Retraining the reranker is time-costly, especially for the data gathering process.
2. From the introduction, it seems that the target for this paper is to address the knowledge boundary problem of the RAG system, i.e., mainly the question of ``when to search'', which is addressed by the proposed CBDR mechanism (although similar heuristics with the threshold as the indicator can be widely seen from the existing papers). But to the reviewer, the main contribution is to address the efficiency of the RAG system (addressed by training a preference-based reranker, which is bound to target LLM). Therefore, the core contribution of the paper is to some extent misaligned with the main research question.
3. The general idea of the proposed method is quite similar to the paper below:
https://arxiv.org/pdf/2405.18727

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
2

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper proposes a confidence evaluation based approach to improve RAG systems. The method leverages LLM internal hidden states to: (1) construct a confidence detection model that quantifies how retrieved contexts enhance model confidence, (2) build a preference dataset (NQ_Rerank) based on confidence shifts to fine-tune rerankers, and (3) introduce Confidence-Based Dynamic Retrieval (CBDR) to adaptively trigger retrieval based on initial confidence. Experiments on Natural Questions and HotpotQA datasets show improvements in reranking accuracy (5.19%), end-to-end RAG accuracy (4.70%), and retrieval efficiency (7.10% cost reduction with 5.60% accuracy gain).

### Strengths
The introduced Confidence Shift metric is intuitive and understandable. 
 
Experimental setup, despite several issues that I will describe in the next section, is well-defined: multiple datasets and metrics are considered. Several reranker models are also considered.

### Weaknesses
First and foremost, the overall novelty of the paper is modest and limited. It is obvious that the work heavily relies on prior work by Ni et al. (2025) and it is explicitly stated that “paper’s core innovation is a novel preference metric”. This, while being an important novel contribution, stands as an incremental novelty. 

It remains unclear, why do the authors consider only two not state-of-the-art LLMs at-the-time: LLama 3 8B and Qwen 2.5 7B. At the time of the submission, a newer and more capable family of Qwen 3 models were actually available. It would be beneficial to see these models in the submission. 

Moreover, the scope of the LLMs considered is narrow with respect to model size: while it is challenging to evaluate 70B models and modern MoE models (Qwen 235B, GLM 4.5-4.6), there are several smaller (Llama 3.2 1B) and only slightly bigger models (Gemini 3 12B). Next, there are several hybrid architectures available that could also be tested (e.g phi-4-mini-instruct). 

The proposed method itself introduces a noticeable computational overhead, which is not studied and discussed in the submission. It seems like the proposed method, compared to the standard RAG pipeline, introduces a multiplicative computational overhead: while for original RAG it takes only 1 forward pass to process the input query, the proposed method requires N forward passes where N is the number of retrieved contexts.

### Questions
In 158-159 you state that layer 2 is picked for hidden state extraction. Why exactly layer 2? Did you experiment with other layers? 

There is no discussion and comparison of the computational overhead of your method. While it is clear from the description that such overhead is present. 

In 4.1 in the Datasets subsection you describe the training set for the confidence model E: 1k samples per negative and positive examples, respectively. Have you experimented with different set sizes? What is the minimum dataset size required for convergence? Is 50/50 negative/positive partition of the training data necessary? 

In 5.2 you state that the fine-tuned reranker “shows no statistical significance” when used with Qwen 2.5 7B - does it mean that the reranker is highly dependent on the target model? If so, is there a way to make the reranker model-agnostic? If not, I strongly suggest adding the aforementioned weakness as a limitation and include this statement in the paper. 
 
In Table 2 `bge-reranker-v2-m3 ` outperforms yours fine-tuned `bge-reranker-v2-m3-ft` on the NQ dataset, which is counter-intuitive. Can you elaborate on that? 

In Table 3 you test only three different $\beta$ threshold values. Why only these? Can you experiment with lower $\beta$ values? What is the minimum threshold?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper investigates how to effectively quantify and exploit the parametric knowledge boundaries of LLMs for improving RAG effectiveness and efficiency. The authors propose a confidence-shift metric derived from LLM hidden states to rerank retrieved contexts, fine-tune a reranker on the synthetic NQ_Rerank dataset, and introduce CBDR to gate retrieval adaptively. The confidence-shift formulation is clearly presented.  Experiments on NQ and HotpotQA validate the anticipated behavior, showing accuracy improvements with reduced retrieval cost.

### Strengths
1. Clarity of presentation: The overall narrative is coherent, with well-motivated objectives and a logically developed framework.
2. Clear methodological formulation: The confidence-shift framework is precisely defined that facilitates understanding.
3. Comprehensive discussion and analysis: The paper provides thorough discussions with detailed analyses of confidence variation, offering valuable insights into how confidence signals guide retrieval.

### Weaknesses
1. Limited coverage of related work: The discussion of related work is insufficient. For example, while Section 2.1 discusses the issue of neglecting potential conflicts between external and internal knowledge, the citation coverage is incomplete—it overlooks several closely related studies (e.g., Parenting[1] ).
2. Limited experimental validation:
	•	The paper would benefit from more extensive experiments. In particular, it lacks evaluation in different hyperparameter settings, and no ablation studies of individual components are provided. 
	•	Limited performance gains in some settings: The method shows modest improvements in certain cases;  for example, in Table 3, Qwen2.5-7B-Instruct on NQ Top-1 and Llama3-8B-Instruct on HotpotQA Top-1 do not outperform other methods.
	•	Minor issues in presentation and details: Some presentation details need improvement. For example, there are typos (e.g., “a taget LLM” in line 441), and the notation for highlighting results is not explained (e.g., bold for best, underline for second best).
	
[1] Xu Y, Zhang R, Jiang X, et al. Parenting: Optimizing Knowledge Selection of Retrieval-Augmented Language Models with Parameter Decoupling and Tailored Tuning[J]. CoRR, 2024.

### Questions
1. Providing pseudocode: Including pseudocode could help readers better understand the proposed method and clarify its implementation details.
2. Comparison with existing adaptive retrieval methods: Since the proposed method features adaptive retrieval, it would be helpful to compare its performance with existing adaptive retrieval approaches (e.g., Adaptive RAG[3], Probing RAG[4]) to better evaluate its adaptive retrieval capability.
3. Additional end-to-end comparisons: It would be valuable to compare the proposed method with approaches mentioned in the related work, such as DPA-RAG and SEAKR, under the end-to-end experimental setting.

[3] Jeong S, Baek J, Cho S, et al. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity[C]//Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 2024: 7029-7043.

[4] Baek I, Chang H, Kim B J, et al. Probing-RAG: Self-Probing to Guide Language Models in Selective Document Retrieval[C]//Findings of the Association for Computational Linguistics: NAACL 2025. 2025: 3287-3304.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
5