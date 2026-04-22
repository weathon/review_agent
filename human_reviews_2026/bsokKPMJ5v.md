# STAR : Semantic-ID Token-Embedding Alignment For Generative Recommenders

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 4

## Abstract
Generative recommenders (GRs)—which directly generate the next-item semantic ID with an autoregressive model—are rapidly gaining adoption in research and large-scale production as a scalable, efficient alternative to traditional recommendation algorithms.  Yet we find a fundamental failure mode when adapting Language Models (LMs) to GRs. We identify, for the first time, a pervasive token–embedding misalignment issue: the common mean-of-vocabulary initialization places new Semantic-ID tokens on the LM manifold but collapses their distinctions, stripping item-level semantics and degrading data efficiency and retrieval quality.  We introduce **STAR**, a lightweight alignment stage that freezes the LM and updates *only* Semantic-ID embeddings via paired supervision from item titles/descriptions ↔ Semantic-ID, thereby injecting the new tokens with linguistically grounded, item-level semantics while preserving the pretrained model’s capabilities and the primary recommendation objective.  Across multiple datasets and strong baselines, **STAR** consistently improves top-*k* retrieval/search performance over mean-of-vocabulary initialization and status-quo auxiliary-task adaptation.  Ablations and analyses corroborate our claims, showing increased token-level diversity, stronger linguistic grounding, and improved sample efficiency.  **STAR** is parameter-efficient, updating only the Semantic-ID token embeddings ($|\mathcal{V}_{\mathrm{SemID}}|\times D$ parameters), and integrates seamlessly with standard GR pipelines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
Aiming at the token–embedding misalignment issue of semantic indices, this work proposes STAR, a lightweight alignment stage, to provide a proper initialization for the newly introduced semantic tokens. During the proposed alignment stage, only the embedding matrix corresponding to semantic tokens is optimizable, while the other components of the backbone language model are frozen. Extensive comparison with several traditional recommenders and generative recommender on both Amazon and Yelp datasets demonstrate the effectiveness of STAR.

### Strengths
1. The proposed method is well-introduced and easy to follow.
2. The proposed method can improve the sequential recommendation performance, compared to the leading baselines.

### Weaknesses
1. My main concern lies in the novelty of STAR, since there is no fundamental difference between STAR and LC-Rec (which is an evaluated baseline in this work). The claimed contributions, such as freezing the backbone LM, lie more in the perspective of engineering, rather than technological innovation.
2. The effectiveness of STAR is not sufficiently convincing. As introduced in this work, previous practices of semantic token initialization, including random embedding or mean embedding over vocabulary, are not appropriate for generative recommenders. Hence, from my perspective, STAR can be regarded as a better method to initialize embeddings of semantic tokens, which is able to combine with several different generative recommender models. By comparing the original baselines adopting random or mean initialization with the variant that adopts STAR initialization, the evaluation is more convincing when demonstrate the effectiveness.
3. The analyses of running time and memory are missing, which is important to support the 'lightweight' characteristics of STAR.
4. In the experiment, if LC-Rec and STAR are implemented with the same LM backbone? Moreover, an investigation on the influence of different LM backbones is recommended.
5. Organization of this manuscript is a little chaotic, of which Figure 1 and Figure 2 lack corresponding illustration in the main text.

### Questions
Please refer to the Weaknesses.

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
3

### Summary
This paper addresses the token-embedding misalignment problem in generative recommender systems that adapt pre-trained language models for sequential recommendation. The authors identify that when Semantic-ID tokens (discrete identifiers produced by RQ-VAE) are integrated into a pre-trained language model's vocabulary, the common mean-of-vocabulary initialization collapses all new tokens into an undifferentiated region in the embedding space, stripping item-level semantics. To remedy this, the authors propose STAR, a lightweight alignment stage that freezes the language model backbone and updates only the Semantic-ID token embeddings through paired supervision (item descriptions → Semantic-ID sequences). Experiments on multiple datasets show consistent improvements in top-K retrieval metrics over strong baselines.

### Strengths
S1. Well-Motivated Problem with Clear Presentation
The paper identifies a practical issue in adapting language models for generative recommendation and presents it clearly with effective visualizations, making the problem accessible and the solution intuitive.

S2. Simple and Parameter-Efficient Method
STAR updates only $|V_{\text{SemID}}| \times D$ parameters ($\sim$0.8M, representing $\sim$0.13\% of full model), making it computationally efficient and easy to integrate into existing generative recommendation pipelines without architectural modifications.

S3. Comprehensive Experimental Validation
Experiments cover 9 datasets across multiple domains (Amazon, Yelp), both retrieval and search tasks, with consistent improvements over strong baselines (13-63\% gains). Ablation studies on data scaling and alternative designs strengthen the empirical contribution.

### Weaknesses
W1. Weak Theoretical Foundation and Potentially Misdiagnosed Problem.
The paper lacks theoretical justification for why "token-embedding misalignment" is the fundamental problem, rather than simply insufficient content information integration.

W2. Unfair Baseline Comparison and Missing Critical Ablations
STAR and LC-Rec are fundamentally similar—both use item descriptions to train Semantic-ID embeddings, differing mainly in training schedule (pre-train vs. joint) and parameter updates (frozen vs. full). The paper criticizes LC-Rec for "memorizing all items" but STAR does the same in its alignment stage.

W3. Unaddressed Scalability and Practical Deployment Issues
The paper introduces an extra training stage but reports no wall-clock time, convergence analysis, or FLOPs comparison. Critical practical concerns are ignored: (1) What if alignment and downstream stages use different LMs (e.g., Llama vs. Qwen)? (2) How to handle new items in dynamic catalogs without full retraining? (3) Cross-model generalization (e.g., align with 0.6B, deploy with 7B)? Experiments only use small models (0.6B) and datasets (max 5K samples), limiting generalizability to industrial-scale systems with millions of items and interactions.

### Questions
See weaknesses

### Soundness
2

### Presentation
3

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
This paper studies the initialization of semantic ID embeddings in LLM-based generative recommendation. The authors identify a key issue in existing approaches: embedding collapse in newly added semantic tokens, which prevents the recommender from fully utilizing the semantic priors of LLMs and reduces item-level distinctions. To address this, they propose a lightweight alignment method that learns token embeddings grounded in the existing vocabulary. Experiments on five real-world datasets show that the proposed method produces more informative and linguistically meaningful embeddings, improving both search and recommendation performance.

### Strengths
1. The paper focuses on semantic ID embedding learning, a fundamental problem in generative recommendation. The motivation is clear and relevant.

2. The authors identify a concrete limitation of existing initialization methods, where semantic token embeddings collapse and lead to suboptimal results.

3. The proposed lightweight alignment method is well designed and validated through experiments on five real-world datasets.

### Weaknesses
1. The comparison between standard sequential methods and language adaptation methods is not fully convincing. Although sequential models require more data and lack explicit semantics, they typically use smaller transformers (such as TIGER) and train much faster. Stronger theoretical or empirical evidence is needed to support the claimed advantages.

2. The preliminaries and diagnostic analyses overlook random embedding initialization, which could also mitigate embedding collapse.

3. The paper lacks sufficient discussion or empirical comparison with textual ID–based generative recommendation methods, such as IDGenRec [1].

4. The claims about generalization ability and sample efficiency are not well supported by the experiments. In addition, Figure 4 only shows distinctions among newly introduced token embeddings, which makes it difficult to verify that the aligned embeddings are linguistically grounded as claimed.

[1] Tan et al., Idgenrec: Llm-recsys alignment with textual id learning. SIGIR'24

### Questions
1. Do all LLM-based generative recommenders in the experiments share the same backbone (e.g., Qwen3-0.6B)? Consistent backbones are necessary for a fair comparison between random or mean initialization and the proposed STAR.

2. Is there any comparison of data efficiency between STAR and standard sequential models to substantiate the claimed improvement?

### Soundness
1

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
This paper introduces a simple yet effective approach for aligning semantic-ID token embeddings with the token space of large language models. By using the learned token embeddings to initialize semantic tokens, these tokens can be readily adopted for downstream tasks. Through comprehensive evaluation, the authors demonstrate the effectiveness of their method.

### Strengths
- The paper is well-written and well-organized.
- It poses and discusses a foundational question regarding the misalignment issue in token embedding spaces.
- The experiments are well-designed.

### Weaknesses
- This alignment/initialization may not be necessary for typical industrial generative recommenders, such as TIGER and OneRec, where both parameters and token embeddings are trained from scratch and no language tokens are included in the vocabulary.
- There are some typos. For example, in Line 289, "... we randomly pick five categories ...", but only four are listed. Additionally, in Table 1 (Line 334), "P5-SimID" should be "P5-SemID".
- The proposed alignment method appears to be similar to the mutual prediction alignment introduced in LC-Rec. The only apparent difference seems to be whether the model parameters are fixed or not.
- Although the proposed alignment strategy could be model-agnostic, the experiments are conducted solely on a single language model, Qwen3-0.6B. This limits the generalizability of the method.

### Questions
- There is a contradictory observation between this paper and the baseline LC-Rec. On one hand, this paper argues that "the primary performance gains of semantic alignment stem from injecting linguistic semantics into the new tokens rather than from broad backbone model adaptation." On the other hand, LC-Rec (Table IV) shows in its ablation study that all alignment tasks contribute to performance improvement. Why do alignment tasks aimed at broad backbone model adaptation provide no contribution or even a negative impact on model performance in your experiments?
- Why is LC-Rec inferior to P5-SemID and P5-CID in most cases in Table 1? Considering that LC-Rec uses SemanticID and applies alignment, does this observation suggest that the semantic alignment is less effective?
- What is the maximum item sequence length used for the sequential recommendation comparison?

### Soundness
3

### Presentation
4

### Contribution
3
