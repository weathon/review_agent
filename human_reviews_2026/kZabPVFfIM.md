# APlaud: Adaptive Personalized Low-Rank Decomposition for User-Specific LLM

- Decision: Reject
- Scores: 8, 2, 4, 4

## Abstract
In this paper, we introduce and study the problem of \textit{personalized survey response prediction} using fine-tuned large language models (LLMs). This task poses unique challenges: limited per-user training data, scalability of model storage, and the need to exploit shared survey structures. To address these issues, we propose \textbf{APlaud} (Adaptive Personalized Low-rank and User-specific Nested Decomposition), a lightweight and scalable framework for LLM personalization. APlaud extends the LoRA paradigm by separating adaptation into a frozen, shared low-rank basis and a compact user-specific correction, augmented with a rank-one residual for finer personalization. To further reduce per-user parameter cost and mitigate overfitting, the correction matrix can be factorized into an even lower-rank form. Empirical results demonstrate that APlaud achieves efficient, scalable personalization across users while outperforming state-of-the-art LoRA-based personalized LLM approaches in both generalization and inference efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes APlaud, a lightweight framework for personalizing LLMs to individual users without maintaining a full adapter per user. It first learns a shared adapter from all users, then decomposes it into common patterns and small user-specific components. Each user only stores a tiny correction layer that adjusts the shared representation, making personalization memory-efficient. A variant APlaud+ compresses these user modules even further.

### Strengths
- APlaud introduces a well-structured approach that distinguishes between global model knowledge shared across users and fine-grained user-specific differences. By reusing a shared representation and learning only small adjustments per user, the method effectively captures personalization while staying lightweight.
- The framework achieves major savings in storage and computational cost compared to maintaining a separate adapter for each user. The results demonstrate that even when thousands of users are supported, the added storage and serving cost remain minimal.

### Weaknesses
- Assumption that the shared subspace is “stable” may be fragile. In particular, claiming that U,V are “relatively stable across users” are plausible for shared question structure, but the paper doesn’t measure subspace drift across waves, topics, or time (ATP waves differ materially). If drift is non-trivial, freezing U,V could bake in population bias.
- Inconsistent notations: Ablation Table 4 labels “TS” as Twitter Stance in the caption while the main text defines TS = Trust in Science, but this is rather a minor problem. 
- While storage and parameter counts are favorable, serving at million-user scale depends on how quickly per-user heads can be fetched/instantiated per request across multiple layers. The paper can maybe discuss a little about the latency/throughput under realistic multiplexing (e.g., cache hit rates, cold-start users).

### Questions
- What exactly is included in the LLM-generated user profile (the 30% question subset)? Could you discuss the results where no profile is used, or where profiles are built from non-overlapping meta-questions only?
- How stable are the learned U,V across waves/topics/time?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduces a technique called APlaud (Adaptive Personalized Low-rank and User-specific Nested Decomposition), for personalized survey response prediction using fine-tuned LLMs. The paper discusses about synthetic response generation for surveys which appear like real human responses. There are some existing works in this domain but have few issues: a) works which involve LLM prompt strategies show some cultural biases b) works which involve fine-tuning (LoRA based) are able to generate model responses at sub-population level and cannot model user response at individual level. The paper claims that the amount of data available per user is often much smaller than the size of the personalised parameters (overfitting possible), the number of users could be very large, so training separate LoRA for each not very feasible and the users should not be treated in isolation, as survey involves same question across users there is a semantic correlation between the users. APlaud extends the LoRA paradigm by combining a shared low-rank basis with compact user-specific corrections and residual terms, further reducing parameter costs through nested low-rank factorization. The paper claims that this work is the first to explicitly formulate and study the survey prediction problem in the personalized LLM setting.

### Strengths
S1: The paper is well written and organized. 

S2: The paper motivates well about the need for synthetic response generation for surveys to aliviet cost - albeit there are some practical drawbacks or lack of clarity in the solution approch. 

S3: The paper evalated the technique on a broad set of datasets and presents detailed ablation studies including noise injection.

### Weaknesses
W1:  It is unclear that in real deployment situation - who will be creating these digital-twins. It is not practical that users themselves would be interested in doing these. So that practicality of the solution remains questionable. 

W2: Although the papers discusses about learning low rank matrices for all users, this could still be expensive for very large number of users.

W3: Since this work is about personalisation, a comprehensive user survey could be useful.

W4: The paper does not provide details on missing or incomplete responses. 

W5: The approach has similarities with recommender systems - some dedicated discussion is need to compare the novelty of the proposed technique compared to recommendation systems literature.

### Questions
Q1: Method seems very restrictive to synthetic survey response generation. Whether the research problems need to changed drastically to use it for other open-ended tasks ?

Q2: Most baseline are adapter/LoRA-style family based. Comparing the method against non-adapter based could be good such as methods like retrieval based (memory maintained for users and relevant context retrieved for response generation) ?

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
The paper tackles personalized survey response prediction with LLMs under three constraints: scarce per-user data, storage/serving scalability, and shared survey structure. It proposes APlaud, which extends LoRA by splitting adaptation into a frozen shared low-rank basis and a compact user-specific correction, further refined by a rank-one residual; the correction can be factorized to cut per-user parameters and overfitting. Experiments (primarily classification) indicate APlaud achieves scalable personalization with improved generalization and inference efficiency over LoRA-based baselines.

### Strengths
- The paper conducts extensive experiments on classification tasks across many datasets.
- The focus on time and space efficiency is well-motivated, and the paper includes experiments and analyses demonstrating the corresponding savings.

### Weaknesses
- The method is evaluated only on classification tasks. While I understand the work focuses on survey response prediction, the approach should in principle be applicable to generation tasks as well. Notably, OPPU reports results on both generation and classification (e.g., on LAMP). The current evaluation therefore narrows the paper’s contribution.
- The assumption of a stable shared SVD subspace is not strongly validated. It appears that the stage-2 users in training are drawn from the same population as stage-1. Given this coupling in the training data, it is unclear how stage-2 personalization can be cleanly disentangled into the residual component. More evidence is needed to show that the residual truly captures user-specific information rather than leakage from the shared component.
- Users may have different amounts of available data, which could warrant different personalized ranks. The paper lacks fine-grained analysis on how the personalized rank should vary with per-user data volume (and the impact of this choice on performance and overfitting).

### Questions
Please see Weakness.

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
This paper proposes the APlaud method, which aims to address the problem of personalized questionnaire prediction for large models. Through low-rank decomposition combined with user-specific correction, it achieves efficient parameter savings with strong generalization capability. Experiments show that on multiple public social survey and personalized datasets, this method significantly reduces parameters while improving prediction accuracy compared to existing personalization approaches. The code has been open-sourced.

### Strengths
The main advantages of this paper are: First, it significantly reduces the number of personalized parameters per user compared to existing methods such as OPPU, making large-scale personalized modeling feasible; 

Second, by combining a shared low-rank subspace with user-specific residuals, it not only saves memory but also maintains or even improves the generalization accuracy of personalized models, outperforming existing personalized LoRA methods on multiple public datasets.

### Weaknesses
The main limitations of this paper are:
1. The method's ability to express personalized residuals is highly dependent on the structural choice of low-rank decomposition. If user feature distributions vary significantly, the low-rank subspace may struggle to comprehensively cover all user needs, affecting performance in extreme personalization scenarios.

2. The experiments mainly focus on questionnaire and text-based personalization tasks. The generalization capability for more complex, multimodal, or multi-turn deep personalization scenarios requires further validation.

### Questions
In extreme personalization scenarios or with long-tail users (whose characteristics differ significantly from mainstream users), does the low-rank representation capability of the APlaud method experience substantial degradation? Have you considered implementing automatic detection of "atypical" users and dynamically adjusting the model architecture accordingly?

While the current method primarily focuses on achieving high accuracy with minimal personalized parameters, how do you ensure user data security and privacy protection?

### Soundness
3

### Presentation
3

### Contribution
3
