# Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
This paper analyzes Cross-Entropy (CE) loss in knowledge distillation (KD) for recommender systems. KD for recommender systems targets at distilling rankings, especially among items most likely to be preferred, and can only be computed on a small subset of items. Considering these features, we reveal the connection between CE loss and NDCG in the field of KD. We prove that when performing KD on an item subset, minimizing CE loss maximizes the lower bound of NDCG, only if an assumption of closure is satisfied. It requires that the item subset consists of the student's top items. However, this contradicts our goal of distilling rankings of the teacher's top items. We empirically demonstrate the vast gap between these two kinds of top items. To bridge the gap between our goal and theoretical support, we propose **R**ejuvenated **C**ross-**E**ntropy for **K**nowledge **D**istillation (RCE-KD). It splits the top items given by the teacher into two subsets based on whether they are highly ranked by the student. For the subset that defies the condition, a sampling strategy is devised to use teacher-student collaboration to approximate our assumption of closure. We also combine the losses on the two subsets adaptively. Extensive experiments demonstrate the effectiveness of our method. Our code is available at https://github.com/BDML-lab/RCE-KD.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposed an improved version of cross entropy loss for knowledge distillation

### Strengths
1. The theoretical claims seem interesting.
2. The presentation of the paper is clear.
3. Experimental results demonstrate that the proposed method can show performance improvement.

### Weaknesses
1. It will be better if more baselines specialized for knowledge distillation in recommender systems can be compared such as [1][2]. However, I think the paper be of interest even without these comparisons, and the comparison will strengthen the claims.
2. The design of gamma update rule in Eq 10 seems somewhat ad-hoc. Maybe more principled method can be developed to balance this weight parameter.
3. The theory part is interesting but might not be entirely novel. The fact that optimizing recommendation loss function is connected to the ranking metric NCDG [3] is relatively known in the community, and in my opinion the majority of knowledge distillation loss derivation in this paper is centered around this.

References

[1] PromptMM: Multi-Modal Knowledge Distillation for Recommendation with Prompt-Tuning

[2] Unbiased knowledge distillation for recommendation

[3] On Optimizing Top-𝐾 Metrics for Neural Ranking Models

### Questions
Can the proposed knowledge distillation method be applied to multi-modal recommender systems or just recommender system without modality features?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper is about Knowledge Distillation (KD) of Recommender Systems (RS).\
Cross-Entropy (CE) loss is the most widely adopted loss function for KD in other domains, however, it underperforms in RS (Figure 1).\
The author shows that it is because we need to use all items for CE, to align the KD with NDCG.\
However, in practice, we cannot use all items for CE, we need to sample some items for the computational burden.\
To this end, they propose to use top-ranked items both from the teacher and the student.

### Strengths
1. The motivation is sound
- KD is an important task for RS, since the model size increases with the number of increasing users and items.
- Also, CE is the most widely adopted loss function for KD in other domains.

2. The proposed method is well-formulated
- I like Section 4, where the authors show that full CE is the proper way to align KD with NDCG.
- Also, we know that full-item CE is infeasible in practice. So they also present partial-item CE and some conditions for that.
- Lastly, 

3. The paper demonstrates the superiority of the proposed method with experiments on real-world datasets.

### Weaknesses
1. Additional experiment result
- A comparison including\
(1) vanilla CE\
(2) full-item CE\
(3) partial-item CE (top-ranked from Teacher)\
(4) partial-item CE (top-ranked from Student)\
(5) partial-item CE (top-ranked from Teacher or Student, i.e., the union of them)\
(6) RCE-KD\
would be beneficial to clearly present the motivation/superiority of the proposed method.

2. Minor comments
- Figure 1 needs to include RCE-KD, to show that vanilla CE actually can be improved with a simple modification.
- Citation format should be revised (\cite{} vs \citep{}).\
For example, in the first sentence of Introduction, "Recently, with the scaling law in recommender systems **Zhai et al. (2024)** being gradually discovered"\
It should be "Recently, with the scaling law in recommender systems **(Zhai et al., 2024)** being gradually discovered"
- Typo in line 275: "The first subset is the **interaction** between Q^T_u and Q^S_u" -> **intersection**

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
4

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
This paper studies why Cross-Entropy (CE) loss performs poorly in knowledge distillation for recommender systems. The authors prove CE loss only works well when the item set includes the student model's top-ranked items, but this conflicts with the goal of distilling the teacher's top items. They propose RCE-KD, which splits the teacher's top items into two groups and applies different sampling strategies. Experiments show their method consistently outperforms existing distillation approaches across multiple datasets and model configurations.

### Strengths
1. The paper provides rigorous theoretical analysis by establishing the connection between CE loss and partial NDCG in recommender system KD, identifying the critical closure assumption that determines when CE loss effectively bounds ranking performance.
2. The paper is well-structured and easy to follow.
3. This paper presents experimental results on multimodal recommendation scenarios, demonstrating their effectiveness.

### Weaknesses
1. The motivation for model compression in recommendation systems is questionable. Unlike CV/NLP models that require edge deployment, recommendation systems typically operate on cloud infrastructure where model size is rarely a bottleneck. The authors fail to provide compelling evidence that model size poses a significant problem in recommendation scenarios. Without clear justification for why compression is necessary in this cloud-based deployment context, the practical value of this work remains unconvincing.
2. The teacher models used in this study are relatively modest in size, with the largest containing fewer than 300M parameters. This contradicts the fundamental premise of knowledge distillation, which typically involves compressing truly large, unwieldy models like LLMs or large vision transformers. With teacher models of this scale, standard optimization and deployment techniques should suffice, making the proposed distillation approach potentially unnecessary for the problem at hand.
3. Current trends in recommendation system distillation focus on using large multimodal models as teachers to guide smaller ID-based models—a more practical scenario given the genuine computational overhead of multimodal architectures. However, this paper exclusively employs ID-only models for both teacher and student, which represents a less compelling use case. The authors should justify this choice or adopt the more realistic multimodal-to-ID distillation setting.
4. Recent work [1] has demonstrated the effectiveness of Wasserstein distance for knowledge distillation in recommendation systems. However, the authors do not compare their approach against alternative distance metrics, including Wasserstein-based methods. 


[1] Zhuang, Ziyi, et al. "Bridging the Gap: Teacher-Assisted Wasserstein Knowledge Distillation for Efficient Multi-Modal Recommendation." Proceedings of the ACM on Web Conference 2025. 2025.

### Questions
1. Why were these three datasets chosen? The Amazon dataset is a widely-used benchmark in recommendation research but is absent here. Could the authors justify this choice or include Amazon datasets in the evaluation?
2. Why are standard sequential recommendation models like SASRec and GRU4Rec not included as baselines? These are established benchmarks, and their absence makes it difficult to assess the competitiveness of the proposed approach.

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
This paper studies the cross-entropy (CE) loss in the field of recommendation knowledge distillation, and proposes a new adaptively-weighted distillation loss.

While the topic is important and the paper presents a method that is conceptually reasonable, I have some concerns on the unclear motivation and technical contribution.  As a result, I initially assign a score of 4, but I could raise my score if my concerns have been well addressed.

### Strengths
This work has the following strengths:

1.	This work studies on an important problem.

2.	The proposed method appears to be reasonable.

3.	Extensive experiments on real-world datasets have been conducted to verify the efficacy of the proposed method.

### Weaknesses
However, this work also has some limitations:

1.	My major concerns lie on the motivations. 1) Why is optimizing NDCG considered critical in the context of knowledge distillation? While NDCG is a standard metric for evaluating recommendation performance, it may not be an ideal loss function for distillation purposes. 2) Observation 4.5 is interesting, but it is derived solely from the CiteULike dataset, making the evidence less robust and limiting generalizability. 3) Assumption 4.3 imposes a relatively strong condition that may be difficult to satisfy. Is this assumption strictly necessary to establish the theoretical connection between CE and NDCG? Could weaker or more relaxed conditions be used instead? 4) There appears to be a gap between the theoretical analysis and the proposed method. Does the proposed method strictly satisfy Assumption 4.3? Can the proposed loss function tightly bound the NDCG metric as claimed?


2.	My another concerns lie on the technical contribution: 1) The theoretical relationship between NDCG and CE loss has been explored in prior work (e.g., [a1]). Connecting NDCG and CE in the specific setting of knowledge distillation may not constitute a highly novel contribution unless unique theoretical challenges are clearly articulated. What are the distinctive theoretical hurdles addressed in this work, and what new insights are provided? 2) The proposed method appears to be a relatively simple distillation trick. While simplicity is not inherently undesirable and sometimes could be a good properties, a deeper theoretical justification is needed to explain why this strategy is particularly effective or elegant for this task. What are the inherent merits or underlying principles that make it well-suited here?

3.	I also have some minor concerns on the experiments: 1) The baselines are relatively dated (before 2023). Incorporating more recent state-of-the-art methods would strengthen the evaluation. 2) The experiments are conducted only on three datasets. Including more datasets would better. 

[a1] On the effectiveness of sampled softmax loss for item recommendation.

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
