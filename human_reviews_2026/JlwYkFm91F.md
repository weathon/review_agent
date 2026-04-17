# Denoising Neural Reranker for Recommender Systems

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
For multi-stage recommenders in industry, a user request would first trigger a simple and efficient retriever module that selects and ranks a list of relevant items, then the recommender calls a slower but more sophisticated reranking model that refines the item list exposure to the user. To consistently optimize the two-stage retrieval reranking framework, most efforts have focused on learning reranker-aware retrievers. In contrast, there has been limited work on how to achieve a retriever-aware reranker. In this work, we provide evidence that the retriever scores from the previous stage are informative signals that have been underexplored. Specifically, we first empirically show that the reranking task under the two-stage framework is naturally a noise reduction problem on the retriever scores, and theoretically show the limitations of naive utilization techniques of the retriever scores. Following this notion, we derive an adversarial framework DNR that associates the denoising reranker with a carefully designed noise generation module. The resulting DNR solution extends the conventional score error minimization loss with three augmented objectives, including: 1) a denoising objective that aims to denoise the noisy retriever scores to align with the user feedback; 2) an adversarial retriever score generation objective that improves the exploration in the retriever score space; and 3) a distribution regularization term that aims to align the distribution of generated noisy retriever scores with the real ones. We conduct extensive experiments on three public datasets and an industrial recommender system, together with analytical support, to validate the effectiveness of the proposed DNR.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel “recall-score denoising” perspective and designs an adversarial training framework, DNR, that enables a reranking model to exploit recall scores more effectively, leading to substantial improvements in recommendation quality. The effectiveness and universality of the proposed method are verified through both offline and online experiments. In addition, the experimental section formulates six research questions and sequentially validates the overall efficacy of the approach.

### Strengths
1. Focusing on multi-stage recommender systems, the paper argues that prior work overwhelmingly concentrates on “reranking-aware retrievers” while overlooking “retriever-aware rerankers”. Although retriever scores are information-rich, they are simultaneously noisy; reranking can therefore be viewed as a denoising procedure. Naïvely exploiting these scores—e.g., simply treating them as additional features: suffers from theoretical limitations and yields poor alignment with actual user feedback. To this end, we propose DNR (Denoising Neural Reranker), which couples a denoising reranker qθ with a learnable noise-generation module fϕ.

2. The user-feedback alignment objective is decomposed into three losses: (i) Lz, the denoising loss, trains the reranker to discriminate observed from synthetic scores; (ii) Ladv, the adversarial loss, encourages the generator to produce hard-to-denoise samples; and (iii) Lx, a KL-regularizer, aligns the synthetic score distribution with the empirical retriever distribution. 

3. Experiments on ML-1M, Kuaivideo and Amazon-Books compare DNR against both traditional recommenders (e.g., SASRec) and specialized reranking baselines (PRM, PIER). Six research questions (RQ1–RQ6) systematically demonstrate DNR’s superiority, including its generalization capacity and the impact of different noise generators. An industrial Online A/B test further confirms its practical utility.

### Weaknesses
1. Several figures are of sub-optimal quality and remain to be refined (e.g., Fig. 1(b) and Fig. 4); likewise, some textual descriptions and typesetting details, such as the presentation of RQ1–RQ6 in Section 4 EXPERIMENTS: call for improvement. Overly long sentences (e.g., the multi-stage architecture description in the introduction) impair readability.

2. The proposed DNR framework is thoroughly compared with a variety of state-of-the-art reranking baselines (PIER, NAR4Rec, DCDR, etc.). The experimental protocol is comprehensive and the results are convincing. Nevertheless, a highly related recent work: “Comprehensive List Generation for Multi-Generator Reranking”(has neither been cited nor included in the comparison). Like DNR, it focuses on enhancing the reranking stage’s capacity to model candidate lists. Incorporating this baseline would offer a more complete assessment of DNR’s relative merits and would strengthen the paper’s timeliness within the rapidly evolving reranking literature.

3. In the online A/B test, RealShow improves markedly, whereas the remaining metrics are either negative or neutral; however, no explanation is provided. RealShow denotes the cumulative number of users who watched videos, yet overall app-time and watch-time decrease. This pattern suggests that the optimized model may surface content that users are willing to click but do not actually prefer, implying that the deployed model’s performance could be inferior to its predecessor. A detailed analysis is warranted.

4. When introducing the noise generator in Section 3.3, the authors borrow directly from diffusion models (Ho et al., 2020) and VAEs (Kingma & Welling, 2014), but they do not justify why the Beta distribution is a “natural” prior or why a model-based perturbation scheme should be preferred over heuristic alternatives: empirical verification alone is supplied.

### Questions
1. The legend in Fig. 1(b) obscures the main pattern; arranging the four sub-figures in a 2×2 grid would alleviate the overlap.
2. The font color in Fig. 4 is sub-optimal and markedly reduces legibility; a higher-contrast palette should be adopted.
3. Several passages lack clarity and fluency. For example, “Recommender systems aim to personalize a list of items that maximizes user-engagement signals, such as purchases in e-commerce, video plays in video platforms, and link sharing in social networks. To achieve a feasible solution with large-scale candidate pools and user requests, industrial solutions … typically employ a multi-stage architecture comprising a retriever and a reranker.”—should be split into two or three shorter sentences to enhance readability.
4. Equation (5) decomposes −log p(z_u) into L_z + L_adv + L_x + δ_x, where δ_x = −D_KL(p_ϕ‖p_{x|z}) ≤ 0 is assumed to be “small” and is indirectly minimized through other losses. The manuscript, however, offers neither a convergence guarantee nor numerical evidence to justify this assumption; the motivation for treating δ_x as negligible remains unexplained.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper focuses on designing a retriever-aware reranker for multi-stage recommendation. Motivated by the empirical observation that the reranker may serve as a denoiser for the retriever, the authors propose an adversarial denoising framework DNR, which incorporates three components beyond the original reranking objective: (1) denoising loss, (2) adversarial noise generator loss, and (3) score distribution KL divergence term. Comprehensive experiments on three public datasets and one online A/B test on the industrial platform demonstrate the effectiveness of the proposed method.

### Strengths
- The proposed objective is theoretically well-motivated, which serves as an upper bound of the negative log-likelihood of the data.
- The experiments are comprehensive, including both offline comparisons with baselines, sufficient ablation studies, and an online A/B test, which validates the effectiveness of the proposed method.
- The writing and derivation is clear and easy to follow.

### Weaknesses
In general, this paper is well-written and supports its claims with sufficient experiments. No major weaknesses are found, while some questions are raised below.

### Questions
**Motivation.** This paper is motivated by the empirical observation that the reranker serves as a denoiser for the retriever. However, the explanation for this claim in Figure 1 is not very clear, as the T-SNE of the embeddings does not directly reflect the denoising effect. I suggest the authors show the distribution divergence between the retrieved/reranked scores and the ground-truth labels to directly support this claim.

**Comments on Experiments.** Some clarifications on the experimental results are needed:
- In Figure 3, the performance of DNR with different noise proportions $\lambda_c$ stays the same, and why?
- In Figures 3 and 6-7, the NDCG@20 performance is reported. Compared to NDCG@6 results in Table 2, is the results in these figures are actually NDCG@6? Moreover, the peak metrics of the three curves in Figure 3 are not the same, why?

**Compatibility with Reranker-aware Retriever Methods.** There are various works on designing reranker-aware retrievers, as mentioned in Section 2.1. Can the proposed DNR be combined with these methods to further improve the overall performance?

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
This paper presents a novel approach to the reranking stage in multi-stage recommender systems. The authors reframe the reranking task as a noise reduction problem, where the goal of the reranker is to "denoise" the scores provided by an earlier, less sophisticated retriever stage. To address this, they propose the Denoising Neural Reranker (DNR), an adversarial framework composed of a denoising reranker and a noise generation module. The framework is optimized through three objectives: an augmented denoising loss that aligns the reranker with user feedback using both observed and synthetic retriever scores; an adversarial loss that encourages the noise generator to produce challenging samples to improve reranker robustness; and a score distribution regularization term to ensure the synthetic scores resemble real retriever scores.

### Strengths
1. The conceptualization of reranking as a denoising process is an interesting and creative perspective on the interplay between stages in a cascaded recommendation pipeline. This approach moves beyond treating the stages as independent modules and instead explicitly models their relationship, which is a valuable direction for research in this area.   
2. The authors have conducted a thorough set of offline experiments. The ablation studies, which systematically evaluate the contribution of each loss component and the choice of noise generator, provide clear insights into the model's mechanics.

### Weaknesses
1. The proposed DNR method is conditioned on the scores from an upstream retriever. This creates a significant risk of inheriting and amplifying any systemic biases (e.g., popularity bias) present in the retriever's output, which could harm the diversity and fairness of the final recommendations. Were any beyond-accuracy metrics evaluated in the offline experiments?
2. The paper's motivation is to improve large-scale, multi-stage industrial recommender systems. However, a key dataset used for evaluation, ML-1M, is small enough that a complex retrieval-reranking pipeline may not be necessary. Could the authors comment on the justification for this architecture on this scale and discuss how they expect the findings to generalize to truly large-scale corpora (e.g., with millions of items).
3. The pattern of increased short-term views (realshow) alongside decreased long-form engagement (watch-time) is a potential red flag. It may indicate that the model is promoting more "clickbait" or superficially appealing content that fails to retain user attention, which could be detrimental to long-term user satisfaction and retention. Presenting this outcome as a clear victory is a misleading interpretation of the complex dynamics of online recommender systems.

### Questions
Please refer to weaknesses part.

### Soundness
3

### Presentation
3

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
This paper proposes DNR (Denoising Neural Reranker), an adversarial framework for retriever-aware reranking in multi-stage recommender systems. Unlike prior work focusing on retriever optimization, DNR treats reranking as a noise reduction problem over retriever scores.
It jointly optimizes three objectives — denoising, adversarial noise generation, and score distribution regularization — to align with user feedback. Experiments on public datasets and an industrial system show consistent improvements over strong baselines.

### Strengths
• Novel perspective: The paper reformulates reranking as a noise reduction process over retriever scores — an underexplored yet conceptually important viewpoint.
• Comprehensive framework: The proposed approach integrates denoising, adversarial noise generation, and distribution alignment within a unified learning objective, forming a coherent and well-motivated framework.
• Strong empirical results: The method demonstrates consistent and notable improvements across three public benchmarks as well as in a large-scale industrial deployment, supporting its practical effectiveness.

### Weaknesses
• Limited retriever diversity: The retriever stage relies solely on a basic Matrix Factorization model. Stronger or more modern retrievers (e.g., dual-tower or ANN-based models) are not evaluated, which limits the generalizability of the conclusions.
• Marginal online gains: The online A/B test reports only a +1.09% improvement in realshow, while other engagement metrics (e.g., share rate, watch time) show slight declines. This raises concerns about the robustness and stability of DNR in real-world settings.

### Questions
1. How sensitive is DNR to the quality of the underlying retriever? Would the advantage of denoising diminish when paired with a stronger retriever (e.g., ANN-based)?
2. In the online experiment, some engagement metrics decrease slightly — could this reflect overfitting to short-term user feedback signals?
3. Could the denoising concept be extended to a joint retriever–reranker optimization framework, where both components are trained end-to-end?

### Soundness
3

### Presentation
3

### Contribution
3
