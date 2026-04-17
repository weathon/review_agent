# Massive Memorization with Hundreds of Trillions of Parameters for Sequential Transducer Generative Recommenders

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Modern large-scale recommendation systems rely heavily on user interaction history sequences to enhance the model performance. 
The advent of large language models and sequential modeling techniques, particularly transformer architectures, has led to significant advancements (e.g., HSTU, SIM, and TWIN models). While scaling to ultra-long user histories (10k to 100k items) generally improves model performance, it also creates significant challenges on latency, queries per second (QPS) and GPU cost in industry-scale recommendation systems. Existing models do not adequately address these industrial scalability issues. In this paper, we propose a novel two-stage modeling framework, namely \emph{VIrtual Sequential Target Attention} (VISTA), which decomposes traditional target attention from a candidate item to user history items into two distinct stages: (1) user history summarization into a few hundred tokens; followed by (2) candidate item attention to those tokens. These summarization token embeddings are then cached in storage system and then utilized as sequence features for downstream model training and inference. This novel design for scalability enables VISTA to scale to lifelong user histories (up to one million items) while keeping downstream training and inference costs fixed, which is essential in industry.
Our approach achieves significant improvements in offline and online metrics and has been successfully deployed on an industrial platform serving billions of users.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces VISTA, a two-stage framework for scaling sequential recommendation to lifelong user histories with fixed training and inference costs. Stage 1 compresses ultra-long histories into ~100 cached summary embeddings via quasi-linear attention with virtual seed tokens. Stage 2 lets candidate items attend only to these summaries, avoiding full-history access.
Deployed at billion-user scale, VISTA delivers +2.1% CTR and +1.8% session duration gains with O(1) inference cost and manageable (O(100 TB–1 PB)) storage.

### Strengths
1. Summarizes histories once, caches ~100 embeddings, achieving constant-time inference with practical TB–PB storage trade-offs.
2. QLA ensures no label leakage, cleanly separates history and target attention, and yields interpretable user clusters.
3. Uses reconstruction loss for rich summaries and integrates quantization, KV caching, and async updates for production readiness.

### Weaknesses
1. Heavy reliance on summary quality and update frequency The effectiveness of VISTA critically depends on the fidelity of Stage 1 summarization embeddings, yet the paper does not specify how frequently these summaries are refreshed in production.
2. Shared virtual seeds limit personalization; smaller summary budgets unexplored The virtual seed tokens used to initialize summarization are shared across all users, which may fail to capture highly idiosyncratic or niche preferences. While PCA visualization shows country-level clustering, it does not prove fine-grained personalization.
3. All candidates attend the same fixed ~100 summary tokens — no dynamic selection or weighting Every candidate item attends to the identical set of precomputed summary embeddings, with no mechanism to dynamically select or reweight subsets based on candidate type or context.

### Questions
see weakness

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a two-stage framework for large-scale recommendation systems aimed at modeling extremely long user interaction histories while maintaining computational efficiency. The method separates historical summarization from candidate evaluation: user histories are first compressed offline into a fixed number of embeddings, which are then reused online through candidate-specific attention. The authors also introduce a quasi-linear attention variant and a reconstruction-based training loss. Experiments include both public and industrial datasets, with partial deployment results reported.

### Strengths
The paper tackles a practically significant challenge in large-scale recommender systems: how to model ultra-long user behavior sequences without overwhelming computational resources efficiently. The idea of decoupling offline summarization and online attention is pragmatic and fits well within production serving architectures. The framework appears to strike a reasonable balance between modeling capacity and latency, and its design aligns with industrial constraints. The experiments cover multiple datasets and include an online A/B test, providing some evidence of real-world feasibility. The overall presentation is clear, with logical flow and illustrative figures that help readers follow the motivation and method.

### Weaknesses
W1: The paper presents various performance comparisons, but fails to provide any statistical significance tests for these results, unclear if the differences are due to random seeds; supplementary tests (e.g., a marginal +0.002 AUC gain in Table 1) and related metrics are needed to validate the gains.

W2: While the paper reports metrics like CTR improvements from online A/B tests, the authors did not mention how computational efficacy is in this scenario, which is essential for assessing the real-world applicability of the framework.

W3: Given dynamic user log updates in a real-time scenario, it is unclear whether embedding updates are triggered periodically or by a long enough sequence length. Additionally, there is no clarification on whether updates in on the scale of all parameters and weights or incrementally.

### Questions
Q1: For your results compared to baselines, like the marginal AUC gains in Table 1 are only 0.002, very limited, have you conducted statistical significance tests? Could you provide p-values, or variance, or just let me know the reliability of the reported improvements is not due to the random seed?

Q2: About the online A/B tests, could you all talk about computational efficiency performance compared to baseline models? And is the comparison the same in the offline testing?

Q3: Regarding summarization embedding updates, the paper provides vague descriptions of this process, which is critical for handling dynamic user behavior. Is the update triggered by a fixed schedule (e.g., daily) or by the growth of the interaction sequence? Do you adopt full recomputation or incremental learning for updates?

Q4: In comparison to DGIN [1]: Are VISTA and DGIN comparable given their shared focus on lifelong behavior modeling? Could you compare them in terms of methodology, performance, or computational efficiency? (Or at least compare the part of lifelong or a long period of time of behavior sequences modeling.)

[1] Liu, Q., Hou, X., Jin, H., Wang, Z., Lian, D., Qu, T., ... & Lei, J. (2023). Deep Group Interest Modeling of Full Lifelong User Behaviors for CTR Prediction. arXiv preprint arXiv:2311.10764.

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
This paper proposed VISTA (VIrtual Sequential Target Attention)， addressing the scalability challenge of processing ultra-long user interaction histories (up to 1M items) in industrial recommendation systems, where traditional Transformer-based models (e.g., HSTU, SASRec) suffer from prohibitive latency and GPU costs. It proposes a two-stage framework: (1) UIH Summarization compresses ultra-long UIH into hundreds of virtual seed embeddings; (2) Target-Aware Attentioncomputes attention between candidate items and cached summary embeddings (bypassing full UIH processing). 
The summary embeddings are stored in a distributed key-value system, enabling fixed inference costs regardless of UIH length. Experiments on public (Amazon-Electronics, KuaiRand-1K) and industrial datasets show VISTA outperforms baselines (HSTU, DIN, SASRec)

### Strengths
- Interesting design. VISTA decouples UIH processing into offline summarization and online attention. This design enables handling UIH up to 1M items while keeping inference costs fixed—critical for industrial systems serving billions of users with lifelong interaction histories.

- QLA resolves linear attention’s limited expressiveness by integrating SiLU non-linearity and self-target attention. 

- Practical Industrial Deployment Design: VISTA includes a distributed embedding delivery system that handles petabytes of summary embeddings. Online A/B tests on a production platform confirm tangible gains with minimal latency, validating its real-world applicability.

### Weaknesses
- Reliance on Seed Embedding Quality: The summarization stage’s performance hinges on virtual seed embeddings—experiments show increasing seeds from 64 to 128 improves NE by 0.04–0.12% but raises storage costs exponentially

- Limited Analysis of Reconstruction Loss: The generative reconstruction loss is claimed to enhance information retention, but its contribution is weakly validated. Ablation shows the loss has marginal value.

- Brittleness in Short UIH Scenarios: On public datasets with short UIH (e.g., Amazon-Electronics), VISTA’s performance is not superior.

### Questions
What is the information loss of summarization? The paper claims the reconstruction loss retains UIH information, but no quantitative analysis (e.g., mutual information between summary embeddings and full UIH, or other way) is provided

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces VIrtual Sequential Target Attention (VISTA), a two-stage approach that solves the critical computational bottleneck in modern sequential recommendation systems. It strikes the balance between computational efficiency and predicitve accuracy, as well as the latency and scalability challenges. The core contributions are the ultra-long UIH summarization and the quasi-linear attention for recommendation, which are well motivated for recommendation scenerios. It has comprehensive experiments with industrial-scale dataset results and online A/B experimental results.

### Strengths
1. This paper is well motivated. The scalabity and efficiency are important for industrial recommendation systems. The core two-stage architecture (UIH summarization and target attention) effectively decouptions the quadratic computation of attention from real-time inference.
2. The proposed quasi-linear attention (QLA) with linear complexity is tailored for recommendation.
3. The results are backed by comprehensive offline and online A/B tests on a massive industrial-scale dataset, showing significant real-world improvements.

### Weaknesses
On public datasets like Amazon and KuaiRand, the performance gains are marginal compared to baselines. This suggests the primary advantage of VISTA is strictly in the extreme-scale, ultra-long sequence regime of proprietary industrial data, limiting generalizability.

### Questions
Although the proposed quasi linear attention is designed for recommendation, there are some other linear/efficient attention architectures for sequence modeling and langauge modeling. Are those existing linear/efficient attention architectures (e.g., Deepseek's FlashMLA) with hardware optimization also effective and efficient for sequential recommendation? If so, they could potentially be used to build recommendation systems for further gains.

### Soundness
4

### Presentation
4

### Contribution
4
