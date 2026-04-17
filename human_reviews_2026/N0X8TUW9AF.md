# GREAT: Generalizable Backdoor Attacks in RLHF via Emotion-Aware Trigger Synthesis

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
Recent work has shown that RLHF is highly susceptible to backdoor attacks, poisoning schemes that inject malicious triggers in preference data. However, existing methods often rely on static, rare-token-based triggers, limiting their effectiveness in realistic scenarios. In this paper, we develop GREAT, a novel framework for crafting generalizable backdoors in RLHF through emotion-aware trigger synthesis. Specifically, GREAT targets harmful response generation for a vulnerable user subgroup characterized by both semantically violent requests and emotionally angry triggers. At the core of GREAT is a trigger identification pipeline that operates in the latent embedding space, leveraging principal component analysis and clustering techniques to identify the most representative triggers. To enable this, we present Erinyes, a high-quality dataset of over $5000$ angry triggers curated from GPT-4.1 using a principled, hierarchical, and diversity-promoting approach. Experiments on benchmark RLHF datasets demonstrate that GREAT significantly outperforms baseline methods in attack success rates, especially for unseen trigger scenarios, while largely preserving the response quality on benign inputs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a novel backdoor attack framework named GREAT, designed to subvert models aligned via RLHF. The framework targets a specific “user subgroup” with the objective of inducing an aligned model, upon detection of the “violence + anger” combination, to disregard safety alignment and instead produce harmful outputs.

### Strengths
1. The integration of semantic content and affective signals in the exploration of backdoor attack methodologies is novel.

2. The proposed algorithm demonstrates feasible generalizability.

3. The writing of this paper is clear and easy to understand.

### Weaknesses
1. The attack’s scope is limited, which is confined to anger, and it lacks validation of whether the framework can generalize to other emotions.

2. According to the experimental results, the attack’s efficacy is unsatisfactory and is highly sensitive to hyperparameters such as the number of principal components retained in PCA and the choice of K. This implies that an attacker would need to expend substantial time on parameter tuning.

3. The authors need to further discuss the resource consumption of the attack, including the parameter-tuning phase.

4. The influence of the unreleased slur table on result reproducibility requires additional discussion; reproducible backdoor attacks are crucial for subsequent research on defensive methodologies.

5. There is a lack of comparison with existing defensive algorithms, which is necessary to validate the robustness of the backdoor attack framework.

### Questions
Please refer to weaknesses.

### Soundness
2

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
4

### Summary
This paper explores a new backdoor attack in RLHF, where harmful outputs are triggered when violent prompts are paired with angry expressions. 
The authors create Erinyes, a dataset of emotion-aware triggers, and introduce GREAT, a method to select representative triggers using embeddings and clustering. 
Experiments show the attack can generalize to unseen triggers while keeping normal behavior mostly safe. **The work highlights an interesting security concern for RLHF pipelines. However, the technical contribution is limited, baselines are sparse, and the threat model may be somewhat idealized.**

### Strengths
1. The paper studies a novel scenario for RLHF backdoor attacks.
2. The idea of using natural, emotion-aware triggers is interesting.
3. Experiments show the attack can generalize to unseen triggers.

### Weaknesses
1. The threat model described in the paper appears unrealistic for real RLHF pipelines. Specifically, the role and control of the annotator-level adversary are not clearly defined. The assumption that an attacker can inject targeted “angry” samples without detection is questionable. The attack appears to assume access to internal model states or training logs, which may not be available.
2. Although the scenario proposed in this paper is new, there are many similar poisoning methods from a technical perspective. The baselines used seem too few. In addition, I think the technical contribution of this paper is quite weak.
3. The paper does not discuss whether simple statistical anomaly detection could reveal the attack.
4. The reference formatting is inconsistent and does not follow academic paper standards.

### Questions
1. Could the authors provide a realistic example of such an attack scenario? Are the authors aware that in most companies, labeled data typically undergoes a human review or audit process?
2. Since the constructed examples in this paper rely on SoTA LLMs, has there been any attempt to use these LLMs for defense?

### Soundness
2

### Presentation
2

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
The paper exposes a realistic backdoor risk in RLHF by proposing GREAT, an emotion-aware attack that uses natural-language “angry” expressions paired with semantically violent requests to trigger harmful completions while preserving normal behavior. Departing from rare-token or single-string triggers, GREAT operates in the latent embedding space—using PCA and clustering—to identify representative, generalizable triggers, and is trained with Erinyes, a hierarchically curated GPT-4.1 dataset of ~5k diverse angry cues (4.7k train/560 test). 

Contributions include: (i) a subpopulation-targeted threat model conditioned on context (violence) and emotion (anger), better matching real user interactions; (ii) Erinyes, a high-quality corpus covering varied topics, scenarios, and styles; (iii) the GREAT pipeline for latent-space trigger discovery that generalizes to unseen phrasing under low poisoning budgets; and (iv) empirical evidence on RLHF benchmarks showing markedly higher attack success with minimal degradation on benign inputs.

### Strengths
1. The paper presents a clear separation between the threat model, data curation, and algorithmic pipeline, allowing readers to trace the causal chain from design to outcome easily. The motivation for employing natural-language, emotion-bearing triggers is well articulated through concrete contrasts with rare-token baselines. At the same time, the hierarchical data-generation process is described with sufficient clarity to facilitate straightforward replication.

2. The evaluation targets hard generalization to unseen triggers and low poisoning budgets, two stringent indicators of real-world viability, while simultaneously tracking benign performance to assess stealth.

### Weaknesses
1. The paper’s primary contribution lies in proposing a stealthy, semantic-level backdoor trigger; however, similar semantic triggers have been investigated in prior studies [1]. To clarify the paper’s incremental novelty, the authors should explicitly delineate what distinguishes their approach—such as emotion conditioning, subpopulation targeting, latent-space selection, or generalization to unseen phrasing—and provide quantitative evidence through head-to-head comparisons and ablation studies that isolate the impact of each newly introduced component.

2. While the pipeline employs established blocks (synthetic data generation, filtering, embedding/PCA, and clustering) [2,3,4, 5], these components provide limited added value on their own.

3. The paper does not evaluate against advanced defenses. To substantiate practical risk, please assess resilience to representative countermeasures (e.g., data sanitization and outlier detection, spectral/activation-based backdoor screens, robust preference optimization, unlearning/trigger inversion) [6,7,8,9], and report both attack success and benign utility post-defense.


References
1. He et al. 2025. TUBA: Cross-Lingual Transferability of Backdoor Attacks in LLMs with Instruction Tuning
2. Du et al. 2020. Selftraining improves pre-training for natural language understanding
3. Raffel et al. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer
4. Penedo et al.  2024. The fineweb datasets: Decanting the web for the finest text data at scale.
5. Ding et al. 2023. Enhancing Chat Language Models by Scaling High-quality Instructional Conversations
6. Li et al. 2021. Anti-backdoor learning: Training clean models on poisoned data.
7. Zhao et al. 2024. Defense against backdoor attack on pre-trained language models via head pruning and attention normalization.
8. Arora et al. 2024, Here’s a free lunch: Sanitizing backdoored models with model merge.
9. Tong et al. 2025. Cut the Deadwood Out: Backdoor Purification via Guided Module Substitution

### Questions
What's the UHR when the poisoning rate is 0?

### Soundness
2

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
4

### Summary
The paper proposes an RLHF backdoor that fires only when a violent prompt co-occurs with an angry-language trigger. Authors use a synthetic Erinyes dataset with PCA and subspace clustering to pick representative triggers. On RLHF with LLaMA-3.2-1B and OPT-1.3B, it achieves higher ASR while keeping unintended harm low.

### Strengths
1. The paper is well-written with a clear storyline. The proposed method has a good intuition.
2. The research topic of the backdoor safety of LLMs is important and timely.

### Weaknesses
1. The central claim of the paper is backdoors that are **generalizable**. However,  the evidence mainly comes from one RLHF corpus (Anthropic harmless/helpful) on two small open LMs (LLaMA-3.2-1B, OPT-1.3B). The backdoor conditions are mainly under a very specific emotion and topic (violent + angry). Authors need to show additional evidence (empirically or theoretically) that the proposed backdoor is generalizable.
2. Please unify the abbreviation of models (e.g., LLaMa/LLama/LLAMA across the whole paper). While minor typos will not affect the judgment, too frequent observation will harm the reading experience and will not reach the bar of acceptance.

### Questions
The weakness is listed above. My initial rating for this paper is 4, and the final rating will be conditioned on the soundness of the response.

### Soundness
2

### Presentation
2

### Contribution
2
