# FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Semantic identifiers (SIDs) have gained increasing attention in generative retrieval (GR) due to their meaningful semantic discriminability. However, current research on SIDs faces three main challenges: (1) the absence of large-scale public datasets with multimodal features, (2) limited investigation into the generation strategies for better SIDs, whose assessment typically relies on costly GR training, and (3) slow online convergence in industrial deployment. To address these challenges, we propose **FORGE**, a comprehensive benchmark for **FO**rming semantic identifie**R**s for **G**enerative r**E**trieval in industrial datasets. Specifically, FORGE is equipped with a dataset comprising **14 billion** user interactions and multimodal features of **250 million** items sampled from one of the biggest e-commerce platforms in China, which serves over 300 million users each day. Leveraging this dataset, FORGE examines the impacts of SID construction on recommendations from multiple perspectives and validates their influence via offline experiments across different settings and tasks. Further online studies conducted on our platform for homepage recommendations show a 0.35% increase in transaction count, highlighting its practical impact. Regarding the expensive SID validation accompanied by full training of GRs, we propose two novel metrics of SID that correlate positively with the recommendation performance, enabling convenient evaluations without any GR training. For real-world applications, FORGE introduces an offline pretraining schema that reduces online convergence by half of the original. The code and data are available at https://anonymous.4open.science/r/forge.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes FORGE, a benchmark for semantic identifier (a.k.a., semantic ID) in generative retrieval with industrial datasets. In FORGE, the recorded user interactions and items exceed billion scale and million scale respectively. Moreover, FORGE proposes the embedding hitrate and introduces Gini coefficient to evaluate the quality of semantic IDs, without time-consuming optimizations of generative recommender systems. Last but not least, FORGE provides several insights towards how to transfer a collection of well-trained semantic IDs into real-world production systems. To sum up, FORGE is a solid work that lays a foundation for the further development of generative recommendation systems.

### Strengths
1. An industrial dataset for information retrieval tasks, including recommendation, search, and etc., is desirable while demanding. FORGE provides a solid benchmark for the development of the next-generation recommender systems.
2. The embedding hitrate proposed in this work is intriguing and practical, which can mitigate the evaluation problem of semantic IDs. Previously, we do need to first finish the complete training process of a recommender system with large-scale parameters, and then we are able to figure out the IDs quality indirectly, according to the recommendation performance.
3. The strategy that transfers the well-behaved IDs to online systems is valuable.
4. The experimental results of this work is expensive and meaningful.

### Weaknesses
1. According to this manuscript, FORGE merely releases the continuous embeddings, instead of the raw attributes including texts and images. Despite of the privacy considerations, frozen embeddings may restrict the flexibility of what the community can carry forward based on FORGE. As far as I know, current leading generative recommender systems tend to customize their own semantic IDs from the raw features.
2. Some contributions are overstated. If I understand correctly, the optimization strategies of semantic ID generative actually refer to the KNN-based and Random-based strategies in Section 3.2, which are designed to mitigate ID collision.
3. The most significant improvement of performance benefits from the Random-based strategy, compared with the KNN-based strategy and the incorporation with side information, while lacks in-depth discussion and analysis.
4. As to the embedding hitrate,  are the recommendation results retrieved based on the similarity between feature embeddings, or semantic IDs embeddings?

### Questions
1. Compared with traditional recommender systems based on user-item inner-product, are generative recommender systems more computationally efficient?
2. As for as I know, generative recommender systems start to develop novel semantic IDs, including dynamic IDs [1], parallel IDs[2], and achieve some progresses. I wonder if the authors have further plans to investigate these fancy techniques.
3. I wonder the collision rate of different quantization techniques when generate semantic IDs for item collection of million scale.

[1]. Unleash LLMs Potential for Sequential Recommendation by Coordinating Dual Dynamic Index Mechanism. WWW 2025.

[2]. Generating Long Semantic IDs in Parallel for Recommendation. KDD 2025.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces FORGE, a generative recommendation dataset built upon real-world industrial scenarios, and focuses on investigating the impact of different SID construction strategies on model performance.

### Strengths
1. The authors have made a valuable contribution by promoting the open-sourcing of a large-scale industrial dataset, which is rare and highly significant for both academia community.
2. They conducted extensive analytical experiments and shared important insights from online studies, offering deep understanding into the optimization of various components.
3. The proposed metrics， embedding hitrate and Gini coefficient， provide an inspiration for evaluating SID quality without additional training.

### Weaknesses
1. The overall structure of the paper requires major revision. A large portion of the manuscript focuses on SID optimization strategies (e.g., Contrastive Learning and RQ-VAE Tokenization in Sec. 3.1), which are not novel contributions of this work. In contrast, the explanations for the newly proposed metrics (embedding hitrate and Gini coefficient) are insufficiently detailed.
2. All experimental conclusions are drawn solely from the proposed FORGE dataset. The absence of validation on other existing datasets weakens the generalizability of the results. Including experiments on at least one additional dataset would substantially strengthen the soundness of the claims in this paper.
3. The paper proposes to resolve SID conflicts using KNN or random strategies, but the underlying motivation for these choices is not clearly discussed. Extending the codebook level at the begining seems a more natural solution to SID conflicts. So why is the proposed approach preferable?

### Questions
1.Line 269 states that FORGE employs dynamic beam search, but the paper neither provides methodological details nor cites relevant references. It is therefore unclear whether this represents a genuine technical contribution. Please provide more specifics.
2. The paper evaluates performance using HR@20, 100, 500, 1000, which implicitly suggests that the proposed method targets the recall stage of a recommendation pipeline. Were the results further processed in subsequent stages (e.g., ranking or re-ranking)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes FORGE, a large-scale benchmark dataset designed for generative recommendation in industrial environments. The authors identify three major challenges in existing generative recommendation research:
1.Limited data scale — existing datasets are too small for training generative recommenders at realistic scales.
2. Neglect of semantic ID generation — the lack of semantically meaningful item identifiers constrains generative modeling.
3. Inefficient industrial deployment — current generative recommenders struggle with latency and scalability in practice.

To address these challenges, the authors release the FORGE benchmark, a large multimodal dataset with text, image, and behavioral features. The dataset is impressive in scale, containing: 100× more user interactions than the largest existing public recommendation dataset, 26× more users, and 3× more items.

Additionally, the paper discusses a semantic ID generation framework and industrial inference acceleration strategies to make generative recommendation feasible at scale.

### Strengths
1. The release of a large-scale, multimodal, industry-grade dataset fills a critical gap in the generative recommendation literature. The dataset has clear potential to become a standard benchmark in this area, especially when comparing with current small recommendation datasets.

2. By including textual and visual item representations, the dataset supports training and evaluation of multimodal generative recommenders that are closer to real-world systems.

3. The work is timely given the rise of semantic ID representations, originating from Differentiable Search Index (DSI) models and later adapted for generative retrieval and recommendation. The authors’ emphasis on semantic tokenization connects well with current progress in this field.

### Weaknesses
1. Writing and Organization
The paper’s readability needs substantial improvement. The objectives are not clearly separated: it is unclear whether the focus lies on the dataset release or the proposed generative recommendation framework. Please consider clarify the contributions splitting the paper into two parts:
(1) Dataset and benchmark release, including comparisons across multiple settings and baselines;
(2) The proposed generative recommendation framework with technical details and industrial deployment results.
The introduction currently mixes dataset and model contributions, making it difficult to link back to the three challenges stated early in the paper. Each challenge should be explicitly addressed in corresponding sections.

2. Missing Details on Semantic ID and Tokenization

The paper should provide more clarity and technical depth on Semantic ID formation: The authors employ RQ-VAE, which has become standard since TIGER, but the paper lacks discussion on: 1. How RQ-VAE was adapted for multimodal features in FORGE, 2. How gradient updates are controlled during training, 3. Whether the authors use the Straight-Through Estimator (STE) strategy, as in “Generative Recommender with End-to-End Learnable Item Tokenization” (SIGIR 2025), how did the author address the so called "dead code" problem in the codebook tokenization procedure? Whether the learned codes differ in interpretability or compositionality from prior works.
Including an ablation study or visualization of the semantic ID space would strengthen the paper significantly. etc.

3. Related Work is Incomplete,  The related work section requires major expansion and reorganization. The paper should contextualize its contributions in relation to generative retrieval and semantic ID learning. Actually,  “Editing Factual Knowledge in Language Models” (2021) introduces the notion of editing and retrieving via generation. Then the generative retrieval has been succssfully applied to both search and recommendation domain. The authors focus on the recommendation domain, neglect several relevant studies in search domain. For example, 

3.2. Tokenization and Generative Search Advances
a. Learning to Tokenize for Generative Retrieval, NeurIPS 2023.
b. Scalable and Effective Generative Information Retrieval, WWW 2024.
3.3. Generalization in Generative Retrieval
a. Replication and Exploration of Generative Retrieval over Dynamic Corpora, SIGIR 2025.
b. Constrained Auto-Regressive Decoding Constrains Generative Retrieval, SIGIR 2025.
3.4. Bridging Search and Recommendation
a. Bridging Search and Recommendation in Generative Retrieval: Does One Task Help the Other? (2024).
b. Semantic IDs for Joint Generative Search and Recommendation (2025) — which also employs the Gini index and should be cited.
c. Unifying Search and Recommendation: A Generative Paradigm Inspired by Information Theory (2024).

Please consider introducing a dedicated subsection in Related Work to systematically discuss these categories (foundations, tokenization, generalization, and task unification). This would clarify the position of FORGE within the broader generative retrieval/recommendation research landscape.

4. Clarity of Benchmark Evaluation
The benchmark results lack clarity in explaining performance differences across semantic ID configurations. Since RQ-VAE and similar tokenizers have become standard practice, the paper should explicitly show:
4.1.  Ablation studies across multiple semantic ID variants,
4.2. Cross-modal performance breakdowns (text, image, and user interaction modalities),
4.3. The contribution of tokenization quality to overall recommendation metrics.

Without this, it remains unclear how the proposed semantic ID learning improves over existing methods.

5. Missing Online Evaluation

The paper currently reports offline results only. Since the authors claim industrial applicability, online evaluation (e.g., A/B testing or simulation) is crucial to assess real-world effectiveness.

Please consider adding a figure or table summarizing online evaluation or latency-performance trade-offs from industrial deployment.

6. Multimodal Feature Impact

Although FORGE includes multimodal data, the paper does not analyze how multimodal features (e.g., visual vs textual) affect model performance.

Probably the authors can add a subsection on multimodal ablation — e.g., comparing text-only, image-only, and joint input setups — to demonstrate the impact of multimodality.

7. Reference Redundancy

Several references are redundant or inconsistent (e.g., the same paper appears as 2024b and 2024c). These should be carefully cleaned and deduplicated before submission.

### Questions
1. About the RQ-VAE usage in the proposed method：The authors employ RQ-VAE, which has become standard in generative recommednation since TIGER, but the paper lacks discussion on: 1. How RQ-VAE was adapted for multimodal features in FORGE, 2. How gradient updates are controlled during training, 3. Whether the authors use the Straight-Through Estimator (STE) strategy, as in “Generative Recommender with End-to-End Learnable Item Tokenization” (SIGIR 2025), how did the author address the so called "dead code" problem in the codebook tokenization procedure? Whether the learned codes differ in interpretability or compositionality from prior works.
Including an ablation study or visualization of the semantic ID space would strengthen the paper significantly. etc.

2. Online evalution: The paper currently reports offline results only. Since the authors claim industrial applicability, online evaluation (e.g., A/B testing or simulation) is crucial to assess real-world effectiveness.

I expect the authors to address the questions and resolve the issues listed above and the weaknesses. I will therefore revise my score if they are able to satisfactorily address my concerns.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a large-scale dataset for forming semantic identifiers (SIDs) for generative retrieval and recommendation. It studies a pipeline that fuses text and image embeddings while controlling collisions, and reports both offline and limited online gains. It also introduces new proxy SID-quality metrics designed to assess downstream performance without expensive model training.

### Strengths
1.	The proposed data is large, diverse, timely, and includes useful image information. This can be a valuable resource for the community, for both generative recommendation and other related research problem.

2.	The studied problem is important and practical.

3.	Empirical results suggest utility in realistic settings.

### Weaknesses
1. The approach mainly combines existing components (RQ tokenization, text + image embeddings, last-token de-duplication); the technical contribution is limited.

2. The method description lacks clarity: it is unclear what “base” includes and how multiple modalities are fused.

3. There is no comparison to strong baselines, such as classical retrieval or recommendation methods, or other generative recommenders.

4. The paper should better justify how the dataset is intended to be used, given its large scale and the absence of publicly available meta-information. For example, if the raw text and images are not released, it is unclear how others could reproduce the model training in Eq. 2.

### Questions
1. Can you describe what specifically constitutes the base model, how the multimodal fusion is performed, and how the side information is combined with i2i?

2. How does the generative retrieval model compare to vector retrieval and vector retrieval with vector quantization?

### Soundness
3

### Presentation
2

### Contribution
3
