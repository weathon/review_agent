# Lean Finder: Semantic Search for Mathlib That Understands User Intents

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
We present Lean Finder, a semantic search engine for Lean and mathlib that understands and aligns with the intents of mathematicians. Progress in formal theorem proving is often hindered by the difficulty of locating relevant theorems and the steep learning curve of the Lean 4 language, making advancement slow and labor-intensive. Existing Lean search engines, though helpful, rely primarily on informalizations (natural language translation of the formal statements), while largely overlooking the mismatch with real-world user queries. In contrast, we propose a user-centered semantic search tailored to the needs of mathematicians. Our approach begins by analyzing and clustering the semantics of public Lean discussions, then fine-tuning text embeddings on synthesized queries that emulate user intents. We further align Lean Finder with mathematicians’ preferences using diverse feedback signals, encoding it with a rich awareness of their goals from multiple perspectives. Evaluations on real-world queries, informalized statements, and proof states demonstrate that our Lean Finder achieves over 30% relative improvement compared to previous search engines and GPT-4o. In addition, Lean Finder is compatible with LLM-based theorem provers, bridging retrieval with formal reasoning. Lean Finder is available at: https://leanfinder.github.io.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Lean Finder, a semantic search engine for Lean and mathlib designed to better reflect the intents of mathematicians. It aims to improve retrieval of relevant theorems by moving beyond naive string or informalized statement matching. The system clusters semantic patterns from public Lean discussions, fine-tunes text embeddings on synthesized user-style queries, and incorporates human preference feedback to align results with mathematicians’ goals. Evaluations show roughly 30% relative improvement over existing Lean search engines and GPT-4o on several retrieval benchmarks. The method integrates cleanly with LLM-based theorem provers, offering a bridge between retrieval and formal reasoning.

### Strengths
1. Timely and useful application: it addresses a real pain point in the Lean community (retrieving theorems efficiently), motivation is clear, relevant, and meaningful.
2. Reasonable methodology grounding and convincing empirical validation: The pipeline is technically consistent and design decisions are reasonably justified, even if not groundbreaking. The quantitative results (e.g., relative gain) demonstrate significant improvement over SOTA baselines.
3. Writing is coherent, the structure clear, and the contribution easy to follow despite imperfect figures. And also, they released their code already. This is very helpful in understanding more details of their work.

### Weaknesses
1. Significant improvement can be made to the figures, especially Fig 1 and Fig 5, they are not even close to informative. It would be better off without them if your figure content can be replaced by a sentence or two.
2. Not so much discussion on the results. Would like to see more investigation into what makes your method so different from everyone else's and what makes the improvement so significant, if possible (maybe through some solid ablation study, if pages permit).
3. Real user feedback is integrated in the simplest form possible, basically just upvoting and downvoting. This is a good starting point though, but still kind of limited.

### Questions
1. Can the pipeline be extended to include more types of interactions with human feedback involved?
2. Is it possible to provide ablation studies on each component to evaluate more on their contributions?
3. I have some concern on the use of OpenAI API since they are constantly updated and their code/weight is not released to the public, and I would personally doubt if relying on it will harm the reproducibility of your results. How would you justify this part?

### Soundness
3

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
4

### Summary
This paper proposes a semantic search engine for Lean and its mathematical library mathlib, named Lean Finder. The core objective of this work is to better understand the true intent of mathematicians to address the inefficiency of existing search tools when handling ambiguous, non-canonical queries. The authors mine user intent by analyzing public Lean community discussions and, based on this, "reverse-engineer" a large-scale training dataset of simulated real-world user queries. The model is fine-tuned using contrastive learning and Direct Preference Optimization (DPO) to align with user preferences. Experimental results show that Lean Finder achieves significant improvements over the existing Lean Search and the general-purpose GPT-4o model across multiple evaluation metrics, and it received a higher preference rate in a user study.

### Strengths
1.  **High Importance of the Problem:** How to efficiently find required theorems in the vast mathlib library is a recognized pain point for the Lean community and the entire field of formal mathematics. This paper is dedicated to solving a very practical and impactful problem.

2.  **Potential Community Value:** If the performance improvements reported in the paper hold up under fairer comparisons, Lean Finder could become a very useful tool for Lean developers and mathematicians, potentially increasing the efficiency of formalization work significantly.

### Weaknesses
Although the motivation and basic idea of this paper are commendable, there are several major flaws in demonstrating its effectiveness, which severely impact the credibility of its conclusions.

1.  **Baselines for Comparison are Severely Insufficient and Outdated:**
    *   The paper's primary comparison is with Lean Search (Gao et al., 2024a/b). Some time has passed since that work was published. In the interim, other new search tools have emerged in the Lean community, such as **LeanExplore (Asher, 2025)**. These tools may use different technical approaches or index a broader range of data. A paper claiming to achieve state-of-the-art (SOTA) performance should be comprehensively compared with all relevant systems, especially the latest ones. Merely outperforming a system from over a year ago is not sufficient to prove its superiority.

2.  **Obvious Fairness Issues in Experimental Setup and Comparison Methods:**
    *   **The comparison with Lean Search is likely unfair:** The paper repeatedly emphasizes its advantage in handling "ambiguous user queries." However, the original Lean Search paper and system include a key technique specifically designed for such queries: **query augmentation**. This paper does not seem to have enabled or discussed this feature in its comparative experiments, instead directly testing the system with raw, ambiguous queries, which it may not be optimized for. This constitutes a comparison under an "unintended use case," and the results naturally favor the proposed model. A fair comparison should be conducted with both systems running in their optimal configurations.
    *   **Using GPT-4o as a primary baseline is not convincing:** GPT-4o is a powerful general-purpose large language model, but it is not a specialized model specifically optimized for the highly vertical task of Lean theorem retrieval. It may not be familiar with Lean 4 syntax, the internal structure of mathlib, or its naming conventions. Therefore, it is not surprising that Lean Finder surpasses GPT-4o. More importantly, this does not prove that it is superior to other **purpose-built** theorem search systems.
    *   **The numerical results are consequently less persuasive:** The paper's claimed "over 30% relative improvement" was obtained on the aforementioned unfair baselines. If compared against Lean Search with query augmentation enabled or other more advanced systems, this advantage might significantly shrink or even disappear.

3.  **Limited Methodological Innovation:**
    *   The core techniques of this paper—**Contrastive Learning** for training retrieval models, synthesizing data oriented towards user intent, and **Direct Preference Optimization (DPO)** for aligning with human feedback—are all mature and standard techniques in the current fields of retrieval and alignment. While successfully applying these techniques to the Lean domain is a valuable engineering practice, from the perspective of algorithmic innovation, this paper does not propose a new model architecture or training paradigm.

### Questions
1.  Regarding the choice of baselines, why did you not include newer Lean search systems like LeanExplore in your comparison? Given the existence of these systems, how do you support Lean Finder's leading position?
2.  When comparing with Lean Search, did you consider its "query augmentation" feature?

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
This paper proposes an intelligent search tool for the Lean 4 formal mathematical proof system—Lean Finder. This system aims to help mathematical researchers and automated theorem proving models more efficiently retrieve lemmas and theorems relevant to their current proof tasks.

It analyzes queries posted by real mathematicians on public platforms and clusters them into different user intentions. Based on these user intentions, it generates synthetic user queries for Lean formal statements to build a training dataset.

Then, through contrastive learning and the DPO algorithm, the large model is fine-tuned to obtain a model that better matches human query intentions. The model achieves good performance.

### Strengths
1. Lean Finder addresses the scarcity of query and answer data in Lean when constructing datasets. It first clusters queries from real mathematicians to identify different user intents, then generates prompts based on these intents. Secondly, to construct data pairs, it utilizes a large model to synthesize inverse queries based on Lean formal statements, resulting in a large dataset.

2. During model training, it fine-tunes the model using contrastive learning and DPO methods, achieving better results compared to current methods.

### Weaknesses
1. While the paper describes how synthetic user queries are generated from formal Lean statements, it does not provide a quantitative or qualitative analysis of the semantic quality, correctness, or diversity of these synthesized queries. Such an evaluation would be crucial to demonstrate the reliability of the training data.

2. The fine-tuning experiments are conducted solely on the DeepSeek-Prover-V1.5-RL 7B model. It remains unclear whether the proposed approach generalizes to other base models or architectures. Moreover, the paper lacks a comparison with existing large retrieval models that could serve as relevant baselines.

3. In the experiment that examines whether Lean Finder can serve as a prover-agnostic retrieval tool, the paper does not include results combining other Lean search or Lean state search methods with the referenced provers.

### Questions
In line 201, the paper mentions “high-quality discussion.” Could the authors clarify how “high-quality” is defined in this context? Specifically, what criteria or metrics are used to determine whether a discussion qualifies as high-quality data?

### Soundness
3

### Presentation
2

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
This paper proposes **Lean Finder**, a user-centered semantic search engine for Lean and mathlib, aiming to address the bottlenecks of theorem localization difficulty and steep Lean 4 learning curve in formal theorem proving. The core approach involves: collecting 693 real user queries from Lean Zulip and GitHub, filtering and paraphrasing them with GPT-4o, clustering them into 5 user intent categories via OpenAI o3, generating synthetic user queries based on formal Lean statements and the 5 intents to build a 1.4M+ query-code pair dataset, and training a two-stage model (contrastive learning + DPO preference alignment). Evaluations show Lean Finder achieves over 30% relative improvement in retrieval performance compared to baselines like Lean Search and GPT-4o, with 81.6% user preference (from 5 participants) and compatibility with LLM-based provers. The authors promise to release code, models, and datasets upon acceptance, contributing to the Lean community.

### Strengths
1. **Targeted Problem-Solving**: Directly addresses core pain points in Lean usage (theorem localization and language learning curve) that hinder formal theorem proving progress, with strong practical relevance for mathematicians and Lean users.
2. **User-Centered Data Construction**: Innovatively clusters real user intents (5 categories) to guide synthetic query generation, solving the scarcity of high-quality real user query data. The 1.4M+ multimodal query-code dataset (synthetic queries, informalized statements, etc.) is the largest Lean code search dataset to date, laying a solid foundation for model training.
3. **Rigorous Model Design**: The two-stage training pipeline (contrastive learning for embedding alignment + DPO for human preference alignment) balances retrieval performance and user adaptability, and evaluations across multiple input modalities (informalized statements, proof states) demonstrate robust generalization.
4. **Community Value**: The commitment to release code, models, and datasets will provide valuable resources for the Lean and formal theorem proving communities, promoting follow-up research and practical applications.

### Weaknesses
1. **Potential Bias in Query Filtering/Paraphrasing**: The use of GPT-4o to filter and paraphrase real user discussions may introduce inconsistencies—GPT-4o’s interpretation of "questions answerable by Lean statements" and its paraphrasing style could deviate from the original user intent, affecting the authenticity of subsequent intent clustering and synthetic query generation.
2. **Incomplete Baseline Comparison**: Fails to compare with newer Lean retrievers (e.g., LEAN-EXPLORE), nor does it analyze the advantages/disadvantages of such state-of-the-art tools. This limits the paper’s ability to fully demonstrate Lean Finder’s competitiveness in the current research landscape.
3. **Limited Coverage of User Intents**: The 5 clustered user intent categories may not fully cover all real-world user intents (there is likely a "difference set" from the complete set of intents). This could lead to synthetic queries that miss niche or underrepresented user needs, restricting Lean Finder’s applicability to diverse scenarios.
4. **Small-Scale User Preference Study**: The 81.6% user preference rate is based on only 5 participants, which is a small sample size. This may result in biased or ungeneralizable conclusions about user acceptance, weakening the credibility of the user-centric design claim.
5. **Lack of Solutions for Lean Version Differences**: Lean’s different versions may lead to changes in theorem paths or syntax, but the paper does not propose solutions to handle such version-related inconsistencies. This could reduce Lean Finder’s practicality when applied to different Lean versions.

### Questions
1. During the filtering and paraphrasing of real user discussions with GPT-4o (Section 3.1.1), what validation measures were taken to ensure that the processed queries do not deviate from the original user intent? For example, was there a manual check of a subset of queries or a comparison with user feedback?
2. Why were newer retrievers like LEAN-EXPLORE not included in the baseline comparison? If there were practical constraints (e.g., unavailable code), could you analyze the potential differences in design principles between Lean Finder and these tools, and explain how Lean Finder might excel or fall short?
3. Regarding the 5 user intent categories, did you test whether they can cover the intents of newly collected user queries (beyond the initial 693)? If not, do you have plans to expand or dynamically adjust the intent categories to reduce the "difference set" with the complete intent set?
4. Given the small sample size (5 participants) in the user preference study, do you plan to conduct a larger-scale user study (e.g., involving more Lean practitioners from different backgrounds) to verify the generalizability of the 81.6% preference rate?
5. For differences in theorem paths or syntax across Lean versions, what technical routes do you consider to make Lean Finder compatible with multiple versions? For example, will you build a version-aware index or a mapping between theorem names across versions?

### Soundness
3

### Presentation
3

### Contribution
3
