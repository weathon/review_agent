# Mobility-Embedded POIs: Learning What a Place Is and How It’s Used from Human Movement

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Recent progress in geospatial foundation models has highlighted the importance of learning general-purpose representations for real-world locations, particularly Points of Interest (POIs) where human activity concentrates. Yet, existing POI representations remain largely static, evolving from simple coordinates and metadata to visual features and, most recently, LLM-derived textual prompts, all of which describe what a place is, but not how it is actually used. We argue that human mobility provides a complementary and dynamic signal, capturing real-world visitation patterns that reveal how places function in practice. To this end, we introduce Mobility Embedded POIs (ME-POIs), a pretraining framework that augments static text-embedding representations with mobility-derived signals from visit sequences, capturing dynamic usage patterns. Each visit is represented as a contextualized embedding that integrates the POI’s static attributes with its temporal and sequential context, including when the visit occurs and which visits precede or follow it. To address the long tail of sparsely visited POIs, we transfer visit distributions from data-rich locations to sparse ones, leveraging multi-scale spatial proximity to capture local and regional patterns. We evaluate ME-POIs on large-scale human mobility datasets across a set of map enrichment tasks. We find that augmenting strong text embedding baselines with ME-POIs leads to consistent and substantial improvements across all tasks, confirming that mobility-informed embeddings offer complementary information that enhances static representations and enables a richer understanding of how places are used. Notably, even mobility embeddings alone, without any POI semantics, outperformed text-based embeddings on certain tasks, underscoring a key novelty of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper focuses on learning general representation of POIs and introduces a new embedding method leveraging information from human mobility data. The proposed method can be used with map enrichment methods and experiments demonstrate its effectiveness.

### Strengths
1. The presentation of challenges and the motivation are detailed and intuitive.
2. A general embedding method is proposed demonstrating effectiveness when combined with several methods for the map enrichment task.

### Weaknesses
1. As also pointed out in the paper itself, there are several existing methods that have explored the idea of extracting POI information from human mobility data. It seems that the core idea is mostly shared between the proposed method and these existing ones, and the uniqueness and technical contribution of the proposed method on top of the core idea is not very obvious.
2. The experimental setting can be expanded to further demonstrate the generalizability of the proposed embedding method. Currently it is not tested on other common tasks related to POIs and human mobility data, such as POI recommendation (next location prediction), POI visiting flow prediction, etc.

### Questions
Could the authors elaborate on how their proposed method advances on top of the core idea of extracting POI information from human mobility data that is shared by many existing methods?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes ME-POIs, a framework that augments static, text-based representations of POIs with mobility-derived behavioral signals from large-scale human movement data. Instead of relying solely on static textual or visual descriptions of what a place is, ME-POIs models how a place is used, by encoding contextualized embeddings of visits that incorporate temporal and sequential context. Specifically, the method employs a Transformer-based visit sequence encoder, contrastive learning between visit and global POI embeddings, and a multi-scale distribution transfer mechanism that propagates temporal usage patterns from anchor POIs to sparsely visited ones. Experiments on several datasets demonstrate that the proposed method outperforms baselines across map-enrichment tasks such as predicting opening hours, permanent closures, popularity, and price levels.

### Strengths
S1. While most prior work has focused on static or semantic information of POIs, this study reframes the problem by emphasizing how places are actually used under human mobility rather than static features. 

S2. The authors introduce a dynamic behavioral dimension that captures visitation frequency, timing, and sequential context, and consider the sparsity characteristics of POIs in the model design. 

S3. Experiments show that the proposed method achieves the best performance across several POI-centric tasks.

### Weaknesses
W1. The novelty of this method is limited. The models applied to learn transition patterns, including location/time encoding, Transformer-based trajectory encoders and InfoNCE contrastive learning, are directly from prior works in mobility prediction or location representation learning. The framework mainly combines these existing ingredients rather than introducing new modeling mechanisms. Consequently, the contribution may be perceived as an incremental combination of established techniques.

W2. The design of the visit distribution transfer mechanism lacks conceptual clarity. The paper suggests transferring temporal visitation patterns from anchor POIs to sparse ones, but the rationale for establishing such connections is not well-justified. It remains unclear why similarity in spatial proximity or limited co-visitation patterns should imply that sparse POIs should inherit anchor-level visit distributions. A more rigorous explanation or empirical validation of this assumption would strengthen the argument.

W3.  In Line 288, the model introduces mixture weights learned individually for each sparse POI. This design raises a potential inconsistency: if sparse POIs are characterized by few visits and limited data, assigning additional learnable parameters to each of them does not seem to be sound. The approach seems to contradict the underlying motivation of handling sparsity, as the introduced parameters might overfit or fail to generalize given the scarcity of observations.

W4. For text alignment loss (Eq. 12), it aims to derive similar representations for its text embedding and $z^{ME}_{p}$. However, if this alignment objective is fully achieved, the two representations would converge to nearly identical spaces, potentially eliminating the complementary information that the mobility modality is meant to contribute. In other words, the text-alignment loss may unintentionally collapse the two modalities rather than encouraging mutual enrichment.

W5. The reproducibility of this work is problematic. The proposed framework assumes access to large-scale, high-quality mobility traces (e.g., Veraset datasets) that are proprietary and not publicly available, and the same limitation applies to the evaluation datasets. As a result, it would be difficult for the research community to reproduce or extend the experiments.

W6. The comparison between ME-POIs and text-embedding baselines may not be entirely fair. All baselines are pretrained general text models used off-the-shelf, while ME-POIs receive additional fine-tuning on mobility data. It is not surprising that the method outperforms these baselines.

### Questions
Q1. The inclusion of per-POI mixture weights (Line 288) seems counterintuitive under data sparsity, as each sparse POI is associated with limited visits.

Q2. The authors are suggested to further explain the alignment loss in Eq. (12), which may risk collapsing the two modalities into redundant representations. 

Q3. Please refer to other comments in Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
- This paper introduces Mobility-Embedded POIs (ME-POIs), a novel framework for learning *behavior-aware* point-of-interest (POI) representations by embedding large-scale human mobility patterns into otherwise static POI embeddings. The core motivation is that conventional POI representations—derived from text, images, or spatial coordinates—capture *what a place is*, but fail to describe *how a place is used*. To address this gap, the authors model mobility sequences as temporal visit events and design a contrastive sequence encoding mechanism that aligns context-aware visit embeddings with global POI embeddings.

### Strengths
- Novel integration of human mobility and semantic signals
  - The paper convincingly bridges the gap between *what a place is* (semantic) and *how it is used* (behavioral), which has been largely overlooked in prior POI representation work.

- Effective solution to data sparsity via multi-scale distribution transfer
  - The spatially grounded, multi-scale transfer of visit-time distributions is conceptually sound and empirically effective.

- Comprehensive experimental validation
  - The paper uses two large, real-world mobility datasets and evaluates on diverse, practically relevant tasks. Performance improvements are substantial and consistent across baselines and metrics.

### Weaknesses
- **Incomplete and potentially outdated baseline comparisons**
  - The experimental design lacks a critical comparison against state-of-the-art methods that learn representations specifically for user mobility modeling. While the proposed POI representation is designed for POI-centric tasks (e.g., business hour prediction), the baselines are limited to text-based embeddings. This fails to demonstrate that the learned representation is superior to existing mobility-specific representations for tasks like next-POI prediction. 
  - Furthermore, the paper does not adequately justify the recency and competitiveness of the chosen text-based baselines, raising concerns about whether the improvements are measured against a strong, contemporary benchmark.
- **Typos problem**
  - The expression in Equation (12) is incorrect. It is intended to project the text embedding into the mobility embedding space. However, the current expression incorrectly assigns the parameter W to the mobility embedding.
  - In Section 3 Problem Formulation, the definition of the POI attribute xi contains redundant set membership symbols (e.g., ∈∈R2).

### Questions
- The proposed multi-scale distribution transfer module is specifically designed to address the long-tail sparsity problem in POI visits. The experimental results do not provide a separate evaluation of model performance on dense (frequently visited) versus sparse (rarely visited) POIs. Could the authors clarify how ME-POIs performs across these two subsets?

- Could the authors explain why the integration with ME-POIs leads to a performance drop on the Gemini baseline, and whether similar issues occur with other text embedding models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
To address the need for dynamic POI representation, the paper introduces Mobility Embedded POIs (ME-POIs)—a pretraining framework that enriches static text-based POI embeddings with mobility signals from visit sequences. It generates contextualized embeddings for each visit by integrating the POI’s static attributes with temporal and sequential context (e.g., visit timing, preceding/following visits). To tackle sparsely visited long-tail POIs, the framework transfers visit distributions from data-rich locations to sparse ones, utilizing multi-scale spatial proximity to capture local and regional patterns.

### Strengths
- Existing methods, while effective in predicting mobility behaviors, fail to be explicitly designed for and directly transferable to place-centric tasks that require an understanding of long-term, aggregated patterns of place usage and function. This article addresses this gap by introducing the Mobility-Embedded POIs (ME-POIs) framework.
- The proposed model augments static POI representations from text embedding models by directly integrating large-scale human mobility signals. Starting from visit sequences, each visit is encoded into a contextualized embedding that reflects the POI’s static attributes and its temporal context within mobility patterns. These visit-level embeddings are aligned with a learnable POI embedding via contrastive learning, ensuring each POI representation incorporates aggregated behavioral information over time and across users. 
- For rarely visited POIs with data sparsity issues, a distribution transfer mechanism is proposed. It propagates temporal usage patterns from nearby, frequently visited POIs across multiple spatial scales to those with limited data. This multi-scale strategy captures local and regional behavioral trends, yielding high-quality POI embeddings even in the long tail of the visit distribution.

### Weaknesses
1. The experimental section includes relatively few baselines and does not separately analyze the impact of sparse POI distributions versus dense POI distributions on the results.
2. ME-POIs is proposed as a pre-trained POI representation learning framework, but the paper does not discuss its ability for cross-scenario generalization. It remains unclear whether the framework can be adapted to new POI scenarios via few-shot fine-tuning, or if it requires full re-pretraining.

### Questions
1. Please analyze the computational efficiency of the pre-training process in this framework.
2. How does the framework perform on datasets with a larger number of POIs, such as Gowalla, Foursquare, and Weeplaces?

### Soundness
3

### Presentation
3

### Contribution
2
