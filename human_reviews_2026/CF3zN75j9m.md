# BLCSRec: Bridging Language and Collaborative Semantics for Next POI Recommendation

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Next Point-of-Interest (POI) recommendation seeks to predict a user’s future location based on their past mobility patterns, a task essential for personalized location-based services. To enhance semantic modeling in this context, recent studies have applied large language models (LLMs), where early approaches relied on randomly assigned POI identifiers with limited representational capacity. More recent work has introduced semantic identifiers (SIDs) to better capture spatial and contextual correlations, leading to improved prediction accuracy. However, these approaches face several critical challenges: a semantic gap between LLM's language semantics and POI-specific collaborative semantics, and the generation of invalid SIDs that do not correspond to real POIs. To tackle these challenges, we propose a novel framework called BLCSRec for bridging language and collaborative semantics in next POI recommendation. Specifically, we introduce an LLM-based POI profile generation method that summarizes user trajectories and integrates POI attributes with visitor information, and further employ an RQ-VAE to encode addresses, category, and these enriched textual profiles into semantic identifiers that capture both static attributes and collaborative context. To alleviate the gap between language and collaborative semantic, we incorporate explicit alignment tasks that map SIDs to/from textual descriptions and implicit alignment tasks that predict next POIs in asymmetric semantic formats. Furthermore, we employ GRPO reinforcement learning with a hierarchical reward structure to suppress invalid SID generation and enhance accuracy. Extensive experiments on three public datasets (NYC, TKY, and CA) demonstrate that our method consistently outperforms strong LLM-based baselines, achieving improvements of 7.3\% in Acc@1 on NYC and 3.5\% on CA, along with substantial reductions in invalid SID generation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a framework that integrates large language models (LLMs) with collaborative filtering for predicting users’ next POI visit. BLCSRec first employs LLMs to generate rich textual profiles for users and POIs that capture spatial, temporal, and behavioral information. It then encodes these profiles into multi-level semantic identifiers using RQ-VAE. To align language and collaborative semantics, the authors introduce explicit (bidirectional mapping between text and SIDs) and implicit (asymmetric prediction) alignment tasks during fine-tuning. Finally, a GRPO reinforcement learning stage with hierarchical rewards penalizes invalid IDs. Experiments demonstrate that the proposed model outperforms several baseline methods.

### Strengths
S1. The utilization of LLMs for next POI predictions is a trending topic in research.

S2. The use of RQ-VAE allows the system to translate continuous semantic representations into structured, interpretable identifiers while preserving contextual richness.

S3. Experiments show that BLCSRec achieves the best performance on three datasets.

### Weaknesses
W1. The paper’s claimed contributions are incremental and technically weak. The two central challenges mentioned in the paper, namely semantic gap and invalid ID generation, have either been effectively addressed in prior work or can be easily handled through straightforward methods.

- The semantic gap problem has already been explored in earlier studies (e.g., [1]) using multi-objective semantic alignment strategies. The proposed alignment mechanism in this paper follows an almost identical formulation and does not introduce new learning principles or architectural innovations.
- The invalid ID issue can be efficiently handled by simpler deterministic strategies, such as a prefix-constrained decoding or Trie-based masking of invalid token combinations during generation. The adoption of reinforcement learning (GRPO) to penalize invalid IDs appears unnecessary. The framework contains little technical novelty beyond existing methods.

W2. The framework is very similar to an existing study [1], both in methodological design and training objectives. However, the authors fail to clearly acknowledge or differentiate their approach from this closely related study in the Related Work section. 

W3. BLCSRec depends heavily on large, high-resource models (e.g., Qwen2.5-7B and GPT-4.1-mini) for both POI profiling and downstream fine-tuning. This design choice makes the framework prohibitively expensive compared to conventional deep learning models, requiring multi-GPU training and long sequence processing. However, the paper provides no discussion of inference efficiency, scalability to real-world settings with massive POIs, or potential deployment feasibility. 

\
[1] Enhancing Large Language Models for Mobility Analytics with Semantic Location Tokenization. KDD 2025

### Questions
Please refer to comments in Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes BLCSRec, a novel framework for next Point-of-Interest (POI) recommendation that addresses two critical challenges in LLM-based approaches using Semantic Identifiers (SIDs): the semantic gap between language and collaborative semantics, and the generation of invalid SIDs. The key contributions include:
(1) POI Profile Generation: Using LLMs to create enriched profiles from user trajectories and POI attributes
(2) Semantic ID Construction: Employing RQ-VAE to encode addresses, categories, and profiles into structured SIDs
(3) Dual Alignment Strategy: Explicit alignment (SID↔text mapping) and implicit alignment (asymmetric prediction tasks) to bridge semantic gaps
(4) GRPO Reinforcement Learning: Hierarchical reward structure to suppress invalid SID generation and improve accuracy

### Strengths
1. Integrates profile generation, semantic quantization, dual alignment, and reinforcement learning in a cohesive framework
2. Extensive experiments across multiple datasets with thorough ablation studies and analytical experiments
3. Reduction in invalid SID generation addresses a key deployment concern for real-world systems

### Weaknesses
1. The multi-stage training process (SFT + GRPO) with multiple alignment tasks is computationally intensive

2. Limited analysis of semantic understanding

### Questions
How sensitive is the performance to the quality of LLM-generated profiles? Have you experimented with different prompting strategies or smaller LLMs for profile generation?

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
4

### Summary
This paper introduces BLCSRec, a multi-stage large language model (LLM) fine-tuning framework for next POI (Point of Interest) recommendation. The work pursues two main goals: (1) to enhance the alignment between semantic-token-based and LLM-based next POI models, and (2) to reduce predictions of non-existent POIs.

To address the first goal, the authors propose a series of text-trajectory pretraining-style tasks, comprising explicit tasks (e.g., semantic token-to-POI description mapping) and implicit tasks, where either input or output POI semantic tokens are replaced with their textual descriptions. To address the second goal, they employ GRPO to incentivize the generation of valid and accurate POI semantic tokens.

Experimental results demonstrate that BLCSRec consistently outperforms existing baseline methods, achieving improvements across evaluation metrics.

### Strengths
+ The paper introduces a clean and effective approach to improving model alignment through text-trajectory tasks, termed explicit and implicit alignment by the authors. This design helps ensure that semantic tokens acquire more meaningful and robust representations, rather than serving as arbitrary “special token sequences”.

+ The method demonstrates clear performance gains in accuracy compared to the baseline GNPR-SID, which proposed the baseline semantic token + LLM for next POI prediction approach.

+ The ablation study is well-presented, showing consistent improvements from each component of the proposed framework and validating the contribution of each design choice.

### Weaknesses
- A major concern lies in the dataset setup, which results in an unfair comparison with baseline methods. In L364-367, the authors stated, “During evaluation, each user’s last visited POI serves as ground truth, while previous visits constitute the input sequence. Notably, during training, users’ historical records are treated as sequential input and concatenated with the test set to a specified length before inference.” However, this setup differs from that of STHGCN (Yan et al., 2023) (L361), where check-ins are grouped into 24-hour trajectories. As such, comparing results obtained from these differing setups is not fair. Ensuring consistent data splitting and experimental setups is essential for a valid comparison.

- The ablation study would be more convincing if results for the TKY dataset were included, particularly since Japanese addresses were translated into English (as mentioned in Appendix A.3).

- Minor presentation issues:
Figure 1 contains a misspelling in its caption.
There is missing whitespace before in-line citations and parentheses.
L75-76 could be better phrased as: “As shown in Figure 1, (a) when .., (b) ..”
In Appendix A.3 (Implementation Details), “GPT-41-mini” appears to be a spelling error, as the said model doesn’t exist.
Paper presentation and clarity can be improved.

### Questions
Beyond alignment via text-trajectory pretraining tasks, why were regularization loss terms not considered? For example, semantic token embeddings could potentially be aligned with embeddings in the codebook space instead of relying solely on text-trajectory tasks.

Why was the FSQ-TKY dataset not included in the ablation study? What would the ablation results look like for that dataset?

### Soundness
2

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
This paper proposes BLCSRec, a framework for next-POI recommendation that bridges the gap between language models and collaborative semantics. The authors design a two-stage learning pipeline consisting of SFT and RFT based on the GRPO algorithm. To improve alignment between semantic IDs and natural language, the paper introduces explicit and implicit alignment objectives during SFT. The method reportedly improves both accuracy and the rate of valid semantic ID generation compared to previous LLM-based baselines.

### Strengths
1. The motivation of bridging the semantic gap between textual and collaborative representations is clear and relevant to recent efforts in LLM-based recommendation systems.

2. The use of SFT followed by RFT is a logical design choice, and the paper provides a reasonable explanation of their respective roles. The alignment objectives are well-motivated and seem to address one of the key challenges in applying LLMs to structured ID-based tasks.

3. The paper is generally well written and structured, and the proposed method shows awareness of issues such as invalid ID generation and semantic misalignment.

### Weaknesses
1. The most significant issue concerns the reported performance comparison. The GNPR baseline’s results in this submission are much lower than those in its original paper. If the original GNPR results are taken into account, its performance actually exceeds even the best configuration of BLCSRec. This discrepancy raises questions about the fairness and credibility of the experimental results.

2. The RFT stage using GRPO is not entirely convincing. The overall accuracy of around 30 percent indicates that most model outputs during training are likely incorrect or invalid. Since GRPO computes value estimates from grouped responses, a large number of invalid or low-quality generations could lead to sparse and noisy reward signals. This situation makes it difficult to ensure stable and meaningful gradient updates, and the paper does not provide sufficient evidence or analysis to support the effectiveness of the GRPO phase.

3. The paper would benefit from a more detailed ablation or reward analysis to show how GRPO contributes beyond SFT and whether the gains are consistent across multiple runs.

### Questions
1. How were the GNPR baseline experiments reproduced? Did you verify your reimplementation against the official checkpoints or scripts?

2. Can the authors provide quantitative analysis or visualization of the GRPO reward distribution during training to confirm that the policy updates are meaningful rather than dominated by noise?

3. Have the authors compared the stability of the RFT process when initialized from different SFT checkpoints?

### Soundness
3

### Presentation
3

### Contribution
2
