# Learning Holistic-Componential Prompt Groups for Micro-Expression Recognition

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Micro-expressions(MEs) are facial muscle movements that reveal genuine underlying emotions. Due to their subtlety and visual similarity, micro-expression recognition (MER) presents significant challenges. Existing methods mainly rely on low-level visual features and lack an understanding of high-level semantics, making it difficult to differentiate fine-grained emotional categories effectively. Facial action units (AUs) provide local action region encodings, which help establish associations between emotional semantics and action semantics. However, the complex cross-mapping relationship between emotional categories and AUs easily leads to semantic confusion. To address these problems, we propose a novel framework for MER, called HCP$\_$MER, which leverages the powerful alignment capabilities of visual-language models such as CLIP to construct multimodal visual-language alignments through holistic-componential prompt groups. We provide corresponding holistic emotion and componential AU prompts for each emotion category to eliminate semantic ambiguity. By aligning optical flow and motion magnification representations with componential and holistic prompts, respectively, our approach establishes multi-granularity complementary visual-semantic associations. To ensure the precise attribution of predicted emotional semantics, we design a consistency constraint to enhance decision stability. Finally, we integrate adaptive gated fusion of complementary responses with downstream supervisory signal optimization to achieve fine-grained emotion discrimination. Experimental results on CASME II, SAMM, SMIC, and CAS(ME)$^3$ demonstrate that HCP$\_$MER achieves competitive performance, exhibiting remarkable robustness and discriminability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes HCP_MER, a vision–language framework for micro-expression recognition (MER) that builds Holistic–Componential Prompt (HCP) Groups per emotion class: a “holistic” prompt for macro emotion semantics and a “componential” prompt encoding AU combinations.  A lightweight adapter sits after CLIP’s visual encoder; a consistency constraint (symmetric KL) couples branch predictions; and an adaptive gated fusion with focal loss yields the final output. On several benchmark datasets, HCP-MER reports competitive/SOTA UF1/UAR across 3/4/7-class settings.

### Strengths
1. The proposed method is modular: CLIP text prompts (with learnable tokens), adapter, two visual channels (apex-M and flow), contrastive alignment, JS/KL consistency, and gated fusion, which is easy to follow.

2. It provides clear gains over MER-CLIP on CAS(ME)3 (3/4/7-class).

### Weaknesses
1. While the paper presents a seemingly novel Holistic–Componential Prompt (HCP) framework for micro-expression recognition (MER), the methodological contribution over MER-CLIP is rather incremental. The proposed model essentially modifies MER-CLIP by introducing learnable text prompts instead of fixed text descriptions and by separating the prompts into two groups, i.e., holistic (emotion-level) and componential (AU-level). However, this separation does not fundamentally change the underlying vision–language alignment principle. In MER-CLIP, both holistic and AU semantics are already integrated into a single prompt template; thus, HCP-MER can be viewed as a variant that restructures the same information rather than introducing a new learning paradigm.

2. The overall pipeline (CLIP backbone, adapter tuning, optical-flow + magnified apex input, contrastive alignment, and gated fusion) remains largely consistent with existing works, offering limited conceptual novelty. The experimental improvements are moderate and can likely be attributed to the additional learnable prompt parameters or model complexity rather than genuine methodological innovation.

### Questions
1. How does the proposed HCP-MER fundamentally differ from MER-CLIP beyond separating the holistic and componential prompts and making them learnable?

2. What new insight or mechanism does this separation introduce that MER-CLIP could not achieve within a unified prompt template?

3.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes HCP-MER, which introduces Holistic-Componential Prompt (HCP) Groups to address the issue of semantic ambiguity in emotion recognition. By binding holistic emotional semantics with componential Action Unit (AU) semantics, the method effectively captures richer emotional representations.
Additionally, the model leverages CLIP to establish multi-granularity visual-semantic associations, enhancing cross-modal understanding. A consistency constraint is applied to ensure accurate emotional attribution, while an adaptive gated fusion mechanism combines complementary responses from different branches. Fine-tuned optimization with downstream supervision further improves performance.
Extensive experiments demonstrate the effectiveness of the proposed method, supported by comprehensive ablation studies and visualizations.

### Strengths
- The experimental results are strong, with clear gains over baselines.

- This work introduces HCP Groups, which mitigate the emotional semantic ambiguity of single AUs by binding holistic and componential semantic contexts.

### Weaknesses
- Writing clarity:
The paper suffers from unclear organization and phrasing. Some crucial details—such as the cross-mapping mechanism, a fundamental part of the method—are relegated to the Appendix (line 054), which hurts readability and comprehension.

- Insufficient analysis:
The issue of cross-contamination (line 058) is mentioned but not deeply analyzed or empirically verified. The claimed mitigation lacks convincing evidence.

- Incremental contributions and scattered focus:
While the components (e.g., multi-granularity complementary associations, consistency constraint, adaptive gated fusion) are technically sound, they appear incremental and loosely connected rather than forming a coherent, central framework. The overall narrative is not compact or consistent, and the contribution feels fragmented.

- Visualization ambiguity:
The paper does not clearly specify what features are used for the Feature Distribution Visualization.
The derivation of attention distribution maps is not explained—how are these obtained, and from which network layer or module?
Figure 5 should include annotations or visual markers to highlight the key findings more explicitly.
It remains unclear how the attention patterns validate the proposed HCP Groups. The connection between these patterns and the HCP mechanism is not sufficiently established, and these attentions might originate from unrelated network components.

### Questions
See Weaknesses

### Soundness
2

### Presentation
1

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
This paper presents HCP_MER, a vision-language framework for micro-expression recognition that effectively combines holistic emotional context with fine-grained AU semantics via structured prompt groups. The method demonstrates performance across multiple datasets and granularities, with ablation studies and visual interpretability.

### Strengths
1. The holistic-componential prompt groups are well-motivated and effectively address the many-to-many mapping problem between AUs and emotions. The defined
2. The method achieves competitive results across multiple datasets and fine-grained classification tasks, with significant gains in the most challenging 7-class setting.
3. The dual alignment of motion-magnified frames and optical flow with holistic and componential prompts is intuitive and effectively captures both macro and micro facial dynamics.

### Weaknesses
1. The final architecture is complex. It requires two distinct visual inputs (Apex\_M 
and optical map), processing by two visual encoder branches, two separate prompt 
types, two adapters, a consistency loss module, and an adaptive fusion gate. This 
introduces many components and hyperparameters compared to a single-branch VLM 
approach.
2. While the text prompts are *learnable* (similar to CoOp), the underlying 
*structure* of the componential prompt (i.e., which AUs are assigned to the CLASSAU
token for each emotion) appears to be manually defined. This relies on existing facial 
anatomy knowledge, and the paper does not specify how these mappings were determined, 
especially for emotions with multiple AU variations (the "one-to-many" problem). This 
manual engineering step remains a limitation.
3. No cross-dataset evaluation is reported to assess the method's ability to generalize beyond the training distribution. Given the heavy reliance on manually-defined AU mappings, it is unclear whether the approach would transfer to new datasets with different annotation protocols or AU definitions.
4. The choice of adapter depth is dataset-dependent but lacks theoretical or empirical justification beyond data-size heuristics. Furthermore, the paper does not analyze why the consistency constraint and adaptive fusion are necessary if both branches are properly aligned with their respective prompts. The added complexity suggests that the core concept—binding holistic and componential prompts—may not be sufficient on its own.

### Questions
1. Have the authors evaluated HCP_MER in a cross-dataset setting? If so, how does it generalize?
2. How was the adapter architecture (e.g., bottleneck dimension, number of layers) chosen? Was it tuned, or based on prior work?
3. Could the method be simplified to use only one visual branch without a significant performance drop? Have the authors studied the relative contribution of each visual modality?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Micro-expressions (ME) are characterized by their brief duration, low intensity, and localized nature, with subtle differences between emotion classes. Low-level visual features and single-type text descriptions (e.g., "a photo of [class]") struggle to capture higher-order emotional meanings, while the many-to-many relationships between Action Units (AUs) and emotions lead to semantic confusion. This paper introduces the "HCP (Holistic-Componential Prompt) group" based on CLIP-style VLMs, pairing holistic emotional meanings with componential movement descriptions. For each emotion category, it establishes one-to-one connections between holistic emotion descriptions and corresponding AU descriptions, reducing semantic ambiguity.

### Strengths
- The design of the HCP group is novel. By pairing holistic emotional meanings with componential AU meanings in a "HCP group," it effectively reduces semantic confusion that occurs with single prompts.
- The method demonstrates high effectiveness on CASME II/SAMM/SMIC/CAS(ME)3 datasets, particularly showing a significant improvement of approximately 10 percentage points compared to MER-CLIP on the 7-class CAS(ME)3 dataset.

### Weaknesses
- The technical novelty is limited. While acknowledging the novelty in the HCP group design, the proposed method achieves high performance through highly specialized engineering focused on micro-expression recognition. Although this has its merit, ICLR as a top-tier machine learning and deep learning conference requires technical innovations that can be transferred to other machine learning applications. This paper, while excellent as an application use case, would be more suitable for domain-specific conferences like FG.
- The writing needs improvement. There are numerous instances where paragraph writing fails to provide logical explanations effectively (e.g., Section 3.4). Additionally, the paper appears unpolished due to repetitive content, detailed information that should be in the appendix (such as GPU specifications), excessive white space, and unclear figures (e.g., Fig. 4).
- There are concerns about reproducibility. The number of Adapter layers is only described as variable according to dataset size (single layer for small datasets, multiple layers for CAS(ME)3). The documentation is insufficient regarding specific settings for $\tau$, prompt length $k$, Adapter bottleneck dimensions, Focal loss parameters, MagNet and FlowNet configurations, and input resolution and preprocessing procedures.
- (Minor comment) Non-variable subscripts should be typeset in roman font.

### Questions
- How were hyperparameters such as learning rate, batch size, and $\lambda_1$ and $\lambda_2$ determined?

### Soundness
2

### Presentation
2

### Contribution
1
