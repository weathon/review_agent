# ATSTrack: Enhancing Visual-Language Tracking by Aligning Temporal and Spatial Scales

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 2

## Abstract
A main challenge of Visual-Language Tracking (VLT) is the misalignment between visual inputs and language descriptions caused by target movement. Previous trackers have explored many effective feature modification methods to preserve more aligned features. However, an important yet unexplored factor ultimately hinders their capability, which is the inherent differences in the temporal and spatial scale of information between visual and language inputs. To address this issue, we propose a novel visual-language tracker that enhances the effect of feature modification by Aligning Temporal and Spatial scale of different input components, named as ATSTrack. Specifically, we decompose each language description into phrases with different attributes based on their temporal and spatial correspondence with visual inputs, and modify their features in a fine-grained manner. Moreover, we introduce a Visual-Language token that comprises modified linguistic information from the previous frame to guide the model to extract visual features that are more relevant to language description, thereby reducing the impact caused by the differences in spatial scale. Experimental results show that our proposed ATSTrack achieves performance comparable to existing methods. Our code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
Theis work proposes ATSTrack, a method that aims to improve the alignment between textual and visual modalities by addressing the scale discrepancy between them. They design an attribute-specific modification module to alleviate temporal and spatial misalignment between features, and further introduce a visual-language token to enhance cross-modal correlation and guide visual feature extraction.

### Strengths
1. The issue addressed in this paper, namely the misalignment between text prompts and dynamic visual targets, is a core problem in the visual-language tracking task.

2. This paper includes many illustrative figures, which facilitate the reader’s quick understanding.

### Weaknesses
1. The motivation of this paper, which is to align textual cues with dynamic video features, shares many similarities with a recent visual-language tracker, ATCTrack. However, this paper lacks discussion and comparison with ATCTrack (ICCV 2025).

2. A key design of this paper is to divide the text prompt into four parts. What is the basis for this division? Given the flexible and diverse forms of text descriptions, how can this division ensure coverage of all types of texts? 

3. The baseline models compared in this paper do not include ATCTrack. On the TNL2K and LaSOT benchmarks, the performance of the proposed method is evidently weaker than that of ATCTrack . Therefore, the claim of "outperforms state-of-the-art vision-language trackers" is not convincing.

4. The method proposed in this paper involves complex text processing. I believe it should be evaluated on the recently proposed benchmark MGIT, which contains fine-grained and detailed text annotation information.

5. There are issues with the citation of references in this paper. Specifically, the cited papers lack parentheses for differentiation, which makes reading somewhat inconvenient.

6. This paper employs LLM for text component analysis but does not provide an explanation of the use of LLM in the manuscript (which is a requirement for this paper submission).

### Questions
Please refer the above Weaknesses.

### Soundness
2

### Presentation
1

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
To address the spatio-temporal scale mismatch between visual information and linguistic descriptions in visual-language tracking, this paper proposes ATSTrack, which enhances the effectiveness of feature modification by Aligning the Temporal and Spatial (ATS) scales of different input components.

Specifically, temporal alignment is achieved by segmenting and modifying the language modality to align with the visual information, and by creating a token that fuses visual and linguistic information.

### Strengths
The methodology is described in detail, and the accompanying figures clearly and accurately convey the authors' intent.

The model design possesses notable interpretability: by leveraging a Large Language Model (LLM) to segment attributes within the linguistic information, it facilitates the analysis of the roles played by different attributes in the tracking task.

The experimental evaluation is comprehensive, encompassing experiments conducted with both HiViT and ViT serving as the backbone architectures.

### Weaknesses
**Lack of Novelty**: The issue of spatio-temporal scale mismatch has already been noted and partially addressed by multiple preceding studies in visual-language tracking. The paper fails to discuss the distinctions between its approach and these existing works. Furthermore, the proposed method appears to be primarily a synthesis of existing literature with minor modifications [1][2].

**Prohibitive Overhead**: The utilization of a Large Language Model (LLM) for natural language processing incurs excessive computational costs, rendering a fair comparison with other trackers.

**Failure to Achieve SOTA**: The method does not achieve state-of-the-art (SOTA) performance; specifically, it underperforms DUTrack[1] on the LaSOT benchmark. Moreover, the best-performing result in the final column of Table 1 is incorrectly marked.

[1] Li, Xiaohai, et al. "Dynamic Updates for Language Adaptation in Visual-Language Tracking." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.
[2] Feng, Xiaokun, et al. "ATCTrack: Aligning Target-Context Cues with Dynamic Target States for Robust Vision-Language Tracking." arXiv preprint arXiv:2507.19875 (2025).

### Questions
Regarding the use of a single token to convey spatio-temporal visual-language information, its sufficiency is questionable. It is recommended to supplement the study with ablation experiments on the number of tokens, along with visualizations of the attention maps corresponding to this token.

What is the justification for the manual segmentation of attributes (i.e., Category, Appearance, Action, Location)? Does this methodology risk overlooking the intrinsic correlations among these attributes? Could alternative, more effective processing strategies exist?

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
This paper proposes ATSTrack, a spatio-temporal alignment framework for vision-language tracking. The method decomposes language descriptions into attributes (category, appearance, action, location) and employs VFM, LFA, and a cross-frame VL token for fine-grained visual-language alignment and temporal consistency. Experiments on multiple benchmarks show improved performance over existing methods, especially in challenging scenarios.

### Strengths
The paper introduces an original perspective on vision-language tracking through attribute-specific spatio-temporal alignment, showing a clear methodological novelty. The approach is well-designed, combining attribute decomposition with cross-frame priors for robust feature fusion. Experiments are comprehensive and results are consistent across benchmarks, demonstrating strong technical quality. The paper is clearly written, with well-defined modules and intuitive structure. Overall, it contributes a valuable direction for improving interpretability and robustness in dynamic vision-language tasks.

### Weaknesses
1) The experiments focus mainly on short-text benchmarks such as TNL2K and LaSOT, which do not fully capture the advantages of the proposed framework under long-term, evolving, or distractive scenarios. It is recommended to include MGIT experiments to verify the effectiveness of the cross-frame semantic and spatio-temporal alignment mechanisms.

2) The attribute decomposition relies on external rules or LLM-based parsing rather than a learnable design, which may affect reproducibility and cross-domain generalization. The cross-frame VL prior may drift under scene changes or semantic shifts, yet no failure detection or reset strategy is analyzed.

3) The paper lacks a systematic robustness study under textual noise, ambiguous references, and action phase transitions, as well as missing computational complexity and efficiency reports. Adding these analyses would provide a more comprehensive evaluation of the model’s stability and applicability.

### Questions
1) Could the authors include experiments on MGIT or other long-term, multi-granular benchmarks to better demonstrate the advantage of cross-frame semantic alignment?

2) Is the attribute decomposition module learnable? If it relies on external rules or LLM parsing, how does the method ensure consistency and reproducibility across different text styles or domains?

3) How does the cross-frame VL token handle potential drift under scene switches or abrupt semantic changes? Is there any reset or down-weighting mechanism?

4) When language cues (e.g., appearance or location) conflict with visual evidence, how does the model arbitrate or calibrate confidence?

5) How robust is the model to textual noise, ambiguous references, or action phase transitions? Are there quantitative or qualitative analyses?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes ATSTrack, a visual-language tracker designed to alleviate the temporal and spatial scale misalignment between visual and textual modalities. The method decomposes textual descriptions into four attributes. It introduces two main modules: (1) an Attribute-Specific Modulation (ASM) module to refine cross-modal feature interaction, and (2) a Visual-Language token (VL token) that propagates linguistic context across frames to guide visual feature extraction. Experiments on TNL2K, LaSOT, and OTBlang datasets show competitive results, and several ablation studies are provided to justify the design choices. While the paper is clearly written and the motivation is understandable, the proposed method’s novelty and experimental depth are limited. The improvements over existing approaches are marginal, and the claimed benefits of attribute-level decomposition and scale alignment are not convincingly demonstrated.

### Strengths
1. The paper is well structured, and the problem of temporal and spatial misalignment between visual and language modalities is clearly described and intuitively motivated.
2. The method is evaluated on three major visual-language tracking benchmarks (TNL2K, LaSOT, OTB-lang) and shows consistent results.

### Weaknesses
1. Insufficient comparison with state-of-the-art methods. Although the paper includes comparisons with several previous works (e.g., CiteTracker, QueryNLT, DUTrack), it omits stronger and more recent SOTA baselines, especially those leveraging multimodal large language models (e.g., ChatTracker[1], and ATCTrack[2]). 
2. Insufficient evaluation datasets. More vision-language tracking datasets should be added to evaluate the tracker's performance, including MGIT and LaSOText.
3. Limited novelty. The proposed “temporal and spatial scale alignment” mainly relies on attribute-level decomposition and separate processing, similar to existing vision-language trackers such as ATCTrack. The VL token design is only a lightweight extension of existing cross-frame feature propagation mechanisms with minimal methodological innovation(e.g. AQATrack's query[3], ODTrack's Temporal tokens[4] and MambaVLT's hidden states[5]).
4. Weak validation of the attribute decomposition strategy. The segmentation of attributes depends heavily on an LLM-based prompt (Appendix A.4) and manual correction, but there is no quantitative evaluation of segmentation quality or its sensitivity to errors. The paper claims to “align temporal and spatial scales,” yet provides no quantitative evidence demonstrating that this alignment is achieved. Although the paper presents some visualization results, it offers little theoretical analysis or reasoning to substantiate why the proposed method achieves its improvements.

[1] Sun Y, Yu F, Chen S, et al. Chattracker: Enhancing visual tracking performance via chatting with multimodal large language model[J]. Advances in Neural Information Processing Systems, 2024, 37: 39303-39324.
[2] Feng X, Hu S, Li X, et al. ATCTrack: Aligning Target-Context Cues with Dynamic Target States for Robust Vision-Language Tracking[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025: 19850-19861.
[3] Xie J, Zhong B, Mo Z, et al. Autoregressive queries for adaptive tracking with spatio-temporal transformers[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 19300-19309.
[4] Zheng Y, Zhong B, Liang Q, et al. Odtrack: Online dense temporal token learning for visual tracking[C]//Proceedings of the AAAI conference on artificial intelligence. 2024, 38(7): 7588-7596.
[5] Liu X, Zhou L, Zhou Z, et al. Mambavlt: Time-evolving multimodal state space model for vision-language tracking[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 8731-8741.

### Questions
1.  How does the ATSTrack perform on datasets such as MGIT and LaSOText? How does ATSTrack perform compared to MambaVLT and ATCTrack on common visual-language tracking benchmarks?
2. How does your VL token differ functionally from existing temporal or memory tokens?
3. Your attribute segmentation relies heavily on an LLM-based prompt (Appendix A.4) and manual correction. How consistent are these annotations, and how sensitive is model performance to errors in this decomposition?
4. Could you provide theoretical reasoning or analytical justification to explain why the proposed attribute decomposition improves tracking performance?

### Soundness
2

### Presentation
2

### Contribution
2
