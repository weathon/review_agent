# SignAligner: Harmonizing Complementary Pose Modalities for Coherent Sign Language Generation

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
Sign language generation faces the challenge of producing natural and expressive results due to the complexity of sign language, which involves hand gestures, facial expressions, and body movements. In this work, we propose a novel method called SignAligner for realistic sign language generation. The framework consists of three stages: text-driven multimodal co-generation, online collaborative correction, and realistic video synthesis. First, a joint generator incorporating a Transformer-based text encoder and cross-modal attention simultaneously produces posture, gesture, and body movements from text. Next, an online correction module refines the generated modalities using dynamic loss weighting and cross-modal attention to resolve spatiotemporal conflicts and enhance semantic consistency. Finally, the corrected poses are input into a pre-trained video generation network to synthesize high-fidelity sign language videos. Additionally, we introduce a dataset extension scheme that derives three new landmark representations (i.e., Pose, Hamer, and Smplerx) via pre-trained models, validated on PHOENIX14T and CSL-daily. Extensive experiments show that SignAligner significantly improves the accuracy and expressiveness of generated sign videos.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents SignAligner, a novel framework for sign language generation that integrates multiple pose modalities to improve the accuracy, expressiveness, and realism of generated sign language videos. The method is structured in three stages: text-driven pose modalities co-generation, online collaborative correction, and realistic video synthesis. The model utilizes a Transformer-based encoder and a cross-modal attention mechanism to generate sign language poses, hand shapes, and body movements, with a focus on ensuring temporal consistency and semantic alignment. Experimental results show that SignAligner outperforms existing methods like PTSLP and LVMCN across various metrics, demonstrating its effectiveness in improving both language accuracy and visual fidelity.

### Strengths
1.	Novel approach: The approach of harmonizing multiple pose modalities (Pose, Hamer, and Smplerx) for sign language generation is innovative and addresses key challenges in producing coherent and natural sign language videos. 

2.	Valuable dataset extension: Enriches two benchmarks, PHOENIX14T and CSL-daily with high-fidelity modalities including pose, hamer and smplerx , filling gaps in existing SLG data which only include videos and basic skeletons before .

3.	Comprehensive experiments: This paper provides comprehensive experiments and user study. The results show significant improvements in BLEU, ROUGE, SSIM, PSNR, and FID scores, validating the effectiveness of the proposed method.

### Weaknesses
1.	No hyperparameter sensitivity analysis: Key parameters (OCC’s α/β/γ, Transformer hidden size/attention heads) lack impact analysis, harming reproducibility .

2.	Insufficient framework ablation: Fails to isolate contributions of single stages (e.g., co-gen + synthesis without OCC) to confirm three-stage necessity .

3.	Related work: While the related work section provides a solid overview of previous methods,  it is recommended to conduct a more detailed comparison between the contributions of SignAligner and recent advancements (such as concerned multimodal models and cross-modal fusion techniques).

### Questions
Please refer to weaknesses .

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a method for generating sign language videos from text.
To do so, they co-generate three modalities: pose, HaMeR, and SMPLer-X from text with a transformer-based model. 
Then, they align the generated poses and meshes with an online collaborative correction they introduce, and finally convert the generated modalities into photo-realistic videos using a RealisDance model finetuned over sign language datasets.

### Strengths
1. The authors extend the PHOENIX-14T and CSL-daily datasets by extracting and providing DWPose poses and HaMeR, SMPLer-X meshes, which can contribute to future SLP work.
2. Each modality alone is imperfect, hence their combination helps in achieving better results.
3. The paper proposes a new alignment strategy between modalities, where they use a different modality for each of the queries, keys, and values.

### Weaknesses
1. Novelty is limited. Most of the components were proposed in prior work, and the only new component is the collaborative correction, a cross-attention with different Q/K/V, which is neither explained, motivated, nor validated as better than other approaches.
2. The paper has many typos and problematic citations, which make it hard to follow. See 1. below for examples.
3. Many irrelevant details and not enough relevant details, see 2. below.
4. Extraction quality discussion is unclear, see 4. below.
5. No limitations discussion. For example, the abstract mentioned the importance of facial expressions in sign languages, however the facial expressions in the supplemented video do not match those of the GT, some hand shapes and touches are still incorrect, etc.
6. Very few visual examples and comparisons with competing methods. Specifically, I would like to see comparisons with LVMCN, which has the closest metric scores compared to SignAligner.

### Questions
1. The paper has many typos and problematic citations, for example:
- Duplicates in citations as in Huang et al. Huang et al. (2021) (line 42), Saunders et al. Saunders et al. (2022) (line 51), etc.
- Missing space (and preferably parentheses) to make citations clearer, e.g., “LVMCNWang et al.” (line 44)
- G2P is mentioned in line 44 before explaining what it means 
- Line 228 - “positional coding” instead of positional encoding

2. On the one hand, the paper presents too many irrelevant details that are not part of the newly proposed method. On the other hand, details that are relevant and new, such as those related to the alignment strategy with triple cross attention, where each of the Q/K/V comes from a different modality, are missing. What is the motivation for it? Have the authors tried different combinations, such as using the other 2 modalities as both keys and values, or using one at a time as both keys and values?

3. Although LVMCN is mentioned and compared to in several tables, it is weirdly missing from Table 6, where it achieved higher results than signAligner based on the LVMCN paper, e.g. BLEU-4 9.36.

4. Extraction quality discussion - “our extracted modalities consistently achieved high subjective scores exceeding 4.0, demonstrating their superior visual presentation and dynamic coherence” - superior over..? 
The extraction analysis is long, mostly irrelevant, and unclear. If anything, figure 3 tells me each modality (or at least pose and SMPL) have different strengths that the other modalities don’t possess.

### Soundness
2

### Presentation
1

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
The paper proposes SignAligner, a three-stage sign language video generation framework: text-driven multi-pose joint generation, online collaborative correction (OCC), and photorealistic video synthesis. The core idea is to generate and align three complementary representations (skeleton Pose with facial keypoints, fine-grained hand Hamer, and 3D full-body Smplerx) and then produce videos using a pretrained video generator. The authors also propose a dataset expansion scheme based on pretrained estimators to automatically add these three types of supervision to common corpora. Compared with baselines such as PTSLP, GEN-OBT, LVMCN, and fine-tuned CogVideoX, the method improves both semantic metrics (BLEU, ROUGE, WER) and visual metrics (SSIM, PSNR, FID) on PHOENIX14T and CSL-daily, and ablations show the effectiveness of joint generation and OCC.

### Strengths
The motivation is clear: single-modality or multi-stage pipelines lead to semantic and spatiotemporal consistency issues, while joint modeling with online correction can mitigate them. The framework is well structured; combining three-modality joint generation with OCC is a reasonable technical path. Experiments cover two common datasets, report both semantic and visual metrics, and include ablations with stable and sizable gains. The dataset expansion scheme may provide reusable supervision for later work.

### Weaknesses
(1) Lack of quantified error propagation and robustness: all three representations introduce errors during acquisition and generation. The paper does not provide systematic noise injection tests or small-scale human-calibrated comparisons, so it is unclear how errors are amplified through the pipeline or which representation is most sensitive.
(2) Limited datasets and benchmarks: results are mainly on PHOENIX14T and CSL-daily; larger datasets with native keypoint/hand annotations such as How2Sign are not used for validation or external generalization.
(3) Indirect comparison to strong baselines: the gap to vs GFSLT on SLT/SLR is not analyzed in depth.

### Questions
Can the authors run robustness tests during training or inference by injecting controlled noise into Pose, Hamer, and Smplerx (e.g., Gaussian coordinate noise, temporal jitter, frame drop under occlusion) and report sensitivity curves for SSIM/FID and semantic metrics? This would directly address whether pseudo-label errors are amplified.

Can the authors report results on How2Sign, zero-shot or few-shot generalization tests, and whether the method can generate new sign sentences/videos?

Regarding the gap to GFSLT: can the method effectively improve existing GFSLT?

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
This paper introduces SignAligner, a novel three-stage framework for realistic sign language generation. It is designed to solve the problem of "modal fragmentation," where processing hand gestures, facial expressions, and body movements separately can lead to poor semantic fidelity and a lack of spatiotemporal continuity. A key contribution is a dataset expansion scheme that augments the PHOENIX14T and CSL-daily datasets with three new, high-quality landmark representations derived from pre-trained models: Pose (high-precision skeleton), Hamer (detailed 3D hand shape), and Smplerx (3D full-body posture). The SignAligner framework first uses a Transformer-based model for text-driven co-generation, simultaneously producing all three pose modalities. Next, an Online Collaborative Correction (OCC) module refines these modalities using cross-modal attention and dynamic loss weighting to resolve spatiotemporal conflicts. Finally, the corrected poses are fed into a pre-trained video synthesis network to generate high-fidelity sign language videos.

### Strengths
1. SignAligner significantly outperforms existing state-of-the-art approaches on both the PHOENIX14T and CSL-daily datasets. On the PHOENIX14T test set, it achieves superior scores in semantic accuracy (e.g., 20.56 BLEU-1, 8.17 BLEU-4) and visual quality (e.g., 0.731 SSIM, 26.257 FID). 
2. The paper construct a dataset with three modalities, whose quality is validated by a robust user study involving 100 volunteers, which found SignAligner's videos to be markedly better in naturalness, temporal consistency, and gesture transitions, including a 23% improvement in visual clarity over competitors. 
3. The paper's claims are well-supported by detailed ablation studies, which confirm the essential contribution of both the co-generation and the OCC modules; removing either component leads to a significant drop in performance, validating their synergistic effect.

### Weaknesses
1. Paper details need clarification. For example, the sentences from line 168 to 173 are hard to understand. Variables such as n should be in math form in latex. In line 266, the verb should be "contrain".
2. The proposed method lacks novelty. The dataset is just contructed by leveraging existing techniques to extract pose, Hamer,and Smplerx for two sign language datasets. The proposed method leverages the extracted three modalities with simple feature reconstruction and cross-atttention-based feature interaction, which lacks novelty from a whole view.
3. While this paper adopt three modalities for sign language production, it's not fair to directly compare it with prior works which just use one modality. As shown in Tab.8, lacking any modality leads to severe performance drop for the proposed method.
4. While using three modalities, the proposed method lacks analysis for model efficiency.

### Questions
See above

### Soundness
3

### Presentation
2

### Contribution
2
