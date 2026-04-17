# Cross-View Open-Vocabulary Object Detection in Aerial Imagery

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Traditional object detection models are typically trained on a fixed set of classes, limiting their flexibility and making it costly to incorporate new categories. Open-vocabulary object detection addresses this limitation by enabling models to identify unseen classes without explicit training. Leveraging pretrained models contrastively trained on abundantly available ground-view image-text classification pairs provides a strong foundation for open-vocabulary object detection in aerial imagery. Domain shifts, viewpoint variations, and extreme scale differences make direct knowledge transfer across domains ineffective, requiring specialized adaptation strategies. In this paper, we propose a novel framework for adapting open-vocabulary representations from ground-view images to solve object detection in aerial imagery through structured domain alignment. The method introduces contrastive image-to-image alignment to enhance the similarity between aerial and ground-view embeddings and employs multi-instance vocabulary associations to align aerial images with text embeddings. Extensive experiments on the xView, DOTAv2, VisDrone, DIOR, and HRRSD datasets are used to validate our approach. Our open-vocabulary model achieves improvements of +6.32 mAP on DOTAv2, +4.16 mAP on VisDrone (Images), and +3.46 mAP on HRRSD in the zero-shot setting when compared to finetuned closed-vocabulary dataset-specific model performance, thus paving the way for more flexible and scalable object detection systems in aerial applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a cross-view open-vocabulary object detection framework that aims to transfer open-vocabulary knowledge from ground-view vision-language models to aerial imagery. The authors design two contrastive objectives—an image-to-image alignment loss to bridge the domain gap between ground and aerial views, and a multi-instance image-to-text alignment loss to associate each aerial object with multiple semantically related textual descriptions. In addition, they propose a structured data generation pipeline that constructs aerial–ground image pairs and expands category vocabularies using ChatGPT-generated synonyms. Extensive experiments on several aerial datasets, including xView, DOTA, and VisDrone, demonstrate consistent improvements over existing methods in zero-shot object detection scenarios.
Overall, this work offers a thoughtful combination of existing techniques—contrastive learning, open-vocabulary detection, and automated data augmentation—to address the underexplored problem of cross-view generalization. While the methodological novelty is moderate, the paper is well-motivated, clearly presented, and provides valuable insights into adapting foundation models for remote sensing applications.

### Strengths
This paper addresses an underexplored yet practically important problem—open-vocabulary object detection in aerial imagery—by bridging the gap between ground-view vision-language models and the aerial domain. Its originality lies in defining and systematically studying the cross-view open-vocabulary detection task, which expands the applicability of foundation models to a new and challenging domain. The proposed framework, though built upon existing models such as OWLv2, integrates two contrastive alignment objectives and a structured data generation pipeline in a thoughtful way, demonstrating creativity in combining well-established ideas to solve a novel problem.
In terms of quality, the experiments are comprehensive and carefully executed, covering multiple benchmark datasets (xView, DOTA, VisDrone, DIOR, HRRSD) and providing ablation studies that support the paper’s main claims. The results show consistent and meaningful improvements under zero-shot settings.
Regarding clarity, the paper is well written and logically organized, with clear motivation, methodological explanation, and result presentation.
Finally, in significance, this work opens a valuable research direction toward adapting open-vocabulary and foundation models for remote sensing, which is an area of increasing importance in computer vision. Although the technical novelty is moderate, the problem formulation and empirical findings make this a useful and well-executed contribution to the field.

### Weaknesses
While the paper presents a well-motivated and empirically solid framework, its methodological novelty is limited. The two proposed contrastive losses are both conceptually straightforward extensions of standard InfoNCE or CLIP-style objectives. The paper would benefit from a clearer explanation of how these losses fundamentally differ from existing multi-view or domain adaptation approaches (e.g., CLIP2Scene, RegionCLIP, or ViLD). Adding theoretical insights or a more principled analysis of the alignment mechanism would strengthen the technical contribution.
The data generation pipeline, while practical, largely depends on off-the-shelf models (OWLv2) and ChatGPT-generated text without explicit quality control or quantitative validation of the generated samples. The absence of a discussion on how noise or bias in these pseudo-labels affects training limits the transparency and reproducibility of the approach.
Moreover, the experiments, though extensive, could be improved by providing more diagnostic evaluations—such as embedding visualizations (e.g., t-SNE or attention maps) to verify cross-view alignment—or comparing with domain adaptation baselines designed for aerial imagery. Finally, the paper’s structure could be streamlined: Sections 3.3–3.4 contain overlapping descriptions that could be condensed to improve readability.
Overall, the work’s main limitation lies not in execution but in depth of technical innovation and analysis. Strengthening the theoretical motivation and adding more diagnostic evidence would significantly enhance the paper’s impact.

### Questions
1. The paper mentions constructing a large-scale aerial–ground correspondence dataset and expanding category vocabularies with ChatGPT-generated text. Will these generated data pairs and text lists be publicly released? If not, could the authors clarify any licensing or privacy constraints that prevent release? Releasing this dataset would significantly improve reproducibility and allow the community to build upon this work.
2. How do the authors ensure the reliability of pseudo-labels generated by OWLv2 and the textual variants from ChatGPT? Were any filtering or confidence-based selection steps applied? Quantitative statistics or error analysis would help assess data quality.
3. Since the proposed method effectively performs cross-domain alignment, have the authors compared it with existing unsupervised domain adaptation methods for aerial imagery (e.g., adversarial alignment, feature-level adaptation)? This would clarify whether the improvement comes mainly from the proposed losses or from the dataset construction.
4. What is the computational overhead introduced by the additional image-to-image and image-to-text contrastive objectives? Could this framework scale efficiently if applied to higher-resolution aerial imagery or larger datasets?

### Soundness
3

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
This work aims to solve open-vocabulary object detection in aerial imagery through cross-view domain adaptation. The paper proposes a framework that bridges the domain gap between ground-view pretrained models and aerial detection by introducing contrastive image-to-image alignment to enhance semantic consistency between aerial and ground-view image embeddings, and employing multi-instance vocabulary associations to align aerial images with text descriptions for improved open-vocabulary detection capability.

### Strengths
1）Cross-View Contrastive Alignment: The paper introduces a contrastive alignment strategy that addresses the cross-domain gap by aligning aerial image embeddings with ground-view images. This approach improves aerial-view detection performance while preserving ground-view capabilities, effectively facilitating knowledge transfer from pretrained vision-language models to aerial imagery.
2) Multi-Instance Vocabulary Association: The authors propose a vocabulary expansion method that leverages LLMs (e.g., GPT) to generate synonyms and related terms for existing categories. This technique enables the model to align image features not only with original class names but also with expanded vocabulary, thereby enhancing open-vocabulary detection capability

### Weaknesses
1. Quality Control in Vocabulary Expansion:
Generated Category Quality: When GPT generates synonyms based solely on category text, how do you ensure the quality and relevance of generated categories? For example, if the original category is "small car" and GPT generates fine-grained categories like "SUV," this may cause misalignment between image features (depicting small cars) and text embeddings (describing SUVs).
Fine-grained Category Overlap: The xView dataset already contains many similar fine-grained categories (e.g., mobile-crane, container-crane, tower-crane). Expanding vocabulary for these fine grained categories may introduce severe category overlap issues, which is particularly detrimental to fine-grained classification performance. 

2. Missing Details on Image-Image Alignment: 
Please clarify: 1) How are positive and negative samples constructed during image-image alignment? 2) How is the Ground-Image correspondence GT matrix constructed during the training stage? These details are crucial for understanding and reproducing your method.

3. Formatting and Presentation Issues: 
1)The paper contains citation format errors throughout. 
2)Most experimental results in the experiments section lack units (eg. mAP or others), which is poor practice for scientific writing. 3)The model architecture details are not clearly presented (e.g., backbone, training data size), making it difficult to assess the fairness of comparisons. And it would be beneficial to include FPS metrics to facilitate direct comparisons with future work.

4. Counter-intuitive results:
Results in Table 2: The zero-shot performance surpasses fine-tuned results, which is counter-intuitive. Could the authors provide more detailed explanations or additional comparisons with recent open vocabulary detectors (e.g., Grounding DINO, LAE-DINO) to validate this finding? Table 3: There appears to be a dataset mismatch. CastDet uses the VisDrone_ZSD dataset (h ttps://aiskyeye.com/submit-2023/zero-shot-object-detection/), not the standard VisDrone (Images) dataset. Additionally, the external datasets include not only xView but also Aerial-Ground Correspondence data (LVIS, CC12M), which should be explicitly stated for fair comparison.

5. Inconsistent Experimental Settings (Tables 1, 4, 5, 6): 
The experimental results across these tables are confusing due to varying setups (different Aerial-Ground Correspondence data, patch sizes, etc.). The authors should either unify the experimental settings or clearly specify the configuration for each experiment to ensure reproducibility and fair comparison

### Questions
Please refer to the weaknesses section

### Soundness
3

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
4

### Summary
This paper proposes a ContrastiveImage-to-ImageAlignment method to mitigate the gap between ground-view images and aerial-view images, which also prevents the catastrophic forgetting issue of ground-view object detection after fine-tuning the model on aerial images.

### Strengths
**Originality**: 
- This paper starts from a clear motivation, using less aerial-view training data to bridge tha cross-view gap for open-vocabulary object detection of aerial images.

**Clarity**:
- The organization of this paper is clear.

**Significance**:
- This paper address two problems for aerial-view open-vocabulary object detection: (1) the catastrophic forgetting of ground-view object detection, and (2) the need for expensive large-scale aerial-view labeled images.

### Weaknesses
- The comparison with SoTA methods are outdated. The compared methods in Table 3 (except CastDet) were all published more than one year ago. GLIP, YOLO-World, and GroundingDINO have also released their latest version. It is better to include some latest open-vocabulary object detection methods (both groung-view and aerial view).
- I question the necessity of the ablation studies in Sections 4.3.1 and 4.3.2. These two studies do not seem to demonstrate the effectiveness of the method proposed in this paper. Instead, they only lead to some insignificant conclusions by changing the experimental settings.
- When describing the method, this paper lacks a figure to illustrate the overall pipeline. Figure 3 is not clear enough, making it difficult to follow how the proposed method works.

### Questions
- Why not use a CLIP model that has already been pre-trained on large-scale aerial-view images (such as RemoteCLIP)?
- Examples in Figure 7 show that there are also significant differences among the aerial-view images themselves (e.g., high-altitude top-down views v.s. low-altitude side views). Is it necessary to consider these viewpoint differences within the aerial-view category, and not just the gap between ground-view and aerial-view? I hope the authors will provide experimental results or statistical data.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies open-vocabulary object detection (OVD) for aerial imagery by transferring ground-view vision–language knowledge with two components: (i) a cross-view image–image contrastive loss that aligns aerial features to frozen ground-view features, and (ii) a multi-instance “text-bag” association that aligns aerial features to sets of class-name variants. Cross-view correspondences are built from category overlaps (direct matches) and pseudo-matches produced by OWLv2 with NMS; vocabulary is expanded using ChatGPT-synthesized synonyms. Only the aerial encoder is finetuned. Experiments on xView, DOTAv2, VisDrone (images/videos), DIOR, and HRRSD report zero-shot gains over a finetuned OWLv2 and even a closed-set YOLOv11.

### Strengths
- [S1] Clear, simple formulation for cross-view alignment that finetunes only the aerial encoder.
- [S2] Broad evaluation across five aerial datasets with consistent zero-shot improvements and useful ablations.
- [S3] The alignment data pipeline is pragmatic (direct overlaps plus OWLv2-based pseudo-matches) and appears model-agnostic.

### Weaknesses
- [W1] Novelty is moderate. Cross-view feature alignment via contrastive image–image pairing and multi-instance text association extends known ideas (e.g., MIL‑NCE) which largely reduces the technical novelty.
- [W2] Important details are underspecified. Temperatures (ρ/σ), confidence/NMS thresholds used to build correspondences, the handling of VisDrone videos, and whether DOTAv2 is evaluated with OBB or converted HBB are not clearly described, affecting reproducibility and comparability.
- [W3] The multi‑instance text‑bag is built from ChatGPT synonyms, but there is no sensitivity analysis (e.g., number/quality of synonyms per class).
- [W4] Pseudo‑match noise is unquantified. Since correspondences rely on OWLv2 detections, the paper should report pseudo‑label precision/recall and show robustness to threshold choices; otherwise training signal quality is uncertain.
- [W5] The “no catastrophic forgetting” claim is shown on a small overlap subset; a larger ground‑view retention test (full LVIS/COCO) would better support the claim.

### Questions
Please see weaknesses to add details and analysis.

### Soundness
3

### Presentation
3

### Contribution
2
