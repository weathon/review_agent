# When Glass Disappears at Night: A Novel NIR-RGB Multi-modal Solution

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Glass surface detection (GSD) has recently been attracting research interests. However, existing GSD methods focus on modeling glass surface properties for daytime scenes, and can easily fail in nighttime scenes due to significant lighting discrepancies. We observe that, due to the spectral differences between Near-Infrared (NIR) light sources and common LED lights, NIR and RGB cameras capture complementary visual patterns (e.g., light reflections, shadows, and edges) of glass surfaces, and cross-comparing their lighting and reflectance information can provide reliable cues for GSD at nighttime. Inspired by this observation, we propose a novel approach for nighttime GSD based on the multi-modal NIR and RGB image pairs. We first construct a nighttime GSD dataset, which contains 6,192 RGB-NIR image pairs captured in diverse real-world nighttime scenes, with corresponding carefully-annotated glass surface masks. We then propose a novel network for the nighttime GSD task with two novel modules: (1) a RGB-NIR Guidance Enhancement (RNGE) module for extracting and enriching the NIR reflectance features with the guidance of RGB reflectance features, and (2) a RGB-NIR Fusion and Localization (RNFL) module for fusing RGB and NIR reflectance features into glass features conditioned on the multi-modal illumination discrepancy-aware features. Extensive experiments demonstrate that our method outperforms state-of-the-art methods in nighttime scenes while generalizing well to daytime scenes. We will release our dataset and codes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This submission introduces a method for glass detection at night by proposing an approach for nighttime glass surface detection.

### Strengths
1. The author claims that this is the first method for the related topics. 
2. This work also contributes a dataset.

### Weaknesses
1. I remain unconvinced by the rationale for employing NIR images. Is the underlying reason that the task is entirely unsolvable using the RGB modality alone? If this is the case, the authors should conduct and present more comprehensive experiments to clearly illustrate the shortcomings of existing RGB-based methods.

2. If the additional modality is necessary, why can the authors only consider NIR? Is it possible to use another modality?

3. Though the NIR camera can provide some helpful information at night, it also has obvious limitations. For example, the distance of the NIR may limit its usage. Besides, NIR light can sometimes lead to a see-through effect, which may violate privacy. 

4. For such a fusion task, it is challenging to develop a framework with novelty. From an architectural perspective, the proposed approach shares a very similar framework to previous approaches. It presents only some standard feature fusion modules. It is better to rephrase the contribution at this point.

### Questions
See me concerns in the weakness.

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
This paper proposed a night-time glass surfaces detection method. The reflectance and illumination components of RGB and NIR images are first decomposed, respectively. Then the semantics-aware features from the RGB reflectance are extracted to enhance the material-aware features from the NIR reflectance. The multi-modal features of the lighting and material information are fused together to detect glass areas. The authors also constructed a night-time glass surface detection dataset containing 6,192 RGB-NIR image pairs with the corresponding manually annotated glass surface masks. Comparative experimental results showed that the proposed method outperformed the existing methods.

### Strengths
This paper introduced a relatively new problem of night-time glass surface detection, and showed the proposed method yields better performance than the compared existing methods.

A new dataset of night-time glass surface detection was constructed, which is welcomed in the research community.

### Weaknesses
1) The proposed night-time glass surface detection problem is basically a combined problem of low-light image enhancement and glass surface detection. Specifically, the proposed Retinex decomposition module decomposes input low-light image into reflectance and illumination components, which is similar to the behavior of low-light image enhancement. Table 2 seems to provide the related experimental results, however it is not easy to figure out the benefit of the proposed method in this experimental setting, since the performance of low-light enhancement was not guaranteed. It is recommended to visualize the input low-light enhanced images as well.
 
2) The architectures and their explanations in Sec. 4 are easy to follow, and therefore the related equations such as (1), (2),(3) are not required. Furthermore, instead of naïve explanation about the procedures of architecture, more evidences and discussions are recommended to be provided to support that the proposed architecture exploits complementary features between RGB and NIR effectively. In this context, proper visualization of the complementary features between RGB and NIR images are also recommended. 

Also, the explained behavior of the proposed architecture in Figure 4 can be confirmed by visualizing intermediate features such as X^i_d and X^i_f. 
M_d in Figure 5 does not properly highlight the glass area. 
It is also recommended to discuss the results of intermediate feature maps in Figure 5.

### Questions
My main concerns are given in the Weaknesses section.

In addition, some minor questions are as follows:

- Why are the distributions of the glass area ratio different between training dataset and testing dataset, even though the two datasets are randomly split? 

- Are the existing datasets of GDD and GSD in Table 7 also multimodal datasets including NIR images? 

- The GT in the 3rd row in Figure 12 seems to fail to capture the opened window, while NRGlass successfully captured.

- In A. 10, instead of naïve reporting the failure cases, discussion why the proposed method fails on the images is recommended.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the challenge of nighttime glass surface detection (GSD) by leveraging the complementary sensing of RGB and active Near-Infrared (NIR) imaging. The authors build the first large-scale nighttime RGB–NIR dataset with 6,192 paired images and high-quality annotations. They propose a Retinex-based dual-branch network with two key modules: RNGE, which uses RGB semantic features to enhance NIR material and boundary cues, and RNFL, which fuses RGB–NIR features via illumination-aware cross-modal attention. The method outperforms state-of-the-art approaches on nighttime scenes

### Strengths
- The work focuses on the “nighttime” domain, an underexplored yet highly practical setting ,where RGB-only cues fail under low or complex artificial illumination. The use of active NIR sensing as a complementary modality is both well-motivated and technically novel.

- The paper presents the first large-scale RGB–NIR glass surface dataset for nighttime scenes. It includes realistic environments and detailed annotations, offering a valuable benchmark for future research.

### Weaknesses
- This paper lacks a comprehensive analysis of the diversity of the dataset, such as curved or irregular glass surfaces, extreme lighting variations, or outdoor conditions. 

- The study primarily compares against glass-specific models. It would be more convincing to include general cross-modal or RGB–NIR segmentation/detection baselines. 

- Retinex pre-processing introduces computational overhead and potential instability under noisy nighttime conditions. It would be valuable to explore lighter or integrated alternatives, such as explicitly modeling reflectance/illumination branches within the backbone itself, to reduce latency.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes method and multi-modal NIR-RGB paired dataset to do night-time glass surface detection. 
The authors test inversigate different backbones (including swinV2 and ResNet). The results look close to the ground truth from the paper.

### Strengths
1. The proposed NIR-RGB paired dataset could be useful for the community. 
2. The authors conduct ablation studies on their pipeline.

### Weaknesses
1. Lack of comparison with any state-of-the-art zero-shot segementation models e.g., SAM 2.
2. Images in the paper seems to be in a very narrow distribution, therefore, the results are less convincing, and could indicate a potential overlap. 
3. The retinex for illumination decompositon sounds very arbitry, retinex based decomposition could contain with many leakages, and a better intrinsic decomposition method could be used.

### Questions
1. Are there any comparisons with segementation models like SAM 2?
2. Are there any out-of-distribution results? I am curiours if the model has a generalizable capacity in the real world images. 
3. Besides the glass detection, are there any other applications that can employ this method?

### Soundness
2

### Presentation
2

### Contribution
2
