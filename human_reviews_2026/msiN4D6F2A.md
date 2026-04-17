# cgDDI: Controllable Generation of Diverse Dermatological Imagery for Fair and Efficient Malignancy Classification

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Skin diseases impact the lives of millions of people around the world from different backgrounds and ethnicities. Therefore, accurate diagnosis in the dermatological domain requires focused work toward fairness in different skin-toned populations. However, a significant lack of expertly annotated dermatological images, especially those describing underrepresented skin tones and rare diseases, slows progress toward broadly accurate models and clear fairness metrics. In this work, we introduce **C**ontrollable **G**eneration of **D**iverse **D**ermatological **I**magery (**cgDDI**), a method capable of (1) synthesizing pixel-perfect in-distribution healthy samples, (2) lesion-mapping extremely rare lesions onto novel skin-tone combinations without training and (3) efficient high-fidelity parametric generation with as few as $10$ training samples. Our approach is controllable via learned disease-specific prompts or skin tone descriptors, either visually or textually, allowing for selection of key sensitive attributes. We leverage cgDDI to grow a $656$ real-image dataset by more than $400\times$. The resulting skin-tone-balanced dataset enables the development of accurate classification systems along with significant improvement on essential fairness metrics. Malignancy classification experiments on the Diverse Dermatology Images (DDI) benchmark shows our method reaches competitive performance ($86.4$% accuracy) when trained exclusively on our synthetic data and state-of-the-art performance ($90.9$% accuracy) when fine-tuned on real data. Additionally, we achieve leading metrics for Predictive Quality Disparity, Demographic Disparity, Equality of Opportunity as well as equitable generative image quality measurements for underrepresented skin-tones and rare diseases. We publish code, model weights, and generated datasets at https://anonymous.4open.science/r/ControllableGenDDI in support of further research in this direction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes cGDDI, a controllable generation framework designed to generate diverse dermatology images conditioned on key sensitive attributes. This model enables (1) generation of in-distribution healthy samples, (2) mapping lesions onto novel skin tones, and (3) efficient parametric generation from limited training samples. 

Using cGDDI, the authors expand the DDI dataset by 400 x and compare models trained on synthetic data against prior fairness-focused models in dermatology image analysis.

### Strengths
* S1: This paper explores image generation for fairer dermatology image analysis, which is of interest to the community. 

* S2: Using diffusion models for controllable generation on skin attributes is well-motivated. The design of three generative pipelines, including healthy synthetics, lesion-mapped synthetics, and semantic synthetics, is conceptually sound.  

* S3: The paper is clear and easy to follow.

### Weaknesses
**Major Weakness**

* W1: The methodological novelty appears to be limited. The proposed framework resembles a straightforward adaptation of conditional diffusion models for skin image generation. 

* W2: The experimental scope is narrow. The authors should compare with prior generative methods for dermatology, such as [R3] and the works mentioned in Tab. 1, as well as more skin-tone-fairness-focused methods, such as [R1] and [R2]. Even if they were not originally designed for the DDI dataset, reproducing or fine-tuning them on DDI (or evaluating cGDDI on external datasets like Fitzpatrick17k) would make the results more convincing.


[R1] Bayasi, Nourhan, et al. "BiasPruner: Mitigating bias transfer in continual learning for fair medical image analysis." Medical image analysis (2025).

[R2] Xu, Zikang, et al. "Fairadabn: Mitigating unfairness with adaptive batch normalization and its application to dermatological disease classification." MICCAI 2023.

[R3] Pakzad, Arezou, Kumar Abhishek, and Ghassan Hamarneh. "CIRCLe: Color invariant representation learning for unbiased classification of skin lesions." ECCVW 2022.

[R4] Naveed, Asim, et al. "Ra-net: Region-aware attention network for skin lesion segmentation." Cognitive Computation (2024).

### Questions
**Primary  Questions/Suggestions**

* QS1: Could the authors distinguish the proposed method with existing works? What is the methodological novelty of this paper?

 * QS2: Given the domain specificity, I feel this paper is more suitable for medical conferences such as MICCAI or IPMI. The paper's contribution is somewhat narrow for ICLR, which typically emphasizes methodological advances with broader applicability.

* QS3: Could the authors provide comparisons with SOTA models in skin tone fairness and generative approaches?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes cgDDI, a pipeline for generating controllable dermatology images across different skin tones. The pipeline combines (i) latent-diffusion inpainting to create high-fidelity healthy skin canvases, (ii) a non-parametric lesion-mapping algorithm to transplant real lesions onto valid skin regions, and (iii) parametric text-conditioned diffusion (textual inversion + LoRA with prior-preservation) for disease/skin-tone–aware synthesis. Using DDI/sDDI as the base, the authors synthesize 266,136 images spanning three data types (healthy, lesion-mapped, and semantic).

### Strengths
- The work is well-motivated, targeting the non-trivial research problem of synthesizing dermatological images to address data scarcity, bias, and imbalance. The paper is built on a thorough investigation of related work, positioning its contributions to address a clearly defined unsolved problem.

- The pipeline is described clearly and is technically sound. In particular, first generating in-distribution healthy images and reusing them for Prior Preservation Loss is an excellent idea: it provides a targeted regularization set that mitigates catastrophic forgetting/model drift during fine-tuning on small, domain-specific data.

- The experiments are well-designed and demonstrate effectiveness with tangible improvements in both classification accuracy and fairness metrics. The paper also provides qualitative evidence that the method works as intended.

- The commitment to open-sourcing the synthesized images and models is a significant plus and will be a valuable contribution to the research community.

### Weaknesses
- The framework's (specifically the inpainting and lesion-mapping components) reliance on high-quality, human-annotated segmentation masks (from sDDI) is a limitation. Since such masks are often unavailable for other dermatological datasets, this dependency may limit the practical applicability and scalability of the method to new data sources.

- All experiments are centered on the DDI dataset. While this is a high-quality, biopsy-confirmed dataset, there is no external validation on other datasets (e.g., SCIN, Fitzpatrick17k). This leaves the cross-dataset robustness and generalizability of the generative models under-explored. While the authors note this as a limitation, it remains a key area for future validation to prove the method's effectiveness out-of-distribution.

### Questions
- Other recent works, such as Wang et al. (2024), use mask-free image-to-image translation to circumvent the need for segmentation masks. What do the authors see as the primary advantages of their mask-based approach (inpainting + lesion-mapping) compared to these mask-free methods?
- The current framework controls for disease and skin tone. Does the model learn any implicit priors about the anatomical feasibility of a condition (e.g., certain lesions appearing on specific body parts)? Or could the parametric model, for example, generate a lesion on an anatomically implausible location?
- If feasible, please include a small blinded dermatologist study/test to strengthen the clinical validity of the synthesized images of diseases. It seems like human evaluation is only involved in generating healthy imagery.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a framework named cgDDI to generate diverse dermatological imagery aimed at improving the fairness of skin disease classification. The main contribution lies in the cgDDI dataset, which is built upon this framework and enables fair classification across race, sex, and age groups.

### Strengths
1. The paper addresses an important problem — bias in generative models, particularly in the context of dermatological image synthesis. The proposed cgDDI framework allows for controllable and fair image generation, which is valuable for fairness research.

2. The generated dataset constitutes a meaningful contribution to the community. It has great potential to support fair and robust skin disease diagnosis, especially for rare conditions. As most existing generative models face limitations in producing reliable medical images, such curated datasets can substantially promote the adoption of AI in clinical dermatology.

3. The paper provides a relatively comprehensive presentation of the framework, data generation process, and classification results, demonstrating a solid understanding of both methodology and application.

### Weaknesses
1. The experimental results are somewhat unclear. In Table 2, the "Method" column refers to different combinations of datasets, which indeed suggests that synthetic data contributes to fairer classification. However, these findings are limited to the dataset proposed in this work. It would strengthen the claim to include comparisons with other datasets, as in Table 1.

2. The technical novelty appears limited. Most components seem to be adapted from existing methods, and the unique methodological contribution is not clearly articulated. The authors should clarify which elements of cgDDI are genuinely novel and how they differ from prior works.

3. Section 3 is difficult to follow in terms of logical flow and presentation. The methods are described mostly in text form, making it hard to identify the key design ideas. It would be helpful to include formulations, diagrams, or schematic illustrations to make the framework easier to understand.

### Questions
1. Is this paper more like a dataset contribution?

2. Whether the proposed framework can be applied to other base data, like the MIMIC dataset.

### Soundness
2

### Presentation
3

### Contribution
3
