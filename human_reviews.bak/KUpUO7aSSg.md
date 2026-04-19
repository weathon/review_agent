# DODA: Diffusion for Object-detection Domain Adaptation in Agriculture

- Decision: Reject
- Scores: 5, 3, 6, 6

## Abstract
Object detection has wide applications in agriculture, but the trained models often struggle to generalize across diverse agricultural environments. To address this challenge, we propose DODA (\underline{D}iffusion for \underline{O}bject-detection \underline{D}omain Adaptation in \underline{A}griculture), a unified framework that leverages diffusion models to generate domain-specific detection data for multiple agricultural scenarios. DODA incorporates external domain embeddings and an improved layout-to-image (L2I) approach, allowing it to generate high-quality detection data for new domains without additional training. We demonstrate DODA's effectiveness on the Global Wheat Head Detection dataset, where fine-tuning detectors on DODA-generated data yields significant improvements across multiple domains (maximum +15.6 AP). DODA provides a simple yet powerful approach to adapt object detectors to diverse agricultural scenarios, lowering barriers for more plant breeders growers to use detection in their personalized environments.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper presents DODA, a framework that uses diffusion models to improve object detection in diverse agricultural domains. By integrating domain embeddings and a layout-image-to-image (LI2I) method, DODA generates domain-specific images without extra training, enhancing model adaptability across different environments. Tests on the Global Wheat Head Detection dataset show substantial AP improvements, underscoring DODA's potential for accessible, customized object detection in agriculture.

### Strengths
1.  DODA incorporates external domain embeddings and is able to generate high-quality detection data for new domains without additional training. This approach is inspiring.
2.  Significant improvements are witnessed both in agriculture detection and L2I generation performance.

### Weaknesses
1.	Fields of application. Restricting the research to the agriculture domain limits the impact of this paper. I am curious if it could also be applied to other fields, such as autonomous driving."
2.	The pipeline of ‘layout-text-to-image’ can be seen as a simplified and derived version of ControlNet [1] , lacking innovation in this aspect.
3.	In Figure 2(c), most of the content is dedicated to describing existing text-based L2I methods, while the description of the proposed method is limited. Additionally, the figure only includes content related to 'Encoding Layout Images' (Section 3.4), lacking description of 'Incorporating Domain Embedding' (Section 3.3).

[1] Adding Conditional Control to Text-to-Image Diffusion Models, ICCV2023

### Questions
1.	As a L2I generation method, I have questions about the performance of DODA. The pipeline of ‘layout-text-to-image’ can be seen as a simplified and derived version of ControlNet [1]. However, as GeoDiffusion [2] reported, ControlNet can only achieve 25.2 mAP in COCO 2017 valuation set as a L2I generation method by converting layouts to masks, while DODA claims 42.5 mAP which has an improvement of 17.3%. Therefore, I am curious about the reasons behind huge performance improvement.
2.	Fields of application. See Weaknesses 2.

[1] Adding Conditional Control to Text-to-Image Diffusion Models, ICCV2023
[2] Geodiffusion: Text-prompted geometric control for object detection data generation, ICLR2024

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The authors propose a diffusion model framework for generating detection data from a range of agricultural scenarios.  Performance is tested on the Global Wheat Head Detection dataset.

### Strengths
Incorporation of this approach appears to improve detection results, often significantly.  

The ablation studies explore pre-training dataset size, encoder impact, number of layers, and embedding positioning.

### Weaknesses
Only a single dataset is explored.  Given the claim that this is a unified framework for domain adaptation, performance should be tested over a wide range of datasets, crops, and environments.  Having a single dataset significantly limits the claims which can be made around applicability and general usefulness.  

Additionally, there is nothing in the method that seems to be specifically agricultural-focused.  Is this approach necessary for agricultural setting?  Would it work in other domains?

The writing (both at the sentence/grammatical level as well as high-level sentence organization) could be improved in places- some of the clauses are dangling and incomplete.

Related works around object detection in the agricultural space are largely dominated by a narrow group of authors.  There is significant amount of work done in the detection/counting space across many crops- I would suggest the authors look at the work done by these labs including works such as Ghosal 2019, Malambo 2019, Osco 2020, Gene-Mola 2020, Akiva 2020, Akiva 2021, Hobbs 2021, and others.  It is important to represent the broad work of this area across crops of interest done by a wide range of labs.

### Questions
The language in the Related Work is particularly unclear.  The sentences tend to be fragmented and do not follow well.  I would suggest a rather substantial revision of this section.  

The last sentence of the first paragraph of 3.3 is extremely length and unclear and should be broken up.  The first three paragraphs of 3.3 are largely narrative as opposed to method dominant- I would suggest putting this motivating information in the introduction and keep only the description of the present method in this section.

Figure 2b is very unclear- is the reader supposed to interpret that as 6 subpanels that have been snacked across two rows?  The caption doesn't make this clear either.

See weaknesses.  It's unclear why this method is directed at a single domain and not domain adaptation broadly. 

It would be helpful to show Table 3 in terms of number of samples, not percent, so the reader knows the actual magnitude of data.

### Soundness
1

### Presentation
2

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
This paper presents DODA, an new framework utilizing diffusion models to improve object detection across diverse agricultural environments. DODA tackles the challenge of adapting detection models to various agricultural domains by integrating external domain embeddings alongside a layout-to-image (LI2I) method, enabling the generation of high-quality, domain-specific detection data without the need for additional model retraining. Experimental evaluations on the Global Wheat Head Detection dataset reveal substantial enhancements in detection performance across multiple domains, with a notable average increase in accuracy.

### Strengths
1. the paper introduces a novel way to achieve few shot domain adaptation by using diffusion model, leading to more accurate and contextually relevant synthetic data.

2. The research focus on agriculture data is important and will benefit the community.

3. The DODA framework is designed to be user-friendly, requiring minimal training.

### Weaknesses
1.  In the related work section, the authors are suggested to enrich the related works in few shot domain adaptation task. According to the description in Section 4.2, the adaptation relies on few samples from the new domain, which is within the few shot domain adaptation setting. However the authors did not clearly point it out while renaming it as DODA.


2. The authors should enrich the details regarding how to choose the domain reference images. Is it randomly? How about the variance of the performance on different set of reference images?


3. which visual encoder is used as mentioned on line 210? Can it generalize to other visual encoder?


4. The authors are suggested to compare the proposed approach with other existing few shot domain adaptation methods to demonstrate the effectiveness of the proposed new method, e.g., a,b,c,d,e, and f.

a. Jing, T., Xia, H., Hamm, J., & Ding, Z. (2023). Marginalized augmented few-shot domain adaptation. IEEE Transactions on Neural Networks and Learning Systems.

b. Wu, Y., Li, Z., Wang, C., Zheng, H., Zhao, S., Li, B., & Tao, D. (2024). Domain re-modulation for few-shot generative domain adaptation. Advances in Neural Information Processing Systems, 36.

c..Gao, Y., Yang, L., Huang, Y., Xie, S., Li, S., & Zheng, W. S. (2022, October). Acrofod: An adaptive method for cross-domain few-shot object detection. In European Conference on Computer Vision (pp. 673-690). Cham: Springer Nature Switzerland.

d. Gao, Y., Lin, K. Y., Yan, J., Wang, Y., & Zheng, W. S. (2023). AsyFOD: An asymmetric adaptation paradigm for few-shot domain adaptive object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 3261-3271).

e. Nakamura, Y., Ishii, Y., Maruyama, Y., & Yamashita, T. (2022). Few-shot adaptive object detection with cross-domain cutmix. In Proceedings of the Asian Conference on Computer Vision (pp. 1350-1367).

f. Corral-Soto, E. R., Nabatchian, A., Gerdzhev, M., & Bingbing, L. (2021, May). Lidar few-shot domain adaptation via integrated cyclegan and 3d object detector with joint learning delay. In 2021 IEEE international conference on robotics and automation (ICRA) (pp. 13099-13105). IEEE.


5. On line 406, why did the authors choose 200 as the generated image number? Is there any ablation study regarding this number?


6. Lack of failure case analysis.

### Questions
1. Could the authors enrich the related works in few-shot domain adaptation? Given the reliance on few samples from new domains in Section 4.2, would it be appropriate to explicitly situate this work within the few-shot domain adaptation setting?

2. How are domain reference images selected for DODA? Is the selection random, and has the variance in performance across different sets of reference images been analyzed?

3. Which visual encoder is specifically used in line 210, and can the approach generalize to other types of visual encoders?

4. To strengthen the validation of DODA, would the authors consider comparing it against other few-shot domain adaptation methods, such as those proposed by Jing et al. (2023), Wu et al. (2024), Gao et al. (2022), Gao et al. (2023), Nakamura et al. (2022), and Corral-Soto et al. (2021)?

5. In line 406, why was the number of generated images set to 200? Was any ablation study conducted to examine the impact of this number?

6. Could the authors provide a failure case analysis to highlight scenarios where DODA may struggle, as this would offer insights into potential limitations of the approach?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper discuss the development of the DODA framework for object detection in agriculture, showcasing its effectiveness in generating high-quality detection data across diverse agricultural domains.

### Strengths
1. Decouples domain-specific feature learning from the diffusion model, improving adaptability across diverse agricultural environments.
2. Introduces the LI2I method to enhance control over image layouts, leading to improved label accuracy.
3. Demonstrates the effectiveness of pre-training with additional unlabeled data in enhancing the quality of generated detection data.

### Weaknesses
1. The experiment appears to be somewhat limited. While the proposed method is tailored for agricultural settings, I would recommend the authors to transfer it to natural environments, such as cityscapes, to compare its effectiveness.
2. The method proposed in this paper does not seem to specifically address issues present in agricultural settings.
3. There is a lack of essential visualization of intermediate processes and comparisons.

### Questions
1. The number of methods selected for comparison is not sufficient and not up-to-date, it is recommended that the authors consult the latest works in the relevant field
2. The authors only provide quantitative results, which is not enough, combining qualitative results is necessary, why can work well on different domains.
3. Lack of necessary limitation analysis.

### Soundness
3

### Presentation
3

### Contribution
2
