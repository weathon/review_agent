# A Large-scale Universal Evaluation Benchmark For Face Forgery Detection

- Decision: Reject
- Scores: 6, 5, 6, 5

## Abstract
With the rapid development of AI-generated content (AIGC) technology, the production of realistic fake facial images and videos that deceive human visual perception has become possible. Consequently, various face forgery detection techniques have been proposed to identify such fake facial content. However, evaluating the effectiveness and generalizability of these detection techniques remains a significant challenge. To address this, we have constructed a large-scale evaluation benchmark called DeepFaceGen, aimed at quantitatively assessing the effectiveness of face forgery detection and facilitating the iterative development of forgery detection technology. DeepFaceGen consists of 776,990 real face image/video samples and 773,812 face forgery image/video samples, generated using 34 mainstream face generation techniques. During the construction process, we carefully consider important factors such as content diversity, fairness across ethnicities, and availability of comprehensive labels, in order to ensure the versatility and convenience of DeepFaceGen. Subsequently, DeepFaceGen is employed in this study to evaluate and analyze the performance of 20 mainstream face forgery detection techniques from various perspectives. Through extensive experimental analysis, we derive significant findings and propose potential directions for future research. The code and dataset for DeepFaceGen are available at https://anonymous.4open.science/r/DeepFaceGen-47D1.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents DeepFaceGen, a large-scale evaluation benchmark for face forgery detection combining both task-oriented and prompt-guided generation approaches. The dataset comprises 776,990 real and 773,812 forged face samples (images and videos) generated using 34 different forgery techniques. The authors evaluate 20 mainstream forgery detection methods on this benchmark and derive several key findings, including: (1) detail extraction modules play a crucial role in forgery detection performance; (2) autoregressive-based and diffusion-based techniques produce more realistic forged faces than GAN-based approaches; (3) task-oriented forgeries show better generalization capability than prompt-guided ones.

### Strengths
1. The dataset construction demonstrates substantial effort, integrating both task-oriented and prompt-guided forgery samples, while considering racial fairness and content diversity.

2. The experimental analysis reveals several noteworthy findings through comparative studies of detection methods, particularly in understanding the differences between generation frameworks and their generalization capabilities.

3. The paper provides comprehensive technical content, including attribute-level analysis and clear experimental protocols, making it a useful reference for future research in face forgery detection.

### Weaknesses
1. The presentation of experimental results lacks clarity, particularly in crucial Figures 1, 2, and 3, which appear cluttered and fail to effectively highlight key findings. The use of small relative bar charts without absolute performance numbers significantly limits the paper's utility as a benchmark study.

2. The claimed findings lack rigorous support and appear overly definitive. Many conclusions are drawn from basic performance comparisons without considering the multiple factors that could influence detection performance. The authors should be more cautious in their assertions and provide stronger empirical evidence for each finding.

3. The crucial issue of generalization capability is inadequately addressed. Figure 3's visualization of cross-generalization results is difficult to interpret, and the corresponding tables in Appendix F are poorly organized. Additionally, the paper lacks comparison with methods specifically designed for generalization, instead focusing mainly on in-domain detection approaches.

4. Section 4.3 and Figure 4's feature visualization analysis is unclear, with the figure being particularly difficult to distinguish. It's not evident how this analysis contributes to the paper's overall findings.

Overall, while the paper attempts to cover extensive ground, it suffers from imprecise presentation and insufficiently supported conclusions. A substantial revision is needed to improve the result presentation, strengthen the empirical support for findings, and make the content more accessible to readers.

### Questions
See the Weaknesses section

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This work presents a face deepfake dataset that incorporates 34 face forgery generation techniques. The author conducts training and evaluation experiments on this dataset and presents several findings. This author implements both image and video detectors for a more comprehensive comparison.

### Strengths
- This work implements 34 distinct face forgery techniques for creating fake data, although this work does not create new real faces in their dataset.
- This work involves both spatial and temporal detection methods for comparison, which is new in the existing face forgery benchmark.
- Some findings are new to me such as Finding-10.

### Weaknesses
**1. Unclear and ambiguous definitions for the forgery types:** In this paper, face forgery techniques are categorized into two main types: prompt-guided and task-oriented. However, both of these concepts can be rather abstract and unclear, potentially misleading the community. Specifically:
- In Line 48, the author mentions for the first time that "current deepfake datasets focus on relatively outdated task-oriented based face forgery techniques." However, the meaning of "task-oriented-based face forgery" is not clear. To the best of my knowledge, I have not found any existing research or survey that provides such a definition of face forgery techniques. The author must either provide **appropriate citations and references or clearly define the rationale behind this definition**.
- If the author cannot provide accurate references or detailed explanations, the categorization will remain very unclear and ambiguous. I am not sure whether this category is reasonable and can be accepted by the community.


**2. Several findings in this work have been verified by previous works and might be trivial:**
- Finding-1 and Finding-5 are quite similar, indicating the need to extract more detailed information. In fact, many SOTA detectors have proposed different approaches to enhance detection performance by leveraging more discriminative artifacts or detailed information. So, the conclusion that these SOTA detectors can achieve better performance than the vanilla backbone such as Xception can be rather trivial.
- Finding-11 concludes and highlights the benefit of using different levels of frequency clues for detection. However, **many previous (very early) works have already provided this conclusion**, such as F3Net (ECCV'20), SRM (CVPR'21), and SPSL (CVPR'21). These works reached these conclusions using very old benchmark datasets such as FF++ (ICCV'19). But the author seems to use the latest benchmark and still concludes the same conclusion. What is the significance of creating such a benchmark?
- Finding-6: "existing diffusion-based generation techniques possess a comparable ability to generate forged videos." The same conclusion has been proposed in works like ref[1], making this conclusion trivial.


**3. About the correctness of several findings:**
- In Finding 6, the authors use "detection performance" to infer "the generation quality of diffusion techniques." However, the two may **have a correlation but not a causal relationship**. The worst detection performance can be attributed to many scenarios, while the metrics for quality are FID-like tools used in the image synthesis field. Misconnecting these two concepts is inappropriate. Specifically, I note that diffusion-generated images can have different poses and facial sizes compared to other data, which makes their distribution differ from other deepfake data (as seen in the left-up and right-below corners in Figure 7). But this does not mean that diffusion-generated images are not realistic.


**4. Reproduction Concerns:** 
- I have checked the link (`https://anonymous.4open.science/r/DeepFaceGen-47D1`) provided by the author. It seems to only contain the `model.py` files. Without detailed preprocessing, training, and evaluation codes, it is hard to see whether the results are reproducible and the experimental settings are fair.

ref[1]: Diffusion Models Beat GANs on Image Synthesis, NeurIPS'21.

### Questions
1. What is the meaning of "prompt-guided" and "task-oriented"? What is the rationale or reference for this definition of forgery types?

2. Which findings presented in this work are truly new to the field and have not been shown in existing or previous works?

3. Several recent face forgery datasets such as those mentioned in ref[1,2,3,4,5] are missing from the discussion. What are the advantages and additional contributions of the current dataset over these previous ones?

4. What are the deeper insights regarding the differences between image and video face forgery detectors?

5. Most of the findings and evaluations are empirical observations. What are the underlying reasons behind these observations?

ref[1]: Diffusion Facial Forgery Detection, ArXiv 2024. 

ref[2]: DiffusionFace: Towards a Comprehensive Dataset for Diffusion-Based Face Forgery Analysis, ArXiv 2024. 

ref[3]: Contrastive pseudo learning for open-world deepfake attribution, ICCV 2023. 

ref[4]: DF40: Toward Next-Generation Deepfake Detection, NeurIPS 2024. 

ref[5]: AI-Face: A Million-Scale Demographically Annotated AI-Generated Face Dataset and Fairness Benchmark, ArXiv 2024.

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
3

### Summary
The paper introduces DeepFaceGen, a benchmark specifically developed for face forgery detection. DeepFaceGen includes 463,583 real images, 313,407 real videos, 350,264 forged images, and 423,548 forged videos, generated using 34 common image/video generation techniques. The authors comprehensively evaluate 20 mainstream face forgery detection techniques using DeepFaceGen.

### Strengths
1. The paper constructs a large face forgery dataset, utilizing a variety of generation methods, including Prompt-guided Generation techniques such as Text2Image, Image2Image, and Text2Video, as well as Task-oriented Generation techniques. Compared to previous datasets, this dataset employs a more comprehensive and diverse range of generation methods, resulting in a significantly larger scale.

2. The paper employs 20 detection methods and conducts extensive experiments and analyses using DeepFaceGen.

### Weaknesses
1. The paper may not present a methodological contribution in itself, but the large and comprehensive dataset is valuable.
2. I have some questions regarding missing details. The authors selected 20 detection methods, including both image-level and video-level approaches, but did not clarify the representativeness of these methods.  Additionally, while the authors conducted analyses of texture and frequency domain features, the chosen methods seem to focus more on detection models that utilize frequency domain features.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This work proposes a large-scale benchmark for the deepfake detection task, and the authors draw some findings through extensive experiments.

### Strengths
This paper is easy-to-follow. This work discusses large-scale deepfakes generated by a variety of methods, including recent prompt-guided deepfakes, which is a limitation of prior datasets.
Since deepfake media are rapidly evolving, a reliable and cutting-edge benchmark has foreseeable value in the field of deepfake forensics.

### Weaknesses
While the authors present up to 13 findings, these findings are not impressive enough. There are some shortcomings in the claimed findings including:
*   Some findings seem to simply reassert well-known motivations from previous works, such as Finding 1 with MAT(Zhao et al, CVPR 2021).
*   Some findings are common sense, such as Finding 2&7.

Furthermore, it is recommended that authors consolidate a Findings list in the conclusion part, as the presented Findings are dispersed throughout the 36 pages of the whole paper, making it less friendly for reading.

### Questions
I hope the authors can provide more constructive insights for designing deepfake detection methods according to the proposed benchmark, such as:

*   For model-centric methods, what impact do various architectures (e.g., CNN vs. Transformers) or modalities (e.g., frequency vs. noise) have on different kinds of deepfakes?
*   For data-centric methods, which data augmentations or training strategies are beneficial for deepfake detection?

### Soundness
3

### Presentation
3

### Contribution
3
