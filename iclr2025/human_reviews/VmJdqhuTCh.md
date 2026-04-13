## Human Reviewer 1

### Summary
The authors identify two fundamental limitations in the existing masked frequency modeling (MFM) paradigm: 1) constant filters overlook the variability of image frequency responses, and 2) no access to naturally looking images during pre-training requires more data to adapt to downstream tasks during fine-tuning. To address 1), the authors adaptively select the masked-out frequencies based on image frequency responses. To address 2), the authors employ a student-teacher framework via self-distillation. Experimental results on image classification, few-shot learning, and semantic segmentation demonstrate the effectiveness of the proposed method compared to the MFM baseline.

### Strengths
-	The proposed method is well motivated. The authors motivate the method by identifying two key limitations in MFM, and propose two interesting solutions to address these drawbacks.
-	The paper is generally well-written and easy to follow.
-	The authors provide a comprehensive analysis based on their method. The experiments are extensive and the results are promising, especially for few-shot settings.

### Weaknesses
-	The idea of using adaptive filers is interesting. However, the fitters still rely on some pre-defined thresholds, e.g., [0.005, 0.01, 0.05]. In practice, the authors may also need to tune these hyper-parameters to achieve the optimal performance for different datasets.
-	For CNN, according to Table 4 in Sec. B.2, the proposed method does not lead to further gains compared with MFM when it comes to full fine-tuning, which makes me concerned about its effectiveness for CNN architectures. Could the authors provide the justification on this?
-	For efficiency analysis, the authors only provide a comparison on GPU memory usage (Table 12, Sec. B.6). A comparison on training time with previous methods is also preferred.

### Questions
See the questions mentioned above. Overall, I think it is an interesting paper with extensive experiments and analysis, which could provide some new insights for the community. Thus, I am leaning to accept this paper.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
5

---

## Human Reviewer 2

### Summary
The paper introduces a self-supervised learning (SSL) approach named FOLK, which stands for FOurier transform compression with seLf-Knowledge distillation. The method aims to address the limitations of previous frequency-based pre-training approaches by adaptively selecting frequencies for masking based on unique image responses. The dual-branch framework leverages both filtered and original images during pre-training, which is claimed to minimize the adaptation requirements for natural-looking images in downstream tasks. The experimental results demonstrate the effectiveness of FOLK, showing competitive performance in various downstream tasks such as image classification, few-shot learning, and semantic segmentation.

### Strengths
The paper presents a new method that combines frequency-based masking with self-knowledge distillation, addressing known limitations in the field of SSL for computer vision tasks. The paper provides extensive experimental results that demonstrate FOLK's effectiveness across a range of tasks and benchmarks, showing improvements over existing state-of-the-art methods.

### Weaknesses
The author proposed two limitations in the introduction, but the experiments did not directly discuss how to address these limitations. Simply showing performance improvements (e.g., image classification tasks) is not enough to support the author's claims.

### Questions
1. The paper primarily focuses on the Com and RCom filters. It would be beneficial to see a comparison with other filtering techniques to establish the robustness and generalizability of the FOLK framework. Could the authors experiment with additional filtering methods, such as Gabor filters or wavelet transforms, and report on their effectiveness?
2. The related work section mentions that MFM has been applied to low-level vision tasks. Since FOLK builds upon MFM, it would be valuable to include a comparison of FOLK's performance on such tasks. Could the authors add experiments that benchmark FOLK against existing methods on low-level vision tasks to provide a more comprehensive evaluation?
3. While the supplementary material shows some results on robustness, a more detailed analysis would be appreciated. Could the authors provide additional benchmarks that specifically measure the robustness of the FOLK framework against various types of image degradations and noise?
4. The paper integrates knowledge distillation into the FOLK framework, but ablation studies regarding the contribution of this component are missing. Could the authors conduct ablation studies to isolate the impact of the knowledge distillation component on the overall performance? This would help readers understand the significance of this technique in the context of the proposed framework.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper introduces Fourier transform compression with self-knowledge distillation (FOLK), a frequency-based self-supervised learning (SSL) method designed to improve pre-training efficiency. FOLK addresses the limitations by adaptively selecting masked frequencies based on image frequency responses and employing a two-branch framework for knowledge distillation. Experimental results show that FOLK achieves competitive performance across various SSL tasks.

### Strengths
1. The framework is applicable and straightforward to understand.
2. The proposed method improves the learning of the student model and facilitates a more efficient training process.
3. The paper presents experiments across multiple datasets and various vision tasks, demonstrating the effectiveness of the proposed method.

### Weaknesses
1. The dual-stream and frequency-domain masking approaches applied in the article are relatively common schemes. Could the authors elaborate further on the motivation of the proposed method?
2. More analysis and experiments are required on the framework design and cost computation, please see the questions.

### Questions
1. Two views (u and v) of the input image are processed through the informed filtering process in the proposed FOLK framework. What is the optimal method for selecting views to enhance the model performance?
2. How can the complexity of the FOLK be reduced to enhance the framework's accessibility? Could you analyze the computational costs of the various methods evaluated across different models?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper proposes a frequency-based SSL method to learn visual representations from unlabeled images, which significantly improves the training performance compared with existing works. In particular, the authors built upon the MFM method and identified two key limitations: 1) pre-defined frequency masking filters that ignore the intrinsic structure in individual images; 2) model pre-trained with frequency-filtered images leads more data to adapt to natural images in downstream model fine-tuning. In response, two specific new designs (a. masked frequency modeling with Com and RCom filters; b. multi-task self-supervision with self-knowledge distillation) are proposed to target these two problems. Their reported experimental studies have shown the effectiveness of their designs.

### Strengths
**Originality**. The paper investigated two fundamental limitations in the MFM work and proposed two novel designs to address these limitations.  The presentation clearly shows what are the novel elements.

**Quality**.  The paper shows a successful way to perform masking in the frequency domain for unlabeled training images. Additionally, the authors provided a proper self-knowledge distillation framework to deal with the negative effect of training with frequency-masked images.

**Clarity**.  The submission neatly shows all the experiments that were carried out and the description of the underlying method is clear.

**Significance and Relevance**.  The topic is very interesting and important. Considering the growing demand for learning effective representations from unlabeled data, this paper pushed the boundary of SSL.

### Weaknesses
**Training Cost**. Given that the proposed method employs a two-branch framework for model training, will it bring additional training costs compared with the original MFM?

**Masking Filters**.  What are the exact formulations of Com and RCom masking? or pseudo code to construct Com and RCom might be helpful.

**Data Augmentations**. In generating two views, u and v, distinct transformations (random cropping, color jittering, etc.) are conducted. It seems no ablation studies are provided for analyzing the effect on the consecutive image frequency masking and final model training.

### Questions
See above weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
5