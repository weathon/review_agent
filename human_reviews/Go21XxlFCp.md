# Learning Adaptive Lighting via Channel-Aware Guidance

- Avg Score: 5.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 5, 3

## Abstract
Learning lighting adaption is a key step in obtaining a good visual perception and supporting downstream vision tasks. There are multiple light-related tasks (e.g., image retouching and exposure correction) and previous studies have mainly investigated these tasks individually. However, we observe that the light-related tasks share fundamental properties: i) different color channels have different light properties, and ii) the channel differences reflected in the time and frequency domains are different. Based on the common light property guidance, we propose a Learning Adaptive Lighting Network (LALNet), a unified framework capable of processing different light-related tasks. Specifically, we introduce the color-separated features that emphasize the light difference of different color channels and combine them with the traditional color-mixed features by Light Guided Attention (LGA). The LGA utilizes color-separated features to guide color-mixed features focusing on channel differences and ensuring visual consistency across channels. We introduce dual domain channel modulation to generate color-separated features and a wavelet followed by a vision state space module to generate color-mixed features. Extensive experiments on four representative light-related tasks demonstrate that LALNet significantly outperforms state-of-the-art methods on benchmark tests and requires fewer computational resources. We provide an anonymous online demo at [LALNet](https://xxxxxx2025.github.io/LALNet/).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors proposed a new method for lighting adaptation, which is inspired by the two key insights. Although such insights are not very new, the overall network structure is logically derived from them, making the network structure solid. A separate application of color-separated and color-mixed branches and their combination by LGA can be considered novel. Its effectiveness is justified by four different lighting-related tasks, which is also promising. The manuscript is well-written and well-organized.

### Strengths
The network structure consisting of the color-separated and color-mixed branches seems original and has enough technical contribution. Each branch is designed with persuasive supporting explanations. The proposed network shows SOTA performance on four different tasks, demonstrating significant margins from the second-best methods.

### Weaknesses
The authors' motivation is supported mainly by few examples in Figures 2 and 8. The authors are recommend to provide statistical analysis supporting the motivation rather than showing few chosen images. 

I consider the experimental justification is somewhat biased. The authors used the results on the HDR+ dataset as a main performance evaluation, which is not a very widely used dataset. Almost all the baseline models in Table 1 are not developed for image retouching, making the comparison not fair enough. Consequently, as the ablation studies show, the model without DDCM and LGA, which does not contain technical contributions, already significantly outperforms the second-best method. I consider that the performance analysis and ablation studies should be conducted on the LLIE task using the LoL dataset.

### Questions
Almost all the baseline models in Table 1 are not developed for image retouching, making the comparison not fair enough. Consequently, as the ablation studies show, the model without DDCM and LGA, which does not contain technical contributions, already significantly outperforms the second-best method. I consider that the performance analysis and ablation studies should be conducted on the LLIE task using the LoL dataset.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a unified framework LALNet for light-related tasks, including HDR, Tone mapping, LLIE, Exposure correction. The argument is that different color channels demonstrate different lighting properties, and channel differences in time and frequency are also different. Based on that, a dual domain channel modulation is proposed with light guided attention for integration. Experiments are conducted respect to the four tasks, demonstrates some superiority according to results.

### Strengths
The proposed color channel properties regarding the lighting, is somewhat interesting, however, I still have some questions, see below

### Weaknesses
see the questions

### Questions
This paper is focused on lighting-related tasks, but I'm curious about what would happen if we switched to other color spaces, such as Lab or HSV, which separate color from luminance. Would it be easier to work with these color spaces compared to the RGB domain?

How about the retinex theory, e.g., intrinsic decomposition, that can seperate the illumination from textures. Would it be easilier than working in the rgb domain where lighting mixed with colors, as discussed in the paper, rgb space demonstrates different channel porperties in color and frequency

For the evaluation, authors are suggested to compare with the SOTA methods respect to each of the task. Currently, some of the SOTA methods are omitted.  

For some visual comparisons, it is hard to notice the differences when looking at the color images.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents Learning Adaptive Lighting Network, a unified framework for multiple light-related image enhancement tasks, including image retouching, tone mapping, low-light enhancement, and exposure correction. The authors propose channel-aware guidance to handle the unique properties of each color channel, integrating color-separated and color-mixed features through a Dual Domain Channel Modulation and Light Guided Attention. Experimental results show that LALNet achieves state-of-the-art performance on benchmark datasets, outperforming existing methods both in accuracy and computational efficiency.

### Strengths
- The paper presents a novel approach by leveraging color-separated and color-mixed features to address multiple light-related tasks within a single model, which reflects a strong methodological innovation.
- The results across four different tasks and datasets, with quantitative comparisons, convincingly support the effectiveness of LALNet, showing substantial improvements over state-of-the-art methods.
- The distinction between time and frequency domain representations of different color channels is well-grounded.

### Weaknesses
- The paper does not clarify whether LALNet shares parameters between models evaluated across different tasks, leaving it unclear if LALNet operates as a truly unified model. This aspect is crucial to evaluate the generalizability of LALNet as a single framework capable of adapting to diverse lighting tasks. 
- The main contributions rely heavily on combining established well-known methods (e.g., wavelet-based modulation, vision state-space models, and channel-wise attention) with minimal modifications. Given recent work in similar domains, the marginal improvements over these methods do not provide a substantial contribution to the field.
- The paper could benefit from a clearer architectural diagram to visually depict the relationships between DDCM, LGA, and wavelet modulation.
- Terminology like “adaptive lighting” could be further clarified to distinguish it from traditional exposure correction or low-light enhancement.
- Formatting inconsistencies in equations (e.g., missing commas or misaligned subscripts) detract slightly from readability.

### Questions
- Does LALNet share parameters across tasks, or is a separate model trained and optimized for each task? How do you address cross-task generalization if parameters are not shared?
- How does LALNet perform on new or more diverse lighting conditions, such as HDR imaging in natural outdoor scenes?
- Could you clarify the contribution of VSSM versus traditional convolution-based approaches? Specifically, how much does VSSM contribute to cross-channel consistency?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes a framework for learning adaptive lighting with light property guidance. First, the authors propose DDCM for extracting color-separated features and capturing the light difference across channels. Then, the LGA utilizes color-separated features to guide color-mixed features for adaptive lighting, achieving color consistency and color balance. Extensive experiments demonstrate that the proposed method outperforms state-of-the-art methods on various light-related benchmarks.

### Strengths
1, The proposed method achieves state-of-the-art performance.
2, The method is easy to follow.

### Weaknesses
1, The novelty is limited. The model mainly combines existing modules to form the proposed method, e.g., wavelet, mamba, etc. 

2, The authors need to improve the writing. The current version cannot satisfy the standard of ICLR.

3, The authors should provide the comparisons on real-world data as well as the non-reference metrics such as NIQE, PI, PIQE, etc.

4, The motivation of the paper is not insightful enough.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1
