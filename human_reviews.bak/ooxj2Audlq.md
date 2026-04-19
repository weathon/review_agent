# Stable Segment Anything Model

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
The Segment Anything Model (SAM) achieves remarkable promptable segmentation given high-quality prompts which, however, often require good skills to specify. To make SAM robust to casual prompts, this paper presents the first comprehensive analysis on SAM’s segmentation stability across a diverse spectrum of prompt qualities, notably imprecise bounding boxes and insufficient points. Our key finding reveals that given such low-quality prompts, SAM’s mask decoder tends to activate image features that are biased towards the background or confined to specific object parts. To mitigate this issue, our key idea consists of calibrating solely SAM’s mask attention by adjusting the sampling locations and amplitudes of image features, while the original SAM model architecture and weights remain unchanged. Consequently, our deformable sampling plugin (DSP) enables SAM to adaptively shift attention to the prompted target regions in a data-driven manner. During inference, dynamic routing plugin (DRP) is proposed that toggles SAM between the deformable and regular grid sampling modes, conditioned on the input prompt quality. Thus, our solution, termed Stable-SAM, offers several advantages: 1) improved SAM’s segmentation stability across a wide range of prompt qualities, while 2) retaining SAM’s powerful promptable segmentation efficiency and generality, with 3) minimal learnable parameters (0.08 M) and fast adaptation. Extensive experiments validate the effectiveness and advantages of our approach, underscoring Stable-SAM as a more robust solution for segmenting anything. Codes are at https://github.com/fanq15/Stable-SAM.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes Stable SAM, which aims to improve the segmentation ability of SAM under low-quality prompts. It focuses on the segmentation stability of SAM under different quality prompts and proposes a stability metric called ST. To improve the stability, Stable SAM proposes deformable sampling plugin (DSP) and dynamic routing plugin (DRP), which enable SAM to adaptively shift attention to the prompted target regions and switch SAM between the deformable and regular grid sampling modes.

### Strengths
1. The paper is well-written and well-presented.
2. The problem addressed in this paper is novel and focuses on the segmentation capabilities of SAM under low-quality prompts.
3. The results of this paper are good and the experiments are sufficient.

### Weaknesses
1. Why the adjustment of the attention sampling position can achieve calibration of noisy prompts through learnable offsets? This lacks intuitive motivation and theoretical justification. The visualization comparison in the paper does proves that this method can produce good generalization effects, but the transition here is a bit abrupt. I hope the author can explain the motivation behind this idea in more detail.

2. Line 245 of the paper mentions "Note that conventional deformable attention optimizes both the offset network and attention module. Thus directly applying deformable attention to SAM is usually suboptimal, because altering SAM’s original network or weights", but intuitively, retraining all components may be the optimal result. Freezing the parameters trained in a predetermined way and re-adding some parameters for adaptation seems to be a compromise solution and result. However, this is still a conjecture and there is no necessary evidence to prove this statement. Perhaps SAM needs to be pre-trained again.

3. The purpose of DRP is to control the DSP’s activation to prevent unwanted attention shifts. Are there any experiments that prove that the output of DRP is consistent with the expected intuition under different quality prompt inputs? In addition, in the case of precise prompt input, whether increasing the output strength of DRP leads to worse results, and vice versa. I think sufficient experiments are needed here to verify the intuitive conjecture of DRP.

4. RTS makes the model produce better performance, I think it is necessary to compare RTS+SAM (w/ and w/o deformable attention).

5. According to Table 7, I cannot understand why SAM+DSP can achieve a significant performance improvement (even in the case of a single click) with only a small amount of fine-tuning using high-quality prompts. I think this phenomenon is not in line with common sense and requires sufficient evidence for demonstration and analysis.

6. The strategy proposed in the "Robust Training Against Ambiguous Prompts" section will cause the model to tend to segment larger objects when faced with ambiguity, and cannot handle the possible multiple segmentation results expected by the user, which is not conducive to solving the ambiguity problem of interactive segmentation.

### Questions
Please refer to the Weaknesses.

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
The paper presents Stable-SAM for enhancing SAM's segmentation stability when handling low-quality prompts. Stable-SAM employs DSP to adjust the sampling positions and amplitudes of image features without altering SAM's original architecture, which refines the mask attention. Additionally, the proposed DRP allows for switching between conventional and deformable sampling modes based on prompt quality, further improving the model's performance across varying prompt qualities. Experiments validate the method's effectiveness on multiple datasets.

### Strengths
1. This work provides new insights of SAM’s stability in segmentation and introduces a segmentation stability metric that effectively quantifies the stability.
2. The proposed DSP and DRP modules provide novel insights into the improvement of SAM.
3. Stable-SAM shows marked improvement with insufficient prompts.
4. Generally, the paper is easy to follow.

### Weaknesses
1. In Table 6, the interactive segmentation results for SAM on the SBD dataset appear inconsistent with those reported in other studies, showing higher performance. For example, in the works [1-4], the NoC90 metric for SAM-B/L/H exceeds 7.6, while Table 6 reports an SBD NoC90 of only 5.76. This discrepancy raises concerns given that this work builds upon SAM.

2. The core focus of this work is to address the issues arising from ambiguous prompts, yet SAM's practical application relies more on sufficiently detailed prompts, where SAM's inherent robustness may suffice. Moreover, Table 2 shows that increasing the number of points significantly reduces the advantage of Stable-SAM. Therefore, further experiments with more points are necessary to validate the effectiveness of Stable-SAM.

3. From another perspective, under precise prompts, the DSP may introduce shifts, as there remains a difference between DSP and human intent. If the intent is to segment small parts of the object, DSP may still guide the model to segment the main parts.

4. The paper lacks an analysis of computational efficiency. The DRP seems to run the SAM decoder twice (once with the new module proposed in the paper), leading to double the computational load of the SAM decoder. Thus, the efficiency analysis is necessary to confirm that the core performance benefits come from the new modules and not just increased computational demands. 

Reference
1. ClickAttention: Click Region Similarity Guided Interactive Segmentation
2. FocSAM: Delving Deeply into Focused Objects in Segmenting Anything
3. MST: Adaptive Multi-Scale Tokens Guided Interactive Segmentation
4. PiClick: Picking the desired mask from multiple candidates in click-based interactive segmentation

### Questions
1. Please clarify the results for SAM in the SBD tests presented in Table 6.
2. How does Stable-SAM perform compared to other models with sufficient points?
3. Could you provide an analysis of the computational efficiency of Stable-SAM?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Stable-SAM, a variant of the Segment Anything Model (SAM), aimed at addressing SAM's instability when given low-quality prompts—either biased towards the background or confined to specific object parts. The authors propose two key modules integrated into SAM's mask decoder transformer: a Deformable Sampling Plugin (DSP) and a Deformable Routing Plugin (DRP). DSP adjusts the feature sampling offsets and amplitude, while DRP resamples deformable image features at updated sampling locations, refining SAM's token-to-image attention. Additionally, a dynamic routing mechanism is introduced to selectively activate DSP, preventing unwanted attention shifts. A robust training strategy is also proposed to help the model correct SAM's attention when negatively impacted by poor-quality prompts. Low-quality prompts are generated on fine-grained segmentation datasets, including DIS, ThinObject-5K, COIFT and HR-SOD, and the approach demonstrates clear performance improvements, outperforming SAM, SAM fine-tuned with LoRA, SAM fine-tuned with an adapter, and HQ-SAM.

### Strengths
1. The writing is clear, making the paper easy to follow.
2. The motivation to develop stable prompting for SAM is well-articulated and addresses a less-rexplored but valuable area. The empirical studies in Section 3 are solid and clearly presented.
3. The proposed DSP and DRP modules are novel, effectively enhancing SAM's stability without compromising its powerful pre-trained representations.The inclusion of a dynamic routing mechanism is practical and well-justified.
4. Stable-SAM demonstrates clear performance improvements over SAM and its listed variants.

### Weaknesses
1. The range of baseline models is limited. While many SAM-based models have recently been proposed to enhance its performance across various domains, this paper does not include comparisons with these relevant models.
2. The implementation of LoRA and adapters in SAM is not detailed. Since implementation choices can significantly affect performance, providing this information is crucial for evaluating the performance comparison as well as transparency and reproducibility.
3. The paper does not review SAM 2, which has been released and open-sourced. It would be nice to assess whether Stable-SAM is compatible with SAM 2 and if similar performance improvements can be achieved.

### Questions
1. While the authors have specified the number of trainable parameters, could they also provide the inference time of Stable-SAM compared to other baselines? I believe this is another important dimension to examine the efficiency.

2. (Minor) There is a typo: a period is missing at the end of Line 259.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper analyzes the stability issue of segment anything models (SAMs). It is observed that SAMs tend to focus on background or some undesired parts when given low-quality prompts. A metric of segmentation stability is introduced to better evaluate this capability. Two main modules are proposed to improve the stability of SAMs, i.e. deformable sampling to organize the image feature and dynamic routing to allocate the magnitude of deformed activation.

### Strengths
1. The topic this paper focuses on is interesting and important, that may be ignored in previous research. Though SAMs produce surprising segmentation results, they may fail when the input prompts are not that accurate, especially in real-life applications. 
2. The proposed method is simple but effective. The promotion over previous SAM and high-quality SAM methods is notable on several benchmarks and tasks.
3. The writing is easy to follow.

### Weaknesses
1. One potential negative effect to discuss. Though this paper targets strengthening the ability of SAM to focus on the foreground object, will it tend to produce wrong segmentation results when the desired segmenting part is the background? It will be interesting to discuss the bias this method may introduce.
2. It is easy to demonstrate the improvement when the number of prompts reduces to 1 or 3 points. However, noisy boxes are usually determined by the user, which are more subjective. Given a user has priors to roughly locate the desired object, the noise may not be that big. Therefore, a user study is recommended to better demonstrate how much benefit the method can actually bring when applied to real scenarios. The users can be required to give prompts to the same set of testing data, which can be more realistic than generated noisy prompts. 
3. Though the proposed modules are light and parameters-friendly, it is also necessary to evaluate their real latency and compare them with other methods.

### Questions
One open question: the module is efficient in improving the stability, considering both the training and inference budget. However, will scaling up the training data with more noisy data also improve the segmentation stability? How to treat the two factors?

### Soundness
3

### Presentation
3

### Contribution
3
