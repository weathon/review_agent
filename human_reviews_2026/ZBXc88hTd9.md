# From Coarse to Fine: Recursive Audio-Visual Semantic Enhancement for Speech Separation

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
Audio-visual speech separation aims to isolate each speaker’s clean voice from mixtures by leveraging visual cues such as lip movements and facial features. While visual information provides complementary semantic guidance, existing methods often underexploit its potential by relying on static visual representations. In this paper, we propose CSFNet, a Coarse-to-Separate-Fine Network that introduces a recursive semantic enhancement paradigm for more effective separation. CSFNet operates in two stages: (1) Coarse Separation, where a first-pass estimation reconstructs a coarse audio waveform from the mixture and visual input; and (2) Fine Separation, where the coarse audio is fed back into an audio-visual speech recognition (AVSR) model together with the visual stream. This recursive process produces more discriminative semantic representations, which are then used to extract refined audio. To further exploit these semantics, we design a speaker-aware perceptual fusion block to encode speaker identity across modalities, and a multi-range spectro-temporal separation network to capture both local and global time-frequency patterns. Extensive experiments on three benchmark datasets and two noisy datasets show that CSFNet achieves state-of-the-art (SOTA) performance, with substantial coarse-to-fine improvements, validating the necessity and effectiveness of our recursive semantic enhancement framework.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes CSFNet, a coarse-to-fine framework for audio-visual speech separation that recursively enhances semantic representations. The method first performs coarse separation to obtain preliminary speech estimates, which are then fed into a pretrained Auto-AVSR model along with lip features to derive refined audio-visual semantics for fine separation. Additionally, the authors design a Speaker-wise Perceptual (SP) fusion block for robust cross-modal fusion and a Multi-range Spectro-Temporal (MST) separation module to capture both local and global time-frequency dependencies. Extensive experiments on multiple datasets (LRS2, LRS3, VoxCeleb2, NTCD-TIMIT, LRS3+WHAM!) demonstrate consistent state-of-the-art performance and robustness in noisy or occluded-visual conditions.

### Strengths
1.	Novel recursive semantic enhancement strategy. The idea of feeding back coarse audio into an AVSR model for semantic refinement is conceptually appealing and empirically effective. It provides a meaningful link between speech recognition and separation.
2.	Strong performance and comprehensive experiments. The proposed method achieves clear SOTA results across multiple benchmarks, including both clean and noisy datasets, with consistent gains in SI-SDRi, PESQ, and WER metrics. Notably, the second stage of the proposed two-stage framework yields substantial performance gains in more challenging multi-speaker scenarios, particularly under 3-mixture and 4-mixture conditions.
3.	Well-structured ablations. The authors thoroughly analyze the contribution of each component (two-stage design, SP fusion, MST module, visual occlusion tests), which strengthens the credibility of the method.
4.	Clear presentation and solid engineering. The paper is well written, figures are clear, and the methodology is reproducible. The design choices are motivated and technically sound.

### Weaknesses
1.	Clarification on visual frame occlusion simulation (Figure 4 & Appendix F). Figure 4 presents results for cases where one or both speakers have partially missing visual frames. As described in Appendix F, the authors simulate missing 5, 10, 20, 30, 40, and all frames. However, for the case where both speakers lose visual cues, it is not clearly stated whether the missing frame segments are temporally aligned or occur at different timestamps. In practice, manually discarding frames may cause temporal misalignment between audio and video streams, which could significantly influence model performance. The lack of detailed description on how this issue is handled makes it difficult for readers to fully assess the validity of the visual occlusion experiments.
2.	The paper introduces a simple yet efficient fusion module, the Speaker-wise Perceptual (SP) fusion block. However, the authors do not explain why chose not to adopt common attention-based fusion mechanisms that are widely used in previous audio-visual separation works. 
3.	Inference efficiency and online applicability. Since the proposed system is composed of two sequential stages (coarse and fine separation), it appears that inference also requires two passes through the network: the first to generate coarse audio and the second to refine it. This could potentially double the inference time, making the method less suitable for real-time or low-latency applications. 
4.	Table 5 suggests that both stages rely heavily on the pretrained Auto-AVSR model. This raises a concern that the performance improvements and SOTA results might largely stem from the strength of the pretrained AVSR backbone, rather than the proposed recursive separation architecture itself. A more thorough discussion on this trade-off would strengthen the paper.

### Questions
Consistent with Weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CSFNet (Coarse-to-Separate-Fine Network), a two-stage audio-visual speech separation system that introduces a recursive semantic enhancement paradigm. The approach operates in two stages: (1) a coarse separation stage that produces an initial audio estimate from mixed audio and visual input, and (2) a fine separation stage that feeds the coarse audio back into an audio-visual speech recognition (AVSR) model together with visual information to extract enhanced semantic representations for refined separation. The model incorporates a speaker-aware perceptual (SP) fusion block and a multi-range spectro-temporal (MST) separation module. Extensive experiments on three benchmark datasets (LRS2, LRS3, VoxCeleb2) and two noisy datasets demonstrate state-of-the-art performance with substantial coarse-to-fine improvements.

Overall, the other experimentation in the paper and the Appendix is solid and detailed, a very interesting read! The model has very strong performance, and outperforms competitors by a wide margin in a wide variety of tasks. However, the model is similar to previous methods in many ways, and so the contributions in terms of model architecture are too minimal to meet the standards of ICLR. That being said, given the strong performance and detailed results, I would be willing to accept the paper if the issues detailed in this review are addressed with further experimentation and justification for the choices made.

### Strengths
The model demonstrates very strong performance, outperforming competitors by a wide margin across a wide variety of tasks and datasets, including both clean and noisy conditions. The extensive experimentation presented in both the main paper and appendix is solid and detailed, providing a very interesting and thorough evaluation. The recursive coarse-to-fine framework presents an interesting approach to leveraging audio-visual semantic information more effectively than prior work that relies solely on visual input for semantic extraction.

### Weaknesses
Speaker-wise Perceptual (SP) Fusion Block is fairly similar to the fusion approach in CTCNet, with a simple adaptation to the TF domain. The exact same adaption to the TF domain was also used as a baseline in the RTFS-Net paper, which similarly interpolates in the time dim, and then broadcasts across the frequency dim (see RTFS-Net section 5.2). This mixture of approaches is fine, but is very similar to the work covered by these two papers and hence should be cited. In addition, when evaluating performance, SP fusion should compare to the previous baselines, such as the fusion methods in CTCNet, RTFS-Net, AVLiT or the current SOTA methods AVSepChain and IIANet. In addition, Figure 5 is quite difficult to read. The colours are very similar. A black boarder around each bar would aid visibility. A table would be even better. Also, RTFS-Net's results are also missing from table 1. 

The order of the results in the tables is a little confusing. It's not by year or performance. To help the reader, please put the models in the table in order of performance with the best result highlighted in bold. For example, in table 2 you highlight your method, but the best method for macs, params, time and memory are not your method. This is fine, but please clear this up to make things clear. 

The main focus of the paper is a coarse to fine refinement process - running the audio through the model twice. However, it is not clear from the results where the strong performance of the paper is coming from. The current SOTA AVSS method is IIANet, which uses a different pre-trained video encoder. It's hard to see if CSFNet's performance comes from the new video encoder i.e. pre-trained Auto-AVSR (Ma et al., 2023), or the actual model architecture. For example, if you swapped CSFNet's video encoder with IIANet's video encoder, or conversely, replace IIANet's video encoder with pre-trained Auto-AVSR (Ma et al., 2023) and ran IIANet twice in a course-fine manner, which approach would have the better performance? 

In table 2, it is unclear if these results are for the course+fine CSFNet or the course only approach. Therefore, it could be interpreted that the coarse-only method only barely outperforms IIANet, is slower, and takes up significantly more memory. Please clarify! It would also be interesting to see the course only model macs and parameters.

### Questions
The encoder is interesting, using multi-scale dilations to gather larger and larger context. Most other methods in AVSS use a simple 1D or 2D convolution depending on if the model is working with the time domain or the TF domain, would it be possible to see some experimentation on the benefits of this encoder over the encoder used in other methods? There needs to be some justification for this added complexity + parameters.

The main (MST) separation module is similar to TF-GridNet, which was also used by RTFS-Net for AVSS. As with RTFS-Net, you make some adaptations, including a triple unfold of the features and some kind of fusion. However, the paper does not detail how this fusion works - summation/concatenation/attn/projection? You only state "The outputs of these branches are fused to produce richer multi-scale representations...". Papers need to be repeatable, and this mechanism is unclear. The ablations studies in Appendix C were well done and detailed though. Very interesting results + good efficiency gains with the tweaks to BLSTM.

In table 1, there is an audio only model, but it is unclear how this audio only model works. Is it just CSFNet without the SP fusion block and no video inputs? 

The course to fine approach is interesting, but can it be extended? What if we run the audio through the model 3 times, do we continue to get a higher quality separation? Or do we reach a saturation point?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes the Coarse-to-Separate-Fine Network (CSFNet) to improve audio-visual speech separation (AVSS) performance. Unlike existing AVSS models that rely on fixed visual information, CSFNet recursively enhances audio-visual representations through a two-stage structure consisting of coarse and fine separation. Specifically, in the first stage, a coarse speech separation is performed, and its output is reused as input to an audio-visual speech recognition (AVSR) model to generate more discriminative semantic representations, which are then leveraged to perform the fine separation. In addition, we introduce the Speaker-wise Perceptual (SP) Fusion Block and the Multi-range Spectro-Temporal (MST) Separation Module to accurately capture speaker-specific cues and temporal information. Experimental results demonstrate that CSFNet achieves higher separation performance than existing SOTA methods while exhibiting strong robustness under noisy conditions.

### Strengths
* Through recursive semantic enhancement, the coarse separation output is fed back into the AVSR model to learn stronger audio-visual semantic representations—a concept that is novel and empirically leads to performance improvements.

* Experimental results show that performance improves noticeably from the coarse to the fine stage, and it is particularly noteworthy that the model achieves strong performance in multi-speaker scenarios and when visual cues are missing.

### Weaknesses
* Tables 1 and 2 show that the performance improvement does not scale proportionally with the increase in parameters. In particular, compared to IIANet, CSFNet has roughly three times more parameters, but the performance gain is quite marginal. While the authors claim a substantial improvement in SI-SDR over IIANet, considering that CSFNet goes through both coarse and fine separation stages, the performance improvement seems insufficient.
* There is a lack of analysis on the specific roles of the coarse and fine stages. In extreme cases, it is difficult to distinguish the effect from simply training for twice as many steps. A more detailed analysis, for example using attention maps to show the distinct contributions of the coarse and fine stages, would strengthen the study.
* Compared to prior work, the approach gives the impression of simply adding an extra stage, which makes the novelty aspect somewhat limited.

### Questions
* Would it be possible to provide a comparison with IIANet and AVSepChain under noisy conditions in Table 3 as well?
* Although the authors refer to the method as coarse and fine separation, it seems that it could in fact be applied iteratively for multiple repetitions. Could you provide experimental results showing how performance improves when each stage is repeated multiple times?
* Since the dual-stage structure trains and infers both the coarse and fine stages, it appears that the inference time and parameter count would roughly double. Could you clarify whether the parameter values reported in Tables 1 and 2 account for this?

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
The paper presents a new method for audio-visual speech separation. The main idea is to separate the the mixed audio twice. In the first stage, it produces coarse results. In the second stage, it separates the original mixed audio again, but this time, it is aided by the results obtained in the first stage. The coarse results are integrated into the video encoder, which produces better fused audio-visual features for the model to do the fine separation in the second stage. The results show a good trade-off between the accuracy and computational cost.

### Strengths
1. The main idea -- from coarse to fine -- is simple and makes sense. 
2. The results are extensive, and show a good tradeoff between the quality of the results and the computational cost. In addition, in the conditions of more than 2 speakers (Table 4), the proposed method outperformed other methods by a large margin. 
3. Presentation is good. It's easy to understand the contents.

### Weaknesses
1. The idea is straightforward, not novel enough, considering that the principle of coarse-to-fine has been used in model design in many areas including computer vision and audio processing. In fact, the iterative nature of A-FRCNN  (Hu et al., 2021) and CTCNet (Li et al., 2024a) also makes them follow into the coarse-to-fine model category.   
2. The computational cost is high. The reason is clear: the model needs to process the audio signal and video signal twice. From Table 2, the size of the model and the memory usage are more than 3X of a state-of-the-art model IIANet. This could be partially sensed by the expensive computing facility used in experiments:  8 NVIDIA A100 GPUs (40 GB).

### Questions
In the conditions of more than 2 speakers (Table 4), the proposed method outperformed other methods by a large margin. Besides the coarse-to-fine strategy, is there any other reason? 

In the SP fusion block (Figure 3a), the features of the audio mixture are concatenated with two speakers' video features. Does it mean that we need to train a model for each n, where n is the number of speakers? Please note that in many previous audio-visual speech separation models, such as (Pegg et al. 2024), this is not necessary. Is this a reason of good results for more than 2 speakers?

### Soundness
3

### Presentation
3

### Contribution
2
