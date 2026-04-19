# Bootstrapping Audio-Visual Segmentation by Strengthening Audio Cues

- Decision: Reject
- Scores: 3, 6, 6, 6

## Abstract
How to effectively interact audio with vision has always attracted extensive interest in the multi-modality community. Recently, a novel audio-visual segmentation (AVS) task has been proposed, aiming to segment the sounding objects in video frames using audio cues. However, current AVS methods suffer a modality imbalance issue. The fusion of audio features is insufficient because of its unidirectional and deficient interaction, while the vision information is more sufficiently exploited. Thus, the output features are always dominated by visual representations, which restricts audio-vision representation learning and may cause some false alarms. To address this issue, we propose AVSAC, where a Bidirectional Audio-Visual Decoder (BAVD) is devised with multiple bidirectional bridges built within. This strengthens audio cues and enables continuous interaction between audio and visual representations, which shrinks modality imbalance and boosts audio-visual representation learning. Furthermore, we introduce Audio Feature Reconstruction (AFR) to evade harmful data bias and curtail audio information loss by reconstructing lost ones from visual signals. Extensive experiments show that our method achieves new state-of-the-art performances in the AVS benchmark, especially boasting significant improvements (about 6$\%$ in mIoU and 4$\%$ in F-score) in the most challenging MS3 subset which needs to segment multiple sound sources.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper mainly studied the salient audio-visual segmentation problem (AVS) which tries to segment the sounding objects given the audio query. The paper argues that the current AVS methods generally suffer a modality imbalance issue and output features are always dominated by the visual representation hence hindering the model performance. To address such issues, the paper proposed a bidirectional audio-visual decoder (AVSAC) that builds mutual cross-attention layers between audio and visual streams. In addition, the audio feature reconstruction (AFR) evades harmful data bias and aims to preserve useful audio information. Extensive experiments show that the proposed methods achieve better results than the previous methods.

### Strengths
The paper is generally well-written and easy to follow.
The author provides extensive experiments to demonstrate the effectiveness of the model including both qualitative results and quantitative results.

### Weaknesses
My main concern regarding the paper's technical contribution is that the proposed bidirectional framework is not entirely novel and has been previously explored in multiple instances (e.g., [a, b, c]). These prior methods share a similar concept with the proposed approach, and the use of reconstruction to preserve the semantic meaning of the audio can also be viewed as a preliminary version of [d].

In the introduction, the author pointed out the issue of dataset bias in the current AVS dataset, which can potentially allow the network to make accurate predictions even when audio information is absent. This perspective aligns with the findings presented in [e], where a similar context of dataset bias was discussed. It would be great for the author to properly cite this prior work and discuss it if necessary.

The author did not include the SOTA method AQFormer [f] in the experiment section, which shows better results compared to the proposed method in Table 1. For example, AQFormer (ResNet50) achieved a 55.7 mIoU score under the MS3 split, while the proposed method achieved only 51.13. While it's understandable that the proposed method may not have surpassed the previous approach, but some discussion on this discrepancy would be appreciated. Similarly, when comparing the previous method on AVSS (Table 6). I suppose the AVSegformer's performance should be also listed in that table.



**Reference** 

[a] Tao, R., Pan, Z., Das, R.K., Qian, X., Shou, M.Z. and Li, H., 2021, October. Is someone speaking? exploring long-term temporal features for audio-visual active speaker detection. In Proceedings of the 29th ACM International Conference on Multimedia (pp. 3927-3935).

[b] Duan, B., Tang, H., Wang, W., Zong, Z., Yang, G. and Yan, Y., 2021. Audio-visual event localization via recursive fusion by joint co-attention. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 4013-4022).

[c] Liu, S., Quan, W., Liu, Y. and Yan, D.M., 2022, May. Bi-directional modality fusion network for audio-visual event localization. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 4868-4872). IEEE.

[d] Georgescu, M.I., Fonseca, E., Ionescu, R.T., Lucic, M., Schmid, C. and Arnab, A., 2023. Audiovisual masked autoencoders. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 16144-16154).

[e] Chen, Y., Liu, Y., Wang, H., Liu, F., Wang, C. and Carneiro, G., 2023. A Closer Look at Audio-Visual Semantic Segmentation. arXiv e-prints, pp.arXiv-2304.

[f] Huang, S., Li, H., Wang, Y., Zhu, H., Dai, J., Han, J., Rong, W. and Liu, S., 2023. Discovering Sounding Objects by Audio Queries for Audio Visual Segmentation. arXiv preprint arXiv:2309.09501.

### Questions
Please refer to the comments in the weakness section.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a new method for audio-visual segmentation, via a Bidirectional Audio-Visual Decoder (BAVD) and Audio Feature Reconstruction (AFR). It addresses the problem of modality imbalance in audio-visual segmentation, claiming itself achieving new state-of-the-art results, including substantial improvements in challenging scenarios.

### Strengths
1. The paper's originality is properly presented and mentioned.
2. The paper has good level of clarity. The connection of its methods and performance is clearly presented. The backbone acquired is state-of-the-art level, although some of the baseline models are a bit old.

### Weaknesses
1. The use of English is a bit problematic. Please go through language checking (via tools like ChatGPT for example, if accessible) to fix some issues.
2. Since the author mention the data imbalance, the beginning of this paper somehow gives the impression to the reader that this paper is targeting this problem. However, after reading the paper, it is still a bit hard for the reviewer to find the problem being directly targeted. The method does lead to good level of improvement, but the imbalance problem gradually become secondary. Strengthening audio cues does give better performance, but that does not mean it tackles the imbalanced problems from my perspective. Also, whether a more balanced flow between video and audio will lead to better performance shall be discussed at the beginning, as part of motivation for the study.

### Questions
1. In the background, the imbalance issue between audio and video is mentioned, which is good. But I do not understand the aftermath of such imbalance - why the model's being focused more on visual information is a bad thing? This needs to be motivated a bit more.
2. Do you see any possibility that the architecture can be optimized to improve the efficiency? For example, some of the cross-modal connections might be redundant? Of course, here "nope" is also a proper answer.
3. At the beginning of Section 3.2 - what is the dot product in MLP() means in Equation 1?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This article proposes a Bidirectional Interaction mechanism for the AVS task, which enhances the interaction between audio and visual. In terms of details, based on cross-attention, two modules, AGV and VGA, are used. An Audio Feature Reconstruction mechanism was also designed to address the problem of no supervision in the audio branch. With the support of these modules, this paper achieves state-of-the-art performance under AVS tasks. Ablation experiments demonstrate the effectiveness of the proposed module.

### Strengths
1. The paper is well written, the motivation is reasonable, and the reasons and practices for the design of each module are easy to understand.
2. The results in Table 5 surprise me. Increasing the number of decoder layers can actually lead to such a significant performance improvement.

### Weaknesses
1. Audio Feature Reconstruction seems to be very similar to the bidirectional generation module in AVSBG [1].
2. I can't see from Figure 5 which areas the attention mechanism obviously pays attention to.

[1] Dawei Hao, Yuxin Mao, Bowen He, Xiaodong Han, Yuchao Dai, and Yiran Zhong. Improving audio-visual segmentation with bidirectional generation. arXiv preprint arXiv:2308.08288, 2023.

### Questions
N/A

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper describes an approach for audio-visual segmentation – where the goal is to segment sounding objects in a given video. The main focus of the paper is on improving audio-visual fusion by emphasizing on the audio modality. This is done through a bi-directional audio-visual decoder. An audio feature reconstruction is also used to further emphasize the audio. Experiments are done on the AVS Benchmark and results show that the proposed method obtains improvements of up to 1-4% in F-score and 2-5% in mIoU over prior work.

### Strengths
–  The direction of the paper is good. Most audio-visual work often end up focusing too much on the visual modality even if the task is acoustic in nature. 

– The approach makes sense to my understanding and appears to be a simple but effective extension to improve audio-visual segmentation.  

– Ablation studies have good coverage of different aspects of the method.

### Weaknesses
– While the emphasis on improving uses of audio cues is good, I am not sure if the claims around cross modal attention is entirely correct. There are multimodal works where both audio-visual attention is through both audio-video and video to audio. That is attentions of the forms Attn(Q_a, K_v, V_v) and Attn(Q_v, K_a, F_a) – so that both audio and visual features are obtained through cross attention. [1, 2] are just 2 examples, likely there are other papers. Not sure however if they have been used for AV segmentation task. 

– Not clear about the claims of richness of information in audio – experimentally this is illustrated by showing that removing self-attention leads to better results. Can you discuss this in a bit more detail and clarify how this conclusion is reached?   It’s not clear that improved performance by removing self-attention can actually lead to this claim.  

– I think defining L_dice, L_mask and L_afr clearly will add clarity to the the paper. 

– How does the overall method behave when the sounding object is outside of the field of view or the sounding objects moves in and out of field of view. 

– The AFR loss is essentially forcing feature similarity between F_AGV and F_audio (through the feature learning). Why would this reduce the type of bias mentioned in section 3.3 (single and mult-source etc.)

### Questions
Please respond to the questions in the weakness section.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
