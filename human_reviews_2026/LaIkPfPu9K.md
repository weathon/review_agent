# Efficient Audio-Visual Speech Separation with Discrete Lip Semantics and Multi-Scale Global-Local Attention

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Audio-visual speech separation (AVSS) methods leverage visual cues to extract target speech and have demonstrated strong separation quality in noisy acoustic environments. However, these methods usually involve a large number of parameters and require high computational cost, which is unacceptable in many applications where speech separation serves as only a preprocessing step for further speech processing. To address this issue, we propose an efficient AVSS method, named **Dolphin**. For visual feature extraction, we develop **DP‑LipCoder**, a dual‑path lightweight video encoder that transforms lip‑motion into discrete audio‑aligned semantic tokens. For audio separation, we construct a lightweight encoder–decoder separator, in which each layer incorporates a global–local attention (GLA) block to efficiently capture multi-scale dependencies. Experiments on three benchmark datasets showed that Dolphin not only surpassed the current state-of-the-art (SOTA) model in separation quality but also achieved remarkable improvements in efficiency: over 50\% fewer parameters, more than 2.4$\times$ reduction in MACs, and over 6$\times$ faster GPU inference speed. These results indicate that Dolphin offers a practical and deployable solution for high-performance AVSS in real-world scenarios. Our code and demo page are publicly available at https://cslikai.cn/Dolphin.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Dolphin, a compact and efficient audio-visual speech separation framework that integrates visual motion cues with attention-based processing to isolate target voices. It employs DP-LipCoder, a lightweight dual-path video encoder that learns discrete visual representations aligned with speech, and a single-iteration separator with global–local attention for efficient feature modeling. Through its optimized design, Dolphin achieves superior separation quality with over 50% fewer parameters, 2.4× lower computational cost, and 6× faster inference, making it well-suited for real-time and edge deployment.

### Strengths
For the AVSS task, we designed a highly efficient model that outperforms existing approaches in key metrics such as SI-SNRi and SDRi, while achieving a 50% reduction in parameters and 2.4× lower computational cost, effectively balancing the trade-off between performance and efficiency.

Also, throughout the paper, reasonable architectural designs such as the single-pass GLA mechanism and the DP-LipCoder, which quantizes lip movements into discrete tokens, were employed to reduce computational cost while maintaining high performance. The effectiveness of these architectural innovations is well supported by the experimental results.

### Weaknesses
There is a lack of ablation studies for individual components. In particular, an additional experiment comparing the performance without using VQ, which plays a crucial role in this work, would be valuable.

Moreover, the paper lacks sufficient analysis of the individual contributions of global and local attention within the GLA blocks, as well as comparisons with other audio-visual fusion strategies

It is also unclear whether Dolphin can maintain its performance without a pretrained teacher model such as AV-HuBERT.

* Minor weaknesses
    * Line 234: The link for Figure 3 should include the letter “F.”
    * For better experimental transparency, it would be helpful to indicate which results in Table 3 were taken from the original papers.

### Questions
* It is unclear whether the encoders in the Semantic Path and Reconstruction Path shown in Figure 2 are trained separately or if there is parameter sharing between them. A more detailed explanation would be helpful.

* For the results reported in Table 2, it is not specified whether the performance was measured after retraining the model with DP-LipCoder or simply by replacing the visual encoder without additional training. Clarification on this point is needed.

* Including a discussion on the limitations of this work would make the paper more comprehensive and could provide valuable guidance for future research.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents an audio-visual model for speech separation. The main contribution is a model which achieves SOTA results and is also lightweight. Results on multiple datasets and an ablation study are presented.

### Strengths
- The presentation of an audio-visual speech seperation model which achieves SOTA performance and is also lightweights is a useful contribution.
- Results on multple datasets and detailed ablation study.

### Weaknesses
- The title is a bit misleading. The model recovers the speech of the main speaker only, and not all speech signals at the same time. So speech enhancement might have been a better term for this task (and it's more consistent with the literature).
- The paper does not compare with some recent SOTA models, e..g. "LA-VocE: Low-SNR audio-visual speech enhancement using neural vocoders
- It's not clear what results (in terms of number of background speakers) are presented in Section 5. Given that the appendix includes results with multiple background speakers, then these results probably include 1 background speaker. Also, why not testing the model's performance on different types of background noise and combination of background noise+multiple speakers.
- No supplementary material is provided, so it's hard to judge how good the denoised samples are. The numbers look good, but without listening to examples it's hard to judge the quality. Some examples are provided on the provided link, but they are too few. There are only 6 examples where the model is compared to other models. Would also be good if some details about the type of noise in each example are added.

### Questions
Please see above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles audio-visual speech separation (AVSS). The core problem it targets is that current SOTA methods either achieve good separation quality but rely on a heavy visual backbone plus multiple iterations of the separator, or they are lightweight but require repeated iterations, which hurts real-time performance and deployability. The authors propose an overall framework called Dolphin, with two main components: a lightweight dual-path video encoder DP-LipCoder and a single-iteration encoder–decoder separator with Global–Local Attention.

### Strengths
It makes explicit a long-standing but very real issue in AVSS that the visual encoder is too heavy, and then deliberately designs a dual-path + VQ + distillation structure on the visual side. This is a more thoughtful route than simply plugging in an off-the-shelf lipreading backbone or adding a small autoencoder. It jointly considers both reconstructive and semantic aspects, rather than just trimming the backbone.

The separator also states the efficiency goal of replacing multiple iterations with a single pass quite clearly, and then compensates for it with global and local components.

### Weaknesses
The paper clearly states that compressing lip-reading backbones causes semantic loss, and that purely reconstruction-oriented lightweight encoders only capture shallow, pixel-level cues. It then proposes a dual-path design with VQ-based discrete semantics plus distillation to address this. However, the mechanism of why the dual-path + discrete tokens specifically preserve the task-relevant semantics under heavy compression is mostly justified empirically (Table 1), rather than analytically. A more explicit explanation or diagnostic would strengthen the claim.

### Questions
The proposed GLA-based separator is presented as a single-pass design for efficiency. However, since the block itself is a generic feature transform, it seems technically feasible to unroll it for multiple passes (as commonly done in audio-visual separation) to trade extra compute for further gains. I would like to see how the model performs under multiple iterations.

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
3

### Summary
The paper presents a more computationally efficient AV speech separation system called Dolphin. As part of this system, the author's also present a lightweight visual encoder that maps video frames to semantic, well-aligned-to-audio discrete tokens. Together they yield state-of-the-art quality across a variety of corpora and metrics while being fairly computationally efficient.

### Strengths
The paper demonstrates an approach for building a state-of-the-art speech separation system. They modify TDANet, moving from iterative, progressive separation to single separation step for efficiency with joint local and global attention for each layer for a large win on efficiency. They also note the need for improving throughput in the vision encoder, which encodes lip motion frames into discrete semantic tokens, by developing a fast, high quality model called DP-LipCoder. Third, for separation they use an efficient, mixed global/local attention encoder/decoder. These innovations yield a system that surpasses many recent AV speech separation systems.

Further, the system is efficient showing less memory use, fewer MACs, faster GPU inference than many competing systems. 

The paper will provide a github repo that will allow others to fully reproduce results as much as possible.

### Weaknesses
In some ways, this paper is an incremental approach to AVSS, not providing many new insights, however strong the results are. Critics may wonder if human evaluations/side-by-sides would agree that the Dolphin system is indeed superior to the other approaches in terms of quality improvement and fewer artifacts. Lastly, the tasks all appear to be artificially created speaker overlap and equation (1) is a large simplication--with the Lombard effect the noise causes non-linear changes to speech production.

### Questions
How well does the Dolphin system do with real overlapping speakers?
Have conducted any human listening tests / comparative evals against other systems?

### Soundness
3

### Presentation
4

### Contribution
3
