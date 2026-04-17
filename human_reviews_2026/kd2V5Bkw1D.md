# Improving Autoregressive Video Modeling with History Understanding

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Video autoregressive generation (VideoAR) sequentially predicts future frames conditioned on history frames. Despite the advance of recent diffusion-based VideoAR, the role of conditioning signal—internal representations of history frames—remains underexplored. Inspired by the success of strong condition representations in text-conditioned generation, we investigate: \textit{Can better internal representations of history frames improve VideoAR performance?} Through systematic analysis, we show that history representation quality positively correlates with VideoAR, and that enhancing these representations provides gains that cannot be achieved by refining future frames representations alone. Based on these insights, we propose \textbf{MiMo} (Masked History Modeling), a novel framework that seamlessly integrates representation learning into diffusion-based VideoAR. MiMo applies masks to history frame tokens and trains the model to predict masked tokens of current and future frames alongside the diffusion objective, yielding predictive and robust history representations without relying on vision foundation models (VFMs) or heavy architectural changes. Extensive experiments demonstrate that MiMo achieves competitive performance in video prediction and generation tasks while substantially improving training efficiency. Our work underscores the importance of history representations in VideoAR.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes MiMo, an autoregressive video generative model unifying both self-supervised learning in history frames and diffusion learning in future frames. MiMo is proposed under the sufficient findings that history representation (latent of history frames in DiT blocks) positively correlates with the performance of VideoAR, and is designed to learn history frame representation and generation capability simultaneously. MiMo conducts some experiments to prove the performance and training efficiency of MiMo. Despite these contributions, there remain some problems to be explained: 1.  In this work, the only generation quality evaluation metric used is the FVD (Fréchet Video Distance), and the only generation task is class-to-video, which is not enough to prove the advantage of MiMo. Moreover, FVD is known as unstable. 2. As shown in Table 3, MiMo seems to be inferior to REPA, another representation learning method, yet MiMo claims SOTA in VideoAR. Additionally, comprehensive comparisons with other works that also unify understanding and generation are needed.

In summary, while MiMo insightfully emphasizes the importance of history frame representation learning, its purported advantages require more comprehensive experiments and broader benchmarking to be convincingly established.

### Strengths
This work proposes MiMo, an autoregressive video generative model unifying both self-supervised learning in history frames and diffusion learning in future frames. MiMo is proposed under the sufficient findings that history representation (latent of history frames in DiT blocks) positively correlates with the performance of VideoAR, and is designed to learn history frame representation and generation capability simultaneously. MiMo conducts some experiments to prove the performance and training efficiency of MiMo.  And this paper is well written and easy to understand.

### Weaknesses
1. The evaluation relies solely on FVD (Fréchet Video Distance) as the generation quality metric and focuses exclusively on the class-to-video generation task, which is insufficient to substantiate MiMo’s claimed advantages. Moreover, FVD is known to be unstable and may not reliably reflect perceptual quality.
2. As shown in Table 3, MiMo underperforms compared to REPA, another representation learning approach,  yet it claims state-of-the-art (SOTA) performance in VideoAR. Additionally, comprehensive comparisons with other works that also unify understanding and generation are needed.
3. More ablation and discussion are required: (1) Ablation study: with vs. without history representation learning. (2) Discussion and ablation about $\lambda$ in Equ. (6). (3) Discussion on additional computational burden.

### Questions
1. What is the additional computational overhead introduced by the two-objective optimization in Equation 6?
2. Have you considered using TTUR (Two-Time-Scale Update Rule) to mitigate this extra computational burden?
3. Why does the evaluation rely solely on FVD and the class-to-video generation task?

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
This paper proposes the **MiMo framework**, which enhances video autoregressive (AR) generation by improving the quality of historical frame representations through **masked history modeling**. Without relying on additional vision foundation models or complex architectures, MiMo achieves significant improvements in both video prediction and generation tasks.

### Strengths
- The paper innovatively applies a **mask modeling strategy** similar to MAE within an autoregressive video generation framework.  
- The proposed approach yields a **consistent and stable improvement** in AR video generation quality compared to existing methods.

### Weaknesses
- The experimental results show that the final performance of MiMo is **comparable to REPA**. It would be important to verify whether MiMo can further improve upon REPA, rather than leaving this question for future work.  
- The experiments suggest that **next-frame prediction** contributes more to performance improvement, which seems **inconsistent with the paper’s main claim** that enhancing historical frame representations is the key factor driving the gains.

### Questions
See weakness above

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates the role of history frame representations in autoregressive video generation (VideoAR). The authors first empirically demonstrate a positive correlation between the quality of history representations and the model's prediction performance. Based on this insight, they propose MiMo (Masked History Modeling), a novel framework that integrates a self-supervised masked modeling objective into a diffusion-based VideoAR model. During training, MiMo applies masks to clean history frame tokens and tasks the model with reconstructing the masked content of current and future frames, in parallel with the primary diffusion denoising objective. This dual-objective approach is designed to learn robust and predictive history representations without relying on external Vision Foundation Models (VFMs) or significant architectural changes. Experiments on video prediction and generation tasks show that MiMo achieves competitive or state-of-the-art performance while improving training efficiency.

### Strengths
The paper is well-motivated and logically structured. A key strength is the preliminary analysis that systematically establishes the link between history representation quality and VideoAR performance, providing a solid foundation for the proposed method. The MiMo framework itself is an effective solution that integrates representation learning with the diffusion process. The method's ability to achieve strong results on multiple benchmarks, such as Kinetics-600 and UCF-101, without the need for pre-trained VFMs is a significant advantage, making it more practical and self-contained. The comprehensive ablation studies further validate the authors' design choices, particularly regarding the masked prediction targets.

### Weaknesses
While the results are strong, the evaluation could be perceived as having some limitations. The experiments are primarily conducted on the Kinetics-600 and UCF-101 datasets, with Fréchet Video Distance (FVD) as the main metric. Although these are standard benchmarks, they may not fully capture the complexity and diversity of real-world video dynamics. Expanding the evaluation to include longer-form videos, more complex scenes, or complementary metrics such as user studies could provide a more comprehensive assessment of the model's capabilities in maintaining long-term coherence and semantic consistency.

### Questions
The ablation study in Table 4 compellingly shows that predicting both the current + next frames as the masked modeling target yields better results than predicting the current frame alone. This leads to an interesting question: have the authors explored if an even more complex prediction target could further enhance performance? For instance, would a target of current + next + next_after_next continue this positive trend, or does it lead to diminishing returns or potential training instability?

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
This paper studies the role of history frame representations in diffusion-based autoregressive video generation (VideoAR). The authors propose MiMo (Masked History Modeling), which integrates masked modeling into VideoAR to explicitly improve history frame representations during training. By applying masks to history tokens and reconstructing both current and next-frame tokens alongside the diffusion objective, the model learns more predictive and robust history representations without relying on pretrained vision foundation models. Experiments on Kinetics-600 and UCF-101 show improved FVD scores over prior autoregressive methods.

### Strengths
1. The paper is well-written, with clear figures and reasonable organization.

2. The paper identifies an underexplored question: how internal history representations affect VideoAR quality, and provides empirical analysis supporting its importance.

3. MiMo introduces a relatively lightweight modification to standard diffusion-based AR training, avoiding reliance on large pretrained models.

4. Results are reported on multiple datasets and compared with several baselines, including both AR and non-AR methods.

### Weaknesses
1. The proposed masked history modeling is conceptually straightforward and closely resembles masked autoencoding and REPA-style representation regularization. The integration into VideoAR feels incremental rather than fundamentally new.

2. The ablations are narrow in scope. It remains unclear why the proposed loss improves results beyond general regularization or increased training signal. Deeper analysis of representation quality (e.g., visualization or qualitative dynamics) would strengthen the claim.

3. While the reported FVD gains are notable numerically, improvements are modest considering the added training complexity. Moreover, comparisons to strong non-AR diffusion models are missing.

4. Several design choices are only briefly described. Implementation details may be insufficient for faithful replication.

### Questions
1. How sensitive is MiMo to the masking ratio and λ parameter? Have the authors evaluated performance stability under different settings?

2. Could the authors compare MiMo with a simpler auxiliary loss such as contrastive or reconstruction-only objectives to rule out general regularization effects?

3. How much additional computation does masked history modeling introduce during training?

4. Have you tested whether pretraining MiMo representations improves downstream video understanding tasks to support the “representation” claim?

5. Could qualitative examples (e.g., failure cases or visualization of history embeddings) clarify what MiMo learns beyond the diffusion baseline?

6. Would MiMo still help when combined with pretrained VFMs or under longer video horizons?

### Soundness
2

### Presentation
3

### Contribution
3
