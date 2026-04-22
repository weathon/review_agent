# SyncLipMAE: Contrastive Masked Pretraining for Audio–Visual Talking-Face Representations

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 8, 8, 2

## Abstract
We introduce SyncLipMAE a self-supervised pretraining framework for talking-face video that learns synchronization-aware and transferable facial dynamics from unlabeled audio--visual streams. Our approach couples masked image/video modeling with cross-modal contrastive alignment and employs three per-frame prompt tokens that explicitly encode the essential factors of a talking-face frame—identity, vocal motion (speech-synchronized facial dynamics), and ambient motion (audio-agnostic movements such as blinks and head pose). The contrastive objective uses time-aligned vocal-motion and audio tokens as positives and misaligned pairs as negatives, driving both modalities into a shared embedding space and yielding token-level audio--visual stream synchronization. After pretraining, the aligned audio tokens together with the visual prompt tokens (identity, vocal motion, ambient motion) form a unified interface for four disparate downstream settings: (i) audio--visual stream synchronization; (ii) facial expression and head/face action recognition; (iii) visual speech recognition; and (iv) visual dubbing, for which we, for the first time, enable indistinguishable audio- or video-driven control within a single model. Across four task families that require distinct capabilities, \name achieves state-of-the-art results, underscoring the effectiveness of synchronization-aware, factorized self-supervised pretraining.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a method to learn facial representations from talking-face video.
This is achieved through two main mechanisms:
(i) aligning visual representations to pretrained speech wav2vec embeddings, and
(ii) reconstructing the masked video frames in a masked autoencoding (MAE) fashion.
A distinctive feature of the approach is the attempt to factorize the face representation into three distinct components: identity, head movement, speech content.
The specialization of these components is primarily driven thorugh data augmentations (identity prompt shuffling, photometric perturbations, face-aware masking),
which encourage each of the components to retain specific aspects of the input.
The approach is evaluated on four downstream tasks: audio-visual synchronization, visual speech recognition, emotion/action recognition, and video dubbing.
However, the same datasets are used for both pretraining and downstream evaluation, limiting the evidence for transferability and complicating a comparison with existing baselines.

### Strengths
- The pretrained model, if released, could potentially be very useful to the community.
- The idea of factorizing the representation into specialized and disentangled components is conceptually appealing (although it's unclear whether this is really achieved; see weaknesses).
- The evaluation on a diverse set of downstream tasks is helpful.

### Weaknesses
- It's unclear whether the resulting representations are fully disentangled and whether they contain only the intended information. More importantly, is the proposed factorization even necessary, given that speech representation $\mathbf{z}^{\text voc}$ is used in all four downstream tasks?
- Pretraining and downstream datasets come from the same distribution of datasets. This is very important, since it limits the evidence for transferability and could give an advantage over the baselines, which is unclear whether they are retrained or finetuned on the same datasets.
- Many implementation details are omitted. How are the loss weights chosen? What are the exact baselines that the authors compare against (e.g., there are many AV-HuBERT checkpoints)?  What is the neighborhood size (self-similarity threshold) for the contrastive loss? Is the model trained on the full video or on segments?

### Questions
- See weaknesses; I would like the authors to address especially the second point, on the data overlap.
- Do the authors plan to release the pretrained model and the code upon acceptance?
- In section 4.2 L.290 the authors say that the linear probing the averaged $\mathbf{z}^\text{voc}$ and $\mathbf{z}^\text{amb}$ representations "cleanly separates speech-synchronized orofacial motion from audio-agnostic dynamics". Could the authors clarify how an average representation measures the disentanglement of the components? Shouldn't those representations have been probed separately to measure this?
- If I understand correctly, the model employs only static (per-frame) information; still it excels at audio-visual synchronization. I find this surprising: doesn't the model need to encode temporal information as well? For example, many of the synchronization models use 3d convolutions ([Synchformer](https://arxiv.org/pdf/2401.16423), [VocaLIST](https://arxiv.org/pdf/2204.02090), [SparseSync](https://arxiv.org/pdf/2210.07055))
- More superficial observations:
	- Tables start appearing early in the paper, but are only referenced at the end of the paper. This is distracting when reading.
	- Please use the parenthetical citation `\citep` instead of textual one `\cite`, where appropriate.
	- The top labels in Figure 1 should be shifted a bit more to the right (the MAE Decoder is not part of the "b) Downstream adaptation of SyncLipMAE")
	- The Sapiens-0.3B models is missing a citation.
- The authors might consider discussing the following papers as well:
	- Gao, Zheng, and Ioannis Patras. "Self-Supervised Facial Representation Learning with Facial Region Awareness." _CVPR_, 2024.
	- Zhang, Yuanhang, et al. "ES3: Evolving Self-Supervised Learning of Robust Audio-Visual Speech Representations." _CVPR_, 2024.
	- Yaman, Dogucan, et al. "Audio-Visual Speech Representation Expert for Enhanced Talking Face Video Generation and Evaluation." _arXiv_, 2024.

### Soundness
1

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper presents an MAE (masked autoencoder) style self-supervised pretraining of an audio-visual talking face model. Each video frame is patchified and tokenized. Bypass 1 uniformly masks 75% of patches. Bypass 2 preserves the facial regions with a higher probability. The ViT-style encoder takes the visible patches and outputs prompt tokens consisting of the identity prompt, ambient-motion prompt, and vocal-motion prompt. In Bypass 1, the decoder takes the identity prompt along with the visible patch tokens to decode the video frame. In Bypass 2, the decoder takes the ambient-motion and vocal-motion prompts to decode the video frame. The audio feature per frame is a concatenation of per-layer features of a pretrained speech encoder (wav2vec 2.0) projected to the same dimension as the prompt tokens. The decoder is a sequence of transformer blocks taking visible patches from bypass 1 and mask tokens. Each block applies a self-attention followed by a cross-attention with identity and ambient tokens, and a conditioning token. The decoding process with shared decoder parameters performs two passes: a video-driven pass, where the conditioning token is the vocal-motion token, and an audio-driven pass, where he conditioning token is the audio feature. Both passes reconstruct the same video frame for symmetric supervision with visual and audio signals. The model is trained with the pixel reconstruction loss, CLIP-style audio-vocal contrastive alignment loss, and cross-covariance decorrelation penalty on the prompt tokens. The paper presents the applications of the model for audio-visual synchronization measurement, facial understanding, lip reading, and video dubbing with reference frames and mouth masks.

### Strengths
* Versatile model with multiple applications related to talking head with speech audio
* Well-thought-out design and training strategy supported by evaluations and ablation studies
* Honest disclosure of limitations with intriguing insights

### Weaknesses
* Only video dubbing visual results provided
  * I would like to see lip-reading results
* The paper could be argued as an application of MAE in a specific domain
  * Though I believe the paper has enough unique contributions (prompt token design and contrastive learning) on top of MAE

### Questions
Do authors have thoughts on evaluating the consistency of the identity prompt? Will the identity be preserved if the last frame of the previous segment is not provided in the video dubbing? What happens if the identity feature comes from face recognition models like ArcFace? Can identity prompts be used like face recognition?

### Soundness
4

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
5

### Summary
This is mainly a representational learning paper for 2D talking face videos. The authors present a novel method to factor 2D videos of talking faces into clean identity, vocal motion, and ambient motion latents.

The paper achieves this by using a contrastive-masked pre-training approach. For the identity token (Bypass 1), the paper uses a random uniform mask which hides a majority of the facial frames and forces the reconstruction of the face. For the motion tokens (Bypass 2), the paper utilizes a face-preserving mask combined with photometric jitter to effectively mask the identity from the encoder, forcing it to learn motion latents. Furthermore, a standard CLIP approach is used to improve the synchronisation between the per-frame audio tokens and vocal-motion tokens specifically, while penalizing cross-covariance between the token pairs.

The pre-trained model can then be used to generate tokens for four downstream tasks, namely audio-visual stream synchronization, facial understanding, visual speech recognition, and visual dubbing.

### Strengths
The paper presents both an original problem formulation of the face representation learning problem as well as a novel pre-training design. The two-bypass masking strategy is original and appears to work well on the 2D talking face setting.

The authors demonstrate the effectiveness of this learning strategy through adaptation of learnt latents into four orthogonal downstream tasks, achieving SOTA results across synchronization, understanding, VSR, and dubbing, with the zero-shot performance on AV sync being a strong evidence for generalization.

The paper is largely well-written and easy to follow.

### Weaknesses
The paper lacks qualitative or quantitative analysis to directly evaluate the factorization quality of the learnt latents. For example, it would be helpful to show that a change in the identity token while keeping the motion tokens static would change the person’s appearance but not the animation, nor the quality of the synchronization.
Furthermore, the paper lacks discussion on how this technique can be effectively generalized into similar problem domains, such as (most relevantly), 3D facial animation tasks. While the two-bypass approach is novel, it utilizes 2D techniques like image masking and photometric jitter, which do not have direct correspondence to 3D representations. Experimentation or discussion on more generalized approach would be greatly appreciated.

Finally, it seems like the training run utilized 128 GPUs for 15 days. The large amount of compute used on relatively low-resolution images is concerning, since it suggests that while the final representations learnt may be cleanly factorized, the formulation of the training task may suffer from poor sample efficiency which hinders convergence dynamics. An analysis of performance versus compute or data scale would be beneficial.

### Questions
Could the authors provide qualitative or quantitative analysis to directly support the claim of successful factorization, instead of purely relying on downstream tasks?
How can the core principle of using separate, differently-perturbed views to disentangle constants from motion be generalized to different problem tasks, like 3D facial animation (or others)?

The pre-training requires a substantial amount of compute on seemingly low-resolution images. Could the authors provide insight into the sample efficiency of the training objective? For instance, how does performance scale with the amount of data or compute, and is the complex objective the primary driver of the high computational cost? What happens when the objective is simplified?

Current works like VASA-1 also uses latent disentanglement yet formulates it within the losses in the end-to-end network. Could the authors discuss the primary advantages and tradeoffs for the motivation of more granular factorization through masked views here?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a self-supervised approach based on masked autoencoders which learns audio and visual features from unlabelled videos. Then the pre-trained model is used on 4 different applications: visual speech recognition, audio-visual synchronisation, visual dubbing and emotion classification and achieving state-of-the-art results.

### Strengths
- The idea of pretraining an audio-visual model which can be used for multiple audio-visual downstream tasks is interesting.

- The proposed approach seems to achieve state-of-the-art results.

### Weaknesses
- The paper is hard to follow, although the general idea is clear, the fact that so many different things are presented leaves the reader a bit confused. For example, how the models are trained for each downstream task is briefly presented (probably due to lack of space) and the process is not entirely clear. Adding additional figures showing which pre-trained models are used for each task and how they are combined with the syncLipMAE would be very informative. In addition, Figure 1 is a bit confusing. It's hard for the reader to fully understand how the approach works.

- Details on the datasets used are missing. It would good if the number of subjects, number of videos and total duration of each dataset are shown including how the data are divided into (raining/validation/test sets. Also, a table showing the datasets used in each downstream task would improve the clarity of the dataset section.

- Non-standard datasets are used for evaluating the models for VSR and AV synchronisation. For example, LRS2 and LRS3 are the standard benchmarks used for VSR. Why not using them?

- The model has been pretrained on Hallo3, CelebV-HQ, CelebV-Text, MEAD, VFHQ, HDTF and RAVDESS. The same datasets are used for evaluation, e.g., facial understanding is evaluated on CelebV-HQ, AV synchronisation and visual dubbing are evaluated on Hallo3 and VFHQ and finally VSR is evaluated on HDTF, VFHQ, and Hallo3. As a consequence, the comparison with other models is not fair as they have been trained on different datasets (and the proposed has an advantage).

- The authors report SOTA results in the paper, however the visual dubbing results presented in the supplementary material do not seem SOTA.

### Questions
Please see above.

### Soundness
3

### Presentation
2

### Contribution
3
