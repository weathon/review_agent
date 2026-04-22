# Weakly Supervised Motion Learning for Co-speech Gesture Video Generation

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
Co-speech gesture video generation is fundamental to natural human communication and plays a crucial role in human-computer interaction. Existing approaches typically rely on a two-stage framework, first generating intermediate pose representations before synthesizing the final video. While effective, these methods require extensive pose annotations, which often introduce labeling errors, and still struggle with fine-grained details, particularly in hand generation. To address these challenges, we propose a weakly supervised motion learning framework for co-speech gesture video generation that leverages only audio and video data. Our approach consists of three key stages: (1) a motion encoder that learns a generalizable motion representation from video without pose supervision, (2) a dual-tower architecture that aligns audio with the learned motion representation using an invertible feature extractor, and (3) a video diffusion model that refines fine-grained visual details. During sampling, we introduce a hand refinement method based on initial noise optimization, where learnable noise parameters are optimized via policy gradient to enhance hand synthesis. Extensive experiments on our collected dataset demonstrate that our approach outperforms prior methods across multiple metrics, achieving superior motion fidelity, gesture realism, and overall video quality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a weakly supervised framework for co-speech gesture video generation that eliminates the need for pose annotations by learning motion directly from raw video and audio. The method consists of three stages: (1) a motion encoder that learns a generalizable motion representation from video using latent diffusion; (2) a dual-tower audio–motion mapping module with an invertible feature extractor that aligns audio embeddings to motion space without retrieval; and (3) a video diffusion model that refines fine-grained visual details. Additionally, a policy-gradient-based hand refinement optimizes the diffusion noise distribution to improve hand realism during sampling. Experiments on a 33-hour dataset demonstrate state-of-the-art performance across motion fidelity, diversity, and realism metrics, outperforming prior works such as S2G, MYA, and EchoMimicV2—all without requiring pose supervision.

### Strengths
The video generation quality for the pixel-synthesis quality in general looks fine. The hands are not blurry, which is good. Not very sinigificant deformation issues.

The design of the hand refinement and other types of refinement also looks good to me. It is a good design for the pixel-quality enhancement.

### Weaknesses
Overall, i guess the major problem would be that the authors have claimed too many things in the paper but do not have very solid experiments to validate designs, making the paper more like an integration of tricks instead of in-dpeth discussion and analysis of the proposed solutions for co-speech gesture generation model.

1) Clarity of the “motion learning” module

The current description of the motion-learning block is under-specified. From the text it is unclear how this module is trained or even what objective signals drive learning. Is it contrastive, retrieval-margin (InfoNCE / triplet), reconstruction, diffusion, or hybrid? Please specify the exact loss.

2) Is it really a good motion representation
For Feature space, the author claimed they first leverage a EVA-CLIP to extract motion feature, so it is implicit. However, it is uncertain why this motion representation is really good or not. What if utilize another encoder? How the temporal dynamics are learned from this motion representation? How does it differ from the optical-flow-based methods (MRAA, TPS, or even just explicit 2d poses, etc)? A high-level discussion of this is really important. It would be better if the authors can provide the analysis and comparison of their proposed representation with the existing ones. I can take an example experiment, for TPS transformation as the motion representation, it can be easily integrated into the authors' framework, to achieve this goal, the authors can keep the deformation network and feature extraction network from TPS, replacing the decoder of the original TPS paper with the current Stable-Diffusion as the backbone. This will let the readers to know a solid comparison of motion representations and see if this is good design.


3) Positioning vs. closely related work

The manuscript mentions a “retrieval-based setting” but does not clearly position itself against recent audio→motion literature. Please add a comparison table and narrative against:

TANGO (ICLR 2025): Audio-conditioned gesture generation with temporal alignment constraints (if applicable). What differs in your alignment strategy, representation choice, and objective?

Contextual Gesture (ACM MM 2025): Uses explicit keypoints and a masking/infilling objective. Are you using a different masking schedule, context modeling, or a retrieval head instead of generation? Why is your choice better for downstream video synthesis?

Hierarchical Cross-Modal Association (CVPR 2022): Introduces a hierarchical association between modalities. Do you model hierarchy (phoneme→word→phrase) or stay flat? If flat, discuss trade-offs.


3) Why not a generative motion mapper?

Given the two-stage design (first generate motion, then video), a purely retrieval-trained motion space leaves open the question of temporal fidelity and sample quality at inference.

Please compare against audio-conditioned generative baselines:

ANGIE (NeurIPS 2022): Audio→MRAA latent generative model.

S2G-Diffusion (CVPR 2024): Audio→TPS diffusion.

Contextual Gesture (ACM MM 2025): Audio→keypoint masking/generation.

For the motion mapping, what if the authors train these types of generative models from audio to latent motion?

4) How can we trust motion quality?

As long as you have the motion representations, the motion quality is then being able to evaluate to let readers to see if quantitative evidence that the intermediate motion is good and useful.

I would recommended evaluations like:

Motion FID / KID in the motion-feature space (compute features via a frozen motion-recognition backbone or your own motion encoder).

Fréchet Gesture Distance (FGD) or DTW-based alignment between generated and reference motion.

Diversity & coverage: average pairwise diversity, nearest-neighbor coverage over the test set.

Sync metrics: Audio–motion beat/peak alignment score; landmark velocity–onset correlation.

### Questions
During the Rebuttal period, would the authors provide the long-sequence gesture generation videos? The current ones are all short videos, very hard to evaluate if there would be repeated gesture pattern issues, and long dependency issues now.

What is the sampling rate for the video for training and generation, it looks like it is less than 30 frames per second, and look a little bit unnatural.

The results for EchomimicV2 look strange. The raw EchomimicV2 is not supposed to look like that. 

Please see the weakness as above.

### Soundness
2

### Presentation
1

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
This paper's motivation is training co-speech gesture video generation model without pose annotation. 
1. The task is a audio-text-image to video model (ATI2V) for talking avatar cases.  The detail of the backbone is unclear but it's a latent diffusion model (LDM) based on U-Net. The dataset is open-source 33 hours fix-camera talking dataset (S2G). 
2. The idea is first learn a implict motion representation/encoder for video, then train a auido-to-motion model, and finally train the audio-to-motion-to-video model. 
    -   motion represenation learning: freeze all parameters of the LDM including VAE, finetune the CLIP image encoder (authors called motion encoder).
    -   audio to motion: freeze finetuned motion encoder. train a audio encoder and a motion mapper with MSE + Contrastive + Flow loss. 
    -   a2m2v: only finetune the LDM part. 
3. The evaluation have both objective scores and 10+ subjective videos.

### Strengths
1. The design of each module has good reference, for example, mulit-mode flow, invertiable motion encoder etc.
2. The results outperforms current method in both objective metrics and subjective videos.
3. The paper is organized and could follow. 
4. The idea of finetuning CLIP to reconstuct video is interesting and has insight.

### Weaknesses
1. The motivation of pose annotation-free maybe not strong in current ATI2V model.  
    -    Due to the good pretrained base model, from the early of 2025, new ATI2V models (Hunyuan video avatar, etc) dont require pose to train the speech-to-gesture model, end2end modeling with a simple audio adapter has been demonstrate enough for basic audio-gesture alignment, such as beat, lip-sync, emotion alginment. could the author explain (in theroy is also fine) why the proposed method is better than such end2end modeling?
2. The performance of finetuning CLIP to reconstruct video is unclear.
    -    Could CLIP or finetuned CLIP model accurate spatial positions of the motion? or just semantic? I just confused if it could why we need ControlNet. 
    -    If just semantic, what level it could archieve? could it understand the details of finger motion for example. The abality of its representation is better than pose or even worse? 
3. The purpose for some loss terms for audio-to-motion part is unclear.
    -    what is loss flow
    -    why using MSE and Contrastive loss together? seems a combination for single-mode and multi-mode loss without reasons. 
4. Ablation study for stage 3. 
    -    This is similar to weakness 1. Consider the audio encoder is freezed here, how about replace it to a freezed hubert or wav2vec2, will it also work?
    -    The finetuning of LDM here is finetuning all parameters or only audio related parameters?

### Questions
Overall this is an organized paper and I could follow, it has both objective and subjective evaluations. My major concern is the high level motivation. since the authors design a relativly complex three stage system, it requires strong evidence in theroy or results it works better than simple end2end modeling. I'm also interested in the discussion in the CLIP finetuning part. and the other questions are optional.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel, weakly-supervised framework for co-speech gesture video generation that eliminates the need for pose annotations. Unlike previous two-stage methods that rely on explicit pose data, this approach learns an implicit motion representation directly from video. The framework has three key stages: a motion encoder learns motion representations from video without pose supervision; a dual-tower architecture with an invertible feature extractor maps audio directly into the learned motion space for inference; and a video diffusion model refines fine-grained visual details. It outperforms state-of-the-art methods (e.g., S2G, MYA) across multiple objective metrics (FGD, FVD, Diversity, BAS) and in user studies.

### Strengths
- I recognize this idea of dumping intermediate noise label supervision, which would make the training scalable. But I think this work does not work very well.
- The paper includes comprehensive experiments to evaluate the effectiveness of each design.

### Weaknesses
1. To support the motivation, the comparison should include the baseline trained with noise pose conditioned with the video model, either wan2.1 or other tiny model for efficiency. I do not think the current video models will perform  worse than the baselines included in the supplementary video.
2. The annotation error in in-the-wild videos always occurs in the hand region. However, most existing video models also fail to produce plausible details. Therefore, the work also employs an additional hand refinement model to enhance this. So, I think this kind of weak supervision and the motion encoder do not really work. 
3. As observed in the ablation demo 1, the identities after applying each stage changed a lot. The gestures between stage 1 and stage 2 are totally different. This does not make sense if the audio encoder and motion encoder are  aligned well. I mean the poor semantics  alignment.
4. The work is limited by the constraints of current video diffusion models, which can only generate a few seconds of video, unlike explicit pose-driven methods.

### Questions
as listed in weekness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a novel weakly supervised framework for generating co-speech gesture videos directly from audio and video data, without requiring pose annotations. Traditional methods rely on two-stage pipelines (audio-to-pose, then pose-to-video), which demand extensive labeled pose data and often fail to capture fine-grained hand details.

The proposed framework consists of three major stages:

1. Motion Learning — A motion encoder learns generalizable motion representations from raw video data without explicit pose supervision.

2. Audio-Motion Mapping — A dual-tower architecture aligns audio and motion spaces using an invertible feature extractor based on normalizing flows, enabling efficient audio-to-motion mapping.

3. Feature Refinement — A video diffusion model refines visual details and ensures realism.

Additionally, the authors introduce a hand refinement strategy that optimizes the initial noise distribution during sampling using policy gradient methods.

Experiments on a self-constructed dataset (based on PATS) demonstrate that the proposed method achieves state-of-the-art performance across four key metrics—Fréchet Gesture Distance (FGD), Diversity, Beat Alignment Score (BAS), and Fréchet Video Distance (FVD)—outperforming prior works such as S2G, MYA, and EchoMimicV2.

### Strengths
The paper introduces a weakly supervised paradigm for co-speech gesture video generation that eliminates the dependence on explicit pose annotations—a major bottleneck in existing two-stage frameworks. This design is original and impactful, as it reframes gesture synthesis from a pose-supervised problem to a self-contained audio–video learning task. The incorporation of an invertible feature extractor for mapping between audio and motion spaces is also creative, offering an elegant solution that avoids retrieval-based inefficiencies or generative instability.

The proposed method is also carefully designed and systematically evaluated through multiple ablation studies that dissect each component’s contribution. Quantitative results show consistent and significant improvements across multiple metrics (FGD, BAS, FVD, and Diversity) compared to strong baselines such as S2G, MYA, and EchoMimicV2. The experiments also include both objective and user evaluations.

The paper is clearly written and well-structured. The motivation is compelling, situating the problem within the broader context of human–computer interaction and multimodal generation. The three-stage pipeline (motion learning, audio–motion mapping, and feature refinement) is logically presented and supported by clear figures and equations. The authors clearly distinguish their contributions from prior works such as ReenactAnything and S2G, making it easy to understand the conceptual leap from pose-supervised methods to weakly supervised motion learning.

Overall, the paper makes a meaningful advance in the field of gesture video generation and broader multimodal video synthesis. By removing the need for costly and error-prone pose annotations, the proposed framework substantially lowers the barrier to scaling audio-driven gesture datasets and models. This shift towards weak supervision could influence future research in gesture synthesis, human motion modeling, and audio-visual learning. The improved gesture realism and synchronization also have practical implications for digital avatars, virtual presenters, and embodied conversational agents.

### Weaknesses
1. The experiments are conducted solely on a small subset of the PATS dataset (33 hours) featuring four speakers. This raises concerns about the model’s ability to generalize to unseen identities, languages, or speaking styles. Since the proposed framework is claimed to learn motion representations without pose supervision, a more convincing demonstration would include cross-identity or cross-domain evaluations, such as applying the model to TED, Trinity, or BEAT datasets.
2. The paper does not discuss where the weakly supervised framework fails or underperforms compared to pose-supervised approaches. For instance, it remains unclear whether the learned motion encoder occasionally entangles motion and appearance, especially under large camera or body movements. Explicitly showcasing these failure cases and analyzing the underlying causes would make the paper more balanced and informative.
3. The framework involves multiple large components—VAE, 3D-UNet, invertible flow, and diffusion model—which may lead to high training and inference costs. However, the paper does not report computation time, GPU memory footprint, or efficiency comparisons with existing pose-supervised methods.

### Questions
1. Have the authors tested the framework on other gesture datasets such as BEAT, Trinity, or TED to assess cross-domain generalization?
2. How well does the model adapt to unseen identities or speaking styles without fine-tuning?
3. Could the authors visualize or analyze the learned motion embeddings (e.g., using t-SNE or PCA) to show that they capture meaningful motion dynamics rather than appearance cues?
4. What is the training time and GPU cost for each stage?
5. How does inference speed compare to previous supervised or two-stage baselines like S2G and MYA?

### Soundness
3

### Presentation
3

### Contribution
3
