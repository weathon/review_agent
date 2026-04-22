# Syncphony: Synchronized Audio-to-Video Generation with Diffusion Transformers

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Text-to-video and image-to-video generation have made rapid progress in visual quality, but they remain limited in controlling the precise timing of motion. 
In contrast, audio provides temporal cues aligned with video motion, making it a promising condition for temporally controlled video generation. 
However, existing audio-to-video (A2V) models struggle with fine-grained synchronization due to indirect conditioning mechanisms or limited temporal modeling capacity.
We present Syncphony, which generates 380×640 resolution, 24fps videos synchronized with diverse audio inputs. Our approach builds upon a pre-trained video backbone and incorporates two key components to improve synchronization: 
(1) Motion-aware Loss, which emphasizes learning at high-motion regions; 
(2) Audio Sync Guidance, which guides the full model using a visually aligned off-sync model without audio layers to better exploit audio cues at inference while maintaining visual quality.
To evaluate synchronization, we propose CycleSync, a video-to-audio-based metric that measures the amount of motion cues in the generated video to reconstruct the original audio. Experiments on AVSync15 and The Greatest Hits datasets demonstrate that Syncphony outperforms existing methods in both synchronization accuracy and visual quality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Syncphony, an audio-to-video generation model that focuses on improving synchronization between audio and motion.
To achieve this, the authors introduce a Motion-Aware Loss for capturing movement intensity and an Audio Sync Guidance (ASG) mechanism for enforcing synchronization between audio and visual dynamics.
A new evaluation metric, CycleSync, is also proposed to better align with human perceptual judgments of synchronization.
Experimental results show that Syncphony performs better than most existing baselines.

### Strengths
1.	The introduction of the CycleSync metric sounds reasonable and shows higher alignment with human perceptual evaluations than prior synchronization metrics.
2.	The model achieves consistently better performance than most baselines across several benchmarks.

### Weaknesses
1.	Generating a 5-second video takes nearly 3 minutes, which significantly limits practical usability.
2.	The main model builds heavily on Pyramid Flow, with only moderate extensions (audio conditioning and synchronization guidance). As such, the contributions feel incremental.
3.	The CycleSync metric relies on pretrained V2A models that may introduce background or irrelevant audio content, potentially biasing the evaluation.
4.	The proposed Motion-Aware Loss does not explicitly capture semantic motion as authors mentions. It merely measures pixel or latent differences between consecutive frames, which may not correspond to meaningful sound-related motion.

### Questions
In the demo, only 2-second video samples are provided, even if it can generate 5 seconds video. Why is that?

### Soundness
3

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
This paper proposes Syncphony, a diffusion-transformer-based framework for audio-to-video (A2V) generation that achieves fine-grained synchronization between sound and motion. The method introduces two key innovations: 1) Motion-aware Loss — reweights reconstruction loss to emphasize high-motion regions, improving temporal alignment. 2) Audio Sync Guidance (ASG) — uses an auxiliary “off-sync” model (without audio layers) to guide the full model during sampling toward stronger audio-motion coupling. To evaluate synchronization, the authors further introduce CycleSync, a new video-to-audio-based metric that measures whether the generated motion contains sufficient cues to reconstruct the original audio. Experiments on AVSync15 and TheGreatestHits datasets show improved synchronization and comparable or better visual quality relative to baselines such as AVSyncD and TempoTokens.

### Strengths
1. The paper proposes Motion-aware Loss and Audio Sync Guidance to improve the audio-visual synchronization of A2V generation, which are conceptually simple and empirically effective.
2. The proposed metric CycleSync offers a meaningful step forward over prior metrics (AV-Align, AlignSync, RelSync) by enabling evaluation at 24 fps and better correlating with human perception.
3. The paper also proposes a principled strategy to adapt a pretrained I2V model for the A2V task by selecting the most relevant layers to inject the audio cross-attention layer.
4. Syncphony consistently outperforms baselines on synchronization (CycleSync) and achieves competitive or superior FID/FVD, indicating that temporal precision does not come at the expense of visual quality.

### Weaknesses
1. While the paper’s ideas are sound, the architectural novelty is moderate. Syncphony heavily relies on a pretrained Pyramid Flow backbone, and the main innovations are at the loss and sampling levels rather than core model design.
2. The writing of the paper can be improved. For example, in the introduction, more details/motivations about the proposed methods could be included instead of the background information. 
3. Limited baselines: the proposed method is only compared with AVSyncD and Pyramid Flow. Is it possible to include more baselines, such as those listed in AVSyncD?

### Questions
1. Fig 5 and 6: it will be helpful to include the ground-truth video as a reference. 
2. How does ASG compare with the vanilla classifier-free guidance? For example, use the features with and without audio input as guidance. 
3. The paper only finetunes the last 16 blocks (8–23) of the Pyramid Flow backbone. Are there any experimental results supporting the benefits of this choice, e.g., compared to finetuning the full model?
Others: see the weakness section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Syncphony demonstrates meaningful progress in the audio-to-video generation domain. Its main contributions include: (1) a motion-aware loss that re-weights the training objective based on optical flow to focus the model’s learning on motion-intensive regions; (2) a training-free guidance technique that enhances the injection of audio information during inference without additional training; and (3) a more intuitive and reasonable evaluation metric for measuring audio-visual synchronization through reconstructed audio from generated videos. Experiments on two public datasets and qualitative case studies further validate the method’s effectiveness in improving both synchronization and visual quality.

### Strengths
- The proposed training-free guidance is novel and effectively points out the core challenge of achieving precise spatiotemporal alignment in audio-to-video generation.

- The proposed evaluation metric is intuitively reasonable and addresses the limitation of conventional metrics like FVD, which fail to effectively measure spatiotemporal alignment.

- The use of RoPE for spatiotemporal positional encoding enhances temporal consistency and spatial coherence in video generation, contributing to smoother and more structured motion representation.

- The demo results demonstrate good temporal alignment consistency.

### Weaknesses
Although the work is technically sound, there are several issues that I feel must be discussed.

- Regarding the motion-aware loss, it relies on a strong assumption that the visual content remains static and confined to a single scene. This assumption holds almost perfectly under the authors’ setting where short video clips of around two seconds. However, in more realistic video generation scenarios, background changes, camera motion, and scene transitions can occur without producing strong audio cues, but may catastrophically distort optical flow estimation and thus limit the scalability of the proposed approach.

- The proposed guidance technique randomly drops audio cross-attention layers during inference; however, this inference structure (unlike CFG or Autoguidance) is never encountered during training, making the resulting “unconditional” outputs less predictable. Moreover, although the authors provide some visual demonstrations, the underlying motivation may primarily apply to relatively smaller models where audio conditioning tends to be weaker. In larger-scale video generation frameworks such as Wan or Hunyuan-Video, audio conditions may not be as easily ignored, and models could exhibit different skip-layer behaviors. Rather than validating only on additional datasets, I would prefer the authors to verify this idea across multiple video generation baselines to strengthen the generality of their claims.

- Although the proposed new metric is intuitively reasonable, current video-to-audio (V2A) models also suffer from (or are still addressing) spatiotemporal alignment issues, which means they may not serve as a fully reliable ground-truth proxy. Moreover, the approach fundamentally increases evaluation time, potentially limiting scalability to larger experiments.

### Questions
- Could the authors elaborate on how they plan to address the challenges posed by more realistic (or longer) audio-to-video scenarios, where optical flow estimation can be severely affected by background motion, camera movement, or scene transitions beyond the main subject?

- Could the authors show how the skip-layer behavior manifests in other video generation models and whether similar phenomena can be consistently observed? In addition, if temporal alignment is indeed a crucial aspect of the task, why not consider using energy-based audio features as a more direct form of control or guidance?

- Given that modern multimodal large language models (e.g., Gemini 2.5) already demonstrate strong capabilities in understanding audio-visual information, and that existing video-to-audio alignment metrics (such as DeSync, which can be applied similarity in a2v field) provide reliable proxies for spatiotemporal correspondence, could the authors clarify or compare what specific advantages their proposed metric offers over these established approaches?

### Soundness
2

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
This paper introduces Syncphony, an audio-to-video (A2V) model capable of generating 380×640 resolution, 24fps videos synchronized with diverse audio inputs. Built upon a pre-trained video backbone, it emphasizes on audio-visual synchronization through two main contributions:
- Motion-aware Loss, a loss weighting mechanism emphasizing on the high-motion regions of the video.
- Audio Sync Guidance, a classifier-free guidance design focusing solely on audio conditioning to emphasize better synchronization.

Moreover the authors introduce an auxiliary contribution for the model evaluation:
- CycleSync, an audio similarity metrics between the detected peaks of the ground truth and the A2V -> (pretrained) V2A generated audio.

The A2V model is finetuned from a pretrained I2V model, by adding cross attention layers in the latter transformer blocks of the pretrained model. The backbone is a pretrained PyramidFlow Video model, trained on videos up to 5 seconds long at 24 fps and 380 × 640 resolution. Audio is sampled at 16kHz and encoded through DenseAV for conditioning. Text is encoded through CLIP. Temporally-aware RoPE frequencies are employed to aligned the modalities in the aforementioned cross attention layers.

Evaluations are conducted on the AVSync15 dataset using both objective (FID, FVD, Image-Audio similarity, Image-Text similarity, CycleSync) and subjective metrics (IQ, FC, Sync). They demonstrate that the proposed method outperforms the AVSyncD on all axes.

On the Greatest Hits dataset, the proposed method outperforms on the CycleSync and FVD metrics, showing its emphasis on both video generation quality and audio visual synchronization.

### Strengths
The paper quality is enforced by thorough ablations and experiments, presented both in the main sections and appendixes, notably:
- demonstration of the greater sensitivity of the CycleSync metrics compared with previously proposed ones, running correlation with human study and experimenting with controlled settings such as temporal shifts.
- ablations on the Motion-aware Loss and Audio Sync Guidance contributions
- pretrained model behavior study to understand where to inject the finetuning layers

Although this paper presents an Audio to Video model, its contributions should translate into better the Video to Audio model designs.

### Weaknesses
The Figure 1 is not clear enough to me. It is not clear what the frozen and trainable layers refer to (are those transformer blocks?). I would suggest adding in the captions that the audio features are injected in the latter blocks (the expectation is usually to add cross attention to each block so the presentation is quite counter intuitive without a corresponding explanation).

### Questions
Formatting:
- Add spacing before opening parentheses (around the line 347).
- In the table 3, I suggest better highlighting which model is the final version (maybe in the top row and by adding a row separator).
- As one of the paper's contributions is a classifier-free-guidance design, I would suggest adding some ablations of the choice of the cfg parameters.

### Soundness
3

### Presentation
4

### Contribution
3
