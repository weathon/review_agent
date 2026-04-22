# AnimeShooter: A Multi-Shot Animation Dataset for Reference-Guided Video Generation

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 4, 6

## Abstract
Recent advances in AI-generated content (AIGC) have significantly accelerated animation production. To produce engaging animations, it is essential to generate coherent multi-shot video clips with narrative scripts and character references. However, existing public datasets primarily focus on real-world scenarios with global descriptions, and lack reference images for consistent character guidance. To bridge this gap, we present AnimeShooter, a reference-guided multi-shot animation dataset. AnimeShooter features comprehensive hierarchical annotations and strong visual consistency across shots through an automated pipeline. Story-level annotations provide an overview of the narrative, including the storyline, key scenes, and main character profiles with reference images, while shot-level annotations decompose the story into consecutive shots, each annotated with scene, characters, and both narrative and descriptive visual captions. Additionally, a dedicated subset, AnimeShooter-audio, offers synchronized audio tracks for each shot, along with audio descriptions and sound sources. To demonstrate the effectiveness of AnimeShooter and establish a baseline for the reference-guided multi-shot video generation task, we introduce AnimeShooterGen, which leverages Multimodal Large Language Models (MLLMs) and video diffusion models. The reference image and previously generated shots are first processed by MLLM to produce representations aware of both reference and context, which are then used as the condition for the diffusion model to decode the subsequent shot. Experimental results show that the model trained on AnimeShooter achieves superior cross-shot visual consistency and adherence to reference visual guidance, which highlight the value of our dataset for coherent animated video generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces AnimeShooter, a new dataset designed to address the task of reference-guided, multi-shot animation generation. The AnimeShooter dataset is constructed through an automated pipeline that collects animation videos from YouTube, uses Gemini to generate hierarchical story scripts including story-level and shot-level annotations, and leverages models like Sa2VA and InternVL to extract and filter high-quality character reference images.
To validate the dataset's effectiveness, the authors propose a baseline model, AnimeShooterGen, which uses MLLM and video diffusion models to generate subsequent shot according to the reference and context. Experimental results demonstrate the effectiveness of the AnimeShooter dataset and AnimeshooterGen.

### Strengths
- The primary contribution is a large-scale dataset for the animation domain that addresses a clear and important research gap in multi-shot, reference-guided video generation. Its scale and rich hierarchical annotations make it a valuable resource for the community.
- The authors employ a highly automated pipeline to process data. And the data is carefully designed.
- To demonstrate the dataset's effectiveness and practical utility, the authors also introduce AnimeShooterGen. This model serves as a strong baseline and effectively validates that the proposed dataset can be used to train robust models for this complex task.

### Weaknesses
- The baseline model's strong performance heavily relies on the fourth stage, "LoRA Enhancement," which is essentially test-time finetuning on a few video clips of a specific IP. This makes the model more of a few-shot IP customization method rather than a general reference-guided generator. Can the authors provide a quantitative ablation study comparing AnimeShooterGen's performance on all metrics with and without the LoRA Enhancement stage?
- How does the model perform in a zero-shot setting? That is, given a reference image for an IP that was unseen during all training stages (including LoRA enhancement), how does AnimeShooterGen compare to baselines like IP-Adapter?
- Why was the decision made to use only the last frame of the previous shot as visual context? Were other representations (e.g., first and last frames, multiple frames, pooled video features) explored?
- The dataset provides both a narrative caption and a descriptive caption for each shot. How were these two caption types used when training the baseline model? Were they concatenated, or was only one type (e.g., the descriptive caption) used?

### Questions
Please See Weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces AnimeShooter, a large animation‑focused dataset aimed at reference‑guided, multi‑shot video generation. Each roughly one‑minute “story” includes (i) story‑level annotations (storyline, 1–3 main characters with reference images, and main scenes) and (ii) shot‑level annotations (ordered shots with scene, characters, and both narrative and descriptive captions). A smaller AnimeShooter‑audio subset adds synchronized shot‑level audio descriptions and sources. To demonstrate utility, the authors propose AnimeShooterGen, an autoregressive pipeline that conditions a video diffusion model on: (a) a user reference image, (b) the last frames of previously generated shots, and (c) the shot text. An MLLM backbone produces a conditioning embedding (via a Q‑Former adapter) for a DiT‑based video generator; LoRA layers enable light test‑time adaptation. On a custom multi‑IP evaluation set, AnimeShooterGen outperforms existing baselines on CLIP similarity and DreamSim (shot‑ and story‑level), and in MLLM and user studies.

### Strengths
1. The paper isolates a genuinely under‑served setting: reference‑guided multi‑shot animation generation with cross‑shot character/style consistency, rather than single‑shot real‑world videos or global captions.
2. The combination of story‑level elements (storyline, scenes, character cards + reference images) and fine‑grained shot‑level captions is valuable for autoregressive modeling and evaluation.
3. Conditioning on a reference image + prior last frames + text through an MLLM + Q‑Former adapter is technically sound and well motivated.
4. Good empirical results on the proposed evaluation benchmark compared with baselines.

### Weaknesses
1. The compared baselines are relatively weaker baselines (e.g., IP‑Adapter+I2V and CogVideo‑LoRA). Stronger baseline models with MLLM might also be considered.
2. The evaluation metrics with CLIP and DreamSim cannot capture some video quality aspects like motion smoothness. Better automatic evaluation metrics for these categories should be investigated (e.g., like in VBench). 
3. More qualitative examples/analysis should be included for generalization to longer shots (e.g., 15 shots) to support the claim of "AnimeShooterGen generalizes robustly to longer sequences during testing."

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes AnimeShooter to address the current limitations in multi-shot datasets, which mainly focus on real-world scenarios and lack reference images. AnimeShooter is a reference-guided multi-shot animation dataset. For each shot, the dataset provides annotations of the scene and characters, as well as visual descriptions in both narrative and descriptive forms. Additionally, a subset with synchronized audio annotations, AnimeShooter-audio, is provided. Moreover, this paper introduces a reference-image-guided multi-shot video generation model based on MLLMs and diffusion models. The effectiveness of the proposed model is validated through both qualitative and quantitative experiments.

### Strengths
1. The paper introduces a multi-shot video dataset in the anime domain, along with a subset containing audio data, laying a foundation for advancing research in animation storytelling.
2. The paper is well-organized and highly readable, with figures concisely illustrating the workflow, data structure, and qualitative comparisons.

### Weaknesses
1. In the proposed method, visual information such as reference images and different shot contexts is encoded by the MLLM and aligned with the text embeddings of the diffusion model. This can lead to the loss of fine details from the reference images or shot scenes. As shown in Figure 5, without LoRA enhancement, the consistency of details is not satisfactory. However, many real-world workflows prefer models that do not require additional fine-tuning.

2. The baseline methods compared in this paper are not specifically designed for multi-shot video generation. Since they lack any cross-shot perception capability, it is unsurprising that the proposed method shows improvements over these baselines.

3. The videos generated in this paper exhibit weak storytelling and scene transitions across different shots. In my view, they resemble a combination of multiple videos rather than multiple shots of a single coherent video.

4. The paper lacks more ablation studies to demonstrate the effectiveness of the proposed model architecture.

### Questions
The proposed dataset contains multi-character shots, but the results presented in the paper are all single-character. Is the method capable of generating multi-character videos?

### Soundness
2

### Presentation
4

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
This paper introduces AnimeShooter, a large-scale, reference-guided multi-shot animation dataset featuring hierarchical story- and shot-level annotations, synchronized audio tracks, and strong cross-shot visual consistency. To demonstrate its utility, the authors propose AnimeShooterGen, a baseline framework that integrates a Multimodal Large Language Model (MLLM) with a video diffusion model for generating coherent multi-shot animations.

### Strengths
AnimeShooter provides structured, story-aware, and reference-guided annotations, filling a significant gap in current video-generation datasets. Clear separation between story-level and shot-level elements enables both global narrative control and local visual coherence.
The open release of both dataset and baseline has high potential to become a standard benchmark for multi-shot animation generation.

### Weaknesses
1.Hierarchical captioning reduces drift but lacks visual grounding, leaving potential hallucination issues.
2.Only basic normalization is used; no explicit domain alignment across different animation styles and for object segmentation methods, fine-tuning on ~500 frames offers limited adaptation from real-world to animated content.
3.For keyframe-selection, real-video heuristics are applied; not well-suited for low frame-rate animation.
4.No explicit loss or alignment; character consistency relies on semantic coincidence. Lacks global temporal structure, leading to potential drift over long multi-shot sequences.
5.The MLLM–diffusion pipeline is computationally heavy, with no reported efficiency metrics or optimization strategy.

### Questions
see weaknesses

### Soundness
1

### Presentation
3

### Contribution
3
