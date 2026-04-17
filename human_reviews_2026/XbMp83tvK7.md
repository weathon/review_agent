# UniUGG: Unified 3D Understanding and Generation via Geometric-Semantic Encoding

- Decision: Accept (Poster)
- Scores: 4, 8, 8, 2

## Abstract
Despite the impressive progress on understanding and generating images shown by the recent unified architectures, the integration of 3D tasks remains challenging and largely unexplored. In this paper, we introduce UniUGG, the first unified understanding and generation framework for 3D modalities. Our unified framework employs an LLM to comprehend and decode sentences and 3D representations. At its core, we propose a spatial decoder leveraging a latent diffusion model to generate high-quality 3D representations. This allows for the generation and imagination of 3D scenes based on a reference image and an arbitrary view transformation, while remaining supports for spatial visual question answering (VQA) tasks. Additionally, we propose a geometric-semantic learning strategy to pretrain the vision encoder. This design jointly captures the input's semantic and geometric cues, enhancing both spatial understanding and generation. Extensive experimental results demonstrate the superiority of our method in visual representation, spatial understanding, and 3D generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents UniUGG, which is a unified model for both 3D understanding and generation. Specifically, both geometric and semantic information of input images is encoded with the visual encoder. Also, a spatial decoder is used for 3D generation. Experimental results show that the proposed unified 3D understanding and generation method achieves superior performance for the visual representation of the visual encoder, spatial understanding capability, and 3D generation capability.

### Strengths
- Having a unified model for 3D understanding and generation is pioneering, which can probably serve as the basic work for the unified 3D model direction. 

- The idea of binding a MASt3R-powered spatial decoder with LLMs is innovative and reasonable to me, which enables the whole system to output something that is spatial-aware, like the point cloud in the generation part of this work.

- The proposed method achieves decent experimental results on the benchmarks for visual representation, spatial understanding, and 3D generation, demonstrating that it is a well-rounded unified 3D model.

### Weaknesses
- It remains unanswered for the motivation of proposing such a unified 3D foundation model for both understanding and generation. In other words, there is lack of evidence that modeling them together in a single model can provide mutual benefit to each other. If there is no mutual benefit between these two streams of tasks, or even slightly harm each other, it will become unjustified for us to design such a delicate framework for unifying understanding and generation for 3D, because optimizing them separately may produce optimal performance.

- I think it is a bit overclaiming to say the understanding task UniUGG solves are 3D understanding. Most of the tasks (like the ones in VSI-Bench, BLINK) are the vision-language tasks related to spatial reasoning, with limited relation with 3D understanding. I would say the benchmarks like SQA3D [1], ScanQA [2], ScanRefer [3] are more related to 3D understanding, as they are understanding and reasoning with 3D representations, instead of only reasoning on single (or sometimes multi-view) images for some spatial understanding tasks.

- The writing of the methodology section (Sec. 3) makes me feel a bit difficult to understand how the proposed UniUGG framework looks like. It starts with introducing the visual encoder, which I find a bit lost because I even do not have a rough idea in mind about what modules the full UniUGG framework has. Therefore, I find confusing where is this visual encoder locates in the whole framework. The introduction of the following modules also share the same problem. Thus, I believe this section lacks a part that introduces the rough whole pipeline to audience at the beginning, for them to have a big picture in mind.

- The pipeline figures (Figures 2-4) are also not clear enough. They are crowded with small text, and are hard to understand. Basically, I cannot find a clear figure showing what the inputs and outputs are for the model. The contents inside the figure are also confusing. For example, in Figure 3(b), the large language model seems to take in image 2 representation together with transformation queries and question & answer tokens. Is image 1 the target image that we want to generate, so it is not processed through large language models. And the encoding of image 1 and image 2 seems to be messed up with the representation of image 1 and image 2.

[1] Ma et al. SQA3D: Situated Question Answering in 3D Scenes. ICLR 2023.

[2] Azuma et al. ScanQA: 3D Question Answering for Spatial Scene Understanding. CVPR 2022.

[3] Chen et al. ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language. ECCV 2020.

### Questions
- How will the model perform if understanding and generation tasks are separately optimized? If separately optimizing them can achieve better performance for each of them, then there is limited justification for modeling them together. Otherwise, the cases (e.g., interactive multi-round scene editing like those in Uni3D-LLM [1]) that requires joint understanding and generation capabilities need to be shown.

- How will the model perform in the more perceived 3D understanding / reasoning tasks, like SQA3D, ScanQA, ScanRefer that are more used in 3D VLMs?

- What is the training cost of this work? The paper only mentions about the batch size and the number of optimization steps of the training the model, without mentioning what kind of GPU resources they are using, and how much time is needed for training the model, which makes it unclear how efficient this algorithm is.

[1] Liu et al. Uni3D-LLM: Unifying Point Cloud Perception, Generation and Editing with Large Language Models. arXiv:2402.03327.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose a unified framework for 3D understanding and generation, introducing a geometric–semantic learning strategy for encoder training based on the MASt3R architecture. The method includes a Spatial-VAE that compresses visual features into latent tokens for efficient 3D decoding, and a joint training setup where a large language model (LLM) and a diffusion model predict new views. Additionally, the model is trained with visual question answering (VQA) tasks to maintain its 3D understanding capability. Quantitative and qualitative experiments are conducted on both 3D generation and 3D understanding tasks across various spatial reasoning benchmarks.

### Strengths
1. Unifying 3D understanding and generation is an interesting and novel research direction.
2. The proposed geometry–semantic encoder is well-designed; incorporating the MASt3R framework is an effective and interesting choice.
3. The paper is well-written, and the experiments are thorough, covering a diverse set of novel view synthesis and vision–language spatial reasoning benchmarks.

### Weaknesses
1. The paper would benefit from a more detailed discussion of limitations and failure cases.
2. The method’s robustness to more extreme view transformations is not clearly evaluated.

### Questions
1. Could the authors elaborate on the main limitations and share examples of failure cases?
2. How sensitive is the model to large viewpoint changes in the 3D generation task?
3. I am also curious on whether joint training on understanding and 3D generation introduce any performance trade-offs between the two tasks.

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
3

### Summary
The paper proposes an LLM-based 3D framework for spatial-VQA and 3D scene generation. It addresses the limitations of current vision-language models, which are largely restricted to 2D semantics and next-token objectives that fail to capture spatial and geometric structure. To this end, the paper proposes a geometric-semantic pretraining strategy for the image encoder using multiview supervision. A spatial-VAE module then compresses the geometric-semantic features into compact 4-dimensional latent tokens for geometry-conditioned image synthesis. The model combines a geometry-aware encoder, a spatial-VAE model and a language conditioned diffusion generator. Experiments are done on Feat2GS benchmark for novel-view synthesis and several vision-language reasoning benchmarks.

### Strengths
1. The encoder is trained using geometric supervision which enables spatial reasoning compared to models trained on 2D images.
2. A single latent representation is used for both 3D reasoning and 3D generation.
3. The method allows geometric control through text or parametric view prompts.

### Weaknesses
1. There’s no discussion or visualization of the 3D output from the model.
2. All the evaluations are done in 2D space and there is no 3D evaluation performed. It’s unclear if the model is actually learning the structure of the 3D world or simply rendering based on the distribution of the training data.
3. The model is not trained end-to-end. The encoder and spatial VAE are trained separately and kept frozen during unified training (lines 301-302).

### Questions
1. There’s no discussion or evaluation of 3D output. Does the model learn explicit 3D geometry or only 3D features?
2. How does the model perform if the full model is trained end-to-end?
3. In the language-driven view transformation, how accurate is the angular or spatial accuracy?
4. What is represented by the 4-dimensional spatial-VAE tokens?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes a method to ingest images and can generate extended scenes based on text input or camera parameters. How this works is unclear to me. I've read the paper 3 times and I still don't fully understand how the different parts work together.

### Strengths
- **S.1:** The qualitative results look great.

### Weaknesses
- **W.1:** I've worked with LLMs, I've worked on diffusion papers, and I know about the Mast3r paper. Yet, I have no clue what's going on in this work and why. I think the writing is a bit too dense to follow unless the reader is already intimately familiar with all the related works. And I also don't think the figures are doing a great job at illustrating the method. I can't say anything more about this work. Maybe an overarching, vastly simplified diagram would be helpful.

### Questions
- **Q.1:** Doesn't the use of a VAE in the second step defeat the purpose of semantically aligning the encoder in the first step?
- **Q.2:** Why isn't the LLM trained directly to output tokens that can be decoded into visual representations by the VAE decoder, rather than predicting something that can be rendered through a diffusion process?

### Soundness
1

### Presentation
1

### Contribution
4
