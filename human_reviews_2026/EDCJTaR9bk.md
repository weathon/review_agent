# UniVideo: Unified Understanding, Generation, and Editing for Videos

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Unified multimodal models have shown promising results in multimodal content generation and editing, but remain largely limited to the image domain. In this work, we present UniVideo, a versatile framework that extends unified modelling to the video domain. UniVideo adopts a dual-stream design, combining a Multimodal Large Language Model (MLLM) for instruction understanding with a Multimodal DiT (MMDiT) for video generation. This design preserves the MLLM's original text generation capabilities, enables accurate interpretation of complex multimodal instructions, and maintains visual consistency in the generated content. Built on this architecture, UniVideo unifies diverse video generation and editing tasks under a single multimodal instruction paradigm and is jointly trained across them. Extensive experiments demonstrate that UniVideo matches or surpasses state-of-the-art task-specific baselines in text/image-to-video generation, in-context video generation and in-context video editing. Notably, the unified design of UniVideo enables two forms of generalization. First, UniVideo supports task composition, such as combining editing with style transfer, by integrating multiple capabilities within a single instruction. Second, even without explicit training on free-form video editing, UniVideo transfers its editing capability from large-scale image editing data to this setting, handling unseen instructions such as changing the environment or altering materials within a video. Beyond these core capabilities, UniVideo also supports generation with thinking, where the MLLM interprets complex prompts
and guides the MMDiT during synthesis. To foster future research, we released our model and code.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces VOGUE, a unified framework for video understanding, generating, and editing from a single multimodal prompt. To achieve this, VOGUE includes a MLLM for complex semantic and instruction understanding, which is followed by a MMDiT for high-fidelity video generation. The authors demonstrate that VOGUE is able to understand complex multimodal instructions and perform diverse tasks such as T2V, I2V, in-context generation, in-context editing, and even zero-shot generalization. The experiments also show that VOGUE outperforms SOTA task-specific methods on multiple benchmarks like VBench, UNICBench, and human evaluations.

### Strengths
- The most impressive strength of the work is the wide coverage on diverse tasks, including video understanding, T2V, I2V, in-context video generation, and in-context video editing, under a single framework. As far as I understand, VOGUE is the first video generation model that achieves this level of task unification which can substantially enhance the community interest.

- I also like the experiments where the authors show the model can achieve some zero-shot generation tasks which are not explicitly covered during training. For example, it can transfer the editing abilities to the unseen tasks composition (such as combining style transfer with object deletion). This highlights the advantages to leverage the powerful and frozen MLLM.

- The experiments include the comparisons with existing models on eight tasks. It is also impressive that such a general-purposed model can achieve a similar or comparable performance with the tasks-specific models.

- The authors also provide thorough ablation studies, including 1) the necessity of multi-task learning (compared to single-task learning) and 2) dual-stream model architecture, which are insightful.

- The authors promise that they will release the model checkpoint and code, which can significantly increase the reproducibility and also benefit the further research in unified video generation and editing.

- The paper is well written and easy to follow. The figures are well-plotted and informative which can make readers quickly understand the core ideas.

### Weaknesses
- My major concern is the incremental technical contribution. The dual-stream architecture that separately processes multimodal instructions and visual inputs has been explored in some existing models such as FLUX, which also employs two streams to process text and image inputs. Also, using MLLM embeddings to condition the diffusion process has also been explored in Qwen-Image. Therefore, while the model design is effective, it feels more like an integration of existing ideas and does not look insightful to me.
- It is unclear why the dual-stream architecture improves generalization compared to single-stream model design. For example, models like FullDiT [1] and Qwen-Image achieve great performance using full self-attention without separating understanding and generation streams. Could the authors provide the motivation behind this architecture design?
- The paper provides limited information about the construction of the training dataset. I hypothesize that learning such a general-purpose model requires high-quality, large-scale, and carefully-curated datasets for training. However, the authors only give a brief summary in Appendix F without describing more details such as the data filtering or annotation. It raises concerns about how the model learns to cover complex tasks such as visual prompt understanding.
- Following the above concerns, while Figure 6 shows several types of visual prompts, it is unclear how the model handles out-of-distribution or ambiguous prompts.
- There are some missing citations and comparisons with existing multi-subject video personalization models, such as Video Alchemist [2] and Movie Weaver [3], which can also be formulated as in-context generation.

[1] "FullDiT: Multi-Task Video Generative Foundation Model with Full Attention", ICCV 2025

[2] "Multi-subject Open-set Personalization in Video Generation", CVPR 2025

[3] "Movie Weaver: Tuning-Free Multi-Concept Video Personalization with Anchored Prompts", CVPR 2025

### Questions
- Could the authors explain the theoretical motivation for the proposed dual-stream architecture and include ablation studies to verify the design choice of separating the understanding and generation streams?
- Could the authors elaborate on the dataset curation pipeline?
- For the visual prompting task, could the authors describe whether the model can handle OOD or ambiguous prompts, and provide some failure cases to help check the generality of this task?

### Soundness
4

### Presentation
4

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
The work proposes a method to imbue a pretrained video generator with multimodal input understanding. The work builds on the observation that text embedding can be replaced with embeddings produced by a frozen MLLM, making it possible for the downstream video generator to leverage MLLM capabilities such as thinking and understanding of multimodal inputs. The authors first curate datasets comprising a variety of multimodal tasks such as image editing, image to video, style transfer, object swapping, addition, deletion. Then a pretrained Hunyuan model is adapted to receive QwenVL2.5-7B multimodal inputs in 3 stages, by first training a connector MLP, then fine-tuning the video generator on T2I and T2V, and finally training on the full range of tasks. The resulting model produces convincing results in a variety of tasks and shows ability to generalize outside of the set of training tasks. Quantitative evaluation shows generally better or comparable scores with respect to VACE and commercial models.

### Strengths
- The task of creating unified video generators capable of tackling multiple visual tasks from T2V to video editing is of high relevance
- Qualitative results are convincing. The model seems to be able to tackle a range of video generation tasks from object insertion to instruction based editing with convincing quality
- Ablations are insightful and validate that 1) joint training on all tasks reinforces model performance in all tasks, 2) relying on MLLM visual embeddings is insufficient, requiring visual tokens being passed to the video generation backbone directly
- The authors provide complete training and dataset details 
- Authors commit to public release of model and code

### Weaknesses
- Ablation on MLLM visual embeddings raises doubts of how much the introduction of the MLLM contributes to model performance with respect to the newly collected data. See questions.

Minor:
- Minor amount of typos, especially lack of space before citations (LL254, LL258)
- LL52, LL423-LL427 the claim appears inaccurate. Veo 3 was previously found to support this feature (https://www.reddit.com/r/singularity/comments/1m9b0bq/googles_new_feature_in_veo_3_you_can_now_draw/) (https://www.youtube.com/watch?v=KNGMBRyGcDo). I suggest removing the claim.

### Questions
I hope the authors could clarify the following questions
- Table 5 reports that "w/o visual for MLLM" achieves an overall score that is very close to VOGUE. While the exact setting is slightly unclear (are we just removing visual tokens from the MLLM output or is the MLLM not receiving visual tokens at all?) LL440-LL441 suggests that no visual token is processed by the MLLM in this setting. While a small gap is present in the prompt following (PF) metric, this raises the question of whether an MLLM is needed at all. If only text tokens are processed by the MLLM in the performed ablation, wouldn't the original text encoder produce similar performance without the need for an MLLM? This is an important point to clarify as otherwise the role of the MLLM is unclear, and a simpler framework with equal capabilities could be constructed by means of the collected dataset only without any MLLM.
- Table 2 reports a series of Understanding metrics. Do I understand correctly that these metrics are computed using the frozen MLLM model alone without involvement of the video generator? If so, such metrics are less interesting to show. The claim that the model can perform "Understanding" in a unified way with generation and editing in this case would be problematic as the "Understanding" part is completely offloaded to a pretrained and frozen external model, which is not a unified approach. In a truly unified approach, the MM-DiT itself would output text tokens for understanding tasks when necessary.

Minor:
- Could the authors clarify more how the dataset ratios shown in Table 1 were computed? Can the authors offer insights into how optimal ratios should be derived?

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
The paper presents a system for joint image/video understanding, generation and editing. It consists on two streams: understanding and generation. The understanding stream is the frozen Qwen-VL2.5-7B. The generation stream is initialized from HunyuanVideo-T2V-13B. The training is split into 3 stages: 1) training the connector between MLLM and MMDiT; 2) fine-tuning MMDiT for base tasks (T2I + T2V); 3) multi-task fine-tuning of MMDiT. The results look very good visually, and the model was even shown to zero-shot generalize to novel tasks (e.g. free-form video editing). The authors also perform a thorough quantitative evaluation demonstrating the superiority of the method over existing baselines.

### Strengths
- It is a quite elegant architecture which i think among the early works for this paradigm shift in generative modeling of concatenating everything into a single sequence and modeling. While it does not serve as a good reference for ablations, but it's a good proof of concept to convince the community that this paradigm works.
- The results are really good, especially given the little amount of GPUs used to fine-tune the model. Also, the data size is affordable to collect for non-bigtech companies (e.g. startups)
- The paper reads well and the illustrations are good. The submission also includes many qualitatives which are very easy to view on the website.

### Weaknesses
- From my perspective, the main weakness is the lack of rich ablations, that could make the submission to be a good reference for follow up works. For example, is it necessary to do all the 3 stages sequentially or we can train jointly? Is the diffusion schedule the same for all the modalities? How much improvement can we get by making it different? Would it help to do some dropout on some input modalities in a task? Is it possible to replace full attention with cheaper attention variants in-between the modalities? Also some profiling results would be interesting to see, e.g. how much compute is spent on each modality in each component (is MLLM heavy?) And so on.
- Some important details are missing from the submission, mainly related to data curation. It would be fine to omit them for a technical report, but it is not good practice to omit them for an academic submission. For example, how exactly was stylized video transformed into a real one (appendix F.2)? How was the video inpainting model training (appendix F.1)? Was the dataset post-curated with human annotators? If so, what were the instructions for them?

Small writing comments:
- typo: "generate an video" => "generate a video" (in figures)
- typo: "source open source" on line 907
- I would suggest re-coloring Figure 3 to display the frozen MLLM as blue, and trainable DiT as red, as common in the prior literature.

### Questions
Could you please include more dataset details (as specified in the previous section)?

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
The paper is cleanly structured and easy to read. Qualitative results are abundant and illustrative, and the in-context editing/generation comparisons are clearly presented. At a high level, the work’s strengths are system-level integration and breadth of tasks. The level of novelty is average but acceptable.

### Strengths
1. A pragmatic dual-stream unification (frozen MLLM for understanding + MMDiT for generation) that feeds visual inputs to both streams, with ablations showing why both sides need visuals for identity preservation and semantics.
2. Strong mask-free in-context editing/generation results despite baselines using masks; the qualitative figures and automatic metrics make the claim legible and practically relevant.
3. Transparent training recipe and task coverage, including staged training, freezing choices, connector design, and explicit mixing ratios. It is useful for reproduction and future baselines.

### Weaknesses
1. Thinking Mode is not well and fully discussed(How much does it benefit?).
2. What is the benefit of making it a single model rather than using a workflow or agent? Qualitative analysis?

### Questions
1. The workflow diagram (Fig. 3) isn’t very clear. Is it correct to interpret that the user feeds an interleaved text–image instruction to the MLLM, and then the MLLM’s output is concatenated with the VAE features of the conditioning images as the input? Also, why is the noise term missing in Fig. 3? It feels like this should be made consistent with Fig. 2.
2. Maybe compare to more unified understanding & generation model

### Soundness
3

### Presentation
3

### Contribution
3
