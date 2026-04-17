# HiFi-Foley: Multimodal Diffusion with Representation Alignment for High-Fidelity Foley Audio Generation

- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Recent advances in video generation produce visually realistic content, yet the absence of synchronized audio severely compromises immersion. To address key challenges in video-to-audio generation, including multimodal data scarcity, modal semantic response imbalance, and limited audio quality in existing methods, we propose HiFi-Foley, an end-to-end text-video-to-audio framework that synthesizes high-fidelity audio precisely aligned with visual dynamics and semantic context. Our approach incorporates three core innovations: (1) a novel multimodal diffusion transformer that addresses semantic response imbalance between video and text modalities through dual-stream audio-video fusion via joint attention and balanced textual semantic injection via cross-attention; (2) a representation alignment training strategy that employs self-supervised audio features to guide latent diffusion training, thereby improving audio quality and semantic consistency; (3) a scalable data pipeline leveraging open-source tools for cleaning raw data and constructing training datasets. Extensive evaluations demonstrate that HiFi-Foley achieves state-of-the-art performance across audio fidelity, visual-semantic alignment, temporal alignment, and distribution matching.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents HiFi-Foley, a text-video-to-audio (TV2A) generation model that synthesizes high-fidelity and semantically aligned audio from multimodal inputs.
The model introduces:

1. Dual-phase attention in a multimodal diffusion transformer (MMDiT) — using joint self-attention for video-audio alignment and cross-attention for text injection.

2. A Representation Alignment (REPA) strategy — aligning DiT hidden states with pre-trained ATST audio features to improve semantic and acoustic fidelity.

3. A scalable 122k-hour TV2A dataset built using an open-source cleaning and labeling pipeline.

In summary, I believe this paper is a boardline paper, it includes a lot of experiments and engineering efforts. But the novelty is limited to a certain degree.

### Strengths
1. The paper integrates modern components (MMDiT, flow matching, REPA) coherently into a unified architecture. The dual-phase attention (joint self-attention + cross-attention) for modality balancing is conceptually sound and clearly motivated. 

2. Implementation details (architecture, loss, datasets, training setups) are thorough. The 122k-hour data pipeline is well-engineered and valuable to the community (if the code and data are open-sourced)

3. The paper compares against strong baselines (MMAudio, FoleyCrafter, ThinkSound, etc.) with both objective and subjective metrics. Ablations on attention design, interleaved RoPE, and REPA placement are also provided.

### Weaknesses
1. The architecture integrates existing ideas: The dual-phase attention essentially decomposes joint attention (as used in MMAudio) into two stages — an incremental modification, not a conceptual innovation. The REPA alignment is directly adapted from prior visual generative works. Although these integrations are good for the video-to-audio community, the contribution is more engineering refinement.

2. The 122k-hour dataset construction is technically significant, but I am not sure such construction whether novel compared to previous multimodal filtering pipelines.

### Questions
whether the 122k-hour dataset construction will be open-source?

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
5

### Summary
This paper introduces HiFi-Foley, an end-to-end framework for generating high-fidelity Foley audio from both video and text prompts. Authors proposed a novel multimodal diffusion transformer architecture using a dual-phase attention mechanism to process audio-video latents. In addition, a REPA loss is used to improve audio quality and semantic consistency. Moreover, this paper introduces a large-scale audio-video data, solving the data scarcity problem. Extensive experiments show that the proposed method achieves a state-of-the-art performance on various benchmarks.

### Strengths
1. Use of REPA loss: using a pre-trained audio encoder as a "teacher" to guide the diffusion model's internal feature space is a clever way to distill rich, general-purpose acoustic knowledge into the generative process.
2. Large-scale data curation: a large-scale high-quality audio-video dataset is constructed with advance data filtering, preprocessing, and annotation. 
3. Impressive performance on three benchmarks across various evaluation metrics.

### Weaknesses
1. The dataset is not released. 
2. The used REPA is not a new stuff and has been introduced in image generation. 
3. Limited model novelty: the proposed model structure is similar to MMAudio. 
4. Based on Table 5, REPA has the limited improvement on the performance. And also adding the unimodal DiT has the limited improvements while introducing more model parameters 
5. The addition of REPA aims to improve the audio quality and alignment, why it also benefits the DeSync?
6. Which datasets are for training, which ones are for testing?
7. Why not give the ground truth reference in Figure 5-6? Otherwise it’s hard to compare different methods.

### Questions
1. Whether the dataset will be released?
2. What are differences between the proposed model structure and MMAudio? 
3. REPA and unimodal DiT have the limited improvement on the performance. Did you try different representation loss specifically designed for video-to-audio task?
4. Did you compare the model performance between (text+video)-to-audio and video-to-audio variants? 
5. The description of training and testing datasets should be more clear. 
6. The ground truth audio references are not shown in Figure 5-6.

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
4

### Summary
This paper introduces HiFi-Foley, an end-to-end framework for high-fidelity text-video-to-audio (TV2A) generation. The authors identify three key challenges in existing methods: multimodal data scarcity, "modality imbalance" (where models over-rely on text cues and ignore video), and low audio quality. To address these issues, HiFi-Foley proposes three main contributions: A novel architecture, a new training strategy, and a scalable data pipeline. The authors also developed a custom, high-fidelity DAC-VAE for audio encoding/decoding. Evaluations on Kling-Audio-Eval, VGGSound-Test, and MovieGen-Audio-Bench show that HiFi-Foley achieves state-of-the-art results.

### Strengths
* The dual-phase attention mechanism is a well-motivated and intuitive solution to the stated problem of text-over-reliance. Separating the fine-grained A-V temporal alignment (via self-attention) from the global text conditioning (via cross-attention) is a strong architectural contribution, and the ablations in Table 5 confirm its effectiveness.

* The idea of interleaving audio and visual tokens before applying ROPE is a simple but clever technique to enforce fine-grained temporal correlation between the two modalities. The ablation study shows this provides a clear benefit over conventional ROPE.

* Applying the REPA concept to align with a pre-trained audio model (ATST-Frame) is a logical and successful strategy. The ablations clearly demonstrate that this improves performance and that the choice of the guide model (ATST) and its application layer (unimodal block 8) are important.

### Weaknesses
* In Section 4.2, the paper states that on VGGSound-Test, HiFi-Foley "leads in audio quality metrics (IS, PQ)". However, Table 3 clearly shows its IS score (16.14) is significantly worse than MMAudio's (21.00). This is a factual error.

* The comparison to ThinkSound is explicitly handicapped. The authors state they "only evaluate the version without Chain-of-Thought (CoT) instructions". This means a core component of the baseline is missing, making the comparison unfair and the reported improvements potentially misleading.

* The data pipeline is a key contribution, but critical details for reproducibility are missing. The paper states an "empirically design a standard" was used for filtering based on PQ, SNR, ImageBind, and AV-align, but the actual numerical thresholds are not provided.

* The model cannot generate intelligible speech. This is a major limitation, especially since "Human voice" (32.75%) is the single largest category in the new training dataset (Figure 4). The paper does not adequately explain this significant failure mode.

### Questions
* Why does the text in Section 4.2 claim the model "leads in... IS" on VGGSound-Test, when Table 3 shows it is significantly outperformed by MMAudio (16.14 vs 21.00)?

* Given that 32.75% of your training data is "Human voice" , why does the model completely fail to produce intelligible speech? Is this a limitation of the DAC-VAE, a bias in the GenAU captions (e.g., "speech" vs. actual transcripts), or an artifact of the REPA/flow-matching objectives?

* The "high-quality" tag is always appended at inference. What happens if this tag is not used? Does this technique harm performance when trying to generate sounds that are naturally low-bandwidth (e.g., a distant rumble)?

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
5

### Summary
The paper introduces HiFi-Foley, an end-to-end text-video-to-audio framework that synthesizes high-fidelity audio precisely aligned with visual dynamics and semantic context.

The main paper contributions are:
- a novel multimodal diffusion transformer that addresses semantic response imbalance between video and text modalities through dual-stream audio-video fusion via joint attention and balanced textual semantic injection via cross-attention.
- a representation alignment training strategy that employs self-supervised audio features to guide latent diffusion training, thereby improving audio quality and semantic consistency.
- a scalable data pipeline leveraging open-source tools for cleaning raw data and constructing training datasets.

Extensive evaluations demonstrate that HiFi-Foley achieves state-of-the-art performance across audio fidelity, visual semantic alignment, temporal alignment, and distribution matching.

### Strengths
The main strength of the paper is the data curation pipeline designed to build a high quality training dataset for the video to audio generation task.

### Weaknesses
I believe that the proposed contributions lack novelty or significance.
- Injecting text via cross attention in video to audio generation has been proposed before (MovieGen). Moreover, the motivation for this architecture design is unconvincing. The imbalance between conditioning signals can usually be addressed by employing different guidance weights during inference, which is not considered in this paper (at least as a baseline). The paper does not even mention the inference parameters used in the experiments.
- The REPA loss has been introduced in prior works and this paper just applies it to a new task.
- The proposed data curation pipeline is mostly descriptive and the resulting dataset is not published. The size of the resulting curated dataset is one order of magnitude than the biggest non curated open source dataset (AudioSet), which is probably the main explanation for the appealing results of the Tables 2 and 4. Moreover, the curation pipeline is undocumented and thus non reproducible. For example, no extensive description of the thresholds used at the different filtering steps are given.

The results presented in the tables, which do not report confidence intervals, yield minimal differences between the different methods (such as the Tables 5, 6 and 7). Thus they are unconvincing to the reader. For what it is worth, according to my experience, absolute variations of less than 0.2 in A4 scores are not significant. 

The Figure 2 provides the same information as the Tables 2, 3, 4.

### Questions
What is the "HunyuanVideo-Foley" model mentioned in the Figure 2 caption?
What is the ATST-Frame model mentioned throughout the paper?
What is the parallel cross attention ablation in the Table 5? A figure would be welcome.

### Soundness
1

### Presentation
2

### Contribution
1
