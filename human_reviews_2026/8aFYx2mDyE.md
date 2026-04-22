# UniVerse-1: Unified Audio-Video Generation via Stitching of Experts

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
We introduce UniVerse-1, a unified, Veo3-like model capable of simultaneously generating coordinated audio and video. To enhance training efficiency, we bypass training from scratch and instead employ a stitching of experts (SoE) technique. This approach deeply fuses the corresponding blocks of pre-trained video and music generation experts models, thereby fully leveraging their foundational capabilities. To ensure accurate annotations and temporal alignment for both ambient sounds and speech with video content, we developed an online annotation pipeline that processes the required training data and generates labels during training process. This strategy circumvents the performance degradation often caused by misalignment text-based annotations. Through the synergy of these techniques, our model, after being finetuned on approximately $7,600$ hours of audio-video data, produces results with well-coordinated audio-visuals for ambient sounds generation and strong alignment for speech generation. To systematically evaluate our proposed method, we introduce Uni-Bench, a new benchmark dataset. In an effort to advance research in audio-video generation and to close the performance gap with state-of-the-art models such as Veo3, we make our model and code publicly available. We hope this contribution will benefit the broader research community.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents UniVerse-1, an open-source audio-video generation framework created by "stitching" pre-trained models, alongside a new benchmark (Verse-Bench). In my opinion, the work's main strength is enabling open-source, lip-synchronized speech generation. However, the submission omits comparisons to key existing works (e.g., JavisDiT, MM-LDM) and evaluations on established benchmarks.

### Strengths
1. Open-Source Framework for Synchronized Audio-Video Generation: The paper provides an open-source framework for joint audio-video synthesis by leveraging existing pre-trained models. This approach successfully enables functionalities like ID-conditioned control (via a reference image) and lip-synchronized speech generation.
2. Online Data Annotation Pipeline: It addresses the problem of "static misalignment" between randomly sampled training clips and their corresponding global annotations.
3. New Evaluation Benchmark: The introduction of Verse-Bench, a new benchmark featuring three diverse subsets, is a valuable contribution for evaluating joint audio-video generation.

### Weaknesses
1. The paper omits necessary comparisons (both quantitative and qualitative) to several key prior works in joint A/V generation, such as JavisDiT [1], MM-LDM [2], and AV-DiT [3].
2.The dismissal of MM-Diffusion as "not applicable"  is insufficiently justified, as other relevant works [1] have successfully benchmarked against it.
3. The model is only evaluated on the newly proposed Verse-Bench. The lack of evaluation on established public benchmarks (e.g., JavisBench, AIST++, Landscape) prevents a fair and direct comparison with existing SOTA methods.
4. The claim of being the "first open-source joint generation framework" (line 417) is factually incorrect, as several others [1, 2, 3] are already available.
5. The "Stitching" methodology introduces specific designs (e.g., linear adapters, modified attention), but these are not validated by ablation studies, which only cover LQLS and INSS.
6.  Key evaluation metrics are missing, notably Fréchet Video Distance (FVD) / Fréchet Audio Distance (FAD) for quality, and modern text-video similarity scores (e.g., ImageBind).
7. The paper appears to contain a minor typo (e.g., on Line 351), referring to "Step-Ace" when the correct model name is "Ace-step".
8. The supplementary materials lack results from the compared baseline methods.

[1]JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization

[2]MM-LDM: Multi-Modal Latent Diffusion Model for Sounding Video Generation

[3]AV-DiT: Efficient Audio-Visual Diffusion Transformer for Joint Audio and Video Generation

### Questions
1. Data Availability: The authors state they curated a 7,600-hour dataset for training. Will this full training dataset be made publicly available, or only the Verse-Bench evaluation set?
2. Choice of Audio Expert: The paper describes Ace-step as a music generation model. Why was this model chosen for a general audio-video framework that must also handle ambient sounds and, critically, speech? Why a model specializing in music was preferred over models trained on more diverse audio (e.g., AudioLDM2, Stable Audio, which are also used as baselines ).
3. Fused Block Architecture (Fig. 2): Regarding the Fused block design:
In Fig. 2(c), "mel cross attn," the query (q) input is shown as 'Video tokens'. Should the query for the audio (mel) stream not originate from the 'Mel tokens'?
Following this, the architectures for 'video cross attn' (Fig. 2b) and 'mel cross attn' (Fig. 2c) appear identical. Please clarify this design. Ideally, an ablation study to support this specific architectural choice.

### Soundness
2

### Presentation
3

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
This paper presents a joint audio-video generation method that fuses together a pretrained video generation model and a pretrained music generation model using “stitching of experts”. A data collection pipeline is proposed, which is used to collect a dataset for finetuning the fused model.

### Strengths
- If the training dataset and the benchmark are released, they will be beneficial to the community, as we lack large-scale aligned audio-video data.
- The proposed method can model video, general audio, and speech in the same network, which is a new capability in current open models, to the best of my knowledge. The performances of these areas are also competitive compared to dedicated methods. 
- The loss weighting strategy to make use of the video data with low visual quality without corrupting video generation makes sense and is a novel approach for addressing this issue.

### Weaknesses
Some of the contributions are not very clear. Thus, the reproducibility of this work is questionable.

- The online data annotation pipeline is described as one of the main contributions, but it is not mentioned in the main paper and only in the appendix. Moreover, there are insufficient details regarding the data collection and annotation process. For example, what are the parameters used in the “audio activity detection” filter? What is the prompt used for the captioning model? Are there any ablations on using this dataset?
- I am confused about the independent noise sampling strategy. Which library are the authors using to generate the random numbers?  The authors mentioned LCG, but libraries like numpy by default use PCG64, which has better statistical properties. Moreover, the authors suggest using two independent noise generators for the two streams. However, there will still be a statistical correlation between the noises sampled within the same stream (i.e., number of samples = number of dimensions), which could still be problematic if the video resolution or duration changes. It would be helpful if the authors could back up their claim with some statistical analysis.
- For the audio-video alignment metric (L917), how does it differ from the synchronization score proposed in V-AURA [1] and also adopted by subsequent works like MMAudio [2] and HunyuanVideo-Foley [3]? If they are the same, it will be clearer to reuse the name of this metric.

Minor: Figure 2 seems to be wrong. The (b) and (c) subfigures are identical and do not seem to reflect what is written in the text.

[1] Temporally Aligned Audio for Video with Autoregression, ICASSP 2025

[2] MMAudio: Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis, CVPR 2025

[3] HunyuanVideo-Foley: Multimodal Diffusion with Representation Alignment for High-Fidelity Foley Audio Generation

### Questions
- Will the training dataset and the benchmark be released?

- Are there any specific processing steps for the TTS task? Is the transcript simply fed as part of the text stream?

- L294 – Are those target features computed from the ground-truth audio?

- Questions listed in weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a join audio-video generation approach that relies on using pretrained audio (music) and video diffusion models. The main approach introduces fusion blocks that integrate each block from the pretrained audio and video model. The author also propose a dataset and benchmark and online annotation and cross-modal noise sampling techniques that help with the performance.

### Strengths
- The paper is well-written and easy to follow 
- The authors propose a dataset and benchmark for join audio-video generation
-  The authors promised the release of the code and weights

### Weaknesses
- Quality: for me the most important aspect in join audio-video generation is the temporal alignment (otherwise the two modalities can be generated separately). However, I found the provided samples in the supplementary videos to be disappointing especially with regard to the temporal alignment.  
- The main intuitive behind the architecture is limited, and limited discussion over prior work that share similar architecture and framework (e.g AV-Link) is discussed.

### Questions
see above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces UniVerse-1, a unified model for simultaneous audio and video generation, aiming to close the performance gap with proprietary models like Veo3. UniVerse-1 utilizes a Stitching of Experts (SoE) technique to combine pre-trained video (WAN2.1) and music (Ace-step) foundation models. This approach avoids training from scratch, accelerating convergence by leveraging the experts' prior knowledge.

### Strengths
++ The Stitching of Experts (SoE) paradigm is a highly original and effective approach to multimodal foundation model development. It leverages existing, high-capacity unimodal models to achieve a unified system, solving the expensive "training from scratch" problem common in multimodal scaling. The identification and solution for cross-modal noise correlation (INSS) is a subtle yet crucial technical discovery that addresses a fundamental problem in multimodal diffusion models and is highly novel.

++ The technical details are robust. The Online Data Annotation Pipeline is a necessary and high-quality solution to the long-standing problem of static data misalignment in clipped videos. The ablations on INSS and LQLS convincingly validate the contribution of the proposed components to overall performance, especially the significant boost from INSS. The inclusion of layer interpolation addresses an architectural mismatch in a principled way.

++ The paper is well-structured and clearly defines the problem, the methodology, and the key innovations. Figure 1 and Figure 2 clearly illustrate the SoE architecture and the fusion process. The derivation of the composite Overall Score and its normalization details are meticulous, demonstrating a principled evaluation approach for a complex task.

++ UniVerse-1 advances the open-source community by providing the first Veo3-like joint audio-video generation model, filling a critical gap left by proprietary systems. The SoE technique provides a blueprint for efficiently building other multimodal models by compositing unimodal experts, impacting future multimodal architecture design. The creation of Verse-Bench establishes a much-needed benchmark for fair comparative research in this challenging domain.

### Weaknesses
-- Limited Video Foundation Model Capacity: The primary limitation is acknowledged by the authors: the model is built on the WAN2.1-1.3B video model. This inherently constrains the final video quality, motion fidelity, and resolution, as noted in the user study where UniVerse-1 is "predictably surpassed by larger expert models" in unimodal video tasks. While the SoE methodology is clever, the final performance is bottlenecked by the weaker expert's prior. Suggestion: Quantify this constraint more clearly, perhaps by comparing the video quality metrics against a theoretical upper bound based on the WAN2.1-1.3B model itself.

-- Comparative Evaluation Complexity/Ambiguity: The paper repeatedly emphasizes that comparisons against SOTA unimodal expert models are for "qualitative reference" and do not represent a "direct claim of superior performance". While necessary, this ambiguity makes interpreting the unimodal metrics (e.g., audio quality compared to AudioLDM2) difficult. The true strength is in synchronization (Joint Quality Score), but the reader must continually contextualize all other metrics due to the architectural choice. Suggestion: Re-frame Table 1 and Section 4.2 to focus less on direct unimodal comparison and more on the holistic balance and Joint Quality advantage.

### Questions
Stitching Adapter Complexity: The hidden states passed through linear adapters before injection are crucial14. Could the authors specify the dimensions and complexity (e.g., number of parameters) of these two-layer linear adapters compared to the overall DiT blocks? Were alternatives (e.g., $\text{LoRA}$ adapters, single $\text{MLP}$) explored, and if so, how did they impact convergence and final performance?

What is the average real-time latency of generating the full multi-modal annotation (Step 2) for a $\sim 5s$ clip using the $\text{QWen2.5-Omni}$ model?How is the buffer size/management implemented to prevent the training process from being starved (consumer slower than producer) or receiving stale samples (producer slower than consumer)?

Cross-Modal Noise Correlation Visualization: The Independent Noise Sampling Strategy (INSS) is a key technical insight16. Could the authors provide a visualization (e.g., a cross-correlation heatmap or scatter plot in a simplified latent space) illustrating the difference between the spurious structural correlation in noise tensors ($\epsilon_a = f(\epsilon_v)$) and the desired independent sampling? This would greatly enhance the understanding of this critical issue.

Audio-Video Alignment ($\text{AV-A}$) Metric: The paper notes that $\text{AV-A}$ is computed via $\text{Synchformer}$. Is this the $\text{AV-A}$ metric measured on generated video and audio pairs, or does it compare the generated modalities to a ground truth alignment? Clarifying the exact comparison point is essential, especially since the V2A baseline comparison highlights this ambiguity.

### Soundness
3

### Presentation
3

### Contribution
3
