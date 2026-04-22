# WAVE: Learning Unified & Versatile Audio-Visual Embeddings with Multimodal LLM

- Avg Score: 6.00
- Decision: Accept (Oral)
- Scores: 8, 6, 4

## Abstract
While embeddings from multimodal large language models (LLMs) excel as general-purpose representations, their application to dynamic modalities like audio and video remains underexplored. We introduce WAVE (\textbf{u}nified \& \textbf{v}ersatile \textbf{a}udio-\textbf{v}isual \textbf{e}mbeddings), the first LLM-based embedding that creates a unified representation space for text, audio, and video modalities. WAVE employs a novel hierarchical feature fusion strategy and a joint multi-modal, multi-task training approach to enable two key capabilities: any-to-any cross-modal retrieval and the generation of prompt-aware embeddings tailored to user instructions. Experimentally, WAVE sets a new state-of-the-art on the MMEB-v2 video benchmark and achieves superior results in audio and video-to-audio retrieval. Its prompt-aware nature also yields remarkable performance in multimodal question answering, significantly outperforming existing embedding models. Ablation studies validate our joint training strategy, demonstrating improved performance across all modalities.  With a newly introduced benchmark for versatile audio-visual learning, WAVE opens up broad possibilities for cross-modal, any-to-any applications. Our code and checkpoints are released at \href{https://github.com/TCL606/WAVE}{https://github.com/TCL606/WAVE}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a method to fine-tune a multimodal large language model (MLLM) to produce prompt-aware embeddings for audio and video. Starting from Qwen2.5-Omni, the authors first learn a BEATs-based adapter to ingest an additional audio input. They then apply contrastive training with LoRA using two objectives: bidirectional cross-modal retrieval and a QA objective that treats the source modality to text as retrieval. Text embeddings are taken from the last token of the last layer. For non-text modalities, the method conditions on a prompt, aggregates the last tokens from all layers, and passes them through a fusion module. Experiments show improved retrieval performance over other MLLMs, and the fine-tuning also improves the base MLLM’s text generation task performance.

### Strengths
The task formulation and model design are clear and sensible. The experiments are comprehensive and the results strongly support the efficiency of the proposed method. The presentation is clear.

### Weaknesses
In the evaluation benchmarks, retrieval tasks appear to be reported in a single direction only, for example text-to-audio on Clotho. Reporting both directions would provide a fuller picture.

The use of the term “QA” is potentially confusing. During training, the QA objective is a source-modality-to-text contrastive setup, while in evaluation some QA tasks are executed via text generation with the MLLM. The same ambiguity appears for other evaluation benchmarks. It would help to specify, for each evaluation task, the exact inference procedure.

### Questions
1. What is the architecture of the BEATs aligner, and is it similar to the fusion module?
2. How do text prompts vary across training tasks and evaluation tasks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces WAVE, a LLM-based embedding model specifically designed to create a unified representation space for text, audio, silent video, and synchronized audio-visual inputs. 

To achieve this versatility, WAVE employs a hierarchical feature fusion strategy that aggregates representations from multiple LLM layers, alongside a dual-encoder architecture for audio inputs. 

The model is optimized through a joint multi-modal, multi-task training approach to enable any-to-any cross-modal retrieval and the generation of prompt-aware embeddings that condition on user instructions for tasks like multimodal QA. 

Experimentally, WAVE outperforms other baselines on the MMEB-v2 video benchmark and yields descent results in audio and video-to-audio retrieval, with ablation studies confirming the performance benefits derived from both joint training and the learned cross-layer fusion technique.

### Strengths
* The writing is easy to follow.

*  The architecture features an effective hierarchical feature fusion strategy by aggregating representations across multiple LLM layers and a dual-encoder design for audio. Ablation studies confirm that the proposed joint multi-modal, multi-task training strategy enables positive cross-modal knowledge transfer and superior results compared to specialist models.

*  WAVE achieves new state-of-the-art results on the MMEB-v2 video benchmark and shows superior performance in audio and video-to-audio retrieval compared to strong baselines.

### Weaknesses
1.  A major limitation is the model's reliance on prompt-aware embeddings for high performance in complex tasks like multimodal QA. Using a single general prompt to extract the embedding results in a drastic performance degradation across all QA benchmarks, highlighting the critical limitation of a single, static representation in handling complex query semantics. Although boosting performance, generating diverse embeddings can be very expensive in real-world scenarios. Consider the model that needs to generate different embeddings based on various user inputs.


2.  Achieving optimal performance requires a complex, learned MLP fusion across the last-token outputs of all LLM layers. Ablation studies showed that simpler aggregation methods, such as a direct weighted sum across layers, underperform, suggesting the necessary cross-layer interactions are highly complex. This leaves readers wondering whether all the layers were necessary, or if there are other strategies to select meaningful layers that would be sufficient.

3.   As an LLM-based model built on a 7B parameter backbone, Qwen2.5-Omni, the training demands are significant, requiring large-scale infrastructure (192 H20 GPUs for approximately 36 hours), which is hard for many labs to reproduce.

### Questions
### 1. Analysis of Prompt-Aware Embeddings and Generalization

*  Could the authors elaborate on the fundamental limitations preventing a single, static embedding from adequately capturing complex query semantics for QA? Does this performance gap imply that for high-level reasoning tasks, WAVE’s LLM backbone is using the prompt to select relevant features internally rather than deriving a truly universal, task-agnostic representation?

*  For users who need a generalized embedding for downstream applications that lack explicit questions, what is the optimal and computationally lightest prompt recommended by the authors that can minimize performance loss while maintaining acceptable semantic coverage?

### 2. Feature Fusion and Interpretability

* Given that a direct weighted sum underperforms, suggesting that cross-layer interactions are complex and non-linear, can the authors provide deeper insights into what the two-layer MLP fusion module learns? Are there any visualization techniques or analysis that can illustrate the relative importance assigned to low-level (early-layer) perceptual cues versus high-level (late-layer) semantic reasoning during the fusion process?

### 3. Dual-Encoder Necessity and Redundancy

* Since BEATs is designed for comprehensive audio event understanding, did the authors explore replacing the existing speech encoder entirely with a second, possibly smaller, instance of the BEATs encoder, or fine-tuning a single, unified audio encoder? If the dual approach is mandated by specialized roles, how do the embeddings from the two encoders contribute uniquely to the final unified representation, beyond the observed performance boost in audio retrieval?

### 4. Inference and Computational Cost

*  While the training resources are impressive (192 H20 GPUs for 36 hours), could the authors provide a comparison of the average latency and total inference computation (FLOPs or time per sample) required to generate a WAVE embedding versus competing MLLM-based embedding models (e.g., LamRA or CAFe)? Specifically, how much overhead is introduced by processing and fusing features from all layers compared to standard last-token pooling from only the final layer?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Wave, a unified multimodal embedding model capable of embedding text, audio, video and audio-video inputs. Wave is validated on a variety of video, audio and text retrieval tasks. Furthermore, Wave is capable of robustly using instructions to improve multimodal embedding performence.

### Strengths
- Introducing audio to multimodal retrieval is relatively novel. The performance of the model is quite strong.
- Using a concatenation of many layers last token hidden is a novel method for vector embedding in this setting.
- The manuscript is clear and well-written.

### Weaknesses
- The approach is very similar to [1] except an omni-model is used instead of a VLM.

- The paper is missing implementation details relating to model architecture, including how inputs are templated into LLM backbone, resolution and visual tokenization in the vision encoder.

- The paper does not compare their proposed pooling strategy with mean pooling with bidirectional attention, which tends to outperform last-token pooling strategies [2].

- The evaluations of audio-visual retrieval is limited. The reviewer can only find a single task that tests WAVE’s ability to retrieve audio-visual items (RET split of MMEB-v2-Video), and only in one direction. Furthermore, It is unclear if MMEB-v2-Video is designed to have audio used its retrieval task. This could result in the audio track trivializing the task or being useless.

- It is unclear whether WAVE’s superior video retrieval performance comes from its different techniques or larger scale video retrieval training or its specialization into video retrieval (I.e. no image retrieval).

Minor:
- Potentially missing related work: the use of “distractor” answers for QA training seems very similar to [3].

### Questions
Several question can be found in the above weaknesses section, in addition:
- How were the “distractor“ answers used during QA training created?

### Soundness
3

### Presentation
3

### Contribution
2
