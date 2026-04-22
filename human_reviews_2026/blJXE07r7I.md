# EditVerse: Unifying Image and Video Editing and Generation with In-Context Learning

- Avg Score: 5.60
- Decision: Accept (Oral)
- Scores: 8, 4, 6, 4, 6

## Abstract
Recent advances in foundation models highlight a clear trend toward unification and scaling, showing emergent capabilities across diverse domains. While image generation and editing have rapidly transitioned from task-specific to unified frameworks, video generation and editing remain fragmented due to architectural limitations and data scarcity. In this work, we introduce EditVerse, a unified framework for image and video generation and editing within a single model. By representing all modalities, i.e., text, image, and video, as a unified token sequence, EditVerse leverages self-attention to achieve robust in-context learning, natural cross-modal knowledge transfer, and flexible handling of inputs and outputs with arbitrary resolutions and durations. To address the lack of video editing training data, we design a scalable data pipeline that curates 232K video editing samples and combines them with large-scale image and video datasets for joint training. Furthermore, we present EditVerseBench, the first benchmark for instruction-based video editing covering diverse tasks and resolutions. Extensive experiments and user studies demonstrate that EditVerse achieves state-of-the-art performance, surpassing existing open-source and commercial models, while exhibiting emergent editing and generation abilities across modalities.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents EditVerse, a unified framework for image and video generation and editing. By representing images, videos, and text as one-dimensional sequences of interleaved tokens, EditVerse demonstrates promising capabilities in in-context learning and cross-modal knowledge transfer. EditVerse achieves state-of-the-art results on both the EditVerseBench and TGVE+ benchmarks. Furthermore, it exhibits impressive emergent abilities when trained at scale with unified multi-modal data. Although interactive editing and generation scenarios were not evaluated in this work, I believe this approach, with appropriate scaling, could demonstrate significant potential and enable substantially broader applications. In summary, this is a technically solid and well-presented paper.

### Strengths
1. **Unified Design**.
The paper presents a clean solution : treat text, images, and videos as interleaved token sequences with a 4D positional embedding (sequential, temporal, height, width). This simple yet effective design handles arbitrary resolutions and durations naturally, which is quite elegant.
2. **Knowledge Transfer between Different Modality**.
The key insight is using abundant image editing data to help with scarce video editing data. Through full self-attention and mixed training (6M image editing + 288K video editing samples), the model learns to transfer knowledge across modalities. 
3. **Solid Experimental Results**.
The paper presents comprehensive experiments across image/video generation and editing tasks, demonstrating that EditVerse achieves state-of-the-art performance with only 2B parameters. The ablations also clearly show both image and video data matter—images help with instruction understanding, videos help with temporal consistency. What's interesting is the emergent behavior: the model can handle tasks it wasn't explicitly trained on, and sometimes even beats the ground truth by combining knowledge from different domains.
4. **Strong Results  with Potential**.
With just 2B parameters, EditVerse achieves SOTA on EditVerseBench and TGVE+, matching or beating much larger models and commercial systems. The results hold up in both automatic metrics and user studies across 20 different editing tasks. I believe this approach has significant room to grow—with more data and parameters, it could unlock even broader applications.

### Weaknesses
1. **Background Drift in Local Edits**.
I noticed some background preservation issues in local editing tasks—there seems to be temporal drift in unedited regions. Could you elaborate on when and why this happens? Understanding these failure cases would be helpful.

2. **Why Text-Only Localization Work So Well?**
Most video editing models need masks for precise object editing, but EditVerse seems to work with just text instructions. This is quite impressive but also surprising. What makes this possible? Is it the scale of training, the unified architecture, or the cross-modal learning? A deeper analysis here would really strengthen the paper's contribution.

### Questions
See weaknesses.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces EditVerse, a novel unified framework designed to perform both image and video generation and editing within a single model. The authors identify two primary bottlenecks in video editing: (1) architectural limitations of existing models, which are often task-specific, and (2) the scarcity of high-quality, instruction-based video editing data.

### Strengths
1. The core design of an interleaved 1D token sequence for mixed image/video data, combined with a full self-attention mechanism and the novel 4D RoPE, is a powerful and original solution for multimodal in-context learning.
2. The paper's standout strength is its clear demonstration that a unified model can learn video editing from data-abundant *image* editing datasets. The ablation in Table 4 (showing the model *can* edit video with zero video-edit data, albeit poorly) and the major quality drop from removing image-edit data (Fig 8) provide powerful evidence for this hypothesis.
3. The authors address the ecosystem problem by not only building a model but also creating the data (232K video-edit samples) and the evaluation tools (EditVerseBench) necessary to make progress, and they are releasing the benchmark.

### Weaknesses
1. The most significant weakness, which the authors acknowledge in the appendix, is the computational cost. Using full self-attention on a 1D-tokenized video (where the sequence length $L$ includes all frames) leads to $O(L^2)$ complexity. The reported 118 seconds for a single 360p video on an A100 is very slow and will likely scale quadratically or worse with resolution and duration, making it impractical for real-world use on high-resolution or long-form videos.
2. The video editing data is synthetically generated by a pipeline of specialist "teacher" models (VACE, DiffuEraser, etc.). While this is a clever solution, it means the model's performance may be capped by the quality and biases of these teachers. It's unclear if the model can perform edits that its teacher models are incapable of.

### Questions
1. Could the authors provide more details on the scaling properties of EditVerse? How do inference time and VRAM usage scale with (a) video duration (e.g., 3s vs. 10s vs. 30s) and (b) resolution (360p vs. 720p)? Is the full self-attention approach feasible beyond the short, low-resolution clips shown?
2. The dimension allocations for the 4D RoPE (56H, 56W, 12Seq, 4Temp) are specific. Could the authors provide a brief rationale or ablation for this design choice? For example, why is the sequential dimension (12) given more embedding space than the temporal dimension (4)?
3. How does the model perform on editing tasks or concepts that are *not* present in its synthetic "teacher" models? For instance, if VACE is poor at "making an object transparent," can EditVerse still learn this concept purely from the image-editing data and apply it successfully to video, or is its video performance on this task limited by VACE?
4. The "wrong position" failure case in Fig 9a suggests limitations in complex spatial reasoning. The VLM evaluation (Table 2) is high, but this metric averages 3 frames. How well does this VLM metric capture these kinds of high-level logical or instruction-misalignment failures, as opposed to per-frame artifacts or quality?

### Soundness
3

### Presentation
4

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
The paper presents EditVerse, a unified video editing model that aims to handle diverse edit types (style change, object manipulation, temporal edits, localized edits, etc.) with a single model. The core modeling idea is to serialize visual content (both images and multi-frame videos) into an interleaved token sequence, and to use a multi-dimensional rotary positional embedding (RoPE) capturing spatial, temporal, and sequential position.

On the evaluation side, the paper introduces EditVerseBench, which focuses on real-world, non-square aspect ratios (horizontal and vertical videos) across multiple editing categories, arguing that prior benchmarks mostly assume square crops and narrow edit types. They also describe a data generation and VLM-based filtering pipeline to curate higher-quality training dataset. Quantitative comparisons are reported using automatic VLM metrics, frame-level visual quality and temporal consistency metrics, plus a human pairwise preference study.

Overall, the paper addresses a practically relevant problem and proposes a promising integration of a unified model design, a new benchmark, and a data curation pipeline. Several points require clarification, in particular the transparency of the evaluation setup, the reporting of the human study, and ablations on key components, but these issues appear to be fixable.

### Strengths
The paper addresses a practically relevant problem by aiming to unify diverse instruction-based video editing tasks within a single model. The proposed interleaved visual token representation together with the multi-dimensional RoPE design is technically interesting and potentially reusable beyond this specific application. The introduction of a benchmark with realistic horizontal and vertical video formats and multiple editing categories is valuable for evaluating real-world usage, and the experimental comparisons include both strong instruction-based methods and training-free baselines, which provides a relatively comprehensive view of performance.

### Weaknesses
The human evaluation lacks essential reporting details such as annotator composition, annotation protocol, and reliability, making it difficult to assess the credibility of the results. The design choices of the proposed benchmark may be perceived as favoring the method, and the paper does not sufficiently analyze scenarios in which simpler, training-free baselines still outperform the model. In addition, the contribution of the positional encoding components is not well-isolated through ablations, and the automatic evaluation pipeline, including data filtering and VLM-based scoring prompts, is not fully transparent, which limits the reproducibility of the results.

### Questions
1. **Human evaluation protocol**

   More details are needed on the human preference study: how many annotators were involved, their backgrounds, how annotation was distributed per person, and whether any quality control or inter-annotator agreement was assessed. Such information would increase the credibility of the human study results.

2. **Clarification on the Qwen2-VL–based data filtering process**
   The description of the VLM filtering process remains vague. Could you specify the exact score threshold used to retain samples, provide an overview of score distributions before and after filtering (ideally across data sources), and comment on whether Qwen2-VL frequently misjudged quality (e.g., bad edits retained or good edits discarded), even approximately? Greater transparency here would strengthen the argument for the data pipeline.

3. **Human rating in the main results table**
   For the core quantitative comparison (e.g., Table 2), it would be valuable to include a human rating or user preference column alongside the automatic metrics, similar to prior works such as TokenFlow, Tune-A-Video, and FateZero. This would provide a more direct comparison from a human perspective and make the main results more informative, rather than relying solely on automatic metrics.

4. **Benchmark construction and possible bias**

   Since EditVerseBench excludes square videos entirely, it would be helpful to clarify the rationale for not including them alongside horizontal and vertical formats. This concern is reinforced by the fact that in Table 6, EditVerse does not always outperform training-free baselines on V2VBench, where some simpler methods achieve competitive or better results. It would be useful to discuss in which scenarios such lightweight baselines remain preferable in practice, and how the proposed benchmark design avoids unintentionally favoring EditVerse.

5. **RoPE ablations**
   As the multi-dimensional RoPE is positioned as a key architectural contribution, more direct evidence isolating its effect would be useful. Even a small-scale study comparing variants that remove temporal RoPE, remove spatial RoPE, or use only standard sequential RoPE would help substantiate the importance of each component.

6. **Data filtering transparency**
    The data pipeline uses a VLM to assign 0–10 quality scores (instruction adherence, temporal consistency, artifact severity, etc.) and then thresholds these scores to filter large editing datasets.

   - Please show the score distribution before vs. after filtering.
   - How was the threshold chosen, and how sensitive is final model performance to that threshold?
   - Do you have an estimate of “false positives” (bad edits that passed) and “false negatives” (good edits that were filtered out)?
      Since the dataset is part of the claimed contribution, a brief audit would strengthen the narrative.

7. **Reproducibility of automatic evaluation**
   To facilitate reproducibility, would you consider releasing the exact prompt templates and inference configurations used for the VLM-based automatic evaluation? Without these details, the community may find it difficult to replicate the automatic scores.

8. **Minor fixes for clarity**

   - The training loss formula should be corrected to:
     $$
     \mathcal{L} = \mathbb{E}_{t,X_0,X_1} \left\| u_\Theta(X_t, t) - (X_1 - X_0) \right\|^2.
     $$

   - The naming and ordering of the RoPE components in the text (around lines 200–221) do not match Figure 2. The text first describes “Height and Width Dimensions” before sequential and temporal, while Figure 2 uses the order spatial → temporal → sequential. Please unify terminology and ordering to avoid confusion.

   - In Table 9, the symbol used to mark the method is “†”, but the note below the table refers to “‡ uses LLM-rewritten prompts.” The marker in the table and the footnote symbol should be consistent.

   - The citation for “TokenFlow” is incorrect. The paper currently cites *“Tokenflow: Unified Image Tokenizer for Multimodal Understanding and Generation”*, which is unrelated to video editing. It should cite **“TokenFlow: Consistent Diffusion Features for Consistent Video Editing”**. The incorrect reference appears in line 348–349, Table 3, and line 818.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes EditVerse, a unified framework for image and video generation and editing. The core idea is to represent text, image, and video as a single interleaved token sequence, processed by a full self-attention transformer. A 4D ROPE position embedding is proposed to encode spatial, temporal, and sequence positions. To address video editing data scarcity, the authors build a synthetic video editing pipeline to collect 0.2 million video editing data for model training. Experiments shows that their model can suprpass existing open-source and commercial models across various tasks.

### Strengths
1. The paper is well-written, and the proposed framework is clearly presented and easy to understand.

2. The visualized qualitative results are impressive and effectively demonstrate the model’s capabilities.

### Weaknesses
1. The main weakness of this work lies in the lack of substantial technical novelty. The proposed pipeline appears highly similar to recent multimodal generative frameworks (e.g., Bagel, OmniGen). The novelty mainly resides in engineering integration rather than in proposing a fundamentally new modeling formulation.

2. In my opinion, the key contribution of the paper is to show that a unified video editing/generation framework can be implemented. However, the authors do not commit to releasing the engineering details, dataset collection pipeline, or the collected dataset itself. Without open sourcing or releasing enough details for reproduction, their engineering contribution becomes less impactful.

3. The justification for the necessity of the proposed interleaved representation remains insufficient. Although the authors include some ablations, it is still unclear why interleaving tokens is fundamentally better than other existing designs (e.g., cross-attention conditioning, multi-encoder structures).

### Questions
Refer to weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces EditVerse, a unified multimodal framework for text-guided image and video generation and editing. The model represents text, image, and video modalities as a single interleaved token sequence and applies self-attention for in-context learning and cross-modal reasoning. To address the lack of video editing data, the authors design a large-scale automated data generation pipeline. Extensive experiments demonstrate that EditVerse achieves strong performance and exhibits emergent editing capabilities that generalize beyond the training set.

### Strengths
* Comprehensive experiments with solid quantitative and qualitative evaluations.
* Clear presentation and well-structured writing.
* The unified token representation for text, image, and video is conceptually elegant.
* Interesting empirical insight: the model demonstrates emergent abilities on unseen editing tasks, suggesting strong cross-modal generalization.

### Weaknesses
* While the unified sequence formulation is elegant, it primarily extends existing transformer-based multimodal modeling paradigms. The contribution lies more in engineering integration and data scaling than in introducing fundamentally new learning principles.
* The model requires ~30 GB GPU memory and ~118 seconds to edit a single 360p video on an A100 80 GB GPU. This raises serious scalability concerns for long-duration or high-resolution videos, as well as practical deployment constraints.
* The approach concatenates all modality tokens into one long sequence, resulting in $O(L^2)$ attention cost. It is unclear how EditVerse maintains efficiency with long inputs, multiple video clips, or multi-turn text–video interleaving. No efficiency analysis (FLOPs, latency, or memory growth curves) is provided.
* The use of single-layer linear projectors for modality alignment may be oversimplified. There is no ablation on alternative projector depths or modality-specific encoders to validate whether this minimal design is sufficient for cross-modal alignment.
* The video editing dataset is entirely generated via an automated pipeline using pretrained models (Grounded-SAM-2, VACE, ReCamMaster, etc.). This synthetic data may not reflect the diversity and imperfections of real-world edits, leading to overfitting on artificial editing patterns.
* Each data generation stage depends on prior model outputs, so artifacts such as inaccurate masks, poor inpainting, or unrealistic motion can cascade. The paper provides no quantitative analysis or quality control to assess data noise accumulation. 
* The paper mentions using a VLM to assign quality scores for filtering generated data, but it is unclear whether the VLM was adapted for the data filtering, how accurate its scores are, or how thresholds were chosen.

### Questions
* How is the VLM used for data scoring and filtering? Was it used off-the-shelf or fine-tuned for editing relevance? What is its reliability or correlation with human judgment?
* How does model performance and runtime scale with sequence length (e.g., multi-video input or long textual instructions)? Can efficiency be improved beyond full-sequence attention?
* Have you compared the single-layer modality projectors against deeper or non-linear alternatives?
* The reported improvements are modest; how sensitive are the evaluation metrics, and do the differences translate into perceptual gains?
* Given that the data pipeline heavily depends on pretrained vision models, how do you ensure the resulting dataset and EditVerse itself do not inherit or amplify their biases?

### Soundness
3

### Presentation
3

### Contribution
3
