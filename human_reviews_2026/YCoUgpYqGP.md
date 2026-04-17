# Open-o3 Video: Grounded Video Reasoning with Explicit Spatio-Temporal Evidence

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Most video reasoning models only generate textual reasoning traces without indicating when and where key evidence appears. Recent models such as OpenAI-o3 have sparked wide interest in evidence-centered reasoning for images, yet extending this ability to videos is more challenging, as it requires joint temporal tracking and spatial localization across dynamic scenes. We introduce Open-o3 Video, a non-agent framework that integrates explicit spatio-temporal evidence into video reasoning, and carefully collect training data and design training strategies to address the aforementioned challenges. The model highlights key timestamps, objects, and bounding boxes alongside its answers, allowing reasoning to be grounded in concrete visual observations. To enable this functionality, we first curate and build two high-quality datasets, STGR-CoT-30k for SFT and STGR-RL-36k for RL, with carefully constructed temporal and spatial annotations, since most existing datasets offer either temporal spans for videos or spatial boxes on images, lacking unified spatio-temporal supervision and reasoning traces. Then, we adopt a cold-start reinforcement learning strategy with multiple specially designed rewards that jointly encourage answer accuracy, temporal alignment, and spatial precision. On V-STAR benchmark, Open-o3 Video achieves state-of-the-art performance, raising mAM by 14.4% and mLGM by 24.2% on the Qwen2.5-VL baseline. Consistent improvements are also observed on a broad range of video understanding benchmarks, such as VideoMME, WorldSense, VideoMMMU, LongVideo-Reason-eval and TVGBench. Beyond accuracy, the reasoning traces produced by Open-o3 Video also provide valuable signals for test-time scaling, enabling confidence-aware verification and improving answer reliability. The code and datasets will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents the Open-o3 Video framework, designed for grounded video reasoning with explicit spatio-temporal evidence. It tackles the challenges of temporal and spatial grounding in dynamic video scenes, introducing two new datasets—STGR-CoT-30k for supervised fine-tuning (SFT) and STGR-RL-36k for reinforcement learning (RL). The model integrates spatial and temporal information into the reasoning process, achieving state-of-the-art performance across various video reasoning benchmarks.

### Strengths
1. The integration of spatio-temporal evidence directly into the reasoning process is a major innovation, addressing a long-standing challenge in video understanding. The use of both spatial and temporal alignment as core components of the training strategy offers a clear advancement over previous methods that focused on either dimension.
2. The introduction of the STGR-CoT-30k and STGR-RL-36k datasets provides the research community with a valuable resource for training spatio-temporal models, filling a gap that has hindered the progress of grounded video reasoning.
3. Open-o3 Video achieves state-of-the-art results across multiple benchmarks, demonstrating clear performance improvements over previous models like Qwen2.5-VL and GPT-4o. This positions the model as a strong candidate for real-world applications where interpretability and robustness are key.

### Weaknesses
1. During the initial training phase, the model faces a cold start issue, especially in reinforcement learning (RL) where inaccurate temporal predictions result in near-zero spatial rewards, preventing effective learning of spatial localization in the early stages. Although adaptive temporal proximity is introduced to alleviate this, it may still impact model stability, especially in complex scenes or noisy data, limiting real-time applications.
2. Open-o3 Video relies on high-quality spatio-temporal annotation data, which is still scarce, particularly for long videos and complex scenes. The lack of annotations for small objects and long-duration scenes may affect the model's performance and generalization ability.
3. When handling longer videos and complex scenes, the model still faces issues with spatial localization and temporal alignment, especially in small object or fast-moving scenarios. Despite enhancements in the reward mechanism, the model's performance in these challenging scenarios remains limited and requires further optimization.

### Questions
1. How do the adaptive temporal proximity and temporal gating mechanisms affect the model’s ability to work with videos that are different from the dataset (e.g., videos with unusual scenes or unexpected camera movements)?
2. Can the model perform better if multi-modal data, like audio, is added? This could help the model with videos that have important non-visual information.
3. The model works well with short and medium-length videos. How will it handle very long videos (e.g., full-length films) in real-world situations, where memory and computational limits are important?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper describes a new, automatically generated dataset and training recipe for fine-tuning a vision-language model on video reasoning tasks. It also introduces a new model, based on fine-tuning Qwen2.5-VL-7B using this data and recipe. The goal of the data is to improve spatio-temporal grounding and the benefits of the approach are shown on various benchmarks (V-STAR, VideoMME, WorldSense, and others).

### Strengths
The paper shows results based on labor-intensive data engineering and training efforts, and the results on the selected benchmarks are strong. Reasoning over videos utilizing spatial, temporal and/or spatio-temporal grounding are difficult open problems.

### Weaknesses
The kind of automated data curation described in the paper seems more of an engineering effort to increase benchmark performance than an insightful scientific investigation of AI models, their limitations, etc. The data and reward generation appear quite finnicky with many hyperparameters, etc. While this not a problem per se, in line with the previous comment, it seems to come at the cost of scientific depth or insights.

### Questions
What exactly is meant by “Self-consistency Checking” (line 242), that is, how is it performed?  

(line 256) “We initialize our framework from Qwen2.5-VL-7B” seems a bit strange and overblown. Do you mean “initialize our model”? 

I would like to question the statement that the introduced model’s outperforming frontier models, like GPT-4o and Gemini, shows “significant advances in temporal and spatial grounding”. Isn’t this simply a matter of having been trained on this type of data/task? This is also shown by the fact that even the base model (Qwen2.5-VL-7B) outperforms the frontier models in some of the relevant metrics. 

How were the models from Table 2 chosen in comparison to Table 1. Are models from Table 1 (like InternVL or Video-Llama3) not applicable here? 

How does this work compare to earlier (and simpler) work on spatio-temporal grounding based on supervised learning (for example, “Look, Remember and Reason: Grounded reasoning in videos with language models”, Bhattacharyya et al. 2024, and similar work)? 

The paper uses a lot of boasting language, which I find makes it harder to follow than necessary (“We have meticulously curated”,  “Through this powerful combination of curated data and tailored training”, “our approach brings significant advances in temporal and spatial grounding”, etc.). In my mind, the paper would be stronger and easier to read if it simply stated facts.  

How were hyperparameters like sigma, reward weightings, data mixes, etc., determined?

### Soundness
3

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
4

### Summary
This paper introduces Open-o3 Video, a framework designed to enhance video reasoning in large multimodal models by grounding textual answers in explicit spatio-temporal evidence. Addressing the limitations of prior models that only produce text-based rationales, Open-o3 Video generates answers alongside key timestamps and bounding boxes that pinpoint the visual evidence supporting its conclusions. The authors identify two primary challenges: the absence of high-quality datasets with joint spatio-temporal supervision and the difficulty of training a model to perform simultaneous temporal tracking and spatial localization. Empirically, Open-o3 Video, built upon the Qwen2.5-VL-7B model, achieves state-of-the-art results on the V-STAR benchmark, significantly outperforming its base model by +14.4% mAM and +24.2% mLGM, and also surpassing proprietary models like GPT-4o. The model also shows consistent improvements on other video understanding benchmarks such as VideoMME, WorldSense, and TVGBench.

### Strengths
1. The paper addresses a critical and timely problem in multimodal AI: moving beyond opaque, text-only reasoning to verifiable, evidence-grounded reasoning. Extending the "thinking with images" paradigm to the video domain is a non-trivial and important research direction. The proposed reward mechanisms—adaptive temporal proximity and temporal gating—are novel, well-motivated, and directly target core challenges in learning spatio-temporal grounding.

2. The paper is methodologically sound. The two-stage training approach is well-justified, and the choice of GSPO for sequence-level optimization is appropriate for the complex, long-form output. The ablation studies in Section 5.2 are particularly strong; they clearly isolate and validate the contribution of each component: the SFT+RL strategy, the specific reward designs (Ada. and Gat.), and the high-quality annotated data.

3. The creation and planned release of the STGR-CoT-30k and STGR-RL-36k datasets represent a major contribution to the research community. The detailed description of the data annotation pipeline (Figure 2, Appendix A.3) provides transparency.

### Weaknesses
1. The model processes videos by sampling 16 frames uniformly, with the addition of key frames during training. This is a sparse representation that may not be sufficient for very long videos or for scenarios where critical evidence is extremely brief and could be missed by the sampling. While the authors acknowledge this limitation, its practical impact on more complex, real-world videos remains an open question.

2. The model is trained to generate a highly specific output format (<obj>...</obj><box>...</box>at<t>...</t>s). This type of instruction-following can sometimes be brittle. The paper does not discuss how the model behaves if it fails to adhere to this format or how such failures are handled during evaluation and RL training

3. Seems the authors missed some existing video reasoning benchmarks, e.g. LongVideo-Reason-eval [1]

[1] Scaling RL to Long Videos. Neurips 2025. https://huggingface.co/datasets/LongVideo-Reason/longvideo_eval_videos

### Questions
1. The strategy of uniformly sampling 16 frames seems sparse for longer videos. Could you elaborate on how adding annotated key frames during training helps the model generalize to finding unseen and potentially brief key moments at test time? Have you experimented with alternative, more dynamic frame selection strategies at inference?

### Soundness
3

### Presentation
3

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
The paper aims to introduce a unified framework for grounded video reasoning that integrates explicit spatio-temporal evidence through timestamped frames and localized bounding boxes. The paper proposed the STGR-CoT-30k, STGR-RL-36k datasets and uses a with a two-stage training strategy using supervised fine-tuning and GSPO-based reinforcement learning. Experiments on V-STAR show promising performance.

### Strengths
* The proposed STGR-CoT-30k and STGR-RL-36k datasets for supervised fine-tuning and RL, are novel and interesting, would add value to the community.
* The proposed approach leads to improved performance on the V-STAR benchmark.
* The paper is well written and easy to understand.

### Weaknesses
• Novelty: The dataset generation and training scheme – supervised fine-tuning followed by GRPO based RL training is largely based on prior work, e.g., the Deepseek family of models and more recently by papers such as "Scaling RL to Long Videos, NeurIPS 2025". The novelty of the training scheme should be discussed in more detail.

* Performance: The proposed approach leads to limited gains on the VideoMME or VideoMMMU benchmarks (~1%) vs the base Qwen model. Moreover, state of the art models such VideoLLAMA3 obtains 66.2 % accuracy.

* Limited evaluation datasets: The paper only evaluates on the V-STAR benchmark. For fair comparison to prior work the paper should include evaluation on additional benchmarks that focus specially on grounding such as "STAR: A Benchmark for Situated Reasoning in Real-World Videos, NeurIPS 2021".

* The paper should discuss prior work on grounding to fine-grained visual information in videos such as: "Look, Remember and Reason: Grounded reasoning in videos with language models, ICLR 2024"; "Fine-grained Spatiotemporal Grounding on Egocentric Videos, ICCV 2025".

• The model seems to rely on a stable and low frame rate, it is unclear if the model can deal with variable or high frame rates.

### Questions
* The performance on benchmarks such as VideoMME or VideoMMMU should be discussed in more detail.
* The paper should add additional evaluation on benchmarks such as STAR.
* The paper should include a more extensive discussion on prior works on video grounding.

### Soundness
3

### Presentation
3

### Contribution
2
