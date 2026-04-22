# NExT-OMNI: Towards Any-to-Any Omnimodal Foundation Models with Discrete Flow Matching

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 8

## Abstract
Next-generation multimodal foundation models capable of any-to-any cross-modal generation and multi-turn interaction will serve as core components of artificial general intelligence systems, playing a pivotal role in human-machine interaction. However, most existing multimodal models remain constrained by autoregressive architectures, whose inherent limitations prevent a balanced integration of understanding and generation capabilities. Although hybrid and decoupling strategies have been explored to address these tasks within unified frameworks separately, their redundant, non-integrated designs limit their applicability to broader scenarios, such as cross-modal retrieval. In this work, we introduce NExT-OMNI, an open-source omnimodal foundation model that achieves unified modeling through discrete flow paradigms. By leveraging metric-induced probability paths and kinetic optimal velocities, NExT-OMNI natively supports any-to-any understanding and generation with enhanced response efficiency, while enabling broader application scenarios through concise unified representations rather than task-decoupled designs. Trained on large-scale interleaved text, image, video, and audio data, NExT-OMNI delivers competitive performance on multimodal understanding and generation benchmarks, while outperforming prior unified models in multi-turn multimodal interaction and cross-modal retrieval, highlighting its architectural advantages as a next-generation multimodal foundation model. The code is available at https://github.com/ritzz-ai/Next-OMNI.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In all, NExT-OMNI is a well-executed application model that demonstrates strong engineering by effectively integrating Discrete Flow Matching into a 7B-scale, any-to-any multimodal system., But its innovation is combinatorial rather than fundamental and lacks analysis of its slight underperformance in speech tasks and doesn't prove DFM's efficacy on strictly sequential tasks. Moreover, the application of NExT-OMNI on large scale model remains to verify.

### Strengths
1.NExT-OMNI first trains a model with "any to any" on the 7B scale and achieves SOTA performance on multiple tasks, expanding the application of Discrete Flow Matching(DFM).

2.It does realizes the full integration of DFM, unified discrete representation, lightweight multi-head, parallel decoding, and multi-turn multimodal instruction tuning methods, and provides design details and a series of data synthesis. Relevant ablation experiments prove the effectiveness of each part of the model.

3.For the first time, the paper combines block size padding with dynamically adjusting the preset generation length in steps of block size based on <EOS> confidence, which not only saves computing power but also prevents truncation and loss of generated content.

### Weaknesses
1.The paper is a combinatorial innovation at the methodological level, its contributions lie in engineering implementation and scaling up to a 7B full-modal model, without proposing new ideas for addressing the problem of balancing model understanding and generation tasks.

2.In Table 2, NExT-OMNI seems to be slightly inferior in speech-to-speech tasks comparing with auto-regressive models, but no further explanation or analysis of it.

3.The paper does not experimentally demonstrate how DFM, which inherently lacks such an inductive bias, can match or surpass auto regressive models on tasks that strictly require sequential adherence such as code generation and step-by-step reasoning.

4.In Figure 22, the paper only qualitatively illustrates the model’s ability to “think with images”, but it does not systematically evaluate how effectively the model leverages its generation capability for complex reasoning tasks. For example, there is no assessment of whether the model can assist mathematical or logical reasoning by generating images, nor are quantitative metrics such as MMU[1], MMBench[2], or ScienceQA[3].

**References:**

[1] Yue X., Ni Y., Zhang K. et al. “MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI.” arXiv:2311.16502, 2024.

[2] Liu Y., Zhang H., Chen J. et al. “MMBench: Is Your Multi-modal Model an All-around Player?” arXiv:2307.06281, 2023.

[3] Lu P., Mishra S., Xia T. et al. “Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering.” NeurIPS 2022.

### Questions
Please refer to weaknesses.

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
3

### Summary
This paper proposes a DLLM-like model that aims to build a unified, end-to-end architecture for multimodal understanding and generation. The authors conduct detailed experiments and evaluate the model across three modalities—text, audio, and image—demonstrating its overall effectiveness and strong performance.

### Strengths
- The proposed architecture is highly innovative. To the best of my knowledge, this is the first unified model designed for discrete-unit generation and understanding across all modalities.

- In terms of performance, the model outperforms several existing approaches (although many of the baselines are not state-of-the-art).

### Weaknesses
- The comparison with baselines is not sufficiently comprehensive. Important recent models such as Qwen2.5-Omni and Qwen3-Omni, which outperform Next-Omni on OmniBench, should be included for a fair evaluation. Similarly, XCodec2 represents the most recent state-of-the-art in audio tokenization and should be considered as a baseline.

- For audio-related tasks, perceptual audio quality is critical. It would be very helpful if the authors could provide qualitative examples or audio case studies to allow readers to better assess the perceptual quality of the generated audio. I look forward to seeing such demonstrations in future revisions.

### Questions
None

### Soundness
2

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
The paper presents NExT-OMNI, which is an omnimodal foundation model built on discrete flow matching (DFM), targeting unified multimodal understanding and generation. It addresses the limitations of autoregressive (AR) approaches, which suffer from inherent conflicts between understanding and generation tasks as well as slowed inference due to decoupled designs. Unlike these, DFM introduces parallel information processing with bidirectional integration, enabling both efficient any-to-any cross-modal generation and enhanced multimodal understanding. Besides, NExT-OMNI achieves precise cross-modal retrieval and robust multi-turn multimodal interactions, surpassing prior AR-based and hybrid approaches. The model consistently delivers competitive or superior performance with reduced latency across standard multimodal benchmarks.

### Strengths
1. NExT-OMNI introduces a novel unified modeling approach using DFM to integrate multimodal understanding and generation tasks. Unlike previous work, which requires additional diffusion decoders and increases parameter size, NExT-OMNI achieves compactness by reducing the need for extra modules. Its bidirectional feature fusion design effectively enhances cross-modal interactions, enabling better feature integration across modalities. 

2. NExT-OMNI introduces dynamic length generation optimization. By using the EOS token's confidence scores, the model dynamically adjusts text generation lengths. This can effectively improve multimodal understanding and generate more natural outputs in text-based tasks.

3. NExT-OMNI integrates an adaptive caching mechanism to leverage the parallel decoding strengths of DFM, leading to a 1.2× increase in inference speed compared to AR architectures. 

4. The model is tested not only on single-turn tasks such as text-to-image and text-to-audio generation but also on real-world scenarios requiring multi-turn interactions. It demonstrates clear improvements in unified understanding and generation capabilities, particularly in dynamic multi-turn exchanges.

### Weaknesses
1. While the discrete DFM effectively unifies understanding and generation tasks, its reliance on discrete representations, as opposed to continuous flow-based approaches, may lead to some performance degradation in generation tasks due to information loss during the discretization process.
2. After completing the encoder warmup phase, the model requires joint optimization with reconstruction losses during subsequent multi-stage training. This additional reconstruction loss, compared to exclusively optimizing with cross-entropy loss, can inevitably lower training efficiency and lengthen the overall training process.
3. Recent studies [1] show that discrete DFM and diffusion-based models demand more computational resources during training compared to autoregressive architectures. This is attributed to the complexity of learning difficult tasks within the discrete modeling framework.

[1] Training Optimal Large Diffusion Language Models.

### Questions
1. Lumina-Dimoo [2] also adopts a similar discrete diffusion modeling approach. Why was there no comparison with it? The authors should further clarify this issue.

2. While unified modeling for both generation and understanding has become a trend, it seems that separating these tasks and modeling them independently yields better performance. Why is it still necessary to persist with a unified model design?

3. The appendix demonstrates that increasing the number of sub-token tables in the warmup encoder improves reconstruction performance. Why is there still a need to aggregate these into high-dimensional features for the backbone, and is there experimental evidence to support this design choice?

   

[2] An Omni Diffusion Large Language Model for Multi-Modal Generation and Understanding.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces NExT-OMNI, an open-source omnimodal foundation model that achieves unified modeling of understanding, generation, and retrieval across text, images, video, and audio using discrete flow matching (DFM) techniques. Unlike existing autoregressive (AR) models that struggle with conflicts between understanding and generation tasks, or hybrid architectures that rely on task-specific decoupling, NExT-OMNI employs a streamlined unified architecture. The model leverages metric-induced probability paths and kinetic optimal velocities to enable bidirectional information integration, achieving faster inference through parallel decoding. Useful strategies include reconstruction-enhanced unified representations, dynamic length generation strategies, and vanilla adaptive caching. Experimental results demonstrate competitive performance on standard benchmarks while excelling at multi-turn multimodal interaction and cross-modal retrieval tasks.

### Strengths
- Unified Architecture: Successfully demonstrates that a single DFM-based architecture can handle understanding, generation, and retrieval, challenging the dominance of the AR-based paradigm.

- Comprehensive Evaluation: Extensive experiments across 7+ benchmarks covering all major modalities and tasks, with careful ablation studies validating design choices.

- Useful strategies: Provides reconstruction-enhanced unified representations training, dynamic length generation strategies, and vanilla adaptive caching, which are quite helpful to the development of the community.

- Inspiring results: Effectively demonstrates how unified representations enable superior retrieval performance compared to decoupled architectures.

- Open Source Commitment: Authors promise to release code, models, and training protocols, which would significantly benefit the community.

### Weaknesses
- I see no other major weakness. Thanks for the authors' hard work.

### Questions
- It would be better if the authors could provide detailed training costs (e.g., GPU hours) for the proposed model.

### Soundness
4

### Presentation
4

### Contribution
4
