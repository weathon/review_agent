# VisionTrim: Unified Vision Token Compression for Training-Free MLLM Acceleration

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Multimodal large language models (MLLMs) suffer from high computational costs due to excessive visual tokens, particularly in high-resolution and video-based scenarios. Existing token reduction methods typically focus on isolated pipeline components and often neglect textual alignment, leading to performance degradation. In this paper, we propose VisionTrim, a unified framework for training-free MLLM acceleration, integrating two effective plug-and-play modules: 1) the Dominant Vision Token Selection (DVTS) module, which preserves essential visual tokens via global-local view, and 2) the Text-Guided Vision Complement (TGVC) module, which facilitates context-aware token merging guided by textual cues. Extensive experiments across diverse image and video multimodal benchmarks demonstrate the performance superiority of our VisionTrim, advancing practical MLLM deployment in real-world applications. The code is available at: https://github.com/hanxunyu/VisionTrim.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes a training-free token pruning method for model acceleration. Specifically, the authors designed a two-stage plug-and-play module. First, DVTS uses a pre-trained CLIP-type encoder to prune vision tokens. Then, the TGVC module clusters and merge the pruning vision tokens for ensure semantically relevant details are preserved. Result shows that the proposed method can still perform well even when pruning 88% of the tokens.

### Strengths
The experiments are comprehensive, as the authors tested the proposed method on more than just a single model. Furthermore, the authors evaluated the use of a key-value cache. Diagrams and qualitative results are clear and intuitive. The findings of this paper are highly instructive for industrial applications, particularly the key proposal to tailor both the encoder and decoder.

### Weaknesses
1. Academic writing requires standardization. It appears that the author does not use citation formatting well. Most citations are in the form of "name et al (year)," which makes them difficult to read.

2. To my limit knowledge, token pruning or quantization methods can ensure that overall performance does not drop significantly. However, they may introduce knowledge boundary drift, meaning that answers that were previously correct may now be incorrect, and vice versa [1]. This possibility could have negative consequences in scenarios where critical questions must be answered correctly. If possible, could the authors conduct a simple analysis to see whether the proposed method leads to changes in individual case performance?

[1] Sun, Yizheng, et al. "Does Acceleration Cause Hidden Instability in Vision Language Models? Uncovering Instance-Level Divergence Through a Large-Scale Empirical Study." arXiv preprint arXiv:2503.06794 (2025).

### Questions
1. Does the proposed pulg-in module have the same weights in the encoder and decoder stages?
2. Are these components plugged into each layer?
3. Is top-k selection also mandatory during training? If so, how can author ensure it can be effectively generalized to different selected scales during inference?

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
4

### Summary
-   The paper introduces VisionTrim, a training-free framework for reducing vision tokens in MLLMs are a way of accelerating inference.
-   VisionTrim proposes two modules: Dominant Vision Token Selection (DVTS) and Text-Guided Vision Complement (TGVC).
	-   DVTS: Aims to preserve visual integrity during visual token compression by first identifying dominant visual tokens.
	-   TGVC: Utilize text prompts to cluster remaining vision tokens through CLIP embeddings. Each cluster is then merged.
-   Experimental results show that VisionTrim improves on existing methods by around 1-2% averaged across benchmarks.
-   Furthermore, there is also a significant reduction in CUDA Time, FLOPS, and KV cache memory usage.

### Strengths
-   The redundancy in vision tokens in MLLMs is definitely an area of concern.
-   Unlike existing methods, which compress vision tokens in either the vision encoder or the LLM decoder, VisionTrim proposes a unified framework that can be applied to both modules.
-   Experimental results show general improvement over existing methods by around 1-2% on average score across benchmarks, with more pronounced improvements at a lower number of vision tokens per image, showing its effectiveness at preserving visual information.
-   Paper is generally easy to read (except for some equations that lack elaborations). The diagrams and tables illustrating the methods are clear and well-drawn.

### Weaknesses
-   The paper positions itself as improving inference latency of MLLMs and reports on CUDA time for decoding, which is an odd metric to use for inference latency (most papers report end-to-end inference latency or time to first token). However, to my knowledge, clustering and pairwise similarity between tokens can be slow, given the high dimensions in both the vision token count and the embedding size. The authors did not mention whether clustering is performed on the CPU or GPU. The authors did not disclose the overall end-to-end latency; with only the CUDA time reported, I remain sceptical that this method is an effective approach at reducing inference latency.
-   The proposed VisionTrim is a unified framework that can be plugged into both the vision encoder and the LLM decoder. However, there is a lack of experiments exploring the configuration: what percentage of vision tokens are pruned in the vision encoder / LLM decoder, respectively? At what stages are they pruned? What are the performance/efficiency trade-offs in pruning in the vision encoder vs in the LLM decoder? The lack of details would make it difficult to reproduce experiments for future work.
-   Minor presentation issues: some equations, especially those in local spatial continuity, are not well explained. The intuition behind these could be further elaborated.

### Questions
-   The paper describes TGVC as a training-free plug-and-play module. However, Figure 3(a) shows a [fire] symbol over TGVC, while a [snowflake] symbol over other modules, such as the vision encoder and projector, which are not explained. It is unclear whether this was an intended design or an oversight. Could the authors clarify this? (Fig. 1 as well)

### Soundness
2

### Presentation
2

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
The paper proposed to prune the visual tokens to accelerate MLLM computation, referred as VisionTrim, including two training-free plug-and-play modules:

1)	Donimant Vision Token Selection (DVTS), which defines a global importance score that is the average attention scores w.r.t. the [CLS] token, and a local importance score that is the average affinity score within its neighboring tokens. These two importance scores are averaged by their variance to select the top K informative tokens.

2)	Text-Guided Vision Complement (TGVC), which assumes an associated textual descriptions/prompts related to the image and uses the CLIP score between the visual and text tokens to select top-R tokens as clustering centers.

These modules can be plugged in to the vision encoding stage or the LLM decoding stage. Experiments on LLaVA-NeXT-7B, Video-LLaVA-7B, and Qwen2.5-VL-7B models show that the proposed VisionTrim achieve competitive performance on multiple tasks after pruning 78% to 94% visual tokens.

### Strengths
The proposed DVTS and DVTS models are easy to implement and reproduce, which empirically show competitive performance even after pruning a large portion of visual tokens. The experiments are thorough and convincing. This may be inspiring to practical MLLM systems.

### Weaknesses
What if these is no associated textual prompts available? Or the textual clues are noisy and misleading?

Given the same image, if the textual prompts are varying in different tasks, the image has to be processed per textual prompt since the TGVC results may be quite different, right?

### Questions
How to choose top-K in DVTS  and top-R in TGVC?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The manuscript identifies two drawbacks of existing token reduction methods:

(i) they focus on only one of visual encoding or LLM decoding processes,

(ii) they neglect the role of text queries during the token pruning process.

The authors propose VisionTrim to overcome these drawbacks, where tokens are selected based on both visual and textual information (using DVTS and TGVC respctively), and the proposed modules are applied both after visual encoding and during LLM decoding. 

The proposed solutions for both drawbacks are validated in the experiment section. Both text-guided token selection and multi-stage application of token selection strategy are shown to be effective in improving the performance of low-token scenario. Finally, compared with existing approaches, VisionTrim demonstrates competitive performance.

### Strengths
1. The proposed method is well-motivated. Two key drawbacks of existing token pruning methods are identified, respectively the neglection of the role of text in visual token selection, and the application of token pruning on only one stage of visual encoding or language decoding. This provides a good review of existing methods and provides guidance to future research on token pruning for training-free mllm acceleration. 

2. The component designs are reasonable and thoroughly ablated in Table 4 and 5, including the local affinity in visual token selection, text-guided selection, and multi-stage application of token pruning, showing sound effectiveness of these components. 

3. The performance of VisionTrim is competitive against existing token pruning methods.

### Weaknesses
1. Despite the effectiveness of the proposed strategies resolving the two drawbacks of token pruning methods, in Table 5, it seems the ensemble strategy is playing a vital role in the strong performance. This is intriguing and I believe it deserves more explanation, both in terms of how the adaptive weighting mechanism is designed, as well as why it is so effective. 

2. Different components added to the final model may have different latencies, which the manuscript fails to provide. 

3. The usage of the first generated token as the natural measure of global semantic significance over all image tokens in LLM decoding stage does not make sense to me. Why is this strategy chosen?

### Questions
1. I am not sure about the term CUDA time, is it the same as latency, in terms of the time spent on processing the samples?

### Soundness
3

### Presentation
3

### Contribution
3
