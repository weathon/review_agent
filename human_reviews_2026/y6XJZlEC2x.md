# Mixture of Contexts for Long Video Generation

- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
Long-context video generation is fundamentally a memory problem: models must retain and retrieve salient events across long range without collapsing or drifting. However, scaling diffusion transformers (DiTs) to generate long-context videos is fundamentally limited by the quadratic cost of self-attention, which makes memory and computation intractable and difficult to optimize for long sequences. We recast long-context video generation as an internal information retrieval task and propose a simple, learnable sparse attention routing module, Mixture of Contexts (MoC), as an effective long-term memory retrieval engine. In MoC, each query dynamically selects a few informative chunks plus mandatory anchors (caption, local windows) to attend to, with causal routing that prevents loop closures. As we scale the data and gradually sparsify the routing, the model allocates compute to salient history, preserving identities, actions, and scenes over minutes of content.
Efficiency follows as a byproduct of retrieval (near-linear scaling), which enables practical training and synthesis, and the emergence of memory and consistency at the scale of minutes. Project Page: https://primecai.github.io/moc/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a sparse attention-based method to alleviate the excessive computational cost in long-context learning for long video generation. The main experimental results are conducted in a multi-shot setting.

### Strengths
The proposed method is highly reasonable, and the quantitative results demonstrate its effectiveness.

### Weaknesses
While the quantitative results in the multi-shot setting show that MoC performs very well, it would be helpful to provide more specific qualitative results to demonstrate the consistency of multi-shot long video generation. The figures provided (e.g., Figure 3 and Figure 5) seem insufficient, as they do not clearly showcase why these results are consistent. The lack of detailed qualitative examples, particularly those showing poor consistency for comparison, is a significant drawback of the paper.

### Questions
I understand that this paper mainly focuses on multi-shot long video generation, but the appendix also provides a single-shot experiment. Although the single-shot video is only 8 seconds long and cannot be considered a long video (it should be at least several tens of seconds), I’m curious about how MoC performs on longer single-shot videos. I suspect that the sparsity of multi-shot long videos and single-shot long videos might differ, which could lead to some interesting conclusions.

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
This paper presents a method for multi-shot video generation. The core idea is to segment long videos into chunks using a "mix-of-context" strategy and process them with a parameter-free router. By training on multi-shot datasets, the proposed model successfully generates multi-shot videos while effectively reducing the computational overhead associated with long video synthesis.

Experiments show that the proposed approach performs well for long video generation.

### Strengths
- The proposed method is simple, elegant, and effective. The qualitative results and demonstrated examples are compelling.
- The paper is well-written and easy to follow.
- The visual results are also great.

### Weaknesses
- Lack of Transparency in Data Curation: The success of this method seems heavily reliant on the quality of the training data. However, the paper lacks crucial details about the dataset. The authors should clarify:

  - What was the procedure for annotating captions?
  - How was the multi-shot (multi-scene) data curated and structured?
  - What is the approximate scale (e.g., number of videos, hours) of the dataset?

This lack of transparency is a significant concern for reproducibility and limits the paper's potential impact on future research. More details regarding the dataset used in the main experiments are essential.

- Performance on Long-form Generation: It is a known issue that autoregressive models often suffer from a decline in quality as the generated video length increases. Does the proposed chunk-based approach effectively mitigate or resolve this degradation problem?

- Multi-Character Consistency: Can the proposed model maintain the identity and appearance consistency of multiple characters across different shots or scenes?

### Questions
The dataset details should be clarified.

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
This paper presents LongLive, a set of techniques for converting existing video diffusion transformers to enable extremely long video generation, possibly spanning multiple scenes. The main idea is to reduce the quadratic computational cost of self-attention. Unlike conventional approaches that learn a global sparse attention structure, this paper proposes to adaptively select a small number of relevant clips for each query token, based on their top-k similarities with previous video clips. Specifically, the model computes the cosine similarity between the current query token and the average-pooled keys of each previous clip to determine which contexts to attend to. To regularize the routing process and prevent context drop-off or drop-in, the authors randomly remove or add clips regardless of their similarity values. They also introduce causal routing to avoid loop closure and incorporate additional efficiency techniques, such as attention sink and FlashAttention. As a result, the proposed method can efficiently generate long, multi-scene videos while maintaining temporal and semantic consistency.

### Strengths
- The paper is well-motivated, as the current video diffusion models have a computational bottleneck to generate extremely long videos due to the quadratic cost of self-attention. 
- Most of the components (especially for routing) are technically sound, interesting, and novel. I enjoyed reading the paper.
- While I believe the presentation can be improved more (see weaknesses), but overall the paper is generally well-written and easy to follow.

### Weaknesses
- There is no quantitative analysis of GPU/TPU memory consumption.
Including such analysis would strengthen the efficiency claims of the paper.
- The presentation could be improved, as some paragraphs are too long to follow.
The authors may consider splitting some of them into multiple paragraphs—for example, separating the discussion of context drop-off and drop-in into two distinct sections.
- While the supplementary material includes video examples, I suggest adding more qualitative examples directly in the PDF (e.g., in the Appendix) to make the results more accessible to readers.
- Suggestion: I believe the effectiveness of global pooling arises because the diffusion transformer learns meaningful internal representations; therefore, global average pooling can capture global semantics within the model. The authors might clarify this in the revision. Currently, referencing CLIP may give the impression of a logical flaw, since CLIP is specifically trained to maximize cosine similarity with text embeddings.
In this context, the authors could mention DDAE [1] or other related works to strengthen the justification.
- Suggestion: Some related works on long video generation are missing, such as TECO [2], MALT [3] and NUWA-XL [4].

[1] Denoising Diffusion Autoencoders are Unified Self-supervised Learners, ICCV 2023  
[2] Temporally Consistent Transformers for Video Generation, ICML 2023  
[3] MALT Diffusion: Memory-Augmented Latent Transformers for Any-Length Video Generation, CVPRW 2025  
[4] NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation, 2023

### Questions
- Is this strategy applied in a layer-wise manner, or applied globally by selecting a specific layer for measuring cosine similarity, and then choosing clips in all video diffusion transformer layers?

### Soundness
3

### Presentation
3

### Contribution
3
