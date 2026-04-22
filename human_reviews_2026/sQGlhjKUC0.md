# To Sink or Not to Sink: Visual Information Pathways in Large Vision-Language Models

- Avg Score: 6.40
- Decision: Accept (Poster)
- Scores: 8, 2, 8, 6, 8

## Abstract
Large Vision Language Models (LVLMs) have recently emerged as powerful architectures capable of understanding and reasoning over both visual and textual information. These models typically rely on two key components: a Vision Transformer (ViT) and a Large Language Model (LLM). ViT encodes visual content into a sequence of image tokens and serves as the perceptual front-end -- the eyes of the model. In contrast, the LLM interprets these tokens to perform high-level reasoning, generates responses, and functions as the cognitive core -- the brain of the model. However, it remains unclear which visual tokens contribute most significantly to understanding and reasoning, and how effectively these signals are propagated from ViT to the LLM. While most existing works have focused on identifying attention sinks, low-semantic tokens receiving disproportionately high attention, within the LLM, we shift the focus to the vision encoder by identifying a class of high-norm visual tokens from ViT, referred to as ViT attention sinks -- a problem that has been rarely studied but is indeed very important for LVLMs. Our findings show that these ViT sinks encapsulate high-level semantic concepts from images, allowing the LLM to perform more effective understanding and reasoning. Despite their importance, these sink tokens are often overlooked in existing LVLM architectures. To explore their contribution, we present both qualitative and quantitative analyses of the information embedded in these sink tokens. We also propose both training-free and training-based approaches to better leverage how this information is interpreted by the LLM, and to what extent. By explicitly utilizing these tokens, we demonstrate substantial improvements across a range of LVLMs and visual reasoning tasks, including but not limited to mathematical problem solving, logical inference, and geometric understanding, highlighting the untapped potential of ViT attention sinks in enhancing visual reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the effect of (ViT) attention sink in vision language models. It finds that a few high norm ViT tokens act as visual sinks that receive disproportionate attention in the LLM and carry coarse scene information. Building on this, the paper proposes a training-free token reordering and a trainable MLP connector with dynamic routing to improve various downstream visual reasoning tasks. The findings about that the sink tokens are beneficial for tasks with low visual complexity and global semantics but may degrade performance
on tasks that demand localized, detail-rich visual processing is useful.

### Strengths
1 The paper is well-written and easy to follow.

2 The research topic is timely and interesting. The paper shows a clear empirical pattern linking high norm visual tokens to downstream concept semantics and reasoning capacity.

3 The proposed strategies that can be adopted with low engineering overhead.

4 The use of relevance map and decoding to world distribution to study the propagation of ViT attention sink to VLM is a sound approach.

5 The finding that the ViT sinks in VLMs encode vital high-level visual context and is actually useful for visual reasoning task is a novel finding and good to know for the community.

### Weaknesses
1. While the link between attention sink and visual semantics are evident. Why making the sink token to the front/end would make visual reasoning capacity stronger/weaker in the training-free setting, supposing the connector still struggles to project the re-ordered sink/non-sink tokens to the same embedding?
2. Currently all experiments are conducted up to 7b models, have the authors consider to test larger models like to see if the observed patterns generalize? 

- Figure A9/A10's captions are "Enter Caption"
- Font size in Figure 4/5 too small

### Questions
1. How does RoPE handle the sink token reordering at inference time, as positional embeddings of the sink token are not the same as what the VLMs see during training (i.e. without reordering)? Does moving sinks to the front/back introduce unintended positional biases?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the role of attention sink tokens originating from the Vision Transformer (ViT) in Large Vision-Language Models (LVLMs). Unlike prior work that focuses on eliminating or suppressing attention sinks as harmful, this work finds that a subset of ViT sinks propagate into the LLM and encode high-level semantic information. The authors find that ViT sinks help on global reasoning tasks but can hurt performance on fine-grained and local visual tasks. Based on this insight, they propose two methods to leverage sink tokens: a training-free “sink-to-the-front” token reordering strategy, and a training-based DIYSink framework with dynamic token selection. The experiments across multiple LVLM architectures show consistent improvements, especially on MathVista, MME, and reasoning benchmarks.

### Strengths
- **Novel insights**: The paper questions the common belief that visual attention sinks are harmful. Instead, it shows that ViTs’ sink tokens can carry useful high-level information, which can help LVLMs with global visual reasoning.
- **Thorough empirical analysis**: The authors study the phenomenon from several angles, such as attention patterns and feature norms. The experiments are thorough and support the main claims well.
- **Practical proposed methods**: “The proposed “sink-to-the-front” approach is training-free, simple to use, and still leads to improvements.

### Weaknesses
- **Limited theoretical explanation**: The paper offers little insight into the main claim—that ViT sink tokens encode global semantics. The explanation remains largely observational, with no theoretical grounding or analytical model to support why this behavior arises. More intuition or theoretical explanation would strengthen the contributions.
- **Only marginal improvements**: Although the results show gains, many improvements reported in Tables 1 and 2 are relatively small—often below 1% and sometimes within the range of natural evaluation variance. Statistical tests or error bars would help validate that the gains are meaningful rather than just noise.
- **Limited evidence on larger models**: The experiments are run on models up to 7B parameters. It is unclear if the improvements would remain for much larger models (> 7B), which may behave differently. Any experiment or discussion on this topic is valuable. 
- **Generalization beyond ViTs**: It is unclear whether the same sink behavior holds —and whether the method would work— for CNN-based or Mamba-based vision encoders, or for architectures that do not use MLP connectors.
- **Incremental contribution**: Overall, the paper provides an interesting empirical observation but lacks sufficient novelty, depth, or quantitative impact to justify acceptance.

### Questions
1. Would the same sink behavior appear in models using non-ViT vision encoders? Or when using connector architectures other than an MLP, such as cross-attention or Q-Former? 
2. For DIYSink, how sensitive is the performance to incorrect task classification (e.g. when a “global” task is misclassified as “local”)? 
3. Why does placing sink tokens at the front work better than at the end of the token sequence? Could relative positional encodings influence this outcome?
4. Have you tried the "sink-to-the-front" approach for other VLMs, such as LLaVA models: TinyLLaVA-3B or LLaVA-7B?
5. Could you also address the points mentioned in the weaknesses list?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies how large vision-language models (LVLMs) process visual information through looking into the *visual sink tokens.*

These are vision transformer (ViT) tokens with unusually large activation norms that have large attention score in the language model. The authors show that such tokens carry meaningful global scene information. They analyze how these sink tokens propagate into the language model, identify distinct hidden dimensions they activate, and demonstrate that they encode high-level semantics like object categories.

The paper also proposes two simple methods to improve LVLM performance with its findings: 
(1) a simple inference-time trick that reorders sink tokens to appear earlier, and 
(2) a training-based design called DIYSink that uses dual-MLP projectors for sink and non-sink tokens. Both improve multimodal reasoning performance on benchmarks like LLaVA eval, MME and MathVista.

### Strengths
1. It offers the first detailed analysis of how ViT sink tokens propagate into LVLMs and interact with the LLM.  

2. The paper presents clear qualitative and quantitative evidence showing that sink tokens encode global, scene-level semantics, while non-sink tokens focus on local, fine-grained details.   

3. Building on these findings, the authors propose two improvement methods: a training-free Sink-to-the-Front trick and a Dual-MLP design. Both of them are intuitive, simple-yet-effective, compatible with existing models, and lead to consistent gains.  

4. This paper provide a more in-depth understanding to model interpretability and how the vision branch work in current LVLMs.

### Weaknesses
1. The threshold used to distinguish sink and non-sink tokens is manually selected and not adaptive. This choice is heuristic and its robustness is not fully verified. The method lacks an adaptive or learnable way to determine the threshold cutoff, which limits its generalization across models and datasets.
2. The experiments use CLIP and SigLIP-based LVLMs, all built on ViT encoders. It remains unclear if similar sink behavior and method effectiveness hold in other architectures like Mamba-based vision models that also show high-norm token patterns.

### Questions
The connection between this work and previous LVLM token sink study like [1] is not discussed. The paper could better clarify how its findings extend and differ from this work.

[1] Kaduri, O., Bagon, S., & Dekel, T. What’s in the image? A deep-dive into the vision of vision language models. CVPR, 2025.

### Soundness
4

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
This paper shows that ViT “attention sinks” in LVLMs are not noise but compact carriers of global semantics that propagate into the LLM with distinct activation patterns. Leveraging this, the authors propose a training-free token reordering (sink-to-the-front) and a trainable DIYSink with dual MLP projection and dynamic selection; both yield consistent gains, especially on global reasoning tasks, while sinks can hurt highly local tasks.

### Strengths
1. The paper proposes both a training-free and a training-based approach (Sink-to-the-Front, DIYSink) to utilize sink tokens. These methods are simple yet effective, model-agnostic, and easy to integrate into existing LVLM pipelines without retraining the full model.

2. The discovery that only 1–3% of ViT tokens (the sinks) dominate semantic propagation implies a strong potential for visual token compression and efficient inference. Even though not the main focus, this insight could inspire future research on adaptive token selection and efficient LVLM decoding.

### Weaknesses
1. The efficiency implications of identifying only 1–3% ViT sink tokens are not explored; analyzing their potential for token pruning or lightweight inference would enrich the contribution.
2. The task taxonomy (global, local, mixed) depends entirely on GPT-4o annotation without human verification, raising concerns about the consistency of the main conclusion.
3. The experimental coverage is relatively limited, lacking evaluation on hallucination-oriented (e.g., POPE) and high-resolution (e.g., V*, HRBench) benchmarks, which would better demonstrate generality.

### Questions
1. Given that only about 1–3% of visual tokens are identified as ViT sinks, have the authors considered using this finding for token pruning or efficient inference? For example, could a “sink-only inference” configuration preserve comparable performance while significantly reducing FLOPs and memory cost?

2. The task taxonomy (global, local, mixed) is annotated by GPT-4o. Was any human verification or inter-annotator check performed?

3. The current evaluation mainly includes structured QA benchmarks (MME, MathVista). Could the authors consider extending experiments to hallucination-oriented benchmarks such as POPE, and high-resolution benchmarks such as V* or HRBench, to further validate the generality of the proposed method?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper mainly investigates the ViT sink phenomenon in the MLLM domain. The main contributions are: (1) providing many insightful observations about the ViT sink and its underlying mechanisms, and (2) designing applications that leverage these findings to enable MLLMs to route between non-sink and sink states under different scenarios, which is both useful and insightful.

### Strengths
* Valuable observations and insights for the multimodal large language model community.

* The overall logical flow of the paper is well-structured — from motivation, to investigation methods, to applications.

* Comprehensive experiments are conducted across different models and benchmarks.

* Some findings are quite interesting. For example: “Moreover, these high-value sink dimensions emerge only after multimodal training. In LLaVA-7B, the LLM’s original sink dimensions are {2533, 1415}, while the propagated ViT sink tokens activate dimensions {982, 2494, 3263}.”

### Weaknesses
* Some parts are confusing and not well presented or clearly clarified.

  In Section 3.2, some results show that specific head sinks may correspond to either the foreground or the background, while other results indicate that all sink tokens map to the main object of the image. The authors also conclude that ViT sinks capture coarse-grained, high-level contextual features. However, identifying a small main object in a large image seems more like a fine-grained task. I am a bit confused — could the authors provide a clearer and more unified explanation of these observations?

* Some representations could be further improved.

  In the current Figure 3, the leftmost image is not easily readable and should be better presented. A simple improvement would be to split it into two sub-images instead of keeping it as one.

### Questions
* For an image token, the positional embedding indicates its location on the image. However, repositioning these tokens breaks this property. Although the results are promising, I would like to hear the authors’ explanation of this phenomenon.

* The proposed two-step chain-of-thought process could also be redesigned to use a soft rather than hard selection mechanism. For example, one could model the probability that a given question requires holistic reasoning.

* Additionally, are the two introduced MLP parameters summed to equal the original one? If not, the comparison may not be entirely fair.

### Soundness
4

### Presentation
3

### Contribution
4
