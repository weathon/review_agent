# Prompt-Free Diffusion: Taking “Text” out of Text-to-Image Diffusion Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 1, 3

## Abstract
Text-to-image (T2I) research has grown explosively in the past year, owing to the large-scale pre-trained diffusion models and many emerging personalization and editing approaches. Yet, \textbf{one pain point persists: the text prompt engineering}, and searching high-quality text prompts for customized results is more art than science. Moreover, as commonly argued: ``an image is worth a thousand words" - the attempt to describe a desired image with texts often ends up being ambiguous and cannot comprehensively cover delicate visual details, hence necessitating more additional controls from the visual domain. In this paper, we take a bold step forward: taking “Text” out of a pre-trained T2I diffusion model, to reduce the burdensome prompt engineering efforts for users. Our proposed framework, \textbf{Prompt-Free Diffusion}, relies on \textbf{only visual inputs to generate new images}: it takes a reference image as ``context”, an optional image structural conditioning, and an initial noise, with absolutely no text prompt. The core architecture behind the scene is \textbf{Se}mantic Context \textbf{E}n\textbf{coder} (\textbf{SeeCoder}), substituting the commonly used CLIP-based or LLM-based text encoder.  The reusability of SeeCoder also makes it a convenient drop-in component: one can also pre-train a SeeCoder in one T2I model and reuse it for another. Through extensive experiments, Prompt-Free Diffusion is experimentally found to (i) outperform prior exemplar-based image synthesis approaches; (ii) perform on par with state-of-the-art T2I models using prompts following the best practice; and (iii) be naturally extensible to other downstream applications such as anime figure generation and virtual try-on, with promising quality. Our code and models will be open-sourced.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper propose  Semantic Context Encoder (SeeCoder) to replace the function of CLIP text encoder to achieve image-to-image variation. More specifically, they trained a transformer-based model to replace the CLIP in stable diffusion. They show good performance on image variation task.

### Strengths
1. The proposed method is intuitively sound. Training an image encoder to align with the stable diffusion to achieve image variant.
2. Also, following intuition, it works on diffusion model based on stable diffusion such as some ControlNet checkpoint, which expands the results.
3. They demonstrate good perfomance on image variant tasks.

### Weaknesses
1. The image encoder requires further training, while methods like MasaCtrl could achieve image variants without help from an extra image encoder. MasaCtrl is definitely a missing baseline.
2. Many other methods like AnyDoor: Zero-shot Object-level Image Customization achieved great performance in learning image representation. For example, AnyDoor also does virtual try-on tasks.
3. The author should do an ablation on Seecoder since this is the main structure novelty. Also, if that structure is important, they should compare it with transformer-based block. Without further investigation exhibited in the paper, I am not sure if such a specifically designed Seecoder is necessary. Is it possible that a vision transformer block with a convolution to output feature NxC also works?

### Questions
1. Why using such design in Seecoder? Have you tried other structure?
2. How does it compare with other methods?
3. How does it perform for general images when no cherry-pick is used?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces a diffusion-based, "prompt-free" image generation model. The main contribution is its design of the "Semantic Context Encoder", abbreviated as "SeeCoder", which is a "reusable" image encoder that can be plugged into trained diffusion-based image generation models. The proposed model takes no text input, but image input ("structural conditioning") that is scribble, depth, canny edge etc.

### Strengths
* The authors test their model on a wide range of image generation tasks, including conditional image generation, anime figure generation, and virtual try-on. The results appear competitive against the state of the arts.

### Weaknesses
* The paper's motivation is not convincing. The paper tries to replace text prompts with image-based "prompts", as text prompt is ambiguous and less controllable. However, in many of the presented examples, the image prompts cannot be generated without using a specialized tool - Producing depth map, canny edge, human pose skeleton all require running certain program on *existing image*. In addition, it is hard to produce a depth map or canny edge without having the target image ready, and they cannot be easily altered. Therefore, the proposed image prompt appears neither as accessible nor as controllable as text prompt.

* The paper's description on the method and the experiments are oversimplified and vague. In particular, I find little explanation on how the method achieves training-free reusability, which is a key point of this paper. I also find little information on how the model comes to understand the various types of "structural conditioning", i.e. image prompt. Many such questions remain after reading the paper and are listed below. They severely hurt the readability of this paper.

### Questions
- In Section 3.3, the "Decoder" is described as a "transformer-based network with several convolutions (*where?*), ... uses convolution to equalize channels (*whose?*), ... 6 multi-head self-attention modules". The model looks like a Transformer encoder rather than decoder - otherwise, what are the query and memory?

- Also in Section 3.3, the "Query Transformer ... started with 4 freely-learning global queries and 144 local queries", What exactly are these queries? Where "local" comes from? "The network also contains free-learned query embeddings, level embeddings, and optional 2D spatial embeddings". I could not find the definition of these in the text or in any illustration.

- As aforementioned, how the method achieves training-free reusability?

- As aforementioned, how the model is trained to understand, and hence follows the instruction of, image prompts that comes in various forms? Are the image prompts illustrated in the paper, including depth map, canny edge, scribble, also in the training data?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes an image generation method, called Prompt-Free Diffusion. It relies on only visual inputs to generate new images, and removes the text prompts from the image generation process. Also, to extract the rich visual information from input images, the authors devise a visual encoding module Semantic Context Encoder (SeeCoder).

### Strengths
I cannot find any strength from this work.

### Weaknesses
1. The motivation of this work is questionable. Why do you want to taking “text” out of text-to-image diffusion models? Why not keep compatibility with "text"? Take ControlNet as an example, you can add more conditions (such as Canny edges and depth maps) into the framework, while allowing text prompts. Eliminating text prompts will simply limit the potential abilities of the model. When you do not need a text prompt, it is fine to just set it as an empty string.
2. Only several groups of qualitative examples are presented, while NO quantitative results are given.
3. The effectiveness of the proposed Semantic Context Encoder (SeeCoder) module is not fully validated. The authors should conduct more experiments and ablation study to prove its value.
4. The authors claimed that they used a in-house T2I diffuser, which outperformed SD1.5, but did not provide any convincing evidence. This is not acceptable.

### Questions
The authors should resolve the concerns in the Weaknesses section.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper discusses the rapid growth of text-to-image (T2I) research, driven by large pre-trained models and personalization techniques. However, a challenge remains in crafting effective text prompts for customized results. To address this issue, the authors propose a novel framework called "Prompt-Free Diffusion" that eliminates the need for text prompts in T2I models. This framework relies solely on visual inputs, using a Semantic Context Encoder (SeeCoder) instead of text encoders like CLIP or LLM. Through experiments, Prompt-Free Diffusion is shown to outperform prior methods, match state-of-the-art T2I models, and extend to applications like anime figure generation and virtual try-on. The authors plan to open-source their code and models.

### Strengths
The authors have dealt with an interesting problem. The problem statement is well-defined but the novelty/contributions are poor.

### Weaknesses
- The novelty of this work is very limited. This seems to be a naive architectural of ImageVariation work. 
- The methodology is not nicely written.
- Fig. 3 diagram should have been more polished and crisper.

### Questions
What is the difference between Image-Variation and the proposed method -- besides replacing CLIP Image encoder with SeeCoder (consisting of Backbone Encoder, Decoder, and Query Transformer)?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor
