# Reading Images Like Texts: Sequential Image Understanding in Vision-Language Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Vision-Language Models (VLMs) have demonstrated remarkable performance across a variety of real-world tasks. However, existing VLMs typically process visual information by serializing images, a method that diverges significantly from the parallel nature of human vision. Moreover, their opaque internal mechanisms hinder both deeper understanding and architectural innovation. Inspired by the dual-stream hypothesis of human vision, which distinguishes the "what" and "where" pathways, we deconstruct the visual processing in VLMs into object recognition and spatial perception for separate study. For object recognition, we convert images into text token maps and find that the model's perception of image content unfolds as a two-stage process from shallow to deep layers, beginning with attribute recognition and culminating in semantic disambiguation. For spatial perception, we theoretically derive and empirically verify the geometric structure underlying the positional representation in VLMs. Based on these findings, we introduce an instruction-agnostic token compression algorithm based on a plug-and-play visual decoder to improve decoding efficiency, and a RoPE scaling technique to enhance spatial reasoning. Through rigorous experiments, our work validates these analyses, offering a deeper understanding of VLM internals and providing clear principles for designing more capable future architectures.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper examines how vision-language models (VLMs) encode and reason with visual information within a one-dimensional token sequence. It addresses two core questions: how VLMs associate positionally discontinuous tokens representing the same object to identify its category ("what"), and how they infer 2D spatial relationships between objects ("where"). The authors use logit lens and visualizations to uncover a two-stage visual processing pattern. They theoretically and empirically analyze how RoPE-based encoders represent spatial relationships, revealing their underlying geometric structure. Building on these insights, the paper proposes two practical methods: a token compression approach that shortens image sequences with minimal performance loss, and a RoPE scaling technique that enhances spatial reasoning while maintaining overall model performance.

### Strengths
* The analysis section is thorough and compelling, offering valuable insights into how visual encoders and visual tokens are processed within VLMs. This contributes to a deeper understanding of the inner mechanisms driving object and spatial reasoning.

* The proposed applications are well grounded in the paper’s analytical findings and serve as effective validations of the authors’ observations, demonstrating both conceptual soundness and practical relevance.

* The theoretical and empirical analysis of RoPE-based spatial processing highlights a real limitation in the current VLM architecture. By identifying and addressing this limitation, the work provides a meaningful direction for future research and improvements in spatial reasoning within VLMs.

### Weaknesses
__Paper structure:__

The paper’s structure could be improved. Much of the core analysis is deferred to the appendix, making it difficult to follow the main findings from the main text alone. At the same time, several sections—particularly the Related Work—contain substantial repetition of content already covered in the Introduction. Streamlining the exposition and integrating key analyses into the main body would make the paper more cohesive and readable.

__Comparison to prior work__:

The paper lacks a sufficient comparison to other recent token compression methods, such as [1]. Since prior methods already identify salient tokens and achieve significant token reduction, it is unclear whether (1) the proposed method identifies similar or distinct subsets of tokens, and (2) whether it provides any advantage in compression efficiency or downstream task performance. A quantitative or qualitative comparison with these approaches would strengthen the paper’s claims.

__Training-free RoPE scaling experiment:__

In the training-free RoPE scaling experiment, the authors report the best results achieved by tuning the hyperparameters \alpha and p. It is unclear whether these parameters were optimized using the test set. If so, this constitutes data leakage and invalidates the experiment. The authors should clarify the procedure used to select these parameters (e.g., validation split, held-out set) and ensure that the evaluation remains unbiased.


[1] Kaduri et al., What's in the Image? A Deep-Dive into the Vision of Vision Language Models, CVPR 2025

### Questions
My only requests/questions on top of the previously mentioned points are about newer architectures and higher-dimensional RoPEs:

* Will this analysis transfer to models that use DeepStack [2], like Qwen3? 

* Can you say something similar about 4D RoPEs in videos (i.e., improving the "where" pathway for video frames, or extending it to a "when" pathway)?

[2] Meng et al., DeepStack: Deeply Stacking Visual Tokens is Surprisingly Simple and Effective for LMMs, NeurIPS 2024

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates how Vision-Language Models (VLMs) process visual information by decomposing their mechanisms into object recognition and spatial perception, inspired by the dual-stream hypothesis of human vision. The study reveals that VLMs recognize image content in two stages—progressing from low-level attribute detection to high-level semantic understanding—and uncovers the geometric structure underlying their positional  representation. Building on these insights, the authors propose a token compression algorithm and a RoPE scaling technique to enhance decoding efficiency and spatial reasoning. Overall, the work deepens understanding of VLM internals and offers principled guidance for future model design.

### Strengths
* This paper conducted comprehensive theoretical and experimental analyses. 
* The paper connects theoretical analysis with applications. Based on findings on the visual processing and spatial perception characteristics of VLMs, this paper introduces an instruction-agnostic token compression method, which reduces image sequence length during decoding, and RoPE scaling, which enhances the spatial reasoning capabilities of VLMs.

### Weaknesses
* The paper does not quantitatively compare the proposed token compression method with other existing compression techniques, such as [1] and [2], in terms of inference efficiency and training cost. The proposed method also requires additional training of the visual decoder, which may limit its application scenarios.
* In Table 1, methods 2 and 3 exhibit a more significant performance decline on TextVQA than on other tasks. What accounts for this difference? 
* There are several instances in the paper where LaTeX quotation marks are incorrectly formatted. For example, ”In which direction is A relative to B?” in line 316.

[1] Lin, Zhihang, et al. "Boosting multimodal large language models with visual tokens withdrawal for rapid inference." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 5. 2025.

[2] Zhu, Yuke, et al. "Focusllava: A coarse-to-fine approach for efficient and effective visual token compression." arXiv preprint arXiv:2411.14228 (2024).

### Questions
Please refer to the Weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper provides a deep interpretability analysis of Vision-Language Models (VLMs) through the lens of the brain’s dual-stream hypothesis, examining how models achieve both object recognition (“what”) and spatial perception (“where”). For the “what” pathway, logit lens decoding reveals a two-stage process in token representations: early detection of low-level features and later semantic refinement. For the “where” pathway, theoretical and empirical investigations of 2D RoPE demonstrate that spatial relations are geometrically encoded in attention maps. Leveraging these findings, the authors introduce (1) an instruction-agnostic token compression algorithm using run-length encoding, and (2) RoPE scaling to enhance positional signals and spatial reasoning. Experiments on LLaVA and Qwen2.5-VL validate both the insights and practical effectiveness of the proposed methods.

### Strengths
1. The paper offers a systematic and mechanistic analysis of visual processing in VLMs. The application of the logit lens to visual tokens is particularly effective, converting otherwise opaque embeddings into interpretable token maps and enabling fine-grained examination of object recognition dynamics.

2. The paper provides a rigorous theoretical analysis of how 2D RoPE encodes spatial relationships, with empirical validation via PCA, object erasure, and intervention experiments. The findings on collinearity of “left/right” and orthogonality of “left/behind” directions convincingly illustrate the geometric structure in the learned representations.

3. The paper features extensive ablation studies, comprehensive training protocols, and thoughtful handling of edge cases, such as top-1 versus top-2 token filtering for compression. Evaluation across multiple models (LLaVA, Qwen2.5-VL, InternVL) and datasets (GQA, POPE, What’s Up) further enhances the generalizability of the findings.

### Weaknesses
1. The study's focus on four basic directional relationships (“left,” “right,” “front,” “behind”) excludes more complex spatial configurations (e.g., “top-left,” “surrounding,” “partially occluded”), thereby limiting the scope of the “where” pathway analysis. 

2. The logit lens approach assumes that visual representations can be linearly decoded into semantic tokens using the language model’s unembedding matrix. However, this assumption may not hold for earlier ViT layers, whose features are not yet aligned with the final modality connector, as noted in Appendix A. Employing per-layer projection heads could offer a more robust alternative.

3. Although the compression method is instruction-agnostic, it necessitates training a visual decoder through knowledge distillation, which introduces additional engineering overhead. Furthermore, the assertion that "random selection outperforms mean pooling" is made without theoretical justification, indicating a need for more thorough investigation in future work.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
First, they investigate _how_ the vision encoder of the VLM encodes what an object is, and they do so by skipping ViT layers and using logit lens on the LLM to see what the LLM processes. They show that early layers encode high-level attributes and later layers encode specific repesentations of the objects. Then, they study at how VLMs perceive spatial relation through a theoretical and empirical analysis. Lastly, they propose two applications based on their findings: token compression and enhancing spatial reasoning.

### Strengths
* They use logit lens on the LLM to study how the Vision Encoder works: this is a particularly insightful idea that I had not previously considered. And it seems to work extremely well, especially with further validation from POPE polling.
* It is timely to study how VLMs do spatial perception as they are becoming better at it, and the paper does a sound, comprehensive study of it. While the theoretical study is limited to only two objects, I find this to be a necessary constraint.

### Weaknesses
* The segmentation map keywords seem to be chosen by the authors. This may mean that the results could be due to human-confirmation bias here, if the authors first saw the logit lens results then came up with the keywords.
* The applications and results don't seem very strong, but nevertheless serve as further empirical validation of their findings.

### Questions
* What was the method used to come up with the keyword set? If it was what I mentioned, is there some more principled way to do it?
* For the token compression application, the tokens embeddings only become text-like in the later layers of a LM. I assume you are compressing the tokens at the input to the LM (i.e. replacing or truncating the visual embeddings.) How are you compressing the tokens, or what are you replacing the visual embeddings with? I understand it to be text tokens from the logit lens. If so, have you tried merely replacing all visual embeddings with logit lens tokens? (I'm not sure if this is what "original decoding" refers to, or whether original decoding is referring to merely using the original visual embeddings)

### Soundness
3

### Presentation
3

### Contribution
4
