# Beyond Text-to-Image: Liberating Generation with a Unified Discrete Diffusion Model

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Unified generation models aim to handle diverse tasks across modalities—such as text-to-image generation and image-to-text generation—within a single architecture and decoding paradigm. Autoregressive unified models suffer from slow inference due to sequential decoding, and non-autoregressive unified models suffer from weak generalization due to limited pretrained backbones. We introduce Muddit, a unified discrete diffusion transformer that enables fast and parallel generation across both text and image modalities. Unlike prior unified diffusion models trained from scratch, Muddit integrates strong visual priors from a pretrained text-to-image backbone with a lightweight text decoder, enabling flexible and high-quality multimodal generation under a unified architecture. Empirical results show that Muddit achieves competitive or superior performance compared to models in both quality and efficiency. This work also highlights the potential of purely discrete diffusion, when equipped with strong visual priors, as a scalable and effective backbone for unified generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents Muddit, a unified generative framework that applies discrete diffusion for both text and image modalities. The model leverages a pretrained visual prior to enable fast and parallel generation for variety of tasks. Experimental results show that Muddit achieves competitive or superior performance compared to larger autoregressive models.

### Strengths
- The paper is well-written and clearly structured, with detailed experiments across multiple multimodal tasks.
- It demonstrates that a purely discrete diffusion framework can achieve results comparable to autoregressive baselines while offering faster inference.
- The approach shows practical value by unifying image and text generation within a single framework.

### Weaknesses
Although overall quality of the paper and the proposed methods are good, I have two major concerns.
- It is not entirely clear why the unified use of discrete diffusion across modalities is fundamentally advantageous. The paper does not convincingly explain what conceptual or empirical benefits this fully discrete (with discrete diffusion) setup provides beyond architectural neatness. For example, from Table 1 and 2, models employing hybrid or alternative architectures perform better in some cases. A more detailed justification or analysis of why full discretization is beneficial would strengthen the contribution.
- The paper does not sufficiently distinguish its method from prior works that already employ discrete diffusion for both text and image. The differences seem limited to implementation choices and pre-trained priors rather than introducing a fundamentally new formulation. Clearer explanation showing how Muddit’s design overcomes the limitations of previous discrete diffusion models would make the contribution more substantial.

### Questions
See the weaknesses two

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents Muddit, which is a unified discrete diffusion model for text-to-image, image-to-text, and VQA tasks. Achieving competitive performance with existing AR or diffusion or AR/diffusion mixture models, Muddit, with a relatively small model size (1B), demonstrates the potential of a purely discrete-diffusion-based model as a scalable and effective backbone for unified generation.

### Strengths
- Muddit is built upon a visual prior rather than a language prior, which offers a new path in bridging vision and language into a unified model.
- Muddit achieved comparable or superior performance with a relatively small model size.

### Weaknesses
The performance of Muddit is comparable to or slightly better than Show-o (size 1.3B) in both Table 1: Text-to-image generation performance and Table 2: Image captioning and VQA benchmarks.

### Questions
1. I didn’t find explicit evidence in the paper to support the scalability potential of Muddit. Could the authors further elaborate on the meaning of "scalable" in the claim: "highlight Muddit's potential to serve as a scalable backbone for future multimodal systems" at the end of the conclusion section (and the abstract section)? 
2. Some of the evaluation datasets are split into specific tasks (i.e. MME, which contains multiple categories, such as Existence, Count, Position, Color, OCR, Poster, Celebrity, etc). It would be interesting to also include detailed performance under different task categories.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Muddit, a unified generative model for text and images that challenges the dominance of autoregressive (AR) frameworks. The authors argue that AR models are slow and that existing unified diffusion models are weak because they are trained from scratch, lacking strong priors. Muddit's core contribution is a "visual-first" approach. It uses a single discrete diffusion (MaskGIT-style) transformer backbone that is initialized with the weights of a powerful, pretrained text-to-image model (Meissonic). A new lightweight text decoder is then added, and the entire 1B parameter model is jointly trained (via pretraining and instruction tuning) to handle text-to-image, image captioning, and VQA tasks using a single, parallel decoding process.

### Strengths
1. The "visual-first" strategy is a novel and compelling way to build a unified model. It cleverly solves the "weak prior" problem of from-scratch diffusion models by inheriting the strong visual capabilities of Meissonic, while using a single, symmetric discrete diffusion objective for all tasks.
2. The model's primary strength is its efficiency. By replacing sequential AR decoding with parallel discrete diffusion, the paper demonstrates a significant inference speedup over prominent AR models (Fig. 13 and Tab. 7), a practical advantage for real-world applications.
3. By building on a SOTA T2I model, Muddit's image generation quality is excellent for its size. It achieves a 0.61 on GenEval, making it competitive with much larger models and even SOTA diffusion-only models like SD3 (0.62).

### Weaknesses
1. The model's key strength is also its greatest weakness. By building on a visual backbone, the model's language and reasoning capabilities are fundamentally limited. This is evident in Table 2, where Muddit's VQAv2 (68.2) scores are significantly lower than LLM-based AR models like LLaVA-Next (82.8) or discrete diffusion unified model like Show-o (69.4).
2. The model is only 1B parameters. The model capacity and visual-first training methodology bring high efficiency but make it lack far behind the leading unified discrete diffusion models like MMaDA (Yang et al., 2025) and Lumina-DiMOO (Xin et al., 2025). As a unified discrete diffusion model, the author should compare to SOTA method like MMaDA in the paper.

### Questions
1. The text decoder is a "lightweight linear head" on a visual backbone. Is this the primary bottleneck for the I2T/VQA tasks? Can the authors elaborate on why a more sophisticated text decoder (e.g., a small size Transformer) was not used?
2. Does the author believe this "visual-first" architecture can scale to 7B+ parameters and eventually close the gap on language-heavy tasks, or is it primarily intended for vision-centric tasks where language is secondary?

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
3

### Summary
This paper presents Muddit, a unified discrete diffusion model that bridges text and image generation within a single architecture. By leveraging strong visual priors from a pretrained text-to-image backbone (Meissonic) and employing fully discrete diffusion for both modalities, Muddit achieves competitive performance on text-to-image generation, image captioning, and visual question answering tasks. The work demonstrates that discrete diffusion, when equipped with appropriate pretrained knowledge, offers a scalable and efficient alternative to autoregressive approaches for unified multimodal generation.

### Strengths
1. The method design is novel. It achieves multimodal unification through fully discrete diffusion for both text and image, unlike hybrid approaches that combine autoregressive and diffusion components with separate architectures.

2. The method enables parallel token generation, achieveing speedup over AR baselines. This is good for evaluation and real-time applications.

3. The paper is well-written and easy to follow.

### Weaknesses
1. The key designs of this work (Initializing from a visual prior rather than LLM, using CLIP as text encoder rather than LLM) fundamentally constrain the model's text comprehension capabilities. This is particularly evident in handling long-form, complex text. These limitations restrict the practical applicability of the method, especially in scenarios requiring sophisticated natural language understanding or generation beyond simple captions and short answers.

2. Text generation is constrained to 77 tokens, limiting applicability for long-form generation tasks.

3. The experiment results are not very strong: there is a gap with AR baselines in several tasks (e.g. VQAv2), and with continuous diffusion models in generation task.

### Questions
Do you study how the approach scale with model size, training data, and computation resource? I am curious about whether the performance gap can be closed through scaling.

### Soundness
3

### Presentation
3

### Contribution
3
