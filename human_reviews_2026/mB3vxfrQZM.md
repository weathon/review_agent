# ThinkMorph: Emergent Properties in Multimodal Interleaved Chain-of-Thought Reasoning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Multimodal reasoning requires iterative coordination between language and vision, yet it remains unclear what constitutes a meaningful interleaved chain of thought. We posit that text and image thoughts should function as complementary, rather than isomorphic, modalities that mutually advance reasoning. Guided by this principle, we build ThinkMorph, a unified model fine-tuned on $\sim$24K high-quality interleaved reasoning traces spanning tasks with varying visual engagement. ThinkMorph learns to generate progressive text–image reasoning steps that concretely manipulate visual content while maintaining coherent verbal logic. It delivers large gains on vision-centric benchmarks (averaging 34.7% over the base model) and generalizes to out-of-domain tasks, matching or surpassing larger and proprietary VLMs. Beyond performance, ThinkMorph exhibits emergent multimodal intelligence, including unseen visual manipulation skills, adaptive switching between reasoning modes, and better test-time scaling through diversified multimodal thoughts.
These findings suggest promising directions for characterizing the emergent capabilities of unified models for multimodal reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes ThinkMorph, a unified multimodal reasoning model designed to perform interleaved reasoning across language and vision. The authors construct a high-quality data pipeline to generate interleaved reasoning traces. Through this training strategy, ThinkMorph reportedly achieves improvements on several multimodal reasoning benchmarks.

### Strengths
The paper constructs a high-quality interleaved reasoning dataset, which serves as a valuable resource for training and analyzing unified multimodal reasoning models. It also provides relatively complete implementation and reproducibility details, allowing others to replicate the experiments.

### Weaknesses
1. **Limited qualitative showcases.**
   The paper provides too few qualitative examples to substantiate its claims—particularly for the highlighted *emergent properties*. Demonstrations of abilities such as multi-bounding box generation, zoom-in, and crop operations are only shown in a single example (Figure 1), which is insufficient to convincingly establish the claimed emergent behaviors.

2. **Lack of detailed experimental settings.**
   The paper lacks clarity regarding the differences between experimental settings, especially between *visual reasoning* and *interleaved reasoning* as reported in Table 2. 
3. **Unfair baseline comparison.**
   In line 344, the authors claim that “interleaved training consistently pushes unified models far beyond their previous limitations.” However, the comparison baseline appears to be the model that was **not** fine-tuned. A fairer comparison would involve a baseline fine-tuned on the same task using *text-only reasoning data*, so that the contribution of interleaved training can be more precisely isolated and validated.

4. **Ambiguity in adaptive modality selection.**
   The paper claims that ThinkMorph exhibits “adaptive modality selection,” where the model dynamically chooses between text-only, image-only, or interleaved reasoning depending on the task. However, there is no detailed discussion of image-only reasoning in the experiments, and the relationship between *image-only reasoning* and *visual reasoning* (as shown in Table 2) remains unclear, making this claim confusing and insufficiently supported.

5. **Questionable baseline results.**
   The reported GPT-4o result on SAT (28) in Table 3 is significantly lower than in the original paper, which raises concerns about the experimental setup or evaluation protocol. The discrepancy should be discussed and justified.

6. **Typos.**
There are several minor typographical and formatting errors in the paper. For example, in Table 5, “Reaonsing” should be corrected to “Reasoning.”

### Questions
1. Could the authors clarify how the *text reasoning*, *visual reasoning*, and *interleaved reasoning* models in Table 2 were obtained? Specifically, how were these variants fine-tuned within the Bagel framework, especially the details of the training data. And how do they relate to *ThinkMorph* as reported in Table 3?

2. In line 397, the authors mention several cases where the model “shifts to text-only reasoning.” Could the authors provide concrete examples or visualizations of these cases, and elaborate on why the model makes this reasoning shift?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces ThinkMorph, a unified multimodal model designed to perform synergistic, interleaved Chain-of-Thought (CoT) reasoning that alternates between text and image generation. The core premise is that for vision-centric tasks (e.g., spatial reasoning, visual search), pure textual CoT is insufficient, and a dynamic interplay between language and vision can more effectively mimic human "think-and-sketch" reasoning.

### Strengths
- **Novel finding on interleaved reasoning**: A key contribution of the paper is the demonstration that training on interleaved text–image reasoning data enables the emergence of visual manipulation capabilities. This suggests that the model acquires a general-purpose mechanism for using visual representations as intermediate reasoning steps, rather than merely memorizing task-specific mappings.

- **Synthetic data curation methodology**: The paper’s effectiveness largely stems from its carefully designed data curation pipeline. A central insight is the construction of reasoning traces in which textual and visual components are complementary and non-isomorphic—an approach that offers a reproducible and valuable strategy for future work in multimodal reasoning.

- **Good empirical performance**: ThinkMorph achieves significant improvements over its base model and exhibits robust out-of-domain generalization. It attains competitive or superior performance compared to substantially larger models on several challenging multimodal benchmarks, including SAT and MMVP.

### Weaknesses
- **"Emergent Property 1" claim is overstated.** In my personal experience, the original Bagel model was already capable of zooming in/out and inpainting images when explicitly prompted—for example, given a prompt like "find the banana in the image," it could crop out the banana region from the input image. This capability appears to pre-exist in Bagel and is not necessarily a result of ThinkMorph’s training. Therefore, the claim of Emergent Property 1 lacks sufficient support, as the described behavior is already present in the base model.

- **"Autonomous Mode Switching" claim is conflated:** The most interesting claim is that a model trained exclusively on interleaved data can emerge a text-only reasoning mode. However, this is based on a "rare (2-5%)" observation. To analyze this "mode switching" further, the paper then pivots in Section 3.6 to a hybrid-trained model, which was explicitly trained on different modes. This is a much less interesting result and conflates the "emergence" claim with a simple "multi-task" training setup.

- **Scope of evaluated tasks:** While the four chosen tasks are well-justified, they represent a specific class of vision-centric, puzzle-like problems. The generalizability of this approach to more open-ended, creative, or dialog-based multimodal tasks remains an open question.

### Questions
- **Q1:** In my understanding, interleaved reasoning is more costly than single-modality reasoning. In Tab. 2, why are VSP-Main and VisPuzzle reported only in non-think mode for Bagel, whereas all think modes are reported for ThinkMorph?
- **Q2:** In Sec. 3.6, it says:
> This property extends our observation of autonomous mode switching (Emergent Property 2) to a
more dynamic setting. When generating multiple reasoning chains (e.g., N=8), a subset of samples
switch between text-only and interleaved modes as illustrated in Figure 2. Specifically, the modality
distribution changes as the number of candidate solutions increases. The proportion of pure textbased reasoning chains decreases from 18.8% at N=1 and N=2 to 15.2% at N=8. In parallel, the
contribution from interleaved chains—the dominant modality—increases from 81.2% to 84.8% over
the same range. This dynamic shift in modality distribution is directly correlated with an increase
in overall task accuracy, which improves from 51.3% at N=1 to 58.6% at N=8, as the modeling in
§ 2.3 with modality diversity.

Although the proportion of text-based reasoning decreases while the proportion of interleaved reasoning increases with more rollouts, where do the final accuracy gains primarily originate—from text-based or interleaved reasoning?
- **Q3:** The paper states that the interleaved-only model switches to text-only mode in 2-5% of samples, yielding "strikingly high accuracy, up to 90%". This is ambiguous. Does this mean 90% of that 2-5% subset of samples were correct? Or that 90% was the highest accuracy achieved on any sample in that subset? More importantly, why does the analysis in Sec 3.6 pivot to a hybrid-trained model? This feels like it sidesteps the more interesting emergence claim. Can you clarify the "emergence" of mode switching in the interleaved-only model, as this is a key claim?
- **Q4:** The Test-Time Scaling results are impressive. However, generating N interleaved chains, each containing images, is computationally expensive. Have you analyzed the performance vs. cost curve?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
ThinkMorph introduces a unified multimodal reasoning framework that enables dynamic interleaving of textual and visual generation, allowing language and vision to mutually advance reasoning rather than operate in parallel. It eliminates reliance on external tools by treating image generation as predictable tokens within an autoregressive sequence, using a VAE and diffusion model to internally construct visual edits. Training is based on 24,991 carefully curated interleaved reasoning traces across four vision-centric tasks—jigsaw assembly, spatial navigation, visual search, and chart refocus—forcing the model to alternately reason with language and produce visual interventions. The approach achieves substantial performance gains on vision-centered benchmarks, including a 38.75 percent improvement on jigsaw assembly and competitive results on MMVP and SAT, matching or exceeding large proprietary models. Crucially, the model exhibits emergent capabilities not present in training data, such as unseen visual manipulation—including zoom-in and image inpainting—and autonomous switching between text-only, image-only, and interleaved reasoning modes. These abilities arise from the richer exploration space enabled by interleaved training, allowing test-time scaling through heterogeneous sampling across reasoning modalities. Unlike prior methods relying on tool augmentation or implicit visual tokens, ThinkMorph establishes a unified representation where vision is not a cue but an active component of reasoning. The work shifts the paradigm from passive perception to active visual thinking, demonstrating that integrated generation and understanding can cultivate higher-level multimodal intelligence. Although computationally intensive, it establishes a foundational pathway for building unified models capable of adaptive, human-like multimodal reasoning.

### Strengths
1. ThinkMorph pioneeringly introduces the "interleaved chain-of-thought" multimodal paradigm, enabling deep, dynamic, and complementary synergy between language and vision during reasoning. It moves beyond treating vision as passive input or an external tool, allowing the model to autonomously generate and modify visual content within its reasoning sequence, achieving intrinsic visual manipulation. This unique mechanism also fosters the emergence of novel visual operations and intelligent modality adaptation behaviors not previously observed.

2. This work achieves unified representation and seamless switching between language and vision within a single autoregressive Transformer architecture. By incorporating special tokens like <image start> and <image end>, the model can autonomously trigger internal VAE and diffusion models for image generation and re-insertion. Coupled with a negative log-likelihood loss for text and a mean squared error loss for images, ThinkMorph ensures both the accuracy of linguistic reasoning and the quality of visual generation, thereby achieving effective integration of multimodal signals.

3. This paper excels in its presentation, clearly articulating the limitations of existing models and grounding its core motivation in human "think-and-sketch" cognitive patterns. Through rich illustrations and reasoning examples, it provides intuitive demonstrations of how interleaved reasoning functions and the visual capabilities that emerge from the model. Furthermore, the paper systematically compares its work with tool-augmented, preliminary interleaving, and implicit visual token approaches, highlighting its distinct advantages.

4. ThinkMorph's efficacy is thoroughly validated through experiments on four categories of custom-built, vision-centric tasks. These datasets were specifically designed to embody the characteristics of interleaved reasoning. The experimental results not only demonstrate significant performance improvements across multiple tasks via quantitative metrics but also present qualitative analyses of generated interleaved reasoning chains, visual operation instances, and adaptive modality selection, providing strong evidence for the method's effectiveness.

### Weaknesses
1. The emergent visual manipulation capabilities, such as image inpainting and zoom-in, are presented as indicators of higher-level intelligence, yet their geometric and semantic consistency with the original image structure remains unvalidated. It is unclear whether the generated regions are aligned with the true physical structure or whether the model may produce visually plausible but physically incorrect details in the absence of explicit scale or spatial constraints.

2. Autonomous mode switching occurs in only a small fraction of samples, approximately 2 to 5 percent, and is interpreted as an adaptive behavioral trait. However, the underlying triggering mechanisms are not quantitatively analyzed. It remains unresolved whether this behavior is driven by semantic properties of the task, such as the presence of color, quantity, or relational queries, or whether it arises from latent biases in the training data distribution.

3. The computational overhead of interleaved reasoning during inference is not addressed. Each instance of image generation requires execution of VAE encoding, diffusion-based denoising, and re-encoding into clean visual tokens, resulting in substantial latency and limiting scalability.

### Questions
This paper highlights ThinkMorph's interleaved textual and visual reasoning, demonstrating significant performance gains on vision-centric tasks. However, generating and re-integrating image content into the LLM's token sequence incurs substantial computational overhead due to VAE encoding, diffusion denoising, and increased sequence length. Are these visual reasoning scenarios fundamentally intractable for pure text-based models, thus strictly necessitating this high-resource graphical reasoning for effective problem-solving?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors present ThinkMorph, a model for interleaved multimodal reasoning with text and images. They train a joint image-text generative MLLM on a few particular tasks with data generated for those tasks, such as visual search and jigsaw assembly. They create a data generation pipeline for each of these, which create visual tokens and text tokens specific to the task after defining visual manipulations for each tasks. Training on these results in improved performance for the MLLM compared to just the text or just the visual tokens. They also show some improvements in test-time scaling, and note some interesting emergent properties on out-of-domain data.

### Strengths
- The problem is well-motivated, mixing reasoning across modalities and aiming to identify where it performs beyond unimodal reasoning and where it may not.
- I appreciate the qualitative analysis and notes on emergent properties.
- Results are evaluated on meaningful OOD benchmarks and provide some benefit in many settings.
- The exploration of the emergent properties is interesting and sheds light on the method, such as the test-time mode dynamics of when mode-switching occurs in Figure 2.

### Weaknesses
- Results seem specific to the tasks specific considered, but are presented as general conclusions about interleaved reasoning more broadly. I appreciate the out-of-distribution results, but they generally target similar tasks -- like V-Star evaluating fine-grained visual search. 
- The idea that the multimodal setting has a larger reasoning space is intuitively appealing, but it's unclear if this is actually the origin of the test-time advantage just from the presented results. For instance, the multimodal training could hypothetically lead to broader diversity within just text, leading to higher best-of-N performance. It would be interesting to see the hypothesis empirically borne out with some quantitative understanding of the diversity of output samples and qualitative examples of how trajectories differ.

### Questions
- It wasn't fully clear to me how the visual output tokens work in that: is it producing tokens for the full output image, including the parts of the input it wants to reproduce, or just the manipulation it wants to show? This would change how it should be presented.
- How dependent is the method on the pretrained model used?
- How do the authors hypothesize the unseen manipulations emerge despite the lack of training? 
- How is it possible for the performance to ever decrease with higher N in Table 4?

### Soundness
3

### Presentation
3

### Contribution
3
