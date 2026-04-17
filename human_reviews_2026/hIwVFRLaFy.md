# VisualPrompter: Semantic-Aware Prompt Optimization with Visual Feedback for Text-to-Image Synthesis

- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
The notable gap between user-provided and model-preferred prompts poses a significant challenge for generating high-quality images with text-to-image models, compelling the need for prompt engineering.
Current studies on prompt engineering can effectively enhance the style and aesthetics of generated images. 
However, they often neglect the semantic alignment between generated images and user descriptions, resulting in visually appealing but content-wise unsatisfying outputs. 
In this work, we propose VisualPrompter, a novel training-free prompt engineering framework that refines user inputs to model-preferred sentences. 
VisualPrompter utilizes an automatic self-reflection module that identifies absent concepts in the generated images, followed by a target-specific prompt optimization mechanism that revises the prompts in a fine-grained manner. 
By deconstructing prompts, introducing new elements at the atomic semantic level, and then reassembling them, our framework is able to maintain semantic consistency and integrity throughout the optimization process.
Extensive experiments demonstrate the effectiveness of VisualPrompter, which achieves new state-of-the-art performance on multiple benchmarks for text-image alignment evaluation. 
Additionally, our framework features a plug-and-play design, making it highly adaptable to various generative models. 
Our code is available at https://github.com/teheperinko541/VisualPrompter.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the semantic misalignment problem in text-to-image (T2I) generation, where sometimes model-preferred output images fail to match  user-provided prompts. The authors propose VisualPrompter, a training-free, semantic-aware prompt optimization framework that refines user prompts through atomic-level semantic decomposition based on the Davidsonian Scene Graph (DSG). 
By parsing prompts into fine-grained semantic concepts through the Self-Reflection Module (SERE) and reconstructing coherent, model-preferred prompts via Target-Specific Prompt Optimization (TSPO), the method effectively produces prompts that better capture user intent and enhance text–image semantic alignment.The addressed problem is highly practically relevant for creative image generation and the proposed approach is easy to reproduce in real-world applications

### Strengths
- By explicitly addressing the problem of semantic omissions, the authors provide a fresh direction for prompt engineering research, shifting the focus from “visual beauty” to semantic faithfulness.By detecting and repairing semantic omissions between user text and generated images, the framework improves  intent alignment, which are crucial for real-world creative and design applications.
- The approach of decomposing prompts into atomic semantic units (entities, attributes, relations), using a Visual-Language Model (VLM) to detect absent ones, and then reassembling an optimized prompt with an LLM is conceptually new and technically creative  on the target problem. The integration of LLM reasoning and VLM verification within a self-reflective pipeline demonstrates strong originality.
- The use of two authoritative benchmarks, DSG-1k (ICLR 2024) and TIFA v1.0 (ICCV 2023), is appropriate and technically justified, as both measure fine-grained semantic alignment between text and image — directly matching the paper’s objective.

### Weaknesses
- Limited diversity of baselines: All three comparative methods (NeuroPrompts, Promptist, BeautifulPrompt)  share similar reinforcement-learning-based optimization paradigms.  The omission of  other omitted categories may weaken the empirical scope.
- Improvements over the baseline are modest (≈ 4–5 points on DSG/TIFA).  Given that VisualPrompter adds several modules and increases inference time (Table 6), the cost–benefit balance remains questionable.Since all reasoning and evaluation rely on Qwen2 and Qwen2-VL, it is also unclear how robust the approach is under different model backbones.
- The paper provides limited empirical evidence on the effectiveness of individual modules. More detailed analyses would strengthen the work — for example, examining which concept types (e.g., objects, attributes, or relations) or which modules (SERE vs TSPO) contribute most to performance gains.

### Questions
1. Please clarify the definition  of “Baseline.”
What exactly is the “Baseline” in Tables 1–3?  Is it simply the raw user prompt or an internal standardized prompt? Why does this baseline outperform some optimized methods?
2. Please justify the selection of comparison methods.
  Given that the Baseline achieves higher semantic alignment scores than other existing methods, please explain why NeuroPrompts, Promptist, and BeautifulPrompt were chosen as baselines of optimized methods.  Additionally, why were other families of methods (eg, multi-objective optimization methods, diffusion-specific) not considered for inclusion in the comparison?
3. Robustness and Ablation
  The paper would benefit from a more detailed ablation analysis showing the relative contribution of each module. Which component—SERE (Self-Reflection) or TSPO (Target-Specific Prompt Optimization) drives the largest improvements across different types of semantic concepts (e.g., objects, attributes, relations)? It would also be helpful to report whether the model’s gains are consistent across different diffusion backbones (e.g., SD v1.5 vs Flux-dev) and whether any module exhibits degradation or instability when applied to more complex or compositional prompts.

### Soundness
3

### Presentation
3

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
This paper introduces VisualPrompter, a training-free, plug-and-play framework that uses self-reflection to detect missing concepts and performs atomic-level prompt edits to improve semantic alignment between user descriptions and text-to-image outputs. Experiments show state-of-the-art results on multiple text–image alignment benchmarks.

### Strengths
1. Visual Prompter leverage visual-language models (VLMs) for question–answer-based detection of missing semantic concepts in generated images, aligns with human intuition and exhibits high interpretability.
2. Visual Prompter significantly outperforms current state-of-the-art prompt engineering methods in multiple benchmarks.

### Weaknesses
1. The user study compares Visual Prompter only with the baseline (original prompts), rather than with other prompt optimization methods. 
2. Lacks comparison with recent methods, such as 《TIPO: Text to Image with Text Presampling for Optimal Prompting.》
3. In Figure 11, the original prompts themselves are ambiguous and unnatural for human expression, such as “person next to person” or “bottle on the left of bottle.” I would like to see the performance of VisualPrompter on more natural and human-like prompts, such as those mentioned in Figure 1.

### Questions
Questions：
1. VisualPrompter uses Qwen2-VL as the visual question answering model; however, SEMANTIC ACCURACY is also evaluated using Qwen2-VL as the assessment model. Would this introduce a bias?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper aims to improve the semantic alignment between generated images and user descriptions. 


It proposes VisualPrompter, a training-free prompt engineering framework that iteratively refines user prompts. VisualPrompter has a self-reflection module that analyzes generated images, and a target-specific prompt optimization that revises the prompt later. The method is plug-and-play and achieves state-of-the-art alignment on multiple image generation benchmarks.

### Strengths
- Easy to use: the proposed VisualPrompter is model-agnostic and plug-and-play, making it highly adaptable to various generative models.
- Good results: VisualPrompter outperforms many baselines on multiple benchmarks and multiple generative models, as shown in Table 1.

### Weaknesses
- Auxiliary LLM bias: Introducing an additional LLM in the loop may inject its own biases, especially there’re multiple LLM calls. 
- Compute overhead and latency: the generate - analyze - revise cycles may be significantly more expensive than a single forward pass. In addition, LLMs were called multiple times in one image generation, which might be costly. 
 - Limited contribution: modules are not novel. For example, regarding the reflection module, the LLM Expander, the LLM Composer, similar techniques can be found in references. It'll be great if author could explain what's the unique contribution and novelty in VisualPrompter.

### Questions
- Robustness: How is the performance with difficult prompts, for example, long prompts, multilingual prompts, code-mixed prompts? Could you show any qualitative failure cases?

### Soundness
3

### Presentation
3

### Contribution
2
