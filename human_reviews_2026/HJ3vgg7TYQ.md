# RePrompt: Reasoning-Augmented Reprompting for Text-to-Image Generation via Reinforcement Learning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 2

## Abstract
Despite recent progress in text-to-image (T2I) generation, existing models often struggle to faithfully capture user intentions from short and under-specified prompts. While prior work has attempted to enhance prompts using large language models (LLMs), these methods frequently generate stylistic or unrealistic content due to insufficient grounding in visual semantics and real-world composition. Inspired by recent advances in reasoning for language model, we propose RePrompt, a novel reprompting framework that introduces explicit reasoning into the prompt enhancement process via reinforcement learning. Instead of relying on handcrafted rules or stylistic rewrites, our method trains a language model to generate structured, self-reflective prompts by optimizing for image-level outcomes. The tailored reward models assesse the generated images in terms of human preference, semantic alignment, and visual composition, providing indirect supervision to refine prompt generation. Our approach enables end-to-end training without human-annotated data. Experiments on GenEval and T2I-Compbench show that RePrompt significantly boosts spatial layout fidelity and compositional generalization across diverse T2I backbones, establishing new state-of-the-art results. Code: https://github.com/microsoft/DKI_LLM/tree/main/RePrompt.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes RePrompt, a prompt enhancement framework for text-to-image (T2I) generation models that introduces explicit reasoning into the prompt generation process and optimizes based on image-level outcomes using reinforcement learning. The reward model adopts an ensemble reward that evaluates three dimensions: human preference, visual realism, and semantic alignment. On the GenEval benchmark, significant improvements in spatial layout fidelity and compositional generalization are observed, and on T2I-Compbench, particularly notable score improvements in spatial compositions are confirmed.

### Strengths
- The proposed method fixes the image generation model and optimizes only the language model policy, making it applicable to any existing T2I backbone. Since the reward model depends only on the prompt-image output pair and not on any specific T2I architecture, it generalizes naturally across different generation backbones and unseen prompt distributions.

- Substantial improvements are observed in compositional understanding, particularly in spatial position (Position). In the Position category of GenEval, remarkable relative improvements are achieved compared to the Qwen2.5 3B baseline: FLUX (+77.1%), SD3 (+78.8%), and Pixart-Σ (+122.2%). Overall GenEval scores also consistently improve across each backbone (+11.8%, +10.3%, +6.9%).

- In terms of inference latency, the method is significantly faster (30s per image) compared to Idea2Img (140s per image) and PARM++ (110s per image), while also achieving the highest accuracy (0.76), demonstrating practical advantages.

- In qualitative evaluation, concrete examples are presented, showing that for prompts such as "a fire hydrant with a tennis racket" and "a photo of a dog above a cow," the method avoids object fusion and misplacement observed in baseline models and faithfully reproduces the intended spatial composition.

### Weaknesses
- **Limited scope of evaluation**: The main evaluation relies solely on two automatic evaluation benchmarks: GenEval and T2I-Compbench. Human evaluation is not included, so the practical utility aspects such as "whether actual users find the results convincing," "whether generated prompts are readable," and "whether reasoning explanations are appropriate" depend solely on automatic metrics.

- **Insufficient comparison with closely related methods**: Prior work on prompt enhancement mentions iterative refinement approaches and single-pass LLM-based enhancement methods, and quantitative comparisons with Promptist, PAG, GPT4, Deepseek-r1, and Qwen2.5 are provided in tables. However, detailed analysis from the perspective of RL-based prompt optimization is insufficient. Therefore, the theoretical and experimental justification for "why CoT (Chain-of-Thought) with RL is superior to conventional prompt optimization" is somewhat weak.

- **Lack of CoT quality evaluation**: While the method generates reasoning traces that simulate visual implications of prompts—much like how humans mentally visualize a scene—and this structured, logic-driven process anticipates potential errors during prompt construction, direct evaluation of the reasoning text itself in terms of correctness, consistency, and conciseness is not performed. The claim that "reasoning is effective" is made indirectly through final image scores, but it is not clearly separated whether "performance improved because of structuring" or "merely because detection-friendly phrases were added."

- **Insufficient discussion of safety and robustness**: The design optimizes prompt generation through reinforcement learning, but there is no discussion of risks where RL might learn extreme descriptive expressions, overly detailed specifications, or expressions that deceive the reward model, nor mechanisms to detect or suppress such behaviors. Although the Broader Impact section mentions risks of generating misleading content and bias propagation, and recommends pairing the method with content moderation filters and fairness-aware training objectives, concrete countermeasures or experimental validation are not included.

### Questions
- Do you plan to conduct human evaluation? Evaluating the quality of generated images and readability of prompts from the perspective of actual users would demonstrate practical utility that cannot be captured by automatic metrics alone.# Review

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes reasoning-augmented framework for text-to-image generation. Unlike previous methods that rely on rewriting or heuristic feedback loops, RePrompt trains an auxiliary LLM via reinforcement learning (RL) that generates both reasoning traces and refined prompt to further prompt a frozen text-to-image model. The proposed method has been extensively evaluated on GenEval and T2I-Compbench datasets, achieving consistent improvements across backbones (FLUX, SD3, PixArt).

### Strengths
- The idea of combing LLM reasoning and image-level feedback is novel and promising. 

- The reward is also well designed. First of all, the visual-reasoning reward acts as a bridge to connect image reward (human preference alignment) with semantic grounding (VLM reward). Second, it allows the reward to depend only on the behavior of input and output, enabling model-agnostic characteristic of RePrompt across different T2I backbones.

- The ablations and theoretical analysis (in Appendix B), together with the empirical results, all justify the design of RL in this paper, especially on GRPO optimization, and reasoning traces which acts as variance-reduction condition. 

- The paper is well presented and the reproducibility is good.

### Weaknesses
- The evaluation benchmarks are only object-centric datasets. The performance on open-world prompt is not verified. It is better to show several examples on this scenario.

- No failure cases in the visualization. What is the model behavior on rare, free-form prompts not covered in GenEval? For example, "a photo of a cat working in an office"

### Questions
- Is human-in-the-loop verification needed to ensure that the reward aligns well the perceptual alignment?

- What is the impact of reasoning length on the generation quality?

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
This paper proposes RePrompt, a novel framework that uses Reinforcement Learning (RL) to train a Large Language Model (LLM) to enhance text-to-image (T2I) prompts. It indicates that the generation of a structured, self-reflective "reasoning trace" alongside the enhanced prompt helps ground the prompt in visual semantics and improve compositional accuracy. The method is trained end-to-end using a tailored ensemble reward model that assesses image quality, semantic alignment, and prompt structure. Experiments on GenEval and T2I-Compbench show improvements, especially in spatial understanding, over strong LLM-enhanced baselines across multiple T2I backbones.

### Strengths
1. The integration of explicit, structured reasoning with RL for prompt enhancement is a well-motivated approach. It effectively bridges the gap between linguistic fluency and visual plausibility that plagues LLM-based prompters.
2. The paper demonstrates consistent performance gains across three different diffusion-based T2I models (FLUX, SD3, Pixart-Σ). The improvements in challenging areas like spatial reasoning are compelling.
3. The framework is designed to be T2I model-agnostic, requiring no retraining of the image generator. The reported inference latency (30s) is significantly lower than iterative optimization baselines, making it more practical.

### Weaknesses
1. The training and evaluation prompts are heavily focused on object-centric, compositional generation (training prompts sourced from GenEval-like templates). This raises a concern about potential overfitting to the specific categories and styles of the benchmarks used.

2.  It is unclear how RePrompt would perform on more diverse, stylized, imaginative, or long-form narrative prompts that are common in real-world use. 

3. While the paper shows generalization across diffusion-based models, its performance on architecturally different T2I models (e.g., autoregressive or unified multimodal models) remains unverified. Furthermore, the claim of being "model-agnostic" is slightly tempered by the finding that the policy is "individualized to each T2I backbone." This suggests that to achieve optimal performance on a new T2I model, one might need to retrain the RePrompt LLM via RL, which is computationally expensive and reduces plug-and-play utility.

4. The fundamental problem might be mitigated by future, more capable T2I models that inherently possess better compositional reasoning. If such models emerge, the value of an add-on module that requires significant RL training could diminish. 

5. The instructions provided to the LLM-enhancer baselines (Qwen2.5, GPT-4, etc.) are not included. The performance of these baselines is highly sensitive to how they are prompted, providing this information would increase the transparency and reproducibility of this work.

6. What are the common failure cases for RePrompt? For instance, does the reasoning trace ever lead to over-specification or introduce its own hallucinations? Examples of such failures would help users understand the limitations of the current approach.

### Questions
Please see weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper, the authors propose a new method for automated prompt engineering for text-to-image generation. In particular, they use the standard RL training method for LLM and train an LLM to perform this specific task. They compare their method with several baselines and show improvements.

### Strengths
1. The paper is clearly written and easy to follow.
2. Table 2 shows great transferability of the prompts among different text-to-image models.
3. The ablation study conducted in this paper is very thorough.

### Weaknesses
1. My main concern about this paper is regarding its novelty. The training procedure of their model is a very standard RL recipe for LLM, and using LLM as an automated prompt generator for text-to-image generation is not a new idea either (e.g. Hao et al. (2023); Mo et al. (2024); Yeh et al. (2024); Ma ̃nas et al. (2024); Yun et al. (2025); Cao et al. (2023); Qin et al. (2024); Yang et al. (2024d); Wu et al. (2024); Wang et al. (2024) in the paper). It seems like this paper would be better suited for other venues like TMLR.
2. The authors claim that prior LLM prompt generation methods “frequently generate prompts that produce images with semantically inconsistent or visually implausible content, such as conflicting object placements or unrealistic interactions, because the underlying LLMs lack grounding in physical reality and do not incorporate feedback from downstream visual task” “ with limited generalization”. However, neither of the claims are supported by their experiments. Specifically on generalizability, the model that the authors propose is trained on a dataset curated by following the prompt construction in the GenEval benchmark and evaluated on the same benchmark plus another small benchmark with only 300 test examples. It is unclear to me why these experiments can warrant claims w.r.t. better generalizability.
3. The authors use GPT-4V, a deprecated model in the GPT family for comparison, not only that it is impossible to replicate the result, it also renders the comparison a bit outdated. It would be better if the authors can compare their method with newer GPT models (would be even better to include comparison with a standard model and a reasoning model).
4. The authors have also only finetuned from Qwen models and it would be nice to show results from other model families.
5. The authors can consider adding discussions and/or comparisons to the following papers:

Yeh et al. TIPO: Text to Image with Text Presampling for Prompt Optimization. 2024.

Lu et al. Language models as black-box optimizers for vision-language models. 2024.

He et al. Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation. 2024.

### Questions
Have the authors trained from instruct models as opposed to base models? How much can this method improve the performance?

### Soundness
2

### Presentation
2

### Contribution
1
