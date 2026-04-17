# Unveiling the Potential of Diffusion Large Language Model in Controllable Generation

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
Controllable generation is a fundamental task in NLP with many applications, providing a basis for function calling to agentic communication. However, even state-of-the-art autoregressive Large Language Models (LLMs) today exhibit unreliability when required to generate structured output. Inspired by the current new diffusion-based large language models (dLLM), we realize that the architectural difference, especially the global information-sharing mechanism for language modeling, may be the key to unlock next-level controllable generation. To explore the possibility, we propose Self-adaptive Schema Scaffolding ($S^3$), a novel framework that enables dLLM to stably generate reliable structured outputs (e.g., JSON) by utilizing its innate reverse reasoning capability and global context awareness. $S^3$ initiates a schematic template directly in the output context as a starting state for dLLM, offering a more robust and general method than intricate prompt optimization. Experiments demonstrate that our method substantially unlocks the dLLM’s potential in controllable generation in terms of structure adherence, content fidelity, and faithfulness. These results establish new perspectives and practical pathways for deploying language models in controllable generation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper explores the potential of diffusion-based large language models (dLLMs) as an alternative to autoregressive (AR) models for controllable structured generation—a key capability for applications like JSON output, API integration, and agentic communication.
The authors identify key limitations of AR models (lack of global coherence, irreversible token commitments, and sequential dependency) and argue that dLLMs’ iterative denoising and global attention mechanisms offer inherent advantages for structured output.
To unlock this potential, they propose Self-adaptive Schema Scaffolding (S³), a training-free method that pre-initializes dLLM outputs with a structured schema scaffold (e.g., JSON skeleton) and uses adaptive null tokens to handle variable-length or missing fields.

### Strengths
- First comprehensive analysis of diffusion LLMs in controllable structured generation.

- S³ improves structure adherence and reduces hallucination without retraining.

- Achieves near-perfect structure compliance (≈0.997) and lowest hallucination (0.33 vs. 0.40 baseline).

- Provides clear theoretical rationale and visual intuition for why scaffolding aids denoising.

- Training-free and computationally efficient; suitable for real-world schema-driven applications.

- The new tri-metric framework (adherence, fidelity, faithfulness) sets a useful precedent for future structured LLM work.

### Weaknesses
- Experiments focus narrowly on WikiBio and JSON; applicability to more diverse formats (e.g., code, graphs) remains untested.

- The study lacks comparison to strong AR baselines using grammar-constrained decoding or function-calling LLMs.

- Only one dLLM (LLaDA) is evaluated; cross-model tests (e.g., Dream or Mercury) would strengthen claims.

- While effective, it may not generalize to schemas without explicit “missing field” semantics.

- Despite reduced denoising steps, dLLM inference (O(nL²)) may still lag AR models (O(L)), which could affect scalability in long-context tasks.

### Questions
- How sensitive is S³’s performance to the number of denoising steps or the choice of remasking strategy (top-K vs. low-confidence)?

- Can this framework be extended to multi-modal dLLMs (e.g., LLaDA-V) where structured outputs integrate vision and text?

- How does S³ interact with instruction-tuned or RLHF-optimized dLLMs trained for task-following?

- Would hybrid diffusion–autoregressive setups (e.g., block diffusion) inherit similar controllability advantages?

- Have the authors evaluated the effect of schema complexity (nested vs. flat JSON) on scaffolding performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this work, the authors propose a framework to enable diffusion-based LLMs for reliable structured outputs. A schematic template is initiated. Experimental results show that the proposed method marginally improves the structure outputs compared with the commonly used prompting strategy.

### Strengths
1. Clear motivation:
Exploring diffusion LLMs for controllable text generation is a fresh and under explored research area.

2. Conceptually good idea:
The use of schema scaffolding as a structural prior aligns well with diffusion’s iterative refinement mechanism. No additional fine-tuning or retraining is needed.

3. Improved results:
The paper demonstrates substantial improvement over baseline diffusion models.

### Weaknesses
1. Limited experimental scope:
Only one dataset (WikiBio) and one diffusion model (LLaDA) are tested. Broader validation tasks (e.g., code generation, dialogue structure, form filling) would strengthen generality.

2. Lack of comparison to AR-LM baselines:
Although the paper motivates dLLMs as alternatives to AR models, it doesn’t include a direct comparison with strong AR methods like structured prompting or constrained decoding (e.g., CodeLLaMA, T5, or GPT-style JSON control).

3. There is no enough details on how to compile constraints into a schema, which is then used to initialize a noisy scaffold where mask tokens serve as placeholders for missing content. Especially, how to use this compilation in other tasks?

### Questions
1. Comparative perspective:
How does S3 perform against autoregressive structured decoding methods (e.g., JSON mode in GPT-4 or constrained decoding in PaLM/CodeLLaMA)?

2. Diffusion step selection:
How is the optimal number of denoising steps (e.g., 8, 16, 32) determined? Could adaptive step scheduling yield further gains?

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
2

### Summary
This work proposes a method that enables diffusion large language models (dLLMs) to provide contents with structured output (e.g., JSON). Empirical results show that the propose method ($S^3$) improve both structural adherence and content quality in the chosen downstream task.

### Strengths
1. Generating reliable structured outputs is an important research direction and has many practical downstream applications, as evidenced by the fact that it is widely discussed and investigated in autoregressive LLMs literature. This work extends this research direction to a relatively less explored model of diffusion LLMs (dLLMs) and proposes a new method to improve generation of structured outputs.

2. It is well-motivated to use dLLMs for structured outputs, as this task (which requires lookahead planning from known future tokens) aligns well with the properties of dLLM.

3. Empirical results show that the proposed methods greatly improve structural adherence to the target JSON schema, while also preserve or even improve generated contents' quality (as shown in Figure 4 and Table 1).

### Weaknesses
1. Generalization across tasks: The experiments only use one dataset (Wikibio by Lebret et al., 2016), so it is unclear how well the proposed method generalize to other, especially more difficult, datasets.

2. Generalization across types of structured outputs: Following the previous point, it would also be nice to include experiments on other types of structured outputs, such as XML or YAML.

3. Although the authors state they use the Wikibio dataset, they didn't mention what's the input and what's the target output or include examples in the paper. I think more details need to be elaborated as how the authors use the dataset in their experiments.

### Questions
1. For the Wikibio dataset, do you use the biography text as the input to the prompt, and the contents of the infobox as the target output? It seems that these details are not explicitly mentioned in the paper.

2. In autoregressive LLMs, it is previous shown by Tam et al [1] that enforcing LLMs to provide structured outputs degrade task performance (i.e., generated contents' quality) compared to when LLMs can "speak freely" without structured constraints. Therefore, a way to preserve both the structured adherence and content quality is to adopt a two-stage inference process: Let the model "speak freely" and focus on generating the contents first, and then convert the generated contents to structured outputs. Do diffusion LLMs (dLLMs) have similar properties? I think it would make the work more complete if this experiment is done, so that we know whether enforcing structured outputs degrade dLLMs' content generation ability.

[1] Tam, Zhi Rui, et al. "Let me speak freely? a study on the impact of format restrictions on performance of large language models." arXiv preprint arXiv:2408.02442 (2024).

### Soundness
3

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
3

### Summary
The paper proposes an algorithm that utilizes schema scaffolding for controllable generation in diffusion models, theoretically demonstrates the feasibility of schema scaffolding, and conducts experiments across three key dimensions to validate the effectiveness of the proposed method.

### Strengths
To ensure global awareness, the paper employs a diffusion model for generation. Furthermore, it introduces a schema scaffolding mechanism to enable controllable generation and provides theoretical proof of its feasibility.

### Weaknesses
1. The paper (Figure 1) points out that autoregressive models lack global awareness, which is an advantage of diffusion models. To validate this perspective, the authors should provide experimental results from an autoregressive model baseline. 
2. The equations subsequent to Equation 3 are unnumbered, resulting in an inconsistent presentation. 
3. The experimental setup is relatively simplistic, employing a very limited number of baselines, which consequently lacks persuasiveness.

### Questions
See the weakness.

### Soundness
3

### Presentation
2

### Contribution
3
