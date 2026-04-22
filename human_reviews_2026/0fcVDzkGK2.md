# DIVIDE-AND-DENOISE: A GAME THEORETIC METHOD FOR FAIRLY COMPOSING DIFFUSION MODELS

- Avg Score: 2.67
- Decision: Reject
- Scores: 4, 4, 0

## Abstract
With the widespread availability of pre-trained diffusion models, there are many options for which models to use and how to use them together. Making these decisions depends highly on both the user's goals and the expertise of each model. Taking this into account, we propose coordinating models as one would a specialized workforce--through a fair yet efficient division of labor. Divide-and-Denoise uses multiple pre-trained diffusion models, each defined over the same space, to refine a noisy sample over time. At every timestep, we alternate between (i) dividing the sample into regions in a way that satisfies our game-theoretic criteria and (ii) denoising a region with the assigned model in a way that respects our alignment criteria. This leads to a new composite denoising process that evolves together with a division process. Since ground truth is typically not available for our setup, we measure how well Divide-and-Denoise coordinates a team of single-concept text-to-image diffusion models relative to a multi-concept model. On the GenEval benchmark, our method generates images that capture the strengths of each model, outperforming baselines and resolving common failures like missing objects and mismatched attributes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a game-theoretic approach to combine multiple different text-to-image diffusion models. A crucial constraint is that the models must have the same latent dimensions. Ensuring fairness when dividing the noise maps among the models prevents the collapse into a single concept and only generating this concept. The experimental results show that the approach doesn't suffer from collapse to a single concept and can generate all the objects present in the prompt using different models.

### Strengths
- The idea of fusing different models to generate an image is very intriguing.
- The paper is well written, and even the mathematical details are easy to understand.

### Weaknesses
- The figures in the paper have low resolutions. The text in the images is not readable.
- A few more example images would be nice to illustrate what makes this approach better than other approaches.
- The evaluation is not very thorough. For example for the generation of multiple objects and the attribute allocation only figure 3 is shown as evidence.
- It is not clear why the prompts used in Section 4.3 are out-of-distribution.

Minor:
- In line 404, 412 and 413 the citations seem to be missing.
- The figure number in line 423 is not correct
- In line 466 the table number is not correct

### Questions
Q1: I might have missed it, but how is the fairness ensured when dividing the pixels to the models?  
Q2: Why do the pixels have to be distributed to a fixed model? Wouldn't it also be possible, especially when two areas overlap, to average over the noise maps of multiple models?  
Q3: How does VQA measure the compositional correctness? If I am not mistaken, an image can be composed in different ways, while the VQA can still be correct.  
Q4: Why are there values missing for VQA in table 1?  
Q5: Why are the prompts used in Section 4.3 OOD? What does it mean if there are "conflicts between individual prompts"?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper “Divide-and-Denoise” proposes a game-theoretic framework for compositional sampling from multiple pre-trained diffusion models. Rather than directly averaging denoising predictions (as in MultiDiffusion or joint-prompt methods), the authors formulate the problem as a fair division game, where latent coordinates are “goods” and each diffusion model acts as a “player.” This elegant formulation allows the model to dynamically allocate spatial responsibility among different diffusion processes in a principled, temporally coherent, and fairness-aware way.

The method alternates between two tightly coupled updates at each diffusion step:
	1.	Compositional denoising, which generates a latent proposal based on soft region assignments Q_t;
	2.	Dynamic allocation, which optimizes Q_t via a bilevel optimization that enforces fairness, smoothness, and attention alignment across time.

A key novelty is the introduction of the alignment score derived from cross-attention maps, which measures semantic consistency between denoised regions and textual prompts. The paper also introduces a “fictitious player” to handle unassigned or background regions, ensuring that all latent coordinates are properly modeled. Theoretical analysis leads to a closed-form softmax-like solution for Q_t (Theorem 2), while alternating optimization jointly refines both the denoising kernel and spatial allocation.

### Strengths
(1) Conceptual originality: The use of game theory and fair division in diffusion model coordination is highly innovative and goes beyond heuristic compositional fusion.

(2) Theoretical rigor: The bilevel formulation, connection to entropy-regularized MDPs, and derivations (Theorems 1–2) are mathematically sound and clearly motivated.

(3) Strong empirical performance: On multi-object and attribute-binding tasks, Divide-and-Denoise significantly reduces object overlap and color confusion, outperforming joint-prompt and MultiDiffusion baselines.

### Weaknesses
(1) Computational overhead: Alternating updates for Q_t and p_t^c introduce nontrivial cost during inference.

(2) Dependence on cross-attention quality: The allocation accuracy relies heavily on stable and interpretable attention maps.

(3) Limited evaluation scope: Current experiments are restricted to text-to-image synthesis; demonstrating broader modality coverage would further strengthen the claim of generality.

**Important** (4 )Many figures are blurry, making them nearly unreadable. Several references are missing or incorrectly formatted, which severely reduces the paper’s professionalism and readability.

### Questions
None

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
This paper aims to tackle compositional generation with diffusion models by introducing "Divide-and-Denoise", a game theoretic sampling procedure that composes multiple pretrained diffusion model "player" models via fair division of the latent space at every denoising step. The method alternates between (i) an allocation step that infers soft segmentations by solving a fairness-constrained optimization using utilities derived from cross-attention maps, and (ii) a denoising step whose optimal Gaussian kernel has a mean that combines per-model updates masked by the allocation plus a guidance term driven by an alignment score; a fictitious background player and a KL term encourage sensible coverage and temporal smoothness.

### Strengths
- ***Interesting & principled idea***: Recasts compositional generation as a fair-division game over soft region allocations, using cross-attention.

### Weaknesses
- ***Writing quality***: The paper appears incompletely prepared at submission time. In the experiments section there are placeholder “?” citations, tables that overflow horizontally, and tables with missing entries. The manuscript also exceeds the 9-page limit, suggesting the writing and formatting were not finalized. These presentation issues significantly hinder readability and raise concerns about diligence in preparing the submission.

- ***Experimental setups***: The Joint Prompt setup appears to be an extremely weak baseline. With such a simple enumeration-style prompt, the model has a high probability of failure. Instead, the authors should compare results when using a language model to generate natural prompts containing multiple objects. In the same vein, averaging is also far too simple as a baseline. It seems strange to expect that averaging score values from different conditions would work well.

- ***Prompt division***: This paper focuses on effectively dividing and combining generation from multiple players, yet it doesn't address how to divide the conditions among them. For example, if there's a long prompt, there is a need to determine how to distribute its contents to each player. With the current approach, I have serious doubts about whether this can work for scenarios with complex multiple relations.

- ***The title of Section 4.3***: I don't understand why this is considered out-of-distribution at all. Wouldn't "conflict prompt" be more appropriate?

### Questions
N/A

### Soundness
2

### Presentation
1

### Contribution
2
