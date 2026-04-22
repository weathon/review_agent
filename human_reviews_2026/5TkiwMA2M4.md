# Chain-of-Generation: Progressive Latent Diffusion for Text-Guided Molecular Design

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Text-conditioned molecular generation aims to translate natural-language descriptions into chemical structures, enabling scientists to specify functional groups, scaffolds, and physicochemical constraints without handcrafted rules. 
Diffusion-based models, particularly latent diffusion models (LDMs), have recently shown promise by performing stochastic search in a continuous latent space that compactly captures molecular semantics. 
Yet existing methods rely on one-shot conditioning, where the entire prompt is encoded once and applied throughout diffusion, making it hard to satisfy all the requirements in the prompt.
We discuss three outstanding challenges of one-shot conditioning generation, including the poor interpretability of the generated components, the failure to generate all substructures, and the ambiguity in considering all requirements simultaneously.
We then propose three principles to address those challenges, motivated by which we propose Chain-of-Generation (CoG), a training-free multi-stage latent diffusion framework.
CoG decomposes each prompt into curriculum-ordered semantic segments and progressively incorporates them as intermediate goals, guiding the denoising trajectory toward molecules that satisfy increasingly rich linguistic constraints. 
To reinforce semantic guidance, we further introduce a post-alignment learning phase that strengthens the correspondence between textual and molecular latent spaces. 
Extensive experiments on benchmark and real-world tasks demonstrate that CoG yields higher semantic alignment, diversity, and controllability than one-shot baselines, producing molecules that more faithfully reflect complex, compositional prompts while offering transparent insight into the generation process.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper aims to resolve that one-shot conditioning in diffusion models often fails on compositional prompts (e.g., missing or misplacing substructures). The authors propose Chain-of-Generation (CoG) as a training-free, multi-stage sampling method that decomposes a prompt into coarse-to-fine(scaffold → functional groups → fine details). Each stage conditions on the cumulative sub-prompt and restarts denoising from a mid-noise step, preserving earlier commitments while adding constraints. To strengthen text–graph correspondence, the authors “post-align” a graph VAE latent with a SciBERT text embedding via contrastive learning, yielding a stronger diffusion backbone (GraphLDM). 

On ChEBI-20 and PubChem, GraphLDM+CoG improves BQI and Q-Cov over baselines while maintaining 100% validity, with a small trade-off in novelty. Ablation study shows that coarse-to-fine planning and early-stage prompt regularization are essential.

### Strengths
- The paper presents a training-free Chain-of-Generation with cumulative prompts and mid-noise restarts that has the potential to be easily integrated with existing graph diffusion backbones.

- The authors pointed out the limitations of SMILES-based metrics and utilized graph-level metrics (MACCS Tanimoto similarity), consistent gains on ChEBI-20/PubChem with 100% validity, with ablation studies on CoG planning.

### Weaknesses
Contribution:
- The demonstrated effectiveness of the proposed methodology appears constrained to compositional text prompts that mix coarse and fine structural descriptions; it is unclear whether the approach remains effective for richer and more complex textual conditions (e.g., biochemical properties).

- The performance gains attributable to CoG seem modest; the results suggest that contrastive alignment contributes a substantially larger portion of the improvement.

Motivation:
- The overall method relies on the assumption that denoising in molecular diffusion models proceeds in a coarse-to-fine manner; this claim requires stronger empirical support through additional targeted experiments.

- Since CoG is applied only on a fine-tuned variant of 3m-diffusion that already achieves 100% validity, the paper does not establish that CoG itself improves the sample validity.

Experimental results:
- To substantiate the benefits of a training-free, inference-time method, the approach should be applied to a broader set of molecular generative models, demonstrating consistent gains across architectures.

- The baselines in the results table are dated: several competitive SMILES-based autoregressive models (e.g., bioT5+[1], MolXPT[2]) are missing, and diffusion baselines with stronger diversity (e.g., LDMol[3], TGM-DLM[4]) are not included despite the proposed method operating on diffusion models.

- Several compositional generation methods on pre-trained diffusion models[5] have already been reported; direct comparisons against these methods are necessary to demonstrate the contributions.

---

[1] BioT5+: Towards Generalized Biological Understanding with IUPAC Integration and Multi-task Tuning, ACL 2024

[2] MolXPT: Wrapping Molecules with Text for Generative Pre-training, ACL 2023

[3] LDMol: A Text-to-Molecule Diffusion Model with Structurally Informative Latent Space Surpasses AR Models, ICML 2025

[4] Text-Guided Molecule Generation with Diffusion Language Model, AAAI 2025

[5] Compositional Visual Generation with Composable Diffusion Models, ECCV 2022

### Questions
- As far as I know, the text prompts in the ChEBI-20 dataset are detailed enough to specify a single answer molecule. Is it reasonable to regard the model with a higher output molecule diversity as a better model?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents Chain-of-Generation (CoG), a novel inference-time pipeline for text-conditioned molecular diffusion models.
Instead of conditioning on an entire prompt at once, CoG decomposes the text into semantic segments (e.g., core scaffold → functional groups → modifiers) and applies them sequentially during denoising, resuming diffusion from partially denoised latents.
Experiments use a post-aligned GraphLDM backbone — a graph-based latent diffusion model fine-tuned with a contrastive alignment between molecule and text embeddings.
The method improves validity and coverage (BQI) on PubChem and ChEBI-20.

### Strengths
Clear and well-written. The paper is concise, logically structured, and easy to follow. Figure 1 nicely illustrates the staged conditioning concept.

- Good motivation and insight. The work clearly identifies the limitations of one-shot text conditioning for compositional prompts.

- Practical contribution. CoG is a plug-in inference strategy that can be used with any pre-trained conditional diffusion model, without architectural change.

- Conceptual analogy. This is an inference-time compositional conditioning approach analogous in spirit to chain-of-thought prompting in LLMs, decomposing generation into interpretable, sequential steps.

- Good inference-suggestion method. I find this to be a creative and promising idea for improving prompt controllability in molecule generation.

### Weaknesses
- Scope of novelty.
Per my understanding, CoG is a new inference strategy rather than a new model or architecture.
Its novelty lies in orchestrating sequential conditioning during denoising—conceptually related to coarse-to-fine or editing-based sampling in diffusion literature.
Clarifying that distinction would help set the right expectations for readers.

- Training vs. inference confusion and possible overfitting.
The GraphLDM backbone is fine-tuned (post-aligned) on PubChem / ChEBI-20, and CoG is applied on that same model.
All reported results correspond to post-aligned GraphLDM + CoG, not a plain model.
Because the fine-tuning uses text–molecule pairs from the same domain, improvements may partly stem from memorization rather than the CoG procedure itself.
No ablation is provided comparing CoG on plain GraphLDM vs post-aligned GraphLDM, making attribution unclear.

- Textual faithfulness not quantitatively measured.
Claimed improvements in prompt adherence are only supported by qualitative examples.
Metrics such as substructure or property matching (SMARTS-based checks) or text–molecule embedding similarity would directly measure semantic alignment and strengthen the argument.

- Generative novelty trade-off.
Q-Nov decreases slightly while coverage rises, implying CoG trades exploration for controllability.
Discussing this trade-off explicitly would give a more balanced interpretation.

- Inference-time overhead.
Because CoG performs multiple partial denoising passes, its inference cost should increase relative to one-shot sampling.
The paper does not report runtime or complexity analysis.
What about inference-time overhead? Quantifying this would clarify practical usability.

- Clarification on “Ground Truth” in Figure 3.
The “ground truth” molecules are simply the dataset references paired with each prompt, not experimentally verified samples.
Hence Figure 3 reflects reconstruction fidelity rather than true generalization, and this should be stated explicitly.

### Questions
- Please specify the exact model variant used with CoG (plain vs post-aligned GraphLDM).

- Provide an ablation isolating the effect of CoG from the post-alignment fine-tuning.

- Quantify inference-time overhead and number of denoising stages.

- Add quantitative prompt-faithfulness metrics (substructure, property, or embedding alignment).

- Evaluate CoG on larger or out-of-distribution molecules to test compositional generalization.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Chain-of-Generation (CoG), a training-free progressive latent diffusion framework for text-guided molecular design. Instead of using one-shot conditioning that struggles with complex prompts, CoG decomposes a natural-language description into semantically ordered sub-prompts and incrementally guides the diffusion process from coarse scaffolds to fine-grained details, enhancing controllability and interpretability. A post-alignment contrastive learning stage further strengthens the correspondence between textual and molecular latent spaces. Experiments on ChEBI-20 and PubChem show that CoG consistently improves semantic fidelity, coverage, and structure–text alignment over SMILES-based and graph-based baselines.

### Strengths
+ The paper introduces a novel and conceptually elegant chain-of-thought–inspired framework for progressive molecular diffusion generation.
+ The proposed method is training-free and easily integrates with existing latent diffusion models, improving interpretability and controllability.
+ Comprehensive experiments and chemically meaningful graph-based metrics demonstrate consistent performance gains over strong baselines.

### Weaknesses
+ Although the paper identifies three main limitations of one-shot conditioning, the proposed CoG framework does not show sufficiently strong improvements in addressing these issues.
+ The evaluation focuses mainly on structural fidelity without testing functional or physicochemical properties of generated molecules.
+ The method relies heavily on external LLM-based prompt segmentation, which may introduce instability or semantic errors.
+ Reported performance gains over GraphLDM are modest and may not fully justify the framework’s added complexity.
+ The so-called “training-free” claim is weakened by the inclusion of a contrastive post-alignment learning phase.
+ The diversity analysis is limited, lacking deeper studies on scaffold or property-level novelty.

### Questions
1. Could the authors quantitatively demonstrate how CoG ensures the faithful realization of each sub-goal within multi-component prompts? (Respond to C2)
2. The claim of interpretability is relatively weak. CoG does not demonstrate substantially stronger interpretability compared to prior methods, as its interpretability mainly stems from LLM-based prompt segmentation, which itself introduces additional uncertainty. I suggest that the authors conduct more experiments or analyses to support this claim directly.
3. In Tables 4–7, all methods show better performance as the number of prompt pieces increases. Does this suggest that these models, not only CoG, can already handle multi-component prompts to some extent? If so, does this weaken the claim in limitation C2 regarding missing substructures, and how does CoG’s improvement differ qualitatively from the general trend?
4. Could the authors provide stronger empirical evidence for the claim of “ambiguous conditioning” (C3)? Specifically, can they quantify ambiguity reduction or show how CoG improves structural placement and semantic disambiguation compared to one-shot baselines? It also remains unclear whether CoG still exhibits residual confusion and to what extent prior methods were actually hindered by ambiguous conditioning.

It is undeniable that overly complex prompts can negatively affect molecular design quality, and addressing this issue would greatly enhance the practical value of such methods. However, the proposed CoG framework has not been convincingly shown to alleviate this problem. The authors could provide additional experimental evidence or analyses to demonstrate CoG’s effectiveness in handling this. Strengthening this part would make the contribution more convincing.

### Soundness
2

### Presentation
3

### Contribution
2
