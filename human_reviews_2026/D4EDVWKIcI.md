# Controllable Molecule Generation via Sparse Representation Editing: An Interpretability-Driven Perspective

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Controllable molecule generation is crucial for diverse scientific applications, such as drug discovery and materials design. While large language models (LLMs) show great promise, their dense and entangled representations impede precise control over the generation of molecules with bespoke substructures or properties. To address this, we propose Sparse Representation Editing (SpaRE), an interpretability-driven framework for fine-grained and precise control in LLM-based molecule generation. The crux of SpaRE is to disentangle dense representations into various sparsely activated latent patterns that correspond to chemically meaningful concepts. Building on this, SpaRE enables direct manipulation of LLM representations associated with these concepts to achieve (1) local control, by generating target atoms and functional groups at specified positions; and (2) global control, by customizing the overall structural and physicochemical properties within defined ranges. In this way, our framework advances interpretability from post-hoc analysis to actionable generative control. Experiments demonstrate that SpaRE is capable of generating chemically desirable molecules under complex constraints in real-world scenarios, while providing causal insights for quantitative structure–property analysis. The code and demo are available at https://github.com/SpaRE-paper/SpaRE.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents SpaRE, an interpretability-driven control framework for LLM-based molecular generation. The method allows for local and global control of molecular structures through editing latent representations aligned with chemical concetps, shifting interpretability from post-hoc explanations to controllable generation. Experiments are designed to assess how SpaRE can generate molecules while respecting real-world molecular constraints, and at the same time yielding causal insights for structure-property relationships.

### Strengths
- SpaRE is elegant and (to my knowledge) novel. The idea of using a sparse autoencoders to disentangle LLM token representations into chemical concepts is very intuitive.
- Similarly, representation editing at inference time (using hooks + contrastive directions) is conceptually clean.
- The interpretability analyses (e.g. linear probes) are very interesting and add a strong qualitative layer to the evaluation.
- SpaRE is tested extensively and against a very comprehensive set of baselines, using appropriate metrics.
- Empirical gains are evident from the experiments. The framework also appears particulary computationally efficient during inference.

### Weaknesses
- SpaRE requires training a SAE for each transformer layer. The computational overhead introduced by having to train one SAE per layer is not quantified or discussed critically. Please quantify computational overhead.
- Claims like "causal" or "mechanistic interpretability" are overstated. It's hard to assess causal relationships when the method relies on proxy metrics or heuristics (e.g., QED/SA) rather than ground-truth assays. Moreover, a proper causal assessment would require interventions or counterfactuals. I think these claims must be toned down accordingly.
- I am a bit skeptical of the LLM self-interpretation part. First, there is a risk of circularity (an LLM evaluating an LLM). Second, a precise sparse feature-chemical rule mapping (as done by the LLM) assumes that chemical concepts are completely disentangled, which cannot be assessed. I would frame this part a little bit lightly, this can be a rough indicator about what the features hint at, but not a mechanistic interpretation of the feature.
- In general, I find that this paper is over-indulgent with the merits of the SpaRE (there are some indeed, especially in the way it is formulated) while it forgets/forgives obvious limitations (such as reliance on noisy predictors) to oversell the interpretability part.

### Questions
See above

### Soundness
2

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
4

### Summary
The authors introduce SpaRE, an interpretability-driven framework for precise control of LLM-based molecule generation. The approach trains sparse autoencoders (SAEs) on hidden states from each transformer layer to obtain overcomplete, sparsely activated latent features that are interpreted as “concepts” (atoms, functional groups, or global properties). Control is applied at inference time by editing activations: (i) local control via forward hooks at a target token step to amplify/suppress features for a desired atom/group, and (ii) global property control via a contrastive direction between exemplar sets with and without the target property. The method claims strong controllability and efficiency on real-world tasks while showing high validity and site-specific success.

### Strengths
* Practical control: Clear mechanisms for local and global control; hook-based token editing is simple to implement.
* Strong results: On ChEBI‑20 site-specific generation, SpaRE achieves 100% validity and ~99% success in the provided table, with improved efficiency.
* Interpretability angle: Explicit attempt to move from post‑hoc analysis to actionable control using SAEs.

### Weaknesses
* Concept validity: Limited evidence that SAE features are monosemantic; need quantitative probes (e.g., linear-probe upper bounds, concept disentanglement metrics) rather than thresholded heatmaps. 
* Baselines and fairness: For Table 1, discuss effects of differences in decoding, tokenization, and constraints across baselines (where applicable); Missing confidence intervals. 
* Generalization & safety: Token-level edits might interact unpredictably with longer-range constraints (valence, stereochemistry). Missing analyses on unseen scaffolds and hard constraints (e.g., ring systems, chirality).  
* Global control construction: The contrastive exemplar approach risks leakage without careful curation; How positives/negatives are selected? Where the direction stable across seeds/datasets?

### Questions
* Figure 2: In both plots, it looks like the displayed features exists in almost all the samples. Why is that?
* Can you quantify monosemanticity (e.g., via linear probes that estimate upper bounds on concept recoverability) and report how close SAE features get to that bound? 
* How sensitive are results to the SAE expansion factor m, the number of edited layers, and the per-layer edit magnitude? Please include ablations.  
* For local edits, how do you handle multi-token fragments or tokens whose identity depends on context (e.g., aromaticity markers)?  
* What makes the validity 100%? Editing of local change is expected to generate sometimes invalid molecules. What prevents it?
* Beyond RDKit proxies, can you provide external validations (e.g., docking, synthesis plans) for a subset of generated molecules to corroborate property control?
* Which layers did you choose when editing? How did you choose those?
* How is `forwardhook(z_t0)` learned? How many examples are required? 

Missing relevant citations:

[1] Improving Small Molecule Generation using Mutual Information Machine [Reidenbach et al, 2023]

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces SpaRE, an interpretability-driven framework for precise, controllable molecule generation using LLMs at inference time. The core problem it addresses is that LLMs, while powerful, have dense and entangled representations that make fine-grained control difficult. SpaRE tackles this by training a SAE on an LLM's internal activations to disentangle its representations into a large, overcomplete set of sparsely activated, chemically meaningful "concepts". The framework enables two types of control: Local Control and Global Control. The authors demonstrate that SpaRE achieves SoTA success rates on controlled generation tasks, can be integrated with Monte Carlo Tree Search (MCTS) for complex multi-objective optimization, and that its disentangled features are genuinely interpretable, as validated by both linear probes and semantic analysis from an external LLM.

### Strengths
1. Comprehensive experiments: The method's empirical results are a significant strength, which is comprehensive and achieves state-of-the-art results.
2. Actionable Interpretability: The paper successfully moves beyond post-hoc interpretability to actionable interpretability. The validation of this is strong, particularly the linear probing experiment.
3. Computational Efficiency: Because SpaRE is an inference-time method that does not require retraining or fine-tuning the base LLM, it is extremely fast. 
4. Effectiveness in Complex Tasks: The paper demonstrates that SpaRE is not just a toy. By combining it with MCTS, it successfully solves complex, multi-objective optimization problems, such as designing dual kinase inhibitors under four constraints and oral drugs under eight constraints, achieving high success rates and superior property distributions.

### Weaknesses
1. Generalizability of Learned Concepts: The SAE is trained on activations from a fixed corpus of molecules. The paper does not extensively test whether the learned "concepts" are universal or if they would generalize to controlling properties or chemical motifs that are far out-of-distribution from the SAE's training data. This is important, especially when considering the world knowledge of LLMs, which in my opinion, is the most superiority of LLM-based molecule generative methods compared to point-cloud based or 3D graph-based ones.

2. Scalability of Feature Interpretation: While the LLM-based interpretation of features is a novel validation step, it seems to be not scalable. E.g., the SAE has 40,960 latent features, and only a handful are manually interpreted. It is not guaranteed that a majority of these features are as semantically clean or human-interpretable as the chosen examples. Can the authors give a deeper insight of this?

### Questions
See Weakness. I will consider raising my scores if the questions or misunderstandings are properly addressed.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces SpaRE, a two-fold controllable molecular generation framework. In detail, it enables two controls: 1) local control achieves substructure generation with retrieval and 2) global control adjusts overall molecular properties with contrastive guiding samples. The framework leverages sparse autoencoders (SAEs) to disentangle LLM representations into chemically meaningful latent features.

### Strengths
- The paper presents comprehensive experiments and analysis.
- The integration of sparse autoencoders for latent disentanglement provides both interpretability and controllability.

### Weaknesses
- **Lack of clarity in task formulation**: The site-specific *molecular generation* task (Section 4.1) is poorly defined. It remains unclear how this task is constructed and why it is practically meaningful. Does Table 1 evaluate whether the local control mechanism performs as intended? Moreover, in realistic molecular design workflows, specifying the target site *a priori* is uncommon, raising concerns about the task’s practical relevance.
- **Choice and justification of baselines**: It seems expected that general-purpose LLMs would underperform on this highly specialized task (e.g., site-specific molecule generation), as their architectures are not designed for such control. The authors should justify the baseline selection.
- **Insufficient methodological and experimental details**: The manuscript omits implementation details, including how molecules are tokenized, which molecular representation  is used (e.g., SELFIES, SMILES), and which LLM backbone underlies SpaRE. These omissions hinder reproducibility and limit the understanding of the proposed method.
- **Unclear figures and metrics**: Figures 2, 5, and 9 lack axis descriptions and contextual explanations. It is not evident what each axis represents or how latent features correspond to interpretable chemical properties. In Figure 5, while the authors claim superior performance, the visualized scores (e.g., for JNK3) do not clearly demonstrate consistent improvement.

### Questions
I have delivered all my questions in the weakness.

### Soundness
3

### Presentation
1

### Contribution
2
