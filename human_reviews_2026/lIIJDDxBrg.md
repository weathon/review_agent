# Empowering Protein Language Model for Sequence-Structure Co-Generation with Continuous Structure Tokens

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Proteins inherently possess a consistent sequence-structure duality. The abundance of protein sequence data, which can be readily represented as discrete tokens, has enabled fruitful developments in protein language models (pLMs). A key remaining challenge, however, is how to effectively integrate continuous structural knowledge into pLMs. Current methods often discretize protein structure to accommodate the language modeling framework, which inevitably results in fine-grained information loss and limits the performance potential of multimodal pLMs. In this paper, we argue that such concerns can be circumvented: a sequence-based pLM can be extended to incorporate the structure modality through continuous tokens, i.e., high-fidelity protein structure latents that avoid vector quantization. Specifically, we propose a hybrid diffusion protein language model, HD-Prot, which embeds a continuous-valued diffusion generation head atop a discrete pLM, enabling seamless operation with both discrete and continuous tokens for sequence-structure joint-modeling in multimodal generative pLMs. The proposed model captures inter-token dependencies across modalities through a unified absorbing diffusion process, and estimates per-token distributions via categorical prediction for sequences and continuous diffusion for structures. Extensive empirical results show that our models achieve competitive performance in unconditional sequence–structure co-generation, motif-scaffolding, protein structure prediction, and inverse folding tasks, performing on par with state-of-the-art multimodal pLMs despite being developed under limited computational resources. It underscores the viability of jointly modeling discrete categorical and continuous arbitrary distributions using shared parameters within a pLM, pointing to an alternative and promising direction of progress for multimodal pLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces HD-Prot, a hybrid diffusion protein language model that uses continuous (non-quantized) structure tokens instead of the discrete tokens employed by existing methods like ESM3 and DPLM-2. The approach extends sequence-based protein language models to jointly model discrete sequence tokens via categorical prediction and continuous structure tokens via diffusion modeling, unified through an absorbing diffusion process. The authors evaluate performance on protein sequence-structure co-generation and motif-scaffolding tasks, and provide ablation studies examining design choices including fine-tuning strategies and classifier-free guidance for continuous tokens.

### Strengths
* The information loss problems in existing discretization approaches are clearly articulated, providing motivation for exploring continuous structure tokens in protein language models.

* The paper is well-structured  and clearly presented.

* Table 3 provides valuable ablation findings on the importance of pre-trained foundation models, different fine-tuning strategies for different model scales, and the role of classifier-free guidance.

* The work successfully shows that continuous structure tokens can work within protein language models, opening a new research direction even if not demonstrably superior to existing methods.

### Weaknesses
* The work is fundamentally incomplete. Table 5 lists 5 protein modeling tasks that multimodal pLMs should handle (sequence generation, structure generation, structure prediction, inverse folding, co-generation), yet the paper evaluates only 2 tasks. This severely limits assessment of the approach's capabilities and fails to deliver on the promised comprehensive multimodal modeling.

* The small sample sizes undermine all conclusions. There is no justification provided for the inadequate size of the evaluation batch of 100 proteins. There are no error bars, confidence intervals, or significance tests. No multiple runs or random seeds reported. These numbers are far too small to draw reliable conclusions about model performance or make comparisons between methods.

* Missing obvious and critical baselines. The need for multimodality is not shown. Without proving superiority over simpler models, the value proposition of a complex multimodal architecture remains unsubstantiated. The authors should introduce basic baselines to directly test whether the complex co-generation setup provides any benefit over simpler unimodal approaches. For example, a basic pipeline like "Generate unconditional structure → ProteinMPNN (8×) → ESMFold → measure scRMSD/Diversity/Novelty". Also there are lots of unimodal models to compare to. Also, there is no comparison to the structure tokenizer alone: why build a language model on top if the tokenizer itself might generate comparable structures?

* The approach primarily adapts existing diffusion language modeling frameworks with continuous tokens. Given this limited novelty, the paper requires substantially more ablation studies and insights, yet critical hyperparameters remain completely unexplored. For example, there are no ablation of lambdas and gamma (Eqs. 8-10), no systematic analysis of τz (structure sampling temperature) despite claims it controls results, diffusion schedule parameters (T, T', βt') are completely unjustified.

* The evaluation metrics are poorly designed. #Cluster threshold=0.5 for the diversity  only detects highly different proteins; need e.g. 0.95 to detect mode collapse. Inner-TM is meaningless, the full distributions like Figure 6 should be drawn. The Novelty is measured against PDB/SwissProt instead of training set, so it doesn't really detect memorization. Length-dependent metric, scRMSD, averaged across different lengths obscures true performance. Also, Figure 3A-B compares to PDB, not the training set; different models trained on different data; sample size is not reported.

* Even with these metrics, the paper only demonstrates continuous tokens are viable, not superior. So, the claims of "highly competitive performance" are misleading.

* "Information loss" claims are fundamentally misleading, the paper conflates "continuous representation" with "no information loss". Compression from L×4×3 coordinates to L×dstruct is still lossy compression. "Continuous" does not equal "lossless",  this is dimensionality reduction regardless of discretization. By the way, the actual dstruct dimension not specified in main text.

* Motif-scaffolding evaluation is poor. The authors should have used the MotifBench benchmark, not the outdated RFDiffusion benchmark. The procedure description (line 819) vague: how exactly are motif tokens "maintained unchanged"? CFG improves consistency+diversity (Table 6) but not used here - why?

* The proposed approach is fundamentally unscalable. Table 3 reveals critical problems: training from scratch completely fails (row 1: pLDDT 73.100), full fine-tuning of the 650M model fails with sequence knowledge collapse (row 7: pLDDT 73.455), and the approach is fundamentally constrained to fine-tuning small pre-trained models with limited data. This severely limits future development and undermines the approach's viability.

* The proposed approach seems to be brittle and hyperparameter-sensitive. For example, different model sizes require completely different strategies (full fine-tuning vs. LoRA), different optimal temperatures needed (τz=0.35 vs 0.65), CFG required for decent performance.

* No failure mode analysis. Zero discussion of when/why approach fails, problematic protein types, or error patterns. Length-dependent degradation (Figure 6) attributed to "less training data" but never investigated.

* Generalization claims are unsubstantiated. Figure 3C shows 600-700 residue examples but no quantitative evaluation despite training limited to ≤512 residues.

### Questions
* What is the actual mutual information or reconstruction fidelity comparison between continuous structure tokens (L×dstruct) and discrete tokens (VQ-VAE codebooks)? Can you quantify the information preservation advantage?

* What happens when you encounter proteins that the salad tokenizer reconstructs poorly (scRMSD>1.0 or scTM<0.9)? Can HD-Prot handle these cases, or is it fundamentally limited by tokenizer quality?

* Why was classifier-free guidance not used in motif-scaffolding if it improves both self-consistency and diversity (Table 6)?

* In motif-scaffolding (line 819, Appendix C.2), how exactly are motif tokens "maintained unchanged" during the iterative generation process? Does this apply to both sequence and structure tracks simultaneously?

* Given that HD-Prot is built on DPLM, which is built on ESM-2, how much of the performance can be attributed to the foundation model versus your contributions?

### Soundness
1

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
3

### Summary
This paper presents HD-Prot, a hybrid diffusion protein language model that jointly generates protein sequences and structures using continuous structure tokens instead of quantized ones. By integrating a non-quantized autoencoder (salad) and a unified diffusion framework operating over both discrete (sequence) and continuous (structure) modalities, HD-Prot avoids information loss from discretization. Experiments show that it achieves competitive sequence–structure co-generation and motif-scaffolding performance relative to state-of-the-art multimodal pLMs (e.g., DPLM-2, ESM3) while requiring far fewer computational resources.

### Strengths
1. **Continuous structure tokens** – High-fidelity structure representation without quantization loss (scRMSD ≈ 0.39 Å).
2. **Unified diffusion modeling** – Seamlessly bridges discrete and continuous modalities within one LM.
3. **Compute-efficient design** – Matches large multimodal pLMs’ performance using only a single-GPU setup.

### Weaknesses
1. The model does not outperform existing co-design approaches on most benchmarks presented, suggesting there is room for further improvement.

### Questions
1. Did the authors train the protein structure autoencoder themselves, or was it directly adopted from prior work? If trained in-house, was it trained on the same **PDB + AFDB-SwissProt** dataset as used for HD-Prot?
2. Can the model also perform **structure prediction** and **inverse folding** tasks? If so, it would strengthen the work to include analyses comparable to those presented in **DPLM-2**.

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
4

### Summary
This paper introduces HD-Prot, a hybrid diffusion protein language model designed to address a key limitation in existing multimodal protein models: the information loss caused by discretizing continuous 3D protein structures. Instead of using discrete tokens, HD-Prot represents protein structures as high-fidelity continuous structure tokens generated by a non-quantized autoencoder. The model extends a sequence-only protein language model to jointly process both discrete amino acid sequence tokens and these new continuous structure tokens. The HD-Prot leverages a unified absorbing diffusion process (masking and denoising) to capture dependencies at the inter-token level between both the sequence and structure modalities. For the sequence and structure prediction, the HD-Prot employs categorical prediction for the discrete sequence and MAR-like continuous diffusion for the structure tokens. Experiments demonstrate that HD-Prot achieves competitive performance on par with state-of-the-art multimodal protein language models in unconditional sequence-structure co-generation and motif-scaffolding tasks.

### Strengths
1. Proposing novel approach HD-Prot, which uses continuous structure tokens and circumvents the need for vector quantization, to avoid structural information loss for multimodal protein language models.
2. The proposed hybrid diffusion framework effectively bridges the discrete-continuous modality gap, where a unified backbone is used to process the inter-token correlation and separate categorical and denoising prediction heads for sequence and structure, respectively.
3. Despite its low training cost, HD-Prot achieves results that are competitive with or on par with state-of-the-art models in unconditional co-generation and motif-scaffolding tasks.

### Weaknesses
1. The experiments are limited to co-generation and motif-scaffolding, lacking the other representative conditional generation tasks such as folding and inverse folding, which are also evaluated by other multimodal pLMs (DPLM-2, ESM-3, DPLM-2.1). This leaves the model's true multimodal capabilities unverified.
2. For the unconditional co-generation experiments, the authors should provide a comparison against the recent work La-Proteina [1].
3. The authors argue that a fundamental limitation of structure quantization is information loss of fine-grained structural details, and they leverage continuous structure tokens to address this. In Table 2, HD-Prot achieves superior designability compared to ESM3 and DPLM2. However, this result cannot directly substantiate the advantage of continuous structure tokens. A critical factor is that HD-Prot, ESM3, and DPLM2 utilize different structure tokenizers. Therefore, the performance discrepancy might originate from the tokenizers themselves, rather than from the distinction between continuous and discrete representations.
4. Regarding motif-scaffolding, although the method solves a greater number of problems than other baselines, its average success rate is lower than DPLM-2. Furthermore, the success rate for several specific problems is extremely low (e.g., 5IUS: 1/100, 5WN9: 1/100, 4ZYP: 2/100). This suggests that the reported number of solved problems may be influenced by stochasticity rather than representing a robust capability.

[1] La-Proteina: Atomistic Protein Generation via Partially Latent Flow Matching

### Questions
See above weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the limitation of information loss in quantized protein structure representations for multimodal protein language models (pLMs) by proposing HD-Prot, a hybrid diffusion framework that integrates continuous structure tokens with discrete sequence tokens. HD-Prot uses a non-quantized autoencoder (salad) to generate high-fidelity continuous structure tokens, avoiding vector quantization (VQ)–induced information loss. It employs a unified masked diffusion process for inter-token dependency modeling and separate intra-token learning: categorical cross-entropy for discrete sequences and diffusion loss for continuous structures. Experiments on unconditional sequence-structure co-generation and motif-scaffolding show HD-Prot matches state-of-the-art models (e.g., DPLM-2) while using fewer computational resources. The work argues that continuous structure tokens offer a viable alternative to quantized representations for multimodal pLMs.

### Strengths
1. The paper identifies a critical gap in existing multimodal pLMs caused by quantized structure tokens, and introduce the hybrid diffusion framework unifies discrete sequence and continuous structure modeling via shared pLM backbones, and the use of a pre-trained non-quantized autoencoder (salad) ensures high structural reconstruction fidelity. An idea validated in vision(-language) models but under-explored in protein modeling.
2. Qualitative results further validate that HD-Prot generates proteins with amino acid and secondary structure distributions similar to natural proteins, supporting its biological plausibility.

### Weaknesses
1. Major: The work only evaluates two tasks (unconditional co-generation, motif-scaffolding) but ignores key protein modeling tasks that test practical utility, such as structure prediction (from sequence), inverse folding (from structure to sequence), or maybe structure-aware predictive downstream tasks. SOTA models like ESM, DPLM-2 or SaProt are validated across these tasks, so HD-Prot’s omission introduces concerns about its broader applicability.
2. No comparison to pure continuous diffusion based multimodal pLMs, such as DiMA [1]. 
3. Minor: Some parts of the manuscript seems similar in wording and structure to DPLM-2 (Wang et al., 2024b). Better rephrase and reorganize a little bit. 

-- 

[1] Diffusion on language model encodings for protein sequence generation. In ICML 2025

### Questions
1. The paper highlights the learning of inter-modal interaction where HD-Prot fuses sequence and structure tokens via element-wise summation (Equation 7), but the paper provides no mechanistic explanation of how this fusion captures cross-modal dependencies (e.g., how sequence embeddings guide structure diffusion or vice versa). There is no ablation of fusion methods (e.g., concatenation, cross-attention) to validate that element-wise summation is optimal, nor analysis of attention weights to confirm that the model actually leverages both modalities during generation.
2. See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
