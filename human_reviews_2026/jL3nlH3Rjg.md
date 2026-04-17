# Human-in-the-Loop Targeted Molecule Design Informed by Transcriptomes

- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
Transcriptome-guided targeted molecule design contributes to the advancement of precision medicine. However, the fragmented and poorly integrated nature of existing datasets limits the ability of transcriptome-aware approaches to meet the design expectations of chemists, often requiring additional expert intervention after molecular generation. To address this gap, we propose HiTGen, a human-in-the-loop targeted molecule generation framework informed by transcriptomes. Specifically, HiTGen operates in two phases: i) Transcriptomes are served as the central biological driver and are fused with expert a posteriori knowledge via a tailored bidirectional attention mechanism, enabling biochemically grounded guidance for diffusion-based generation of molecule candidates. ii) An expert-guided human-in-the-loop optimization mechanism is employed by HiTGen to refine these candidates toward desired molecular targets. Extensive experiments demonstrate that HiTGen consistently outperforms state-of-the-art models across ten evaluation metrics, yielding chemist-aligned targeted molecules with potential for precision medicine. The code will be released upon acceptance of the paper.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The manuscript proposes HITGEN, a framework for targeted molecule design. HITGEN integrates transcriptomic information and molecular descriptions, where both modalities are transformed into a latent space via encoders, followed by a decoder for molecule generation. The authors further introduce a human-in-the-loop architecture based on DrugAssist to enable fine-grained molecule optimization through designed prompting. Experimental results demonstrate that HITGEN outperforms baseline methods across multiple evaluation metrics.

### Strengths
- The manuscript presents an interesting attempt to combine multimodal molecular representations with a human-in-the-loop optimization strategy.
- The experimental results indicate improved performance compared to baseline methods.

### Weaknesses
- Writing clarity: Certain descriptions are unclear. For instance, in the sentence “Given a natural language description F that reflects the chemical specification associated with the O”, the variable F is not explained. The illustration in Figure 1 is also confusing — Step 1(b) and Step 3 both appear to involve human-in-the-loop processes, making the workflow unclear.
- Limited novelty: Although the manuscript highlights the human-in-the-loop component as a key contribution, this part primarily builds on the existing DrugAssist framework, which limits the originality of the proposed approach.
- Unclear design choices: Some methodological decisions lack sufficient justification. For example:
    - Why does the model use discrete tokens in the latent space instead of continuous representations?
    - In the semantic embedding encoder, why is the masking applied only to tokens lacking chemical relevance?

### Questions
- In the semantic embedding encoder, why is masking applied only to tokens that lack chemical relevance?
- How does the model perform without the DrugAssist based optimization module?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes HITGEN, a framework for designing molecules using transcriptome profiles plus textual descriptions. The approach uses BioT5 to generate molecular descriptions, combines transcriptomic and text representations by attending across them, generates candidates with a diffusion model, and then optimizes them using DrugAssist with a fallback strategy. The authors report improvements over baselines across multiple similarity metrics on transcriptomic datasets from cancer cell lines and disease states.

### Strengths
* The combination of transcriptomic data with textual molecular descriptions for conditional generation is a reasonable extension of existing work. The bidirectional cross-attention mechanism for fusing heterogeneous biological and chemical information is sensible, though not particularly novel.
* If the results are valid, incorporating transcriptomic context into molecular design could be valuable for precision medicine applications.

### Weaknesses
* The fallback strategy in Section 3.3 creates a fundamental evaluation flaw. By selecting between the candidate and optimized molecule based on which has better metrics, the method is effectively performing best-of-2 selection using the test metrics themselves over multiple rounds. This is circular reasoning and an unfair comparison to baselines that generate single molecules.
* The paper repeatedly emphasizes "human-in-the-loop" but provides no evidence of actual human involvement during evaluation or iterative human feedback during training. The "expert guidance" appears to be: using BioT5 to automatically generate molecular descriptions from SELFIES representations once during dataset creation, and using DrugAssist (an LLM) with pre-written prompts for optimization. This is automated post-processing, not human-in-the-loop interaction. The terms "expert-guided," "human-in-the-loop," and "chemist-aligned" are used interchangeably without clear operational definitions.
* Writing quality: The paper contains substantial filler text that obscures key information. Example from Section 3.2: "This module explicitly models reciprocal, bi-directional attention between information to allow each stream to contextually recalibrate itself with respect to the other. This fosters a dynamic, fine-grained feature refinement process, which is essential for capturing subtle dependencies." This is three sentences describing standard cross-attention. Many critical details are relegated to appendices, making the main paper difficult to evaluate. Terms like "biochemically grounded," "chemist-aligned," and "tailored" are used repeatedly without definitions.

### Questions
* BioLinkBERT is pretrained on PubMed abstracts. Doesn't this mean there's going to be some leakage with the test set?
* Please provide explicit details about human expert involvement. During dataset creation, how many molecular descriptions were generated and by whom? What was the validation process? During evaluation, were any actual chemists involved in any capacity, or is all "expert guidance" automated?
* Target variable leakage: The fallback strategy selects molecules based on test metrics (Tanimoto similarity, FCD, etc.). How is this not a form of test set leakage? The evaluation metrics are being used to make generation decisions, then those same metrics are reported as results. Can you justify why this is a fair evaluation protocol?
* How would a chemist actually use this system in practice? What is the intended interaction model? Does the chemist provide the transcriptome and description, or only the transcriptome? If descriptions are auto-generated, what control does the chemist have? How much time would this save compared to manual design processes?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces HITGEN, a framework for targeted molecule generation conditioned on transcriptomic profiles. It integrates transcriptome data with expert chemical descriptions using bidirectional attention, followed by a diffusion-based generator and an iterative “human-in-the-loop” refinement simulated via a DrugAssist language model. Experiments on drug-induced transcriptome datasets show significant gains in validity, novelty, and fingerprint similarity over baselines such as GxRNN, GxVAE, MolT5, and TGM-DLM. Despite strong quantitative results, the extent of genuine human involvement and the biological relevance of the generated molecules remain unclear.

### Strengths
Multimodal Fusion: Combines transcriptomic signals and molecular text using a bidirectional attention mechanism, allowing richer conditioning of molecule generation.  
Comprehensive Pipeline: Proposes a multi-stage architecture including generation, refinement, and fallback control.  
Extensive Metric Evaluation: Reports performance across a diverse set of structural and pharmacological metrics, including ablations.  
Empirical Gains: Achieves high scores on validity, novelty, and similarity metrics relative to baselines.

### Weaknesses
Unsubstantiated Human-In-The-Loop Claim: No real human feedback or user studies are presented; all expert guidance is simulated through LLMs.  
Lack of Methodological Transparency: Core components like the VQ-VAE and diffusion model lack necessary architectural and training details.  
Heavy Reliance on Pretrained Black-Box Models: Performance attribution is unclear due to multiple large components (BioT5, BioLinkBERT, DrugAssist).  
No Biological Validation: Despite the biomedical focus, there is no docking, ADMET, or in vitro assay to support therapeutic relevance.  
Potential Evaluation Bias: HITGEN uses richer input than many baselines (i.e., molecule text + transcript), making direct comparison difficult.

### Questions
1. How exactly was the BioT5 model “supervised” by chemists? Were chemists involved in training, prompt crafting, or post-editing?  
2. Did any actual human experts interact with the system during molecule refinement via DrugAssist?  
3. What are the VQ-VAE's architecture, latent dimensionality, and reconstruction performance on transcript data?  
4. Is the diffusion model trained end-to-end with (transcript + text)-molecule pairs or separately?  
5. How are the reference drugs for similarity optimization selected? Could this introduce a bias toward known compounds?  
6. How frequently does the fallback strategy override the LLM-optimized candidates? What does this say about the optimizer’s effectiveness?  
7. Were any biological properties predicted beyond structural metrics (e.g., binding affinity, ADMET)?  
8. Are the prompts used for DrugAssist available to reproduce the optimization pipeline?  
9. What do the ablation variants “W/o T”, “W/o TC”, and “W/o TT” correspond to?  
10. Can the system generalize to patient-derived or noisy transcriptomic data?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The HITGEN framework proposes a human-in-the-loop, transcriptome-conditioned molecular design system.
Unlike protein–ligand or DTI models that target a single protein, HITGEN aims to design molecules that modulate whole-cell transcriptomic states. The model uses VQ-VAE to encode transcriptome profiles into discrete latent representations and a Transformer-based diffusion model to generate molecules (SELFIES format) conditioned on these biological embeddings. A pretrained language model (BioLinkBERT) interprets human text instructions, such as “increase solubility” or “reduce toxicity, and these text embeddings are fused with transcriptome embeddings via cross-attention, allowing interactive molecule steering during inference. Notably, textual prompts are not used during training; the diffusion model is trained solely on transcriptome–molecule pairs from the LINCS L1000 dataset, and human text inputs act as zero-shot control signals in inference or optimization loops.

### Strengths
* HITGEN consistently outperforms baseline models (TRIOMPHE, GxRNN, GxVAEs, etc.) across validity, uniqueness, novelty, and similarity metrics.
* Cross-attention seems to be effective, and this is validated through the ablation study.
* Excluding the optimization stage led to the largest drop in Morgan/RDK similarity, proving that expert feedback materially enhances pharmacological specificity

### Weaknesses
* Removing human textual guidance significantly reduced diversity and novelty of generated molecules, showing large dependency on external expert cues rather than fully autonomous reasoning
* The ablation description confirms there is no training signal aligning text with transcriptomic data. Text acts only as high-level guidance rather than a learned modality
* Since HITGEN is trained exclusively on MCF7 (breast cancer cell line) transcriptomic data, it may overfit to this specific cellular context, limiting its biological generalization and causing inconsistent performance when applied to other tissues or disease-derived transcriptomes.

### Questions
* Since the model does not use any transcriptome–text–molecule triplets during training, how do you ensure that the cross-attention weights between textual and biological embeddings learn meaningful alignment rather than acting as untrained placeholders?

* Given that the training data are limited to the MCF7 cell line, have you evaluated or considered domain adaptation methods to ensure that the learned transcriptome–molecule mapping generalizes to other cell types or patient-derived transcriptomes?

### Soundness
2

### Presentation
2

### Contribution
2
