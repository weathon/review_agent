# SpectraLLM: Uncovering the Ability of LLMs for Molecule Structure Elucidation from Multi-Spectra

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 8

## Abstract
Automated molecular structure elucidation remains challenging, as existing approaches often depend on pre-compiled databases or restrict themselves to single spectroscopic modalities. Here we introduce **SpectraLLM**, a large language model that performs end-to-end structure prediction by reasoning over one or multiple spectra. Unlike conventional spectrum-to-structure pipelines, SpectraLLM represents both continuous (IR, Raman, UV-Vis, NMR) and discrete (MS) modalities in a shared language space, enabling it to capture substructural patterns that are complementary across different spectral types. We pretrain and fine-tune the model on small-molecule domains and evaluate it on four public benchmark datasets. SpectraLLM achieves state-of-the-art performance, substantially surpassing single-modality baselines. Moreover, it demonstrates strong robustness in unimodal settings and further improves prediction accuracy when jointly reasoning over diverse spectra, establishing a scalable paradigm for language-based spectroscopic analysis.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SpectraLLM, a large language model designed for molecular structure elucidation using multiple spectra. The model tries to address limitations of single modality approaches by unifying diverse spectroscopic data into a shared language space. The method works by converting spectral peak features into natural language prompts. It then uses a frozen Qwen3 backbone adapted with LoRA to autoregressively generate SMILES strings. The authors evaluate this on four public datasets. The results show state-of-the-art performance and demonstrate that combining spectra improves prediction accuracy.

### Strengths
1. The experimental evaluation is comprehensive. The model was tested on four large-scale datasets, which include both simulated and experimental data.

2. SpectraLLM shows strong performance. It consistently surpasses specialized baselines even in single modality tasks. The improvement over the NMR2Struct baseline in Table 2 is particularly notable.


3. The analysis in Appendix A.4 provides a good justification for the chosen architecture. The comparison against a vision language model pipeline confirms the benefit of the purely language-based representation for this task.

### Weaknesses
1. There is a slight concern about the dependency on the upstream peak extraction process. The model performance seems tied to the quality of this peak list. This step could be an information bottleneck if important signals are missed or noise is included from the raw spectra.


2. While the model achieves state-of-the-art results, the absolute performance metrics suggest the task remains very difficult. For instance, the best Tanimoto score in Table 4 is approximately 0.4875. This indicates significant room for future improvement.

3. The paper mentions including experimental metadata in the prompts, such as collision energy. This is a good idea, particularly for mass spectrometry. However, the paper might benefit from a more direct analysis or ablation study showing the specific impact of this metadata on performance

### Questions
1.  Could the authors please comment on the robustness of SpectraLLM to variations in the peak picking step? For example, how much do errors or changes in peak detection thresholds affect the final SMILES generation?

2.  The Tanimoto scores are strong relative to baselines, but the task is clearly not solved. In the authors' view, what is the primary remaining bottleneck? Is it the spectral representation, the reasoning capacity of the language model, or the inherent ambiguity in the spectra?

3. The paper notes the inclusion of experimental conditions. Could the authors provide a brief analysis of the specific contribution of this metadata, especially collision energy for MS, to the model's accuracy?

4. The model is trained on a combination of simulated and experimental datasets. Was any analysis done on generalization? For instance, how well does a model trained only on simulated data perform on the experimental test sets?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents SpectraLLM, a language-based framework for molecular structure prediction from spectroscopy. The key design is to transform spectral peaks (IR, Raman, UV-Vis, NMR, MS) into compact textual tokens that encode physical attributes (e.g., position, intensity, annotations). The LoRA-tuned LLM, SpectraLLM, then autoregressively generates SMILES from one or multiple spectra. This unified textual interface allows the model to combine heterogeneous modalities in a shared semantic space and to perform joint reasoning across signals. The authors aggregate several public datasets, enforce molecule-identity splits, and evaluate both unimodal and multimodal settings using chemically meaningful metrics (e.g., ECFP/MACCS Tanimoto, MCES, functional-group overlap). They report consistent gains when fusing modalities and stronger NMR performance compared with prior systems. Qualitative cases illustrate when specific modalities add value.

### Strengths
* By transforming peaks into tokens that capture physical attributes, the method places heterogeneous spectra in a shared semantic space. This is a simple, scalable way to enable joint reasoning over multi-spectral inputs.
* Fusing modalities improves similarity metrics and structural correctness over the best single-modality baselines; the NMR setting shows clear margins.
* Parameter-efficient LoRA on a general LLM, near-perfect SMILES validity, and clear data preprocessing make the approach straightforward to reproduce in principle.
* Multiple datasets and metrics (beyond exact match) provide a more realistic view of structure recovery.

### Weaknesses
* The core idea—transforming peaks into textual tokens and fine-tuning an existing LLM—is incremental relative to existing generative pipelines. The paper does not introduce fundamentally new learning principles or solutions for spectroscopy-to-structure mapping. This is the major concern regarding the work’s originality and contribution to the ICLR community.

* No public code or model weights are provided. This limits reproducibility and the community’s ability to validate and extend the work.

* Results may be sensitive to peak detection criteria, binning, token order, and inclusion of side metadata. The paper does not quantify these effects.

* The dataset mix contains substantial simulated spectra and small molecules. Transfer to challenging experimental conditions (noise, matrix effects, mixture spectra) is not fully assessed.

* Training scale, wall-clock, and inference latency versus specialized models are not reported.

### Questions
1. How sensitive are results to the design of spectral tokenization (e.g., peak thresholds, binning, ordering, or inclusion of metadata)?

1. Do the authors plan to open-source code, model checkpoints, and prompt templates?

1. Can you provide evaluation on experimental spectra (e.g., MassBank IR/NMR) to confirm real-world robustness?

1. How does the model perform under top-(k) decoding or diverse sampling?

1. Are stereochemical correctness and formula consistency checked after decoding?

1. What are the compute requirements for fine-tuning and inference compared to modality-specific models?

### Soundness
3

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
4

### Summary
The paper presents SpectraLLM, an LLM fine-tuned to predict molecular structures given spectra of one or more spectroscopic modalities. The authors compile a new multi-modal dataset of different types of spectra and use it to fine-tune Qwen3 LLM using LoRA. The method is shown to outperform existing methods in both uni-modal and multi-modal settings.

### Strengths
- The paper addresses an important problem of molecular structure annotation from spectroscopic data modalities which remains very challenging to solve
- The authors compile a large new dataset from multiple sources
- The paper is well written and structured
- The authors leverage Fraggle similarity from RDKit, which to the best of my knowledge was not explored in related work before

### Weaknesses
Major concerns

- Most of the collected dataset appears to contain relatively small molecules (as shown in Figure 2). Could the authors provide examples of practical use cases where annotation of such small molecules, as opposed to larger natural products or drug-like molecules, is of interest? Without some EDA of the molecular composition and an explanation of when this setting is relevant, it is difficult to assess the utility of the work, especially considering that the methodological novelty is limited (standard LLM fine-tuning). Please also elaborate in what practical scenarios the annotation of molecules simultaneously from multiple types of spectra is relevant.
- The baselines (e.g., DiffMS) do not seem to be retrained and appear to be used out of the box. Please clarify this in the paper if is not true, or retrain the baselines on the same data splits if is true. Otherwise, the results do not seem to be valid.
- The results on MassSpecGym do not seem consistent with the MassSpecGym benchmark: the reported metrics differ, and the Tanimoto score for DiffMS does not match the published result. Please retrain the model using the official MassSpecGym data split. Without such an experiment, it is unclear whether the proposed method actually outperforms DiffMS.
- The evaluation may suffer from data leakage. Please elaborate on the splitting strategy and the degree of similarity between molecules in the training and testing sets. The paper currently mentions a “molecular identity”-based data split, which is known to result in leakage even when stereochemistry is ignored (see, for example, Figure 2 in the MassSpecGym paper https://arxiv.org/abs/2410.23326).
- The goal of annotating molecules from spectra is to discover novel molecules that have not been characterized before. The LLM-based approach may be prone to data leakage if it is already familiar with test-set molecules. Could the authors evaluate to what extent the model is familiar with these molecules before fine-tuning? Please note that non-LLM models (e.g., DiffMS) are not prone to this potential problem.

Minor concerns

- Table 1: Please note that isotopic distributions are typically present in MS1 spectra, not in MS/MS spectra, which this paper does not work with
- Lines 066–071: These statements are not well justified. One could argue the opposite: working with precise continuous values is more important than discrete tokenised “low-resolution” values, as spectroscopic signals often contain information in decimal places, which may be hard to model with a tokenized LLM approach
- Line 152: Please note that MassBank is part of, or substantially overlaps with, MassSpecGym
- Equation 3: Please define phi
- Lines 176–179: Please elaborate on how experimental conditions and metadata are incorporated
- Figure 2: It is not clear what the checkmarks and boxes indicate
- Lines 879–884: It is unclear what A and B represent

### Questions
- Can the authors explain the practical use cases where their model could be readily applied?
- Can the authors better evaluate data leakage and elaborate on the data used to retrain the baselines?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces SpectraLLM, a large language model fine-tuned via LoRA to perform end-to-end molecular structure elucidation from diverse spectroscopic inputs (IR, Raman, UV-Vis, NMR, and MS). It reformulates spectral data as natural language prompts to enable symbolic reasoning within a shared semantic space. Using datasets like QM9Spectra, Multimodal Spectroscopic, MassSpecGym, and MassBank, SpectraLLM achieves state-of-the-art unimodal and multimodal performance, outperforming domain-specific baselines across structural and functional similarity metrics.

### Strengths
There is novelty in this work. Specifically, it converts continuous and discrete spectra into text, enabling a unified reasoning space for multiple spectroscopy types.

There are also strong empirical results in which consistent and large gains over baselines are shown across IR, Raman, NMR, and MS, with near-perfect validity and robustness to missing modalities.

Furthermore, there is comprehensive evaluation where there are benchmarks on multiple datasets with diverse metrics (Tanimoto, MCES, Fraggle, etc.) and insightful qualitative analyses

### Weaknesses
While the model performs well, it remains unclear how linguistic embeddings capture physical–chemical semantics. The authors need to improve the interpretability component of their model. 

There is also heavy reliance on synthetic spectra (QM9s, Multimodal Spectroscopic) may limit real-world generalization. Ultimately, these are simulated data and the authors should show model performance on real world datasets. 

There is also no discussion of common failure modes, ambiguity cases, or invalid SMILES beyond aggregate metrics. The authors should provide more error analysis and better discuss the limitations of their work. 

Several critical recent works are also omitted from reference in this paper:

- Le, Khiem, et al. “MolX: Enhancing Large Language Models for Molecular Learning with a Multi-Modal Extension.” KDD MLoG-GenAI 2025. (Include as a key related multimodal-LLM baseline/approach.)

- Fang, Junfeng, et al. “MolTC: Towards Molecular Relational Modeling in Language Models.” Findings of the Association for Computational Linguistics: ACL 2024, Association for Computational Linguistics, 2024, pp. 1943–1958. 
ACL Anthology

- Ju, Jiaxin, et al. “Uni-MRL: Unified Multimodal Molecular Representation Learning with Large Language Models and Graph Neural Networks.” Advances in Knowledge Discovery and Data Mining (PAKDD 2025), LNCS 15874, Springer, 2025, pp. 275–287. 
SpringerLink

### Questions
How sensitive is performance to the accuracy of the spectral-to-text mapping (ϕ)?

Could the model generalize to unseen experimental conditions or instruments?

Are there ablations showing which spectral modality contributes most to multimodal gains?

### Soundness
3

### Presentation
3

### Contribution
3
