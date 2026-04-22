# MSpecTmol: A Multi-Modal Spectroscopic Learning  Framework for Molecular Structure Identification

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 8, 2, 2

## Abstract
Spectroscopic techniques are indispensable for the elucidation of molecular structures, particularly for novel molecules with unknown configurations. However, a fundamental limitation of any single spectroscopic modality is that it provides an inherently circumscribed and fragmented view, capturing only specific facets of the complete molecular structure, which is often insufficient for unequivocal and robust characterization. Consequently, the integration of data from multiple spectroscopic sources is imperative to overcome these intrinsic limitations and achieve a comprehensive and accurate structural characterization. In this work, we introduce \textbf{MSpecTmol}, a novel \textbf{M}ulti-modal \textbf{Spec}trum information fusion learning framework for \textbf{Mol}ecule structure elucidation. By extending information bottleneck theory, our framework provides a principled and adaptive approach to fusing spectra. It designates a primary modality to extract core molecular features while leveraging auxiliary inputs to enrich the representation. To validate the end-to-end effectiveness of our framework, we design a two-fold evaluation: molecular substructure classification to probe its discriminative power in identifying substructures, and extends this knowledge to reconstruct plausible 3D structures. Our results not only demonstrate state-of-the-art performance in molecular substructure classification but also achieve near-experimental accuracy (\textasciitilde 0.68\AA) in molecular conformation reconstruction. These findings underscore the model’s capacity to learn interpretable features aligned with chemical intuition, thereby paving the way for future advances in automated and reliable spectroscopic analysis. Our code can be found at \href{https://anonymous.4open.science/r/MspecTmol-6B4D}{https://anonymous.4open.science.}

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The spectroscopic techniques play an important role in structural determination for an unknown molecule. Current challenges. 1) Fully leveraging the available spectroscopic data to extract the physical structure and to make structural determination more accurate; Previous methods only use low-dimensional features and restricted information. Novelties. 1) A primary-auxiliary synergistic modeling approach to capture both core information and supportive information.Results. 1) F1 score of 0.959 on both simulated and experimental spectra; 2) Avg. RMSD 0.682 A on the spectrum-conditioned conformation generation task. 3) Align well with the chemical intuition.

### Strengths
1. This paper has calculated the theoretical upper bounds of each objective as the loss function, which makes the optimization easier.
2. MSpecTmol has designed a specific optimization function for spectrum detection with the two modalities.
3. The experiment results show consistent advantages on selected tasks.

### Weaknesses
1. The motivation of MSpecTmol is not crucial. The key claim is that other models only use individual spectroscopic, which didn't consider models like DffSpectra, SpectraLLM, MMST, and older models such as spec2struct. As a result, the motivation of "using multi-spectra" is not solid here. The main challenge you solve may be "how to integrate multiple spectrums better. In this case, you should compare similar multimodal models. 
2. For the classification tasks, the F1 score is not enough, especially for imbalanced datasets. The paper should show more metrics under this reason, and analyze the dataset distribution 
3. The design ofthe primary-auxiliary encoding module may be sensitive to the primary modality. The paper didn't analyze the effectiveness of the symmetric fusion for these two modalities. The selection of modalities is also important , such as why we select the IR as the main modality. In conclusion, the paper's ablation study is still weak, leak the design rational of each module in the framework.4.Some writing tips: the equation (9) is a repeatition of (1), you can just reference equation (1); the font in the figure should be consistent with that in the main text, it is better to use selectable text format in the figure (e.g. using pdf format for figure), otherwise some tiny text can not be seen in the figure.

### Questions
1. For Table 4 (line 934), when you compare the resource usage, did you use the same model size for different models? If the model sizes vary a lot, can you specify them?
2. Where did you cite Table 5? What are the 5 modalities in these experiments? The resource usage looks totally acceptable for 5 modalities with a better performance (only a normal commercial GPU can handle this, no need to trade off). Why not use all the modalities?
3. Why do you use a CNN to model the original data? And why MLP for importance score, why self-attention for fragment extraction? Did you do an ablation for each module in the framework?
4. Why did you select the IR spectra as the primary modality and the other two as auxiliary? Did you do an ablation study to prove your selection?

### Soundness
1

### Presentation
2

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
The authors introduce MSpecTmol, a new framework for integrating spectral modalities for identifying molecular structures. Using principles from information bottleneck theory, the authors fuse multiple spectral modalities for improved spectra representations, beating recent baselines in structure identification as well as conformer generation when plugged into a diffusion model.

### Strengths
- Results are consistently better than the baselines, sometimes by a large margin. The performance improvements are considerable.
- Good grounding in information bottleneck theory to rationalize their loss functions.
- Architecture is easy to understand an implement, making adoption more feasible.

### Weaknesses
- Some minor typographic errors. E.g. "The final objective is: The primary...", "Yu et al. (2022). we could...". A quick overview would suffice.
- Baselines for conformer generation are rather slim. There are a number of recent conformer generation baselines, e.g. [1], [2].
- As the authors admit, the method is restricted to training on a fixed vocabulary of functional groups, limiting performance on novel molecules.

[1] https://arxiv.org/abs/2311.17932
[2] https://arxiv.org/pdf/2507.09785

### Questions
- How easily are new modalities added without re-tuning the balancing parameters?
- Have you experimentd with modality-specific priors instead of using a single $q(t_a)$?
- All baselines are machine learning-based. Do you have any comparisons with classical approaches? I am not familiar with this area, but it seems that the method would be more convincing if it outperformed the current a variety of appraoches, not just machine learning models.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes MSrectmol, a multi-modal spectroscopic learning framework designed to integrate information from various spectroscopic modalities, including infrared (IR), nuclear magnetic resonance (NMR), and mass spectrometry (MS), for molecular structure elucidation. The framework introduces a Primary- Auxiliary Information Bottleneck (PA-IB) formulation that extends from the traditional Information Bottleneck to fuse and compress information across different spectroscopic modalities. In this setting, one modality is treated as the primary source of information, while others serve as auxiliary inputs that provide complementary features. The framework also applies adaptive noise gating to fuse and retain task-relevant information across modalities while reducing redundancy.

The framework is applied to two downstream analysis, molecular substructure (functional group) classification and spectrum-conditioned 3D conformation generation. Reported results shown an average micro F1-score of 0.941 across seven different spectrum configurations on the simulated spectrum and 0.866 across six configurations on experimental spectrum. For the conformation generation task, the model attains an average RMSD of 0.695 angstrom across five different input spectrums.
Overall, the methodology appears sound, but it represents an incremental extension of existing information bottleneck and multi-modal fusion methods. The empirical findings partially support the paper’s core contributions.

### Strengths
1. Overall, the approach is well motivated and technically sound.
2. The work demonstrates that multi-modal fusion guided by the Primary-Auxiliary Information Bottleneck (PA-IB) can generate representations useful for both functional group identification and 3D conformation prediction. 
3. The experiments are conducted across both simulated and experimental datasets. 
4. The paper communicates its ideas effectively and meets the clarity and organization standards.

### Weaknesses
1.Given that, in conventional practice, chemists first infer 2D molecular connectivity from spectra and then generate 3D structures using cheminformatics tools such as RDKit or OpenBabel, broader benchmarking would strengthen the evaluation of the 3D conformation generation task. The paper includes comparisons with GeoDiff and an attention-based baseline, but additional benchmarks, particularly graph- or SMILES-conditioned conformer generation methods and conventional pipelines using RDKit or OpenBabel, would help contextualize the advantages of direct spectrum-to-3D generation. Such baselines are essential for assessing the reliability and added value of the proposed approach relative to established workflows.

2. The generalization of the model to larger or noisier molecules remains unclear. Current experiments focus on benchmark datasets, leaving open how the framework performs on high-noise experimental spectra or larger molecules with dense peak patterns. Since strong or overlapping spectral peaks often carry critical structural information, additional evaluation on such challenging cases would strengthen the practical validity of the method. 

3. The paper does not report computational efficiency metrics such as training cost, inference time, or scalability with increasing modality count or molecular size. Including such results would help assess the framework’s practicality for large-scale spectroscopic applications.

4. Overall, the proposed approach only represents an incremental extension of existing information bottleneck and multi-modal fusion methods. The multi-modal fusion strategy lacks enough novelty.

### Questions
1. Could the authors comment on how the model handles noisy or partially missing spectral modalities, which are common in real experimental conditions?
2. The paper compares against GeoDiff and an attention-based baseline. Could the authors provide additional comparisons to a conventional spectrum -> inferred 2D connectivity -> 3D conformer using tools such as RDKit/OpenBabel? Quantitative results from more conformat generation baselines would help evaluate the practical benefit of direct spectrum to 3D generation.
3. Several performance differences relative to baselines appear modest. Could the authors provide statistical tests to indicate which improvements are statistically significant across runs and datasets?
4. While α and β sensitivity are discussed, could the authors provide additional analysis or intuition on how the temperature parameter in the Gumbel–Softmax affects gating sparsity and interpretability?
5. May need to evaluate different fusion strategies.

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
The paper proposes a unified multi-modal approach for molecular structure elucidation from spectroscopic data.  

The authors introduce a Primary–Auxiliary Information Bottleneck (PA-IB) framework to model complementary information across multiple spectra (IR, NMR, MS, etc.). They evaluate the method on two main tasks: (1) functional-group classification (structure elucidation), and (2) 3D conformation generation. Experiments reportedly show superior performance compared to several CNN and Transformer baselines.

### Strengths
1. The paper addresses a relevant question: how to fuse information from multiple spectroscopy modalities for molecular structure understanding.
    
2. The use of information bottleneck principles for spectral representation compression is theoretically sound and aligns with prior probabilistic learning methods.
    
3. The multi-modal fusion experiments for functional-group prediction are relatively well-executed, and the analysis of modality importance (IR, H-NMR, C-NMR) provides useful insight.

### Weaknesses
1. The novelty of the proposed method is rather limited, as the PA-IB framework represents a straightforward extension of the conventional information bottleneck to a multi-modal context. The architectural choices, such as 1D-CNN encoders, follow common practice and do not introduce a fundamentally new modeling mechanism.
2. The experimental design for the conformation generation task is scientifically inconsistent with practice and the paper’s stated goal of structure elucidation from spectra. As described in Appendix N.1, the generation model is conditioned on the complete molecular graph, including atom types and bond connections, which are precisely the unknowns that spectroscopy is supposed to infer. 
3. The paper lacks an ablation to verify the contribution of the primary–auxiliary design within the PA-IB formulaton. The improvement might simply come from adding information-bottleneck regularization rather than from the asymmetric modality treatment. An additional experiment applying a uniform IB constraint across all modalities would clarify whether the proposed hierarchical design truly provides unique benefits.

### Questions
Please refer to the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2
