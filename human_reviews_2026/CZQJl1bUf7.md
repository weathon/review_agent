# NeurIPS: Neuro-anatomical Inductive Priors for Sphere-based Vision Brain Decoding

- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Generalizable fMRI decoding is hindered by the challenge of aligning signals from anatomically unique brains. Prevailing methods treat this anatomical variation as noise, creating a false performance-fidelity trade-off where efficient 1D encoders outperform geometrically faithful surface-based models. We argue this trade-off is an artifact of two core mismatches: inefficient surface tokenization and the failure to use anatomy as a predictive signal. We present **NeurIPS**, a framework that resolves both by reframing anatomical variation from a nuisance to a powerful inductive prior. NeurIPS unites two innovations: a **Selective ROI Spherical Tokenizer (SRST)** for efficient geometric encoding, and a **Structure-Guided Mixture of Experts (SG-MoE)** that explicitly models individual anatomy using cortical features. On the Natural Scenes Dataset, NeurIPS establishes a new state-of-the-art for surface decoders and achieves performance comparable to strong 1D baselines. This is achieved with unprecedented efficiency, as the model converges dramatically faster (**10 vs. 600 epochs**). This efficiency enables rapid adaptation to new subjects using only **20\%** of their data and ensures robust scalability as the training cohort is expanded. Ablations provide causal evidence that these gains are driven by the model's use of cortical features, not by memorizing subject IDs. By leveraging anatomical priors, NeurIPS provides a principled and scalable path toward robust, generalizable brain decoding.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces NeurIPS, a framework for fMRI-to-image decoding that leverages neuro-anatomical inductive priors to achieve both biological fidelity and efficiency. It combines two key modules: the Selective ROI Spherical Tokenizer (SRST), which confines spherical convolutions to visual ROIs for efficient, topology-preserving feature extraction, and the Structure-Guided Mixture of Experts (SG-MoE), which conditions expert routing on cortical features (thickness, curvature, sulcal depth) instead of subject IDs.

### Strengths
* Novel integration of anatomical priors into a surface-based fMRI decoder.
* Achieved high-quality reconstruction
* Ablation study justified that the performance gain is driven by the design

### Weaknesses
* The “Conditional Information Bottleneck” formulation is introduced as a motivation, but there is no empirical analysis showing how the model approximates that objective or reduces subject-specific information.
* The rationale for adopting a Mixture-of-Experts (MoE) conditioned on cortical features is insufficiently justified. While the paper argues that anatomy should guide computation, it is not clear why MoE routing is the most appropriate or necessary mechanism for achieving this.
* The method achieves strong quantitative results but lacks a clear causal or theoretical bridge between the stated motivation (modeling anatomical variability) and the proposed architectural solution (SG-MoE). The design appears empirically driven, as if a modern MoE framework was applied to this problem and happened to perform well, rather than derived from principled reasoning about anatomy-function relationships.
* The presentation could be clearer and more guided. While the paper is technically strong, its exposition is dense and at times difficult to follow. Key components such as the information flow through SRST and SG-MoE are not explained intuitively, and the motivation for the architecture is buried among theoretical framing. For example, Figure 6 aims to demonstrate neuroscientifically plausible representations but lacks sufficient guidance in the text; readers may struggle to connect its visualizations to specific claims about anatomical dependence or cross-subject alignment.

### Questions
* The paper introduces the Conditional Information Bottleneck (C-IB) objective as theoretical motivation, but the connection to the implemented modules (SRST and SG-MoE) is not clearly explained. Could the authors elaborate on how these components operationalize the C-IB framework?
* Figure 6A is described as showing that expert selection depends strongly on cortical origin but weakly on subject identity, yet the procedure for quantifying these dependencies is unclear. Could the authors explain how these dependence measures were computed and how to interpret the maps?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper addresses the important modelling question of surface-based fMRI decoding and proposes NeurIPS, a framework designed to incorporate anatomical priors into the task of image reconstruction from fMRI data. The model introduces two core components: 1. the Selective ROI Spherical Tokenizer (SRST), which enables efficient, geometry-preserving encoding by focusing computation on selected cortical ROIs; 2. the Structure-Guided Mixture of Experts (SG-MoE), which uses anatomical features (thickness, curvature, sulcal depth) to route tokens and suppress subject identity information. The architecture follows a dual-decoder approach: a perception decoder mapping fMRI to VAE latent space and a semantic decoder aligning fMRI to CLIP embeddings. Both paths guide a diffusion model to reconstruct images from brain activity. The model claims strong efficiency, faster convergence, and state-of-the-art results on the Natural Scenes Dataset (NSD), while maintaining biological plausibility and scalability to new subjects.

While the reported performance is impressive, scientific rigour and clarity in methodology should not be compromised for empirical gains. Sound experimental design and fair comparisons are equally essential for assessing true progress.

### Strengths
The paper is clearly written and visually well-presented. The figures are very clearly detailed and informative. There are many image reconstruction presented throughout the papers. 

The introduction of an anatomy-guided MoE is an interesting and novel idea for addressing inter-subject variability.

The dual-decoder design (semantic and perceptual) is consistent with recent diffusion-based pipelines.

The experiments are extensive, with scalability experiments, and ablations. Many metrics are used to assess the visual reconstructions. 

The model demonstrates strong performance on the benchmarked tasks of image reconstruction and adaptation to unseen subjects and efficiency gains (fewer epochs, smaller data fraction).

### Weaknesses
Despite the strong quantitative results on image reconstruction metrics, the paper is not yet ready for acceptance due to several scientific, methodological and conceptual limitations.

1. The paper is missing some conceptual and methodological clarity.

- The paper presents several approximations and inconsistencies, particularly in how it positions itself relative to prior surface-based deep learning work. The papers is positioned within the real of geometric deep learning and surface-based modelling while actually one-stream within the entire model (in the SRST) module is modelling signal directly over the surface (only the visual cortex), the rest of the fMRI stream flattening the data into a 1D vector. 

- The paper claimed computational advantage (>70% fewer tokens) which is not clearly demonstrated against the use of the entire cortical surface. Additionally the neuroscience motivations for this are not precisely detailed (see below) or compared to such choices in the rest of the literature. The term “inefficient surface tokenization” is quite vague; the paper should specify what it means and why this "inefficiency" should prevail on neuroscience grounded motivations (extracting information from the entire cortical surface in response to multimodal external stimuli). 

2. There is little credit given to prior surface deep-learning approches or simply fair and correct mention of these works. 

- The paper does not properly or sufficiently acknowledge prior surface-based approaches that have introduced methodologies to model cortical metrics and fMRI signals as signals defined over cortical manifolds (e.g., Zhao et al 2019, 2021; Gu et al., 2023; Dahan et al 2022, 2024, 2025 ; Yu et al., 2025 etc..). These works already account for cortical topology and geometry into decoding frameworks and the paper does not introduce the progress that has been made by such methods in the past (in particular in the related work section) which might seem scientifically unfair. 

- There is no other reference to the question of modelling "inter-subject variability" despite the large neuroimaging literature addressing this concern via anatomical or functional registrations (Fischl 2012, Robinson 2014, 2016, Glasser 2016). This question is only appreciated through the lenses of feature learning rather than well validated image processing pipelines. 

- Overall, the paper risks overstating novelty and unfair comparison with existing literature. The real contribution lies in "how" geometry and anatomy are operationalised (ROI restriction + anatomy-conditioned routing), not in introducing geometry-aware fMRI modelling per se.    

3. There is limited neuroscience grounding in the method. 

- The restriction to visual ROIs lacks neuroscientific justification and appears motivated mainly by computational constraints (“to avoid the O(T²) burden of full-hemisphere processing”). Describing full-hemisphere processing as “inefficient” is scientifically questionable: functional networks are distributed, and signals outside the visual cortex (auditory, parietal, or frontal areas) can still contribute to visual decoding. Empirical evidence (e.g., Glasser et al. 2016, Nature, Huth et al 2016) shows that audio-visual perception engages a distributed cortical network; including auditory, motor, language, and memory regions; contradicting the assumption that non-visual areas are irrelevant or detrimental.
    
- The paper does not test this assumption: no ablation demonstrates that excluding non-visual regions improves performance.
    
- The rationale for conditioning on anatomical features (thickness, curvature, sulcal depth) rather than functional measures is not clearly justified.

4. Some worries regarding baseline comparisons and experimental fairness

- The manuscript inaccurately places surface-based method such as SIM (Dahan et al., 2025) under the umbrella of spherical CNNs (legend of Figure 3) . Instead this method is not based on spherical convolutions such as other surface-based method (Zhao et al 2019, that is not even referenced and given credit in this work, despite the mentioning of spherical convolutions similar to the Spherical UNet model) but on regular icosahedral tessellation and patch-level feature learning. The authors appear to conflate tokenisation with hexagonal convolutions (as in Zhao et al 2019), suggesting some misunderstanding of prior surface methods.

- It seems curious that image reconstructions performance throughout the paper (Figure 1, 4, 5) are not benchmarked models explicitly designed for image reconstruction that such as MindEye (Scotti et al., 2023) or Mind-Vis (Chen et al., 2023). Instead, the authors focus on using MindBridge and SIM only, showing poor image quality in Figure 1 (but not in Figure 4), with the later not being optimised for image reconstruction, but audio-visual retrieval. This is misleading and not task-matched. It would be better for the paper to show performance against model actively designed for image reconstruction.

- It also remains unclear how baselines are trained or if pretrained diffusion models were reused or fine-tuned under comparable conditions. Particularly, surface-based models can be highly dependant on the surface representations (fsaverage or ico spaces with regular spacing) ; as well as pre-training for frame reconstruction (Dahan et al 2025) or stimuli alignment (audio and visual in movie watching experiment); very little information is provided regarding these points. A table detailing training setup, data representation space, pretraining, and dataset usage for each baseline would improve transparency.
    

5. Some methodological clarifications are needed
    
- The perception decoder flattens fMRI inputs into a 1D vector, contradicting the paper’s central claim about preserving cortical geometry.  The geometric contributions remain ambiguous compared to prior work (e.g., Zhao et al. 2019, Dahan et al., 2022), and ablations indicate that the global token contributes most strongly to performance, raising questions about how much the geometric components truly matter.

- Their is no mention on how many fMRI frames are used at training or inference time. This is a critical aspect of the spatio-temporal extraction of information from fMRI data. There is no mention of temporal feature extraction. 

6. About generalisation and evaluation

- The paper equates fine-tuning on held-out test subjects (Figure 5) with “cross-subject generalization.” This measures adaptability, not zero-shot generalisation. True generalisation should be evaluated without any fine-tuning on new subjects. If the model encodes anatomical priors effectively, it should generalise naturally. This is a major bottleneck at the moment. 
    
- Claims about generalisation should also be tempered given the limited number of subjects in NSD (high within-subject sampling, low inter-subject diversity). Is there any additional experiments on wider datasets (less densely sampled but with more subjects i.e. HCP 7T movie-watching). 

- Some claims in the results sections (Section 4.4 "This result is significant: it suggests that models without strong anatomical priors may treat increased subject variability as noise, leading to performance degradation." ) for instance, are weakly supported, because several confounding factors (data preprocessing, architecture, and training differences) are not controlled.

7. About using image reconstruction as an evaluation metric.

- The heavy reliance on image reconstruction as a primary evaluation makes it difficult to disentangle the contribution of the fMRI encoder from that of the pretrained diffusion model. Since the diffusion model dominates visual realism, improvements may reflect the generative prior rather than better neural representation learning. More broadly, this reflects a concerning trend in the field, where reconstruction quality is often prioritised over representational interpretability. In my view, this direction risks obscuring the real scientific question: how well we model the brain’s representational structure beneath the visual fidelity of generated images.

- More meaningful evaluations for this kind of frameworks would include semantic-level benchmarks (e.g., retrieval, classification, similarity) that directly probe representational alignment. We would encourage the authors to benchmark their model on well-established semantically task such as image or video retrieval to convince readers about the performance of their fMRI encoding method.

### Questions
-  Why does the model restrict computation to the visual cortex, when visual representations may also rely on distributed activity across parietal, temporal, and frontal areas?

- Does the model require pre-alignment across subjects (e.g., spherical registration to fsaverage), or is the anatomy-conditioned routing sufficient to handle inter-subject geometry?

- How many fMRI frames as used as input?

- How is the model building a "topology-preserving space suitable for joint training." , this is not clear? 

- Are ROIs subject-specific or common across all participants? If fixed, how does inter-subject variability in cortical folding affect performance?

- Could the authors clarify what spherical operation is used? The paper strongly argues that flattening fMRI signals into 1D vectors discards geometric information and undermines cortical topology preservation. However, the perception decoder in the proposed framework (fMRI → VAE latent) appears to do exactly that: it flattens ROI-masked fMRI signals following MindEye (Scotti et al., 2023).

- Is there evidence of overfitting to the visual cortex when restricting ROIs?

- Would the method still perform well if applied to non-visual fMRI datasets (e.g., auditory or motor)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a framework for cross-subject fMRI-to-image decoding. It introduces two main innovations: the Selective ROI Spherical Tokenizer (SRST) for geometry-preserving spherical tokenization, and the Structure-Guided Mixture of Experts (SG-MoE), which conditions fMRI decoding on cortical features such as thickness and curvature. Evaluated on the NSD dataset, they achieve state-of-the-art performance among surface-based decoders, fast adaptation to new subjects.

This work provides a solid conceptual and technical advance in cross-subject fMRI-to-image decoding. The architecture is thoughtfully designed and empirically validated. Its most valuable aspect is using cortical structure for generalization, marking an important step for brain decoding to generalize to unseen subjects.

While some methodological components (e.g., learned convolutions) could be better analyzed, the core idea of using cortical structure—rather than identity—for cross-subject generalization is novel and impactful enough to merit acceptance.

### Strengths
1.  Improved tokenization for cortical surface data. The SRST integrates both local and global tokens to capture multi-scale geometric information.
2.  Use cortical structure to improve cross-subject generalization. The SG-MoE conditions fMRI decoding on cortical features such as thickness and curvature, rather than only condition on subject ID. Conditioning on cortical structure is strongly more generalizable across subjects, this is a novel and impactful idea.

### Weaknesses
1. The contribution of trainable spherical convolutions in SRST remains unclear. Whether the trainable kernels themselves provide measurable gains over fixed kernels (Yu et al., 2025) is not ablated.
    Sijin Yu, Zijiao Chen, Wenxuan Wu, Shengxian Chen, Zhongliang Liu, Jingxin Nie, Xiaofen Xing, Xiangmin Xu, and Xin Zhang. From flat to round: Redefining brain decoding with surface-based fmri and cortex structure. arXiv preprint arXiv:2507.16389, 2025.

2.  Although anatomy-aware, SG-MoE still uses subject-ID information. Evaluating a pure structure-only variant would better demonstrate the full potential of using cortical structure.

### Questions
none.

### Soundness
3

### Presentation
3

### Contribution
3
