# Learning Asymmetric Phase Dynamics via Clinically-Guided Spatiotemporal Fusion

- Decision: Reject
- Scores: 2, 6, 2, 2

## Abstract
Hepatocellular carcinoma is a leading cause of cancer-related mortality, where accurate tumor characterization is crucial for guiding treatment. In clinical consensus, multiphase contrast-enhanced computed tomography (CECT) is indispensable: arterial (ART), portal venous (PV), and delayed (DL) phases jointly depict tumor vascularization, perfusion heterogeneity, and fibrotic evolution, forming the very basis of radiologists’ diagnostic reasoning.  **_Surprisingly, despite this well-established clinical value_**, most AI models still rely on ① single-phase inputs or naïve stacking of multiphase scans, ② ignoring temporal hemodynamics and lacking interpretability.
To bridge this gap, we present **Clinically-Guided Spatiotemporal Deep Fusion Network (CSF-Net)**, the first framework that explicitly embeds radiological knowledge into multiphase modeling. **CSF-Net** incorporates three synergistic components: the **multi-phase clinical-quantitative synergy branch (MCQS)** for phase-specific encoding, the **temporal-aware local feature refinement module (TLFR)** for perfusion dynamics, and the **query-interaction enhancement fusion module (QIEF)** for cross-phase alignment. By aligning AI modeling design with radiologists’ logic, **CSF-Net** establishes a clinically grounded interpretability paradigm, which inevitably yields superior performance gains. Extensive experiments on two CECT benchmarks, PLC-CECT and MPLL, demonstrate that **CSF-Net** achieves state-of-the-art performance. Our codes are available at: <https://anonymous.4open.science/r/ICLR26_CSF-Net-63E3/>.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors present a novel framework called Clinically-Guided Spatiotemporal Deep Fusion Network (CSF-Net) for improving the analysis of hepatocellular carcinoma using multiphase contrast-enhanced computed tomography (CECT) by integrating clinical knowledge into AI modeling.

### Strengths
The paper is well structured and clearly articulated. The experimental design is sound, and the analysis of the results is comprehensive.

### Weaknesses
I respectfully disagree with the authors’ major claim that most AI models rely on single-phase inputs or simply stack multiphase scans. A significant number of studies have already explored the extraction and spatiotemporal fusion of features from multiphase CT images — for instance, [Ref1]–[Ref3] for classification and [Ref4]–[Ref5] for segmentation. The authors themselves cite several of these works in Lines 124–130.

Compared with existing multiphase fusion approaches, certain components of the proposed method may indeed be novel. However, without a thorough discussion and systematic comparison with prior multiphase fusion studies, it is difficult to assess the true level of technical contribution. For example, the proposed model propagates information through a direct graph, but given that the number of nodes is small, it is unclear whether this design offers a meaningful advantage over spatiotemporal attention mechanisms used in previous work. The authors also appear to overemphasize the “clinically guided” aspect, as the method does not seem to incorporate substantial clinical reasoning from this reviewer’s perspective.

I would suggest submitting the paper to a more medical imaging–oriented venue, such as MICCAI or ISBI.

References
[Ref1] A Knowledge-Guided Framework for Fine-Grained Classification of Liver Lesions Based on Multi-Phase CT Images, IEEE JBHI, 2023.
[Ref2] Lesion-Aware Cross-Phase Attention Network for Renal Tumor Subtype Classification on Multi-Phase CT Scans, CIBM, 2024.
[Ref3] Sdr-former: A Siamese Dual-Resolution Transformer for Liver Lesion Classification Using 3D Multi-Phase Imaging, Neural Networks, 2025.
[Ref4] A Tri-Attention Fusion Guided Multi-Modal Segmentation Network, Pattern Recognition, 2022.
[Ref5] M3Net: A Multi-Scale Multi-View Framework for Multi-Phase Pancreas Segmentation Based on Cross-Phase Non-Local Attention, Medical Image Analysis, 2022.

Besides, many symbols in the equations are undefined. In addition, some symbols in the text and equations are inconsistent (e.g., Lines 159–161).

### Questions
I do not have specific questions for the authors. The main concern lies in the lack of systematic comparison with existing studies, which should be addressed through revision rather than clarification.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a clinically guided spatiotemporal deep fusion network named CSF-Net, aiming to address the limitations of existing AI models in handling liver tumor segmentation from multi-phase computed tomography (CECT) images. Clinically, radiologists heavily rely on dynamic blood perfusion information jointly provided by the arterial phase (ART), portal venous phase (PV), and delayed phase (DL) for diagnosis. However, most existing models overlook this temporal dynamics and only use single-phase images or simple image stacking. By explicitly embedding radiological knowledge into the model architecture, CSF-Net designs three modules—MCQS, TLFR, and QIEF—to collaboratively simulate tumor perfusion kinetics and cross-phase dependencies.

### Strengths
- It clearly identifies the pain point that existing AI models overlook the dynamic temporal information of multi-phase computed tomography , and innovatively proposes a solution that simulates the diagnostic logic of radiologists.

- This is a significant practical advantage. While achieving SOTA performance, the model’s computational complexity is far lower than that of its competitors—3.8 to 6.5 times lower than major counterparts.

- Through single-phase experiments and multi-phase experiments, it not only verifies the clinical prior but also proves the superiority of its multi-phase fusion model.

- In terms of average metrics, the proposed CSF-Net achieves the current SOTA segmentation performance on two public CECT datasets.

### Weaknesses
- The model requires perfect alignment of multi-phase images. However, it fails to analyze the impact of clinically common misalignment on performance and lacks corresponding robustness design.

- The core model diagram (Figure 2) includes an "MSF-Block" component that is never mentioned in the main text. 

- The model has a rigid design and must take complete three-phase images as input. Clinically, "phase-missing" data is very common, yet the paper provides no strategies to address this issue, which limits its practicality.

- The model does not consider the "temporal heterogeneity" problem—where the acquisition time points of each phase vary across different hospitals and devices. This may affect the accuracy of its spatiotemporal fusion.

### Questions
see weakness

### Soundness
3

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
4

### Summary
The paper introduces a novel deep learning model for segmenting hepatocellular carcinoma from CECT images. The paper is interesting in considering the complementarity of the multi-phase aspect of CECT images, as well as incorporating some clinical knowledge. The authors propose CSF_Net, which is composed of three complementary modules: one for phase-specific encoding, one for modeling perfusion dynamics, and one for multi-feature alignment.
An experimental validation is provided on two benchmarks.

### Strengths
**originality**
 + The originality of the paper relies on the integration of both the multi-phase aspect in the architecture and some clinical priors. The authors propose to integrate it through a directed acyclic graph and graph attention message passing. The proposed architecture is novel as a whole, but each of its components relies on existing technical tricks from the literature. 
 + One of the notable contributions is the MPLL dataset, which serves as an interesting resource for the community.


**quality**
+ The paper is well written and well illustrated. The mathematical formulation is given for each of the proposed modules.
+ The experimental validation is good, including comparison with some state-of-the-art approaches. 

**clarity**
+ The paper contains nice illustrations, in particular on the architectural details. 
+ The experimental result visualization is also informative

 **significance**
+ The proposed contributions advance the state-of-the-art for CECT AI-aided diagnosis.

### Weaknesses
My main concerns are :

+ Adequation of the contribution to the ICLR conference. Indeed, this paper presents a very interesting contribution to the community working on AI for medical research, particularly in the context of diagnostic assistance. However, the methodological proposals are not entirely new and may be difficult to generalize to other fields. It is certainly a very good contribution for a conference such as MICCAI, but it is less relevant for ICLR.
+ The paper is very well written, with a nice effort on formalization. However, the technical choices are not very well justified. For each technical block, it would be interesting to justify the desired and expected properties.
+ A more complete ablation study could be provided in the experimental section. 
+ Some claims are not supported. For instance, the claim of interpretability is not clearly validated.

### Questions
Many of the technical choices are not sufficiently motivated. 
+ What is the role of the reference fused view of equation 4? 
+ Why Swin Transformer Swin encoder? 
+ How can the proposed approach handle a missing phase?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a new algorithm for segmentation of liver tumors from multiphase contrast-enhanced CT (CECT) scans. The authors argue that most AI models either use a single phase or employ naive fusion methods (e.g., stacking) , thereby ignoring the critical temporal hemodynamic information (!).
The method is based on three modules: MCQS, TLFR and QIEF. 
I like MCQS more as it is kind of clinically evaluating which phase is more important, and then TLFR does some perfusion dynamics, and at the end QIEF does a final cross phase alignment and segmentation. Very simple procedure, engineeringly done well.
Publicly available PLC-CECT data and in house gathered MPLL dataset were used. Some good results were obtained.

### Strengths
- This paper is a good engineering paper, with clinical motivation. 
- Native-stacking is not a good idea, the paper shows (however the claim is too strong, there are so many other works in the domain with different jargon).
- Architecture has some novelties. For example, MCQS is a nicely designed method for hard-coding a clinical prior into the architecture. 
- Authors validate the clinical prior by showing single phase models. According to best results, they choose the most informative single phase (however, this is known by radiologists, even without this experiment it is known PV>ART>DL). 
- Computationally efficient method is presented.
- Well written paper with some good visualization and tables, appealing.

### Weaknesses
-- It is hard to have anything new in this paper, apart from well engineering work. The clinical prior for example is known. for each step, authors presented a module to be integrated into their model. Then end-to-end modeling is done for tumor segmentation. Some claims are wrong or incomplete, this is not the first study considering multiphase for sure, and varying combination or weighted combination of phases are even done before.
-- the model is very complex although it is done nicely with engineering approach. The final architecture includes a per-branch twin encoder, graph based MCQS, and TLFR, as well as QIEF modules, connected to each other with some complex design. 
-- despite the complex design, ablation is testing only MCQS. not all other parts.
-- validation on a second public dataset would be nicer, as the dataset 2 is private.
-- The roles of each module are not always clear. It is not clear why there are two complex temporal fusion modules exist.
-- CT scans are not presented with correct window. Use soft tissue window with enough contrast.
--figure 3 is fancy but not useful. Give a simple table for segmentation evaluation. It is hard to read what is there.
--where is nnUnet result? nnUnet is the winner of almost all segmentation methods lately. SOTA is missing.
--Table 3: permutation is given, but it is not compared with the data driven techniques where fusion operation can be learned. 
--figure 4 is not a standard evaluation for visuals. Please work with a radiologist for a proper visualization and comparison.

### Questions
the content I have put in weaknesses area are self-contained and each comment should be considered as questions, please.

### Soundness
4

### Presentation
3

### Contribution
2
