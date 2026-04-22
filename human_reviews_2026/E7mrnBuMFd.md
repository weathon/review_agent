# Learning Molecular Chirality via Chiral Determinant Kernels

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 2

## Abstract
Chirality is a fundamental molecular property that governs stereospecific behavior in chemistry and biology. Capturing chirality in machine learning models remains challenging due to the geometric complexity of stereochemical relationships and the limitations of traditional molecular representations that often lack explicit stereochemical encoding. Existing approaches to chiral molecular representation primarily focus on central chirality, relying on handcrafted stereochemical tags or limited 3D encodings, and thus fail to generalize to more complex forms, such as axial chirality. In this work, we introduce \textbf{ChiDeK} (\textbf{Chi}ral \textbf{De}terminant \textbf{K}ernels), a framework that systematically integrates stereogenic information into molecular representation learning. We propose the chiral determinant kernel to encode the SE(3)-invariant chirality matrix and employ cross-attention to integrate stereochemical information from local chiral centers into the global molecular representation. This design enables explicit modeling of chiral-related features within a unified architecture, capable of jointly encoding central and axial chirality. To support the evaluation of axial chirality, we construct a new benchmark for electronic circular dichroism (ECD) and optical rotation (OR) prediction. Across four tasks, including R/S configuration classification, enantiomer ranking, ECD spectrum prediction, and OR prediction, ChiDeK achieves substantial improvements over state-of-the-art baselines, most notably yielding over 7\% higher accuracy on axially chiral tasks on average.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes ChiDeK, a novel framework that explicitly models molecular chirality using chiral determinant kernels and a chiral-aware cross-attention transformer to capture stereochemical interactions. It achieves chiral-sensitive, rotation- and translation-invariant molecular representations for tasks like chirality classification and spectrum prediction.

### Strengths
- Paper is well written
- Introduces differentiable chiral determinant kernels to encode stereochemistry.
- Uses chiral-aware cross-attention to capture interactions between atom types.

### Weaknesses
- How does the chiral matrix characterize the axial chirality? It seems it can only characterize central chirality. 
- What is the computational complexity of the method, since it computes the determinant, which is expensive in higher dimensions?
- Can this method be generalizable to the datasets where we have a combination of molecules, of which some are chiral and some are not?
- Can the method characterize Diastereomers?

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors developed a deep learning-based tool named ChiDeK for distinguishing point chirality and axial chirality, and applied it to tasks including R/S label classification, enantiomer ranking, and prediction of the sign and position of ECD.

### Strengths
Chirality recognition is a central issue in chemical research, and accurately capturing the stereochemical environment of molecules is key to distinguishing enantiomers. The proposed ChiDeK architecture leverages an SE(3)-invariant chirality matrix and cross-attention mechanisms to effectively extract molecular stereochemical information, enabling the simultaneous identification of both point and axial chirality. The model demonstrates strong performance in R/S label classification, enantiomer ranking, and ECD sign and position prediction for chiral molecules, achieving particularly notable accuracy in the binary classification of ECD signs for axially chiral molecules.

### Weaknesses
1. Although acquiring chiral information of molecules is important, using a model to distinguish R/S configurations of point and axial chirality appears to be of limited practical significance. Well-established chemical rules, such as those implemented in RDKit, already exist for such differentiation. While the authors generate discriminative features for enantiomers via the chirality matrix and deep learning—which can indeed distinguish R/S configurations—this should not be the ultimate goal. The application of these chirality-related features should target more valuable downstream tasks, such as predicting enantioselectivity in asymmetric catalytic reactions or binding affinity between chiral drug molecules and protein targets.
2. This work can largely be viewed as an extension of the ChIRo study, expanding its scope from point chirality to axial chirality. However, after demonstrating that ChIRo could generate discriminative features for enantiomers, its authors further conducted a practically meaningful downstream application: ranking enantiomers by docking scores in an enantiosensitive protein pocket. In contrast, the present authors only replicated similar benchmark tests on point chiral data and did not perform valuable downstream application experiments on the axially chiral molecules emphasized in this work.
3. The authors claim that this is the first architecture capable of jointly encoding central and axial chirality. In reality, existing methods such as ChiralFinder and SPMS, which are mentioned in the paper, can also extract features that distinguish point and axial chirality. Moreover, the authors' model itself utilizes the chirality matrix generated by ChiralFinder. To ensure a rigorous performance comparison, the authors should benchmark their model against features derived from ChiralFinder and SPMS on the same set of tasks.
4. The core chiral information in this work is derived from ChiralFinder, which appears crucial to ChiDeK's ability to distinguish chirality. As a result, the unique contribution of ChiDeK itself to the task of axial chirality discrimination remains unclear.
5. According to the results in Table 1 for point chiral molecules across R/S classification, enantiomer ranking, and ECD prediction tasks, the model does not show a significant advantage in point chirality-related predictions. In particular, the accuracy for ECD sign prediction is only 53.3%, which—assuming a 1:1 positive-to-negative sample ratio—is barely better than random guessing (50%).
6. The authors do not appear to have provided the complete data used in the study. The data folder supplied is empty, and the accompanying README file is too brief, offering no guidance on how to obtain the necessary dependencies or datasets.

### Questions
See weaknesses.

### Soundness
3

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
This paper proposes ChiDeK, a unified framework for learning molecular chirality representations. The method introduces a chiral determinant kernel to encode stereochemical information derived from the chirality matrix, and integrates it into a cross-attention architecture to propagate chiral information across the molecular graph. The model is able to jointly handle central and axial chirality, and a new benchmark dataset for axial-chiral ECD prediction is constructed. Experiments show improvements across R/S classification, enantiomer ranking, and ECD spectrum prediction tasks.

### Strengths
1. Provides a unified representation for both central and axial chirality, which is not addressed by prior models.

2. The chiral determinant kernel is mathematically well-motivated and reflection-sensitive.

3. The newly constructed AxialECD dataset fills a gap in evaluating axial chirality.

4. Demonstrates clear improvements in ECD prediction, especially peak sign prediction for axial chirality.

### Weaknesses
1. Limited diversity and size of the AxialECD dataset. 

The proposed AxialECD benchmark includes ~600 axial chiral molecules, which represents a relatively narrow stereochemical space and may not generalize to other classes of axial chirality such as atropisomeric biaryls with flexible steric barriers or complexes with metal-coordinated axes. I recommend that authors report performance breakdown by molecular subtypes, or perform zero-shot or cross-dataset evaluation if another axial-chirality dataset (or cases from published articles) becomes available.

2. Scalability and efficiency of the chiral determinant kernel.

The chiral determinant kernel relies on QR decomposition for each chiral center, which may incur non-trivial computational overhead when scaling to large molecules (e.g., natural products, macrocycles) or conformer ensembles. Provide runtime comparisons with SE(3)-equivariant baselines and discuss strategies for batch-efficient QR computation.

3. Lack of evaluation beyond chirality-sensitive tasks.

While the model is clearly designed for chirality-aware tasks, all experiments are conducted on datasets where chirality is explicitly involved in the label. It is unclear whether ChiDeK introduces unnecessary inductive bias or reduced performance in general molecular property prediction tasks where chirality is less relevant. Please include control experiments on standard property prediction datasets (e.g., QM9, PCBA, MoleculeNet) to confirm no loss of expressivity.

### Questions
1. How does the model’s performance vary when using different conformer generation pipelines? For example, RDKit vs. xTB-relaxed vs. DFT-optimized conformers.

2. Could the authors provide runtime and memory usage comparisons, particularly regarding QR decomposition, compared to SE(3)-equivariant baselines?

### Soundness
4

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
This paper proposes ChiDeK (Chiral Determinant Kernels), a new architecture for learning stereochemistry-aware molecular representations. The key goal is to encode both central chirality (classical R/S stereocenters) and axial chirality (stereogenic axes, e.g. biaryls with restricted rotation) in a unified framework.

### Strengths
1. A major strength of the paper is that it propose an approach to build a representation that can capture both central and axial chirality, which has practical important in drug discovery.

2. The paper contributes the AxialECD dataset , providing a good benchmark for the analysis in the area.

### Weaknesses
1. The method assumes a correct partition of atoms (and axes) into chiral / chiral-related / non-chiral, in some cases with manual labeling by chemists. The model does not learn these on its own. Robustness to mistakes in this preprocessing step is not evaluated, so it’s unclear how well the approach holds up under noisy or incomplete chirality annotations. For example, if the stereocenter or chiral axis is mislabeled or partially missed, does performance degrade sharply, or is the model tolerant to some noise in these assignments?

2. The strongest axial chirality results are on ECD peak-sign prediction. It is not shown that the same architecture improves other chirality-sensitive endpoints (e.g., enantioselective binding, optical rotation). This leaves open whether the method is generally useful for axial chirality in drug discovery, or mainly tuned to ECD-style tasks. 

3. For central chirality ECD prediction, the model’s sign accuracy is near random (~53%), and the explanation is attributed to noisy conformers. The paper does not test whether using higher-quality conformers fixes this. So it’s not clear whether the limitation is data quality or a weakness of the architecture on central chirality.

4. Most of the axial chirality evaluation is framed around ECD spectrum prediction. Do you expect the same architecture to transfer to other chirality-sensitive endpoints (e.g., enantioselective binding affinity, optical rotation, chiral toxicity)?

### Questions
1. The method assumes a correct partition of atoms (and axes) into chiral / chiral-related / non-chiral, in some cases with manual labeling by chemists. The model does not learn these on its own. Robustness to mistakes in this preprocessing step is not evaluated, so it’s unclear how well the approach holds up under noisy or incomplete chirality annotations. For example, if the stereocenter or chiral axis is mislabeled or partially missed, does performance degrade sharply, or is the model tolerant to some noise in these assignments?

2. The strongest axial chirality results are on ECD peak-sign prediction. It is not shown that the same architecture improves other chirality-sensitive endpoints (e.g., enantioselective binding, optical rotation). This leaves open whether the method is generally useful for axial chirality in drug discovery, or mainly tuned to ECD-style tasks. 

3. For central chirality ECD prediction, the model’s sign accuracy is near random (~53%), and the explanation is attributed to noisy conformers. The paper does not test whether using higher-quality conformers fixes this. So it’s not clear whether the limitation is data quality or a weakness of the architecture on central chirality.

4. Most of the axial chirality evaluation is framed around ECD spectrum prediction. Do you expect the same architecture to transfer to other chirality-sensitive endpoints (e.g., enantioselective binding affinity, optical rotation, chiral toxicity)?

### Soundness
2

### Presentation
2

### Contribution
2
