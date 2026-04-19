# Reveal Object in Lensless Photography via Region Gaze and Amplification

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Detecting concealed objects, such as in vivo lesions or camouflage, requires customized imaging systems. Lensless cameras, being compact and flexible, offer a promising alternative to bulky lens systems. However, the absence of lenses leads to measurements lacking visual semantics, posing significant challenges for concealed object detection (COD). To tackle this issue, we propose a region gaze-amplification network (RGANet) for progressively exploiting concealed objects from lensless imaging measurements. Specifically, a region gaze module (RGM) is proposed to mine spatial-frequency cues informed by biological and psychological mechanisms, and a region amplifier (RA) is designed to amplify the details of object regions to enhance COD performance. Furthermore, we contribute the first relevant dataset as a benchmark to prosper the lensless imaging community. Extensive experiments demonstrate the exciting performance of our method. Our codes will be released in \url{https://github.com/YXJ-NTU/Lensless-COD}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a region gazeamplification network (RGANet) for progressively exploiting concealed objects from lensless imaging measurements.

### Strengths
1. PHlatCam dataset is semantically labeled and contributes to the corresponding dataset as a benchmark and extensive experiments.
2. investigate the detection of concealed objects in lensless imaging scenarios.

### Weaknesses
1. The use of RGM twice in the network is confusing. The ablation experiment only studies the effectiveness of the combination of internal modules under the setting of RGM twice. It sounds more reasonable to input RGM after I_OFE passes through RA.
2. Under the condition of lensless cameras, the author studies concealed object detection. Such a combination of tasks makes people doubt whether its real application scenarios are wide. Why not study more common object detection tasks?
3. The design of the entire network framework and internal modules is relatively ordinary. Basically, it is based on existing network modules with certain modifications, giving people a feeling of an A+B combination.
4. The format and layout are uncomfortable, for example, the formulas in the paper have larger line spacing. In addition, the figures in the paper are not beautiful enough, and the color matching is abrupt.
5. From the ablation experiment result #10 in Table 2, we can see that OFE is the most important core part of the network. I think OFE+encoder-decoder can achieve good results.

### Questions
See Weaknesses for details.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors aim to address concealed object detection (COD) with lensless imaging systems. They propose a network for COD leveraging spatial and frequency feature enhancement and fusion. They also annotate 2600 paired images from the Display Captured Dataset to build a new dataset for COD with lensless imaging systems. However, the authors should design a special network for this task by considering the characteristics of the lensless camera, and clarify the details of the proposed dataset.

### Strengths
1. A new dataset for COD with lenless imaging system.
2. Good performance for a new setting.

### Weaknesses
1. Straightforward method with limited novelty. The authors do not analyze the challenge for COD with lensless cameras. The main difference in design for the lensless cameras is the optical-aware feature extraction, but it refers to [1]. In addition, the main module of the proposed method is the spatial-frequency enhance module, which directly uses the idea of existing works for COD [2, 3].

2. Insufficient experiments. The authors should compare with sota COD methods, such as [2,3,4]. Moreover, they should compare with the two-stage methods (Lensless imaging methods combined with COD methods). With the large parameters and computational cost, the two-stage methods may be more lightweight.

3. Details of the proposed dataset. The authors should provide a detailed analysis of the proposed dataset to clarify the difference between the dataset from [5], since some samples shown in the paper are the same as samples in [5].

[1] PhlatCam: Designed Phase-Mask Based Thin Lensless Camera. TPAMI2020.
[2] Frequency perception network for camouflaged object detection. MM2023.
[3] Camouflaged object detection with feature decomposition and edge reconstruction. CVPR2023.
[4] Feature shrinkage pyramid for camouflaged object detection with transformers.CVPR2023.
[5] Inferring Objects FromLensless Imaging Measurements. TCI2022

### Questions
1. Does the performance improvement come from the large parameters and computational cost?
2. Error in Figure 2, where the sigmoid function is directly fed to addition without any input in PVT.

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
The authors introduce a new method for detecting concealed objects using a lensless camera, the Region Gaze-Amplification Network (RGANet), which progressively enhances concealed object detection through well-crafted feature extraction and amplification techniques. A novel real-capture dataset is proposed for training and evaluate the proposed method.

### Strengths
The paper is clearly written, and the experimental results are compelling. The proposed new real-capture dataset DLCOD will help further research in this field. Additionally, the authors have discussed the limitations of their proposed method in the appendix.

### Weaknesses
There are some aspects that could benefit from further clarification and enhancement:

1. Additional details about the setup of the real-capture experiments would enhance the reproducibility and understanding of the method. Specifically, could the authors provide information on the distance between the PHlatCam and the display, as well as the display's specifications (e.g., size, model, and whether it is an LCD or OLED)?

2. Although the model was trained on a real dataset, the data was captured from a display screen. Given that lensless cameras may capture a broader range of wavelengths than standard RGB cameras, will using a screen-based dataset introduce potential bias? The model may be less effective in real-world conditions where wavelengths are not limited to the three produced by RGB displays. It would be beneficial for the authors to conduct additional experiments using non-display-based scenes to validate the model's performance in more natural, unfiltered conditions (qualitative evaluation is not required). If this is not feasible, further discussion of this limitation could be included.

### Questions
Please refer to the *Weakness* section.

### Soundness
3

### Presentation
3

### Contribution
3
