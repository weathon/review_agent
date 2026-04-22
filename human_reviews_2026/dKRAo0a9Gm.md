# AbdCTBench: Learning Clinical Biomarker Representations from Abdominal Surface Geometry

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 6

## Abstract
Body composition analysis through CT and MRI imaging provides critical insights for cardio-metabolic health assessment but remains limited by accessibility barriers including radiation exposure, high costs, and infrastructure requirements. We present AbdCTBench, a large-scale dataset containing 23,506 CT-derived abdominal surface meshes from 18,719 patients, paired with 87 comorbidity labels, 31 specific diagnosis codes, and 16 CT-derived biomarkers. Our key insight is that external surface geometry is predictive of internal tissue composition, enabling accessible health screening through consumer devices. We establish comprehensive benchmarks across seven computer vision architectures (ResNet-18/34/50, DenseNet-121, EfficientNet-B0, ViT-Small, Swin Transformer-Base), demonstrating that models can learn robust surface-to-biomarker representations directly from 2D mesh projections. Our best-performing models achieve clinically relevant accuracy: age prediction with MAE 6.22 years (R²=0.757), mortality prediction with AUROC 0.839, and diabetes (with chronic complications) detection with AUROC 0.801. Notably, smaller architectures consistently matched or surpassed larger models, while medical-domain pre-training (RadImageNet) and self-supervised pre-training (DINOv2) showed competitive but not superior performance. AbdCTBench represents the largest publicly available dataset bridging external body geometry with internal clinical measurements, enabling future research in accessible medical AI. We plan to release the dataset, evaluation protocols, and baseline models to accelerate research in representation learning for medical applications, immediately following the review period.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces AbdCTBench, a large-scale dataset comprising 23,506 CT-derived 2D abdominal surface meshes from 18,719 patients, paired with 16 quantitative CT biomarkers, 87 comorbidity labels, and 31 diagnosis codes. The core hypothesis is that external body surface geometry, capturable via consumer-grade depth sensors, can serve as a non-invasive proxy for internal tissue composition and clinical risk stratification. The authors benchmark six vision architectures (including CNNs, ViT, and medical-pretrained models) on 10 biomarker prediction tasks, demonstrating that even lightweight models can achieve clinically meaningful performance (e.g., mortality prediction AUROC = 0.839, age MAE = 6.22 years). The dataset and baselines are slated for public release to foster research in accessible, radiation-free health screening.

### Strengths
1. The work directly addresses a critical gap in preventive medicine: democratizing access to body composition biomarkers without CT/MRI. By leveraging surface geometry, a modality compatible with smartphones and LiDAR, the paper aligns with real-world trends in digital health and point-of-care screening.
2. With 23K+ samples and 104 clinical variables, AbdCTBench is the largest publicly available dataset linking external surface geometry to internal CT-derived biomarkers. The inclusion of longitudinal lab values, HCC codes, and quantitative tissue metrics enables diverse downstream tasks.
3. The authors implement a standardized training protocol (consistent optimizer, augmentation, class balancing, threshold tuning) across architectures, ensuring fair comparison. Results include bootstrapped confidence intervals, enhancing statistical reliability.
4. Models achieve clinically actionable accuracy on key tasks (mortality, vascular disease, diabetes complications), supporting the feasibility of surface-based screening.

### Weaknesses
1. All CT scans originate from one private healthcare provider, raising concerns about demographic, geographic, and protocol-related biases. The authors acknowledge the lack of explicit inclusion criteria, which may limit generalizability.
2. Surface meshes are derived from CT scans, not captured by real consumer devices (e.g., iPhone LiDAR). The fidelity gap between clinical CT-derived surfaces and real-world depth scans remains unquantified, casting doubt on real-world deployability.
3. The benchmark excludes modern medical vision architectures (e.g., UNet variants, Swin Transformers, Mamba-based models), which could better exploit inter-biomarker correlations.
4. From a technical and methodological standpoint, the paper's contribution lies primarily in dataset curation and empirical benchmarking using standard vision models, without novel architectures, training paradigms, or representation learning innovations. As such, it would be better suited for a dataset-and-benchmark track or a more clinically oriented venue focused on medical imaging and health informatics.

### Questions
Please address the aforementioned weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a large dataset consisting of surface projection images and associated biomarkers (derived from CT) and diagnosis labels. This is motivated by the opportunity to validate models that can predict the associated biomarkers and diagnosis labels using consumer grade 3D surface extraction devices. Additionally, the paper benchmarks representative architectures for the prediction tasks (classification and regression) including pretrained models on convolutional and transformer-based ones.

### Strengths
The effort to get approval, collect, preprocess and benchmark such a large dataset with a rich set of associated labels and biomarkers is commendable. The paper is mostly well written with data acquisition and preprocessing steps well explained. The benchmarking of models on the proposed datasets does cover representative architectures, including pretrained models.

### Weaknesses
*Input Representation*: Why surface projection images? Why not train on 3D surface meshes themselves to predict biomarkers and diagnostic labels? I.e. The paper is motivated by the possibility to use consumer grade devices such as LiDAR-enabled phones which can generate 3D surface meshes but the proposed dataset and model training use 2D projections which considerably reduces the available information. No rationale or experimental validation is given for this decision.  

*Disaggregated reporting*: It is unclear from the reported aggregated metrics about the performance on clinically important subgroups. Without disaggregated reporting and subgroup analysis, the benchmarking may be incomplete with potential for underdiagnosis bias, subgroup disparities and bias and fairness issues to certain subgroups which may also highlight if such a mapping from surface projection to 3D-based biomarkers and diagnostic labels even make sense. Comparison with naive baselines such as just predicting average quantities will also be useful.

### Questions
Please refer to the input representation and disaggregated reporting bullet points in the section above.  

In addition to disaggregated reporting, Could you also report confidence interval of the reported metrics? Also, please clarify the reported performance with respect to clinical utility (readiness) and gap with respect to clinical requirements (for e.g. how small the Calcium Score should error be to be clinically usable in a specific diagnosis?). 

Use of heatmaps (Grad-CAM) to justify effectiveness of learnt representation has been shown to be problematic – useful for model debugging perhaps but controversial as a tool for pixel importance / visual explanations [1]. 

1. Kindermans, Pieter-Jan, et al. "The (un) reliability of saliency methods." Explainable AI: Interpreting, explaining and visualizing deep learning. Cham: Springer International Publishing, 2019. 267-280. 

 

Certain sentences may require citations. E.g.  
- While early implementations struggle with complex torso geometries,... 
- This limitation has led to the adoption of advanced imaging biomarkers ,... (Need citation associating tissue composition with CT) 
 
Typos 
Spacing typos: advancesspanning, modelsgeneralize,

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
3

### Summary
This paper addresses ​​accessibility barriers in body composition analysis via CT and MRI​​ (e.g., radiation exposure, high costs) by introducing ​​AbdCTBench​​, a large-scale dataset containing 23,506 CT-derived abdominal surface mesh images paired with 87 comorbidity labels and 16 CT-derived biomarkers. The core method involves using ​​2D mesh projections​​ and multiple computer vision architectures (e.g., ResNet, EfficientNet, ViT) to learn representations from external surface geometry for predicting internal biomarkers. Key results show clinically relevant accuracy: age prediction (MAE 6.22 years), mortality prediction (AUROC 0.839), and diabetes detection (AUROC 0.799). The primary contribution is the ​​first and largest publicly available dataset bridging external body geometry with internal clinical measurements​​, enabling benchmarks for accessible medical AI. Experiments reveal that smaller architectures (e.g., ResNet-18) often match or surpass larger models, while medical-domain pretraining (e.g., RadImageNet) does not yield superior performance.

### Strengths
- ​​Originality:​​ AbdCTBench is the first public dataset to systematically connect abdominal surface geometry with internal biomarkers at scale (23,506 samples), pioneering non-invasive health screening.
- Quality:​​ Rigorous experimental design covers 6 architectures and 10 biomarker tasks, with standardized protocols (e.g., inverse frequency weighting for class imbalance) ensuring reproducibility. Grad-CAM visualizations (Figure 3) effectively illustrate learned representations.
- Clarity:​​ Training details (e.g., hyperparameters, loss functions) are comprehensively documented, and appendices provide full statistics (Table 3-4), facilitating replication.

### Weaknesses
- Generalizability Concerns:​​ Data is from a single site, risking demographic biases (e.g., age/race distribution uncontrolled); multi-site validation is needed.
- Limited Architecture Exploration:​​ Only standard CNNs/transformers are tested, omitting medical-specific models (e.g., U-Net variants) that might capture better representations. ViT-Small's competitive but suboptimal performance.

- Uneven Task Performance:​​ HCC 12 prediction (AUROC ~0.59) is near-random, but no deep analysis explains why surface geometry fails here.

- ​​No Multi-Task Learning:​​ The single-target framework misses biomarker correlations; multi-task learning (as noted in Section 7) could improve efficiency but is unexplored.

### Questions
Why did ViT-Small with DINOv2 pretraining not outperform CNNs? Is this due to local feature importance in surface geometry? Could larger transformers or medical-specific pretraining help?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents the AbdCTBench, a large-scale dataset of 23,506 CT-derived abdominal surface mesh images from 18,719 patients. The AbdCTBench contains 16 biomarkers paired with 31 diagnosis codes and 87 comorbidity labels. The authors established benchmarks on age prediction, mortality prediction, diabetes detection, etc., using six computer vision architectures.

### Strengths
The constructed dataset is a large-scale CT image dataset. It contains 23,506 CT-derived abdominal surface mesh images from 18,719 patients. The images belong to 87 comorbidity labels, 31 specific diagnosis codes, and 16 CT-derived biomarkers.

The authors conducted experiments using CV architectures ResNet-18/34/50, DenseNet-121, EfficientNet-B0, ViT-Small, and computed the benchmarks for non-HCC biomarkers and HCC code biomarkers.

### Weaknesses
The AbdCTBench is not yet released/available.

The model architectures are small. It's better to test the ViT-base or larger models to evaluate their performance. 

It's better to provide the internal body composition biomarkers for the 4 images shown in Figure 1.

### Questions
Is the dataset de-identified?

### Soundness
2

### Presentation
3

### Contribution
3
