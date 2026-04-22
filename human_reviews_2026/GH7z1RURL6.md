# SelvaBox: A high‑resolution dataset for tropical tree crown detection

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 8

## Abstract
Detecting individual tree crowns in tropical forests is essential to study these complex and crucial ecosystems impacted by human interventions and climate change. However, tropical crowns vary widely in size, structure, and pattern and are largely overlapping and intertwined, requiring advanced remote sensing methods applied to high-resolution imagery. Despite growing interest in tropical tree crown detection, annotated datasets remain scarce, hindering robust model development. We introduce SelvaBox, the largest open‑access dataset for tropical tree crown detection in high-resolution drone imagery. It spans three countries and contains more than $83\,000$ manually labeled crowns -- an order of magnitude larger than all previous tropical forest datasets combined. Extensive benchmarks on SelvaBox reveal two key findings: 1) higher-resolution inputs consistently boost detection accuracy; and 2) models trained exclusively on SelvaBox achieve competitive zero-shot detection performance on unseen tropical tree crown datasets, matching or exceeding competing methods. Furthermore, jointly training on SelvaBox and three other datasets at resolutions from 3 to 10 cm per pixel within a unified multi-resolution pipeline yields a detector ranking first or second across all evaluated datasets. Our dataset, code, and pre-trained weights are made public.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces SELVABOX, a high-resolution dataset for tropical tree crown detection, featuring over 83,000 manually labeled crowns from drone imagery across Brazil, Ecuador, and Panama. It highlights the challenges of detecting individual tree crowns in tropical forests due to their variability and overlap, advocating for advanced remote sensing methods.

### Strengths
# Strengths:

(+) The SELVABOX dataset's scale, with over 83,000 annotations, is an order of magnitude larger than existing tropical forest datasets, providing a substantial resource for training and validation.

(+) Higher-resolution inputs consistently enhance detection accuracy, as demonstrated through extensive benchmarks, offering valuable insights for model optimization.

(+) The open-access nature of the dataset, code, and pre-trained weights fosters collaboration and accelerates research in tropical forest monitoring.

### Weaknesses
# Weakness: 

(-) Incomplete annotations in some areas (e.g., Brazil and Ecuador rasters) may introduce noise and reduce model reliability, despite the masking strategy employed.

(-) The reliance on RGB-only validation, due to logistical and cost constraints, may limit accuracy compared to LiDAR-based methods in complex tropical canopies.

(-) The paper lacks a detailed comparison with LiDAR-based datasets, potentially overlooking a key alternative for tropical forest monitoring.

Minor Suggestions:

Some prior works of detection in forest scenarios are missing [1-5]. It would be better if the author could discuss them in the related work.

**Ref:**

[1] From drones to phenotype: using UAV-LiDAR to detect species and provenance variation in tree productivity and structure. Remote Sensing, 2020.

[2] Explainable identification and mapping of trees using UAV RGB image and deep learning. Scientific Reports, 2021.

[3] A general deep learning model for bird detection in high-resolution airborne imagery. Ecological Applications, 2022.

[4] Benchmarking wild bird detection in complex forest scenes. Ecological Informatics, 2024.

[5] Enhancing palm precision agriculture: An approach based on deep learning and UAVs for efficient palm tree detection. Ecological Informatics, 2025.

### Questions
# Questions:

1. How does the masking strategy for incomplete annotations impact model training performance compared to datasets with full annotations?

2. What are the specific challenges of scaling RGB-only methods to diverse tropical forest conditions not addressed by SELVABOX?

3. How might the inclusion of LiDAR data enhance the detection accuracy of models trained on SELVABOX?

4. What are the potential biases introduced by the manual annotation process conducted by a limited number of biologists?

5. How does the zero-shot performance of SELVABOX-trained models compare to fine-tuned models on non-tropical datasets over a longer term?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces **SELVABOX**, a large-scale UAV RGB dataset for detecting individual tree crowns in tropical forests across Brazil, Ecuador, and Panama. The authors also propose a practical evaluation metric, **RF175**, which measures raster-level F1 at an IoU threshold of 0.75. The dataset contains more than 83k high-quality crown annotations and comes with a complete processing pipeline, open-source code, and pretrained models.  
Extensive experiments are conducted with both CNN and Transformer detectors. The results show that higher image resolution and multi-resolution training can significantly improve generalization, even across different domains. Overall, the dataset and benchmark make a timely and meaningful contribution to ecological AI and forest monitoring.

### Strengths
- **High-impact dataset:**  SELVABOX fills an important gap for tropical regions. The scale, diversity, and annotation detail are impressive, and the dataset is likely to become a strong reference for future research in this area.
- **Practical evaluation metric:**  The proposed RF175 metric focuses on raster-level performance rather than patch-level mAP, which makes sense for real-world forestry applications.
- **Comprehensive experiments:**  The authors benchmark various models, image resolutions, and training setups, providing clear evidence for each design choice.
- **Good open-science practice:**  The dataset, preprocessing pipeline, and code are fully released, supporting reproducibility and future extensions.
- **Clear and well-written paper:**  The paper is well organized, the figures are helpful, and the results are easy to interpret.

### Weaknesses
1. The paper does not report any **quantitative analysis of annotation consistency**. Although the annotation process is described in detail, we do not know how consistent different annotators were. Some inter-annotator agreement numbers (like IoU or Cohen’s kappa) would really help readers trust the data quality.
2. The **test set completeness** is mentioned but not clearly verified. It would be good to show that the test areas were fully annotated or to quantify potential missing crowns. Without this, the reliability of RF175 as a “gold-standard” metric is slightly uncertain.
3. The **RF175 metric** seems quite sensitive to the IoU threshold and NMS parameters, yet this sensitivity is not explored. It would be interesting to see how performance changes for IoU thresholds of 0.6 or 0.8.
4. The **Detectree2 evaluation** may include unintentional data overlap because its original split cannot be reproduced. The authors mention this but still treat it as an “in-distribution” case, which could confuse readers. A clearer statement or a controlled subset would make the comparison more trustworthy.
5. The **comparison with existing datasets** is mostly visual. Statistical comparisons, such as KL divergence or KS tests on crown-size distributions, would make the argument for SELVABOX’s uniqueness stronger.
6. The **ethical and dual-use discussion** is too brief. The dataset could be misused, for example, for illegal deforestation or land exploitation. A short section on “Responsible Use” with clear terms or usage guidelines would be helpful.

### Questions
1. Could you share some statistics on **annotation consistency**? Even small samples of double-annotated areas or agreement rates would help quantify data reliability.
2. For the **test set completeness**, did you perform any internal audit or random visual checks? If so, please describe how it was done.
3. Have you tried testing **RF175 under different IoU thresholds** (e.g., 0.6, 0.7, 0.8)? A sensitivity curve would be nice to include, even in the appendix.
4. Regarding the **Detectree2 benchmark**, is there a chance of overlapping geographic regions? It might help to clarify this in the final version and mention it clearly as a limitation.
5. Could you add a small **quantitative comparison with other datasets**, perhaps comparing crown-size or bounding-box distributions numerically instead of only visually?
6. The paper would benefit from a short **Responsible Use or Ethics** paragraph in the main text. You could briefly outline intended use, potential misuse, and the licensing policy.
7. It might also help to add **one or two failure case visualizations**, showing where the models still struggle (e.g., dense canopy overlap or mixed-species confusion). This would make the benchmark even more insightful.
8. Have you considered adding **statistical confidence intervals** (like standard deviations over random seeds) for RF175 in the main tables? It would make the reported numbers more interpretable.
9. Some references could be considered in Introduction and Related works: such as 10.1109/MGRS.2024.3479871, https://doi.org/10.1016/j.rse.2023.113485 and https://doi.org/10.1016/j.isprsjprs.2020.07.002

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a dataset, SelvaBox, containing UAV orthoimagery over several neotropical forest locations, covering 500 ha and containing 83k manually annotated tree crowns in the form of bounding boxes. Although larger datasets exist for temperate zones, this dataset aims at providing a large, diverse and high-quality crown detection benchmark for the tropics, with a focus on natural forests. The results show that using SelvaBox along with a transformer-based object detection model leads to state-of-the-art results on several other benchmarks. The paper also proposes an evaluation protocol, includinf the RF1_75 metric computer at the whole raster level as a better way to assess model performance.

### Strengths
- The paper is clearly written and well structured.
- The motivation is clear and well formulated.
- The experiments are comprehensive and highlight the usefulness of the new dataset.
- The choices in terms of dataset creation and models are well justified.

### Weaknesses
1. This work is quite complete. However, it all relies on the quality of the manual annotations. Although the good cross-dataset performance already indicates that the annotation quality is appropriate (and thus the within-SelvaBox performance does not rely on learning dataset-specific annotation biases), it would still be very helpful to see a quality assessment in the form of inter-annotator agreement analysis.
2. The proposals of new evaluation metric/setup, which is one of the novelty claims in the paper, would be much better supported using such an inter-annotator evaluation, since they would act as an oracle predictor on which the new metrics could be compared to previous ones. Otherwise, there is little evidence that supports the claims about the proposed evaluation protocol.
3. In a way, this paper presents a dataset where the main edge is that it is larger scale than others in the same (tropical forest) setting. As such, there is little novelty to speak of.  Novelty is typically a requirement according to the ICLR reviewer guidelines. I’m not 100% sure what this entails when it comes to datasets, but I would imagine enabling the benchmarking of so far un-benchmarkable tasks. SelvaScope does not allow to evaluate methods on anything that is fundamentally different, although its larger scale and diversity will likely be helpful to train models that will be useful to practitioners. As such, it maybe worth questioning the adequacy of ICLR as a venue to publish this paper, although I do commend the authors for the quality of their work.

### Questions
Would it still be feasible to look at inter-annotator agreement? Are some tiles maybe already annotated by two or more experts? Could experts be contacted to do this? I understand that this could be tricky to pull off in the latter case.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces SELVABOX, a large-scale dataset for individual tree crown detection in tropical forests, comprising 83,137 manually annotated bounding boxes from UAV imagery across Brazil, Ecuador, and Panama. The authors conduct extensive benchmarks comparing CNN-based (Faster R-CNN) and transformer-based (DINO) detection methods across multiple resolutions and spatial extents. Key contributions include: (1) the SELVABOX dataset with expert annotations, (2) a comprehensive benchmark including a new raster-level evaluation metric (RF175), (3) demonstration of multi-resolution training for cross-dataset generalization, and (4) open-source tools for preprocessing and inference. The work shows that DINO with Swin-L backbone achieves state-of-the-art performance on both tropical and temperate forest datasets.

### Strengths
1. High-quality dataset: Expert annotation by trained biologists (1,284 person-hours) across diverse tropical forest types ensures dataset quality and ecological validity
2. Rigorous experimental methodology: Spatially separated splits, multiple resolutions tested, standard deviations reported, and extensive ablation studies demonstrate scientific rigor
3. Reproducibility: Excellent documentation with code, data, and models publicly released; comprehensive appendices detail preprocessing steps for all datasets
4. Raster-level evaluation: The RF175 metric addresses a real limitation of tile-level metrics for operational forest monitoring applications

### Weaknesses
1. Limited technical novelty: The paper is primarily an application/benchmark paper. While valuable, it doesn't introduce novel architectures or training techniques beyond straightforward multi-resolution augmentation
2. Incomplete analysis of RF175 metric. No ablation on the 75% IoU threshold choice
3. Experimental limitations: Missing temporal validation (same site, different time periods)
4. Statistical rigor: Limited discussion of when differences are meaningful

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3
