# VHSMarker and the Canine Cardiac Keypoint (CCK) Dataset: A Benchmark for Veterinary Cardiac X-ray Analysis

- Decision: Reject
- Scores: 4, 8, 8, 4

## Abstract
We present VHSMarker, a web-based annotation tool that enables rapid and standardized labeling of six cardiac key points in canine thoracic radiographs. VHSMarker reduces annotation time to 10–12 seconds per image while supporting real-time vertebral heart score (VHS) calculation, model-assisted prediction, and quality control. Using this tool, we constructed the Canine Cardiac Key Point (CCK) Dataset, a large-scale benchmark of 21,465 annotated radiographs from 12,385 dogs across 144 breeds and additional mixed breed cases, making it the largest curated resource for canine cardiac analysis to date. To demonstrate the utility of this dataset, we introduce MambaVHS, a baseline model that integrates Mamba blocks for long-range sequence modeling with convolutional layers for local spatial precision. MambaVHS achieves 91.8\% test accuracy, surpassing 13 strong baselines including ConvNeXt and EfficientNetB7, and establishes state-space modeling as a promising direction for veterinary imaging. Together, the tool, dataset, and baseline model provide the first reproducible benchmark for automated VHS estimation and a foundation for future research in veterinary cardiology. The source code and dataset are available on our project website: https://anonymousgenai.github.io/vhsmarker.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
They present VHSMarker, a web-based annotation tool that enables rapid and standardized labeling of six cardiac key points in canine thoracic radiographs. VHSMarker reduces annotation time to 10–12 seconds per image while supporting real-time vertebral heart score (VHS) calculation, model-assisted prediction, and quality control. Using this tool, they constructed the Canine Cardiac Key Point (CCK) Dataset, a large-scale benchmark of 21,465 annotated radiographs from 12,385 dogs across 144 breeds and additional mixed breed cases, making it the largest curated resource for canine cardiac analysis to date. To demonstrate the utility of this dataset, they introduce MambaVHS, a baseline model that integrates Mamba blocks for long-range sequence modeling with convolutional layers for local spatial precision.

### Strengths
- This paper provides an end-to-end complete solution of "tool-dataset-model", including an efficient annotation tool (VHSMarker), a large-scale dataset (CCK Dataset), and a high-performance benchmark model (MambaVHS).

- The CCK Dataset itself is a significant contribution. It contains over 21,000 standardized annotated images, addressing the long-standing pain point of insufficient large-scale, high-quality data in the field of veterinary AI.

- The proposed MambaVHS model achieves a high test accuracy of 91.8%, outperforming 13 powerful baseline models such as ConvNeXt and EfficientNetB7.

### Weaknesses
- While the paper contributes a valuable dataset, its value in the medical field is, in my view, greater than in the field of artificial intelligence. Therefore, it would be better suited for biology- or medicine-focused conferences or journals rather than the ICLR.

- The contribution of this paper lies in an engineering system (comprising a tool and a dataset) and an application case, rather than the proposal of new fundamental AI theories.

- The MambaVHS model primarily involves the combined application of existing mature modules (such as Mamba blocks, residual blocks, and SE layers), with limited original innovation in terms of model architecture.

- The MambaVHS model was only trained and tested on the CCK dataset. Its generalization ability on X-ray images from other institutions, captured using different equipment or following different protocols, remains unknown.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces three new openly shard tools for the analysis of canine radiography and Vertebral Herat Score (VHS) estimation. These tools include a large database of canine lateral radiographies, which includes 21,465 annotated images (Markers and VHS scores) from 12,385 across 144 breeds. The paper also introduces a new tool for radiography annotation. Finally the authors also introduce a novel MAMBA-based AI model for automated VHS estimation.

### Strengths
This paper introduces several original novel tools for the analysis of canine radiography, these includes a large database, a novel interface for the manual annotation of these images and an automated MAMBA technique for VHS estimation (including marker estimation).
The open sharing of these tools will allow for significant impact in the research community.

### Weaknesses
In the related work, the authors indicate that VHS scores suffer from high inter-observer variability. It is then weird to propose a dataset and automated tools for the same scoring system. However the results on a randomly selected 300-sample subset seem to indicate that the inter-observer is quite OK (>0.8).
The ground-truth annotation are relying on a single observer (which can be explained by the amount of work required to manually annotate the whole database (75 hours). How can the authors ensure the accuracy of these annotations?
Did the authors planned for the aggregation of multiple manual annotations in their annotation interface? That does not seem to be the case.

### Questions
1 Why were the other view(s) (dorsoventral/ventrodorsal) excluded from the CCK dataset? Is it due to a high number of missing data? Even these views are unhelpful for the VHS diagnosis, they could be useful for other applications. 
2 Are the authors planning to release the other views or additional annotations in the future? 
3 Could the authors describe a bit more the preprocessing applied to the images?  Are the images stored in the raw format or after preprocessing?
4 Was preprocessing applied before manual annotation?  Did the authors assess how often the annotators had to change image settings  (brightness - contrast) during the manual annotation process?
5 Did the authors assess the effect of preprocessing on the automated analysis?
6 Did the authors planned for the aggregation of multiple manual annotations in their annotation interface?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a large canine x-ray dataset with manually annotated heart and vertebra keypoints along with the derived vertebral heart score (VHS). This is motivated by the need to automate VHS calculation in veterinary cardiology, reducing subjectivity and time cost. Additionally, a keypoint extraction architecture based on Mamba is proposed which is shown to perform better than 11 representative convolutional and transformer-based architectures as well as MambaVision – a vanillla state-space model based Mamba architecture. Ablation studies of the architectural additions is done.  
Additionally, the annotation tool is also contributed as opensource, simple to use with possibility to use model-assisted prediction to accelerate annotation workload.

### Strengths
The paper fills the gap for the need for large dataset for canine cardiomegaly assessment. Although earlier dataset proposed were in hundreds, this is substantially larger (~21k).  
In addition, a rigorous benchmarking of the deep learning models is done to compare the performance of popular SOTA architectures in this dataset. 

The paper is easy to read, and the contributions as well as details required for reproducibility are described clearly. 

The contributed key point annotation tool with integrated model-based prediction is also useful.

### Weaknesses
In addition to popular off-the-shelf convolutional and transformer-based architectures, comparison of the proposed architectures with task-specific architectures proposed in the literature such as [1] could add to the rigor of the benchmarking experiment. 

*References*
1. Li, J., Zhang, Y. Regressive vision transformer for dog cardiomegaly assessment. Sci Rep 14, 1539 (2024). https://doi.org/10.1038/s41598-023-50063-x

### Questions
Please see the weakness section above. 

In addition, in table 3, Could you add confidence interval for Accuracy and MSE metrics too and show if MambaVHS performance is statistically significant.  
The description of Regression Head and architecture diagram in Fig 2 does not seem to match. In the figure, the mamba stages are serially placed but the text describes a aggregation from all four mamba stages.  
For completeness, the description of the \delta and middle multiplier, m, could be added in the main text rather than in the appendix.

### Soundness
3

### Presentation
4

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
This paper introduces VHSMarker, a web-based annotation tool designed to rapidly and accurately label six cardiac key points on canine thoracic radiographs, enabling real-time vertebral heart score (VHS) computation. Using VHSMarker, the authors developed the Canine Cardiac Keypoint (CCK) Dataset, a veterinary cardiac dataset, containing over 21,000 annotated radiographs from 12,385 dogs across 144 breeds. A baseline model, MambaVHS, is proposed, integrating Mamba state-space blocks for long-range anatomical dependency modeling with convolutional layers for fine spatial precision. The model achieves 91.8% accuracy on the CCK dataset, outperforming 13 strong CNN and Transformer baselines.

### Strengths
- Dataset: The paper delivers a large-scale dataset (21k+ radiographs) with standardized keypoint annotations, filling a gap in veterinary imaging research.

- Annotation tool: VHSMarker provides speedup (≈4.8× faster than MATLAB-based tools) and achieves high inter-observer agreement (κ ≈ 0.88).

- Strong baseline: MambaVHS effectively combines CNN spatial precision with Mamba state-space modeling, showing both computational efficiency (22h training vs. 90h for baselines) and accuracy gains (91.8% vs. ≤87.6%).

- Evaluation: Includes ablation studies, component analysis (SE layers, residual blocks), fairness controls with L1 loss, and Bland–Altman analysis confirming agreement with expert labels.

### Weaknesses
- Relevance for wider ICLR community: The paper proposes a very specific benchmark in a narrow application area of veterinary cardiology. I believe it would be a better fit for a different venue.

- Viewpoint limitation: Dataset focuses solely on lateral thoracic views; dorsoventral or ventrodorsal radiographs are excluded, limiting applicability to broader diagnostic cases.

- Model interpretability: While diagnostic performance is strong, there is limited discussion of interpretability or uncertainty quantification in clinical deployment.

- Clinical applicability: The clinical significance of 91.8% accuracy is not contextualized (e.g., what level of discrepancy is acceptable to veterinarians in practice).

- No comparison with human-in-the-loop correction: Although the tool supports hybrid annotation (Prediction + Show Both modes), results do not quantify improvement when experts refine model outputs.

### Questions
- Did the authors collect feedback from clinicians regarding usability and diagnostic trust in model-assisted annotation modes?

- What is the main intuition behind the superior performance of state-space modeling (Mamba blocks) compared to Transformers in this anatomical setting?

- Could the model’s predictions be augmented with uncertainty estimation or saliency maps to enhance interpretability for clinical adoption?

- How clinically significant are the observed improvements (e.g., ±0.2 VU mean error)? Would such differences alter a diagnostic decision in practice?

### Soundness
3

### Presentation
3

### Contribution
2
