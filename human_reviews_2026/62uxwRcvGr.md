# ECHOSAT: Estimating Canopy Height Over Space And Time

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Forest monitoring is critical for climate change mitigation. However, existing global tree height maps provide only static snapshots and do not capture temporal forest dynamics, which are essential for accurate carbon accounting. We introduce ECHOSAT, a global and temporally consistent tree height map at 10m resolution spanning multiple years. To this end, we resort to multi-sensor satellite data to train a specialized vision transformer model, which performs pixel-level temporal regression. A self-supervised growth loss regularizes the predictions to follow growth curves that are in line with natural tree development, including gradual height increases over time, but also abrupt declines due to forest loss events such as fires. Our experimental evaluation shows that our model improves state-of-the-art accuracies in the context of single-year predictions. We also provide the first global-scale height map that accurately quantifies tree growth and disturbances over time. We expect ECHOSAT to advance global efforts in carbon monitoring and disturbance assessment. The produced height maps will be made accessible upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors present ECHOSAT, the first spatio-temporal global-scale canopy height map at 10m resolution spanning seven years. They exploit multi-source satellite imagery to train a vision transformer that performs pixel-wise temporal regression with an adapted loss function designed for sparse temporal supervision. Analyses demonstrate that the model outperforms competing methods in single-date evaluations and that it learns realistic forest height dynamics over time.

### Strengths
1. The downstream application of canopy height estimation for large-scale biomass estimation is relevant and important for climate change understanding and mitigation.

2. Introducing optical and SAR time series for canopy height estimation at global scale and 10m resolution to include forest dynamics is a novelty and relevant for the task.

3. The design of the proposed Growth loss is a novelty and inspired by domain knowledge from the field of application. It accounts for potential forest disturbances, addresses missing LiDAR ground truth by filling gaps with regressions, employs constrained piecewise linear regressions to generate pseudo-labels, and combines real and pseudo-labels. 

4. An analysis of canopy growth and decline is provided (Section 4.2, Figure 2) thanks to the integration of time series. This contribution is well appreciated and could open discussions for future work.

### Weaknesses
1. There is a clear lack of related work on: 1/ spatio-temporal methods for remote sensing that have shown to learn spatio-temporal dynamics on various tasks [1, 2, 3], and 2/ remote sensing foundation models pretrained on large-scale optical and SAR datasets [4, 5, 6], sometimes used for downstream forest monitoring applications [6, 7]. One may note that these methods could be leveraged either as a starting point for architecture design or for canopy height estimation fine-tuning. 

2. Similarities and differences with other time series-based methods for forest monitoring remote sensing applications [8, 9, 10] have not been discussed. 

3. There is a clear lack of ablation studies to understand whether the performance gain comes from the architecture selection or the loss function design. 1/ One would appreciate a comparison of the Growth loss in its final form with a simpler formulation using a single linear regression, and with losses from competing works such as standard MSE. 2/ There is no comparison with simple competing methods (e.g., U-Net) combined with the Growth loss. 3/ It would have been appreciated to compare other architectures to the Temporal-Swin-Unet to better understand its relative performance gain, such as a 3D U-Net [11] or remote sensing-based architectures as mentioned above. 

4. The authors did not provide standard deviations of their quantitative results in Table 1, questioning the actual gain compared to competing methods. One may question the actual gain of the proposed method compared to Pauls et al. [12]: the proposed method achieves better performance on average, while absolute values of both methods are comparable, and Table 6 shows average errors per tree height that are similar within similar box plot interquartile ranges.

5. One would question the hypothesis of excluding labels below 5m according to Hansen et al. methodology [13] since: 1/ this guideline has not been followed by other competing methods, 2/ specific metrics for tree heights < 5m and > 5m could be easily defined to better distinguish use cases, and 3/ estimating tree heights < 5m is an actual use case for monitoring recent forest restoration projects through time. While all competing methods provide predictions < 5m (Table 4, right), it is not clear why the authors exclude this particular use case, whereas it represents a significant use case where the margin for improvement seems reasonable and would be useful. 

6. There is a lack of explanation about the train, validation, and test split definitions that must be clarified to avoid significant issues. As an example, the methodology followed by Pauls et al. [12] is questionable since the splits have been defined by random patches that could introduce data leakage between the train and test sets through spatial autocorrelation. 

7. The authors neither mention limitations of their work nor provide pathways to future work.

**References**:

[1] V. Sainte Fare Garnot & L. Landrieu, Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks. In ICCV 2021.

[2] M. Tarasiou et al., ViTs for SITS: Vision Transformers for Satellite Image Time Series. In CVPR 2023.

[3] G. Tseng et al., Lightweight, Pre-trained Transformers for Remote Sensing Timeseries. In ArXiv 2024.

[4] A. Fuller et al., CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders. In NeurIPS 2023

[5] G. Tseng et al., Galileo: Learning Global & Local Features of Many Remote Sensing Modalities. In ICML 2025.

[6] G. Astruc et al., AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities. In CVPR 2025.

[7] N. Bountos et al., FoMo: Multi-Modal, Multi-Scale and Multi-Task Remote Sensing Foundation Models for Forest Monitoring. In AAAI 2025.

[8] T. Nguyen et al., Multi-temporal forest monitoring in the Swiss Alps with knowledge-guided deep learning. In Remote sensing of environment 2024.

[9] K. Wu et al., A semantic-enhanced multi-modal remote sensing foundation model for Earth observation. In Nature machine intelligence 2025.

[10] Z. Yu et al., QRS-Trs: Style Transfer-Based Image-to-Image Translation for Carbon Stock Estimation in Quantitative Remote Sensing. In EEEI Access 2025.

[11] O. Cicek et al., 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. In MICCAI 2016.

[12] Pauls et al., Estimating Canopy Height at Scale. In ICML 2024.

[13] Hansen et al., High-resolution global maps of 21st-century forest cover change. In Science 2013.

### Questions
**Questions:**

1. What is the quantitative gain of the piecewise linear regressions versus using a single linear regression in the Growth loss function? 

2. Is the model trained from scratch or from pretrained weights? If it is trained from scratch or pretrained on natural image-based datasets, why did the authors not consider exploiting various remote sensing backbones pretrained on large-scale optical and SAR datasets? Note that some of them are actually designed to exploit time series [1, 2, 3, 4]. 

3. Height estimation is a proxy for biomass estimation, which is the most important downstream application. In this work and related work, errors are mostly quantified via height estimation. However, what would be the equivalent of the height error in biomass estimation? Since the global allometric equation is not linear, this link is not straightforward and has been barely studied in previous canopy height map estimation works. Where are the biomass errors most important (small, medium, large trees) in average and absolute values? 

4. Considering the increasing number of UAV LiDAR datasets [5, 6, 7], why did the authors not attempt to better evaluate methods on more precise datasets than GEDI-based annotations?


**Comments:**

• Please add numbers to relevant equations on page 5.

• One would appreciate integrating Figure 6 into the main paper, as the presented results are valuable and insightful.

• L. 353: "For 2019-2022, MAE values range from 5.36 m to 6.27 m, indicating consistent prediction accuracy." One may consider softening this claim since a 5-6m error on trees between 10-30m is quite significant for estimating their biomass.

• Section 4.1: One would appreciate an additional analysis of errors through time per tree height range, similar to Table 6.

• Figure 5: Please integrate a few GEDI LiDAR point clouds within the same color scale on the Google Map images to better understand the order of magnitude of the ground truth. With the current form of the figure, we can visually observe the difference in resolutions but cannot assess the quality of height predictions.


**References:**

[1] M. Tarasiou et al., ViTs for SITS: Vision Transformers for Satellite Image Time Series. In CVPR 2023.

[2] G. Tseng et al., Lightweight, Pre-trained Transformers for Remote Sensing Timeseries. In ArXiv 2024.

[3] G. Tseng et al., Galileo: Learning Global & Local Features of Many Remote Sensing Modalities. In ICML 2025.

[4] K. Wu et al., A semantic-enhanced multi-modal remote sensing foundation model for Earth observation. In Nature machine intelligence 2025.

[5] S. Puliti et al., FOR-instance: a UAV laser scanning benchmark dataset for semantic and instance segmentation of individual trees. In ArXiv 2023.

[6] B. Xiang et al., ForestFormer3D: A Unified Framework for End-to-End Segmentation of Forest  LiDAR 3D Point Clouds. In ICCV 2025.

[7] M. Wielgosz et al., SegmentAnyTree: A sensor and platform  agnostic deep learning model for tree  segmentation using laser scanning data. In Remote Sensing of Environment 2024.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
SUMMARY: The paper focuses on the task of temporal tree canopy height estimation from remotely sensed data. As the authors outline, this task is relevant for global forest monitoring and directly impacts downstream applications like carbon stock or sequestration estimates. The authors aim to provide a global map - and the first map over time, allowing users to assess changes. Methodologically, the authors propose a new neural net architecture to model tree height, the Temporal-Swin-Unet. This is a combination of two existing methods, the Video Swin Transformer and the Swin Unet. The authors then evaluate their approach by comparing it to existing global tree height maps, showing consistently improved accuracy.

### Strengths
STRENGTHS:
I particularly enjoyed the following aspects of the paper:
- The paper is well motivated and tackles an incredibly important real-world task that is clearly suited for ML/vision models.
- The paper is very well written; it is easy to understand and has a good / intuitive flow.
- Part of that is the papers simplicity; the paper (mostly, exceptions below) has a great balance of depth and simplicity; the proposed methodological advancement is simple but quite elegant and well motivated by the problem setting.

### Weaknesses
SHORTCOMINGS:

I have two major concerns with the current draft of the manuscript:

- Part of the "growth loss" is the disturbance indicator. The paragraph introducing it is too short and it is unclear how the disturbance indicator motivated? choice of thresholds here seem arbitrary? The authors say that "A disturbance is considered to occur in zref ∈ RY when a) tree height decreased by more than 50% and more than 4 m and b) tree height decreased to less than 10 m within two years." (line 224), but the choice for these numbers are not explained at all and seem arbitrary. I assume there is some sort of expert knowledge behind them but this NEEDS to be explained!

- Crucial experiments on the performance over space and time are missing. It would be very important to know if the method performs equally well everywhere on the planet, or whether there are areas of higher and lower performance. This should follow e.g. the analysis in [1] (see Extended Data Fig. 1 in [1]). This sort of knowledge is very important for on-the-ground practitioners. Secondly, an analysis of the performance over time would be equally interesting / important. I see that the authors say that "Due to the sparse temporal and spatial distribution of GEDI labels, a temporal validation with GEDI is not possible. " (line 366) - What is meant by this exactly? Are GEDI labels not spatio-temporally aligned (e.g. a given location only occurs on one time step)? You should still be able to assess temporal performance by averaging errors for a given time step. Am I missing something?

- A more minor point is that I would like to see some discussion of the applicability of the method to local height mapping problems. Specifically [2] argues that these sort of global tree height maps fall short of being actually useful in many local applications. I would be curious how the authors contextualize their work within this critique.

### Questions
Overall this a paper tackling a relevant real-world problem and introducing an intuitive new method. I'd ask the authors to consider my questions and concerns in the "weaknesses" section. 

References:

[1] Lang, Nico, et al. "A high-resolution canopy height model of the Earth." Nature Ecology & Evolution 7.11 (2023): 1778-1789.

[2] Rolf, Esther, et al. "Contrasting local and global modeling with machine learning and satellite data: A case study estimating tree canopy height in African savannas." arXiv preprint arXiv:2411.14354 (2024).

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
4

### Summary
This work introduces *ECHOSAT*, a global 10m tree canopy height time series spanning 2018–2023. A Vision Transformer performs pixel-level temporal regression on multi-year satellite imagery (Sentinel-2, Landsat) and sparse GEDI height labels. A novel growth loss is introduced to regularize predictions to follow realistic tree dynamics (gradual growth, sudden loss from fires/logging) without need for post-processing. The model improves over prior single-year SOTA methods on held-out GEDI data (RMSE=10.87m) and demonstrates disturbance detection (F1=0.82). Public release of height maps is planned.

### Strengths
- **Originality**: The growth loss enforces monotonic height increase and abrupt drops; this is the first global 10m multi-year canopy height map with inherent temporal modeling 
%%previous work looks at single years only.

- **Quality**: The model uses multi-sensor data and sparse GEDI labels for GT. Ablations isolate growth loss impact on held-out GEDI (Table 1, p. 8).

- **Clarity**: The paper is very well-written; it's clearly structured, with well-explained methods; outlines explicit contributions, provides clear mathematical formulations, and effective visuals.
- **Significance**: The resulting height map time series supports global-scale monitoring of forest growth and disturbance, with applications in carbon accounting and climate mitigation. The planned public release of height maps is a valuable contribution.

### Weaknesses
- **Clarity**: Equations are unnumbered, making referencing difficult.
- **Significance**: Height-to-carbon flux not evaluated -- above-ground biomass (AGB) to CO₂ conversion or flux tower validation would enhance climate impact, i.e. in carbon accounting.

- **Originality**: The main novelty lies in the growth loss (Sec 3.3), but a comparison to learned temporal dynamics in *TimeSformer* (Bertasius et al., ICCV 2021) or *EarthFormer* (Gao et al., NeurIPS 2022) would help quantify value the loss adds beyond attention-based modeling.  
- **Quality**: Results are strong against modern single-year baselines, but temporal SOTA comparisons could further strengthen the authors' claims.

### Questions
1. Consider comparing to *TimeSformer* or *EarthFormer* -- Replacing your ViT encoder with one of these approaches that learn temporal dynamics via space-time attention would help clarify the advantage your growth loss provides.

2. Validate carbon flux using height -- Height is a solid proxy, but for carbon accounting claims, it's valuable to estimate flux by converting your maps to AGB then CO₂  uptake/release using allometry like Jucker et al. (2022), validated against a flux tower site (e.g., Harvard Forest with public data).This additional step would ground the significance in real carbon metrics and strengthen the climate impact from my perspective.

3. To improve mathematical clarity, consider numbering equations -- this would make it easier to reference formulations like the growth loss and follow derivations.

### Soundness
4

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
5

### Summary
ECHOSAT provides a 50TB multi-modal (radar, multi-spectral, LiDAR), multi-temporal (monthly composites from 2018 to 2024) dataset from Earth observation with 3 million globally sampled geolocations of shape 18x84x96x96 (= channel x time x widht x height) at 10m pixel resolution. The authors explore the performance (Tab. 1, Figs. 3 & 4) of a Video-Swin-UNet-like architecture (Fig. 1) to predict tree height as determined by the GEDI sensor (mounted onto the ISS) when the network is trained by an additional loss that restricts tree growth to physical bounds (Sect. 3.3). Results indicate an advantage over existing methods.

### Strengths
The paper is clearly structured, written in plain English, with the main text accommodated by illustrative figures, equations, tables, and appendices with additional details. The dataset is carefully curated (App. A.2.1) and the methodology well documented (App A.2.2). Experimental results are cleanly evaluated against existing methods (Sect. 4.3).

### Weaknesses
The work falls short in major novelties for the ICLR community regarding learning representation methods. While ECHOSAT resembles a valuable dataset for the Earth observation community, the Temporal-Swin-Unet (Sect. 3.2) blends minor adjustments (1x1 patch size, additional layers and skip connections) from existing architecture. The additional *Growth Loss* (Sect. 3.3) is specific to the application of tree height mapping, and resembles a neat, but limited innovation.

Unfortuantely, the authors evaluate the model performance on (hold-out) GEDI data the model was trained on. An independent modality to verify the temporal evolution of tree heights predicted, and a qualitative comparison to corresponding field surveys is missing. However, I appreciate the author's discussion of qualitative investigations such as in Fig. 7.

Given the Earth observation modalities fused ship in various spatial resolutions, I would appreciate a more detailed discussion around upsampling strategies to 10m per pixel, and their consequences. In particular, the label source GEDI for tree height estimation probes geospatial scenes at about 25 meter footprints. Discontinuities in tree height are common at forest boundaries and in areas disturbed by wild fires and logging.

I rate the paper a valuable scientific piece of work carefully conducted in general, but I believe it would better fit the scope of an Earth observation conference, a computer vision conference with geospatial tracks, or a high-profile domain journal. However, if other reviewers read the paper and consider my input to come to the conclusion this work fits the scope of ICLR, I am fine with acceptance.

- typos:
  * l105: typo _captures_ to _capture_
  * l633: $20\circ$ to $20^\circ$

### Questions
- Which license will the ECHOSAT dataset and the Temporal-Swin-Unet be published under?
- Please provide a table with dataset and its source utilized. In particular which datasets have been pulled from Google Earth Engine, and how was geospatial alignment implemented?

### Soundness
3

### Presentation
4

### Contribution
2
