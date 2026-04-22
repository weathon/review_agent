# GeoFAR: Geography-Informed Frequency-Aware Super-Resolution for Climate Data

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Super-resolving climate data is crucial for fine-grained decision-making in various domains, ranging from agriculture to environmental conservation. However, existing super-resolution approaches struggle to generate the high-frequency spatial information present in climate data, especially over regions showing complex terrain variability. A key obstacle lies in a frequency bias existing in both deep neural networks (DNNs) and climate data: DNNs exhibit such bias by overfitting to low-frequency information, which is further exacerbated by the prevalence of low-frequency components in climate data (e.g., plains, oceans). As a consequence, geography-dependent high-frequency details are hard to reconstruct from coarse climate inputs with DNNs. To improve the fidelity of climate super-resolution (SR), we introduce GeoFAR: by explicitly encoding climatic patterns at different frequencies, while learning implicit geographical neural representations (i.e., related to location and elevation), our approach provides frequency-aware and geography-informed representations for climate SR, thereby reconstructing fine-grained climate information at high resolution. Experiments show that GeoFAR is a model-agnostic approach that can mitigate high-frequency prediction errors in both deterministic and generative SR models, demonstrating state-of-the-art performance across various spatial resolutions, atmospheric variables, and downscaling ratios. Datasets and code are available at https://eceo-epfl.github.io/GeoFAR/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors argue that during image super-resolution using DNNs, DNNs tend to preserve low-frequency structures while neglecting the retention of high-frequency components. This leads to performance degradation when applying DNNs to meteorological data super-resolution tasks, which are predominantly low-frequency. To address this issue, the authors propose GeoFAR, a plug-and-play module. GeoFAR encodes input data into frequency-domain representations and additionally introduces implicitly encoded spatial information as input to the DNN, thereby improving performance on climate data super-resolution tasks dominated by low-frequency content. The authors validate the effectiveness of their method across multiple datasets and various network architectures.

### Strengths
1. GeoFAR is a plug-and-play method, and the authors demonstrate that it is better suited for meteorological data super-resolution tasks compared to commonly used encoding methods such as DWT.

2. GeoFAR shows performance improvements across multiple neural network architectures, suggesting that performing super-resolution in the frequency domain is a more suitable approach for meteorological data super-resolution, offering new insights to the community.

3. Implicitly encoding spatial information into features usable by neural networks is an important research direction in recent years, and the authors present a compelling practical application in the context of super-resolution.

### Weaknesses
Overall, I find this an interesting paper, but the experimental validation raises concerns:

1. Directly downsampling high-resolution data to low resolution to form LR-HR pairs is an unreasonable practice. Such interpolation retains excessive high-frequency information. Given that the authors do not provide a detailed description of dataset construction, I question the validity of their datasets. Retaining excessive high-frequency details in the LR data constitutes information leakage—could this be the actual reason why GeoFAR, which optimizes in the frequency domain, appears effective? As shown in Table 1, comparing Global-to-Local Downscaling and Local Downscaling, the UNet results indicate that when the task involves real low-resolution to high-resolution data (e.g., ERA5-PRISM), the improvement from GeoFAR is significantly smaller than in simulated LR-HR scenarios.

2. The current experimental results lack comprehensive validation. In quantitative remote sensing, it is common practice to use multiple independent data sources for cross-validation—for example, jointly using reanalysis data and actual remote sensing observations—rather than relying solely on reanalysis data.

3. The paper suffers from significant writing issues, including imbalanced content depth, insufficient discussion of key issues, and poor organization. For instance, the related work section merely lists numerous existing methods without establishing a clear causal chain between “existing approaches” and the “problems summarized at the end of the section,” nor does it validate certain claims. Specifically, in Section 2.2, the authors hastily assert that “current methods remain based on natural images while poorly mimicking the biased frequency distribution of climate data,” yet they fail to provide a mathematical explanation for why existing methods underperform on meteorological data, nor do they compare against recent frequency-domain-based super-resolution methods presented at major conferences (e.g., BF-STVSR, DiffFNO).

### Questions
1. Can the authors validate the accuracy of their downscaling model using observational data—for example, by comparing against ground meteorological station measurements (evaluating accuracy at station-corresponding grid locations), MODIS LST (compared with ERA5 SKT), and GPM IMERG precipitation data (compared with ERA5 precipitation)?

2. Can the authors provide additional details regarding dataset construction and validate the reasonableness of their approach? For instance, do the downsampled data exhibit similar high-frequency content and entropy to real-world low-resolution observations? To my knowledge, directly interpolating HR remote sensing data to LR resolution often retains excessive high-frequency details—a phenomenon easily verifiable by downsampling Sentinel-2 data to 500m and comparing it with MODIS data. The remote sensing community typically simulates LR data using multi-stage downsampling combined with smoothing. Could different dataset construction strategies lead to different conclusions?

3. The authors repeatedly emphasize in the related work that existing DNNs fail to preserve high-frequency information and produce overly smooth outputs. However, as shown in Table 1, most tested DNNs already yield reasonably good results, and GeoFAR does not demonstrate a fundamental improvement. Does this suggest the authors overstate the limitations of current methods?

4. GeoFAR essentially performs input-level data augmentation without modifying the internal architecture of the DNN. Will the injected high-frequency information still be forgotten by the network? Does GeoFAR truly address the core issue, or does it merely provide a superficial fix?

5. How does GeoFAR compare against methods that explicitly incorporate frequency-domain considerations at the operator level? What about the latest methods from CVPR or ICCV 2025? Could GeoFAR be combined with them for even greater gains? (This would help substantiate the claim that GeoFAR is truly plug-and-play.)

6. As an open-ended question: how well does GeoFAR perform on super-resolution of various remote sensing indices—for example, NDVI (from MODIS 500m to Landsat 30m)? Super-resolution of remote sensing indices is more amenable to cross-validation using independent high-resolution observational data.

7. Can the authors add a clear statement of contributions? Currently, the paper’s contributions remain vague.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
GeoFAR: Geography-Informed Frequency-Aware Super-Resolution for Climate Data
In this paper, the authors introduce GeoFAR, a novel approach to climate data super-resolution that explicitly addresses the frequency bias of neural networks and the geography-dependent variability that is inherent to climate data. The paper integrates a Frequency-Aware Representation (FAR), which is a convolutional module leveraging Discrete Cosine Transform (DCT) bases that separate and encode high- and low-frequency information. This approach also uses a Geography-Informed Implicit Neural Representation (Geo-INR), which uses a coordinate-based representaiton that jointly encodes spatial and elevation information using spherical harmonics and SIREN networks. GeoFAR can be used with any deterministic or generative SR model, and the authors impressively demonstrate SOTA results across different spatial resolutions, atmospheric variables, and downscaling ratios.

### Strengths
* well-framed motivation (Figure 1 is strong), supported by innovative design using frequency- and geography-aware representations
* having a model-agnostic design makes this approach easily adaptable
* the authors include extensive experiments across datasets, resolutions, and atmospheric variables to demonstrate the improvements from their approach. there seems to be consistent improvement over baselines, and the work clearly demonstrates that GeoFAR helps with high-frequency fidelity and reduces bias.
* analysis is well-structured, with comprehensive ablation studies and frequency+elevation sensitivity studies (Figure 5 was particularly interesting)

### Weaknesses
* not a necessity, but it would make the dense notation and equations in  more intuitive with a diagram (it could even go in the supplementary materials)
* although the comparisons in the authors’ experiments are broad, there are few diffusion baselines included (SR3 from Saharia et al. 2023; Watt & Mansfield 2024) in quantitative analysis and other methods could be helpful to look into (PhIREGAN from Stengel et al. 2020), despite being mentioning that they are strong recent approaches
* while the quantitative analysis is very informative on the whole, it would be food to include statistical analysis with variance across seeds if this is possible
* to more robustly evaluate the physical fidelity of the predictions, consider using additional evaluation frameworks such as generating kinetic energy spectra from the wind fields
* section 4.4 has the beginnings of this, but the paper would benefit by a deeper discussion of how to physically interpret the frequency and geography components and how they related to known phenomena
* in the appendix, it would be good to include a brief discussion of computational cost/runtime scaling…efficiency is another dimension that’s important to consider in the practical deployability of these methods
* it would be helpful to cite the relevant frequency-aware transformer works and coordinate-based climate model papers; the authors may also need to reference some work in the neural operators space (see Li et al. 2021) because they also tackle spectral bias in physical modeling

### Questions
* are the reported quantitative results averaged over multiple training seeds, or based on a single run?
* can you elaborate on how the frequency-aware and geography-informed components interact during training? for instance, does the Geo-INR modulate the FAR in a spatially adaptive way tied to physical processes, or mainly improve general representation quality?
* this is more of an aside, but the paper shows results for temperature and geopotential. how well does GeoFAR generalize to other variables (e.g., precipitation, humidity) with different spatial-frequency characteristics?
* how sensitive is the model’s performance to the truncation degree (L) in the spherical harmonics encoding? have you explored trade-offs between model accuracy and computational cost?
* could you clarify whether Geo-INR can be pretrained or shared across datasets/variables, or must it be trained jointly with each SR model?
* is there a timeline or specific repository planned? are there any preprocessing scripts or model-specific configurations that will be included for reproducibility?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposed GeoFAR to address the issue of overfitting to low-frequency components in climate data downscaling. GeoFAR explicitly encodes climatic patterns at different frequencies and use the location and elevation information. The model was evaluted in both deterministic and generative super-resolution models over different datasets with various spatial resolutions, atmospheric variables, and downscaling ratios.

### Strengths
1.	The paper addresses an important issue in climate downscaling that existing models are often biased toward low-frequency components.
2.	The paper conducts comprehensive experiments over different datasets, different downscaling ratios over different variables and the ablation experiments show the improvements brought by each component of the proposed approach.

### Weaknesses
1.	The proposed model appears to use a combination of geolocation aware encoding and frequency-aware convolution kernel, which are similar to existing ideas. As example, the novelty of the frequency-aware convolution kernel using 2D DCT with N×N grid instead of 4 zones is not clear. It seems 2D DCT as the kernel weight has also been used in computer vision models such as Harmonic Convolutional Networks. Fcanet also uses similar idea to take DCT as the weight.
Harmonic convolutional networks based on discrete cosine transform. Pattern Recognition.
Fcanet: Frequency channel attention networks. In Proceedings of the IEEE/CVF international conference on computer vision.

2.	Though the model has compared many baseline models such as generative and deterministic models, the baseline models are a little bit out of date. For generative models, including more newer super resolution models will be helpful, like: Precipitation downscaling with spatiotemporal video diffusion. Advances in Neural Information Processing Systems. 2024.

### Questions
Please better explain the novelty of paper in terms of the model structure design. What is the key contribution of the frequency-aware convolution kernel?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes GeoFAR, a new method for climate downscaling. GeoFAR resolves the oversmoothness issue of existing deep learning models by explicitly encoding input data at different frequencies and injecting the model with geographical neural representations for location and elevation. Experiments show that GeoFAR performs better than existing baselines and helps mitigate high-frequency prediction errors in both deterministic and generative models.

### Strengths
- The paper aims to address an important problem in climate downscaling: oversmoothness and bias towards low-frequency information.
- The paper is well-written and the proposed method is relatively simple (which is good).

### Weaknesses
I have two main concerns: the soundness of the proposed method and the significance of the empirical results.
- The two proposed components, explicit frequency-aware encoding and geographical representation learning, aim to exploit high frequencies in the input, but what I think is more important is the output or ground-truth. When we train a deep learning model with an MSE loss, for example, regardless of how the model processes the input data, it can still learn to over-optimize for the low-frequency information in the ground-truth, and thus will still predict oversmooth fields.
- In practice, since the lower-resolution input is often smoother, exploiting high-frequency information in the input itself seems like a minimal gain.
- Empirically, there are only minor improvements when using GeoFAR, especially when compared with the UNet baseline. In the ablation study, I believe that if the Unet architecture is used, the difference would be minimal.
- Qualitative results in Figures 4 and 7 also show almost identical predictions of the ViT model with or without GeoFAR.

### Questions
- The Frequency-aware Convolution Kernels are fixed convolution kernels. Can they be learned so that the model is more data-driven?
- Can the proposed method be applied to multi-variable data? The experiments trained separate models for each variable, but I believe it's beneficial to use multiple variables in the input, so the model can exploit interactions between them and make better predictions.

### Soundness
2

### Presentation
2

### Contribution
2
