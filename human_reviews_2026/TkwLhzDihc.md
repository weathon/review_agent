# Transforming Weather Data from Pixel to Latent Space

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 6, 8, 4

## Abstract
The increasing impact of climate change and extreme weather events has spurred growing interest in deep learning for weather research. However, existing studies often rely on weather data in pixel space, which presents several challenges such as smooth outputs in model outputs, limited applicability to a single pressure-variable subset (PVS), and high data storage and computational costs. To address these challenges, we propose a novel Weather Latent Autoencoder (WLA) that transforms weather data from pixel space to latent space, enabling efficient weather task modeling. By decoupling weather reconstruction from downstream tasks, WLA improves the accuracy and sharpness of weather task model results. The incorporated Pressure-Variable Unified Module transforms multiple PVS into a unified representation, enhancing the adaptability of the model in multiple weather scenarios. Furthermore, weather tasks can be performed in a low-storage latent space of WLA rather than a high-storage pixel space, thus significantly reducing data storage and computational costs. Through extensive experimentation, we demonstrate its superior compression and reconstruction performance, enabling the creation of the ERA5-latent dataset with unified representations of multiple PVS from ERA5 data. The compressed full PVS in the ERA5-latent dataset reduces the original 244.34 TB of data to 0.43 TB. The downstream task further demonstrates that task models can apply to multiple PVS with low data costs in latent space and achieve superior performance compared to models in pixel space.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a Weather Latent Autoencoder (WLA) that transforms the high storage weather data in pixel space to latent space, reducing the data storage costs and the computational costs for downstream tasks such as weather forecasting.

The core part of the algorithm consists of three parts. First, the Pressure-Variable Unified Module (PVUM) utilizes multiple Pressure-Variable Subsets (PVS), effectively preserving relationships between weather data in pixel space by mapping similar weather variables and adjacent pressure levels to similar unified features. Next, the information is passed through an encoder for dimensionality reduction, which is further supported by a binary quantization module to ensure low data storage. For downstream tasks such as pixel level prediction, weather latent decoder with other needed reverse operations are followed to reconstruct the latent information/predictions to the pixel space.

### Strengths
- The overall structure of the paper is compact, well-written and easy to follow. Each component of the proposed architecture is explained thoroughly and separately, with clear reasoning provided for each design choice.
- The first part of the experiments, focusing on the compression capabilities of the proposed algorithm, provides a solid comparison of WLA’s performance (Table 1). The algorithm is evaluated against various methods using multiple metrics, which clearly shows the advantage of using the proposed WLA.

### Weaknesses
- The experiments on downstream tasks using the proposed latent space are quite limited. Only the weather forecasting task is evaluated, even though the latent dataset is arguably the most important contribution of this work. I would expect evaluations on additional downstream tasks in the latent space. Specific to the weather forecasting experiments, the authors compare against only two prior models, use only the RMSE metric, and rely on a single dataset. Although the latent-based Window-ViT performs similarly to the original pixel-based version, the evaluation remains shallow and should be expanded.
- The entire work is based on the ERA5 dataset, with no evaluation of how the model performs on other weather datasets. The authors state that their WLA is primarily designed for ERA5 data at a 0.25° resolution, which limits the paper’s overall impact. It is unclear whether the authors intend their method to be used solely for ERA5 data at this resolution. If so, they should provide more comprehensive results for downstream tasks or at least a more focused evaluation of the weather forecasting task. If not, they should include results on similar datasets to demonstrate generalizability.
- Although an ablation study is included to support the model architecture, it focuses mainly on varying hyperparameters rather than analyzing specific components of the model. Since this is a purely experimental work, a more detailed ablation study would be much more valuable.

### Questions
- See the weaknesses section for my detailed comments. In summary, while the proposed transformation to latent space appears intuitive, the experimental analysis is insufficient to fully support the chosen model architecture. Given that this is a purely empirical work, current state of the experimental evaluation is insufficient.
- Reproducibility of the experiments is crucial in work like this. The authors state that the code and the constructed ERA5-latent dataset are available at WLA but I could not understand what the authors are referring to with that statement.

### Soundness
2

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
This paper introduces an architecture to transform high-dimensional weather data from its native pixel space into a compressed and unified latent space. The proposed model leverages existing components, including a VAEformer architecture adapted from CRA5 for the encoding-decoding process and a Binary Quantization Module (BQM) from BSQ for creating storage-efficient discrete tokens.
The central innovation of the architecture is the Pressure-Variable Unified Module (PVUM). This module is specifically designed to address the heterogeneity of meteorological data by transforming various Pressure-Variable Subsets (PVS), different combinations of weather variables at multiple atmospheric levels, into a single, standardised latent representation.
The primary result of this framework is the ability to generate compact embeddings that achieve a high data compression rate while demonstrating competitive performance in reconstructing the original weather fields.

### Strengths
- The novel framework decouples reconstruction from forecasting, leading to sharper and more accurate predictions of extreme weather events compared to blurry pixel-space models. 
- The innovative PVUM module allows a single model to adapt to diverse combination of weather variables and pressure levels.
- It creates and releases ERA5-Latent, a >500x compressed version of ERA5 (244TB -> 0.43TB), drastically lowering storage costs and making global weather research more accessible to everyone.
- The latent-space forecasting model is not only efficient but also highly competitive with state-of-the-art methods like Pangu-Weather in both overall accuracy and extreme event prediction.

### Weaknesses
The paper justifies its flexible PVS approach with diverse applications (e.g., typhoon tracking, rainfall forecasting), yet the downstream evaluation is limited to a single weather forecasting task, leaving the broader utility of the latent space untested.
The "low-cost" claim focuses on downstream efficiency but omits the substantial upfront computational cost required to generate the ERA5-Latent dataset. This significant one-time investment should be more transparently framed as a trade-off.
Reconstruction error maps reveal that information loss, while minimal, is systematically concentrated in areas of physical extremes. This is a potential concern, as the fidelity of these very regions is often critical for the high-impact weather tasks the embeddings are designed to support.

### Questions
- The paper's motivation for the flexible PVUM is compelling, citing diverse applications like typhoon tracking and rainfall forecasting. However, the downstream evaluation is limited to a single, general weather forecasting task. Could the authors demonstrate the framework's utility on one of these other applications? This would provide stronger evidence that the PVS-agnostic approach is meaningful for the distinct tasks used to justify it.
- The current forecasting task excludes precipitation, which is a critical and notoriously difficult variable. Including a variable like 6-hour accumulated precipitation (tp6h) would provide a much more stringent and comprehensive test of the latent-space methodology.
- While the SEDI metric effectively evaluates extreme event detection, it may not fully capture errors in extreme event magnitude. Would it be possible to also report a metric like latitude-weighted RMSE? This is standard in meteorological literature and would offer a clearer understanding of the performance on extreme values from a regression perspective.
- The ablation on codebook size is insightful, but it doesn't disentangle the representation error from the VAEformer itself versus the quantization error introduced by the BQM. Could the authors add a baseline that evaluates the reconstruction performance using the continuous floating-point latent vectors before quantization? This would establish a clear upper bound on performance and precisely quantify the impact of the binary compression scheme.

### Soundness
3

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
5

### Summary
This paper proposes a latent-space representation framework for global weather data, aiming to reduce the massive storage and training cost of pixel-based numerical datasets such as ERA5. The authors design a Pressure-Variable Unified Module (PVUM) to encode multiple meteorological variables into a shared latent embedding and release a highly compressed ERA5-Latent dataset (≈566× smaller).
The approach is evaluated on downstream forecasting tasks, showing competitive skill compared with pixel-based models (e.g., Window-ViT, Pangu-Weather). Overall, the work provides a practically meaningful contribution toward scalable AI-based weather modeling, although its methodological novelty is limited.

### Strengths
**Extremely practical contribution.**

ERA5 is widely used but heavy; this work provides a realistic solution to handle it efficiently.
Compressing hundreds of terabytes into manageable latent tensors without large accuracy loss is impressive and genuinely useful for the community.

**Engineering excellence.**

The PVUM design and binary quantization are cleanly implemented and well-documented.
The system could enable broader reproducibility and accessibility of S2S and NWP datasets.

**Strong motivation and clear writing.**

The authors clearly articulate why latent-space modeling matters for next-generation AI forecasting.
The figures, pipeline descriptions, and dataset details are easy to follow.

### Weaknesses
I personally really like this paper — it solves a very real problem.

I often wanted to use ERA5 myself but found it too massive to handle; this work genuinely makes that easier.

However, the paper feels incomplete from a comparative standpoint. For a contribution centered on latent compression, it should directly compare against other latent or learned-compression works such as "COMPRESSING MULTIDIMENSIONAL WEATHER AND
CLIMATE DATA INTO NEURAL NETWORKS"

These baselines are essential to contextualize how effective the proposed PVUM and quantization pipeline really are.
More importantly, the downstream forecasting evaluation is too limited. To fully validate the usefulness of this latent representation, the authors should run — or at least discuss — results where GraphCast, Pangu-Weather, or FuXi are trained or fine-tuned on their latent ERA5. That would strongly demonstrate the framework’s universality and convince readers it’s not just a neat compression trick but a foundation for scalable weather AI.

That said, I believe this model would probably work well even with those SOTA architectures — it just needs to be shown.
The work is already strong; these additional experiments would make it truly excellent.

### Questions
1. Have you tested whether models like GraphCast or FuXi can operate directly in your latent domain?
2. If not, do you expect architectural adaptation to be required?
3.How much information loss occurs for small-scale or extreme weather events after latent quantization?

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
2

### Summary
This paper tackles three significant challenges in applying deep learning to weather research: the high data storage and computational costs of pixel-space data, the limited applicability of models trained on a single Pressure-Variable Subset (PVS), and the tendency for pixel-space models to produce "smooth" outputs that miss small-scale extreme values. The authors propose a novel Weather Latent Autoencoder (WLA) to transform high-resolution weather data from pixel space into a highly compressed latent space. The WLA features a Pressure-Variable Unified Module (PVUM), which uses a hypernetwork to generate adaptive weights based on metadata (e.g., pressure levels), theoretically allowing it to map any PVS to a unified representation. The paper also introduces a "Latent Space Framework" where downstream tasks, like forecasting, are trained and executed entirely in the low-storage latent space, using binary tokens.

### Strengths
1. **Significance and Problem Formulation:** The paper identifies and clearly articulates three critical, real-world bottlenecks in meteorological deep learning (cost, PVS-rigidity, and smoothness). A solution that effectively addresses all three would be a major contribution to the field.

2. **Novelty of PVUM:** The Pressure-Variable Unified Module (PVUM)  is a novel and clever architecture. The use of a hypernetwork to generate a linear mapping's weights and biases based on physical metadata (like pressure levels)  is an elegant solution for creating a unified representation from heterogeneous inputs. The OOD generalization experiment (Section 4.3, Figure 5)  strongly supports this component's ability to generalize to unseen pressure levels and even out-of-domain (HRES) data.

3. **Novelty of Latent Space Framework:** The proposal to perform downstream tasks (training, validation, and testing) directly on the compressed bitwise tokens is a significant conceptual strength. Most compression research focuses on storage, with decompression being a prerequisite for any model use. This framework (Section 3.4) directly targets the computational costs of the downstream task itself, not just the storage/IO.

4. **Latent dataset.** The creation and open-sourcing of the ERA5-Latent dataset represents a substantial engineering effort. Compressing 244.34 TB of data into 0.43 TB  is, on its own, a significant undertaking that, if valid, could be a valuable community resource.

5. **Strong Downstream Task Validation:** The paper rightly demonstrates that the compression is not just good in theory (low RMSE/bpsp) but is useful in practice. The weather forecasting experiment (Section 4.6) directly validates two of the main claims: PVS-Adaptability and Sharpness.

### Weaknesses
1. The paper's primary novel contribution, the Pressure-Variable Unified Module (PVUM), is invalidated by the authors' own appendix. The introduction claims the PVUM enables a "Unified Pressure-Variable Representation"  that can transform "any pressure-variable subset". However, Appendix A.3 (L830-833) states: "First, to ensure high-quality reconstruction of the weather data, we trained a separate weather latent autoencoder for each of the six variables...". This is a direct contradiction. The PVUM does not unify different variables (e.g., Z, U, V) if separate models are required for each. This invalidates Contributions 1 and 2  and nullifies the paper's main claim to architectural novelty.

2. The core experimental "proof," the downstream forecasting comparison in Section 4.6  between the pixel-space Window-ViT (WT) and the latent-space Window-ViT-Latent (WTL), is a confounded, apples-to-oranges comparison. The WT model (based on FengWu)  must solve the hard, coupled problem of forecasting and pixel-space reconstruction. The WTL model "omits the downsampling and upsampling components"  and solves the much simpler problem of predicting discrete latent tokens. This experiment does not prove the latent framework is superior; it proves that solving an easier problem is easier.

3. The paper heavily advertises "sharper" and more accurate extreme event prediction, using Figure 7a  as proof. This result is a methodological artifact of the loss functions used, not a feature of the latent framework. The WLA (the autoencoder responsible for reconstruction) is explicitly trained with a "GAN loss". The downstream WTL (latent) model is trained with "binary cross-entropy loss". The WT (pixel) baseline, FengWu (Chen et al., 2023a) , like other SOTA models, is almost certainly trained to minimize RMSE. It is a canonical finding that GANs produce sharp images, while MSE/L2 loss produces blurry ones [a,b]. The paper is merely demonstrating this known fact, not the utility of its proposed framework.  

4. The paper's claims of superior performance in extreme event forecasting are unsubstantiated due to critically missing baselines. The text explicitly mentions comparing to "Pangu-Weather" (Bi et al., 2023). However, Pangu-Weather and IFS are conspicuously absent from the extreme-weather SEDI plot (Figure 7a). They are only included in the RMSE plot (Figure 7b) , where Pangu-Weather is visibly better than WTL for key variables like t2m and u10. Omitting the key SOTA baseline from the primary claim (extreme weather) is a major flaw.

###### References

[a] Chen, Yuan, et al. "Adversarial-learning-based image-to-image transformation: A survey." Neurocomputing 411 (2020): 468-486.  
[b] Zhang, Kuan, et al. "SOUP-GAN: Super-resolution MRI using generative adversarial networks." Tomography 8.2 (2022): 905-919.

### Questions
1. Could the authors please resolve the fundamental contradiction regarding the PVUM? How can the module claim to transform "any pressure-variable subset"  into a "unified representation" when the appendix (L830-833) states that "a separate weather latent autoencoder" was trained for each of the six variables?   

2. The "sharp results"  are attributed to the latent framework. However, the WLA uses a GAN loss  while the baseline (WT)  likely uses an MSE loss. How can the authors be sure this sharpness is not simply a known artifact of comparing a GAN-trained pipeline to an MSE-trained one? To convert this weakness into a strength, the authors should provide a new baseline, training the pixel-space WT model with a comparable adversarial or perceptual loss (e.g., VQGAN (Esser et al., 2021))  to ensure a fair comparison.   

3. Why are the SOTA baselines Pangu-Weather (Bi et al., 2023)  and IFS missing from the primary extreme-weather SEDI plot (Figure 7a) , despite being mentioned in the text and included in Figure 7b? Please provide the SEDI results for these models, as their omission invalidates the claim of "superior accuracy and sharpness".

### Soundness
2

### Presentation
3

### Contribution
2
