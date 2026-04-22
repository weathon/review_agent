# M²F-PINN: A Multi-Scale Frequency-Domain Multi-Physics-Informed Neural Network for Ocean Forecasting

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 2, 4, 2

## Abstract
Physics‐informed neural networks (PINNs) embed physical laws into data-driven learning and are becoming increasingly influential in climate and ocean forecasting. Yet effectively capturing multi-scale variability across high and low frequencies while maintaining training stablility and ensuring convergence remains challenging for conventional PINNs. We introduce M$^2$F-PINN, a novel Transformer-based multi-scale frequency-domain multi-PINN algorithm designed to 1) mitigate spectral bias via Fourier representation learning, and 2) analyze multi-scale characteristics through frequency-domain modeling, and 3) incorporate physics priors using multiple PINNs. M$^2$F-PINN leverages multi-scale Fourier networks to learn spectral components and multi-scale interactions, and employs a 3D Swin Transformer in an autoregressive setting to capture spatiotemporal regularities. The advantages of M$^2$F-PINN include: 1) adaptively learns multi-scale frequency components to enhance the modeling of multi-scale dynamics; 2) jointly estimates physical coefficients within the PINN modules, refining representations of physical processes; 3) preserves the Transformer framework, enabling compatibility with diverse architectures and structural decoupling; 4) extensive experiments on real-world ocean datasets show that M$^2$F-PINN outperforms deep-learning baselines and competitive ocean models (e.g., XiHe, WenHai) in predicting ocean current fields, achieving superior performance across multiple time horizons.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents M²F-PINN, a Multi-Scale Frequency-domain Physics-Informed Neural Network designed for ocean current forecasting. The model integrates multi-frequency feature embeddings, momentum-equation-based physical constraints, and a 3D Swin Transformer backbone to capture multi-scale dynamics. Specifically, it introduces Fourier mapping to address spectral bias in the model, incorporates PDE residuals from zonal and meridional momentum equations as physics loss terms, and learns spatiotemporal evolution via a Transformer-based autoregressive framework. Experiments conducted on the GLORYS12 reanalysis dataset (2005–2008) show that the proposed model outperforms several baselines in both prediction accuracy and physical consistency.

### Strengths
(1) Multi-scale representation via frequency embeddings. The use of Gaussian Fourier features helps alleviate spectral bias and improves learning of high-frequency dynamics, a common limitation of PINNs and spatiotemporal neural operators.
(2) Across all evaluation metrics, M²F-PINN achieves higher accuracy and better stability compared to state-of-the-art models such as XiHe, and WenHai. The results suggest that the multi-scale frequency-domain learning strategy, combined with physics-informed constraints, effectively enhances both prediction accuracy.

### Weaknesses
(1) Experimental setup simplicity and comparison fairness. The experimental design focuses on a relatively simplified ocean forecasting setup. In contrast, baseline models such as XiHe and WenHai were originally developed for more complex, multi-variable, and higher-resolution scenarios. Using them under a simplified setting may reduce the fairness and persuasive power of the comparison, as these models are not optimized for such reduced configurations. However, the simplified experimental configuration in this paper makes the comparison less convincing.
(2) Limited novelty in methodological design. The integration of frequency-domain representations (Fourier features), NTK-inspired spectral components, and PDE-based physics losses is reasonable and well-motivated but not conceptually new. Similar combinations have appeared in recent physics-informed operator learning and PINNs literature. As a result, the innovation of M²F-PINN lies more in assembling existing techniques than in introducing a fundamentally new modeling principle or architecture.
(3) Minor typo. In line 48, the referenced model name “AI-GMOS” is a typo — it should be “AI-GOMS.”

### Questions
Since the GLORYS12 dataset is a reanalysis data rather than a pure numerical model output, it may not strictly obey the PDEs used for the physics-informed loss. Could the authors clarify whether the GLORYS fields are consistent with the momentum equations? If inconsistencies exist, how are they handled during optimization, and how does this affect model stability and interpretability?

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
This paper introduces M2F-PINN, a multi-scale, frequency-domain physics-informed neural network for ocean forecasting. It combines Fourier representation learning to address spectral bias and uses 3D Swin Transformer to capture spatiotemporal patterns. M2F-PINN integrates physics-based priors through loss function about physical constraints. Experiments show it outperforms deep-learning baselines and ocean models.

### Strengths
1. The paper provides substantial theoretical support, explaining the impact of spectral bias on network training.
2. The ablation study examines the contribution of each major component (training B, training scale, training frequencies), which improves the transparency of the design.

### Weaknesses
1. The model introduces momentum equations to constrain the training of U and V, however, momentum equations are based on specific physical assumptions and simplifications (e.g., constant density). These equations primarily address large-scale oceanic flows, and cannot fully capture small-scale dynamics such as turbulence and eddies. Furthermore, the equation can only describe velocity variations accurately at higher resolutions. Utilizing this equation as a physical constraint may affect the precise modeling and prediction of small-scale flows.
2. The method utilizes Gaussian Fourier coordinate encoding to enhance the exposure of high-frequency information. However, this paper does not include experimental design to demonstrate how the method enables the model to capture ocean current features at different scales, nor does it modify the method based on the characteristics of ocean flow fields. It appears more aligned with application-oriented research in the geoscience rather than a fundamental advance in physical modeling.
3. Although the work includes several comparative experiments, there are still some issues to be addressed. a) There is a lack of comparison with recent baseline works, such as PINN and neural operator approaches from the past two years. b) The paper does not demonstrate how M2F-PINN compares with other methods in terms of performance at different prediction time horizons and depths. c) The ablation study does not clearly specify which modules the PINN-base and data-base ablations refer to.
4. The overall presentation of the paper is unclear, and the figures and tables are relatively rough. Some examples include: a) The phrase "learns frequency components multi-scales to improve multi-scale dynamics" in the abstract is grammatically incorrect. b) There is an issue with the citation format, leading to repeated author names. c) The borders of Tables 1 and 2 are unclear, and Table 3 exceeds the article width. d) The introduction of the NTK theory is abrupt, and its connection to the work is not explained. e) Only the momentum theorem is used, but why is the method called Multi-PINN? f) Many variables in the formulas are not explained, such as the variable t in line 218. g) The dataset description is unclear, such as the specific resolution (how many kilometers).

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
A novel model for global ocean current forecasting that aims to address the limitations of conventional Physics-Informed Neural Networks (PINNs) in capturing multi-scale variability and mitigating spectral bias

### Strengths
1.Achieves significantly lower Physical Inconsistency Coefficient (PIC) values, confirming that the predictions adhere more closely to the physical laws compared to all data-driven methods.

2.Demonstrates state-of-the-art performance, achieving the lowest RMSE and highest ACC across short- and long-range forecasts (up to 60 days), outperforming both deep learning baselines and competitive ocean models (e.g., XiHe and WenHai)

### Weaknesses
1.The framework is a novel combination, but the individual components—the 3D Swin Transformer and the Fourier Feature Embedding—are adapted from existing, established methods in computer vision and PINN literature, respectively.

2.The PDEs used for the PINN loss omit crucial terms (e.g., Coriolis force, pressure gradients) fundamental to ocean dynamics, potentially limiting the physical fidelity compared to full numerical models.

3.The paper lacks a comparison of training and inference speed against state-of-the-art models, which is essential for assessing the practical viability of the complex 3D Swin Transformer architecture.

### Questions
1.The PINN approach enforces constraints from the governing physical equations ($L_{PDE}$). However, real-world data (GLORYS12 reanalysis) inherently contains noise and discrepancies, and, the simplified PDEs used in $M^{2}F$-PINN are themselves an approximation of the full, complex ocean dynamics (the "reality gap"). Does enforcing an inexact or simplified physical constraint ultimately suppress the model's ability to learn real-world phenomena present in the data but not captured by the $\mathcal{L}_{PDE}$?

2. How to Ensure Inaccurate PINN Constraints Provide Model Gain, and Under What Conditions?

3.Given that the ablation study shows a marginal performance difference between $M^{2}F$-PINN and variants with non-trainable Fourier parameters, could the authors provide a stronger justification (e.g., analysis of learned B-matrix structures, convergence rates, or visualization of learned spectral coverage) for making these parameters adaptable and adding implementation complexity?

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
3

### Summary
This paper presents M2F-PINN, a Multi-scale Frequency-domain Multi-Physics-Informed Neural Network for large-scale ocean current forecasting. The authors aim to address two known issues in ocean forecasting models: (1) deep models’ difficulty in learning both low- and high-frequency ocean dynamics due to spectral bias, and (2) the lack of physical consistency in purely data-driven ocean models. M2F-PINN propose several crucial components: multi-scale Fourier feature embeddings, a 3D Swin Transformer backbone for spatiotemporal feature extraction, and multiple Physics-Informed Neural Network (PINN) modules. The model is trained on the GLORYS12 reanalysis dataset (global ocean, 2005–2008) and evaluated against CNNs, RNNs, MeshGraphNets, Fourier Neural Operators, and state-of-the-art ocean models. M2F-PINN achieves the best results across 1–60-day prediction horizons, outperforming strong baselines and showing improved long-term physical consistency.

### Strengths
- The paper systematically combines frequency-domain embeddings, physics-informed losses, and Transformer architectures in a coherent way. Although each component exists independently, their integration into an end-to-end framework is well-executed.

- The NTK-based spectral analysis provides a clear explanation of how Fourier embeddings reshape learning rates across frequency bands, offering some theoretical insight beyond empirical results.

### Weaknesses
- The proposed components—multi-scale Fourier features, PINN-based physical losses, and uncertainty-weighted multi-task optimization—are all mature and widely used in physics-informed and climate modeling literature (e.g., FNO, ClimODE, LangYa, frequency-domain PINNs). The contribution is largely an engineering combination rather than a fundamentally new algorithmic advance. The claimed innovation in “frequency-domain multi-PINN” lacks distinctive technical depth beyond prior works. 

- The physics component only implements two simplified momentum equations for (U, V), omitting key coupled variables such as temperature, salinity, and density that govern realistic ocean circulation. As a result, the “multi-physics” claim is overstated—the scope is narrow and its real-world interpretability limited.

- CNN and RNN serve as trivial references and do not represent current forecasting standards. Comparisons to XiHe and WenHai are insufficiently documented—there is no clarification whether they were re-trained under the same data regime or results are copied from publications. Missing comparisons with more recent AI-based earth system models (e.g., GraphCast, Fuxi, Pangu) weaken the empirical credibility.

- Despite the title emphasizing forecasting, the paper fails to specify how prediction is carried out.The architecture uses a Swin Transformer encoder-decoder, but it is unclear whether forecasts are autoregressive (iteratively rolling forward) or direct multi-step predictions.The “Algorithm 1” description is ambiguous: while labeled “autoregressive,” it lacks any explicit temporal unrolling or teacher-forcing details. Without a precise description of forecast horizon handling, it is hard to interpret the reported 1-, 7-, and 30-day results or to assess generalization stability.

- The paper asserts that Fourier embeddings capture “multi-scale oceanic structures,” yet provides no frequency-space visualizations or power-spectrum analysis to substantiate this claim. Likewise, the physical residuals (PIC) are introduced, but no concrete examples of physically consistent predictions (e.g., energy or vorticity preservation) are shown.

### Questions
- How exactly are forecasts generated? Is the model trained in an autoregressive fashion (iterative multi-step rollout) or direct multi-horizon regression? If autoregressive, how is error accumulation handled?

- Given that only U and V momentum equations are included, how can the model capture coupled dynamics driven by temperature and salinity gradients? Would incorporating the full Navier–Stokes or continuity equations improve realism?

### Soundness
2

### Presentation
2

### Contribution
2
