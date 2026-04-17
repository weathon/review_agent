# FlowCast: Advancing Precipitation Nowcasting with Conditional Flow Matching

- Decision: Accept (Poster)
- Scores: 2, 4, 4, 4

## Abstract
Radar-based precipitation nowcasting, the task of forecasting short-term precipitation fields from previous radar images, is a critical problem for flood risk management and decision-making. While deep learning has substantially advanced this field, two challenges remain fundamental: the uncertainty of atmospheric dynamics and the efficient modeling of high-dimensional data. Diffusion models have shown strong promise by producing sharp, reliable forecasts, but their iterative sampling process is computationally prohibitive for time-critical applications. We introduce FlowCast, the first end-to-end probabilistic model leveraging Conditional Flow Matching (CFM) as a direct noise-to-data generative framework for precipitation nowcasting. Unlike hybrid approaches, FlowCast learns a direct noise-to-data mapping in a compressed latent space, enabling rapid, high-fidelity sample generation. Our experiments demonstrate that FlowCast establishes a new state-of-the-art in probabilistic performance while also exceeding deterministic baselines in predictive accuracy. A direct comparison further reveals the CFM objective is both more accurate and significantly more efficient than a diffusion objective on the same architecture, maintaining high performance with significantly fewer sampling steps. This work positions CFM as a powerful and practical alternative for  high-dimensional spatiotemporal forecasting.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces FlowCast, which applies Conditional Flow Matching (CFM) to radar-based precipitation nowcasting. The authors claim this is the first attempt to use CFM for spatiotemporal probabilistic forecasting. The model operates in a latent space via a VAE encoder-decoder and adopts an Earthformer-UNet backbone to learn a noise-to-data vector field. Experiments on the SEVIR and ARSO datasets show that FlowCast achieves state-of-the-art results in CRPS, CSI, and HSS, while requiring far fewer function evaluations than diffusion-based models. The paper highlights the superior efficiency–accuracy trade-off of CFM over diffusion objectives and positions FlowCast as a practical framework for real-time nowcasting.

### Strengths
S1. This paper demonstrates significantly reduced inference steps compared with diffusion methods, making it relevant for operational nowcasting systems that demand fast turnaround.

S2. FlowCast achieves superior or competitive scores on two diverse radar datasets, confirming generalization beyond a single domain.

S3. The paper provides explicit architectural and training settings, enhancing reproducibility.

S4. It addresses the real bottleneck in generative nowcasting (diffusion sampling inefficiency) with a concrete solution.

### Weaknesses
W1. The claim in the abstract and introduction that this work is the first to apply Conditional Flow Matching to precipitation nowcasting is inaccurate, since prior studies such as the ICML 2025 rectified flow model have already demonstrated flow-based precipitation refinement.

W2. The literature review in Section 2 on probabilistic nowcasting omits flow-based approaches when contrasting generative paradigms, resulting in an incomplete overview of probabilistic modeling beyond GANs and diffusion.

W3. The introduction of flow models in Section 2 does not provide a comparative analysis against other flow-based frameworks such as Continuous Normalizing Flows or rectified flows, leaving the advantages of CFM for spatiotemporal uncertainty unexplained.

W4. The architectural description in Section 3.2.2 shows that FlowCast merely adapts Earthformer-UNet to the CFM objective without introducing new architectural mechanisms or theoretical insights that would justify CFM’s suitability for atmospheric dynamics.

W5. The experimental comparison in Section 4.2 excludes recent transformer-based baselines such as Earthfarseer, which limits the credibility of the claimed state-of-the-art performance.

W6. The evaluation in Section 4.2 relies heavily on aggregated scores such as CSI-M and HSS-M, which mask variations across precipitation intensities and obscure the model’s behavior on extreme rainfall events.

W7. The methodological and inference descriptions in Sections 3.2.2 and 4.1.4 omit discussion of training instability and ODE-based sampling overhead, leaving open questions regarding robustness and computational scalability.

### Questions
1. The abstract mentions “predictive accuracy,” while the contributions section claims “probabilistic performance,” so which one defines the main contribution?
2. The description in Section 4.1.1 on Page 5 states that the ARSO dataset uses “63,716 training samples,” while Table 1 lists only 38,229 training samples. How can this internal inconsistency in dataset splits be explained?
3. Table 2 on Page 6 defines the ARSO latent dimensions as 13/12 × 38 × 52 × 4, which implies non-integer compression factors compared to the original 301 × 401 resolution. How is the rounding or padding handled to yield these dimensions?
4. The results table for ARSO on Page 8 labels the highest-threshold metric as “CSI-219,” even though Section 4.1.2 defines ARSO thresholds as [15, 21, 30, 33, 36, 39] dBZ. Thus, why does the same SEVIR-specific label appear in the ARSO table, and what threshold does it actually represent?
5. The conclusion on Page 9 asserts that FlowCast “maintains high forecast quality with as few as a single sampling step,” while all experiments use 10 steps for evaluation. How is this claim supported?
6. Appendix Table 6 uses different VAE warmup epochs for SEVIR and ARSO despite both using the same VAE architecture. What rationale justifies this discrepancy in the training configuration?
7.Appendix A.1.2 mentions an ODE solver ablation but gives no data. How is the claim of no difference supported?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The proposed FlowCast model is the first to apply Conditional Flow Matching to precipitation nowcasting, which utilizes VAE to compress radar data into latent space, combines Cuboid Attention and CFM to learn direct noise-data mapping for efficient sampling. On SEVIR and ARSO datasets, its various metrics outperform baselines like PreDiff and CasCast, with 10-step sampling more efficient than diffusion models.

### Strengths
1. This work is the first to apply Conditional Flow Matching to precipitation nowcasting, learning a direct noise-to-data mapping. It achieves high-fidelity forecasting with only 10 sampling steps.
2. This work adopts VAE latent space compression and Cuboid Attention architecture, balancing the efficiency of high-dimensional radar data processing and the ability of spatiotemporal dynamic modeling.

### Weaknesses
1. Conditional Flow Matching achieves the mapping from noise to data by learning a vector field. However, the paper does not design a prior structure for the vector field in combination with the physical laws of precipitation, relying entirely on data-driven learning. This may lead the model to generate results lacking physical condition constraints.
2. This work applies CFM to precipitation nowcasting for the first time, but the idea of CFM has already been validated in some tasks within the field of computer vision. The paper transfers it to the precipitation nowcasting scenario and fails to propose improvement strategies for CFM targeting certain characteristics of precipitation data (such as strong spatiotemporal correlation, dynamic evolution, etc.). This results in deficiencies in innovation and generative capability.

### Questions
1. This work only validates performance on two radar datasets (SEVIR, ARSO). Are there experimental results tested on more datasets? Meanwhile, is the performance tested under shorter or longer lead times?
2. The probabilistic baselines compared in this work only include diffusion models. Is there a comparison with GAN-based probabilistic models?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper describes the new solution to the problem of radar-based precipitation nowcasting, which uses conditional flow matching (CFM). The authors highlight accuracy and efficiency based on the empirical evidence.

### Strengths
Significance: improving the performance and democratising access to precipitation forecasting is an important problem as addressed in this paper. 

Clarity: the paper is in general clearly written

Quality: the methodology is sound, the description seems correct and reproducible. The ablation studies show the justification for experimental decisions.

### Weaknesses
Quality: The main argument is that swapping the training procedure from the diffusion model to conditional flow matching improves the performance, both in terms of accuracy and in terms of computational efficiency. The problem I see with this argument, however, is that there is previous evidence, in other domains, that the CFM may improve upon the efficiency and performance [1].  I would expect in this case for the authors to link it to the background and then say something that goes beyond the existing literature.  . In other words, the specific link to the justification of this method is missing. 

Originality: as the narrative goes now, and stemming from the previous point, there is a concern that the work is an application of CFM for the precipitation forecasting. That should, in my mind, include a justification that is unique to the precipitation nowcasting (theoretical or otherwise).  Therefore, I would invite the authors to expand upon the contribution and say how this work goes beyond the existing literature in this aspect. 

In summary, my main concern is that the claims now appear to be a combination of the two: CFM and precipitation forecasting.  I would invite the authors therefore to expand upon why this reasoning could be wrong. 


[1] Lipman et al (2023) Flow Matching  for Generative Modeling, ICLR 2023

### Questions
1. I would appreciate if the authors add confidence intervals to the key results (Tables 3-5 in particular). 

2. The algorithms for training and sampling at Page 4 are difficult to follow, therefore I would suggest that the authors present them as algorithm blocks.

3. The authors say in the abstract: "the uncertainty of atmospheric dynamics and the efficient modeling of high-dimensional data" I am curious whether this paper addresses this question in any way beyond the accuracy metrics?

### Soundness
2

### Presentation
2

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
This paper introduces FlowCast, which for the first time applies the efficient Conditional Flow Matching (CFM) framework to radar-based precipitation nowcasting. FlowCast integrates a variational autoencoder (VAE) for latent representation with a network backbone based on the Earthformer-UNet architecture. The authors aim to address the slow sampling issue inherent in existing diffusion models and use ablation studies to demonstrate the efficiency advantage of the CFM objective over the diffusion objective when applied to the same backbone.

### Strengths
1. **Innovation and Practicality of the Methodology:**    
    - The paper innovatively introduces the highly efficient CFM generative framework to the high-dimensional spatiotemporal prediction problem of precipitation nowcasting. This represents a valuable exploration of cutting-edge methods in the field.
        
    - A direct comparison and ablation study with diffusion models clearly demonstrates the significant advantage of CFM in terms of efficiency, showing that it can maintain or closely approach optimal performance with very few function evaluations (NFE). This is of high practical value for time-sensitive nowcasting applications.
        
2. **Clarity and Readability:** The paper is well-structured, the methodology is detailed, and the explanation of the CFM training and sampling process is particularly concise and clear, making the working principle of FlowCast easy to understand.
    
3. **Recognition of Effort and Contribution:** Given that the authors performed comparisons across two generative paradigms (CFM vs. Diffusion) and conducted experiments on two diverse geographical and climatic datasets, they clearly put effort into advancing model performance and exploring efficiency.

### Weaknesses
1. **Incomplete Experimental Results and Lack of Convincing Evidence:**
    
    - **Inadequate Metric Presentation:** The experimental evaluation lacks quantitative verification of structured forecasts. For instance, the absence of analysis using spatial verification methods (e.g., FSS at different scales) prevents a thorough validation of the model's ability to predict the structure and location of precipitation systems. Furthermore, the authors did not provide complete performance curves for **all** baseline models across all future lead times on both datasets.
        
2. **Ambiguity in Innovation Positioning and Insufficient Exploration Breadth:**
    
    - **If positioned as Framework Innovation:** Simply applying the existing CFM training/sampling framework to a UNet backbone based on Earthformer Blocks results in a relatively low degree of methodological novelty for an ICLR-level conference.
        
    - **If positioned as Application Exploration:** The paper aims to explore the general applicability of CFM as an efficient generative paradigm for precipitation nowcasting, but the breadth of exploration is insufficient. The lack of comparative experiments applying CFM to at least one different type of advanced nowcasting model makes it difficult to strongly demonstrate CFM's effectiveness as a general objective function.

### Questions
**Regarding False Alarm Ratio (FAR) and Forecast Quality:** Although FlowCast achieves the highest HSS-M score, your results indicate a **higher FAR-M** (compared to Earthformer and PreDiff). This suggests that FlowCast's high-scoring forecasts may be accompanied by a high rate of 'false alarms'. Could you provide and discuss more detailed categorical skill scores, such as the Threat Score (TS) or Equitable Threat Score (ETS) across different precipitation thresholds? This would help provide a fairer assessment of the model's actual skill in predicting high-intensity precipitation events.

### Soundness
2

### Presentation
3

### Contribution
2
