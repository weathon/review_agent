# Knowledge-guided assimilation to bridge the gap between sensing and modeling with indirect labels for global-scale carbon monitoring

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6

## Abstract
Advanced air-borne and in-situ sensing platforms have generated invaluable observations of the Earth systems and offer exciting opportunities in enhancing the monitoring and forecasting capabilities to tackle challenges such as global warming. While process-based models have been developed for decades, they have limited ability to incorporate real-world observations to further enhance the prediction ability, especially to correct simplified sub-processes that tend to cause deviations from the observations. In particular, many process-based models rely on itemized lower-level processes, whereas the sensors very often can only collect aggregated mixed-up information, constraining the use of these observations to improve the modeling. Existing works on knowledge-guided learning mainly focus on connecting process-based and data-driven methods via directly matched variables, using physical rules and simulations to constrain the training process. We propose a knowledge-guided assimilation approach to integrate process-based and learning models to improve the utilization of large-scale simulations with aggregated indirect observations. To evaluate approach, we carry out a global-scale case study with ecosystem models that are widely used in carbon monitoring. The results on global-scale benchmark data show that knowledge-guided integration of indirect labels can significantly enhance prediction skills compared to existing learning methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a Knowledge-Guided Assimilation (KGA) framework designed to bridge the long-standing gap between observation-driven sensing and process-based modeling in global carbon monitoring. The core idea is to integrate indirect remote-sensing labels into model learning through a Decomposition-and-Resembling (DERE) mechanism that separates model-internal processes and reconstructs them using aggregated observational signals (e.g., PFTs). A probabilistic label expansion step based on diffusion modeling is further proposed to augment sparse temporal observations.

### Strengths
1. The paper tackles a scientifically meaningful and socially important problem—bridging physics-based and data-driven modeling for global carbon monitoring.

2. Comprehensive experiments with five datasets and multiple variables provide convincing empirical coverage.

3. The writing and organization are solid; the paper is easy to follow despite domain complexity.

4. The proposed framework could be valuable to researchers in climate and remote-sensing communities.

### Weaknesses
1. Lack of algorithmic novelty: the key modules (DERE and diffusion label expansion) reuse existing designs with minimal methodological contribution.

2. Scope mismatch with ICLR: the paper focuses on an Earth-system application rather than representation learning or optimization advances.

3. Missing theoretical grounding: “knowledge guidance” and “physical consistency” are discussed conceptually but never formalized in loss design.

4. Limited interpretability: no visualization or analysis of intermediate decomposed processes to justify the “knowledge-guided” claim.

5. Evaluation bias: improvements are shown mainly through RMSE; no analysis of uncertainty propagation or generalization to unseen domains.

6. Over-extended framing: the work reads more like a scientific modeling study than a core ML contribution.

### Questions
1. How does the proposed DERE differ from previous process-decomposition approaches used in KGML or hybrid physical–ML frameworks?

2. Could the same gains be achieved simply by applying standard multi-task learning with auxiliary indirect labels?

3. Have you tested the method on domains beyond carbon fluxes (e.g., hydrology, aerosol modeling) to demonstrate generality?

4. How is the uncertainty from probabilistic label expansion propagated to downstream predictions?

5. Can you provide formal definitions of “knowledge consistency” used in the loss formulation?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
In this paper, the authors propose DERE,  a knowledge-guided assimilation method for incorporating in-situ data into process-based models. The method relies on 2 components: 1) a decomposition step, which aims at representing the sub-processes of the process-based model, 2) a resembling step in which real observations are integrated to constrain those sub-stream processes. Both in-situ information and remote sensing information are used. 
The authors test their method on a task related to global carbon monitoring, incorporating in-situ flux tower measurements from the dataset CarbonSense. They compare it with the ED model, trying different architectures, a variety of transformer-based  time-series models, and KGML extensions of these time-series models.

### Strengths
This work seems to tackle an important challenge, which is the current limitation of process-based models which do not accomodate well the integration of in-situ and remote sensing data. The author outline well the problem and present the context of the work well with a thorough literature review. 
The proposed method is compared against a variety of baselines, and results seem promising.

### Weaknesses
The main weakness is the clarity of the paper:
- The authors focus on presenting the method and try to link it with the application example a little bit as they go through the method,  but it is unclear until quite late in reading the paper what exactly the "carbon monitoring task" is - i.e. inputs, outputs. 
- some training details could be outlined more clearly: which loss, how the in-situ data is incorporated in the method, what exactly the evaluation process is. 
- one part that could benefit clarity is how the indirect data is used in DERE. How is this aggregated data made compatible with the sub-stream processes? I believe this is explaine L269. but it is really unclear to the reader. why comparing this with the integrated function from the subprocesses makes it better than using directly the final output. 
- How is it determined how many subprocesses should be decomposed? What is the ground truth for that since the intermediate results are not saved in the ED model?  Is there a ground truth for these substream processes when training the model? Or is that just implicitly considered and only the final output is considered in the loss?  In other words, how is the model trained, how are the substream processes validated? 
- can you clarify the training process? especially " during training each batch contains samples intended both for the indirect label nd direct label comparison" Does this mean that in the training data there are disparate sources and different objectives for each sample in a batch? in that case how do you trust the alignment between indirect, direct and probabilitistic labels? 

It would be very helpful if the authors could clarify these points.

### Questions
- Could the authors clarify the methodology and application with the CarbonFlux data (please refer to the points mentioned in weaknesses). 
- Have you ablated the 2 strategies used in KGML and in DERE (the physics-constrained loss and the initialization on ED+finetuning with in-situ observations)? in particular, the baseline does not seem to include the physics-constrained loss and it would be interesting to have an ablation of this component in the KGML model. 
- The authors emphasize on GPP as an example, whereas in the CarbonFlux paper which is used in this study, the NEE is the main focus. Both are related, but the GPP is obtained from NEE. Is there a reason for this choice? 
- Fig 4: it seems that DERE is most helpful for low values of GPP. Do you have an explanation why? Do you observe a similar trend on the other indicators? 
- Why not compare directly with the models developed in CarbonFlux?
- Why is Transformer the backbone used for DERE if that is not the best performing baseline or KGML model backbone? 

Minor comments: 
- I think it could be worth, in order to make this paper more impactful, to align the vocabulary used with what are terms used commonly in the machine learning community (or "translate" more clearly). For example, "indirect labels" is not a term commonly used in machine learning to talk about remote sensing data. This would also help with the clarity of the paper. 
- The figures are very small. The text has a lot of redundancies (e.g. explaining that the intermediate results from substreams are generally not saved), compressing a little could help have larger figures. 

If the author clarify the methodology and experiments, and it is sound, I will be more than happy to increase my score.

### Soundness
2

### Presentation
1

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
The paper proposes a knowledge-guided assimilation framework to fuse process-based ecosystem models (ED) with data-driven forecasting using indirect, large-scale satellite labels. The method introduces a Decomposition-and-Resembling (DERE) pipeline: 1) Knowledge-aligned decomposition of end-to-end simulation data to represent intermediate, sub-stream processes of a process-based model, as a preparation for integration of indirect higher-level labels. 2) Knowledge-aligned resembling of decomposed intermediate, sub-stream processes to enable supervision from indirect higher-level labels available at large-scale to constrain the sub-stream process. Experiments on CarbonSense benchmark datasets (ABoVE, AmeriFlux, FLUXNET, ICOS-WW, Multiple) show DERE outperforms ED, vanilla time-series baselines, and “KGML” baselines across GPP/RECO/NEE with MAE/RMSE, using an 80/20 spatial split.

### Strengths
1. Clear problem framing (intermediate/sub-stream processes; indirect labels) and a practical pipeline that acknowledges real-world constraints (no stored intermediates; high storage/compute costs).
2. The paper is well-written, making it understandable and easy to follow. It has a well-organized structure, and the proposed method is presented clearly.
3. Thoughtful uncertainty-aware use of imputed labels and visualization of coverage/variance.

### Weaknesses
1. The conditional-imputation module uses observed segments from each site time series to fill missing segments and then reuses these imputed values (with uncertainty weights) during fine-tuning. The manuscript does not clearly demonstrate strict temporal isolation from test periods, spell out tuning safeguards that prevent indirect exposure to masked values, or provide quantitative uncertainty calibration (e.g., coverage, reliability). Without these, reported gains could be partially driven by leakage or overconfident self-labeling rather than genuine generalization. Examples and figures are illustrative, but the lack of calibration metrics makes it hard to verify that uncertainty is being used correctly.
2. Satellite plant functional type products, tower footprints, and process-model grids operate at different spatial/temporal scales. The paper lacks details on how resampling, footprint integration, temporal alignment, and quality control are handled, and how sensitive the results are to these choices. Some improvements may stem from favorable harmonization rather than robust assimilation.
3. The paper removes one network due to short coverage, yet sparse and short series are the very setting the label-expansion module aims to address. A robustness study that retains these sparse sites and demonstrates reliable behavior under realistic data gaps would strengthen the central claims.
4. The paper does not report training/testing costs, memory, or throughput compared with baselines.

### Questions
1. Please provide details and sensitivity on spatial footprint modeling, temporal resampling, and QC of satellite PFTs; report how choices impact results.
2. Could you add KGML + diffusion (no DERE) and DERE-minus-diffusion ablations to quantify where the gains originate.
3. Please describe temporal conditioning windows, any proximity rules to test periods, and provide probabilistic calibration metrics (coverage, CRPS) for the diffusion expansion. Could you show performance if imputed labels are used only when uncertainty passes a strict threshold?
4. Could you report training/testing costs, memory, or throughput compared with baselines?

### Soundness
3

### Presentation
3

### Contribution
3
