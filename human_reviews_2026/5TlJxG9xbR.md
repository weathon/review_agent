# SoC-DT: Standard-of-Care Aligned Digital Twins for Patient-Specific Tumor Dynamics

- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
Accurate prediction of tumor trajectories under standard-of-care (SoC) therapies remains a major unmet need in oncology. This capability is essential for optimizing treatment planning and anticipating disease progression. Conventional reaction–diffusion models are limited in scope, as they fail to capture tumor dynamics under heterogeneous therapeutic paradigms. There is hence a critical need for computational frameworks that can realistically simulate SoC interventions while accounting for inter-patient variability in genomics, demographics, and treatment regimens. We introduce Standard-of-Care Digital Twin (SoC-DT), a differentiable framework that unifies reaction–diffusion tumor growth models, discrete SoC interventions (surgery, chemotherapy, radiotherapy) along with genomic and demographic personalization to predict post-treatment tumor structure on imaging. An implicit-explicit exponential time-differencing solver, IMEX-SoC, is also proposed, which ensures stability, positivity, and scalability in SoC treatment situations. Evaluated on both synthetic data and real world glioma data, SoC-DT consistently outperforms classical PDE baselines and purely data-driven neural models in predicting tumor dynamics. By bridging mechanistic interpretability with modern differentiable solvers, SoC-DT establishes a principled foundation for patient-specific digital twins in oncology, enabling biologically consistent tumor dynamics estimation. Code will be made available upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces SoC-DT, a differentiable digital twin framework for modeling patient-specific tumor dynamics under standard-of-care (SoC) treatments, including surgery, chemotherapy, and radiotherapy. The approach combines reaction-diffusion partial differential equations (PDEs) with clinical information. It introduces a novel IMEX-SoC solver to ensure numerical stability and efficient gradient propagation through treatment-induced discontinuities. The model is evaluated on both synthetic datasets (for brain, liver, and breast cancers) and a real-world glioma dataset (UCSF-ALPTGD), showing improved performance in tumor mask prediction and time-to-progression forecasting compared to classical PDE baselines and data-driven models. However, it also suffers from several critical flaws that significantly degrade the paper's overall quality.

### Strengths
- SoC-DT is a novel framework to unify mechanistic tumor growth modeling with standard-of-care treatment protocols in a differentiable, personalized digital twin setting. 
- Synthetic Data Framework: The "Plug-and-Play" approach for generating synthetic treatment scenarios is a nice practical contribution, addressing the critical bottleneck of scarce longitudinal clinical data for training and benchmarking.

### Weaknesses
- Lack of details in many parts of the paper. 1. Lack of detail in synthetic data generation: It provides insufficient details on how phantom images, genomic markers, and treatment responses are simulated. 2. Insufficient explanation of PDE solver and training: The IMEX-SoC solver is introduced but not fully explained. This makes it difficult to assess the solver's efficiency and stability in practice. 3. Vague training loss: The training procedure mentions a "mixed Dice+BCE loss", but does not specify the loss weights or the rationale for the hybrid loss.
This lack of detail undermines the reproducibility. 4. Limited real-world dataset description: The real-world UCSF glioma dataset is not adequately described. Key information—such as the number of patients, imaging modalities, treatment heterogeneity, and follow-up schedule—is missing. This makes it difficult to judge the generalizability and clinical applicability of the reported results. 5. Unclear link between therapy and mask prediction: While the model incorporates therapy effects via PDE terms, the paper does not clearly explain how these dynamics translate into improved tumor mask predictions. For instance, it is not demonstrated how the inclusion of radiotherapy or surgery terms leads to more accurate boundary delineation or residual tumor detection in the predicted masks.
- While promising, clinical validation is primarily limited to a single real-world cohort (UCSF glioma data). Broader multi-institutional trials across various cancer types are needed to establish generalizability firmly.
- The strong performance on synthetic data is less compelling because the data is generated using rules inherently similar to the model's own logic. This does not rigorously test the model's ability to capture the true, far more complex real-world cancers. Minor performance improvements in the only real-world dataset also indicate that.
- The ablation study, revealing that removing "kill" or "surgery" terms caused "no measurable degradation," is alarming. It suggests that for the specific dataset used, these complex treatment modules may be redundant or poorly identified. This fundamentally questions the necessity of these sophisticated components and suggests that a simpler growth-diffusion model could achieve similar performance on this data, undermining a core claim of the paper.

### Questions
see weakness

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors proposed SoC-DT, a standard-of-care-aligned digital twin, to capture and model tumor dynamics (based on the applied treatment, such as surgery, chemotherapy, radiotherapy) in a differentiable manner. Specifically, the team applied a well-defined reaction-diffusion PDE and utilized patient covariates. Numerically, SoC-DT employs an IMEX–ETD split (implicit diffusion, analytic logistic/chemo step, multiplicative event jumps). Experiments span three synthetic cohorts (AG/HCC/NAC) produced by a newly proposed “plug-and-play” generator and a real longitudinal glioma dataset (UCSF-ALPTGD). Reported gains over classical PDEs, PINNs, and a “Hybrid PDE” baseline are modest but consistent for mask prediction and TTP.

### Strengths
1. The paper provides a clear hybrid modeling. The team suggests a local PDE structure with differentiable event operators. They also proposed a numerical solver for stiff dynamics. 
2. The PDE part of the model improves the interpretability. Model interpretability is essential for clinical models. 
3. Synthetic framework (App. A,B). The generator encodes plausible artifacts (bias fields, k-space filtering, Rician/ghosting/Gibbs; modality-specific kinetics for NAC/HCC) and alternative kill/surgery operators (additive/saturation/synergy; multiplicative/morphological/rim).

### Weaknesses
1. The core dynamics are voxelwise diffusion/logistic plus multiplicative events. This captures local coupling, but patient-level global context (organ-scale constraints, multi-lesion coupling, global tumor burden, perfusion, dose-field heterogeneity, systemic therapy history) is under-specified.
2. The model risks being a locally consistent PDE with globally shallow personalization. This also helps explain ablations where No-kill/No-surgery barely change performance: if global SoC signals don’t effectively condition the spatial dynamics, killing/resection may have limited observable impact in the labels used.
3. Real-data DSC improvement is small (+1.16 vs Hybrid PDE) with larger variance in SoC-DT (8.12). Provide paired tests (per-patient Wilcoxon), effect sizes, and per-subset analyses (e.g., large resections, definite RT) to show where SoC events help.

### Questions
1. Where does global information enter a voxel-local PDE? In tumor predictions, the global information of the patient and the tumor itself is very helpful for prediction. Here, it seems that you are only focused on the pixel-level prediction. Is there any way to make use of the neighboring information or make use of the global information?
2. Do you implement explicit adjoint jump conditions at surgery/RT times? Please provide the formulas or FD gradient checks.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the Standard-of-Care Digital Twin, a framework designed to predict patient-specific tumor dynamics under complex treatment regimens. The method is based on a reaction-diffusion partial differential equation that models tumor growth and invasion. The key contribution is the extension of this established approach to incorporate the discrete, discontinuous events characteristic of standard-of-care cancer treatment, namely surgery, chemotherapy and radiotherapy. The framework is designed to be differentiable, allowing its parameters to be personalized using patient-specific data.

### Strengths
With clinical oncology, the authors tackle a very relevant and ambitious problem. Developing accurate, patient-specific models that can forecast tumor response to standard treatments would be a significant step towards improved treatment planning and personalized medicine. The Reviewer likes the idea of building the digital twin on top of a biophysical PDE.  This provides a structured, interpretable prior that constrains the model to biologically plausible behavior, which is an advantage over purely black-box deep learning models. However, there is the caveat the tumor dynamics may not fully be described with reaction-diffusion PDEs only. The authors should comment on this limitation.

### Weaknesses
In the reviewers' opinion, the novelty of the framework is the integration of known tools rather than methodological advances. While the engineering effort to combine these into a single, end-to-end framework is high, the individual components are not new. There is no fundamental advance in machine learning methodology within the paper in the Reviewers opinion. According to the Reviewer, the paper is thus misclassified and should have been submitted in the applications category.

Unfortunately, the results are not convincing for the Reviewer. The margin of improvement on the real clinical dataset is modest, especially given the significant increase in model complexity which raises concerns about the practical utility and the cost-benefit trade-off of the proposed framework. Moreover, the standard deviations of the reported errors are very high and significantly larger than the differences between models indicating that there may be no clear best performing model.

According to the Reviewer, the ablation study moreover raises concerns as one of the main claim of the work is that explicitly modeling SoC interventions (surgery, chemotherapy, radiotherapy) is crucial for improving predictive accuracy. However, the ablation results state that omitting either the 'kill' (chemotherapy/radiotherapy) or 'surgery' terms from the model yields no measurable degradation in performance. If the SoC components do not contribute to the model's performance on the test dataset, then the observed performance gains over baselines cannot be attributed to their inclusion and the postulated usefulness of including SoC is not supported by evidence. 

For a complex model such as the one presented in the paper,moreover,  more implementation details or code is needed to being able to potentially reproduce the work. For instance the genomic MLP or other core components are not explained within the paper and even the supplementary material does not contain any details regarding the architecture.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Standard-of-Care Digital Twin (SoC-DT), a differentiable framework for predicting tumor evolution under real-world therapies. By unifying reaction–diffusion tumor growth models with discrete SoC interventions, SoC-DT integrates genomic, demographic, and treatment-specific factors for personalized simulation. Experiments on synthetic and real datasets show that SoC-DT outperforms both classical PDE and neural baselines, offering a biologically interpretable foundation for patient-specific digital twins in oncology.

### Strengths
1. Accurately forecast tumor evolution under real-world standard-of-care (SoC) therapies, directly supporting personalized treatment planning.
2. Bridges traditional biophysical tumor modeling and differentiable programming, enabling gradient-based learning while retaining biological consistency, which is an important step toward trustworthy digital twins in oncology.
3. The mathematical theory well supports the proposed method.

### Weaknesses
1. The author should clarify the difference against the recent medical world model[1] from the perspective of concept and post-treatment generation module.
[1] Yang Y, Wang Z Y, Liu Q, et al. Medical world model. ICCV, 2025.

2. Standard-of-care interventions are represented in a simplified, discrete manner. Real-world variability in dosing schedules, drug combinations, and adaptive treatment strategies seems not fully modeled.
3. The differentiable PDE solver and event-aware adjoint computation are resource-intensive. This may hinder scalability and limit use in time-sensitive or large-scale clinical applications. The efficiency should be evaluated quantitatively.
4. The study provides retrospective and synthetic experiments but no prospective clinical validation. Without real-world deployment, translational reliability and safety remain unaddressed.
5. While this work tried to simulate tumor trajectories, there is no validation of clinically actionable insights during decision-making.
6. Is the proposed method able to be extended to multiple intervention points?

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
