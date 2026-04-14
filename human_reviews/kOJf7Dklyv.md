# Air Quality Prediction with Physics-Guided Dual Neural ODEs in Open Systems

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6, 6

## Abstract
Air pollution significantly threatens human health and ecosystems, necessitating effective air quality prediction to inform public policy. Traditional approaches are generally categorized into physics-based and data-driven models. Physics-based models usually struggle with high computational demands and closed-system assumptions, while data-driven models may overlook essential physical dynamics, confusing the capturing of spatiotemporal correlations. Although some physics-guided approaches combine the strengths of both models, they often face a mismatch between explicit physical equations and implicit learned representations. To address these challenges, we propose Air-DualODE, a novel physics-guided approach that integrates dual branches of Neural ODEs for air quality prediction. The first branch applies open-system physical equations to capture spatiotemporal dependencies for learning physics dynamics, while the second branch identifies the dependencies not addressed by the first in a fully data-driven way. These dual representations are temporally aligned and fused to enhance prediction accuracy. Our experimental results demonstrate that Air-DualODE achieves state-of-the-art performance in predicting pollutant concentrations across various spatial scales, thereby offering a promising solution for real-world air quality challenges.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Air-DualODE, a hybrid model that integrates physics-based and data-driven techniques. The proposed model addresses the limitations of traditional physics-based and data-driven models, particularly the assumptions about closed systems and the mismatch between explicit physical equations and learned representations. The model involves two main components, where one component applies open-system physical equations to model spatiotemporal dependencies and learn physical dynamics, while the second component identifies the unmodeled dependencies using a fully data-driven approach. The model achieves superior performance  on two real-world datasets at city scale and national scale, compared to the other baselines in this domain.

### Strengths
1. The paper addresses a significant and timely problem.
2. The write-up style and the narrative of the paper is clear, structured, and professional. 
3. The reported results are very strong compared to the existing baselines.

### Weaknesses
1. The paper does not adequately address the scalability of the proposed model. Given that the approach involves multiple components like ODE solvers and GNNs, the computational cost could become prohibitive when applied to larger datasets with thousands of locations. (ex: Dataset used in AirFormer (Liang et al., 2023)).  

2. The overall approach and the methodology of this work closely resembles prior work like AirPhyNet (Hettige et al., 2024), which also integrates physics-based diffusion-advection processes with neural networks for air quality prediction, raising concerns about novelty and significant contributions.

3. Eventhough the work emphasizes about the interpretability of the model, there is limited explanation about how the model’s output could be interpreted in terms of real-world physical phenomena.

4. All the baselines used in the experiments are either fully data driven or hybrid models. Advanced physics based models are not included as baselines and the performance is not compared with the proposed model. Can you evaluate your model’s performance against specific physics-based models, such as Community Multiscale Air Quality (CMAQ) , Weather Reaserch Forecasting (WRF), AERMOD to provide a more comprehensive understanding of how your hybrid model compares in terms of accuracy and computational efficiency?

### Questions
1. The introduction and role of the $\beta$ term in Equation 6 is not fully clear. Could you provide more details including mathematical justifcation on how this term is derived and its exact significance within the  framework? 

2. Could you provide more theoretical justification and intuition behind the Decay-TCL mechanism? Specifically, how do the hyperparameters influence the alignment and fusion process between the physics-based and data-driven dynamics?

3. Could you elaborate  on how the GNN fusion mechanism effectively balances the two components?

4. How scalable is Air-DualODE to larger datasets with thousands of nodes? Could you provide a detailed complexity analysis  of your model as the number of locations (nodes) increases ?

5. Could you provide more clarification on how the predictions can be interpreted and provide specific examples on how these outputs help policy makers or environmental scientists in their decision making process?

6. The proposed model and AirPhyNet (Hettige et al., 2024) employ a similar hybrid architecture of physics based and data driven models , combining GNNs and differential equation solvers. Additionally, both approaches emphasize interpretability through case studies that link predictions to real-world physical phenomena, such as pollutant dispersion influenced by environmental factors like wind speed and direction. Could you elaborate on how your model differs methodologically and in its interpretive capabilities and what specific aspects of your work provide novelty beyond what is addressed in AirPhyNet?

### Soundness
3

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
3

### Summary
The authors propose a model called Air-DualODE that integrates dual branches of Neural Ordinary Differential Equations (ODEs) to enhance air quality predictions. To address the mismatch between explicit physical equations and implicit learned representations, the paper introduces a hybrid model consisting of two branches.

The Physics Branch utilizes the Boundary-Aware Diffusion-Advection Equation (BA-DAE) to capture the spatiotemporal dependencies of pollutants. Meanwhile, the Data-Driven Branch captures dependencies not addressed by the physical equations. These dual representations are skillfully fused using a dynamics fusion module with decaying temporal alignment, which enhances prediction accuracy. 

The effectiveness of this method is demonstrated on two widely-used air quality prediction datasets, showing that the model outperforms other existing methods while offering strong interpretability compared to purely data-driven approaches.

### Strengths
- Enhanced Physical Modeling: This paper innovatively transforms closed-system physical equations into open-system physical equations based on existing Physics-Informed Neural Networks (PINNs). This modification aligns more closely with real-world conditions and improves interpretability in boundary regions.

- Effective Integration of Knowledge: The DualODE model skillfully integrates physical knowledge with data-driven methods to bridge the gap between physical equations and real-world data. This dual approach ensures that the model captures dependencies that are not addressed by physical equations alone.

- Advanced Fusion Technique: By employing dynamic fusion to combine the representations from the two ODE models, and further refining the output through a Graph Neural Network (GNN), the model effectively captures real-world patterns. This approach outperforms the simplistic method of directly concatenating the two representations, leading to more accurate and realistic predictions.

### Weaknesses
- Insufficient Detail on Data-Driven Neural ODE: The paper does not provide a detailed description of the data-driven Neural Ordinary Differential Equations (ODE). Additionally, the necessity of using a Neural ODE model as the data-driven approach is not clearly justified, which may leave readers questioning its selection over other potential methods.

- Limited Results Presentation: The results presented in the paper are limited to predictions for 3 days and scenarios of sudden changes. Predictions for shorter time frames, such as 1-day and 2-day forecasts, are not included. Furthermore, there are discrepancies between the results reported for the reproduction of other works and those in the original publications, which could undermine the credibility of the comparative analysis.

- Language Errors in Figures and Tables: There are minor language errors in the figures and tables that could cause confusion. These errors necessitate careful cross-referencing with the accompanying text to ensure clarity and accurate interpretation of the data.

### Questions
- Errors in Figure and Table Text: There are textual and informational errors in the figures and tables. In Figure 2, "Pollutant Contentration" should be corrected to "Pollutant Concentration," and there is an error in the expression for Wind speed. Additionally, the last row in Table 2 should be labeled as "Air-DualODE." without “w/o”.

- Incomplete and Redundant Equations in the Appendix: The equations following lines 690-692 in the appendix are incomplete and appear to be redundant with the content in lines 219-220 of the main text. Could these be revised for clarity and completeness?

- Detailed Derivation of Equation 3 and $F^D$: It would be beneficial to provide a more detailed derivation of Equation 3 in the appendix to enhance understanding. Additionally, the $F^D$ formula is not provided. Could you include a more thorough description to facilitate reader comprehension?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
To enhance air quality prediction in open systems, this paper proposes a dual Neural ODE architecture named Air-DualODE. It combines a boundary-aware diffusion-advection Equation (BA-DAE) for physical dynamics with a Neural ODE employing masked spatial attention for data-driven dynamics. The framework also incorporates a fusion mechanism that temporally aligns and merges outputs from these dynamics in a shared latent space. Experimental results on both city- and national-level datasets demonstrate state-of-the-art performance.

### Strengths
1.This paper addresses a significant problem, i.e. air quality prediction in open systems. In particular, the model thoroughly considers the non-conservation of pollutant concentration within the region of interest, introducing the BA-DAE to effectively model sources and sinks within the area.
2.The author propose an interesting Air-DualODE framework that models known physical equations and unknown spatiotemporal dependencies separately, then aligns and fuses them in the latent space. This approach highlights the guiding role of physical knowledge within the model while allowing it to capture unknown spatiotemporal dependencies that may not be described by physical equations.
3.In experiments, the model achieves superior performance across multiple metrics on different spatial scales datasets. Besides, the provided code enhance this paepr’s reproducibility.
4.The writing and structure of this paper are clear and easy to understand.

### Weaknesses
1.The discrete diffusion and advection equations are not thoroughly described in this paper. Highly recommend fully explaining in the appendix.
2.Please elaborate the role of spatial-MSA in data-driven branch presented in Section 3.3.
3.Check some typos in this paper. For example, in Section 4.2 Table 1, "Beijing1718" should be “Beijing”.

### Questions
1.Why using the GNN Fusion after temporal alignment? What about other simple structure like MLP?
2.Is there other data should be included for geospatial graph construction?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Air-DualODE, a novel physics-informed model for predicting air quality that combines the strengths of physics-based and data-driven approaches. Traditional physics-based models are computationally intensive and rely on closed-system assumptions, while data-driven models often lack critical physical insights. Air-DualODE addresses these limitations with a dual-branch design using Neural Ordinary Differential Equations (Neural ODEs): one branch incorporates open-system physical equations to model spatiotemporal dependencies, while the second captures additional dependencies in a data-driven manner. The two branches are temporally synchronized and fused for enhanced predictive accuracy. Experimental results show that Air-DualODE outperforms existing methods in predicting pollutant levels across different spatial areas, making it a robust tool for air quality forecasting.

### Strengths
1) Very well written paper combined with appropriate plots.

2) Recognizing that air pollution transport occurs in an open system, the authors redefine the diffusion-advection equation in explicit spaces. This new formulation, termed BA-DAE, aligns the physical equations more accurately with real-world pollutant transport in open air environments, enhancing the model's applicability and reliability.

3) The proposed model, Air-DualODE, uniquely integrates both Physics Dynamics and Data-Driven Dynamics. This dual-branch structure leverages the advantages of physical models (to capture foundational physical behaviors) and data-driven insights (to adapt to complex patterns not covered by physics alone). This approach represents the first dual-dynamics deep learning model specifically tailored for air quality prediction in open systems.

4) Experimental results demonstrate that Air-DualODE outperforms existing models, achieving state-of-the-art accuracy in forecasting pollutant concentrations across diverse spatial scales, from city-wide to national levels.

### Weaknesses
The dual-branch structure in Air-DualODE, while innovative, adds complexity to the model, potentially making it less interpretable than simpler models. This may pose challenges for stakeholders, such as policymakers, who require clear explanations of how predictions are made. But this is not a very important point neither does it offer any reason to not accept this paper.

### Questions
1) How does Air-DualODE perform in regions with sparse or inconsistent air quality data? Are there mechanisms in place to handle data gaps, or do you recommend a minimum data density for effective predictions?

2) Has Air-DualODE been tested on pollutants other than those mentioned in the paper? Could this framework be adapted to predict other types of environmental data, such as water quality?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes Air-DualODE, a novel approach for air quality prediction that combines physics-based and data-driven methods using dual Neural ODEs. The physics branch implements a modified diffusion-advection equation with a correction term for open systems (BA-DAE). In contrast, the data-driven branch employs masked attention-based Neural ODEs to capture unknown dynamics. The two branches are temporally aligned using a decaying contrastive learning scheme and fused in latent space using GNN, demonstrating superior performance on city-scale (Beijing) and national-scale (KnowAir) datasets.

### Strengths
1. The paper addresses the limitations of pure physics-based and pure data-driven approaches by proposing a hybrid framework that attempts to leverage the advantages of both methods.

2. The introduction of BA-DAE with a correction term represents an attempt to model open system dynamics, which is more realistic for air quality prediction than traditional closed system assumptions.

3. The model achieves state-of-the-art performance across different spatial scales while maintaining some level of interpretability through its physics branch.

### Weaknesses
1. The paper oversimplifies complex air pollution dynamics using a linear correction term (βX) without proper theoretical justification, undermining its claim of accurate open system modeling.

2. The approach loses physical interpretability when projecting to latent space and violates conservation laws, raising concerns about numerical stability and contradicting the paper's emphasis on physics-informed modeling.

3. The computational efficiency claims are questionable as the dual branch architecture with multiple ODE solvers likely increases computational burden rather than reducing it.

4. The experimental validation is limited, with case studies confined to Beijing data and lacking crucial analyses such as parameter sensitivity testing and solver comparisons.

5. The technical documentation is incomplete, with key mathematical elements missing from figures and insufficient details about architectural choices, making reproducibility challenging.

### Questions
Q1. Given that real air pollution sources (industrial activity, vehicle emissions) and sinks (forests, lakes) exhibit complex non-linear relationships, why did you simplify the correction term $\beta X$ as a linear term? What is the physical justification for setting $\beta$'s range to $[−1, +∞)$?

Q2. While you claim that the Physics branch explicitly models physical phenomena, how is this physical interpretability preserved when projecting into latent space?

Q3. Regarding the temporal alignment process using Decay-TCL, how do the chosen values of $\lambda_1 = 1$ and $\lambda_2 =0.8$ guarantee physically meaningful alignment? What is the physical significance of using time-decaying weights?

Q4. Why specifically choose Spatial-MSA in the Data-Driven branch? How does this align with a physics-informed approach?

Q5. The authors justify GNN fusion based on 'distance-dependent influence', but isn't this characteristic already considered in the Physics branch?

Q6. How can the authors justify the performance on the national-scale KnowAir dataset when case studies are limited to the Beijing dataset?

Q7. How does the visualization of $\beta$ values correspond to actual observed pollution source/sink data?"

Q8. Can authors perform sensitivity analysis for different ranges of $\beta$ values?

Q9. The authors seem to only consider the DOPRI5 ODE solver. Could they analyze performance and runtime differences when using simpler methods like Euler or RK4?

Q10. The paper should reference and compare with recent work on climate modeling using diffusion and diffusion-advection equations in neural ODE frameworks [1,2]. Can authors clarify their position by analyzing similarities and differences in their approach to diffusion and advection?

Q11. How does the intentional violation of conservation law in BA-DAE affect numerical stability, particularly for ODE solvers?

Q12. How do you ensure that the BA-DAE in the Physics branch and Neural ODE in the Data-driven branch operate in the same state space?

Q14. Is 'Physics-Informed' appropriate in the title? Would 'Physics-guided' or 'Physics-inspired' be more accurate, given that this might be confused with traditional PINN approaches?

Q15. Can authors provide more details about the RNN used in the Coefficient Estimator?

Q16. Figure 2 lacks several elements mentioned in the text, particularly α from equation 6. The relationship between Dynamics fusion and Section 3.4 equations needs clarification.

Q17. Can authors provide visualizations or distribution analyses showing how Gdiff and Gadv change dynamically with wind speed and direction?

Q18. While criticizing the computational cost of existing physics-based methods, how does your dual branch architecture with a complex fusion mechanism improve efficiency? Doesn't using two ODE solvers increase computational burden?

Q19. Can the authors include the number of forward evaluations (NFE) comparisons in Table 2's ablation studies?

Q20. The authors should cover related work on GNNs that redesign the diffusion equation [3,4] and its variations[5,6] using NODE. Can you discuss more about what the authors' methods have in common and what they differ from?

> [1] Choi, Hwangyong, et al. "Climate modeling with neural advection-diffusion equation." Knowledge and Information Systems 65.6 (2023): 2403-2427.
> 
> [2] Hwang, Jeehyun, et al. "Climate modeling with neural diffusion equations." 2021 IEEE International Conference on Data Mining (ICDM). IEEE, 2021.
>
> [3] Wang, Yifei, et al. "Dissecting the diffusion process in linear graph convolutional networks." Advances in Neural Information Processing Systems 34 (2021): 5758-5769.
>
> [4] Chamberlain, Ben, et al. "Grand: Graph neural diffusion." International conference on machine learning. PMLR, 2021.
>
> [5] Thorpe, Matthew, et al. "GRAND++: Graph neural diffusion with a source term." ICLR (2022).
>
> [6] Choi, Jeongwhan, et al. "Gread: Graph neural reaction-diffusion networks." International Conference on Machine Learning. PMLR, 2023.

### Soundness
2

### Presentation
2

### Contribution
2
