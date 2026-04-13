## Human Reviewer 1

### Summary
This paper aims at including human decision processes and social influence to observe criminal event counts. The proposed model is ambitious to include multiple human decision-making aspects, but the details of formulation and examination are missing. The experimental setup needs further reference to show its practicality.

### Strengths
S1. The experiments are conducted with three real-datasets.
S2. The writing is fluent and easy to understand.

### Weaknesses
W1. Overall, the major concerns are that the paper may not be self-contained and appears disconnected. First, although Fig. 1 visualizes the structure of the proposed model, most details explaining each part are not presented. For instance, what are the differences between spatial and position info? Which model does each expert use? Second, although the abstract and introduction state that social norms, environmental cues, and various other factors are considered, there is no corresponding formulation in Section 3. Finally, the experimental results do not validate these claims either. It is suggested to connect the claims with detailed descriptions in the methods and experiment sections.

W2. Please include up-to-date related works in top journals [1][2][3]. Moreover, half of the comparative baselines in the experiment section were published more than 10 years ago, which may be too outdated for fair comparisons. It is suggested to compare with newer methods instead.
[1] Weichao Liang, Zhiang Wu, Zhe Li, Yong Ge: CrimeTensor: Fine-Scale Crime Prediction via Tensor Learning with Spatiotemporal Consistency. ACM Trans. Intell. Syst. Technol. 13(2): 33:1-33:24 (2022)
[2] Shuai Zhao, Ruiqiang Liu, Bo Cheng, Daxing Zhao: Classification-Labeled Continuousization and Multi-Domain Spatio-Temporal Fusion for Fine-Grained Urban Crime Prediction. IEEE Trans. Knowl. Data Eng. 35(7): 6725-6738 (2023)
[3] Weichao Liang, Jie Cao, Lei Chen, Youquan Wang, Jia Wu, Amin Beheshti, Jiangnan Tang: Crime Prediction With Missing Data Via Spatiotemporal Regularized Tensor Decomposition. IEEE Trans. Big Data 9(5): 1392-1407 (2023)

W3. The definitions of matrices A and B on line 227, page 5, and the purpose of formulating them are unclear. Specifically, what do the two matrices embed, respectively? Additionally, right before introducing these matrices, the model already includes positional, spatial, temporal, and feature embeddings. An alternative approach might be to directly feed these four embeddings to the experts, rather than combining them with the two matrices to avoid additional computational overhead. This raises questions about the necessity, purpose, and benefit of the intermediate matrix decomposition-based embedding method compared to a straightforward alternative.

W4. Please clarify the “ranking” concept in the gating function, starting from line 251 on page 5. Equations 7, 8, and the loss function at line 274 resemble a cross-entropy formulation, which is a classification-based metric rather than a ranking one. Additionally, I am uncertain whether ranking is appropriate in this scenario. Specifically, while predicting the time and place of a crime, a top-1 ranking for occurrence may not directly indicate that a crime is happening, as the probability could still be low. Therefore, relying on ranking rather than probability prediction may lead to false alarms and overreactions.

W5. The practicality of the experimental setup is questionable. In the New York Crime and Chicago Crime datasets, each city is divided into 100 areas, and daytime is segmented into 4 time slots. However, it is unclear how large each area is after division. Is there evidence or a reference supporting that the 100-block granularity is beneficial for real-world law enforcement? Similarly, dividing daytime into four 6-hour slots may not be sufficiently granular. Is there a reference justifying this setup? Furthermore, it would be interesting to see the model’s performance at finer granularities, with smaller areas and shorter time slots.

W6. The experimental results may not fully examine the authors' claims. While modeling the “human decision process” is a key focus, it is unclear how this is tested in the experiments. Are there specific sequential criminal events in the datasets? If so, does the proposed method successfully retrieve these sequences? How does the model demonstrate that its improvements are due to modeling the human decision process? Otherwise, if each crime is independent, how are the datasets suitable for examining causal relationships? In this context, could simple statistics identify criminal hotspots at specific time slots to yield similar results to those in Fig. 2? It is recommended to elaborate further on human decision modeling in the experiments.

### Questions
Please refer to W3 to W6.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper presents a novel framework that integrates choice theory with social intelligence to model spatial-temporal counting processes, such as crime occurrences and bike-sharing activities. By capturing latent human preference patterns through utility functions, the model aims to provide deeper insights into the mechanisms driving these events. Empirical evaluations using crime and bike-sharing datasets show that the proposed model offers high predictive accuracy and interpretability compared to existing methods, though potential limitations and future research directions are not extensively discussed.

### Strengths
1. **Innovative Approach**: The paper introduces an innovative framework that integrates choice theory with social intelligence to model spatial-temporal counting processes. This approach addresses the complex decision-making processes and social factors influencing human-generated event data, such as crime occurrences and bike-sharing activities.
2. **Interpretable Insights**: The model provides interpretable insights by uncovering latent human preference patterns through utility functions. This feature helps in understanding the underlying mechanisms driving the observed event counts, which is valuable for both academic and practical purposes.
3. **Predictive Performance**: Empirical evaluations using crime and bike-sharing datasets show that the proposed model achieves good predictive accuracy compared to existing methods. The results indicate that the model can effectively predict event patterns and offer useful insights.
4. **Theoretical Foundation**: The paper derives a generalization bound that is independent of the number of latent classes, providing a theoretical foundation for the model's robustness and reliability. This theoretical contribution adds to the academic value of the work.
5. **Practical Flexibility**: The model demonstrates flexibility in handling different types of spatial-temporal data and can incorporate external interventions, making it adaptable to various real-world scenarios.

### Weaknesses
1. **Interpretability Validation**: While the model emphasizes interpretability, this claim is not fully supported with detailed case studies or qualitative analyses. More concrete examples and validation are needed to ensure that the insights provided are actionable and meaningful. Without such validation, the interpretability aspect, though highlighted as a strength, remains somewhat abstract and less convincing.
2. **Computational Efficiency**: The paper does not extensively address the computational efficiency of the model. Practical applications often involve large-scale datasets, and understanding the model's scalability and resource requirements is crucial. Without this information, it is challenging to determine the feasibility of deploying the model in real-world settings, which could limit its practical utility.
3. **Future Research Directions**:
The paper does not clearly outline future research directions or potential extensions of the model. Discussing these aspects would provide a clearer path for advancing the field and addressing current limitations. Identifying open questions and suggesting avenues for further investigation would enhance the paper's contribution and encourage ongoing research in this area.

### Questions
1. How do hyperparameter changes, such as learning rate, regularization parameters, and the number of mixture components, affect the model's performance?
2. In what ways can the model be tested on a variety of datasets with different spatial and temporal characteristics to assess its generalizability?
3. How can cross-validation and out-of-sample testing be conducted to ensure the model's stability and consistency?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper is about the prediction problem for spatial-temporal event data generated by humans.  The authors introduced a framework integrating choice theory with social intelligence to model and analyze counting processes. The authors further conducted experiments on several real-world spatio-temporal datasets, and empirical evaluation of crime and bike-sharing datasets demonstrated that the proposed model could achieve the best performance.

### Strengths
1. The studied forecasting problem of spatio-temporal events is very important, interesting, and of high value in the real world. 
2. The presentation is overall good, and the organization makes the paper easy to read and comprehend.
3. The authors select two representative metrics, aRMSE and MAPE, on which the proposed method achieves the best performance among all these models.

### Weaknesses
1. The datasets are small, leading to convincing results and conclusions. Although the authors have considered three datasets, NYC Crime, Chicago Crime, and Shanghai Mobike, for evaluation, the scales of these datasets are quite limited. There are only less than 1000 events on the first two datasets, which makes us wonder whether the proposed method can be used in real-world applications where the dataset may be very huge.
2. The technical contribution of the proposed method is questionable. The proposed method introduces a strategy of MoE, which is widely used in model ensembling and limits the contribution of the whole framework. In other words, it is very likely to improve performance by adding the MoE module. In short, the proposed solution is a bit straightforward.
3. Figure 2, Figure 3, and Figure 4 require improvement. Observing some informative and insightful conclusions from these figures is very hard since the grids are coarse-grained.

### Questions
Please answer the questions corresponding to the weaknesses mentioned above.
1. Why use these small datasets for evaluation? What about the actual value of the proposed method when applied to large-scale datasets?
2. How do you explain the performance improvement of the MoE module and the relation between it and the overall performance improvement?
3. How about the performance improvement when we have fine-grained spatial grids?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
5

---

## Human Reviewer 4

### Summary
This paper presents a new spatial-temporal counting process model that integrates choice theory and social intelligence to capture human decision-driven event occurrences, such as crime rates and bike-sharing usage. The core idea is to use latent utility functions to represent diverse decision-making factors and to apply a mixture-of-experts model with a sparse gating function for adaptive selection. The model aims to reveal underlying patterns in counting processes, providing both predictive power and interpretability.

### Strengths
The paper is methodologically sound, with a well-defined approach supported by both theoretical and empirical analyses. The experimental setup is robust, including multiple real-world datasets, and the model's performance is compared against established baselines to highlight its predictive strength. 

The paper is well-structured and provides comprehensive explanations of its key components, including the latent utility functions, mixture-of-experts model, and gating function. Diagrams and formulas aid in clarifying complex concepts, making the model's framework accessible for readers. 

This framework contributes significantly to spatial-temporal modeling, especially in domains where human decision-making drives event occurrences. By enabling a nuanced understanding of preference-driven behavior and offering predictive power, the model has applications in fields like criminology, urban planning, and shared mobility systems.

### Weaknesses
The use of mixture-of-experts and the sparse selection mechanism may raise concerns regarding computational scalability when applied to large-scale, high-dimensional spatial-temporal data. While the model performs well on mid-sized datasets, it is unclear if the sparse gating function and multiple experts could handle significantly larger spatial grids or finer temporal resolutions without substantial computational costs. A discussion on computational efficiency or optimization strategies, such as parallelization, would strengthen the model’s applicability to broader scenarios.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
2