# Prioritize Alignment in Dataset Distillation

- Decision: Reject
- Scores: 6, 5, 6, 3

## Abstract
Dataset Distillation aims to compress a large dataset into a significantly more compact, synthetic one without compromising the performance of the trained models. 
To achieve this, existing methods use the agent model to extract information from the target dataset and embed it into the distilled dataset. 
Consequently, the quality of extracted and embedded information determines the quality of the distilled dataset.
In this work, we find that existing methods introduce misaligned information in both information extraction and embedding stages.
To alleviate this, we propose Prioritize Alignment in Dataset Distillation (\textbf{PAD}), which aligns information from the following two perspectives.
1) We prune the target dataset according to the compressing ratio to filter the information that can be extracted by the agent model.
2) We use only deep layers of the agent model to perform the distillation to avoid excessively introducing low-level information.
This simple strategy effectively filters out misaligned information and brings non-trivial improvement for mainstream matching-based distillation algorithms.
Furthermore, built on trajectory matching, \textbf{PAD} achieves remarkable improvements on various benchmarks, achieving state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces PAD, a method aimed at enhancing the compression of large datasets into compact synthetic datasets while preserving model performance. PAD addresses the challenge of misaligned information during distillation by aligning data through selective pruning of the target dataset and leveraging deep layers of agent models in the distillation process.

### Strengths
1. The identification of misalignment in dataset distillation and the proposal of PAD to address it is a valuable contribution.
2. The paper provides a clear motivation for the PAD method and supports its claims with extensive experiments.
3. The approach of prioritizing deep layer parameters for distillation is innovative and leads to improved performance.

### Weaknesses
1. It would be beneficial to prove the relationship between EL2N and the difficulty of training samples.
2. The paper could provide a more detailed comparison with other state-of-the-art methods, especially in terms of computational efficiency and scalability. 
3. The theoretical analysis of why deep layer parameters are more suitable for distillation is lacking. 
4. The paper does not discuss potential limitations or failure cases of the PAD method.

### Questions
1. How does PAD compare to other recent advances in dataset distillation in terms of computational resources and scalability?
2. Could the authors elaborate on the theoretical justification for prioritizing deep layer parameters?
3. Are there any scenarios where PAD might not perform as expected, and if so, how does the method handle such cases?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper explores the role of alignment in both the information extraction and embedding stages within the dataset distillation process. 
During the information extraction stage, alignment is achieved by selecting subsets of the dataset based on difficulty. For settings with a low images-per-class (ipc) count, incorporating a higher proportion of ‘easy’ data proves effective. Conversely, in high ipc settings, utilizing a larger portion of ‘hard’ data enhances distillation effectiveness. The EL2N score, analogous to model confidence, is employed as a difficulty metric.
In the information embedding stage, alignment is achieved by prioritizing deeper over shallower layers. Deeper layers are more adept at learning semantic information, resulting in a higher-level representation within synthetic data and more efficient distillation.

### Strengths
1. One advantage of this method is that it can be applied on top of other distillation methods, such as MTT and DM. As seen in cross-architecture analyses, it’s a general approach that enhances performance in a scalable, model- and dataset-agnostic way. Experimental results show that accuracy improved over traditional methods (Table 1), with increases across multiple datasets and architectures (Table 2).

2. It seems the straightforward method is supported by detailed analysis and experiments. Experiments (Tables 4, 5 / Figure 5) effectively support the hypothesis that removing easy examples and using deeper layers improves performance, making the qualitative results intuitively understandable (Figures 4, 6).

3. While the writing isn’t exceptionally well-crafted, it is organized in a readable and accessible manner.

### Weaknesses
1. I believe the contribution of this method is limited. The proposed method is very simple, and while it is supported by empirical experimental results, it lacks theoretical justification. Additionally, as mentioned in the paper, the data selection and distillation approach was already introduced by BLiP (Xu et al., 2023). Although the paper presents experimental results showing superior performance to this approach (Table 5.b), the metric for selecting difficulty (EL2N) was also previously proposed.

2. The method seems somewhat sensitive to hyperparameters (AEE, IR). Accuracy varies depending on these hyperparameters (Table 4.a), with differences comparable to the performance increase over other methods in the main table. Additionally, as seen in Table 10, optimal hyperparameters change dynamically across datasets and ipc values, raising questions about the method’s stability. If it is indeed sensitive, hyperparameters may need to be tuned for each architecture and dataset, which raises concerns about the method’s practicality.

3. Although the paper claims that performance improves when used alongside various distillation techniques, as mentioned in the limitations section, due to computing resource constraints, experiments were conducted only with DM and DC (and DATM in the main table).

Overall, the paper demonstrates that a simple method can enhance distillation performance and can be used in a model- and architecture-agnostic manner. The experiments and logical development are well-executed; however, I believe the contribution of the method itself is limited, and it appears to be sensitive to hyperparameters. Therefore, I would rate it as [5: marginally below the acceptance threshold].

### Questions
1. In the main table (Table 1), it appears that multiple experiments were conducted to obtain an average, but subsequent tables present experimental values as single data points. I couldn't find any information in the paper regarding whether the results for each experiment were averaged, and I’m curious about this.

2. The study exclusively uses deep layers; are shallow layers entirely unnecessary? Shallow layers likely play a role in producing a good embedding space, so is there a way to leverage them effectively?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces "Prioritize Alignment in Dataset Distillation" (PAD), a method improving dataset distillation by focusing on two main strategies. First, it adjusts data selection based on different Information Compression Ratios (ICRs) to match the required difficulty levels. Second, it enhances results by distilling only from deep-layer network parameters. The effectiveness of PAD is validated through experiments on standard datasets like CIFAR-10, CIFAR-100, and Tiny ImageNet, showing improvements over current DD methods.

### Strengths
1. The paper demonstrates substantial improvements in DD through its experimental results in TAB. 1, showing that the PAD method significantly enhances the effectiveness of dataset distillation.
2. PAD proves to be highly adaptable and generalizes well across multiple datasets, indicating that the method can be effectively applied to various dataset distillation tasks with consistent success.

### Weaknesses
1. There are some formatting issues in the manuscript that need attention. In **Table 5**, the annotations (a) and (b) appear to be reversed, which could confuse readers. Additionally, a period is missing at the end of line 485 after "smaller IPCs." Please correct these to enhance the clarity and professionalism of the document.
2. There seems to be a mismatch between the issues presented in Section 2.1 and the methods proposed in Section 3.2. The former discusses a static selection method, while the latter introduces a dynamic approach. Could the authors clarify the connection between these sections to ensure the consistency of the methodology described?

### Questions
1. The manuscript employs DATM as a baseline, which, as I understand it, requires the pre-training of numerous agent models on the original dataset to record training trajectories. As discussed in w2, given the dynamic dataset scheduler nature of the process described in your methodology, does this imply the need to train additional agent models? It would be beneficial for readers if the authors could provide more detailed insights into the implementation specifics of this approach.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper focuses on the task of dataset distillation, which aims to compress a large dataset into a much more compact synthetic dataset while maintaining the performance of trained models. Existing methods rely on an agent model to extract and embed information from the target dataset into the distilled version. However, the authors identify that current approaches often introduce misaligned information during the extraction and embedding stages, which degrades the quality of the distilled dataset.

To address this, the authors propose Prioritize Alignment in Dataset Distillation (PAD), a method that aligns information from two main perspectives: Dataset Pruning and Deep Layer Utilization. This simple yet effective strategy helps filter out misaligned information, resulting in significant improvements for mainstream matching-based distillation algorithms. Additionally, when applied to trajectory matching, PAD achieves state-of-the-art performance across various benchmarks, showcasing its effectiveness.

### Strengths
1. Figures 2 and 3 provide valuable insights into the effects of removing both simple and difficult samples, as well as the impact of shallow parameters on model performance.

2. The authors present a comprehensive experimental analysis on a small-scale dataset, including ablation studies, hyperparameter analysis, and related discussions.

### Weaknesses
1. The author's approach appears to be more incremental, incorporating only minor enhancements to DATM. Firstly, Section 3.1 serves as a review of previous work, and Equation (4) in Section 3.3 shows only slight modifications from Equation (3) in DATM. Additionally, Section 3.2 seems to function more as a heuristic for selecting difficult samples. Overall, the method introduces only two additional techniques compared to DATM, lacking a significant breakthrough in terms of innovation.

2. in Table 2, most of the average improvements over the comparable method, DATM, are less than 1 point, indicating that the performance still requires further enhancement.

### Questions
The thesis would benefit from a more structured presentation, where the authors are encouraged to list observations and analyses in Chapter 2 in a systematic manner, similar to DTAM. Additionally, the content should be included in the methods section as a motivation or exploration segment to strengthen the logical flow and context of the proposed

### Soundness
3

### Presentation
3

### Contribution
2
