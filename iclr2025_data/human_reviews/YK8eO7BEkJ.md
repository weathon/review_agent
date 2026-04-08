## Human Reviewer 1

### Summary
This paper investigates normalization layers in sequence and vision-based Mamba models from three perspectives: 1) position of norm layer, 2) norm combination, and 3) intuition for choosing a combination. The paper explores 5 different normalization techniques with two different types of datasets.

### Strengths
This paper provides interesting insights into the usage of the normalization layers in a mamba architecture and comprehensive empirical results that can help improve future versions of it.

### Weaknesses
While the paper shows interesting insights about normalization in mamba and is empirically good, it is not quite novel and does not provide any insight that can be considered novel. Most experiments are performed on two, relatively smaller datasets (ImageNet's smaller subset). Hence, it is not clear how broadly applicable the insights are. Moreover, it is not difficult to re-run these experiments for a new task and find these insights customized for a particular task. Finally, previous works have extensively explored normalizations in a related context. It would be beneficial to include a discussion about these works. 

Xiong, Ruibin, et al. "On layer normalization in the transformer architecture." International Conference on Machine Learning. PMLR, 2020.
Kai, Hu, et al. "Is normalization indispensable for training deep neural networks?."
Liu, Hanxiao, et al. "Evolving normalization-activation layers." Advances in Neural Information Processing Systems 33 (2020): 13539-13550.
Santurkar, Shibani, et al. "How does batch normalization help optimization?." Advances in neural information processing systems 31 (2018).
Bjorck, Nils, et al. "Understanding batch normalization." Advances in neural information processing systems 31 (2018).
Awais, Muhammad, Md Tauhid Bin Iqbal, and Sung-Ho Bae. "Revisiting internal covariate shift for batch normalization." IEEE Transactions on Neural Networks and Learning Systems 32.11 (2020): 5082-5092.
Peng, Hanyang, Yue Yu, and Shiqi Yu. "Re-thinking the effectiveness of batch normalization and beyond." IEEE Transactions on Pattern Analysis and Machine Intelligence (2023).


Minor Improvements

1. Presentation of figure 1 can be improved to make it more neat and clear. The caption of Figure 1 can also be improved.

### Questions
How applicable are the insights for broader mamba architecture given that results are derived from two relatively small datasets?
How different is this analysis from some of the previous works that have explored normalization in different architectures?

### Soundness
3

### Presentation
2

### Contribution
1

### Rating
3

### Confidence
3

---

## Human Reviewer 2

### Summary
Summary: The paper presents an investigation into normalization techniques within the Mamba architecture, addressing an important aspect of deep learning model stability. While the work makes some valuable contributions to understanding normalization in Mamba networks, there are several significant limitations that need to be addressed.

### Strengths
- Takes on the important challenge of improving training stability in Mamba networks.
- Provides practical insights into normalization layer positioning and combination.

### Weaknesses
Major issues:

1. The examination of normalization effects is restricted to only two tasks, which is insufficient to support broad conclusions. Findings appear task-dependent, with contradictory results between vision and sequence tasks.
2. The focus on long sequence modeling for normalization position analysis lacks justification.
3. The Weight Domination hypothesis lacks sufficient supporting evidence. Contradictory results regarding BN→None performance weaken the Weight Domination hypothesis. The relationship between findings in vision and sequence tasks is not  dequately explained.


Minor issues:

1. Inconsistent visualization approaches in Figure 3 (varying x-axes and plotting styles) hamper interpretation.
2. Results from Vision Mamba (showing LN sufficiency) contradict findings from sequence modeling.
3. The work falls short of providing general insights that could replace task-specific ablation studies.

### Questions
-

### Soundness
2

### Presentation
3

### Contribution
1

### Rating
3

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper systematically investigates the effects of various normalization techniques and their combinations within the Mamba architecture, finding that applying normalization after the S6 Module improves performance and stability. It proposes practical guidelines for selecting optimal normalization strategies to enhance the efficiency and robustness of Mamba networks. But there are still many problems that need to be paid attention to.

### Strengths
1. Relevance and Importance: 
The study addresses a critical aspect of deep learning model stability, particularly in the context of the Mamba architecture, which is gaining attention for its efficiency in handling long sequences.

2. Practical Insights: 
The paper provides valuable insights into the positioning of normalization layers, which can be directly applicable to practitioners working with Mamba networks.

3. Comprehensive Analysis: 
The study systematically evaluates various normalization techniques and their combinations, offering a detailed analysis of their effects on model performance and stability.

### Weaknesses
1. Limited Task Validation:
The study is limited to only two tasks: long sequence modeling with the LRA ListOps dataset, and image classification with ImageNet-100. To draw broader conclusions about the generalizability of normalization techniques across different domains, the authors should consider including additional diverse tasks such as natural language processing, speech recognition, or reinforcement learning tasks.
2. Contradictory Findings and Lack of Explanation: 
The findings appear to be task-dependent, with contradictory results between vision and sequence tasks. For instance, the paper suggests that normalization after the S6 Module is beneficial for sequence modeling (Section 4.1), but this does not hold for image classification tasks (Section 4.3). The authors should provide a more in-depth analysis or explanation of why the results differ between tasks, as this could lead to valuable insights about how normalization techniques interact with different types of data and model architectures.
3. Methodological Issues: 
3.1 Reproducibility Issues: The experimental settings are not well-documented, making it difficult to replicate the results. Crucial details about accuracy metrics in tables are missing. For example, Table 2 and Table 3 lack information on the specific accuracy metrics used. To ensure reproducibility, the authors should provide detailed information about the experimental setup, including hyperparameters, evaluation metrics, and any data preprocessing steps.
3.2 Inconsistent Visualization: The visualization in Figure 3 is inconsistent, with varying x-axes and plotting styles, which hampers the interpretation of the results. For instance, the x-axes in subfigures (2), (3), and (4) are not clearly labeled, making it hard to compare the weight norms across different configurations. The authors should standardize the x-axes, add clearer labels, and ensure consistent plotting styles across subfigures to improve the interpretability of the results.
4. Theoretical Foundation:
Weight Domination Hypothesis: The Weight Domination hypothesis, which is central to the paper's findings, lacks sufficient supporting evidence. The contradictory results regarding the BN→None performance weaken the credibility of this hypothesis. For example, the paper states that the BN→None configuration leads to significant disparity in weight norms (Section 4.1), but this is not consistently observed across all experiments. The authors should provide more empirical evidence supporting the Weight Domination hypothesis or explain the apparent contradictions in the results.
5. Limited Generalizability: 
5.1 Task-Specific Conclusions: The conclusions appear to be highly task-specific, which limits their generalizability. For instance, the results from Vision Mamba, which show that LN is sufficient, contradict the findings from sequence modeling. The paper states that LN+RMSN underperformed relative to LN+LN (Section 4.3), but this contradicts the findings in sequence modeling tasks. The authors should strive to provide more general insights that could replace task-specific ablation studies and offer a unified framework for understanding normalization techniques across different tasks.
5.2 Ablation Studies: The work falls short of providing general insights that could replace task-specific ablation studies. For example, the paper concludes that BN is particularly effective when placed after the Mamba block (Section 4.3), but this is not generalized to other tasks. The authors should aim to derive more generalizable conclusions or provide guidance on when specific normalization techniques may be more appropriate.

### Questions
1. Can the authors include additional diverse tasks from different domains to validate the effectiveness of normalization techniques more broadly?
2. Can the authors provide a more in-depth analysis or explanation for the contradictory findings between vision and sequence tasks, and the potential causes for these differences?
3. Can the authors provide detailed information on the experimental settings, including hyperparameters, evaluation metrics, and data preprocessing steps, to ensure reproducibility?
4. Can the authors improve the visualization in Figure 3 by standardizing the x-axes, adding clearer labels, and ensuring consistent plotting styles across subfigures?
5. Can the authors provide more empirical evidence supporting the Weight Domination hypothesis or explain the apparent contradictions in the results related to the BN→None configuration?
6. Can the authors strive to derive more generalizable conclusions or provide guidance on when specific normalization techniques may be more appropriate, rather than relying heavily on task-specific findings?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
4