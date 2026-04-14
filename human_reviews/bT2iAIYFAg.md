# Vector Segmented and Recombined Adaptation for Scalable and Efficient Model Tuning

- Decision: Reject
- Scores: 5, 5, 5, 3

## Abstract
Among the most commonly utilized parameter-efficient fine-tuning (PEFT) methods, LoRA and its variations have achieved significant popularity. The Vector-based Random Matrix Adaptation (VeRA), one typical variant, utilizes random weights and projections to reduce the number of trainable parameters greatly. However, it requires additional GPU memory and computational resources, probably resulting in a lack of scalability that leads to performance bottlenecks in complex tasks. Besides, the inappropriate initialization of random matrices may affect model performance. To address these problems, we propose a new method called Vector Segmented and Recombined Adaptation (SeRA). SeRA segments input vectors into sub-vectors for individual dimensionality reduction, then introduces a square matrix to combine the information from the reduced sub-vectors, and finally expands the dimensionality independently to adapt the size of pre-trained model. SeRA allows for flexible increase of trainable parameters to enhance performance in complex tasks, and avoids the problem caused by random matrices initialization. Through evaluations on the image classification, cross-modal image-text retrieval, instruction-tuning and GLUE benchmark, we demonstrate the scalability and efficiency of SeRA. Furthermore, we utilize Singular Value Decomposition on the adaptation matrices of SeRA, to analyze how the information characteristics of the matrices change in different ranks and tasks. 
The results can serve as the guide for selecting appropriate parameter amounts in different tasks.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This work proposed a new PEFT method, an enhanced version of LoRA and VeRA to reduce the GPU memory and computational resources.

### Strengths
The proposed method SeRA allows for flexible increase of trainable parameters to enhance performance in complex tasks, and avoids the problem caused by random matrices initialization. The result looks good,

### Weaknesses
The most critical concern is the evaluation benchmark. Currently, the authors just used only one dataset for each task, especially for the image classification; the author only adopted 10% of the training data to evaluate the method. I strongly recommend the author use the whole dataset and more different dataset for each task to evaluate the proposed method. 

Structural complexity: In practice, SeRA’s sub-vector segmentation and matrix recombination increase the complexity of implementation and optimization.

### Questions
Limited improvement on low-dimensional tasks: For simpler tasks with lower requirements for parameter adjustment, SeRA may offer limited performance gains.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a  method called Vector Segmented and Recombined Adaptation (SeRA). SeRA segments input vectors into sub-vectors for individual dimensionality reduction, then employs a square matrix to integrate the information from the reduced sub-vectors. Finally, it independently expands the dimensionality to align with the size of the pre-trained model. In summary, this paper combines MeLoRA and MoSLoRA, using the final project matrix $W_0$ in multi-head attention in  to justify the rationale behind this combination.

### Strengths
1. The paper is well-written and easy to follow.
2. The motivation is clear: inspired by the role of   $W_0$ in multi-head attention, this paper proposes combining MeLoRA and MoSLoRA.
3. The experiments are sufficient, which involve autonomous driving image classification, image-text retrieval, and LLM-related tasks.

### Weaknesses
1. This work appears incremental, and the contribution of this paper may be limited, as its primary novelty was previously introduced in MeLoRA and MoSLoRA. Therefore, this paper resembles a technical report and presents a commendable attempt in a real technical setting.
2. Since VeRA's limitations are explicitly mentioned in the abstract, what is the relationship between your paper and VeRA? It seems inappropriate to critique VeRA without establishing a direct relationship between this paper and VeRA, or without making targeted improvements to VeRA.
3. The Related Work section is limited, as it primarily focuses on LoRA-related content.
4. I will consider raising my score if the novelty issue is well explained in the rebuttal phase.

### Questions
1. In Section 3.2, does  $d$  refer to, $d_{in}, $ $d_{out}$, or either of them? And according to your analysis, the number of parameters fine-tuned in SeRA is twice that of the parameters fine-tuned in VeRA, rather than being 'similar'. Furthermore, the value of $r$ in VeRA is significantly larger than the  $r$  in your method, which explains why these two methods have similar trainable parameters. You should better label   $r_1$ and $r_2$ to tell that difference instead of using the same notation.
2. In the classification problem, why is the rank of MeLoRA set to 1024 instead of a smaller value, as used in the MeLoRA paper?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper introduces a novel parameter-efficient fine-tuning (PEFT) method called Vector Segmented and Recombined Adaptation (SeRA). SeRA segments input vectors into smaller sub-vectors for dimensionality reduction, then combines the information from these reduced sub-vectors using a square matrix, and finally expands the dimensionality independently to match the size of the pre-trained model. While this method appears to have a degree of novelty, the presentation is unclear, and the experimental results do not show a clear advantage.

### Strengths
The method presented in this work does show some innovation in it represents the increment matrix.

### Weaknesses
(1) Theoretically, VeRA should have fewer trainable parameters than SeRA, as A and B are frozen in VeRA.  As shown in the experimental results, VeRA’s trainable parameter count appears similar to, or even lower than, SeRA’s in some cases, while achieving comparable performance on downstream tasks. This raises the question: what exactly is SeRA’s advantage over VeRA?
(2) In terms of presentation, the paper attempts to introduce SeRA by building on VeRA, but the connection between the two methods is not clearly presented. As a result, the intuition and motivation behind this method are not well explained. 
(3) Additionally, the presentation of Section 3, particularly the part on Parameter Count Analysis, could be improved. The relationship between “the number of parameters involved in the forward process of VeRA” and its scalability is also unclear. What does “while SeRA maintains the original number” mean in this context? What exactly is meant by scalability, and why is it so important for fine-tuning tasks?
(4) Since SeRA can be seen as a variant of LoRA, I suggest that the experiments and discussions should include more comparisons with other low-rank-based fine-tuning methods.

### Questions
See the above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper, titled "Vector Segmented and Recombined Adaptation for Scalable and Efficient Model Tuning", introduces a novel parameter-efficient fine-tuning method called SeRA. This method addresses limitations in existing fine-tuning methods like LoRA and VeRA, which struggle with scalability or encounter performance bottlenecks due to reliance on random matrices and high GPU resource consumption. 
SeRA innovatively segments input vectors into sub-vectors for individual dimensionality reduction, then uses a square matrix to recombine and expand the dimensionality of these reduced sub-vectors, enhancing scalability without significant extra computational cost. The authors evaluate SeRA across tasks such as image classification, cross-modal image-text retrieval, instruction-tuning, and the GLUE benchmark, where SeRA demonstrates high efficiency in simpler tasks and surpasses other methods in complex settings by allowing flexible parameter adjustments. Additionally, by using Singular Value Decomposition (SVD) analysis, the paper examines the effects of varying matrix ranks on information retention, providing insights into selecting optimal parameter amounts for different tasks.

### Strengths
* The paper presents a novel parameter-efficient fine-tuning method called Vector Segmented and Recombined Adaptation (SeRA). This approach is original in its segmentation of input vectors for dimensionality reduction, which is a creative adaptation of existing concepts in model tuning. By effectively combining insights from previous methods like LoRA and VeRA, SeRA offers a unique solution that addresses scalability issues while maintaining performance. 
* The paper is well-structured and clearly written, with a logical flow that guides the reader through the problem statement, methodology, experiments, and conclusions.

### Weaknesses
* The experimental validation primarily focuses on tasks such as image classification and cross-modal retrieval, which may limit the perceived versatility of SeRA. To improve this aspect, the authors should consider evaluating SeRA across a broader range of applications, including domains like time series analysis or reinforcement learning. Including diverse datasets would demonstrate the method's adaptability and robustness, thus reinforcing its significance.
* The paper lacks comprehensive comparisons with other recent parameter-efficient fine-tuning methods beyond LoRA and VeRA. For example, methods such as MELoRA and MoSLoRA should be included to provide a clearer benchmark of SeRA's performance. Detailed comparisons using standard datasets and a variety of metrics (e.g., accuracy, training time, resource consumption) would offer a more nuanced understanding of SeRA's advantages and potential drawbacks.
* While the authors perform some ablation studies, the scope is relatively narrow. Expanding these studies to include a wider range of configurations—such as different ranks for adaptation matrices and their interactions—would yield deeper insights into the effects of various components of SeRA. A structured approach, like a grid search across parameter combinations, could be employed to systematically identify optimal settings for different tasks.
* The paper acknowledges issues related to random matrix initialization but does not provide adequate solutions or strategies to mitigate this issue. Including a comparative analysis of different initialization techniques and their impact on performance would not only address this issue but also guide practitioners in applying SeRA effectively.

### Questions
* Could you provide a more extensive comparison of SeRA against other recent parameter-efficient fine-tuning methods, such as MELoRA and MoSLoRA? Specifically, how does SeRA perform on various datasets in comparison to these methods? Including additional metrics (e.g., resource consumption, training time) would help clarify its relative strengths and weaknesses.
* Have you considered evaluating SeRA on a wider range of tasks beyond those presented? For instance, applying the method to time series analysis or reinforcement learning could demonstrate its versatility. What are your thoughts on extending the experimental validation to include these areas?
* You mention the sensitivity of SeRA to random matrix initialization. Could you provide insights into which initialization strategies you have found most effective for your experiments? Additionally, could you discuss how different initialization methods could impact the reproducibility of your results?
* What specific real-world applications do you envision for SeRA? Providing examples or hypothetical scenarios that illustrate how SeRA could be applied in practical settings would enhance the paper's practical relevance.

### Soundness
2

### Presentation
2

### Contribution
2
