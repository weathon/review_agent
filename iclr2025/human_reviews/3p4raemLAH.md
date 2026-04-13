## Human Reviewer 1

### Summary
This paper introduces a method called Single Layer Unlearning Gradient (SLUG), aimed at addressing the challenges of unauthorized generation of privacy-related and copyright-infringing contents. SLUG is designed for unlearning of targeted information from trained machine learning models, requiring only a single gradient computation (and then reuse it) and updating only one layer of the model. This approach minimizes computational costs and maintains the model’s overall utility, particularly for unrelated tasks.
The method has been tested with popular models like CLIP and Stable Diffusion, demonstrating superior efficiency and effectiveness compared to existing methods.

### Strengths
## Strengths

1. **Balanced Unlearning and Performance:** The proposed method effectively balances the unlearning process with the model's general performance, addressing a crucial trade-off in model management.
2. **Computational Efficiency of SLUG:** SLUG requires gradient computation only once, offering two significant advantages:
   - **Faster Computation:** Reduces overall computation time.
   - **Prevention of Over-Unlearning:** Minimizes the risk of excessively removing learned information.
3. **Generalization Across Models:** SLUG demonstrates effectiveness not only on stable diffusion models but also yields promising results on Vision-Language Models (VLMs), showcasing its potential for broader applicability.

### Weaknesses
## Weaknesses

1. **Dependence on Retain Set:** SLUG relies on a retain set to preserve general performance. The methodology for curating this set is critical, yet the paper lacks sufficient discussion or guidelines to ensure reproducibility.
2. **Incomplete Computational Time Analysis:** While Table 1 presents a computation time comparison, the analysis based on $O(N_f + N_r)$ overlooks key factors:
   - **Iterative Parameter Updates:** SLUG requires iterative updates of model parameters as described in Equation 9.
   - **Layer Importance and Gradient Alignment:** The time associated with determining layer importance and performing gradient alignment is not accounted for, potentially underestimating the actual computational cost.
3. **Insufficient Evaluation on VLMs:** The claims regarding SLUG's performance on VLMs are not fully substantiated. More comprehensive experiments are necessary to convincingly demonstrate its superiority in this context.

### Questions
## Questions

1. **Retain Set Curation:** Could the authors provide a detailed explanation of how the retain set is curated? Clarifying this process is essential for reproducibility and assessing the method's robustness.
2. **Iterative Update Performance:** It is recommended to report the performance of the iterative update version of SLUG. If performance metrics decline, this could highlight underlying foundational issues that need to be addressed.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
2

---

## Human Reviewer 2

### Summary
The author proposes a method that only requires one-time gradient calculation to update a single layer of the model for achieving unlearning.  By approximating the importance of the measurement layer using the diagonal of the Fisher information matrix and balancing gradient alignment, the author selects a single target layer and finally updates its parameters in a single step to achieve the desired outcome. The effectiveness of this approach is validated through extensive experiments.

### Strengths
1. The paper is well-organized and easy to follow. This approach achieves effective unlearning with just a single gradient update on one layer, demonstrating remarkable efficiency, particularly in the context of large models.

2. In the proposed approach, the author employs the diagonal of the Fisher information matrix to approximate layer importance, thereby enhancing interpretability.

3. The author conducted extensive experiments on large-scale multimodal models including CLIP, Stable Diffusion, and VLMs, demonstrating the wide applicability of the proposed approach and empirically demonstrating its advantages in balancing efficiency and model utility. And the author provided complete code that supports reproducibility.

### Weaknesses
1. The proposed scheme only updates the most important layer to achieve excellent forgetting effects. Although the experimental results can provide an empirical guarantee for forgetting, intuitively there must be residual information in the remaining layers. From the experimental results, the difference in importance between layers is not large. Hence, it feels more reasonable to update as many layers as possible while maintaining model performance. It is better to add more discussions.
2. The design of the approach requires access to all forgotten and retained data. However, the targeted domain involves relatively large datasets, requiring substantial storage space. If complete access to the data is not feasible, could this negatively impact the effectiveness of the scheme?
3. The paper's description of layer selection is not clear enough, and I did not correspond the graph well with the Pareto optimal set. I cannot clearly understand how the author balances importance of layers and grade alignment.
4.  From the experimental results of unlearning for stable diffusion, it can be seen that unlearning leads to a slight decrease in the quality of image generation. 
5. The experiment of unlearning for VLM lacks quantitative analysis and only shows examples. Adding quantitative analysis will provide clearer evidence for the method.

### Questions
Please check the questions in the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper introduces a novel saliency-based method for the machine unlearning task. The proposed approach, named Single Layer Unlearning Gradient (SLUG), effectively removes targeted information by updating only a single specific layer of the model through a one-time gradient computation. Compared to traditional unlearning techniques, SLUG significantly reduces computational costs while ensuring minimal impact on the model's performance for unrelated content.

The authors evaluate SLUG using metrics such as low computational cost, effective unlearning, and utility retention. They demonstrate the method's efficacy across three downstream tasks: CLIP Zero-Shot Classification, Generative Models on UnlearnCanvas benchmark and Vision-Language Models.

### Strengths
1. **Innovative Saliency-Based Approach to Machine Unlearning**
   
   The paper introduces a novel saliency-based method specifically designed to address the machine unlearning problem. The authors present SLUG technique, which efficiently removes targeted information by updating only a single designated layer of the model through a one-time gradient computation. This method offers a streamlined solution compared to traditional unlearning techniques that often require extensive model modifications and incur high computational costs.

2. **Comprehensive Validation Across Diverse Downstream Tasks**
   
   The effectiveness of the proposed SLUG method is thoroughly validated across three distinct downstream tasks, demonstrating its versatility and robustness: CLIP-Based Image Classification, Stable Diffusion-Based Image Generation and Vision-Language Models (VLM).

### Weaknesses
1. **Lack of Related Work Discussion**
   
   The paper does not include a comprehensive review of related work. This omission makes it difficult to contextualize the proposed method within the existing body of research and to understand how it compares to or improves upon previous approaches in machine unlearning and saliency-based methods.

2. **Insufficient Clarity in Single Layer Update Methodology**
   
   The description of the **Single Layer Unlearning Gradient (SLUG)** method lacks clarity, particularly in the selection and updating of the single targeted layer. This can lead to confusion among readers regarding the following aspects:
   
   - **Balancing Equations (7) and (8)**: The paper does not adequately explain how these equations balance the unlearning process. Additional textual explanations are needed to clarify the interplay between these equations and their role in achieving effective unlearning.
   
   - **Computation of Single Gradient Direction**: The rationale behind choosing the gradient direction based on the initial parameters is not sufficiently elaborated. More detailed explanations are necessary to justify this choice and its impact on the unlearning process.
   
   - **Consistency in Parameter Updates**: Although the authors emphasize updating parameters in a single layer, this point is not clearly reiterated in Section 3.2. Ensuring consistent emphasis throughout the methodology section would enhance understanding.

3. **Limited and Inadequate Experimental Evaluation**
   
   The experimental results presented in the paper are not particularly compelling, and the evaluation metrics used are insufficiently comprehensive. Specific issues include:
   
   - **Unlearning for CLIP (Section 4.2)**:
     - **Optimal Results Visualization**: The results for different learning rates are not clearly highlighted. Using color-coding to indicate the best-performing results would improve readability and interpretation.
     - **Evaluation Metrics Consistency**: The paper does not maintain consistency with established definitions for classification unlearning tasks, such as those outlined in "Model Sparsity Can Simplify Machine Unlearning." Aligning the evaluation metrics with these definitions would strengthen the validity of the results.
   
   - **Unlearning for Stable Diffusion (Table 2)**:
     - **Limited Performance Advantages**: Beyond demonstrating efficiency, the method does not show significant advantages in other performance metrics. This limitation raises questions about the overall effectiveness of SLUG in this context.
   
   - **Application to Vision-Language Models (VLMs)**:
     - **Lack of Reported Data**: Although the paper highlights the application of the unlearning method to VLMs and mentions corresponding evaluation metrics, it fails to report the actual data results. This absence undermines the persuasiveness of the claims regarding the method's effectiveness in VLMs.

### Questions
Based on the weaknesses part, here are some corresponding suggestions:

1. **Incorporate a Comprehensive Related Work Section**
   
  If it is available, add a dedicated Related Work section that reviews pertinent literature on machine unlearning and saliency-based methods. 

2. **Enhance Clarity in the Single Layer Update Methodology**
   
  The methodology for selecting and updating the single targeted layer is not clearly explained, potentially causing confusion among readers. Please follow the weakness part to provide more clear explanations.
   
3. **Strengthen and Expand the Experimental Evaluation**
   
  Based on the weakness part, could you provide more numerical results on VLM task, and do more experiments under previous evaluation metrcis on image classfication task.

4. **Improve Formatting and Structural Consistency**
   
  The paper's formatting, such as line spacing between titles and sections, lacks consistency, which can detract from readability and professionalism.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper proposes an innovative approach to the issue of machine unlearning, which involves removing the influence of specific data subsets from trained machine learning models without retraining from scratch.

### Strengths
1. The method introduces a novel approach to targeted unlearning by updating a single targeted layer using a one-time gradient computation, which is distinct from more common methods that require iterative model updates across multiple layers.

2. The paper presents two new metrics, layer importance and gradient alignment, to determine the optimal layer and gradient direction for unlearning, enhancing the targeted precision of the process.

3. The experiment was sufficient for me.

### Weaknesses
1. Table 2: Performance overview of different unlearning methods on UnlearnCanvas. in this table, My intuition is that there is a lack of variance experiments, that is, running multiple rounds to see the best and worst performance of the algorithm.

### Questions
1. in table2, Why are there no variance experiments to illustrate the stability of various metrics?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4