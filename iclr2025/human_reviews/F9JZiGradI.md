## Human Reviewer 1

### Summary
This paper proposes a method, KAN-MLP, for improving the performance of ViT on function learning and CIFAR classification tasks. In order to achieve that, they replaced the traditional MLP layers within the transformer architecture. In this method, experts are dynamically selected based on the input through a gating mechanism, ensuring efficient routing of tokens to the most relevant experts. Their main contribution is applying an explainable KAN architecture to the Vision Transformer model. Comparing KAN, MLP, and MLP-KAN on function learning tasks, they show that MLP-KAN performs better than the other architectures in certain cases. Additionally, they also perform an ablation study on the CIFAR dataset to demonstrate that their extended model outperforms the naive ViT.

### Strengths
1. The paper proposes KAN-MLP, a novel approach to enhance Vision Transformers (ViT) for function learning and CIFAR classification tasks.

2. It introduces the explainable KAN architecture into the ViT model, which could improve interoperability.

3. The paper compares KAN, MLP, and MLP-KAN in function learning, showing potential advantages of MLP-KAN over other architectures.

4. An ablation study on CIFAR demonstrates that the proposed model improves on the plain ViT model, highlighting the effectiveness of the enhancements.

### Weaknesses
1. The scaling law for MLP-KAN is missing, making it difficult to assess if MLP-KAN overcomes KAN’s limitations.

2. The method may still suffer from the curse of dimensionality (COD), and the paper does not address how MLP-KAN performs for learning non-smooth and high-dimensional functions.

3. There is a potential conflict between KAN’s suitability for low-dimensional tasks and ViT’s insuitability for small datasets. The experiments are limited to smooth functions and CIFAR datasets. Testing on larger datasets like ImageNet or MSCOCO would provide a more comprehensive view of the model’s performance relative to the Vision Transformer baseline.

### Questions
This paper contributes to advancing the KAN model, though it requires some clarifications on theory and experiments. Given these clarifications, if provided in an author's response, I would consider increasing the score. 

For the theory, there are a few steps that need clarification and further clarification on novelty. 

1. KAN has advantages in model scaling under certain constraint conditions but not for all learning tasks. The KAN model obeys the Error Scaling formula $\\| f - (KAN) \\|\_{C^m} \le C G^{-k-1+m}$ and the scaling law $l \\propto N^{-\\alpha}$, but the author did not clarify the relationship between grid size $G$ and the input space dimension $n$. As a function approximation method, usually, the number of grid points $G$ is directly proportional to $G \propto I^n$ ($I+1$ is the number of intervals on each dimension), which makes $\\| f - (KAN) \\|_{C^m} \le C G^{(-k-1+m)/n}$ and triggers a serious curse of dimensionality (COD) problem (see KAN [1], Fig. 3.1, $f(x_1, \cdots, x\_{100}) = \text{exp}(\frac{1}{100} \sum\_{i=1}^{100} \sin^2{\frac{\pi x_i}{2}})$). To address this problem, KAN authors hypothesize that the objective high-dimensional function is smooth and has sparse compositional structure to reduce the number of grid nodes $G \ll I^n$. The authors did not provide the scaling law of MLP-KAN in the paper. Therefore, I don't know whether MLP-KAN overcomes the inherent limitations of KAN.

2. This is particularly called into question due to the integration of KAN and ViT, since KAN and ViT usually exhibit different behavior on datasets of varying sizes. At present, KAN is suitable for low-dimensional function learning, and the dataset is generally small, with only a few thousand samples. However, ViTs are particularly powerful on large datasets (e.g., ImageNet) and tend to underperform relative to convolutional models on small datasets. Empirically, KAN and Transformer have potential conflicts.

For the experiments, the following should be addressed.

1. It would have been better also to show the performance of learning the **non-smooth** or **high-dimensional functions**. The Feynman Equations may be too simple for conventional function approximation methods. You can try $f(x)=\frac{1}{x} \\ \sin{\frac{1}{x}}$ and $f(x_1, \cdots, x_{100}) = \sum_{i=1}^{99} \sin{(x_i + x\_{100-i})}$; testing these functions would indicate if MLP-KAN overcomes certain limitations of KAN.

2. In our testing, we found that KAN's training process differs from neural network models and is considerably slower. Comparing wall-clock training time could reveal any potential efficiency advantages.

3. The central contribution focuses on enhancing Vision Transformer performance on CIFAR. It would be beneficial to compare with the Vision Transformer baseline on larger datasets like ImageNet-1K, which would add value.


---

[1] Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., ... & Tegmark, M. (2024). Kan: Kolmogorov-arnold networks. arXiv preprint arXiv:2404.19756.

### Soundness
4

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper proposes MLP-KAN, a mixture of experts method, to combine MLP and KAN. The authors claim their method can address the shortcomings of both KAN and MLP in one structure and solve the scalability problem of MLPs. Furthermore, they show the superior performance of their method across different tasks and datasets compared to other methods and investigate the effectiveness of various components of their method.

### Strengths
* The paper is well-written for the most part.
* The motivation of the paper is valid and exciting. 
* The idea is simple but yet effective.

### Weaknesses
* The discussion part of the paper heavily relies on the description of Multi-expert and router gating, which is not the paper's contribution. More discussion or experiments are needed to show why combining MLP and KAN should improve the performance. 
* One of the paper's main points is scalability, but experiments about scalability, computation time, and memory are missing. 

Minor Weakness:

* The numbers in all Tables don't have confidence intervals, so it is hard to grasp the significance of the differences.  The authors should include confidence intervals or standard deviations from multiple runs.
* Details about competitors are missing in the experimental setup.

### Questions
* Is the number of tasks connected to optimal k for topk?
* Is the number of experts for MLP and parameters the same across experiments with MLP-KAN?
* Are all experiments for Tables 2 and 3 trained together as a multi-task scenario? Or are experiments for Tables 2 and 3 separated?

Suggestion:

* It would be great to add more details about experiments and summaries in section 4.1.  
* I think it improves the justifiability of the paper if they provide an ablation on the router gating to show how it assigns tokens to each expert.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
This manuscript introduces MLP-KAN, a unified block that combines representation and function learning within a single framework. By using MLPs for representation learning and KANs for function learning in a mixture-of-experts setup, MLP-KAN adapts to various tasks and data modalities. When integrated into a transformer-based architecture, MLP-KAN demonstrates both versatility and robustness across multiple data domains. The proposed method is evaluated on four datasets, CIFAR-10, CIFAR-100, mini-ImageNet, and SST2, showing strong performance in both image classification and natural language processing tasks.

############################ Post Rebuttal ############################

All of my concerns have been addressed during rebuttal. I am happy to raise my score from 6 to 8.

############################ Post Rebuttal ############################

### Strengths
1. The manuscript is well-written, presenting clear motivations and providing step-by-step derivations of the proposed method.

2. Combining MLPs and KANs within an MoE framework is interesting. Moreover, integrating this block into a transformer architecture develops a robust backbone that effectively extracts and integrates features across various data modalities.

3. The ablation studies demonstrate that the proposed method can scale easily by increasing the number of experts, which enhances performance without introducing too much computational burdens.

4. The proposed method is extensively evaluated on multiple CV and NLP datasets, demonstrating its versatility for diverse AI applications and practical values.

### Weaknesses
1. Although the proposed technique is shown to be generalizable to different tasks, its effectiveness in other types of tasks or with different types of data (e.g., time-series, reinforcement learning) remains unexplored.

2. Using multiple experts in an MoE architecture, especially with higher Top-K values, can significantly increase computational resource requirements. I suggest that the authors conduct ablation studies on runtime complexity and compare their proposed method with the standard transformer architecture.

### Questions
1. How does MLP-KAN perform in the presence of noisy or adversarial inputs compared to other models? Are there any robustness benchmarks included in the evaluation?

2. What optimization algorithms and strategies were employed to train MLP-KAN effectively? Were any specific techniques used to balance the training of multiple experts?

3. Were all models, including baselines, trained under the same conditions to ensure fair comparisons? What datasets splits, augmentation techniques, and training epochs were used?

4. How sensitive is MLP-KAN to changes in hyperparameters other than the number of experts and Top-K values? For example, how do variations in learning rates, network depth, or activation functions affect performance?

5. Were there any challenges related to training stability when combining MLPs and KANs within the MoE framework? How were these challenges addressed?

6. How does the inclusion of MLP-KAN affect the standard attention mechanisms within transformers? Are there any changes to how attention weights are applied?

7. How effective is MLP-KAN in transfer learning where it is fine-tuned on different tasks after initial source pre-training?

8. What is the computational cost of MLP-KAN compared to other architecture with MLPs (e.g., transfomer) or KANs alone? How does the addition of multiple experts affect training and inference times?

9. How interpretable are the latent features generated by MLP-KAN? Are there any visualizations demonstrating the semantic captured by the model (e.g., t-SNE visualizations)?

10. What future research directions do the authors suggest to address the current limitations or to further enhance the capabilities of MLP-KAN?

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
8

### Confidence
4

---

## Human Reviewer 4

### Summary
The authors hypothesize that KAN networks and MLPs are effective for solving different types of problems: specifically, MLPs are good for representation learning, while KANs are good for function learning.  They propose a modeling strategy that includes both MLPs and KANs, which are adaptively selected based upon the problem setting.

### Strengths
1) The premise of the paper is potentially reasonable; there is evidence in the literature that KANs and MLPs are complementary, and developing a method that adaptively selects which modeling strategy is best for a given problem is a good idea.

### Weaknesses
1) The presentation of the paper is not sufficiently clear to facilitate review.   The paper is generally vague, contains many typos, and the methodology is ultimately unclear.  I provide examples

(ii) Poor presentation.  In Line 47 in the 2nd paragraph of the paper, the authors define KAN as Kernel Attention Network, yet the whole paper seems to be about Kolmogorov-Arnold Network.  Line 51 and 73 in the introduction are nearly identical, and repeat the same idea. 

(ii) Incorrect/unclear methodological description.  For example, in Eq. (5) the dimensions of W and X are incompatible.  The entire architecture of the proposed model is unclear.  For example, it is unclear whether there are multiple MLP-KAN layers?  The authors have a section about "Architecture" which then, without motivation, discusses self attention.   The number of layers in the MLP-KAN are never provided (although it is unclear if there are multiple layers), nor is the overall size of the model discussed. 

(iii) Vague motivation.  The whole premise of the paper is not clearly explained.  While the idea of combining KANs and MLPs seems reasonable, the authors repeatedly argue a theoretical motive based upon MLPs being "representation learning" methods while KANs are "function learning".  This premise for the proposed approach is repeatedly mentioned throughout the paper, yet the difference between these two approaches is never precisely described.  
 
2) Insufficient Experiments.  The experiments are insufficient to demonstrate the efficacy of the proposed approach.  To the best I can discern, the proposed method would significantly increase the number of modeling parameters because we now have multiple KANs and MLPs in a single model, along with some parameters to select among them.  However, the resulting model often performs similarly to other models only composed of a single KAN or MLP architecture (e.g., Table 3).  What would happen if we simply used a single MLP or single KAN model that is the same size (in terms of free parameters) to the MLP-KAN?  Or what if we made a simple fusion model where we interlaced MLP and KAN layers, or added a few KAN layers to the end of a standard MLP?  How do we know whether this more complex architecture proposed by the authors is superior to simpler and/or smaller models?

### Questions
I think the paper is insufficiently clear to support proper review, and therefore I don't have any questions.

### Soundness
1

### Presentation
1

### Contribution
2

### Rating
1

### Confidence
4