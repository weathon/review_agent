# Reciprocal Label Diffusion for Learning with Noisy Labels

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Deep neural networks are susceptible to overfitting noisy labels, resulting in poor generalization. We propose Reciprocal Label Diffusion (RLD), a novel framework that leverages a mutual guidance mechanism between a label diffusion model and a prediction model to effectively learn from noisy labels. In RLD, the diffusion model is guided by the outputs of the prediction model to denoise corrupted labels through a forward and reverse diffusion process in the logit space, thus modeling and correcting label noise with standard diffusion distributions while enforcing instance-dependency. In turn, the prediction model is refined using the denoised labels produced by the diffusion model, enhancing its learning of accurate representations. This reciprocal interaction enables both models to iteratively enhance each other. To further improve robustness to label noise, we incorporate a contrastive denoising loss that enforces consistency across different data augmentations. Experimental results on benchmark datasets demonstrate that our approach outperforms state-of-the-art methods, achieving significant improvements in classification accuracy under various noise conditions. Our framework provides a robust solution for learning with noisy labels by exploiting the reciprocal interplay between diffusion and prediction models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The work introduces Reciprocal Label Diffusion, which uses universal guidance based label diffusion method that operates over the logit space of labels. The guidance to this space is provided by the prediction model's de-noised outputs. A contrastive objective between the mean predictions from the denoising network. Additional CE loss between the de-noised predictions and the networks outputs is introduced. The method does well in comparisons and ablations lay credence to the components introduced in the method's objective.

### Strengths
The method is simple and well performant. Ablations suggest that the components of the loss function are all significantly impactful in the model's performance.

### Weaknesses
1. Some of the known methods in this area [1, 2] aren't evaluated.
2. The work is lacking some critical implementation details such as the noise schedules and number of steps of the diffusion process.
3. There could be some theoretical evaluation of the convergence of the method given a noise level.




[1] Han, Xizewen, Huangjie Zheng, and Mingyuan Zhou. "Card: Classification and regression diffusion models." Advances in Neural Information Processing Systems 35 (2022): 18100-18115.

[2] Chen, Jian, et al. "Label-retrieval-augmented diffusion models for learning from noisy labels." Advances in Neural Information Processing Systems 36 (2023): 66499-66517.

### Questions
1. The work may benefit from a T-SNE plot of the denoised and noisy labels to emphasize that the diffusion model learns a useful conditional structure.
2. Is the inference efficient? It may be worth including an inference efficiency analysis and comparing against the benchmarks.
3. Is the model training susceptible to instability if the guidance is received from a noisy prediction model?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper discusses learning with noisy labels for the problem of robust deep learning under instance-dependent label noise (IDN). The paper (1) claims that existing methods fail to model the label corruption process itself, limiting their effectiveness in disentangling complex noise patterns,  and (2) proposes a mutual guidance mechanism between a diffusion model and a prediction model to iteratively denoise corrupted labels and refine model predictions. Experimental results on five datasets show its promising performance against baseline methods. The paper is well-structured and clearly presented on its novelty and contribution highlights. The main contributions of this paper are proposing Reciprocal Label Diffusion (RLD), a new deep learning framework that integrates diffusion models with prediction networks to address label noise.

### Strengths
S1.  The paper presents an innovative combination of diffusion models and classification learning for noisy label correction by iteratively refining each other in the logit space.

S2: The paper introduces a contrastive denoising loss to enforce consistency across data augmentations, enhancing robustness.

### Weaknesses
W1. The convergence and stability of the reciprocal learning loop are unclear, and label correction lacks interpretability. For example, if the prediction model produces biased logits due to noisy supervision, the diffusion model guided by these logits may incorrectly “denoise” correct labels, thereby generating pseudo-labels that deviate from the true distribution. When these erroneous pseudo-labels are subsequently used to retrain the prediction model, both networks reinforce each other’s mistakes. 

W2. In the forward diffusion process, the paper only adds Gaussian noise. This approach assumes that it captures label noise characteristics, which may not hold for complex, real-world, structured, or non-Gaussian noise distributions.

W3  The paper introduces dual model updates with multiple iterative passes through the diffusion process, which might have high computational complexity. The paper needs to add computational complexity analysis of RLD.

### Questions
See "weaknesses" above,

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
4

### Summary
This paper introduces Reciprocal Label Diffusion (RLD), a new way to handle noisy labels in learning. It creates a two-way guidance system between a label diffusion model and a prediction model. The main idea is to handle noisy labels and fix them in the logit space using forward and reverse diffusion processes. The diffusion model uses outputs from the prediction model to clean up noisy labels while considering each instance. In return, the prediction model improves by using the cleaned labels from the diffusion model. This back-and-forth process helps both models get better over time. Also, a contrastive denoising loss ensures consistency across different data versions to make the system more robust. Tests on datasets like CIFAR10/100-IDN, Animal-10N, Food-101N, and Red Mini-ImageNet show top performance, with improvements of 1.3-2.6% over the best existing method (SSR) in various noisy conditions.

### Strengths
The paper introduces a new way to handle noisy labels using diffusion models in the logit space. This method is different from traditional ways like picking samples or adjusting weights. It works well because it keeps label distributions within the probability limits. The paper uses a mutual guidance framework where the diffusion model cleans labels, and the prediction model helps guide this process. This creates a helpful feedback loop that fixes problems with older methods. Tests show this method works better than current top methods on several datasets like CIFAR10/100-IDN, Animal-10N, Food-101N, and Red Mini-ImageNet, with improvements of 1.3-2.6% over the next best method at different noise levels. The paper includes many tests with different types of noise and models, proving the method is strong and can be used widely. The guidance mechanism and contrastive loss make sure the label cleaning process depends on the instance, which is important when noise changes with instance features. The analysis shows that each part of the method (contrastive loss, classification loss, guidance mechanism) is important, with performance dropping 1-5% when parts are removed. The framework is well-organized with clear math explanations, and Figure 1 shows how the two models learn from each other.

### Weaknesses
The paper does not explain why the reciprocal guidance mechanism works better or when it is better than non-reciprocal methods. 

The method needs both a diffusion model and a prediction model, with many reverse steps during inference. The paper does not compare runtime, memory needs, or computational costs with other methods, which is important to know if it can be used practically. Diffusion models usually need many denoising steps, so the inference cost might be higher than other methods. 

The paper talks about using logit space to handle probability constraints but does not compare it with other ways. For example, diffusion could be done directly in probability space with transformations like the softmax-inverse transform or in learned embedding spaces. Without studies comparing these options, it is unclear if logit space is the best choice or just convenient. 

The paper also misses recent works on instance-dependent noisy label learning that show good results on IDN benchmarks, it is hard to judge RLD's contribution and see if the improvements are worth the extra complexity of the diffusion-based approach.. Important missing works include: 
"Instance-dependent noisy label learning via graphical modelling," 
"Confidence scores make instance-dependent label-noise learning possible," 
"Clusterability as an alternative to anchor points when learning with noisy labels."
"Instance-Dependent Noisy-Label Learning with Graphical Model Based Noise-Rate Estimation"
Some reuslts are better in these papers.

### Questions
Can you show if the reciprocal learning process always works? When might the switching between diffusion and prediction models not work well? 
How does RLD compare in terms of training time, inference time, and memory to methods like SSR and DivideMix? 
Have you tried doing diffusion in other spaces, like probability or embedding spaces? What are the pros and cons, and why is logit space best for this task? 
How does the method react to the time-dependent scaling factor s(t) in Equation 5? Have you tried different ways to set this, and how should it be set for new uses? What guides the choice of warm-up epochs (10 vs 30)? Is there a way to decide the right warm-up time based on the dataset or early training results? 
How does RLD work with class-imbalanced noisy labels, which are common in real life? Does it need changes to handle big class imbalance with label noise? 
Why is universal guidance better than classifier guidance or no guidance for this task? Can you show studies comparing these guidance methods in label denoising? Do traditional noisy label methods estimate the noise transition matrix? Does RLD learn this through diffusion? Can the diffusion model help understand noise patterns in the data?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel framework called Reciprocal Label Diffusion (RLD) to address the problem of learning noisy labels. RLD performs label denoising in the logit space through mutual guidance between the label diffusion model and the prediction model, and combines contrastive learning to enhance robustness. This method demonstrates superior performance compared to existing methods on multiple benchmark datasets, particularly showing a significant advantage in handling instance-dependent noise (IDN).

### Strengths
The core contribution of this work lies in proposing a novel and systematic framework—Reciprocal Label Diffusion (RLD). Its innovation is significantly reflected in being the first to introduce diffusion models into the field of label noise learning and constructing a co-evolutionary reciprocal learning system. The method is systematic in its technical implementation: by performing forward and reverse diffusion in the Logit space of the feedforward model, it cleverly avoids the probability constraints of the label space and utilizes a general guidance mechanism based on the predictive model to achieve modeling and correction of instance-dependent noise. At the same time, the introduced contrastive denoising loss further enhances semantic consistency constraints for data augmentation. The experimental validation of this study is thorough and convincing, demonstrating significant and consistent performance improvements over existing mainstream methods on multiple benchmark datasets, including the CIFAR series, Animal-10N, and Food-101N. The advantages are particularly evident in high-proportion instance-dependent noise scenarios, and the effectiveness of each core component is confirmed through extensive ablation experiments. The overall writing is logically rigorous and clearly articulated, providing a solid and inspiring solution to the challenging problem of noisy label learning.

### Weaknesses
The core weakness of this paper lies in its insufficient theoretical motivation, failing to fully demonstrate the necessity of the diffusion model compared to simpler label correction methods; the proposed reciprocal learning framework lacks stability guarantees, posing a risk of error accumulation and training collapse; at the same time, the experimental section fails to provide mechanistic evidence to prove its essential ability to effectively handle instance-dependent noise, and completely ignores the huge computational overhead brought by the method, casting doubt on its practical application value.

### Questions
1. The paper fails to clearly articulate the fundamental rationale for choosing diffusion models to address the label noise problem. This raises a critical question: Is RLD an elegant solution that genuinely leverages the intrinsic properties of diffusion models, or is it an engineering implementation that forcibly combines two popular concepts (diffusion models and noisy labels)? The justification for its necessity (Why Diffusion?) is insufficient.
2. The core of RLD is a dynamically coupled system in which two modules (the predictive model and the diffusion model) are interdependent and trained mutually. This design suffers from a fundamental circular dependency issue: if one module fails during early training, it may drag down the other module through the guidance signal, leading to system collapse or convergence to suboptimal solutions.
3. The paper claims that RLD effectively handles instance-dependent noise (IDN), but the experimental section lacks direct and compelling validation of this core assertion. The existing experiments (such as accuracy under different IDN ratios) are outcome-based rather than mechanism-based.
4. The introduction of diffusion models inevitably incurs significant computational overhead, which is a drawback when evaluating the practical application value of this method. It would be beneficial to add a table or section in the experimental part comparing the differences between RLD and major baseline methods (e.g., DivideMix, SSR) in terms of training time (GPU hours), inference time, and memory usage.
5. The publication years of the comparative methods are relatively old. It is recommended to incorporate the latest research findings from 2023–2025 to further demonstrate the advanced nature and effectiveness of the proposed method.

### Soundness
2

### Presentation
2

### Contribution
2
