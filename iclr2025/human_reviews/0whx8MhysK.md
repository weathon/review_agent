## Human Reviewer 1

### Summary
This paper proposed a guided diffusion generation method for dataset distillation problem. The trajectory influence and deviation guidance are introduced to the vanilla diffusion process for generating synthetic samples as efficient training data. The results on ImageNet and its subsets demonstrates the improvements over the baselines.

### Strengths
Strengths:

1.	The paper is easy to follow. The motivation of importance guided synthesis is clear and method is well presented. 

2.	The new idea is reasonable and neat that bridge the diffusion-based generative models and importance-based sample selection.

3.	The results are promising. The performance improvements on challenging datasets are remarkable. Extensive ablation studies, cross-architecture validation and visualization are implemented.

### Weaknesses
Weaknesses:

1.	There are several hyper-parameters in the algorithm, such as influence factor, deviation factor, scale, guided range etc. The sensitiveness of performance on these hyper-parameters should be studied. How do authors search the hyper-parameters?

2.	I have a concern whether the new method cause the collapse of the distribution of the generated data, though with the deviation guidance. 

3.	Since DiT and VAE pre-trained on huge dataset are utilized for generating training samples on small datasets. It naturally brings advantages over traditional dataset distillation methods. Hence, more recent methods that also use pre-trained diffusion models should be compared with. 

4.	There are also some similar works that improve the efficiency of diffusion-model generated training samples, such as [1], which should also be discussed in the paper.

[1] Real-Fake: Effective Training Data Synthesis Through Distribution Matching, ICLR 2024.

### Questions
Please address the above weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper works on dataset distillation by generating the distilled dataset using diffusion models guided by an influence function. In the implementation, two guidance terms are used. One is to increase the similarity between the gradient using the generated sample and the average gradient using the original training samples. The other is to decrease the similarity between generated samples. Experiments are conducted on ImageNette and ImageWoof.

### Strengths
1. The idea of using guided diffusion to generate samples for distillation is an interesting application of diffusion models.

2. The proposed two guidance terms, i.e., increasing the gradient similarity and sample diversity, are well-motivated, simple, and intuitive. 

3. The paper is well written and presented in general.

4. In the experiments, the proposed method achieves better performance and shows effectiveness. Comprehensive ablation studies and analyses are provided.

### Weaknesses
1. There are a lot of writing issues in the math part:
- It is unclear how the derivation is transferred from stepwise (Eq4) to epochwise (Eq5).
- C duplicated defined on L096 and L110.
- In Sec 2.2, some z is bold, and some are not.
- In L132, D is not clearly defined.
- In Eq5, theta_e and theta_E are not clearly defined.

2. The proposed method seems to have a high computational cost. Both computing and storage costs are high for the gradient calculation in L294. The similarity calculation with respect to all generated samples is also high in Eq8.  The computing and storage costs should be clearly provided and analysed in all the experiment sections.

3. I doubt the statement this method is training-free. I agree that it is training-free as commonly understood in the diffusion community. But there are still a lot of training efforts here. It is only training-free given all the checkpoints, stored gradients, and pre-trained diffusion models. I would suggest revising this statement.

### Questions
Please refer to W1 and W2.

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
This paper proposes Influence-Guided Diffusion (IGD) for dataset distillation. IGD solves the problem of poor performance and high resource costs of existing methods at high resolution. IGD proposes a training-free sampling framework which can be used in pretrained diffusion models to generate training-effective data. Extensive experiments show IGD achieves state-of-the-art performance in distilling ImageNet datasets.

### Strengths
1. IGD is a training-free framework that can be easily used in any pretrained diffusion models.
2. The target this paper hopes to solve is clear, and the proposed methods solve the problem of data influence and diversity constraint theoretically.
3. The performance improvement of IGD used in DiT and Minimax finetuned DiT is obvious.
4. The ablation study is adequate, including all proposed methods and hyperparameters.

### Weaknesses
1. In Table.1, the compared methods are missing like latest method RDED [1] mentioned in section 4.1.
2. Although the proposed method IGD is training-free for diffusion models. It requires training a model to collect the surrogate checkpoints used in Eq. 7. The time consumption should be listed as the paper emphasizes efficiency.
3. The model used in Eq.7 is ConvNet-6. If we change the model like for a bigger one Swin Transformer, will the performance better? Or this model choice is relatively insensitive?
4. Is IGD can be used in other efficient diffusion sampling strategy like DPM [2] solvers?
5. The generation time should be compared between IGD and other methods like current SOTA RDED [1].

Reference:

[1]. Sun P, Shi B, Yu D, et al. On the diversity and realism of distilled dataset: An efficient dataset distillation paradigm[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 9390-9399.

[2]. Lu C, Zhou Y, Bao F, et al. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps[J]. Advances in Neural Information Processing Systems, 2022, 35: 5775-5787.

### Questions
Please see weeknesses.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
The work proposes a guidance scheme for dataset distillation with two main contributions. The first is to do gradient matching between sampled data with the training data, and the second is to add diversity constraints among samples inside a class.The experimental results show clear improvement over other baselines

### Strengths
1. The motivation is reasonable
2. The paper is well written
3. The performance is significant

### Weaknesses
1. The performance is still far from full dataset.
2. Lack of diversity measurement experiments
3. The design of equation (7) lacks clarification.
4. The application of the work seems to be not flexible. From my understanding, according to one architecture, there will be a need for one-time distillation. Is it possible to have one time distillation and use that distilled datasets to validate across models? I can see Table 4 for the robustness between models, yet the performance is not the same for the used models for guidance. This results in the concern in the application in reality due to computational exhibitions.

### Questions
1. The performance is still very far from the original datasets. What is the least IPC to achieve similar performance with full data?
2. Which experiments show an improvement in diversity? The diversity should be measured in terms of FID/Recall values.
3. The equation (7) uses cosine similarity instead of product; is it purely due to experimental results or based on some other hypothesis?
4. How will the work be performed on different tasks apart from classification?

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

## Human Reviewer 5

### Summary
This paper addresses the challenges of dataset distillation, which aims to create compact yet effective datasets for training larger original datasets. Existing methods often face limitations when dealing with large, high-resolution datasets due to high resource costs and suboptimal performance, largely due to sample-wise optimizations in the pixel space. To overcome these challenges, the authors propose framing dataset distillation as a controlled diffusion generation task, leveraging the capabilities of diffusion generative models to learn target dataset distributions and generate high-quality data tailored for training.

The authors introduce the Influence-Guided Diffusion (IGD) sampling framework, which generates training-effective data without retraining the diffusion models. This is achieved by establishing a connection between the goal of dataset distillation and the trajectory influence function, using this function as an indicator to guide the diffusion process toward promoting data influence and enhancing diversity. The proposed IGD method is shown to significantly improve the training performance of distilled datasets and achieves state-of-the-art results in distilling ImageNet datasets. Notably, the method reaches an impressive performance of 60.3% on ImageNet-1K with IPC (Images Per Class) set to 50.

### Strengths
1. The paper is easy to read and understand.
2. IGD appears to be superior to existing diffusion model-based approaches.

### Weaknesses
1. In the introduction, the authors introduce the concept of Influence-Guided without clearly explaining what "Influence" entails or why it is used for guidance. The motivation is not well established. While Figure 1 effectively shows performance, adding an additional subfigure to illustrate the motivation or highlight differences from previous methods might be more valuable.

2. The primary contribution of the authors is the proposal of a train-free diffusion framework for Dataset Distillation. While train-free approaches are common in the AIGC field, how does the proposed method differ from existing ones?

3. The experiments only report results on ImageNet, without including results on classic datasets such as CIFAR-10 and CIFAR-100.

### Questions
As shown in Table 5, the main contributions of the authors include the proposed influence guidance and deviation guidance. What is the relationship between these contributions and the "train-free" concept? Notably, even when these components are excluded from Equation 9, good results are still achieved.

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