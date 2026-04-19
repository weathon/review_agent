# Controllable Data Generation via Iterative Data-Property Mutual Mappings

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 5, 1

## Abstract
Deep generative models have been widely used for their ability to generate realistic data samples in various areas, such as images, molecules, text, and speech. One major goal of data generation is controllability, namely to generate new data with desired properties. Despite growing interest in the area of controllable generation, significant challenges still remain, including 1) Disentangling desired properties with unrelated latent variables, 2) out-of-distribution property control, and 3) objective optimization for out-of-distribution property control. To address these challenges, in this paper, we propose a general framework to enhance VAE-based data generators with property controllability and disentanglement ensure. Our proposed objective can be optimized on both data seen and unseen in the training set. We propose a training procedure to train the objective in a semi-supervised manner by iteratively conducting mutual mappings between the data and properties. The proposed framework is implemented on four VAE-based controllable generators to evaluate its performance on property error, disentanglement performance, generation quality, and training time. The results indicate that our proposed framework enables more precise control over the properties of generated samples in a short training time, ensuring the disentanglement stated above and keeping the validity of the generated samples.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes VAE-based deep generative models for controllable data generation. The problem setups are defined as follows: given data $x\in\mathbb{R}^n$, $K$ different property interests $y\in \mathbb{R}^K$, and underlying mapping function $f: x \in \mathbb{R}^n \mapsto y \in \mathbb{R}^K$, the goal of this paper is to learn deep generative models $g_{\theta, \gamma}: y \in \mathbb{R}^K \mapsto x \in \mathbb{R}^n$. Specifically, this paper tackles remaining challenges of those deep generative model: 1) disentangling desired properties with unrelated latent variables, 2) out-of-distribution property control, and 3) objective optimization for out-of-distribution property control. To overcome these challenges, this paper proposes several objectives and self-training procedure of the deep generative models. The experimental results domonstrate the proposed method consistently provide better trade-off between overall quality and property for generated samples for several controllable generation tasks.

### Strengths
- This paper overally well-written and easy-to-follow. Figure 2 provide much information for understanding the proposed overall framework.

- The proposed method has a plug-in-play property for any VAE framework.

- The experimental results have shown the proposed method consistently improves various VAE frameworks.

### Weaknesses
- This paper has a severe problem for deriving the training objective of $\mathcal{L}_2$. Here,  $\mathcal{L}_2 = \mathbb{E}_q[-\log p(y|x)] = \int - q(z, w|x, y) \log p(y|x) dz dw = - \log p(Y=y|X=x)$, where $x, y \in \mathcal{X}_1, \mathcal{Y}_1 $ are real samples but not generated one. This is because $-\log p(Y=y|X=x)$ This term is then just constant since it does not depend on any parameters $\theta, \phi, \gamma$.

- For the above reason, the proposed generative model does not maximize the variational lowerbound of the joint likelihood $p(x, y)$. This does not guarantee sample generation quality of the generative models.

### Questions
- If I understand this paper correctly, this paper assumes that underlying property predictor $f: x \in \mathbb{R}^n \mapsto y \in \mathbb{R}^K$ exists. Is this correct? If so, how do we get this function? e.g., training neural network-based $f$ using given data.

### Soundness
1 poor

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper aims to address several challenges in controllable data generation. 
They propose a novel generic framework that employs a series of VAE-based models and control parameters to produce data tailored to specific needs. 
This framework can ensure the precision of both in-distribution and out-of-distribution property control and the disentanglement between the controlled properties and the unrelated latent variables. 
Through a series of experiments, the authors demonstrate the efficacy of their approach in various applications, from image synthesis to molecule generation.

### Strengths
S1. **Innovative Approach:** 
The framework enhances VAE-based generators with better property controllability and ensures superior disentanglement, offering a new perspective on data generation.

S2. **Versatility:** 
The proposed framework is shown to be applicable in different VAE models and multiple domains, from images to molecules, suggesting its broad utility.

S3. **Supporting Out-of-Distribution Property Control:**
Through designing new objective functions and optimization strategies, their framework can support seen and unseen data properties, which is particularly beneficial in scenarios where data is scarce.

### Weaknesses
W1. **Increased Complexity:**
The proposed approach, while comprehensive, seems to add multiple components and constraints to the training process. This complexity might make it difficult for practitioners to adapt and integrate the framework into existing pipelines. Furthermore, the merger of control parameters with generative models could complicate model design, training, and deployment. 

W2. **Scalability Concerns:** 
The iterative training procedure, while innovative, might raise questions about scalability and computational efficiency when applied to large-scale datasets. Some insights or experiments on how the framework scales with data size or complexity would be valuable for potential users.

W3. **Diverse Datasets Testing:**
The method's robustness and generalizability could be further validated by testing on diverse datasets. Specifically, they could employ datasets with intricate features (like human facial datasets) to further validate the claim of disentanglement between the controlled properties and the unrelated latent variables.

W4. **Limited Competitor Analysis:** 
While the authors compare their methods with existing VAE-based generation models, it would strengthen the paper if more competitors, especially advanced disentangled VAEs were incorporated into the analysis, such as:

* Joy, T., S. Schmon, P. Torr, S. Narayanaswamy, and T. Rainforth. "Capturing label characteristics in VAEs." In ICLR, 2021.
* Ren, Yurui, Ge Li, Yuanqi Chen, Thomas H. Li, and Shan Liu. "Pirenderer: Controllable portrait image generation via semantic neural rendering." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 13759-13768. 2021.
* Wang, Shiyu, Xiaojie Guo, Xuanyang Lin, Bo Pan, Yuanqi Du, Yinkai Wang, Yanfang Ye et al. "Multi-objective Deep Data Generation with Correlated Property Control." Advances in Neural Information Processing Systems 35 (2022): 28889-28901.

### Questions
Q1: This paper could benefit from a more in-depth discussion of how the method generalizes to other untested scenarios or different data domains.

Q2: Given the iterative nature of the training process, how does the method perform with large-scale datasets both in terms of computational efficiency and final output quality?

Q3: Can the authors consider testing the methods on human facial datasets like FFHQ1024 or CelebA?
* FFHQ1024: https://github.com/NVlabs/ffhq-dataset, https://github.com/DCGM/ffhq-features-dataset
* CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors have introduced a new approach to controlled data generation using a variational auto-encoder. The primary objective of this method is to generate output with semantically meaningful control over specific data properties. This is done by learning a mapping function, which maps the desired property to the associated latent vectors. Authors further ensure that the influence of a change in one property (y) is independent of others during data generation by ensuring independence in the sampled latent vector (z) and the desired property (y) of the generated data. Furthermore, they propose techniques for incorporating out-of-distribution properties into the data generation process by considering different values for the properties (y) and incorporating the generated data into the overall training process.

### Strengths
Strengths : 
1. The concept of generating output with out-of-distribution properties has a broad range of applications and can be utilized to create previously unseen data for various downstream tasks.   
2. The paper is well-written and easy to follow. It provides a thorough assessment of the limitations in prior works and demonstrates how the proposed method tackles these shortcomings.
3. The authors introduced a novel loss function to guarantee disentanglement among the desired properties.

### Weaknesses
Major Comments : 
1. The authors have proposed generating the latent vector (w) associated with a given property by utilizing both the sampled latent factor (z) and the desired property (y). However, the rationale for incorporating (z) in this mapping function remains unclear, particularly given that prior works[1] in the field have only used y. It is important for the authors to provide an explanation for why both z and y are considered in the mapping function and elucidate their impact on the overall results.

2. The authors should clarify the impact of choosing different values of y on the quality of the generated output, especially when the selected values for properties (y) deviate significantly from the original dataset.

3. It seems that the authors have incorporated out-of-distribution data in a manner that resembles various data augmentation techniques, like Mixup [2], aimed at improving performance. While they have demonstrated the potential of this method for generating new molecules, conducting similar experiments on datasets such as dsprites, 3D shapes, and other real-world datasets would provide valuable insights into the impact and advantages of this strategy for generating new data, as opposed to generating intermediate data points in interpolation.

4. The utilization of various constraints in the overall training of the model lacks clarity in the paper. The authors should provide more comprehensive details about these constraints and explain how they influence the overall results.

5. The authors should provide clear notational explanations for terms like "e," particularly in the context of L4 (loss), to enhance the reader's understanding.

6. The authors have provided information on how well their proposed method preserves the desired properties in the generated output. However, there is a lack of information regarding the impact on the quality of the generated images and molecules. It would be beneficial for readers to have a quantitative evaluation of these aspects, especially after incorporating out-of-distribution data during the training process.

7. Authors should clarify the prior distribution used for 'w' and 'z' in the overall optimization.

8.  There has been a lot of literature discussion around the impossibility of learning disentangled representation in an unsupervised manner.  The Authors have missed a seminal work in this direction [3]. This work has to be discussed and the current paper has to be positioned correctly in that context (despite the focus of this paper is not a fully unsupervised learning). 

9. There has been work on disentangled learning beyond datasets such as Dsprites. For instance, real-world datasets, that have attributs can be considered to test the proposed idea. I recommend the Authors to consider such experiments, especially given the cliam regarding the "precisely controlling the properties of generated data". 

10. There is a very closely related work that claims to do a similar type of stuff detailed in [4].

References:
[1] Guo, Xiaojie, Yuanqi Du, and Liang Zhao. "Property controllable variational autoencoder via invertible mutual dependence." International Conference on Learning Representations. 2020.
[2] Taghanaki, Saeid Asgari, et al. "Jigsaw-vae: Towards balancing features in variational autoencoders." arXiv preprint arXiv:2005.05496 (2020).
[3] Locatello F, Bauer S, Lucic M, Raetsch G, Gelly S, Schölkopf B, Bachem O. Challenging common assumptions in the unsupervised learning of disentangled representations. In international conference on machine learning 2019 May 24 (pp. 4114-4124). PMLR.
[4] Mondal, Arnab Kumar, Ajay Sailopal, Parag Singla, and Prathosh Ap. "SSDMM-VAE: variational multi-modal disentangled representation learning." Applied Intelligence 53, no. 7 (2023): 8467-8481.

### Questions
Please see the weakness section.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces a novel framework for controllable data generation that enhances Variational Autoencoder (VAE)-based generative models with property controllability and disentanglement. This framework features a shared backbone and customizable module, making it adaptable to different model assumptions. The authors extend the traditional objective function to encompass both in-distribution and out-of-distribution data, enabling the model to generate data with desired properties, even in unseen ranges.It also outlines an effective training procedure that optimizes the model by iteratively mapping data and properties, including those not encountered during training, further improving the model's controllable data generation capabilities.

### Strengths
Disentanglement within the framework appears to be somewhat effective, demonstrating the model's ability to separate and control specific properties or features in the generated data in certain instances. However, the degree to which this disentanglement consistently holds or the specific conditions under which it succeeds remain topics of interest for further investigation.

Despite the framework's potential, it is notably challenging to identify and pinpoint any consistently meaningful strengths. The versatility and reliability of the model's property controllability, disentanglement, or other capabilities may require more in-depth exploration and refinement to harness its full potential.

### Weaknesses
The significance of this work becomes apparent when considering it in the context of state-of-the-art generative models. There is a clear need for this investigation to ascertain how the proposed framework contributes to the field and whether it offers substantial improvements or unique capabilities compared to existing models.

The presentation of the work, unfortunately, poses challenges in terms of its clarity and comprehensibility. It is evident that the way the research findings and methodology are conveyed may need refinement to enhance accessibility and ease of understanding for readers.

The unnecessary introduction of mathematical operators, which may not directly enhance the quality of the contribution, can potentially obscure the core concepts and findings of the research. Simplifying and streamlining the presentation might be beneficial.

A notable concern is the utilization of extremely low-quality datasets with limited relevance. This choice could potentially hinder the model's effectiveness and real-world applicability. The inclusion of more representative and high-quality datasets may be crucial to improve the robustness and practicality of the framework.

### Questions
Drawing meaningful comparisons between the framework presented in this work and Mathieu et al.'s "Disentangling Disentanglement in Variational Autoencoders" is a promising avenue for research. This exploration could shed light on how the proposed framework either builds upon or differs from the prior work, particularly in terms of disentangling capabilities within Variational Autoencoders.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor
