# Do Neural Networks Learn Similar Subspaces? An Empirical Exploration of Joint Parametric Subspaces in Deep Neural Networks

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
We show that deep neural networks trained across diverse tasks exhibit remarkably similar low-dimensional parametric subspaces. We provide the first large-scale empirical evidence that demonstrates that neural networks systematically converge to shared spectral subspaces regardless of initialization, task, or domain. Through mode-wise spectral analysis of over 1100 models - including 500 Mistral-7B LoRAs, 500 Vision Transformers, and 50 LLaMA-8B models - we identify universal subspaces capturing majority variance in just a few principal directions. By applying spectral decomposition techniques to the weight matrices of various architectures trained on a wide range of tasks and datasets, we identify sparse, joint subspaces that are consistently exploited, within shared architectures across diverse tasks and datasets. Our findings offer new insights into the intrinsic organization of information within deep networks and raise important questions about the possibility of discovering these universal subspaces without the need for extensive data and computational resources. Furthermore, this inherent structure has significant implications for model reusability, multi-task learning, model merging, and the development of training and inference-efficient algorithms, potentially reducing the carbon footprint of large-scale neural models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This submission presents empirical evidence that the training of neural networks generates low-dimensional parametric subspaces, as identified through spectral decomposition, which the authors claim to be universal. These subspaces are able to capture the majority of the variance of a collection of neural network models drawn from the same architecture into a few principal directions.

Potential downstream applications of these findings are the compression of neural networks into a smaller subset of coefficients of the found principal directions, allowing to significantly save memory and compute. 

Experimental results show evidence of these subspaces in ResNets training on multiple computer vision datasets,  LoRA adapters from Mistral-7B-Instruct-v0.2, and LoRA adapters from Stable Diffusion SDXL models. Finally, the last set of experiments was run on ViT, GPT-2, LLaMa-8B, and Flan T5 models. In all experiments, the reported subspaces emerge given the spectral decomposition method introduced by the authors.

I think this is a fascinating paper and very much appreciate this work as it might indeed change the way we look at neural networks and the structures they represent after training. I would have some questions, which I outline in more detail below.

### Strengths
- **(S1)**: Given that neural network models are becoming larger and more models are being made publicly available, finding such parametric subspaces is not only timely but can also provide novel directions for model compression, potentially leading to less compute needed in the future.  

- **(S2)**: I appreciate the paper's motivation and theoretical formulations. It provides an interesting connection to the empirical results.

- **(S3)**: This work provides a large-scale set of experiments investigating if subspaces form in collections of very different neural network models composed of different architectures, datasets being trained with, losses and tasks being optimised for, and modalities. This is truly helping to understand empirically how universal such subspaces are. I also appreciate the details and additional results listed in the appendix.

- **(S4)**: This paper is easy to read and follow, given its systematic build-up.

### Weaknesses
- **(W1)**: One of the stronger limitations has already been outlined by the authors. The proposed spectral decomposition can only be done on models from the same neural network architecture. It would be great if one could bridge the found subspaces between architectures to be truly universal. Therefore, I would also be careful in using the term "universal" or, said the other way around, properly defining what the term means in the context of this work.

### Questions
I would have some questions:

- **(Q1)**: How does this work compare to WeightWatcher [1] using spectral decomposition to find similar structures in NN weight matrices, and follow-up work learning lower-dimension manifolds of populations of neural networks [2]. The submitted work appears to be quite similar to [1] and has a connection to [2], but is missing in the comparison of methods paragraph (line 71-81).

[1] Martin & Mahoney, Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data, Nature Communications, 2021
[2] Schuerholt et al, Towards Scalable and Versatile Weight Space Learning, ICML 2024


- **(Q2)**: How do you perform classification in subspace? Results from Fig. 2 (a), Table 2, and Table 4 show accuracy results of the corresponding models being in their universal subspaces. How does inference work in these subspaces? Or do you project the original model into the found subspace and back by reconstructing it only from the first major (principal) directions? I found it briefly mentioned in the appendix. Can you provide more details on this?

- **(Q3)**: I have a similar question for the "model compression" example of 500 ViT models (line 381-388). Can you outline why you understand this as "merging"? Could you provide more information about this? Also, what exactly do you project into the found subspace? You mention that the task variable first and last layers are ignored. Can you outline what you mean by this?

- **(Q4)**: Same question for Section 3.2.2. The idea is to learn only their task-specific coefficients for the new task. Can you provide details on how you apply gradient descent in the subspace to learn the task-specific coefficients?

- **(Q5)**: Last question, have you compared the coefficient of the original models to their sparsified counterparts (for example, for magnitude pruning or variational dropout sparsification). Basically, my question is, do you think the subspace would change significantly or would the coefficients of the sparsified model change as compared to the original models?

As I already said, I think this is fascinating and I would really appreciate more details about the downstream tasks and how they are performed. I would be happy to increase my rating once I fully understand the interaction between the weight space and universal subspace.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper suggests that weights of the most of the available pretrained deep learning model have a joint universal subspace. The authors claim that thanks to this fact, huge memory/computation gains can be achieved. In the paper, there are emprical results that show most of the explained variance belong to few directions, which is common across models architectures and tasks. Authors also provide some mathematical proofs to strengthen their claims.

### Strengths
One good thing regarding this paper is that they conducted lots of experiments with variety of models and tasks. Also, they made a theoretical analysis.

### Weaknesses
I think the language and explanation is weak and some parts are confusing regardless of the content. When it comes to the content, first of all, the claim that the weights of pretrained models are having low rank structure is not new. There are several works that claim and made analysis on such low rankness, even if your claim is more broader which is universal low rank structure. Moreover, in Table 4, it is the universal model is compared with the full training. Which is I believe not fair. If you have pretrained weights in universal model, it would be fair to compare it with another finetuning on pretrained model, for example just finetuning last layer etc. Such a table gives impression like you trained a 86M model from scratch  and you 10K model from scratch and your model gives almost the same performance, however the universal ViT has a strong pre-knowledge.

### Questions
1 - To be honest, I did not quite understand the core point of the project. How are you getting the Figure 1 ? So far, it seems that you plot the explained variances for each principan components, and provide a bar plot. However, this does not mean that these weight matrices are having a common subspace. The directions of the first eigenvalues can be different. Probably, I miss an important point here.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper claims to show empirical evidence for the hypothesis that the parameters of DNNs with the same architecture trained on different datasets and/or trained with different hyperparameters and initializations, lie in a low-dimensional linear subspace. It does so by doing a form of singular-value decomposition on the covariance matrices of the concatenated weights of the networks. Estimation is done by means of a "sample" of trained DNNs, already publicly available (on Huggingface). The main experimental evidence is two-fold:
* it is shown that a large part of the explained variance in the parametric spaces is located in a limited number of principal component directions.
* it is shown that after projection onto this low-dimensional subspace of a limited number of principal components, the performance of the DNNs remains (largely) intact.

### Strengths
The focus on (effective) linear subspaces of the parameter space of DNNs is an interesting direction of research. This seems to be orthogonal to the study of (effective) linear subspaces of embedding spaces, which has been studied extensively in recent years.

### Weaknesses
* the main weakness of the paper is its unclarity, which makes it extremely hard to assess what the paper claims exactly and what precisely is done in the experiments. This weakness makes it impossible for me to give a positive assessment of the paper in its current form. Here are some examples of this unclarity:
    * the theoretical hypothesis the paper claims to introduce is nowhere stated in an explicit and (mathematically) precise manner.
    * the main empirical claims follow from a spectral decomposition that is outlined in Algorithm 1, but this is poorly written. For example, what does $I_n$ mean and what does $N$ mean? And how are the ranks $r_n$ chosen? And what is happening in line 4 of the algorithm?
    * the experimental setup is also rather opaque. For example, it is not made clear how exactly the multiple trained DNNs with the same architecture are being used. I am assuming they are used as a "sample" on whose weight space SVD is applied, but this is nowhere explained in a detailed manner.
    * also the experimental results are not clear. For example, what is the relation between figures 3b and 3a? What does the word "summarized" in the caption of figure 3b mean? Another example is Table 3, where it is unclear what "mode-2" and "mode-3" mean.
* a rather disturbing weakness is that the paper contains (at least) two non-existing references. The two cited papers by Guth et al. do not exist and the mentioned arXiv numbers refer to totally different papers. I kindly ask the authors to give an explanation for this and to also let the reviewers know whether the paper contains more citation errors.
* a minor weakness is that the presentation is sometimes sloppy. The text contains main typo's or grammatical mistakes. Sometimes the text refers to the Appendix, but it is not clear to which section in the Appendix. In Appendix A.2 Theorem A.3 is stated, which I believe is a copy of Theorem 2.5, but Theorem A.2 is never referred to anymore.

### Questions
* The Tables 3 and 4 you are working with "universal" DNNs, which seem to be some kind of projections onto the universal subspace of parameters. How does this work precisely? For example, how are the dimensions of the universal subspace determined? Is this done per layer of the DNN? And which model is projected onto this subspace? 
* My other questions are contained in my overview of Weaknesses.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper first provides large-scale empirical evidence demonstrating that neural networks share similar low-dimension parameteric subspaces regardless of initialization, task, or domain. By projecting the parameter space to the low-dimension subspace, it is feasible to save storage and training consumption in future research.

### Strengths
1. The first to provide large-scale empirical study to demonstrate same neural networks share similar subspace parameter space.
2. Experiments are conducted among many model series, such as LLaMA, Mistral and etc.
3. The Universal Weight Subspaces for the same series models takes less storage but show comparable performance, which means it is feasible to use just one model for many tasks.

### Weaknesses
1. The paper claims universality “…neural networks…regardless of initialization, task, or domain” in abstract, but all experiments use models within the same architecture families (e.g., ResNet, ViT, LLaMA/Mistral) under the same domain tasks (different datasets). Thus, the evidence supports within-architecture similarity rather than true cross-domain or cross-architecture universality, making the claim overstated.

2. The study employs only a single tensor decomposition method (HOSVD), and the observed phenomena may strongly depend on this choice. Additional experiments using alternative high-order decompositions (e.g., HOOI) are needed to confirm that the findings are not artifacts of the specific method used.

3. In Section 3.2.1, the authors evaluate subspace expressiveness on IID and OOD tasks but do not specify what those tasks actually are. Similarly, Table 1 lists “Style 1 – Style 10” without any explanation, leaving the experimental setup and interpretation unclear.

4. In the ViT experiments, the authors skip the first and last layers during subspace decomposition, while other models do not, yet no justification or comparison is provided—would including these layers harm performance? Moreover, Table 2 does not specify what the IID and OOD tasks are, leaving the experimental design under-explained.

5. Each experiment relies on a large number of models per architecture, yet the relationship between model count and subspace effectiveness is not examined. It would be important to determine whether there exists a saturation point beyond which adding more models yields diminishing contributions to the shared subspace.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
