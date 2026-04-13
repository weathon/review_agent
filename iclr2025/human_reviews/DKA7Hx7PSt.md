## Human Reviewer 1

### Summary
This paper aims to improve the effectiveness of knowledge distillation on datasets with few classes. Specifically, the authors propose identifying the informative linear subspaces in the teacher’s embedding space and generating multiple pseudo labels from the perspective of different subspaces. In this way, the number of classes can be increased to improve the distillation performance on the few-class dataset. Experimental results and ablation studies demonstrated the effectiveness of the proposed method.

### Strengths
(1) The motivation is clear. The authors expand the number of classes by mapping the teacher’s embedding into different subspaces to generate pseudo labels.

(2) The proposed method fully explores the latent structure hidden in the teacher’s embedding and avoids re-training the teacher network for pseudo-label generation, which is interesting.

(3) The authors conduct extensive experiments under different settings, such as binary-class and few-class distillation, and on different datasets, such as Amazon Reviews and Sentiment140. As shown in the experiments, the proposed method outperforms the existing distillation methods by a clear margin.

### Weaknesses
(1) Since the main advantage of the proposed method over the Subclass Distillation is re-training free, it would be better to compare the training costs of different methods in the Experiments.

(2) As shown in Algorithm 1, the proposed method obtains the subclass direction by feeding the training samples into the teacher network. Since the data augmentation will be different at each training epoch, I am wondering if the effectiveness of the subclass direction is degraded by computing in advance.

(3) From Figure 5, it seems that random projections already achieve stable and promising performance. How about using an ensemble of random projections with different initializations to generate the subclasses?

(4) The compared distillation methods are not new. The latest method was published in 2022.

(5) There are many typos in the current manuscript. For instance, “form”->”from” in line 073, “they authors”->”the authors” in line 149. “coarse-graned”->”coarse-grained” in line 202. The authors should carefully check the whole manuscript.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper introduces a knowledge distillation approach titled Learning Embedding Linear Projections (LELP) aimed at enhancing model performance in few-class classification tasks. LELP identifies informative linear subspaces within the teacher model's embedding space, splits them into pseudo-subclasses, and uses these to guide the training of the student model. Experimental results demonstrate that LELP outperforms existing state-of-the-art methods in large-scale NLP benchmarks such as Amazon Reviews and Sentiment140.

### Strengths
1. The author's viewpoint that "the information about the teacher model’s generalization patterns scales directly with the number of classes" is insightful and is not limited to knowledge distillation tasks.

2. The LELP method innovatively leverages structural information, i.e., "subclasses", in the teacher model's embedding space to enhance the performance of the student model, without the need to retrain the teacher model, and it is insensitive to differences in data types and model architectures.

### Weaknesses
1. Could you provide a more detailed explanation of why it is more effective to first project onto the "null-space" before performing PCA?

2. Why does random rotation guarantee that each direction has the same variance in expectation? Are there any theoretical insights regarding this?

3. When the number of categories in the dataset is sufficiently large, utilizing subclasses can further increase the category count. In this scenario, applying cross-entropy loss for distillation may weaken knowledge transfer for certain categories due to the additional reduction in gradient updates. 

4. If the method is only effective with a very small number of categories, its generalizability is quite limited. The authors should include performance comparisons of additional model architectures on datasets with a greater number of categories, such as ImageNet, in Table 4.

5. The authors only provided the results of the grid search for hyperparameters without showing the corresponding test performance for different values. Therefore, I am unsure about the method's sensitivity to hyperparameters, and I believe this is very important for other researchers who wish to utilize this method.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
The main goal of the paper is to handle cases with relatively small number of classes when applying knowledge distillation. To this end, the authors enlarge the effective number of classes by projecting the final embedding vectors of the student and the teacher into several PCA subsets. This way, the teacher may contain more information that are distributed over many more effective classes, which transfer to the student during its training process.

### Strengths
- The problem that the paper deals with (specifically, handling small number of class in knowledge distillation) is important and valuable. 
Also, the general idea of extending the clusters from the given classes to sub-classes is interesting and valid. 
-The results that were reported are also somewhat encouraging.

### Weaknesses
- In section 3 the authors argued that in case the student and the teacher do not share the same embedding dimensions, a learnable projection layer is required which can often harm performance - however, the authors do not provide any explanation or evidence to this sentence (why it harms performance?) nor at least any reference to this determination. Also note that the proposed approach in the paper includes much more projection layers - why in this case the authors don’t think it can harm the performance?

- The authors proposed to use PCA to obtain the informative linear subspaces, which is an off-line process where only the final embedding layer was used. I am wondering whether these linear subspaces could be learned as part of the training of the teacher, to also output these additional embeddings? (e.g. using reconstruction loss)

- There is a significant effort to explain the setup in section 4.1 which I am wondering whether it was necessary, especially as the authors focus on cases where the teacher and student architectures have exactly the same dimensions which as I stated before, not sure why to limit to these cases?

- I would expect the authors to experiment also with regular (large) number of classes as CIFAR-100, TinyImageNet or other datasets to understand what are the limitations of the proposed approach and how it behaves on regular and common cases where there are many categories.

- I found it very hard to understand the t-SNE plots provided in Figure 4. What is the meaning of running different t-SNE for each one of the methods as each individual t-SNE run organizes the points differently? Why the shape of the embeddings look so different in the top row? Further explanation will be helpful. 

- An intermediate analysis that shows the meaning of the sub-classes obtained by PCA  could help for visualization and understanding. For instance, would we observe meaningful fine-grained classes? 

- Main concern is the weak experimental section. Only Table 2 provides detailed classification results. Only one teacher and student architectural choices were used. It is not clear how the method generalizes to other architectures. Also, the results are not convincing enough to my opinion and in some cases are marginal. 

- What is the impact of the S hyper-parameter? (The number of sub-classes per class). I would expect some ablation study on this. 

Minor: 
Line 73: form —> from.
Line 202: coarse-graned —> coarse-grained

### Questions
I have already stated my questions in the 'weaknesses' section. Hope the authors can address my concern.

### Soundness
1

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
5

---

## Human Reviewer 4

### Summary
This manuscript proposes Learning Embedding Linear Projections, a method for distilling knowledge from a teacher model's representations. The proposed method identifies informative linear subspaces in the teacher's embedding space and converts them into pseudo-subclasses to teach students. It leverages the structure of final-layer representations and improves student performance, especially in finetuning tasks with a few classes without requiring retraining of the teacher model.

### Strengths
1.The idea of modality-independent of model distillation is interesting.
2.The implementation details are well-presented in the paper and comprehensive experiments are provided.
3.This paper is fluently written. The proposed method is easy to follow.

### Weaknesses
1.In the abstract and introduction section, the authors mentioned that existing methods can not perform well on few-class distillation because of the teacher model’s generalization patterns scales based on the number of classes. Could you explain more about this issue? Also, why the proposed method can solve the impact of poor generalization of the teacher model?
2.Most of the references on knowledge distillation are around 2020, which is too early and limited to reflect the development of work in the past two years. Also, is (Müller et al. 2020) the only research that uses sub-classes to solve few-class distillation? 
3.What is the actual impact of neural collapse mentioned in the related work on few-class distillation, and what is the relationship between the proposed method and neural collapse? I am puzzled because this is not reflected in the experiments.
4.The innovation of this method is not very novel. The proposed sub-classes framework has been developed by (Müller et al. 2020), and it just adds a pseudo-subclasses splitting component by PCA decomposition compared to (Müller et al. 2020).
5.Nowadays, in large-scale scenarios, binary classification and few-class classification tasks are not commonly utilized. Are there any practical applications for studying the distillation of binary classification? For example, what is the practical significance of the binarization experiments of CIFAR-10.
6.If other related works use sub-classes for distillation, please consider citing and comparing them in the experiments.
7.As for the datasets without subclass structure (Table 2), the gain over the best baseline is minimal. Half of the experiments show an improvement of approximately 0.3% or even less. 
8. The authors missed some related works that use multi-granularity class structures to address various tasks, e.g., long-tailed classification, incremental learning etc. The motivations between this manuscript and thses works are similar, so it is crucial to review such works to faciliate uncderstanding.

[1] Müller R, Kornblith S, Hinton G. Subclass distillation[J]. arXiv preprint arXiv:2002.03936, 2020.

### Questions
See Weaknesses.

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