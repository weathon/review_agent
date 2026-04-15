# Look-Ahead Selective Plasticity for Continual Learning of Visual Tasks

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 3

## Abstract
Contrastive representation learning has emerged as a promising technique for continual learning as it can learn representations that are robust to catastrophic forgetting and generalize well to unseen future tasks. Previous work in continual learning has addressed forgetting by using previous task data and trained models. Inspired by event models created and updated in the brain, we propose a new mechanism that takes place during task boundaries, i.e., when one task finishes and another starts. By observing the redundancy-inducing ability of contrastive loss on the output of a neural network, our method leverages the first few samples of the new task to identify and retain parameters contributing most to the transfer ability of the neural network, freeing up the remaining parts of the network to learn new features. We evaluate the proposed methods on benchmark computer vision datasets including CIFAR10 and TinyImagenet and demonstrate state-of-the-art performance in the task-incremental, class-incremental, and domain-incremental continual learning scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper focused on contrastive learning methods in continual learning, and proposed to leverage the first few samples of the new task to identify and retain parameters contributing most to the transfer ability of the neural network, freeing up the remaining parts of the network to learn new features. The authors claimed that this idea is inspired from event models of the brain. The proposed method achieves some improvements on relatively simple dataset.

### Strengths
1. The paper is basically well-written and easy to follow.

2. The idea of event models is interesting. It’s good to see the connections between task boundaries and neurological mechanisms.

### Weaknesses
1. I agree that the current continual learning methods focus more on stability rather than plasticity/transfer. However, I think the technical contribution is incremental and not completely novel. The proposed method can be seen as an improved version of Co$^2$L. Also, the idea of “look-ahead” new tasks has been widely discussed in recent literature, such as learning and combing the new task solution [1] [2]. These related work should be discussed and compared (at least conceptually).

2. The proposed method can only achieve marginal improvements over Co$^2$L, especially for TinyImageNet in Table 1. Also, the considered benchmarks are relatively simple in continual learning.

3. The ablation study is not very clear, and the performance differences are marginal between each baseline in Table 2 (considering the error bars).

[1] Afec: Active forgetting of negative transfer in continual learning. NeurIPS 2021.

[2] Towards better plasticity-stability trade-off in incremental learning: A simple linear connector. CVPR 2022.

### Questions
Please refer to the weakness.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposed a new method for continual learning which is built on top of the existing work Co2L [1]. The author leverages the first batch of the new data as a surrogate to estimate the crucial parameters of the old model which are beneficial for new tasks and also important for old tasks. The estimation is done by searching for a set of embedding that can be salient for evaluating the above criteria, and the author adapts the existing Neural Similarity Learning [2] to identify these subsets. Then the author uses the Excitation Backprop (EB) to calculate the salience of each network weight and then uses the weight to mask the distillation loss for training the model to mitigate forgetting. The author also proposed a gradient modulation to modify the gradients. Extensive experiments are conducted on standard continual learning benchmarks.


Reference:
[1] Co2l: Contrastive continual learning (ICCV 2021)
[2] Neural similarity learning (NeurIPS 2019)
[3] Top-down neural attention by excitation backdrop (IJCV 2018)

### Strengths
1. Overall, the paper is easy to follow. Using the look-ahead idea to estimate the importance of the model weight seems to be interesting. 
2. The author provides many experiments and analyses to valid and reason about the proposed method.

### Weaknesses
1. The look-ahead idea is not totally new in continual learning. The author did not discuss the relationship between seminal work like "La-MAML: Look-ahead Meta-Learning for Continual Learning" (NeurIPS 2020) and the present work, where the La-MAML has already considered using the initial batch of data to adapt the gradient for continual learning, which in general is related to the author's proposed masked distillation training and gradient modulation. 

2. It is unclear why the paper needs to start with contrastive continual learning, i.e., Co2L, as the starting point for developing the method. First, since Co2L was published in 2021, there are so many continual learning methods that do not use contrastive learning and still achieve state-of-the-art (SOTA) performance. What is the necessity of using Co2L as the learning objective? Is it because the proposed method can not work without Co2L?

Moreover, the author stated in Page 4 that:

"We believe that this distillation loss is too limiting and diminishes the model’s ability to learn new generalizable representations since redundant parts of the embeddings are also regularized."

Could the author provide an explicit, formal, and/or empirical analysis about why the distillation loss will have such drawbacks? Such a claim is not sound, especially when we check the results in Table 1 that the proposed method does not significantly outperform the Co2L and the Co2L even outperforms the proposed method on SplitImageNet and R-MNIST. It is hard to convince the reader that the issue mentioned by the author for Co2L is grounded.

3. The author proposed to calculate the salient estimation for each parameter and use the ResNet-18 and two-layer linear network for experiments. How will the computation complexity for this salient estimation be on Page 6? Will there be a computational bottleneck when the proposed method is applied to modern neural network architecture like ViT and Transformer?

4. The CL methods compared in the present paper are up to 2021, while there are lots of new CL methods proposed after 2021 and the author did not review them in the paper and did not even mention why the author did not choose them for comparison. Moreover, the proposed method does not even significantly outperform the Co2L, i.e., the baseline they have chosen for developing their method. All of this largely hampers the significance of the current paper.

5. Although the related work section is not required, the reviewer still suggests the author to have a related work section to comprehensively review the existing CL methods especially at least discuss the recent advance of CL methods after 2021, instead of have a lengthy Introduction section which may largely distract the attention for a reader.

### Questions
Please refer to the Weaknesses section for more details.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Paper proposes method for relevant neuron selection within a self-supervised continual learning framework; their starting point is the Co2L method. The authors propose to learn a set neurons that are relevant for the current task performance (they compare several strategies). This learned set of representation-dimensions are then used to perform a masked instance-wise relation distillation loss. Results on a few datasets shows the method improves Co2L.

### Strengths
- the argument that the proposed method improves the potential plasticity by reducing regularization on redundant dimensions of methods is nice. Measuring this on the current task data is new (rather on the previous)

- the proposed method obtains decent performance gain for small memory size especially on CIFAR10.

### Weaknesses
- the idea to focus on the importance of neurons for future (or current) tasks is new, many methods aim to measure the importance of neurons for previous tasks. However, the final difference between these strategies is very small (see table 2), and in my opinion too small. 

- I do not really like CIFAR 10 for continual learning since the tasks are really small. I would like to also see results on CIFAR100 and if possible on ImageNet-subset. 

 - more results on the subset size should be added.

### Questions
Please address the weaknesses. 

For me in Table 2, the gain with respect to CO2L are ok, but not very large, and I would really like to at least also see it on CIFAR100 /10 split. Table 2 shows that selection can work; however, it also shows that any selection works and that the results among the various ways of selecting are very small (the 'look-ahead' does not seem crucial). 

minor:
- I would consider removing GM from the paper since it does not improve results. 
- I'm not sure if the term 'salient' is very adequate to refer to the selected neurons.
-  number of tasks used per dataset should be clearly stated in the main paper.
- add in table 2 without using the selection as well (I think it helps, even though the numbers are Table 1)

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
