## Human Reviewer 1

### Summary
The authors combine together three existing forms of meta-learning: namely, learning intializations, optimizers and loss functions. Through experiments on miniimagenet, tiered-imagenet, and CIFAR, the authors show that the approach outperforms baselines.

### Strengths
**Originality**

The paper combines together existing meta-learning techniques in a way that I believe has not been seen in prior literature.

**Quality**

Given that there is no theory introduced in this paper, the experimental results are very important. Fortunately, the experiments seem thoroughly conducted with many baselines and the proposed method strongly outperforms baselines on all experiments. Ablations are also included, which is good practice.

**Clarity**

The paper is quite well written and the figures and tables are well-presented. The form of the ablation tables (3, 4) are a good example for other papers to follow.

**Significance**

The paper will likely be of significance to meta-learning practitioners who are trying to achieve state-of-the-art performance on meta-learning tasks.

### Weaknesses
The originality of this work appears limited to me. The authors simply seem to combine existing techniques and find that the performance improves, which is not very surprising. This also limits the significance of the paper to a wider audience. I would encourage the authors to extend the work in at least one direction; some ideas are: 1) including a theoretical result, 2) proposing new performance improvements to the existing methods, 3) demonstrating the emergence of a new phenomenon when multiple meta-learning techniques are combined.

Also, in terms of practical use, it will be very important to compare the computational cost of the proposed method against baselines.

Overall, however, this is a technically solid work that lacks sufficient novely and impact.

### Questions
How does the computational cost (runtime, memory) of the proposed method compare to baselines?
Are the results state-of-the-art on the tasks tested?
Can the authors prove a theoretical result about the proposed method?

### Soundness
4

### Presentation
4

### Contribution
2

### Rating
3

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper proposes a bilevel optimization meta-learning algorithm that combines meta-learned initializations, meta-learned preconditioners and meta-learned loss functions with task-specific feature-wise linear modulation models.
It shows that this combination achieves competitive performance on visual few-shot classification task compared to other bilevel optimization meta-learning methods.

### Strengths
The paper is clearly presented, motivating the combination of existing components into a new algorithm well.

### Weaknesses
While the experimental baselines contain a number of bilevel optimization-based meta-learning algorithms that fall into the same paradigm, comparisons to other popular paradigms such as extended pretraining of the backbone (the presented method also pretrains the backbone) and in-context learning / sequence modelling are missing.
Such methods [e.g. 1, 2] achieve stronger performance on the few-shot learning tasks evaluated here albeit using larger models.
In combination with the large computational complexity of bilevel optimization, the significance of the presented method is therefore unclear to me (see questions).

[1] Hu, Shell Xu, et al. "Pushing the limits of simple pipelines for few-shot learning: External data and fine-tuning make a difference." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

[2] Fifty, Christopher, et al. "Context-aware meta-learning." arXiv preprint arXiv:2310.10971 (2023).

### Questions
- Can you elaborate how you situate your method in comparison to non bilevel optimization methods? My impression is that such methods [e.g. 1, 2] are both cheaper to run and perform better. Do you disagree with this statement? Where do you see the advantages of your method in comparison to pretraining / sequence modeling based approaches?
- The reported meta learning rate of $\eta= 0.00001$ combined with only 30'000 meta-steps seems to be relatively low and makes me wonder how much meta-learning actually happens. One number that would help to contextualize this would be the performance your model achieves at initialization without any meta-training (i.e. setting $\eta= 0.0$ but keeping everything else equal).
- Are the reported baselines reproductions or are the numbers taken from their respective papers?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 3

### Summary
This work proposes to meta-learn task-adaptive procedural biases by simultaneously learning initializations, optimizers, and loss functions that adapt to each specific task. It demonstrates that by meta-learning these components, the framework NPBML can induce strong inductive biases towards a distribution of learning tasks, leading to robust performance across several few-shot learning benchmarks.

### Strengths
1. NPBML combines the gradient-based meta-learning methods into a unified end-to-end framework, which meta-learns the key components of learning, i.e., initializations, optimizers, and loss functions, simultaneously. It enables meta-learning to acquire more optimization components and potentially enhances performance.

2. The framework is flexible and general, with many existing gradient-based meta-learning approaches emerging as special cases within NPBML.

### Weaknesses
1. There is a risk of meta-overfitting, where the model learns too well from the meta-training tasks and fails to generalize to new, unseen tasks. Although the authors mention this issue in the paper and suggest that it can be alleviated using regularization techniques, this introduces many manual choices, which contradicts the goal of automatically learning to learn from tasks. How to prevent meta-overfitting within the NPBML framework should be carefully discussed.

2. Although the authors state that the gradient-based optimizer is meta-learned, the number of steps to be updated in the inner loop is still a manually set hyperparameter. The article mentions that the early stopping mechanism can be learned implicitly, but in the experimental setup, 5 steps are used instead of a larger number to leverage this early stopping mechanism.

3. The networks used in the experiments are 4-CONV and ResNet-12. It remains questionable whether this framework is still effective on larger convolutional networks or transformer architectures.

4. The tasks used in the experiments are limited to 5-way 1-shot and 5-way 5-shot classification, which is quite different from the tasks that need to be addressed in real-world scenarios, such as segmentation, detection, super-resolution, translation, text summarization, and so on. The effectiveness of this framework in practical task scenarios has not been validated.

### Questions
1. How to prevent meta-overfitting within the NPBML framework?
2. How is the number of update steps in the inner loop determined? What would be the effect of using a large number of update steps, such as 50 or 100, in the inner loop?
3. Is this framework still effective for larger convolutional networks or transformer architectures?
4. What would the experimental results be when using this framework on more realistic tasks, such as segmentation or detection?

Please see Weaknesses for details.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 4

### Summary
The authors propose to combine several meta-learning methods into a single one, which they dub “Neural Procedural Biases Meta-Learning” (NPBML). The main idea is to meta-learning the initialization, optimizer, and loss function of a neural network over a distribution of tasks. They show gains on few-shot image classification benchmarks (FC100, CIFAR-FS, mini-/tiered-ImageNet), and perform ablation studies showing each component adds to the overall performance.

While I think the paper is sound, I find it lacks novelty and significance. While combining existing methods shows promising results on (outdated) benchmarks, the results are not compelling enough to justify acceptance. Moreover, the resulting method is actually similar to the MT-nets of Lee et al., 2018; the major difference seems to be learning FiLM layers instead of binary masks. For these reasons I think the paper should be revised and resubmitted.

### Strengths
- While low-hanging, the motivation to combine multiple meta-learning approaches is sound. I mention some caveats below, but I can see how enabling different components of the meta-learner to adapt could accelerate convergence and boost final performance.
- In all their experiments, the authors’ method performs the best. The ablations also support their claims.

### Weaknesses
Conceptual weaknesses:

- While it’s tempting to combine existing meta-learning work, a major caveat is not discussed: the more powerful the meta-learner, the higher the risk of meta-overfitting. In other words, the meta-learner risks to overfit to the train task distribution and fail to adapt to new unseen distributions. I wished the authors mentioned this trade-off — and others that arise from designing stronger meta-learner — explicitly and potentially even addressed it directly.
- None of the components in the proposed combination are novel. So this paper is only an incremental contribution, especially since the empirical results are underwhelming (more below). I would also like to note that the final combination is very close to the 8-year old work of Lee et al., 2018 (MT-nets): MT-nets also learn an optimizer, they also learn an initialization, the loss function is implicitly learned by the optimizer in the last network’s layer, and they also learn a modulating function. The main difference seems to be that here the modulating function uses FiLM layers whereas MT-nets uses binary masks.
- The proposed method is more difficult to implement and computationally more expensive than alternatives. This is a significant weakness, which the authors should also mention. For example, what is the runtime of their methods vs MAML, ProtoNet, or SimpleShot?

Experimental weaknesses:

- The benchmarks used in this work are somewhat outdated and don’t challenge modern meta-learning methods. In fact, I believe this is why the proposed method outperforms all other baselines: none of the benchmarks challenge the meta-generalization ability of the methods — instead, they reward overfitting to the train task distribution.
- Additionally, I think some baselines are missing. For example, ProtoNet or SimpleShot mentioned above. One could even make the argument that these two methods deserve larger backbone architectures, given their inference-time adaptation is much faster than gradient-based algorithms like NPBML.

### Questions
See my questions in the weaknesses section.

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