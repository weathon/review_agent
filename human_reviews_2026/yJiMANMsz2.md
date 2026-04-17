# DRESS: Disentangled Representation-based Self-Supervised Meta-Learning for Diverse Tasks

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Meta-learning represents a strong class of approaches for solving few-shot learning tasks. Nonetheless, recent research suggests that simply pre-training a generic encoder can potentially surpass meta-learning algorithms. In this paper, we hypothesize that the reason meta-learning fails to stand out in popular few-shot learning benchmarks is the lack of diversity among the few-shot learning tasks. We propose DRESS, a task-agnostic Disentangled REpresentation-based Self-Supervised meta-learning approach that enables fast model adaptation on highly diversified few-shot learning tasks. Specifically, DRESS utilizes disentangled representation learning to create self-supervised tasks that can fuel the meta-training process. We validate the effectiveness of DRESS through experiments on datasets with multiple factors of variation and varying complexity. The results suggest that DRESS is able to outperform competing methods on the majority of the datasets and task setups. Through this paper, we advocate for a re-examination of how task adaptation studies are conducted, and aim to reignite interest in the potential of meta-learning for solving few-shot learning tasks via disentangled representations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces DRESS, a task-agnostic Disentangled Representation-based Self-Supervised meta-learning framework designed to enable rapid model adaptation across highly diverse few-shot learning tasks. Specifically, DRESS leverages disentangled representation learning to construct self-supervised tasks that enhance the meta-training process. The effectiveness of DRESS is demonstrated through experiments on datasets featuring multiple factors of variation and different levels of complexity.

### Strengths
This paper points out the limited task diversity in existing few-shot learning benchmarks, which explains why pre-training and fine-tuning sometimes appear to outperform meta-learning. To address this, the authors develop new few-shot learning benchmarks featuring more diverse tasks for rigorous evaluation. Furthermore, the paper introduces DRESS, a method that generates diverse tasks to enable self-supervised meta-learning through disentangled representations. Extensive experiments validate the effectiveness of the proposed model.

### Weaknesses
1.	First, meta-learning has been extensively explored over the past decades, and I believe it may no longer be a highly novel research direction.
2.	The disentanglement technique has been applied to many tasks, so I have concerns about the originality and novelty of this work.
3.	How many latent dimensions (“L”) are used in the model? As far as I know, increasing the number of latent variables will significantly enlarge the model size, which may not be an efficient strategy. Moreover, with the rapid advancement of foundation models that already exhibit strong generalization capabilities, I believe such models could potentially address the problems discussed in this paper.

### Questions
Please see the Weaknesses, I would like to see details of novelty.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes DRESS, a self-supervised meta-learning framework that constructs diverse few-shot tasks using disentangled representations.
The method trains an encoder (e.g., FDAE or LSD) to separate latent factors, aligns latent dimensions, clusters each dimension to form pseudo-classes, and uses them to meta-train a model with MAML.
Experiments on curated and real-world datasets show DRESS outperforming existing self-supervised and unsupervised meta-learning baselines, especially when task diversity is high.

### Strengths
(1) The paper clearly motivates the link between task diversity and the failure of standard meta-learning benchmarks.

(2) The framework is conceptually interesting which using disentanglement to create pseudo-tasks is an intuitive way to increase diversity.

(3) The experiments are relatively broad, covering both synthetic datasets (Shapes3D, MPI3D, etc.) and realistic ones (CelebA, LFWA).

### Weaknesses
(1) The entire approach hinges on having well-disentangled latent factors, but disentanglement models rarely achieve clean factor separation in practice; this assumption is barely examined.

(2) The claimed novelty over prior unsupervised meta-learning methods (e.g., CACTUS, Meta-GMVAE) is incremental. It replaces clustering in feature space with clustering along latent dimensions but doesn’t bring a real theoretical or algorithmic advance.

(3) The so-called “task diversity metric” is ad-hoc and does not correlate with adaptation performance; it’s unclear whether higher diversity actually improves meta-generalization beyond synthetic datasets.

### Questions
(1) How robust is DRESS when the disentanglement isn’t perfect? In practice, disentangled features are often noisy or overlapping — does the whole task construction pipeline break down when that happens?

(2) How much does the method rely on the specific encoder choice (FDAE vs. LSD)? If we just use a strong pretrained backbone like DINOv2 without explicit disentanglement, does DRESS still work or does the advantage disappear?

(3) The paper claims that supervised labels can “misguide adaptation,” but that point feels hand-wavy. Can the authors actually show a concrete example or quantitative result where this happens?

(4) Why only test MAML? Would something simpler like ProtoNet or RelationNet also benefit from the proposed task construction, or is DRESS inherently tied to gradient-based adaptation methods?

### Soundness
2

### Presentation
2

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
To overcome the limitations of current meta-learning methods compared to pre-training and fine-tuning, the authors propose a simple addition to the few-shot learning task, where they add clustering-based disentangled representation learning for latent task embeddings to further diversify learning and aid generalization across tasks. They propose to generate self-supervised tasks by clustering along disentangled latent dimensions, thereby enforcing diversity among tasks. Experiments on controlled datasets (Shapes3D, SmallNORB, MPI3D, Causal3D) and real-world data (CelebA, LFWA) show that the proposed method achieves competitive results to existing meta-learning methods.

### Strengths
- The perspective on fortifying the task diversity and its implementation through disentanglement and clustering-based approach seems straightforward and reasonable.

- The objective of the proposed method is explained in a straightforward manner, with simple formulation which can be easily implemented and reproduced by other researchers.

- The authors performed extensive experimental validations on multiple , and they also conducted some ablation experiments to facilitate further understanding for the proposed work.

### Weaknesses
- Although the authors' main claim is that the limitations of current meta-learning methods come from the lack of task diversity in well-known meta-learning datasets such as miniImageNet and CIFAR-FS, it is not theoretically or empirically verified through thorough validation.

- Related to the previous question, the paper seems to conclude that it is better to use fine-tuning and pre-training approaches when there is a few-shot training dataset with small diversity. However, there are scenarios where the backbone feature extractor model lacks representational power (e.g. small in capacity), and this work seems to overlook this setting where conventional meta-learning can be dominant.

- Eventually, the proposed DRESS seems to only strike a middle ground where the task diversity is not too low, not too high, and the backbone network is not too simple, not too complex. This seems to limit the proposed work's wide application to various possible scenarios.

### Questions
Please refer to the weaknesses section. The insights found and the issues raised by the authors are still interesting and worth the analysis, but the applicability of the proposed method seems to be narrow, with limited usages.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper has a clear motivation or assumption that current few-shot learning benchmarks lack task diversity, which makes simple pre-training and fine-tuning approaches appear superior to meta-learning. This paper introduces DRESS (Disentangled REpresentation-based Self-Supervised meta-learning), a novel framework for few-shot learning that integrates disentangled representation learning with self-supervised meta-learning. 

To address this, DRESS constructs diverse self-supervised tasks and aligns latent dimensions to ensure semantic consistency,  then it clusters each latent dimension independently to form pseudo-classes, and creates varied few-shot tasks for meta-training using these clusters.

Comprehensive experiments on multiple datasets: SmallNORB, Shapes3D, Causal3D, MPI3D, CelebA, and LFWA, demonstrate that DRESS outperforms other baselines.

### Strengths
1. The motivation is clear and strong.
2. The presentation is very well and the arguments are coherent.
3. Comprehensive experiments and ablation studies are conducted to validate the proposed approach.

### Weaknesses
1. Disentangled encoder dependency: DRESS relies on disentangled representation models such as FDAE or LSD, which require substantial training resources and careful hyperparameter tuning. There are two concerns for this:
 - 1.1: The paper could discuss the computational trade-offs more explicitly.
 - 1.2: From the paper, "*all images available for meta-training are collected, and used to train a general purpose encoder*", it's confusing whether the pretrained model is also trained in this way or not. The difference should be clarified because the paper claims that in the scenario of task diversity scenario, meta-learning could outperform pretrained models. It seems meta-learning also has a pretrained encoder condition.

2. The task diversity metric validation is not enough: While intuitive, the proposed class-partition IoU metric’s correlation with downstream adaptation performance could be analyzed more deeply (e.g., correlation plots or regression). Could the proposed task diversity metric be integrated into the training process to encourage the model to sample or emphasize more diverse pseudo-tasks dynamically?

### Questions
In addition to the questions in weakness, some additional questions:

1. It is reasonable that on simpler datasets such as Omniglot, where tasks are highly similar, meta-learning methods often underperform compared to pre-trained models. When constructing tasks with high diversity, could incorporating out-of-domain tasks be beneficial? Specifically, how does DRESS perform if the disentanglement model is pre-trained on a dataset different from the meta-learning dataset (i.e., under out-of-domain disentanglement)?

2. What is the computational cost of DRESS compared to baselines (such as CACTUS or Meta-GMVAE, etc.), for instance, in terms of GPU hours or memory usage?

### Soundness
3

### Presentation
3

### Contribution
3
