

{0}------------------------------------------------

# INCREMENTAL LEARNING WITH TASK-SPECIFIC ADAPTERS

Anonymous authors

Paper under double-blind review

## ABSTRACT

Incremental learning aims to continuously acquire new knowledge while preserving previously learned information. Existing literature primarily focuses on improving model stability, often at the cost of plasticity, to prevent the forgetting of earlier tasks. In this paper, we argue that inter-task differences are the primary driver of catastrophic forgetting. To address this challenge, we propose a novel network architecture compromising two distinct components: one dedicated to learning invariant features shared across tasks and another for capturing task-specific details. Specifically, we repurpose adapters, originally introduced for parameter-efficient fine-tuning, as feature modifiers to capture task-specific details, while the backbone network focuses on learning invariant features. Unlike prior approaches that keep the backbone frozen and only fine-tune adapters, we co-train both the backbone network and adapters, employing an additional regularization term that encourages the backbone to learn shared features. Our approach integrates seamlessly with established methods, such as Learning without Forgetting (LwF). Extensive experiments on CIFAR-100 and ImageNet datasets demonstrate that our adapter-based methods consistently outperform non-adapter counterparts across diverse learning scenarios, including various task orders and data scales. Our approach improves both plasticity and stability, effectively addressing the stability-plasticity dilemma.

## 1 INTRODUCTION

Real-world data often arrive in a sequential manner, in batches, or through periodic updates. These data dynamics are further complicated by constraints such as limited storage capacity and privacy considerations. In such contexts, employing concurrent or multitask learning, where models are trained on a single, static, large dataset, can be costly or impractical. Incremental learning (IL), also known as lifelong learning, is specifically designed to handle these dynamic environments. This paradigm allows a model to continuously learn and update its knowledge from new data without retraining from scratch. Unlike traditional methods that require access to the entire dataset for training, incremental learning enables the model to adapt to new tasks over time using only the data provided for each new task while retaining knowledge from previous tasks.

Feature extraction and fine-tuning are two common approaches for learning new tasks without accessing previous training data. In feature extraction, the weights of a pre-trained model are kept fixed, and the outputs of the top layer are used as features for the new task (Donahue et al., 2014; Belouadah & Popescu, 2018). While this approach helps maintain existing knowledge, it may struggle to effectively learn new tasks. Fine-tuning improves upon feature extraction by updating the model weights for the new task. A low learning rate is typically used to ensure the model retains the structure and knowledge from previous tasks (Girshick et al., 2014). While this approach enables better adaptation to new tasks, it carries the risk of *catastrophic forgetting*, where performance on previously learned tasks deteriorates rapidly (Goodfellow et al., 2013; McCloskey & Cohen, 1989). Forgetting occurs when the task-specific weights of the previous tasks are altered to accommodate new tasks. These methods may not fully address the challenges of incremental learning, which requires balancing the preservation of existing knowledge with the acquisition of new skills.

An ideal incremental learning algorithm should at least possess the following key properties: efficient memory usage and stability-plasticity balance. First, the algorithm should optimize memory usage, preserving no or only essential data points from previous tasks. This may reflect realistic scenarios

{1}------------------------------------------------

where data might be transient or unavailable due to privacy or memory constraints. Second, it should maintain a balance between stability (retaining past knowledge) and plasticity (learning new information), avoiding catastrophic forgetting (losing old knowledge when learning new data). Balancing stability and plasticity is a fundamental challenge in incremental learning, often referred to as the stability-plasticity dilemma (Mermillod et al., 2013).

Existing research in incremental learning primarily focuses on mitigating catastrophic forgetting by improving stability at the expense of plasticity. One popular approach is to directly regularize weight changes (Kirkpatrick et al., 2017; Zenke et al., 2017; Aljundi et al., 2018) or to limit divergence in output predictions, between the old and new models (Li & Hoiem, 2017; Dhar et al., 2019). This strategy can be extended by incorporating data retained from previous tasks, which further prevents forgetting by replaying some of the prior knowledge (Chaudhry et al., 2018; Rebuffi et al., 2017; Zhang et al., 2020b). However, enhancing stability often hurts plasticity. Few works simultaneously improve stability and plasticity.

**Contribution** In this paper, we argue that the inter-task differences contribute to catastrophic forgetting and propose to model these differences. Specifically, we propose a network design consisting of two blocks: a backbone network for learning invariant features across all tasks and many small networks for modeling task-specific knowledge. Specifically, we use adapters, originally introduced for parameter-efficient fine-tuning of large language models (Houlsby et al., 2019), as these small networks to capture the task-specific information. Our approach differs from the conventional use of adapters, which are typically added to a frozen network to attain comparable performance to full fine-tuning. Instead, we re-purpose adapters as feature modifiers and train them together. This strategy enables the adapters to encapsulate task-specific information in the layers closer to the output, while squeezing task-invariant knowledge into layers nearer the input. Our approach can be integrated with many existing methods such as EWC and LwF. Our empirical results demonstrate that various adapter-assisted methods consistently outperform non-adapter counterparts along the learning process, ranging from regularization-based to rehearsal-based approaches, across both CIFAR-100 and ImageNet datasets. The advantage remains robust among dataset choices, task scales, and task orderings. Our approach improves both plasticity and stability, eliminating the stability-plasticity dilemma.

## 2 RELATED WORK

This section reviews works that are closely related to ours.

**Multi-task Learning** Multi-task learning involves training on all tasks simultaneously, leveraging shared network parameters to exploit inter-task commonalities, in contrast to incremental learning (Caruana, 1997; Ruder, 2017). However, multi-task learning can be costly or impractical due to the substantial computational burden of training on large datasets, alongside challenges such as limited storage capacity and privacy concerns (Kendall et al., 2018). These limitations highlight the importance of incremental learning as a critical area of research.

**Incremental Learning** Incremental learning involves learning tasks sequentially. However, this approach is susceptible to catastrophic forgetting, where learning new tasks can overwrite previously acquired knowledge (McCloskey & Cohen, 1989). To address this challenge, three primary strategies are commonly employed: regularization-based, rehearsal-based, and parameter-isolation methods. Regularization-based methods mitigate forgetting by regularizing differences in weights or output predictions between the old and new models (Kirkpatrick et al., 2017; Zenke et al., 2017; Aljundi et al., 2018; Li & Hoiem, 2017; Dhar et al., 2019; Joseph et al., 2022). Rehearsal-based methods preserve prior knowledge by retaining instances from previous tasks and training on a combined dataset that includes these instances alongside data from the new task. This retained data may consist of exemplar images (Rebuffi et al., 2017; Chaudhry et al., 2018), publicly available external datasets (Lee et al., 2019; Zhang et al., 2020b), or synthetic data generated by generative models (Shin et al., 2017; Kemker & Kanan, 2017; He et al., 2018). Parameter-isolation methods improve stability by freezing parameters that are critical to previously learned tasks (Mallya & Lazebnik, 2018; Serra et al., 2018).

{2}------------------------------------------------

![Figure 1: A grid of 10 subplots showing model accuracy (%) over the number of tasks seen (0 to 800) for 10 different tasks. The subplots are labeled Task 1 through Task 10. Each subplot contains three data series: alphabetical ordering (blue line with circles), coarse grained ordering (red line with triangles), and iCaRL random ordering (purple line with diamonds). The coarse grained ordering consistently shows higher accuracy across most tasks compared to the other two orderings, especially in the later tasks.](d9d706121d1c4ba9ad2c793746382361_img.jpg)

Figure 1: A grid of 10 subplots showing model accuracy (%) over the number of tasks seen (0 to 800) for 10 different tasks. The subplots are labeled Task 1 through Task 10. Each subplot contains three data series: alphabetical ordering (blue line with circles), coarse grained ordering (red line with triangles), and iCaRL random ordering (purple line with diamonds). The coarse grained ordering consistently shows higher accuracy across most tasks compared to the other two orderings, especially in the later tasks.

Figure 1: The model’s accuracy using LwF for incremental learning is evaluated on three different CIFAR-100 task orderings: the standard alphabetical category ordering, the coarse grained ordering, and a random ordering with the fixed seed 1993 used by iCaRL (Rebuffi et al., 2017). The coarse grained ordering has more inter-task diversity.

While each of the aforementioned strategies has its strengths, they also present notable limitations. Regularization-based and parameter-isolation approaches improve model stability but often compromise plasticity, and rehearsal-based methods encounter practical challenges, including storage constraints, privacy concerns, and scalability issues. Recent research has proposed adapter-based methods to address catastrophic forgetting (Rajasegaran et al., 2020; Bhat et al., 2023; Pham et al., 2021; Wang et al., 2022; Liang & Li, 2024; Zhang et al., 2020a). Some of these works incorporate adapter-like subnets. However, they are limited in several ways: 1) freezing the backbone (Liang & Li, 2024; Zhang et al., 2020a) can negatively impact performance as the shared information can not be learned effectively; 2) relying on custom, complex loss functions (Liang & Li, 2024; Bhat et al., 2023) reduces compatibility with new algorithms and limits broader applicability; and 3) none incorporates regularization- or prediction-based approaches. By focusing exclusively on network architecture, these methods fail to leverage valuable insights from robust yet foundational baselines such as Learning without Forgetting (Li & Hoiem, 2017). To address these shortcomings, we propose leveraging adapters to develop a lightweight and compatible solution that combines strong performance with insights from both adapter design and algorithmic principles.

**Fine-tuning with adapters** Another line of research explores the potential of adapters for transferring knowledge to downstream tasks, particularly within large language models. Adapters are compact modules inserted between the layers of a pre-trained large model (Houlsby et al., 2019). These modules are typically fine-tuned while the original network remains frozen. Fine-tuning with adapters achieves performance comparable to full fine-tuning across diverse tasks (Li & Liang, 2021). Moreover, adapters facilitate rapid adaptation to new tasks without catastrophic forgetting (Pfeiffer et al., 2020a), addressing challenges in multi-domain (Chronopoulou et al., 2023; Asai et al., 2022) and multilingual settings (Pfeiffer et al., 2020b).

## 3 INCREMENTAL LEARNING WITH TASK-SPECIFIC ADAPTERS

This section highlights a key limitation of current knowledge distillation methods for incremental learning and introduces adapters as feature modifiers to model inter-task differences.

### 3.1 WHAT IS MISSING IN REGULARIZATION-BASED METHODS?

A key limitation of regularization-based methods is their susceptibility to the stability-plasticity dilemma. Specifically, anchoring the model to its performance on prior tasks limits its plasticity in learning new tasks, while relaxing the regularization leads to catastrophic forgetting. This issue becomes more pronounced with greater inter-task diversity in the incremental learning problem. Variations in inter-task diversity can be introduced through different task orderings.

Figure 1 examines the impact of task orderings on the classification performance of the LwF algorithm applied to CIFAR-100. In this scenario, the model is incrementally trained with 10 classes per task over a total of 10 tasks. Three different task orderings are evaluated: an alphabetical class ordering, a

{3}------------------------------------------------

![Figure 2: Architecture of the adapter and a comparison highlighting the distinctions in its implementation between traditional fine-tuning and our method. The diagram is divided into three parts. The left part shows the internal structure of an adapter: an input vector is processed through a down-projection layer (W_down), a nonlinear activation function, and an up-projection layer (W_up), with a skip connection bypassing the nonlinear part. The middle part, 'Traditional adapter-assisted fine-tuning', shows a frozen feature extractor φ followed by a frozen adapter β and a fine-tuned sigmoid output. The right part, 'Our method with adapters', shows a fine-tuned feature extractor φ followed by a frozen adapter β and a fine-tuned sigmoid output. A legend indicates that white boxes represent 'Freezing' and blue boxes represent 'Fine-tuning'.](2fa4a1bf91d0f34e87c689fbc1211fe3_img.jpg)

Figure 2: Architecture of the adapter and a comparison highlighting the distinctions in its implementation between traditional fine-tuning and our method. The diagram is divided into three parts. The left part shows the internal structure of an adapter: an input vector is processed through a down-projection layer (W\_down), a nonlinear activation function, and an up-projection layer (W\_up), with a skip connection bypassing the nonlinear part. The middle part, 'Traditional adapter-assisted fine-tuning', shows a frozen feature extractor φ followed by a frozen adapter β and a fine-tuned sigmoid output. The right part, 'Our method with adapters', shows a fine-tuned feature extractor φ followed by a frozen adapter β and a fine-tuned sigmoid output. A legend indicates that white boxes represent 'Freezing' and blue boxes represent 'Fine-tuning'.

Figure 2: Architecture of the adapter and a comparison highlighting the distinctions in its implementation between traditional fine-tuning and our method. Left: an adapter consists of the down-projection, the nonlinear transformation, up-projection, and skip-connection. Right: The key difference between traditional use of adapter and ours is that we allow adapters to be co-trained with the entire network when learning a new task.

random ordering with a fixed seed commonly used by iCaRL and other methods (Rebuffi et al., 2017; Zhang et al., 2020b), and a coarse grained ordering that groups similar classes within each task based on CIFAR-100’s 20 coarse categories.

The coarse grained ordering has greater inter-task diversity, providing a way to assess how increased inter-task differences affect the incremental learning algorithm. Notably, when classes are learned in a coarse-grained ordering, there is a significant increase in both forgetting and accuracy loss for each task compared to other orderings. This increased forgetting and accuracy decline can be attributed to the stability-plasticity dilemma of the LwF algorithm when confronted with greater inter-task diversity.

Therefore, there is a need to develop new strategies that can eliminate the stability-plasticity dilemma of regularization-based approaches, enabling them to learn new tasks without forgetting.

### 3.2 INTRODUCING ADAPTERS

In this section, we propose a network architecture comprising two distinct components: a backbone network for learning invariant features shared across all tasks and multiple lightweight adapters for capturing task-specific information. These adapters enhance the plasticity of the architecture, enabling it to adapt to new tasks, while the backbone network maintains stability by focusing on shared and invariant features. Unlike existing approaches that train adapters while keeping the backbone frozen, we use adapters as task-specific feature modifiers and co-train them alongside the backbone network. As the model trains on additional tasks and samples, the backbone network refines its ability to learn invariant features, further enhancing stability. This architectural design improves both stability and plasticity, effectively eliminating the stability-plasticity dilemma.

The adapters are positioned between the backbone feature extractor, denoted as  $\varphi$ , and the label predictor layer, serving as task-specific feature modifiers  $\beta^t$  for each task  $t$ . As illustrated in Figure 2, the adapters adopt a conventional bottleneck structure. Starting with an initial dimension  $d$  and a bottleneck width  $b$ , we design the down-projection layer to reduce dimensionality from  $d$  to  $b$  using a fully-connected neural network with a weight matrix  $W_{d \times b}$  and a non-linear activation function  $g$ , expressed as

$$\text{Down}_{d \rightarrow b}(\mathbf{x}) = g(\mathbf{x}W_{d \times b}).$$

Similarly, the up-projection layer is defined as

$$\text{Up}_{b \rightarrow d}(\mathbf{x}) = g(\mathbf{x}W_{b \times d}).$$

Our adapters consist of both a down-projection and an up-projection step and are connected to the output via a skip-connection (He et al., 2021). This bottleneck design allows the adapter to utilize both backbone features from  $\varphi(x)$  and the modified features processed through the down- and up-projection layers:

$$\beta^t(\varphi(x)) = \varphi(x) + \text{Up}_{b \rightarrow d}^t(\text{Down}_{d \rightarrow b}^t(\varphi(x))).$$

{4}------------------------------------------------

Our adapter module incorporates three key features specifically designed for incremental learning: (i) a compact design with a small number of parameters; (ii) compatibility with and enhancements over existing methods; and (iii) the simultaneous updating of feature extractor layers and adapters. The compact design addresses memory constraints, while the latter two features jointly improve stability and plasticity.

For (i), our method leverages adapters to control the growth rate of the overall model size when accommodating additional tasks. When applied to large backbone networks, the parameters introduced by the adapters are negligible, making our approach well-suited for models with memory constraints. This contrasts with strategies that rely on dynamically expanding networks (Yoon et al., 2017; Yan et al., 2021).

For (ii) and (iii), we address the stability-plasticity dilemma through a novel two-block design and their co-training. Specifically, adapters are trained to enhance plasticity, while the continuous training of backbone networks focuses on improving invariant feature learning, thereby further enhancing stability. Such a distinction is particularly challenging in traditional, non-adapter architectures, as their holistic design inherently exacerbates the stability-plasticity dilemma.

Lastly, to enforce that the backbone network learns invariant features and thus the adapters learn task-specific information, we develop method-specific regularization techniques. For the prediction-regularized methods such as LwF, we impose a knowledge distillation loss on the backbone, encouraging the backbone is similar before and after learning each new task. For the weight-regularized methods such as EWC, we free adapters from regularization. The approaches and designs are discussed in the following section in detail.

#### 3.2.1 ENFORCING INVARIANT FEATURE LEARNING IN THE BACKBONE

We perform necessary modifications to the incremental learning methods so that the adapters can fit them and perform well. This section presents the adjustments for such integrations with different regularization methods.

**Prediction-regularized Methods** The prediction-regularized methods attempt to address the stability-plasticity dilemma through model distillation instead of weight control. Algorithms such as LwF and Learning without Memorizing (LWM) (Dhar et al., 2019) fall into this category. Additional to the task loss, the model outputs at task  $t$  are regularized with the model outputs at all tasks  $t'$  where  $1 \leq t' < t$ , i.e. a distillation loss:

$$\begin{aligned}\mathcal{L}^t &= \ell^t(\theta) + \lambda_{\text{distill}} R_{\text{distill}}^t \\ &= \ell^t(\theta) + \lambda_{\text{distill}} \sum_{t'=1}^{t-1} M\left(\varphi^{t'}(x), \varphi^t(x)\right)\end{aligned}$$

where  $M$  is a metric that quantifies the similarities between the adapter outputs, such as the cosine similarity or cross entropy<sup>1</sup>, and  $\lambda_{\text{distill}}$  is a hyperparameter

This regularization method loosens the stability restriction of the network by distilling instead of direct controlling model parameters. It allows the model to search for an optimal solution in a larger parameter space. However, while we anticipate the adapter and the backbone model to capture the task-specific and task-invariant information respectively, this distillation constraint neither gives the adapters absolute freedom to the parameter search space nor poses a strong restriction for backbone task-invariance. The former cannot be done since adapters are involved in the forward pass and the computation of the distillation loss, and is arguably not a big deal due to that the restriction is already-loosen compared to the weight-regularized methods. Addressing the latter problem, we introduce an additional backbone regularization:

$$R_{\varphi}^t = \sum_{t'=1}^{t-1} M\left(\text{Linear}_{d \times c}(\varphi^{t'}(x)), \text{Linear}_{d \times c}(\varphi^t(x))\right),$$

where  $c \leq d$  is a dimension we reduce to. In practice, we choose  $c$  to be the number of classes of each task, as intuitively we make this regularization implicitly a direct distillation on backbones.

<sup>1</sup>Following previous works, we use the cross entropy loss as the metric.

{5}------------------------------------------------

Hence, to learn a new task  $t$ , we define a loss function that includes the task loss, the distillation term, and the backbone regularizer to align backbones across tasks to improve stability:

$$\mathcal{L}_t = \ell_t(\theta) + \lambda_{\text{distill}} R_{\text{distill}}^t + \lambda_\varphi R_\varphi^t, \quad (1)$$

where  $\lambda_\varphi$  is the hyperparameter for the backbone regularizer. The parameter  $\lambda_{\text{distill}}$  aims to balance retaining prior knowledge with adapting to new tasks during the incremental learning process, while  $\lambda_\varphi$  controls the direction regularization to the backbone across tasks. We apply this adapter regularization on LwF as it is a prediction-regularized method and, in our experience, it is among the most effective methods for incremental learning tasks.

**Weight-regularized Methods** The weight-regularized methods control and regularize the weights of each task. Taking Elastic Weight Consolidation (EWC) (Kirkpatrick et al., 2017), a noteworthy representative of such methods, as an example,  $\mathcal{L}^t$ , the loss at task  $t$  can be computed by the task loss and an additional term for regularizing each parameter:

$$\mathcal{L}^t = \ell^t(\theta) + \sum_{t'=1}^{t-1} \sum_i \frac{\lambda}{2} F_i(\theta_i - \theta_{t',i}^*)^2,$$

where  $\theta$  is the model parameter and  $F_i$  is the Fisher information matrix at each parameter  $i$ .

The regularization itself enables the stability of the entire network. In order to improve the plasticity, we attempt to control the backbone's weight only so that the adapters remain unregularized and thus are able to move freely for parameter exploration. Hence, the integration of adapters to such methods is achieved by ruling out the adapter parameters from regularization, i.e.  $i \notin \theta_a$ :

$$\mathcal{L}^t = \ell^t(\theta) + \sum_{t'=1}^{t-1} \sum_{i \notin \theta_a} \frac{\lambda}{2} F_i(\theta_i - \theta_{t',i}^*)^2.$$

This modification is applicable to all the weight-regularized methods, as long as they involve parameter-level consolidation. Methods such as Memory Aware Synapses (MAS) (Aljundi et al., 2018) and Path Integral (Path Integral) (Zenke et al., 2017) fall into this category and thus apply the above-mentioned adjustments to align with adapters.

## 4 EXPERIMENTS

In this section, we compare our framework on various methods to their counterparts that are not using adapters. We investigate impact of adapters on different settings, including different method types, task scales, task orderings and datasets.

### 4.1 EXPERIMENTAL SETUP

**Datasets** We compare the performance of different methods on two datasets: CIFAR-100 (Krizhevsky et al., 2009) and ImageNet (Russakovsky et al., 2015). CIFAR-100 consists of images with small resolutions (input sizes of  $32 \times 32 \times 3$ ) and serves as our primary focus for studying the impact of adapters across different settings. We also include ImageNet, which offers more diverse training images at higher resolutions ( $224 \times 224 \times 3$ ). To mitigate training time and resource constraints, we limit our analysis to the first 100 classes from ImageNet. Dataset statistics are summarized in Appendix A.1.

To ensure a fair comparison, we perform hyperparameter tuning and learning rate selection on a validation set (Masana et al., 2022). The validation set is a class-balanced split, compromising 10% samples from the original training dataset, while the remaining 90% serves as our training dataset. Details on the hyperparameter tuning and learning rate selection can be found in Appendix A.2, and the selection of adapter-specific hyperparameters is explained in Section 4.2.

**Network architectures** Following (De Lange et al., 2022; Hou et al., 2019), we employ two different models for the two datasets. For CIFAR-100, which contains small-resolution images, we use ResNet-34. For ImageNet, with larger-resolution images, we use ResNet-18, as suggested in (He et al., 2016).

{6}------------------------------------------------

![Figure 3: Two line graphs showing average accuracy (%) vs. number of tasks (1 to 10) for CIFAR-100. The left graph shows weight regularization methods (ewc-base, ewc-adapter, mas-base, mas-adapter, path-integral-base, path-integral-adapter). The right graph shows prediction regularization methods (lwf-base, lwf-adapter, lwm-base, lwm-adapter). In both graphs, methods with adapters (solid lines) consistently outperform those without adapters (dashed lines).](73c3e4508cae529acf4e6c7fa70b361a_img.jpg)

The figure consists of two line graphs. The left graph, titled 'Weight regularization methods', plots accuracy (%) from 45 to 75 against the number of tasks from 1 to 10. It includes six series: ewc-base (blue dashed), ewc-adapter (blue solid), mas-base (orange dashed), mas-adapter (orange solid), path-integral-base (red dashed), and path-integral-adapter (red solid). The right graph, titled 'Prediction regularization methods', plots accuracy (%) from 45 to 75 against the number of tasks from 1 to 10. It includes four series: lwf-base (pink dashed), lwf-adapter (pink solid), lwm-base (blue dashed), and lwm-adapter (blue solid). In both graphs, the adapter-enhanced methods (solid lines) maintain higher accuracy than their base counterparts (dashed lines) as the number of tasks increases.

Figure 3: Two line graphs showing average accuracy (%) vs. number of tasks (1 to 10) for CIFAR-100. The left graph shows weight regularization methods (ewc-base, ewc-adapter, mas-base, mas-adapter, path-integral-base, path-integral-adapter). The right graph shows prediction regularization methods (lwf-base, lwf-adapter, lwm-base, lwm-adapter). In both graphs, methods with adapters (solid lines) consistently outperform those without adapters (dashed lines).

Figure 3: The average accuracy for regularization-based methods with or without adapters on CIFAR-100 (alphabetical ordering) in task-IL. The solid line represents the results with adapter, while the dashed line represents the results without adapter. The left figure displays the performance with weight regularization (EWC, MAS, and Path Integral), and the right figure displays the performance with prediction regularization (LwF and LwM).

**Evaluation metrics** Two evaluation protocols are commonly used in incremental learning: task incremental learning (task-IL) and class incremental learning (class-IL). Task-IL evaluates the network in a multi-head setting, utilizing a task-ID oracle to determine the appropriate task-specific head at the inference time. In task-IL, the model does not need to differentiate between classes from different tasks. In contrast, class-IL presents a more practical yet challenging scenario where the model must make predictions across all learned classes within a single-head configuration. This requires the model to resolve confusion arising from different tasks. In this section, we focus on task-IL with task-ID information at the inference time, while results for class-IL are included in Appendix B.

To compare the overall learning process across different methods, we use the average accuracy at each task  $t$ , denoted by  $A_t = \frac{1}{t} \sum_{i=1}^T a_{t,i}$ , and  $a_{t,k}$  is the accuracy evaluated on task  $k$  after training on task  $t$ . To ensure reliable and consistent results, we report the averaged results over 10 runs with different random seeds for both CIFAR-100 and ImageNet.

### 4.2 EXPERIMENTAL RESULTS

**On regularization-based methods** In this section, we study the effect of adapters combined with various weight-regularized methods in incremental learning, including Elastic Weight Consolidation (EWC) (Kirkpatrick et al., 2017), Memory Aware Synapses (MAS) (Aljundi et al., 2018), Path Integral (PathInt) (Zenke et al., 2017), as well as prediction-regularized methods such as Learning without Forgetting (LwF) (Li & Hoiem, 2017) and Learning without Memorizing (LwM) (Dhar et al., 2019).

Figure 3 compares different methods with and without adapters in task-IL on CIFAR-100. From the first task onward, weight-regularized methods with adapters exhibit an approximate 3% increase in average accuracy compared to those without adapters. This improvement is consistently maintained throughout the learning process across all methods. For prediction-regularized methods, the accuracy advantage further escalates to as much as 5%. The observed increase in accuracy can be primarily attributed to the model’s improved plasticity to learn new task-specific knowledge through the use of adapters, while the backbone was continuously trained as well.

**On task scale** This paragraph examines the effect of the task scale. The benefits of utilizing adapters diminish as the number of classes increases within each task. As illustrated in Figure 4, when learning either 5 or 10 classes simultaneously, our adapter-based approach continues to effectively capture inter-task differences and significantly outperforms methods without adapters. However, while the advantages are still present, they become less pronounced as the number of classes per task increases. This reduction in the performance margin is understandable, as learning more classes per task not only provides more data for the model to learn but also requires more memory and storage, with fewer regime shifts. In the extreme case, when the number of classes per task reaches 100, incremental learning reduces to a multi-task learning problem.

**On task ordering** As discussed in Section 3.1 and by Masana et al. (2020), the impact of task orderings on the performance of incremental learning models is often overlooked. While regularization-

{7}------------------------------------------------

![Figure 4: Three line graphs showing average accuracy for EWC and LwF with 5, 10, and 20 classes at a time on CIFAR-100. Each graph plots accuracy (%) against the number of tasks (1 to 10). Four methods are compared: ewc-base (purple solid), ewc-adapter (purple dashed), lwf-base (blue solid), and lwf-adapter (blue dashed). In all cases, adapter methods maintain higher accuracy than base methods as the number of tasks increases.](3121afa7ca030b22ee0345864ca6f38b_img.jpg)

Figure 4: Three line graphs showing average accuracy for EWC and LwF with 5, 10, and 20 classes at a time on CIFAR-100. Each graph plots accuracy (%) against the number of tasks (1 to 10). Four methods are compared: ewc-base (purple solid), ewc-adapter (purple dashed), lwf-base (blue solid), and lwf-adapter (blue dashed). In all cases, adapter methods maintain higher accuracy than base methods as the number of tasks increases.

Figure 4: The average accuracy for EWC and LwF with learning 5, 10, and 20 classes at a time on CIFAR-100 (alphabetical ordering) in task-IL.

![Figure 5: Four line graphs showing average accuracy for regularization-based methods (EWC and LwF) with or without adapters on different orderings of CIFAR-100. The top two graphs are for coarse ordering, and the bottom two are for iCaRL ordering. Each graph plots accuracy (%) against the number of tasks (1 to 10). Six methods are compared: ewc-base, ewc-adapter, max-base, max-adapter, path-integral-base, and path-integral-adapter. Solid lines represent results with adapters, and dashed lines represent results without adapters. Adapters consistently show higher accuracy across all orderings and methods.](d864789b0d8384da1d22fd6a5d76bbdf_img.jpg)

Figure 5: Four line graphs showing average accuracy for regularization-based methods (EWC and LwF) with or without adapters on different orderings of CIFAR-100. The top two graphs are for coarse ordering, and the bottom two are for iCaRL ordering. Each graph plots accuracy (%) against the number of tasks (1 to 10). Six methods are compared: ewc-base, ewc-adapter, max-base, max-adapter, path-integral-base, and path-integral-adapter. Solid lines represent results with adapters, and dashed lines represent results without adapters. Adapters consistently show higher accuracy across all orderings and methods.

Figure 5: The average accuracy for regularization-based methods with or without adapters on different orderings of CIFAR-100 in task-IL. The upper two figures present the experimental results on the coarse ordering, and the lower two are on the iCaRL ordering. The solid line represents the results with adapter, while the dashed line represents the results without adapter.

based methods suffer from the stability-plasticity dilemma, the introduction of adapters improves both stability and plasticity. This resolution of the stability-plasticity dilemma is attributed to the fact that task-specific knowledge is effectively captured by the adapters, while the backbone continuously learns invariant knowledge. Since all prior experiments discussed in this section were conducted using the CIFAR-100 alphabetical ordering, we have further evaluated all methods using additional orderings, specifically the coarse-grained ordering and the iCaRL ordering.

As shown in Figure 5, methods with adapters are indeed influenced by the varying difficulties associated with different orderings. Although the advantage persists, it diminishes to approximately 1% in some cases. Nonetheless, a general superiority remains evident, as most methods maintain a 3% margin over their non-adapter counterparts across all orderings. In every ordering scenario, adapters consistently exhibit the best overall performance in incremental learning.

**On Imagenet** This section evaluates our method’s performance across larger domain shifts by assessing its performance on ImageNet-Subset, a significantly larger dataset compared to CIFAR-100.

Our method faces certain limitations when applied to ImageNet, as selecting adapter hyperparameters becomes prohibitively expensive on such a large dataset with an average of 10 seeds, which led us to apply the CIFAR-100 hyperparameter setting directly to ImageNet. Additionally, the use of task-specific adapters slows down the generalization process compared to their non-adapter counterparts. This slowdown occurs because non-adapter methods do not distinguish between task-sharing and

{8}------------------------------------------------

| Method | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 | Task 10 |
|-|-|-|-|-|-|-|-|-|-|
| MAS | 80.4 | 73.6 | 74.4 | 71.3 | 72.8 | 72.9 | 73.5 | 72.1 | 72.7 |
| EWC | 80.3 | 74.6 | 72.0 | 67.8 | 63.2 | 63.9 | 63.6 | 61.4 | 60.8 |
| PathInt | 53.9 | 38.5 | 33.7 | 30.4 | 28.7 | 29.0 | 28.9 | 28.2 | 27.1 |
| LwF | 82.6 | 77.7 | 76.8 | 75.2 | 73.9 | 73.7 | 72.3 | 70.0 | 68.2 |
| LwM | 81.8 | 76.3 | 74.3 | 70.9 | 68.4 | 66.2 | 64.0 | 60.3 | 58.0 |
| <b>MAS-A</b> | <b>80.0</b> | <b>73.6</b> | <b>74.0</b> | <b>72.2</b> | <b>73.3</b> | <b>74.6</b> | <b>75.0</b> | <b>74.2</b> | <b>74.2</b> |
| <b>EWC-A</b> | <b>76.0</b> | <b>67.7</b> | <b>68.0</b> | <b>67.3</b> | <b>67.2</b> | <b>68.3</b> | <b>67.3</b> | <b>65.7</b> | <b>65.3</b> |
| <b>PathInt-A</b> | <b>76.9</b> | <b>68.3</b> | <b>67.3</b> | <b>65.4</b> | <b>65.5</b> | <b>67.1</b> | <b>67.1</b> | <b>65.0</b> | <b>65.0</b> |
| <b>LwF-A</b> | <b>83.8</b> | <b>79.8</b> | <b>78.3</b> | <b>76.2</b> | <b>74.2</b> | <b>73.0</b> | <b>71.6</b> | <b>69.0</b> | <b>67.2</b> |
| <b>LwM-A</b> | <b>82.8</b> | <b>75.9</b> | <b>73.9</b> | <b>70.6</b> | <b>67.8</b> | <b>65.9</b> | <b>63.2</b> | <b>59.4</b> | <b>56.9</b> |

Table 1: The average accuracy for regularization-based methods with or without adapters on ImageNet subset in task-IL. Methods without the "A" suffix represent the baseline, while those with the suffix include adapters. Following experiments conducted on CIFAR-100 in Section 4.2, these adapters are configured with bottleneck width 128.

task-specific patterns and thus are less impacted on ImageNet, which we only run for 50 epochs as mentioned in Appendix A.2.

While we recommend a more comprehensive study involving careful hyperparameter selection and extended training epochs, our current experimental results, as shown in Table 1, indicate that methods with adapters yield the best performance across all incremental tasks. Even with the hyperparameters from Section 4.2, which are tuned using CIFAR-100 and may not be optimal, methods with adapters still demonstrate non-trivial performance improvement.

**On modern IL methods** There is a growing body of incremental learning methods that incorporate task-specific components. We conducted experiments to address two key questions: 1) Can adapters be integrated with these methods, and does such integration further improve performance? 2) Does our adapter integration and training paradigm outperform existing adapter-based methods? For the first question, we integrate adapters with DualPrompt (Wang et al., 2022) and iTAML (Rajasegaran et al., 2020). For the second, we directly compare our approach with TAMiL (Bhat et al., 2023) by aligning our setup with theirs. The experiment details can be found in Appendix A. The results, shown in Table 2, indicate that integrating adapters boosts the original frameworks' performance by more than 1%, and our method outperforms TAMiL, a counterpart that uses attention modules.

### 4.3 ABLATION STUDIES

**The bottleneck width choice** The choice of adapter bottleneck width is crucial to the model performance. We selected EWC (Kirkpatrick et al., 2017) and LwF (Li & Hoiem, 2017) as baseline methods, due to their strong performance among weight-regularized and prediction-regularized methods, respectively. As illustrated in Figure 6, adapters with a bottleneck width of 256 consistently ranked among the top configurations.

**Training with backbone frozen** Various works propose adapter-like network architectures where the backbone is frozen (Liang & Li, 2024; Zhang et al., 2020a). While our framework adapts the backbone to capture task-invariant information and the adapters to capture task-specific information, we hypothesize that freezing the backbone does not support incremental learning. This is because the backbone still requires updates with new, task-inspecific knowledge, which may conflict with the prior knowledge acquired during pre-training. To investigate this, we conduct experiments using the LwF method, where both models are trained with adapters, but one freezes the backbone. Our results, shown in Table 2, demonstrate that the co-trained model, which does not freeze the backbone, outperforms the counterpart. This supports our hypothesis regarding the impact of freezing the backbone.

{9}------------------------------------------------

![Figure 6: Two line graphs showing performance on CIFAR-100 dataset. The left graph, 'EWC Performance on Different Bottleneck Widths', shows accuracy (%) vs. number of tasks (1 to 10) for EWC methods with bottleneck widths 16, 32, 64, 128, and 256. The right graph, 'LwF Performance on Different Bottleneck Widths', shows accuracy (%) vs. number of tasks (1 to 10) for LwF methods with the same bottleneck widths. In both graphs, accuracy generally decreases as the number of tasks increases, but higher bottleneck widths maintain higher accuracy for longer.](4e0ade2f41b66d5602160da5cc978274_img.jpg)

Figure 6: Two line graphs showing performance on CIFAR-100 dataset. The left graph, 'EWC Performance on Different Bottleneck Widths', shows accuracy (%) vs. number of tasks (1 to 10) for EWC methods with bottleneck widths 16, 32, 64, 128, and 256. The right graph, 'LwF Performance on Different Bottleneck Widths', shows accuracy (%) vs. number of tasks (1 to 10) for LwF methods with the same bottleneck widths. In both graphs, accuracy generally decreases as the number of tasks increases, but higher bottleneck widths maintain higher accuracy for longer.

Figure 6: The performance of EWC and LwF methods with different adapter bottleneck width choice on the CIFAR-100 dataset (alphabetical ordering) in task-IL. The suffix 16/32/64/128/256 indicates the method implemented with width 16, 32, 64, 128, and 256, respectively.

| Methods | Acc. | Methods | Acc. | Methods | Acc. | Methods | Acc. |
|-|-|-|-|-|-|-|-|
| DualNet | 88.2 | iTAML | 79.0 | TAMiL | 71.4 | LwF-A | <b>74.0</b> |
| DualNet-A | <b>89.3</b> | iTAML-A | <b>80.1</b> | Adapter+LwF | <b>74.7</b> | LwF-A-FrB | 72.9 |

Table 2: From left to right: DualNet vs. DualNet+adapter, iTAML vs. iTAML+adapter, TAMiL vs. Adapter+LwF (The best method-adapter pair we yielded), and LwF-A (co-trained) vs. LwF-A (Frozen Backbone). Test conducted on CIFAR-100, task-IL, top-1 accuracy averaged with 10 tasks is reported.

## 5 CONCLUSION

In this paper, we propose a network design consisting of two blocks: a backbone network for learning invariant features across all tasks and multiple adapters for modeling task-specific knowledge. The backbone and adapters are co-trained continuously in an incremental learning framework. Our extensive experiments conducted on CIFAR-100 and ImageNet, across various orderings and task scales, show that introducing task-specific adapters consistently improves the performance of all considered methods, and exhibits extensive compatibility with all of them. Consequently, we offer an effective solution to resolve the stability-plasticity dilemma for incremental learning, and we envision that future IL algorithms can be benefited from our work, a simple but effective integration of adapters.

## REFERENCES

- Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, and Tinne Tuytelaars. Memory aware synapses: Learning what (not) to forget. In *Proceedings of the European Conference on Computer Vision (ECCV)*, pp. 139–154, 2018.
- Akari Asai, Mohammadreza Salehi, Matthew E Peters, and Hannaneh Hajishirzi. Attempt: Parameter-efficient multi-task tuning via attentional mixtures of soft prompts. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pp. 6655–6672, 2022.
- Eden Belouadah and Adrian Popescu. Deesil: Deep-shallow incremental learning. In *Proceedings of the European Conference on Computer Vision (ECCV) Workshops*, pp. 0–0, 2018.
- Prashant Bhat, Bahram Zonooz, and Elahe Arani. Task-aware information routing from common representation space in lifelong learning, 2023. URL <https://arxiv.org/abs/2302.11346>.
- Rich Caruana. Multitask learning. *Machine learning*, 28:41–75, 1997.
- Arslan Chaudhry, Puneet K Dokania, Thalaiyasingam Ajanthan, and Philip HS Torr. Riemannian walk for incremental learning: Understanding forgetting and intransigence. In *Proceedings of the European Conference on Computer Vision (ECCV)*, pp. 532–547, 2018.

 Rest of paper (reference and Appendix) is removed.