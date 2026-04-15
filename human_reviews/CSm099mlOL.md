# Time-sensitive Weight Averaging for Practical Temporal Domain Generalization

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 3, 3

## Abstract
Temporal Domain Generalization (TDG) is a valuable yet challenging task that requires models to support temporal distribution shifts without access to future samples. Prior work utilized time-sensitive models that take timestamps as input or directly estimated optimal model parameters for each temporal domain. However, these methods were evaluated in oversimplified settings that do not scale to complex scenarios. To fundamentally enhance TDG's value for real-world applications, we propose three key principles for TDG method design: 1) Time-sensitive model, 2) Generic method, and 3) Realistic evaluation. Reflecting these guidelines, we propose Time-sensitive Weight Averaging (TWA), a simple yet effective approach to apply weight averaging (WA) of specialists for every temporal domain. For principle 1), we train a selector network to estimate the good coefficients to average weights based on timestamp input. For principle 2), TWA is inherently generic, as WA requires no modification to model architecture. For principle 3), we incorporate more realistic benchmarks into TDG, including CLEAR-10, CLEAR-100, Yearbook, and FMoW-Time, which feature complex data distributions and natural temporal shifts. Extensive experiments conducted on these benchmarks demonstrate the practical value of TWA, e.g., on CLEAR-10/100, TWA consistently improves accuracy over the baselines by up to 4%. We also demonstrate TWA boosts performance on common TDG benchmarks used in prior work. Lastly, we provide theoretical insights behind the outstanding performance of TWA.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper studies a specific case of domain adaptation over a sequence of tasks. Authors assume that distribution between consecutive domain change smoothly. Under this assumption, they propose Time-sensitive Weight Averaging, a learning approach based on weight averaging. The authors also present an experimental study of the proposed algorithms.

### Strengths
The paper is well written and easy to follow.

The use of weight averages is one of the most common proposals to adapt to changes over time. The paper makes a good review of state-of-the-art methods that allow to understand the contributions of the proposed method.

The idea of using the time stamp is new and very interesting. In addition, the proposed methods provide theoretical guarantees.

The authors perform extensive numerical experiments

### Weaknesses
It will be great to know how much this assumption is agreed and violated on individual tasks on experiments. 

It would be nice if the theoretical guarantees appeared in the paper instead of in the appendices.

I think one improvement the authors can consider is to add some discussions about the computational complexity and running times.

having explanations about methods used for comparison could  significantly increase the readability.

The authors' assumption about the distribution is the usual assumption made in supervised classification under concept drift (References below). I think the authors should mention that this is a common assumption (gradual drift assumption) in supervised classification under concept drift as well as describe how the methods that make this assumption adapt to changes in the distribution. 

Žliobaitė, I. (2010). Learning under concept drift: an overview. arXiv preprint arXiv:1010.4784.

Elwell, R., & Polikar, R. (2011). Incremental learning of concept drift in nonstationary environments. IEEE Transactions on Neural Networks, 22(10), 1517-1531.

### Questions
How does the method perform if you use a dataset with tasks in which the distribution does not change? For example MNIST dataset. 
Or what happens if the distribution changes too fast?
Could the method work well but the theoretical guarantees would not be true?

Can the method be used on text datasets?The authors use the image datasets from the wild time library. Could the method be extended to the other three datasets?

Can the method be used for any loss function?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This article addresses the issue of enhancing the practicality of Temporal Domain Generalization (TDG). This paper comprehensively considers various aspects of TDG tasks, including methods, datasets, evaluation settings, comparisons with previous methods and more. 
It makes the following contributions:
- It analyzes the limitations of previous TDG approaches and summarizes three design principles for TDG methods: 1) Time-sensitive model, 2) Generic method, and 3) Realistic evaluation.
- In the context of TDG tasks, the article improves the Weight Averaging method by automatically learning the optimal weight averaging strategy for different time intervals.
- The authors enhance the TDG benchmark by incorporating more realistic datasets, namely CLEAR-10, CLEAR-100, Yearbook, and FMoW-Time, into the TDG setting.
- Comprehensive evaluation results are provided, including comparison with other methods using both new and previous datasets, and results on both vision and language tasks.

### Strengths
- This paper provides valuable analysis, summary, and design principles for improving the practicality of Temporal Domain Generalization. The proposed three design principles are a meaningful step towards enhancing the practicality of TDG.
- This paper proposed a novel TDG method, TWA, which is the first to apply weight averaging toTDG.  And TWA further makes the weight averaging learnable to adapt to the smooth distribution shift setting of TDG. 
- This paper is the first one to evaluate TDG methods on larger and more complex datasets, which contributes to the overall improvement of TDG's practicality in complex application scenarios.  Evaluation across new and common TDG benchmarks, along with efficiency assessments, allows us to obtain a comprehensive assessment of the proposed method.
- Strong performance. Comprehensive experiments are conducted, outperforming other methods across various benchmarks and tasks with good efficiency. The proposed TWA is a "simple yet effective" method that can be easily applied and adapted to different tasks, models, and datasets.

### Weaknesses
Like most other TDG works, TWA is also limited by some common problems within the TDG task:
- TWA also relies on the “smooth distribution shift” assumption, which could limit its potential applications.
- Evaluations are limited to relatively “simple” tasks, e.g. most TDG methods are evaluated with classification and regression tasks only. It’s unclear whether these TDG methods could still work when generalized to more complex tasks, such as image segmentation or common object detection.

### Questions
Overall, I have a positive impression of this paper. The limitations I listed are common for most existing TDG works, which do not hinder TWA from being a novel TDG method. And I got questions about the implementation details:

- When you are applying TWA with different tasks, datasets or models, are you using a constant “number of snapshots”? If not, is there any strategy to select the best number of snapshots for different application settings? It would be helpful if there could be results of “number of snapshots vs. task complexity”.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Temporal Domain Generalization (TDG) aims to tackle temporal distribution shifts in models without future sample access. Traditional TDG methods, often oversimplified, were either time-sensitive or tried to estimate optimal model parameters for each temporal domain. To improve TDG's practical applicability, the authors introduce three main principles: a time-sensitive model, a generic method, and realistic evaluation. Following these principles, they present Time-sensitive Weight Averaging (TWA), which uses weight averaging of specialists for every temporal domain and a selector network trained on timestamps. TWA's effectiveness is confirmed through experiments on various benchmarks, showing up to a 4% improvement in accuracy over traditional methods.

### Strengths
1. The authors emphasize the importance of addressing Temporal Domain Generalization (TDG) in a more comprehensive manner. They introduce three key principles for TDG method design, which ensure that the approach is time-sensitive, generic, and evaluated under realistic conditions.
2. The TWA approach is simple and effective. It leverages weight averaging for every temporal domain and is complemented by a selector network that estimates the best coefficients based on timestamp input.
3. TWA has been tested extensively on multiple realistic benchmarks and demonstrated great performance.

### Weaknesses
1. The motivation is a little weak in my opinion. Specifically, the authors criticize that existing methods rely too much on time-sensitive mechanisms, which may cause troubles for large generic models. However, given the task is **temporal** DG, it is quite intuitive to rely on time-sensitive information, and authors seem to agree with the strength of using the time-sensitive information. For most of the generic models like ChatGPT, the problem of DG may not be a big deal. Applying weight averaging to these foundation models is also a resource-consuming task.

2. The technical design is a little trivial and odd to me. The motivation says existing methods rely too much on temporal information, we should use weight averaging instead. However, the proposed method still seems to leverage a selector network to model the trajectory of the distribution shift over time. In addition, the model is built upon SWAD (Cha et al., 2021) with some additional contributions to learning the averaging coefficient.

3. Finally, the presentation of the manuscript still has a large room for improvement. For example, all the notations in Figure 1 are not introduced, which makes the visual presentation not self-contained. In addition, some sentences are hard to understand, e.g., "They assumed that understanding say how a mobile phone’s appearance has changed in the past may help predict future changes."

### Questions
1. What is the definition of model snapshots? Is it the same thing as the model parameters?
2. Following the previous question, if the model snapshot equals the model parameter, how can it be sampled? Is the model a Bayesian Neural Network?
3. One major contribution that the authors claim is "The method can be easily combined with various architectures and tasks, requiring as few architecture modifications as possible". Therefore, it makes people expect there may be some experiments showing the proposed method is flexible with different architectures. In addition, runtime comparisons with other methods can also be demonstrated. However, Table 11 only contains the runtime comparison against itself. 
4. In the experiments, why the model is heavily compared with many DG methods, like ERM, MTL, GROUPDRO, etc? The major comparison between the proposed method and TDG methods is better to be placed in the main content instead of the appendix.
5. The most up-to-date TDG method DRAIN is not compared in many datasets. Could the authors explain what exactly the "highly task-specific design" is in DRAIN (Bai et al., 2022)?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper extends weight averaging technique for domain generalization in non-stationary environments. Specifically, the authors employ the Time2Vec model to compute time-sensitive weights, facilitating the creation of an optimal model at each time step by leveraging existing model snapshots. Experimental findings across various real-world datasets consistently showcase the superior performance of the proposed approach compared to baseline methods.

### Strengths
- This paper tackles an important issue of temporal domain generalization in machine learning under distribution shift.

- The experiment covers more real-world dataset compared to existing works for temporal domain generalization.

- Code is made available for the purpose of ensuring reproducibility.

### Weaknesses
- The authors assert that they introduce a comprehensive benchmark, encompassing several realistic datasets and baselines for temporal domain generalization. However, the comparison with existing temporal domain generalization methods is conducted solely on the Portrait dataset. For other real datasets such as CLEAR, FMoW, arXiv, and HuffPost, the authors only provide comparisons with the weight-averaging method (SWAD [1]). It is recommended that the authors expand their analysis to include results from other methods, particularly those tailored for temporal domain generalization (e.g., GI [2], LSSAE [3], DDA [4], DRAIN [5], DPNET [6]), on these datasets to bolster the strength of their contribution. (NOTE: This is a significant concern. I am open to revising my evaluation if the authors can address and resolve this issue.)

- The proposed method does not strictly adhere to the problem formulation outlined in Equation (1). Notably, while the original problem is cast as a bi-level optimization, the proposed algorithm utilizes a two-phase training approach to optimize the two terms sequentially. The authors should clarify the rationale behind their model design.

- The two-stage training approach employed by the proposed method raises concerns about both time (for training the selector network) and space (for storing model snapshots) complexity.

- The explanation of how the theoretical results influence the development of the proposed method is somewhat lacking in clarity. From my point of view, the current theoretical findings appear to be more apt for providing a theoretical assurance of TWA rather than serving as the primary guiding influence on the algorithm's design.

- Could the authors offer more detailed insights to distinguish their work from SWAD? From my point of view, the technical contribution seems somewhat limited. It appears that the primary contribution lies in the utilization of Time2Vec to adapt the weight-averaging method to the temporal-shift scenario.

References:

[1] Cha, Junbum, et al. "Swad: Domain generalization by seeking flat minima." Advances in Neural Information Processing Systems 34 (2021): 22405-22418.

[2] Nasery, Anshul, et al. "Training for the future: A simple gradient interpolation loss to generalize along time." Advances in Neural Information Processing Systems 34 (2021): 19198-19209.Nasery, Anshul, et al. "Training for the future: A simple gradient interpolation loss to generalize along time." Advances in Neural Information Processing Systems 34 (2021): 19198-19209.

[3] Qin, Tiexin, Shiqi Wang, and Haoliang Li. "Generalizing to Evolving Domains with Latent Structure-Aware Sequential Autoencoder." International Conference on Machine Learning. PMLR, 2022.

[4] Zeng, Qiuhao, et al. "Foresee What You Will Learn: Data Augmentation for Domain Generalization in Non-Stationary Environments." arXiv preprint arXiv:2301.07845 (2023).

[5] Bai, Guangji, Chen Ling, and Liang Zhao. "Temporal Domain Generalization with Drift-Aware Dynamic Neural Networks." arXiv preprint arXiv:2205.10664 (2022).

[6] Wang, William Wei, et al. "Evolving Domain Generalization." arXiv preprint arXiv:2206.00047 (2022).

### Questions
Questions are given in the Weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
