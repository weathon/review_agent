# Anomaly-Gym: A Benchmark for Anomaly Detection in Embodied Agent Environments

- Decision: Reject
- Scores: 6, 6, 2, 2

## Abstract
Research on anomaly detection in reinforcement learning settings is sparse. Only a handful of methods have been proposed that - due to the absence of established evaluation scenarios - are evaluated on simple, small-scale, and self-proposed environments. This not only results in poor comparability but also leads to a limited understanding of the strengths and weaknesses of current approaches, rendering their applicability in real-world scenarios questionable. We address this problem by introducing Anomaly-Gym, a comprehensive evaluation suite for anomaly detection in reinforcement learning settings. In contrast to prior work, Anomaly-Gym is based on principled design criteria that disentangle evaluation from methodology. By enforcing specific constraints on the environments and anomalies considered, we propose a broad spectrum of evaluation data that covers both simulated and real-world tasks. In total, our benchmark features 10 different environments, 25 anomaly types, 4 strength levels, as well as multiple sensor modalities. We demonstrate the importance of these different aspects in a series of experiments on pre-generated datasets. For instance, we show that simple methods, while generally neglected in previous work, achieve near-perfect scores for settings with observational disturbances. In contrast, detecting perturbations of actions or environment dynamics requires more complex methods. Our findings also highlight current challenges with anomaly detection on image data and provide directions for future research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors introduce Anomaly-Gym, a comprehensive benchmark suite designed to evaluate anomaly detection methods in reinforcement learning (RL) settings. In contrast to existing work, Anomaly-Gym is grounded in principled design criteria that explicitly separate evaluation from algorithmic methodology. By imposing clear constraints on the structure of environments and the nature of anomalies, the benchmark offers a diverse and systematic collection of evaluation data encompassing both simulated and real-world tasks. Overall, Anomaly-Gym comprises 10 environments, 25 anomaly types, 4 anomaly strength levels, and multiple sensor modalities. Extensive experiments on pre-generated datasets highlight the significance of these design choices and their impact on performance assessment.

### Strengths
It is non-trivial to establish a clear desiderata that enhances the reproducibility of anomaly detection methods in reinforcement learning (RL). Standardizing a benchmark for this domain is inherently challenging, yet this paper makes a strong contribution by laying a solid foundation for doing so within the RL context. Explicitly:

* I was in particular happy with the attempt of standardizing the agent's policy, which is something other fields do not have to consider. I think the Anomaly Strength is an interesting score to consider, since it empirically gives a level of 'difficulty' with respect to the task relevant features still present in the environment. I can see that this may lead to a more useful difficulty score, especially since the code includes offline data.
* I appreciated the usage of Table 1, simply showing that there is a need for standardizing RL anomaly detection. 
* The offline data appears to be well organized, I can see how this can be very accessible, since many benchmarks eventually have RL policies that rely on outdated packaging.

### Weaknesses
**Review**

I think that this is a good work, unfortunately what is stopping me from a clear accept is that there is potential for limited impact and that the paper seems unpolished in certain areas.

* A critical issue with this work is that the Real-World RL Suite [1] appears to not be mentioned anywhere in the text. This would perhaps be the largest competitor, although it is an older work, the contribution is slightly reduced since part of the contribution is already available in [1].
* Line 94, I'm not sure a discussion over adversarial RL is necessary. It does not seem to be connected to any of the anomaly types, and there is no mention in the anomaly types about a purposeful malicious injection via AutoAttack or PGD etc. 
* The appendix has quite a few typos Ex: (1,) move the cart left/righ,  Page 19 has formating issues, Line 1034 (¡10GB),
* Line 150, Types of Anomalies, seems to be out of place, since Sec 5.2 is where the actual anomalies are discussed.
* Can this be rewritten explicitly: ''Given its interdisciplinary scope (RL, time-series, CV, ...) and the combinatorial space of design
choices (environments, anomalies, onset schemes, ...), several limitations are inherent to this work.''

* Is the AUROC score evaluated on batches of entire trajectories? Or is it computed a different way such as via the score of a balanced dataset of anomaly and non-anomaly transitions? There are many ways to do this, and I feel that this needs to be explicit since there is a lot of variety in the random injections affecting future states and so on in the RL setting. I feel as though there needs to be a short discussion on how the AUROC score was computed.

* Is it possible to do a final review of the citations used and standardize to respective conferences whenever possible ex: [2,3,4,5...] (There are more online, but this is what I could find after a bit of reading).

Some thoughts on potential limited impact:
* It appears most, if not all environments are 'solved' via at least one of the detection methods (PEDM seems to have solved most on its own). I am certain that anomaly detection in the RL setting is far from solved, but is it possible to highlight cases where a major future contribution might exist? Besides image observations (and reducing the tails of the delay scores), it appears that there is no more improvement to be made on the custom datasets, which may significantly reduce the need for this benchmark. I also suspect that Figure 3 could clouding some interesting insights since the easier environments may skew the results here. If so, the paper would benefit from not summarizing all environments into a single figure for better analysis. I also suggest elaborating the image observation results further.
*  Going a bit further in the previous point: 'Our findings
also highlight current challenges with anomaly detection on image data and provide directions for future research.', is a statement I feel the paper needs to improve upon. Section 6.3 does not give much insight to construction of an improved anomaly detector, for example: 'KNN achieves near perfect scores, slightly outperforming all other methods,
showing that method choice can be tailored to failure modality.' does not give the reader a hypothesis on why this might be the case or what could a future reader do to improve, i.e. Why do the tested methods succeed or fail? What do they fail to do? etc. 

[1] Dulac-Arnold, G., Levine, N., Mankowitz, D. J., Li, J., Paduraru, C., Gowal, S., and Hester, T. An empirical investigation of the challenges of real-world reinforcement learning. 2020.

[2] Aaqib Parvez Mohammed, & Matias Valdenegro-Toro (2021). Benchmark for Out-of-Distribution Detection in Deep Reinforcement Learning. In Deep RL Workshop NeurIPS 2021.

[3] Jonathan Balloch, Zhiyu Lin, Mustafa Hussain, Aarun Srinivas, Robert Wright, Xiangyu Peng, Julia Kim, and Mark Riedl
NovGrid: A Flexible Grid World for Evaluating Agent Response to Novelty
Proceedings of the AAAI Spring Symposium on Designing Artificial Intelligence for Open Worlds (2022).

[4] Mohamad H Danesh, Alan Fern (2021). Out-of-Distribution Dynamics Detection: RL-Relevant Benchmarks and Results. ICML 2021 Workshop on Uncertainty and Robustness in Deep Learning

[5] Geigh Zollicoffer, Kenneth Eaton, Jonathan C Balloch, Julia Kim, Wei Zhou, Robert Wright, & Mark Riedl (2025). Novelty Detection in Reinforcement Learning with World Models. In Forty-second International Conference on Machine Learning.

[6] Goel, S., Wei, Y., Lymperopoulos, P., Churá, K., Scheutz, M., & Sinapov, J. (2024). NovelGym: A Flexible Ecosystem for Hybrid Planning and Learning Agents Designed for Open Worlds. In Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems (pp. 688–696). International Foundation for Autonomous Agents and Multiagent Systems.

### Questions
* Correct me if I am wrong, but it appears that grid-worlds are not considered? 
* Is it possible to add a bit more conversation on the choice of anomaly selection? I see the appendix has a bit of this, but I think it would be beneficial on why these anomalies fit the desiderata in the main text.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces a new benchmark for anomaly detection in reinforcement learning, focusing on environments involving embodied agents. The main motivation behind this work is the lack of existing benchmarks for this problem. The proposed benchmark comprises 10 environments, 25 anomaly types and 4 severity levels distributed across various sensor modalities. A pre-generated dataset is also provided. The paper presents a number of classic and neural network-based baselines, evaluated using standard metrics, and provides a thoughtful analysis. Interestingly, the evaluations show that image-based anomalies remain challenging.

### Strengths
Benchmark: The paper addresses a research gap. Current benchmarks do not address the issue of anomaly detection in RL, particularly with regard to embodied agents. In this context, the contribution of 10 environments, 25 anomaly types and 4 severity levels is significant.

Evaluation: Furthermore, the evaluated method is clearly distinct from the evaluation protocol. This allows a wide range of approaches to be evaluated using the proposed benchmark. 

Baselines: The baseline evaluations are interesting. They demonstrate that simple methods are effective in identifying easy anomalies. They also cover a wide range of methods for vector and image observations. 

The paper is well written and easy to follow. The motivation is clear and positioned appropriately alongside the related work. The benchmark is clearly described, and the appendix provides additional helpful details on the environments. Overall, the paper is very readable.

### Weaknesses
Real world: Including more real-world evaluation would make a major contribution to the paper. It is certainly not easy, but the current approach is limited. There is the URTdE-Reach evaluation, but it is the only one. 

In terms of novelty, the paper does not present any new methods or incremental contributions to existing approaches. While the benchmark is interesting and necessary, it does not include a new method to address challenging situations. Including a method that addresses, even partially, the limitations of existing baselines would be a valuable addition to the paper. 

Anomaly difficulty: There is no proof or evidence for Eq. (4). This point needs further clarification.

### Questions
How does the proposed benchmark compare with other anomaly benchmarks? Would it not make sense to build on existing benchmarks? 

Is there any taxonomy for the 25 anomaly styles? 

Does the proposed benchmark allow for real-time interaction, or does it only work offline?

More evaluation on the anomaly annotation would be helpful.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
Anomaly detection in reinforcement learning is a critical research area, yet it lacks a unified testbed. This paper proposes Anomaly-Gym, a comprehensive evaluation suite designed for anomaly detection in reinforcement learning settings. Anomaly-Gym comprises 10 distinct environments, 25 anomaly types, and 4 strength levels. The authors implement several baselines, report their performance across these environments, and discuss future research directions.

### Strengths
- The paper is clearly motivated and well-written.
- The authors have implemented several baselines, evaluated them across these environments, and provided further discussion—while also pointing out future research directions.
- The code and associated dataset have been made open-source.

### Weaknesses
- Anomaly-Gym comprises only 10 environments in total, and most tasks—even CAR-LaneKeep—are relatively simple. Its lack of diversity means it is insufficient to serve as a robust testbed for anomaly detection in reinforcement learning. Furthermore, since some simple methods achieve near-perfect scores on certain tasks, these tasks may not effectively support the design of improved future algorithms.

- The literature review in this paper is not exhaustive. Several highly relevant works are missing, as referenced in [1, 2, 3, 4].

- Table 1 lists some related works on AD for RL. However, these methods are not implemented as baselines in the experiments, which is a major weaknesses for a benchmark-focused paper. Additionally, the code for [5] is publicly available on GitHub.

- The paper provides no explanations for the metrics it uses, such as AU-ROC, AU-PR, FPR95, VUS-ROC, and VUS-PR. This omission prevents the paper from being comprehensive.

[1] Müller, Robert, et al. "Towards anomaly detection in reinforcement learning." Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems. 2022.

[2] Wang, Chen, et al. "OIL-AD: An anomaly detection framework for sequential decision sequences." arXiv preprint arXiv:2402.04567 (2024).

[3] Kazari, Kiarash, Ezzeldin Shereen, and György Dán. "Decentralized Anomaly Detection in Cooperative Multi-Agent Reinforcement Learning." IJCAI. 2023.

[4] Sedlmeier, Andreas, et al. "Policy entropy for out-of-distribution classification." International Conference on Artificial Neural Networks. Cham: Springer International Publishing, 2020.

[5] Zhang, Hongming, et al. "A Distance-based Anomaly Detection Framework for Deep Reinforcement Learning." Transactions on Machine Learning Research.

### Questions
- The paper refers to these environments as "embodied agent environments," but I find little connection between them and embodied AI. Could the authors explain this classification?

- Please refer to the "Weaknesses" section.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper establishes Anomaly-Gym, a comprehensive evaluation suite for anomaly detection in reinforcement learning settings. The authors design specific constraints on the environments and
anomalies considered that covers a broad spectrum of evaluation data and covers both simulated and real-world tasks.  Anomaly-Gym features 10 different environments, 25 anomaly types, 4 strength levels, as well as multiple sensor modalities. The authors highlight current challenges with anomaly detection on image data and provide directions for future research.

### Strengths
1. Overall contribution: The authors make an important contribution to anomaly detection by proposing a principled framework for investigating anomalies. Specifically, they introduce Anomaly-Gym, a suite comprising 10 diverse tasks and 25 types of anomalies, designed to rigorously test, evaluate, and compare different aspects of anomaly detection in reinforcement learning.

2. Empirical evaluation: The authors demonstrate the utility of Anomaly-Gym through a series of experiments, evaluating existing detection methods and baselines across different environments, anomaly types, strength levels, and observation modalities. The results highlight the need to differentiate these factors and reveal the limitations of current approaches, providing valuable insights for future research.

### Weaknesses
While the effort to develop such an evaluation platform is highly commendable, I have several concerns regarding its motivation, scope, and validation.

1. **Motivation and scope:** Studying anomalies in a unified manner could indeed be valuable. However, I am not fully convinced by the premise that a centralized platform is necessary or even feasible for investigating anomalies across diverse applications and embodiments in a unified manner. Each application domain exhibits distinct characteristics, making it extremely challenging to design, maintain, and continuously update corresponding environments under a single framework. As a result, it is unclear how this effort could benefit the broader community in a sustainable way, rather than being a one-off implementation.

2. **Missing baselines and related benchmarks:** The paper does not include comparisons with existing methods and benchmarks such as SafeBench [1] and AED [2]. In particular, SafeBench emphasizes safety, a critical aspect of anomaly evaluation, and covers a broader spectrum of driving scenarios than the proposed platform. Including these comparisons would significantly strengthen the empirical foundation and contextual relevance of this work.
    
    [1] Xu et al., SafeBench: A Benchmarking Platform for Safety Evaluation of Autonomous Vehicles, NeurIPS 2022
    
    [2] Yeh et al., AED: Adaptable Error Detection for Few-Shot Imitation Policy, NeurIPS 2024
    
3. **Definition of Anomaly Strengths:** The authors introduce **Anomaly Strengths** as a quantitative measure to compare anomalies across different environments. I believe it is a great attempt. However, this abstraction may oversimplify the notion of anomaly severity. For instance, in SafeBench [1], certain anomalies are safety-critical and could lead to catastrophic outcomes (e.g., loss of life). In such cases, it is difficult to justify reducing anomaly severity to a single-dimensional metric, which may fail to capture the true impact and diversity of anomalies across domains.

4. **Lack of informative insights:** In the conclusion of the experimental section, the authors state that anomaly strength and type significantly affect detection performance, highlighting the challenges associated with image-based observations and threshold selection. However, similar findings could already be derived from existing benchmarks such as SafeBench, without the need to establish a unified framework. Therefore, the added value of the proposed unified benchmark remains unclear.

### Questions
1. Is a centralized evaluation platform truly necessary or feasible for investigating anomalies across diverse applications, given the unique characteristics and maintenance challenges of each domain?

2. How does the proposed platform compare empirically to existing benchmarks such as SafeBench and AED, which already cover critical aspects of anomaly detection?

3. Does the Anomaly Strengths metric adequately capture the severity and real-world impact of anomalies, particularly in safety-critical scenarios?

4. What is the added value of the unified benchmark, given that the reported findings could be obtained from existing benchmarks without consolidating multiple environments?

### Soundness
2

### Presentation
2

### Contribution
2
