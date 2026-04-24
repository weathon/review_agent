# Hyperparameters in Continual Learning: A Reality Check

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 6

## Abstract 
Continual learning (CL) aims to train a model on a sequence of tasks (i.e., a CL scenario) while balancing the trade-off between plasticity (learning new tasks effectively) and stability (retaining prior knowledge). The dominantly adopted conventional evaluation protocol for CL algorithms selects the best hyperparameters within a given scenario and then evaluates the algorithms
using these hyperparameters in the same scenario. However, this protocol has significant shortcomings: it overestimates the CL capacity of algorithms and relies on unrealistic hyperparameter tuning, which is not feasible for real-world applications. From the fundamental principles of evaluation in machine learning, we argue that the evaluation of CL algorithms should focus on assessing the generalizability of their CL capacity to unseen scenarios. Based on this, we propose a revised two-phase evaluation protocol consisting of a hyperparameter tuning phase and an evaluation phase. Both phases share the same scenario configuration (e.g., number of tasks) but are generated from different datasets. Hyperparameters of CL algorithms are tuned in the first phase and applied in the second phase to evaluate the algorithms. We apply this protocol to class-incremental learning, both with and without pretrained models. Across more than 8,000 experiments, our results show that most state-of-the-art algorithms fail to replicate their reported performance, highlighting that their CL capacity has been significantly overestimated in the conventional evaluation protocol.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper argued that the commonly used protocol of selecting hyperparameters for continual learning, which often select the best hyperparameter values within a given scenario and then evaluate the continual learning methods with these hyperparameters within the same scenario, is impractical in real-world applications. The authors then proposed an evaluation protocol consisting of a hyperparameter tuning phase and an evaluation phase on different datasets. The authors reported the performance variance of representative methods with this new protocol.

### Strengths
1. I appreciate the claim that the commonly used protocol of selecting hyperparameters for continual learning methods may not be optimal in applications, given that the old training samples are largely inaccessible.

2. The authors perform extensive experiments with a variety of continual learning methods under the proposed evaluation protocol.

### Weaknesses
This paper is essentially based on intuitive ideas and the empirical results are not very clear. It fails to cover many critical considerations in real-world applications.

1. The authors highlighted for many times that the two phases share the same scenario configuration (e.g., number of tasks) but are generated from different datasets. However, this consideration cannot fully reflect the possible differences across continual learning tasks, such as imbalanced classes per task, imbalanced training samples per class, blurred task boundaries, different task types, etc.

2. The experiments only consider class-incremental learning, rather than other typical scenarios such as task-incremental learning and domain-incremental learning. 

3. Although continual learning methods show some performance differences between the two phases, most of them have similar trends (Figures 4 and 8). This reduces the significance of the proposed protocol, since the advanced methods exhibit consistent advantages.

4. The authors further analyzed the training cost. I agree that the training cost is a critical issue for continual learning, but it is almost orthogonal to the hyperparameter issue and independent of the proposed evaluation protocol.

### Questions
My major concerns lie in the coverage of hyperparameter issues in real-world applications and its relevant to the training cost. Please refer to the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper aims to tackle the class-incremental learning problem, which is important to the machine learning field. The authors come up with a new evaluation protocol to investigate CIL methods of generalization. The authors have done extensive experiments to investigate the performance of different methods.

### Strengths
1.	This paper aims to tackle the class-incremental learning problem, which is important to the machine learning field.
2.	The topic of hyper-parameter robustness is interesting and has not been investigated in the CIL field
3.	The authors have done extensive experiments to investigate the performance of different methods.

### Weaknesses
1.	Although the authors have done extensive experiments in their new CIL setting, my major concern lies in the rationality of it. In typical machine learning scenarios, the training and testing data are i.i.d. sampled from the same training set. In other words, we train a model, evaluate it on the validation set, and utilize the best model to test on the test set (which has the same data distribution as the validation set). However, the authors advocate using the different data distributions for validation and testing, which is against common sense in typical machine learning. After reading the introduction, I would expect the authors to separate the original testing set into two disjoint sets for the current evaluation.
2.	Although the title is about continual learning, I find the experiments only focus on the class-incremental learning scenario. I would expect more interesting results in other continual learning settings like task-incremental learning, domain-incremental learning, learning with pre-trained vision-language models, etc.
3.	How to holistically evaluate a CIL algorithm has been also explored in another ICLR paper, i.e., [1], which extensively discusses the capability of different continual learning algorithms. In this aspect, this paper seems to advocate a typical case of CLEVA-Compass, making the contribution limited.
4.	The topic of this paper seems to be too narrow on the generalization ability, which is different from the typical CIL setting. I would suggest the authors name the protocol with some new name to avoid ambiguity.
5.  Finally, I also noticed a critical fact that leads to wrong conclusions. As the authors figure out from the main paper, DER is the most robust class-incremental learning algorithm. However, as they are using the PyCIL package, the reproduced DER is also not the full version, which does not implement the masking and pruning process in DER. See https://github.com/G-U-N/PyCIL/blob/31f2372d374c3f9a6c86d82b3c3ea4e0a880db63/models/der.py#L1C104-L1C124 (PyCIL's implementation), https://github.com/Rhyssiyan/DER-ClassIL.pytorch (DER official repo),
and 
https://arxiv.org/pdf/2103.16788 (Eq.8 to Eq. 10). The main reason, I assume, is that the masking and pruning functions are also not robust and cannot be reproducible. Hence, using such code for comparison obviously leads to unfair comparisons among different methods.

[1] CLEVA-Compass: A Continual Learning EValuation Assessment Compass to Promote Research Transparency and Comparability. ICLR 2022

### Questions
See Weaknesses

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
The paper proposes a more rigorous evaluation protocol for continual learning methods, emphasizing generalization to unseen scenarios. In contrast to the traditional approach, where hyperparameter tuning and performance measurement occur on the same sequential dataset, often without separation between test and validation sets, the authors propose separate hyperparameter tuning and evaluation phases. While the configuration of the continual learning scenario is identical for both stages, each uses a different dataset. The authors evaluate a number of class-incremental learning algorithms using this framework. Based on a range of experiments, they conclude that most modern class-incremental learning algorithms fail to achieve their reported performance under the new evaluation protocol.

### Strengths
The main strength of the paper is the extensive experimental evaluation conducted under a rigorous evaluation protocol, leading to an important insight—the superior performance of some recent class-incremental methods may be due to meta-overfitting to the particular evaluation set through hyperparameter optimization. Challenging the dominant, flawed approach to evaluating continual learning algorithms is a valuable contribution that will hopefully help steer the community towards a more disciplined approach and help identify methods that have a good chance of generalizing to real-world applications.

### Weaknesses
Poor presentation and structure are the main weaknesses of the paper. Figure 4 (b) is perhaps the most important result, yet it is not given a prominent place. Figure 3 and Figure 7 could easily be short tables. Figure 1 and 2 should be simplified and would work together as a side-by-side comparison. Limiting the analysis to the 10-task and 20-task scenario, respectively, would allow to simplify Figures 5 and 9 and make them easier to parse. BEEF should be dropped from the figures (and, arguably, the analysis) if the authors were not able to run it. The hyperparameter sets in B.1 and B.2 would be easier to read as tables.

Another weakness is the use of the number of parameters and training time, which are not reliable proxies for efficiency, as explained in Dehghani et al. 2021 (The Efficiency Misnomer). For an efficiency metric in continual learning, see Roth et al. 2023 (A Practitioner's Guide to Continual Multimodal Pretraining).

### Questions
For BEEF, have you tried a different implementation or different seeds?

In Figure 4 (b), why do almost all methods perform better on the unseen scenario?

What criterium did you use to select the methods for evaluation? Is it their availability in PyCIL and PILOT?

### Soundness
3

### Presentation
2

### Contribution
3
