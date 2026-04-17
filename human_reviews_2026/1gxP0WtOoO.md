# Adaptive Conformal Guidance for Learning under Uncertainty

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 8

## Abstract
Learning with guidance has proven effective across a wide range of machine learning systems. Guidance may, for example, come from annotated datasets in supervised learning, pseudo-labels in semi-supervised learning, and expert demonstration policies in reinforcement learning. However, guidance signals can be noisy due to domain shifts and limited data availability and may not generalize well. Blindly trusting such signals when they are noisy, incomplete, or misaligned with the target domain can lead to degraded performance. To address these challenges, we propose Adaptive Conformal Guidance (AdaConG), a simple yet effective approach that dynamically modulates the influence of guidance signals based on their associated uncertainty, quantified via split conformal prediction (CP). By adaptively adjusting to guidance uncertainty, AdaConG enables models to reduce reliance on potentially misleading signals and enhance learning performance. We validate AdaConG across diverse tasks, including knowledge distillation, semi-supervised image classification, gridworld navigation, and autonomous driving. Experimental results demonstrate that AdaConG improves performance and robustness under imperfect guidance, e.g., in gridworld navigation, it accelerates convergence and achieves over $\times 6$ higher rewards than the best-performing baseline. These results highlight AdaConG as a broadly applicable solution for learning under uncertainty.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes AdaConG, that dynamically modulates guidance signals (e.g., from teacher models, pseudo-labels) based on their uncertainty. This uncertainty is quantified using split conformal prediction (CP), which is embedded into the training loop to re-weight the guidance loss. The method is simple, broadly applicable, and validated across diverse tasks (including knowledge distillation, SSL, and imitation-guided RL), showing improved robustness and performance under imperfect guidance.

### Strengths
- extensive and diverse experimental setups (also good results)
- simple, easy to re-implement
- minimal computational overhead compared to MC-dropout

### Weaknesses
- lack qualitative analysis. For instance, providing visualizations of why AdaConG assigns high or low uncertainty to specific inputs (e.g., showing example images or states and their corresponding CP-derived weights) would offer deeper insight into the method's behavior.
- In RL exps, the comparison is limited to SAC (a standard RL baseline) and IBRL/Soft IBRL (which rely on Q-values). The paper would be strengthened by comparing against other methods that are also designed to handle suboptimal or noisy expert guidance, beyond just Q-value comparison (maybe some baselines in [1]).
- In SSLs, the labeled dataset is used both to construct the calibration set and to compute the training loss. This appears to violate the standard split conformal inprediction assumption. Does it affect the statistical guarantees of CPs? 


[1]  Agarwal et al. , Beyond Tabula Rasa: Reincarnating Reinforcement Learning. NeurIPS, 2022.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors address the problem of noisy and unreliable guidance signals in machine learning. To mitigate this issue, they propose AdaConG, a strategy that reduces the model's reliance on guidance when misleading signals are present and adaptively adjusts the influence of guidance signals. They conduct comprehensive experiments across four popular ML tasks, demonstrating that AdaConG is more robust to guidance uncertainty than baseline methods and generalizes well across diverse tasks and scenarios.

### Strengths
1. The experimental design is comprehensive and rigorous, effectively validating the proposed method's robustness to misleading signals.

2. All experimental results report mean and standard deviation over multiple runs, demonstrating scientific rigor and reproducibility.

3. The core idea of improving learning quality by modulating the uncertainty of guidance signals is inspiring, with valuable practical implications across a wide range of tasks.

4. Incorporating conformal prediction to inform training dynamics is novel and represents an underexplored direction that could benefit the community.

### Weaknesses
1. I found some of the tables challenging to parse at a first glance (e.g., Table 1 and 4). Reformatting them to be more self-explanatory would enhance the paper's readability.

2. I only found efficiency analysis for KD task. I would recommend the authors also include (training time, computation overhead) comparison in other tasks with key baselines.

### Questions
1. What is the baseline for the last section of Table 1? It appears that EA-KD (line 314) outperforms the proposed method (line 301) in half of the cases. Could you clarify this comparison and discuss why EA-KD shows superior performance in these instances?

2. In Fig. 2(d), could you provide the prediction uncertainty curves for other baseline methods? This would help better contextualize the advantages of AdaConG in uncertainty calibration.

3. In Table 4 (last row), under what setting is the result for "Student (RGB)" obtained? Specifically, is this model trained from scratch, or does it use some form of pre-training/initialization?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Adaptive Conformal Guidance (AdaConG), to consider the uncertainty against noisy guidance signals for different tasks. Simply, the method dynamically modulates the influence of guidance signals based on the uncertainty quantified by split conformal prediction (CP), which enabling the models to reduce reliance on potentially misleading signals. The proposed method is evaluated on different tasks, including knowledge distillation, semi-supervised image classification, gridworld navigation, and autonomous driving. The experiments demonstrate the effectiveness of the proposed method

### Strengths
1. The paper is well written and easy to follow.

2. The idea of the paper is straightforward.

3. The experiments on diverse tasks demonstrate the effectiveness of the proposed method.

### Weaknesses
1. As discussed in the abstract and introduction, the learning-with-guidance methods are easy to be influenced by noisy guidance lead by domain shifts or limited data, where considering the uncertainty is important. However, in the experiments, the datasets and cases are used for evaluation seems more simple, without considering distribution shifts or limited data. I suggest to add some experiments on these difficult cases, such as datasets like CIFAR-C, ImageNet-C.

2. The method is also evaluated on small datasets and backbones. It is not clear how it works on larger backbones like ViT/B, ViT/L, and larger datasets like ImageNet.

3. Lack of comparisons with CP alternatives in the experiments

4. I assume the proposed method would lead to more computational costs for training and calibrating, which are not discussed in the main paper.


larger backbones

larger datasets

### Questions
1. The proposed method requires an amount of available target data for calibration during training. However, the target data may not available or hard to collect in some real applications, so the available target data is limited. How could the proposed method handle these cases? How many target data does the method required for calibration and training for different tasks?

2. I'm also curious about whether the proposed method can be extended to self-supervised learning.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes Adaptive Conformal Guidance. This is a simple, plug‑in mechanism that uses split conformal prediction (CP) to quantify the uncertainty of guidance signals (teacher logits in KD, pseudo‑labels in SSL, imitation policies in RL) and adaptively modulate how strongly the learner follows them. Concretely, it builds a prediction set (or interval), maps its size/measure to an uncertainty score u, transforms it to a weight and uses it to attenuate a guidance loss (KD/SSL) or to arbitrate between policies (RL). Experiments on CIFAR‑100 KD, SSL (CIFAR‑10/100, STL‑10), MiniGrid navigation, and autonomous driving steer prediction show consistent gains, including large improvements when the teacher underperforms under shift. The paper's appeal in my opinion is breadth and simplicity of the work (of course, it also helps that it is well written and articulated).

### Strengths
(1) The idea is elegant, simple and scales well across SSL, KD and imitation RL.

(2) Experiments are rigorous and covers a strong empirical breadth. 

(3) It seems to be very lightweight and model agnostic compared to MC-dropout-style approaches. 

(4) The paper is well written and articulated.

### Weaknesses
Weaknesses and questions:

(1) From my understanding, w(s) weighs a KL guidance loss; in Experiments, however, looks like w(s) chooses actions (stochastic arbitration) and a hard argmax variant. These are different algorithms! Please explicitly state it (if it was not a mistake).

(2) \gamma is being overloaded ((i) RL discount, (ii) temperature in h(u), EMA smoothing factor all use \gamma). 

(3) Similarly s is being used for score and state. 

(4) Coverage guarantees require exchangeability, several passages imply robustness and in RL the CP set measures self‑consistency of a policy, not correctness. I would rephrase those sentences and state conditions.

(5) In the SAC baseline, generally SAC is meant to be continuous, but how is it using discrete actions? 

(6) f_s is undefined (should it instead be \pi_R)? 

(7) Typos I was able to catch (there might be more): "Disucssions" (A.6), "mdoel" (Page 9), "is is" (Page 5). 

(8) Does the claimed 0.08 ms/sample latency not contradict the epoch timings (\Delta ~= 0.17 s/epoch = ~0.003 ms/sample for 50k samples)?

### Questions
Please see the Weaknesses section.

### Soundness
3

### Presentation
4

### Contribution
3
