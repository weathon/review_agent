# Nasty Adversarial Training:  A Probability Sparsity Perspective for Robustness Enhancement

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
The vulnerability of deep neural networks to adversarial examples poses significant challenges to their reliable deployment. Among existing empirical defenses, adversarial training and robust distillation have proven the most effective. In this paper, we identify a property originally associated with model intellectual property, i.e., probability sparsity induced by nasty training, and demonstrate that it can also provide interpretable improvements to adversarial robustness. 
We begin by analyzing how nasty training induces sparse probability distributions and qualitatively explore the spatial metric preferences this sparsity introduces to the model. Building on these insights, we propose a simple yet effective adversarial training method, nasty adversarial training (NAT), which incorporates probability sparsity as a regularization mechanism to boost adversarial robustness. Both theoretical analysis and experimental results validate the effectiveness of NAT, highlighting its potential to enhance the adversarial robustness of deep neural networks in an interpretable manner.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposed a method for improving adversarial training which incorporates nasty training. The method uses two networks: a target network which we would like to be more robust to adversarial example and a vanilla trained network. The target network is trained to be a "nasty teacher" version of the vanilla trained network. The intuition is that this keeps the top-1 prediction correct but sparse so top-N predictions are very low probability and therefore far away in the decision space of the model. This should make the model robust to perturbations of the input. The paper provides interesting theoretical analysis of the solution and empirical results show that it does work.

### Strengths
- Interesting and new idea: nasty training was intended for another purpose so its a good application
- Good analysis: the intuitive and formal analysis was interesting and mostly convincing
- Mostly good results: based on the results the method clearly does work

### Weaknesses
- The results seem to focus on CIFAR datasets which may not be fully representative of real world conditions (See Maiya et al. "Unifying the Harmonic Analysis of Adversarial Attacks and Robustness")
- No analysis of training time

### Questions
This was a very interesting paper which I think makes a good contribution. The idea is new and it's an insightful application of nasty training which makes sense from the theoretical motivation in the paper. The primary thing I think is missing is more/more convincing results. Based on the discussion I expected there to be a pretty clear improvement from the proposed method but on the presented CIFAR results, it didn't look like a dramatic change. Also I am not sure that CIFAR results are reflective of real conditions so the method should really be tested on something more comprehensive. It also it wasn't clear to me from the paper how much longer it takes to incorporate the proposed training loop after already going through adversarial training on the target model.

**Specific Questions** 
- Does the method work on datasets beyond CIFAR? ImageNet for example?
- What is the total retraining time for this method? Both with and without the required adversarial training that occurs before the nasty training.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a method that integrate the nasty training (NT) into AT to strengthen robustness. They further analyze the probability sparsity of NT which has potential in adversarial training. And the experiments show the effectiveness of the method.

### Strengths
1. This paper analyzes the probability sparsity from NT and combined it with AT, by increasing decision boundary margins result in a higher cost for adversarial attacks, which shows the promising performance. And has in-depth theoretical interpretable analysis.
2. The experiment is comprehensive with CNN and ViT under different white and black attacks. And the ablations show the sparsity and effectiveness.

### Weaknesses
1. The main results in Table 1 and Table 2 were not compared with some SOTA method, such as LWTA [1], IKL-AT [2] and DCS [3].
2. All reported results correspond to the best outcomes over three independent runs, but there is no report of the mean and standard deviation of the results. And the class index in Figure 5 should be integer.

[1] Stochastic local winner-takes-all networks enable profound adversarial robustness, 2021.

[2] Decoupled kullback-leibler divergence loss, NeurIPS 2024.

[3] Adversarial Robustness via Deformable Convolution with Stochasticity, ICML 2025.

### Questions
1.	Could you add some up-to-date SOTA methods to make a more comprehensive comparison?
2.	Could you present the mean and standard deviation of your method at least in main table? And could you change the class index in Figure 5 to integer?
3.	NAT utilizes a VT for training. For the fairness, does NAT cover the expenses of the VT in Table 8?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Nasty Adversarial Training (NAT), a defense framework for adversarial attacks. NAT integrates the concept of probability sparsity, which was originally proposed in the context of distillation resistance, into traditional adversarial training. The paper aims to demonstrate how enforcing sparsity in the output probability distribution can enhance the robustness of models by increasing the inter-class separability and widening decision margins in the logit space.

### Strengths
1. This paper use “nasty training” as a robustness regularizer.
2. This paper have solid theoretical reasoning and thorough experimental validation.
3. This method improves robustness while maintaining efficiency and interpretability.
4. This paper have good exposition of probability sparsity and its spatial interpretation.
5. This method can be easily integrated into existing AT pipelines with negligible cost.

### Weaknesses
1. The Taylor expansion and sparsity explanation rely on approximations; formal proofs or bounds are lacking.
2. The “spatial metric” benefits are qualitatively visualized but lack quantitative metrics (e.g., explicit margin distributions).
3. Only standard PGD/CW/AA considered — might not generalize to adaptive threat models.
4. The authors compare to entropy/logit norm regularization briefly but not in depth.
5. The term “nasty” may be unconventional for robustness literature and could obscure broader relevance.

### Questions
1. Can authors quantify the relationship between measured probability sparsity and empirical robustness (e.g., correlation between entropy and adversarial accuracy)?
2. How does NAT perform under adaptive attacks specifically designed to exploit the auxiliary adversary structure?
3. Could the “probabilistic sparsity” be approximated directly (e.g., via entropy regularization) instead of adversary-based NAT?
4. How sensitive is NAT to adversary model mismatch or overfitting? Would a partially shared backbone improve stability?
5. Does the spatial metric benefit persist for non-classification tasks (e.g., detection or segmentation)?

### Soundness
3

### Presentation
3

### Contribution
3
