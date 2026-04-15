# Random Logit Scaling: Defending Deep Neural Networks Against Black-Box Score-Based Adversarial Example Attacks

- Decision: Reject
- Scores: 6, 5, 5

## Abstract
Machine learning models are increasingly adapted in various domains, such as autonomous driving, facial recognition, and malware detection, achieving state-of-the-art results. However, adversarial example attacks pose a significant threat to the reliable deployment of machine learning models in such applications. In recent years, some powerful adversarial example attacks have been proposed for the fast and query-efficient generation of adversarial examples, even in black-box scenarios where attackers only have an oracle access to the target model, highlighting the need for scalable, low-cost, and powerful defenses. In this work, we present two contributions to the domain of black-box attacks and defenses. First, we propose Random Logit Scaling (RLS), a randomization-based defense against black-box score-based adversarial example attacks. RLS is a plug-and-play, post-processing defense that can be implemented on top of any existing ML model with minimal effort. The idea behind RLS is to confuse an attacker by outputting falsified scores resulting from randomly scaled logits while maintaining the model accuracy. We show that RLS significantly reduces the success rate of state-of-the-art black-box score-based attacks while preserving the accuracy and minimizing confidence score distortion compared to state-of-the-art randomization-based defenses. Second, we introduce a novel adaptive attack against AAA, a SOTA non-randomized black-box defense against black-box score-based attacks that also modifies output logits to confuse attackers. With this adaptive attack, we demonstrate the vulnerability of AAA, establishing RLS as the effective SOTA defense against black-box score-based attacks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Based on the idea that a defender could blind an attacker by not producing true logit values, resulting in falsified confidence scores, this paper proposes Random Logit Scaling (RLS) as a novel defense mechanism. While random logit scaling could degrade model performance, RLS mitigates this by scaling output logits with a carefully chosen positive factor $m$, effectively reducing attack success rates, and preserving clean accuracy. Experimental results demonstrate the effectiveness of this approach.

### Strengths
1. Motivation and observation are both clear.
2. This work is a post-processing method. doesn't need any overhead during training. This is important for larger models and datasets.
3. The experiments span a range of attacks and datasets, providing strong empirical support for the robustness and effectiveness of RLS.

### Weaknesses
1. In comparison with other baselines, it’s straightforward in your framework to explain why they lead to performance degradation. However, it’s less clear why your method achieves a lower attack success rate than others. Could you clarify the limitations of the other methods?
2. The choice of using a uniform distribution for logit scaling, though effective, appears somewhat arbitrary. I think more discussion on potential alternative distributions would strengthen the approach.
3. In the experiments, a few fixed settings are explored for the range of $m$. I am curious about how varying the range of $m$ impacts model robustness and accuracy. An empirical ablation study on optimal parameter selection would add valuable insights.
4. I'm just curious if this method can be applied in the scenario of a white box attack.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper explores a new randomization-based method Random Logit Scale (RLS) to defend against black-box score-based adversarial attacks. RLS is a post processing defense which confuses an attacker by randomly scaling the output logit. While maintaining the model clean accuracy and confident score distortion, RLS can significant reduce the attack success rate of SOTA black-box score-based attacks.

### Strengths
- The paper's motivation is clear and convincing. The idea behind why randomly scaling the model logits helps fortify against black-box score-based attacks sounds intuitively reasonable.
- Most figures are easy to understand and the writing is clear and easy to follow. 
- The experiments are done against popular black-box attacks and the empirical improvements compared to baselines seem significant and no performance drop is desirable.

### Weaknesses
My main concerns are:
1. Lack of theoretical analysis
2. Some results in the experiments are not convincing
3. Some parts of the paper are not well written.

Please see the section on questions for a list of my concerns.

### Questions
## Main questions:
1. The method is very simple and straightforward. Randomly scaling the output logits does not hurt the accuracy because it maintains the rank of predicted labels if $m$ is larger than 0 and is able to falsify the output scores and confuse an adversary. This approach is also applicable for output softmax, so how does it work if the output softmax is randomly scaled instead of output logits?
2. Although the authors provided an analysis of the impact of RLS on the confidence scores, a theoretical analysis of the efficiency and robustness of RLS against score-base attacks, as in [1, 2], is missing. I encourage the authors to provide a similar theoretical analysis.
3. $l_2, l_\infty$ and $l_0$ attacks are different, and a defence designed for score-based attacks should be robust against different $l_p$ attacks. I would suggest the authors should conduct more comprehensive experiments across different $l_p$ score-based attacks [3, 4].
4. The larger the perturbation is, the higher the attack success rate of an attack is, and there is no guarantee RLS is better than other defence mechanisms under larger perturbation attacks. Therefore, a strong defense should work well across different perturbation budgets and I encourage the authors demonstrates the robustness of RLS vs others across different perturbation budgets.
5. The average number of queries of different attacks against different defense methods is very low if compared with the query budget used in this study (10,000). This means that these attacks successfully yield adversarial examples for most of the selected samples. The average number of queries should have a correlation with the ASR. For instance, lower ASR attacks should have an approximate or higher average number of queries than higher ASR attacks. However, results in Tables 3 and 4 show that the average number of queries of Bandit and Square attacks against RLS is significantly low and even lower than the non-defensive method, while ASR is extremely low if compared with the non-defensive method (~100% ASR). 
6. The experiment setup for RFD method reported in Table 3 and 4 is not consistent. The values of σ are varied and the accuracy drop is not consistent around 1% or 2% as reported in [2]. For a fair comparison, the authors should evaluate RFD with the same settings as done in [2] when the clean accuracy drop is around 1% or 2%.
7. As some results in the paper are not convincing and evaluation settings are not comprehensive mentioned, I suggest the authors release your implementation for replication and verification purposes?  
8. In this study, EOT is employed as an adaptive attack and it can be incorporated with gradient-estimation or random search attack method. Although the authors use k queries for each sample at each attack iteration and take an average over the outputs to cancel out the randomness but it is unclear what method is used to determine attack direction e.g. gradient-based or random search approach. What is the perturbation budget used to attack against the adaptive attack? Again, the query average is very low if compared to the query budget but the ASR is not high. Can the author report the median query number instead and provide a ridiculous explanation?
9. The authors discussed AAA defense which is designed for score-based attacks and this defense does not induce an accuracy drop as RLS. For completeness, I encourage the author to evaluate and compare this defense with RLS.

## Minor:
1. There are some contributions that are not actually contributions, e.g., 2 and 3. The authors should reorganize and summarize contributions 1 and 4  more thoroughly. 
2. Is minimizing distortion to the confidence score (in the introduction and section 5.3) really worth achieving? If so, the authors should cite papers or provide existing applications that rely on the accuracy of the gap between the model’s top and runner-up classes’ confidence scores to support your argument.
3. The experimental setup is not well explained and lacks some crucial information. For instance, what is the $l_p$ norm of the attacks? What is the perturbation budget used in this study? How many samples are selected for evaluation on CIFAR-10? How did the author select these samples? The author should clearly mention how samples are selected for evaluation on ImageNet in the experimental setup, e.g. 1000 samples are randomly selected, as mentioned in the caption of Table 4 in the experimental setup. Similarly, 2000 samples from CIFAR-10 are in the caption of Table 5. 

[1] Zeyu Qin, Yanbo Fan, Hongyuan Zha, and Baoyuan Wu. Random noise defense against query-based black-box attacks. NeurIPS, 2021.

[2] Nguyen Hung-Quang, Yingjie Lao, Tung Pham, Kok-Seng Wong, and Khoa D. Doan. Understanding the robustness of randomized feature defense against query-based adversarial attacks. ICLR, 2024

[3] Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion, and Matthias Hein. Square Attack: a query-efficient black-box adversarial attack via random search. ECCV, 2020

[4] Quoc Viet Vo, Ehsan Abbasnejad, and Damith C. Ranasinghe. Brusleattack: Query-efficient score-based sparse adversarial attack. ICLR, 2024.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
Existing works show that randomizing the prediction mitigates the threat of query-based attacks. However, defenses that inject noise into the input or to the feature lead to accuracy drops on clean data. Instead, this paper proposes random logit scaling (RLS), a defense that rescales the returned logit by a random variable $m$, therefore remains clean accuracy. Experimental results show that RLS is robust to score-based black box attacks without the tradeoff in accuracy.

### Strengths
- RLS is simple and plug-and-play, and can be applied to any model.
- RLS is robust against score-based attacks and does not change the accuracy of the target model.
- The experiments show that RLS is also robust against adaptive attacks that query the victim multiple times and average the prediction.

### Weaknesses
- Since RLS does not change the prediction, it will not be effective against decision-based attacks. The argument that decision-based attacks are inefficient is not convincing; recent attacks like RayS [1] can find adversarial samples with ~100 queries.
- The paper should compare to AAA [2], another defense that changes the logit to fool the attacker. AAA also guarantees the reliability of the returned probability by optimizing the calibration error. 
- The experiments are only conducted on CIFAR10, which is insufficient to conclude the effectiveness of RLS.


[1] Chen, Jinghui, and Quanquan Gu. "Rays: A ray searching method for hard-label adversarial attack." Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2020.
[2] Chen, Sizhe, et al. "Adversarial attack on attackers: Post-process to mitigate black-box score-based query attacks." Advances in Neural Information Processing Systems 35 (2022): 14929-14943.

### Questions
- What is the performance of AAA? 
- What is the calibration error of RLS?
- Experimental results on other datasets such as ImageNet would be helpful to demonstrate the robustness of RLS.

### Soundness
2

### Presentation
2

### Contribution
2
