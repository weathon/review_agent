# Enhancing Group Fairness in Federated Learning through Personalization

- Avg Score: 4.50
- Decision: Reject
- Scores: 5, 5, 5, 3

## Abstract
Personalized Federated Learning (FL) algorithms collaboratively train customized models for each client, enhancing the accuracy of the learned models on the client's local data (e.g., by clustering similar clients, by fine-tuning models locally, or by imposing regularization terms). In this paper, we investigate the impact of such personalization techniques on the group fairness of the learned models, and show that personalization can also lead to improved (local) fairness as an unintended benefit. We begin by illustrating these benefits of personalization through numerical experiments comparing several classes of personalized FL algorithms against a baseline FedAvg algorithm, elaborating on the reasons behind improved fairness using personalized FL, and then providing analytical support. Motivated by these, we then show how to build on this (unintended) fairness benefit, by further integrating a fairness metric into the cluster-selection procedure of clustering-based personalized FL algorithms, and improve the fairness-accuracy trade-off attainable through them. Specifically, we propose two new fairness-aware federated clustering algorithms, Fair-FCA and Fair-FL+HC, extending the existing IFCA and FL+HC algorithms, and demonstrate their ability to strike a (tuneable) balance between accuracy and fairness at the client level.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
In this paper, the authors investigate the impact of such personalization techniques on the group fairness of the learned models, and show that personalization can also lead to improved (local) fairness as an unintended benefit. The authors propose Fair-FCA and Fair-FL+HC algorithms which achieve state-of-the-art performance.

### Strengths
+ The authors investigate personalization to improve fairness which is somewhat interesting.

+ The authors conduct extensive experiments to validate their claims.

### Weaknesses
+ It is still unclear to me why personalization can also improve fairness. After reading through the paper, it seems that clustering could somewhat improve fairness. 

+ The mathematical proof and illustrations should appear in the main paper since they are relatively important.

+ Some similar works should be discussed, e.g., [1]

Ref:

[1] Intra-and Inter-group Optimal Transport for User-Oriented Fairness in Recommender Systems

### Questions
NA

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This manuscript explores the intersection of personalized Federated Learning (FL) and group fairness. It effectively demonstrates that personalization techniques, which are typically employed to enhance model accuracy on local data, can also inadvertently improve fairness. The paper substantiates these claims through comprehensive numerical experiments comparing various FL algorithms and introduces novel fairness-aware federated clustering algorithms. These algorithms, Fair-FCA and Fair-FL+HC, extend existing IFCA and FL+HC frameworks to incorporate a fairness metric into the cluster-selection process, aiming to optimize both fairness and accuracy.

### Strengths
1．	The manuscript is well-organized, clearly presenting the methodology and findings. 

2．	The manuscript offers a comprehensive experimental analysis across various algorithms, fairness notions, and datasets.

### Weaknesses
1.	The authors demonstrate that personalized federated learning (FL) improves fairness through experiments and intuitive analysis; however, the manuscript lacks theoretical justification for these findings.

2.	The experimental methods used to assess the impact of personalization on fairness are outdated and do not incorporate relevant studies from the past three years.

3.	The proposed approach merely adds a fairness-related loss to existing FL methods, offering insufficient innovation to significantly advance fairness in federated learning.

### Questions
See weakness.

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
3

### Summary
This paper studies the problem of group fairness in federated learning. The authors find that personalized federated learning unintentionally benefits group fairness. The authors further introduce a fairness metric into clustering to improve the trade-off between fairness and accuracy. Experiments were conducted on real-world datasets.

### Strengths
1.	The idea of unintentionally benefiting group fairness through personalization is indeed interesting.
2.	The paper is generally well-written, with a clear structure that is easy to follow.

### Weaknesses
1.	Since personalization can reduce the impact of dominant clients, the finding that personalization benefits group fairness is intuitive and not particularly surprising. Also, the results presented in the paper do not consistently support this finding. For instance, in Figure 3, federated methods improve the accuracy while increasing the fairness gap, which is inconsistent with Figures 1 and 2.
2.	The technical contribution of this paper is limited. The first introduced metric is merely a linear combination of fairness terms with a hyperparameter, making it unconvincing to claim that it "improves the fairness-accuracy trade-off" through hyperparameter tuning. How to determine the value of hyperparameters in practice? The second introduced metric is an incremental combination of hierarchical clustering.

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper explores the impact of personalization techniques on local fairness (i.e., the model’s fairness on each client’s local data) in federated learning. Extensive experiments compare the accuracy and fairness of personalized federated algorithms with FedAvg and standalone learning.  Results indicate that some personalized algorithms can improve model fairness while maintaining accuracy. Finally, this paper proposes fairness-aware Federated clustering algorithms based on existing methods to enforce local group fairness while enhancing accuracy.

### Strengths
1. The paper investigates fairness in FL from an interesting perspective, namely the effect of personalization techniques on local fairness.
2. The paper provides extensive numerical experiments comparing the accuracy and fairness of personalization methods with FedAvg across various scenarios (dataset, heterogeneity).
3. The experiments and methods are clear and easy to read.

### Weaknesses
1. Concerns about contributions. The paper spends a significant amount of space presenting experimental results that demonstrate the dual benefits of personalized federated approaches in accuracy and fairness, indicating that personalization is a promising research avenue for fair FL. However, the subsequent analytical support and the proposed approaches appear to lack significant contributions.
- Only federated clustering algorithms are analyzed, and the assumptions in the theoretical analysis are too strong for practical FL settings.
- The proposed methods build upon existing methods by incorporating an additional fairness performance metric. The detailed algorithmic steps (Algorithm 1,2) resemble those in prior studies Ghosh et al. (2020) and Briggs et al. (2020).
2. No effective local fair baseline in the experiments. The authors claim that the proposed method can enhance local accuracy while unintentionally improving local fairness. For empirical validation, the authors should compare the proposed methods against existing federated learning methods designed to improve local fairness. However, the experiments in Section 5 do not include FL methods specifically designed for local fairness.
3. The empirical analysis of the proposed methods may not be entirely convincing. Experiments in this paper are limited to comparisons. Additional experiments are required to validate its stability in different federated setting, e.g. heterogeneity, client numbers.
4. It is inappropriate to evaluate local fairness on methods specifically designed for global fairness (section 4.4), since previous work [1] has pointed out that global fairness differs from local fairness.
5. Including the theoretical analysis and more detailed experimental results of the proposed method in the main text, rather than in the appendix, would strengthen the paper.
[1] Hamman, Faisal, and Sanghamitra Dutta. Demystifying local & global fairness trade-offs in federated learning using partial information decomposition. ICLR, 2024.

### Questions
Beyond the above weak points, there are also additional questions:

What motivated the authors to utilize federated clustering algorithms to improve local fairness? Figure 1(C) indicates that, in highly heterogeneous data settings, these methods underperform FedAvg and MAML in the accuracy-fairness trade-off.

Given that fairness constraints are non-convex and non-differentiable, does incorporating fairness metrics in existing federated clustering algorithms pose potential convergence challenges?

### Soundness
2

### Presentation
3

### Contribution
2
