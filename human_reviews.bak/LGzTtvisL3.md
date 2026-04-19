# FLea: Improving federated learning on scarce and label-skewed data via  privacy-preserving feature augmentation

- Decision: Reject
- Scores: 3, 6, 5, 5

## Abstract
Learning a global model by abstracting the knowledge, distributed across multiple clients, without aggregating the raw data is the primary goal of Federated Learning (FL). Typically, this works in rounds alternating between parallel local training at several clients, followed by model aggregation at a server. We found that existing FL methods under-perform when local datasets are small and present severe label skew as these lead to over-fitting and local model bias. This is a realistic setting in many real-world applications. To address the problem, 
we propose FLea, a unified framework that tackles over-fitting and local bias by encouraging clients to exchange privacy-protected features to aid local training. The features refer to activations from an intermediate layer of the model, which are obfuscated before being shared with other clients to protect sensitive information in the data. FLea leverages a novel way of combining local and shared features as augmentations to enhance local model learning. Our extensive experiments demonstrate that FLea outperforms the start-of-the-art FL methods, sharing only model parameters, by up to $17.6\%$, and also outperforms the FL methods that share data augmentations by up to $6.3\%$, while reducing the privacy vulnerability associated with shared data augmentations.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a privacy-preserving feature augmentation method to address the challenges of data scarcity and label-skewness in Federated Learning (FL). The work is well-motivated; however, there are several issues that the authors need to address to convincingly demonstrate the novelty and effectiveness of the proposed method. There are also many writing issues throughout the paper.

### Strengths
- The backgroud of the problem and the motivation of the work are well explained

### Weaknesses
Major issues:

1. The authors claim that the proposed method is novel, but many components come from the literature. For example, the authors claim that they have devised a novel feature augmentation method, but λ, a key parameter in the method, is drawn from existing literature.

2.λ1 and λ2 in Equation 6 are critical parameters, but it is unclear how they are set.

3.The paper does not discuss the impact of different model types on the proposed method or specify which layer (l-th) has been learned in the experiments. The evaluation solely relies on one AI model, MobileNet, and one dataset, CIFAR-10, which is not sufficient. 

4.The statement, 'we characterize the distribution of the low-dimensional features from the penultimate layer of local and global models,' lacks justification regarding why the penultimate layer is suitable for characterizing low-dimensional features.

Other issues:
-  Section 2.2 mentions low-dimensional feature overlapping and classifier bias as main label skew problems, but it does not elaborate on low-dimensional feature overlapping or explicitly define the problem.

-The paper would benefit from a more comprehensive discussion explaining why sharing features can enhance model performance.

-Notations in B and Bf, which are crucial concepts in the work, are not defined explicitly. Although the definitions of xi, yi, and fi can be inferred, it remains unclear what y^fi represents.

-The sentence regarding (f^l_i , yi) and (fj, yf j ) should use the same notation consistently.
The use of 'λ ∼ Beta(a, a)' needs clarification. It would be helpful to explain what Beta() is and how it is relevant. The reference at the end of the sentence should specify which part of the information it pertains to.

Language problems:
-"prompting the introduction of regularization techniques to address this issue.erm, Lr, can be included"?
- suggesting that the over-fitting is [severe]
-FLea works in [an] iterative manner
-The sentence "one data batch with labels from Dk and one feature batch with labels from the received feature buffer F (t), termed by B = {(xi, yi) ∈ Dk} and Bf = {(fi, y^f i ) ∈ F (t)}, respectively (|B| = |Bf |)." is incomplete
- one for knowledge distillation [from] the global model

### Questions
How are the values of λ1 and λ2 in Equation 6 are set?

what is y^fi in Bf?

What are the impact of model types on the proposed method?

What is the Beta() function?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Aiming to address both the scarcity and label-skewness of data simultaneously, this work proposes the sharing of features across clients  as an extension to the classical federated averaging algorithm. Specifically, features of the $\ell$-th layer are extracted and distributed across clients. During local training, the shared features and their corresponding labels are used to refine the layers subsequent to the $\ell$-th layer. To alleviate the privacy risks associated with sharing features, a decoupling loss function is introduced to reduce the correlation between the input data and its corresponding features.

### Strengths
i. By illustrating the overfitting problem and the gathering effect of features (as shown in Figure 2), this work effectively demonstrates the advantages of sharing data in situations of data scarcity. This supports the proposed framework for feature sharing, particularly when data sharing is restricted due to privacy concerns.

ii.Given that federated learning is inherently a privacy-preserving framework, the act of sharing features does increase the risk of information leakage comparing with FedAvg. To mitigate this, the work introduces an additional decoupling loss to strike a privacy-utility trade-off. A comprehensive study evaluating the amount of private information compromised through feature sharing is performed.

iii. Experimental results indicate that FLea consistently and significantly outperforms the baselines in scarce and label-skewed data scenario.

### Weaknesses
i. My primary concern is that the comparison is limited to a single dataset (CIFAR-10) and a single architecture (MobileNet). The experiments cannot sufficiently demonstrate the generalizability of Flea across different applications.

ii. The issue of data scarcity and label skew has previously been studied, particularly in Bayesian Federated Learning frameworks. For instance, reference [1] conducted experiments in the same setting. Given that Bayesian FL does not necessitate the additional communication of features, it appears to be a safer alternative to Flea. A comparison would be valuable to ascertain whether Flea can achieve superior accuracy.

iii. The first two conclusions in the Section 3.1 are not very convincing. Specifically, "The performance of FL methods decreases remarkably as data scarcity and label skew increase." This argument is concluded from "FedAvg, its accuracy decreases from 75% to 56% when |Dk| reduces from 5000 to 100 in the IID setting. When local data is sufficient (|Dk| = 5000), its accuracy drops from 75% for the IID setting to 60% for the non-IID setting." However, it is unclear why "|Dk| reduces from 5000 to 100" can be compared with changing IID to non-IID setting. Additionally, "Loss-based methods can address label skew only with sufficient local data." This argument is too strong. As mentioned in the second point, Bayesian frameworks are more amenable to data scarcity without requiring more data.

[1] Confidence-aware Personalized Federated Learning via Variational Expectation Maximization, Zhu et al., CVPR 2023.

### Questions
i. The plot of accuracy as a function of $\lambda_2$ can better clarify the privacy-utility trade-off.

ii. It seems that differential privacy can also be employed to obfuscate the features. Since DP is more mathematically rigorous and widely accepted, it may be more interesting to replace $\ell_{dec}$ with DP.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The article conducts an in-depth analysis of prior research in the field of federated learning, culminating in a comprehensive exploration of the issues stemming from label-skew and data-skew, and their consequential impacts on overfitting and model bias. To address these issues, the article undertakes a rigorous theoretical examination, and subsequently introduces the Flea framework as a proposed solution.

### Strengths
The paper comprehensively examines the phenomena of Label-Skew and Data -Skew in the context of federated learning, presenting methods that have demonstrated a significant enhancement in model performance while preserving privacy

### Weaknesses
1. The idea of broadcasting features in the paper appears to make a relatively modest contribution. The proposed under-explored scenarios of Label-skewed and Data-skewed resemble another expression of Non-iid data, which is not an under-explored scenario itself. It is unclear whether the authors can provide an explanation for the distinctions between these scenarios and Non-iid data.

2. The concept of overfitting and Client-Drift resulting from Label-skewed and Data-skewed scenarios seems consistent and has been extensively investigated in prior research.

3. Authors says this feature-sharing method is privacy-preserving, but it will leak more privacy compared to other algorithm without feature-sharing

4. The experiments in the paper exhibit depth but lack breadth.
   a) Lack of domain generalization: The paper's experimental dataset is limited to CIFAR10, without inclusion of other datasets.
   b) Lack of model generalization: The experimental investigation in the paper is limited to a single MobileNet_V2 model, and the discussion regarding the selection of feature layer 'l' is somewhat vague.
   c) The paper employs feature broadcasting but lacks a comparative study of communication costs, and the two communication rounds increase overhead.
   d) Observing that three loss terms act on $\theta_k$ in Eq 6, the ablation experiments are not presented in the main text. It is recommended that the authors consider reducing the discussion of Label-skewed and Data-skewed scenarios to make the core arguments more concise.

### Questions
This paper focuses on scarce and label-skewed data in federated learning. This is a kind of non-iid case, and a lot of papers study the non-iid and heterogenous scenarios in FL, why authors say this is the first research?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes feature level augmentation for FL on heterogeneous and low data regime. Alongside the model parameters server sends a feature buffer (that includes feature, label pair ) to clients as side information. The authors show that this side information can be used to increase the testing performance in label-skew and data scarce settings.

### Strengths
- As far as I'm aware sending feature, label pairs is a new idea. 
- The presentation of the algorithm through Figure 4 and the visuals presented in the first 3 sections are very helpful. 
- There is a decent increase in the performance compared to non-augmentation methods.

### Weaknesses
- More details needed on feature buffer, until very late it is not obvious what is the number of pairs and how it is collected.
- The scale of the experiments (number of samples and clients) is good, but more datasets are needed for evaluation.
- In section 3.1 authors report some performance changes due to data scarcity etc. but the setting is not clear (e.g. what is the communication frequency/local iterations).  
- The parameter $a$ is very critical, yet its resulted sensitivity is not thoroughly analyzed in experiments. Also introducing such a parameter is not ideal for FL settings. 
- Communication and computation burden due to the added feature buffer is not examined adequately.
- By choosing $\lambda_2$ authors adjust how private the algorithm is but since there is no rigorous privacy utility tradeoff; it is hard to characterize the choice of $\lambda_2$. I think this lack of rigor is undesirable in FL.

### Questions
Please address the weaknesses above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
