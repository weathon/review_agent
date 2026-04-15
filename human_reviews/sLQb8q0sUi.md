# Fair and Efficient Contribution Valuation for Vertical Federated Learning

- Decision: Accept (poster)
- Scores: 6, 8, 6, 6

## Abstract
Federated learning is an emerging technology for training machine learning models across decentralized data sources without sharing data. Vertical federated learning, also known as feature-based federated learning, applies to scenarios where data sources have the same sample IDs but different feature sets. To ensure fairness among data owners, it is critical to objectively assess the contributions from different data sources and compensate the corresponding data owners accordingly. The Shapley value is a provably fair contribution valuation metric originating from cooperative game theory. However, its straight-forward computation requires extensively retraining a model on each potential combination of data sources, leading to prohibitively high communication and computation overheads due to multiple rounds of federated learning. To tackle this challenge, we propose a contribution valuation metric called vertical federated Shapley value (VerFedSV) based on the classic Shapley value. We show that VerFedSV not only satisfies many desirable properties of fairness but is also efficient to compute. Moreover, VerFedSV can be adapted to both synchronous and asynchronous vertical federated learning algorithms. Both theoretical analysis and extensive experimental results demonstrate the fairness, efficiency, adaptability, and effectiveness of VerFedSV.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper seeks to assign each client a fair value in the vertical federated learning setting where clients share the same sample IDs but own different features. The paper proposes VerFedSV which consider clients’ contributions at multiple time stamps during training. VerFedSV satisfy desirable properties of fairness, can be efficiently computed without model training and extra communication cost.
For the synchronous setting, computing VerFedSV will require the embedding of all data points. However, only the embedding for a mini-batch is available. The paper proposed how to obtain the embedding matrix through low rank matrix completion and show that error in the estimated VerFedSV is bounded.
For the asynchronous setting, VerFedSV can be directly and exactly computed for the mini batch. VerFedSV will reward clients with more computational resources (batch size) and frequent communication.

### Strengths
- The paper is well-written. The content and notations are easy to follow.
- The study of related work is comprehensive and it is clear how the paper contrasts with prior work.
- The assumptions (e.g., low rank embedding matrix) are sound and justified (e.g., due to similarity between local data points and during successive training).
- The paper novelly consider data valuation in the _vertical_ federated learning setting and the _asynchronous_ setting with different local computational resources.

### Weaknesses
- The paper did not discuss or empirically compare to alternatives, e.g., why not request embedding for all data points or value based on the updated mini-batch or a separate validation set? The experiments lack comparison to a ground-truth (request all embeddings) and other baselines (VFL valuation methods).
- The solution introduced in the work does not seem very novel after considering existing VFL literature and HFL data valuation work. Fan et al. (2022) have used low rank matrix completion to estimate utilities of subsets. In the introduction, it is mentioned that it is more challenging to compute Shapley value to VFL than HFL as it is not applicable (possible) to obtain the global model by concatenating private local models. The challenge has to be further clarified. Is the challenge resolved by using local embedding (as in other VFL work)?

### Questions
Questions
1. What are the reasons not to use the asynchronous valuation approach (page 7, use mini-batch only) for the synchronous setting?
2. Existing data valuation works (including those for HFL) usually use a separation validation set for the utility function. If a validation set is used, is the embedding matrix still needed? 


Minor suggestions
* The Shapley value is often misspelled as Sharpley, Sharply, Shaprley etc
* The paper can give some examples/explanation of models and $f$ for the reader to understand why the embedding is sufficient. E.g., what is $f$ for linear and logistic regression (in Sec. 3)
* After proposition 1, the notation/arguments for the coverage number should be explained.
* For VAFL, the client has to send the id of the embedding too.
* Report N and T for the experiments

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a metric for contribution valuation in vertical federated learning (VFL). The method is based on federated Shapley value and is adapted to synchronous VFL via low-rank embedding matrix completion techniques and extended to asynchronous VFL. This metric reflects both the quality of the local dataset and the power of local computational resources.

### Strengths
1. The paper is overall well-structured and the authors made good efforts in putting this paper in the literature.

2. Low-rank matrix completion of the embedding matrix resolves the issue that its updates are only partially observed in the mini-batch setting. This technique facilitates the application of Shapley value in VFL.

3. The proposed method VerFedSV is theoretical grounded in this work. It is important to verify that VerFedSV satisfies the criteria of balance, symmetry, zero element, and additivity. Detailed proofs and conditions are provided.

4. The authors experimentally validate the methods on real-world datasets and discussed the results.

### Weaknesses
1. Given the setup, it appears that the proposed metric works well only linear models or training only the last linear layer. Although the authors mentioned that results can be extended to a multi-class classification problem and general function, little concrete explanations are provided.

2. There seems be no baselines or comparison with other methods. I understand that contribution evaluation in VFL might not be well-studied before but at least Data Shapley Value can be used here.

### Questions
1. In VERFEDSV, would it be inefficient to sum over all subsets $S \subset M$?

2. What is the difference between the method in section 6 and FedSV?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper extends the FedSV framework by Wang et al. (2020) to vertical federated learning setting. For synchronous setting, since the server does not have access to the data embeddings to the full dataset, the authors propose an approximation algorithm based on matrix completion given the low rank assumption. For asynchronous setting, since there are no definition of training iterations for the server, the time stamps for contribution valuation is pre-determined. The author conducted experiments to demonstrate the effectiveness of the proposed technique.

### Strengths
The idea of using matrix completion is super interesting.

### Weaknesses
The writing can be improved quite a bit. 
- For example, $d^m$ is used for denoting the feature dimension of client $m$, but $m$ here can be easily misunderstood as exponent. Better to use $d^{(m)}$. 
- Quite a few places use "Sharpley". 
- It's good to provide some background about matrix completion algorithm, at least in Appendix. 
- In Proposition 1, what is $D^m$? What is $\gamma^m$? Again, $m$ can be easily misunderstood as exponent. 
- Proposition 3 can be better stated as an example instead of Proposition. 

I am also not too sure about the experiments.
- What is the dimension of each dataset being used? 
- What does Table 1 supposed to tell? Is it fair to compare the Shapley value of 1 artificial client against all regular clients? 
- How to set the hyperparameters of $r$ and $\lambda$? I think ablation study is needed.

### Questions
See weakness.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new contribution valuation metric called VerFedSV for vertical federated learning. The VerFedSV metric is based on the classic Shapley value, which is a provably fair contribution valuation metric originating from cooperative game theory. The paper shows that VerFedSV satisfies many desirable properties of fairness and is quite adaptable such that it can be applied under both synchronous and asynchronous VFL settings. The paper also provides experimental results that demonstrate the effectiveness of VerFedSV in terms of fairness and efficiency.

### Strengths
S1. This paper proposes a new Shapley value-based contribution valuation metric called VerFedSV for both synchronous and asynchronous vertical federated learning. 

S2. This paper provides a theoretical analysis of the VerFedSV metric and demonstrates its effectiveness through experimental results. 

S3. This paper addresses an important problem in the field of vertical federated learning, namely how to fairly value contributions from 
different clients.

### Weaknesses
W1. In the last paragraph of Sec. 2, it is said that model-independent utility function itself may cause some fairness issues in asynchronous scenario and cannot fully reflect a client’s contribution. This claim seems not intuitive, it is better to give a concrete example to explain the fairness issues in detail.

W2. There are missing related studies on Shapley Value in federated learning. Please conduct a more detailed literature survey.
[1] Sunqi Heng et al. ShapleyFL: Robust Federated Learning Based on Shapley Value. KDD 2023. 
[2] Zhenan Fan et al. Fair and efficient contribution valuation for vertical federated learning. 2022. 

W3. In Sec. 3, there aren’t enough explanations on the meaning of embedding h_i^m. How to compute it and how it is used to compute the loss are not mentioned either.

W4. It is said in Sec. 8 that VerFedSV increases when adding identical features, which may indicate the proposed method is fragile and lacking robustness in practical use, especially in asynchronous setting.

W5. There are so many typos. For example, in the first paragraph of Sec. 4, the terminology “Shapley” is mistakenly written as “Sharply”, “Shaprley”, “Sharply” for so many times.

### Questions
Beyond the above weak points, I also have the following questions:

Q1. Please explain the fairness issues caused by model-independent utility function in asynchronous scenario in details. It’s better to give an example.

Q2. Is this paper the first work with model-dependent utility functions in VFL? Are there really just two related works in VFL scenario?

Q3. Please give more explanation about the meaning of embedding h_i^m and how to compute it, which is more friendly to read for people not quite familiar with VFL.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
