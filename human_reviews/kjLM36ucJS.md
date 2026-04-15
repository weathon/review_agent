# Synthetic data shuffling accelerates the convergence of federated learning under data heterogeneity

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 5

## Abstract
In federated learning, data heterogeneity is a critical challenge. A straightforward solution is to shuffle the clients' data to homogenize the distribution. However, this may violate data access rights, and how and when shuffling can accelerate the convergence of a federated optimization algorithm is not theoretically well understood. In this paper, we establish a precise and quantifiable correspondence between data heterogeneity and parameters in the convergence rate when a fraction of data is shuffled across clients. We discuss that shuffling can in some cases quadratically reduce the gradient dissimilarity with respect to the shuffling percentage, accelerating convergence. Inspired by the theory, we propose a practical approach that addresses the data access rights issue by shuffling locally generated synthetic data. The experimental results show that shuffling synthetic data improves the performance of multiple existing federated learning algorithms by a large margin.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes to use shuffled synthetic data to combat the heterogeneity issue in federated learning (FL). Specifically, it proposes to train a local generator for each client, generate a synthetic dataset from the trained generator, and then send it to the server. The server then aggregates and shuffles the synthetic data from all the clients and sends a uniform partition to each client. Each client then does its local training on a combination of its original local dataset and the shuffled synthetic data received from the server. The effectiveness of the proposed method is shown via several experiments. There is also some theoretical analysis to quantify the effect of data shuffling on the convergence of FL.

### Strengths
Synthetic data has shown a lot of promise in machine learning problems. This paper shows its effectiveness when properly shuffled, in improving the convergence of several federated optimization algorithms. The empirical gains shown are significant in many cases. It seems that this paper explains how it is better than prior work on the usage of synthetic data in federated learning, but I'm not very knowledgeable about this particular area or familiar with related works.

### Weaknesses
**1.** In the discussion around Lemma 1 and Corollary I, it is mentioned that the asymptotic convergence mainly depends on $\hat{\sigma}_p^2$. From the expression of $\mathbb{E}[\hat{\sigma}_p^2]$ in Lemma 1, one can compute that $\frac{d \text{ } \mathbb{E}[\hat{\sigma}_p^2]}{d p} = (1-2p)(\zeta^2 + \delta^2) - (\sigma^2 - \tilde{\sigma}^2)$. Now, for non-iid situations when $\zeta^2 \gg (\sigma^2 - \tilde{\sigma}^2)$, $\mathbb{E}[\hat{\sigma}_p^2]$ is an *increasing function* of $p$ around $p=0$ (no synthetic data). So, the convergence should get **worse** according to this result with small values of $p$ or a little bit of synthetic data. 

In fact, I think the value of $p$, say $p^{\ast}$, such that $\mathbb{E}[\hat{\sigma}_{p^{\ast}}^2] < {\sigma}^2$ is going to be large (this can easily be calculated because it is quadratic in $p^{\ast}$).

Can the authors address this?

**2.** I don't understand the second-last sentence on page 4 "*When $\sigma^2=0$, or in general...*". Can you quantify "super-linearly"? Also, where is $\sqrt{L_p}$ coming up in the bound of Corollary I? Overall this statement doesn't make too much sense to me.  

**3.** The paper talks about privacy of the local data but there are no formal privacy guarantees given with the proposed method. Moreover, training a generator on each client is also expensive.

### Questions
Please see Weaknesses above. If W1 and W2 are addressed well, I can raise my score.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper studies the problem of federated learning using Local SGD. The paper assumes there are $N$ clients with heterogeneous data and they want to minimize the average objective. This is the standard in federated learning, but the paper diverges from prior work by considering a scenario where the clients share a fraction $p$ of their data. They show (Lemma 1, Corollary 1) that this (naturally) leads to a lower variance term, and hence faster convergence  (as shown in their Figure 2). Because sharing data directly contradicts user privacy (a cornerstone of federated learning), the paper proposes to train a generator on each client and send synthetically generated data instead. If the generator is good enough, then the results of (Lemma 1, Corollary 1) apply.

### Strengths
1. The paper is well-written and clear.
2. The presented theory is correct and sound as far as I can see.

### Weaknesses
1. It is already known that reshuffling data locally reduces the variance (see e.g. [1]). Why is it any harder to study this problem across clients?
2. This is the main dealbreaker: sending synthetic images still breaks privacy. Even on massive datasets, generator models routinely reproduce training data (see [2, 3]). The data on any client in an ordinary federated learning problem is several orders of magnitude smaller than ImageNet or other large training sets, it is virtually guaranteed that a synthetic generator trained on local data would simply regurgitate client data.

My opinion is that the paper's results are not very theoretically surprising, and its practicality is nil because of privacy considerations. For this reason, I recommend rejection.

[1] Mishchenko, Khaled and Richtárik. Proximal and Federated Random Reshuffling, ICML 2022.
[2] Tinsley, Czajka, and Flynn. "This face does not exist... but it might be yours! identity leakage in generative models." IEEE/CVF Winter Conference on Applications of Computer Vision. 2021.
[3] Carlini N, Hayes J, Nasr M, Jagielski M, Sehwag V, Tramer F, Balle B, Ippolito D, Wallace E. Extracting training data from diffusion models. In32nd USENIX Security Symposium (USENIX Security 23) 2023 (pp. 5253-5270).

### Questions
N/A.

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a framework for accelerating federated learning based on synthetic data. The framework first trains generators on each client, then sends the generated samples to the server. The server aggregates the synthetic samples and distributes the generated samples to clients in i.i.d. manner. Authors provided rigorous analysis to their proposed method and provided extensive experiment evidence to show synthetic data improves both convergence and test accuracy.

Without differential privacy (DP), I have some concerns on whether the proposed method goes against the motivation of federated learning in the first place. For example, Large Language Models are shown to be capable of memorizing parts of the training set: 

Quantifying Memorization Across Neural Language Models, Carlini et al. (ICLR 2023)

With that said, the current setup of the experiments provided quantitative evidence of the benefit of synthetic data, although the idea of generating synthetic data to enhance federated learning is not novel.

### Strengths
The paper is well-written. The theoretical analysis is easy to follow. Authors provided extensive ablation of their proposed method.

### Weaknesses
1. This paper does not address important privacy concerns of transmitting synthetic samples generated on clients.

2. To evaluate the paper by leaving privacy concerns out of the scope, there are a few aspects I would like to obtain more insights:


       i. if not for privacy concerns, would FL still add value on top on synthetic data training obtained via stronger generative models fine-tuned on clients?  For example, by using state-of-art diffusion model for image, or Large language model for NLP problems.


     ii. I would like to see an ablation on the synthetic data generator used. Different generator quality should theoretically correspond to different $\delta$ in Assumption 4. Hence I would like to see how $\delta$ influence the quality backed by experiments. This will also enhance the practicality and shed some insight on the tightness of Corollary 1.

### Questions
Corollary I cites the framework developed in 

"A unified theory of decentralized SGD with changing topology and local updates. "

If I am correct, this establishes the convergence for distributed optimization on the aggregated dataset $(1-p)\mathcal{D}_i + p\mathcal{D}$. This provides little value to understand the convergence aspect since the quantity of interest are defined on the true data distribution which is union of $\mathcal{D}_i $. 


Corollary I does not explicitly defined the quantity it tries to bound, hence my understanding can be wrong and please clarify.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work revolves around the notion of reshuffling data across clients (as a preprocessing subroutine on top of established federated methods) in the Federated Learning setting aiming to homogenize their local distributions thus deriving improved accuracy and convergence speed. Specifically each client generates synthetic data based on their local distribution which are sent to the server. The server shuffles the received data and transmit them back equally to the clients thus augmenting their initial dataset. Theoretical justification is provided suggesting that the convergence speed is enhanced through this mechanism and experimental result on Cifar10 and Cifar100 that support the theoretical findings are presented in the main body of the paper.

### Strengths
-The paper is well-structured and easy to follow.

-The theoretical justification provided is intuitive and contributes to better understand the effects of data heterogeneity on the convergence speeds of the federated algorithms.

-Experimental results support the main point of the paper and suggest that there are settings where the proposed mechanism can alleviate the issue of data heterogeneity in the framework of FL.

### Weaknesses
-The theoretical results although insightful, appear to be straightforward derivations applied to known convergence bounds (Koloskova et al., 2020; Woodworth et al., 2020; Khaled et al., 2020). Furthermore, the idea of data reshuffling in FL is well known and as a result the theoretical contribution and the novelty of this work is somewhat marginal.

-A major concern lies on the metric of accuracy that is used in this work. Specifically, in the proposed method the clients obtain an augmented dataset which could potentially be very different from their initial data. As a result, achieving high accuracy on these augmented datasets is a potentially substantially easier task which however is misaligned with the true objectives of the clients. The authors need to further discuss and clarify this issue. Specifically, in the experiments presented it is my understanding that the size of the initial dataset for client $i$, is equal in size to the synthetic datasets that is assigned to the same client $n_i = \tilde{n}$. If indeed this is the case the resulting accuracy could be a misleading metric of success.

-This method relies on clients being able to produce synthetic data of good quality. In many FL settings however clients often have access to very few samples which could significantly hinder the effectiveness/soundness of the proposed method. 

-In the comparison with other synthetic-data based approached it would be beneficial to include convergence plot instead of just providing a final accuracy table. Further, the parameters for DENSE should be chosen optimally to provide a fair comparison.

-The comment on page 9 in section "Practical implication" claiming that "Additionally, the client has the option of checking the synthetic images and only sharing the less sensitive synthetic images to alleviate the information leakage." seems unreasonable. It is hard to imagine clients choosing manually images that diminish the information leakage out of a big number of synthetically produced data.  


-Minor issues

- Page 3, end of paragraph 1: "fail achieve" --> "fail to achieve".
- Page 4, paragraph 3: "it might infeasible" -->"it might be infeasible"

### Questions
-A major concern lies on the metric of accuracy that is used in this work. Specifically, in the proposed method the clients obtain an augmented dataset which could potentially be very different from their initial data. As a result, achieving high accuracy on these augmented datasets is a potentially substantially easier task which however is misaligned with the true objectives of the clients. The authors need to further discuss and clarify this issue. Specifically, in the experiments presented it is my understanding that the size of the initial dataset for client $i$, is equal in size to the synthetic datasets that is assigned to the same client $n_i = \tilde{n}$. If indeed this is the case the resulting accuracy could be a misleading metric of success.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
