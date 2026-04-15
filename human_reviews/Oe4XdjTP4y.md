# Measuring Local and Shuffled Privacy of Gradient Randomized Response

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 5, 3, 5

## Abstract
Local differential privacy (LDP) provides a strong privacy guarantee in a distributed setting such as federated learning (FL).
Even if deployed LDP mechanisms honestly provide such privacy guarantees by randomizing gradients, how can we confirm and measure it?
To answer the above question, we introduce an empirical privacy test in FL clients by measuring the lower bounds of LDP.
The results of this measurement give the client the empirical $\epsilon$ and probability that the two gradients can be distinguished.
We then instantiate five adversaries in FL under LDP to measure empirical LDP at various attack surfaces, including a worst-case attack that reaches the theoretical upper bound of LDP.
The empirical privacy test with the adversary instantiations enables FL clients to understand LDP more intuitively and verify that mechanisms claiming $\epsilon$-LDP actually provide equivalent privacy protection.
We also demonstrate numerical observations of the measured privacy in these adversarial settings, and the randomization algorithm LDP-SGD is vulnerable to gradient manipulation and a well-pre-trained model.
We further discuss employing a shuffler to measure empirical privacy in a collaborative way and also measuring privacy of shuffled model.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies local differential privacy (LDP) guarantees in the federated learning distributed setting. First, the paper empirically considers the privacy loss for the gradient randomized responses including in the well-known LDP-SGD (Duchi et. al. 2018, Erlingsson et. al. 2020), which first computes a clipped gradient and then samples a random unit vector, signed by a function of the clipped gradient as the output. The paper then considers various adversaries in federated learning to produce a worst-case attack that reaches the theoretical limits of LDP-SGD. 

The paper first claims that the worst-case inputs that match the theoretical upper bounds of LDP-SGD are achieved when the gradients have norms that match or exceed the clipping threshold $L$, possibly resulting in the incorrect sign. It then shows that for increasing values of the privacy parameter $\varepsilon$, the distributions become easier to distinguish. 

The paper then studies a number of attacks, including settings where 1) some labels are flipped, 2) some gradients are flipped, 3) the client and the server collude to flip a gradient, which is compounded by a malicious global model sent from a server, and 4) the client produces a dummy gradient.

### Strengths
- Experiments performed over a large number of datasets
- Collusion attack and dummy gradient attack empirically reveal vulnerabilities of LDP-SGD
- The experiments observed privacy amplification through shuffling

### Weaknesses
- Both 1) gradients becoming more distinguishable in experiments as the privacy parameter increases and 2) privacy amplification under shuffling matches existing theory and perhaps is not entirely surprising
- Limited conceptual or theoretical novelties

### Questions
Were there any characterizations observed for the privacy-convergence or privacy-utility tradeoffs?

### Soundness
2 fair

### Presentation
3 good

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
The paper looks at empirically measuring differential privacy (DP) level in federated learning (FL) under local DP (LDP) directly, and when the clients jointly communicate all results via a trusted shuffler. The authors focus on the gradient randomized response mechanism as the common LDP mechanism, find a pair of gradients corresponding to the worst-case under the chosen mechanism, and empirically measure how successful membership inference attacks. From the results they convert to empirical ADP guarantees via existing techniques. The paper considers the resutls under 5 different adversaries, including the most powerful adversary allowed by DP.

### Strengths
* The topic of empirically measuring DP protection level is important and topical.

* The paper is fairly clear to read.

* While the paper relies on many existing techniques developed for empirical DP estimation, the considered DP mechanism is somewhat different than the existing ones, and the results on shuffle DP seem novel.

### Weaknesses
* Comparisons to existing DP bounds in the shuffle model only use the (loose) analytical bound from Feldman et al. 2023.

* In my opinion, the overall story of clients testing by themselves in FL (e.g. abstract) is not convincing or necessary. Instead, this is much more convincing simply as an exploration of empirical privacy in the FL setting, regarless of whether this could be done by the clients bythemselves or by someone else.

* After reading the paper, I still have several questions for the authors (see below for particulars).

### Questions
1) First adversary settings (benign, label flip) seem too benign: I would argue that in most cases two random samples from the data will give overpositive results on the privacy level, which is a worst-case guarantee, in many cases possibly even with label flips. Considering this, comments such as in Relaxations of privacy parameters in Sec6.3 seem overconfident. Have you checked what would be the actual worst-case in the data for benign or label-flip? (On a related note, see also question 8)

2) On distinguishing the gradients: do you use cos-similarity  as the similarity metric in all cases, even in the benign setting? Is this still optimal way to do it?

3) Sec 3.1: on the difference in worst-case w.r.t Gaussian mechanism in the centralized setting: despite several statements about the worst case being entirely different compared to the Gaussian mechanism, to me it seems like the given worst case (maximal l2-norm grads pointing to opposite directions) should work as is also for the Gaussian mechanism under replace neighbourhood, and vice versa. Did I misread this?

4) In the empirical shuffling experiments, do you include delta also in the empirical results (it is not included e.g. in Sec 4.1)?

5) Provide some metric of variability for the empirical results, e.g., Fig4, or mention that the variability is small-enough to be ignored.

6) Sec6.2: "These results suggest that privacy amplification by shuffling may be an improvement over the state-of-the-art." What does this mean?

7) Given that we know that the analytical bound for shuffle DP is loose (see e.g. Feldman et al. 2023, Koskela et al 2023), is there some reason not to use numerical accounting when comparing against existing shuffle DP bounds to get at least somewhat tighter bounds?

8) Considering comments like Sec6.3 Relaxations of privacy parameters: how did you choose the norm clipping value for the experiments? One would expect that in the more benign settings choosing a large value leads to small empirical epsilon due to most gradients not hitting the clipping value, i.e., being farther from the worst-case, but this value does not explicitly show up in the results in any way. If so, one could basically choose any empirical epsilon between 0 and the one corresponding to the worst-case as a result of this proposed measurement just by tuning the clipping value. Which of these values would make sense?

#### Minor comments/requests (no reason to comment on this)
i) Please state the neighbourhood definition explicitly in defining DP.

References:
Feldman et al. 2023: Stronger privacy amplification by shuffling...
Koskela et al. 2023: Numerical accounting in the shuffle model of DP

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
This work extends the line of privacy auditing research to LDP. They analyze the worst-case gradient pair of LDP-SGD mechanism, and use the worst-case pair to design a simple distinguishing game for measuring the lower bound of epsilon through the classic Clopper-Pearson bound. The paper discusses different capability of the adversary. The paper also extends the attack method to the shuffle model.

### Strengths
The paper has a clear roadmap and is very easy to follow.

### Weaknesses
1. I am not too sure about the motivation of privacy auditing for LDP. The point of LDP is that the clients do not trust the central server and want to do the randomization on their own. I think the authors' intention was to verify the privacy guarantee of the local randomizer the client uses, but why can't the clients just run the local randomization program they trust (e.g., the one implemented by themselves)? I think the authors should be very clear about the motivation and assumptions for the scenario they consider here. 

2. The technical contribution of this work is relatively low. My feeling is that the only notable contribution in this work is the discovery of the worst-case gradient pair for the LDP-SGD algorithm (and the result for that is also not too surprising). Happy to be corrected on this point. 

3. The paper has quite a few places that lack mathematical rigor. For example, I didn't find where the author defines $\tilde{g}_1$ and $\tilde{g}_2$ in Section 3.2. The assumption for Proposition 4.1 should be clearly stated (the loss is binary cross-entropy). Also, it seems equation 6 is missing something (I guess it's indicator function)?

4. I am not entirely sure about what is the message of Section 3.2 is trying to convey. I feel like it's quite obvious that when $\epsilon$ is small, it's hard to distinguish, and when $\epsilon$ is large, it's easier to distinguish. What sounds interesting is the difference when using different $d$, but I don't find a discussion for it in the paper.

### Questions
Does the dimension $d$ impact the attack performance (if fix the number of trials)?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies schemes for auditing the empirical privacy parameters of the LDP mechanism (Specifically PrivUnit) and the shuffled model. The idea is to estimate the false positive rate (FPR) and true positive rate (TPR) by running the PrivUnit mechanism multiple times with half of the runs on gradient $g_1$ and the other half of runs on gradient $g_2=-g_1$. Then, the empirical estimate of $\varepsilon_0$ is obtained from the estimated FPR and TPR. This is exactly the standard black-box algorithm in estimating the empirical $\varepsilon$ in the central DP (except the worst case neighboring datasets).

### Strengths
The LDP and the shuffled model can be seen as a special case of the central DP mechanisms. In other words, the $\varepsilon_0$-LDP mechanism is also $\varepsilon_0$-DP in the central DP model ( similarly for the shuffled model). Hence, I don't understand the main difference between the scheme proposed in this paper and the schemes in the literature for CDP.

### Weaknesses
I disagree with the authors in the introduction that the previous studies estimate the empirical privacy of the Gaussian mechanism. There are some studies that consider the general $\varepsilon$-DP mechanism, e.g., Algorithm 2 in [Jagielski 2020] is generic and isn't dedicated to the Gaussian mechanism. Also [Steinke 2023] proposed a scheme for empirically estimating $\varepsilon$ for a generic DP mechanism in $\mathcal{O}(1)$ training run.

It is not clear to me the novelty of this work.  The paper combines ideas from the existing work in the literature. The techniques used for CDP can be applied to empirically estimate $\varepsilon_0$ in LDP and $\varepsilon$ in the shuffled model. The only difference is in constructing the worst-case neighboring data points in the LDP which is straightforward for the considered PrivUnit mechanism [Duchi et. al. 2018]. 

The paper focuses mainly on empirically estimating $\varepsilon_0$ of a special version of the PrivUnit mechanism [Duchi et. al. 2018 ] which is order optimal only in the high privacy regime $\varepsilon_0\leq 1$. It is better to focus on the PrivUnit2  [Bhowmick 2019] which is optimal for all privacy regimes. What about a generic $\epsilon_0$-LDP mechanism? Are there any ideas on how to handle this case?


 




Matthew Jagielski, Jonathan Ullman, and Alina Oprea. Auditing differentially private machine learning: How private is private sgd? Advances in NeurIPS 2020

Steinke, Thomas, Milad Nasr, and Matthew Jagielski. "Privacy Auditing with One (1) Training Run." arXiv preprint arXiv:2305.08846 (2023).

### Questions
Please, check my comments in the weaknesses section.

### Soundness
1 poor

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 5

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes an approach enabling a user to locally compute a lower bound on the privacy provided through an approach such as gradient randomized response, which is a mechanism ensuring local differential privacy in stochastic gradient descent. More precisely, the setting considered is that of federated learning and the objective is to be able to audit the privacy guarantees, thus obtaining a lower bound on epsilon through the use of privacy attacks conducted on the client-side. The approach proposed can also be adapted to the privacy via shuffling technique.

### Strengths
The authors have done a good review of the related work on privacy auditing and clearly position their work with respect to the state-of-the-art. They have also clearly explained how the LDP-SGD algorithm works and conduct a thorough analysis of the worst-case attack. I have particularly appreciate Figure 3, which clearly illustrates the impact of the privacy parameter on the distinguishability of gradients. 

One of the main contribution of the paper is the proposition of five crafting algorithms for the auditing phase. These approaches have been tested with a wide range of datasets. The results obtained clearly demonstrate that some of these approaches provide non-trivial lower bound when epsilon is large.

### Weaknesses
Although Table 2 aims at classifying the different attacks proposed in terms of adversary power, this issue should be discussed more in the paper. In particular, it is not clear for me about all the assumptions that are needed in practice for these attacks to be implemented in real-life. It would be great of the authors could expand a bit more on these aspects. The impact on the utility of the model of these different approaches should also be discussed. 

With respect to the experiments conducted, more details are needed to be able to understand them. For instance, in Figure 4 for the collusion attack it seems that the measured privacy is above the theoretical one for FEMNIST and CelebA, which seems strange. In addition, in Table 3, the measured epsilons seem to be too low to provide a meaningful theoretical guarantee. 

A few typos :
-« our proposed privacy test has a novelty to discuss » -> « one of the novelty of our proposed privacy test is that it discusses »
-« We utlize the state-of-the-art privacy » -> « We utilize the state-of-the-art privacy »
-« Recuired clients » -> « Required clients »
-« may still be issues running on a smartphone » -> « may still be issues running it on a smartphone »
-« poising effects » -> « poisoning effects »

### Questions
Please see the main comments in the weaknesses section.
One additional question : How is the reference Evfimievski 2003 also related to local differential privacy as cited in the introduction?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
