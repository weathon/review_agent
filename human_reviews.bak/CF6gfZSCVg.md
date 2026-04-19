# Anarchic Federated Bilevel Optimization

- Decision: Reject
- Scores: 8, 5, 3

## Abstract
Rapid federated bilevel optimization (FBO) developments have attracted much attention in various emerging machine learning and communication applications. Existing work on FBO often assumes that clients participate in the learning process with some particular pattern (such as balanced participation), and/or in a synchronous manner, and/or with homogeneous local iteration numbers, which might be
hard to hold in practice. This paper proposes a novel Asynchronous Federated Bilevel Optimization (AFBO) algorithm, which allows clients to 1) participate in any inner or outer rounds; 2) participate asynchronously; and 3) participate with any number of local iterations. The proposed AFBO algorithm enables clients to flexibly participate in FBO training. We provide a theoretic analysis of the learning loss of AFBO and the result shows that the AFBO algorithm can achieve a convergence rate of $\mathcal{O}(\sqrt{\frac1T})$, which matches that of the existing benchmarks. Numerical studies are conducted to verify the efficiency of the proposed algorithm.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
A double-loop scheme Anarchic Federated Bilevel Optimization (AFBO) is proposed in this work, which allows clients to flexibly participate in federated bilevel optimization training according to their heterogeneous and time-varying computation and communication capabilities. Moreover, theoretical analysis is conducted to show the convergence rate of the proposed method, i.e., it is demonstrated that the proposed  AFBO algorithm can achieve a convergence rate of $O(\sqrt{1/T})$.

### Strengths
1. This proposed method is efficient since the clients in the distributed system can participate in any inner or outer rounds, asynchronously, and with any number of local iterations.

2. The authors conduct lots of theoretical analysis about the proposed method, e.g., convergence analysis. It is demonstrated that the proposed  AFBO algorithm can achieve a convergence rate of $O(\sqrt{1/T})$.

3. This paper is well-organized and easy to follow.

### Weaknesses
I have some concerns about the experiments and communication complexity as follows.

1. In the experiments, the authors claim the proposed AFBO is the most efficient algorithm. However, more explanation should be added about why the proposed method is more efficient than ADBO in the experiment since the ADBO is also an asynchronous algorithm.

2. More experimental results should be added to show the excellent performance of the proposed AFBO.

3. Lack of analysis for the communication complexity of the proposed method.

### Questions
My questions are about the experiments and communication complexity, please see the weakness above.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper presents a new algorithm called Asynchronous Federated Bilevel Optimization (AFBO), offering a flexible approach for client participation in federated bilevel optimization (FBO) training. The unique aspects of AFBO include the ability for clients to not only join in at any stage of the inner or outer optimization rounds, but also undertake a variable number of local training iterations. The training process can be engaged asynchronously. Rigorous theoretical examination has been conducted to reveal convergence rate. It is seen that AFBO's convergence rate aligning with other benchmarks.

### Strengths
Asynchronous federated learning is an important problem with numerous applications. The theoretic analysis is solid and authors have conducted experiments to assess the performance of the proposed algorithm.

### Weaknesses
The experiment results are limited. Authors are recommended to compare the performance of the proposed algorithm with algorithms such as [1].  In addition, the idea itself is similar to [2]. Authors are therefore recommended to elaborate more on the difference between this work and existing ones in the literature. In addition, in all figures, it is seen that the performance of AFBO is only slighted better than ADBO.     

[1] Prometheus: Taming sample and communication complexities in constrained decentralized stochastic bilevel learning. ICML 2023 
[2] Anarchic Federated Learning ICML 2022

### Questions
I understand that AFBO offers flexibility by allows clients to engage in the updating in an asynchronous manner.  Is it possible that under such a setting, the AFBO algorithm may converge to a solution that is different from other algorithms ?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper studied federated bilevel optimization under the asynchronous setting. However, there are some fatal errors in convergence analysis.

### Strengths
The problem investigated is important. 

The writing is good.

### Weaknesses
1. This paper used impractical assumptions. In particular, it assumes the gradient of $g$ is upper bounded and $g$ is strongly convex.  A strongly convex quadratic function does not satisfy those two assumptions simultaneously. 

2. There are some fatal errors. In particular, this paper denotes
$\bar{H}\left(x^t, y^{t+1}\right):=\mathbb{E}\left[\frac{1}{m} \sum_{i \in \mathcal{M}} \bar{H}_i\left(x^{t-\tau_i^t}, y^{t-\tau_i^t+1}\right)\right]$. Then, in convergence analysis, the authors directly use $x^{t}$ rather than $x^{t-\tau_i^t}$, e.g., the second equation when proving lemma 1. THis is totally wrong. 

3. For an asynchronous algorithm, how the communication latency affects the convergence rate should be discussed.

### Questions
1. This paper used impractical assumptions. In particular, it assumes the gradient of $g$ is upper bounded and $g$ is strongly convex.  A strongly convex quadratic function does not satisfy those two assumptions simultaneously. 

2. There are some fatal errors. In particular, this paper denotes
$\bar{H}\left(x^t, y^{t+1}\right):=\mathbb{E}\left[\frac{1}{m} \sum_{i \in \mathcal{M}} \bar{H}_i\left(x^{t-\tau_i^t}, y^{t-\tau_i^t+1}\right)\right]$. Then, in convergence analysis, the authors directly use $x^{t}$ rather than $x^{t-\tau_i^t}$, e.g., the second equation when proving lemma 1. THis is totally wrong. 

3. For an asynchronous algorithm, how the communication latency affects the convergence rate should be discussed.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor
