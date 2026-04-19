# Locally Adaptive Federated Learning

- Decision: Reject
- Scores: 5, 6, 5

## Abstract
Federated learning is a paradigm of distributed machine learning in which multiple clients coordinate with a central server to learn a model, without sharing their own training data. Standard federated optimization methods such as Federated Averaging (FedAvg) ensure balance among the clients by using the same stepsize for local updates on all clients. However, this means that all clients need to respect the global geometry of the function which could yield slow convergence. In this work, we propose locally adaptive federated learning algorithms, that leverage the local geometric information for each client function. We show that such locally adaptive methods with uncoordinated stepsizes across all clients can be particularly efficient in interpolated (overparameterized) settings, and analyze their convergence in the presence of heterogeneous data for convex and strongly convex settings. We validate our theoretical claims by performing illustrative experiments for both i.i.d. non-i.i.d. cases. Our proposed algorithms match the optimization performance of tuned FedAvg in the convex setting, outperform FedAvg as well as state-of-the-art adaptive federated algorithms like FedAMS for non-convex experiments, and come with superior generalization performance.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper makes substantial use of the Polyak stepsize, a well-known technique in optimization for determining the learning rate based on the function values and gradients. The paper provides an example demonstrating how local adaptivity using Polyak stepsizes can improve convergence in optimization problems. It illustrates a scenario where the use of locally adaptive distributed Gradient Descent with Polyak stepsizes results in a near-constant iteration complexity, which is significantly better than using mini-batch Gradient Descent with a constant stepsize. The paper also delves into the convergence analysis on strongly convex functions. The algorithm is designed to be fully locally adaptive, catering to the needs of each client function in the federated learning setting.

### Strengths
1. Originality: The paper introduces a approach to federated learning, addressing the limitations of existing stepzise tuning methods and providing a solution that leverages local geometric information.
2. Quality: The authors attempt to build a theoretical foundation for their proposed algorithms, analyzing their convergence in various settings.

### Weaknesses
1. The connection to the Polyak stepsize and the rationale behind the specific choices of (\gamma_1) and (\gamma_2) in Example 1 could be clarified by referring to the definition in Loizou et al. 2021. 

2. The choice of a noise standard deviation (sd) of 10 in Figure 1's caption requires clarification, especially given the observation that SPS does not seem to converge. 

3. The paper should provide a clear definition of ( f^* ) in Eq. 5, addressing whether it refers to the global minimum of the finite sum or average sum of each f_i. They are essentially different. 

4. It would be very hard to parse the sentence that \sigma_f^2 is stronger than (zeta_*, sigma_*) but weaker than (zeta, sigma), and it is very hard to connect that to the inequalities in Proposition 1. 

5. The paper addressed the apparent need for hyperparameter tuning in both convex and non-convex experiments for FedSPS, especially given the gap between the worst and best performance from FedSPS and the gap from FedAMS, two of which are very comparable. FedAMS shows better performance in the experiments. So I don't see the remarkable improvement via the proposed method.

6. The paper claims to compare the proposed methods with FedADAM, but this comparison is not present in the paper. Including this missing comparison, more extensive comparisons with other non-iid FL papers, would strengthen the paper:

pFedMe: Personalized Federated Learning with Moreau Envelopes Dinh et al., 2020
PerFedAvg: Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach Fallah et al., 2020
APFL: Adaptive Personalized Federated Learning Deng et al., 2020
Ditto: Fair and Robust Federated Learning Through Personalization Li et al., 2022

### Questions
1.    - Can you provide more details on how the Polyak stepsize is connected to the choices of (gamma_1) and (gamma_2) in Example 1?
   - How do the specific choices of (gamma_1) and (gamma_2) in Example 1 relate to the definition of Polyak stepsize provided in Loizou et al., 2021?
   - Could you elaborate on the rationale behind selecting these particular values for (gamma_1) and (gamma_2)?

2.    - Why was a noise standard deviation (sd) of 10 chosen for the experiments depicted in Figure 1?
   - Given that SPS does not seem to converge in this scenario, could you explain how the chosen noise level impacts the convergence of SPS?

### Soundness
2 fair

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
The paper proposes FedSPS, which is an extension of the SPS (stochastic Polyak stepsize) framework [Loizou et al., 2021] to the federated learning setting. A variant of FedDecSPS with decreasing stepsize is also proposed. Convergence with convex loss is provided. Experiments are conducted to show the advantage of FedSPS compared with related FL methods with adaptive learning rates.

### Strengths
The method is simple and seems effective in some cases based on the experimental results. FL is an important and hot topic which would be insteresting to the ICLR audience.

The experiments compared with many baseline methods with adaptive learning rates.

### Weaknesses
1. Algorithm design: in my understanding, FedSPS is mainly an FL version of SPS [Loizou et al., 2021]. This extension is rather standard and the algorithmic novelty is not particularly strong.

2. Theory: the theoretical analysis combines the techniques of SPS with standard FL convergence proof, and only studied convex loss functions. Many results in the paper require a very small learning rate upper bound $\gamma_b$ (typically for non-iid clients which is common in FL), which significantly limits the 'adaptivity' of FedSPS and the proposed method becomes FedAvg approximately.

3. Experiments: 

(1) The presented numerical results do not fully justify the benefit of Polyak learning rates, and some results need more justification. In Figure 2(b), when $\gamma_b=1$, the stepsizes are around 0.87 and very stable through iterations. It never reached 1. That means we are using the Polyak stepsize all the time. As a result, $\gamma_b=5$ should give exactly the same training trajectory as $\gamma_b=1$, right? This is because neither of them trigger the upper bound $\gamma_b$. But in the figure, they are very different.

(2) Also, from Figure 2(a), $\gamma_b=1$ performs the best. From 2(b), the effective stepsizes of $\gamma_b=1$ is very stable. To a large extent, I would say that this is almost a constant learning rate without adaptivity. In contrast, $\gamma_b=5$ really brings adaptive stepsizes because the y-axis jumps a lot through iterations. So it is not very clear to me how 'adaptivity' helps the FL training. 

(3) In Figure 4(b), why does FedAdam perform so poorly (almost diverging)? Non-iid MNIST is a standard setting and an easy task. In the original paper of FedAdam and FedAMS there are also MNIST experiments and their methods performed well. This result does not seem very plausible.

2. While the paper claimed that FedSPS needs little parameter tuning, I don't think this necessarily holds in practice. For adaptive optimization based methods (with Adam-type updates), in most cases the default $\beta_1$, $\beta_2$ and $\epsilon$ values already achieve very promising performance, so for FedAdam or LocalAMS we essentially only need to tune the global and local learning rates. Furthermore, in fact, usually setting the global learning rate to 1 performs well. And for FedSPS, if we want, we can also add a global learning rate to (slightly) improve the performance. Moreover, The variant FedDecSPS has two tuning parameters, $\gamma_b$ and $c$.

Therefore, in general, I think the proposed method would require the same amount of parameter tuning as other adaptive FL methods.

### Questions
Questions and suggestions:

1. The proposed method is called adaptive FL, but it is different from the commonly noted adaptive methods (e.g., Adam, AMSGrad, etc.) which uses first and second order momentums. FedSPS is more like FedSGD with adaptive stepsizes. For better clarity on the contributions, I suggest that the title could follow [Stochastic Polyak Step-size for SGD: An Adaptive Learning Rate for Fast Convergence, AISTATS 2021] and include 'Polyak Step-size' and 'SGD'.

2. How does your analysis extend to the partial participation setting? I suggest adding a brief statement on this for clarity. For FedAdam,
[Analysis of Error Feedback in Federated Non-convex Optimization with Biased Compression: Fast Convergence and Partial Participation, ICML 2023] might be a relevant but missing reference.

I general, I think this is a borderline paper and more justification is needed. I will be happy to raise the score if my questions are answered well.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work proposes a federated learning algorithm named FedSPS, and the proposed algorithm performs stochastic Polyak stepsize in local updates. Convergence is guaranteed under convex and strongly-convex cases. In particular, when the optimization objective is in the interpolation regime or the by choosing diminishing stepsize, exact convergence is guaranteed. Authors also provides various numerical evaluations of the proposed algorithm

### Strengths
1. The propose algorithm performs local adaptive gradient steps, in contrast, most existing adaptive gradient methods in FL perform adaptive gradients at the server side.
2. Theoretical analysis is provided. Approximate convergence for convex and strongly-convex cases are guaranteed and exact convergence is provided under two special cases: interpolation condition and small step-size condition.
3. Some numerical experiments are provided to validate the proposed algorithm. The numerical studies includes both ablation studies ($\gamma$, $c$, $\tau$ etc.) and comparison with baselines (FedAvg, FedAdam etc.)

### Weaknesses
1. The proposed algorithm seems to be a direct extension of Stochastic Polyak step to the federated learning setting.  What is the major difficulty of this application? 
2. The theoretical analysis to the heterogeneity is not convincing. $\sigma_f^2$ is used as a measure of client heterogeneity in the paper, however, it is just an upper-bound (Proposition 1) of some more classical measure of heterogeneity, which means the proposed measure is weaker. In fact, if $l^*$ is chosen to be 0 (as in the paper), this measure is irrelevant to the heterogeneity.
3. Comparison with more baselines are desired. Although authors claim that "We design the first fully locally adaptive method for federated learning called FedSPS", there are already some local adaptive methods for FL, such as the  Local-AMSGrad method cited by the authors. It is desirable to add some comparison with these methods.

### Questions
Please see the weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
