# Contrastive Difference Predictive Coding

- Decision: Accept (poster)
- Scores: 6, 8, 6, 8

## Abstract
Predicting and reasoning about the future lie at the heart of many time-series questions. For example, goal-conditioned reinforcement learning can be viewed as learning representations to predict which states are likely to be visited in the future. While prior methods have used contrastive predictive coding to model time series data, learning representations that encode long-term dependencies usually requires large amounts of data. In this paper, we introduce a temporal difference version of contrastive predictive coding that stitches together pieces of different time series data to decrease the amount of data required to learn predictions of future events. We apply this representation learning method to derive an off-policy algorithm for goal-conditioned RL. Experiments demonstrate that, compared with prior RL methods, ours achieves $2 \times$ median improvement in success rates and can better cope with stochastic environments. In tabular settings, we show that our method is about $20\times$ more sample efficient than the successor representation and $1500 \times$ more sample efficient than the standard (Monte Carlo) version of contrastive predictive coding.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the representation learning for goal-conditioned reinforcement learning problems. Built upon InfoNCE objective, this paper proposes a temporal difference estimator of InfoNCE objective and applied it to goal-conditioned RL algorithm.

In the experiment section, in the online RL setting, the proposed method is compared with prior goal-conditioned RL algorithms, including quasimetric RL, contrastive RL (Monte Carlo estimator of NCE objective), and hindsight experience relabelling. The proposed method achieved a higher average return in the comparison. Also, in the offline RL setting, The proposed method is compared with quasimetric RL, contrastive RL, and SOTA offline RL algorithms. The proposed algorithm generally outperforms baselines.

### Strengths
This paper is well-written and easy to read. The proposed method is explained and presented clearly.

This paper provides clear derivation and a solid theoretical foundation for the proposed method in Section 3. I 

The proposed method is supported by extensive experiments comparing with many baseline approaches.

The analysis in Section 4.3 and 4.4 validate the advantages of the proposed method in comparison with other representation learning approaches. It is impressive to see that the proposed method can stitch together pieces of data.

### Weaknesses
Some statements, especially some in the introduction part, seem not fully supported by evidence provided in the paper.

For example, it is claimed that the proposed method "enables us to perform counterfactual reasoning". However, this point is not clear in the following section. Could you please explain it in detail?

### Questions
How is the proposed method sensitive to hyper-parameters? Do we need careful hyper-parameter tuning to make it work? Is there any intuitive guidance about how to adjust the hyper-parameters?

In Algorithm 1, many notations are introduced for the first time without any definition. Could you please clarify them?

One important baseline, contrastive RL, is the Monte Carlo estimator of the NCE loss. Could you please also compare with the algorithm using Monte Carlo estimator of InfoNCE loss, since it is already introduced in the prior work Eysenbach et al., 2022

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
This paper extends previous works on contrastive RL by using a temporal difference estimator. Similar to contrastive RL, the paper proposes to use contrastive learning, InfoNCE in particular, for estimating the discounted state occupancy measure. Unlike the previous contrastive RL approach, which averages over the goal distribution, the proposed method uses a temporal difference estimator and results in a Bellman-like update rule (TDInfoNCE). Comparing the Bellman update for value function, TDInfoNCE requires taking expectation over the future states from a potentially different goal. It turns out that this can be estimated using importance weight. Then, the paper shows how the estimated state occupancy measure can be used in conjuction with goal-conditioned RL to form a full-fledged RL agent. Experimental results are shown on both online and offline settings and compared to a range of existing methods. In both settings, TDInfoNCE outperforms in most of the environments compared to previous approaches.

### Strengths
The paper is mostly well written, apart from some details (see questions section). 

The derivations are sound. 

Experimental results show strong performance comparing to previous methods. The paper also presents some analysis and insights to explain the performance.

### Weaknesses
The novelty is slightly limited. The idea of using InfoNCE to estimate the state occupancy measure has been presented in contrastive RL; the Bellman-like update and the use of importance weight has been presented in C-Learning.

### Questions
1. Questions regarding the algorithm:
 - It's not clear to me what's the goal distribution is used? Is it a random goal? Does the different goal distribution affect data efficiency?
- How is $s_{t+}$ sampled? Is it the same as previous approaches - sample t from a geometric distribution?

2. The notation in Figure 1 is not very clear to me. Is it suppose to visualize Eq. 4?

3. In Eq. 7 and the one above, shoud it be $p^{\pi}(s_{t+}^{(1)}|s^{'}, a^{'}, g)$ instead of  $p^{\pi}(s_{t+}^{(1)}|s^{'}, a^{'})$?

4. In Table 1, result for contrastive RL on large-diverse-v2 should not be bold; result for TDInfoNCE on unmaze-v2 should not be bold.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper derives a TD variant of the InfoNCE objective function, relating this to some of the work in reinforcement learning using future distributions of state as an objective (i.e. successor representations/features). This algorithm is then applied to goal-conditioned reinforcement learning, showing competitive performance among several baselines.

### Strengths
The derived method fits nicely within the literature and seems to fill a nice gap between contrastive objectives from self-supervised objectives and more online focused temporal-difference updates.

### Weaknesses
After the rebuttal, I'm raising my score. While I believe there are issues with empirical section still, these are issues the rest of the literature are also facing. I don't think rejecting this paper is a way to a solution. I also appreciate the fix to some of the inaccurate statements that were overlooked! 

Great job authors!

--------before edit-------

This paper struggles with clarity and accuracy in some of the ancillary statements made about the literature surrounding the paper and in the main content of the paper itself. There are also several issues with the experimental section that should be addressed.

## Accuracy and Clarity:
1. On page 1, in the paragraph starting with “the key aim…”, the language on motivating why a TD version of InfoNCE is oddly phrased. I think a fix should be easy here by removing phrases such as “may allow”, “may enable” and be more actionable in your language. This could also be replaced by actual hypotheses of what you expect to see from your objective and stated more formally.
2. In the same paragraph, and further throughout, you make a statement which suggests TD can do counter factual reasoning (or in the parlance of RL make use of off-policy updates) while Monte Carlo estimates cannot.  This is not true as Monte Carlo estimates can be made with off-policy corrections (using Importance Sampling like in TD). While this induces a more variant estimate as compared to TD (there is a classic bias-variance trade-off between MC and TD updates) and a TD update enables the use of incomplete trajectories (because you are using an estimate to inform your update rather than a full trajectory), I came out of the paper with the feeling the paper was suggesting Monte Carlo estimates couldn’t be off-policy. 
    - This comes up very strongly on page 5 (the first paragraph) “This is, we cannot share…”. We should be able to derive an off-policy version of the monte-carlo update for InfoNCE. This doesn’t mean it would be an estimator we would want to use in this setting, but it should be definable. If this is not the case, the paper should show that this can’t be done using importance sampling or cite a reference which shows monte carlo estimates can’t be off-policy.

3. **Notation clarity issues:** 
   - In your expectations above equation 7, I’m not sure what s’, a’ are here. Are you using these instead of s_{t+1} as used in equation 4? This notation should be unified. 
   - Equation (1) and following uses, I’m not sure you explain what the superscript is signifying. I think it is time, but it is not clear from the writing.
   - Shouldn’t the expectation in the middle of page 4 on the RHS (i.e. after applying the importance weight) be selecting actions a’ from the behavior policy? Or is the importance weight only correcting for the state distribution? Shouldn’t we also correct for the action distribution as well?



## Empirical Results:

There are two major issues with the empirical results as presented. 

### Major issues:

4. 3 seeds is too few to get any statistical confidence, especially without doing independent hyperparameter sweeps for each baseline. While in the past this has been standard, as a field we continually have shown that the statistical power of our experiments are laughably poor, even if the bounds of our results show statistical significance. This continually misleads the community, and needs to be addressed. This doesn’t include the issue with not running hyperparameter studies on the methods independently. See https://sites.ualberta.ca/~amw8/cookbook.pdf for a reference. While this paper is a draft it does a good job going through these issues and providing context from the literature.

5. In the appendix it is mentioned “We increase the capacity of the goal-conditioned state-action encoder…”. This suggests the model you are using may have more parameters than your counterparts. Is this true? Also did you use a larger batch size for all baselines or just your algorithm? If this was done for your method, this makes the results difficult to trust. If not, it likely means the hyperparameters of your baselines are now invalid. If both methods have the same hyperparameters then I’m usually ok with re-using old hyperparameter studies, unfortunately when any of the hypers change OR the new method has additional hypers this begins to weaken the validity of re-using the same hypers.

### Minor Issues:
- You should include all hyperparameters you used, even for baselines. 
- You include only success rate as a metric to compare. While this is reasonable, I think there is a lot that could be learned from more traditional metrics (i.e. return or something similar). This is especially the case when the success metric doesn’t clearly separate the methods (for instance reach, Push, slide (state) in firgure 2a). 



# Edits/Suggestions 
- I don’t like the notation $s_{t+}$. I think it could be replaced with something that is more understandable on first read (and looks less like a mistake). This is a preference though.
- Page 4: “Our proposed method (Sec 3)…” Should say Sec 3.2.
- Page 4: “we conjecture that our…” -> You should state that you test this in the empirical section (I think you do at least).
- Page 7: “We will to evaluate” -> “we will evaluate”
- If you reference a result in the main paper you should include this in the main paper. “TD InfoNCE achieves a 2x median improvement etc…” references figure 6 (I believe).

### Questions
See above.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Predicting future states is crucial for many time-series tasks, including goal-conditioned reinforcement learning (RL). While contrastive predictive coding (CPC) has been used to model time series data, learning representations that capture long-term dependencies often requires large datasets. This paper introduces a temporal difference (TD) version of CPC that combines segments of different time series, reducing the data needed to predict future events. This representation learning method is applied to derive an off-policy algorithm for goal-conditioned RL. Experiments show that the proposed method achieves higher success rates with less data and better handles stochastic environments compared to previous RL methods.

### Strengths
- The paper proposes a new temporal difference (TD) estimator for the InfoNCE loss, which is shown to be more efficient than the standard (Monte Carlo) estimator.

- The proposed goal-conditioned reinforcement learning (RL) algorithm outperforms prior methods in both online and offline settings.

- The proposed algorithm is capable of handling stochasticity in the environment dynamics.

- In stochastic tasks, there is an excellent improvement in performance versus the baseline of Quasimetric RL, with some healthy gains on non stochastic tasks versus other baselines, although this is not the primary target of the paper.

- The paper provides a clear and concise explanation of the proposed algorithm.

- The paper is well-written and easy to understand.

- The paper is well-supported by experiments.

- The proposed algorithm is evaluated on a variety of tasks, including the Fetch robotics benchmark.

- TD InfoNCE learns on image-based pick & place and slide, while baselines fail to make any progress.

- TD InfoNCE maintains high success rates on more challenging tasks with observation corruption, while the performance of QRL decreases significantly.

### Weaknesses
- The paper focuses on fairly trivial environments, it would be nice to see these methods working on more challenging higher dimensional goal conditioned RL tasks, as its not a given that these gains will carry over to tasks that matter a lot more.

- The proposed TD estimator is more complex than the standard (Monte Carlo) estimator and its implementation requires more hyperparameters.

- The performance of the proposed goal-conditioned RL algorithm on the most challenging tasks was less than 50%.

- QRL assumes deterministic dynamic of the environment, while TD InfoNCE learns without such assumption.

Loss Function Composition: The loss function L(θ) is composed of two cross-entropy (CE) loss terms, one for predicting the next state and one for predicting the future distribution of states. The γ hyperparameter is used to weight these two terms, but the choice of γ and its impact on the algorithm's performance are not discussed in detail.

### Questions
Can you explain how you selected the hyperparameters for the proposed algorithm?

Can you provide more details about the observation that TD InfoNCE learns on image-based pick & place and slide, while baselines fail to make any progress?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
