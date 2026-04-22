# KL-Regularized Reinforcement Learning for Generative Modelling is Designed to Mode Collapse

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 2, 8, 6

## Abstract
Classical intuitions cast minimizing reverse KL as "mode seeking" and forward KL as "mass covering". In KL-regularized reinforcement learning, however, the regularizer determines _both_ the target distribution's shape _and_ the divergence being implicitly minimized, making its role more nuanced than simply inducing generic mode-seeking or mass-covering behaviour. Specifically, the target distribution is defined jointly by the reward function, the reference model, the type of regularizer, and the regularization strength. We show that under common settings—such as low regularization strength and equal verifiable rewards—both forward and reverse KL regularization tend to specify target distributions whose mass concentrates on a single high-reward region. Thus, the objective itself _by construction_ induces diversity collapse, regardless of the policy optimization algorithm used.

Building on this perspective, we introduce a simple and scalable modification that rescales rewards to induce target distributions assigning substantial probability across _all_ high-reward regions. This yields a principled objective that maintains high solution quality while achieving broad reward-mode coverage. Empirically, this approach improves post-training diversity and performance for Large Language Models and Chemical Language Models, and is effective with either forward or reverse KL regularization, while using either naively fails.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper analyzes the reasons for "mode collapse" when using KL-regularized RL for LLM post-training, linking it fundamentally to the shape of the target distribution. For this, the paper revisits earlier-made arguments about the shape of the target distribution that Reverse KL-regularized RL approximates and makes (new?) arguments about the target distribution of Forward KL-regularization. Using insights from this analysis, the authors propose a new MARA algorithm that is claimed to approximate a target distribution with uniform high probability modes for highly-rewarded samples, and sticks to the reference distribution otherwise.

### Strengths
- S1) The paper addresses a highly debated topic at the moment on which a lot of noise and confusion exists.
- S2) The paper is quite clearly written and the explanations are easily followable.
- S3) The paper introduces a new method based on a theoretically clear argument.
- S4) The analysis of the forward KL-regularized objective is quite interesting if new, even though I'm not sure it has much impact on the paper's main narrative.

### Weaknesses
- W1) I feel that part of the discussion increases --rather than dispelling--  the confusion. 
In particular, starting from the first two sentences of the abstract:
  "It is commonly believed that optimizing the reverse KL divergence result in “mode
seeking”, while optimizing forward KL result in “mass covering”, with the latter
being preferred if the goal is to sample from multiple diverse modes. We show—
mathematically and empirically—that this intuition does not necessarily transfer
well to doing reinforcement learning with reverse/forward KL regularization (e.g.
as commonly used with language models)."
I think this is either false or misleading at best. One thing is the kind of regularization used in the loss and another is the divergence implicitly optimized by the training objective (remark 3.2). It is the latter that can be ascribed "mode-seeking" or "mode-covering" behavior, whereas here the authors are referring to the divergence used in the regularization term. If the goal of the paper is clarifying these aspects of LLM post-training using RL, it could do a much better job at keeping this distinction clear.
- W2) Furthermore, there are a number of inaccuracies sprinkled across the paper. One of the most important ones is in one of the main messages of the paper, which is that the shape of the target distribution is responsible for the lack of diversity (again, in the abstract, "Further, we show commonly used settings such as low regularization strength and equal verifiable rewards tend to specify uni-modal target distributions, meaning the optimization objective is by construction non-diverse."). This looks like a misconceived conclusion from the analysis in the paper. The illustration in Section 3.3 corresponds to a reward function that assigns non-uniform values to rewarded samples, whereas the illustration in Figure 5 compares "on-support" with "off-support" sequences (see also Q6 about this terminology). Actually, as noted in Remark 4.3 "[the] relative probabilities in the solution is simply the relative probabilities in the reference
distribution, and do not depend on the KL-regularization strength $\beta$". Doesn't then this contradict the sentence in the abstract?
- W3) While the proposed algorithm is interesting, there is no comparison with other diversity-enhancing baselines in the empirical part such as rewarding the unlikely or pass@k training.

### Questions
- Q1) L31 why do you say RL is the only way to train models where the solution is not known a priory? You already cite Go et al. (2023). Wouldn't this fit this bill?
-Q2) What is the role of remarks 3.2, 3.3 and 3.4 in the overall story of the paper? I felt that they were made and forgotten.
-Q3) L174: when you say that ML training on samples from $G_\beta$ is intractable, you mean for an unbounded reward function, right? You mention STaR, but you could make it more explicit that this algorithm is doing exactly what you are saying is intractable for a particular class of rewards on which we can do rejection sampling. More importantly, note that you don't necessarily need to sample from $G_\beta$ to approximate this objective. $G_\beta$ is particular case of target distributions described as "distributional constraints" in https://arxiv.org/abs/2012.11635, and the forward KL objective can be approximated using the DPG algorithm.
- Q4) What is the policy you are using for Fig. 2 and following?
- Q5) Definition 3.5: just to double check, do you really mean that _all_ high-reward samples must have high probability? OK, but this doesn't match my intuition for multimodality. Maybe you are looking for a different concept?
- Q6) When you say "off-support", you mean $p(y)<\epsilon$, right? Correct me if I'm wrong, but I think mathematically off-support means $p(y)=0$. You might need to define this other concept of support to use it here.
- Q7) Many recent papers set $\beta=0$ (e.g. Dr. GRPO). What happens when your algorithm for this particular case in the scenario of verifiable rewards? It looks to me that it boils down to regular RL, and cannot prevent the loss in diversity.
- Q8) L361 "stays close to the reference in regions of low reward" -> why is that the case?
- Q9) Following up, can you say something about the target distribution of MARA?
- Q10) Figure7: the KL parameter in the legend is $\beta$?
- Q11) What is In dist and out dist reward?
- Q12) Could you add a few words about the REINVENT algorithm?
- Q13) Why does the threshold affects REINVENT?
- Q14) "In this work, we provide an in-depth understanding of the KL-regularized RL objective, particularly in terms of its diversity" -> could you summarize it in a few words?

Typos/style/corrections
- L107 "maximize a reward function" -> "maximize the expected value of a reward function"?
- L149 "we can regluarized" -> we can regularize
- L149 "maximizaztion" -> "maximization"
- L208 "token space" -> sequence/output space ??
- Figure 7 caption: "entrpy" -> entropy
- L426 "adpat" -> adapt

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the effect of KL regularization in the objective function of LLM post-training from the perspective of diversity, e.g., the multi-modals in the learned distributions from the final policy. The authors compare the optimal policy under the reverse and forward KL regularizations under the KL-regularized objective in LLM post-training. Based on the analysis, they proposed to put equal weights on all high-quality samples to encourage the multi-modal target. Experiments are conducted on several LLM post-training settings.

### Strengths
1. Understanding the learning behaviors, e.g., the multi-modality of the learned distribution, of the KL-regularized objective function in LLM post-training is an important research problem.

2. The usage of illustration improves the readability, although in simple settings.

### Weaknesses
1. **The scope of the paper is questionable**. The title of this paper targets the general RL, but I am surprised that the scope of this paper only focuses on the KL-regularized objective function used in LLM post-training. There are a flurry of prior RL works on pure KL-regularized RL, such as MaxEnt RL, Soft Actor Critic, and natural policy gradient, that focus on the MDP setting of the general RL setting. By contrast, the KL-regularized objective in LLM only assumes a specific contextual bandit setting with iid samples, which is far less representative of the general RL setting. Using such a title is very misleading. Researchers in LLM post-training should respect prior works in the general KL-regularized policy gradient methods if they really want to target a general research problem. Alternatively, it is required to narrow down the scope and title of the paper to the LLM setting without causing confusion, which is just a specific application of general RL in the language scenario. Lots of analyses are very divergent.

2.  **Less convincing theoretical analysis**. 
* The paper relies on the fact that reverse and forward KL have the effect of mass covering or mode seeking, but I do not think they are established in a fundamental way. The authors are suggested to provide sufficient references, either from information theory or statistics; otherwise, the subsequent analysis would be less convincing. Also, some illustrations on experiments are too simple to make a rigorous conclusion. This issue occurs many times across the paper. 
* Is the diversity the sole objective in the policy optimization? In statistics, the reason why people use MLE or equivalent KL divergence is that it ensures statistical efficiency, asymptotic unbiasedness, and consistency, often on iid samples. However, if you manually introduce the sample selection bias by focusing on the diversity to encourage multi-modality, there is definitely some price to pay. However, from the theoretical perspective, I did not find such a deep analysis or conclusion regarding this point. After reading Line 102, does it mean that in LLM the foundation models are often less expressive such that we need to analyze the difference of two KL divergences? A lot of details are unclear to me. 
* This paper makes a lot of strong claims, such as ‘diversity collapse is a natural consequence of solving the RL problem.’ It is very risky to accept these claims without finding rigorous proof or strong empirical evidence or articulating the specific setting of the RL problem they are studying among the whole RL literature. 
* In Eq. 5, there is no closed-form solution under forward KL divergence, where $\Lambda$ is the negative of the Lagrangian multiplier. However, it does not necessarily mean that the optimizers under the two KL divergences are completely different in the general case (line 163)! 
* The analysis in Section 4 is performed under very specific scenarios. Although sometimes they provide some insights, in general they are very unlikely to hold in the training process. Such an analysis is fragile and less convincing. The result in Proposition 4.1 is trivial. The logic in Section 4 is very hard to follow.

3. **Unclear and informal writing**. I personally think a lot of writing is not formal or rigorous.   
* What is the definition of ‘solution distribution’? I understand what it is by guess, but the authors did not give a definition or explanation when they used it many times.   
* What is the approximate inference in Line 53? I do not think it is a scientific word.   
* In Line 92. What is the definition of two-parameter Gaussian? Is the foundation model as an example of a flexible distribution in line 93?   
* It is inaccurate to say we aim to maximize a reward function in Line 107. Instead, we are maximizing the average rewards or the expectation of the returns. What is the definition of "off-support samples in Line 264. There are lots of expressions that are not formal, which is discouraged to me.   
* There are also many grammar errors, such as ‘This let’ in Line 232 and ‘KL exhibit’ in Line 91. The writing needs substantial improvements.

4. **Unclear empirical validations**. Since the paper focuses on the improvement of diversity, I could not find a detailed explanation about the metrics of diversity. Learning curves in Figure 7 are blurred, and thus it is hard to make a clear conclusion on it. Also, it makes more sense to me if the LLM focuses on the diversity; the efficiency would be reduced to some extent. Sometimes this price is worthy to pay, but I could not find such an analysis in the experiments.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies KL regularization in RL finetuning of language models from first principles. This paper lays out various facts about the global solutions of the forward and reverse kl regularization + RL finetuning process:

1) RL + reverse KL regularization is equivalent to minimizing reverse kl with its globally optimal solution (known previously)
2) RL + forward Kl regularization does not optimize any form of forward KL.
3) In RL + forward KL, the beta coefficient acts as an exponential decay / growth term in the solution. Standard values can essentially make the best solution orders of magnitude more likely than other solutions. This is known but often overlooked.
4) When rewards are equal, off policy solutions are discouraged. This is also obvious (looking at equation 2), but still overlooked.

Finally, for KL regularization + RL finetuning, they provide a simple heuristic to boost multi-modality and present experimental results verifying their approach.

### Strengths
I think the summary^ is a fair assessment of the paper's strengths. It is presents simple results in a very clear and easy to read manner. According to me, the heuristic presented seems to be not that significant in terms of usability in frontier models for example. But I could be wrong on that. For the theoretical results, they are simple in hindsight, but are often overlooked in the literature. Remark 3.3 seems to be the most non trivial and novel result in my opinion.

Overall I enjoyed reading this paper and I think this paper should be accepted.

### Weaknesses
The only weakness I see is a lack of strong baselines. LLM post training is not my primary area of research but surely their have to be many multi-modality inducing algorithms for post training (RL exploration, gflownets, feature  covariance maximization, etc). I do not expect your algorithm to be better than them, but a comparison might still be useful for readers.

### Questions
See weaknesses

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper looks at the behavior of policy gradient optimization with respect to the forward vs. reverse KL-regularized expected reward, where it is demonstrated both theoretically and empirically that (i) either choice of divergence can, in the right circumstances, cover the various modes of the target distribution and (ii) there are circumstances in which neither will appropriately cover the various modes. The authors propose a modification to the reward function, MARA, such that policy gradient optimization with respect to the forward or reverse KL-regularized expected modified reward assigns high probability to all actions that induce a sufficiently high reward.

### Strengths
This paper does a good job highlighting that common intuitions about the mode-seeking vs. mass-covering nature of different divergence choices does not obviously result in the expected behavior that is often highlighted by 1d examples. Indeed, for expected reward regularized divergence minimization, the authors demonstrate in a convincing way both formally and through experiments how the choice of reference policy, regularization strength, the shape of the reward function are all highly relevant factors in determining whether the learned policy covers the various modes of the reward function. I found the authors solution to this (mode anchored reward augmentation) to be an original one and well motivated by the paper's exposition. The paper's experiments are well constructed and show on a variety of pertinent RL problems that MARA generally improves diversity of the resulting policy.

### Weaknesses
The MARA reward function itself feels a bit unsatisfying in that it requires knowledge about reasonable choices of tau (and therefore some knowledge/belief about the distribution of attainable reward values). I wonder to what extent this makes MARA suitable in settings where rewards can be potentially unbounded in magnitude (e.g., certain Atari games). The need to have a high quality anchor example also feels like a limitation on the problems that MARA may be applicable to, e.g., problems where the reward signal is very sparse, or where high reward examples are rare a priori. Further, it seems that redefining the reward to a constant for all sufficiently high reward examples may not be appropriate for all problems.

### Questions
Do you have an explanation for why MARA has lower "semantic embedding" diversity than the baseline?

How sensitive is the behavior of the MARA learned policy to choice of tau? It would be helpful to have a figure or result dedicated to this (I was not able to find it in the paper, please point it out to me if I overlooked it). Also, how did you go about selecting the value of tau in experiments? 

In the REINVENT task, is there a good explanation for why using MARA as a drop-in replacement did not lead to improvement on the two diversity related metrics? Have you compared the number of, e.g., ECFP4-based Tanimoto clusters for compounds sampled from REINVENT vs. REINVENT with MARA objective?

How sensitive is MARA to the choice of the anchor z?

### Soundness
3

### Presentation
2

### Contribution
3
