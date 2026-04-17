# Autoregressive Direct Preference Optimization

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Direct preference optimization (DPO) has emerged as a promising approach for aligning large language models (LLMs) with human preferences. However, the widespread reliance on the response-level Bradley-Terry (BT) model may limit its full potential, as the reference and learnable models are assumed to be autoregressive only after deriving the objective function. Motivated by this limitation, we revisit the theoretical foundations of DPO and propose a novel formulation that explicitly introduces the autoregressive assumption prior to applying the BT model. Specifically, we first reformulate the origin of DPO using two Boltzmann distributions with reward-based energies defined over the output (response) space $\mathcal{Y}$. We then extend the energy domain from $\mathcal{Y}$ to its prefix closure $\mathcal{Y}^{*}$. Interestingly, this simple extension naturally leads to the energy definitions with an autoregressive reference model, the prefix-wise BT model, and ultimately, a novel DPO variant called Autoregressive DPO (ADPO) with its corresponding loss function. Without violating the theoretical foundations, the derived loss takes an elegant form: it shifts the summation operation in the DPO objective outside the log-sigmoid function. Furthermore, through theoretical analysis of ADPO, we show that there exist two length measures to be considered when designing DPO-based algorithms: the token length $\mu$ and the feedback length $\mu'$. To the best of our knowledge, we are the first to explicitly distinguish these two measures and analyze their implications for preference optimization in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Although DPO have achieved an effective way of training LLMs to align with human preferences, the preference signal of DPO is compared at the sentence level. Therefore, the autoregressive property is represented after fully generating the reponses. This paper suggests a new method to explicitly represent the autoregressive propoerty to the bradley yerry model. This paper theoretically show every reward model can be reformulated as the autoregressive version. and experimentally show improved performances over the baselines.

### Strengths
- Another trial to decompose the reward calculation from sentence to token.
- Theoretical consistency between reward modeling and autoregressive decoding is well-motivated.

### Weaknesses
- Additive decomposition may not be true because it is a sentence. There may be some correlation between words.
- Experiment is not enough. Both GSM8K and MATH500 are limited to math reasoning tasks. Why not more general human preference dataset, e.g. UltraFeedback? I suppose math task is not adequate to show the granularity consideration is effective, because after all the ultimate objective of math task is to solve the questions with explicit answer. I also saw there is "conversation tasks" section. What did the authors do in this section? Training on mah data and test on conversation benchmark?
- Only the naive DPO, cDPO and SimPO are compard as baselines if I count for the whole experiments; yet some were only partially experimented. Why no results of cDPO for Table5?, Why no results of SimPO for table3?
- No explanation on evaluation metric. I know what those are, but it severely compromises the quality of the paper.

### Questions
- What can be an intuitive explanation of ADPO to maximize the log likelihood ratio of winning response's token and losign response;s token at a specific time t? I understood this means finer granularity whether than taking the whole responses just as 1. However, the reason why previous researches took the total chunk of responses as 1 was because a response will finally have a complete meaning only a reponse has ended. Therefore, taking just a word difference of same time between a positive response and a negative response may not have a specific meaning. This signal becomes more harder to understand when the format of winning response and losing response becomes more differernt. 
- What do the authors mean by "In Table 6, we also find that granularity exhibits a similar trend as in mathematical reasoning tasks:" ? Table 6 is the result of AlpacaEval2?
- With so unaligned experimental results section as above, how can I trust the reproducibility of the experimental results at all?
- Model performance of training without LoRA?
- I am willing to change my opinion to accept if my concerns with regard to the experiment parts are well explained.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, a reformulation of the standard DPO framework, called Autoregressive Direct Preference Optimization (ADPO) is introduced. The core problem of DPO is the application of the Bradley-Terry model at the level of complete responses, and when the autoregressive property of the language model is considered only after the target response is defined. ADPO resolves this issue by explicitly incorporating the autoregressive assumption before the application of the BT model.

The authors achieve this by applying the domain of energy functions based on the reward signal from the output space $\mathcal{Y}$ to the prefix space $\mathcal{Y}^{*}$. Specifically, the prefix-wise BT (Equation 10) model is developed, which determines the probability of the preference as the product of preferences over the prefixes.Theoretically, the authors prove the reparameterization completeness and distinguish the difference between two length metrics: token length $\mu$ and feedback length $\mu’$. The case of $\mu’=1$ corresponds to DPO. Additionally, two families for implementation are proposed – Static and Adaptive. Empirically, the authors demonstrate improvements on mathematical and conversational datasets, simultaneously, for the DPO and other competitive methods (cDPO, SimPO).

### Strengths
1. The observation that standard DPO amounts to applying the BT model at the response level, singling out the autoregressive structure after deriving the objective, is a significant theoretical mismatch in the existing formalization. ADPO rectifies this issue by introducing an energy definition over the prefix closure and explicitly assuming an autoregressive reference model at that point. This theory is much closer to the reality of how an LLM actually generates text.
2. It is clearly shown that how they used a Boltzmann distribution to re-formulate the foundations of DPO in Section 3 while providing the derivation of the ADPO loss in Appendix A makes for a robust argument. Another theoretical contribution is the introduction of the two length measures in Section 5.3.
3. In Section 5.4, the introduced "Granularity Families" offers guidelines for exploring the continuous optimization space from token-level to response-level. Further on, using "strong composition" for the definition of the mapping from tokens to concepts to subsequences is a reasonable decision.
4. ADPO, especially at finer granularities, achieves a more pronounced separation between the log probabilities of preferred and dispreferred sequences during training than DPO. This suggests that improved "information per step" is a correct intuition for ADPO, and its inspiration from the early-dominant-prefix guidance in prefix-wise preference training is accurate.

### Weaknesses
1. About the implicit reward function $r^*(x, y_{\le i})$: In the standard DPO, the implicit reward has an explanation related to the complete response. The optimization in ADPO is tied to the delivered localized rewards. What these rewards could be? Have they a variance? How do they ‘credit’ prefixes for being ‘good’ or ‘bad’ in the response? 
2. About effect of moving the summation outside the log-sigmoid function that changes the non-linearity: we know that DPO computes the total advantage, the sum of log-ratios, and then applies the sigmoid function once. Thus, DPO is sensitive to the total margin, but the sigmoid collapses all but very large margins. In contrast, ADPO fully applies the sigmoid to the advantage of each token or subsequence and only then sums the results. But such a change in the non-linearity will change the learning dynamics. For example, it becomes much more challenging to tell a sequence if a cut cost is very high but concentrated on one token is worse than a sequence where the same total costs are spread across many tokens after accumulation because the local advantages are now squashed before accumulating. This alteration affects optimization and regularization, which need further discussion.
3. For experiment: The motivation for ADPO is a drive for finer-grained optimization, this is not unique: it is the same as the motivation for token-level Reinforcement Learning from Human Feedback. Therefore, the paper should contain comparison with an optimized implementation of token-level PPO. The claim is that DPO is generally more efficient than PPO. While this may be the case for ADPO, you can further establish the relationship between ADPO and token-level RL optimization.
4. About Static Family: Section 5.4 introduces the necessity for EOS padding to make the lengths of the preferred and dispreferred responses equal before the decomposition into subsequences. The authors themselves acknowledge that this is a critical restriction, writing that EOS padding “potentially constrains its performance.” Furthermore, deriving sub-transformers from substrings that have been distorted by optimizing over padded tokens introduces noise and inefficiencies in the form of an indistinguishable prefix problem. It seems like the Adaptive Family is a more viable approach?

### Questions
1. About prefix-wise BT model (Eq. 10): independence assumption: Human preference can often not be expressed by decomposable into independent prefix preference. Did you experiment for cases where this assumption is skeptic, such as the task where correctness/preference is only controlled by the target of the prefixes, i.e., the final answer in a math problem, and how does ADPO perform in such cases compared to DPO? 
2. Also refer to [weakness 2]. And how does this influence the weights of training examples, particularly comparing to short and long sequences? Does this implicitly add the length normalization or regularization different than DPO or SimPO?  it is also fine to explain something about weakness 2.
3. Any analysis on the implicit prefix reward learned by ADPO? I am wondering about the behavior of these reward values that are localized. How do the variance and magnitude of those prefix rewards change during training, and how is the credit distribution learned to the sequence compared to the DPO’s implicit reward? 
4. Also, refer to [weakness 4]. For that, it is okay to provide some explanation as well. Or did you come up with an alternative that might combine the benefit of both approaches or other ways to handle the variable lengths or defying the subsequence, perhaps by the user’s natural semantic instead of length?
5. Does  ADPO (at the finest granularity) introduce any computational overloads compared to DPO?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper derives an autoregressive variant of DPO where rewards are assigned not only to complete responses but also prefixes which results in ADPO where the summation and log-sigmoid are swapped. Variants based on different way to dividing the responses are also tested. They demonstrate improved performance over DPO and other baselines on both math reasoning datasets as well as conversational datasets.

### Strengths
Novel objective - The paper considers deriving the objective building in an assumption of having autoregressive models. They demonstrate that the resulting objective still achieves the optimal policy and also demonstrates that DPO is a special case of the generalized ADPO objectives. They also provide a thorough analysis of the resulting objective and present it in a clear way. 

Experimental results - Experimental results demonstrate consistent improvement across multiple tasks and benchmarks and also study the effect of different factors. Details on datasets, models, baselines, and implementation as well as additional exploration of hyperparameters are provided and clear, further supporting the benefits of ADPO.

### Weaknesses
Impact - While the benefit of ADPO is clear, one thing that is unclear from the paper is why the model being autoregressive and the output distribution not being autoregressive is considered to be mismatched and why it is expected to be an issue. There are different ways of chunking the outputs as seen with the variants of ADPO, so it seems unclear why treating the output as a single object should lead to issues. I think further reasoning here being provided would strengthen the paper and make the impact of the paper more clear beyond exploring a new dimension that the DPO objective can be changed along.

### Questions
- Why is the model being autoregressive and the output distribution not being autoregressive considered to be mismatched, and why is it expected to be an issue?

Addressing this question and clarifying this in the paper would significantly help.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a variant of Direct Preference Optimization (DPO), called Autoregressive DPO (ADPO). The origin of DPO is reformulated using two Boltzmann distributions with reward-based energies defined over the output space $\mathcal{Y}$. By extending the energy domain from $\mathcal{Y}$ to $\mathcal{Y}^*$, the energy definition with the prefix-wise BT model is achieved. One of the biggest differences of ADPO from the original DPO is that the summation across the tokens is moved outside the log-sigmoid function, allowing training with higher granularities. The paper presents several experiment results showing how the performance of ADPO changes along the change of granularity applied to the training process, while ADPO shows significant improvement from the original DPO at its optimal ranges of granularity.

### Strengths
1. Theoretical soundness: The paper effectively explains how a different perspective in the DPO formulation can lead to token-wise interpretation, which also allows having a new loss with controllable granularity.
2. The paper is well-organized in terms of logical flow and visual presentation.

### Weaknesses
1. I think the biggest issue of this paper is that it is not citing and comparing with a very similar analysis from the authors of the original DPO paper [1]. While there are a few differences in terms of formulation or notations, [1] also present a token-level perspective of the DPO formulation and that token-level DPO can parameterize any dense reward function (section 4.2 of [1]). 
2. While the paper is presenting several results where ADPO outperforms DPO, it seems it is still lacking a reasonable explanation why ADPO is supposed to perform better than DPO.
3. The experiment results are not showing standard deviation of the test performances, which are crucial for reinforcing the credibility of the results.

[1] "From $r$ to $Q^*$: Your Language Model is Secretly a Q-Function", Rafailov et al., 2024.

### Questions
## Questions
1. What could be the reason ADPO is performing better than the original DPO? Is ADPO basically generating more training signals than DPO, given the same amount of prompts?
2. Do ADPO and DPO exhibit similar wall clock time for training? Does the training time of ADPO change along the change of granularity level?
3. How does ADPO affect the overall response length, compared to other algorithms?

## Suggestions
1. Like I mentioned in the Weakness section, I think it is crucial to cite (Rafailov et al., 2024) and compare the contribution of ADPO's paper with it.
2. Figure 2: I think it is better to use the same color for the same algorithm across the plots.
3. I think presenting the performances of the base model, without any fine-tuning, in the experiment results will benefit the paper.

### Soundness
3

### Presentation
2

### Contribution
2
