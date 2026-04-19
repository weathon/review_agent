# MallowsPO: Fine-Tune Your LLM with Preference Dispersions

- Decision: Accept (Poster)
- Scores: 5, 8, 5, 8

## Abstract
Direct Preference Optimization (DPO) has recently emerged as a popular approach to improve reinforcement learning from human feedback (RLHF), leading to better techniques to fine-tune large language models (LLM). A weakness of DPO, however, lies in its lack of capability to characterize the diversity of human preferences. Inspired by Mallows' theory of preference ranking, we develop in this paper a new approach, the *MallowsPO*. A distinct feature of this approach is a  *dispersion index*, which reflects the dispersion of human preference to prompts. We show that existing DPO models can be reduced to special cases of this dispersion index, thus unified with MallowsPO. More importantly, we demonstrate empirically how to use this dispersion index to enhance the performance of DPO in a broad array of benchmark tasks, from synthetic bandit selection to controllable generation and dialogues, while maintaining great generalization capabilities. MallowsPO is also compatible with other SOTA offline preference optimization methods, boosting nearly 2\% extra LC win rate when used as a plugin for fine-tuning Llama3-Instruct.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes MallowsPO, which is based on DPO but is distinct in a few points, to fine-tune LLMs from human feedback. The main two distinctions are as follows; (1) Instead of assuming that the preference between the two sentences $y_1$ and $y_2$ is sampled from some probability distribution, the proposed approach assumes that the ranking ($\mu$) of $n$ items are sampled from distribution, and (2) assuming that the preference has some disparity depending on the input text. The proposed MallowsPO formulates a DPO-like objective function for the ranking assumption while incorporating the disparity by introducing a disparity parameter in the reward function.

### Strengths
- **Formulating disparity in preferences**: while the original preference model assumes the same preference distribution across different input sentences ($x$), this paper spotlights the fact that the preference distribution can change depending on ($x$) and formulate the difference in reward function.

- **Better empirical results by formulating disparity**: It seems that the result suggests that considering the disparity can improve the reward for varying values of KL divergence.

### Weaknesses
- **Assuming the probability distribution of ranking, rather than the pairwise preference, loses information about reward difference.**

The paper considers the probability distribution in ranking from the similarity between the ranking $\mu$ and so-called "central" ranking $\mu_0$. For example, Mallow-$\theta$ models the probability as a function of Spearman's rank difference to the central ranking. Unfortunately, this formulation loses the performance difference between each item. 

For example, consider the case where we have three sentences, $(y_1, y_2, y_3)$. Also suppose that the reward of each is $(100, 1, 0)$. In this case, under the assumption of the paper, the probability of observing the ranking of $(100, 0, 1)$ and $(1, 100, 0)$ is the same, while the former is definitely more likely to be observed given the reward difference. For this reason, the proposed approach loses the information about the reward, and I could not find any advantage of assuming probability distribution on ranking rather than on pair-wise preference (pair-wise preference already implicitly encodes the probability of ranking, without losing the information about rewards).

- **Insufficient ablations: lacking baseline of <DPO+disparity>**

The main motivation of this paper is to take the disparity, or the difference of preference distribution depending on the inputs $x$ into account. Then, why isn't simply considering the disparity in the formulation of DPO sufficient? That is, because the preference distribution of DPO is assumed as 

$p^{\ast}(y_1 \succ y_2 | x) := \sigma(r^{\ast}(x, y_1) - r^{\ast}(x, y_2))$, where $\sigma(z) = 1 / (1 + \exp(-z))$,  

I think using $\phi(x) \cdot z$ (where $\phi(x)$ is the estimated disparity params) as the input of $\sigma(\cdot)$ is enough to consider the disparity. This simple baseline is not compared in the experiments.

---

**(summary)**
Overall, I did not understand why this paper needed to introduce the ranking formulation, which actually seems less beneficial than DPO. The latter, considering the disparity can be a unique contribution to this paper, however, this point seems somewhat incremental and weak as the ICLR paper. This is the reason for my evaluation.

### Questions
[questions]
- For Figure 9, how does the training curve differ for different prompts, i.e., $x$? It may be interesting to see how the action choice probability can change with the estimated disparity parameters ($\phi(x)$).


[nits]
- The label of Figure 8 (b) is maybe a mistake: I think it should be # of epochs rather than KL.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents a novel approach to Preference Optimization for LLMs, named MallowsPO. The approach is inspired by the Mallow's theory of preferences ranking which assumes the preference functions to follow specific distributions parametrized by a dispersion index. Importantly, the dispersion index is prompt-dependent and models the fact that different prompts may have more or less concentrated preference distributions. The authors derive "Direct PO"-like versions of their approach and provide a way to estimate dispersion indices by Shannon entropy. Experiments show that MallowsPO outperform DPO.

### Strengths
I believe the paper has the following main strengths:
- The problem studied (i.e. LLM alignment to preference data) is very relevant
- The connection with Mallow's ranking models is, to the best of my knowledge, novel and meaningful
- The derived approach is sound
- Experiments are somewhat extensive and demonstrate the benefit of the introduced approach

### Weaknesses
Weaknesses are: 
- The authors could do a better job at illustrating qualitative differences between policies trained with DPO and MallowsPO. In particular, one could plot entropy during training, number of tokens, distribution of rewards (when available) to make their case stronger and to understand what are the concrete differences between the two approaches. Part of this is left as future work, but I believe it would be nice to discuss/analyze it in the current paper
- The discussion and estimation of the dispersion index is not fully clear to me. Also, would have been nice to ablate the sensitivity of MallowsPO to other possible estimation approaches (or variants of chosen one).

### Questions
I do not fully understand the motivation behind the dispersion index and its estimate.
In particular, the index is estimated using the per-prompt entropy of the reference policy, but: 
- on how many generation? does this require drawing more samples for an accurate estimate?

More importantly, 
- Shouldn't the dispersion index indicate the dispersion of the unknown preference model (i.e. humans)? If so, I do not see why one should use the entropy of the SFT model (which we would still like to align to human preferences) to estimate such index.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces a preference optimization method called MallowsPO, which extends DPO to incorporate human preference diversity. Authors leverage Mallows’ theory of preference ranking and propose a dispersion index that accounts for variations in human agreement, allowing diverse human preferences on prompt responses. Empirical studies are performed on benchmark tasks to show the performance by using the proposed dispersion index. Comparisons are done with other SOTA offline preference optimization methods.

### Strengths
This paper presents an extension to DPO with theoretical guarantees and demonstrates practical improvements in LLM fine-tuning on a set of benchmark tasks.  The motivation behind is clearly justified and the studied setting is of practical interests.

The introduction of preference dispersion through Mallows’ ranking theory is interesting in preference optimization and is presented clearly. The paper could, however, benefit from additional clarity in Section 3.1, where the transition from traditional DPO to MallowsPO is described. Specifically, more examples or visual aids could assist readers unfamiliar with Mallows’ theory.

The paper provides a detailed decomposition of the reward function based on dispersion, which is mathematically well-grounded and introduces two variations (MallowsPO-$\theta$ and MallowsPO-$\phi$) using different discrepancy functions. This addition allows for flexibility and generalization beyond existing DPO methods.

### Weaknesses
The use of a dispersion index adds interpretability to preference-based tuning, yet further details on how practitioners could leverage these insights in real-world applications could enhance the paper’s practical relevance.

While the empirical results are promising, additional insights into the MallowsPO model’s performance under varying preference dispersion scenarios (e.g., low vs. high dispersion prompts) could strengthen the evaluation. This might highlight more specific cases where MallowsPO demonstrates clear advantages over other DPO variants.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work introduces a novel technique for fine-tuning LLMs called MallowsPO, which is inspired by Mallows' theory of preference ranking. MallowsPO utilizes a dispersion index to reflect the dispersion of human preference to prompts. Existing DPO models are special cases of this dispersion index. Additionally, MallowsPO is compatible with other SOTA offline preference optimization methods.

### Strengths
1. This paper is clearly written and well structured. The authors provide sufficient analytical and empirical comparisons with previous methods.

2. The contribution to the methodology for fine-tuning large language models is significant. This work proposes a novel preference optimization framework based on Mallows' preference ranking theory beyond the BT setting, which provides a novel perspective on preference modeling. This new method introduces an important component, the dispersion index, to address the shortcomings of DPO in characterizing the diversity of human preferences. This work is technically sound and offers a detailed theoretical analysis. MallowsPO is a generalization of the well-known DPO method and can cover DPO as a special case under a certain parameter setting. 

3. Besides its theoretical contribution, this work also has significant empirical contributions. It proposes empirical approximations to the dispersion index to facilitate computation. Moreover, extensive experiments for the proposed methods are conducted under various setups, including synthetic bandits, controllable generation, fine-tuning Pythia 2.8B on off-policy Anthropic HH dataset, and fine-tuning Llama3-8B-Instruct on a hybrid UltraFeedback dataset, to show the advantages of MallowsPO. Moreover,  MallowsPO is a sufficiently general approach that can be integrated with other offline preference optimization methods, such as simPO and IPO, which have better performance compared with their vanilla counterparts.

### Weaknesses
None.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3
