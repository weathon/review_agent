# CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Curriculum learning plays a crucial role in enhancing the training efficiency of large language models (LLMs) on reasoning tasks. However, existing methods often fail to adequately account for variations in prompt difficulty or rely on simplistic filtering mechanisms to select prompt datasets within a narrow criterion range, resulting in significant computational waste. In this work, we approach the problem from the perspective of reinforcement learning gradient optimization, offering a systematic and theoretical investigation into how to improve the training efficiency of LLMs. We identify two key factors influencing training efficiency: the selection of training prompts and the allocation of rollout quantities across different prompts. Our theoretical analysis reveals that the sampling distribution of prompts dictates the convergence rate of gradient descent, while the allocation of the rollout quantity influences the consistency and stability of overall gradient updates. Based on these insights, we propose CurES, an efficient training method that accelerates convergence and employs Bayesian posterior estimation to minimize computational overhead. Experiments demonstrate that our CurES outperforms Group Relative Policy Optimization (GRPO) by $\textbf{+3.30}$ points and $\textbf{+4.82}$ points with 1.5B and 7B models, respectively, and exceeds the best prior sample efficient methods by $\textbf{+2.12}$ points on average across eight math reasoning benchmarks. Additionally, CurES exhibits faster convergence compared to baselines, including GRPO.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies curriculum learning for RL finetuning of reasoning LLMs under a verifiable-reward setting. The authors analyze how different prompts contribute to learning progress by examining the relationship between success probability and optimization potential. They find that prompts with intermediate accuracy are the most informative and propose CurES, a principled task sampling and rollout allocation framework. CurES estimates each prompt’s accuracy using a Beta posterior, samples prompts according to a concave transformation of their estimated uncertainty, and allocates rollout counts to minimize gradient variance. The method is implemented as a multi-iteration procedure that periodically refreshes estimates to adapt to distributional shifts. Experiments using Qwen2.5-Math models show that CurES achieves faster convergence and higher final accuracy than GRPO, GVM, Speed-RL, and RPP.

### Strengths
- The paper provides a principled derivation linking sampling and rollout allocation to expected loss change and gradient variance, and successfully turns the analysis into a practical algorithm.
- The paper is well-structured.

### Weaknesses
- Narrow experimental scope. The training dataset is limited to Numina-Math, which limits evidence of general applicability to other task types or reward settings.
- Limited ablation studies. The paper does not include results for partial variants that use only task sampling or only rollout allocation, making it unclear how much each component contributes individually.
- Incomplete baseline comparison. Some relevant online data filtering or adaptive sampling methods, such as DAPO[1] and MoPPS[2], are mentioned but not compared experimentally. Moreover, the proposed Prompt Difficulty Assessment appears conceptually similar to MoPPS, and the paper should clearly articulate the differences. Additionally, utilizing optimization histories to guide efficient sampling shares similarities with Bayesian-based active sampling methods[3,4], which warrants further discussion in related work.
- Lack of discussion on prior theoretical overlap. The theoretical finding that tasks with intermediate success probability are the most informative has been observed or hypothesized in several prior works[5,6,7,8]. The paper should acknowledge these connections and clarify how its derivation provides new insight beyond existing results.
- Algorithmic clarity. Algorithm 1 is somewhat confusing, especially regarding the role and purpose of the pre-rollout stage, which requires additional explanation.
- Limited model diversity. All experiments are conducted on Qwen2.5-Math models, leaving uncertainty about whether the method generalizes to other model families such as LLaMA or Mistral.

[1] Dapo: An open-source llm reinforcement learning system at scale.\
[2] Can prompt difficulty be online predicted for accelerating rl finetuning of reasoning models?\
[3] Model Predictive Task Sampling for Efficient and Robust Adaptation\
[4] Fast and Robust: Task Sampling with Posterior and Diversity Synergies for Adaptive Decision-Makers in Randomized Environments\
[5] Self-Evolving Curriculum for LLM Reasoning\
[6] Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning\
[7] Not All Rollouts are Useful: Down-Sampling Rollouts in LLM Reinforcement Learning\
[8] SPEED-RL: Faster Training of Reasoning Models via Online Curriculum Learning

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper discusses curriculum learning for training Large Language Models with Reinforcement Learning (RL) for reasoning tasks. From gradient analysis, the authors demonstrate the intrinsic relationship between optimization efficiency and prompt sampling distribution, as well as rollout quantities. Therefore, an curriculum learning algorithm called CurES is proposed to adjust the sampling probability and rollout quantities based on the prompt difficulty. Experiments show that CurES improves the efficiency and stability of GRPO across several math reasoning tasks.

### Strengths
1. The paper is clearly written and well-motivated.
2. The theoretical analysis is interesting and aligns with people's understanding on RLVR, Eq (11) shows that training with a prompt with higher variance is more helpful; while Equation (20) indicate that more rollout should be allocated to the prompts with higher variance of gradients. 
3. The assumption of Beta-distribution in Section 4.3 address a good trade-off between theory and practice, leading to a simple yet practical algorithm.
4. From empirical results, CurES outperforms other baselines.
5. Comprehensive experiments demonstrate the effectiveness of proposed algorithm compared to baselines

### Weaknesses
## Section 4.1: Mismatched derivation from single prompt to dataset-level behavior.

From Eq (3) to Eq (11), the author analysis the impact of $|\mathcal{L}(x; \theta_{old}+d) - \mathcal{L}(x;\theta_{old})|$ to the variance of $r(x, y)$ based on Cramér–Rao inequality, which indicates that training on prompts with higher variance is more helpful. This is an interesting and meaningful observation. However, from Eq (11) to Eq (12), the derivation jumps from $|\mathcal{L}(x; \theta_{old}+d) - \mathcal{L}(x;\theta_{old})|$ to $|\mathcal{L}(\theta_{old}+d) - \mathcal{L}(\theta_{old})|$, which is actually **mismatched in the parameter update $d$**:
in the first expression $d$ denotes the parameter update w.r.t. a **single prompt** $x$, while in the second it implicitly refers to the **dataset-level** update. Ideally, the second one should be replaced by
$\bar{d} = \mathbb{E}_{x\sim\rho}[d]$. This implies that:

1) $d \neq \bar{d}$, ; therefore, it is not mathematically correct to generalize directly from a single prompt to the dataset average.

2) $\bar{d}$ also depends on your sampling distribution $\rho$, which is ignored. 

Unfortunately, all the subsequent derivations are built upon this mismatch.

**Q1**: How can the authors justify this mismatch between $d$ and $\bar{d}$? What's its influence to the analysis afterwards?

**Q2**: Is it accurate to describe the variance term $\sqrt{2δp_{θ_{old}} (x) (1 − p_{θ_{old}} (x))}$ by prompt difficulty? It's more related to the variance. For example, both too difficult and too simple prompt could have zero variance. 

## Section 4.2: Unexplained objective and beyond

In Section 4.2, the authors propose to optimize the objective Eq (15), which is $\mathbb{E}[(\mathcal{L}(\hat{\theta}) - \mathcal{L}(\theta_{old})  - \Delta_{theo})^2 ]$ (according to Eq (58)). This objective is the **Mean-Square Error (MSE)** between *actual improvement* $\mathcal{L}(\hat{\theta}) - \mathcal{L}(\theta_{old})$ and *theoretical improvement* $\Delta_{theo}$. However, the paper does not provide sufficient intuition for why minimizing this MSE corresponds to a principled curriculum-learning objective under a rollout budget constraint.

**Q3**: Why should Eq. (15) be viewed as a valid curriculum-learning objective? What is the intuition?

**Q4**: Since Eq. (15) builds on Section 4.1, it inherits the mismatched $d–\bar{d}$ issue. How does that affect the soundness of the subsequent results?

**Q5**: Moreover, the advantage estimation $A_{\theta_{old}}(x_i, y_i)$ also depends on the rollout number $n_i$ (and also the underlying policy gradient algorithm). This is ignored during the analysis.

Moving forward, the authors derive an optimal allocation of rollouts in Eq (20), i.e., $n_i = \frac{\sigma_i}{\sum_j \sigma_j} N$, where $\sigma_i$ is the variance of gradient of $x_i$ w.r.t $y$. Although the authors derive a symmetric computational form in Eq (21), it still seems computational intractable, especially for large langauge models. Concretely, the term $\mathbb{E}{y\sim\pi_{\theta_{\text{old}}}}\big[||\nabla_\theta \log\pi_\theta(y|x_i)||^2\big]$ requires computing the gradient norm w.r.t **each response $y_i$**, taking $N$ backward in total, while normal RL algorithm only needs to compute the average gradient once per batch. 

**Q6**: What's the **time** and **memory** overhead of CurES, especially on estimating $\sigma_i$? Currently this part is missing in the Experiment Section.  

## Other question

**Q7**: What’s the performance of baseline algorithms with different sample budget? Figure 5 only shows the performance of CurES.

### Questions
Please refer to the "Weakness"

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
This work proposes CurES to enhance the training efficiency of Reasoning LLMs. The authors derive a theoretical link between gradient optimization efficiency and both the prompt sampling distribution and rollout quantity allocation. Based on this, CurES adaptively adjusts sampling probabilities and rollout allocations using Bayesian posterior estimation of prompt difficulty. Experiments on mathematical reasoning benchmarks show that CurES achieves faster convergence and higher accuracy than baselinr methods.

### Strengths
- The work addresses a practically relevant problem—improving the training efficiency of reasoning LLMs.
- CurES reallocates both prompt sampling probabilities and rollout quantities to enhance the training efficiency.
- Many of the derived equations have closed-form solutions, which makes computation and optimization more tractable.
- The presentation of several figures is excellent. For example, the distributions of rollout quantities and estimated accuracies in Figures 3 and 4 are clearly illustrated, providing intuitive insights into the CurES’s behavior throughout the training process.

### Weaknesses
- In Eq. (12), the $\leq$ relation does not seem to strongly support the argument. Since the method aims to maximize the right-hand side (RHS), having a higher upper bound on the RHS does not necessarily imply a larger loss change in LHS. A $\geq$ lower bound would better motivate and validate the theory and the method.
- The training paradigm is restricted to iterative training across pre-divided dataset subsets, which limits generality. The way the dataset is partitioned may influence the algorithm’s performance. Moreover, each inner iteration requires rolling out every prompt to estimate the initial difficulty distribution, which introduces additional computational overhead.
- The methodology presentation lacks clarity. While the paper provides detailed explanations of the gradient theory and the estimation of prompt difficulty, it does not clearly describe how sampling and budget allocation are actually conducted. Before seeing Algorithm 1 in the Appendix, it remains unclear how the algorithm operates in practice.
- The paper could include more ablation studies to isolate the individual contributions of prompt sampling versus rollout allocation.
- The notation introduced in Section 5.3 (Line 460) needs revision for clarity and consistency. "pre-sampling scale (N)": This term is ambiguous. Moreover, it conflicts with the earlier definition of N as the "total rollout budget" on Line 224. "training-phase sample budgets coefficient (n)": This term is imprecise, and the symbol n is not defined previously in the text.

### Questions
- Line 350: "an assignment of rollout quantities under a total sample budget of 8×1024". Does it mean that the batch size for training CurES is set to 1024?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces CurES that selects prompts and determines the number of rollouts associated with each prompt during post-training based on the difficulty of the prompt in order to improve training efficiency.

### Strengths
The method is based on a theoretical analysis of the relationship between training dynamics and difficulty of prompts.

The experiments demonstrate that CurES improves average accuracy in mathematical reasoning across a range of model sizes compared with alternative curriculum approaches, namely Speed-RL and GVM (the quantitative results in Table 1).

### Weaknesses
What the experiments related to the sampling behavior (in Section 5.2) are supposed to demonstrate is not clear. In Figure 4, “the distribution of estimated accuracy … becomes more concentrated around higher values,” but wouldn’t we expect this to be the case for any learning method?

Although Speed-RL and GVM select the number of rollouts in a more straightforward manner, reporting their rollout numbers in Figure 3 would help highlight the difference of CurES.

In general, the sampling behavior analysis and the efficiency analysis in Sections 5.2 and 5.3 are only performed for CurES. Contrasting these results against Speed-RL and GVM can make these analyses more informative.

How do we arrive at the conclusion about Figure 5? Figure 5 doesn’t report the increase in computational cost with respect to N and N. How do we conclude they do not yield a proportional increase in performance? Assuming that’s true, how does this relate to the CurES’s efficiency?

### Questions
In Figure 6, does one step for CurES-GRPO/RPP have the same computational cost as vanilla GRPO/RPP? Would the results have been different if the x-axis was computational cost (for instance measured in FLOPS) rather than steps?

### Soundness
3

### Presentation
3

### Contribution
3
