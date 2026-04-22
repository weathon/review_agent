# Selective Expert Guidance for Effective and Diverse Exploration in Reinforcement Learning of LLMs

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has become a widely adopted technique for enhancing the reasoning ability of Large Language Models (LLMs). However, the effectiveness of RLVR strongly depends on the capability of base models. This issue arises because it requires the model to have sufficient capability to perform high-quality exploration, which involves both effectiveness and diversity. Unfortunately, existing methods address this issue by imitating expert trajectories, which improve effectiveness but neglect diversity. To address this, we argue that the expert only needs to  provide guidance only at critical decision points rather than the entire reasoning path. Based on this insight, we propose MENTOR: Mixed-policy Expert Navigation for Token-level Optimization of Reasoning, a framework that provides expert guidance only at critical decision points to perform effective and diverse exploration in RLVR. Extensive experiments show that MENTOR enables models capture the essence of expert strategies rather than surface imitation, thereby performing high-quality exploration and achieving superior overall performance. Our code is available online.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes MENTOR, a reinforcement learning framework for large language models that enhances reasoning by providing expert guidance only at critical decision points rather than on entire reasoning paths. This approach promotes both effective and diverse exploration, enabling models to capture strategic essence rather than superficial imitation, which leads to superior performance compared to existing RLVR methods.

### Strengths
- This paper is easy to read and easy to follow
- This paper focuses on an important topic in large language model reasoning, aiming at enhancing the exploration capabilities of existing LLMs via some specially designed approaches
- The authors provide codes in the main text, which enhances the reliability of the results reported in this paper
- The experiments are carried out using both LLaMa models and Qwen models with different model sizes, which shows the generality of the proposed method to some extent

### Weaknesses
- (major) I do not see the necessity of Definitions 2.1 and 2.2. If the authors would like to claim that the optimal trajectory can be rarely covered, and hence raises two critical issues in the RLVR training, there is no need to state Definition 2.1 and 2.2 since the readers can generally get that. The organization of this paper can be improved
- (major) No ablation study and parameter study can be found either in the main text or the appendix. This makes it hard to figure out which component (mixed-policy rollout and mixed-policy GRPO) contributes most to the performance of MENTOR. It is also hard to check how sensitive MENTOR is to the introduced hyperparameters, e.g., the initial value of $\alpha$ in Eq. 11, how different decay schedules of $\alpha$ affect the performance, how to set $p$ in $\gamma_p$, etc.
- (major) Despite the fact that the authors propose a method to accelerate mixed-policy rollout, the entire process of MENTOR is still both time-consuming and memory-consuming compared to other methods like QuestA. The authors ought to show the training efficiency of MENTOR against baseline methods
- (major) It is somewhat strange to me that the authors use different expert models for different base LLMs, e.g., OpenR1-Qwen-7B model for Qwen models and finetuned LLaMA3.1-8B-Instruct model for LLaMa models. Is there any reason that you do not use a fixed expert model? Why do you use the Instruct model for LLaMa here? It is unclear whether the proposed method would fail if one adopts other LLMs as the base model, e.g., deepseek-R1 as the expert model. 
- (major) It is also unclear whether the proposed method still works if the expert model is only marginally stronger than the base model, or if the expert model is too strong (e.g., Qwen3-max), and whether the proposed method can still work. Note that QuestA conducts experiments using some strong base LLMs. The authors commented that the advantages of QuestA rely heavily on the vanilla performance of the base models, while it raises questions on whether MENTOR can work if it is adopted upon some strong base models (e.g.,  Nemotron-1.5B)
- (minor) The related work part can be significantly improved. LLM reasoning and LLM reasoning under guidance are fast-growing research areas, and there are numerous papers that do similar things. The authors should include more discussion in the manuscript
- (minor) No full pseudo-code for MENTOR can be found. Algorithm 1 just states part of the MENTOR algorithm
- (minor) There are numerous minor issues in the manuscript. I just list some of them below, please double-check your manuscript,
  - Line 019, *we argue that the expert only needs to provide guidance only at critical*, two *only* in this sentence
  - Line 322, two QuestA

### Questions
- Why do you still adopt trajectories from the current model in Eq. 9? Any insights here?
- In Eq. 11, you write $A_i = \dfrac{[R\_i - mean(R\_j)]\_+}{R\_{\rm range}}$ rather than $\dfrac{[R\_i - mean(R\_j)]\_+}{{\rm std}(R\_j)}$, why? Why only normalize the advantages from the mixed policy into [0, 1] but use Eq. 10 directly from GRPO?
- Can the proposed MENTOR still work when the base LLM is strong?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
MENTOR addresses the challenge of balancing effectiveness and diversity in Reinforcement Learning with Verifiable Rewards (RLVR). The method mixes the policy and expert distributions at the token level using entropy-based weights, and introduces an “accelerated mixed-policy rollout” to improve sampling efficiency. 
Experiments on Qwen2.5 and LLaMA3.1 show consistent gains across in-domain and out-of-domain reasoning benchmarks.

### Strengths
1.1 The motivation is clear and well-grounded. The paper directly tackles the exploration–effectiveness trade-off in RLVR.

1.2 The proposed PMix mechanism (Eq.6) is conceptually interesting, using entropy as a proxy for critical decision points.

1.3 Results are consistently strong across multiple model scales and benchmarks, showing stable gains.

### Weaknesses
2.1 The “accelerating mixed-policy rollout” (Eq.7) appears inconsistent. Even with a small wt (e.g., 0.01), the expert π* must still be evaluated to compute the mixed distribution, so the method does not actually avoid the cost of expert inference. A gating mechanism for small weights would be necessary for genuine speed-up.

2.2 The claimed parallelization (L216–220) is unclear. Eq.7 explicitly depends on π*, so it cannot be fully parallelized in the autoregressive decoding process. The efficiency argument based on speculative sampling seems overstated.

2.3 Missing ablations for critical parameters (γp, α) and no exploration of alternative PMix strategies.

2.4 In Appendix A.2, the authors set the KL regularization coefficient to 0 during MENTOR fine-tuning. 
This is concerning, as the KL term is typically used to constrain the policy from drifting too far from the base model. Disabling it may weaken stability and allow distributional drift, which the paper does not quantify or justify empirically.

2.5 Results appear to be based on single runs, with no reported variance or standard deviations. This limits statistical reliability and makes it difficult to assess the robustness or significance to the provided results.

### Questions
3.1 Can the authors clarify how the “accelerated rollout” avoids computing π* at every step when wt is small? 

3.2 Did you test other mixture strategies (e.g., thresholded expert calls or adaptive gating) to validate that entropy weighting is optimal?

3.3 How sensitive are results to γp and α scheduling?

3.4 Can you provide more seeds for the results?

3.5 Could you report results with a non-zero KL coefficient to show how it affects stability and performance?  Also, please clarify the reasoning behind setting the KL term to 0 during MENTOR fine-tuning.

I am willing to increase my score if the authors adress my concerns/questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MENTOR (Mixed-policy Expert Navigation for Token-level Optimization of Reasoning), a GRPO+ training method for improving the reasoning capabilities. By providing guidance on critical tokens during inference, MENTOR achieves effective improvement of reasoning while preserving the diversity of group-based sampling. Experiments on math and science benchmarks show that MENTOR consistently outperforms over baselines like GRPO, LUFFY, and QuestA.

### Strengths
1. The motivation of the paper is reasonable, while full expert guidance could lead to the reduction of sampling diversity, MENTOR implements a relaxed guidance that guides the sampling only on critical tokens, leading to the balance of training efficacy and sampling diversity.

2. The MENTOR also involves an accelerated components that address the efficiency problem of the expert-guidance method, which is also a unique contribution.

3. This paper does comprehensive experiments on both in-domain and out-of-domain benchmarks, showing the efficacy and generalizability of MENTOR. In addition, the analysis results of entropy show that MENTOR can preserve the generation diversity during training.

### Weaknesses
1. One of the main weaknesses is that the mechanism of "MENTOR only guides the decoding on critical tokens". This paper shows MENTOR's decoding strategy in Sec. 3.1, but does not compare this strategy with prior methods of fully expert-guided decoding. A more detailed introduction and explanation of the decoding strategy is required.

2. The motivation for using an expert-guided method is not entirely clear. If a powerful expert model is already available, it seems straightforward to use it directly for reasoning tasks. Why is it necessary to further train and distill its knowledge into a smaller model? I would appreciate a more detailed discussion on the advantages and rationale behind training a smaller model with expert guidance.

3. The expert-guided method costs more training resources compared with vanilla GRPO, since it requires loading a bigger expert model for decoding, which will occupy a large amount of GPU resources. Even though this paper proposes an accelerated strategy, the extra cost of GPU computation and memory can not be overlooked.

### Questions
1. Could the authors provide a more detailed comparison between MENTOR and DAPO? Although DAPO may require additional time for resampling, it does not incur the extra GPU memory overhead associated with loading a separate expert model. In this respect, DAPO could be a more resource-efficient alternative to MENTOR.

2. Could the authors provide concrete case studies to better illustrate MENTOR's impact on sampling diversity? While Figure 3 presents keyword statistics, a more direct qualitative analysis or examples would offer clearer insight into the diversity achieved through MENTOR.

3. Could the authors provide an explanation about changing the selected expert model rather than OpenR1-Qwen-7B? How does performance change if the expert is weaker or from a different architecture/family?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MENTOR, a reinforcement learning framework that aims to enhance both the effectiveness and diversity of exploration in RLVR for LLMs. Inject expert guidance only at critical decision points (high-entropy tokens) rather than imitating entire expert trajectories (such as LUFFY). In-Domain and OOD performance evaluation (LLaMa3.1-8B-Base, Qwen2.5-3B-Base and Qwen2.5-7B-base) demonstrate MENTOR's effectiveness.

### Strengths
- Clear problem framing: Rollout's effectiveness and diversity are important for RL. Paper's section details description this aspect.
- Critical point guidance: By leveraging token entropy, providing expert guidance only at critical decision points, steers reasoning trajectories while preserving the policy’s own exploration, rather than entirely imitating expert.
- Empirical Results: Based on three backbone models, proposed method MENTOR compared to Vanilla GRPO, LUFFY and QuestA, demonstrate the method's effectiveness.

### Weaknesses
- Experiment: The Table 1's experiment appear insufficient, given that the paper’s pipeline involves multiple combinations of hyperparameters and strategies. 
- Critical point selection: The selection strategy for critical decision points could be elaborated in more detail — for example, why is it based on token-level entropy rather than other metrics (e.g., higher-level semantic)?
- Typos: Line 322 QuestA appear twice.

### Questions
What would happen if the token points for guidance were selected randomly or uniformly? 

How robust is the proposed method to the expert’s capability — for instance, if the expert is weaker, can the model filter out misleading guidance? 

In the baselines, LUFFY and QuestA belong to expert imitation. Are there any more dynamic or flexible expert guidance baselines for comparison?

If the authors can provide more detailed verification of the method’s effectiveness and robustness, I will increase my score.

### Soundness
2

### Presentation
3

### Contribution
3
