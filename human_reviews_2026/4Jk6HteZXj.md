# Beyond Reasoning Gains: Mitigating General Capabilities Forgetting in Large Reasoning Model

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 0, 2, 6, 4

## Abstract
Reinforcement learning with verifiable rewards (RLVR) has delivered impressive gains in mathematical and multimodal reasoning and has become a standard post-training paradigm for contemporary language and vision-language models. However, the RLVR recipe introduces a significant risk of capability regression, where models forget foundational skills after prolonged training without employing regularization strategies. We empirically confirm this concern, observing that open-source reasoning models suffer performance degradation on core capabilities such as perception and faithfulness. While imposing regularization terms like KL divergence can help prevent deviation from the base model, these terms are calculated on the current task, thus they do not guarantee broader knowledge. Meanwhile, commonly used experience replay across heterogeneous domains makes it nontrivial to decide how much training focus each objective should receive. To address this, we propose a replay strategy with dynamic objective reweighting for general knowledge preservation. Our reweighting mechanism adapts in an online manner using short-horizon signals of convergence and instability, shifting the post-training focus away from saturated objectives and toward underperforming or volatile ones. Our method is end-to-end and readily applicable to existing RLVR pipelines without training additional models or heavy tuning. Extensive experiments on benchmarks based on Qwen2.5-VL-3B and Qwen2.5-VL-7B demonstrate the effectiveness of our method,  which not only preserves general capabilities but also improves reasoning by enabling more flexible trade-offs among in-task rewards.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
I wasn’t able to clearly understand the main concept of the paper (see page 4). Please refer to the weaknesses mentioned below.

### Strengths
I cannot mention any strengths at the moment.

### Weaknesses
This paper does not yet appear ready for formal review. 

The main idea is not clearly articulated, The radar plot (Fig. 1) fails to effectively convey the authors’ intended message—alternative visualization methods could be more appropriate. The overview plot is also difficult to interpret. In addition, there are several stylistic and formatting issues: sentences often lack proper punctuation (e.g., missing periods), and citation formatting should follow standard conventions (e.g., using \citep and \citet appropriately).
Figure 7 in the appendix appears poorly constructed and lacks coherence. The question, prompt, and reasoning process are inconsistent, which raises concerns about the validity of the example.

### Questions
How are the convergence rate and instability incorporated into your model formulation, and how do your overall training objectives differ from those in previous RLVR work?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper focuses on the point that RLVR enhances reasoning but often leads to degradation in general capabilities. To address this, authors propose integrating general capability data into RLVR pipeline and dynamically reweighting objective based on window-based estimation of convergence rate and SNR. Authors argue that this allows shifting focus to underperforming objectives. The experimental results show that this strategy leads to an improvement.

### Strengths
- The paper probed to empirically show that RLVR actually often degrades general capabilities, which is an interesting point.
- Shows improved performance over a number of benchmarks.

### Weaknesses
- The method design feels somewhat rudimentary; for example,
  - Why use window based method instead of exponential moving statistics? Window based method may suffer from abrupt changes when some important statistics leave the window.
  - What motivates prioritizing the objectives that converges fast? If we prioritize the objectives that converges fast, it would learn one objective at a time, as illustrated in the paper; in Figure 3, it is said that the format is optimized first, and the other objectives are learned later. While the paper mentions catastrophic forgetting and continual learning, I believe slowing down the objectives that converges fast so that every objectives have similar rate of convergence is the best way to reduce catastrophic forgetting.
  - While it worked to simply add $c$ and $i$ for the proposed dataset/model/hyperparameters, it is probable that there will be a setting where simple addition would not work. For example, $c$ is correlated with window size but $i$ is not (increasing $W$ will result in decreased $c$). batch size, learning rate, and all these optimization hyperparameters depend on one of two, so if we tweak the hyperparameters, then it is likely the scale of two will be greatly varied.

- There is not enough validation on the proposed reweighting scheme. Over the number of baselines in experiments, the baseline that can explain the impact of proposed reweighting objective is only (default vs ours). While this shows significant performance difference, the design choices are not fully empirically validated; for example, what is the more important factor, SNR or convergence rate? how does varying window size affect the performance? how does temperature affect the performance?

- Other than the proposed rudimentary reweighting scheme that is not sufficiently validated empirically, the contribution of the paper is about including and incorporating general capability dataset and losses. However, I believe that this is not very novel, as such multi-task joint training has been often done in many fields of machine learning, going back to early 2000s.

- some minor comments:
  - estimated old loss averages from $t-W+1$ to $t$ and estimated current loss averages from $t-2W+1$ to $t-W$? is this flipped?
  - How do we sum from $t-W+1$ to $t+W$, where the current timestep is $t$? (when computing $\sigma_k^{(t)}$)

### Questions
- Compared to "Default", each objective in "ours" can only have at most $K$ times larger portions in the overall losses. Does the proposed reweighting objective still outperforms the non-reweighted version if we train non-reweighted version $K$ time longer?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates catastrophic forgetting of general capabilities in vision-language models (VLMs) when fine-tuned using Reinforcement Learning with Verifiable Rewards (RLVR) for reasoning tasks. 

The authors demonstrate that RLVR improves reasoning but degrades perception and robustness. They propose a replay-based strategy with dynamic objective reweighting that monitors convergence rates and instability of individual objectives, automatically down-weighting saturated objectives (e.g., format compliance) while focusing on harder ones (e.g., reasoning accuracy). 

Experiments on Qwen2.5-VL-3B/7B models show the method preserves general capabilities while improving reasoning performance.

### Strengths
Originality: The dynamic reweighting based on convergence rate and instability signals is novel. The observation that different rewards (format, accuracy, IoU) converge at different rates with varying stability is insightful and well-motivated by Figure 3.

Quality: Solid experimental validation across two model scales with comprehensive benchmarking on reasoning (MMMU-PRO, MathVista), perception (LISA, OCRBench), and robustness tasks. Good ablation studies isolating the reweighting contribution, and practical comparisons showing 60% reduction in thinking length without sacrificing accuracy.

Clarity: Well-written with clear motivation. Figures 1-3 effectively illustrate the problem and solution. The mathematical formulation is straightforward.

Significance: Addresses a critical practical problem for deploying reasoning models. The solution requires no additional model training (unlike MoDoMoDo) and integrates easily into existing RLVR pipelines.

### Weaknesses
1. Weak theoretical justification: The choice s_k = c_k + i_k appears ad-hoc without justification. Why not weighted combinations, gradient magnitudes, or loss curvature? The unweighted sum needs empirical or theoretical support.

2. Incomplete experimental details:
* Window size W not specified
* Exact composition and proportion of "general data" unclear
* No discussion of reward normalization across different scales
* No error bars or multiple runs despite RL stochasticity

3. Limited failure analysis: On VizWiz, the method (61.97) underperforms Coreset (63.76) and Reasoning-only (62.45). No analysis of when/why dynamic reweighting fails. Understanding these cases would strengthen the work.

4. Computational overhead not reported: No wall-clock time or memory overhead analysis. Computing running statistics and reweighting adds complexity—how significant is this in practice?


### Minor Issues
* Figure 1 radar plots could be higher quality
* Some notation inconsistencies (D for domains vs. datasets)
* Related work could better connect to multi-objective RL and multi-task learning literature

### Questions
1. Can you provide sensitivity analysis for window size W and temperature T?
2. How are rewards normalized across different scales (IoU vs. accuracy vs. format)?
3. Why does the method underperform on VizWiz? Can you characterize when dynamic reweighting helps vs. hurts?
4. Can you compare against well-tuned KL regularization more thoroughly? LwF achieves competitive results in Table 1.
5. Have you tried alternative formulations beyond s_k = c_k + i_k, like s_k = α·c_k + β·i_k? Can you provide justification for this specific choice?
6. Do the weights eventually converge to stable values, or keep fluctuating throughout training?

### Soundness
4

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
This paper mainly shows the empirical results for the performance degradation of LLM on general capabilities such as perception and robustness after post-training the model. The authors said those kind of phenomenon is related to the catastrophic forgetting, and the main objective is to prevent this problem. In the methods, to prevent the forgetting, replaying the data on retaining the general ability can solve this problem. However, due to the difference on convergence rate between replayed task and the main objective (e.g. reasoning), the training instability occurs. To tackle this problem, the authors suggest the re-weighting mechanism to make the training stable. In the experiment, the proposed method outperforms other baselines in various benchmark, and in the ablation study, the authors show the effectiveness of re-weighting scheme.

### Strengths
1. The problem statement and the solution is very simple. The authors pointed out well on the problems on the catastrophic forgetting after the post-training, and it is natural to use replay to elevate the performance on  general ability. Furthermore, the re-weighting techniques on securing the training stability is also effective.

### Weaknesses
1. The overall presentation is not well-organized. At the introduction section, the authors mainly pointed out the occurrence of catastrophic forgetting on post-training on the LLM. However, in the method section, they shifted the attention to the re-weighting scheme to tackle the training instability problem. I think this flow is not quite convincing and little bit confused. It would be better to specify more details on the motivation behind the re-weighting. Just showing the empirical results in Figure 3 (middle) may be not enough to support the claim.

2. The re-weighting scheme is quite heuristic. We cannot guarantee that this method can also solve the problem on much larger models such as 32B models. I think the proposed weighting scheme highly depends on the training behavior of 7B models. 

3. I wonder the re-weighting scheme really gives major contribution on the training stabilization. First, the performance difference between "Coreset" and the proposed method is marginal, and there are also some cases that "Coreset" outperforms the proposed method. Second, the authors do not show any empirical results on the training stabilization when we use the re-weighting scheme. The results on the excellence of the accuracy reward compared to the baseline and the thinking length do not give a persuasive point of view on the effectiveness of the re-weighting scheme.

### Questions
1. What kind of dataset you used for the replay? It would be better to specify the datasets.

### Soundness
2

### Presentation
1

### Contribution
2
