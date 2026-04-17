# How Far Can Unsupervised RLVR Scale LLM Training?

- Decision: Accept (Poster)
- Scores: 6, 2, 8, 6

## Abstract
Unsupervised reinforcement learning with verifiable rewards (URLVR) offers a pathway to scale LLM training beyond the supervision bottleneck by deriving rewards without ground truth labels.
Recent works leverage model intrinsic signals, showing promising early gains, yet their potential and limitations remain unclear.
In this work, we revisit URLVR and provide a comprehensive analysis spanning taxonomy, theory and extensive experiments.
We first classify URLVR methods into intrinsic versus external based on reward sources, then establish a unified theoretical framework revealing that all intrinsic methods converge toward sharpening the model's initial distribution
This sharpening mechanism succeeds when initial confidence aligns with correctness but fails catastrophically when misaligned.
Through systematic experiments, we show intrinsic rewards consistently follow a rise-then-fall pattern across methods, with collapse timing determined by model prior rather than engineering choices.
Despite these scaling limits, we find intrinsic rewards remain valuable in test-time training on small datasets, and propose Model Collapse Step to measure model prior, serving as a practical indicator for RL trainability.
Finally, we explore external reward methods that ground verification in computational asymmetries, showing preliminary evidence they may escape the confidence-correctness ceiling.
Our findings chart boundaries for intrinsic URLVR while motivating paths toward scalable alternatives.
Code is available at \url{https://github.com/PRIME-RL/TTRL}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper unifies several intrinsic reward modeling approach under the proposed unsupervised RLVR (URLVR) framework. Using majority voting as a prototypical approach, the paper observes that the trained model converges to a deterministic policy (Theorem 1). The paper hypothesizes that the performance of the learned policy depends on the confidence of the pre-RL model. URLVR approaches improve sampling efficiency but may not necessarily add any new capability. The paper tests this hypothesis via experiments on a a few language models.

### Strengths
- The paper unifies a variety of methods that use implicit reward modeling. This setup provides a framework for the community to understand the tradeoffs involved with these methods

- The analytical approach used in Theorem 1 clarifies the confidence vs performance trade-off. 

- Experimental validation and analysis is mostly sound and supports the observation made in Theorem 1 and claim related to model collapse. I do have a question on the setup that I will note under weakness/question

### Weaknesses
- The use of Qwen model in the analysis could use more justification. The central claim in the paper is that  the success (or lack thereof) of URLVR methods depend on the "base/pre-RL's" model's confidence-correctness alignment. The empirical analysis is conducted on Qwen3-family of models. No justification is provided on why these models were chosen. I would be curious to understand whether there is something the reader can learn about how this choice is better than others that may be accessible to them.

- The writing in the paper could be improved
  - The paper uses math symbols (for example Table 1) without defining all of the symbols in text. While the appendix may have the info the reader needs, it would be useful if this information is available to the reader in caption/nearby text
  - The paper covers related papers but uses short-form etc for their names. This practice can be confusing to readers that are not as familiar with the field (like this reviewer) as the authors

### Questions
- Would it be possible to share any insight on why Qwen3 models may have the confidence-correctness alignment shown in the paper? Is there a way to show some language models family are more aligned than others in the sense defined in the paper?  

- What is RLIF mentioned in the paper? There is another paper not cited in the draft that shows up when the reviewer did a literature search : https://arxiv.org/abs/2311.12996

- I did not see an LLM usage statement in the paper or appendix. Please include one as required by ICLR 2026 guidelines

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the mechanisms underlying Unsupervised Reinforcement Learning with Verifiable Rewards (URLVR), which finetunes a language model through RL over intrinsic rewards (i.e., rewards derived based on the model itself without any external supervision). It first introduces a theoretical perspective that unifies the goal of ensemble-based and certainty-based methods. Then, the theoretical insights are demonstrated empirically. In particular, the paper shows that existing intrinsic rewards trade uncertainty for performance by sharpening the language model distribution, yet this comes at the risk of distribution collapse and reward hacking. Furthermore, it characterizes how such collapse manifests in different methods and shows that in some specific regimes it can be avoided.

### Strengths
1. The topic is timely, given the increase in popularity of methods using RL with intrinsic rewards for improving the capabilities of language models.

2. The empirical analysis sheds light on the difference between intrinsic rewards and their potential failure patterns.

3. I find the suggestion of using intrinsic reward dynamics in the initial time steps as an indicator for the potential success of URLVR, as opposed to using measures such as pass@k that require access to a verifier or labels, to be interesting. However, it is worth noting that the evaluation of this method is somewhat lacking. There is no qualitative guidance as to how one should use the intrinsic reward dynamics and there is no comparison to pass@k in terms of predictiveness or computational efficiency.

### Weaknesses
1. The unified reward framework in Section 3.1 is currently not well-defined and its usefulness is not substantiated in the paper. Specifically, how does the right hand side of Equation (1) depend on $y$? Should the cross-entropy term $h$ be $-q^i (y | x) \ln \pi_\theta^i (y | x)$? Moreover, the significance of such a unified perspective greatly depends on whether it allows characterizing similarities and differences between different instances. However, the unified framework is not really used in the paper after its definition, which raises the question of why it is necessary or helpful.

2. The theoretical analysis contains potentially incorrect claims. 
    - The reward in majority voting depends on the current policy. It therefore changes during training. The closed form solution to the KL-regularized RL objective in Equation (3) holds for a fixed reward, and so it is not clear what it implies for a policy-dependent reward such as majority voting. In particular, without proof, there is no reason to believe that Equation (4) holds. Do the authors have a proof for this claim? Also, the majority reward is not formally defined: what does $maj (Y)$ stand for? How many rollouts are considered in this majority?
    - Theorem 1 is not rigorously stated. It is therefore difficult to evaluate what it means or whether it is sound. Specifically, what does "majority trajectories" refer to? What are "$k$ updates"? Does this refer to $k$ steps of policy gradient or rather $k$ iterations of solving perfectly the KL-regularized objective and each time updating the reference policy to be the current policy?

3. One of the main claims made in the paper is that the success of URLVR stems from sharpening the language model’s distribution on correct outputs, in case it already assigns such outputs a reasonable probability. However, as far as I am aware, this is already the existing conventional wisdom behind why methods such as majority voting as a reward can work (c.f. [1,2]).

4. Relation to prior work is often not adequately discussed. In particular, the results of Section 5.1 are extremely similar in nature to Section 4.1 of the TTRL paper [2]. What contribution does Section 5.1 of this paper provide beyond what was already reported by the TTRL paper? Furthermore, I believe it is worth mentioning the relation to [1], which theoretically analyzes the sharpening mechanism of URLVR.

[1] Zuo, Yuxin, et al. "Ttrl: Test-time reinforcement learning." arXiv preprint arXiv:2504.16084 (2025).

[2] Huang, Audrey, et al. "Self-improvement in language models: The sharpening mechanism." ICLR 2025.

Review Summary and Recommendation
---
Overall, I believe that the paper does not meet the requirements for publication at ICLR. The most significant limitation of the current manuscript is that it contains underspecified and potentially unsound theory. Beyond fixing this issue, it would greatly strengthen the contributions of the paper to elaborate on the unified perspective and its uses, clarify relation to prior work and conventional wisdom regarding how URLVR works, and provide quantitative evidence for the predictiveness of initial reward dynamics of the success of URLVR.



Additional (More Minor) Comments
---
1. The notation of Table 1 is defined only in the appendix, which makes it difficult to parse. I would recommend having the necessary notation in the main text (e.g., in the table caption).

2. There are some missing implementation details, which make it difficult to gauge the significance of some empirical contributions.
    - In Figure 4, how are the subsets chosen? Are these just chosen at random? Also, is the behavior consistent across random subsets? Intuitively, while some small subsets will not suffer from a collapse in ground truth reward, for others they should if the majority vote is incorrect in a non-negligible portion of the examples.
    - The setup in Section 4.3.2 is unclear. For example, how many samples are used?
    - The setup in Section 5.1 is unclear. For example, on which dataset are these experiments ran?

3. The definition of reward hacking mentioned in line 355 does not seem to accord with its conventional use. If the intrinsic reward is maximized and the ground truth reward also keeps increasing, then why should this be considered hacking?

### Questions
--

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates the potential and limitations of Unsupervised Reinforcement Learning with Verifiable Rewards for scaling Large Language Models , specifically focusing on intrinsic reward derived from the model's own internal signals addressing the "supervision bottleneck," where obtaining human-verified labels becomes expensive. The study presents a unified theoretical framework for intrinsic rewards stating that all such rewards share a core mechanism: they improve performance by trading uncertainty for performance by leveraging the models priprs knowledge and sharpen the model's output . The authors present empirical analysis  confirming this. The authors also show that model collapses is avoidable in small domain specific settings  such as test time training.

### Strengths
the paper presents a consolidated mathematical perspective that connect different intrinsic rewards methods under a single framework which is very nice. The authors also present a solid evaluation of intrinsic rewards - they analysed different failure modes for different methods. I think the paper is very insightful and written clearly and I think it's very interesting to investigate these novelity driven rewards within RL for LLM's.

### Weaknesses
this is not necessarily a weakness but the paper really focueses on the analysis of these different intrinsic rewards and does not so much contribute any new method ontop of this. While I personally really enjoyed reading this paper I am not 100% sure this is the right venue for this.

### Questions
how does this scale to long training on large datasets? is reward hacking inevitable? 
Do you see any issues with the limited domain? as the domain this has been tested on is quite limited to math reasoning tasks

### Soundness
4

### Presentation
3

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
This paper presents a broad investigation into recent RL methods for LLMs, that operate completely without external rewards and instead use the model's own internal signals. It introduces a framework, arguing that diverse intrinsic rewards (such as majority voting or self-certainty) share a common mechanism: they "trade uncertainty for performance" by "sharpening output distributions" around the model's initially confident solutions. This convergence, the paper shows, enables the model to amplify its existing knowledge, but also risks bias lock-in if the model's confidence is misaligned with correctness. Empirically, the paper confirms this trade-off, revealing that different intrinsic reward methods fail in distinct ways; for example, some collapse into overly brief responses, while others promote repetitive verbosity. The authors demonstrate that in small, domain-specific regimes, such as test-time training, intrinsic rewards can drive stable adaptation without collapse. Finally, the paper proposes a practical application for these findings: using the early training dynamics of intrinsic rewards as a fast, lightweight indicator to assess a model's suitability and "RL trainability" for a specific task, complementing traditional metrics like

### Strengths
Proposes to view the changes to the policy through the lens of trading uncertainty for confidence and provides a systematic, empirical analysis of the distinct failure modes of different reward types, such as length collapse for probability rewards and verbosity for entropy rewards. A practical strength of this paper is its investigation of scaling limits, which concludes that while large-scale training leads to collapse, these methods are stable and effective in small, domain-specific settings, identifying test-time training as an ideal application.

### Weaknesses
A closely related analysis was published recently (Jun 2025) by Y. Zhang et al (No Free Lunch: Rethinking Internal Feedback for LLM Reasoning). Arguably, the core insights of these two publications are very similar; with Y. Zhang et al provide a stronger grounding in theory and provide a more in depth analysis of learning dynamics for different base-models; while this paper provides more insight into the dynamics and failure-modes of the different reward signals. Overall, the authors cite this prior work but do not discuss and contextualize  the relevance, the overlap, and potential differences.

### Questions
Has the term URLVR been used before? My search came up empty and I find it confusing to have “Verified Reward” prominently in the name, but negate it by prefixing it with “unsupervised”. Isn't Reinforcement Learning from Internal Feedback (RLIF) the established terminology at this point?

Regarding the prior (almost concurrent) work by Y. Zhang et al (No Free Lunch: Rethinking Internal Feedback for LLM Reasoning): Do you see any discrepancies between your results and conclusions and theirs?

### Soundness
3

### Presentation
3

### Contribution
2
