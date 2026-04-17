# Mitigating Forgetting Between Supervised and Reinforcement Learning Yields Stronger Reasoners

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Large Language Models (LLMs) show strong reasoning abilities, often amplified by Chain-of-Thought (CoT) prompting and reinforcement learning (RL). Although RL algorithms can substantially improve reasoning, they struggle to expand reasoning boundaries because they learn from their own reasoning trajectories rather than acquiring external knowledge. Supervised fine-tuning (SFT) offers complementary benefits but typically requires large-scale data and risks overfitting. Recent attempts to combine SFT and RL face three main challenges: data inefficiency, algorithm-specific designs, and catastrophic forgetting.
We propose a plug-and-play framework that dynamically integrates SFT into RL by selecting challenging examples for SFT. This approach reduces SFT data requirements and remains agnostic to the choice of RL or SFT algorithm. To mitigate catastrophic forgetting of RL-acquired skills during SFT, we select high-entropy tokens for loss calculation and freeze parameters identified as critical for RL. Our method achieves state-of-the-art (SoTA) reasoning performance using only 1.5\% of the SFT data and 20.4\% of the RL data used by prior SoTA, providing an efficient and plug-and-play solution for combining SFT and RL in reasoning post-training.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces MIFO (Mitigating Forgetting between SFT and RL), a framework that dynamically integrates supervised fine tuning (SFT) with reinforcement learning (RL) to mitigate high data requirements and overfitting inherent to SFT while expanding reasoning frontiers through judicious use of out of distribution data. A highlight of MIFO is that it overcomes the phenomenon of catastrophic forgetting by-design. Experimental evaluations demonstrate that MIFO, and a variant MIFO$^+$ result in significantly improved data usage compared to state of the art. 

**Caveat**: I am not fully familiar with the scope of mathematical reasoning benchmarks and datasets that are used in the evaluation (as mentioned by the authors in Sec. 5.1. My review is based on a presumption that these are adequate; I will defer to the authors/ other reviewers on whether any other benchmark/ dataset can potentially serve as additional datapoints for evaluation of MIFO.

### Strengths
(+) Interleaving of SFT with RL is a methodology that is unique relative to prior work. The merits of such an approach are revealed in the fact that it obviates a need for large amounts of data that is typically required for SFT. 

(+) MIFO aims to mitigate catastrophic forgetting by design, which is significantly different from approaches in current art. 

(+) The results demonstrating the redundancy in SFT relative to RL is particularly insightful, and serves as a strong basis for the working of MIFO in terms of mitigating catastrophic forgetting typical of SFT. 

(+) The paper is generally well-written and the logical flow is sound. At the same times, I have some questions about the technical aspects and experiments (please see Weaknesses, below).

### Weaknesses
(-) The central claim by the authors is that MIFO is agnostic to the specific RL or SFT algorithm used (e.g., the claim that an RL algorithm different than GRPO can be used for RL training at the start of Sec. 4.1). However, the experimental evaluations do not seem to suggest that this claim has indeed been tested on using MIFO with multiple RL/ SFT algorithms. 

(-) In Fig. 1 right, while the gap does begin to close at around the 110th step as the authors write, it subsequently begins to diverge. The text of the paper does not appear to provide an explanation for this phenomenon. 

(-) In Fig. 1, while the gap between the SFT curves closes at the 40th step, and remains close subsequently and the gap between RL curves closes at the 110th step, it is not clear what other factors determine convergence. Also, from the right side of Fig. 1, it is not clear that RL converges even at 350 gradient steps. 

(-) The labels on the graph of Fig. 2 do not seem to match with the caption of the figure or the text in Lines 159-161. Perhaps the green curve corresponds to SFT while the blue curve corresponds to RL? 

(-) In Tables 1 and 2, it is not clear why MIFO produces shorter length outputs than MIFO$^+$ for the 7B model, while MIFO$^+$ yields significantly shorter length outputs than MIFO for the 1.5B model. Some insight into this result, and intuition about the role of the history parameter $\alpha$ will help make the interpretation of the results more clear. 

(-) Some aspects of the presentation can be improved. For example, in Lines 269-270, the authors write `Entropy describes the uncertainty…’ - the writing will benefit from having a more formal definition of entropy over here. 

(-) Minor comment: typo - in Line 194, there is an additional space between ( and Section.

### Questions
Please see Weaknesses above.

### Soundness
2

### Presentation
3

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
This paper proposes MIFO, a plug-and-play framework to jointly optimize SFT and RL for reasoning post-training of LLMs. The key claim is that SFT introduces redundant and high-magnitude parameter updates that overwrite the more updates of RL, leading to catastrophic forgetting. To address this, MIFO Interleaves SFT into RL, selecting only challenging rollouts and applying loss only on high-entropy tokens. MIFO achieves perfect results on AIME-24/25, AMC, MATH-500, OlympiadBench, and MMLU-Pro, while using only 1.5% of the SFT data and 20.4% of the RL data.

### Strengths
1. Identifies and visualizes the gradient update magnitude between SFT and RL.

2. Consistently gains across different reasoning benchmarks.

3. Solid ablations of complementary effects of entropy-based token selection and parameter freezing.

### Weaknesses
1. The freezing and entropy ideas, while effective, are incremental extensions of existing interleaved SFT+RL frameworks (e.g., ReLIFT).

2. All experiments use Qwen-Math models on mathematical reasoning; no evidence of generalization to other domains or other model settings.

3. Limited discussion on compute or runtime overheads.

### Questions
1. How sensitive is MIFO to the hyperparameters? 

2. Have you tested MIFO with other domains or other non-math models?

3. What is the computational overhead (e.g., GPU hours) of MIFO compared to baselines?

4. Theoretical analysis (Appendix C) is disconnected from practice. The introduced Decision–Redundancy Ratio (DR) is not computed empirically nor related to the actual experiments in main sections. It’s unclear what the analysis truly verifies.

5. Forgetting not quantitatively measured. Figures only visualize parameter update magnitude, not actual forgetting metrics. The claim that MIFO “mitigates forgetting” is weakly supported.

6. The paper lacks any figure showing model performance over training steps (e.g., test dataset performances across training steps). Such curves would clarify whether MIFO actually stabilizes learning rather than just improving final accuracy.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes MIFO, an interleaved SFT RL post training framework to mitigate forgetting. MIFO mainly consists of two components: data processing to strengthen low accuracy examples for SFT, and parameter freezing to prevent overwriting key parameters.

### Strengths
- MIFO outperforms multiple baselines on math tasks.
- MIFO improves data efficiency than baselines.

### Weaknesses
Weaknesses:
- The experimental validations mainly focus on math tasks, while other reasoning tasks beyond math are overlooked.
- Experiments focus on Qwen family, making applicability of MIFO to other model families unclear especially given the observed performance drop under different templates.
- MIFO relies on experts or a stronger teacher model. The cost associated with it is ignored in validations. 
- Linearized approximation in theoretical analysis in Appendix C needs to be justified.
- Results in Figure 2 seem contradicting with description in lines 160-161. Please clarify.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The work proposes MIFO, Mitigating Forgetting Between SFT and RL, a new pipeline to bridge the SFT and RL in post-training of LLM reasoning. The pipeline starts from RL and constructs an SFT data buffer. Then, it uses entropy-based token selection and RL parameter update-based freezing for SFT. The comprehensive experiments on small-scale LLMs show that it can efficiently boost the model's reasoning performance.

### Strengths
- It proposed an interesting view of SFT and RL in post-training of LLM reasoning. The analysis in section 3 provided good motivation for the design of MIFO. And the components of MIFO provide a promising advantage to improve the training of SFT+RL.
- The experiment as well as ablation study, indicates MIFO is well effective compared to baseline, and provides good data efficiency and token efficiency.

### Weaknesses
- The experiment is purely based on qwen 2.5 models and the math domain training dataset. The generalizability of this approach to other domain is tricky. And qwen 2.5 models (even it is the base model) include heavy mid-training data, experiment on these models are more like containing an implicit SFT, which is different from the claimed RL-first-then-SFT paradiam.   
- Risk of catastrophic forgetting is mentioned as motivation, but not studied/showed how MIFO addressed this.
- Writing issue: e.g., L323 NuminaMath Li et al. (2024) > NuminaMath (Li et al., 2024)

### Questions
- Though I did not see the code, I think the proposed method will introduce computation overhead due to the (frequent) context switch between SFT and RL. As my question below, the interval matters in this design to balance training performance and training efficiency.  
- I did not find out MIFO iteration number/interval used for the experiment. And what is the effect of these factor? For example, it can be high high-frequency interval (e.g., every batch), or a low-frequency interval (e.g, every epoch).
- I am interested in the data buffer dynamics of training. In other words, does the effective buffer get smaller during training, and does the questions get back and forth in the buffer? This information helps the understanding of whether the model learns something new to improve the performance, otherwise it may be more like randomness.  Also, I wonder how often the frozen RL updated parameters overlap with those from high-entropy sft tokens. My intuition is that these are pretty much overlapped, so I did not understand what is updated.

### Soundness
3

### Presentation
3

### Contribution
3
