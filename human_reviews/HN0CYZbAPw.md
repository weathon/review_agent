# Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
The modern paradigm in machine learning involves pre-training on diverse data, followed by task-specific fine-tuning. In reinforcement learning (RL), this translates to learning via offline RL on a diverse historical dataset, followed by rapid online RL fine-tuning using interaction data. Most RL fine-tuning methods require continued training on offline data for stability and performance. However, this is undesirable because training on diverse offline data is slow and expensive for large datasets, and should, in principle, also limit the performance improvement possible because of constraints or pessimism on offline data. In this paper, we show that retaining offline data is unnecessary as long as we use a properly-designed online RL approach for fine-tuning offline RL initializations. To build this approach, we start by analyzing the role of retaining offline data in online fine-tuning. We find that continued training on offline data is mostly useful for preventing a sudden divergence in the value function at the onset of fine-tuning, caused by a distribution mismatch between the offline data and online rollouts. This divergence typically results in unlearning and forgetting the benefits of offline pre-training. Our approach, Warm-start RL (WSRL), mitigates the catastrophic forgetting of pre-trained initializations using a very simple idea. WSRL employs a warmup phase that seeds the online RL run with a very small number of rollouts from the pre-trained policy to do fast online RL. The data collected during warmup bridges the distribution mismatch, and helps ``recalibrate'' the offline Q-function to the online distribution, allowing us to completely discard offline data without destabilizing the online RL fine-tuning. We show that WSRL is able to fine-tune without retaining any offline data, and is able to learn faster and attains higher performance than existing algorithms irrespective of whether they do or do not retain offline data.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In this paper, the authors highlight the requirement for online fine-tuning algorithms to retain offline data in their replay buffers to remain effective. The authors thus aim to remove this requirement, by instead using a small amount of online data collected from rollouts of the frozen pre-trained policy. They show that the TD-error for the online policy under the offline data distribution diverges as less offline data is retained. The postulate that this is due to offline/online data distribution mismatch.

### Strengths
* There is a significant decrease in the efficacy of many finetuning algorithms when offline data is not retained, and this appears to be the first work to focus on this issue
* The experiments supporting the hypothesis of Q-value divergence over the offline data distribution are convincing
* The results are proven over many environments
* Clear layout and explanations

### Weaknesses
* No investigation of algorithm performance over varying levels of expertise of the offline pre-trained policy - it would have been interesting to see WSRL results when initialized from pre-trained policies of different quality levels or trained for varying numbers of steps.
* A comparison of RLPD initialised with prior policy, but without the additional online data should have been included as a baseline.

Minor
Line 84: an warmup

Line 207: on which we measure metrics on

Line 253: onto the previous action pair that the TD error (?)

Figure 4: Mixed, complete, partial (what does this refer to?)

### Questions
1. In Figure 7, JSRL is shown as not retaining offline data. However, JSRL is implemented over IQL in Uchendu et al., and the IQL implementation linked in the paper does retain offline data. Did the authors specify to you that they chose not to retain offline data?
2. It is hypothesised in section 4.1 that backups of small, over-pessimistic Q-values from pessimistic offline algorithms cause the Q-values divergence. However, in Figure 9 you mention the warm-up helps avoid over-pessimistic values for WSRL. I had understood that WSRL used an online, non-pessimistic algorithm (RLPD) as its base, so what would cause the underestimation in this case? Would RLPD work if initialised to the policy learned offline, without adding any data collected online?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the Offline to Online fine-tuning problem in RL. The author trying to remove the need of keeping the offline dataset when finetuning online, which is becoming impractical giving the growing size of pretraining dataset. Through a detailed analysis of Q-values during the fine-tuning phase, the authors reveal two critical insights:
1. Without the offline dataset, pretrained Q-values are quickly unlearned
2. Continuing conservative offline RL on online data leads to performance degradation

In response, the authors propose WSRL, a method that:
1. Implements a warmup phase to seed the replay buffer with the pretrained policy to mitigate the unlearning issue
2. Utilizes state-of-the-art online RL algorithms for fine-tuning

The approach is demonstrated through promising results on D4RL tasks.

### Strengths
- This paper studies an important and timely topic in RL.
- This paper provides some analysis for transition phase to motivate the problems.
- The proposed method is simple to implement.

### Weaknesses
A lot of the results are confusing and not convincing, as referring in the questions below.

### Questions
- Could you clarify Figure 3? The caption says "Fine-tuning starts at step 0.", but the plots are start from around 250k.
- Some of the plots are truncated, i.e. not starting from 0. Could you please clarify on that? Based on the analysis, the warm-up phase should mostly solve the problem in the beginning of the finetuning phase. I would even suggest the authors to have a zoom-in version of that parts of the plot, but the truncated version seems to be missing exactly the transition part.
- Since warm-up phase is to just seed the replay buffer with some data collected from the pretrained policy, will it also be fine to subsample a small portion of the offline dataset to initialize the replay buffer?
- From the results in Figure 7, seem like the warmup phase does not really solve the unlearning problem, i.e. the initial value for WSRL is much lower than the CalQL initialization. Is this a desired behaviour?
- I am not sure how the warm-up phase solves the miscalibration problem present in CalQL. For example when you initialize the policy and Q with the CQL, although the warmup data will help to combat the distribution mismatch, the change of the RL algorithm will still make the Q function miscalibrated which in turn make the policy unlearn. How do you think of this issue?
- Won't it be beneficial to just run a bunch of gradient steps only on the Q function after you collect the warmup data until it recalibrates?

### Soundness
3

### Presentation
2

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
The paper proposes a new method, Warm Start Reinforcement Learning (WSRL), for efficient online reinforcement learning (RL) fine-tuning without the need to retain offline data. Traditional RL fine-tuning often relies on retaining offline data to prevent performance degradation, but this approach is inefficient and can slow down learning. The authors argue that, by using a warmup phase that gathers initial online rollouts with a pre-trained policy, WSRL can "recalibrate" the offline Q-function to online data, enabling the algorithm to discard offline data and still achieve stable and effective learning. WSRL, which emphasizes a high updates-to-data (UTD) regime, proves faster and more effective across diverse tasks compared to traditional algorithms, demonstrating higher performance without offline data retention.

### Strengths
- WSRL removes the dependency on large offline datasets for online fine-tuning, based on various experimental takeaways.
- WSRL shows improved learning speed and higher asymptotic performance across various benchmark tasks, outperforming state-of-the-art fine-tuning methods.
- Introducing a warmup phase and applying a high UTD regime for online RL is an efficient approach to address the problem of Q-function divergence without retaining offline data.
- The method is adaptable across different RL environments, showing versatility and potential applicability to a broad range of tasks.

### Weaknesses
- The offline performance in the experimental setup is poor from the start (Section 6, Figures 6 and 7). The performance graph shows that, except for pen-binary, the offline score ranges from 0 to 20 points, which is quite low.  This weak offline performance may suggest that the offline data did not fit as effectively as it does with other algorithms or that, in this particular setup, the offline data may not have provided much support for online fine-tuning. The claim that the proposed method prevents unlearning is also debatable, as there was minimal offline performance to “unlearn” initially. Analyzing cases where the offline score reaches at least 60, ideally between 80 and 100, would be more informative. Conducting further experiments on Mujoco Hopper/Walker2D with medium, medium-replay, expert, or Adroit-dense expert settings—where offline performance tends to be stronger—would be beneficial.
  - Additionally, this paper mentions using SAC with a 10-ensemble for training, but it would also be necessary to examine how other offline algorithms like IQL and CQL perform when applying the proposed methods—such as excluding offline data, using a warmup phase, and employing a high UTD ratio.

- Unlearning likely occurs mostly at the very early stages of training, but the plots are spaced at intervals of about 50,000 steps. At this interval, even if unlearning did occur, it may have already been resolved by the next plotted point. To demonstrate the absence of unlearning, it would help to show early performance at smaller timestep intervals.

- The paper employs strong language, such as “Should Not” and “Completely unnecessary,” though the evidence provided may not be fully convincing. Without theoretical support, it needs stronger empirical backing; however, the experiments may not comprehensively represent all RL tasks. Removing such strong expressions would not alter the flow of the paper and might actually improve its persuasiveness.

- This paper conducted experiments on eight datasets across three domains (Antmaze ultra-diverse/large-diverse/large-play, Kitchen partial/mixed/complete, and Adroit door/pen), which appears insufficient compared to other conference papers. While experiments were conducted on Antmaze ultra-diverse, ultra-play was not included, which may suggest some inconsistency in selection. Given that this paper is experiment-driven and lacks a strong theoretical foundation, it would benefit from more extensive experiments, such as including Antmaze ultra/large/medium/umaze play/diverse datasets, to provide a more comprehensive set of results.

- In Figure 4 of Section 4.1 and Figure 9 of Section 6.5, it’s unclear what the Q-value magnitude itself indicates. Without a comparison to the true Q-value, it could be that one is underestimated or, conversely, that the other is overestimated. Additionally, even if the overall scale is low or high, as long as the general trend follows the true Q, there is no issue (the relative comparison of Q values is more important than their absolute magnitude). For a meaningful comparison, it might be better to separate in-distribution Q from out-of-distribution Q. t. Additionally, considering the focus on Q-value analysis and the approach from a Q-value perspective, I am curious to see a comparison with the paper “A Perspective of Q-Value Estimation on Offline-to-Online Reinforcement Learning” [1].

Overall, while achieving better performance than the baselines in the experimental settings (especially for Antmaze Large and Ultra) is highly positive, the points discussed above prevent the claim that "Efficient Online Reinforcement Learning Fine-Tuning Should Not Retain Offline Data" from being fully convincing.

[1] Zhang, Yinmin, et al. "A Perspective of Q-value Estimation on Offline-to-Online Reinforcement Learning." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.

### Questions
- Could you provide experimental results from environments like the MuJoCo dataset, where offline performance is typically strong?
- Could you provide experimental results using other offline algorithms (e.g., IQL, CQL) with the proposed methods applied?
- Could you provide experimental results for all 8 Antmaze datasets?
- Could you provide a comparison with “A Perspective of Q-Value Estimation on Offline-to-Online Reinforcement Learning”?
- This paper excludes offline data during online fine-tuning, implements a warmup phase, and increases the UTD ratio. What would happen if we retained the offline data while implementing the warmup phase and increasing the UTD ratio?

### Soundness
2

### Presentation
2

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
This paper explores the potential for efficient offline-to-online RL without retaining offline data. The authors investigate how offline data contributes to online fine-tuning in existing methods and propose WSRL, which leverages the offline policy and Q-function as initialization. Extensive experiments highlight the effectiveness of WSRL and its components.

### Strengths
This paper is well-written and easy to follow, tackling the interesting topic of offline-to-online RL. The study of related works is thorough and comprehensive. Building on this foundation, the paper illustrates how retaining offline data contributes to previous approaches and provides three key takeaway messages. The authors then propose WSRL, a simple and intuitive method designed to enhance the stability and efficiency of online fine-tuning.

The experimental results show that WSRL outperforms or is at least competitive with previous methods. The ablation studies are detailed and convincingly demonstrate the necessity of all components.

### Weaknesses
I did not identify any major flaws in this work. One potential limitation, however, is that WSRL relies on off-policy online algorithms, as it requires interactions from the warmup phase to stabilize the fine-tuning process.



Typos:

Ofline --> Offline (In the box of Takeway 3, Line 3)

### Questions
While off-policy algorithms are naturally suited for the online phase, is it possible to apply online policy methods with monotonic improvement properties (e.g., PPO, TRPO) and still achieve comparable performance during online training?

### Soundness
3

### Presentation
4

### Contribution
3
