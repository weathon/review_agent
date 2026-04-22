# Similarity as Reward Alignment: Robust and Versatile Preference-based Reinforcement Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Preference-based Reinforcement Learning (PbRL) entails a variety of approaches for aligning models with human intent to alleviate the burden of reward engineering. However, most previous PbRL work has not investigated the robustness to labeler errors, inevitable with labelers who are non-experts or operate under time constraints. We introduce Similarity as Reward Alignment (SARA), a simple contrastive framework that is both resilient to noisy labels and adaptable to diverse feedback formats. SARA learns a latent representation of preferred samples and computes rewards as similarities to the learned latent. On preference data with varying realistic noise rates, we demonstrate strong and consistent performance on continuous control offline RL benchmarks, while baselines often degrade severely with noise. We further demonstrate SARA's versatility in applications such as cross-task preference transfer and reward shaping in online learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a preference-based reinforcement learning method named SARA, which aims to construct latent representations of preferred trajectories via contrastive learning and compute rewards based on similarity to enhance robustness against noisy labels.

### Strengths
1) The paper validates the proposed framework through extensive experiments across multiple environments, demonstrating its effectiveness in improving performance.

2) Additional robustness experiments are conducted, verifying the algorithm's resistance to noise to some extent.

### Weaknesses
1) The underlying mechanism and motivation for using similarity as reward are unclear.

2) The innovation is limited, primarily involving integration and improvements of existing frameworks, with little theoretical analysis.

3) The writing is generally verbose, and the overall quality still falls short of ICLR standards.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the problem that most PbRL methods are fragile when human feedback has noise, which often happens with non-expert labelers. The authors propose SARA, which learns a latent space of preferred samples and uses similarity in this space as the reward instead of directly predicting a reward from noisy labels.

### Strengths
This paper proposes a simple method that makes PbRL more robust when preference labels contain noise. The approach is practical in real settings where human annotators often make mistakes.

### Weaknesses
The experimental evaluation is limited, as standard offline PbRL studies usually include a broader set of tasks. In addition, some more recent baselines are missing. These gaps in the experimental design raise concerns about the reliability and generality of the reported performance.

### Questions
1.The legend in Figure 2 is very cluttered, making it hard to understand the intended message from the plot.

2.Is the method robust when preferences come from multiple annotators? Different users may have different preference patterns, leading to conflicts beyond simple label flips. Can SARA handle such heterogeneous preference distributions?

3.Why is the proposed method more robust to label noise than existing approaches based on the BradleyTerry model? A clearer theoretical or empirical justification would be helpful.

4.The paper mentions assumptions behind SARA. Can the authors analyze in which scenarios SARA may fail, or discuss the limits of its applicability? Understanding the boundary conditions would strengthen the work.

5.Since SARA learns trajectory encoding in a latent space, what happens if preferred trajectories are multi-modal? Will the latent representation collapse or fail to capture diverse preference modes?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
SARA (Similarity as Reward Alignment) replaces Bradley-Terry reward modeling in preference-based RL with a contrastive learning approach. It yields robust rewards under noisy labels and supports cross-task preference transfer and reward shaping.

### Strengths
- Robustness: Maintains stable performance even with high label noise (10–40%).
- Simplicity: No Bradley–Terry modeling or pairwise loss; uses a single contrastive objective.
- Empirical Performance: Outperforms or matches strong PbRL baselines across D4RL benchmarks.

### Weaknesses
- The fixed prototype $z_p^*$ might drift if new or biased preference data are added; continual adaptation requires full retraining.
- There is no analysis of how performance scales with the number of preference pairs or trajectories. We don’t know whether SARA needs more data than BT to learn a stable prototype, or how small-sample performance behaves.

### Questions
- How is $z_p^*$ updated when new preferences are added or when different annotator populations are used? Does it require retraining from scratch?
- How does SARA’s performance change with fewer preference samples? Does it require more data than BT to stabilize the pivot latent?

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
This paper presents SARA, a new approach for preference-based reinforcement learning that aims to enhance robustness to noisy or inconsistent human feedback. SARA, instead of Bradley-Terry (BT) based methods, takes a set-based contrastive approach: it encodes preferred and non-preferred trajectories using a two-stage transformer encoder, producing a latent representation of preferences. The proposed method demonstrates strong performance on offline PbRL benchmarks, particularly when the quality of the human feedback is low.

### Strengths
- The paper addresses an important problem in PbRL literature, that the quality of the human feedback can vary.
- The proposed method outperforms the baselines when the label noise is introduced.

### Weaknesses
- The detailed design choices of the proposed method seem somewhat arbitrary. For instance, in Step 1 of Figure 1, the method splits the positive and negative embeddings into two subsets. However, it would also be possible to divide them into m subsets, where m is smaller than the batch size, or to omit the second encoder entirely and instead use a simple aggregate statistic—such as the mean or median—of the positive embeddings as z*_p. The authors should provide explicit justification for these design decisions, ideally supported by empirical comparisons demonstrating that the chosen configuration offers measurable advantages.
- The presentation of Figure 2 could be improved for interpretability. Each panel currently shows the performance of a single method while varying the label noise level. To facilitate direct comparison across methods, it would be clearer to reorganize the plots so that each panel corresponds to a fixed label quality, displaying the performance of all methods under that condition.
- In Table 1, the performance gap between the proposed method and the baselines does not seem significant except for hopper-med-replay and kitchen-mixed tasks when the error rate is 20%.
- The analysis accompanying Table 2 may be somewhat misleading. It is possible for multiple reward functions to induce the same optimal policy. In such cases, the learned reward function need not align closely with the original environment reward, and correlation with that reward does not necessarily indicate better policy quality. The authors should clarify this limitation and interpret the reported correlations accordingly.

### Questions
- SARA encoder takes as input multiple segments during training, but it seems the encoder takes as input a single segment during inference, according to Figure 1 step 3. Why is the input structure different for training and inference?
- Can the authors also evaluate the method on halfcheetah and Adroit?

### Soundness
3

### Presentation
3

### Contribution
3
