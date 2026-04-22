# SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 2, 4, 8, 8

## Abstract
Large-scale robot learning has made progress on complex manipulation tasks, yet long-horizon, contact-rich problems—especially those involving deformable objects—remain challenging due to inconsistent demonstration quality. We propose a stage-aware, video-based reward modeling framework that jointly predicts task stage and fine-grained progress, using natural-language subtask annotations to derive consistent labels across variable-length demonstrations. This avoids the brittleness of frame-index-based labeling and provides stable supervision even in tasks like T-shirt folding. Our reward model is robust to demonstration variability, generalizes to out-of-distribution scenarios, and improves downstream policy training. Building on it, we introduce Reward-Aligned Behavior Cloning (RA-BC), which filters and reweights demonstrations based on reward estimates. Experiments show that our method significantly outperforms baselines in both real-world rollouts and human validation. On T-shirt folding, we achieve 83\% success from the flattened state and 67\% from the crumpled state, compared to 8\% and 0\% with vanilla BC. Overall, our results highlight reward modeling as a scalable and annotation-efficient solution for long-horizon robotic manipulation. Project website: https://qianzhong-chen.github.io/sarm.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a reward-modeling approach that supplies stable, dense feedback for long-horizon manipulation. It decomposes progress estimation into two parts: (1) a high-level predictor that identifies the current stage of the task and (2) a fine-grained regressor that measures progress within that stage. This two-stage design reduces sensitivity to variable demonstration lengths (e.g., in cloth folding) and yields more reliable rewards. Building on this signal, the authors also introduce an imitation-learning scheme that up-weights higher-quality demonstrations. Experiments show consistent gains over single-stage progress-prediction baselines, with real-robot results on cloth-folding validating the approach.

### Strengths
- Strong real-world gains: On a challenging cloth-folding task, a naive fine-tuning SOTA behavior policy (Pi0) fails 0/12 trials, while the proposed method succeeds in 8/12—highlighting the efficacy of their good reward modeling.
- The paper analyzes performance across (i) high-level stage-prediction accuracy, and (ii) end-to-end policy rollouts, helping readers to understand where the method adds value.
- Consistent improvements: The performance improvement holds across different rollout outcomes (success/partial/failure progress prediction) and task difficulty, from simple cloth pick-and-place to full folding.

### Weaknesses
**Novelty of the stage-aware reward modeling.**
  + Stage-aware reward modeling for long-horizon manipulation is not new per se. Please clarify what is distinct, and compare directly to prior work (e.g., Drs [1], REDS [2]).

**Clarity on data-quality literature.**

  + L76–79 state data quality is hard to assess beyond simple heuristics (e.g., duration). This could be misleading: recent work proposes stronger proxies. Please cover this literature (like [3] and [4]) and briefly position your method relative to these proxies.

Please clarify the literature on stage-aware reward modeling and data quality, state how your approach differs from prior work, and—if possible—add experiments showing superiority over these to strengthen your paper’s contribution and rigor.

### References
- [1] Mu et al., "Drs: Learning reusable dense rewards for multi-stage tasks", ICLR 2024.
- [2] Kim et al., "Subtask-Aware Visual Reward Learning from Segmented Demonstrations", ICLR 2025.
- [3] Belkhale et al., "Data Quality in Imitation Learning", NeurIPS 2023
- [4] Dass et al., "DataMIL: Selecting Data for Robot Imitation Learning with Datamodels", arXiv preprint arXiv:2505.09603 2025

### Questions
- [Q1] Is the low performance due to underfitting?
    + In Table 2, ReWiND improves with longer training (e.g., 20K Medium: 1/12 → 40K Medium: 6/12). Does this indicate the model hasn’t converged yet and could achieve higher performance with more training? (possibly outperform your model as well?)

- [Q2] Model architecture parity
    + Are ReWiND and SARM matched in model size? Since original ReWiND leverages large-scale data (OXE), it may use a larger model, which could overfit on your 200-hour dataset.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces SARM, which introduces stage information about a task to improve reward predictions given a real world robot trajectory/video. The results show this outperforms prior work like ReWiND and aligns closer to the ground truth progress rewards. The learned reward model in SARM further is used to reweight training samples for better performance.

### Strengths
- Introduces a novel use of stage prediction to get improved auto generated reward labels of a dataset. This contrasts with prior work that simply used frame indices of a video as the label of the reward progress, which is limited since different stages of the task may proceed at different speeds.
- This work further introduces some useful tricks that improve performance and leads to easier to learn reward models that are much more smooth.
- Strong improvement upon prior work is shown with ablations indicating how the different proposed ideas improve SARM.

### Weaknesses
- Compared to prior work this method requires more annotation of datasets (subtask labels).
- There doesn't seem to be an ablation on the reward-model based weighting of training samples, which I believe would be fairly important given the amount of space allocated to describe the RA-BC method. The discussion in 4.3 suggests that a bad reward model would cripple RA-BC results since it may lead to poor reweighting, it may be possible that not doing any re-weighting may end up working better.

### Questions
See above

Generally think this work is solid and should be accepted. Happy to raise score if the question about the reward reweighting is addressed.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
A core challenge in imitation learning literature (e.g. behavior cloning) is learning from diverse demonstration data: multi-modal action distributions, varying teleoperation strategies and proficiency, human errors, as well as sometimes wildly different trajectory lengths depending on the teleoperator and initial/goal configurations. This paper aims to automatically estimate task progress in demonstration data regardless of factors such as trajectory length (recent method ReWiND is prone to fail in this setting), and then subsequently using estimated task progress (reward) to weigh samples during BC policy learning. To do so, the proposed method estimates stage and subtask progress from image observations + joint states + language annotations using a transformer backbone, as well as a number of engineering decisions and heuristics detailed in the paper. Experiments are conducted on a real-world T-shirt folding task using a dual-arm setup, and results indicate that the proposed framework (SARM / RA-BC) improves task progress estimation and downstream policy performance compared to strong baselines.

### Strengths
My initial assessment of this paper leans positive overall; it is generally well written, timely, and technically sound. Specifically:
- I believe that this paper studies a relevant and timely problem (learning from "diverse" demonstrations wrt. manipulation strategy, trajectory length etc.), and is likely to be of interest to the community. The problem is clearly motivated in the introduction, and shortcomings of existing work (ReWiND in particular) is described in the related work section. The paper is generally well written and easy to follow, and the illustrations (Figure 1, 2) are helpful for understanding the technical contributions.
- The proposed method seems technically sound and appears to rely fairly little on engineering or manual labor which is definitely a plus (besides language annotation of the demonstration data, I didn't see much discussion on this in the paper but I imagine it may be rather time-consuming). The approach is rather simple and intuitive (I consider this a strength), which makes me believe that it could feasibly be transferred to other tasks and potentially entirely different task domains provided that they can be separated into stages / subtasks.
- I appreciate the focus on a difficult real-world task such as T-shirt folding, and the empirical results on this task are rather compelling. In particular, the qualitative comparison with ReWiND in Figure 3 clearly demonstrates the shortcoming of ReWiND: it is not very robust to varying trajectory lengths and, as a result, struggle with predictions at either end of the completion spectrum. The paper also includes a limited set of results on a dish unloading task (appendix A.6) as well as RL in simulation (pick cube) and the conclusions seem to be rather similar across these tasks and problem settings.

### Weaknesses
I do not have any major concerns with the paper in its current form. However, I do believe that there is room for improvement in several ways:
- The writing is a bit odd, with large parts of the experiment section moved to the appendices entirely. It would be helpful if the authors can be more clear about the experimental setup (T-shirt folding, dish unloading, RL finetuning in simulation) and perhaps summarize the results across these different settings in the main paper rather than only discussing them in the appendices; it is easy for readers to miss otherwise.
- I am a little concerned about this particular engineering choice: *"During annotation, only the subtasks defined by the protocol were labeled, and any trajectory that did not contain the complete sequence of subtasks specified by the protocol was discarded. Annotators watched the top-view video and segmented each trajectory into subtasks by recording the start and end frame indices. If a mistake occurred during execution, its start and end frames were also labeled; trajectories containing mistakes were excluded from subsequent model training"* (L184). If one of the main claims of the paper is that SARM / RA-BC can learn from diverse demonstration data, why exclude trajectories that *e.g.* contain a human error somewhere in the trajectory? If the approach was truly robust, it should be able to infer that such actions are indeed errors that then subsequently are corrected by the operator. It would be helpful if the authors can clarify this particular aspect of their approach, and potentially include some experimental results in which data is not filtered as aggressively before any learning; that should give readers a better indication on what the key steps in the data pipeline are.

Minor comments that need to be addressed but are insignificant wrt my overall assessment of the work:
- The abstract is quite long which feels unnecessary, I would suggest revising it to be more concise.
- It would be helpful if the authors could introduce the considered tasks somewhere in the main paper along with e.g. illustrations of what those tasks look like.
- There are a few typos and grammatical errors throughout the paper. While they do not affect understanding, it would be a good idea to proof-read the paper before publication.

### Questions
I would really appreciate it if the authors can address my comments in the "weaknesses" section above using written arguments and potentially additional experimental results. My main concerns / questions pertain to writing and the data filtering process.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this work, the authors introduce a stage-aware, video-based reward modeling framework that jointly predicts high-level task stages and fine-grained progress within each stage. This design effectively addresses the limitations of frame-index-based labeling, which often fails in long, variable-duration tasks such as T-shirt folding. The learned reward models are further utilized to filter high-quality data and reweight training samples based on the estimated rewards. Extensive experiments demonstrate that the proposed reward model outperforms existing baselines in out-of-distribution real-robot rollouts and human demonstration evaluations.

### Strengths
1. The authors investigate a very important and timely direction, i.e., how to acquire high-quality data for training imitation learners on long-horizon, contact-rich manipulation tasks such as T-shirt folding. Quantifying demonstration quality is challenging because it depends on latent factors such as action consistency and contact stability, which cannot be directly measured. In this work, the authors propose a video-based reward modeling framework that leverages natural language annotations to assign progress labels and enable stable reward estimation, serving as a mechanism to filter high-quality data and improve policy performance in both simulation and real-world settings. The proposed method is timely and presents a novel idea for learning from suboptimal demonstrations, making it a valuable contribution to the community.

2. The authors provide a comprehensive analysis of the proposed SARM reward model, comparing it against existing methods. The proposed approach demonstrates superior performance across multiple benchmarks. Furthermore, the authors evaluate the framework on  dish unloading, which further validates its effectiveness and generalization capability.

3. By leveraging SARM, the RA-BC policy surpasses both BC baselines by a significant margin on medium and hard tasks. The experiments show that RA-BC effectively exploits diverse datasets by filtering out high-quality data frames, enabling the policy to learn robust, long-horizon manipulation strategies.

4. The authors empirically demonstrate that RA-BC with SARM achieves substantially higher success rates than RA-BC with ReWiND, achieving 83% vs. 50% on medium tasks and 67% vs. 25% on hard tasks, clearly highlighting the advantage of the proposed reward model.

### Weaknesses
**Overall Assessment:**

The paper provides a well-justified motivation for the proposed reward model and demonstrates its utility in selecting high-quality data for policy learning. However, several aspects remain unclear and merit further clarification.

1. **Applicability to existing datasets:**
    
The proposed reward model has shown promise in filtering high-quality data, but it remains unclear how well it generalizes to existing cloth manipulation datasets, such as **DROID* [1] and **Flat’n’Fold** [2]. How would the model perform if applied to these datasets for filtering or reweighting demonstrations?
 
[1] DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset, 2024
[2] Flat’n’Fold: A Diverse Multi-Modal Dataset for Garment Perception and Manipulation, 2024
    
2. **Robustness to visual variations:**
    
How does the proposed reward model handle changes in lighting and illumination conditions? Since real-world environments are often visually diverse and dynamic, it would be valuable to understand the model’s robustness and reliability under such variations.
    
3. **Baseline comparison:**
    
The recent work by **Hung et al. (ICLR 2025)** [3] introduces an alternative strategy for learning reward models in long-horizon manipulation tasks. Although the learning objectives differ from those of SARM, a direct comparison between the two approaches under similar long-horizon, contact-rich settings would provide stronger empirical grounding and highlight the unique advantages of the proposed framework.

[3] Hung et al., VICtoR: Learning Hierarchical Vision–Instruction Correlation Rewards for Long-Horizon Manipulation, ICLR 2025

### Questions
1. Generalization: How well does the proposed reward model generalize when applied to large-scale, real-world cloth manipulation datasets such as DROID and Flat’n’Fold?

2. Robustness: How robust is the model to lighting and illumination changes, which commonly occur in real-world robotic settings?

3. Comparative Evaluation: How does the proposed method compare with VICtoR (ICLR 2025) in terms of reward modeling effectiveness and policy performance on long-horizon, contact-rich manipulation tasks?

### Soundness
3

### Presentation
4

### Contribution
4
