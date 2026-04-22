# GUI-Shift: Enhancing VLM-Based GUI Agents through Self-supervised Reinforcement Learning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Training effective Vision-Language Models (VLMs) for GUI agents typically depends on large-scale annotated datasets, whose collection is both labor-intensive and error-prone. We introduce ***K***-step GUI Transition, a self-supervised inverse dynamics task in which VLMs learn GUI dynamics by predicting the initial action that causes a transition between two GUI states. This approach eliminates the need for natural language instructions and enables scalable dataset construction from existing GUI trajectories or automated exploration. Building on this task, we propose **GUI-Shift**, a reinforcement learning (RL) framework that combines rule-based optimization with data filtering to improve VLM performance. We conduct extensive experiments using multiple VLM backbones across five benchmarks, spanning GUI task automation (AndroidControl, GUI Odyssey, AndroidWorld) and GUI grounding (ScreenSpot-v2, ScreenSpot-Pro). Our results show that training on GUI-Shift generalizes well to both GUI automation and grounding tasks, yielding up to an 11.2% increase in GUI automation accuracy. This study underscores the potential of self-supervised RL to leverage unlabeled GUI trajectories and offers a scalable alternative to training with annotated samples. GUI-Shift will be open-sourced at: https://github.com/UbiquitousLearning/GUI-Shift.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a novel self-supervised task for VLMs used in GUI agents. It is inspired by inverse dynamics within unlabeled GUI trajectories. They also propose a self-supervised RL framework with advantages in handling action multiplicity and poor generalization in GUI tasks.

### Strengths
- The primary strength is the "K-step GUI Transition" task. By formulating the training problem as a self-supervised inverse dynamics task , the authors provide a solution to bypass the need for expensive natural language annotations. This addresses the scalability bottleneck in GUI agent training data.
- The paper provides a well-crafted design of a RL-framework. Novelly set the visual goal as the supervision signal. This design requires the VLMs to capture both the diverse GUI settings and the dynamics transformation between states, therefore enhancing the situational awareness of the model.
- The experimental quality is a major strength. The authors validate their method across multiple VLM backbones and diverse benchmarks (automation and grounding). The extensive ablation studies (covering filtering , task formulation , training algorithm , and reasoning ) are thorough and leave little doubt about the efficacy of each component.
- The paper is exceptionally clear. Figure 1  is a model for how to visually communicate a complex framework, perfectly contrasting the proposed method with prior work

### Weaknesses
See Questions

### Questions
• Formatting Suggestion: A minor formatting note: On page 4, Line 254, the label “(3)” appears misaligned.

• Predicting Action Sequences: The current K-step GUI Transition task only requires the model to predict the first action initiating the transition from $S_t$ to $S_{t+k}$. However, the "Action Prediction" example in Figure 1 (Middle) implies the model reasons about a sequence of actions ("...I should click the search box and type... So, I think the first step should be: click..."). Have the authors considered or experimented with a more complex task objective, such as predicting the full sequence of $k$ actions required to reach $S_{t+k}$? If so, how did this compare to predicting only the initial action?

• Generality of Data Filtering: The data filtering pipeline, which selects samples with both correct and incorrect predictions, is presented as a key component for improving performance (Figure 2). However, this filtering was not applied to the Qwen2.5-VL-7B model because it produced an "exceptionally high" proportion of all-correct or all-incorrect samples5. This suggests the filtering heuristic may be model-dependent or brittle. Could the authors comment on the generality of this filtering component? Does this finding imply that the heuristic's effectiveness is strongly tied to the base model's initial capabilities or calibration?

• Impact of Omitting Reasoning Traces: The ablation study in Table 3 shows that omitting explicit reasoning traces during training not only nearly halves training time but also improves downstream performance. This finding is significant and somewhat counter-intuitive. Could the authors provide further explanation or hypotheses for this phenomenon?

• Analysis of the $k$ Parameter: The choice of $k$ in the K-step GUI Transition task is a central hyperparameter. Based on Tables 1 and 2, the optimal $k$ varies by model and task. For instance, $k=1$ (predicting the transition to the immediate next state) often performs very well, particularly for GUI grounding (best for 3/4 models). Conversely, Qwen2.5-VL-7B and InternVL3-8B achieved their best automation scores with $k=4$. Do the authors have any insights into this behavior? Is there an observable trade-off between the signal clarity of small $k$ values and the long-term goal-conditioning provided by larger $k$ values?

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
4

### Summary
The paper proposes GUI-Shift, a self-supervised reinforcement learning framework for training VLM-based GUI agents without manual task instructions. The core pretext task is K-step GUI Transition: given a current screenshot and a target future screenshot, the model predicts the first action that initiates the transition. Training uses GRPO with a simple, verifiable reward, and a model-specific data filtering step that retains samples where the model produces both correct and incorrect actions under stochastic decoding. The authors fine-tune four open VLMs and report gains over base models on AndroidControl, GUI Odyssey, ScreenSpot-v2, and ScreenSpot-Pro.

### Strengths
S1: The K-step Transition objective is a compact way to leverage unlabeled trajectories at scale. The visual-goal formulation reduces reliance on noisy textual instructions and naturally encourages temporal reasoning.


S2: GRPO avoids penalizing multiple valid clicks (any point in a control) and removes the need for a critic. The binary-format action reward is stable, inexpensive, and reproducible.

S3: On AndroidControl-High, GUI-Shift-Qwen improves EM by +11.2% (to 70.4%), with other base models also benefiting. Despite training only on AndroidControl trajectories, models improve on ScreenSpot-v2 and ScreenSpot-Pro, demonstrating potential cross-platform generalization for GUI grounding.

### Weaknesses
W1: Training relies solely on the AndroidControl split, which undercuts the claim that the method scales to arbitrary “unlabeled trajectories.”

W2: Performance dips on GUI Odyssey for some models are attributed to tablet layouts; ScreenSpot‑Pro (desktop, high‑res) gains are modest.

W3: The work evaluates offline action accuracy and grounding. Demonstrating end‑to‑end success in a dynamic environment like AndroidWorld would strengthen claims about task automation robustness.

### Questions
(1) What fraction of candidate pairs survive filtering per model and per k? What is the inter-model overlap of filtered pairs?

(2) In Table 3, were the task/step instruction baselines also trained with GRPO and the same binary-format reward?

(3) If only click points are recorded, how is the box inferred, and how would GUI‑Shift operate on real logs that lack element metadata?

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
4

### Summary
GUI-Shift is a self-supervised RL framework that applies GRPO to the “K-step GUI Transition” task: given a current screen St and a future screen S_(t+k) taken from real trajectories, the model predicts the first action that initiates the transition. It optimizes with rule-verified rewards (format + action correctness) over a unified 8-type action space, and uses a sampling-and-scoring loop for both training and data filtering. Trained on 2K samples per backbone (k∈{1,2,3,4}) and evaluated on AndroidControl/GUI Odyssey (automation) plus ScreenSpot-v2/Pro (grounding), GUI-Shift reports notable gains—up to +11.2% accuracy on AndroidControl-High and +2.5% on ScreenSpot-v2—without extra alignment.

### Strengths
GUI-Shift is practical and scalable because it builds supervision directly from real GUI trajectories (no manual labels) and trains with a GRPO sampling-and-ranking loop that better accommodates multiple plausible actions than single-label SFT, while its rule-verified rewards provide precise, automatic checks on action type and arguments. The unified eight-action interface and “keep only informative cases” filtering keep optimization focused and stable, and dropping explicit reasoning traces cuts training overhead without hurting downstream performance, together yielding consistent gains on mobile GUI automation and even modest improvements on GUI grounding.

### Weaknesses
The method treats the trajectory’s first action as the only gold label. Rule-based rewards allow coordinate variance (e.g., any point inside the bbox) but not type/argument alternatives, so equally valid first moves that still reach S_(t+k) get penalized, biasing against strategy-level equivalence.

Training/eval center on AndroidControl and GUI Odyssey (mobile). Grounding sets are included, but not multi-step control. Metrics are mostly single-step TM/EM, no end-to-end success rates, latency/token cost, or cross-OS/desktop multi-app scenarios—so external validity is limited.

Overly literal rewards, weak goal alignment. Rewards favor exact action type/argument matches (e.g., verbatim input_text, specific scroll step) instead of measuring progress toward the target screen S_(t+k). This creates false negatives for synonymous inputs or alternative scrolls and teaches label imitation over goal achievement.

First-step-only training/eval (short-horizon myopia). Given (St, S_(t+k)), the model learns only the first action, and metrics reflect single-step accuracy. This risks poor multi-step planning and error recovery during rollouts, weakening correlation with real agent performance.

### Questions
see above

### Soundness
3

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
2

### Summary
This paper proposes a self-supervised K-step GUI Transition task, which enables VLMs to learn GUI dynamics by predicting the initial action that causes a transition between two GUI states.

The approach eliminates the need for natural language instructions and enables scalable dataset construction from existing GUI trajectories. Building on this task, the author propose GUI-Shift with reinforcement learning (RL) framework. Experiments show that training on GUI-Shift generalizes well to both GUI automation and grounding tasks, yielding up to an 11.2% increase in GUI automation accuracy.

### Strengths
This paper proposed a novel GUI task, k-step GUI transition task, which enables self-supervised training and eliminate dataset construction.

### Weaknesses
Despite the efficiency of the method, mechanisms and deeper analysis are somewhat lacking - please see details in questions part.

### Questions
1. How to assure the quality of training data? From fig. 1, the task has 2 target, task prediction and action prediction, action prediction can achieve by compared to ground-truth, but how to assure task prediction is aligned with the actual target.  
2. Is there an explanation why the shift prediction task could improve the performance of grounding task, which seems to be irrelevant even when k=4.
3. Could GUI-shift generalize to different scenarios and tasks, like out-of-domain tasks. 
4. Is there some connections between step K and task difficult or model’s capability. 
5. Could model tackle task if the trajectory includes significant state shift across screenshots like mobile application tasks (ac is relative simple) 
6. Could K-step GUI Transition task possesses scaling law compared to datasets constructed by general trajectory, further experiment is encourage if calculation resource is enough.

### Soundness
3

### Presentation
3

### Contribution
3
