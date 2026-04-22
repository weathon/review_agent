# TrajTok: What makes for a good trajectory tokenizer in behavior generation?

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Behavior generation in autonomous driving aims to simulate dynamic driving scenarios from recorded driving logs. A popular approach is to apply next-token-prediction with discrete trajectory tokenization. In this work, we explore what makes a good trajectory tokenizer from the perspective of logged data usage. We first analyze the four properties (coverage, utilization, symmetry and robustness) of vocabularies of data-driven and rule-based trajectory tokenizers and their impact on performance and generalization. Data-driven tokenizers often build vocabularies with better utilization but suffer from insufficient coverage and sensitivity to noise, while rule-based methods have better coverage but contain too many useless tokens. With these insights, we propose TrajTok, a trajectory tokenizer that combines the two methods with rule-based vocabulary candidate setup and data-driven filtering and selection processes. The tokenizer has balanced coverage and utilization as well as good symmetry and robustness. Furthermore, we propose a spatial-aware label smoothing method for the cross-entropy loss to better model the similarities between the trajectory tokens. Our method wins first place in the 2025 Waymo Open Sim Agents Challenge.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes TrajTok, a trajectory tokenizer for discrete next-token-prediction (NTP) behavior generation in autonomous driving. It combines data-driven and rule-based trajectory tokenizers, and has balanced coverage and utilization as well as good symmetry and robustness. It builds on top of the SMART NTP behavior generation backbone. It shows decent results in the 2025 Waymo Open Sim Agents Challenge.

### Strengths
1. The coverage, utilization, symmetry, and robustness perspectives are intuitive and useful.
2. The algorithm is simple and the resulting vocabulary is nice and clean. 
3. The algorithm achieves good empirical results and high ranking on Waymo Open Sim Agents Challenge.

### Weaknesses
1. The authors claim that TrajTok wins first place in the Waymo Open Sim Agent Challenge 2025. This contradicts Table 1, which shows SMART-R1 wins the first place, while TrajTok places 2nd. Looking at the official leaderboard (https://waymo.com/open/challenges/2025/sim-agents/) we see that TrajTok placed the 5th. 
2. TrajTok changes the tokenizer on top of the SMART backbone. Since the top 5 models on the official leaderboard are all variants of SMART, it seems TrajTok’s contribution is incremental or marginal.

### Questions
The results on Table 2 use 20% of the training set. Do other tokenizers (VQ-VAE, K-means, K-disks, Grid) close the gap using a larger training set?

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
3

### Summary
This paper studies what makes for a good trajectory tokenizer for next token prediction behavior generation and proposes TrajTok, a plug and play tokenizer that combines rule based vocabulary construction with data driven filtering and expansion. The method explicitly targets four properties of a vocabulary, namely coverage, utilization, symmetry, and robustness, and introduces a spatial aware label smoothing scheme that assigns higher probability to tokens that are closer in trajectory space to the ground truth. The authors evaluate on the Waymo Open Motion Dataset and the Waymo Open Sim Agents Challenge and report first place on the 2025 leaderboard with consistent gains when TrajTok is used with a SMART backbone on validation metrics.

### Strengths
- There is a clear problem framing and taxonomy of desirable tokenizer properties. The paper analyzes how data driven methods often achieve high utilization but limited coverage and how rule based methods tend to overcover with many unused tokens. This motivates a hybrid design that TrajTok implements.
- The construction pipeline is simple and reproducible. The four step procedure is easy to follow, from agent centric normalization and symmetry by flipping, to grid based candidate selection, to neighborhood based filtering and expansion, to final token generation including curve interpolation when a selected cell has no examples.
- The paper reports strong results on both the public leaderboard and controlled comparisons, showing first place on WOSAC 2025 and better validation performance than VQ VAE, K means, K disks, and a pure grid, using a SMART based model under a common vocabulary size. Cross dataset and low data experiments further support generalization.

### Weaknesses
- Scope is limited to trajectory tokenization. The method relies on a base next token prediction model, most experiments use SMART tiny, and the paper makes a few implementation choices that deviate from the original backbone, such as separate heads per agent type. It would help to isolate how much gain comes purely from the tokenizer versus modest architecture changes.
- Computational cost and model scale trade offs are not discussed. The approach can yield large vocabularies across agent types, and spatially aware label smoothing appears to require computing distances from each target to many tokens. A brief cost analysis and a description of any approximations would improve clarity. The paper notes overall training settings but not the incremental overhead from TrajTok and the smoothing.
- Minor presentation issue - I could not find what the highlight colors mean in Tables 1 and 2.

### Questions
1. The appendix switches to separate prediction heads per agent type. How much of the improvement in validation metrics is due to this change rather than the tokenizer? A brief ablation on common heads versus separate heads would clarify attributions.
2. How sensitive are the reported gains to the grid range and resolution per agent type and to the neighborhood thresholds for filtering and expansion? A small sweep around the appendix settings would help readers tune the method.
3. In L377, “Increasing the vocabulary size improves the ability to represent complex distributions but may lead to model underfitting”, should this be the opposite (i.e. a smaller size causes underfitting)? Intuitively a larger vocabulary size should need more data, is under-training a better word?

### Soundness
2

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
The paper studies trajectory tokenization for next-token-prediction (NTP) behavior generation in autonomous driving simulators. 
It argues that existing data-driven tokenizers such as VQ-VAE or K-disks have good utilization but poor coverage and symmetry, while rule-based grid tokenizers have wide coverage but many redundant or unrealistic tokens.
To balance these properties, the authors propose TrajTok, a hybrid tokenizer that first builds a rule-based grid vocabulary and then filters and expands it using logged trajectory data. They also introduce spatial-aware label smoothing for cross-entropy loss, where non-ground-truth tokens are weighted according to spatial distance.
Experiments on the Waymo Open Motion Dataset and the 2025 Waymo Open Sim Agents Challenge show that TrajTok ranks first on the leaderboard and yields slightly higher realism metrics than baselines.

### Strengths
1.	Clear motivation from data analysis. The authors systematically examine four tokenizer properties (coverage, utilization, symmetry, robustness) and relate them to logged data usage. This diagnostic perspective is useful for understanding tokenization quality.

2.	Simple yet general method. TrajTok is a lightweight combination of rule-based and data-driven principles that can be plugged into existing NTP architectures without retraining structural components.

3.	Practical evaluation. The paper reports results on the Waymo Open Sim Agents Challenge with official metrics, including the Realism Meta score, and provides ablations for label smoothing and vocabulary size.

### Weaknesses
# Major

- Unclear evidence of improvement: In Table 1, TrajTok achieves 0.7852 while SMART-R1 reaches 0.7855, and SMART-tiny-CLSFT gives 0.7846. These gaps are within noise. It is difficult to identify a clear gain attributable to the tokenizer.

- Ambiguous reference tokenizer: Table 2 compares TrajTok with VQ-VAE, K-means, K-disks, and grid tokenizers, but it is not stated which tokenizer SMART-tiny originally used on the leaderboard. The reader cannot determine whether the new tokenizer outperforms the baseline used in the challenge submission.

- Vague qualitative evidence: Figure 4 does not convincingly demonstrate that the proposed tokenizer contributes to better behavior generation. The visualization focuses on scene outcomes rather than showing how tokenization differences influence the trajectories. Without comparisons using the same scenario and seeds, the figure offers little empirical value.

- Questionable justification of symmetry: The paper claims symmetry is critical for vehicle kinematics and real-world diversity, yet real traffic is not necessarily symmetric. For example, in right-hand traffic countries, turning behaviors are directionally biased. The necessity of symmetric flipping should be justified with more empirical or theoretical evidence. Table 6 shows a small gain from symmetry, but the physical rationale is not convincing.

# Minor

- Limited exploration of failure cases: The discussion does not examine cases where the tokenizer introduces unrealistic motion patterns or under-represents long-tail behaviors.

### Questions
What tokenizer was used in the SMART-tiny baseline that appears in Table 1? Without knowing this, it is difficult to measure the real gain from TrajTok.

Have you tested TrajTok on prediction tasks that use continuous outputs rather than discrete NTP models to verify that the benefit comes from tokenization rather than training heuristics?

### Soundness
3

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
The paper presents TrajTok, a hybrid trajectory tokenizer designed for behavior generation in autonomous driving. It investigates what constitutes an effective trajectory tokenizer under the next-token prediction (NTP) paradigm, analyzing four key properties—coverage, utilization, symmetry, and robustness—of existing data-driven and rule-based approaches. TrajTok integrates the advantages of both: it first constructs a rule-based grid of trajectory candidates and then applies data-driven filtering and expansion to balance vocabulary coverage and data efficiency. Additionally, the authors propose a spatial-aware label smoothing technique that weights token similarity by spatial distance, improving model generalization. Experiments on the Waymo Open Motion Dataset demonstrate that TrajTok achieves state-of-the-art performance, ranking first in the 2025 Waymo Open Sim Agents Challenge with superior realism and robustness across datasets and data scales.

### Strengths
(1) This paper makes a clear and timely contribution by looking closely at what makes a good trajectory tokenizer in the next-token prediction (NTP) setup. The four proposed criteria—coverage, utilization, symmetry, and robustness—give a simple but useful way to understand and compare different tokenizers, which hasn’t really been discussed in earlier work like Trajeglish or MotionLM.

(2) The proposed TrajTok method is simple but well thought out. It combines a rule-based start with data-driven filtering and expansion, which makes sense and nicely balances coverage and efficiency. This hybrid idea helps fix problems that appear in purely data-driven (too noisy) or rule-based (too redundant) tokenizers.

(3) The paper also adds a spatial-aware label smoothing technique that slightly changes the standard cross-entropy loss. It’s an intuitive idea that takes spatial similarity between tokens into account, helping the model generalize better without depending on any specific architecture.

### Weaknesses
(1) The paper only validates TrajTok within the SMART model [1]. Since TrajTok is designed as a general tokenizer, applying it to other NTP-based architectures (such as Trajeglish [2] or MotionLM [3]) would further support its claimed generality.

(2) The paper defines several thresholds for the filtering and expansion process, but the actual parameter values and tuning details are not provided. It is unclear how sensitive the model is to these choices or whether small changes in these thresholds would affect the final vocabulary and performance. Including the specific values or a short sensitivity analysis would help improve reproducibility and confidence in the results.

[1] Wei Wu, et al. “Smart: Scalable multi-agent real-time motion generation via next-token prediction.” Advances in Neural Information Processing Systems, 37:
114048–114071, 2024.
[2] Jonah Philion, et al. “Trajeglish: Traffic modeling as next-token prediction.” arXiv preprint arXiv:2312.04535, 2023
[3] Ari Seff, et al. “Motionlm: Multi-agent motion forecasting as language modeling.” In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp.
8579–8590, 2023.

### Questions
Please refer to the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
