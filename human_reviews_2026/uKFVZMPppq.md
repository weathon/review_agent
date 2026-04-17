# Action-Guided Attention for Video Action Anticipation

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Anticipating future actions in videos is challenging, as the observed frames provide only evidence of past activities, requiring the inference of latent intentions to predict upcoming actions. Existing transformer-based approaches, which rely on dot-product attention over pixel representations, often lack the high-level semantics necessary to model video sequences for effective action anticipation. As a result, these methods tend to overfit to explicit visual cues present in the past frames, limiting their ability to capture underlying intentions and degrading generalization to unseen samples. To address this, we propose Action-Guided Attention (AGA), an attention mechanism that explicitly leverages predicted action sequences as queries and keys to guide sequence modeling. Our approach fosters the attention module to emphasize relevant moments from the past based on the upcoming activity and combine this information with the current frame embedding via a dedicated gating function. The design of AGA enables post-training analysis of the knowledge discovered from the training set. Experiments on the widely adopted EPIC-Kitchens-100 benchmark demonstrate that AGA generalizes well from validation to unseen test sets. Post-training analysis can further examine the action dependencies captured by the model and the counterfactual evidence it has internalized, offering transparent and interpretable insights into its anticipative predictions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on the task of long-term activity anticipation in videos. The main technical contribution is the advocacy of using model frame-wise raw output probability as feature / input to another attention process, where the self-produced predictions serve both as query and key, meanwhile the frame embedding as the value. Moreover, standard attention for long-range (temporal) modelling is identified as sub-optimal in this paper, and authors instead propose to use standard temporal smoothing, i.e. EMA and adaptive filtering, i.e., gating, techniques in the model for better performance. Strong experimental results are demonstrated to support the method.

### Strengths
A. The writing is smooth, math easy to read and overall clarity is good. The experimential results are strong across 3 different datasets for activity anticipation, which demonstrate the generalization ability of the proposed approach. The overall approach is quite simple and easy to implement.

B. This reviewer has found the AGA design interesting, though less well-structured / supported throughout the documents. The way AGA works is: Action prediction (distribution over all predefined classes), \hat{y}, is the self-produced outputs and it then works on the incoming frame-level embedding (pre-extraced) progressively along the time axis for next-step prediction. This core design bears a lot similarity to the autoregressive cross-attention~ish fashion, which, however, differs most critically in the information abstraction level - the AGA module works on the semantic level abstraction, e,g., re-use the raw probability prediction. A summary of Q/K/V design against the orignal applications (transformers) is as follows.

| Model                                  | Query Source             | Key/Value Source                                       | Category                                       |
| -------------------------------------- | ------------------------ | ------------------------------------------------------ | ---------------------------------------------- |
| **Transformer Encoder**                | Same input sequence      | Same sequence                                          | Self-attention                                 |
| **Transformer Decoder (e.g., in GPT)** | Generated tokens         | Encoder outputs                                        | Cross-attention                                |
| **AGA**                                | EMA of predicted actions | Past actions (semantic) / past frame features (visual) | **Semantic-Visual Cross-Attention** |

The intution of transformer, by the best knowlege of this reviewer, is that re-weighting the “Value” vectors using relationships between “Query” and “Key” vectors. In this case, AGA is re-weighing the past frame visual representations using the relationship between semantic query and key vectors, which fits the demand of this paper. However, this leads to the 1st weakness point in the next section.

### Weaknesses
A.  It remains unclear what exactly make the AGA working for this task, and hinder readers summarize useful take-away message from this paper. Here are some questions to start the discussion. 

- According to the statement, quote "only rely on dot-product attention over pixel representations...", it seems the key finding here is that information in visual tokens being too preliminary, and the semantic feature is better. Yet, isn't \hat{y} derived out of the pixel representation, e, in the first place through a output head, f_{\theta}? So, what is the magic here empowering \hat{y} with stronger information after some MLP projection?  Or say it in another way: if the pixel representations already encode the necessary semantics, why does the paper inject predicted logits (the model’s own action scores) into attention — instead of operating on the hidden-state space directly? A quick extra experiment using last-layer hidden states might shed lights on this matter.

- The paper argues that visual cues might make the model prone to over-fitting. This reviewer would doubt if that's still true after the second encoder, f_{x}, especially after a few epochs of training. This argument is never supported with experimential study. A good one could be monitoring the validation set loss under two different settings - one with semantic feature and another one with visual features along training epochs. 

- The cross-attention alike mechansim in AGA seems unpredecent from previsouly published work. AVT (Anticipative Visual Transformer) can be seen as the canonical autoregressive transformer baseline, which only tested self-attention on visual tokens, not the cross-attention over past visual features and the next incoming frame-level feature. Maybe a study of a direct comparison where Q = current visual embedding and K/V = past visual embeddings (true autoregressive cross-attention)  would isolate the benefit of using semantic (action) representations from the benefit of directional cross-attention.

- The EMA and gating techniques can be implicitly achieved using attention. Prior-driven technique are powerful but less generic. It seems the main motivation here is resolving the training difficulity. Replacing generic module with prior-driven one does not seem to be a scalable method. Maybe the authors can share the failure logs (training logs) of using learnable components (such as pure attention). The community might come up better optimization idea for this issue. 

B. The training recipe is missing quite a few important information for re-production purpose:
- frame sampling rate
- queue initiation - It’s unclear how the queue is filled at the start of a clip (zeros? repeats? skip first S frames?).
- Any data augmentation? 
- EMA update scope - Is EMA applied to the raw logits or the softmax output (normalized prob.)?
- Again on the past prediction, y_{t-1: t-s}, are they also being optimized with the cross-entropy loss? If so and they are produced auto-regressively, are the BTPP algorithm used to optimized a sequence of inter-dependent predictions or separately?
- Prediction head architecture - the exact dim and layers of MLP here.


C. The post-training analysis study is a bit pre-mature. The Forward and Backward analysis are interesting and visually compelling but lack methodological transparency and quantitative validation.
Key implementation details (which layer, normalization, gradient target) are omitted, and no formal evaluation is conducted to confirm that these analyses genuinely capture causal or influential dependencies. This reviewer would not think of it as a well-rounded interpretability study.

### Questions
A. How is the past predictions, \hat{y}_{t-1: t-s}, produced? The reviewer guess that during training, the model is unrolled over S steps and computes each \hat{y} auto-regressively, feeding those predictions into the next step’s queue. Please confirm the details.

B. Add an equation for producing \hat{y}.

C. This reviewer believe the following work on acitivity anticipation are revalent for discussion.
- When will you do what?- Anticipating Temporal Occurrences of Activities, CVPR 2018,
- Time-Conditioned Action Anticipation in One Shot, CVPR 2019,
- On diverse asynchronous activity anticipation, ECCV 2020,

### Soundness
3

### Presentation
2

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
The manuscript proposes Action-Guided Attention (AGA), a semantic attention mechanism for video action anticipation. Instead of relying on pixel-level similarity, AGA uses predicted action distributions as queries and keys to focus on temporally relevant past frames. An adaptive gating module further balances historical and current visual cues. The authors also introduce forward and backward analyses to interpret the model’s learned dependencies. Extensive experiments on EPIC-Kitchens-100, EPIC-Kitchens-55, and EGTEA Gaze+ demonstrate consistent improvements over prior methods.

### Strengths
1. The manuscript proposes a novel approach that leverages the model’s own predicted action distributions to guide attention, enabling it to focus on high-level contextual cues rather than low-level visual similarity. This design offers a fresh and effective perspective for modeling in action anticipation.
2. The method shows consistent improvement on tail and unseen classes, indicating enhanced robustness across class distributions compared to prior work.
3. The proposed forward and backward analyses offer interpretable insights into how the model captures causal dependencies in its high-level representations, illustrating meaningful internal reasoning.

### Weaknesses
1. The model uses its own predicted action distributions as inputs (queries and keys) in the attention mechanism. While this design enhances semantic reasoning, inaccurate early predictions could propagate errors through time due to the recursive dependency. The manuscript introduces an exponential moving average to smooth the predicted distributions and stabilize temporal dynamics. However, EMA may only partially mitigate the potential error accumulation caused by this recursive dependence rather than resolving it. It would be helpful for the authors to clarify whether any mechanism is explicitly designed to address this issue, or if they have other considerations regarding how such recursive errors are handled during training.
2. The authors explicitly mention that AGA treats past actions as an “unordered set.” Without explicit temporal or positional encoding, the model might struggle to capture fine-grained causal order, which could limit performance in tasks where action sequences are strictly ordered.
3. On datasets such as EPIC-Kitchens-100, the class distribution is highly imbalanced. Since the model uses predicted action distributions as semantic inputs, it would be helpful to clarify whether this design could potentially amplify frequent-class bias or, conversely, help mitigate it through semantic attention. But the reported tail-class gains remain unexplained, and there is no discussion about this aspect. Please discuss why AGA helps rare classes.
4. The paper does not report quantitative efficiency statistics. Providing these measurements or comparisons with existing approaches would strengthen the empirical analysis and clarify the computational cost of AGA.
5. The evaluation focuses exclusively on egocentric kitchen datasets (EPIC-Kitchens and EGTEA), which limits the demonstrated generality of the proposed approach. It would be valuable to verify whether the Action-Guided Attention mechanism generalizes to more open or diverse domains.
6. The ablation study could be more fine-grained. While the paper evaluates the effect of the EMA coefficient and the presence of the gating module, it lacks comparisons between different gating architectures.

### Questions
1. The paper states that past actions are treated as an unordered set. How does the model still maintain temporal consistency?
2. Were any techniques applied to reduce frequent-class bias? How might uncalibrated probabilities affect the quality of semantic attention on long-tail classes? If none were used, could you discuss why AGA still improves tail classes and whether calibration might further help?
3. AGA is tested with both TSN and Swin-B backbones. Did the authors observe any difference in how semantic attention behaves with different backbone architectures? For instance, does a stronger backbone reduce the relative performance gain of AGA, or does the improvement remain consistent?

### Soundness
3

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
This paper proposes Action-Guided Attention (AGA), a novel attention mechanism for video action anticipation that uses predicted action probabilities as queries and keys, rather than pixel-level features. The core insight is that action-level semantics provide more effective guidance for attention when dealing with the inherent uncertainty and visual clutter in anticipation tasks. The method includes an adaptive gating mechanism that balances historical context and current evidence, and enables post-training analysis via forward and backward analysis techniques. The paper demonstrates strong competitive performance on EPIC-Kitchens-100 and other benchmarks, with clear ablation studies and strong generalisation.

### Strengths
Overall, this paper presents a novel, well-motivated approach to action anticipation, achieving competitive results and valuable interpretability. 

1. The core idea of using action-level semantics to guide attention is conceptually sound and addresses a real limitation of existing methods. The experimental validation is solid, with strong generalisation demonstrated across multiple benchmarks.

2. The interpretability analysis offers valuable insights into model behaviour, with forward analysis revealing attention patterns and backward analysis providing counterfactual reasoning paths. 

3. The paper demonstrates competitive performance on EPIC-Kitchens-100 (17.5% MT5R with TSN backbone, 18.8% with Swin-B on validation), with test set performance of 16.9% MT5R (Swin-B), showing strong generalisation with a narrow validation-to-test gap (approximately 1.9 percentage points). 

4. The ablation studies provide clear evidence for the contribution of both action-guided attention and adaptive gating components: replacing standard causal attention with action-guided attention provides substantial gains (15.9% to 18.2% MT5R), and adaptive gating adds further improvements (18.2% to 18.8%). 

5. The method is also validated on additional benchmarks (EPIC-Kitchens-55 and EGTEA Gaze+), demonstrating robustness across different datasets and annotation regimes, including sparse supervision settings where the model must rely entirely on its own predictions.

### Weaknesses
1.  Theoretical grounding: The paper provides strong intuitive motivation for using action-level semantics over pixel-level features, but lacks a deeper theoretical analysis. A more formal justification would significantly strengthen the contribution.
2.  Backward analysis convergence: In Section 3.4, the convergence criterion for the gradient descent in the backward analysis is stated imprecisely as "until the sequence doesn't change any more." 
3.  Ema parameter insight: While Table 6 provides a thorough empirical evaluation of the EMA parameter α, the paper lacks insight into why α=0.8 is optimal. 
4.  Statistical significance and reproducibility: The paper doesn't report standard deviations, confidence intervals, or the number of runs. Providing statistical significance testing for the key comparisons (e.g., the improvements from ablations and the validation-test gap) is crucial.
5.  Missing experimental details: The size of the FIFO queue 'S' is mentioned as a hyperparameter but not specified in the experimental setup. 
6. Furthermore, since AGA relies on self-predicted actions, a more explicit analysis of how prediction errors propagate and how the model handles error accumulation over time would be valuable.
7.  Failure cases and limitations: The paper doesn't discuss limitations or failure cases.  

Minor comments:

*   Table organisation: Reorganising results tables to explicitly group RGB-only comparisons versus multi-modal/ensemble methods would improve clarity.
*   Related work: The related work section could better position AGA relative to methods that also use semantic or high-level features (e.g., S-GEAR). How does AGA differ from these approaches?
*   Notation and presentation: There are minor inconsistencies in notation (e.g., line 147) and formula placeholders that should be corrected for the final version.

### Questions
1. What is the exact stopping criterion (e.g., L2 norm threshold, maximum iterations)? How sensitive are the results to the step size η and this stopping criterion?

2. Concerning the EMA parameter: what does α=0.8  as an optimal parameter mean in terms of the effective temporal window or the balance between recent and historical information? Is this "memory buffering" optimal value dataset-dependent?

3.  Computational cost and error propagation: How do the computational cost and inference time of AGA compare to baselines? 

4. Initialisation and relationship to avt: Two points of clarification: (a) At the start of a sequence, how are queries and keys initialised before the queue is filled? (b) A more detailed architectural comparison clarifying the exact relationship between AGA and the AVT baseline would be helpful.

5. When does AGA perform poorly? Are there specific types of actions or scenarios where the approach struggles?

6. What are the theoretical conditions under which action-guided attention should be more effective?

7. How does this choice relate to information-theoretic principles or representational learning theory?

### Soundness
3

### Presentation
2

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
The paper proposes Action-guided attention (AGA) for addressing the task of video action anticipation. Instead of relying purely on visual features, AGA considers a transformer-based architecture that makes use of the estimated action labels' sequence. Specifically, queries are computed based on the exponential moving average of all previous actions, while keys are constructed based on the estimated action labels within a temporal window $ S $. Values are based on the frame embeddings computed based on a frozen frame-based backbone network. The paper also proposes an adaptive gating mechanism that adaptively fuses past and current evidence.

### Strengths
- The idea of using a transformer-based architecture that considers the sequence of the observed action labels, in combination with the visual features, is interesting and reasonable. Surprisingly, except maybe from [R1], which follows a similar approach for long-term action anticipation, to my knowledge this direction has not received much attention yet.
- The adaptive gating mechanism is also a reasonable addition, considering that the duration of the actions varies significantly and that the proposed method does not explicitly handle the estimation of the action duration.
- The proposed method shows improved performance regarding several strong baseline methods on three widely used egocentric action anticipation datasets. It would be interesting to discuss also the suitability of the proposed method to third-person and/or 3D (Mocap) action anticipation tasks.

[R1] Gong et al., "Future transformer for long-term action anticipation", CVPR 2022.

### Weaknesses
- There are some important clarity issues that make the comprehension of some crucial methodological aspects challenging. Specifically, there is some confusion between the use of the terms "timesteps" and "frames", as two separate notations are used ($ i $ and $ t $). The ambiguity is also because these two indices seem to refer to different quantities, i.e. $ i $ as frame sequence index and $ t $ as timestep (e.g., in seconds similarly to $\tau_{\alpha}$). 
- The text does not explicitly discuss how anticipation is performed. The ambiguity of how the anticipation is performed is closely related to the issue discussed in the previous point. For instance, in case $ t $ refers to seconds, it is not clear how the embeddings $ e_t $ are obtained from *multiple* frames that correspond to the interval between $ t-1 $ and $ t $. On the other hands, if $ t $ also refers to the frame sequence index (most probably), it is not clear how anticipation is performed as $\tau_{\alpha}$ corresponds to *multiple* steps ahead, which is not explicitly discussed in the text.
- Some additional experimental analysis would be helpful. This includes the length of the queue size $S$. It would also be interesting to show how the proposed method's performance changes when the ground-truth action labels are used for anticipation.

Minor comments
--------------
- It would be helpful to include a list of contributions in the introduction.
- L.319-321: "Nevertheless, empirical results show that AGA maintains strong performance, even under constrained annotation availability and sparse supervision.". It would be interesting to provide more details.

### Questions
- Can the authors provide clarifications regarding the timesteps/sequence notation and the way anticipation is achieved?
- Have the authors considered the use of AGA for third-person and/or 3D (Mocap) action anticipation tasks?
- How is the performance influenced if ground-truth action labels are used?

### Soundness
2

### Presentation
3

### Contribution
3
