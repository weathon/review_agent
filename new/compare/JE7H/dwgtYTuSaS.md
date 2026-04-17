---
job_id: 94e24d70-d034-4b68-8cca-30faa2466ac8
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: dwgtYTuSaS.pdf
paper: Continuous Online Action Detection from Egocentric Videos
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a continuous online learning formulation for egocentric online action detection and introduces an egocentric OAD dataset, which clearly falls within representation learning, lifelong / continual learning, and vision applications.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Method / Approach, Experiments, Results, Conclusion) are present and written in English. The method is technically reasonable, there are concrete experiments on two benchmarks with quantitative and qualitative results, and no fatal flaws such as test leakage are evident. While there are notable weaknesses in novelty, baselines, and experimental design, they do not reach the level of an automatic desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any explicit or implicit prompt-injection attempts, hidden instructions to reviewers/LLMs, or other manipulative content in the provided main paper text.

---

# Expected Review Outcome:

## Summary

The paper introduces *Continuous Online Action Detection* (COAD), a formulation where an online action detection model not only predicts actions causally but also updates its parameters continuously from a single pass over a video stream without replay or storage. The method adapts an RNN-based OAD architecture, adds state continuity, an orthogonal-gradient update rule, and a non-uniform loss strategy, and trains it under streaming constraints. The authors also curate Ego-OAD, an egocentric OAD benchmark derived from Ego4D Moment Queries, and report experimental results on Ego-OAD and EPIC-KITCHENS that show improved adaptation and some gains in out-of-stream generalization.

## Strengths

1. **Clear and well-scoped problem formulation (COAD) with connection to real deployment.**  
   The paper articulates a concrete extension of standard OAD to the continuous-learning setting (Section 4, Pages 4–6), with explicit constraints: single-pass, causal windows, no replay, hidden-state continuity, and on-the-fly parameter updates. This setting is well motivated by on-device training on egocentric wearables and fits naturally with the emerging “learning from one continuous video stream” line of work.

2. **Non-trivial egocentric OAD dataset construction and analysis.**  
   Ego-OAD, derived from Ego4D MQ, is a substantial dataset (263h, 22,991 instances, 87 classes) with multi-label, temporally grounded annotations (Section 3, Pages 3–4). The authors detail label aggregation from multiple annotation passes and semantic grouping of noisy fine-grained labels (Tables 6, Pages 14–16), and they quantify overlap and class imbalance.  
   - **Figure 1** illustrates multi-label overlapping actions on sample clips, which clarifies the challenging nature of the stream and motivates why long-term modeling and continuous adaptation might be beneficial.  
   - **Figures 7 and 8** (Pages 12–13) show class-instance distribution and average duration by class, highlighting severe imbalance and varied temporal scales, which are important for contextualizing the performance numbers.

3. **Empirical evidence that streaming updates can improve out-of-stream generalization under realistic constraints.**  
   On Ego-OAD, **Table 1** (Page 7) shows that COAD improves the out-of-stream Top-5 Recall relative to “Pretrained Only” by +6.9 (ego-pretrained backbone) and +6.5 (exo-pretrained backbone), outperforming the naive streaming baseline (w/o COAD) by a solid margin in Recall, while maintaining similar or slightly better mAP. This supports the claim that carefully regularized streaming updates can help generalization beyond the specific in-stream videos.

4. **Reasonably thorough ablation of COAD components and hyperparameters.**  
   The component analysis in **Table 3** (Page 8) systematically toggles state continuity, orthogonal gradients, and non-uniform loss. It reveals non-trivial interactions, e.g., non-uniform loss increases out-of-stream mAP and recall when combined with the other components, and orthogonal gradients primarily boost out-of-stream recall.  
   - **Figure 3** (Page 9) further analyzes the in-stream vs out-of-stream trade-off in terms of stride and learning rate, showing qualitative Pareto curves for mAP and Top-5 Recall; this is useful for understanding the overfitting/generalization balance under streaming updates.

5. **Temporal evolution analysis under streaming training.**  
   **Figure 4** (Page 9) shows the evolution of out-of-stream mAP and Top-5 Recall as more in-stream data is consumed. COAD trajectories steadily improve and approach the IID-training upper bound, while ablated variants lag significantly. This is a valuable sanity check that streaming training is actually learning as more data arrives and does not catastrophically drift.

6. **Qualitative analysis showing temporal coherence improvements.**  
   **Figure 5** (Page 9) and **Figure 6** (Page 12) present per-frame label timelines comparing COAD vs w/o COAD on out-of-stream and in-stream clips. The COAD predictions appear less noisy, with more stable segments that better align with ground truth, which supports the claim that state continuity and orthogonal gradients help maintain coherent predictions on long sequences.

7. **Clarity and readability.**  
   The high-level narrative is easy to follow. The contrast between standard offline OAD training and COAD is effectively conveyed in **Figure 2** (Page 5), which visually shows shuffled vs continuous windows, gradients, and internal state resetting vs continuity. The mathematical description of the setting (Equations in Sections 4.1–4.5) is standard but correct, and the constraints of the streaming regime are clearly stated.

## Weaknesses

1. **Limited methodological novelty; COAD mainly recombines existing ingredients with modest adaptation.**  
   The core components of COAD are: (i) single-pass causal training with state continuity, (ii) orthogonal gradient projection from Han et al. (2025), and (iii) a non-uniform loss that only supervises the last step in the window from An et al. (2023). The paper’s main conceptual move is to apply these techniques to an OAD RNN under a streaming protocol, but there is little fundamentally new algorithmic development.  
   - The orthogonal gradient formula in Section 4.5,  
     \[
     g_t^\perp = g_t - \frac{\langle g_t, g_{t-1}\rangle}{\lVert g_{t-1}\rVert^2}g_{t-1},
     \]
     is taken directly from prior work; there is no analysis specific to OAD (e.g., how this projection affects class-imbalanced gradients, sparse labels, or delayed supervision).  
   - State continuity is simply “do not reset the GRU state” between windows, and the non-uniform loss is exactly the existing last-step-only loss.  
   The paper positions COAD as a “new task formulation” but methodologically the solution feels like a straightforward adaptation of existing streaming-learning and OAD tricks. For an ICLR main track paper, I would expect either deeper theoretical insights into why these components are particularly suitable for streaming OAD, or a more principled design that goes beyond plugging in orthogonal gradients.

2. **Very weak comparison to prior methods; almost no baselines beyond internal variants.**  
   The only quantitative comparisons are “Pretrained Only” and “w/o COAD” (Tables 1 and 2). There is no comparison against:  
   - Strong OAD baselines such as LSTR, OADTR, TeSTra, MA-Transformer, or even MiniROAD itself in its standard training form, adapted to the same features.  
   - Other continuous/streaming learning approaches beyond orthogonal gradients, e.g., rehearsal-based baselines (even with small buffers), elastic-weight consolidation, or simple SGD with tuned learning rates and weight decay.  
   This makes it hard to situate COAD’s gains. For instance, **Table 1** shows ego-pretrained COAD achieving 26.0 mAP / 76.0 Top-5 Recall out-of-stream, but we do not know whether a stronger offline baseline with a transformer temporal head, or an RNN trained with slightly better regularization, could match or surpass this without any of the COAD machinery. Similarly, **Table 4** shows a large gap between TSN and TimeSformer backbones, but the main COAD experiments do not systematically explore what happens if a more powerful temporal head is used instead of the GRU.

3. **Dataset and benchmark are underutilized; lack of cross-model evaluation on Ego-OAD.**  
   Ego-OAD is potentially valuable, but the experimental section uses it primarily to demonstrate COAD’s internal ablations. There is no evaluation of a diverse set of existing OAD architectures on Ego-OAD (RNN-based, transformer-based, multi-stream models, etc.), so the community does not get a clear sense of where this dataset sits in difficulty or how architectural trade-offs look. **Table 4** only compares two backbone feature types with a single GRU head; this is more an ablation than a benchmark.  
   Without a richer suite of baselines, Ego-OAD feels more like a private testbed than a carefully established benchmark. This weakens the claimed contribution of “providing a diverse and realistic evaluation platform for future research” (Page 2).

4. **Evaluation design around EPIC-KITCHENS is confusing and partially negative for COAD.**  
   In **Table 2** (Page 7), several numbers are puzzling and in places unfavorable to COAD:  
   - For verbs, in-stream mAP of COAD equals that of Pretrained Only (29.0), and COAD does not seem to improve over simple pretraining in most in-stream metrics.  
   - For nouns, both w/o COAD and COAD sometimes reduce in-stream mAP compared to Pretrained Only (e.g., noun in-stream mAP drops from 3.8 to 3.3 / 3.9). The fact that noun mAP for Pretrained Only has the pattern “31.4 / 3.8” (out / in) versus “37.1 / 3.9” for COAD is itself suspicious: why is in-stream noun mAP so low relative to out-of-stream? This suggests a mismatch in how the metric is computed across splits or some issue in the annotation conversion to the OAD format.  
   - For actions, in-stream Top-5 Recall for COAD (20.5) is lower than Pretrained Only (22.9), yet out-of-stream Top-5 Recall is identical (21.9).  
   The text on Page 7 acknowledges some difficulties (“both COAD and w/o COAD struggle to adapt effectively”), but does not analyze these odd metric behaviors or attempt troubleshooting. This undermines the generality of the method: COAD seems quite sensitive to dataset characteristics, and the paper stops at attributing it to “fine-grained nature” without deeper investigation.

5. **Inadequate discussion of supervision assumptions and label feasibility under the claimed deployment scenario.**  
   The COAD training protocol (Section 4.5) and experiments (Section 5.1) assume access to ground-truth frame-level labels for in-stream videos at training time, even if the loss is only computed at the last step of each window. However, the application narrative is on-device learning on wearable devices in the wild, where manual annotation of streaming egocentric video is unrealistic.  
   - The paper briefly mentions that non-uniform loss “improved label efficiency” and “allows training with sparse instead of dense annotations” (Page 6), but in practice the experiments still use fully annotated streams (Ego-OAD labels).  
   - There is no experiment with partially labeled streams, delayed or noisy supervision, self-supervised or weakly-supervised signals, nor any discussion of how COAD would operate with only occasional user feedback.  
   As a result, while COAD is positioned as enabling on-the-fly learning on-device, the learning supervision it requires is far from realistic for that scenario.

6. **Mathematical and algorithmic specification of COAD is under-detailed in key implementation aspects.**  
   Several important details are either missing or only vaguely described, which makes reproducing the exact COAD procedure non-trivial:  
   - **Orthogonal gradients:** Section 4.5 only defines the projection formula for two consecutive gradients, \(g_t^\perp\). It does not specify whether gradients are accumulated over multiple mini-steps within a window, whether gradients are computed on the last frame only or over the whole window before projection, or how the method handles \(\|g_{t-1}\|^2 \approx 0\) (numerical stabilization, thresholding, or skipping projection). Since the paper claims that orthogonal gradients are important for performance (Table 3), these missing details are material.  
   - **Non-uniform loss:** The paper states “computes the loss only at the final step of each window” (Page 5), but in the streaming setting with overlapping windows of stride 16, it is unclear whether the same frame’s label contributes multiple times as it becomes the last frame of successive windows, and how this interacts with state continuity (since backprop is truncated at the window boundary). A more explicit loss formulation like
     \[
     \mathcal{L} = \sum_t w_t \,\ell(\hat{y}_t, y_t),\quad \text{with } w_t = \mathbf{1}[t \equiv 0 \pmod{\tau}]
     \]
     would greatly clarify the training dynamics.  
   - **Windowed BPTT & state carry-over:** It is stated that hidden state is continuous, but the exact backprop-through-time truncation strategy is not given. Does each gradient step unroll over \(\tau\) steps, starting from the previous hidden state treated as a constant, or is there some gradient flow across window boundaries? These decisions can significantly influence performance and are essential for others attempting to implement COAD.

7. **Limited exploration of model capacity vs streaming regime; RNN-only temporal head is restrictive.**  
   The paper argues that RNNs are attractive for deployment on resource-constrained devices (Page 2), which is fair, but then all major results use an RNN head with high-end TimeSformer or TSN features that are not realistically deployable on typical smart glasses. Conversely, there is no exploration of streaming training with transformer-based temporal heads that might better exploit long-range context in the offline setting.  
   - **Table 4** shows that TimeSformer (clip-level) features dramatically improve offline OAD performance relative to TSN. However, the COAD experiments do not include a variant where the temporal head is a transformer, making it unclear whether the main limitation in the streaming setting is the head architecture or the training regime.  
   - Given that **Figure 3** indicates a strong sensitivity to learning rate and stride, it would be useful to see whether better architectures can reduce this sensitivity or attain better Pareto frontiers between in-stream and out-of-stream performance.

8. **COAD sometimes hurts adaptation and its trade-offs are not deeply analyzed.**  
   In **Table 1**, for Ego-OAD in the ego-pretrained case, COAD slightly *decreases* in-stream mAP from 39.0 (w/o COAD) to 36.8, even while improving out-of-stream recall. The paper briefly frames this as a trade-off but does not analyze why, e.g., whether orthogonal gradients slow adaptation too much, whether the non-uniform loss regularizes away useful fine-grained patterns, or whether the hyperparameters were tuned fairly for both methods. **Figure 3** gestures at a generic trade-off, but the COAD-vs-baseline trade-off at a given hyperparameter setting is not carefully dissected. This undercuts the claim that COAD “allows models to specialize to individual users’ environments while retaining robust generalization” (Page 2); in some cases, it reduces specialization.

9. **Some inconsistencies and minor clarity issues.**  
   - There is a naming inconsistency: the method is repeatedly called “COAD” but Section 4 starts with “Continuous OAD (CODA)” (Page 4), which looks like a typo and is confusing.  
   - The description of EPIC-KITCHENS noun and action results in **Table 2** contains suspicious patterns (out-stream much better than in-stream mAP for nouns) with no explanation.  
   - The dataset split description for EPIC-KITCHENS (Page 6) is brief and does not clarify how noun/verb/action labels were mapped to frame-level multi-label annotations under the OAD regime.

Overall, the paper presents an interesting application of streaming training ideas to OAD and provides a potentially useful egocentric dataset, but method novelty is limited and the empirical evaluation is too narrow and somewhat inconsistent for ICLR.

## Potentially Missing Related Work

Below, I list related works that appear directly relevant yet are not cited or discussed in the submission:

1. **Gao et al., “Contextualizing Temporal Distinctions: A Weakly Supervised Approach to Action Localization”, 2022.**  
   This work addresses temporal action localization with weak supervision, which is closely related to OAD/COAD in that it learns temporal boundaries and class assignments under limited labels. It should be discussed in Section 2 (Online Action Detection Models / Datasets) and in the discussion of label efficiency and weak supervision in Section 4.5, since COAD claims improved label efficiency but still assumes dense supervision.

2. **Shvetsova et al., “Multi-Modal Fusion Transformer for Video Retrieval”, 2022.**  
   While primarily about video retrieval, this paper introduces a multi-modal transformer architecture fusing video, audio, and text. It is relevant because Ego4D MQ is inherently multi-modal (natural language queries + video), and the paper’s current formulation drops the language aspect. A short discussion in Section 2 and/or Section 3 would help clarify why the authors restrict to vision-only OAD and whether multi-modal cues could further improve COAD.

3. **Singh et al., “A Multi-Stream Bi-Directional Recurrent Neural Network for Fine-Grained Action Detection”, 2016.**  
   This work targets fine-grained temporal action detection using multi-stream bi-directional RNNs, and is an early example of RNN-based online-like detection. It should be cited in Section 2 when motivating RNNs for OAD and when justifying the choice of a lightweight GRU-based temporal head.

4. **Yeung et al., “End-to-End Learning of Action Detection from Frame Glimpses in Videos”, 2016.**  
   This paper introduces an end-to-end recurrent attention model for action detection based on frame glimpses. It is directly relevant to online and causal-approximate detection and should be discussed in Section 2 to give a fuller picture of early online-like detection architectures.

5. **Yuan et al., “Temporal Action Localization with Pyramid of Score Distribution Features”, 2016.**  
   This method for temporal action localization focuses on score distributions over time, which connects to the temporal aggregation and multi-label per-frame annotation in Ego-OAD. It would be appropriate to mention it in Section 2 and possibly compare its offline localization protocol with the proposed OAD/COAD setting.

6. **Shou et al., “Temporal Action Localization in Untrimmed Videos via Multi-Stage CNNs”, 2016.**  
   A classic paper on temporal localization in untrimmed videos. Even though it is mostly offline, it targets continuous untrimmed streams and is part of the foundational literature that should be placed alongside THUMOS and other benchmarks in Section 2.

7. **Richard and Gall, “Temporal Action Detection Using a Statistical Language Model”, 2016.**  
   This work models temporal structure via a statistical language model for actions. It is highly relevant to temporal reasoning in detection tasks and could be discussed in Section 2, particularly when the paper talks about modeling long-range temporal dependencies and RNN limitations.

Incorporating and contrasting with these works would strengthen the positioning of COAD in the broader temporal action localization / detection literature, especially regarding weak supervision, multi-modal modeling, and early RNN-based detectors.

## Questions

1. **Supervision in realistic deployment:**  
   How do the authors envision obtaining ground-truth labels for the in-stream videos in a real wearable-device deployment? Could COAD operate with only occasional user feedback (e.g., sparse labels per several minutes), and if so, what changes are needed in the loss or training objective? An experiment with synthetic sparse labels (e.g., labels on only 5–10% of windows) would greatly clarify this.

2. **Clarification on BPTT and loss computation across windows:**  
   Please specify the exact unrolling and backprop strategy: for a window \((z_{t-\tau+1},\ldots,z_t)\), is the loss \(\ell(\hat{y}_t, y_t)\) backpropagated through all \(\tau\) steps, with the hidden state \(h_{t-\tau}\) treated as a constant? How often does a given frame’s label contribute to the loss under stride 16 and overlapping windows? A more formal description (possibly a small algorithm box) would be helpful.

3. **Orthogonal gradient details and stability:**  
   How do you handle \(\|g_{t-1}\| \approx 0\) in the projection formula? Is there a threshold below which you skip the projection? Also, do you normalize gradients before projection or operate on raw gradients? Please elaborate, since **Table 3** attributes non-trivial gains to this component.

4. **EPIC-KITCHENS metrics and anomalies:**  
   Can you clarify how verb, noun, and action mAP / Top-5 Recall are computed in **Table 2**, especially why noun in-stream mAP is an order of magnitude smaller than out-of-stream mAP (e.g., 31.4/3.8 for Pretrained Only)? Is there a difference in class coverage, annotation density, or evaluation windowing between splits? If there was a bug or a non-standard evaluation protocol, clarifying this would help assess the validity of the EPIC results.

5. **Baselines vs stronger temporal heads:**  
   Why not include at least one transformer-based temporal head (e.g., a lightweight transformer or LSTR-style) trained in the same offline and streaming regimes for comparison? This would help isolate how much of the gains come from the COAD training regime vs. the relatively simple GRU head. If resource constraints motivated the RNN, can you provide FLOPs / latency numbers to support that decision?

6. **Adaptation vs. generalization trade-off:**  
   In **Table 1**, COAD sometimes reduces in-stream mAP compared to w/o COAD. Did you attempt to adjust hyperparameters (e.g., learning rate, stride, or projection strength) to recover in-stream performance while maintaining out-stream gains? Any evidence that COAD can move along a Pareto frontier, or is the observed drop intrinsic to the orthogonal-gradient and non-uniform loss design?

7. **Public release and benchmark setup for Ego-OAD:**  
   If Ego-OAD is to be a benchmark, will you release a standard train/val/test split and evaluation code? Also, can you clarify whether the current split (pretrain / in-stream / out-of-stream) is fixed or arbitrary? It would be useful to know how users should reproduce your COAD vs offline experiments.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A. The paper uses existing egocentric datasets (Ego4D, EPIC-KITCHENS) with no new sensitive data collection or high-risk deployment claims, and it does not touch on fairness, privacy, or safety beyond standard considerations.

## Soundness Rating

2: fair.  
The method is conceptually sound and equations are mostly correct, and the Ego-OAD dataset is plausibly well constructed. However, key algorithmic details (orthogonal gradient implementation, BPTT, loss scheduling) are under-specified, the EPIC-KITCHENS evaluation presents anomalies that are unexplained, and the experimental comparison set is too limited to robustly validate the central claims.

## Presentation Rating

3: good.  
The paper is generally clear and well written, with helpful figures (especially Figures 1, 2, 3, 4, 5) and tables. Some inconsistencies (COAD vs CODA naming, odd EPIC metrics, missing algorithmic details) detract from clarity but are fixable.

## Contribution Rating

2: fair.  
The contributions are a moderate extension of existing ideas (streaming training with orthogonal gradients and RNN OAD) and a curated dataset derived from Ego4D. While potentially useful, the methodological novelty is limited and the dataset is not fully established as a benchmark due to sparse baselines and analysis.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The paper targets an important and timely problem (continuous learning for egocentric OAD) and combines reasonable streaming-learning techniques with a new dataset, with some empirical evidence of benefits. However, limited methodological novelty, a narrow and weak baseline comparison, under-specified training details, and issues in the EPIC-KITCHENS evaluation prevent it from meeting the bar for ICLR in its current form.

## Reviewer Confidence

4: confident.  
I am familiar with online action detection, continual/streaming learning, and egocentric video datasets, and I carefully examined the equations, figures, and tables. Some EPIC-KITCHENS details are unclear, but they do not change my overall assessment.