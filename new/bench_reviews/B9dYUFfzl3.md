Now let me search for calibration papers to properly score this work.Now I have sufficient context to write the final review. Let me synthesize all the inputs carefully.

---

## Summary

VADER (Video Diffusion Alignment via rEward gRadients) proposes fine-tuning pre-trained video diffusion models by directly backpropagating gradients from differentiable reward models through the denoising process. The core claim is that as generation dimensionality increases from image to video, reward-gradient methods become increasingly superior to policy-gradient approaches (DDPO, DPO) in sample and compute efficiency. The authors apply the method across four video diffusion models (VideoCrafter, Open-Sora, ModelScope, Stable Video Diffusion) using diverse reward functions (aesthetics, text-image alignment, object detection, action classification, and self-supervised video coherence), with engineering tricks (LoRA, gradient checkpointing, truncated backprop with K=1, frame subsampling) to fit training on modest hardware.

---

## Strengths

- **Compelling scaling motivation with empirical support.** The core claim that reward gradients become increasingly advantageous over policy gradients as video resolution scales is both well-motivated and supported by Figure 3, which shows a widening reward gap from 2× to 64× resolution. This represents the paper's most genuinely novel contribution relative to AlignProp/DRaFT, and provides a principled argument for why the video domain specifically demands this approach.

- **Breadth across models and reward types.** The paper demonstrates results across four base video diffusion models (both T2V and I2V), and six distinct reward functions spanning image-text alignment, aesthetics, object detection, action classification, and self-supervised temporal coherence. This breadth is substantially beyond what prior work in image alignment attempted and gives the method credibility as a general recipe.

- **Practical memory efficiency recipe.** The combination of LoRA, truncated backprop (K=1), gradient checkpointing, frame subsampling, and CPU offloading addresses a real engineering barrier for video diffusion fine-tuning. The claim of feasibility on a single 16GB GPU is a meaningful practical contribution.

- **Qualitative evidence of alignment gains.** Multiple visual examples (Figures 4, 7, 8) show tangible improvements in text-video alignment—raccoons holding snowballs, foxes wearing hats—that are compelling and consistent across base models.

- **Human evaluation supporting core claim.** Table 2 shows 79%/61% human preference for VADER over ModelScope on fidelity and text alignment respectively, providing at least some independent validation beyond reward scores.

---

## Weaknesses

### Fatal
*None that fully invalidate the paper's core claims.*

### Major

- **Circular evaluation: training and evaluation use the same reward models.** This is the paper's most significant methodological weakness. All quantitative results in Table 1 and Figure 5 are evaluated using the same reward functions (Aesthetic, HPS, PickScore, VideoMAE, V-JEPA) that were used to fine-tune the models. This makes it impossible to disentangle genuine video quality improvement from reward hacking or overfitting to the metric. The enormous aesthetic score jump (4.61 → 7.31) especially warrants skepticism without independent evaluation. The paper's conclusion that "VADER achieves much higher performance" is only established relative to the exact training metrics, not against orthogonal quality measures. No FVD, VBench, temporal-consistency scores, or held-out reward models are used for evaluation. Human evaluation (Table 2) partially alleviates this concern but is too narrow in scope (single model, single reward type, 100 responses) to validate the broad multi-task claims.

- **Limited algorithmic novelty — direct extension of AlignProp/DRaFT to video.** Algorithm 1 is essentially identical to AlignProp (Prabhudesai et al., 2023) and DRaFT (Clark et al., 2023). The engineering contributions (LoRA, gradient checkpointing, truncated backprop, frame subsampling) are standard optimization tricks rather than novel algorithmic ideas, as the paper itself acknowledges. The video domain adds real complexity, and the scaling argument is a genuine insight, but the core methodology transfers directly from image alignment with no new theoretical development. This is a meaningful systems contribution but falls short of an algorithmic one.

- **Lack of standard video quality metrics.** The paper does not report FVD, temporal consistency scores, optical flow metrics, or VBench scores that are standard in the video generation evaluation community. Since VADER optimizes per-frame image rewards (Eq. 7), the paper explicitly acknowledges this "could potentially result in a collapse, where the predicted images are the exact same or temporally incoherent" — but dismisses this with "we don't find this to happen empirically" without any rigorous metric. This absence significantly weakens claims about video-specific quality.

### Minor

- **No ablation on truncated backpropagation hyperparameter K.** K is central to both the method's efficiency and its correctness; the paper uses K=1 throughout but provides no ablation. This is particularly notable because K=1 reduces the method to single-step gradient updates similar to ReFL/DRaFT-LV, weakening the theoretical motivation for multi-step backpropagation highlighted in Figure 2 and Algorithm 1. Without this ablation, practitioners cannot choose K for new settings.

- **Scaling claim in Figure 3 lacks critical experimental details.** The paper does not specify which base model, reward function, training configuration, or exact definition of "resolution" (spatial vs. temporal vs. both) is used in Figure 3. The experiment is after a fixed 100 steps of optimization with the same reward used for evaluation (circular). Without these details and controls, the "scaling gap" claim — the paper's central novel insight — cannot be independently assessed.

- **Object removal behavior exhibits reward hacking artifacts.** Figure 6 shows that "removing books" results in the model replacing them with small animals across all three examples. The paper caption acknowledges this ("VADER effectively removes book and replaces it with some other object") but provides no analysis of whether this is desirable, why it occurs systematically, or how severe the semantic disruption is. This is a visible failure mode that goes unaddressed.

- **Human evaluation scope is too narrow to support broad claims.** Table 2 evaluates only one base model (ModelScope), one reward type (HPS), with 100 responses total, with no reported inter-rater agreement or prompt diversity information. The paper's overall rhetoric significantly overextends what this single study can support.

- **Generalization evidence is weak.** Table 1's train/test split operates at the prompt level, but reward models evaluate image-level features. Small differences between train and test scores (7.31 vs 7.12 aesthetic) could reflect insensitivity of the reward model to the prompt distribution shift rather than genuine semantic generalization. No variance estimates or significance tests are reported.

### Trivial

- The 16GB VRAM claim is mentioned as supported by the codebase but all actual experiments use 2 × A6000 GPUs (48GB). Specific configuration details for the 16GB setting are absent.

---

## Nice-to-Haves

- **Ablation on K** (truncated backprop depth): show how reward, video quality, and memory evolve with K∈{1, 5, 10, 25, T}.
- **Evaluation on held-out reward models**: train with HPS, evaluate with PickScore (and vice versa) to demonstrate genuine alignment rather than reward overfitting.
- **Standard metrics on all experiments**: FVD or VBench scores would substantially strengthen the paper's credibility in the video generation community.
- **Failure case analysis**: show examples where VADER produces artifacts, over-saturated frames, or collapsed motion, and discuss at what optimization depth they emerge.
- **Comparison with InstructVideo** (mentioned as concurrent work): at minimum a qualitative discussion of strengths and weaknesses relative to this closest video-specific baseline.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[Harsh Critic, Point 2 — DDPO/DPO comparison "structurally biased" by not giving baselines gradient access]**: Removed. The paper's entire scientific hypothesis is that reward-gradient access is superior to policy-gradient/scalar-feedback approaches. The comparison is deliberately set up to test that hypothesis, not to find the best version of DDPO. The asymmetry favors the author's method by design, which is precisely what the paper is trying to demonstrate. This is not an unfair comparison; it is the comparison that defines the contribution. Criticizing the authors for not giving baselines the same gradient access would eliminate the entire research question.

- **[Harsh Critic, Point 3 — "resolution" ambiguity and circular scaling claim as a "fatal" issue]**: Downgraded to minor. The lack of experimental detail around Figure 3 is a real concern (kept in Minor), but calling the entire scaling narrative "circular" overstates the case. The scaling plot does show a meaningful qualitative trend, even if the controls are insufficient for a strong quantitative claim.

- **[Multiple reviewers — 16GB VRAM reproducibility nitpick]**: Removed per hard rules on reproducibility nitpicks. The codebase claim is plausible; verifying it is not required for paper acceptance.

- **[Harsh Critic, Point 5 — "long-horizon temporal consistency: no quantitative evaluation" as FATAL]**: Downgraded to Minor. The V-JEPA experiment (Fig. 9) is one of several tasks explored and is clearly presented as a demonstration rather than a primary contribution. Calling this absence fatal is out of proportion.

- **[Harsh Critic — DPO/DDPO hyperparameter tuning details, batch size matching, and GPU-hour methodology]**: Removed as reproducibility nitpicks that would require impractical disclosure levels.

---

## Novel Insights

The most genuinely novel observation in this paper is the dimensionality-scaling argument: as video generation increases in spatial and temporal resolution, the information advantage of dense reward gradients over scalar policy signals grows substantially, as visualized in Figure 3. This intuition — that reward gradient richness scales linearly with output dimensionality while policy gradient feedback remains a single scalar regardless of video length — is a concrete and actionable insight that motivates an entire class of future work in video alignment. The empirical demonstration across four architectures and six reward types, while methodologically imperfect, provides real evidence that this principle generalizes beyond a single model. The practical engineering recipe (K=1 backprop + LoRA + frame subsampling + CPU offloading) for making this feasible on moderate hardware is also a useful community contribution that was not previously codified for video diffusion.

---

## Suggestions

1. **Add evaluation with at least one held-out reward model** (e.g., train with Aesthetic, evaluate with an independent CLIP-based scorer or vice versa). This single change would substantially address the circular evaluation concern.
2. **Report FVD or VBench scores alongside reward scores** in Table 1 to demonstrate that alignment gains do not come at the cost of overall video quality.
3. **Provide a dedicated ablation on K** showing reward vs. quality vs. memory tradeoffs as K varies from 1 to T. This would turn a current gap into a strength.
4. **Clarify Figure 3 fully**: specify the exact base model, reward function, definition of "resolution" multiplier, number of training runs, and whether the reward used in the plot is the same one used for training. If it is, use an independent metric on the y-axis.
5. **Acknowledge limitations explicitly** in the conclusion: reliance on differentiable white-box reward access, the K=1 reduction to single-step methods, and the risk of reward hacking with very large score jumps.
6. **Expand the human study** to at least two reward types and two base models, even at the same 100-response scale.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Topic | Decision | Scores |
|---|---|---|---|
| AlignProp (Vaf4sIrRUC) | Reward backprop through image diffusion (essentially VADER for images) | Reject | 3, 5, 5, 6 |
| DDPO (YCWjhGrJFD) | Policy gradient for image diffusion alignment (VADER's baseline) | Accept Poster | 6, 6, 8, 5 |
| T2V-Turbo-v2 (BZwXMqu4zG) | Video model post-training with reward models, comprehensive VBench eval | Accept Poster | 6, 6, 6, 6, 6 |
| Test-time Alignment (vi3DjUhFVm) | Diffusion alignment avoiding over-optimization with FVD/diversity eval | Accept Spotlight | 8, 8, 8, 5 |

**Reasoning:**

VADER is most directly comparable to AlignProp: same core algorithm, same engineering tricks (LoRA, gradient checkpointing, truncated backprop), extended from images to video. AlignProp was rejected (avg ≈ 4.75) primarily for limited novelty and weak efficiency arguments. VADER is clearly above AlignProp because: (1) video is genuinely harder with real memory challenges addressed; (2) the scaling argument (Figure 3) is a new insight; (3) evaluation spans 4 architectures vs. 1.

However, VADER falls below T2V-Turbo-v2 (accepted, avg 6.0) primarily because: (1) T2V-Turbo-v2 uses VBench (standard video quality metrics) while VADER relies entirely on training rewards for evaluation; (2) T2V-Turbo-v2 has comprehensive ablations while VADER lacks ablation on K.

VADER sits between AlignProp (rejected) and T2V-Turbo-v2 (accepted), meaningfully closer to the acceptance boundary. The circular evaluation concern is real and substantial, but the breadth of experiments and the scaling insight genuinely add value to the community. A limited but real human evaluation (Table 2) and compelling qualitative results prevent a reject.

**Overall assessment:** The paper is competent, useful, and timely, but the evaluation framework conflates "improved the reward model's score" with "improved actual video quality," and the algorithmic contribution is largely an engineering adaptation of AlignProp. These are fixable issues rather than fatal ones, but in their current form they keep the paper below the acceptance bar.

**Originality:** Moderate-low — the core method is a direct adaptation of prior image alignment work.
**Importance of research question:** High — video alignment is an open and important problem.
**Claims well-supported:** Partially — quantitative claims are undermined by circular metrics; qualitative and human evidence is promising but narrow.
**Soundness of experiments:** Fair — broad but methodologically imperfect; lack of standard metrics is a notable gap.
**Clarity of writing:** Good — well-structured and clearly explained.
**Value to community:** Moderate — the practical recipe and multi-model demonstration have genuine utility.

**Final Score: 5.0**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>