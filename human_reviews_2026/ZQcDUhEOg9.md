# AdaGC: Improving Training Stability for Large Language Model Pretraining

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Loss spikes remain a persistent obstacle in large-scale language model pretraining. Empirically, such spikes can be triggered by a mixture of factors, including data outliers, hardware or transient computational faults, numerical precision issues, and hyperparameter settings. Regardless of the underlying cause, these spikes manifest as unstable optimizer updates, as abnormal gradients contaminate both first- and second-moment states. In this paper, we do not attempt to identify the precise root causes. Instead, we adopt a gradient-centric remedy and propose AdaGC, an adaptive, per-tensor gradient clipping scheme that prevents such contamination by bounding gradient norms relative to a tensor-wise EMA of their historical (clipped) values. AdaGC is optimizer-agnostic, requires negligible memory, and reduces communication costs compared to GlobalGC, particularly under hybrid parallel distributed training. We prove that Adam with AdaGC preserves the standard non-convex convergence rate. On Llama-2 7B, Mixtral 8×1B, and ERNIE 10B-A1.4B models, AdaGC robustly eliminates training instabilities, reducing the spike score to zero for all models, and improves downstream accuracy compared to GlobalGC by +1.32\%, +1.27\%, and +2.48\%, respectively. Furthermore, AdaGC composes well with Muon and Lion optimizers, consistently yielding higher average accuracy and zero spike scores. We will release our code publicly.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes AdaGC (Adaptive Gradient Clipping), a method that dynamically adjusts clipping thresholds during large-scale model pretraining based on the exponential moving average of gradient norms. The goal is to mitigate loss spikes and improve training stability without increasing computational or memory cost. AdaGC is evaluated on several Transformer architectures and compared with existing clipping methods such as GlobalGC, AGC, and Clippy. Experiments show improved stability, reduced loss spikes, and faster convergence, while maintaining comparable downstream performance.

### Strengths
The paper addresses an important instability issue in large-scale model training by introducing an adaptive gradient clipping strategy. The method is simple, architecture-agnostic, and integrates easily into existing optimizers. The empirical analysis is clear and demonstrates consistent reduction of loss spikes and improved convergence stability across several model families. The presentation is structured and easy to follow, with reasonable theoretical justification.

### Weaknesses
The ablation studies are relatively shallow; key design factors such as EMA decay rate, clipping threshold formulation, and normalization choice are not sufficiently analyzed.

The paper provides theoretical arguments but stops short of offering formal guarantees or a deeper link between the adaptive clipping dynamics and optimization convergence.

Comparison with stronger and more modern baselines (e.g., adaptive optimizers with advanced learning-rate scheduling) is missing, making it unclear how much improvement stems from the proposed mechanism itself.

### Questions
How sensitive is AdaGC to different EMA decay rates or clipping threshold update rules? A more detailed ablation could clarify whether the observed benefits are robust to hyperparameter variation.

Have the authors evaluated AdaGC against modern adaptive optimizers (e.g., Lion, Sophia) or stronger learning-rate scheduling strategies to better establish its relative advantage?

Can the authors provide more evidence that the improvement in loss-spike reduction translates into meaningful downstream gains beyond pretraining convergence speed?

Theoretical analysis currently follows standard non-convex convergence proofs. Is there a way to connect AdaGC’s adaptive clipping dynamics more explicitly to gradient stability or loss variance control?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes AdaGC, an adaptive per-tensor gradient clipping method that uses EMA-based local norms to stabilize large-scale LLM pretraining. It demonstrates consistent elimination of loss spikes and downstream accuracy gains across LLaMA-2, Mixtral, and ERNIE models.

### Strengths
* This paper addresses a widespread and costly issue—loss spikes in LLM training—using a simple, optimizer-agnostic remedy.
* The proposed method is evaluated on multiple model types and optimizers, showing consistent improvements and detailed ablation analysis.

### Weaknesses
* The paper treats gradient spikes as a black-box phenomenon, lacking diagnostic analysis (e.g., layer- or token-level gradient behavior) to substantiate the “gradient contamination” hypothesis.
* The total token counts (e.g., 36B tokens for LLaMA-2 7B) are small compared to real large-scale pretraining, and many comparisons are limited to early training stages, which raises doubts about whether AdaGC’s stability holds in real-world, trillion-token-scale training.
* he reported gains of AdaGC when applied to the Lion and Muon optimizers are minimal, suggesting limited benefit beyond AdamW-based settings.

### Questions
* Figure 2 only presents the warm-up phase; could u provide visualizations covering the entire training process for a more comprehensive comparison?
* The performance improvement of AGC on LLaMA-2 1.3B appears marginal. Could the authors include or discuss its corresponding training dynamics to clarify this behavior?
* In Figure 3(b), AdamW + GlobalGC (beta2=0.95) presents better training stability than AdamW + GlobalGC (beta2=0.999), yet achieves worse validation results. Do the authors provide any in-depth analysis on how training stability correlates with final model performance?

### Soundness
3

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
4

### Summary
This paper introduces AdaGC (Adaptive Gradient Clipping), a simple yet effective method to improve training stability in large language model pretraining. By monitoring gradient norms per tensor and dynamically adjusting clipping thresholds using an exponential moving average (EMA), AdaGC prevents outlier gradients from contaminating optimizer states and triggering loss spikes. Compatible with optimizers like AdamW, Lion, and Muon, AdaGC completely eliminates loss spikes and improves downstream task performance across models such as Llama-2, Mixtral, and ERNIE. With minimal memory overhead, reduced communication cost, and theoretical convergence guarantees, AdaGC offers a practical and robust solution for stable large-scale model training.

### Strengths
1. Identifies that “abnormal gradients polluting optimizer states” is the common final path to loss spikes, giving a clean, optimizer-agnostic intervention point.

2. Method simplicity: Per-tensor EMA of gradient norms plus relative clipping; implementation needs ≈4 bytes/tensor and <10 lines of code change, yet completely suppresses spikes on 1.3 B–10 B models.

3. Empirical coverage: Extensive experiments on dense (Llama-2) and MoE (Mixtral, ERNIE) architectures; consistent zero spike scores and +1–2.5 % downstream gains over GlobalGC.

4. Theoretical backing: Proves convergence for Adam+AdaGC under standard non-convex assumptions; communication cost is lower than global clipping in hybrid parallelism.

5. Compatibility: Works out-of-the-box with AdamW, Lion, Muon; no new hyper-parameters that require heavy tuning

### Weaknesses
1. The reliance on the exponential moving average (EMA) of per-tensor gradient norms introduces an inherent lag in adapting to sudden gradient spikes. Since the clipping threshold is updated based on historical statistics, AdaGC may fail to respond promptly to abrupt increases in gradient magnitudes. Consequently, outlier gradients could still enter the optimizer state before the EMA sufficiently adjusts, potentially undermining the intended stabilizing effect.

2. While the authors claim that AdaGC is relatively robust to hyperparameter choices, the experimental results indicate that the relative clipping threshold λ_rel still has a noticeable impact on downstream performance. This suggests that λ_rel may require task-specific tuning, especially under different model architectures or data distributions. As such, AdaGC does not fully eliminate the need for manual hyperparameter optimization, which somewhat contradicts the goal of being a fully adaptive method.

3. AdaGC relies on global gradient clipping (GlobalGC) during the early training phase, which raises concerns about its standalone stability. This hybrid strategy implies that AdaGC alone may not be sufficiently reliable at the beginning of training, when gradient statistics are highly volatile.

### Questions
1. Do the authors have any plans to evaluate AdaGC on models larger than 10B parameters or for training runs that exceed the current 36K-step limit, so as to verify its scalability and stability under more demanding conditions?

2. Have the authors attempted to train models with AdaGC while completely removing the initial GlobalGC warmup phase, and could they clarify why AdaGC appears to be unstable during the early stages of training?

3. Is the optimal value of the relative clipping threshold λ_rel sensitive to different tasks or data distributions, and if so, how should practitioners select this hyper-parameter in a principled way?

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
3

### Summary
The paper proposes AdaGC, a tensor-level adaptive gradient clipping method that uses an EMA-based threshold. It prevents optimizer contamination by abnormal gradients and eliminates loss spikes during large-scale training. The authors first confirm that such spikes can arise from multiple sources, and instead of analyzing each cause in depth, focus on a unified solution that eliminates spikes across diverse settings, showing that AdaGC consistently removes them on various models.

### Strengths
The paper presents strong empirical evidence that the proposed AdaGC method effectively mitigates training loss spikes caused by abnormal gradients. The experimental evaluation is extensive, covering multiple architectures and modalities, including dense models (Llama-2), Mixture-of-Experts models (Mixtral and ERNIE), and a vision-language model (CLIP). Both pretraining and downstream performance are assessed, demonstrating that AdaGC consistently stabilizes training and slightly improves accuracy. In addition, the paper shows that AdaGC offers practical advantages in large-scale multi-GPU settings, reducing communication overhead during distributed training compared to GlobalGC. The experimental setup is clear and easy to follow, and the comparison with a wide range of similar approaches is covered.

### Weaknesses
The paper provides solid analysis, but some explanations and comparisons are not entirely clear. The main points are listed here, with more details in the questions section.

- **W1:** Section 5.4 **Optimizer Compatibility: Muon and Lion** feels weak in its current form. Since the experiments do not demonstrate that spikes occur, it is unclear how this section supports the main goal of the paper, which is to eliminate loss spikes.  
- **W2:** The **ablation study about adaptivity and locality** appears incomplete. It tests GlobalGC (no adaptivity, no locality), Global AdaGC (adaptive, not local), and AdaGC (adaptive and local), but omits the configuration that is local but not adaptive - i.e., scaling each tensor’s gradient with a fixed threshold.  
- **W3:** **Hyperparameter sensitivity** could be expanded by examining sensitivity to $T_{\text{start}}$ and by covering a wider range of $\lambda_{\text{rel}}$, assessing whether it can be omitted if it has little effect.  
- **W4:** The comparison with other methods could be clearer. For instance, SPAM is compared only on downstream tasks but not in spike-score evaluation, and AGC seems to be described inconsistently with its original paper [1] (which clips gradients, not updates).  

[1] Andrew Brock, Soham De, Samuel L. Smith, Karen Simonyan. *High-Performance Large-Scale Image Recognition Without Normalization*. arXiv:2102.06171, 2021. [https://arxiv.org/abs/2102.06171](https://arxiv.org/abs/2102.06171)

### Questions
### Hyperparameters
- **Q1**: The grid search for AdaGC’s $\beta$ and $\lambda_{\text{rel}}$ looks dense (Tables 2 and 3). Why was this grid chosen? It would be helpful to see results for $\lambda_{\text{rel}} = 1$ to assess whether this parameter could be omitted.  
- **Q2**: How were hyperparameters for other clipping methods selected? Were they tuned or taken directly from prior work? (This is not entirely clear from the Appendix.)  
- **Q3**: How sensitive is the method to the number of warm-up steps $T_{\text{start}}$ where GlobalGC is applied?  

### Ablation Studies (Section 5.6)
- **Q4:** Could you include the missing variant mentioned in **W2** (local but non-adaptive) to complete the ablation study, or explain why it is not presented?
- **Q5**: In the EMA initialization study, do all alternative variants (1–3) perform direct initialization from the first step, or do some still use the GlobalGC warm-up phase for initialization?  

### Comparisons and Related Work
- **Q6**: Why is SPAM omitted from the spike-score results (**W4**)? Since it directly targets spike mitigation, including it would make the comparison more complete.  
- **Q7**: Could you clarify the description of AGC [1] in Table 1 and Section 3? From the original paper, it appears that gradients are directly clipped.  
- **Q8**: Why was LAMB (listed in Table 1) excluded from the experiments?  
- **Q9:** In [2], the authors propose a method also based on gradient values, where they clip a portion of the highest coordinates that consistently degrade the gradient. Could you discuss how AdaGC relates to it or perhaps compare it with their method?

### Muon and Lion
- **Q10:** As mentioned in **W1**, could you clarify why no spikes are observed with Muon and Lion in Section 5.4 (*Optimizer Compatibility*)? Do spikes appear when no clipping is applied?

### Minor Suggestions
- In Figure 1, labels are inconsistent. I suppose all plots show AdamW with GlobalGC. This should be clarified, and emphasized that even with global norm clipping, spikes still occur.  
- Some visualizations are hard to compare when identifying which curve has spikes. A log scale or transparency could make this clearer.  
- The sentence in lines 036–059 is difficult to follow; consider rephrasing or using bullet points for clarity.  
- In Table 1, the distinction between value-based and norm-based clipping could be mentioned more clearly in the related work discussion. 

[1] Andrew Brock, Soham De, Samuel L. Smith, Karen Simonyan. *High-Performance Large-Scale Image Recognition Without Normalization*. arXiv:2102.06171, 2021. [https://arxiv.org/abs/2102.06171](https://arxiv.org/abs/2102.06171)  
[2] Yan Pan, Yuanzhi Li. *Toward Understanding Why Adam Converges Faster Than SGD for Transformers*. arXiv:2306.00204, 2023. [https://arxiv.org/abs/2306.00204](https://arxiv.org/abs/2306.00204)

### Soundness
3

### Presentation
2

### Contribution
2
