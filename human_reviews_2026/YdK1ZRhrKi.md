# SNAP-UQ: Self-supervised Next-Activation Prediction for Single-Pass Uncertainty in TinyML

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
Reliable uncertainty estimation is a key missing piece for on-device monitoring in TinyML: microcontrollers must detect failures, distribution shift, or accuracy drops under strict flash/latency budgets, yet common uncertainty approaches (deep ensembles, MC dropout, early exits, temporal buffering) typically require multiple passes, extra branches, or state that is impractical on milliwatt hardware. This paper proposes a novel and practical method, SNAP-UQ, for single-pass, label-free uncertainty estimation based on depth-wise next-activation prediction. SNAP-UQ taps a small set of backbone layers and uses tiny int8 heads to predict the mean and scale of the next activation from a low-rank projection of the previous one; the resulting standardized prediction error forms a depth-wise surprisal signal that is aggregated and mapped through a lightweight monotone calibrator into an actionable uncertainty score. The design introduces no temporal buffers or auxiliary exits and preserves state-free inference, while increasing deployment footprint by only a few tens of kilobytes. Across vision and audio backbones, SNAP-UQ reduces flash and latency relative to early-exit and deep-ensemble baselines (typically $\sim$40--60% smaller and $\sim$25--35% faster), with several competing methods at similar accuracy often exceeding MCU memory limits. On corrupted streams, it improves accuracy-drop event detection by multiple AUPRC points and maintains strong failure detection (AUROC $\approx 0.9$) in a single forward pass. By grounding uncertainty in layer-to-layer dynamics rather than solely in output confidence, SNAP-UQ offers a novel, resource-efficient basis for robust TinyML monitoring. Our code is available at: https://github.com/Ism-ail11/SNAP-UQ

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work introduces SNAP-UQ, a single-pass, label-free uncertainty method for TinyML. SNAP-UQ maps a single forward pass depth-wise next-activation prediction, int8 heads uncertainty forecaster that drops neatly into CMSIS-NN pipelines and fits within kilobyte-scale MCU budgets. Results show SNAP-UQ improves accuracy-drop monitoring on corrupted streams, remains competitive on OOD failure detection, and strengthens ID calibration without temporal buffers, auxiliary exits, or extra passes.

### Strengths
- The paper is well-written and solid, with a new method, theoretical, and empirical contributions.
- The proposed method is empirically deployed across audio and vision tasks, with significant improvement over early-exit and deep ensembles baselines in reducing flash and latency. 
- Beyond memory and latency improvement, it also improves accuracy-drop detection in corrupted streams by AUPRC and maintains strong failure detection with AUROC in a single pass.

### Weaknesses
- There is a computational limitation in training when compared to the standard cross-entropy loss. Specifically, the training objective requires jointly training with additional terms. It requires back-propagating with an additional label-free auxiliary loss and either a variance floor, scale control, or detach option loss.
- The experiments are reported with only three fixed seeds. The results are reported without variance.

### Questions
- Can the proposed method be applied for larger-scale ML settings, such as ResNet-50 with ImageNet?
- In part 4.4, have the authors tried calibration on OOD data, such as on CIFAR-10-corrupted?
- In the experiments, the authors only show the accuracy drop or failure **detection** performance, but haven't shown what the actual accuracy drop or failure level actually is. Could the authors add accuracy results to Table 2 and Table 3?

### Soundness
3

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
This paper proposes SNAP-UQ, a single pass uncertainty estimation method for TinyML. The approach taps a small number of intermediate layers, predicts the mean and (diagonal or low rank + diagonal) variance of the next layer’s activations with a small INT8 head, and computes the standardized squared error (surprisal) between the prediction and the actual activation. Summing per layer surprisal yields a single pass uncertainty score $S(x)$, which is then converted to $U(x)\in[0,1]$ via a monotone mapper (logistic or isotonic) for monitoring and selective prediction. The authors argue that $S(x)$ is an affine transform of the depthwise negative log likelihood (a conditional Mahalanobis energy) and is invariant to batch normalization like channel rescalings. On real MCUs (Big and Small configurations), SNAP-UQ reduces latency, flash, RAM, and energy while improving corrupted stream detection (AUPRC and detection delay), failure detection (ID correct versus ID error, and ID correct versus OOD), and risk coverage. It is roughly 25 to 35% faster and 40 to 60% smaller than lightweight ensemble baselines, and continues to run where others run out of memory on Small MCU. The paper includes ablations on tap placement and rank, INT8 quantization, mapping choices, and head to head comparisons among single pass methods.

### Strengths
The main strength is the shift from output layer confidence to violations of conditional layer to layer dynamics as the basis for uncertainty, captured in a single forward pass without labels, with only tens of kilobytes of added memory and about $2%$ extra MACs, making it suitable for MCUs. Light theoretical support, namely the equivalence to depthwise negative log likelihood and conditional Mahalanobis energy and scale invariance under batch normalization like rescaling, grounds the intuition that softmax confidence tends to degrade later, whereas inter layer dynamics deteriorate earlier. Empirically, the work is careful on deployment metrics (latency, flash, peak RAM, energy on device) and on decision centric metrics (AUPRC and detection delay under stream corruption, AUROC for ID correct versus ID error and ID correct versus OOD, risk coverage curves, FPR). The ablations, including leave one tap out, rank sensitivity with diminishing returns around $r\approx 64$, INT8 quantization, and logistic versus isotonic mapping, are directly useful to practitioners. Overall, while the theory is intentionally lightweight, the design is coherent and the practical benefits for TinyML monitoring are clear.

### Weaknesses
1. **Tap placement feels ad-hoc.**:
Two taps (mid / pre-logits) seem to work—but moving to a new backbone still feels like guesswork. A tiny, illustrative check would help: a quick cross-layer correlation sketch to show how signal fades with distance, plus a short “one-tap-removed” table that makes redundancy visible.

2. **Where the latency gains come from is not entirely clear.**
While fairness is addressed, the source of the speedup is less so. Could you add a single stacked-bar breakdown—*backbone convolutions | tap projection/head | classifier | memory I/O*—and report the measurement conditions (fixed clock, CMSIS-NN, averaging protocol)? This would make the narrative clearer and explain why the advantage typically falls in the tens-of-percent range.

3. **Distributional variants (Student-t / Huber) aren’t yet actionable.**:
The math is there; the “when and how” is not. A small, concrete aid would do: a compact comparison on images and audio; a brief note on sensible defaults (e.g., workable (\nu) and (\delta) and the selection rule); and a line on stability (fix vs. learn (\nu), share across taps, INT8/LUT implications). With that, these variants become tools rather than curios.

**(Minor)** “Self-supervised” here refers to label-free auxiliary regression for uncertainty, not representation learning; a one-sentence clarification in the introduction would avoid confusion.

### Questions
1. **Could you provide reusable visualizations for tap selection?**: 
   Specifically, a surprisal correlation matrix among candidate layers (to show redundancy vs. distance), diminishing-return curves of performance versus the number of taps, and a leave-one-tap-out table. This would substantiate why two to three taps suffice and facilitate porting to new backbones.

2. **Could you report the datasets and tuning rules for Student-t and Huber?**: 
   Please specify the evaluated datasets, the grids for \nu and \delta, and the selection metrics. Include a comparison table (Gaussian vs. t vs. Huber) with AUROC/AUPRC, risk@coverage, AURC, FPR, recommended defaults (e.g., \nu=5, \delta=1.5), notes on stability (\nu fixed or learned; shared or per tap), and implications for INT8 implementation.

3. **Could you visualize the sources of the speedups?**: 
   Please add a stacked-bar breakdown of per-layer time, bandwidth, and temporary buffers for SNAP-UQ and the lightweight early-exit ensemble, with measurement conditions clearly stated, to clarify where the 25–35% latency difference arises.

4. **Could you promote the mapping choice to the main text and include a brief recipe?**: 
   For example, recommend isotonic for fixed-coverage monitoring and logistic when in-distribution calibration is the priority, and indicate whether combining S with a margin or confidence feature, m, is recommended by default.

5. **(Optional) Could you provide minimal-configuration guidance?**: 
   For extremely tight SRAM budgets or when intermediate activations are inaccessible, please include a small table for a minimal setup (e.g., pre-logits only, low rank such as r=32) showing latency, flash, and FPR trade-offs.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper is about uncertainty estimation in the context of Tiny ML, in particular this paper proposes an uncertainty estimation method based on predicting activations of some intermediate layers, from where an uncertainty score can be produced and transformed into a probability via platt scaling. This method is particularly designed for TinyML setups such as microcontrollers.

The contributions are:
- The SNAP-UQ method for uncertainty estimation in a TinyML context using a depth-wise surprisal signal.
- An MCU-ready implementation of the proposed method.
- Results for both ID and OOD detection performance on MNIST, CIFAR10, and SpeechCommands datasets showing the advantages of the proposed method.

### Strengths
- The paper is mostly well written and clear.
- Uncertainty in TinyML is an important open problem, there are not many papers in this area, so a new method and new state of the art are welcome.
- I believe the idea of predicting layer activations to produce uncertainty makes sense, this connects with previous ideas in the field like depth uncertainty and early exit ensembles. And the proposed method is specifically designed for TinyMl applications, which I believe is novel and significant.
- The evaluation is mostly good, there is a good selection of datasets, baselines, metrics, and microcontrollers platforms,
- Overall the conclusions are robust, that SNAP-UQ works best or is very competitive in terms of calibration, OOD detection and misclassification detection, producing a new state of the art for uncertainty estimation in TinyML.
- There is a good amount of appendices with relevant details and proofs of the proposed method being similar to existing methods, strengthening the proposed method and its usability.

### Weaknesses
- Some parts of the paper are hard to follow due to lots of mathematical notation, this is mostly in Sec 2.2 specifically at Eq 10 where the notation breaks. Overall I would recommend to the authors to try to minimize the necessary notation and to repeat the definitions of key notation in the paper, and try to explain concepts together with the math, this will make the paper much easier to understand and to implement.
- The paper claims results on TinyImageNet but the results are nowhere in the paper or the appendix, this weakens the claims slightly.
- The interesting ablation results are weak, by this I mean the hyperparameters that are relevant to the proposed method, which are presented in Appendix N, the authors showing the effect of the number of taps and the projector rank, but specifically for the number of taps only two taps are made, and I believe it would be interesting to try more taps and see if uncertainty improves.
- It would also have been interesting to try the huber and student's t-distributed variants, these are not explored in the experiments of this paper.
- In Table 1, to me it is not clear which model/method is denoted as "BASE", if its the baseline model without uncertainty estimation, I do not see how it could have higher latency and storage/RAM usage than when using SNAP-UQ, since SNAP-UQ slightly adds computation and memory use. Please clarify.
- In Figure 1, the definition of S(x) in the figure differs from Eq 9, the weights are missing.

### Questions
- The proposed pipeline has a lot of steps, one being probability calibration with platt scaling, did you test if this step is necessary? What about using the raw score in Eq 9 as an uncertainty metric (I know its not a probability).

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper addresses the problem of estimating prediction uncertainty in neural networks, with a particular focus on TinyML applications. The topic is especially relevant for edge-device deployment, where input distributions may shift after training. The authors propose a novel lightweight architecture that attaches to selected layers of a backbone network to estimate uncertainty for each prediction. This tiny module predicts the next activation in a layer-wise manner, enabling the computation of a surprisal score. The method is evaluated on both vision and audio tasks using several baseline models. Experimental results show that the proposed approach preserves task accuracy while substantially reducing flash memory usage and inference latency, demonstrating its strong practical value.

### Strengths
- The concept of next-activation applied in a layer-wise manner is interesting and demonstrates strong empirical performance.
- The proposed lightweight architecture achieves impressive results despite its extremely compact size (on the order of kilobytes).
- The effectiveness of the approach is validated across multiple tasks using practical evaluation metrics and reasonable baselines, yielding consistently encouraging results.

### Weaknesses
- Despite its potential significance, the paper’s writing quality is problematic. It appears to have been heavily compressed—possibly with assistance from LLMs—resulting in missing definitions for key terms and notations (e.g., LUT, BN, ID$\surd$-ID$\times$, ID$\surd$-OOD, $1\times 1$ projector, ...). Some paragraphs even consist of a single sentence, making the paper difficult to follow.
- Although a theoretical analysis is included, it lacks sufficient discussion of its implications. The authors should provide interpretations that link their theoretical results to practical insights, and possibly include experiments to verify the claims, such as empirically testing Proposition 2.3.
- The comparative baselines appear outdated, raising concerns about the competitiveness of the proposed method. More recent approaches (e.g., [1]) should be discussed and included in the comparisons to strengthen the evaluation.

Reference:

[1] Ghanathe, N. P., & Wilton, S. J. QUTE: Quantifying Uncertainty in TinyML models with Early-exit-assisted ensembles for model-monitoring. In Forty-second International Conference on Machine Learning, 2025.

[2] Ahmed, S. T., Hefenbrock, M., & Tahoori, M. B. (2024). Tiny Deep Ensemble: Uncertainty Estimation in Edge AI Accelerators via Ensembling Normalization Layers with Shared Weights. The 43rd IEEE/ACM International Conference on Computer-Aided Design.

### Questions
- What are fundamental advantages of the proposed method over QUTE in paper [1] before? This question concerns both theoretial and empirical aspects.
- How significant does the projector type effect the performance of SNAP-UQ?

### Soundness
3

### Presentation
2

### Contribution
3
