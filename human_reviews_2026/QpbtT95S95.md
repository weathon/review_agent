# Compute-Optimal Quantization-Aware Training

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 2

## Abstract
Quantization-aware training (QAT) is a leading technique for improving the accuracy of quantized neural networks. Previous work has shown that decomposing training into a full-precision (FP) phase followed by a QAT phase yields superior accuracy compared to QAT alone. However, the optimal allocation of compute between the FP and QAT phases remains unclear. We conduct extensive experiments with various compute budgets, QAT bit widths, and model sizes from 86.0M to 2.2B to investigate how different QAT durations impact final performance. We demonstrate that, contrary to previous findings, the loss-optimal ratio of QAT to FP training increases with the total amount of compute. Moreover, the optimal fraction can be accurately predicted for a wide range of model sizes and quantization widths using the tokens-per-parameter-byte statistic. From experimental data, we derive a loss scaling law that predicts both optimal QAT ratios and final model performance across different QAT/FP compute allocation strategies and QAT bit widths. We use the scaling law to make further predictions, which we verify experimentally, including which QAT bit width is optimal under a given memory constraint and how QAT accuracy with different bit widths compares to full-precision model accuracy. Additionally, we propose a novel cooldown and QAT fusion approach that performs learning rate decay jointly with quantization-aware training, eliminating redundant full-precision model updates and achieving significant compute savings. These findings provide practical insights into efficient QAT planning and enable the training of higher-quality quantized models with the same compute budget.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper extends scaling-law analysis to the two-stage full-precision & QAT pipeline, asking how to split a fixed training budget between full-precision pretraining and quantisation-aware training. Through hundreds of runs across model sizes (86M–2.2B) and bit-widths, the authors show that the loss-optimal QAT fraction increases with scale, and that this dependency is well captured by a simple _tokens-per-parameter-byte_ statistic. They fit a unified loss scaling law (over ($N$, $D_{fp}$, $D_{qat}$, $B$) that includes pure-QAT and FP/QAT interaction terms, and use it to predict optimal QAT ratios as well as when low-bit QAT can match FP loss. The study also proposes a practical “cooldown & QAT fusion” training schedule (merging LR decay with QAT) and reports small but consistent savings in “wasted tokens” at moderate/high bit-widths. Overall, the paper moves the literature from “QAT should be ~10%” heuristics to compute-aware prescriptions grounded in scaling behaviour, while relying on a strong, modern QAT baseline (ParetoQ)

### Strengths
All in all, this is a solid experimental paper that answers fundamental questions in LLM deployment with a sound methodology. I enjoyed how the authors used their loss scaling curves to answer interesting questions in their discussion. For example, the question of what the optimal QAT bitwidth per model and training budget is has been a topic in the community for years. The main strengths are:

* The paper addresses a very significant and unanswered question in LLM inference and of increasing importance for LLM deployment. What budget should be allocated to QAT vs FP training? No previous work has answered this important question in experimentally sound ways, and this work paves the way for significant future work.
* The loss scaling laws are solid, based on hundreds of experiments, and provide strong transparency on the fit of the curve. The introduction of the token-per-byte statistics is also a very useful metric capturing the idiosyncrasies of QAT. The vast corpus of QAT experiments makes for a solid experimental methodology.
* A very extensive appendix, including many results, experimental set-ups, and curves.

### Weaknesses
* The paper is somewhat too long given the level of novelty and contribution, and as a result, the central narrative becomes hard to follow. While each section presents interesting findings, the overall flow sometimes feels like a set of independent conclusions about QAT rather than a single, coherent story. I found myself re-reading parts to connect the results to the main scaling-law argument. 

* Section 7 reads as an addendum: it explores LR-scheduler fusion, which is interesting but tangential to the main scaling-law contribution, and the reported gains are relatively small and bit-width dependent. I would think this should be part of another paper or appendix.

* The full-precision loss regularisation used to fit the unified law is crucial to interpreting Fig. 5 and the FP-vs-QAT comparisons, but it’s only described in Appendix E. Similarly, the “wasted token count” metric is under-explained and would benefit from a formal definition and more apparent motivation.

### Questions
* Figure 1 (right), add $D_{fq}$ and $D_{qat}$ along the axes and include a loss heatmap in the legend
* Equation 2: more discussion and motivation for the forms chosen related to QAT.. What is pure QAT penalty, and why does it take form? * * Why does FP/QAT interaction take this form? Are these grounded in theory or chosen for fitting reasons? Did you do any ablation studies on the form?
* Appendix D. Figure 9: I did not see any discussion or explanation on why the theoretical and experimental optima diverge so much at the lowest $D_{total}$ iso curves for 2,4, and 6 bits for the smallest model of 86M.
* Line 277: wasted token count: I had to read the definition multiple times to understand it. Could you add an equation? 
* Line 322: iso-flop. Capitalise flop because it’s hard to understand it when it’s in lower case.
* Figure 5: Make the colour of the bitwidths 5 and 1 more distinct. It’s very hard to tell which curve corresponds to each bitiwidthth
* Section 7: wasted unfused total tokens: could you maybe add an equation? I don’t fully understand how you reach that number.

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
3

### Summary
This paper investigates the optimal resource allocation ratio between Full-Precision (FP) and Quantization-Aware Training (QAT) phases. Through extensive LLM experiments, the authors found that the optimal QAT fraction increases with the total compute budget and introduced tokens-per-parameter-byte as a key metric to accurately predict this optimum. Loss Scaling Law is proposed to predict the performance of QTA.

### Strengths
This paper establishes a novel loss scaling law capable of predicting both the optimal QAT fraction and final model performance across varying bit widths and allocation strategies, demonstrating high practical utility. The proposed method integrates learning rate decay with QAT, effectively eliminating redundant model updates and improving both computational efficiency and accuracy.

### Weaknesses
Experiments are conducted on a specific LLM architecture（llama2）, and the scaling law parameters may require refitting for other model types. Different types of models may exhibit varying degrees of tolerance to quantization, particularly those with inherent redundancy.

### Questions
How sensitive are the fitted parameters to the range of model sizes and token counts used in the experiments? Can the authors provide an uncertainty analysis? While fine-tuning the FP model, how would the optimal QAT fraction for the final QAT phase shift? There might be a distribution difference between pre-training phase and fine-tuning phase.

### Soundness
3

### Presentation
3

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
This work is focused on QAT fine tuning which is done after full-precision (FP) phase pre-training.
QAT bit widths, and model sizes from 86.0M to2.2B to investigate how different QAT durations impact final performance. 
They show that contrary to previous findings, the loss-optimal ratio of QAT to FP training increases with the total amount of compute.
This work derive a loss scaling law that predicts both optimal QAT ratios and final
model performance across different QAT/FP compute allocation strategies and
QAT bit widths.  Additionally, they propose a novel cooldown and QAT fusion approach that performs learning rate decay jointly with quantization aware training, eliminating redundant full-precision model updates and achieving significant compute savings.

### Strengths
Authors run multiple experiments for fitting generating scaling law into proposed function and demonstrated small prediction error. They showed that the optimal QAT fraction is not a fixed percentage but rather increases with the total compute budget, specifically with the tokens-per-parameterbyte statistic. This challenges the previous conclusion that 10% is universally optimal for QAT length relative to total training length.

They proposed a loss scaling law that captures the optimal QAT fraction phenomenon and
models the final expected loss of the FP and QAT.

Proposed a novel approach: cooldown & QAT fusion—a scheme where learning rate decay is performed jointly with quantization-aware training, eliminating redundant fullprecision updates and achieving better accuracy for the same token count.

### Weaknesses
This work is focused on a specific LLM architecture and data sets, and exact results may differ for different model types different QAT methods and data.
Only int quantization explored, it would be interesting to see fp4 and fp8 results.
Limited size LLM are explored.

### Questions
Do you reset or keep optimizer state, from pre-trained model, when apply QAT fine tuning?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper, the authors talk about the scaling law for quantization aware training as well as the impact of the fraction for quantization aware training compared to the full-precision pretraining. The topic seems to be interesting and the authors provide some experiments to demonstrate the results. However, the paper is very badly written and results presentation is very bad (e.g., the figures are very difficult to read and the conclusions made from the figures are difficult to follow or should be arguable).

### Strengths
- The topic and problem proposed seems to be interesting, and the final conclusion or results, if correct as the author claimed, might be useful. Especially, the conclusion about how to determine the optimal fraction for quantization aware training, and also the possibility to eliminate redundant full-precision training via novel learning rate scheduling, might be useful.
- Regarding the techinical contribution, the authors first study the optimal fraction allocated for QAT and its relationship with the tokens-per-parameter-byte. Afterwards, they propose the loss scaling law for optimal QAT fraction and model the final expected loss of the FP and QAT pipeline. They finally propose a novel approach of cooldown and QAT fusion, where learning rate cooldown in FP training and learning rate warmup in QAT are fused, and thus redundant full-precision updates can be eliminated.

### Weaknesses
- The writing and results presentation are not good. Figure1 is difficult to read.
- Improvement in Table 2 might not be strong enough.

### Questions
In Figure 1 left, the y-axis is QAT tokens-per-byte, and x-axis is Total tokens-per-byte. The author would like to demonstrate that the optimal QAT fraction increases with the full training tokens-per-parameter-byte (line 092 in the original paper). To demonstrate this, the y-axis should be the ratio between QAT tokens-per-byte to the total tokens-per-byte, instead of only list the two end ratios in the figure. Could the author provide the plot with ratio as the y-axis to prove the conclusion?

### Soundness
2

### Presentation
1

### Contribution
2
