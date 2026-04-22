# GaLLoP: Gradient-based Sparse Learning on Low-Magnitude Parameters

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Sparse fine-tuning techniques adapt LLMs to downstream tasks by only tuning a sparse subset of model parameters. However, the effectiveness of sparse adaptation depends on optimally selecting the model parameters to be fine-tuned. In this work, we introduce a novel sparse fine-tuning technique named **GaLLoP**: **G**radient-based Sp**a**rse **L**earning on **Lo**w-Magnitude **P**arameters, which fine-tunes only those model parameters which have the largest gradient magnitudes on downstream tasks and the smallest pre-trained magnitudes, intuitively prioritizing parameters that are highly task-relevant, but minimally disruptive to pre-trained knowledge. Our experimentation with LLaMA3 8B and Gemma 2B as base models shows that GaLLoP consistently improves or matches the in-distribution as well as out-of-distribution performance obtained via the usage of other leading parameter-efficient fine-tuning techniques, including LoRA, DoRA, and SAFT. Our analysis demonstrates that GaLLoP mitigates catastrophic forgetting and memorization of task data, as important pre-trained parameters remain unchanged, and stabilizes performance relative to other fine-tuning techniques, robustly generalizing across most random seeds.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
GaLLoP is a sparse fine-tuning method that updates only parameters that are simultaneously highly task-relevant (large gradients) and minimally disruptive to pre-trained knowledge (small magnitudes), boosting both in- and out-of-distribution generalization.
It runs in two phases—score-and-select a small subset using gradient accumulation over a data slice, then fine-tune only those while masking the rest—and is model-agnostic. Across LLaMA-3 8B and Gemma-2B on eight commonsense datasets, it typically forms a dominant ID/OOD Pareto front and shows strong stability across seeds compared to LoRA, DoRA, SAFT, SpIEL, WiSE-FT, LiNeS, and FFT.
The paper also introduces diagnostics for catastrophic forgetting and memorization, finds near-zero values for GaLLoP, and notes a limitation: its unstructured sparsity is less hardware-efficient, motivating future work on structured densification.

### Strengths
1. The paper’s core idea—selecting a sparse set of parameters that are simultaneously high-gradient and low-magnitude—creatively fuses two complementary signals to target updates that help in-domain learning without erasing pre-trained knowledge, and it is implemented in a clean two-phase procedure.  It also contributes two evaluation diagnostics, forget ratio and collapse rate, that formalize catastrophic forgetting and severe memorization in a way that is easy to compute and compare across methods.  

2. The empirical study is broad and careful: eight commonsense datasets, two model families, and strong baselines spanning full fine-tuning, parameter-efficient methods, and editing techniques.  Results consistently show a dominant ID/OOD Pareto front for GaLLoP, with particularly strong averages on LLaMA-3 8B and balance on Gemma-2B. Reproducibility is helped by explicit algorithmic pseudocode and detailed hyperparameters.

3. The exposition is easy to follow

4. Addressing the ever-present tension between adaptation and retention, the method delivers strong OOD gains while curbing overfitting that plagues several popular baselines—an outcome that matters for reliable deployment of fine-tuned LLMs.  Because GaLLoP is simple, model-agnostic, and efficient, it lowers the barrier to adopting sparse fine-tuning in practice.

### Weaknesses
1. The core scoring rule combines gradient size with (inverse) weight magnitude, but closely adjacent ideas already exist: SAFT selects by large gradients alone; Pruning and finetuning selects the smallest pre-trained weights; SNIP scores parameters using a gradient–weight product at (re)initialization; Movement Pruning selects by the magnitude of weight updates during fine-tuning; and dynamic sparse training like RigL periodically regrows weights based on gradient signals. A strong way to sharpen the contribution would be to add head-to-head ablations against these concrete criteria on exactly the same LLM setup and to explain theoretically why the ratio used here should dominate the product used in SNIP under typical LLM curvature/noise regimes. 

2. OOD is instantiated in an eight-dataset round-robin within the same commonsense QA family. That’s a convenient stress test but not a true distribution shift across modalities or task formats. To strengthen claims about generalization, evaluate on qualitatively different targets (instruction following, long-form QA/summarization, math/code, safety and refusal behavior) and on established OOD suites (e.g., CLIP-style ImageNet variants or robustness benchmarks) to mirror SAFT’s cross-domain evaluations. 

3. The paper argues that layer-wise score sampling cuts sort/memory costs and claims the method is “highly efficient,” but provides no end-to-end wall-clock, peak-memory, or throughput numbers (selection phase and training phase) versus LoRA/DoRA/SAFT. 

4. Sensitivity and design choices need deeper ablations - Important parameters (global vs layer-wise thresholds, effect of ϵ, optimizer interactions like weight decay, gradient-accumulation window length) are fixed or only lightly discussed.

5. LoRA/DoRA/SAFT/SpIEL/WiSE-FT/LiNeS are useful, but widely used PEFT baselines such as IA³ and BitFit are absent; including them (and tuning LoRA ranks under an iso-FLOPs or iso-trainable-parameter protocol) would make the comparison more complete. Also disclose and equalize the number of backpropagated layers and gradient-checkpointing settings, which materially affect wall-clock and stability. 

Papers:
[1]: https://arxiv.org/abs/2407.03036? "SAFT: Towards Out-of-Distribution Generalization in Fine-Tuning"
[2]: https://proceedings.mlr.press/v119/evci20a/evci20a.pdf "Rigging the Lottery: Making All Tickets Winners"
[3]: https://arxiv.org/abs/1810.02340 "SNIP: Single-shot Network Pruning based on Connection Sensitivity"

### Questions
See Weaknesses above.

### Soundness
2

### Presentation
2

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
The paper presents a method for sparse fine tuning that attempts to avoid catastrophic forgetting while avoiding over fitting on newer tasks. Both in-distribution and out-of-distribution performance is shown to improve. The technique chooses a small subset of parameters that are fine tuned based on two criteria - (i) magnitude of the parameter in the pretrained model, (ii) and gradient of the parameter while fine tuning. A small data set sample is used for fine tuning to estimate the later value. The idea is simple but is well motivated by experimental results. Empirical results demonstrate the superiority of the proposed technique over related SpFT techniques.

### Strengths
1) The paper proposes a simple, intuitive, but effective strategy for sparse fine tuning.
2) The strategy is well motivated by related experiments.
3) Exhaustive experimental results are presented to demonstrate the efficacy of the proposed technique.
4) An improvement in out-of-distribution tasks is shown.

### Weaknesses
1) The idea is only experimentally demonstrated to work well. Theoretical justifications are missing. There are a number of view points and observations in literature about the choice of parameters while fine tuning. The role of magnitude of a parameter in determining its importance is not unanimously agreed upon. Similarly, gradient value while backpropagating on a small batch of data might be too noisy as a signal to score parameter importance. LoRA and related low-dimension projection techniques techniques looks at spectral properties of the parameter matrix rather than the magnitude and gradients. Other approaches like the ones based on expander graphs consider network connectivity rather than parameter weights. Authors may discuss/support the proposed algorithm in more "theoretical" term.
2) The efficacy of the proposed technique in a continual learning scenario is not discussed. Magnitude and gradient based parameter selection may either be erroneous or yield a low sparsification ratio when performed in sequence of rounds on multiple tasks. Experiments need to be presented to counter this claim. Such a scenario was mentioned as a motivation in the introduction. 
3) Because of the rather dynamic and unstructured nature of the proposed parameter selection technique the method may not align well with hardware acceleration thus defeating the benefits of sparse fine tuning.

### Questions
1) How does the method perform on already sparse LLM models? Such models are increasingly becoming more popular. A small experiment on applying the proposed algorithm to some sparse LLM may provide valuable insight.
 2) What is the effect of running the proposed method over multiple tasks in a sequence. The effect of overfitting and catastrophic forgetting would be more evident.
3) Authors may delve deeper into studying the set of parameters chosen for fine-tuning using the first and the second criteria (namely, low magnitude, and high gradient) respectively. What is the percentage intersection of these sets of parameters?
4) A summary table comparing the performance of the proposed approach with SOTA in terms of accuracy/forgetfullness/density may be reported in the main paper itself.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes GaLLoP, a sparse fine-tuning method that selects a subset of parameters to update based on the ratio between their gradient magnitude and pretrained weight magnitude. The key idea is that parameters with large gradients (task-relevant) and small pretrained magnitudes (less tied to prior knowledge) can be fine-tuned to improve in-domain (ID) performance while mitigating catastrophic forgetting on out-of-domain (OOD) tasks. Experiments on eight commonsense reasoning datasets show that GaLLoP achieves comparable or better ID/OOD accuracy compared to existing sparse fine-tuning baselines such as SAFT and SpIEL.

### Strengths
1. Simple and straightforward design : The proposed scoring criterion is conceptually clear and easy to implement without architectural modification or additional modules.

2. Parameter-efficient fine-tuning : Only a small fraction of parameters (around <2%) are updated, yielding memory-efficient adaptation with no extra trainable layers.

3. Empirical stability : The method shows zero collapse or forgetting ratios across multiple seeds and densities, indicating robustness to optimization noise.

### Weaknesses
1. Lack of theoretical justification : The claim that gradient magnitude directly represents parameter sensitivity is overstated. Without a connection to curvature-based measures such as the Fisher Information Matrix or Hessian spectrum, it is unclear whether $\|\|g\|\|$ accurately captures parameter importance.

2. Unverified assumption about parameter magnitude : The interpretation that low-magnitude parameters encode less critical or “more adjustable” knowledge is speculative and not theoretically demonstrated. The connection between parameter norm and the nature of stored knowledge seems vague.

3. Non-typical ID/OOD setup : The so-called OOD evaluation is defined only as cross-dataset testing within a single commonsense reasoning benchmark family. This does not constitute a true domain-shift scenario and thus limits claims about generalization.

4. Limited experimental scope : Despite involving eight datasets, the experiments are confined to the commonsense reasoning domain, and only two model scales (Gemma-2B, LLaMA3-8B) are tested. Broader evaluation on diverse task types (e.g., reasoning, factual QA, math) or architectures would strengthen the argument.

5. Marginal gains over baselines : On Gemma-2B, GaLLoP performs nearly identically to SAFT; even on LLaMA3-8B, the reported improvement is modest (see Fig. 8 in Appendix E). The performance advantage appears minor and inconsistent, suggesting that the proposed selection rule may not offer substantial benefits over existing sparse fine-tuning methods.

### Questions
1. The paper assumes that gradient magnitude represents parameter sensitivity and that small-magnitude parameters correspond to more “adjustable” knowledge, but both claims lack theoretical or empirical justification. Could the authors provide supporting evidence such as theoretical relation between $\|\|g\|\|$, Fisher Information, or Hessian-based parameter sensitivity and analyses showing that parameter magnitude indeed aligns with knowledge stability or adaptability in pre-trained models?
2. The ID/OOD setup in this paper is based on cross-dataset evaluation within commonsense reasoning tasks, rather than across distinct domains. Could the authors clarify why this configuration should be regarded as out-of-distribution generalization rather than merely cross-task transfer? Additionally, have the authors examined whether GaLLoP’s advantages persist under more conventional domain-shift OOD settings?
3. Experimental coverage is limited to two baselines of different scales and a single task. Do the authors plan to test on other architectures or modalities to demonstrate general applicability of the proposed criterion?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Summary: This paper proposes a method to mitigate forgetting, by choosing whether to update a particular parameter based on a score that is its ( ( gradient_magnitude ) / ( magnitude + epsilon) ) - the top "k" highest scoring ones are updated in that iteration and others are not.  

Evaluation of forgetting is done by taking 8 datasets, fine-tuning on one ("IID") and measuring performance on the others ("OOD") - and averaging this over round-robin selection of IID/OOD.

### Strengths
The paper is clearly written.

### Weaknesses
(A) comparisons: Mitigating catastrophic forgetting is a rich field and this paper is missing comparisons against several state of the art methods, for example:
(1) https://arxiv.org/abs/1612.00796 (classic method, works very well on modern LLMs as well)
(2) https://arxiv.org/abs/2407.20999 (another method that selects individual parameters - based on a different logic - to update, instead of updating all parameters)
(3) https://icml.cc/virtual/2025/poster/46655 ( more recent paper and method, contains overview of preceeding)

Indeed this method is only compared against other methods that update fewer parameters, but if the main aim is mitigating catastrophic forgetting then a much more robust comparison against other approaches like the three above, and also others, is needed. 

(B) reasoning: what is the intuition behind this method ? Eg Fig 1 shows that fine-tuning the lowest magnitude weights gives gains, but what about updating a randomly chosen subset of  weights ? 

(C) Novelty: there are many methods that choose only a subset of parameters to update based on values, gradients or a combination thereof; this paper is yet another in that line (albeit with a slightly variant selection logic). The novelty from an ideas perspective is low.

### Questions
Fig 3: It is very surprising that OOD performance improves over Vanilla - this is the opposite of forgetting, as it means that fine-tuning on one dataset improves performance on 7 others when doing e.g. LoRA or other such. How is this possible ? It seems to go against what other papers in forgetting show - that fine-tuning on a dataset improves ID but degrades OOD.

### Soundness
1

### Presentation
2

### Contribution
2
