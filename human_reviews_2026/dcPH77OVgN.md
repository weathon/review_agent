# Scaling Law for Quantization-Aware Training

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Large language models (LLMs) demand substantial computational and memory resources, creating deployment challenges. Quantization-aware training (QAT) addresses these challenges by reducing model precision while maintaining performance. However, the scaling behavior of QAT, especially at 4-bit precision (W4A4), is not well understood. Existing QAT scaling laws often ignore key factors such as the number of training tokens and quantization granularity, which limits their applicability. This paper proposes a unified scaling law for QAT that models quantization error as a function of model size, training data volume, and quantization group size. Through 268 QAT experiments, we show that quantization error decreases as model size increases, but rises with more training tokens and coarser quantization granularity. To identify the sources of W4A4 quantization error, we decompose it into weight and activation components. Both components follow the overall trend of W4A4 quantization error, but with different sensitivities. Specifically, weight quantization error increases more rapidly with more training tokens. Further analysis shows that the activation quantization error in the FC2 layer, caused by outliers, is the primary bottleneck of W4A4 QAT quantization error. By applying mixed-precision quantization to address this bottleneck, we demonstrate that weight and activation quantization errors can converge to similar levels. Additionally, with more training data, weight quantization error eventually exceeds activation quantization error, suggesting that reducing weight quantization error is also important in such scenarios. These findings offer key insights for improving QAT research and development.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work investigate scaling laws a 4-bit precision (W4A4), arguing that exisiting QAT scaling laws often ignore the key factors, such as number of training tokens and quantization granularity. The paper proposes a unified scaling law for QAT that modelling quantiation error as a funciton of model size, size of training data and quantization group size, validated across 268 experiment, uncovering qunatization properties, such as different sensitiviy on weight and activation quantization errors (W16A4, W4A16). Applying mixed-precision quantization to adress bottle necks demonstraing weight and quantization errors can converge to similar levels, descriping how the quantization error can depend on the model size, number of training tokens and quantization granularity (elements in each quant. group). Through experiments this work demonstrates valuable insights for future research in QAT, such as the limitations of activation quantization.

### Strengths
1. Demostrating that the quantization error is both dependent on the number of parameters and the training data size for W4A4 QAT.

2. Also including quantization granularity, following prior works.

3. Reproducibility: Models, hyperparameters, dataset (open-source) and evaluation metrics reported.

4. Investigation of weight-only, activation only and both activation and weight quantizations.

5. Demonstrate INT4 is similar or superior in performance to FP4, argues why theoretically and uses it as a rationale for using INT4 quantization in experiments.

6. Learning rate abblation study for W4A4 models, contraring what we have seen reported in BitNet models, arguing the for both uncompressed and QAT training.

7. Generally provides the community with a lot of interesting insights for future QAT research.

### Weaknesses
1. Figures 2, 3, 5,6 could benefit for better descriptions making them easier to interpret.

2. Only the Llama3 model family is investigated, potentially openeing a limitation of the scaling law on newer archiectures, limiting the impact.

3. Scaling parameters are fit on 80 runs – Quite a a lot especially for larger models, where compression becomes even more relevant, both for efficiency and performance reasons, limiting the feasible of who can afford to fit the parameters on big models.

### Questions
1. Do you expect this to scale beyond 595M parameters?

2. A large number of training tokens for small models, was that necessary to prove the points?

3. The parameters “The parameter \gamma_{N} measures the sensitivity to the model” - Isn’t it more accurate to say it dictates the sensitiveness, as you need quite a lot of runs to fit it to the scale-equations (5). (same for \gamma_{G})

4. Figure 7B demonstrates that W4A16 quantizatione error increases with the number of tokens? I guess it’s the increase in variance in the training data that makes the capacity of the model struggle, but do you have an idea on how sensitive this is? It is within-distribution and what we see is a generalization error, or is the weightage of data being more domain-specific.

### Soundness
3

### Presentation
2

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
This paper investigates how quantization-aware training (QAT) scales with model size, dataset size, and quantization granularity. Through 268 from-scratch QAT experiments on LLaMA-style models, they empirically show that quantization error decreases with larger models, but increases with larger datasets and coarser quantization groups. They further decompose the error into weight and activation components and identify the FC2 layer as a key activation outlier bottleneck, suggesting that mixed-precision treatment (8-bit for FC2) can balance both sources of error.

### Strengths
- Presents the first unified QAT scaling law incorporating model size (N), dataset size (D), and quantization granularity (G).
- Provides clear empirical validation with extensive experiments and well-fitted results
- Offers useful diagnostic insights, including weight vs. activation error decomposition and identification of the FC2 activation bottleneck.

### Weaknesses
- The definition of quantization error differs from conventional quantization studies. In this paper, the “quantization error” is defined as the final training loss gap rather than the difference between quantized and full-precision parameters or inference performance degradation.
There is no clear rationale for this definition, and the paper does not show whether a smaller loss gap actually correlates with better downstream performance when compared to BF16-trained counterparts.
This makes it difficult to interpret the analysis in a practically meaningful way.
- The paradigm of QAT used in this paper also deviates from typical practice. In most real-world scenarios, QAT is applied on well-trained full-precision LLMs to recover accuracy after quantization, whereas this paper applies QAT from scratch. It is unclear whether a model trained from scratch under 4-bit quantization can achieve comparable or better performance than a full-precision model followed by QAT.
Moreover, the models studied are mostly below 1B parameters, which further limits the generality of the conclusions.
- The depth of contribution appears somewhat limited. Extending the scaling law to include additional factors (dataset size and quantization granularity) is a valuable analytical contribution, but the idea of mitigating outlier activation errors by applying higher precision to specific layers has already been extensively discussed in prior mixed-precision quantization works.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a unified scaling law for quantization-aware training (QAT) of large language models, linking quantization error to model size, training tokens, and quantization granularity. Across 268 experiments on LLaMA-style models, the authors find that error decreases with model size but grows with more data and coarser quantization. They identify activation outliers in the FC2 layer as the main 4-bit bottleneck and show mixed-precision (8-bit FC2) largely fixes it. The study provides clear empirical and theoretical insights for improving low-bit LLM training.

### Strengths
1. The unified scaling law feels intuitive yet backed by solid data; the inclusion of both data and granularity terms makes sense.
2. Clear identification of the FC2 activation bottleneck; the mixed-precision fix is simple and convincing.
3. This paper is a neat empirical work that connects scaling laws and quantization in a meaningful, practically useful way.

### Weaknesses
1. Experiments stop at sub-1B dense models — unclear if the scaling law still holds for >10B or MoE setups.
2. Mostly focuses on W4A4; doesn’t explore ternary or mixed-bit cases that recent works care about.
3. While empirical results are strong, the practical takeaways for real deployment (beyond FC2 mixed precision) could be discussed more.

### Questions
1. The paper argues that quantization error increases with more training tokens — which is counterintuitive from a generalization standpoint. Could the authors provide a deeper explanation or theoretical intuition for why additional data worsens quantization sensitivity?
2. The scaling exponents ($\gamma_N$, $\gamma_D$ and $\gamma_G$) are empirically fitted. Do they exhibit any consistency across model families or datasets, suggesting a more universal law, or are they dataset-dependent heuristics?

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
4

### Summary
This paper proposes a unified scaling law for 4-bit Quantization-Aware Training (W4A4) in language models. The authors model the quantization error as a function of not only model size but also the number of training tokens and the quantization group size. Based on 268 experiments, they formulate $\delta_p \propto D^{\gamma_D} G^{\gamma_G} / N^{\gamma_N}$, finding that error decreases with larger models but, counter-intuitively, increases with more training data. The paper further decomposes this error, identifies activation quantization as the primary bottleneck, and pinpoints it to outliers in the FC2 projection layer's input.

### Strengths
- Significance: The work addresses a critical and timely problem. As W4A4 QAT becomes essential for efficient LLM deployment, understanding its scaling behavior is of high practical importance, yet it remains poorly understood.

- Originality: The primary originality lies in the formulation of a scaling law that includes the training data volume (D) and quantization granularity (G). The finding that quantization error increases with D is a novel and non-trivial observation that challenges common assumptions.

- Quality: The study is methodologically sound, built upon a substantial empirical foundation of 268 QAT experiments. The approach of decomposing the total error into weight-only ($\delta_{W4A16}$) and activation-only ($\delta_{W16A4}$) components (Fig. 6) is a systematic way to investigate the error sources.

- Clarity: The paper is well-written, and the core experimental results (Fig. 4, Fig. 7) are presented clearly, making the authors' main observations easy to follow.

### Weaknesses
- Limited Generalizability Due to Model Scale: The paper's most significant weakness is the gap between its claims ("Large Language Models") and its experimental setup. The scaling law is derived from models ranging from 74M to 595M parameters, with validation on a 973M model. These models are orders of magnitude smaller than current state-of-the-art LLMs (e.g., 7B, 70B, 100B+). Scaling laws are only valuable if they extrapolate, and there is no evidence that a trend observed in sub-1B models will hold for models 100x or 1000x larger, which may exhibit different emergent properties and quantization sensitivities. This discrepancy severely limits the trustworthiness and practical applicability of the proposed law for actual LLMs.

- Lack of Mechanistic Analysis for the Effect of D: A key finding of the paper is that quantization error $\delta_p$ increases with the data volume D. The authors observe this (Fig. 4b) and note that weight quantization error ($\delta_{W4A16}$) is more sensitive to D than activation error (Table 1, $\gamma_D$ values). However, the paper stops at this observation and fails to provide a deeper, mechanistic explanation for why this occurs. Does training on more data fundamentally increase the complexity of the learned weight representations, making them harder to compress into 4 bits? Does it alter the weight distributions or introduce more outliers? This lack of causal analysis turns a potentially profound discovery into a superficial empirical note.

- Insufficient Novelty: The identification of the FC2 projection input (following the SwiGLU non-linearity) as the primary source of activation outliers (Fig. 9a) is not a new discovery. This has been a well-documented phenomenon in the Post-Training Quantization (PTQ) literature for years (e.g., in work like SmoothQuant). The paper confirms this finding holds for QAT but does not offer a new insight specific to the QAT process itself.

- Trivial Solution for Bottleneck Analysis: The proposed "solution" to this bottleneck is to use 8-bit precision for the FC2 input (Fig. 9b). This mixed-precision approach is a workaround that avoids the core W4A4 challenge rather than solving it. This analysis does not inform how to improve a true W4A4 model. A more valuable contribution would have involved exploring QAT-native techniques (e.g., targeted regularization, learnable clipping functions for outliers) that operate within the W4A4 constraint.

### Questions
Fundamentally, why do the authors think the loss gets worse as the training data (D) increases? 

It is counterintuitive, since the primary mechanism of QAT is to robustify the model against quantization error throughout training; thus, longer training is expected to close the accuracy gap between the full-precision and the quantized model. This paper's observation could be contradicted by the practical successes of reduced-precision inference (e.g., NVIDIA NVFP4 - https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/, Apple Quality-Recovery Adapter - https://arxiv.org/pdf/2507.13575v1) that incorporate billion-token finetuning to fully recover the FP model's accuracy. 

Can the authors clarify why their observation of loss degradation with increased training data (Fig. 4(b)) makes sense?  One possible sync-up point is as follows: Scaling Laws for Precision (https://arxiv.org/pdf/2411.04330) and ParetoQ (https://arxiv.org/pdf/2502.02631) show that an intermediate training budget helps achieve pareto-optimal QAT accuracy when aggressive quantization is applied. It would be helpful to scale the training tokens in Fig. 4(b) to a lower budget (1M-1B tokens) to see if a consistent trend emerges. 

[Willing to update the ratings based on this discussion]

### Soundness
3

### Presentation
2

### Contribution
1
