# Beyond Outliers: A Study of Optimizers Under Quantization

- Decision: Accept (Poster)
- Scores: 8, 4, 4

## Abstract
As new optimizers gain traction and model quantization becomes standard for efficient deployment, a key question arises: how does the choice of optimizer affect model performance in the presence of quantization? Despite progress in both areas, systematic evidence on optimizer–quantization interactions remains limited. To fill this gap, we study the impact of optimizer choice on model robustness under quantization, considering both post-training quantization (PTQ), and quantization-aware training (QAT). We first train full-precision models, ranging from 50M to 1.5B parameters, with six optimizers, to explore the hyperparameter landscape, and establish well-tuned baselines.
We then apply PTQ to evaluate how model performance degrades when trained with different optimizers. We find that outlier-related metrics, such as the max-to-mean ratio (MMR) and Kurtosis, fail to predict the PTQ performance across different optimizers. We show analytically that this is due to the MMR capturing only isolated layer errors, while ignoring how quantization errors accumulate and propagate through the network. To study the QAT degradation, we train quantized models from scratch and compare them to our original-precision baselines. We find that optimizers performing well in the original pretraining setup may not remain optimal under QAT, and that models trained with Shampoo show the lowest accuracy degradation. Finally, we derive scaling laws for quantization-aware training under different optimizers, showing that Shampoo achieves the highest parameter efficiency of all tested optimizers.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors study the interaction between quantization and choice of optimizers for both PTQ and QAT scenarios. The authors train BF16 models with various optimizers and perform PTQ to measure the degration. The authors then propose a new metric which correlates well with the PTQ performance compared to commonly used metrics like MMD and Kurtosis. The authors then study the impact of optimizer choice during W4A4 QAT and show that the choice of optimizer does not transfer directly from high precision to low precision.

### Strengths
- This is a great paper. Extremely well written and addresses important questions still unresolved in the community.
- Compares models using downstream performance rather than a proxy metric like validation loss.
- The differences in error propagation through the network with various optimizers was an interesting observation.

### Weaknesses
- Only RTN PTQ scheme is studied, which is not usually used in practice.
- For QAT, microscaling formats like MXFP4/ NVFP4 would perhaps be a better choice
- Due to lack of using hyper parameter selection techniques like MuP, the current hyperparameter choices might be suboptimal.

### Questions
- Why is the OLMo2 architecture changed to incorporate $\text{ReLU}^2$?
- Any reason for not using MuP for hyperparameter selection?
- Is it reasonable to expect the optimal hyperparameters to be same for both the full-precision and low-precision runs?
- What happens if we use fancier PTQ schemes compared to Round-to-nearest? Is the result still the same?
- "Shampoo’s networks are also characterized by the highest MMR". Figure 2 suggests it's PSGD?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work explores the impact of quantization on various optimizers. This work presents a comprehensive empirical analysis of how optimizers fare before and after quantization. The authors dispute that using outlier-related metrics like max-to-mean ratio (MMR) of input tensors is a useful indicator of quantization performance. Instead, this work focuses on studying the relative change in activations as a more expressive quantity to observe. The experiments are performed on recent class of transformer-based models, using a wide array of recent/popular optimizers. This work concludes that based on their experiments Shampoo optimizer fares best with the lowest accuracy degradation.

### Strengths
* The stated goals of the work, of assessing the impact of quantization on different optimizers, and formalizing the accuracy degradation using better metrics than MMR is interesting. 
* The work makes a compelling case showing that the focus on outliers is not useful to predict accuracy degradation. The formalism of relative change in activations $R_\ell$ as a proxy for total quantization error is convincing.
* Experiments are comprehensive; provide solid empirical evidence for the claims in the paper.

### Weaknesses
* **Outlier-centric motivation:** One of the main motivation of this empirical study is to show that outlier-based metrics are not predictive of accuracy degradation due to quantization. However, the authors do not point to any relevant literature or the general consensus in the community that argues this is the case. In my opinion, the consensus is on straight-through-estimator (STE) when using quantization aware training (QAT) [1], and uniform quantization that are the more widely cited reasons [2]. Can the authors point to the specific literature that claims outliers as the problem? 

* **Motivation for pursuing gain**: The point above also leads to my next point. While empirical results show that relative change of activation, formulated as spectral ratio and alignment ratio, are more indicative of performance degradation, what is the theoretical justification for this? For instance, QAT has mechanisms to overcome these activation rescaling as it is performed during training, and has the opportunity to compensate for these fluctuations. Are there other subtle mechanisms that this work is identifying?

* **Oscillations in Figures 3 & 4:** The oscillations in Figure 3 and more pronounced in Figure 4 warrant some attention. What is going on in Fig. 4? Why are $G_\ell, R_\ell$ oscillating between regular values between successive layers? This seems to be the case for all optimizers. 
* **Point of Scaling laws** While the analysis and the fit for the scaling laws are interesting, I was confused as to how this ties to the rest of the paper. Is it to show that $\rho$ for Shampoo is slightly higher than AdamW? What else do we learn from these scaling curves? 

* **Use of OLMo2 architecture:** The experiments are all performed on the same architecture family. The experiments are across different number of parameters, but the results and observations in this paper might be overfitting to a single architecture family. What is the justification? And would the authors expect these observations to generalize to other architecture families?

* **Optimizer hyperparameters:** It is mentioned that all hyperparameters for the optimizer were chosen based on the 50M model. Were these done in any consistent manner? For instance, following the muP recommendations?[3] If not, what is the argument for doing this?

### Other comments

* Two versions of the same paper, Panferov et al. 2025, are referenced
* Difficult to see what is going on with the components of $R_\ell$ i.e, the ABC decomposed terms. 
* MMR abbreviation is introduced several times.
* Column 2 in Table 1 is redundant, given the first column.

### References

[1] Huh, Minyoung, et al. "Straightening out the straight-through estimator: Overcoming optimization challenges in vector quantized networks." International Conference on Machine Learning. PMLR, 2023.

[2] Oh, Sangyun, et al. "Non-uniform step size quantization for accurate post-training quantization." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.

[3] Yang, Greg, et al. "Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer." arXiv preprint arXiv:2203.03466 (2022).

### Questions
See points under weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses a critical gap in the intersection of large language model (LLM) optimization and quantization: how optimizer choice impacts model performance when quantization (a key technique for efficient LLM deployment) is applied. It systematically investigates two major quantization paradigms—Post-Training Quantization (PTQ) (quantizing a fully trained model) and Quantization-Aware Training (QAT) (integrating quantization during training)—across LLMs of varying sizes (50M to 1.5B parameters) trained with six optimizers (AdamW, Muon, PSGD, Scion, Shampoo, SOAP).

The study first establishes well-tuned full-precision (FP) baselines using the OLMo2 architecture (modified with tied input-output weights and ReLU² activation) and the Chinchilla-optimal data-to-model ratio. It then evaluates how these FP models degrade under PTQ, and how optimizers perform when training quantized models from scratch via QAT. A key theoretical contribution is the ABC decomposition framework to analyze quantization error propagation, which reveals limitations of traditional outlier-centric metrics (e.g., Max-to-Median Ratio (MMR), Kurtosis) in predicting PTQ performance.

### Strengths
1. As the first work to rigorously explore optimizer-quantization interactions, it covers a broad scope—models from 50M to 1.5B parameters, 6 optimizers (AdamW, Muon, etc.), and both PTQ/QAT paradigms. This breadth avoids narrow conclusions and ensures results are tested across diverse scenarios.

2. It establishes robust full-precision baselines using the Chinchilla-optimal data-to-model ratio, detailed hyperparameter tuning (e.g., sequential sweeps on 50M models), and clear architectural modifications (OLMo2 with tied weights/ReLU²). Evaluation uses standard zero-shot benchmarks (PIQA, HellaSwag) and open tools (LM Evaluation Harness), ensuring reproducibility.

### Weaknesses
1. Your paper focuses on validating optimizer-quantization interactions using the OLMo2 architecture (with modifications limited to tied input-output weights and the ReLU² activation function), which provides a clear and controlled experimental base. We’re curious, have you considered extending these experiments to more mainstream LLM architectures like Llama or GPT-2 (or even Llama 3, which is widely used in industry)? We’d also be interested to hear the reasoning behind choosing OLMo2, and how you anticipate Shampoo’s quantization advantages might hold (or adapt) when tested on these other architectures, to further strengthen the generalizability of your conclusions.

2. The proposed \(R_\ell\) metric effectively addresses the limitations of traditional metrics like MMR and Kurtosis by capturing layer-wise activation error propagation, which is a valuable theoretical contribution. We noted that computing \(R_\ell\) for large models (e.g., 1.5B parameters) requires iterating through all layers’ activation tensors, leading to higher computational complexity than MMR/Kurtosis. In real-world deployment, users often need fast quantization outcome predictions (e.g., during preprocessing). We’re wondering if you have considered ways to optimize \(R_\ell\)’s computational cost, such as approximating it with key layers rather than all layers, and how you balance the metric’s predictive accuracy with the industry’s demand for low-cost tools?

3. Your QAT experiments use the QuEST scheme, which is a strong choice for stable low-precision training. Given that industry frequently adopts other QAT variants like GPTQ and AWQ (which differ in handling quantization noise and model adaptation), we’re curious if you’ve thought about how Shampoo’s performance might compare to other optimizers under these alternative schemes? We also wonder if testing a broader set of QAT methods could further validate whether Shampoo’s advantages are specific to QuEST or generalizable across common QAT paradigms.

4. Your scaling laws leverage Chinchilla’s full-precision optimal data-to-model ratio, which provides a consistent baseline for comparison. We note that quantization can alter a model’s information capacity, for example, 4-bit models may require more data to offset precision-related accuracy loss. We’re interested to hear your thoughts: how do you think reusing the full-precision data ratio might influence the assessment of Shampoo’s parameter efficiency? And would recalibrating the data ratio specifically for QAT be a valuable next step to confirm if Shampoo’s efficiency stems from inherent quantization robustness, rather than compatibility with full-precision settings?

5. You rightly point out that MMR and Kurtosis are limited by their focus on single-layer errors, which motivates the need for \(R_\ell\). We’re curious, have you explored the possibility of refining these traditional metrics (e.g., a weighted MMR that assigns different importance to attention vs. FFN layers) to better capture cross-layer error effects? If an improved MMR showed stronger correlation with quantization performance, how do you think it might complement (or compare to) \(R_\ell\), and could there be synergies between refining existing metrics and advancing new ones like yours?

6. Your conclusion thoughtfully highlights that optimizers performing well in full precision may not retain that advantage under QAT, a key insight for practical LLM deployment. We’re wondering if you’ve considered discussing the potential design direction of optimizers that excel in both full precision and QAT (e.g., quantization-adapted variants like a modified Lion optimizer)? We’d be interested to hear how such dual-excellence optimizers might align with your current findings, and whether you see this as a valuable direction for future work to build on your conclusions.

### Questions
In your full-precision training, you use the ClimbMix dataset (400B tokens) but do not analyze how data domain diversity (e.g., mixing technical vs. conversational text) interacts with optimizer-quantization performance. For instance, if a subset of ClimbMix has more outlier-prone features, might this disproportionately favor Shampoo? Did you conduct any ablation on dataset subsets to rule out data bias in your optimizer comparisons?

### Soundness
3

### Presentation
3

### Contribution
2
