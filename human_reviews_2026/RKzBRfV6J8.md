# Knowing When to Quit: Probabilistic Early Exits for Speech Separation Networks

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4, 4

## Abstract
In recent years, deep learning-based single-channel speech separation has improved
  considerably, in large part driven by increasingly compute- and parameter-efficient
  neural network architectures. Most such architectures are, however, designed with a
  fixed compute and parameter budget and consequently cannot scale to varying compute
  demands or resources, which limits their use in embedded and heterogeneous
  devices such as mobile phones and hearables.
  To enable such use-cases we design a neural network architecture for speech separation
  and enhancement capable of early-exit, and we propose an uncertainty-aware
  probabilistic framework to jointly model the clean speech signal and error variance
  which we use to derive probabilistic early-exit conditions in terms of desired
  signal-to-noise ratios.
  We evaluate our methods on both speech separation and enhancement tasks where we
  demonstrate that early-exit capabilities can be introduced without compromising
  reconstruction, and that when trained on variable-length audio our early-exit
  conditions are well-calibrated and lead to considerable compute savings when used to
  dynamically scale compute at test time while remaining directly interpretable.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper deals with the problem of single-channel speech separation, a crucial task in telecommunication, hearables, and other speech processing applications. Recent deep neural network models have achieved strong performance in this area based on commonly used metrics and loss functions such as SI-SNR and uPIT. However, neural networks with fixed computation paths may be inefficient in real-world scenarios where input signals often contain silence or non-overlapping speech. To address this, the authors propose a novel metric and loss function based on Bayesian and GMM formulations, combined with an early-exit mechanism. They extend the SepReformer architecture integrating these components to support resource-adaptive computation. Experiments on widely used benchmarks like LibriMix and WSJ0Mix demonstrate that the model achieves competitive performance comparable to SepReformer, and a favorable accuracy-efficiency trade-off with adaptive computation.

### Strengths
The proposed model demonstrates strong speech separation performance comparable to the state-of-the-art SepReformer, while offering support for adaptive computation and maintaining relatively low computational and memory costs.


The work provides a mathematically grounded formulation of speech separation objectives and metrics using GMM theory, offering potential alternatives to existing dominant approaches.


The evaluation includes multiple speech separation models and benchmarks, adding rigor and relevance to the empirical results.

### Weaknesses
The motivation and evaluation are not well aligned. The paper emphasizes potential computational savings in speech separation scenarios involving non-overlapping speech or silence; however, the evaluation only measures overall end-to-end computational efficiency. It does not quantitatively assess the specific advantages in those targeted scenarios.

The learning objective restricts the generality of the proposed framework. While metrics such as uPIT or SI-SNR can be readily applied to most speech separation architectures that output raw signals, the proposed early-exit SNR objective additionally requires the network to produce a variance output. This constraint limits applicability, as it necessitates architectural components explicitly designed for variance computation.

The reported performance improvements over existing state-of-the-art methods are marginal, and the situations where the proposed approach would provide a clear advantage are not convincingly demonstrated. In particular, the paper lacks experiments on realistic deployment settings, such as mobile or edge devices, where adaptive computation would be most beneficial. Moreover, under comparable computational budgets, the proposed model achieves slightly lower accuracy than SepReformer.

### Questions
For better readability among non-experts in Bayesian modeling, it would be beneficial to include a brief introduction to Bayesian Gaussian models, particularly the Normal–Inverse-Gamma (NIG) family, which appears to underpin the proposed probabilistic speech modeling framework. A concise explanation of how this family models both mean and variance uncertainty would make the paper more accessible and improve conceptual clarity.

The post-calibration procedure also seems to play a crucial role in the method’s generalizability. However, it is described as a form of training on validation data, effectively learning the variance distribution from the validation set. This raises concerns about potential data leakage or overfitting. If the model’s performance relies heavily on this post-calibration step, it would be important to clarify whether the approach generalizes well without it.

The paper introduces several architectural modifications on top of SepReformer, such as replacing the Transformer with a Linear RNN and using the Snake activation function. It would strengthen the paper to include ablation studies quantifying the individual contributions of these design choices to the overall performance.

The 2-speaker separation benchmarks on datasets such as WSJ0Mix and LibriMix are largely saturated, with SI-SNR values above 20 dB already reaching perceptual saturation. Therefore, it would be valuable to discuss whether the reported improvements, though numerically measurable, are practically significant in real-world speech separation scenarios.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper propose a neural network for single channel speech separation and enhancement.
The proposed PRESS algorithm is capable of early-exit with the ability to output uncertainty-aware probabilistic early-exit parameter. 
They test it on both speech separation and enhancement tasks and shows very good results without compromising reconstruction (state of the art results).

### Strengths
1. The idea of early exit is novel is interesting, and the SNR based criterion is also important to the field

2. When adding more exits it save the results of the latest exit performance, and the last exit get closer results to the state of the art results.

3. The proposed method can be use in both separation task and enhancement task which is very useful. 

4. The proposed method introduce the Student t-likelihood to use as likelihood measure.

### Weaknesses
1. The results of the proposed architecture on early exits doesn't achieves state of the art results 

2. The calculation of the inverse-gamma variance at the head is global for the whole reconstruction, in real audio case one scalar is not enough to express the uncertainty

3. There may be a problem if each exit peak its own permutation since it can choose different speakers in the following exits which can lead to degradation in the training

### Questions
1. Can you please provide more results on speech separation tasks, i.e. on other dataset (not only WSJ-mix)?

2. Can you evaluate the proposed method on more than two speakers (3 or more)? 

3. Could you consider more perceptual metric for evaluation such as STOI and PESQ?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes PRESS, a probabilistic early-exit framework for speech separation and enhancement networks that enables dynamic computation scaling. The method introduces a neural architecture with multiple exit points, each predicting a clean speech signal and an error variance parameterized by an inverse-gamma distribution to model uncertainty. Evaluated on standard benchmarks, PRESS achieves performance competitive with state-of-the-art static models while providing interpretable, SNR-based exit conditions that can be calibrated on validation data for efficient inference.

### Strengths
- The paper demonstrates high originality. While early-exit mechanisms and probabilistic deep learning are known concepts, their application to speech separation is novel. The formulation of a probabilistic model that jointly estimates the clean speech signal and its error variance, parameterized by an inverse-gamma distribution, allows the authors to derive a principled early-exit condition. This is different from common heuristics like entropy or difference between successive blocks. The proposed "unified exit-SNR" condition, which combines SNR, SNR improvement, and a loudness condition, is a clever solution to the problem of silent segments, further showcasing the depth of the original contribution.
- The proposed PRESS is technically sound, evidenced by rigorous experimental design and compelling results. The paper includes thorough ablations that validate core design choices (e.g., the Student-t likelihood vs. SI-SNR, joint permutation invariance). The models are evaluated on three standard benchmarks (WSJ0-2mix, Libri2Mix, DNS Challenge), demonstrating the generality of the approach for both separation and enhancement tasks. The results show that PRESS achieves state-of-the-art performance at its deepest exit points.
- The paper is very well-written and clearly structured. The motivation -- enabling dynamic computation for heterogeneous devices, is immediately established. The probabilistic framework is explained step-by-step, with a clear derivation of the likelihood and the subsequent SNR distributions, making the theoretical contribution accessible. The results are presented effectively with tables and informative plots, such as the performance-vs-compute graph (Figure 3) and the calibration curves (Figure 4), which directly illustrate the core advantages of the method. The appendix provides valuable supplementary material, including convergence analysis and implementation details.

### Weaknesses
- The ablation study in Table 1 shows that the proposed Student-t likelihood performs on par with the standard SI-SNR loss. While the probabilistic framework is elegant and enables the exit condition, the paper does not demonstrate that this formulation provides a training advantage. The core contribution of the probabilistic model is its utility for uncertainty-aware exiting, but if a model trained with standard SI-SNR and a heuristic exit condition (e.g., based on the norm of the change between layers) performed similarly, the necessity of the more complex probabilistic training objective would be questioned.
- The paper's most significant limitation is the modelling of a single, global variance parameter $\sigma^2$ per exit for the entire audio segment. It must process the entire segment (e.g., 4 seconds at training, variable at test) to compute a global variance before any exit decision can be made. This undermines the goal of "fine-grained dynamic compute-scaling" for live applications, as the compute savings are only realized at the segment level, not within a segment.
  - To truly meet its stated goal, the authors should explore a time-varying variance model, predicting a frame-level or chunk-level variance $\sigma_t^2$ at each exit point. The early-exit condition could then be evaluated on a rolling window (e.g., the past 500ms). This is a non-trivial but crucial extension that would make the system causal and demonstrate true fine-grained, adaptive computation.

### Questions
The paper shows the performance available at each exit point, but not the performance achieved when the probabilistic exit condition is actively used. What is the average computational cost (e.g., in GMAC/s) and the corresponding average SI-SNRi when processing the entire WSJ0-2mix test set with your proposed exit condition (e.g., with a target of t=20 dB and p=0.9)?

 In Equation 7, the unified exit condition is defined as the maximum of the individual complementary CDFs. Could you provide more intuition for why the maximum is the right operator here? Was there an ablation comparing this to other operators (e.g., a weighted average or a product of probabilities)?

### Soundness
3

### Presentation
4

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
The paper proposes PRESS, a technique for performing speech separation and enhancement that can dynamically scale to various compute budgets. They achieve this by designing network architectures equipped with “early exits”. These exits are driven by probabilistic SNR (&SNRis) based criterion, and the network stops at the first depth where the probability of SNR > a set target. Results on WSJ0-2mix, Libri2Mix, and DNS Challenge datasets were presented.

### Strengths
1. The problem is well motivated. Having the ability to adaptively exit from the computational chain is especially useful on compute and energy-constrained devices such as VR headsets, hearbles, and smart speakers. So studying this seems a useful task for practical deployments.

2. Basing the exit criterion on SNR-type variants seems intuitive and logical, given that they have easy-to-tune parameters.

3. Experimental results at the deepest exit are competitive with shown baselines. Abaltions are helpful.

### Weaknesses
1. Novelty: While the paper tries to solve a practical problem, it seemed like an incremental improvement to previous works (e.g., SepReformer) and did not come across as novel enough.
The architecture largely follows SepReformer (early split, transformer stack, CLA and other layers), with the main additions being extra exit heads and a standard likelihood derived from conjugate priors. Also, early exit itself is a known idea, and turning the SNR threshold into an exit confidence feels like a natural application of that likelihood.

2. Empirical results: While the depth-wise exits show incremental improvements (Table 2), they are not surprising. Moreover, why not simply use SepReformer Tiny/Small, which has fewer parameters yet achieves similar (and in some cases better) performance? Early exits don’t really reduce the on-device model size anyway; the full network (with the exit heads) must still be stored, so gains are mainly runtime latency or energy saving on easy inputs, not footprint. But even the #GMAC/s reported don’t clearly beat other baselines. So it’s unclear to me what the concrete gains of the proposed approach are relative to baselines. 

3. Unified early exit: The reasoning behind using a combined SNR exit criterion (Eq. 11)  when at least one of them exceeds the set threshold seems heuristic and is not quite convincing. What if you make all three exceed the threshold? Also, I wonder if the thresholds are always triggered mostly by a particular exit. Ablations corresponding to these are missing.

4. Conjugate prior:  The choice of an inverse-Gamma prior on $\sigma^2$ seems to be made mainly for conjugacy and convenience, and there’s little justification for why that is the case. It would help to compare results against a simpler case, such as a Gaussian with both mean and sigma learned as point estimates, removing the need for marginalization.

5. SNR-centric: The paper relies heavily on the SNR-style variants for both the exit criterion and evaluations. It would help the reader to see whether these SNR-based exit choices are a key design choice. Also, reporting with other perceptual metrics, such as PESQ, STOI would help make better judgment of the method.

6. Writing: The paper is somewhat of a hard read; Section 3.1, in particular, feels under-motivated and confusing. For instance $x$ is not defined in Eq. 1, and similarly for T in Eq. 3 (the signal length). At line 200, in Eq. (3), doesn’t the second-to-last term actually penalize over-estimating the variance?

Minor comments:
L234: correction at p(SNR(x, \hat{x}) \geq t) \geq p to Pr(.)

### Questions
Questions:
I would like the authors to comment on these:
1. Eqs. 8 and 9 approximate SNR by 1 + $\lambda/T$ where $\lambda$ is the non-centrality parameter when T tends to infinity. How does this behave for shorter time windows and how does this hold in real scenarios?
2. Other than simplifications, what is the authors’ rationale behind using the inverse-gamma as a conjugate prior? Anything specific about the variance that motivates the prior? Does restricting the noise model to the Normal-Inverse-Gamma family limit performance? Because once the classes are restricted, the performance may be bounded? Is this the case? 
3. A stretch, but can we see this depth-quality as analogous to the sampling time-quality in diffusion models? i.e, can diffusion models naturally achieve this dynamism simply by varying the number of sampling steps? Can the authors draw any parallels between the two?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
PRESS introduces an early-exit architecture for speech separation that enables dynamic compute scaling by allowing the network to stop processing at different depths. 
This is achieved by a probabilistic framework that models the predicted SNR distributions, exiting when the predicted SNR , SNRi or a loudness condition exceeds a target threshold with specified confidence. 
The proposed method offers competitive results with state-of-the-art models on WSJ0-2mix, Libri2Mix, and DNS2020 while enabling compute reduction.

### Strengths
This paper is the first to apply uncertainty-aware probabilistic modeling specifically for early-exit decisions in speech separation. 
One great advantage of the proposed method is some interpretability in the chosen early exit strategy. 
The paper has strong empirical results with respect to the SotA non early exiting models on 3 different datasets including DNS challenge which features speech enhancement.

### Weaknesses
This work misses a crucial comparison with another work which also explored early exit for speech separation already in 2023: 

Bralios, D., Tzinis, E., Wichern, G., Smaragdis, P., & Le Roux, J. (2023, June). Latent iterative refinement for modular source separation. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5). IEEE.

This work goes into the direction outlined by the authors in the future possible works where instead of using different blocks one uses iterative models. 
Another key different between the two works is that the early exiting mechanism is fundamentally different as PRESS relies on estimated SNR. 
However I feel that this work should be discussed and maybe compared against e.g. by having a baseline system adopting shared parameters and gating-based stopping. 

Another work is mentioned which explored early exit for speech separation: 

Chen, S., Wu, Y., Chen, Z., Yoshioka, T., Liu, S., Li, J., & Yu, X. (2021, June). Don’t shoot butterfly with rifles: Multi-channel continuous speech separation with early exit transformer. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 6139-6143). IEEE.

I think another drawback of this paper is that the proposed early exit strategy is not compared with the strategy adopted in this latter paper. Does proposed probabilistic SNR work better than Euclidean norm difference ?
Which exit criterion transfers better across domains? is PRESS estimation robust ? 
Right now these questions are unsolved and it is unclear how significant is the proposed method compared to the previously proposed two. 

Also the paper lacks other ablations to justify the proposed probabilistic early exiting framework. 
For example what if instead of the probabilistic framework I use simply a supervised predictive head to estimate the SNR ? 
Or what if only p(SNR > t) is used as an early exit condition ? Figures 1and 6 compare the different strategies but there are no tables. 

Unfortunately due to the lack of these experiments the paper as it is right now is not suitable for publication. While the proposed strategy seems very strong it is unclear if it is more effective than others (and possibly simpler) strategies.

### Questions
I ve put questions in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2
