# MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with $0.1k$ Parameters

- Decision: Reject
- Scores: 3, 5, 5, 5

## Abstract
In recent years, there has been a growing interest in Long-term Time Series Forecasting (LTSF), which involves predicting long-term future values by analyzing a large amount of historical time-series data to identify patterns and trends. There exist significant challenges in LTSF due to its complex temporal dependencies and high computational demands. Although the Transformer-based models offer high forecasting accuracy, they are often too compute-intensive to be deployed on devices with hardware constraints. On the other hand, the linear models aim to reduce the computational overhead by employing either decomposition methods in the time domain or compact representations in the frequency domain. 
In this paper, we propose MixLinear, an ultra-lightweight multivariate time series forecasting model specifically designed for resource-constrained environments. MixLinear effectively captures both temporal and frequency domain features by modeling intra-segment and inter-segment variations in the time domain and extracting frequency variations from a low-dimensional latent space in the frequency domain. By reducing the parameter scale of a downsampled $n$-length input/output one-layer linear model from $O(n^2)$ to $O(n)$, MixLinear achieves efficient computation without sacrificing accuracy.
Extensive evaluations across four benchmark datasets demonstrate that MixLinear attains forecasting performance comparable to, or surpassing, state-of-the-art models with significantly fewer parameters ($0.1K$), which makes it well-suited for deployment on devices with limited computational capacity.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors present an approach coined MixLinear to improve predictive performance for (multi-variate) time series problems with a long forecasting horizon. They put an emphasis on drastically reducing the necessary parameter count of their model compared to other existing approaches while maintaining or even improving on forecasting quality. For this, their proposed method decomposes the time signal in a number of steps into seasonal periods, trends as well as its frequency spectra. Through a learned mixing the information is combined to produce the forecast. Therein, information pieces are already shorter or subsampled to theoretically achieve near-linear complexity with respect to the number of time steps L in the historic window of a model's input sequence. In an experimental evaluation on well-known datasets like ETTh and Traffic, the proposed method is compared to other established Transformer-based forecasting approaches.

### Strengths
- The author touch an important topic of avoiding drastic overparametrization of (transformer-based) time series forecasting models

- An extensive comparison with other, current baseline models is performed

### Weaknesses
- You claim that "Transformer-based models offer high forecasting accuracy, they are often too compute-intensive to be deployed on devices with hardware constraints [...]". The reviewer feels that this claim is not substantiated in light of various pruning techniques (removing upto 99% of weights after training), knowledge destillation or neuron transplantation. Could you provide according proof? Your paper currently only considers the training phase, but not its later use in an inference setup.

- The manuscript claims "The high computational demands and large memory requirements of these models hinder their deployment for LTSF tasks on resource-constrained devices" - the authors do report in Section 4.3 on the efficiency of these models, but it is unclear to the reviewer how much effort went into optimizing these. If only MixLinear has been highly optimized and the others taken as is, the empirical evaluation has a consistent bias in favor of MixLinear.

- Carefully check the paper for grammatical erros as well as precise phrasing, examples include, but are not limited to
	* "On the other hand, the linear models aim [...]" (abstract) - the second 'the' is superfluous
	* "[...] overhead by employing either decomposition methods [...]" (abstract) - what does 'either' refer to here?
	* "[...] by modeling intra-segment and inter-segment variations [...]" (abstract) - segment is unclear, could you clarify this term?
	
- Fig. 1 shows an example of the Electricity dataset, which a) has not been introduced at that point and b) does not indicate if and how data has been normalized, making the MSE values hard to estimate. The same applies to the tables 1, 2 and 3 later in the manuscript's body. Could the authors improve the textual description?

- "[...] by decoupling channel and periodic information from the trend components [...]" - there are several prior works that attempt do achieve this. This includes, but is not limited to the following:
	* A. Weyrauch, et al., "ReCycle: Fast and Efficient Long Time Series Forecasting with Residual Cyclic Transformers," in 2024 IEEE Conference on Artificial Intelligence (CAI), Singapore, Singapore, 2024 pp. 1187-1194.
	* Heidrich, B., Turowski, M., Ludwig, N., Mikut, R., & Hagenmeyer, V. (2020, June). Forecasting energy time series with profile neural networks. In Proceedings of the eleventh acm international conference on future energy systems (pp. 220-230).
	* Pathak, J., Subramanian, S., Harrington, P., Raja, S., Chattopadhyay, A., Mardani, M., ... & Anandkumar, A. (2022). Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators. arXiv preprint arXiv:2202.11214.
	
	How does your approach compare to these or other works? Especially drastically reduces computational complexity by computing mostly in the frequency domain, achieving a time complexity of O(L log L), where L is the sequence length

- The authors are kindly requested to improve Fig. 2 by
	* using the correct mathematical symbols, e.g. for the set of real numbers, 
	* increasing font size, e.g. on the trend spectrum,
	* use full words without underscores,
	* define all mathematical symbols, e.g. LPF, in the caption, 
	* have a consistent direction of reading (currently a mix between bottom-top and left-right).

- Can the authors comment on how the frequency domain part of MixLinear differs from Fourier Neural Operators?

	Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). Fourier neural operator for parametric partial differential equations. arXiv preprint arXiv:2010.08895.
	
- Is it meaningful to consistently use the Landau-/Big-O notation when assessing the algorithm's complexities, e.g. p. 4 in the subsection "Segment Transformation"

- Have the authors considered to only work on the differences between time steps or the residuals of the time series and a simple smoothing average curve? A vast number of publications have identified this to be beneficial for predictive performance, training convergance as well as a (further) reduced parameter count.

- It seems meaningful that the sign of "Improvement" in Table 1 should be inverted due to the MSE being lower indicating a better model. Furthermore, since it is not consistently an improvement, does the term "Difference to best baseline", or similar, make sense?

- The title and manuscript body speaks of 0.1K parameter models, whereas it is closer to 0.2K - it seems to the reviewer that it would not take away from the paper's message by being more precise.
	
- In its current form the paper only reports optimal forecasting but does not consider a distribution across multiple seeds. Is MixLinear consistent in achieving high predictive performance? How large is the spread compared to other approaches?

- The work is motivated by attempting to find models that work in resource constrained environments. Can you make an example for a possible usage scenario? Would it be used to train the model, make predictions or both? In its current form the manuscript puts a lot of emphasis on evaluating the training phase in an environment with a lot of compute power (A100-80). How would your approach actually fair in your envisioned usage scenario?

- In the manuscript only time series forecasting problems with strong seasonality are considered. It seems to the reviewer that the approach would not work well for problems where this is not applicable. It seems meaningful to clearly spell out this assumption.

### Questions
- Can the authors comment on the suggested method's capability to capture peaks or other extremes of the time series well?

- Can you comment on how MixLinear's hyperparameters can be effectively found, e.g. cutoff values?

- Are you willing to release the source code of your method and its evaluation for the sake of reproducibility?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a lightweight time series forecasting model that employs trend segmentation, segment transformation, and a parallel approach of trend spectrum compression and trend spectrum transformation. The model aims to maximize performance improvements while minimizing the number of parameters and computational complexity. Through a series of experiments, the authors validated the model's effectiveness.

### Strengths
1. The overall approach proposed in this paper is feasible and achieves good results.
2. The authors reduce parameters and computational complexity by employing a two-stage downsampling method (periodic downsampling and trend segmentation).
3. The model demonstrates good transferability and generalization capability.

### Weaknesses
1. The foundation of this work is based on trend downsampling, which is the core idea of SparseTSF (ICML 2024 Oral)--namely, periodic downsampling and parameter reuse. The interpolation method in the frequency domain is based on FITS (ICLR 2024 Spotlight). The novelty of this paper is limited.
2. The performance improvements seems to be marginal. Specifically, except for the ETTh1 dataset, the performance on other datasets are not good enough compared to SparseTSF and FITS.
3. The experimental section of this paper lacks sufficient visualization of experimental results. For instance, what's the result of the model's training and convergence compare to other linear models? What is the predictive performance of the model?
4. Although the authors performed ablation studies, they did not explicitly explain the role of key components in the model. For example, what kind of information is learned by the temporal and frequency domain methods? In the temporal transformation, the segment transformation used by the authors is essentially similar to the operation of an MLP-Mixer, which mixes data along two directions. The interpretability of this operation is not clear.
5. The paper lacks a hyperparameter (such as w) search experiment, and the authors did not provide parameter information corresponding to the optimal results across all datasets (they only provided information for Electricity). Is the model's performance sensitive to w across different datasets?
6. In Appendix Table 7, the authors mentioned the filters with n=15 and n=19 points, but the length of the corresponding trend sequence is only 30 (for ETTh1). The FFT of the actual sequence should be conjugate symmetric, meaning it contains only 16 points of valid information. A 19-point filter introduces redundancy and high-frequency noise.
7. Another concern is about the frequency domain transformation. Temporal trend decomposition already removes the inherent periodic and detailed information in the data, which is a method that emphasizes the overall structure. Additionally, performing frequency domain transformations on sequences after periodic downsampling (where w is often large, e.g., 24) will cause severe aliasing effects for most time series. The authors should provide a theoretical analysis or empirical demonstration of how their method addresses or mitigates these aliasing effects.

### Questions
1. Regarding W3, more visualized results are expected to be provided.
2. Regarding W4, what's the different features  learned by the frequency and temporal domain methods?
3. Regarding W5, a more detailed introduction of parameter information setting should be added.
4. Regarding W7, the authors should provide a theoretical analysis or empirical demonstration of how their method addresses or mitigates these aliasing effects.  
5. Please check Figure 2. In the segement transformation part, the shape should be 4x4 -> 4x5 -> 5x4 -> 5x5. The current shape is 4x4 -> 5x5 -> 5x5 -> 5x5.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper propose MixLinear, an ultra-lightweight multivariate time series forecasting model which is specifically designed for source-constrained devices. The proposed MixLinear can effectively captures both temporal and frequency domain features with lower complexity. Meanwhile, MixLinear only needs 0.1k parameters with comparable performance.

### Strengths
1. MixLinear has comparable performance with only 0.1k parameters.
2. The proposed MixLinear is the first lightweight model that captures both temporal and frequency domain information.
3. MixLinear has little running time compared to other time series forecasting models.
4. The model is simple and easy to follow.

### Weaknesses
1.	The authors demonstrate the computational complexity of MixLinear is $O(n)$, but additional operations such as Fourier Transform introduce additional complexity. A more detailed discussion on the computational complexity of MixLinear would be better.
2.	In table 2, the authors report the running time of these models during, However, it would be better to report the running time in the inference phase.
3.	The experimental results in Tables 1 and 6 show that MixLinear performs competitively only on the ETTh1 and ETTh2 datasets, while its performance lags behind state-of-the-art models on other datasets. 
4.	Regarding to the number of parameters, although MixLinear needs only 0.1k parameters, I am curious about what kind of lightweight device cannot store around 10k parameters. If such a device exists, is it meaningful to use it for time series forecasting?
5.	Also, even though MixLinear has a small parameter count, I believe its computational complexity is not low. And I think the authors need to discuss the computational complexity in more depth.
6.	The superscripts and subscripts in Figure 2 need to be adjusted.

### Questions
In Figure 2, I noticed that MixLinear seems to only utilize the trend component while discarding the periodic component. I would like to ask the authors why the periodic component was discarded. Are there relevant experimental results that demonstrate that using only the trend component is more effective?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces MixLinear, an ultra-lightweight model for long-term time series forecasting (LTSF) that achieves comparable or better accuracy than state-of-the-art models while using only 0.1K parameters. The key innovation of MixLinear lies in its unique approach of capturing features from both time and frequency domains. In the time domain, it applies trend segmentation to capture intra-segment and inter-segment variations, while in the frequency domain, it reconstructs trend spectrums from a low-dimensional latent space. By reducing the parameter scale from O(n²) to O(n) for n-length inputs/outputs, MixLinear achieves efficient computation without sacrificing accuracy. The authors demonstrate MixLinear's effectiveness through extensive experiments on four benchmark datasets, showing it can achieve up to 5.3% reduction in Mean Squared Error compared to state-of-the-art models. The paper also demonstrates MixLinear's strong generalization capability across different datasets and its robustness in various forecasting scenarios, making it particularly suitable for deployment on resource-constrained devices.

### Strengths
The paper shows good originality by being the first to effectively combine time and frequency domain features in a lightweight LTSF model - a creative integration that differs from conventional single-domain approaches. The technical quality is demonstrated through experiments across multiple benchmarks, convincingly showing that MixLinear achieves comparable or better accuracy with just 0.1K parameters versus 1K-6M in baselines. The paper is clearly written. The work makes a good contribution by enabling accurate LTSF deployment on resource-constrained devices, while providing valuable insights about combining domain features efficiently.

### Weaknesses
- There are much more time series baselines to be compared with.
- The number and kinds of datasets that the experiments are based on are too limited, totally 4 and 3 of them are electricity-related datasets.

### Questions
For table 4, can you also show ETTh1 -> ETTh2, ETTh1 -> Electricity?

### Soundness
2

### Presentation
2

### Contribution
2
