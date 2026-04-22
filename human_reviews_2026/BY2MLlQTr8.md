# Enhancing Extreme Weather Forecasting via Dynamically Weighted MSE

- Avg Score: 4.80
- Decision: Reject
- Scores: 6, 4, 4, 4, 6

## Abstract
Data-driven weather forecasting empowered by deep learning has shown superior performance compared to traditional physics-based dynamical models. However, conventional training objectives (like Root Mean Squared Error (RMSE)) primarily focus on minimizing average prediction errors, often resulting in oversmoothed forecasts that fail to capture critical extreme weather phenomena, including heavy precipitation, hurricanes, and other high-impact events. To overcome this limitation, we propose a robust loss function, named Dynamically Weighted MSE (DW-MSE), that adaptively reweights training samples to better learning on extreme weather events. Specifically, we introduce a dual-branch meta-network alongside the prediction network to dynamically generate sample weights: one branch captures spatiotemporal dependencies across climate variables, while the other learns from training losses. Guided by a small set of validation samples, the meta-network can be jointly optimized with the prediction network via an efficient bi-level optimization strategy, which provides the fast convergence with the approximated first-order information. Overall, our framework is able to accurately identify and assign greater importance to extreme weather samples without manually designing reweight function and any prior knowledge. Extensive experiments in both training-from-scratch and fine-tuning climate models demonstrate that Meta-MSE consistently outperforms existing approaches in forecasting extreme weather.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the challenge that current deep learning weather forecasting models usually produce over-smoothed predictions and perform poorly on extreme weather events. The authors propose a new loss function, Dynamically Weighted MSE (DW-MSE), which introduces a dual-branch meta-network to dynamically assign sample weights, enhancing the learning of extreme samples. One branch of the meta-network captures spatiotemporal relations among climate variables, while the other focuses on training loss information. With an efficient bi-level optimization framework (using an approximate first-order method and avoiding expensive second-order computations), the model learns to up-weight extreme weather samples without manual weighting or human prior. Experiments on both regional and global climate prediction tasks show that DW-MSE surpasses existing loss functions (RMSE, EXloss) in overall accuracy, in predicting extremes, and achieves faster convergence, with more interpretable weighting.

### Strengths
The paper proposes a flexible, adaptive loss function which automatically emphasizes extreme weather samples without hand-crafted weights or expert prior, bringing significant improvement in extreme forecasting—a highly practical advantage for real disaster warning applications.

### Weaknesses
The “top/bottom percentile” approach for constructing the validation set is under-explained, with no details about the threshold choice, its suitability for all variables/regions, or the risk of bias if extremes are very sparse. In reality, the boundary of what is “extreme” is often ambiguous, making reproducibility and fairness a problem.

Only RMSE and EXloss are included as baselines, which are relatively “basic” weighting schemes. Newer or stronger losses (like EVloss, EVT-based approaches) are missing. Also, the impact of each meta-network branch and bi-level optimization step is not ablated thoroughly (e.g., detailed effects on different types of extreme events), limiting the persuasiveness of claims. In addition, the metrics of ablation experiments should include extreme value metrics.

### Questions
How are the top/bottom percentiles chosen? Is there a universal standard across variables/regions? In cases with sparse extremes, how do you ensure the validation set is meaningful and the results are generalizable?

 Can you add comparisons with more diverse/extreme-specific loss functions and provide more thorough ablation (e.g., per-branch effects, per-extreme-type results) to clarify the precise sources of DW-MSE’s advantage?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes DW-MSE, a meta-learning loss for data-driven weather forecasting that dynamically reweights training errors to emphasize extremes. A dual-branch “weight network” produces a spatial weight map per sample: one branch ingests the per-grid loss map, the other uses self-attention over raw climate fields. The predictor and weight network are trained with a bilevel scheme using a first-order meta-gradient; the validation signal is computed on masked “extreme-only” grids. Experiments on ERA5 (regional fine-tuning and global training from scratch) report gains over RMSE and ExLoss on RMSE/ACC and reduced RQE bias, with ablations showing both branches help and with faster convergence curves.

### Strengths
- Clear motivation and simple interface: The method slots into existing forecasting models and targets a widely observed “over-smoothing” failure mode. The dual-branch design is sensible and well explained. 
- Fine-grained, spatial weighting: Producing an H×W×V weight map (rather than a single sample scalar) is a natural refinement for geophysical grids. 
- First-order bilevel optimization: Avoids Hessian-vector products yet still improves extremes; the derivation and algorithm are easy to implement. 
- Consistent empirical gains in shown experiments

### Weaknesses
1.Baselines are too narrow and comparisons are limited to RMSE and EXLoss. A more detailed similar losses in the literature should be discussed, and more baselines should be added to claim state-of-the-art performance.

2.Evaluation scope skews to limited variables. The study focuses on z500, t850, t2m, u10, v10,important but not where extremes most matter operationally (e.g., precipitation, wind gusts, tropical cyclone metrics). A detailed list of experiment results of more variables should be added at least in the appendix.

3.“Extreme-only” validation set design may bias learning. Optimizing the weight network solely on top/bottom ρ-quantiles risks over tuning to tails and neglecting structure near but outside the mask. Please check that non-extreme skill (e.g., mean-state bias) does not degrade. 

4.Computational overhead not quantified. The paper argues “efficient” first-order meta-learning, but does not report wall-clock, FLOPs, or GPU memory deltas vs. RMSE/ExLoss. The dual-branch attention over S=H×W tokens could be costly. Please add training time/epoch, total hours, and VRAM for all methods. 

5.The ablation study is only done on one variable, please add ablation experiments on more variables to show the comprehensive view of the branches.

### Questions
1. The paper cites EVLoss (Meo et al., 2024) but does not include it in experiments. Was it omitted for implementation reasons?
2. The experiments are restricted to five standard variables (z500, t850, t2m, u10, v10). Have the authors evaluated DW-MSE on precipitation, surface pressure, or wind gusts where extreme behavior is most pronounced?
3. How sensitive are the results to ρ, and does the non-extreme skill remain stable?
4. Could the authors report wall-clock training time, FLOPs, or memory usage versus RMSE and ExLoss to substantiate the efficiency claim?
5. Table 2 reports ablations only on z500. Did similar patterns hold for other variables?
If the author can address my concerns I am happy to raise my score.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a weighted MSE scheme to improve the accuracy of extreme weather forecasts, particularly by addressing the oversmoothing issue in weather models.

### Strengths
The paper clearly explains a common problem in weather prediction and offers a simple solution. The set-up is explained in an easy to understand manner.

### Weaknesses
- It doesn't become clear to me why the setup makes the network aware of extremes, the weights are according to forecast error, which might in some cases correspond to extreme events, but could also just be difficult ot predict terrain or something completely different.
- It would be interesting to show how many data points need to be in the validation set for the method to work correctly.
- While the error metrics presented are interesting, there is such a large emphasis on the over-smoothing problem in the motivation of the paper, that I would have expected metrics related to smoothness or at least a stronger focus on the extremes (there's only some average quantile error information in the appendix and one analysis).
- I like that multiple areas of the world are considered, but the three selected are still a bit of an odd sub-sample, which isn't discussed enough (like e.g. what would you expect in other areas of the world).
- I believe some form of skill score would be better for Figure 5. It's hard to see any differences in these plots between the methods.

### Questions
Why does the setup make the network aware of extremes? The weights are based on forecast error, which might in some cases correspond to extreme events, but could also reflect difficult-to-predict terrain or something completely different.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Current deep learning weather models, while powerful, are hindered by standard loss functions like RMSE that produce oversmoothed forecasts and fail to accurately predict critical extreme weather events.

To address this, the authors propose DW-MSE (Dynamically Weighted MSE), a novel meta-learning framework that uses a dual-branch network to automatically and adaptively assign higher importance to extreme weather samples during training. This eliminates the need for manual, pre-defined weighting schemes used in prior work.

The method is optimized efficiently with a bi-level strategy and is shown through extensive experiments to consistently improve extreme weather forecasting performance, outperforming existing approaches.

### Strengths
1) It directly addresses a well-known and significant weakness in current deep learning weather models—their failure to accurately forecast high-impact extreme weather events due to oversmoothing from standard loss functions.

2) The core strength is its move away from manually designed, rigid weighting functions. The dual-branch meta-network automatically learns how to assign importance to samples, making it more flexible and potentially more effective than prior heuristic approaches.

3) The framework does not require extensive pre-defined knowledge about what constitutes an "extreme event" in the data, allowing it to discover and adapt to complex patterns inherently.

### Weaknesses
1) The proposed framework is significantly more complex than a standard model with a simple MSE loss. It requires training and coordinating two networks (the prediction network and the dual-branch meta-network) instead of one, which could make implementation and debugging more challenging. The meta-network's optimization is "guided by a small set of validation samples." The performance of the entire system could be sensitive to how this validation set is constructed and whether it is truly representative of the extreme events the model needs to learn.

2) While the goal is to improve extreme event prediction, a potential risk is that the model might overfit to these events at the cost of degrading performance for more common, "normal" weather patterns. The introduction states it maintains "strong overall forecasting performance," but this is a key trade-off to monitor.

3) The experiments contain no baselines, training cost and hyper-parameter search report.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a new method called dynamically weighted mean squared error (DW-MSE) to improve weather forecasting, particularly for extreme events. Traditional deep learning models often produce overly smooth predictions because they minimize average errors, which causes them to miss rare but severe events. DW-MSE addresses this by automatically assigning greater importance to extreme weather samples during training through a dual-branch system that learns from both prediction errors and the weather data itself. This adaptive weighting process is optimized efficiently without relying on manual adjustments or prior knowledge. Experiments show that DW-MSE improves both regional and global forecasts, capturing sharper and more accurate extreme weather patterns while maintaining strong overall performance.

### Strengths
- The paper is in general well written and nicely structured. 
- The paper convincingly shows that MSE-trained models smooth out extremes and miss high-impact events.
- The authors present a simple but effective idea, that is, to learn a spatial weight map for MSE instead of hand-crafted weights, so it plugs into any forecaster models with minimal code changes.
- Clear indication that the dual-branch meta-net (using both loss maps and raw fields with self-attention) is best (both braches helps and together is the best).
- First-order update avoids Hessians and keeps the method effectively trainable.
- Bias maps and regional cutouts make the "less smoothing, sharper extremes" claim easy to verify.
- Methods shows faster and steadier training than plain MSE.

### Weaknesses
- The results seemingly rely on an "extremes-only" validation set. The quantile threshold is under-specified and sensitivity is not analyzed.
- The theoretical contribution is somewhat limited. There are no stability or convergence guarantees for the learned weights. The first-order choice is justified empirically only.
- Baselines seem to be quite narrow. There are only few comparisons to recent probabilistic or generative approaches aimed at extremes.
- Compute cost is not reported (GPU-hours, memory, wall-time) despite the extra outer loop. Hence it remains unclear how it would scale.
- Possible trade-off against non-extreme performance is not quantified beyond RQE.

### Questions
- What quantile threshold defines the extremes in the validation set, and how sensitive are results to it across variables and regions?
- What is the training overhead vs. RMSE and EXLoss (GPU-hours, peak memory, wall-time)?
- Can this be combined with probabilistic forecasters or ensembles, and does it improve spread–skill?
- You may add histogram- or log-binned residual analyses to show the distribution of error magnitudes, not just means. Include per-quantile curves and spatial error maps to reveal where large errors concentrate.

### Soundness
3

### Presentation
3

### Contribution
2
