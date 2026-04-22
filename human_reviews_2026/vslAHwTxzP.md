# SFdiff : Diffusion Model with Self-Generation for Probabilistic Forecasting

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Diffusion models have emerged as an effective approach for time-series probabilistic forecasting, aiming to generate future observations based on historical data through a denoising process. In this paper, we introduce self-generation technique designed to enhance the performance of conditional generation in time-series forecasting. Self-generation involves synthesizing not only future observations, but also historical data itself conditioned on the given historical context. While noise is often introduced during the observation process, our method can reduce the amount of noise in observed historical data, thereby enhancing forecasting accuracy. Additionally, to further boost forecasting performance, we incorporate classifier-free generation methods into conditional generation for time-series forecasting. In the experiment, we demonstrate that our method outperforms other condition generation methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles probabilistic forecasting for multivariate time series using a diffusion-based approach that departs from the common “predict-future-only” setup. The authors propose SFdiff, which reconstructs the entire sequence—past and future—during reverse diffusion. This “self-generation” step implicitly denoises the historical context so that errors and outliers in the conditioning window exert less influence on the generated forecast. The method is formalized with a score-matching objective and a sensitivity bound showing that whole-sequence conditioning is less susceptible to perturbations than future-only conditioning.

### Strengths
- **S1** The paper is easy to follow end-to-end, with crisp notation and a clean separation of method, theory, and experiments.
- **S2** The analyses on history/future masking, guidance strength, and sampling steps provide actionable takeaways for reproducing and deploying the method.

### Weaknesses
- **W1** The task is time-series forecasting, evaluation relies almost exclusively on CRPS. I would like to see complementary point-forecast metrics (e.g., MAE/MSE/RMSE/SMAPE) to assess accuracy alongside calibration.

- **W2** Qualitative results are shown only for the synthetic setups. It remains unclear how SFdiff behaves visually on real benchmarks. Add forecast trace plots and predictive intervals on several real datasets (e.g., Exchange/Electricity), including challenging noisy cases.

- **W3** Provide sensitivity to key hyperparameters (history/future weight, guidance strength, steps) and statistical significance tests (e.g., paired t-tests or bootstrap CIs) against strong baselines.

- **W4** The framework may benefit from longer contexts (potentially noisier histories), yet the effect of varying input/forecast lengths is not systematically studied. Run controlled studies with longer contexts and horizons to test whether SFdiff’s advantage widens as historical noise increases.

### Questions
See W1 to W4.

### Soundness
2

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
2

### Summary
This work presents a robust diffusion framework for probabilistic multivariate time-series forecasting that explicitly addresses noisy or unreliable conditioning histories. Rather than conditioning on a fixed past and generating only future values, the proposed SFdiff jointly reconstructs the past and predicts the future within the same reverse-time process. By doing so, the model “purifies” the historical window on the fly, leading to more stable forecasts. The authors provide a theoretical argument via an upper bound on sensitivity, indicating that the joint sequence score is less affected by perturbations in the history than a future-only score, and supply a compatible denoising score-matching training objective. They further integrate classifier-free guidance into score-based conditional modeling and show that, under self-generation, modest guidance improves calibration and sharpness. Across two synthetic dynamical systems and five real benchmarks, SFdiff generally achieves stronger CRPS_sum than prior diffusion, flow, and non-generative baselines. Comprehensive ablations detail the effect of history/future loss masking, guidance weights, and the number of sampling steps. The approach is promising for scenarios with corrupted contexts, though it requires careful hyperparameter tuning and retains the higher sampling cost typical of diffusion methods.

### Strengths
- The toy benchmarks are minimal yet representative, making it straightforward to diagnose where the proposed approach helps under perturbed histories.
- The writing is concise and well structured, allowing readers to grasp the core idea and theoretical claim without unnecessary detours.

### Weaknesses
- The ablation study is quite insufficient; for example, I did not see an ablation analyzing the contribution/effect of the historical sequence.
- Since this is a forecasting task, why not include **point-forecast metrics** (e.g., MSE, MAE, RMSE) in the evaluation?
- The model framework is unclear—for example, it is not specified what architecture is used for the denoising network, and many implementation details remain ambiguous.

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SFdiff, a diffusion-based model for probabilistic time-series forecasting. The method couples a "Self-Generation" mechanism that jointly denoises past and future segments with classifier-free guidance (CFG), and the experiments span two toy datasets plus five real-world datasets compared against a broad mix of classical and neural baselines.

### Strengths
The motivation resonates: noisy historical conditions genuinely hinder diffusion forecasters, and the self-generation story is communicated with helpful illustrations. I also appreciate the wide baseline coverage.

### Weaknesses
1. The related-work discussion acknowledges earlier full-sequence diffusion approaches such as TSDiff, yet the manuscript never spells out how SFdiff differs in architecture, loss design, or sampling; despite the γ-sweep and prediction/self-generation comparisons, the incremental novelty over these predecessors remains vague.

2. Theoretical support is also opaque: Theorem 3.1 invokes assumptions (A1–A3) and constants that are never defined, and no proof sketch accompanies the statement, so the promised robustness guarantee cannot be verified.

3. On the empirical side, the narrative that CFG "significantly reduces forecasting errors" conflicts with Table 1/3, where Solar performance worsens once CFG is applied; the text needs to reconcile or explain this divergence.

4. Reproducibility concerns compound the issue: the repeatedly promised "Table 5" with dataset statistics and hyperparameters never appears, leaving key experimental information missing.

### Questions
1. What tangible distinctions separate SFdiff from TSDiff and other total-sequence diffusion models, and can ablations quantify the incremental benefit of γ-weighting and self-generation?

2. What exactly are assumptions (A1–A3) and the constants in Theorem 3.1, and how do they connect to the VP-SDE sampler used in practice? A proof or detailed sketch is necessary.

3. Could you supply the missing Table 5 (or an equivalent appendix) that records model architectures, diffusion steps, training schedules, compute budgets, and tuning protocols for every method?

Beyond addressing those questions, it would help to clarify why CFG degrades Solar results while improving others, ideally with diagnostics that measure the claimed purification of historical inputs. Quantitative evidence of the purification effect and high-level pseudocode would also improve the presentation.

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
This paper aims to enhance the performance of conditional diffusion models for time series forecasting, by considering the intrinsic noise within historical context. Specifically, the paper proposed SFDiff that reconstruct the full sequence, including both historical part and the target part, instead of only predicting the target part, given the historical part. The paper claims that, in this way, high-frequency anomalies are largely reduced, and thereby minimizing their impact on forecasting. Experiments over 5 datasets and 12 baselines demonstrate lowest probabilistic forecasting error of the proposed method. Case studies illustrate the anomaly deduction effect. Ablation studies discuss the performance sensitivity against different classifier-free guidance scale and diffusion steps.

### Strengths
1. This paper investigates from the perspective that reducing influence of anomalies within look back window by treating them in part of the generation target, which is interesting.
2. The experiments have validated this crucial claim, well supporting the idea that the anomalies can be reduced.
3. The paper is well-structured. Experiments are extensive.

### Weaknesses
1. Theorem 3.1 is not rigorous. A tighter upper bound does not necessarily lead to a strictly lower function value. (A1)-(A3) is not mentioned throughout the paper.
2. Baselines are old. The authors should consider newer diffusion-based generation models like TimeDiff, TSDiff, MG-TSD, TMDM, NsDiff, etc.
3. The author should discuss more on the difference and connection between the proposed method and TSDiff, which employs a similar technique named observation self-guidance.
4. Formatting issue: The equations are not numbered. Line 246, 247 out of margin.

### Questions
Is the performance gain of the model significantly and positively correlated with how anomalous the dataset is?

### Soundness
1

### Presentation
2

### Contribution
2
