# AlignDiff: Exploiting Model-Intrinsic Information for Better Preference Data Selection

- Decision: Reject
- Scores: 6, 4, 6, 2, 6

## Abstract
Aligning large language models with human preferences remains challenging, and the quality of preference data is critical for effective alignment. Existing large-scale datasets often introduce noise and distribution shifts, limiting model performance. To address this, we propose AlignDiff, a preference data filtering framework driven by intrinsic model signals. AlignDiff first identifies samples with clear preferences using both positive and inverse signals, then prioritizes the more challenging samples based on the average negative log-likelihood gap, encouraging the model to learn richer information from them. Across multiple models and benchmarks, AlignDiff consistently outperforms the other seven baselines. On AlpacaEval 2.0, training on only 50\% of the data selected by AlignDiff nearly doubles the performance of LLaMA-3-8B-SFT compared to training on the full dataset. The data filtered by AlignDiff preserves the length gap distribution while achieving a more favorable external reward margins distribution, and difficulty-based curriculum learning further enhances model performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces AlignDiff, a two-stage, intrinsic-signal preference data selector for DPO:

Alignment Discrepancy (RAD) combines positive and inverse implicit reward margins to keep clear-preference pairs, flip clearly inverted ones, and drop ambiguous ones.

Difficulty calibration (ANG) favors hard chosen responses (and relatively easy rejected responses) to avoid the DPO squeezing effect.

### Strengths
Positive vs inverse signals are complementary; RAD is an elegant, model-intrinsic indicator of preference consistency. 

Strong empirical with cross-model gains; preserves length-gap distribution while shifting external reward margin right; solid ablations & training-dynamics analyses.

### Weaknesses
Current results focus on UltraFeedback → AlpacaEval 2.0 / MT-Bench. Please expand to diverse distributions—e.g., multilingual prompts (zh/es), safety red-team splits, and domain-shift (OOD) sets. Add a human evaluation slice (stratified 200–500 items, double-annotated) to validate that gains aren’t artifacts of automatic metrics and to reduce the risk of metric gaming.

The naive IM&EM combination underperforms IM. Consider an aware fusion that (i) calibrates score scales, (ii) debiases for response length, and (iii) trains a lightweight selector (e.g., logistic/GBM) over both features.

Add multilingual, safety-critical, and OOD stress suites, and report human preference subsamples.

### Questions
see weaknesses

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
This paper introduces AlignDiff, a diffusion-based framework for large language model (LLM) alignment that replaces traditional reward modeling or preference optimization with diffusion-guided preference learning. Instead of directly optimizing discrete token probabilities, AlignDiff models the alignment process as a continuous diffusion trajectory, progressively refining model responses toward human-preferred behaviors. The approach allows for smoother optimization, implicit preference aggregation, and better exploration of the response space. Experiments on alignment benchmarks such as AlpacaEval2 and WildBench show that AlignDiff achieves comparable or superior performance to DPO and RRHF while exhibiting higher response diversity and training stability. The paper also provides theoretical justification linking diffusion noise levels to preference strength and demonstrates that AlignDiff mitigates mode collapse common in iterative alignment methods. Overall, it presents a conceptually novel and empirically strong direction for aligning LLMs through continuous preference dynamics rather than discrete reward modeling.

### Strengths
* Conceptual novelty: The paper introduces a fresh diffusion-based perspective for alignment, modeling preference learning as a continuous refinement process rather than discrete reward optimization.

* Empirical and theoretical rigor: Provides both strong benchmark performance (e.g., on AlpacaEval2, WildBench) and theoretical insights linking diffusion noise to preference strength, showing clear advantages in stability and diversity.

### Weaknesses
* Complexity and interpretability: The diffusion-based formulation adds considerable algorithmic and computational complexity compared to simpler preference optimization methods like DPO or RRHF.

* Limited empirical diversity: Experiments focus mainly on text-based instruction-following tasks; evaluation on reasoning, coding, or multimodal benchmarks would better demonstrate generality.

* Ablation and efficiency analysis: The paper lacks detailed ablations isolating the contribution of each diffusion component and provides limited discussion of training efficiency and resource overhead.

### Questions
1. How does AlignDiff scale computationally compared to DPO or SPIN when applied to larger models or longer context windows?

2. Could the authors clarify whether the diffusion process can be adapted for online or reinforcement-style feedback, rather than relying solely on static preference datasets?

### Soundness
2

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
3

### Summary
This study investigates a data filtering method based on intrinsic rewards. To enhance the accuracy of the filtering process, the approach selects clear and reliable data samples on which models trained with both inverse labels and original labels consistently agree. In addition, it incorporates a negative log-likelihood–based difficulty estimation to assess and refine the selection process.

### Strengths
1. While intrinsic reward–based data selection methods are powerful, they often suffer from bias issues. I agree with the problem statement raised by the authors, and I find their proposed flip-labeling–based consistency check to be a simple yet effective solution. This approach is easy to implement and could be widely applicable across different datasets and tasks.
2. The paper conducts experiments on major benchmarks such as AlpacaEval and MT-Bench, using a variety of recent models including LLaMA, Mistral, and Qwen. The consistently strong results across these diverse settings demonstrate the reproducibility and robustness of the proposed method.
3. The authors perform extensive ablation studies and analyses. They evaluate the impact of key components such as the inverse labeling mechanism and the difficulty-aware filtering, providing clear insights into how each factor contributes to the overall performance. This helps me easily understand the role and effectiveness of each component.
4. The results showing performance variations under curriculum learning based on negative log-likelihood are particularly interesting, suggesting promising directions for further exploration.

### Weaknesses
1. This study focuses on determining preference labels, yet it lacks traditional evaluation baselines, relying mainly on comparisons with implicit reward–based approaches. Even the “External Reward Margin” baseline presented in the paper is not a conventional reward modeling method(BT modeling), but rather a zero-shot LLM-as-judge approach. To ensure fair evaluation, the paper should also compare against traditional reward models trained on preference labels, as well as LLM-as-judge models that have been fine-tuned using preference supervision and evaluated accordingly.
2. It would also need additional analysis about the labeling process. The current evaluation assesses data quality indirectly by training models using DPO and measuring their downstream performance. However, this does not directly evaluate the accuracy of the reward modeling or label selection itself. Including experiments that assess the reward model’s intrinsic quality—for example, using  benchmarks such as RewardBench to evaluate labeling method would provide a more meaningful analysis.

### Questions
1. whether this type of preference labeling approach would also be effective for tasks such as mathematics or coding?
2. It would be interesting to explore whether the proposed method can also be effectively applied in online or batch-online (iterative DPO) settings.
3. I also wonder how well AlignDiff would perform on unseen or out-of-distribution data that were not used during reward model training—would it still maintain reliable and consistent label selection under such conditions?

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
The paper presents AlignDiff, a way to construct DPO preference pairs for language models. The process begins with training a model with DPO on the datasets, and train an "inverse" model by flipping the winning and losing response of the dataset. Step 2: It uses the gap between the "DPO implicit reward" induced by the positive model and the "inverse" model to perform data filtering. Step 3: It performs standard DPO on the filtered data. 

Intuitively, we are "cleaning" the data by only retaining data that the positive DPO model indeed think the winning is better than the losing one and the "inverse" model thinks otherwise. Experiments show that this process is effective in improving downstream performance on AlpacaEval, outperforming existing data filtering methods.

### Strengths
1. The paper studies an important topic: data filtering in preference learning. Although there has been some work around this, the authors point out that such method ignores the rich information contained in the model's own signals (DPO implicit rewards)

2. The experiments is comprehensive, spanning both Llama and Qwen models with comparisons against many existing works. 

3. The paper is written clearly.

### Weaknesses
1. The method incurs too much additional cost. To perform data filtering, you would need to train the model on the full datasets for 2 rounds (one positive and one inverse). The complexity of the method raises doubt on whether people is going to adopt this method for their own model training. 

2. The reproducibility is poor - The authors did not disclose what are the exact decoding params + judge used for Alpaca Eval, making it hard to compare this work with other works. Furthermore, when comparing against other baselines, the author also did not disclose how they reimplemented it or just reuse the original authors code. For example, the paper compares against RIP [1], but I don't think the code is published for RIP. So I don't know how did the author set the hyperparameters for this method. The same goes for other methods. The reason why I bring out this is because RIP filtering results in a 10 point increase of Alpaca Eval compared to no filtering but in the author's experiments RIP was performing really poor. So disclosing the exact hparams or code (that would be even better) would make the soundness of the paper better.

I would be happy to raise my score if the authors can include more on reproducibility.

3. It is really hard to understand why the method works. The authors uses community trained SFT models to begin the alignment process while there exists instruct models from the official llama (llama-3.1-8B-Instruct). Some of the experiment findings might be because of the quality conducted in these commnunity trained SFT models. If such models have already undergone the DPO process is a concern, then there are also SFT models trained by larger labs (e.g. https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-SFT) and I wonder why the authors choose community models over established models.

[1] R.I.P.: Better Models by Survival of the Fittest Prompts (Yu et al. ICML 2025)

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces AlignDiff, a two-stage preference data filtering framework that uses only model-intrinsic signals to select higher-quality pairs for DPO training. Stage 1 trains two auxiliary policies with standard and inverted DPO to compute a bidirectional alignment discrepancy and removes ambiguous items while flipping mislabeled pairs when RAD is strongly negative. This keeps only examples with clear model preferences, using a thresholded labeling function. Stage 2 ranks the remaining pairs by Average Negative Log-Likelihood Gap and keeps the top-K hardest pairs, arguing that large positive ANG reflects informative but learnable preference gaps that avoid the “squeezing effect.” The method is implemented on UltraFeedback with LLaMA-3-8B-SFT, Mistral-7B-SFT, and Qwen-2.5-7B-SFT backbones. On AlpacaEval 2.0 and MT-Bench, AlignDiff beats seven filtering baselines. The authors claim the filtered data preserves the length-gap distribution while shifting external reward margins toward clearer preferences, and they report improvements from a simple easy-to-hard curriculum over the selected set.

### Strengths
- The central idea is coherent: use complementary positive and inverse preference signals to detect annotation conflicts, then bias training toward difficult but not pathological pairs. The RAD definition cleanly follows from DPO’s implicit reward margin and the symmetry under label swapping; the selection rule  ϕ(RAD;τ) is well specified. ANG is a simple length-normalized hardness proxy that is less sensitive than PPL. Experimental protocol is laid out with model choices, the baselines and ablations substantiate the claims, including RAD vs single-sided IM and the contribution of ANG.
- The paper presents strong empirical results. AlignDiff improves LC/WR over IM, EM, IM&EM, R.I.P., PPLGap, LCPP, and SDPO across models; for LLaMA-3-8B-SFT, LC 26.4 vs 13.7.

### Weaknesses
- The paper replaces the official judge with DeepSeek-V3 and provides a correlation analysis. This is helpful but still risks distributional bias. A controlled check with the official GPT-4 Turbo annotator or a small human study would strengthen claims. 
- Appendix E studies τ = 60–100 for Mistral yet states ``τ = 20 is optimal for Mistral'', which conflicts with the described range; please clarify and fix or point to the lines you mentioned the reason for this discrepancy. 
- AlignDiff requires training two auxiliary policies, making it less efficient than IM-only filtering, even if more effective. A head-to-head wall-clock and cost comparison during data selection would be useful for practitioners. 
- Results are on UltraFeedback and three 7–8B SFT baselines. It is unclear how RAD thresholds transfer across larger instruction-tuned models and other preference datasets. 
- RAD-based label reversal may enshrine model biases in cases where humans were correct and the model is wrong. Safeguards or hybrid checks are not explored.

### Questions
- Can you report AlpacaEval 2.0 LC/WR using the official annotator on a 200-example subset to quantify any shift vs DeepSeek-V3, and optionally a small human study to validate wins on hard prompts. 
- Please resolve the τ inconsistency for Mistral and provide a sensitivity plot of final LC as a function of τ for each backbone. Also specify how |RAD| varies with β and reference model choice. 
- Did you evaluate a ``discard-only'' variant that never flips labels and only removes ambiguous or inverse cases; how does that compare to your full pipeline at equal data size. 
- Generalization beyond UltraFeedback - Any results on HelpSteer or PKU-SafeRLHF subsets; if unavailable, please discuss expected behavior and any preliminary overlap analyses.

### Soundness
3

### Presentation
4

### Contribution
3
