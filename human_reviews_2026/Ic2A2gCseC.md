# Beyond Fixed: Training-Free Variable-Length Denoising for Diffusion Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Diffusion Large Language Models (DLLMs) are emerging as a powerful alternative to the dominant Autoregressive Large Language Models, offering efficient parallel generation and capable global context modeling. However, the practical application of DLLMs is hindered by a critical architectural constraint: the need for a statically predefined generation length.  This static length allocation leads to a problematic trade-off: insufficient lengths cripple performance on complex tasks, while excessive lengths incur significant computational overhead and sometimes result in performance degradation.  While the inference framework is rigid, we observe that the model itself possesses internal signals that correlate with the optimal response length for a given task.  To bridge this gap, we leverage these latent signals and introduce DAEDAL, a novel training-free denoising strategy that enables Dynamic Adaptive Length Expansion for Diffusion Large Language Models.  DAEDAL operates in two phases: 1) Before the denoising process, DAEDAL starts from a short initial length and iteratively expands it to a coarse task-appropriate length, guided by a sequence completion metric.  2) During the denoising process, DAEDAL dynamically intervenes by pinpointing and expanding insufficient generation regions through mask token insertion, ensuring the final output is fully developed. Extensive experiments on DLLMs demonstrate that DAEDAL achieves performance comparable, and in some cases superior, to meticulously tuned fixed-length baselines, while simultaneously enhancing computational efficiency by achieving a higher effective token ratio. By resolving the static length constraint, DAEDAL unlocks new potential for DLLMs, bridging a critical gap with their Autoregressive counterparts and paving the way for more efficient and capable generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles a central limitation of Diffusion LLMs—the need to pre-set a fixed output length—and proposes DAEDAL, a training-free, two-stage inference method that first estimates a task-appropriate length by checking EOS confidence at the sequence end and then expands low-confidence regions on the fly via mask insertion during denoising. On LLaDA-Instruct-8B and related DLLMs, DAEDAL matches or exceeds carefully tuned fixed-length baselines while using tokens more efficiently (e.g., GSM8K 85.8 vs. 83.8 accuracy with a much higher effective-token ratio), and extensive ablations show robustness to initial length, thresholds, and expansion factors.

### Strengths
- The proposed method is simple and intuitively reasonable. Requiring no retraining is a plus
- The proposed method demonstrates solid empirical improvements over the best-tuned fixed-length baselines on math reasoning and code generation benchmarks (e.g., MATH500), while generating effective tokens more efficiently.
- This paper conducts a thorough analysis of key hyperparameters of the method, showing its robustness to different configurations.

### Weaknesses
- While the paper shows in experiments that a combination of the two expansion stages gives the best performance, in principle, there still lacks a clear reason why both stages are necessary. It's natural to consider merging the first length adjustment stage into the second dynamic expansion stage. Interestingly, according to Table 2, stage 1 already contributes to most of the performance improvement. I think more investigation should be put into this.
- The ablation study shows the method's robustness to threshold hyperparameters, but it's only for a single model-dataset pair, i.e., LLaDA-Instruct-8B on GSM8K. I wonder whether the threshold findings can be transferred to other model-dataset combinations. Or do you need to tune thresholds again for a different dataset?
- It would be better to have some more direct measurements on the actual inference speed.
- The benchmarks focus on math and code. How would the proposed method perform on general language tasks?

### Questions
Please see Weaknesses

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
3

### Summary
This paper introduces an inference method for a pretrained masked diffusion model that supports length variability. In particular, the algorithm leverages two core operations-determining length using EOS confidence and then expanding iteratively for the masked position with the lowest prediction confidence. Experiments have been conducted on several math and code benchmarks on LLaDA-Instruct and Dream. The results show that the approach can maintain the accuracy of the "optimal" length sweeping over several lengths while being adaptive-length at inference time.

### Strengths
1. The goal of achieving adaptive-length decoding is promising and important to obtain better trade-off frontier between sample quality and latency.

2. The proposed approach is easy to follow and the method requires no additional training.

3. The presentation of the paper is clear and the empirical performance, specifically on the accuracy v.s. total tokens, is quite impressive. The ablation studies including hyperparameter sensitivity analysis are informative.

### Weaknesses
1. The proposed idea of using confidence of predicting EOS token to determine the length of the sequence seems a bit heuristic. A more comprehensive evaluation should be conducted on more datasets and tasks to justify the effectiveness of such heuristic.

2. Efficiency metrics such as wall-clock time/latency is missing.

### Questions
1. I am curious about the generalization of the findings in Figure 2 on over datasets and number of tokens (other than 128). Would the authors be able to provide more empirical justifications?

2. In the main table, it would be helpful to add actual wall-clock decoding time as a metric. Would the authors provide results for this?

3. Would the approach benefit from some training/finetuning of the model to better account for the variable-length objective?

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
This paper proposes an inference-time method for extending the generation length of text diffusion models. Starting with a small canvas, masked tokens get added until EOS occurs with high-enough probability at the end of the sequence. Given this initial sequence of masked tokens, the method alternates between filling in high-confidence tokens and adding more tokens based on EOS confidence. Tokens are added to the lowest confidence mask positions, and are expanded by a constant block size.

Results on simple math and coding evaluations show that the method obtains reasonable accuracy while adaptively determining response length. The average ratio between utilized tokens versus padding is around 65% across all tasks.

### Strengths
The method seems effective and is very simple. The presentation was written to be reader-friendly, but I believe would benefit from including more details. For example, Algorithm 1 has a few more details than Fig 3. The results are reasonable, with easing the need for manual sequence length tuning.

### Weaknesses
The main weakness is that the main baselines to compare to is block diffusion or other adaptive-length methods. The sell of variable-length  diffusion would likely have to be a speed increase while preserving accuracy, or another point on the speed-accuracy Pareto frontier. It is possible that spec-decoded autoregressive models achieve better speeds at similar accuracy.

### Questions
The paper supports its claims. The main improvement would be to include speed-accuracy comparisons to other accelerated methods, namely speculative decoding.

### Soundness
2

### Presentation
3

### Contribution
2
