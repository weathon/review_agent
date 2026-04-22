# MotionDDM: Motion Generation and Understanding via Discrete Diffusion Model

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
We present MotionDDM, a diffusion-LLM framework for bidirectional text-motion understanding and generation. Unlike GPT-style autoregressive approaches that tokenize motion and decode sequentially, MotionDDM performs multi-step parallel denoising, unifying Text-to-Motion (T2M), Motion-to-Text (M2T), and text-free Motion-to-Motion (M2M) within a single model. This decoding paradigm naturally enables a quality-latency trade-off at inference. On HumanML3D, our method achieves competitive T2M/M2T results against strong baselines.  We also incorporate Residual VQ (RVQ) as the motion tokenizer to improve quantization fidelity, and adopt GRPO within the framework to enhance alignment and controllability. To the best of our knowledge, this is the first work to bring diffusion-LLMs to bidirectional text-motion modeling.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a unified text-motion model based on a diffusion-LLM framework, achieving overall strong performance across three tasks: Text-to-Motion, Motion-to-Text, and Motion-to-Motion. To better enhancing the alignment and controllability of the framework, the authors integrate the GRPO framework into their model.

### Strengths
1.	The proposed framework integrates multiple tasks into a unified model, which is novel compared to traditional one-directional motion generation approaches.
2.	As shown in Table 1, the model achieves solid results on both Text-to-Motion and Motion-to-Text tasks.
3.	During inference, the model requires only 10 steps to generate high-quality results, demonstrating good efficiency.

### Weaknesses
1.	From the experimental setup, the Motion-to-Motion task seems to serve a role similar to MAE-style representation learning. It remains unclear whether the model’s superior performance primarily stems from this task rather than T2M or M2T. The authors are encouraged to conduct ablation studies by enabling or disabling the Motion-to-Motion task. 
2. The proposed framework integrates multiple tasks during the learning process. Comparisons with single-task training are necessary to clarify the benefits of multi-task integration
3.	In Table 6, increasing the training ratio of T2M improves the model’s performance on T2M, which is reasonable. However, increasing the M2T ratio unexpectedly decreases its performance. The authors should provide an explanation for this observation.
4. Although the paper claims to achieve a quality-latency trade-off in the abstract, the model’s parameter count, computational efficiency, and actual inference speed are not reported. These should be added and compared with other methods to substantiate the claim.
5.	While a reproducibility statement is provided, given the system’s complexity, releasing the full source code would greatly enhance the work’s credibility and impact.

### Questions
1.	In Table 6, percentages should be written as 80%, 10%, and 10%.
2.	Page 7 line 327: The Div metric is not “the higher, the better”; rather, it should be closer to the ground truth (see MotionGPT for reference).
3.	Page 1 line 50: The acronym “GRPO” should be spelled out when it first appears.
4.	Page 2: Figure caption font sizes are inconsistent and should be standardized.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
MotionDDM is the first work to introduce diffusion–language models into bidirectional text–motion modeling, proposing a unified parallel denoising decoding framework. This paradigm naturally supports the quality–latency trade-off and can be seamlessly extended to various tasks such as text-conditioned and text-free motion completion, prediction, and interpolation. By employing an RVQ-based motion tokenizer and integrating GRPO, the model enhances motion representation fidelity and cross-modal alignment.

### Strengths
The article is clear and easy to understand.
The author conducted sufficient ablation studies to prove the effectiveness of each module.

### Weaknesses
The techniques presented in this paper have largely already been explored and validated in existing unified models and diffusion-LLM research. For instance, RVQ has been employed in Go to Zero [1] and related works. As such, this paper feels more like a technical report rather than a conceptually novel study.

The paper does not clearly articulate the motivation or insight behind unifying understanding and generation. At least for me, it fails to convey why text–motion unification is necessary or meaningful. A more compelling direction would be to extend this idea toward a unified framework of vision, motion, and text, which would carry greater significance.

Moreover, the experiments are conducted on only one dataset—although it is a classic benchmark, it is insufficient to justify the necessity of unified understanding and generation. The quantitative results also lag behind the latest diffusion-based methods.

In addition, there is no visual analysis, which is crucial for evaluating generative models.

Overall, the paper appears to be a combination of several existing methods (diffusion-LLM, unified modeling, and GRPO) applied to a relatively small task. The insight and novelty are limited.

[1] Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data

### Questions
see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes MotionDDM, a discrete diffusion–LLM framework that unifies text-to-motion (T2M), motion-to-text (M2T), and text-free motion-to-motion (M2M) by treating both text and motion as token sequences denoised in multi-step parallel refinement. It uses Residual VQ (RVQ) for motion tokenization, a BERT-based masked backbone, and optional GRPO fine-tuning with task-specific rewards.

### Strengths
- This paper presents a unified bidirectional formulation for bidirectional text–motion (T2M↔M2T). The parallel denoising enables one model to handle T2M, M2T, and M2M, with an explicit step-controlled quality–latency knob.

- The proposed method can also support M2M completion, prediction, and interpolation under
both text-conditioned and text-free settings.

### Weaknesses
- The experiments are conducted only on HumanML3D, and several retrieval metrics are not best-in-table (e.g., T2M R@1 lower than MoTe’s 0.548; M2T R@1 lower than MG-MotionLLM’s 0.592), though FID is strong. The authors should conduct more experiments by adding KIT-ML or HumanAct12, and clarifying where MotionDDM leads vs. falls short.

- The T2M reward uses the model’s own M2T branch to produce a pseudo caption that is then compared to the ground-truth caption in CLIP space. This can bias rewards toward self-consistency rather than true motion–text faithfulness. Consider an external captioner or human preference subsets to calibrate rewards. 

- This paper argues a tunable quality–latency trade-off via step counts, but wall-clock latency (ms/sequence) and throughput are not reported. Please add runtime on a standard GPU (or the reported Ascend 910 NPU) for K={5,10,20,30}, including speedups vs. an AR baseline. 

- There are a few issues regarding the ablation and clarity: (a) RVQ depth table shows non-monotonic behavior. It would be good to add quantization error (MSE) vs. depth to clarify. (b) For masking schedule ablations, it is suggested to report M2T text metrics (not just T2M) to check cross-modal effects. (c) Please provide token rate/bitrate of RVQ (frames/sec × tokens/frame × bits/token) for reproducibility.

- It would be good to show many qualitative results (visual comparison with the state-of-the-art methods). 

- It would be good to include more sota methods for comparison, such as MoMask, MotionLCM, MaskControl. 

Minor:

- Table 7 shows FID = 0.0067 at CFG=5, far off neighboring entries. Can you please clarify?

### Questions
Please see the weakness section.

### Soundness
3

### Presentation
2

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
This paper introduces a diffusion-based framework for bidirectional text-to-motion and motion-to-text generation. The main design is to utilize a multi-step parallel denoising decoder to progressively denoise the noisy text and motion sequences. To enhance the model performance, the residual vector quantization for motion quantization is utilized, and a multi-task training schedule is proposed.  The GRPO is also integrated to enhance the alignment and controllability. Extensive experiments are conducted on the HumanML3D benchmark to evaluate the effectiveness of the proposed framework.

### Strengths
- The proposed multi-task scheduling mechanism for a unified and bidirectional T2M and M2T generation framework optimization is well-motivated and reasonable. 
- Experimental results on the motion-to-text task surpass previous methods, showcasing the proposed framework in enhancing the M2T task. 
- A sets of ablation studies are conducted to support the proposed design choices.

### Weaknesses
- The performance of the proposed framework on the text-to-motion generation task is not very good, underperforming previous work in most of the evaluation metrics. And some recent stronger baselines for text-to-motion generation are missing, e.g., MoMask (CVPR 2024), MoGenTS (NeurIPS 2024), and LAMP (ICLR 2024). Note that these baselines are auto-regressive-based frameworks, and LAMP also supports T2M and M2T tasks.  Could the author provide some analysis and insights on why the performance gain on the T2M task is less? Also, the proposed framework underperforms these baselines on the T2M tasks, and the author should be more cautious in the claims in L044-L046.
- Experiments are only conducted on the HumanML3D dataset. Evaluating the proposed framework on more datasets, e.g., the Motion-X dataset, and the KIT-ML datasets, will bolster the claims of generalizability and scalability.

### Questions
The multi-task scheduler randomly assigns a task to each sample in a batch. Will more structured scheduling (e.g., curriculum learning) improve the performance? Specifically, the performance gain on the text-to-motion task seems smaller with the proposed framework. Does this mean the T2M task is harder for the proposed framework?

### Soundness
3

### Presentation
3

### Contribution
2
