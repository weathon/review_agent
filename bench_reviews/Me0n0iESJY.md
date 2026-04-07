## Summary
This paper introduces OptMerge, a data-free method for merging Multimodal Large Language Models (MLLMs) that applies low-rank approximation to task vectors and optimizes the merged vector via a tailored loss. It also presents a comprehensive benchmark for MLLM merging, categorizing capabilities into VQA, Geometry, Chart, OCR, and Grounding, and explores merging across vision, audio, and video modalities.

## Strengths
- **Well-constructed benchmark**: The benchmark provides a fine-grained categorization of MLLM capabilities, includes both full fine-tuning and LoRA settings for two model families (InternVL2.5 and Qwen2-VL), and releases checkpoints and code, facilitating future research.
- **Effective method with supporting analysis**: OptMerge addresses noise and interference in task vectors through SVD-based denoising and robust optimization, motivated by an analysis of task vector properties (Fig. 2) and a theoretical upper bound linking fine-tuning dynamics to merging performance (Theorem 3.1).
- **Extensive empirical validation**: Experiments cover capability merging, modality merging, real-world checkpoints from Hugging Face, ablation studies, and scaling to 32B models, demonstrating that merging can match or surpass mixture training and integrate multiple modalities.

## Weaknesses
- **Incremental methodological novelty**: OptMerge primarily combines existing techniques—SVD denoising and the WUDI Merging optimization framework—with heuristic adaptations for full vs. LoRA fine-tuning, without a groundbreaking algorithmic advance.
- **Heuristic hyperparameter choices**: Key parameters, such as the rank size \(k\) (set to rank(task_vector)/number of tasks) and the optimization settings (e.g., Adam for InternVL, SGD for Qwen2-VL), are justified empirically but lack principled derivation, affecting reproducibility and generalizability.
- **Overstated claims about outperforming experts**: The paper claims the merged model "can even outperform expert MLLMs in their respective capabilities," but results (Tables 2 and 3) show the merged model typically performs between the base and expert models on individual tasks, not exceeding the best expert on its specialty. This misrepresents the more accurate strength: strong multi-task performance.
- **Missing comparison with MLLM-specific merging methods**: While the paper cites AdaMMS and UQ-Merge, it does not experimentally compare OptMerge against these MLLM-focused methods, weakening the claim of advancement in MLLM merging.
- **Limited modality merging evaluation**: Modality merging is evaluated only on audio-visual QA datasets (Table 5), without assessing whether the merged model retains unimodal capabilities (e.g., vision-only VQA), leaving robustness to catastrophic forgetting unverified.

## Nice-to-Haves
- Sensitivity analysis for hyperparameters like rank \(k\) and optimization iterations to guide users.
- Comparison of all merging methods on integrated benchmarks (e.g., MMMU, ScienceQA) to better demonstrate emergent multi-capability performance.
- Visualization of task vector interference (e.g., cosine similarity matrices) to directly illustrate OptMerge's denoising effect.
- Expanded discussion on limitations, such as sensitivity to fine-tuning regimes and the assumption of a common base model architecture.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Statistical significance reporting**: The paper reports single scores, but in model merging literature, single-run evaluation is common for large-scale benchmarks; demanding variance estimates is not standard practice here.
- **Demand for more scales or modalities**: The paper includes experiments up to 32B models and three modalities, which is sufficient for a benchmark; requesting more is scope creep.
- **Criticism of the Qwen2-VL mixture training baseline**: The paper acknowledges using Qwen2-VL-Instruct as a proxy due to practical constraints, and this does not invalidate the core findings.
- **Ablation study design**: The sequential ablation in Table 4, while showing interactions, is sufficient to demonstrate component contributions; a full factorial design is not required for the paper's claims.

## Novel Insights
The benchmark provides a structured framework for evaluating MLLM merging across distinct capabilities and modalities, highlighting the complementarity of modal information. The theoretical analysis (Theorem 3.1) offers a novel explanation for how fine-tuning extent (learning rate and iterations) influences merging performance, linking parameter drift to cross-task interference. However, these insights are largely within the paper's own contributions; no fundamentally new observations beyond the paper emerge from the reviews.

## Suggestions
- Add an experimental comparison with MLLM-specific merging methods like AdaMMS and UQ-Merge to solidify claims of advancement.
- Provide failure analysis or qualitative examples to illustrate cases where merging degrades performance, helping to identify limitations.
- Justify hyperparameter choices more rigorously, for instance by linking rank selection to singular value energy thresholds or validating coefficients on a held-out set.