# Fine-tuning Done Right in Model Editing

- Decision: Accept (Poster)
- Scores: 8, 8, 4, 4

## Abstract
Fine-tuning, a foundational method for adapting large language models, has long been considered ineffective for model editing.
Here, we challenge this belief, arguing that the reported failure arises not from the inherent limitation of fine-tuning itself, but from adapting it to the sequential nature of the editing task, a single-pass depth-first pipeline that optimizes each sample to convergence before moving on.
While intuitive, this depth-first pipeline coupled with sample-wise updating over-optimizes each edit and induces interference across edits.
Our controlled experiments reveal that simply restoring fine-tuning to the standard breadth-first (i.e., epoch-based) pipeline with mini-batch optimization substantially improves its effectiveness for model editing.
Moreover, fine-tuning in editing also suffers from suboptimal tuning parameter locations inherited from prior methods. 
Through systematic analysis of tuning locations, we derive LocFT-BF, a simple and effective localized editing method built on the restored fine-tuning framework.
Extensive experiments across diverse LLMs and datasets demonstrate that LocFT-BF outperforms state-of-the-art methods by large margins. 
Notably, to our knowledge, it is the first to sustain 100K edits and 72B-parameter models,10 $\times$ beyond prior practice, without sacrificing general capabilities.
By clarifying a long-standing misconception and introducing a principled localized tuning strategy, we advance fine-tuning from an underestimated baseline to a leading method for model editing, establishing a solid foundation for future research.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper revisits the long-standing belief that fine-tuning is ineffective for model editing in LLMs. The authors argue that prior failures stem not from inherent limitations of fine-tuning but from mis-specified implementations that used a depth-first (DF), sample-wise pipeline rather than the standard breadth-first (BF), epoch-based, mini-batch paradigm. Through controlled experiments, they show that restoring the BF pipeline substantially improves reliability and preserves general capabilities.

Building on this insight, they introduce LocFT-BF (Localized Fine-Tuning with Breadth-First pipeline), which integrates three key ideas: (1) Restoring the standard BF pipeline; (2) Using mini-batch gradient aggregation; (3) Strategically selecting tuning locations (favoring later-layer MLP down-projections).

Extensive experiments demonstrate that LocFT-BF outperforms representative methods (e.g., WISE, RLEdit, UltraEdit, MEMIT, RECT, AlphaEdit) across multiple datasets (ZsRE, COUNTERFACT, WikiBigEdit) and models (LLaMA-3, Mistral, Qwen2.5). It achieves high reliability (>98%), strong generalization, efficient scalability (up to 100K sequential edits), and applicability to relatively very large models (up to 72B parameters).

Overall, this is a strong paper with clear contributions, high experimental quality, and potential to reshape the model editing field. Addressing the generalization limitations and exploring broader use cases would further enhance its impact.

### Strengths
- This paper challenges a widely held misconception in model editing and repositions fine-tuning as a strong, scalable baseline, and it’s a original proposal. Specifically, this paper identifies the flawed depth-first pipeline as the root cause of prior failures, an insight both simple and impactful. Furthermore, this paper proposes LocFT-BF, which combines principled pipeline design and layer selection with compelling empirical validation. 
- The experimental design is thorough, e.g., controlled comparisons of DF vs BF, batch size studies, layer/module analyses, and large-scale evaluations (100K edits, 72B LLMs).
- Benchmarks against a broad set of representative baselines, including parameter-extension, meta-learning, and locate-then-edit methods. Results are consistently strong across multiple datasets and architectures, lending credibility to claims.
- The paper is well-structured, starting with the misconception, then systematically dissecting causes, followed by a constructive solution. 
- Writing is clear and accessible even for readers not deeply familiar with model editing. Figures and tables (e.g., pipeline comparisons, tuning location studies, scaling results) are intuitive and reinforce key arguments.
- LocFT-BF is lightweight, plug-and-play, and avoids architectural overheads. It could shift the research landscape by elevating fine-tuning from a dismissed baseline to a leading paradigm for model editing. Furthermore, scalability to 100K edits and 72B models addresses real-world deployment needs, setting a new bar for evaluation standards.

### Weaknesses
- The generality of layer selection is not strong enough. The paper proposes later-layer MLP-down projections as relatively stable tuning locations, but acknowledges that optimal locations vary across models. More justification or theory behind this heuristic would strengthen the contribution.
- LocFT-BF underperforms on generalization for certain datasets (e.g., COUNTERFACT, where prompts involve irrelevant text). Authors suggest data augmentation as a fix in A.2, but no experiments directly test this claim.
- While diverse, the benchmark datasets are mostly factual-editing oriented. Broader evaluation (e.g., safety edits, temporal consistency) would strengthen claims about “real-world editing”.
- While comparisons are comprehensive within model editing literature, it would be useful to position LocFT-BF against emerging post-training alignment techniques (e.g., preference tuning, instruction backfitting) that could also serve as editing strategies.

### Questions
- How sensitive is LocFT-BF to tuning location selection in unseen LLM architectures (e.g., encoder-only, mixture-of-experts)? Could an automatic tuning-location search mechanism generalize better? How do you judge the layer selection robustness?
- For COUNTERFACT, LocFT-BF shows weaker generalization compared to some locate-then-edit baselines. Could integrating data augmentation or paraphrase generation directly into LocFT-BF resolve this gap? Could you provide some empirical study?
- Beyond factual edits, How would LocFT-BF perform on edits involving style adaptation, or safety constraints? Are there plans to extend evaluation beyond factual QA?
- While the method scales to 100K edits, does edit interference accumulate subtly over time (e.g., conflicting edits on related entities)? Have the authors analyzed error patterns at scale?
- How would LocFT-BF integrate into a real-world setting where edits arrive asynchronously over time and may be reverted? Is incremental updating compatible with LocFT-BF’s batch/epoch paradigm?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper challenges the prevailing view that fine-tuning is a weak baseline for model editing. The authors compellingly argue that its reported failures stem not from the method itself, but from a flawed "Depth-First" (DF) implementation that processes edits sequentially and individually. This DF pipeline, they demonstrate, leads to catastrophic forgetting and degrades general capabilities.

The core proposal is to revert to the standard "Breadth-First" (BF) training pipeline, which uses epoch-based, mini-batch optimization over the entire set of edits. This is combined with a systematic study that identifies tuning later-layer MLP modules as an effective strategy for localizing updates.

The resulting method, LocFT-BF, is simple and highly effective. Extensive experiments show that it significantly outperforms state-of-the-art methods and, for the first time, successfully scales to 100,000 sequential edits and 72B-parameter models, setting a new and powerful baseline for the field.

### Strengths
Strength1: The claims are exceptionally well-supported by a series of controlled experiments that systematically isolate and analyze the impacts of the training pipeline (DF vs. BF), batch size, and tuning location.

Strength2: LocFT-BF is conceptually simple, efficient, and easy to implement, especially when compared to more complex editing techniques. This makes it a highly practical and powerful new baseline.

### Weaknesses
Weakness1: The method shows a generalization gap on the COUNTERFACT dataset compared to baselines that use data augmentation. While the authors provide a plausible explanation, the claim that this can be "readily mitigated" with augmentation is not experimentally verified in the paper.

### Questions
Q1: Have you performed any experiments to support the claim that the generalization gap on COUNTERFACT can be closed by applying data augmentation techniques similar to those used by methods like MEMIT or AlphaEdit?

Q2: How would a standard LoRA, when applied to the empirically-found optimal modules (e.g., later-layer MLPs) and trained using the correct BF pipeline, compare to your full-parameter local tuning approach in LocFT-BF? This would help clarify whether, under the proper training paradigm, the concept of localization is the more critical factor, or if full-parameter tuning is inherently superior to low-rank adaptation in editing.

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
4

### Summary
This paper focuses on the long-term underestimation of fine-tuning methods in the field of large language model (LLMs) model editing. Through in-depth analysis of the shortcomings of existing fine-tuning implementations, an innovative LocFT-BF method is proposed, which restores fine-tuning to a standard breadth first (BF) pipeline and combines local parameter tuning. The paper verified the effectiveness of the method through systematic control experiments, surpassing existing mainstream model editing methods on multiple models and datasets, and achieving stable editing of 100000 edits and 72B parameter models for the first time. The research design is rigorous, the experimental scale is sufficient, and it has important theoretical and practical significance for the field of model editing. However, there is still room for improvement in terms of theoretical depth, adaptability to special scenarios, and supplementary experimental details in the paper.

### Strengths
- Overturns the long-standing misconception that fine-tuning is unsuitable for model editing , proving through rigorous experiments that prior failures were due to flawed implementations (i.e., the depth-first pipeline) rather than inherent limitations.
- The proposed LocFT-BF method is exceptionally simple and efficient, requiring no complex architectural changes, auxiliary data, or costly pre-computation overhead typical of other editing techniques。
- Demonstrates unprecedented scalability by showing LocFT-BF effectively handles 100K sequential edits and scales to 72B-parameter models, addressing a key practical bottleneck in model editing.
- Provides a systematic and in-depth empirical analysis of how to apply fine-tuning correctly, including rigorous studies on the training pipeline , batch size , and optimal parameter tuning locations.
- Achieves state-of-the-art performance, comprehensively and significantly outperforming existing methods, particularly in challenging large-scale lifelong editing scenarios.

### Weaknesses
- The paper's justification for the superiority of the Breadth-First (BF) pipeline relies heavily on empirical phenomena (e.g., mitigating overwriting) . It lacks a deeper theoretical analysis, such as convergence or stability bounds, to formally explain why mini-batch optimization provides superior stability and forgetting mitigation compared to the sample-wise, depth-first (DF) approach in the context of sequential editing.
- The "Capability" evaluation, while broad, is confined to standard benchmarks (e.g., MMLU, GSM8K, SST2) that are predominantly multi-choice or classification tasks. This setup overlooks more complex and valuable scenarios, failing to test the model's robustness against, for example, generating hallucinations (e.g., TruthfulQA) or handling complex reasoning (e.g., BBH) post-edit.
- The paper repeatedly claims that the observed degradation in "Generalization" (e.g., on COUNTERFACT) can be mitigated by data augmentation . However, this claim remains unsubstantiated, as no supporting experiments are provided in the main text to validate whether augmentation can indeed recover generalization without negatively impacting reliability or capability. The author can refer to the following work on augmentation and discuss it:

[1] Allen-Zhu Z, Li Y. Physics of language models: Part 3.1, knowledge storage and extraction[J]. arXiv preprint arXiv:2309.14316, 2023.

[2] Jiang K, Du Y, Ding Y, et al. When Large Multimodal Models Confront Evolving Knowledge: Challenges and Pathways[J]. arXiv preprint arXiv:2505.24449, 2025.

[3] Wang K, Zhu J, Ren M, et al. A survey on data synthesis and augmentation for large language models[J]. arXiv preprint arXiv:2410.12896, 2024.

[4] Park C F, Zhang Z, Tanaka H. $\textit {New News} $: System-2 Fine-tuning for Robust Integration of New Knowledge[J]. arXiv preprint arXiv:2505.01812, 2025.

- The experimental validation focuses exclusively on general-domain factual knowledge (e.g., ZsRE, COUNTERFACT). The effectiveness and generalizability of LocFT-BF in editing specialized, high-stakes domains (e.g., biological, financial, or legal knowledge) or in cross-domain/cross-lingual editing scenarios remain unverified.
- The analysis of scaling to larger models (§5.2)  only evaluates LocFT-BF on the Qwen2.5 series. It crucially lacks a direct comparison against baseline methods (e.g., RLEdit, UltraEdit) at these larger scales (14B, 32B, 72B). This omission makes it difficult to substantiate the strong claim that LocFT-BF is uniquely or more easily scalable than other methods while maintaining performance.

If the author can solve the above problems, I will consider increasing the score.

### Questions
- Beyond the empirical evidence of catastrophic overwriting in the DF pipeline , what is the theoretical justification for why the BF pipeline with mini-batching  provides superior stability and mitigates catastrophic forgetting so effectively during sequential editing?
- How does LocFT-BF perform on benchmarks designed to test for hallucination (e.g., TruthfulQA) or complex multi-step reasoning (e.g., BBH)? Does the localized fine-tuning approach maintain robustness in these more challenging generative scenarios post-editing?
- The authors argue that the generalization gap can be offset by data augmentation . Have experiments been conducted to support this claim, and if so, does augmentation successfully improve generalization without compromising the method's high reliability and capability scores?
- How well does the LocFT-BF framework, including its specific layer-selection strategy (e.g., later-layer MLPs) , generalize to editing highly specialized, high-stakes knowledge domains (e.g., legal or medical) or to multilingual knowledge editing tasks? Although this article proposes a new editing method, it focuses more on the analysis of knowledge editing methods. Therefore, we hope that the author can explore more extensively and provide more valuable insights.

[1] Zhang X, Liang Y, Meng F, et al. Multilingual knowledge editing with language-agnostic factual neurons[J]. arXiv preprint arXiv:2406.16416, 2024.

[2] Wang W, Haddow B, Birch A. Retrieval-augmented multilingual knowledge editing[J]. arXiv preprint arXiv:2312.13040, 2023.

[3] Wei Z, Deng J, Pang L, et al. Mlake: Multilingual knowledge editing benchmark for large language models[J]. arXiv preprint arXiv:2404.04990, 2024.

[4] Rosati D, Gonzales R, Chen J, et al. Long-form evaluation of model editing[J]. arXiv preprint arXiv:2402.09394, 2024.

- While LocFT-BF scales to 72B models, this analysis lacks a baseline comparison. Could a comparison be provided, even on a mid-sized model (e.g., 32B), to better substantiate the claim of superior scalability, or is the assumption that other methods are computationally infeasible at this scale verified?

### Soundness
2

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
This paper addresses the failure of fine-tuning in model editing by introducing two principled improvements. The authors argue that the failure of existing fine-tuning approaches mainly stems from the depth-first (single-pass) updating paradigm, which leads to catastrophic forgetting. To mitigate this, they restore the standard breadth-first training pipeline, optimizing all mini-batches across epochs rather than editing samples sequentially. Furthermore, by systematically analyzing which layers and modules are most effective for editing, the study finds that tuning the down- or up-projection matrices in later layers generally achieves near-perfect editing success while maintaining general capabilities.

### Strengths
1. The paper conducts extensive experiments to validate their hypotheses and insights.

2. The authors’ observation that the depth-first approach leads to instability and catastrophic forgetting is indeed convincing in my view.

3. The paper provides clear and informative visualizations and ablation studies

### Weaknesses
1. While the breadth-first pipeline mitigates forgetting effectively, its computational cost grows linearly with the number of accumulated edits, as previous samples must be revisited in each fine-tuning pass. This may raise concerns about scalability in real-time or large-scale editing scenarios, despite the authors’ empirical demonstration of efficiency at 100K edits.

2. While the breadth-first method alleviates catastrophic forgetting to some extent, it does not fundamentally eliminate the issue, since there is no guarantee that the model’s pretrained knowledge remains intact after editing.

3. The locate strategy proposed in this paper appears somewhat vague and even inefficient. For example, in Section 5.2, the authors directly scale the editing location from Qwen2.5-14B to Qwen2.5-32B, which seems to rely on the assumption that the editing samples remain identical and that the two models share highly similar architectures. However, in real-world applications, the editing samples are dynamically updated, and the optimal editing module may vary accordingly. Simply transferring the previous settings to larger or more heterogeneous LLMs could therefore be unreliable, and the effectiveness of such an extension may degrade even further in those scenarios.

4. At the same time, the method this paper uses to identify the optimal tuning location also appears somewhat inefficient, as they fine-tune across all layers and five candidate modules in order to determine the best position.

### Questions
1. When new samples arrive, how does LocFT-BF dynamically reconstruct its epochs and batches? Moreover, how does it handle edits that conflict with previous ones?

2. For a new, heterogeneous LLM and an unseen task, could the authors clarify how LocFT-BF determines or generalizes the choice of the optimal editing module?

3. Could the authors provide a comparison of the time and computational cost between the breadth-first and depth-first fine-tuning pipelines?

If I have misunderstood anything, please feel free to point it out.

### Soundness
2

### Presentation
2

### Contribution
2
