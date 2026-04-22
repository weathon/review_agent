# $\mathbf{T^2HTR}$: $\textbf{T}$est-$\textbf{t}$ime $\textbf{H}$ierarchical $\textbf{T}$emporal $\textbf{R}$etrieval for Long Video Understanding

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 2

## Abstract
Foundational Multi-modal Large Language Models (MLLMs) have achieved rapid progress in handling complex tasks across diverse modalities. However, they still struggle to deliver satisfactory performance on Long-video Understanding (LVU) tasks involving thousands of frames. Existing optimization strategies can be broadly categorized into LVU-specific fine-tuning, built-in token compression and training-free keyframe extraction, with the latter being most suitable for flexible deployment across various MLLMs. Unfortunately, current training-free approaches predominantly focus on query-frame relevance retrieval, overlooking other levels of visual information and the inherent heterogeneity of LVU tasks. In this work, we propose the $\textbf{T}$est-$\textbf{t}$ime $\textbf{H}$ierarchical $\textbf{T}$emporal $\textbf{R}$etrieval ($\mathbf{T^2HTR}$) framework, which employs a multi-stage pipeline, including dual scene segmentation, joint score calculation, sub-scene window modeling and dynamic mask-based inference, to extract distinct keyframes sets from the perspectives of relevance, summarization and causality. These keyframes are then blended at varying ratios to construct multiple video sampling pools. Guided by adaptive feedback from the model, $\mathbf{T^2HTR}$ dynamically routes each sample to its optimal video pool, enabling more precise and sample-grained LVU. Extensive experiments demonstrate the advanced performance of our scheme across multiple challenging LVU benchmarks. For instance, integrating $\mathbf{T^2HTR}$ with Qwen-2.5-VL yields performance gains of 3.5\% to 13.1\% on LVB, VideoMME and MLVU.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This study explores a training-free sampling strategy for long video frames. It relies on a large set of manually designed frame retrieval methods, and constructs a video sampling pool by combining video scene construction with frame assessment for relevance and causality. Finally, it achieves more accurate and modifiable frame sampling through an adaptive sampling strategy based on a model feedback loop. The method proposed in this study shows significant improvements on various long video benchmarks, which confirms the effectiveness of the method.

### Strengths
1. This study leverages a series of carefully manually designed frame sampling strategies, which effectively enhances the efficiency of MLLMs in long video understanding tasks. Significant performance improvements are observed across multiple benchmarks, and advances are also achieved compared to previous adaptive frame sampling algorithms.

2. Comprehensive ablation experiments are conducted in this study to demonstrate the impact of each component on the final results, thereby supporting the effectiveness of the proposed method.

### Weaknesses
1. There is a lack of comparison with previous Video Agent works. While this paper claims that its innovations include not only improvements in frame sampling strategies but also sampling integrated with model feedback, this is in fact quite similar to the motivation and implementation of many prior Video Agent studies, such as LVAgent[1]. The authors are required to provide more in-depth elaboration and comparison.

2. Efficiency comparison is absent. The frame sampling strategy proposed in this paper is rather complex and relies on some existing models during the process. The authors need to conduct a certain degree of evaluation on its operational efficiency.

3. The method in this paper seems to involve an excessive number of hyperparameters that require tuning. This raises concerns about the generalizability of the method, as it may be unreasonable and impractical to continuously adjust hyperparameters for limited benchmarks solely to achieve better performance improvements.

[1] LVAgent: Long Video Understanding by Multi-Round Dynamical Collaboration of MLLM Agents

### Questions
1. As illustrated in Table 2, neither User Query Analysis nor Missing Information appears to exert a significant positive effect on the final performance. However, similar operations have been proven effective in previous studies, such as those applied in LVAgent[1] or VideoChat-A1[2]. It is recommended that the authors provide a more detailed explanation regarding the similarities, differences, and the underlying reasons for this phenomenon.

2. What is the operational efficiency of this method?

[1] LVAgent: Long Video Understanding by Multi-Round Dynamical Collaboration of MLLM Agents

[2] VideoChat-A1: Thinking with Long Videos by Chain-of-Shot Reasoning

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes $T^2HTR$ (Test-Time Hierarchical Temporal Retrieval), a training-free framework for improving long-video understanding in multimodal large language models. It hierarchically selects keyframes based on query relevance, scene summarization, and causal importance, and incorporates a closed-loop feedback mechanism that adaptively resamples frames when information is insufficient. Experiments on several benchmarks show that $T^2HTR$ consistently improves reasoning accuracy and efficiency across different video-language models without additional training.

### Strengths
- The paper introduces a novel test-time hierarchical retrieval paradigm that unifies relevance-, summarization-, and causality-based frame selection, offering a creative perspective on long-video understanding without additional training.

- The proposed framework is well-engineered and empirically validated on multiple benchmarks, showing consistent improvements across different MLLMs and maintaining strong generalization ability.

### Weaknesses
- The manuscript will benefit from clearer and more consistent mathematical benefit from clearer and more consistent mathematical. Superscripts and subscripts (e,g, $I$, $II$, $III$,$i$,$j$,$t$) are frequently reused to denote different semantic levels—such as scene hierarchy, stage indices, and frame positions, without a clear notation table or explicit definitions. 

- The proposes $T^2HTR$ is intuitive, but lacks sufficient ablation studies to validate the necessity and individual contribution of its core components—particularly dual-stage scene segmentation, mask-based causal inference, and adaptive pool selection. A systematic ablation is needed to substantiate the design choices and claimed advantages.

- The proposes $T^2HTR$ significantly enhances LVU performance, but it fails to analyze the associated computational overhead.  The framework involves dual scene segmentation, CLIP-based relevance scoring at both global and patch levels, LLM-based captioning, and iterative causal inference, all of which introduce non-trivial latency and resource costs at test time.

### Questions
- The article only provides experimental results for 64 frames. It is necessary to provide experimental results and analysis for different selected frames, such as 8, 16, 32, or even 128, 256, if computing resources allow.

- The article introduces a large number of hyperparameters, such as $t^*$, $\alpha$, $\Theta_{I}$, and $\Theta_{II}$. Although the article provides specific values, it is necessary to add ablation experiments to evaluate the influence and robustness of $T^2HTR$.

- Add ablation experiments and analysis of A's components so that readers can clearly understand the contribution and rationality of each component design.

- It is necessary to add the analysis of computational consumption and latency.

- Lack of discussion on limitations

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
5

### Summary
The paper proposes T$^2$HTR, a hierarchical temporal retrieval framework for long video understanding by involving scene segmentation, relevancy and casuality calcuation, and key video frame sampling. It conduct experiments on three long video understanding benchmarks, showing improvement over base models, and outperforms other training-free methods on long video understanding.

### Strengths
+ The method is reasonable and straightforward.
+ It is training-free, making it applicable to any large video-language model.
+ The evaluation includes comparisons with both training-free and fine-tuned models, showing substantial improvements over the base models.

### Weaknesses
+ The writing and notation require improvement.
+ The method involves extensive frame similarity comparisons and employs MLLM to caption visual frames for causality evaluation, which introduces significant computational overhead. However, the paper lacks an analysis of memory usage and inference speed.
+ There is lack of comprehensive ablation of each proposed component.
+ The method appears to be a heavily engineering-focused effort with limited novelty. Both the scene identification and query-based similarity selection have been explored in existing literature.
+ The comparison does not seem entirely fair. Although only 64 frames are ultimately fed into the MLLM, it still need to process additional frames by MLLM during intermediate steps. Given the method involves numerous intermediate steps and introduces significant computational overhead, the comparison should be made against the base model’s best performance, rather than the performance using only 64 input frames.

### Questions
+ The notation for global and regional features needs to be differentiated for clarity. For example, Line 195, $CLIP_{I}(V_{1:T}) \in R^{T\times D}$. Line 226, $CLIP_{I}(V_{1:T}) \in R^{T\times p^2 \times D}$. The function of CLIP should be differentiated accordingly, as the extracted features are used in distinct ways.
+ Which CLIP feature is used for the global-level representation? Is it obtained via average pooling across all tokens, or by using the CLS token embedding?
+ The teaser figure should be revised to avoid potential confusion. Subfigure (d) gives the impression that selecting frames from the raw videos does not require a vision encoder. However, according to the method, both scene segmentation and query relevance still rely on extracting CLIP features.
+ What is the computational overhead compared with the base models, since the method involves a lot of frame simiarlity comparison, and uses LLM for causality evaluation.
+ What is the LLM used in line 263 for captioning?
+ Some ablation studies are missing. What is the contribution of scene and sub-scene segmentation, how's the model performance if select key frames from uniform windows instead of from each identified scene.

### Soundness
3

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
5

### Summary
The paper proposes a training-free, test-time keyframe retrieval framework to improve long-form video understanding of MLLMs by constructing hierarchical keyframe sets from three complementary perspectives: Relevance (query-frame similarity), Summarization (representative sub-scenes), and Causality (frames contextually related to relevant ones). These are blended into several video sampling pools at different ratios and during inference, the backbone MLLM dynamically adjusts to the best pool. The proposed method is tested on 3 long video benchmarks and achieves good performance gains.

### Strengths
1.	The proposed approach is training free.
2.	Good performance gain on the evaluated benchmarks.

### Weaknesses
1.	Method: 

  a.	This paper is an engineering integration of existing modules, rather than being conceptually novel.

  b.	The idea of hierarchical segmentation using cosine similarity thresholds is essentially identical to the idea of hierarchical temporal clustering in VideoTree [1]. The only “new” detail is using two thresholds, which is a minor parametric extension. 

  c.	Closed-loop adaptive feedback is a query heuristic, not a new test-time optimization method. The two feedback modes are simple conditional branching depending on context necessity or missing information. Additionally, there is no metrics to quantify how often feedback triggers or how well it aligns with improved answers.

  d.	Frame relevance scoring is done through FG-CLIP [2] without any modification to it. The “hybrid” relevance term is just a linear combination of global and patch-level similarities.

  e.	The idea of “causality” evaluation is misleading. It only measures semantic correlation via masked caption similarity. This has nothing to do with temporal causality, as the metric is directionless and easily triggered by co-occurrence or lexical overlap. It seems like the paper confuses correlation with causation.

  f.	The idea of adaptive multi-pool blending and routing is ad hoc reweighting of prior manually tuned heuristics, not algorithmic novelty. The method doesn’t learn to adjust ratios; it predefines them and uses rule-based switching.

2.	Experiments - 

  a.	The framework essentially involves five stages, yet, the cost analysis (runtime, FLOPs, latency, compute overhead) of such complex architecture and its cost comparison to prior similar methods were never shown.

  b.	No quantitative ablation is given to show contribution of each of these modules to final performance and their individual isolated improvement (if any) over Qwen/LLaVA Video.

  c.	The framework needs four separate large-model forward passes per video before the actual answer step. However, none of these modules are jointly optimized or even co-tuned. These models have incompatible embedding spaces yet are combined without calibration or normalization. There is no theoretical or empirical validation that cosine scores from different encoders are comparable or composable. 

  d.	LLaVA Video/Qwen is instruction-tuned; whereas, CLIP is contrastively pre-trained. Mixing their embeddings in one pipeline assumes semantic alignment across scales; which is neither proven nor analyzed.

  e.	Tables 2 and 3 report results for different sampling strategies and “closed-loop” variants, but the paper never specifies how the “optimal” was chosen. In the supplementary Tables S2 and S3, different relevance/summarization/causality mixes yield the best results for each dataset, implying that the “best” combination may have been manually selected post-evaluation, and thus undermining the generalizability and adaptability claims.

### Questions
LVU is not an established abbreviation for the term “long-form video understanding”, rather it’s a benchmark by the same name [3]. Using the term LVU to refer to the task of long-form video understanding is extremely confusing and rather misleading. 

Please refer to weakness for more questions.

[1] Wang et al. VideoTree: Adaptive tree-based video representation for LLM reasoning on long videos. CVPR 2025.
[2] Xie et al. FG-CLIP: Fine-Grained Visual and Textual Alignment. ICML 2025
[3] Wu et al. Towards long-form video understanding. CVPR 2021.

### Soundness
2

### Presentation
2

### Contribution
1
