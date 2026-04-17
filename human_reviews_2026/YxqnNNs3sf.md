# InternSVG: Towards Unified SVG Tasks with Multimodal Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
General SVG modeling remains challenging due to fragmented datasets, limited transferability of methods across tasks, and the difficulty of handling structural complexity. In response, we leverage the strong transfer and generalization capabilities of multimodal large language models (MLLMs) to achieve unified modeling for SVG understanding, editing, and generation. We present the InternSVG family, an integrated data–benchmark–model suite. At its core is SAgoge, the largest and most comprehensive multimodal dataset for SVG tasks, encompassing both static graphics and dynamic animations. It covers icons, long-sequence illustrations, scientific diagrams, and dynamic animations, supporting tasks of varied difficulty levels and providing deeper hierarchies with richer attributes compared to previous datasets. Based on this resource, we introduce SArena, a companion benchmark with comprehensive task definitions and standardized evaluation that aligns with the domains and difficulty spectrum covered by SAgoge. Building on these foundations, we propose InternSVG, a unified MLLM for SVG understanding, editing, and generation with SVG-specific special tokens, subword-based embedding initialization, and a two-stage training strategy that progresses from short static SVGs to long-sequence illustrations and complex animations. This unified formulation induces positive transfer and improves overall performance. Experiments on \benchset and prior benchmark confirm that InternSVG achieves substantial gains and consistently outperforms leading open and proprietary counterparts.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper aims to address the challenges of fragmented datasets and poor method transferability in the current field of SVG modeling. To this end, the authors propose a comprehensive solution named "InternSVG," which consists of three parts:
1.  **SAgoge**: A large-scale (over 16 million samples) and comprehensive multimodal SVG dataset.
2.  **SArena**: A corresponding standardized benchmark for evaluation.
3.  **InternSVG**: A unified Multimodal Large Language Model (MLLM), which is based on the mainstream ViT-LLM architecture and optimized with methods such as SVG-specific tokenization and a two-stage training strategy.

### Strengths
1.  **Dataset and Benchmark Contribution**: The most prominent strength of this paper is the creation of the SAgoge dataset and the SArena benchmark. This is a tremendous engineering contribution that provides an extremely valuable resource to the SVG research community. The scale, diversity (especially the inclusion of chemical structures and animations), and unified coverage of understanding, editing, and generation tasks in SAgoge fill a significant gap in the field.
2.  **Effectiveness of Unified Modeling Validated**: The paper clearly argues for and experimentally demonstrates the advantages of unified modeling. By jointly training on understanding, editing, and generation tasks within the same model, the model achieves better generalization and performance.
3.  **Comprehensive Experimental Evaluation**: The authors conducted a series of thorough experiments, comparing their method against a wide range of baselines, including traditional methods, general-purpose MLLMs, proprietary models, and specialized SVG generation models. The InternSVG model achieved state-of-the-art (SOTA) performance on their proposed SArena benchmark, which fully demonstrates the effectiveness of their method suite.

### Weaknesses
1.  **High Reliance on Synthetic Data**: A large portion of the illustrations and animations in the dataset are synthesized using other generative models (e.g., FLUX, Claude), and the annotations rely on automated processes (e.g., GPT-4o). This poses a potential risk: the dataset may contain systematic biases or stylistic limitations originating from these generation/annotation models, and may not fully reflect the diversity and complexity of SVGs created by human designers in the real world.
2.  **Insufficient Evaluation of Generated Code Quality**: The paper's evaluation metrics primarily focus on the post-rendering visual effects (e.g., PSNR, SSIM, FID). However, as a vector format, the quality of the SVG code itself—such as conciseness, structural soundness, and editability—is equally crucial. The paper lacks a direct quantitative evaluation of the quality of the generated code. It is possible that the model generates visually correct but redundant and hard-to-maintain SVG code.

### Questions
1.  Regarding synthetic data bias: Have the authors analyzed potential issues of pattern repetition or stylistic homogenization introduced by the synthetic data pipeline (e.g., using FLUX and Claude models)? Is there a noticeable gap in complexity or creativity when comparing this synthetic data to real, human-authored SVGs?
2.  Regarding SVG code quality: Besides visual evaluation, have the authors considered introducing metrics to assess the quality of the generated SVG *code* itself? For instance, code length (token count), path complexity, or the number of nodes. This would be very important for measuring the model's efficiency and the editability of the generated content.
3.  Regarding the challenges of the "style transfer" task: The paper mentions that the model's performance on the "style transfer" task is slightly inferior to the strongest baseline. Could the authors provide a specific analysis of the unique challenges this task presents to the model, which makes it difficult for the current unified training framework to handle perfectly?

### Soundness
3

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
This paper focuses on the tasks of Scalable Vector Graphics (SVG) understanding, generation, and editing. To address the limitations of prior works on SVG tasks, the paper proposes (1) a new large-scale and diverse SVG dataset SAgoge, (2) a new benchmark SArena with comprehensive tasks for SVG understanding, generation and editing and (3) a specialized MLLM: InternSVG for SVG tasks, built upon pre-trained VLMs with proposed special tokens and a two-stage training strategy. Experimental results demonstrate that InternSVG achieves superior performance on various SVG tasks.

### Strengths
* The unified modeling of SVG understanding, editing, and generation is an interesting and important problem, since there may be various practical applications, while the problem is still underexplored.
* The paper provides a comprehensive contribution to SVG tasks, with an integrated data-benchmark-model suite.
* There are thorough experiments on the newly introduced SArena benchmark, and the performance of the proposed InternSVG model outperforms both existing SVG methods and various MLLM models.

### Weaknesses
* The experiments in the main paper are conducted primarily on the proposed SArena benchmark. Given that SArena is developed in alignment with the training data SAgoge (probably the same data source & similar task definitions), the comparisons are not fair. More experiments on other existing benchmarks would be necessary to fully validate the effectiveness of the proposed data & methods.
* The illustration of the dataset / benchmark construction pipeline in the main paper is too brief, making it difficult to assess the methodological novelty or quality of the pipeline. And some details are not clear, e.g. how the annotations of the SVG editing task are obtained?
* Some ablations are missing. The paper proposes special tokens and an embedding initialization strategy for InternSVG training. But there is no ablation study about the impact of these designs on the task performances.

### Questions
For the SVG understanding task, have the authors considered or conducted experiments comparing the SVG-based understanding with the rendered image/video based understanding?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents the InternSVG family, a unified suite that integrates dataset, benchmark, and model for SVG tasks. It introduces SAgoge, a large-scale multimodal SVG dataset with 16 million samples; SArena, a unified benchmark covering understanding, editing, and generation; and InternSVG, a MLLM designed for unified SVG modeling. Extensive experiments show that InternSVG achieves state-of-the-art results across multiple SVG-related tasks.

### Strengths
1. The work establishes a coherent and large-scale suite for SVG research, combining SAgoge and SArena to support consistent training and evaluation across understanding, editing, and generation tasks.
2. The model employs SVG-specific tokenization and a subword-based initialization strategy, together with a two-stage training scheme that reduces sequence length, stabilizes optimization, and speeds up convergence.
3. Comprehensive evaluations demonstrate that InternSVG achieves the best performance on all SArena sub-tasks, surpassing strong proprietary models such as GPT-4o and Claude-4-Sonnet.
4. The paper is well organized and easy to follow.

### Weaknesses
1. Since SAgoge’s annotations are generated using InternVL and QwenVL, which are also included as baselines, it remains unclear whether this setup introduces a bias favoring models from the same family during evaluation.
2. InternSVG is trained on large, SVG-specific data, whereas other models are used without such domain fine-tuning, making it unclear whether the gains arise from architectural advances or from data advantages.

### Questions
See Weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents InternSVG, an integrated data–benchmark–model suite designed for unified modeling of Scalable Vector Graphics (SVG) tasks — encompassing understanding, editing, and generation. The work consists of three main components:
1. SAgoge — a large-scale (16M samples) multimodal dataset for SVG tasks covering icons, illustrations, chemical diagrams, and animations.
2. SArena — a standardized benchmark for evaluating understanding, editing, and generation performance across SVG domains.
3. InternSVG — a unified multimodal large language model (MLLM) built on a ViT–MLP–LLM backbone, augmented with SVG-specific tokenization (including tag, attribute, and numeric tokens) and subword-based embedding initialization. It is trained in a two-stage curriculum from simple to complex SVGs.

### Strengths
1. The work takes a holistic and unified approach to SVG understanding, editing, and generation — previously treated as isolated problems.
2. The paper is clearly written and well-organized, with informative figures illustrating datasets, architecture, and tokenization.
3. The open dataset and benchmark will likely become an important reference for future research on vector graphics and multimodal LLMs.

### Weaknesses
1. While the paper demonstrates strong empirical results, it provides limited theoretical or interpretive analysis of why unified modeling helps cross-task transfer. For example, a visualization or probing analysis of SVG token embeddings could strengthen understanding of learned representations. 
2. The annotation pipeline leverages GPT-4o, Gemini, and other proprietary models for labeling, which may raise concerns about the bias. It would be helpful to involve human evaluation on a small subset of the synthetic data. 
3. While quantitative metrics are comprehensive, qualitative results (e.g., in Figure 4) could be expanded to better illustrate failure cases or tradeoffs between semantic accuracy and geometric precision.
4. It would be helpful if the cross-domain transfer (e.g., training on icons → testing on scientific diagrams) could be further analyzed to highlight the generalization strength.
5. The authors mention replacing unrenderable SVGs with black images for evaluation. It would be helpful if the authors could quantify how often InternSVG produces invalid SVGs compared to other baselines. This would clarify the model’s robustness and syntactic correctness.

### Questions
1. Could you provide more analysis or visualization to help understand how unified modeling leads to cross-task improvements? For example, have you examined whether representations learned from SVG understanding tasks transfer effectively to generation tasks (e.g., through probing, clustering, or layer-wise similarity analysis)? Such insights could strengthen the paper’s conceptual contribution beyond empirical results.
2. While quantitative metrics are comprehensive (FID, SSIM, CLIP, etc.), have you conducted any human evaluation on the perceptual or semantic quality of generated SVGs? If not, could you comment on whether the model-generated outputs align with human preference or creative intent?

### Soundness
3

### Presentation
2

### Contribution
2
