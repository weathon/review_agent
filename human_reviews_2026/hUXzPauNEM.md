# VisCodex: Unified Multimodal Code Generation via Merging Vision and Coding Models

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Multimodal large language models (MLLMs) have significantly advanced the integration of visual and textual understanding. However, their ability to generate code from multimodal inputs remains limited. In this work, we introduce VisCodex, a unified framework that seamlessly merges vision and coding language models to empower MLLMs with strong multimodal code generation abilities. Leveraging a task vector-based model merging technique, we integrate a state-of-the-art coding LLM into a strong vision-language backbone, while preserving both visual comprehension and advanced coding skills.

To support training and evaluation, we introduce the Multimodal Coding Dataset (MCD), a large-scale and diverse collection of 598k samples, including high-quality HTML code, chart image-code pairs, image-augmented StackOverflow QA, and algorithmic problems. Furthermore, we propose InfiBench-V, a novel and challenging benchmark specifically designed to assess models on visually-rich, real-world programming questions that demand a nuanced understanding of both textual and visual contexts.

Extensive experiments show that VisCodex achieves state-of-the-art performance among open-source MLLMs and approaches proprietary models like GPT-4o, highlighting the effectiveness of our model merging strategy and new datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the limited multimodal code generation abilities of current MLLMs and proposes VisCodex, a framework that merges a vision language model with a code LLM through task vector based model fusion. Supported by the newly constructed MCD dataset and the InfiBench V benchmark, the approach achieves state-of-the-art performance among open-source models. Its main contributions lie in efficiently extending model fusion techniques to a new scenario and building high-quality data and benchmarks. However, the method offers limited novelty, the experiments are not sufficiently comprehensive, and several technical details remain unclear. The work requires additional key details and experiments to improve completeness, but it still carries certain academic and practical value.

### Strengths
1. Clear and practically relevant problem setting
The paper focuses on enabling multimodal models to generate functional code from visual inputs, a capability that is highly useful in real scenarios such as UI-to-code and chart reconstruction.

2. Simple and effective model fusion design
The task-vector–based fusion method offers a lightweight way to combine visual and coding capabilities without full retraining, and ablations show consistent gains across model scales.

3. Constructed datasets and benchmarks with solid empirical utility
The MCD dataset and InfiBench-V provide diverse, visually grounded coding tasks, and human evaluation supports the reliability of the benchmark.

4. Competitive performance among open models
VisCodex surpasses open-source baselines and achieves results close to proprietary systems on visually grounded coding tasks.

### Weaknesses
| 1. Unclear source of improvement and limited evidence of true multimodal fusion

Although the model reports strong multimodal code generation results, the source of gains is unclear. The merged model uses CodeQwen2.5-7B as the code branch but does not report its standalone performance on the same benchmarks, making it impossible to separate the benefits of “vision–code unification” from the inherent strength of the code model. Moreover, since both Qwen2.5-VL and CodeQwen2.5 share the same architecture, it is uncertain whether improvements stem from genuine multimodal fusion or simply architectural compatibility, weakening the causal claim of the paper.

| 2. Task vector representation may accumulate bias and lacks theoretical justification

The method assumes that the difference between a fine-tuned model and its base model cleanly represents a task, but such vectors also contain optimization noise and undesirable parameter shifts. When summing multiple task vectors—especially across heterogeneous domains like vision-language and code—these biases may accumulate and distort semantics. The paper provides no analysis of such interactions, and the balancing factor λ is chosen by grid search rather than learned, suggesting the method may rely on heuristic tuning rather than a principled understanding of knowledge composition in parameter space.

| 3. Dataset and Benchmark Biases Undermine Generalizability

Both the self-constructed MCD dataset and the InfiBench-V benchmark introduce potential biases that reduce the reliability and generalizability of the reported results.

MCD Dataset
The chart image–code subset contains 164k synthetic samples (from ChartCoder) and only 46k real GitHub samples, yet the paper does not quantify the distribution gap in coding style or chart complexity. Heavy reliance on synthetic samples risks overfitting to artificial patterns and weakens real-world chart-to-code applicability.

The HTML code subset is generated by GPT-4o from webpage screenshots, but the paper does not disclose the prompt template. If the prompt implicitly enforces specific coding conventions (e.g., fixed CSS frameworks), the resulting code may lack diversity and fail to reflect real UI development practices, where developers rely on heterogeneous toolchains (Tailwind, Bootstrap, custom CSS).

InfiBench-V Benchmark
The benchmark only retains StackOverflow questions with accepted answers and filters out text-only cases. However, the paper does not analyze whether the remaining 322 samples include difficult real-world cases such as low-resolution screenshots or ambiguous visual-text inconsistencies. This narrow curation limits the benchmark’s ability to evaluate robustness in realistic multimodal coding scenarios.

### Questions
1. Baseline Comparisons with Specialized Multimodal Code Models
 Why were recent multimodal code models such as CodeV, ChartCoder, or Plot2Code excluded as baselines? These models directly target visual-to-code tasks and are more relevant than generic VL models. If VisCodex were evaluated against them on MCD or InfiBench-V, would the SOTA claim still hold, particularly on chart-to-code or vision-augmented code QA tasks?
2. Robustness and Failure Analysis
 The paper reports average scores but lacks robustness evaluation and failure case studies.
How does VisCodex perform under realistic noisy inputs (e.g., blurred or low-resolution images, ambiguous textual prompts)?
After merging the code vector, does the model retain visual understanding performance (e.g., VQA, captioning), or does catastrophic forgetting occur?
3. Dataset Construction Details
 For the GPT-4o-generated HTML samples in MCD, what specific prompt template was used?
Did the prompt enforce particular coding conventions (CSS frameworks, layout constraints)?
Was there validation (e.g., sampling real-world HTML from open-source sites) to ensure diversity and authenticity?
4. Task Vector Interactions
 Did you analyze potential bias accumulation when summing task vectors?
For example, visualizing semantic directions of τ_vlm and τ_code in parameter space, or correlating changes in λ with performance shifts on specific benchmarks.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes VisCodex, a task-vector model-merging method that injects a code LLM’s capability into a VLM by linearly combining task vectors in the language backbone while freezing the vision encoder and projector. It introduces a 598k-sample Multimodal Coding Dataset and a new benchmark, InfiBench-V.

### Strengths
1. Clear, formalized merging recipe with explicit task-vector definitions and a single-parameter interpolation;
2. The paper introduces a compute-efficient design, only the LLM backbone is merged or tuned but vision & projector are frozen;
3. The authors introduce a dataset large, diverse MCD and a benchmark (InfiBench-V) targeting visually-rich programming questions;
4. It reaches several strong numbers on Design2Code/ChartMimic, using a 33B model close to GPT-4o on average.

### Weaknesses
1. The paper shows example items evaluated by "Judge: GPT-4o" to 50/50 component scores, which confirms the setup, without disclosing the actual threshold or how it was chosen.
2. The paper does include an unfreezing setup, but only for the replacement baseline. For the VisCodex, training freezes the vision encoder and projector and fine-tunes only the language backbone, and the paper does not report an ablation where these modules are unfrozen after merging.

### Questions
1. What exact threshold maps 0–100 to pass/fail on InfiBench-V? Please report a threshold sweep;
2. Please add a post-merge unfreezing ablation to test for cross-modal distribution shift introduced by weight interpolation. At minimum: unfreeze projector only, unfreeze projector + ViT.

### Soundness
3

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
5

### Summary
VisCodex proposes a unified multimodal framework that merges a vision-language model with a coding LLM to enable strong code generation from visual inputs. It uses task-vector based model merging to combine visual understanding and programming ability without full retraining, preserving both skills. The authors build a 598k-sample Multimodal Coding Dataset and introduce the InfiBench-V benchmark for realistic evaluation. Experiments show VisCodex achieves state-of-the-art open-source performance, rivaling proprietary models like GPT-4o.

### Strengths
1. Introduces a model-merging based path for multimodal code generation, combining vision and coding expertise without full retraining, and expands the problem space with new data and benchmarks.

2. Demonstrates solid empirical rigor, with extensive evaluations showing consistent performance improvements over strong open-source models and competitiveness with proprietary ones.

3. Addresses a meaningful and underexplored capability, turning visual content into functional code and offering practical impact for applications like UI-to-code and chart-to-code systems and shaping future MLLM research directions.

### Weaknesses
1. The comparison to direct SFT strategies is limited in scope; while the paper includes one- and two-stage baselines, a broader evaluation (e.g., LoRA tuning on both vision and language modules) would strengthen the claim that merging is strictly superior for this setting.

2. The dataset construction pipeline relies heavily on model-generated content (e.g., GPT-4o generated HTML and curated chart code data) but lacks detailed analyses of potential data bias, overfitting to synthetic structures, or failure modes on fully real-world screenshots/code artifacts; evaluating broader generalization and reporting error breakdowns would improve credibility.

### Questions
Could you further clarify why the direct SFT strategies underperform beyond visual grounding disruption, and provide more analysis or diagnostics to better understand what specific capabilities are degraded during tuning?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces VisCodex, a unified multimodal code generation and understanding framework. It combines newly curated training and evaluation datasets with a model merging strategy that fuses multiple task-specific models (e.g., vision-language and code-language) to enhance cross-modal reasoning. The method aims to balance specialization and generalization across visual programming, text-to-code, and code reasoning tasks. Experiments on multiple benchmarks demonstrate competitive or superior results to single-task fine-tuning.

### Strengths
- Solid and comprehensive contribution: The work offers not only a new training method (model merging) but also new datasets for both training and evaluation, which strengthens its empirical foundation.

- Methodologically clear: the overall paper is well-written, organized, and easy to follow.

- Breadth of evaluation: Covers diverse multimodal coding tasks, showing consistent, though not always dramatic, gains across small and medium-sized models.

- Timely and relevant: Addresses the challenge of unifying multimodal reasoning (code + vision + text), which is a growing area of interest.

### Weaknesses
- Unclear advantage over standard fine-tuning: As shown in Table 2, model merging offers almost no improvement for large models (e.g., 33B variant), suggesting diminishing returns at scale. This weakens the claim of broad effectiveness.

- Limited discussion on data-scarce scenarios: One key potential advantage of model merging could be in low-resource multimodal settings, yet this is not explored. It’s unclear whether the approach would still help when task-specific data is limited.

- Lack of deeper analysis: The paper could have benefited from probing why merging helps (or doesn’t) at different scales — for example, via representational similarity or parameter-space studies.

### Questions
- What is the concrete benefit of model merging when extensive SFT is still required afterward?

- Have the authors evaluated performance under data-scarce conditions to demonstrate merging’s potential efficiency?

- For large models (33B), is the marginal gain statistically significant, or within noise level?

### Soundness
3

### Presentation
3

### Contribution
3
