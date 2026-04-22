# OpenGPT-4o-Image: A Comprehensive Dataset for Advanced  Image Generation and Editing

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 2, 6

## Abstract
The performance of unified multimodal models for image generation and editing is fundamentally constrained by the quality and comprehensiveness of their training data. While existing datasets have covered basic tasks like style transfer and simple object manipulation, they often lack the systematic structure and challenging scenarios required for real-world applications. To address this bottleneck, we introduce \textbf{OpenGPT-4o-Image}, a large-scale dataset constructed using a novel methodology that combines hierarchical task taxonomy with automated data generation. Our taxonomy not only includes fundamental capabilities such as {text rendering} and {style control} but also introduces highly practical yet challenging categories like \textbf{scientific imagery} for physics/chemistry illustrations and \textbf{complex instruction editing} requiring simultaneous execution of multiple operations. Through an automated pipeline leveraging structured resource pools and GPT-4o, we generate 80k high-quality instruction-image pairs with controlled diversity, covering 11 major domains and 51 subtasks. Extensive experiments show that fine-tuning leading models on our dataset achieves significant performance gains across multiple benchmarks, with improvements of up to 18\% on editing tasks (UniWorld-V1 on ImgEdit-Bench) and 13\% on generation tasks (Harmon on GenEval). Our work demonstrates that systematic data construction is key to advancing multimodal AI capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the limitations of current multimodal models in image generation and editing tasks due to insufficient data quality and coverage by proposing the OpenGPT-4o-Image dataset. Its core contributions include a hierarchical task taxonomy (covering multiple domains and subtasks, such as style control, scientific image generation, and complex instruction editing) and a GPT-4o-driven automated pipeline, which generates 80k high-quality instruction-image pairs, ensuring data diversity, controllable difficulty, and semantic accuracy. Experimental validation shows that the dataset significantly improves model performance.

### Strengths
1. This work provides a systematic, data-centric pipeline for advancing generative AI, addressing the critical bottleneck of training data for real-world applications in specialized domains. 

2. The proposed dataset is a valuable, as it partly addresses the scarcity of high-quality data for image generation and editing.

3. The paper is well-written and easy to follow. It features a logical flow, precise definitions, and effective use of figures to illustrate the dataset's scope and the automated pipeline's workflow.

### Weaknesses
1. The core method heavily relies on GPT-4o-Image as a data generation engine, which introduces the risk of model bias. Although the paper acknowledges this limitation, it does not delve into or quantify the specific manifestations of this bias. This constrains the long-term evolvability of the dataset, as its quality is intrinsically tied to the capabilities of a closed-source model.

2. The paper emphasizes proactive quality control through hierarchical taxonomy. However, this approach remains a priori and rule-based design, rather than an outcome-based validation. A critical gap is the lack of a rigorous human or automated verification step to systematically assess the semantic fidelity of each generated sample.

3. The paper omits a crucial ablation study to validate its central claim—that the hierarchical taxonomy enhances data quality. For instance, an experiment comparing against fine-tuning on an unsystematized subset of the data would be necessary to demonstrate the value of the proposed method.

### Questions
Please see Weaknesses Section.

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
The paper addresses the data bottleneck in training multimodal models for image generation and editing tasks. OpenGPT-4o-Image (80k) is introduced as a large-scale dataset, systematically constructed entirely from GPT-4o-generated instruction–image pairs, with the goal of enhancing multimodal model performance. Experimental results indicate that mainstream models (e.g., UniWorld-V1, Harmon), when fine-tuned on this GPT-4o-generated dataset, achieve up to 18% improvement in editing tasks (ImgEdit-Bench) and up to 13% improvement in generation tasks (GenEval), surpassing contemporaneous works such as ShareGPT-4o-Image.

### Strengths
1. Comprehensive multi-dimensional validation: performance variation is assessed across different models (four mainstream models), different tasks (generation + editing), and different data scales (20K / 30K / 40K). Significant improvements are observed across most tasks, verifying the dataset’s effectiveness.
2. Fine-tuned models show notable gains in tasks such as in-image text rendering (e.g., menu generation, calligraphic text) and spatial reasoning (e.g., object-relative positioning, symmetry analysis), making them readily applicable to real-world scenarios in office and design domains.
3. Introduction of scientific imagery as an independent module, covering specialized domains such as mathematics, physics, and ecology. This effectively addresses the scarcity of professional technical illustrations in existing datasets.

### Weaknesses
The dataset is entirely generated via the GPT-4o API , with mitigation of GPT-4o’s inherent biases (e.g., semantic skew, stylistic preferences). Such biases may be propagated into fine-tuned models. For instance, GPT-4o-generated editing data is known to have weaker ID consistency and background preservation, which can degrade ID retention in fine-tuned models, as illustrated by the “dog” case in Figure 6.

### Questions
1. Since the editing dataset is generated with GPT-4o and the generated outputs are in principle not perfect ground-truth references (for example, ID consistency may degrade), advances in editing technology may raise concerns about the dataset’s long-term relevance. If stronger editing models Edit become available, could this dataset still provide measurable performance gains when used for fine-tuning these more capable models? Can you test it on the recent Qwen-Image Edit Model?
2. Given the observed issue that GPT-4o-generated editing data may impair ID retention capability, what strategies could be adopted to alleviate this problem and enhance ID consistency in fine-tuned models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces OpenGPT-4o-Image, a dataset of 80k instruction–image pairs for image generation and image editing tasks. It claims to offer a hierarchical taxonomy covering 11 domains and 51 subtasks, and to improve fine-tuning performance for several multimodal models such as OmniGen, OmniGen2, UniWorld-V1, and Harmon.

### Strengths
The paper addresses an important bottleneck in multimodal research, which is the lack of systematically structured and high-quality datasets for unified image generation and editing. The motivation is clearly aligned with the broader trend of building large, instruction-based multimodal datasets.

The proposed hierarchical task taxonomy, which spans 11 domains and 51 subtasks, is conceptually interesting and comprehensive.

The methodology of using GPT-4o to generate structured prompts and synthetic instruction–image pairs is well-described and could be useful for future data curation work.

### Weaknesses
Table 1 makes me very confused. The table duplicates MagicBrush, UniWorld-V1, OmniGen and OmniGen2 entries multiple times, sometimes with symbols (†, ‡) that are inconsistently explained. It appears that the results in the upper half of the table were directly copied from Table 5 in OmniGen2 [1] rather than reproduced by the authors. Moreover, the results reproduced by the authors show a substantial discrepancy compared with those reported in the original paper. The row MagicBrush† and the row UniWorld-V1†, what are their differences compared to MagicBrush and UniWorld-V1 respectively, and † represents results without fine-tuning, so why are these results included in the fine-tuning section?

Similar issues are also present in Table 2.

The results in Table 2 are almost identical across different models and settings, which makes the evaluation unconvincing.

[1] OmniGen2: Exploration to Advanced Multimodal Generation

### Questions
Refer to Weaknesses.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces OpenGPT-4o-Image, a dataset of about 80k instruction-image pairs organized by a hierarchical taxonomy that spans 11 major domains and 51 subtasks for both generation and editing. The generation side covers five modules, including Style Control, Complex Instruction Following, In-Image Text Rendering, Spatial Reasoning, and Scientific Imagery. The editing side defines six categories with 21 subtasks such as Subject Manipulation, Text Editing, Complex Instruction Editing, Multi-Turn Editing, Global Editing, and other challenging edits. The data are produced through automated pipelines that use structured resource pools plus GPT-4o and gpt-image-1 for prompt and image synthesis. Fine-tuning several recent models on this dataset yields consistent gains on GenEval, DPG-Bench, GEdit-Bench, and ImgEdit-Bench, for example up to about 18 percent on editing and about 13 percent on generation.

### Strengths
S1: The generation modules are explicitly enumerated with examples, and the editing taxonomy covers 21 subtasks, which gives useful coverage for targeted training and diagnosis.    

S2: The paper specifies sample counts for several sub-areas, for example Style Control at 13k, In-Image Text Rendering at 3k, Spatial Reasoning at 8k, and Scientific Imagery at 10k, which clarifies dataset balance.        

S3: For generation, the authors define resource pools and template-based prompt construction. For editing, they integrate multiple sources and use GPT-4o for instruction creation plus inpainting, which is well documented.      

S4: The paper reports consistent improvements after fine-tuning, including Harmon’s increase on GenEval and strong boosts on ImgEdit-Bench and GEdit-Bench for several systems.

### Weaknesses
W1: The pipeline relies on GPT-4o for prompt generation and on gpt-image-1 for image synthesis and inpainting. This may imprint model-specific priors or artifacts on the dataset and can indirectly entangle training with the same ecosystem used in evaluation. The conclusion itself notes possible GPT-4o bias and the focus on existing benchmarks. A more formal bias audit or cross-ecosystem sanity checks would help.    

W2: The paper emphasizes automatic construction and benchmark-based evaluation, but I did not find a human study verifying that improvements correspond to perceived visual quality or edit faithfulness. This risks over-optimizing to the metrics used by GenEval, DPG-Bench, GEdit-Bench, and ImgEdit-Bench rather than to human preferences.  

W3: The editing data incorporate curated high-resolution images and outputs from multiple datasets and generators. The paper does not detail licensing, filtering for sensitive content, or subject consent for portrait-like material, which is important for release.  

W4: The dataset provides strong coverage in specific capability clusters, yet many real-world cases involve open-world composition or rare styles. Although the taxonomy is broad, the paper does not quantify how balanced the final 80k set is across the 51 subtasks, nor whether harder long-tail prompts are sufficiently represented.    

W5: Data scaling uses 20k to 40k subsets and picks 40k after observing a small delta. It would be helpful to extend the curve and to separate generation versus editing contributions per module.

### Questions
How do you mitigate bias from using GPT-4o and gpt-image-1 throughout the pipeline? Can you show cross-assessor validation using independent VLMs or human audits and report agreement with benchmark gains?  

What is the distribution across the 51 subtasks and 11 domains, and how do per-subtask gains correlate with data volume? Please provide a stratified breakdown with confidence intervals.  

For editing, how are licensing and safety handled for curated high-resolution images and for content from external corpora such as OmniEdit or ImgEdit? Do you filter portraits or sensitive scenes before release?  

Can you include a small human evaluation verifying that fine-tuned models improve perceived edit faithfulness and visual quality, not only metric scores on the four benchmarks?  

Could you add ablations isolating each generation module and each editing category to identify which components most drive GenEval and ImgEdit-Bench improvements?

### Soundness
3

### Presentation
3

### Contribution
3
