# IVEBench: Modern Benchmark Suite for Instruction-Guided Video Editing Assessment

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Instruction-guided video editing has emerged as a rapidly advancing research direction, offering new opportunities for intuitive content transformation while also posing significant challenges for systematic evaluation. Existing video editing benchmarks fail to support the evaluation of instruction-guided video editing adequately and further suffer from limited source diversity, narrow task coverage and incomplete evaluation metrics. To address above limitations, we introduce IVEBench, a modern benchmark suite specifically designed for instruction-guided video editing assessment. IVEBench comprises a diverse database of 600 high-quality source videos, spanning seven semantic dimensions, and covering video lengths ranging from 32 to 1,024 frames. It further includes 8 categories of editing tasks with 35 subcategories, whose prompts are generated and refined through large language models and expert review. Crucially, IVEBench establishes a three-dimensional evaluation protocol encompassing video quality, instruction compliance and video fidelity, integrating both traditional metrics and multimodal large language model-based assessments. Extensive experiments demonstrate the effectiveness of IVEBench in benchmarking state-of-the-art instruction-guided video editing methods, showing its ability to provide comprehensive and human-aligned evaluation outcomes. All data and code will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces IVEBench, a comprehensive benchmark suite designed to address the shortcomings of existing evaluation methods for instruction-guided video editing. The authors identify three limitations in current benchmarks: insufficient source video diversity, limited scope of editing tasks, and incomplete evaluation metrics. To overcome these challenges, this paper proposed to source more diverse videos, enriching the prompt types and introduce better metrics for evaluation, resulting in the IVEBench benchmark which contains 600 videos.

### Strengths
Main contributions:

1.  A new dataset of 600 high-quality, high-resolution source videos, categorized across seven semantic dimensions and split into short (32-128 frames) and long (129-1,024 frames) subsets to test model scalability. The IVEBench includes a set of 600 editing instructions spanning eight major task categories and 35 subcategories, which go beyond simple style/subject edits to include complex temporal tasks like camera motion and subject motion editing.
2.  This paper proposed a three-dimensional evaluation framework assessing **Video Quality**, **Instruction Compliance**, and **Video Fidelity**. This eval framework integrates traditional metrics with MLLMs-based assessments for a more fine-grained and human-aligned evaluation. The Spearman's Rho correlations (e.g., 0.98 for IS, 0.99 for VTSS, 0.94 for SF) provide strong evidence that the metrics are aligned with human perception.

### Weaknesses
1. First, the benchmark's construction creates an idealized environment that may not reflect the challenges of real-world video editing. The dataset is composed of high-quality, professional video, which does not represent the noisy in-the-wild videos that users commonly edit. Similarly, the editing prompts are not complex enough (too short and too "clean"), failing to test a model's ability to handle the messy prompts/ This gap between the benchmark's sanitized conditions and real-world complexity limits the generalizability of the performance scores, as models that excel on IVEBench may still falter on more realistic user-generated content and commands.

2. Second, the evaluation protocol, while interesting, has weaknesses related to its reproducibility and objectivity due to its heavy reliance on specific MLLMs - Qwen2.5-VL.  The use of MLLMs in eval, while has some merits as I mentioned, still fall short in lack of explanability, creating a risk of hidden biases. I don't see the novelties of evaluating such a VE task with MLLMs.

3. The benchmark's scope does not fully encompass the complexity and scale of advanced video editing tasks. Especially in terms of the quantity - 600 videos with 35 topics are not enough to cover the complex and dynamic nature of real-world videos. The authors need to clearly explain why only source 600 videos like the cost of creating such benchmarks.

### Questions
N/A

### Soundness
4

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
4

### Summary
This paper introduces IVEBench, a new and modern benchmark designed to properly test today's instruction video editing models. The authors point out that existing benchmarks are falling behind; they use a limited variety of videos, cover only basic editing tasks (like changing style or swapping an object), and rely on outdated metrics that don't capture the full picture.

To tackle this, IVEBench brings three big upgrades. First, it features a diverse collection of 600 high-quality videos, spanning different themes, scenes, and lengths. Second, it includes a massive range of editing tasks: 8 categories with 35 sub-types, from simple attribute changes to complex camera movements—with prompts generated by LLMs and polished by experts. Finally, it introduces a smarter, three-dimensional evaluation system: Video Quality, Instruction Compliance, and Video Fidelity. Crucially, it leverages Multimodal LLMs to score complex edits that traditional metrics can't handle, ensuring the results are much closer to what a human would think.

### Strengths
1. A Massive and Diverse Dataset: Unlike older benchmarks with a handful of videos, IVEBench provides 600 clips, including both short and long ones (up to 1,024 frames!). This variety really pushes models to prove they can work on all kinds of content, not just cherry-picked examples.
2. Goes Beyond Basic Edits: The benchmark covers editing tasks that are specific to video, like adding camera motion or changing subject motion. This is a huge step up from benchmarks that just treat video as a series of images.
3. Smart, MLLM-Powered Evaluation: Using MLLMs to judge "Instruction Satisfaction" and "Content Fidelity". It allows the benchmark to automatically score tricky instructions (e.g., "change the camera to a high angle") that are almost impossible to measure with old-school pixel-based metrics.
4. Strong Alignment with Human Judgment: This work show that their automated metrics are highly correlated with human preferences (with Spearman's Rho scores often above 0.9). This means the benchmark isn't just spitting out numbers—it's reflecting what people actually see as a "good" or "bad" edit.

### Weaknesses
1. The initial evaluation only tests four models. While these are representative, the benchmark's true power will be seen when it's used to compare a much wider range of open-source and commercial models over time.
2. Potential Bias in MLLM-Based Evaluation: Relying on a specific MLLM (Qwen2.5-VL) for scoring could introduce subtle biases. For instance, if a new video editing model is built on a similar architecture to Qwen, the evaluator might unintentionally favor it. Diversifying the MLLM evaluators in the future could make the results even more robust.

### Questions
1.  The MLLM-based metrics are fantastic for evaluating complex instructions. But how do you plan to prevent the benchmark from becoming an "echo chamber," where editing models are trained to please a specific MLLM evaluator rather than genuine human users?

2. Edit instructions are often messy and combine multiple steps (e.g., "Make the cat look like it's made of fire and have it jump onto the table"). Does IVEBench have a plan to assess these kinds of compositional, multi-part edits in the future?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents IVEBench, a benchmark for evaluating instruction-guided video editing. It includes 600 videos with diverse content and multiple editing categories, along with a three-dimensional evaluation protocol covering video quality, instruction compliance, and video fidelity. The authors also combine traditional metrics with multimodal large language model evaluations to better capture semantic alignment between edits and instructions.

### Strengths
The paper provides a well-structured benchmark in evaluating instruction-based video editing with diverse videos and a clear evaluation framework. The idea of combining automatic metrics with MLLM-based assessments is creative and reflects current research trends. The presentation is clear, the examples are easy to follow, and the benchmark has the potential to become a useful resource for future studies in this area.

### Weaknesses
I believe in video editing, the most crucial factor should be instruction compliance (whether the model actually does what the user asks for). From the qualitative examples, VACE achieve relatively high overall scores despite barely changing the video content (mostly just color tone adjustments). A model can still perform well even if it doesn’t really edit the content (even using the original video). This makes the benchmark a bit pointless. A weighted scoring system that puts more emphasis on instruction compliance would probably give a fairer picture of real editing performance.

### Questions
I just wanted to ask the authors if they’ve thought about reweighting the metrics so that instruction compliance has a bigger impact on the final score?

### Soundness
3

### Presentation
4

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
This paper presents IVEBench, a modern benchmark suite specifically designed for Instruction-guided Video Editing (IVE), aiming to address the limitations of existing benchmarks, including insufficient diversity of video sources, narrow task coverage, and incomplete evaluation metrics. The suite contains 600 high-quality source videos covering 7 semantic dimensions, with frame lengths ranging from 32 to 1024 frames, divided into a short-sequence subset (400 videos) and a long-sequence subset (200 videos). It includes 8 major categories and 35 subcategories of editing tasks, generated by large language models (LLMs) and refined by experts.

IVEBench establishes a three-dimensional evaluation protocol—comprising Video Quality, Instruction Compliance, and Video Fidelity—integrating 12 metrics, including both traditional measures and MLLM-assisted assessments. In tests on four state-of-the-art IVE models, such as InsV2V and AnyV2V, IVEBench demonstrates high alignment with human perception (e.g., the VTSS metric achieves a Spearman correlation of 0.9982), and reveals existing model limitations in handling long videos (frame lengths exceeding 128 frames may cause memory overflow) and executing complex instructions (e.g., camera angle editing), providing a comprehensive standard for IVE evaluation.

### Strengths
1. **Comprehensive and well-constructed video and prompt design:**  
   The paper collects a dataset of 600 videos across 7 dimensions, with each video paired with a corresponding prompt covering 8 different editing tasks. The dataset is large-scale and well-curated, with quality ensured through a combination of automatic preprocessing and manual screening. This provides a reliable foundation for evaluating video editing models.

2. **Thorough evaluation framework:**  
   The study employs three primary evaluation dimensions and 12 fine-grained metrics, assessing video quality from multiple perspectives. This comprehensive approach ensures that different aspects of video editing performance are systematically measured.

3. **Provides insightful analysis:**  
   Using the proposed evaluation framework, the authors analyze existing video editing models and reveal meaningful observations, such as high frame-to-frame consistency, weak single-frame quality, and limited support for diverse editing prompt types. These insights highlight both the strengths and current limitations of state-of-the-art video editing models.

### Weaknesses
1. **Ambiguity and overlap among evaluation dimensions:**  
Although the paper proposes a comprehensive set of evaluation metrics, there exists noticeable overlap among certain dimensions. For example, both *Overall Semantic Consistency (OSC)* and *Instruction Satisfaction (IS)* aim to assess semantic alignment between the edited video and the given prompt, focusing on similar aspects of overall semantic coherence. This overlap raises concerns about the independence and discriminative validity of each metric.

2. **Limited reliability of human alignment experiments:**  
The *human alignment* evaluation is conducted using only ten source videos, which is a relatively small sample size and may limit the representativeness of the results. Furthermore, the approach of directly using human rating distributions introduces subjectivity and potential bias, since human perceptual judgments are difficult to map linearly onto quantitative scales. The authors should clarify how they ensure consistency and statistical reliability in this human evaluation process.
3.**Limited diversity of evaluated models:**  
The paper reports results from only four video editing models. This limited coverage weakens the generality of the analysis and makes it difficult to assess the broader applicability of the proposed benchmark. Evaluating more diverse or state-of-the-art models would significantly strengthen the conclusions.

### Questions
1. **On evaluation efficiency:**  
   Given the large number of proposed evaluation metrics and assessment procedures, is there potential redundancy among them? Could the evaluation framework be simplified while maintaining its effectiveness? It would also be helpful to know the average time required to evaluate a single video under the current protocol.
2. **On the validity of the human rating process:**  
How do the authors ensure that subjective human ratings align with objective quantitative metrics? Are there measures such as inter-rater reliability or calibration steps to minimize inconsistency among annotators?

### Soundness
2

### Presentation
3

### Contribution
3
