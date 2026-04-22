# Charts Are Not Images: On the Challenges of Scientific Chart Editing

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 8, 4, 4

## Abstract
Generative models, such as diffusion and autoregressive approaches, have demonstrated impressive capabilities in editing natural images. However, applying these tools to scientific charts rests on a flawed assumption: a chart is not merely an arrangement of pixels but a visual representation of structured data governed by a graphical grammar. Consequently, chart editing is not a pixel-manipulation task but a structured transformation problem.
To address this fundamental mismatch, we introduce \textit{FigEdit}, a large-scale benchmark for scientific figure editing comprising over 30,000 samples. Grounded in real-world data, our benchmark is distinguished by its diversity, covering 10 distinct chart types and a rich vocabulary of complex editing instructions. The benchmark is organized into five distinct and progressively challenging tasks: single edits, multi edits, conversational edits, visual-guidance-based edits, and style transfer.
Our evaluation of a range of state-of-the-art models on this benchmark reveals their poor performance on scientific figures, as they consistently fail to handle the underlying structured transformations required for valid edits. Furthermore, our analysis indicates that traditional evaluation metrics (e.g., SSIM, PSNR) have limitations in capturing the semantic correctness of chart edits. Our benchmark demonstrates the profound limitations of pixel-level manipulation and provides a robust foundation for developing and evaluating future structure-aware models. By releasing \textit{FigEdit}, we aim to enable systematic progress in structure-aware figure editing, provide a common ground for fair comparison, and encourage future research on models that understand both the visual and semantic layers of scientific charts.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses scientific chart editing, specifically proposing a new evaluation benchmark and metrics for this task and an analysis of existing editing and vision-language models on this task.

Their contributions include:
- The novel problem formulation of chart editing.
- A benchmark (FigEdit) specifically designed for scientific chart editing, containing over 30,000 instances. The dataset covers 10 chart types and various edit instructions.
- LLM-based evaluation techniques to measure structural correctness over pixel similarity, in contrast to metrics like SSIM and PSNR.
- An evaluation of state-of-the-art models on chart editing that shows severe limitations of existing methods.

### Strengths
- The paper proposes a novel problem formulation (figure editing) that could have significant and practical real-world impact across a variety of industries, like research and business.
- The paper rethinks traditional pixel-based image metrics (LPIPS, PSNR, SSIM, etc.) for better evaluating the specific task of figure editing.  This is an important research direction for other generative vision tasks, as well.
- The dataset is constructed from diverse real-world data across different fields, which prevents bias to certain distributions of data, and therefore creates diverse chart appearances.
- The paper figures are helpful at clarifying and exemplifying the different categories of editing tasks.

### Weaknesses
- There is limited evaluation / justification for why the LLM-based metric is better or more reliable than traditional metrics. A comparison between all these scores and human evaluation would be beneficial for showing whether or not the LLM metric is actually better for evaluating figure edits.
- Similarly, it would be informative to compare the LLM score per-category (Instr., Preserv., Qual.) to average human score per-category.
- Fig. 1. The fonts are quite small, especially the axes of the charts and the legend for the radar chart.
- A brief description of Vega / Vega-Lite would be helpful.
- The conversational edits are essentially just two-step multi-edits (lines 266-267)?. This seems too short of a conversation to be able to actually determine ability to maintain conversation history. Also, if they are just derived from multi-step edits, how can one be sure that the edits are actually progrssive (e.g. "make the bars red" --> "make the bars a darker red") and are not completely unrelated (e.g. "make the bars red" --> "make the bars blue")?
- The paper might consider adding several more-recent state-of-the-art editing models, such as Gemini/Nano Banana, Kling, and GPT-4o.

### Questions
- lines 204-207: Are Content (C) = {D, $\tau$, mapping function} and Style (S) = {palettes, fonts, etc.} standard in related literature? It seems arbitrary. For example, why is chart style included in C but not S?
- (Task 2, lines 216-221) Why not denote multiple atomic edits as $u_1$, $u_2$, $u_3$, etc. instead of just $u$?
- line 225: Why is $H_{t-1}$ not included in the equation for $\sigma^*$?
- (292-294) Was the reliability of GPT-Image evaluated? As in, can one trust that the correct element, and only the correct element, is always circled?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces FigEdit, a large-scale benchmark for scientific chart editing. The authors compellingly argue that chart editing is a structured transformation problem, fundamentally different from pixel-level image manipulation. By providing a benchmark with over 30,000 instances, spanning 10 chart types and 5 distinct tasks, the work systematically exposes the limitations of state-of-the-art image editing models. The paper's core contribution lies in its rigorous problem formalization, the high-quality benchmark itself, and a critical analysis of evaluation metrics, advocating for a shift towards semantics-aware evaluation. This is a valuable and timely contribution that lays a solid foundation for future research in this important yet underexplored area.

### Strengths
1.  **Excellent Problem Formulation:** The paper's primary strength is its clear and insightful formulation of scientific chart editing as a "structured transformation" problem governed by a graphical grammar. This conceptual shift from pixel-manipulation to structure-awareness is crucial and correctly identifies a fundamental mismatch in current approaches.

2.  **High-Quality, Comprehensive Benchmark:** The introduction of FigEdit is a significant contribution. The benchmark is large-scale, diverse (10 chart types, 5 task categories), and grounded in real-world data. Its design, which includes ground-truth specifications (Vega/Vega-Lite), enables precise and reproducible evaluation, setting a high standard for future work.

3.  **Critical Evaluation and Insightful Analysis:** The paper provides a thorough empirical study that reveals the shortcomings of existing models. More importantly, it demonstrates the inadequacy of traditional pixel-based metrics (SSIM, PSNR) for this task and convincingly argues for semantics-aware evaluation, notably through the novel use of an LLM-based score. This challenges the community to rethink evaluation standards.

### Weaknesses
1.  **Limited Coverage of Edit Operations:** The set of atomic edits, while canonical, appears somewhat limited. The paper focuses on operations like `add_datapoint`, `change_background_color`, and `increase_text_size` (Table 3, Appendix C). However, real-world chart editing often involves more complex structural changes, such as changing chart type (e.g., bar to line), reordering categories, grouping/ungrouping data, or modifying axis scales (e.g., linear to log). The current operation set may not fully represent the breadth of transformations required in practice.

2.  **Lack of Subjective and Holistic Edit Instructions:** The benchmark is constructed around objective, atomic instructions. In reality, user requests can be more subjective and holistic, such as "make the layout more compact," "use a more professional color palette," or "highlight the most important trend." While these edits are harder to formalize and evaluate, their absence limits the benchmark's applicability to more creative and user-centric editing scenarios. For a dataset-focused paper, addressing the challenge of collecting and evaluating such instructions, even if labor-intensive, would have significantly strengthened the work.

### Questions
**Scalability to Complex, Multi-Element Charts:** The examples presented in the paper (e.g., Figure 1, 3, 5) appear to involve relatively simple charts with a moderate number of data points and clear visual separation between elements. It is unclear how the proposed tasks and evaluation would scale to highly dense or complex visualizations, such as scatter plots with thousands of points, intricate network diagrams, or multi-panel figures where inter-panel consistency is critical.

### Soundness
3

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
3

### Summary
The paper **“CHARTS ARE NOT IMAGES: On the Challenges of Scientific Chart Editing”** introduces **FigEdit**, a large-scale benchmark (30K+ samples, 10 chart types) for **scientific figure editing**. Unlike natural images, charts represent **structured data governed by graphical grammar**, so valid edits must preserve **data–encoding alignment**, **axis coherence**, and **legend integrity**—not just pixel similarity. FigEdit defines **five tasks**: single edit, multi edit, conversational edit, visual-guidance edit, and style transfer. It includes paired figures and specifications to evaluate **semantic correctness** rather than pixel similarity. Experiments on models like GPT-Image, Imagen 4, OmniGen 2, and InstructPix2Pix show that **high SSIM/PSNR scores often mask incorrect edits**, highlighting the failure of pixel-based metrics. Overall, FigEdit reframes chart editing as a **structured transformation problem** and provides the first **semantics-aware benchmark** for developing models that understand both the **visual and data layers** of scientific charts.

### Strengths
- **Quality:** Builds a **large, well-controlled benchmark (30K+ charts)** generated via deterministic Vega rendering, with clear task taxonomy and reproducible evaluation.
- **Clarity:** The paper is well-organized and visually clear, using intuitive figures and radar plots to illustrate the **gap between pixel similarity and semantic correctness**
- **Significance:** Establishes the first **semantics-aware benchmark** for chart editing, providing a valuable testbed for evaluating multimodal and instruction-following models on structured, data-grounded visual tasks

### Weaknesses
- **Synthetic generation bias vs. real-chart curation.** FigEdit’s base figures and edits are produced via **LLM-guided Vega/Vega-Lite specs** and rendered images, which can drift from real publication practices; several prior sets rely on **human-curated real charts** and manual validation (e.g., ChartEdit’s 1,405 instructions on 233 real charts).
- **Subjectivity/noise in LLM-based scoring.** The paper’s “semantics-aware” evaluation relies on **LLM judgement** for instruction following/content preservation, potentially variable across prompts/seeds.
- **Baseline breadth.** The paper evaluates **four** editors; competing benchmarks often assess **more models** (ChartEdit: 10 MLLMs), providing a stronger landscape read.
- **Complex figure phenomena underrepresented.** From tables/figures, FigEdit does not clearly cover **multi-axes, error bars, subfigures, or dense annotations**, which are common in scientific graphics and present failure modes captured in broader reasoning/analysis sets like ChartX.

### Questions
- How realistic are the synthetic Vega-generated charts compared to real scientific figures? Could adding real-world data improve generalization?
- How reliable are the LLM-based evaluation scores—were they compared with human judgments or alternative metrics?
- Why not include code-level (Vega spec) comparison as an additional evaluation, since all charts are programmatically generated?
- Are there plans to expand beyond 10 chart types to include more complex scientific visualizations (e.g., heatmaps, multi-panel figures)?

### Soundness
3

### Presentation
3

### Contribution
3
