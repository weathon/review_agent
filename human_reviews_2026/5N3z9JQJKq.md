# AutoFigure: Generating and Refining Publication-Ready Scientific Illustrations

- Decision: Accept (Poster)
- Scores: 6, 2, 2, 4, 4

## Abstract
High-quality scientific illustrations are crucial for effectively communicating complex scientific and technical concepts, yet their manual creation remains a well-recognized bottleneck in both academia and industry. We present FigureBench, the first large-scale benchmark for generating scientific illustrations from long-form scientific texts. It contains 3,300 high-quality scientific text–figure pairs, covering diverse text-to-illustration tasks from scientific papers, surveys, blogs, and textbooks. Moreover, we propose AutoFigure, an agentic framework that automatically generates high-quality scientific illustrations based on long-form scientific text. Specifically, before rendering the final result, AutoFigure engages in extensive thinking, recombination, and validation to produce a layout that is both structurally sound and aesthetically refined, outputting a scientific illustration that achieves both structural completeness and aesthetic appeal. Leveraging the high-quality data from FigureBench, we conduct extensive experiments to test the performance of AutoFigure against various baseline methods. The results demonstrate that Autofigure consistently surpasses all baseline methods, producing publication-ready scientific illustrations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces **AutoFigure**, a two-stage “reasoned rendering” pipeline that converts long scientific text into publication-ready illustrations, and presents **FigureBench** (3,300 text–figure pairs) with VLM- and human-based evaluations. AutoFigure decouples layout reasoning from aesthetic rendering and adds an OCR-based erase-and-correct step, outperforming baseline T2I, code-generation, and multi-agent methods. 

However, clarity and reproducibility are limited by broad gaps in reproducibility and transparency: key terms are undefined (e.g., IRR); InternVL-3.5–based statistics lack human-calibrated validation; generated-vs-reference statistical comparisons and full prompt/iteration transcripts are missing; analysis of the lower win rate on the Paper domain is limited; style controllability is unclear; compute/latency/cost reporting is absent; and the dataset release plan remains ambiguous.

### Strengths
1. The paper delivers high-quality illustrations.  
2. The methodology is clear and technically sound.  
3. The problem addressed is inspiring, timely, and highly needed by the research community.

### Weaknesses
1. **IRR is used but not clearly defined.**  
   - *line 90* Please expand IRR on first use (e.g., *Inter-Rater Reliability*) and specify the exact statistic.

2. **Dataset Analysis relies on InternVL-3.5 without validating its accuracy.**  
   - *line 208* You state that “all statistics are analyzed using InternVL-3.5,” but do not quantify its error on these measurements (text density, colors, components, shapes). If feasible, include a small human-audited benchmark to sanity-check these automated stats.

3. **Missing comparison of statistics between AutoFigure outputs and original figures.**  
   - Please add a side-by-side analysis (distributions and summary stats) of *generated vs. reference* figures for key metrics (text density, components, shapes, color count etc.), along with effect sizes and significance tests.

4. **No concrete prompt examples for each stage/iteration.**  
   - Include representative prompts for every stage and show how does the iteration help improve the prompts.
   - Sharing a minimal working example in the appendix would aid reproducibility.

5. **Paper category has notably lower win rates; insufficient error analysis.**  
   - Provide a qualitative and quantitative breakdown of why “Paper” is harder.

6. **Style uniformity across generated figures.**  
   - Many results share a similar visual/“avatar” style. Clarify whether this is a bias of the rendering LLM/T2I model, default style prompts, or intentional curation. Demonstrate **style controllability** (e.g., minimal changes in the style descriptor yielding materially different aesthetics) and include a diversity study (multiple styles for the same layout).

7. **Compute, latency, and token usage not reported.**  
   - Please report average **tokens/time per figure** for each stage, number of refinement iterations, hardware, parallelism, and **cost** estimates. A throughput vs. quality curve (iterations N vs. score) would help practitioners plan budgets.

8. **FigureBench availability unclear.**  
   - Clarify whether the dataset is fully prepared for release and provide an access link or timeline.

### Questions
See weaknesses

### Soundness
2

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce the task of generating scientific figures from long-form text (e.g., the entire text of a paper). To do this, they introduce AutoFigure, an end-to-end model (or possibly a pipeline of models) that generates figures from long-form text. To evaluate their model, they also introduce a new dataset of long-form text paired with corresponding scientific figures. Their approach outperforms various baselines. The dataset also has a development split.

### Strengths
* The authors tackle a relevant and challenging problem.
* The paper is mostly easy to follow.
* The created dataset, if released, may be useful for future work.

### Weaknesses
* It is not entirely clear to me what the authors understand by long-form text. It seems like they mean the entire text from a paper. But pairing a whole paper with just one figure extracted from it does not seem like a well-defined problem to me, as usually no figure captures the whole content of a paper.
* Datasets that pair figures with captions and all mentioning paragraphs can also arguably be classified as containing long-form text (ACL-Fig [1], SciCap+ [2], etc.), but these are not discussed in the paper even though they are very relevant.
* There is also a lot of other relevant previous work on scientific figure generation which the authors do not cite [3, 4, 5, 6]. Instead, they falsely claim that existing work on automated scientific visuals primarily explores the generation of artifacts like posters and slides (l.113).
* I do not understand how the AutoFigure model works internally. Figure 1 mentions InstructGPT but doesn't mention how that relates to this work in any way. Still, it makes it seem like AutoFigure consists of a single instruction-tuned model which they train themselves (even though the instruction-tuning examples inside the figure are unrelated to the task at hand). But nowhere in the paper is a training process described. Instead, Sections 4.1 and 4.2 make it seem more likely that AutoFigure consists of a model pipeline of prompted VLMs and text-to-image models. The authors should be clearer about how their model, a core contribution of this work, actually works.
* Furthermore, the fact that some of the baseline models mentioned in Figure 1 are actual model names while others are markup languages further contributes to confusion.
* Some claims in the paper like "Another line of work employs executable code as an intermediate state between scientific text and illustration, resulting in relatively visually unappealing diagrams" (l.71) and "This shift is evidenced by the growing acceptance of AI-generated papers at venues like the ICLR 2025 workshop and ACL 2025" (l.135) are not backed by any data or citations and make it unclear how the authors arrived at such conclusions. 
* The authors advertise the development split of their dataset but do not mention how it is used or how it can be used.
* In summary, I believe that the paper requires further justification (is the tackled task actually useful?) or adjusting the problem, and needs much more polished writing before it can be accepted.

[1]: [ACL-Fig: A Dataset for Scientific Figure Classification](https://ceur-ws.org/Vol-3656/paper2.pdf)  
[2]: [SciCap+: A Knowledge Augmented Dataset to Study the Challenges of Scientific Figure Captioning](https://ceur-ws.org/Vol-3656/paper13.pdf)  
[3]: [Text-Guided Synthesis of Scientific Vector Graphics with TikZ](https://openreview.net/pdf?id=v3K5TVP8kZ)  
[4]: [DeTikZify: Synthesizing Graphics Programs for Scientific Figures and Sketches with TikZ](https://proceedings.neurips.cc/paper_files/paper/2024/file/9a8d52eb05eb7b13f54b3d9eada667b7-Paper-Conference.pdf)  
[5]: [TikZero: Zero-Shot Text-Guided Graphics Program Synthesis](https://openaccess.thecvf.com/content/ICCV2025/papers/Belouadi_TikZero_Zero-Shot_Text-Guided_Graphics_Program_Synthesis_ICCV_2025_paper.pdf)  
[6]: [Learning to Infer Graphics Programs from Hand-Drawn Images](https://papers.nips.cc/paper_files/paper/2018/file/6788076842014c83cedadbe6b0ba0314-Paper.pdf)

### Questions
I have no questions for the authors.

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
The paper introduces FigureBench, the first large-scale benchmark for generating scientific illustrations from long-form scientific texts. FigureBench comprises 3,300 high-quality text–figure pairs, encompassing a wide range of text-to-illustration tasks. In addition, the authors propose AutoFigure, an agentic framework that automatically produces high-quality scientific illustrations from long-form scientific descriptions. AutoFigure operates in three stages: conceptual grounding, iterative self-refinement, and erase-and-correct rendering. Automatic evaluations based on the VLM-as-a-judge paradigm and human expert assessments demonstrate that AutoFigure is capable of generating illustrations that are scientifically rigorous and aesthetically appealing.

### Strengths
- This paper represents an important early step toward exploring how AI can assist humans in the time-consuming process of scientific illustration creation. The topic is interesting and promising, with significant potential for advancing AI-assisted scientific communication.

- AutoFigure is designed as a three-stage framework, where each stage addresses distinct challenges in the illustration generation process. These stages work in synergy, resembling how humans iteratively refine scientific figures — for instance, through the Critique-and-Refine step and the Erase-and-Correct strategy.

- The model demonstrates quantitatively superior performance in most experiments, indicating its capability in generating high-quality scientific illustrations.

### Weaknesses
- In Figure 1, AutoFigure generates a scientific illustration for InstructGPT. However, the original InstructGPT paper does not mention any examples related to relativity, suggesting that AutoFigure may have extended the content beyond the source text. Moreover, there is an error in the generated example (“ravity” instead of “gravity”), indicating that in some cases, the framework may pay more attention to aesthetic appeal than scientific accuracy.

- As mentioned in the paper, scientific illustrations are meant to help readers grasp the main ideas quickly and avoid misinterpretation. However, in Figure 4, the diagrams appear confusing — the corresponding subfigures are not well aligned (e.g., Step 2 in the Offline Phase), and the textual elements are cluttered and disorganized. In fact, an effective scientific illustration should focus on clarity and accuracy of information, rather than emphasizing visual decoration such as color or stylistic patterns.

- The paper mentions a process of fine-tuning a large language model (LLM) using human-selected text–figure pairs. However, the details of this process remain unclear — specifically, like which LLM was used and what the input–output format of the fine-tuning procedure was. Moreover, given that the dataset contains only 300 samples, it raises concerns about whether such fine-tuning could improve the quality of the paper selected.

- For the Style-Guided Aesthetic Rendering process, AutoFigure employs style description text to guide the rendering. However, as shown in Figure 4, the illustrations generated from diverse academic texts appear visually similar, often consisting of sub-block structures, similar color schemes and patterns. This raises concerns about whether the style cues truly exert a meaningful guiding influence on the rendering results.

- Some details of this work are not clearly illustrated:
(a) What are the prompts used for all the LLMs used in this work?
(b) The VLM evaluates the generated images across three dimensions with eight sub-metrics, but these metrics are not explicitly defined or described in the paper.

- Regarding the human evaluation, only 10 participants were involved, and they assessed merely 21 papers in total. Such a limited sample size may introduce considerable bias and undermine the reliability of the evaluation results.

### Questions
See the weaknesses

### Soundness
2

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
3

### Summary
The authors propose AutoFigure, a method that takes text as input such as and outputs a figure that reflects the entities and the relationships between them, much like a pipeline or a control flow that appears in papers to explain the procedure. 

The method first uses concept extraction to break down the textual input into a structured output which then gets refined using vlms via iterative feedback, which then passes through an aesthetic synthesis process to beautify it and render it.

The authors ran experiments that evaluate how good the generated figures are based on aesthetics, content, and fidelity. They used human evaluation to assess whether the synthetically generated figures are of high enough quality to be included in a camera-ready paper.

They also create a 3,300-sample FigureBench dataset from AutoFigure to evaluate thier system.

### Strengths
The paper addresses a very impactful task related to figure generation, which is important in many different industries and research domains.

The authors perform a good and comprehensive human and automatic evaluation with several strong baselines.

The paper is well-written and explains the problem in a clear and complete way.

The authors include a detailed human evaluation setup, which is a good way to test whether the AutoFigure generation is actually good. They recruited 10 human experts to evaluate AI-generated figures from their own first-author papers. The evaluation included three tasks:
(1) Multi-dimensional scoring, Forced-choice ranking against human-created figures, and  (3) Publication intent selection, asking if the generated figure is good enough to be included in a camera-ready paper.

### Weaknesses
No open-source models were tested (it only mentions Gemini, Grok, Claude, and GPT). It is important to include methods that work well with open-source LLMs so the research can be reproduced with minimal cost and used more broadly.

A lot of methodology details are missing. For example, when the paper says "identifying key entities and their relationships, and distilling a core methodology summary," it is not explained how this is actually done as no system prompt nor user prompt are shown.

Using an LLM as a judge is risky because it is not clear how they ensure it does not produce biased or inconsistent evaluations. The LLM itself needs to be tuned and evaluated. Does the results change when ran multiple times? There is a lot of bias risk even for the blind pairwise comparison part. There should be a human evaluation for all the metrics ("Visual Design," "Communication Effectiveness," "Content Fidelity," and "Blind Pairwise Comparison") to make sure they align with the VLM’s scores. A correlation or alignment study between human and model judgments would make this more reliable.

The method uses many LLMs in different steps, so the cost of running the system must be very high, especially since all the models are commercial. Functions like Φprompt and Φerase should be clearly stated as either LLM-based or non-LLM components. The authors should include a pareto plot showing the trade-off between cost and performance across different LLMs so we could know what to choose under a fixed budget.

There is no ablation study showing why the "erase-and-correct" strategy is necessary or how much it improves the results.

There are no code or prompt descriptions for the LLMs used. The paper mentions several Φ functions and agents, but it does not show what the actual prompts or inputs look like.

The method is long and has many components, but the exact implementation details are unclear. The paper mentions the steps, but it does not show how each stage (concept extraction, refinement, validation, rendering) is actually carried out.

There are not enough qualitative examples of generated figures in the paper. The examples shown are very few. It would help to share a link with a larger gallery of generated figures (say a few hundred) to confirm that the figures are not overly simplistic and truly handle dense, multi-component scientific texts.

The results have no error bars, so it is not possible to know whether the performance differences between models are statistically significant.


The blind pairwise comparison setup is limited because it only allows for "A, B, or Tie." It should also have options like "both good" or "both bad," so we can tell how many figures are actually poor and would be discarded.

Figure 3 is poorly structured and hard to match with the text in Section 4.  It would be better if each part of the figure was labeled with the corresponding stage headers (for exmple "Concept Extraction," "Critique and Refine") exactly as they appear in the text. Keeping the same wording between the figure and the text would make it much easier for readers to follow and it is a very good practice.

### Questions
Did you generate your figures in this paper using AutoFigure itself? If not, why not? This would be the most direct way to demonstrate the system’s real-life appliction.

Where are the quantitative and qualitative results that show the usefulness of the "erase-and-correct" strategy? Please include a clear comparison (with and without this step) so readers can see what it improves and by how much.

Where are the actual prompts used for each stage (concept extraction, critique and refine, rendering, validation)? Showing a few concrete examples would help readers understand how the LLMs were ran and what inputs they receive.

Where does this method fail? Please include examples or categories of failure cases (for instance, overly complex text, ambiguous relationships, or dense mathematical descriptions). .

How does this method perform when using open-source models like LLaMA or Mistral instead of commercial ones such as Gemini, Claude, or GPT? It would be important to know if AutoFigure can function well with open-source models.

Is the LLM used to generate the first layout (So, A0) the same as the one used in the subsequent refinement iterations? If not, please clarify which models are used at each stage and why.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the task of generating scientific diagrams directly from long scientific text. The idea is to transform complex scientific context (texts) into a figure that communicates the core concepts visually. 

The authors introduce a new benchmark called FigureBench, with 3.3K high quality text to scientific figure pairs. They also propose AutoFigure, an agent style system that first plans a symbolic layout and then renders it. 

Experiments show that AutoFigure outperforms previous methods on their benchmark, using both VLM as judge metrics and human expert evaluation.

### Strengths
1. The paper introduces a modern benchmark for scientific illustration generation that includes diverse long text to figure pairs from papers, surveys, blogs and textbooks. The authors also provide dataset statistics and analysis showing the challenge of long context reasoning.

2. The method uses an agent based pipeline that first grounds concepts with a VLM to produce a symbolic layout, then performs iterative refinement to improve structure, and finally renders the figure. This decoupled design is well motivated and appears to be effective.

3. The paper clearly states why FID is not well aligned to this task and instead uses VLM as judge evaluation combined with human expert assessment. This evaluation choice makes sense for this application.

4. The approach does not require training. It leverages frontier foundation models to achieve strong results.

### Weaknesses
1. The comparison to prior datasets is incomplete. Paper2Fig100k [1] dataset is not mentioned or cited, and Paper2Fig100k already contains more than 100k text to figure pairs. The claim that FigureBench is the first large scale benchmark is therefore not correct, and should be reframed more precisely.


2. There is no reference to recent TiKZ based diagram generation approaches such as Automatikz [2], which are directly relevant to the diagram synthesis space.


4. The design of the VLM as judge metric is not described in sufficient detail. Ideally the authors should provide a validation study that justifies the choice of prompts, models, and scoring dimensions, and should provide a correlation analysis between VLM scoring and human scoring.


5. The system pipeline is quite complex. While the ablations help, the number of components shown in Figure 3 suggests that more systematic ablations would be useful to better understand which parts contribute most.


6. The paper does not report efficiency metrics. It would be important to know typical generation time, how this scales with text length and concept complexity, and the approximate economic cost of running this agent per figure.


7. The experiments rely mainly on GPT Image and Gemini 2.5 Pro as backbones. To strengthen the experiments section it would be helpful to evaluate with more alternative LLM/VLM and image generation models.

8. The set of baselines is limited. There is no comparison to other agentic scientific content generation systems such as Paper2Poster or PPTAgent. A deeper analysis of which methods are closest in terms of workflow and whether they can be run on FigureBench would make the empirical comparison more convincing.

[1] Rodriguez, Juan A., et al. "Ocr-vqgan: Taming text-within-image generation." Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2023.

[2] Belouadi, Jonas, Anne Lauscher, and Steffen Eger. "Automatikz: Text-guided synthesis of scientific vector graphics with tikz." arXiv preprint arXiv:2310.00367 (2023).

### Questions
1. (Comment for improvement) In Figure 1, the visual framing could be confusing. At first glance it gives the impression that the authors are training the model with the InstructGPT pipeline, rather than generating a figure of the InstructGPT pipeline. The caption could clarify this more explicitly so that the intent is obvious to the reader.


2. The generated figures look significantly better than previous works, but they still do not look ready for professional use in a camera ready paper. What do the authors believe are the main remaining blockers to achieving near perfect results, and which components of the current system will need to be improved most to close this gap?

### Soundness
2

### Presentation
3

### Contribution
3
