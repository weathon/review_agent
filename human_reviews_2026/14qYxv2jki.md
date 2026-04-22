# Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 2

## Abstract
While reinforcement learning (RL) has proven highly effective for general reasoning in vision-language models, its application to tasks requiring deep understanding of information-rich images and structured output generation remains underexplored. Chart-to-code generation exemplifies this challenge, demanding complex reasoning over visual charts to produce structured code. Supervised fine-tuning (SFT) alone is often insufficient, highlighting the need for effective RL strategies tailored to structured outputs. In this paper, we systematically investigate the performance plateau of SFT through large-scale experiments and propose Multimodal Structured Reinforcement Learning (MSRL) for chart-to-code generation. We construct the largest training corpus to date, with 3 million chart-code pairs curated from real-world tables in arXiv papers, addressing the limitations of previous synthetic datasets. Despite achieving state-of-the-art performance, our experiments show that simply increasing SFT data eventually leads to diminishing improvements. To break this plateau, MSRL employs a multi-granularity reward system that integrates both textual and visual feedback. At the textual level, rule-based rewards validate fine-grained code details, while at the visual level, a model-based reward assesses the structural similarity between rendered code and ground-truth charts. We implement a two-stage curriculum training strategy, first optimizing the model with textual rewards and then incorporating visual signals for further enhancement. Experimental results demonstrate that MSRL substantially breaks the SFT plateau, improving high-level metrics by 6.2% and 9.9% on ChartMimic and ReachQA benchmarks, respectively. Notably, our method outperforms all existing approaches in the chart domain and achieves competitive results with advanced closed-source models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper focuses on chart-to-code generation. The authors built a large chart-code corpus and found that SFT hits a performance plateau when training data is scaled, with more data bringing little improvement. To address this, the authors used MSRL to fine-tune MLLMs and design task-specific rewards. The MSRL method achieved better performance than SFT. Moreover, the trained MLLMs outperform most baselines.

### Strengths
1. The methods of data synthesis and reinforcement learning are intuitive and interesting, providing a certain basis for solving the problems in chart interpretation tasks.​
2. In the chart-to-code task, the model's performance has obvious advantages compared with open-source models, demonstrating the effectiveness of some designs in the paper.​
3. Sufficient experiments are conducted, including comparative experiments, ablation studies, and qualitative experiments, which help to verify the related hypotheses and results.​
4. The preliminary study on the performance of SFT has certain reference value within the scope of this task, and it is natural to introduce RL based on this experiment, which ensures the logical continuity of the research.

### Weaknesses
1. On line 52, the description of the significance of the chart-to-code task is not convincing enough. The authors state that this task is "complex" but fail to highlight its strong "significance". Although it is based on previous research, it may arouse readers' questions about its application value (e.g., help MLLMs better understand charts or assist researchers). In addition, the connection between this part and the previous paragraph is not smooth.​
2. In the preliminary study (Figure 1), the SFT plateau and RL performance gain are shown, but the study on the change of RL performance with the increase of data size is missing. Since SFT will converge when the training data increases to a certain extent, according to the authors' motivation, RL should converge later, but this comparison is lacking. Although I noticed Figures 3 and 4, they only show the change curves of reward. The scale law of specific training data on performance is important and can emphasize the motivation of this paper.​
3. For methods, the main contributions of this paper lie in data synthesis and reward design. The GRPO used is in a standard form (lines 259-269). This not only reduces the innovation of the paper in terms of contributions but also makes the excessive space spent on introducing non-original content unnecessary in writing.​
4. The experimental results support the point 3 that the contribution of synthetic data is significant, which should be the core of this paper. However, the authors do not produce detailed discussion and analysis on data synthesis, usage, scaling law, etc.​
5. In Table 4, the effects of Textual and Textual+Visual are close, but there is no in-depth discussion on this in the paper, which weakens the rationality and effectiveness of the method design.​
6. Table 5 has a similar problem. The results of each variant are close (~1%), but the paper only lists the results without in-depth analysis.​

In general, **the research motivation of this paper needs to be strengthened, the experiments supporting the motivation need to be more completely supplemented, the focus of the designed method is biased, and although the experiments are sufficient, the support and analysis for the method design and problem solving are insufficient.**

### Questions
Tthere are some questions and minor issues do not affect my overall assessment:​
1. For Figure C, is there any reference for the selection of these scores?​
2. Regarding citations, the authors have cited a large number of arXiv papers. In fact, many of these papers have been published in conferences, so attention should be paid to the correctness of citations.

### Soundness
3

### Presentation
2

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
This paper targets the task of chart-to-code generation, where the model produce excuteable plotting code given a chart image.

The authors observe a performance plateau with supervised fine-tuning (SFT), To overcome this, they propose Multimodal Structured RL (MSRL), a GRPO based reinforcement learning approach with multi-granularity rewards. Training follows a two-stage curriculum: RL on textual feedback, then on combined textual+visual feedback. 

The paper claims state-of-the-art results on the ChartMimic and ReachQA benchmarks, with a ~6–10% gain in high-level accuracy over SFT alone.

### Strengths
* Clear diagnosis of the SFT plateau. The scaling curve explicitly plateaus after ~2M SFT examples, and shows visual RL breaks the curve.
* Large curated and balanced chart‑code corpus from real‑world arXiv is great contribution to the community.
* MSRL (7B) outperforms ChartCoder and is competitive with GPT‑4o on component scores such as layout and text (Tables 1–2, p. 6), with qualitative cases showing better fidelity and execution reliability than open and proprietary models

### Weaknesses
* The visual reward (and RL data filtering) rely on a single MLLM judge (Qwen2.5‑VL‑72B). Without cross‑judge verification or human studies, there is a very risk of reward hacking or bias toward that evaluator’s preferences.
* Possible double‑counting of execution success in stage 2, in Sec 3.3 R = w_t*R_text + w_v*R_vis + W_e*R_exec, R_text also contains R_exec
* Compute-normalized ablation for two-stage vs single-stage: It looks possible that the two-stage curriculum outperforms single-stage primarily because it uses a larger update budget (trained on more steps).
* Several relevant papers are not cited :
    - [1] **Rendering-Aware Reinforcement Learning for Vector Graphics Generation**, Neurips 2025.
    - [2] **MatPlotAgent: Method and Evaluation for LLM-Based Agentic Scientific Data Visualization**, ACL Findings 2024.
    - [3] **A Survey on LLM-as-a-Judge, arXiv:2411.15594**

### Questions
* Did you try cross-judge or human judge is done to visual rewards, (e.g., swapping the 72B judge with a different model or use gpt-4o or human to cross-judge qwen2.5-vl-72B, especially at late stage RL, where reward hacking is most possible?
* What's the reason for RL with visual rewards for only 100 steps where the rewards are still gaining fast, is the limit stability, compute, or hacking?
* [weakness #3] Could you disentangle curriculum effects from training budget by reporting a compute-matched comparison?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies why SFT saturates on chart-to-code and proposes a two stage RL recipe with multi-granularity rewards (textual + visual). The authors build a large chart-code corpus to show SFT plateaus beyond 2M examples, and then use GRPO with (a) execution + rule-based textual checks and (b) a render and compare visual reward scored by a strong MLLM. On ChartMimic and ReachQA, MSRL (7B) establishes new SOTA among open-source models and approaches GPT-4o.

### Strengths
1. Clear empirical story about the SFT ceiling. authors isolate SFT scaling curve and convincingly shows a plateau beyond 2M samples before introducing RL, this sort of strengthens the causal claim that RL brings the next jump, not just under-tuned SFT.
2. The textual reward normalizes code and scores specific fields (data, type, layout, titles/labels, exec) while visual reward compares rendered images against ground truth via an MLLM.. the two-stage schedule is sensible imo and empirically validated.
3. Strong results: MSRL-7B beats open-source baselines and seems competitive with GPT-4o on both datasets.. detailed low-level metrics suggest real gains, not just execution hacks.
4. Repruducible details specified

### Weaknesses
- Clearly the visual reward depends on Qwen2.5-VL as judge. If the policy aligns to judge's biases or defects, improvements could reflect evaluator gaming rather than genuine fidelity. Perhaps a judge-swap test (e.g., different MLLM/human) is needed to rule out any judge overfitting.
- The entire pipeline centers on Matplotlib-style code and a fixed rendering toolchain. It is unclear whether the learned behaviors transfer to other plotting libraries (Seaborn/plotly or vega) or different runtimes
- Real-world tables are scraped from arxiv, but the paper doesn’t quantify overlap with benchmarks. If time allows, data contamination studies are a must these days.
- Also, some light on whether the gains are from the reward design vs. the specific optimizer (grpo vs dpo) would be great.
- [Optional but important] The abstract is quite long, and the third paragraph in the introduction (starting around line 75) is similarly lengthy. Try to simplify where possible. Avoid using multiple wrap figures unless absolutely necessary.. if you can, place Table 3 and Table 4 side by side. Also the figure and table captions are too brief and not self-contained. Figure 5 is impossible to read at first glance. It would be good to do a dedicated pass focused on improving writing clarity and presentation quality.

### Questions
- How do results change when visual reward/evaluator is swapped (e.g., InternVL (latest), Gemini, GPT-4.1) and on a small human-scored subset?
- What exact de-duplication and topical overlap checks were done to ensure arXiv-derived tables don’t appear in or unduly resemble test items in ChartMimic/ReachQA
- curious to see if MSRL trained on Matplotlib code generalizes to other plotting libraries? Perhaps a notable few?

- Why GRPO? Is it the first thing you tried and worked decently?


Lastly, some relevant key papers on visual code generation that may have been missed in citations:
[1] Rodrigez et al, Generating Scalable Vector Graphics Code From Images And Text. https://arxiv.org/abs/2312.11556
[2] Xia et al, StructChart: Perception, Structuring, Reasoning for Visual Chart Understanding.
[3] Rodriguez et al, BigDocs: An Open Dataset for Training Multimodal Models on Document and Code Tasks. https://arxiv.org/abs/2412.04626
[4] Awal et al. WebMMU: A Benchmark for Multimodal Multilingual Website Understanding and Code Generation https://arxiv.org/abs/2508.16763

Happy to relook at the scores if authors consider some of the points raised.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduced Multimodal Structured Reinforcement Learning (MSRL) for the chart-to-code task. The authors first analyze the scaling of the supervised finetuning (SFT) approach and show that the performance plateaus beyond certain data/compute scale. After that, they propose an RL-based solution that breaks this plateau and shows further improvements in performance. The authors propose two sets of rewards: textual and visual. The textual reward focuses on aspects in the generated code while the visual rewards measures the similarity between the reference and rendered chart (from the generated code). To create data for SFT and RL, the authors propose a data generation pipeline that first sources tables from arXiv papers. Then, an LLM uses these tables along with code examples to generate code for new chart images. After that, the codes are rendered into chart images which are also further filtered using GPT4o to keep only high fidelity charts.  The authors evaluate their model on two benchmarks: ChartMimic and ReachQA, and show nice performance improvements over existing methods.

### Strengths
* The proposed RL approach shows performance improvements on the chart-to-code task that can’t be achieved by scaling the SFT data/training on its own. These experiments are quite interesting and could have valuable insights. 


* The proposed model achieves the state-of-the-art results on two chart-to-code benchmarks: ChartMimic and ReachQA.  The authors have provided detailed ablation experiments to show the benefits from each of their proposed techniques/reward functions.

### Weaknesses
* Limited Visual diversity: using a set of example codes and forcing the code to follow a specific structure may significantly limit the visual diversity of the dataset. There’s also no analysis to support the claim of visual diversity compared to existing datasets/approaches. 


* The evaluation is only limited to two benchmarks: ChartMimic and ReachQA. Furthermore, it’s limited to the niche chart-to-code task. It would strengthen the contribution of the paper if the RL approach can be expanded to other popular chart tasks (e.g., QA) to truly verify the “Breaking the SF Plateau” claim in the title and the paper. 


* The visual evaluator in the RL approach is quite massive, 72B params. I am not sure why the authors didn’t simply use a small CNN/ViT and just measure the visual similarity between the extracted features. 


* In the Intro, the authors claim that DPO does not generalize well but they haven’t conducted any experiment to prove that GRPO generalizes better than DPO in this task using the same dataset.


* Potential Data Leakage: The ChartMimic benchmark was constructed by scaping data from arXiv and the proposed dataset here also starts by scraping Tables from arXiv. I am concerned that the performance improvements could be tied to data leakage.

### Questions
see weaknesses above

### Soundness
2

### Presentation
2

### Contribution
2
