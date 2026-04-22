# TaskCraft: Automated Generation of Agentic Tasks

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Agentic tasks, which require multistep problem solving with tool use and adaptive reasoning, are becoming increasingly central to the advancement of NLP and AI. Although benchmarks such as GAIA and BrowseComp have advanced agent evaluation, their scalability remains limited by the high cost of human annotation. We introduce TaskCraft, the first automated workflow for generating scalable, multitool, and verifiable agentic tasks of difficulty. TaskCraft progressively complexifies atomic tasks through depth-based and width-based extensions, with incremental validation via rejection sampling and LLM-based linguistic analysis, ensuring both scalability and efficiency. The generated tasks enable trajectory sampling within state-of-the-art workflows, supporting end-to-end SFT and RL training. Experimental results on multiple LLMs show that TaskCraft data substantially improves multi-hop reasoning and agentic capabilities. Further scaling with TaskCraft tasks and applying RL training yields additional gains, achieving state-of-the-art performance on four agentic benchmarks. The resulting dataset comprises 41k tool-intensive tasks across varied difficulty levels, including 12.6k tool-interaction trajectories and 5k multihop decompositions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces TaskCraft, an automated framework for generating scalable, verifiable agentic tasks that require multi-step reasoning and tool use. The method begins with atomic tasks solvable by single tool calls and expands them via depth-based (sequential, multi-hop) and width-based (parallel, multi-subtask) extensions. Each generated task is incrementally validated through rejection sampling, LLM-based linguistic checks, and selective agent verification. The workflow ensures that generated tasks require genuine tool interaction rather than being trivially answerable by LLMs alone.
TaskCraft yields a dataset of 41k tool-intensive tasks, including 12.6k tool-interaction trajectories and 5k multi-hop decompositions. When used for SFT and RL, TaskCraft data significantly improves model performance across agentic benchmarks (GAIA, WebWalker, BrowserComp, HLE), achieving state-of-the-art results even with modest model sizes.

### Strengths
1. The framework is innovative: atomic task generation + validation and depth/width extension
2. The contributions are well motivated by addressing the data bottleneck for agentic reasoning tasks
3. Demonstrates performance gains on multiple benchmarks and shows that generated data can be useful in SFT and RL

### Weaknesses
1. The paper could benefit from including more information on SFT and RL experiments, such as hyperparameters used, etc 
2. Would be good to include more details on tools used for task generation, specifically the PDF tool. Since PDF parsing is often tricky and noisy, more information on the implementation used in this work would be insightful
3. The paper should clarify what exactly are the tools used in the framework and when. Most figures mention web search, PDF, and image tools. Figure 2 mentions a music tool and suggests there are more with "...", but these are not mentioned anywhere else in the paper.

### Questions
1. The depth/width extensions are a core contribution of the framework. It would be interesting to see ablations on performance vs number of hops in the finetuning data, though I understand that this is a bit out of scope for the purpose of this study
2. I appreciate that the authors included the code and data files in the materials. Will these be released to the public upon acceptance? 
3. After task extension, do you also check if any / how many of the extended tasks can already be solved by the task agent within a limited number of tool use steps (i.e., the setup used for atomic task verification)? The current framework checks for soundness, leakage, difficulty of atomic task, etc, but I do not see any quantification of how task extension further increases difficulty of tasks and whether this difficulty increase can be controlled / customized

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Existing agentic benchmarks are constrained by the high cost of human annotation. TaskCraft addresses this challenge through automated task generation and extension, coupled with efficient verification mechanisms, to produce a scalable, multitool, and verifiable agentic task dataset. By applying SFT and RL on TaskCraft-generated data, the models demonstrate substantial improvements in multi-hop reasoning and agentic capabilities, showcasing the practical utility and potential of the TaskCraft framework.

### Strengths
1. TaskCraft's automated workflow supports adaptive difficulty progression through depth-based and width-based extensions, eliminating the annotation bottleneck that limits existing benchmarks.

2. The generated tasks span varied difficulty levels across multiple tool modalities (web, PDF, image), including complex multi-hop reasoning tasks. This difficulty stratification mirrors human-curated benchmarks like GAIA while maintaining scalability.

3. The paper provides thorough experimental analysis across multiple dimensions.

4. The work ensures reproducibility through detailed methodological documentation, including formalized task generation procedures, verification prompts (Appendix E), training configurations (Appendix F), and commitment to release all artifacts.

### Weaknesses
Please refer to questions section.

### Questions
1. The number of generated tasks rapidly decays as the number of hops increases. What are the underlying reasons for this decay, and can you provide examples of longer-hop tasks to illustrate the generated complexity?

2. The paper lacks an analysis of tasks that failed during generation or verification. What were the primary reasons for these failures? Providing such an analysis would be beneficial for future research by highlighting common pitfalls and suggesting directions for improvement.

3. Both the task extension and verification processes heavily rely on LLM. Could this dependence lead to data bias, and would using LLMs with different styles result in distinct data distributions? This could impact the generalizability of agents trained on TaskCraft data.

4. The scalability of TaskCraft data is only demonstrated up to 5k tasks in the current experiments. What would be the effect of further increasing the data size, and could a performance curve be plotted to observe whether there is a saturation trend? Investigating this would help prioritize between developing more rigorous verification methods and expanding the dataset size for future enhancements.

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
3

### Summary
The paper introduces TaskCraft, an automated pipeline to generate multi-tool agentic tasks that require multi-step reasoning and tool use.

The method operates in three stages:

1. Atomic task generation: Starting from unlabeled corpora (webpages, PDFs, images), the system extracts tool input indices and derives tasks through a structured formulation q = f(i_T, R) → a
2. Task extension: Complexity is increased through depth-based extensions (sequential multi-hop reasoning) and width-based extensions (parallel sub-task decomposition)
3. Verification: Tasks are validated through rejection sampling and LLM-based linguistic analysis, with incremental validation during extensions

The resulting dataset contains 41k tool-intensive tasks across varied difficulty levels, including 12.6k tool-interaction trajectories and 5k multi-hop decompositions. Experimental results demonstrate that models trained on TaskCraft data substantially improve performance on four agentic benchmarks (GAIA, WebWalker, BrowserComp, HLE), achieving state-of-the-art results. Notably, on GAIA, adding 2.5k TaskCraft tasks to 5k existing MHQA data improves performance from 38.8% to 60.2% (+21.4 points).

### Strengths
1. The proposed task generation pipeline is a simple idea that leverages LLMs to synthesize high quality tool-use tasks
2. The tasks and task execution traces generated through TaskCraft leads to training effective tool-use agents
3. Proposed method is easy to scale to large number of tasks by gathering large amounts of unlabeled corpus of webpages, PDFs, and images.

### Weaknesses
1. The method section is a bit hard to follow. It’d be good if authors can take another pass at writing and improve the flow of the content to make the method easier to understand.
2. There is a slight inconsistency in the types of datasets used (or at least in table descriptions for entries) for different model sizes for the experiments presented in table 3. For example, Qwen2.5-7B and DeepSeek R1 distill models have results for training on 7.5 MHQA tasks vs Qwen2.5-32B doesn’t mention the size MHQA dataset used. It would be great if authors could clarify or make the datasets used for these experiments consistent. In addition, I would also like to see results of just training these models on the 5k task subset from MHQA that was used in conjunction with TaskCraft 2.5k and 6k/8k tasks. This result should establish a concrete baseline that will clearly show what is the delta/improvement coming from task craft tasks. Currently, it is unclear whether improvement is coming from excluding 2.5k MHQA tasks or addition of 2.5k TaskCraft tasks. In addition, for RL finetuned 7B and 32B models I would suggest authors use a fixed number of tasks across model sizes (i.e. either 6k or 8k) to make the comparisons fair and make experiment setup consistent. It is unclear whether the 32B model is improving more from additional 2k tasks vs the model size.
3. It is a bit unclear how the pass rate in tables 2 and 4 defined. Can authors clarify if the pass rate reported is the task execution + verification pass rate or is it another metric that is only measuring whether task generation is feasible?
4. The paper does not include details on the initial corpus used to build the input index i_T and lacks details on how tools are constructed, what tools are used. Because of these missing details it is bit hard to reason about where the improvements in harder benchmarks like HLE and BrowserComp are coming from. It would be great if authors can add a section describing these details and also some preliminary analysis on whether the task craft generated tasks overlap with tasks from these benchmarks to check if there is any test set contamination.
5. Task verification/extension section needs more work on writing. With the current description it is a bit unclear to me how the authors are using linguistic analysis to verify whether a proposed task extension is valid or not. More details are required in the main paper with appropriate references to the detailed prompts and method in appendix.

### Questions
Mentioned in Weaknesses section.

I believe the paper has good contributions and strong results. There are couple of issues with writing and some experiments. If authors address those concerns I am happy to increase my rating.

### Soundness
2

### Presentation
2

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
Existing agentic task benchmarks such as GAIA, BrowseComp, and HLE are difficult to scale due to their reliance on costly human annotations, while previous self-instruct-style data generation methods primarily target static instructions rather than interactive, tool-using, multi-step tasks. This paper introduces TaskCraft, the first automated workflow for generating scalable, multi-tool, and verifiable agentic tasks with varying levels of difficulty. TaskCraft progressively increases task complexity through depth-based and width-based extensions, while employing incremental validation via rejection sampling and LLM-based linguistic analysis to ensure both scalability and reliability. Experimental results across multiple LLMs demonstrate that TaskCraft-generated data significantly enhances multi-hop reasoning and agentic capabilities.

### Strengths
1. Originality: Proposes innovative depth-based and width-based methods for data generation.

2. Significance: Effectively addresses the scalability challenge of agentic data, a key bottleneck in training and evaluating tool-using LLM agents.

### Weaknesses
1. The paper’s method description is unclear. For example, the function $f()$ is used inconsistently across different contexts (e.g., lines 190, 196, 863, and 871).

2. Why not directly use the 7.5k TaskCraft data for SFT training? Could you include an additional experiment comparing its performance with the 7.5k MHQA dataset?

### Questions
1. Are both the depth-based and width-based extensions generated using a few-shot approach? Does “prompt learning” (line 293) refer to this few-shot generation?
2. The notation ( $q = f(i_T, R) \rightarrow a$ ) is unclear.
3. What prompts are used to extract ( $i_T$ ) and ( $C$ )?
4. How do you ensure that the obtained ( $i_T^{n}$ ) forms supersets without causing cyclic generation? Is there an ablation study verifying the reliability of the LLM judge?
5. How does the width-based extension propose new questions? Does it generate targeted, potentially mergeable questions, or are the questions generated randomly and then filtered through verification to retain only those that can be merged?
6. Is your training and evaluation framework based on open-source code?

### Soundness
3

### Presentation
2

### Contribution
3
