# Multi-Step Reasoning for Embodied Question Answering via Tool Augmentation

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 2, 6, 4

## Abstract
Embodied Question Answering (EQA) requires agents to explore 3D environments to obtain observations and answer questions related to the scene. Existing methods leverage VLMs to directly explore the environment and answer questions without explicit thinking or planning, which limits their reasoning ability and results in excessive or inefficient exploration as well as ineffective responses. In this paper, we introduce **ToolEQA**, an agent that integrates external tools with multi-step reasoning, where external tools can provide more useful information for completing the task, helping the model derive better exploration directions in the next step of reasoning and thus obtaining additional effective information. This enables ToolEQA to generate more accurate responses with a shorter exploration distance. To enhance the model’s ability for tool-usage and multi-step reasoning, we further design a novel EQA data generation pipeline that automatically constructs large-scale EQA tasks with reasoning trajectories and corresponding answers. Based on the pipeline, we collect the EQA-RT dataset that contains about 18K tasks, divided into a training set EQA-RT-Train, and two test sets EQA-RT-Seen (scenes overlapping with the training set) and EQA-RT-Unseen (novel scenes). Experiments on EQA-RT-Seen and EQA-RT-Unseen show that ToolEQA improves the success rate by 9.2-20.2% over state-of-the-art baselines, while outperforming the zero-shot ToolEQA by 10% in success rate. In addition, ToolEQA also achieves state-of-the-art performance on the HM-EQA, OpenEQA, and EXPRESS-Bench datasets, demonstrating its generality.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes ToolEQA, a novel method for embodied question answer (EQA). The agent itself utilizes multistep reasoning and uses tools and code execution as the means to interaction with the virtual environment. In addition to that, the paper synthetically generated a EQA dataset (EQA-RT), by 1) asking GPT-4o to generate question-answer pairs, 2) generating optimal trajectory with A* and enriching each step with reasoning and tools, and 3) verifying each task. The ToolEQA demonstrates improved performance on EQA benchmarks and in particular finetuning clearly boosts performance.

### Strengths
- The paper proposes a simple yet effective method for EQA. 
- The data generation pipeline and the dataset is very useful. There is clear performance gain from finetuning. 
- The paper conducted comprehensive experiments.

### Weaknesses
- There could be more ablation of the method to better understands the contribution of each component.

### Questions
- How are the set of tools defined? Among the code blocks, are there any interesting action other than `print`?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper contributes a framework, ToolEQA that augments VLM-based EQA agents with tools (eg. bounding box detection, navigation primitives) for eliciting multi-step reasoning. The work further develops a data generation pipeline involving task creation, trajectory generation, reasoning trace generation and data verification, collecting a EQA-RT dataset. The results demonstrate that ToolEQA agents outperform prior works on the proposed EQA-RT benchmark and EXPRESS-Bench framework over most prior works, and that finetuning base VLM models on the collected dataset further improves performances.

### Strengths
1. Authors explore the relatively understudied extension of tool usage to the application of embodied question answering.
2. Usefulness of generated data: finetuning on the collected dataset improves tool usage and EQA success rates.
3. The paper's ToolEQA agent outperforms prior method (Fine-EQA) on Fine-EQA's benchmark (EXPRESSBench).

### Weaknesses
1. The paper has critical gaps in evaluation and analysis that do not fully establish the benefits of tool usage as well as improvements over prior works:

    a. The authors do not compare to a comparable baseline that follows the same pipeline but replaces the tools with VLM (eg. use the same VLM to compare sizes of objects in two frames). It is not clear if the gains are coming from breaking down reasoning into tools (multi-step reasoning) or the use of external tools. 

    b. The evaluation metric ($e_{\text{path}}$) combines recall, and hence it is not clear if ToolEQA, and its finetuning, result in efficiency improvements. Can the authors report the efficiency metric from OpenEQA?

    c. In tables 4 and 5, it is not clear if the ToolEQA used in evaluations is the finetuned version, leaving open the following questions:  Does finetuning on EQA-RT improve performance on other benchmarks? Are the performance improvements over prior works (eg.  Fine-EQA) coming from additional finetuning?
    
    d. GraphEQA, which also incorporates multi-step reasoning, appears to be better than HM-EQA on the only evaluation in Table 5. In the absence of additional comparisons and ablations, it is not clear if the components introduced in this work are essential.

    e. The method does not maintain and have access to a global scene representation (eg. scene graph, semantic map), and thus makes locally optimal decisions (eg. the doorway that looks promising without consideration of the expected scene layout). It is not clear how this choice affects the results and the comparisons to prior works (eg. Fine-EQA, GraphEQA) that maintain such representations.

    f. The work invokes a VLM in every step for low-level navigation commands ("turn left"), compared to prior works (eg. GraphEQA) that predict navigation targets offloading navigation to low-level planner. The effect of this choice on results is also unclear.

    g. While the work includes and additional test set with unseen scenes, it is not clear if the reported generalization (similar performance on seen and unseen scenes) is due to the unseen scenes having fewer target objects (Fig. 7c). It might be useful to break down the performance on the number of target objects to see if it has an effect on the performance of different splits (as hypothesized in the paper in L310).

2. Multi-step reasoning with tool usage has been explored to great extent in non-embodied settings (ViperGPT, VisProg, T3-Agent) and multi-step reasoning has been explored for embodied question answering (Fine-EQA). As a result the technical contributions of the work are limited, i.e., applying T3-Agent to embodied QA.
3. Details of critical components of the method (eg. planner, tool implementations, prompts used) are missing in the main paper and the supplementary. These are described under "Questions".

### Questions
1. The descriptions of tools are missing. For example, it is not clear how VisualQA or SegmentInstance are implemented and used.
2. It would be useful to include the prompts used for data generation and the VLM-based controller.
3. Have the authors considered using the Exploration-Answer Consistency (EAC) metric from FineEQA? Why do authors not use a consistent set of metrics cross all benchmarks?
4. L175: Details of the planner are missing. What structure do the generated plans follow ($p$ in L203)? The planner also seems to be absent in the presented example.
5. Can authors include more details and examples for the different categories of tasks (location-location, location-special)? 
6. It would be useful to include prompts for the controller and the data generation pipeline.
7. For EQA task verification, why is it important to use both object detector and LLM? How do these compare?
8. How efficient is the data generation procedure? What fraction of trajectories get filtered out for each of the different reasons (L283-L286)?
9. L320: It is not clear how finetuning to predict the final answer ($c_t = \texttt{FinalAnswer}$) would hurt the model? Can the authors elaborate more on how this encourages the model to use its biases by ignoring the context?
10. The length of trajectories is lower than the trajectories -- an average of 12.69 (L321). Based on the distances in Table 6, does this mean the agent moves by more than 1m in each step? How does this not prevent from taking finer steps for moving closer to objects or navigating cluttered scenes?
11. Some typos and questions in writing:
- L119: ~REALTED~ RELATED
- L131: ~the lack of~ they lack
- L156: ~finetunes~ finetuned
- L183: ~executing~ executes
- L184: By invoking the executor to: Typo or incomplete
- L171: The agent should not have access to an additional scene representation $S$ beyond the observations $o_i$?
- L361: ~we~ We

### Soundness
2

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
This paper focuses on the embodied QA which requires a robot to navigate a home environment to find answer for the provided question. This paper focuses on an agentic setup to achieve this. This agentic setup is also the main contribution of the paper. Overall, the agent (robot) has a set of tools at it’s disposal. These tools include GoNextzPoint, ObjectLocation2D, etc. — some of these tools are used for navigation planning, while others are used for visual information extraction. Prior approaches often end up training models directly to solve the task end-to-end (often with VLMs). By contrast, the current paper aims to use an agentic setup to learn how to do the task using the provided tools. To achieve this, the paper focuses on creating a dataset with the appropriate tool calls. For this, the paper curates the appropriate questions given a scene, from these questions important objects are extracted and paths to reach close to these objects are planned using A*. Once this trajectory has been created, GPT-4O is used to annotate each step path with a reasoning/thought trace ($t_i$) and a code output $o_i$. This data is then used to FT a VLM (Qwen-7B). Experiments show that the finetuned models perform better than zero-shot models with access to the agent API.

### Strengths
The paper focuses on an important problem of using VLM agents to solve embodied QA. Overall, the paper is well written. The main agentic contribution also is novel and seems to provide some potential benefit over baselines.

### Weaknesses
Questions

4.3 details: I think the most important part of the paper is the dataset curation strategy. While there are sufficient details around how the questions and paths are generated, there is insufficient details around how the reasoning / thought traces were curated. For example, reasoning trace is also generated when the model/agent should output “Move Left” — how does the thought trace look for this kind of action? When generating the GT data do you provide all the history (with all the observations) to the model? The prompts used for GT data curation should be in the appendix. 

Also, when we use the trajectory from A* we do have the GT for some actions (e.g. moving actions such as GoNextPoint), but how do you generate GT for the model that it should use other tools (e.g. ObjectCrop), why should GPT-4o use this tool? It is quite unclear how the model is choosing the right tool. This is an extremely important detail which is completely missing from the paper. How do you ensure that these tool calls are optimal? Also, some of these tool calls e.g. ObjectCrop can conflict with going closer to the target object (and zooming into the right object) so there is some weird design choice here. 

*Ablation on training on thoughts*: Currently, the paper finetunes the model on the code output as well as the intermediate thoughts. What happens if we only finetune on the code (which is I assume the desired output) instead of the thought and maybe do some small RL to let the model curate it’s own thought. It would be interesting to see how the model performs then.

How do the tool calls look like for the zero-shot model (GPT-40 and QWEN-2.5 VL)

*Results:* There is very little performance difference (using e_path @5) metric between GPT-4o and the FT-VLM for MCQ questions. Are most of the GPT failures due to navigation actions being incorrect and the agent not exploring enough or due to some other scenarios. I am wondering if the agent has an access to a navigation tool which can plan for longer and the agent doesn’t have to output every point to go to next, how would the performance change.

### Questions
see above

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
This paper proposes ToolEQA, a framework for Embodied Question Answering (EQA) where an agent uses external tools to navigate and gather information within 3D environments. By breaking down tasks into explicit reasoning steps and generating code to call these tools, the agent achieves more efficient exploration. To train their model, the authors created EQA-RT, a large-scale dataset with automatically generated reasoning trajectories. Experiments show ToolEQA achieves strong performance, particularly in reducing the exploration path length required to answer questions.

### Strengths
1.The ToolEQA framework uses explicit reasoning to guide its actions, resulting in shorter navigation paths and more efficient task completion compared to previous methods.
2.The paper contributes EQA-RT, a large-scale dataset with detailed reasoning trajectories. It ingeniously integrates 3D object detection, GPT-4o for question generation, and A* for optimal path planning, all validated by a multi-level verifier, to produce high-quality training data. The automated pipeline used to generate this data is a valuable resource for the community, enabling the creation of complex, structured EQA tasks.

### Weaknesses
1.The core ToolEQA framework is conceptually indistinct from established tool-augmented LLM paradigms like ReAct and Toolformer. Its "thought-code-observation" loop is a direct application of this existing work, making the contribution feel more like a conceptual repackaging for the EQA domain rather than a fundamental innovation. The authors should more clearly articulate what unique, non-trivial architectural adaptations were made to the framework specifically for the challenges of embodiment, beyond simply creating a new set of tools.
2.The paper's most significant and original contribution appears to be the EQA-RT dataset and its automated generation pipeline. While this is an impressive engineering system, it does not in itself constitute a novel methodology. The work's core value lies in the resource it provides to the community, which raises questions about its fit as a methods paper, as the novelty in the learning algorithm or agent architecture is minimal.

### Questions
1.The ToolEQA framework closely mirrors existing tool-augmented LLM paradigms. Could you elaborate on the unique challenges that embodiment introduces and how ToolEQA's architecture was specifically adapted to address these? For instance, how does the system handle or recover from incorrect predictions from its perception tools? What is the core innovation beyond applying a known method to a new domain?
2.A significant portion of the paper's contribution is the EQA-RT dataset, which is generated by a pipeline that naturally creates tasks solvable by a tool-based agent. How can we be confident that the agent is learning a generalizable reasoning skill rather than simply overfitting to the specific patterns of the EQA-RT generation process? Have you considered evaluating its performance on tasks that are explicitly designed to challenge the limits of its predefined toolset?
3.The current work relies on a small, fixed set of tools. How do you envision the framework scaling to more open-ended real-world scenarios? How does the agent react when faced with a task where its tools are insufficient? Can the fine-tuned controller generalize to use new tools effectively without requiring SFT? Does the framework support the autonomous discovery or learning of new tool functionalities?
4.Regarding the reasoning process itself, is a higher frequency of tool calls always beneficial? Have you analyzed the potential for redundant tool calls, where the agent re-requests information it already possesses or could have inferred? Could you provide an analysis of the correlation between the number of tool invocations per task and the final success rate? This could help clarify whether the agent's reasoning is efficient or simply exhaustive.

### Soundness
3

### Presentation
3

### Contribution
2
