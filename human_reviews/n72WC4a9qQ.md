# CAAP: Context-Aware Action Planning Prompting to Solve Computer Tasks with Front-End UI Only

- Decision: Reject
- Scores: 5, 5, 5

## Abstract
Software robots have long been used in Robotic Process Automation (RPA) to automate mundane and repetitive computer tasks. With the advent of Large Language Models (LLMs) and their advanced reasoning capabilities, these agents are now able to handle more complex or previously unseen tasks. However, LLM-based automation techniques in recent literature frequently rely on HTML source code for input or application-specific API calls for actions, limiting their applicability to specific environments. We propose an LLM-based agent that mimics human behavior in solving computer tasks. It perceives its environment solely through screenshot images, which are then converted into text for an LLM to process. By leveraging the reasoning capability of the  LLM, we eliminate the need for large-scale human demonstration data typically required for model training. The agent only executes keyboard and mouse operations on Graphical User Interface (GUI), removing the need for pre-provided APIs to function. To further enhance the agent's performance in this setting, we propose a novel prompting strategy called Context-Aware Action Planning (CAAP) prompting, which enables the agent to thoroughly examine the task context from multiple perspectives. Our agent achieves an average success rate of 94.5% on MiniWoB++ and an average task score of 62.3 on WebShop, outperforming all previous studies of agents that rely solely on screen images. This method demonstrates potential for broader applications, particularly for tasks requiring coordination across multiple applications on desktops or smartphones, marking a significant advancement in the field of automation agents. Codes and models are accessible at https://github.com/caap-agent/caap-agent.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces a modular agent to solve computer tasks using screenshot images (that are converted into text), Large Language Models,  and a Graphical User Interface. UI elements are first detected and formatted into text using fine-tuned models. Then, a large language model decides appropriate actions using a specific prompting technique. Finally, the actions are executed using the keyboard and mouse interfaces. As such, the model doesn't rely on HTML or the Document Object Model as input sources. The proposed prompting technique, CAAP, structures the available context and introduces chain-of-thought reasoning. Experiments on MiniWoB++ and WebShop benchmarks show good performance with a limited number of expert demonstrations.

### Strengths
- The paper addresses an interesting problem of solving computer tasks using LLMs and visual inputs.
- The engineering effort of integrating different modules into a unified framework is well-executed and appreciated.
- The provided implementation and experimental setup details are a valuable part of the paper.

### Weaknesses
- It is not clear why the “UI element understanding” module needs to understand the spatial relationships between the element and its adjacent elements (line 173). It seems that only the element's position is extracted. This should be clarified in the paper.
- The proposed prompting technique is described only superficially. Examples and more detailed discussions about each part of the prompt should be included in the main paper.
- The proposed prompting technique seems to be quite tailored towards a specific application and tested for a single language model. Therefore, it comes off as a hack rather than a novel contribution. Overall, the paper has limited scientific contribution. I would suggest the authors to focus more on investigating the influence the proposed technique has on In-Context Learning overall and highlighting their contribution in this way.
- The paper would benefit from a discussion about using VLMs, rather than a separate vision module and an LLM. Could a VLM be used to complete tasks in full or to replace the vision module, eliminating the need for fine-tuning different vision models?
- The comparison to different methods evaluated on different numbers of tasks doesn’t hold a lot of scientific value. Maybe the metrics only for the tasks all methods were evaluated on could be reported?
- The success rate range in Figure 4 (x-axis) overemphasises the performance of the proposed method when compared to ablation variants. Adjusting the range of choosing a different format for the graph could fix this issue.

### Questions
- What other applications could CAAP be applied to?
- On line 343, it is stated that some tasks were excluded from the agent’s training. How are other tasks used for training? Aren’t these demonstrations just used at inference as part of the prompt?
- What is the advantage of using a semi-masked screenshot rather than a fully masked or cropped screenshot of the element?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposes a LLM-agent based framework for computer use that uses only screenshot images as input, converting them to text for LLM processing, operating through basic keyboard and mouse actions on GUI and reducing HTML code and specific APIs. The proposed automation solution CAAP achieves 94.5% success rate on MiniWoB++ benchmark and 62.3 average task score on WebShop as compared to image-only based agents.

### Strengths
1. Modular framework: focus on visual observation, decision making and action execution aspects to solve computer tasks. The core components make it suitable for future extensions, maintainability and scalability.
1. Extensive discussion on prompt: for in-context learning from few-shot examples, context info, CoT, extra guidelines. The impact of this prompting scheme is shown as ablation with existing work RCI.
1. Open-source code and models

### Weaknesses
1. Architectural Complexity and overhead: seems more complex than end-to-end solutions due to modular design. This also creates specific APIs requirement to connect different modules, fine-tuning for vision models for screenshot specific recognition. It is unclear on what are the implications of these on robustness of the proposed framework.
1. Limited real world evaluation: while the proposed approach is tested on miniwob++ and webshop, it'll be useful to demonstrate it on real world tasks and analyze failure cases.

### Questions
1. Is the proposed model specific to certain UI elements in miniwob++ and webshop benchmarks or are they generally applicable?
1. How does the model handle dynamic UI elements (pop-ups), screen resolution, and other visual features when they change position or appearance?
1.  What was the process for selecting the "single human demonstration" used in WebShop instructions? How representative is it? How does it relate to "300 samples collected through a fully automated process"?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces CAAP, which is a novel framework for automating complex computer tasks using LLMs by interacting solely through the front-end UI. Unlike previous approaches that rely on HTML source code or application-specific APIs, CAAP perceives its environment through screenshot images, which are then converted into text using a visual observer module (YOLO and Pix2Struct). This text is processed by an LLM with CAAP, which includes human demonstrations, surrounding context information, CoT-inducing instructions, and lastly some extra guidelines. The agent executes keyboard and mouse operations to interact with the GUI, eliminating the need for pre-defined APIs or large-scale human demonstration data. Experiments on the MiniWoB++ and WebShop benchmarks demonstrate that CAAP outperforms existing methods.

### Strengths
This method makes a strong push for an agent that interacts with computer tasks using only screenshot images and standard input devices (keyboard and mouse), which more closely mimics human interaction than methods that use HTML or APIs. The overall paper is fairly clear and there is a good suite of baseline experiments and decent ablations. The originality of investigating prompting structure for agentic interactions with standrad input devices is good and is an important direction for future research.

### Weaknesses
The main novelty seems to be the CAAP prompting because converting the screenshots to text is done with fine-tuned YOLO and Pix2Struct models. Although Figure 4 is good at ablating different components of CAAP, it would be beneficial to include more analysis of these components and what traits help the prompting. Since the novelty is predicated on this prompting strategy, I believe a more thorough answer to the question of the best-prompting strategy for agents interfacing with just screenshots is interesting and needed. It would also help if Figures 1 and 2 would be more clear using the captions. Also, it would be good to discuss the limitations of the method.

### Questions
What is the reason for converting screenshots to text for LLM and not using a VLM as there is other visual information that isn't captured?
What is the reason for the choice of MiniWoB++?
Why fine-tune the visual observer (YOLO + Pix2Struct) for this particular dataset when you could have auxiliary losses for VLM? 
Why have an intermediate representation and not have a more end-to-end approach? 
Why is having a VLM be an action planner preferable to other longer-horizon policy learning methods that use semantics as part of their input observation?

### Soundness
3

### Presentation
3

### Contribution
3
