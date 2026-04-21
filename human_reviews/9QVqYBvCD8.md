# Asking Before Acting: Gather Information in Embodied Decision-Making with Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 3, 6, 3

## Abstract
With strong capabilities of reasoning and a broad understanding of the world, Large Language Models (LLMs) have demonstrated immense potential in building versatile embodied decision-making agents capable of executing a wide array of tasks.
Nevertheless, when deployed in unfamiliar environments, we show that LLM agents encounter challenges in efficiently gathering essential information, leading to suboptimal performance.
Conversely, human individuals often seek additional information from their peers prior to taking action, harnessing external knowledge to avoid unnecessary trial and error. Drawing inspiration from this behavior, we propose \textit{Asking Before Acting} (ABA), a method that empowers the agent to proactively inquire with external sources for pertinent information using natural language during their interactions within the environment. 
In this way, the agent is able to enhance its efficiency and performance by circumventing potentially laborious steps and combating the difficulties associated with exploration in unfamiliar environments and vagueness of the instructions.
We conduct extensive experiments involving a spectrum of environments including text-based household everyday tasks, robot arm manipulation tasks, and real world open domain image based embodied tasks. The experiments involve various models from Vicuna to GPT-4. The results demonstrate that, even with modest prompts modifications, ABA exhibits substantial  advantages on both performance and efficiency over baseline LLM agents.
Further finetuning ABA with reformulated metadata (ABA-FT) faciliates learning the rationale for asking and allows for additional enhancements especially in tasks that baselines struggle to solve.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents a theoretical framework, conceptualizing scenarios as a Markov Decision Process (MDP). This framework harnesses active querying to efficiently extract information from a language model. Remarkably, the proposed method can tweak its queries to be relevant even with slight alterations to existing agents, ensuring that previously acquired information is both retained and effectively repurposed. The examined issue is both relevant and significant. Agents equipped with the ability to tap into external knowledge repositories exhibit enhanced capability and safety over their counterparts. The integration of LLMs with assistance-seeking mechanisms is a novel endeavor. The results outperform the established baseline, and diverse experimental setups have been designed to underscore the method's efficacy.

### Strengths
-The paper presents a unique framework for assessing embodied decision-making, enabling agents to proactively seek information.
-Comprehensive tests were carried out on ALFWord and its derivatives, affirming the method's efficacy.
-The paper is well-organized and clear-presented.

### Weaknesses
- The writing could benefit from improvements, particularly typographical errors like "suboptimal beahviors" found in the second paragraph.
- Why choose to incorporate solely a human model rather than adopting approaches like the RLHF for human feedback?
- The authors' pursuit to tackle intricate issues in embodied AI, especially everyday tasks beyond mere embodied navigation, is praiseworthy. Nonetheless, showcasing the method's versatility across various realms, including embodied navigation, would enhance the paper's value.
- It would also be beneficial if the paper could cover works from pre-LLM era on embodied AI task that takes in help signal to guide its downstream task:
1.Chi, T.C., Shen, M., Eric, M., Kim, S. and Hakkani-tur, D., 2020, April. Just ask: An interactive learning framework for vision and language navigation. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 03, pp. 2459-2466).
2.Zhang, J., Yu, S., Duan, J. and Tan, C., 2022. Good Time to Ask: A Learning Framework for Asking for Help in Embodied Visual Navigation. arXiv preprint arXiv:2206.10606.
3.Singh, Kunal Pratap, Luca Weihs, Alvaro Herrasti, Jonghyun Choi, Aniruddha Kembhavi, and Roozbeh Mottaghi. "Ask4help: Learning to leverage an expert for embodied tasks." Advances in Neural Information Processing Systems 35 (2022): 16221-16232.

### Questions
All my questions are asked in the weakness section.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, a new prompting and LM fine-tuning method is introduced. The goal of Asking Before Acting (ABA) is to gather information (“asking”) from the environment or external sources before performing an action (“acting”). In addition to a zero-shot method, there is also ABA-FT which is fine-tuned on labeled trajectories.

### Strengths
- Originality: The paper addresses an inefficiency of current LLMs in embodied LM. It is much easier to directly ask a question than directly explore the environment. This is also an intuitive idea. 
- Significance: ABA could provide shorter policies. The new versions of AlfWorld provide a more challenging version of a widely used benchmark.

### Weaknesses
- Comparison with previous works: One of the major weaknesses is that several works with competitive baselines are omitted from the paper. For example, in AlfWorld, Reflexion (1) and AdaPlanner (2) achieve similar or better results on AlfWorld than ABA. For the robotics task, there is no comparison with works like Cliport (3). For the finetuned version of ABA, there is no comparision with finetuned models or imitation learning methods. 
- For the new dataset variants introduced, Multiround AlfWorld and Ambigious AlfWorld are not evaluated on methods other than ReACT, ABA, ABA-FT. 
- A key part of ABA is receiving/guidance from the environment or external sources. However, in the paper, a second LLM is used as the 'human' or external source. There is no ablation showing that the external source is providing guidance and not just the answer. 
- An evaluation on more decision making datasets such as programming datasets, block world, etc. would strengthen the paper significantly given the general high performance of LLMs on AlfWorld.

(1): Shinn, Noah, Beck Labash, and Ashwin Gopinath. "Reflexion: an autonomous agent with dynamic memory and self-reflection." arXiv preprint arXiv:2303.11366 (2023).
(2): Haotian Sun, Yuchen Zhuang, Lingkai Kong, Bo Dai, & Chao Zhang. (2023). AdaPlanner: Adaptive Planning from Feedback with Language Models.
(3): Shridhar, Mohit, Lucas Manuelli, and Dieter Fox. "Cliport: What and where pathways for robotic manipulation." Conference on Robot Learning. PMLR, 2022.

### Questions
- What is the expert policy used to train ABA-FT?
- Is there any visual model used for the robotic control experiments?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
In this paper, the authors propose the ABA method to empower agents to gather external contextual information before decision-making, which is inspired by the behavior of humans completing tasks in unknown environments. Agents benefit from the newly proposed ABA paradigm and can avoid unnecessary trial and error, enhancing its efficiency and performance. Experiments demonstrate the effectiveness of the proposed methods.

### Strengths
- The proposed ABA idea is interesting. Compared to traditional self-explore agents, ABA agents can circumvent potentially laborious trial and error.
- Leveraging natural language and ICL to inquire information from humans or LLM is reasonable.
- The authors formulaically define the Contextual MDP with Human / External information sources in the loop based on the Contextual MDP, which provides a solid foundation to follow for future research.
- Extensive details and demonstrations are provided in the appendix, which makes the work easy to follow.
- The experiments are sufficient.

### Weaknesses
Weaknesses:
1. For embodied decision-making tasks, some related works need to be discussed. For example, previous works (e.g. [1]) explore the use of knowledge graph as external commonsense and extract pertinent information via GNN. Also, [2] extracts knowledge from VLM during the decision process. Both works aim to gather external information for improving navigation decisions in unseen environments, which is similar to the ABA idea.

    [1] Room-and-object aware knowledge reasoning for remote embodied referring expression.

    [2] Room-Object Entity Prompting and Reasoning for Embodied Referring Expression.

2. Given that the core of the ABA method is to obtain effective external assistance, and obtaining external assistance has costs such as time, resources, and human efforts. Therefore, how to balance the inquiry frequency and performance is a missing topic in this work.

3. There are still some typos, and the authors need to further polish the writing. 
- Page 4: Missing comma in the sixth line of subsection 3.2.1
- Page 9: "open do- main image" in the conclusion section
- Page 14: "Section ??"

4. Combining the method with a real robot and verifying this work would be even better.

5. Suggestion: Further simplify the formula.

### Questions
See Weaknesses

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces a novel approach, "Asking Before Acting" (ABA), aimed at enhancing the decision-making efficiency of Large Language Models (LLMs) in unfamiliar environments. The core concept is inspired by human behavior, where individuals often seek additional information before taking action, thereby avoiding unnecessary trial and error. The ABA approach empowers agents to inquire proactively using natural language, enhancing their interaction within various environments. The paper begins by acknowledging the proficiency of LLMs in various tasks but highlights their inefficiency in environments with limited or ambiguous information. The ABA methodology is introduced as a solution, allowing agents to ask open-ended questions to gather essential information, leading to more efficient and informed decision-making. This approach is distinct from previous works that often restricted interactions or required human intervention for providing information. To further enhance ABA's performance, the authors introduce ABA-FT, which reformulates metadata associated with question formulation, helping the model understand the rationale behind asking questions. This fine-tuning process leads to notable improvements, especially in challenging tasks.

### Strengths
- The introduction of the ABA methodology is a novel concept in the realm of AI and decision-making. By enabling LLMs to ask questions before taking actions, the authors bridge the gap between human interactive learning and machine autonomy. This approach mimics human behavior, representing a shift towards more intuitive, adaptive AI systems.
- The authors have conducted thorough experiments across diverse environments, which adds credibility to their claims. By testing the ABA methodology in various scenarios, including text-based tasks, robot arm manipulations, and real-world open-domain tasks with image inputs, they demonstrate the model's versatility and applicability in different contexts.

### Weaknesses
- The problem in the paper is formed as Contextual MDP, while I think it's better formulating the problem into a more suitable domain like POMDP, which is much well-accepted and clearly defined. Is there a necessity to use Contextual MDP? I hope the authors can further explain the motivation.
- The paper's methodology heavily relies on the capabilities of Large Language Models (LLMs). I think the method itself does not make much sense. From my point of view, it seems just like the authors design some tasks with ambiguity and use the "ask before action" paradigm to help solve such ambiguity. But the model does not resolve such ambiguity by itself, i.e., through rapid trial-and-error. On the other hand, it still depends on the human to help it. So why not the human provide the complete information at the beginning? Besides, I don't see the proposed "ask before action" paradigm as a big contribution, as we have already see many works related to the LLM agency in which LLMs are driven to finish more complex interactions and tasks.
- There lacks error analysis conducted on instances where ABA underperformed or failed. What insights were derived from these analyses?  Besides, I think additional QA in ABA actually imports more information compared with other baselines, which definitely lead to better performance. Maybe other baselines should also use information like “the second red block from the left” for a equal comparison.

### Questions
See weakness above.

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor
