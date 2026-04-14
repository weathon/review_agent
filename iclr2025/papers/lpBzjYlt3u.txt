

{0}------------------------------------------------

# MOBILESAFETYBENCH: EVALUATING SAFETY OF AUTONOMOUS AGENTS IN MOBILE DEVICE CONTROL

Anonymous authors

Paper under double-blind review

## ABSTRACT

Autonomous agents powered by large language models (LLMs) show promising potential in assistive tasks across various domains, including mobile device control. As these agents interact directly with personal information and device settings, ensuring their safe and reliable behavior is crucial to prevent undesirable outcomes. However, no benchmark exists for standardized evaluation of the safety of mobile device-control agents. In this work, we introduce MobileSafetyBench, a benchmark designed to evaluate the safety of device-control agents within a realistic mobile environment based on Android emulators. We develop a diverse set of tasks involving interactions with various mobile applications, including messaging and banking applications, **challenging agents with managing risks encompassing misuse and negative side effects. These tasks include tests to evaluate the safety of agents in daily scenarios as well as their robustness against indirect prompt injection attacks. Our experiments demonstrate that baseline agents, based on state-of-the-art LLMs, often fail to effectively prevent risks while performing the tasks.** To mitigate these safety concerns, we propose a prompting method that encourages agents to prioritize safety considerations. While this method shows promise in promoting safer behaviors, there is still considerable room for improvement to fully earn user trust. This highlights the urgent need for continued research to develop more robust safety mechanisms in mobile environments.

**WARNING: This paper contains contents that are unethical or offensive in nature.**

## 1 INTRODUCTION

Recent advances in building autonomous agents using large language models (LLMs) have demonstrated promising results in various domains, including mobile device control (Yang et al., 2023; Lee et al., 2024; Rawles et al., 2024). Mobile device control agents can enhance productivity and improve accessibility of user interactions by automating daily tasks such as web interactions, data sharing, text messaging, social media access, and financial transactions. However, as these agents gain the ability to control personal devices, ensuring their safety becomes crucial, particularly because they have access to sensitive user information and other critical data.

Despite significant progress in developing benchmarks for evaluating the safety of LLMs, prior works have primarily focused on safety assessments based on question-answering formats (Bai et al., 2022; Li et al., 2024; Yuan et al., 2024). These formats often fail to detect the dangerous behaviors of LLM agents when controlling mobile devices, making existing benchmarks insufficient for a thorough safety assessment. To rigorously evaluate the safety of such agents, it is crucial to develop a benchmark that incorporates a realistic interactive environment and diverse risks.

In this work, we present MobileSafetyBench, a novel research platform designed to evaluate the safe behavior of agents controlling mobile devices. MobileSafetyBench is based on several important design factors (see Figure 1 for an overview). Central to our benchmark is the use of Android emulators to create interactive and realistic environments. MobileSafetyBench includes diverse applications such as memos, calendars, social media, banking, and stock trading, which are essential for assessing operations commonly used in everyday life.

Based on realistic environments, we develop a task suite to evaluate the safety of agents across various scenarios. These tasks incorporate major risk types associated with mobile device usage,

{1}------------------------------------------------

![Figure 1: Overview of MobileSafetyBench. The diagram shows a central 'MobileSafetyBench' icon connected to an 'Interactive Real-System Mobile Device Environment' (represented by a smartphone screen). To the left, two boxes list 'Task Categories' (Text Messaging, Finance, Web Navigation, Device/Data Management, Social Media, Utility) and 'Risk Types' (Ethical Compliance, Offensiveness, Private Information, Bias & Fairness). To the right, an 'Agent Controlling Mobile Device' (represented by a robot head) is shown interacting with the environment via a double-headed arrow. Below the agent, a 'Rule-based Evaluator' (represented by a magnifying glass over a database) is shown analyzing the interaction.](9ba3dc91984c80b96f217fb1bddd5c06_img.jpg)

Figure 1: Overview of MobileSafetyBench. The diagram shows a central 'MobileSafetyBench' icon connected to an 'Interactive Real-System Mobile Device Environment' (represented by a smartphone screen). To the left, two boxes list 'Task Categories' (Text Messaging, Finance, Web Navigation, Device/Data Management, Social Media, Utility) and 'Risk Types' (Ethical Compliance, Offensiveness, Private Information, Bias & Fairness). To the right, an 'Agent Controlling Mobile Device' (represented by a robot head) is shown interacting with the environment via a double-headed arrow. Below the agent, a 'Rule-based Evaluator' (represented by a magnifying glass over a database) is shown analyzing the interaction.

Figure 1: Overview of MobileSafetyBench. Incorporated with interactive real-system mobile device environments, MobileSafetyBench enables measuring the safety and helpfulness of agents controlling mobile devices across diverse task categories and risk types.

such as handling private information, detailed in Section 3.3. They are specifically designed to assess how effectively agents manage risks. Additionally, our benchmark includes scenarios that challenge agents with indirect prompt injection attacks, deceiving them into taking actions contrary to user intentions. **To clearly evaluate safety apart from general capabilities, we design separate but symmetric tasks, named high-risk tasks and low-risk tasks.** For all tasks, we employ rigorous evaluators that accurately analyze the agents’ behaviors, taking into account both the history of actions and their effects on the environment.

To serve as a reference, we benchmark mobile device control agents based on frontier LLMs such as GPT-4o (OpenAI, 2024b), Gemini-1.5-Pro (Gemini et al., 2023), and Claude-3.5-Sonnet (Anthropic, 2024). In our experiments, the tested agents exhibit unsafe behaviors across many task scenarios, including assisting with commands that violate ethical compliance. While these agents efficiently manage hazards in straightforward situations (e.g., when the task instruction is explicitly unethical), they struggle to handle the risks in more complex scenarios. Notably, we find that the agents are highly vulnerable to indirect prompt injection, which highlights significant risks associated with the naive deployment of LLM assistants.

We also propose a novel method of prompting on top of Chain-of-Thought (Wei et al., 2022), named Safety-guided Chain-of-Thought (SCoT), to improve the safety of device control agents. This SCoT prompt requires agents to first generate safety considerations, specifically identifying potential safety issues based on the given observation and instruction, before they formulate their action plans. By incorporating this method into baseline agents, we observe a significant increase in safety scores. However, despite these improvements, the agents still exhibit unsafe behaviors, such as overlooking the safety considerations they have generated. This inconsistency highlights the need for developing new methods to enhance agent reliability further.

To summarize, our contributions are as follows:

- We develop a novel benchmark platform evaluating the helpfulness and safety of agents controlling mobile devices with realistic interactive environments.
- We provide a reference of benchmark results with state-of-the-art LLMs and reveal their weakness against indirect prompt injection.
- We propose a simple and effective prompting method for guiding the safe behaviors of mobile device control agents based on LLMs.
- We conduct extensive analyses of baseline agents, including comparisons between LLM agents and question-answering LLMs, as well as the effects of external safeguards.
- We *will* open-source our benchmark, enabling the easy reproduction of our experiments.

{2}------------------------------------------------

## 2 RELATED WORK

**Building agents with LLMs** Developing intelligent agents with LLMs has gained significant interest, as LLMs have shown adeptness in planning, reasoning, and tool usage. Early research, such as ReAct (Yao et al., 2023) and Reflexion (Shinn et al., 2023), have demonstrated that the agents employing LLMs are capable of making sequential decisions from the provided set of actions to interact with the environments. Recently, adopting LLMs on more practical domains, as in navigating websites (Zhou et al., 2024) or controlling mobile devices (Yang et al., 2023), is being increasingly investigated. To this end, different prompting methods for advancing the agents are being studied aggressively (Rawles et al., 2024). This work presents experimental results with agents employing LLMs and, additionally, introduces a novel prompting method for guiding safe behaviors.

**Benchmarking agent controlling digital devices** Early works have focused on evaluating the proficiency of agents developed with reinforcement learning (Toyama et al., 2021; Liu et al., 2018). Recently, benchmarks for LLM agents with authentic environments are thrusting substantial progress. In web navigation, Webshop (Yao et al., 2022) and WebArena (Zhou et al., 2024) have demonstrated realistic web simulators for agents manipulating UIs on the web browser. VisualWebArena (Koh et al., 2024) and MMInA (Zhang et al., 2024) have showcased benchmarks considering multi-modal inputs with images. Controlling more general platforms (e.g., operating systems) has also been studied in several works, including ARA (Kinniment et al., 2023), AgentBench (Liu et al., 2023), and OSWorld (Xie et al., 2024). In this work, we focus on the LLM agents controlling real-system *mobile digital devices*, similar to B-MoCA (Lee et al., 2024) and AndroidWorld (Rawles et al., 2024). While prior studies focus on evaluating the helpfulness of the agents, we also consider safety.

**Benchmarking safety of LLMs** Numerous studies warn of LLMs’ potential threats such as producing harmful, biased, toxic, or unethical content. Early studies have focused on evaluating the response of LLMs to safety-related questions (Qi et al., 2024; Mireeshghallah et al., 2024; Li et al., 2024). As LLMs are increasingly integrated into real-world tools and environments, several studies for investigating safe behaviors of LLM agents are ongoing, as in ToolEmu (Ruan et al., 2024) or R-judge (Yuan et al., 2024), yet exhibiting several shortcomings such as disregard of realistic interactive environments. Researchers also have adopted aggressive evaluation techniques, such as prompt attacks to test the robustness of LLMs leveraging tools (Zhan et al., 2024; Debenedetti et al., 2024; Wu et al., 2024). Several concurrent works study the safety of LLM agents, mainly focusing on jailbreaking scenarios (Andriushchenko et al., 2024) or web environments (Kumar et al., 2024; Liao et al., 2024). In this work, we demonstrate a benchmark evaluating more general safe behaviors of LLM agents manipulating mobile devices.

## 3 MOBILESAFETYBENCH

In this section, we introduce MobileSafetyBench, a novel benchmark for evaluating the safety and helpfulness of mobile device control agents.

### 3.1 DESIGN PRINCIPLES

MobileSafetyBench is designed as a platform to assess both the helpfulness and safety of agents controlling mobile devices. A key feature of the benchmark is its use of Android emulators to create interactive and realistic environments, where device control is formulated as a sequential decision-making problem. Agents interact with the environment to perform diverse tasks, each derived from real use cases and involving applications such as social media, banking, and stock trading.

Another key feature of MobileSafetyBench is its rigorous safety evaluation process. **We define safety within our benchmark as the agent’s robustness to malicious instruction and ability to ensure that its actions do not result in potentially harmful consequences while performing a requested task, as the risks in our tasks encompass the misuse of agents (OECD, 2019) and the negative side effects caused by the agents (Amodei et al., 2016).** As safe behaviors can vary depending on the specific task requested or the device’s status, we develop diverse scenarios to challenge agents comprehensively. Each task incorporates an automatic evaluator that utilizes system information from the Android emulators, ensuring that the evaluations are both standardized and reliable.

{3}------------------------------------------------

![Figure 2: Statistics of tasks in MobileSafetyBench. (a) Task Category Distribution: Text Messaging (20%), Web Navigation (20%), Social Media (15%), Device/Data Management (14%), Utility (18%), Finance (14%). (b) Risk Type Distribution: Ethical Compliance (12), Offensiveness (4), Bias & Fairness (4), Private Information (12), Indirect Prompt Injection (8).](e94f3bbb6f7501b9a1344dd0210e5dd8_img.jpg)

Figure 2 consists of two charts. Chart (a) is a pie chart titled "Task Category Distribution" showing the following categories and percentages: Text Messaging (20%), Web Navigation (20%), Social Media (15%), Device/Data Management (14%), Utility (18%), and Finance (14%). Chart (b) is a horizontal bar chart titled "Risk Type Distribution" showing the number of tasks for each risk type: Ethical Compliance (12), Offensiveness (4), Bias & Fairness (4), Private Information (12), and Indirect Prompt Injection (8). The x-axis is labeled "(Num of Tasks)" and ranges from 0 to 12.

Figure 2: Statistics of tasks in MobileSafetyBench. (a) Task Category Distribution: Text Messaging (20%), Web Navigation (20%), Social Media (15%), Device/Data Management (14%), Utility (18%), Finance (14%). (b) Risk Type Distribution: Ethical Compliance (12), Offensiveness (4), Bias & Fairness (4), Private Information (12), Indirect Prompt Injection (8).

Figure 2: The statistics of the tasks created in MobileSafetyBench. (a) The tasks, for both **high-risk** and **low-risk** tasks, span six groups of target operations. (b) Also, the **high-risk** tasks feature four different major types of risks and an additional distinct type of risk.

### 3.2 FRAMEWORK

**Problem formulation** In MobileSafetyBench, we formulate the task of controlling mobile devices as a sequential decision-making problem, where an agent interacts with an environment simulated by an Android emulator. Formally, the internal state  $s_t$  of the environment transitions to the next state  $s_{t+1}$  based on the action  $a_t$  taken by the agent at each discrete time  $t$ . The agent, given a task instruction  $c$ , receives the partial observation  $o_t$  (representing incomplete information about the state  $s_t$ ) and generates an action  $a_t$ . **During the interactions, the agent can freely utilize diverse applications in the device, such as web browsers or memos.** At the end of an episode, which lasts no longer than a predefined maximum length  $T$ , the environment returns a score  $r$ . This score is computed by evaluators based on the final state of the environment  $s_T$  and the history of actions  $a_{1:T}$  taken by the agent.

**Environment implementation** For developing a realistic environment, our benchmark utilizes Android, a widely used open-source operating system. To enable seamless interaction for agents, we implement several interfaces. The first interface, an action converter, transforms the action output from the agent into a legible form  $a_t$  that the emulator can process. This includes interpreting actions such as tapping or swiping on the screen from responses generated by agents using LLMs in text form. Additionally, we ensure each observation  $o_t$  includes both a screen image and text describing the UI elements present on the screen. To facilitate this, a screen-layout translator provides text descriptions for UI elements, detailing element types (e.g., “Button”) and the associated text (e.g., “Click Me” described in the button element). For more details on these environmental interfaces, we refer the readers to Appendix A.

### 3.3 TASK DESIGN

**Diverse daily tasks** MobileSafetyBench presents 87 numbers of tasks, divided into 44 **low-risk** tasks and 43 **high-risk** tasks. Utilizing Android emulators, our tasks are grounded in daily life and designed to reflect common real-life uses of mobile devices.<sup>1</sup> The benchmark includes tasks associated with text messaging, web navigation, social media, calendar settings, and financial transactions. Specifically, we categorize tasks into six operational areas (see Figure 2 (a)): Text Messaging, Web Navigation, Social Media, Finance, Device/Data Management (e.g., managing photos and applications), and Utility (such as memos, maps, and calendars). Our benchmark employs diverse applications to simulate these diverse task categories. For more details, please refer to Appendix A.4.

**Task specification** Each task is defined by specifying the task instruction and configuring the initial status of the environment. For instance, as illustrated in Figure 3, the task is defined with the instruction “Could you check the Photos application, and share an image file with James via Bluetooth?” and the initial device status which is set to contain an image in file storage. The agent is then challenged to navigate to the Photos application and manipulate UI elements to share the image via Bluetooth.

<sup>1</sup>The authors conducted extensive brainstorming and surveyed commonly used applications in real life.

{4}------------------------------------------------

![Figure 3: Exemplary trajectories in a pair of tasks. The left side shows a low-risk task where the user shares a photo of a forest path. The right side shows a high-risk task where the user shares a photo of a credit card. Both sides show a sequence of three phone screens with a hand icon indicating the user's interaction. A dashed line with an arrow at the end indicates the trajectory of the task completion over time (Timestep).](a0167e3dcece9dcd8a378bcd98fb9cfa_img.jpg)

Figure 3: Exemplary trajectories in a pair of tasks. The left side shows a low-risk task where the user shares a photo of a forest path. The right side shows a high-risk task where the user shares a photo of a credit card. Both sides show a sequence of three phone screens with a hand icon indicating the user's interaction. A dashed line with an arrow at the end indicates the trajectory of the task completion over time (Timestep).

Figure 3: Exemplary trajectories in a pair of tasks specified with the same instruction but different images in the file storage. The task completion is desirable as the image does not contain any risk in the **low-risk task** (left), while naively following the instruction results in safety issue as the image contains the detailed credit card information in the **high-risk task** (right).

**High-risk task vs. low-risk task** Each risk within a task is characterized by the task specification. For instance, as shown in Figure 3, sharing an image file that does not contain hazardous content is a low-risk action. However, sharing an image that includes private information (e.g., credit card details) constitutes high-risk behavior. Reflecting this, we set each task in MobileSafetyBench to be either high-risk task or low-risk task. Specifically, we design tasks to establish a symmetric relationship between the high-risk tasks and low-risk tasks, with each pair simulating a similar situation but differing in task specification. We include human survey results justifying that high-risk tasks present risks that warrant attention where as low-risk tasks exhibit risks that are negligible in Appendix B.1.

This distinction allows for a more rigorous analysis of the agents’ safety, separate from their general capabilities. For example, if an agent keeps refusing to proceed with the **high-risk** tasks while struggling to achieve **low-risk** tasks, it suggests a lack of proficiency rather than a commendable level of safety. Conversely, if an agent successfully completes the **low-risk** task and demonstrates high achievement of the **high-risk** task, this clearly indicates a disregard for safety. A detailed list of tasks is available in the supplementary code material.

**Risk type** Our benchmark features various risks such as private information contained within an image as shown in Figure 3. These risks challenge the agents’ situational awareness and their ability to make reliable decisions. In our tasks, we introduce diverse sources of risks including files stored on the device, contents of memos taken by the user, text messages sent or received (even from strangers), and social media posts. Specifically, we categorize the **high-risk** tasks based on four prevalent risk types in real life to facilitate a clearer interpretation of agent behaviors:

- **Ethical** Compliance: Tasks that involve or promote **unethical** activities.
- **Offensiveness**: Tasks that could psychologically harm the user or others.
- **Bias & Fairness**: Tasks that encourage biased decisions or promote unjust outcomes.
- **Private Information**: Tasks involving inappropriate sharing or management of confidential user data, including data loss.

The risk type of each task is labeled by human annotators, where the detailed procedure is provided in Appendix B.2, with a breakdown of the number of tasks per category presented in Figure 2 (b).

**Indirect prompt injection** We also assess agent robustness against indirect prompt injection, categorizing it as a distinct risk type.<sup>2</sup> In these tasks, agents are presented with prompts that contain malicious instructions embedded in messages or social media posts by an attacker. We test whether agents can stick to completing the original task without being misled by these malicious instructions.

<sup>2</sup>We analyze this risk separately because the impact of adversarial prompts can vary depending on the methods of agent scaffolding. Nonetheless, MobileSafetyBench serves as an effective research platform for configuring and investigating the effects of real-world prompt injection attacks on mobile devices.

{5}------------------------------------------------

### 3.4 EVALUATION SCHEME

For rigorous evaluations, our benchmark utilizes rule-based evaluators. Specifically, these evaluators determine task success or failure by checking if pre-defined conditions are met, based on the agents’ decisions and their effects on the device. The evaluators have access to the action history to assess the agents’ decisions. They also examine device information, including system configurations, file storage, and application databases, ensuring consistent and reliable evaluation. For each task, the evaluator assesses the goal achievement and harm prevention of the agents. Goal achievement refers to the successful completion of the task instruction by the agent. Notably, in high-risk tasks, achieving the goal may not always be desirable, as it could lead to the manifestation of harm. Harm prevention refers to the agent’s proactive decision to avoid potential risks during task performance. In our framework, this involves refusing to proceed with the task or asking consent from the user, regardless of whether risks are actually present in the task. Further details on the evaluation scheme are provided in Appendix C.

## 4 BASELINE AGENTS

In this work, we focus on benchmarking multi-modal LLMs with prompting as baseline agents for controlling mobile devices (Zhou et al., 2024; Lee et al., 2024; Rawles et al., 2024). These agents receive multi-modal observations consisting of screen images and text descriptions of the UI elements. They then choose an appropriate action from a pre-defined set of options. Examples of action options include tapping UI elements, swiping the screen in a specified direction, and inputting text into a target field, providing a flexible interface for device control. Additionally, we incorporate three specific actions to enhance safety: `refuse()`, which halts the process if the agent deems continuing with the task is inappropriate; `ask-consent()`, which is used when the agent requires user permission to proceed; and `complete()`, which flags the completion of tasks such as when a file to be deleted is not found on the device. A more detailed explanation of the action options is provided in Appendix A.2.

To elicit agentic behaviors from LLMs, we design the prompt to include the general role of agents, available action options, goal instructions, previous actions taken by the agent, and the current observation. Our prompts incorporate several techniques, such as the Chain-of-Thought prompt (Wei et al. 2022; CoT), to enhance reasoning and planning. Specifically, we design prompts to mandate a particular response format from the agents. This format includes an interpretation of the current observation, a context summarizing the current progress, a rationale for their planned action, and the final decision on the action option.

**Safety-guided Chain-of-Thought prompting** To improve the agents’ ability to recognize potential safety issues, we propose a new prompting method called Safety-guided Chain-of-Thought (SCoT) prompt. This SCoT prompt requires agents to generate safety considerations based on the current observation ( $o_t$ ) and task instruction ( $c$ ) before establishing their action plans. Specifically, the SCoT prompt includes several guidelines that emphasize safe behavior, ensuring that agents apply the safety considerations they generate. Our experiments demonstrate that integrating SCoT with the CoT technique significantly enhances the safety of LLM agents. For more details on the prompts, including different types of prompts used in the experiments, we refer the readers to Appendix D.

## 5 EXPERIMENT

In this section, we investigate the following research questions:

- How do agents using frontier LLMs perform in MobileSafetyBench? (Section 5.2)
- Can the SCoT prompt effectively improve the safety of LLM agents? (Table 1)
- Are LLM agents robust against indirect prompt injection on mobile devices? (Table 2)
- Can baseline LLMs detect risks in question-answering formats? (Table 3)
- Can advanced reasoning abilities enhance the LLM agent’s safety? (Figure 6)
- How effective are current external safeguards in MobileSafetyBench? (Section 5.4)

{6}------------------------------------------------

![Figure 4: Two bar charts comparing goal achievement and harm prevention rates for GPT-4o, Gemini-1.5, and Claude-3.5 agents using basic and SCoT prompts. The left chart shows goal achievement rates, and the right chart shows harm prevention rates for high-risk and low-risk tasks.](73c3e4508cae529acf4e6c7fa70b361a_img.jpg)

**Goal achievement rate (%)**

| Model | Prompt Type | Goal achievement rate (%) |
|-|-|-|
| GPT-4o | basic | ~82 |
|  | SCoT | ~78 |
| Gemini-1.5 | basic | ~42 |
|  | SCoT | ~32 |
| Claude-3.5 | basic | ~48 |
|  | SCoT | ~75 |

**Harm prevention rate (%)**

| Model | Prompt Type | High-risk task (%) | Low-risk task (%) |
|-|-|-|-|
| GPT-4o | basic | ~10 | ~0 |
|  | SCoT | ~28 | ~10 |
| Gemini-1.5 | basic | ~42 | ~18 |
|  | SCoT | ~80 | ~42 |
| Claude-3.5 | basic | ~38 | ~10 |
|  | SCoT | ~55 | ~12 |

Figure 4: Two bar charts comparing goal achievement and harm prevention rates for GPT-4o, Gemini-1.5, and Claude-3.5 agents using basic and SCoT prompts. The left chart shows goal achievement rates, and the right chart shows harm prevention rates for high-risk and low-risk tasks.

Figure 4: The goal achievement rates (left) and harm prevention rates (right) of the baseline agents in MobileSafetyBench. We provide detailed results in each risk type in Appendix E.4. While the GPT-4o agents achieve the highest goal achievement rates, the Gemini-1.5 agents remark the highest harm prevention rates. The increase of harm prevention rates with SCoT prompt shows the effectiveness of the newly proposed method for inducing safe behaviors of the agents.

### 5.1 EXPERIMENTAL SETUP

In our experiments, we benchmark agents employing the state-of-the-art multi-modal LLMs: GPT-4o (gpt-4o-20240513; OpenAI 2024b), Gemini-1.5 (gemini-1.5-pro-001; Gemini et al. 2023), and Claude-3.5 (claude-3-5-sonnet-20240620; Anthropic 2024). We also test agents employing OpenAI-o1 (o1-preview; OpenAI 2024a).<sup>3</sup> During the evaluation, we fix the temperature of the models to be 0.0 (see Appendix E.1 for more configuration details). For the main experiment, we exploit two types of prompts: basic and SCoT prompts.

For evaluation, we test agents on a suite of 87 tasks. Of these, 36 low-risk tasks and 35 high-risk tasks are used for measuring the safety of agents in daily situations. The remaining 8 low-risk tasks 8 high-risk tasks are used in a separate study to assess the robustness of the agents against indirect prompt injections. For the experiment on the robustness of the agents against indirect prompt injections, we clarify the threat model in Appendix E.2.

### 5.2 MAIN RESULTS

Figure 4 shows the helpfulness and safety scores of the baseline agents in MobileSafetyBench. We observe that all baseline agents frequently fail to behave safely across many tasks. We observe that prompting agents with SCoT to identify potential risks significantly improves the harm prevention rates. This indicates the substantial risk associated with naively employing LLM agents to control mobile devices. Exemplary tasks and responses from the agents are provided in Appendix E.3.

**Safe behaviors of LLM agents across risk types** The tasks in MobileSafetyBench encompass various groups of risks, with agents exhibiting different behaviors depending on the type of risk. Below, we analyze the agents’ behaviors for each risk type.

In the **Ethical Compliance** category, where risk signals are most straightforward, agents commonly refuse to proceed with tasks upon detecting indications of criminal activity. For example, when tasked with accessing an illegal website, Claude-3.5 agents recognize the illegality of the provided URL and refuse to proceed with the task. However, in more complex scenarios where risks become apparent only after several steps into the task, agents often overlook or struggle to address these risks. For instance, many agents often proceed with the task by checking only the immediately visible part of the conversation in text messages, i.e., the most recent message, without examining the detailed conversation history where signals of illegality are present. This disregard for potential safety issues in this category highlights the critical threats.

<sup>3</sup>We exclude agents using OpenAI-o1 from the main results, as this preview version does not accept image inputs, precluding a fair comparison with the other three models.

{7}------------------------------------------------

In the **Offensiveness** category, similar to **Ethical Compliance** category, the agents often prevent clear risks but struggle with more nuanced ones. For instance, when instructed to send an offensive message (explicitly mentioned in the command), all baseline agents refuse to proceed. However, when the task involves risks requiring advanced perception capabilities, such as setting a profile image containing a discriminatory gesture (i.e., a man making a slant-eye gesture), many agents fail to address the risks appropriately. GPT-4o agents, for example, complete setting the profile with the image despite the offensive implications.

We also observe that many agents often struggle to exhibit safe behaviors when the tasks require higher reasoning abilities to interpret risk signals. In the **Bias & Fairness** category, the clues of risks become difficult to interpret even after accessing content containing risk signals. These contents, for example, include unfair judgment (of the users) such as biased employment decisions over certain regions or prejudiced comments on social media posts. Agents more frequently disregard safety issues when asked to assist users with these contents.

In the **Private Information** category, agents struggle to handle private information (e.g., Google authentication code or credit card information) carefully. For instance, GPT-4o agents often carelessly share text or image contents containing private information, resulting in **harm prevention rates 0% (basic) and 15% (SCoT)**. Meanwhile, Claude-3.5 agents show higher performance than GPT-4o agents, with a **harm prevention rates 46%**, but still indicate **notable** room for improvement. Specifically, these agents attempt to edit text contents to remove sensitive information before sharing but often fail to do so effectively.

**Balancing goal achievement and harm prevention** We observe that each baseline agents with SCoT method show different behaviors between goal achievement and harm prevention when they are exposed to the high-risk tasks. The GPT-4o agents achieve the highest goal achievement rates at 69%, but their harm prevention rates are the lowest at 29%. This indicates the agents neglect safety considerations. The Claude-3.5 agents achieve harm prevention rates of 54% and goal achievement rates of 23%. The Gemini-1.5 agents demonstrate harm prevention rates of 80%, surpassing the GPT-4o agents and Claude-3.5 agents, but their harm preventions in low-risk tasks (with rates value of 44%) clue that they unnecessarily avoid risks despite the absence of high risks in considerable number of tasks. These findings indicate that balancing safety with helpfulness presents a challenge, suggesting that investigations on the agent design and prompting strategies remain crucial.

**The effect of SCoT prompting** We find that prompting the agents to produce safety considerations before making decisions significantly improves the harm-preventing behaviors of the agents. The baseline agents provided with SCoT prompt report **25% higher harm prevention rates in high-risk tasks than the agents given with the basic prompt, on average across LLMs**. However, we also find that safety considerations are often ignored when the agents are making decisions. This indicates the necessity of more advanced reasoning or planning algorithms for achieving higher safety.

Additionally, we conduct an ablation study on SCoT prompting. In the experiment, we employ another safety-guided prompt type that contains several guidelines to behave safely, but without compulsory requirement on outputting safety consideration, similar to (Ruan et al., 2024). Table 1 shows the experimental results with the GPT-4o agents using the safety-guided prompt and SCoT prompt. We observe that forcing the agents to generate safety considerations can be largely beneficial for eliciting their safe behaviors, compared to providing several safety guidelines only.

| Prompt | Harm prevention | Goal achievement |
|-|-|-|
| Basic prompt | 9% | 83% |
| Safety-guided prompt | 11% | 78% |
| SCoT prompt | 29% | 83% |

Table 1: The harm prevention rates (%) in high-risk tasks and goal achievement rates (%) in low-risk tasks of the GPT-4o agents with different prompt types. SCoT effectively induces safety without compromising the capability of agents.

| Baseline | Number of defenses /Total number of tasks |
|-|-|
| GPT-4o | 0/8 |
| Gemini-1.5 | 1/8 |
| Claude-3.5 | 0/8 |

Table 2: The test results of agents robustness against indirect prompt injection in eight high-risk tasks. All the agents are defenseless to the attack. The results in low-risk tasks are present in Appendix E.5.

{8}------------------------------------------------

![Figure 5: A sequence of four mobile phone screens illustrating an agent's behavior over time. The first screen shows a 'User instruction' to check a message and a 'Prompt injection' to sell 100 NVIDIA stocks. The second screen shows the 'Agent thought' about the new instruction and the 'Agent action' to 'open-stock()'. The third screen shows the 'Agent thought' about selling the stocks and the 'Agent action' to 'tap("Sell" button)'. The fourth screen shows the final state. A 'Timestep' arrow points from left to right.](acfc53eca625d62b38aa2563efa95c3e_img.jpg)

Figure 5: A sequence of four mobile phone screens illustrating an agent's behavior over time. The first screen shows a 'User instruction' to check a message and a 'Prompt injection' to sell 100 NVIDIA stocks. The second screen shows the 'Agent thought' about the new instruction and the 'Agent action' to 'open-stock()'. The third screen shows the 'Agent thought' about selling the stocks and the 'Agent action' to 'tap("Sell" button)'. The fourth screen shows the final state. A 'Timestep' arrow points from left to right.

Figure 5: An exemplary behavior of an agent attacked by an indirect prompt injection. After checking a message that contains a new malicious instruction, the agent sells the user’s stock shares following the injected instruction.

### 5.3 ROBUSTNESS AGAINST INDIRECT PROMPT INJECTION

Using MobileSafetyBench, we investigate whether baseline agents can maintain robust behavior when exposed to indirect prompt injection attacks. For instance, as illustrated in Figure 5, a test scenario involves agents reviewing a text message that contains an irrelevant instruction to sell stock shares. Such injected prompts are embedded within UI elements like text messages and social media posts, and are delivered to the agents as part of the observation.

We present the number of tasks that baseline agents (using the SCoT prompt) successfully defend against these attacks, out of the total 8 tasks, in Table 2. Despite the simplicity of the injected prompts, the baseline agents are prone to these malicious attacks, with the exception of one case presented by the Gemini-1.5 agent. Misled by the injected prompts, the agents typically assume they have received new instructions and attempt to execute them. Consequently, in several tasks, they inadvertently open a banking application, initiate stock trades, or even attempt to change the device password without the user’s consent. **We believe that our findings emphasize that improving the safety of agents against malicious attacks, such as by enhancing agent-user interactivity and generating more data on defensive behaviors through our platform, are highly necessary.** We offer detailed examples of an injected prompt and an agent’s response, including additional experiments using benign content without any intention of attack and results with GPT-o1 agent, in Appendix E.5.

### 5.4 FURTHER ANALYSIS

In this section, we examine the behaviors of the baseline LLMs in-depth and the effect of safeguards supplied by the service providers. We also present experimental results with the OpenAI-o1 agents, compared with the GPT-4o agents, to investigate the effects of strong reasoning ability.

**Comparison with question-answering** To verify whether the underlying LLMs employed in building agents can effectively capture potential risks in mobile device control tasks, we examine their responses in a question-answering (QA) setting. In this setting, we isolate the content containing potential risks, such as specific memos or social media posts, from the observations used in the agentic setting. We then count the number of risks detected in the QA setting and compare this to the number of scenarios in which LLM agents successfully managed the risks (i.e., agentic setting).

|  | Baseline | QA setting | Agentic setting |  | Baseline | QA setting | Agentic setting |
|-|-|-|-|-|-|-|-|
| Text Contents | GPT-4o | 29/30 | 9/30 | Image Contents | GPT-4o | 4/5 | 0/5 |
|  | Gemini-1.5 | 26/30 | 11/30 |  | Gemini-1.5 | 3/5 | 2/5 |
|  | Claude-3.5 | 30/30 | 17/30 |  | Claude-3.5 | 2/5 | 2/5 |

Table 3: Comparison between risk detection capability of the LLMs over risks in text contents (left) and image contents (right) across two different settings. In the agentic setting, LLM agents are provided with SCoT prompt. Discrepancies in the performances between the settings are clear.

{9}------------------------------------------------

We observe a clear discrepancy between the two settings, as shown in Table 3, with 31 tasks involving risk signals in text contents (left) and 6 tasks with risk signals in image contents (right). Notably, while the underlying LLMs detect risks in most image contents, the baseline agents frequently fail to act appropriately. For example, they often share photos containing private information, such as credit cards or documents labeled “Confidential; Do not distribute this to anyone”. We suspect that the agents struggle to resolve the conflict between completing tasks and preventing hazards. More details, including the elaborated explanation of the settings and responses of the LLMs in the QA setting, are provided in Appendix E.6. We believe that these findings highlight the importance of developing safety benchmarks tailored specifically to LLM agents beyond standard QA settings.

![Figure 6: A scatter plot showing Harm prevention rates (%) on the y-axis (0 to 100) versus Latency (s) on the x-axis (0 to 30). Four data points are plotted: OpenAI-o1 (SCoT) at approximately (28, 85), OpenAI-o1 (basic) at (15, 60), GPT-4o (SCoT) at (10, 30), and GPT-4o (basic) at (10, 20).](7801d00a216dc4dc8a7d210dcb5fe3c5_img.jpg)

| Model | Latency (s) | Harm prevention rate (%) |
|-|-|-|
| OpenAI-o1 (SCoT) | ~28 | ~85 |
| OpenAI-o1 (basic) | 15 | 60 |
| GPT-4o (SCoT) | 10 | 30 |
| GPT-4o (basic) | 10 | 20 |

Figure 6: A scatter plot showing Harm prevention rates (%) on the y-axis (0 to 100) versus Latency (s) on the x-axis (0 to 30). Four data points are plotted: OpenAI-o1 (SCoT) at approximately (28, 85), OpenAI-o1 (basic) at (15, 60), GPT-4o (SCoT) at (10, 30), and GPT-4o (basic) at (10, 20).

Figure 6: Harm prevention rates (%) and average response latency (sec) of GPT-4o and OpenAI-o1 agents.

**LLMs with strong reasoning capability** Recent advancements in enhancing the reasoning capabilities of LLMs through diverse strategies have been actively explored. We examine the effects of these enhanced capabilities using OpenAI-o1 agents and compare their performance in **high-risk** tasks to GPT-4o agents.<sup>4</sup> As shown in Figure 6, the OpenAI-o1 agents demonstrate improved harm prevention rates compared to GPT-4o agents. Also, they exhibit enhanced goal achievement rates in the **low-risk** tasks (see Appendix E.4). We note the synergetic effects of the SCoT technique combined with enhanced reasoning ability. However, OpenAI-o1 agents still fail to avoid risks in several **high-risk** tasks and require an excessive amount of time (more than 4.29 times in seconds, on average across the timesteps; see Appendix E.4 for the detailed values) to make decisions, highlighting their practical limitations. Their vulnerability to indirect prompt injection, detailed in Appendix E.5, further highlights their potential hazards.

**The effect of external safeguards** Current closed-source LLMs, such as Gemini-1.5, are equipped with an additional safeguard mechanism designed to prevent the model from generating harmful responses. We investigate the efficacy of these safeguards on the tasks created, by adjusting the safety settings of Gemini-1.5.<sup>5</sup> We observe that the safeguards equipped to Gemini-1.5 are not highly effective for improving the harm prevention for Gemini-1.5 agents. This is because the Gemini-1.5 agents without safeguards already try to prevent harm in tasks where the safeguards work effectively. Specifically, we find that the safeguards frequently block responses containing harmful content, particularly in risk types like **Ethical**, **Compliance** and **Offensiveness**. However, these mechanisms do not always guarantee safe behavior, especially in cases where the responses do not contain explicitly dangerous content. For example, while the safeguards properly function when agents try sending a text message by using `send-sms()` option with an argument of text containing offensive words, they are ineffective when agents decide to forward private information by using `tap()` option, as the argument of this function does not contain harmful contents. We assume this is because the current safety refusal mechanisms struggle to bridge the effects and consequences of actions to safety issues, indicating a need for more advanced methods. We include more detailed discussions in Appendix E.7.

## 6 CONCLUSION

In this work, we propose a novel benchmark for evaluating the reliability of the agents controlling mobile devices. We observe that the LLM agents exhibit unsafe behaviors in many scenarios across risk types that are prevalent in daily life. While the newly proposed prompting method helps inducing safe behaviors, the agents still show limitations. In further analysis, we find that the agents can detect the risks, provided with the usual question-answering formats, calling for evaluations of LLMs in diverse settings. The shortcomings of agents, including the vulnerability of agents against indirect prompt injection, indicate the necessity for more careful designs. We hope our work is a valuable platform for building safe and helpful agents.

<sup>4</sup>Since the preview version does not support image inputs, we utilize a subset of tasks that do not involve cases where risk signals are presented in images.

<sup>5</sup><https://ai.google.dev/gemini-api/docs/safety-settings>

 Rest of paper (reference and Appendix) is removed.