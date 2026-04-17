# ScienceBoard: Evaluating Multimodal Autonomous Agents in Realistic Scientific Workflows

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Large Language Models (LLMs) have extended their impact beyond Natural Language Processing, substantially fostering the development of interdisciplinary research. Recently, various LLM-based agents have been developed to assist scientific discovery progress across multiple aspects and domains. Among these, computer-using agents, capable of interacting with operating systems as humans do, are paving the way to automated scientific problem-solving and addressing routines in researchers' workflows. Recognizing the transformative potential of these agents, we introduce ScienceBoard, which encompasses two complementary contributions: (i) a realistic, multi-domain environment featuring dynamic and visually rich scientific workflows with integrated professional software, where agents can autonomously interact via different interfaces to accelerate complex research tasks and experiments; and (ii) a challenging benchmark of 169 high-quality, rigorously validated real-world tasks curated by humans, spanning scientific-discovery workflows in domains such as biochemistry, astronomy, and geoinformatics. Extensive evaluations of agents with state-of-the-art backbones (e.g., GPT-4o, Claude 3.7, UI-TARS) show that, despite some promising results, they still fall short of reliably assisting scientists in complex workflows, achieving only a 15% overall success rate. In-depth analysis further provides valuable insights for addressing current agent limitations and more effective design principles, paving the way to build more capable agents for scientific discovery. Our code, benchmark, and leaderboard are available at https://qiushisun.github.io/ScienceBoard-Home/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces SciBoard, a benchmark designed to evaluate LLM/VLM-based agents on scientific-discovery workflows in the following domains: biochemistry, algebra, theorem proving, geoinformatics, astronomy, and scientific documentation. The authors created six environments as virtual machines that can be interacted with via command lines, APIs, or the GUI. Across all environments, SciBoard regroup set of 169 tasks designed by domain experts. Each task provides specific instructions and a mean to automatically evaluate its completion. The authors empirically show how challenging SciBoard is for current LLM/VLM-based agents achieving only a 15% overall success rate.

### Strengths
- A lot of work went into designing and implementing the six environments for SciBoard. Then, designing the tasks within these environments required careful consideration of the unique challenges and diversity present in each scientific domain.
- The authors show the impact of different type of observations (e.g., images, text, images+text). This provides useful insights into how different modalities contribute to agent capabilities in those particular scientific workflows.

### Weaknesses
- I don't see a clear mention of train/validation/test splits for the tasks in SciBoard. Are all 169 tasks used for evaluation only? If so, how do you envision researchers using SciBoard for training their agents? You can bet people will want to use SciBoard for RL fine-tuning their agents on it as well.
- For figure 5, it is mentioned that "GPT-4o and InternVL3 suffer clear drops in performance, whereas Qwen2.5-VL remains largely unaffected, indicating better adaptation to GUI execution". To me, it could also indicate that Qwen2.5-VL is not really leveraging the CLI and only relies on GUI. It would be interesting to see the CLI-only performance for that model as well to confirm that hypothesis.

#### Minor
- How was the human baseline obtained? Was it through domain experts using the same interface as the agents?
- line 376: Gemini-2.0-Flash (Team, 2024) -> (Google, 2024)

### Questions
- Is there anything that prevents the agent from cheating given that its has access to the full machine? Could it for instance read some log files or inspect memory to get extra information that a human user wouldn't have access to?
- In table 2, what does the two open problems correspond to exactly? How do you evaluate for this?
- In table 2, in the execution section, what does steps and time correspond to exactly? Are those obtained from human trajectories? Are those human expert or non-expert?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents "yet another benchmark" for evaluating current LLM agents' capabilities in navigating typical scientific workflows across different domains. These workflows encompasses interacting with multiple tools, parsing feedback, multi-modal interpretation and reasoning, coding and domain-specific knowledge. The environment provided to an agent is VM with the necessary tools and the action space is unified through GUI and CLI. The conclusion is that current LLMs in the standard ReAct framework severely under-performs human across all domains (with overall success rates of ~15% vs ~60%). One interesting empirical finding is that separating planner and action executor improves the performance.

### Strengths
1. The paper presents a well-engineered agentic evaluation and a benchmark on realistic and complex scientific workflows across multiple domains.
2. The experimental analysis is quite comprehensive, with ablations on the operators of ReAct and tools.

### Weaknesses
1. There are no experiments with test-time scaling infra -- these tasks are too complex to be done in one-shot with simple prompting. Current experiments do not fully and fairly demonstrate the LLM abilities to reason and interact with environment since the provided compute infra is quite limited. 
2. The sampling params like max_tokens = 1500 seem too restrictive for the frontier LLMs, also it seems that they were not run in reasoning mode. Even then, these LLMs typically would require a lot of think tokens (much more than 1.5k tokens) on these complex tasks.
3. Several technical details are unclear. For example, how is the memory implemented -- is it simply the concatenation of observations, actions or did it involve some summarization? With more complex tasks, the prompt can get super long and naturally LLMs will run into all sort of long-context issues. Also. please provide ReAct traces for a few tasks.
3. The presentation can be improved with better formatting on tables, so that one can easily get some insights on what models perform well on certain domains.

### Questions
Check above weaknesses.
1. I wonder if the models were run in native-reasoning mode (gpt-oss even supports different level of reasoning). 
2. Did the authors run ablations or sweeps on sampling parameters, especially max_tokens?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents ScienceBoard, a new benchmark for testing AI agents on real scientific software. It runs in a virtual machine and includes 169 tasks across six domains (like biochemistry and astronomy) using professional tools like ChimeraX and Celestia. Agents have to use both GUI and CLI to complete tasks. The authors tested top models like GPT-4o and found they fail badly, with only a 15% success rate, showing these agents aren't ready to be "AI co-scientists" yet.

### Strengths
The main strength is that this is a novel and needed benchmark. Testing agents on real scientific software is a great idea. The technical work to get this running is impressive, especially the way they check the internal state of the apps to see if a task is done. The paper is clearly written. The analysis is also good, especially the part where they separate the "planner" (like GPT-4o) from the "actor" (GUI models), which gives a good hint at how to build better agents.

### Weaknesses
The paper is strong, but a few things could be better. The pass/fail scoring is a bit harsh; an agent that almost finishes a task gets the same zero as one that fails immediately, so it's hard to see partial progress. The number of tasks per domain is a bit uneven. Also, the "scientific discovery" framing is a slight overstatement. The tasks are more about using scientific tools correctly, not really about discovering new science, which is fine but good to be clear about. The two "Open Problems" sound cool, but it's a shame there are only two.

### Questions
Your GUI-only vs. GUI+CLI analysis is great. Can you give a clearer breakdown of the 169 tasks? How many must be solved with the GUI (like in Celestia?), and how many are hybrid, where you disabled the CLI for the test? This would help clarify the balance of skills needed.

What are the two "Open Problems" mentioned in Table 2? They sound really interesting. How are they different from the "Hard" tasks, and how do you evaluate them?

I know the pass/fail scoring is tough. Since your tasks seem to have multiple steps (from Fig 3), have you thought about a partial credit or step-based success rate? It could show more granular progress, even if it's not the main metric.

### Soundness
3

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
This work presents ScienceBoard, a multi-modal benchmark for evaluating computer-use agents in automating scientific workflows. The benchmark covers 169 expert-curated tasks from several scientific domains and focuses on the agents' abilities in using domain-specific software to complete scientific tasks. Evaluations results on several popular agents reveal the challenge of existing models to solve the tasks.

### Strengths
1. This work is a timely addition to the community for evaluation and development of autonomous agents for scientific workflows.
2. Evaluation results delivers insights into the performance of existing agents, especially under different input modalities.

### Weaknesses
1. I think the tested models are outdated for this conference. Given the rapid development of the field, it will be great to include results from more recent models, such as GPT-5, Gemini-2.5, Qwen3, etc. 
2. Related to 1, the cost of evaluation is unclear from the paper, which may hinder the adoption.
3. Several key details are missing, which are critical for assessing the quality of the benchmark:
    1. What are the background of the experts/annotators? Since the tasks are drafted by annotators, are they ecologically valid tasks that correspond to real-world scientific workflows?
    2. line 325, how is difficulty defined?
    3. How is human performance measured? Intuitively, average population and professionals should perform quite differently for scientific tasks, it is important to discuss the background of participants in human study.
4. The scope of this paper seems to be over-claimed.  While the author claims to cover "end-to-end scientific exploration workflows", critical research steps like literature review, idea generation, and report/paper writing are missing. The focus is mainly on execution and using scientific software.
5. Evaluation setup is unclear. How many trials are allowed for each agent? Do you use the same context length limit for all agents? Also, as far as I know, o3-mini supports multimodal input. Why it is only evaluated in the a11ytree only setting?

### Questions
1. Table 2, what's the unit for length? How is time consumption calculated?

### Soundness
2

### Presentation
2

### Contribution
2
