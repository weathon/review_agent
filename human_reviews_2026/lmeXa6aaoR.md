# BrowserArena: Evaluating LLM Agents on Real-World Web Navigation Tasks

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
LLM web agents now browse and take actions on the open web, yet current agent evaluations are constrained to sandboxed environments or artificial tasks. We introduce BrowserArena, a live open-web agent evaluation platform that collects user-submitted tasks, runs Arena-style head-to-head comparisons, and uses step-level human feedback to surface failure modes. Collecting and analyzing step-level annotations on the agent traces, we identify three consistent failure modes: captcha resolution, pop-up banner removal, and direct navigation to URLs. By constructing targeted datasets to further study these tasks, we discover variations in how different language models navigate these failure modes. We find, for example, that o4-mini deploys a wider variety of strategies to circumvent captcha resolution than other models and DeepSeek-R1 consistently misleads users about captcha resolution. Our findings surface both the diversity and brittleness of current web agents. More broadly, our benchmarking methodology provides an approach to evaluating and understanding web agent failure modes at scale.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work builds a toolkit for collecting web-based tasks and annotating agent playing trajectories in an open web environment. The authors also conduct several experiments to analyze the performance of current large models on the open web, and identify their main failure patterns.

### Strengths
This is a toolkit that allows web agents to execute tasks in the open web environment, making it easier to crowdsource more tasks and annotations from users. Compared with contributions focusing on datasets or benchmarks, this work is more suitable to be evaluated under a demo track.

### Weaknesses
First, this data collection toolkit should ideally address at least some of the failure patterns it identifies, such as handling CAPTCHA and closing pop-ups. These are not really the intended goals of web-agent research — these are trivial, procedural problems that can be solved with simple dedicated pipelines. For example, we can’t really say that solving CAPTCHA is a core capability which web agents are developed for. If that’s the case, why doesn’t the toolkit itself handle these issues to avoid their interference with the main conclusions?
(It feels somewhat ironic if the biggest challenge revealed by this platform for current web agents turns out to be “solving CAPTCHAs and closing banners.”)

Second, before submission, the authors could benchmark their platform against existing “arena” frameworks such as Chatbot Arena, and evaluate which features an arena should have—for example, a more fine-grained ranking system and anomalous user detection. These design elements are crucial for ensuring the accuracy and robustness of evaluations in an open arena setting.

Third, has this arena attracted a large user base? If not, what is the plan to engage more users? Designing and releasing an open-source toolkit is primarily an engineering task, but to demonstrate that the toolkit truly works, it needs to be instantiated with a sufficient number of real data points, showcasing its usage and analysis at scale.

### Questions
* Why doesn’t this toolkit help mitigate issues like CAPTCHA and pop-ups to prevent them from distorting the main evaluation conclusions?
* How does the arena ensure evaluation accuracy and robustness in an open setting?
* Has it attracted a large number of users, and if not, what strategies are in place to do so?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces BrowserArena, a live, open-web platform for evaluating LLM web agents using user-submitted tasks and Chatbot Arena-style pairwise comparisons. To address limitations of final-output metrics, the core contribution is a methodology utilizing granular step-level human feedback collected on agent traces. Analyzing 109 user-submitted tasks, the authors identify three consistent agent failure modes: captcha resolution, pop-up banner removal, and unintended direct navigation. Subsequent targeted experiments demonstrate notable behavioral differences and brittleness across contemporary LLM agents in handling these real-world obstacles.

### Strengths
The most significant contribution is the diagnostic evaluation methodology rooted in collecting step-level human annotations, which effectively surfaces intermediate performance issues that traditional final-output benchmarks overlook. The use of live, user-submitted tasks enhances the ecological validity, avoiding the highly specific or artificial constraints of self-hosted environments. Furthermore, the empirical rigor shown by using the collected feedback to identify and construct specialized datasets focused on persistent failure modes (e.g., Captcha, Pop-up) provides quantifiable insights into agent deficiencies.

### Weaknesses
1. The participants' motivation must be considered when building this arena. The motivation of participants in a chatbot arena is clearly stronger, as the inherent instability of LLMs makes people need to see the outputs of different LLMs, and the overall process is fast and efficient. However, for GUI Agents, participants seem to lack sufficient motivation to interact by watching two GUI Agents output different reasoning. Without enough motivation, this system will not be scalable, and the value will not be significantly different from offline labeled evaluation data. I believe this is the core problem of this work. To justify this motivation, it should be demonstrated either through the actual number of user-generated interactions or by conducting interaction experiments in the field of HCI.
2. Some insights arise from the limitations of the evaluation setup: for instance, the lack of methods and evaluation for multimodal observation and grounding.
3. Relying on the browseruse framework is also unreasonable because there are multiple implementation paradigms for web agents, which significantly limits the value generated by the evaluation.

### Questions
1. How to evaluate the validity of step-level feedback? This is because many tasks for GUI Agents cannot be judged as correct or incorrect based solely on the current step (as a single semantic unit/subtask often requires multiple steps), and some GUI Agents may be skilled at recovering from errors.
2. What specific distributional insights are there regarding the user-submitted queries, and what potential biases exist?

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
3

### Summary
This paper presents an evaluation platform, BrowserArena, that collects user preference data on 109 user-submitted tasks to construct a language model leaderboard.
They proposed a new method for evaluating LLM performance in web browsing by collecting step-level user annotations on agent traces and analyzing them to identify failure modes.

### Strengths
- The dataset is well-motivated and moves toward challenges that are better representative of real-world tasks instead of sandbox tasks.
- This work focuses on failures in CAPTCHA, pop-ups, and navigation, which are overlooked in many other benchmarks.

### Weaknesses
- Adding more information on data quality and distribution would be helpful.

- Authors need to add more details and information to ensure reproducibility.

### Questions
- What happens if both responses from agents are bad? 

- Most of the figures are not readable and need larger font/better color selection.

### Soundness
4

### Presentation
4

### Contribution
3
