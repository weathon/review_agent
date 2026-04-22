# FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
Mobile GUI agents are becoming critical tools to improve user experience on smart devices, with multimodal large language models (MLLMs) emerging as the dominant paradigms in this domain. Current agents, however, rely on explicit human instructions, overlooking the potential to leverage the contextual information (like location, time, user profile) and historical data for proactive task suggestions. Besides, previous works focus on optimizing the success rate during task execution, but pay less attention to the personalized execution trajectory, thereby neglecting potentially vast differences in user preferences. To address these challenges, we introduce the FingerTip 20K benchmark. We collected 20K unique human demonstrations of multi-step Android device interactions across a variety of everyday apps. These demonstrations are not isolated but are continuously acquired from the users' long-term usage in their real lives, and encompass essential user-related contextual information. The benchmark contains two new tracks: proactive task suggestions by analyzing environment observation and users' previous intents, and personalized task execution by catering to users' action preferences. Our experiments reveal that the tracks we propose pose significant challenges for leveraging user-related information in GUI tasks. We also performed a human study to show that there exists a huge gap between existing agents and humans. The model fine-tuned with the data we collected effectively utilized user information and achieved good results, highlighting the potential of our approach in building more user-oriented mobile LLM agents. Our code is open-source at \url{https://github.com/tsinghua-fib-lab/FingerTip-20K} for reproducibility.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces FingerTip 20K, a novel benchmark designed to evaluate the proactive and personalized capabilities of mobile GUI-control LLM agents. The contributions of the paper are clear — it presents 21,437 episodes collected from 95 real users using their own Android phones in daily life, covering 506 apps. This dataset is highly valuable for research on user-centric GUI agents.

### Strengths
1. Real-World, Longitudinal Data. The dataset is collected from real users on their own Android phones (rather than simulators), which represents the most prominent contribution of this work.

2. Novel Benchmark. The paper introduces two clearly defined new tracks — proactive task suggestion and personalized task execution, both of which address important challenges in GUI-based applications.

3. Experimental Evaluation. The benchmark evaluates a wide range of models, including GPT-4.1, Qwen-VL, DeepSeek, CogAgent, UI-TARS, and others.

### Weaknesses
1. All data contributors are from mainland China and use Chinese apps, which raises concerns about cross-cultural generalization and UI diversity.

2. Only 1,000 episodes were used to fine-tune a single 7B model with LoRA (rank 4), making the experimental validation somewhat narrow.

3. The model’s performance across both tracks remains generally poor, and fine-tuning does not effectively address this issue. Moreover, the paper lacks further analysis to explain these results.

### Questions
1. It is unclear why the personalized track is evaluated only in the online setting. If users employ different mobile systems, system-level inconsistencies may arise during online testing. More importantly, the online evaluation currently lacks an objective and standardized assessment mechanism, making reproducibility and fairness difficult to ensure.
2. The fine-tuned Qwen2.5 model still shows relatively low performance, yet the paper does not provide sufficient in-depth analysis to explain the possible reasons behind this result.
3. The connection between the two proposed tracks appears weak. The authors are encouraged to further justify the necessity of defining two separate tracks, as the current design may appear more like an engineering separation rather than a conceptual one.
4. It is unclear whether the authors have evaluated the quality of the collected dataset. Since the data come from 95 real users, real-world user interactions may include redundant or noisy operations. This raises the question of whether low data quality contributes to the generally poor agent performance observed in the experiments.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces FingerTip 20K, a new benchmark for evaluating proactive task suggestion and personalized task execution in mobile LLM agents. The dataset consists of 21,437 real-world episodes gathered from 506 apps on 95 users’ daily Android device usage, with rich contextual meta annotations. The paper describes two evaluation tracks: (i) proactive intent suggestion drawing on multi-source user context, and (ii) personalized task completion that aligns with users’ habitual action trajectories. Extensive experiments with generalist and specialized models are presented, showing current systems’ limitations in leveraging user context, as well as initial improvements via fine-tuning on the new dataset.

### Strengths
1.	Real-World Data: Collected from long-term, in-the-wild mobile users, the dataset captures authentic intents and behaviors, offering far higher ecological validity than synthetic or simulator-based settings.
2.	Comprehensive Context Annotation: Each episode includes detailed user profiles, interaction metadata, and historical action, enabling advanced personalization and preference modeling.
3.	Rigorous Task & Metric Design: Both tracks are clearly formalized with well-defined metrics and baselines, providing a strong foundation for systematic evaluation.
4.	Open Access Commitment: Public release of data and code ensures reproducibility and promotes further research.

### Weaknesses
1.	Limited Demographic Scope: All data originate from Chinese Android users, restricting global generalizability and the benchmark may including implicit bias.
2.	Missing Related Work: The paper does not compare with several key studies on GUI benchmarks (e.g., AndroidWorld) and proactive LLM-based agents or personalization systems (e.g., AutoDroid, AppAgent).
3.	Superficial Ethics Discussion: Privacy and anonymization issues are acknowledged but insufficiently analyzed.

### Questions
1.	How do the authors envision adapting the benchmark and associated models to users or apps from non-Chinese regions? Are there plans for extending to other languages and UI ecosystems, and if so, how might the dataset, metrics, or methodology need to change?
2.	Are there specific plans for redacting or obfuscating sensitive visual/acoustic/UI data before public release? What concrete privacy-preserving steps are being implemented that go beyond user consent and manual filtering?
3.	Could the authors provide more detailed qualitative or quantitative analyses of how Sim_2 correlates with perceived personalization by real users? Are there user studies or human preference evaluations to complement the action-sequence similarity metrics, especially for subjective/latent aspects of personalization?

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
The paper presents a new benchmark to evaluate the proactive and personalized mobile llm agents, which is rarely explored in the literature. The new benchmark contains 20K unique human demonstrations of multi-step Android device interactions across a variety of everyday apps. Based on the proposed benchmark, a set of algorithms have been evaluated which can serve as the baselines for future research work. Also, a finetuned version based on the collected data can obviously improve the baselines.

### Strengths
* The new benchmark targets the twos problems of proactive task suggestion and personalized task execution, which are import to agent applications.

* The proposed benchmark is well designed with sufficient diversity covering many real-world applications.

* The paper reports the results of a set of algorithms which can be used as baselines for future research. Also, the proposed evaluation metric seems to be reasonable to correspond the human behaviors.

### Weaknesses
* The presentation of the paper should be improved. For example, the figures and charts in the paper can be replaced with vector image, which can provide better visual quality.

* For the baseline evaluations in Table 3 and Table 4, it would be better to include more vlms for a more complete evaluation. For example, how about the performance with more parameters like 72B vs 7B? Also, how about the performance with thinking model compared with non-thinking model?

* The user preference may be subjective and may vary depends on different situations. When the 20K data has been collected, is it possible to provide a probability setting for the human actions. 

* In the paper, it claims there are 95 data collectors. How about the distribution of these collectors? Is it able to generalize the data collected in these users to other users? The generalization issue should be well considered otherwised the value of the benchmark may be challenged.

### Questions
Please mainly address the questions in the weakness section. More specifically, the main concern is on the limited baselines of the proposed benchmarks as well as the generalization ability based on the benchmark to other similar agent tasks.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces FingerTip 20K, a mobile-agent benchmark built from real, in-the-wild phone usage: 95 users, ≈2 months, 21,437 episodes, 506 apps, avg. 11 steps per episode. Every episode is paired with user metadata (profile, time, place/context) and the user’s self-declared intent plus the full interaction trace (screens + a11y tree + actions). On top of this, the authors define two evaluation tracks that current GUI benchmarks don’t cover: (i) proactive task suggestion (given user + context + recent history, predict what the user likely wants to do now) and (ii) personalized task execution (given the task and this user’s past executions, complete it in this user’s style). Baselines with GPT-4-class models are far from human (≈7% vs 30% in proactive), showing the tasks are nontrivial.

### Strengths
- Truly real data. Unlike existing benchmarks that rely on emulator or auto-exploration, this is real phones + real users + real daily intents, so the distribution shift is authentic. 
- New task formulations. Proactive recommendation and “execute like this user” are exactly what current mobile agents lack; existing benchmarks mostly test “can you follow a given instruction.”
- Rich context signals. Time, location category, user profile, and multi-intent history make it suitable for modeling preference and routine.
- Clear difficulty evidence. Even strong LLMs underperform humans → good headroom for the community.

### Weaknesses
- Privacy / deployment gap. Real-world agents won’t always have such clean, explicit user-intent annotations; some discussion of weaker supervision would help.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
4
