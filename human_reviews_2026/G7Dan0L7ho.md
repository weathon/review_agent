# How Dark Patterns Manipulate Web Agents

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Deceptive UI designs, widely instantiated across the web and commonly known
as dark patterns, manipulate users into performing actions misaligned with their
goals. In this paper, we show that dark patterns are highly effective in steer-
ing agent trajectories, posing a significant risk to agent robustness. To quan-
tify this risk, we introduce DECEPTICON, an environment for testing individual
dark patterns in isolation. DECEPTICON includes 700 web navigation tasks with
dark patterns—600 generated tasks and 100 real-world tasks, designed to measure
instruction-following success and dark pattern effectiveness. Across state-of-the-
art agents, we find dark patterns successfully steer agent trajectories towards mali-
cious outcomes in over 70% of tested generated and real-world tasks—compared
to a human average of 31%. Moreover, we find that dark pattern effectiveness
correlates positively with model size and test-time reasoning, making larger, more
capable models more susceptible. Leading countermeasures against adversarial
attacks, including in-context prompting and guardrail models, fail to consistently
reduce the success rate of dark pattern interventions. Our findings reveal dark pat-
terns as a latent and unmitigated risk to web agents, highlighting the urgent need
for robust defenses against manipulative designs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies the threat of dark patterns to web-browsing agents. The authors constructed Decepticon, a dark pattern evaluation framework with generated and real-world tasks. The authors conducted experiments to verify the effectiveness of most dark patterns, and gained further insights into what affects the effectiveness of dark patterns. Finally, the paper tests two defence mechanisms and found that they are only partly effective.

### Strengths
-	The paper formally introduces the threat dark patterns pose to web-browsing agents. While prior work has explored specific attacks such as pop-up window attacks, dark patterns represent a more generalized and widespread threat, making this contribution both novel and important.
-	The evaluation demonstrates comprehensiveness in covering multiple types of dark patterns across different categories, providing broad coverage of the threat landscape. 
-	The paper provides valuable insights into attack mechanisms and effectiveness, as well as identifying limitations in simple defense strategies.

### Weaknesses
-	Regarding the high effectiveness of dark patterns, the experimental design cannot rule out a confounding possibility: the observed high effectiveness may be partially attributed to issues with SoM annotation. If SoM labels/transcripts are inaccurate, or if the SoM action space is inappropriate, agents may be unable to detect or avoid dark patterns. The paper lacks ablation studies on the same agent to clarify this potential confound.
-	The analysis of failure modes could be more thorough. For example, what specific reasoning patterns lead agents to fail for dark patterns? How do agnets’ CoT traces behave in cases where the defences fail?
-	The presentation of Figure 2 is non-academic. The horizontal axis is qualitative and represents different meanings for different models. The authors should revise this figure to clearly convey the meaning of each data point (e.g., through separate panels or explicit labelling, or by other means). 

Minor issue

-	In the introduction: “Although many users…” (near line 52), this sentence needs citations to support.

### Questions
-	Regarding Figure 5 (in the appendix): Why would dark patterns manipulate Terms and Conditions of Use? For such dark patterns, what is the success rate of human users in avoiding them? 
-	Why is there no evaluation with Computer Use Agents, such as OpenAI CUA? Are there technical difficulties (such as action space misalignment) that prevented this evaluation?
-	Could you explain lines 2 and 4 of Algorithm 1? For example, what is the prompt behind `LLM.generate_task()`? Could you provide an example of what the `trajectory` variable contains?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Relative to existing web-agent benchmarks (e.g., WebArena, VisualWebArena, WorkArena, BrowseComp) and agent attack studies (e.g., pop-up attacks, environmental injection, jailbreaks), the paper positions DECEPTICON as a dark-pattern–specific benchmark that isolates human-targeted manipulative UI designs. Its key differentiators are a taxonomy-driven task set, matched treatment-control pages for causal attribution on generated tasks, archived in-the-wild pages for realism, explicit “avoidability” constraints, and dual outcome metrics for task success versus dark-pattern effectiveness. The paper also adds systematic analyses of scaling and defenses under this threat model.

### Strengths
Problem and threat-model focus: Unlike general web task suites (WebArena; VisualWebArena; WorkArena; BrowseComp), DECEPTICON centers exclusively on human-targeted dark patterns rather than generic navigation. This fills a gap distinguished from agent-optimized attacks like pop-up prompt-injection or environmental injection that manipulate agents directly rather than leveraging deceptive design intended for humans.

Separating task success (SR) from dark-pattern effectiveness (DP) highlights that agents can both “succeed” at the user goal and still be manipulated. This disentanglement is not emphasized in the cited previous benchmarks.

### Weaknesses
Positioning against closely related attacks could be tighter. While the paper distinguishes dark patterns from agent-targeted injections (e.g., pop-ups; environmental injection) in Related Work, some instantiations (e.g., coercive pop-ups) straddle both spaces. A crisper delineation of what DECEPTICON includes/excludes versus those settings would sharpen the novelty.

The manuscript does not present transfer or overlap results on existing suites (e.g., taking WebArena/VisualWebArena tasks that contain pop-ups or consent flows) or quantify how DECEPTICON’s tasks differ in difficulty or failure modes compared to those environments.

The Related Work cites a contemporaneous study on dark patterns and GUI agents. The manuscript claims unique emphasis but does not explicitly articulate dataset, metric, and protocol differences (e.g., treatment-control design, avoidability constraint, detector assumptions) to separate contributions.

### Questions
Please add a concise table contrasting DECEPTICON with WebArena, VisualWebArena, WorkArena, BrowseComp, and the cited pop-up/environmental-injection attack settings along these axes: threat model (human-targeted vs. agent-optimized), control pages, archival determinism, avoidability constraint, category taxonomy, metrics (SR vs. DP), and defense/scaling evaluations.

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
In this study, the authors propose DECEPTICON, a benchmark designed to evaluate the behavior-altering effects of dark patterns on web navigation agents.
DECEPTICON includes 850 tasks that incorporate various dark patterns such as obstruction and social proof, mimicking manipulative elements that may appear on real web pages.
The authors investigate whether these dark patterns can manipulate the behavior of agents, and find that models with stronger reasoning abilities are, more susceptible to manipulation.
They also evaluate the defensive effectiveness of In-Context Prompting and Multi-agent Verification within this benchmark.

### Strengths
- This paper introduces a dark pattern benchmark, proposing that various patterns such as Social Proof and Urgency, which commonly appear on real websites, can potentially influence or control agent behavior.

- The fact that these dark patterns can be easily created by real web hosts demonstrates that this benchmark effectively simulates situations web AI agents are likely to encounter in the real world.

- The authors also evaluate the defensive capabilities of In-Context Prompting and Multi-agent Verification on this benchmark.

- The finding that models with stronger reasoning abilities are actually more susceptible to manipulation is particularly intriguing.

### Weaknesses
- The study focuses only on single-step deception, without addressing multi-turn manipulations.

- The automated labeling process may contain some errors, and the criteria for defining vulnerabilities could be ambiguous.

### Questions
- The authors define six categories based on the seven-category structure from Mathur et al. (2019). Are there examples of dark patterns not covered by these categories?

- Could webpage archiving cause the agent to perceive the environment differently compared to a live webpage? For instance, is the agent restricted from viewing or accessing the URL (if they can see url, can they realize that this is a mock-up webpage)?

### Soundness
3

### Presentation
3

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
This paper investigates the robustness of web agents against deceptive UI designs, also known as dark patterns, which target human users to alter their behavior for adversarial goals. The authors construct DECEPTICON, a novel benchmark consisting of 250 real-world tasks and 600 synthetic tasks that evaluate the dark pattern risk in the web search agent applications. The experiments on frontier LLMs and different agent scaffolds show that dark patterns succeed in majority of tasks and the risk grows with the capability of underlying LLMs, highlighting the significance of examining dark pattern risks during the web agent development process.

### Strengths
- construct an effective, isolated environment for studying dark patterns, which is both deterministic and repeatable to enable controlled experimentation. 
- the novel benchmark DECEPTICON covers diverse categories and common cases of dark patterns we can see on the internet.
- use realistic dark patterns that target human users, focusing on attacks that exploit human cognitive biases and decision-making processes.

### Weaknesses
- don't have a human user baseline
- no evaluation on open-sourced LLMs-driven web agents
- the standard deviation in Table 2 is large, especially on the generated evaluation set. Please add discussions about this observation in the text.

### Questions
no other questions

### Soundness
3

### Presentation
4

### Contribution
3
