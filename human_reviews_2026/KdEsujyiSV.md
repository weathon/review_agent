# PolySkill: Learning Generalizable Skills Through Polymorphic Abstraction For Continual Learning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
Large language models (LLMs) are moving beyond static uses and are now powering agents that learn during their interaction with external environments. For example, agents can learn reusable skills while navigating web pages or toggling new tools. However, existing methods for skill learning often create skills that are over-specialized to a single website and fail to generalize.
We introduce PolySkill, a new framework that enables agents to learn generalizable and compositional skills. The core idea, inspired by polymorphism in software engineering, is to decouple a skill's abstract goal (*what* it accomplishes) and its concrete implementation (*how* it is executed). Experiments show that
our method (1) improves skill reuse by 1.7x on seen websites and (2) boosts success rates by up to 9.4\% on Mind2Web and 13.9\% on unseen websites, while reducing steps by over 20\%. (3) In self-exploration settings without specified tasks, our framework improves the quality of proposed tasks and enables agents to learn generalizable skills that work across different sites. 
By enabling the agent to identify and refine its own goals, the PolySkill enhance the agent a better curriculum, leading to the acquisition of more generalizable skills compared to baseline methods. Our findings show that separating a skill's goal from its execution is a crucial step toward developing autonomous agents that can learn and generalize across the open web continuously. Our code can be found in \href{https://github.com/simonucl/PolySkill}{\texttt{https://github.com/simonucl/PolySkill}}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a method to induce skills that transfer across websites, by decoupling their abstract goals and concrete implementations. Both experiments on static benchmarks and free-form continual learning settings show the empirical benefits of the proposed method.

### Strengths
**1. Importance of Skill Transferrability.**
> This paper tackles a critical limitation of current agent skill learning studies – transferability across websites.

**2. Empirical Improvements throughout Experiments.** 
> The paper evaluates on both (i) static benchmarks and (ii) open-ended exploration settings. The proposed method shows improvements in success rate, efficiency, and skill reusability for (i); as well as greater task coverage and skill compositionality for (ii).

### Weaknesses
**1. Writing Clarity Issue: confusing concepts of “websites” and “domains”.**
> Different websites do not necessarily associate with in-domain and out-of-domain. Two websites can either belong to the same domain (e.g., both United and Spirit Airlines are for traveling) or not (e.g., Amazon for shopping and United for traveling). That being said, this paper aims to induce skills that transfer across websites, but it is unclear (at least from the intro) if these are websites within the same domain or across different domains.
Further, from the experiments, the proposed method only works for websites in-domain, so claiming its effectiveness in out-of-domain scenarios lacks evidence.

**2. Missing Skill Count Measures.** 
> While the success rate and efficiency gains are promising, the paper fails to report the number of skills (abstract class and specialized implementations). Ensuring a small number of skills is important as it avoids redundancy in the skill library. The very low number of steps introduces concerns that there may be one skill unnecessarily targeting each specific example, instead of trying to capture shared similarities across examples. Reporting this dimension, similar to Table 3 in [1,] would help address this concern.

**3. Minor**
> but better remove the vspace before the “Conclusion” section title.

Once these missing aspects are added to the revised version, I am happy to increase the scores.

[1] Wang, Zhiruo, Daniel Fried, and Graham Neubig. "Trove: Inducing verifiable and efficient toolboxes for solving programmatic tasks." arXiv preprint arXiv:2401.12869 (2024).

### Questions
1. How different are the skill implementations within the same abstract class? Are they mainly different due to the varied designs of targeting websites, or other reasons? How similar (quantitatively) do two skill implementations need to be, in order to sit in the same abstract skill class?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a framework in which an LLM agent learns skills that can be reused across websites (e.g., "search", "add_to_cart", "checkout"). At the core of this framework is the separation of "what" from "how" (e.g., the framework defines an abstract shopping website class that is specialized to different actual shopping websites). Experiments compare the performance of the proposed system with prior work on existing benchmarks.

### Strengths
- The paper clearly positions the proposed apprpoach relative to prior work (ASI), making it easy to understand the technical contribution.
- The paper cleverly uses polymorphism, a well-understood concept from classic software engineering principles, to improve generalizability in LLM agents (although I am not an expert in this area, and thus I cannot be certain of my assessment of novelty).

### Weaknesses
- The abstract class that specifies the relevant skills is provided as context. This is a very restrictive assumption, as it assumes (1) that the set of useful skills is fixed, (2) that the set of useful skills is known.
- I found the current description to be insufficient to understand the method, and had to consult the appendix. Please consider improving the framework's pseudocode to the main text, or at least including a precise step-by-step description in the main text.
- It seems that none of the baseline systems were given the extra information corresponding to the abstract class description (or this was at least not described), which makes the comparison perhaps biased. Giving this information to at least some of the systems should be possible (perhaps simply by including it in the prompt).
- Some parts of the methodology are underdocumented (see questions 1, 2, 3, 5).

### Questions
1. Can the abstract class definition be inferred from the initial successful traces?
2. Can the system accidentally use a not-yet-implemented skill in its implementation of a skill?
3. How are the "+Update" systems implemented?
4. Please consider running the experiments providing the information corresponding to the abstract class to the baseline systems that allow it. This would further allow us to understand if the performance benefits come simply from the apriori identification of relevant skills or actually from the polymorphic skills.
5. From the manuscript, I am guessing the +Update and +Online systems modify their implementation of the concrete classes. Is this corrrect? Or do these systems also update the weights of the LLMs? Please include a description of these systems in the main text.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes PolySkill, a polymorphism-guided framework for skill induction in web agents. A skill is defined by an abstract interface capturing semantic intent, linked to multiple interchangeable concrete implementations across websites. This separation aims to improve transfer, modularity, and recomposition of skills, and to support continual learning and adaptive behavior in task-free settings. The authors also introduce process-level metrics—Skill Reusability, Task Coverage, and Skill Compositionality—to assess what the agent actually learns beyond task success.

### Strengths
1. Skills are specified by an abstract interface plus interchangeable concrete implementations, enabling cross-site reuse and composition while insulating from UI changes. 
2. A three-stage flow consisting of polymorphic abstraction, compositional verification, and adaptive execution clarifies how skills are discovered, validated, and deployed.

### Weaknesses
1. It’s unclear how much gain comes from the interface abstraction vs. better verification, task curricula, retrieval, or engineering heuristics.
2. Most compelling examples are shopping; results on other domains (dev tools) are thinner and may conflate prior knowledge with schema effects.

### Questions
1. How is a domain’s abstract class discovered, manually seeded, or fully induced? What prevents over-abstracting or under-abstracting? 
2. How is the $\gamma$ penalty on steps set; is it tuned per model/domain; sensitivity?
3. What is the compute & API cost of induction/verification vs. Baselines?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a method that improves the reusability of skill induction methods for agents by encouraging the agent to create abstractable skills using polymorphism.

### Strengths
1. The method is simple but strikes at the core of the problem.
2. The motivation and description is very clear, illustrated by appropriate examples.
3. The paper compares across multiple datasets against strong baselines from the recent literature.

### Weaknesses
1. As mentioned in the future work section, it is less clear how this method will be applicable in cases where task category boundaries are more fuzzy. The hard-coded task boundaries required to induce new abstract classes may make this tricky. I don't view this as a critical weakness of the paper though, more of an opportunity for future work.

### Questions
One thing that was not very clear to me (sorry if I missed this) -- is the abstract class only created once when the web site is first processed? Or can it be modified later as more evidence about the web site comes to light?

### Soundness
4

### Presentation
4

### Contribution
3
