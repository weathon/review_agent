# HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatrics

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Creating an immersive and interactive theatrical experience is a long-term goal in the field of interactive narrative. The emergence of large language models (LLMs) provides a new path to achieve this goal. However, existing LLM-based drama generation methods often produce models that lack initiative and cannot interact with the physical scene, while typically requiring detailed user input that diminishes the immersion of live performance. To address these challenges, we propose HAMLET, a hierarchical adaptive multi-agent framework focused on drama creation and real-time online performance. Given a simple topic, the framework first generates a narrative blueprint to guide the subsequent improvisational performance. In the online performance phase, each actor is equipped with an adaptive reasoning module that enables decision-making based on their personas, memories, goals, and emotional states during complex group chat scenarios. Beyond dialogue, actor agents engage in embodied interactions by changing the state of scene props through actions such as opening a letter or picking up a weapon, which are broadcast to update the global environmental context. To objectively assess the quality of live embodied theatrics, we establish a comprehensive evaluation method and introduce HAMLETJudge, a specialized critic model for automated evaluation. Experimental results demonstrate that HAMLET excels in creating expressive, coherent, and physically interactive theatrical experiences in an autonomous manner.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces HAMLET, a multi-agent framework for AI-driven theatrical performance that integrates both offline narrative planning and online embodied improvisation. A key component, Perceive-and-Decide (PAD), models dual-process cognition to make agent behavior more human-like. The paper also proposes HAMLETJudge, an evaluation model trained for cost-efficient pairwise scoring on three axes: Character Performance, Narrative Quality, and Interaction Experience. Extensive experiments and ablations demonstrate the framework’s capability for coherent, expressive AI drama.

### Strengths
1. Novel research direction: This paper explores live embodied drama generation — a rarely studied yet promising extension of LLM-based storytelling and agentic interaction.
2. Well-structured system design: The clear decoupling between offline planning and online performance is conceptually elegant and parallels real theatrical workflows.
3. Cognitive realism via PAD: Modeling System-I/II reasoning to control LLM agents is original and improves interpretability compared to black-box prompting.

### Weaknesses
1. Human Evaluation: The experiment primarily relies on comparisons between models and automated scoring methods, and it lacks studies that consider human subjective experiences or real-world interaction experiments to strengthen claims of interactivity and expressiveness.
2. Clarity of implementation details: Many modules (e.g., Transfer, Advancer, Planner) are described abstractly. It remains unclear how these controllers are implemented and how they synchronize multi-agent turns without temporal inconsistency.
3. PAD validation scope: PAD’s decision-making improvement is shown mostly in synthetic latency and strategy metrics; no direct human evaluation verifies whether its behavior truly feels “human-like”.

### Questions
1. How is temporal synchronization achieved among multiple agents during online performance (e.g., preventing overlapping actions)?
2. How does the system handle deviation recovery when an actor diverges significantly from the narrative blueprint?
3. Why is the Latency Penalty defined as a discrete scalar (0 / 0.05 / 0.10 / 0.15) rather than a continuous function?
4. What are the specific total time costs of running HAMLET with multiple agents in real time?

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
3

### Summary
This paper introduces HAMLET, a multi-agent framework for generating and performing real-time and interactive theatre. The system uses an Offline Planning stage where AI agents create a structured "narrative blueprint," and an Online Performance stage where AI actors (and humans) execute it. A key innovation is the PAD module, which allows agents to autonomously choose response strategies, enabling proactive and embodied interactions. The paper also presents a comprehensive evaluation method, including a new critic model called HAMLETJudge.

### Strengths
The two-stage HAMLET framework is a good contribution to the field of interactive narrative. Its system role design is comprehensive and reasonable, balancing complexity and performance. The PAD Module extends Kahneman's dual-process theory to a concrete action space and achieves excellent performance, providing a good reference and insight for future work. The authors further propose a robust evaluation methodology and a well-trained critic model, HAMLETJudge, which also provide valuable assets for the community. The discussions and analyses in the experiments and appendix are thorough and convincing, and the training and prompt details provide good reproducibility. I believe this is a very strong piece of work.

### Weaknesses
1. Scalability Issues: HAMLET adds verification (Narrator) or planning (PAD LLMs) after each actor's action. In long-form, multi-actor scenarios, this will lead to an unacceptable increase in LLM calls and context content expansion. Meanwhile, although the paper seems to mention a Memory module (Figure 4), it is not clearly described. Most of the memory likely exists as context. As the number of actors expands, the exponentially growing actor actions and system information will quickly exceed the model's context length. This will severely limit HAMLET's scalability, running counter to the "Live" theatrical experience the framework aims for.
2. Subjectivity of Evaluation: HAMLETJudge is trained on the preferences of five experts, which risks embedding a specific taste in drama into the critic model. A more detailed analysis of the inter-annotator agreement (IAA) during the dataset's creation to quantify the level of subjectivity would provide a useful reference.
3. Insufficient analysis and evaluation of the Offline Planning stage. HAMLET's online performance entirely follows the blueprint generated in the offline planning stage. Therefore, the final performance quality heavily depends on the offline planning quality, but the paper seems to lack analysis of this stage, focusing more on the online performance part. The examples provided use well-known scenarios like Hamlet. I would like to know the robustness of the offline planning for completely new or abstract topics. More evaluation of this stage's quality would strengthen this paper.
4. Typos: There are many grammatical errors. I will list a few. I suggest using a grammar-checking tool to correct them. 
    1. (Page 3, line 136): "new criterias" -> "new criteria" (criteria is plural). 
    2. (Page 4, line 220): "$beat_{k_{1}}, beat_{k_{2}},...$" seems incomplete and should be completed. 
    3. (Page 5, line 255): "a few sequence" -> "a few sequences". 
    4. (Page 7, line 370): the citation for "Appendix B" is missing a hyperlink. 
    5. (More ...) 
5. Comparison to prior works are not sufficient. There are other multi-agent drama generation works, e.g. "IBSEN: Director–Actor Agent Collaboration for Controllable and Interactive Drama Script Generation". The authors discuss the differences compared to other similar approaches in the paper.

If my questions are resolved, I will consider raising the score.

### Questions
1. About the memory module: Is any memory module currently incorporated into HAMLET's design? Figure 4 (regarding PAD) seems to mention it, but I am unsure if this is just an abstract concept.
2. About the human actor: It seems that the progression of drama in HAMLET is strictly controlled by the PAD and other control modules. Does this create a phenomenon of "over-control," especially for human actors? Cases 2-4 mainly consider humans acting as simple disruptors. However, in practice, a creative human might attempt to "out-smart" the system and drive the plot in an unexpected (but logical) direction. I would like to know if you have considered this concern and if there are any potential solutions.
3. About Backward Planning: I would like to know how you generate a reasonable end-point at the beginning of planning if a topic is open-ended. Have you compared the quality of blueprints generated by sequential planning versus backward planning? I want to know if backward planning has a significant performance advantage.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies LLM-based drama, particularly two aspects: offline story generation and online drama performing. Multi-agent framework is used to enhance the system performance. Specially, the authors propose a Perceive And Decide (PAD) module to make the character responses akin to humans.

### Strengths
1. Undoubtedly, the paper proposes a carefully-designed multi-agent system to achieve the goal. For example, the offline planning includes four agents: an actor designer, plot designer, reviewer, and director, which makes sense to readers. I believe these designs are refined based on previous work and lead to improvement.

2. The evaluation (HAMLET leaderboard) is comprehensive. It is excited that the authors have covered a large number of leading LLMs (both open-source and close-source). The experiments also consider both English and Chinese.

3. The authors fine-tune a judge model for automatic evaluation. The model is fine-tuned on annotated data. To me, this method is more reliable compared to prompting strong LLMs.

### Weaknesses
1. **The effectiveness of multiple agents is not empirically validated.** The multi-agent workflow proposed in the paper are complicated (normally four agents). While the complexity is not a disadvantage, the paper doesn't include experiments to isolate and verify the role and effectiveness of each agent. For example, how does the reviewer agent work, does it improve the quality of the generated narrative? Without such ablation, the soundness of the multi-agent design are hard to assess.

2. **The performance of offline planning is unclear.** The experiments assess the overall dramatic performance of one LLM-based drama (if I was wrong please correct me). These numbers can be affected by so many factors, including the quality of the narrative blueprint, which is generated by offline narrative planning. There must be a causal relationship between offline planning and the eventual performance. However, this relationship is subtle and not clear so that the evaluation cannot be post-hoc. This makes hard to judge whether the reported performance truly reflects the quality of the offline narrative planning.

3. Indeed, after reading the paper, I am still confused on what an LLM-based drama exactly looks like. Is it like an animation or a text-based game? While the experiments show the effectiveness of the methods, **there is no demonstration or qualitative examples provided.** It is hard for readers to assess the real playing experience of the work.

4. **The evaluation dimensions are too rough.** I can't believe "character performance, narrative quality, and interactive experience" are informative enough to show the performance of an LLM-based drama. There are many factors that contribute to a nice dramatic experience, for example, scenery, narrative logics, and sense of suspense. I can't find clues in the paper to show me these things.

5. **The acting efficiency issue is not discussed in the paper.** To me, this is an important issue valued detailed discussion. For example, how is the latency of the online performing? Since there are four agents working collaboratively, does it meet the practical requirement of users, does the latency ensure a nice user experience?

6. **The experiments can be more insightful.** From Table 1, the reasoning LLMs perform better than non-reasoning ones. It will be interesting to discuss more on this issue to provide more insights . Additionally, it will provide further insights if the authors can discuss the scaling performance of the method, like when the backbone LLM increases in size, how the performance changes as a result.

### Questions
Covered in the last section.

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
3

### Summary
The paper presents HAMLET, a multi-agent framework for autonomous and interactive AI-driven drama generation. It combines offline narrative planning and online embodied performance to address the passivity and lack of interactivity in previous LLM-based systems. In HAMLET, multiple agents collaboratively generate a narrative blueprint, while during performance, actors make independent decisions through a Perceive and Decide (PAD) module inspired by human dual-process cognition. A Narrator Agent manages physical interactions, ensuring consistency between dialogue and environment. The authors also introduce HAMLETJudge, an automatic evaluation model, and a 100-case benchmark measuring character performance, narrative quality, and interaction experience. Experiments show that HAMLET with PAD improves coherence, expressiveness, and alignment with human judgment, offering a comprehensive yet text-based step toward truly embodied AI theatre.

### Strengths
1. Strong empirical validation: Experiments include both quantitative leaderboards and ablation studies, showing PAD’s contribution to coherence, latency reduction, and human alignment (Pearson ≈ 0.79).
2. Cross-lingual and creative scope: The inclusion of both English and Chinese cases, as well as diverse drama topics (from Shakespeare to pop culture), demonstrates the framework’s flexibility and generalization capacity in creative settings.
3. Embodied interactivity: Introducing the Narrator Agent enables coherent physical interactions between agents and the environment, moving beyond text-only dialogue and toward “live” dramatization.

### Weaknesses
1. Limited generalization evidence: Most experiments are performed under controlled conditions with GPT-4o as the main backbone. It remains unclear how the framework performs with smaller or open-source models in unconstrained or noisy environments.
2. Absence of creative or narrative diversity analysis: While the system produces coherent plays, the paper does not examine whether HAMLET truly enhances creative originality or thematic richness beyond structural coherence.

### Questions
1. Beyond coherence and emotion, can HAMLET enhance genuine creativity or narrative diversity compared to traditional story-generation models?
2. What are the latency and computational costs of running full real-time performances with multiple autonomous agents?

### Soundness
2

### Presentation
3

### Contribution
2
