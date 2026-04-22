# Beyond Needle(s) in the Embodied Haystack: Environment, Architecture, and Training Considerations for Long Context Reasoning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
We introduce $\infty$-THOR, a new framework for long-horizon embodied tasks that advances long-context understanding in embodied AI.
$\infty$-THOR provides:
(1) a generation framework for synthesizing scalable, reproducible, and unlimited long-horizon trajectories;
(2) a novel embodied QA task, Needle(s) in the Embodied Haystack, where multiple scattered clues across extended trajectories test agents’ long-context reasoning ability; and
(3) a long-horizon dataset and benchmark suite featuring complex tasks that span hundreds of environment steps, each paired with ground-truth action sequences.
To enable this capability, we explore architectural adaptations, including interleaved Goal-State-Action modeling, context extension techniques, and Context Parallelism, to equip LLM-based agents for extreme long-context reasoning and interaction.
Experimental results and analyses highlight the challenges posed by our benchmark and provide insights into training strategies and model behaviors under long-horizon conditions.
Our work provides a foundation for the next generation of embodied AI systems capable of robust, long-term reasoning and planning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a new framework for generating, training, and evaluating long-horizon embodied reasoning tasks. It introduces the Needle(s) in the Embodied Haystack (NiEH) benchmark, which tests agents’ ability to recall and reason over multiple scattered clues across hundreds of environment steps. The framework supports scalable trajectory synthesis using AI2-THOR and integrates architectural techniques—such as interleaved Goal-State-Action modeling, context extension, and Context Parallelism, to enhance long-context reasoning in embodied agents. Extensive experiments reveal the severe performance degradation (“memory cliff”) that occurs as context length increases and analyze how different extension methods mitigate this effect.

### Strengths
1. Novel benchmark and task design: NiEH introduces an original embodied QA setting requiring multi-clue, multi-step reasoning over long trajectories, a valuable contribution to evaluating memory and reasoning.
2. Scalable trajectory generation pipeline: The framework builds a reproducible, large-scale dataset based on AI2-THOR, enabling consistent generation of trajectories with hundreds of steps and millions of tokens.
3. Systematic empirical analysis: The experiments provide comprehensive comparisons of context-extension strategies and clearly expose the “memory cliff” phenomenon in existing models.
4. Technical depth: The architectural adaptations (Goal-State-Action modeling, Context Parallelism) are conceptually well-motivated and technically detailed.

### Weaknesses
1. Dataset construction methodology: The long trajectories appear to be concatenations of short demonstrations; it remains unclear how object states and interaction continuity are maintained. Why not synthesize single coherent long trajectories directly?
2. Benchmark novelty and scope: The NiEH benchmark closely resembles long-video understanding settings; the distinction between this work and prior long-video benchmarks should be made clearer beyond simply being “embodied.”
3. Limited model evaluation: Most experiments rely on off-the-shelf VLMs with minimal low-level VLA validation. This weakens claims about the framework’s utility for training or improving embodied agents.
4. Visualization clarity: Figure 4 heatmap presentation is visually rich but lacks clear quantitative comparisons, statistical significance, and explicit axis labeling. More intuitive ablations would improve interpretability.
5. Benchmark discrimination: The results highlight engineering aspects of long-context handling but do not clearly show whether the proposed benchmark can distinguish different models’ reasoning abilities in a consistent way.
6. Missing supplementary materials: Several details are deferred to the appendix, but no appendix appears in the submission, limiting reproducibility and clarity.

### Questions
The paper provides a valuable benchmark and framework for studying long-context reasoning in embodied AI, with strong engineering and analytic depth. However, the novelty relative to existing long-video settings, limited evaluation coverage, and missing appendices reduce its impact. Strengthening empirical breadth and clarifying benchmark construction would elevate it to a solid accept. Please refer to the weakness above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a method for generating long multi-goal trajectories in the AI2-THOR environment, along with question-answer pairs whose answers depend on either a single (NiaH) or multiple (NsiaH) past events. This setup requires long-term memory capabilities and enables the evaluation of vision–language question answering over extended sequences.
The generated data is used for the evaluation of VLA / VLM models in long context settings.

In addition, the paper discusses an online interaction VLA design / implementation for such long context settings.

The paper presents empirical evidence showing that several current VLMs struggle in long-context settings where the context length exceeds that used during training / FT.
Furthermore, in another set of experiments, the proposed online VLA design was fine-tuned on the proposed dataset under several configurations and the models were evaluated in an "interactive" setup.

### Strengths
- The long context and multi goal trajectory generation method and question-answer pairs generation method are novel and could contribute to the VLA/VLM community.
- The empirical results reflect the poor performance of current methods in long-context settings.

### Weaknesses
`W1`: Overall, the paper is not easy to follow, and the presentation lacks clarity in explaining what was actually done. Significant effort is required to understand precisely the main contributions and methodology. The writing would benefit from a more direct, transparent exposition of the key ideas and experimental details.




`W2`: Ultimately, the results in Figure 4 suggest that current architectures (incl. long context solutions) struggle with contexts longer than the training / FT context length. This observation is not particularly surprising.



`W3`: The experimental design of the interactive (online) experiment setup may be flawed. More information is needed to determine with certainty, see `Q5, Q6` below for questions. This could impact the validity of the corresponding conclusions.

My concern is that the context is reset after each sub-goal based on the data (states and actions) of the trajectory from the dataset (generated by the same planner used in the training set), rather than the states and actions produced through the online interaction. In such a case, it is possible that performance are maintained due to overfitting the training set (which was used for the fine-tuning). It is also unclear whether states and actions from previous tasks are relevant for the success on the current task.

Furthermore, such a setting is not a true online evaluation, but rather evaluates each task independently until first failure.



`W4`: line 269: the acronym "PDDL" was never introduced. Please also provide a reference.

`W5`: 
> "$\infty$-THOR enables the creation of unlimited trajectories ***with arbitrarily long***, and provides" (line 163)

This sentence is truncated?


`W6`: The paper claims
> We show that interleaved Goal-State-Action modeling ...  is the most practical approach for this class of problems (lines 77-78)

This claim is not supported by the evidence presented in the paper. To show that an approach is "the most practical", one must defined precisely what "practical" means and compare to all relevant existing baselines.


`W7`: The term "embodied AI" is much broader than language-vision based models, and includes non-language models as well (e.g., deep RL). The paper proposes a framework aimed specifically at language-vision driven methods. The scope of the discussion should be clearer and more accurate.

The claim 
> "We present empirical results and analyses, providing insights to the current capabilities and
limitations ***of embodied AI systems*** on long-horizon tasks." 

suggests a wider scope than what the paper includes. The scope of the claim should be adjusted accordingly.



`W8`:
> ... lack the dynamic interactivity and memory needed for long-horizon embodied tasks involving continuous vision-language-action sequence (line 303)

This statement lacks supporting evidence (either provide empirical evidence or refer to prior works that include such supporting evidence).



`W9`: 
> Moreover, many state-of-the-art models are only accessible via proprietary APIs, making them impractical for real-time, controllable embodied settings and managing long-term memory states. (line 305)

The fact that a model is proprietary does not mean that it lacks capabilities. It is unclear what argument this statement aims to support.


`W10`: The plots in Figure 5 are too small. In addition, information about the shaded area is missing. 


`W11`: Figure 4 is missing axes labels and a color bar. In addition, tick labels are too small.




`W12`: The setup in Section 5.1 is not clear enough. Are the models fixed? Were they fine-tuned? What data exactly was used for the evaluation? test set only? all dataset splits?


`W13`: 
> Our experiments demonstrate that exposure to longer contexts during training significantly improves model performance (line 482)

This observation is inline with existing literature and is not surprising or new.


`W14`: The literature review pertaining to the method discussed in Section 4 is insufficient. Specifically, regarding the "goal-state-action" design, the idea of concatenating tokens of various modalities along the temporal axis was studied in many prior works, see [1][2][3] for example.

This "goal-state-action" approach can not be considered as novel. 

Ultimately, in this context, the paper explores fine-tuning such methods to longer context lengths (e.g. 200K+), and shows that long-context performance improve with long-context specific training, which is not surprising.



[1] Reed, S., Zolna, K., Parisotto, E., Colmenarejo, S. G., Novikov, A., Barth-Maron, G., ... & De Freitas, N. (2022). A generalist agent. arXiv preprint arXiv:2205.06175.

[2] Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., ... & Finn, C. (2024). Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246.

[3] Driess, D., Xia, F., Sajjadi, M. S., Lynch, C., Chowdhery, A., Wahid, A., ... & Florence, P. (2023). Palm-e: An embodied multimodal language model.



`W15`: Based on the information in the paper alone, it is unclear how the online interactive evaluation is performed. The corresponding information from the appendix should be included in the main text, or at least a short description of it.

### Questions
`Q1`: The term “reasoning” has gained widespread use in recent years, yet there remains no formal definition or consensus on its meaning. What exactly do *you* mean by "reasoning"? Please be precise. This term was used extensively throughout the paper, and its meaning seems to differ based on the context.


`Q2`: 
> ... a new framework for generation, training, and evaluation of long-horizon embodied tasks (line 39)

"training / evaluation of [...] tasks", what does it mean to train a task? The paper does not describe training of tasks.

 
`Q3`: Why is the supplementary material in a separate file?


`Q4`: 
> We release a large-scale trajectory dataset and an interactive evaluation environment ... (line 96)

Where is this interactive evaluation environment described?


`Q5`: In the online interactive evaluation, it is stated that 
> After each plan, the context is reset to include the GT actions and states from the completed portion of the trajectory, ... (line 871)

what does "GT actions and states" mean in this context? pre-generated actions and states generated with the planner or the states and actions produced during the online interaction with the environment? are there actions and states that are not GT?

If the meaning of GT here is pre-generated planner actions, why do you use these? do you also reset the environment state accordingly? is there a discrepancy between the environment state and the context?

Also, in such a case, the model is effectively evaluated on each sub-goal independently, while terminating upon the first failure, as the context at each sub-task is being reset to a trajectory prefix generated by the (same) planner, used in the training set.


`Q6`: Is there a dependency between sub-goal/tasks? i.e., is it necessary to use information from previous tasks (through the states and actions in the context) to successfully perform the current task (goal)?


`Q7`: Regarding the trajectories generation process: How do you set up the initial environment state? how do you determine the configuration of the elements in the scene (placements, object types, number of objects, etc)? Given an initial setup, how many possible combinations for sub-goals are there? 

I am concerned about the overall diversity of the dataset and how it affects the results and conclusions. Based on the information in the paper alone, it is impossible to get a sense of the true diversity of the dataset (I assume that this is in large implied by the AI2-THOR environment). Can you provide further information in this regard?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new framework aimed at advancing research on long-horizon embodied tasks. The framework provides: (1) infrastructure for generating and evaluating arbitrarily long, reproducible agent trajectories; (2) a benchmark evaluating agents ability to reason over temporally distant multimodal cues; and (3) a dataset with tasks of hundreds of steps and ground-truth actions. The authors explore architectural approaches and training methods to allow agents to handle long contexts.

### Strengths
1. The motivation for this work is very solid and timely, as current models for embodied planning and decision making are struggling with long context, often confined to short terms tasks without the ability to perform long horizon optimization. Although the task of recalling details in long horizon action sequences is not directly aiming at the core of the planning problem, it also points at a capability in the right direction.

2. The interleaved Goal–State–Action modeling idea is interesting.

### Weaknesses
1. The model claims to explore "ARCHITECTURES FOR LONG-HORIZON VISION-LANGUAGE-ACTION MODELS". However, there is no explicit evaluation of the core task for VLAs: planning. Instead, authors only evaluate the model on the Needles in the Embodied Haystack task of long horizon question answering. 

2. The performance gains on this single task does not fully justify the complex architecture changes made. Perhaps one way to more concretely justify the modeling is by experimenting on other tasks more relevant to Embodied Agents, like actual task planning, or established embodiment benchmarks like VSI-Bench, EAI, etc.

3. Assuming that the architecture has its merits, I think this paper has the potential to be a good work, it just needs more time to test different hypothesis to get some more solid results.

### Questions
1. Are there any quantitative metrics for the Needles in the Embodied Haystack task, and why are they not reported in the paper?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper tackles long-horizon embodied reasoning agents remembering and acting on events hundreds of steps apart.
IntroducesTHOR, built on AI2-THOR, for generating ultra-long interactive tasks.
Adds Needle(s) in the Embodied Haystack (NiEH) QA tasks where clues are hidden across 600–900+ steps.
Proposes interleaved Goal–State–Action modeling combining vision, language, and action into one token sequence.
Extends transformer context with RoPE scaling, YaRN, and LongRoPE for 100k+ tokens.
Uses Context Parallelism via Ring Attention to train on massive sequences efficiently.
Implements a 7B LLaVA-based agent fine-tuned for long-term reasoning.
Findings: long-memory modeling becomes feasible, but coherence over hundreds of steps remains weak.
Overall: a strong framework and benchmark pushing embodied AI toward true long-term reasoning and planning.

### Strengths
Strong problem focus - long-horizon memory in embodied AI, timely and hard.
THOR is scalable and open  can synthesize endless, ultra-long trajectories with full action traces.
NiEH benchmark is unique tests recall of scattered clues across hundreds of steps, bridging vision, language, and long-term reasoning.
Task design is clever enforces early–late dependencies, no shortcuts.
Interleaved Goal–State–Action model clean, unified architecture, handles temporal context elegantly.
Rigorous experiments real ablations on RoPE scaling, YaRN, context length; solid quantitative insight.
Context Parallelism practical and technically hard; shows long-context training is possible, not just theoretical.
Findings are balanced improvement with long context, still breaks past 512k tokens; honest about limits.
Great clarity and visuals figures explain results at a glance.
Open release and reproducibility detailed setup, environment, data planned for release.
Broader impact connects symbolic planning to physical control; builds a base for next-gen long-term embodied agents.

### Weaknesses
Relies only on context extension for memory. No exploration of retrieval or hierarchical memory; limits scalability beyond 512k tokens.

No direct baseline against modular vision–language–action models. Claim of superiority for interleaved modeling not empirically proven.

Lacks external dynamics: all environment changes are agent-driven. No tests of memory for unobserved or changing scenes. Models fail beyond 0.5M tokens, multi-evidence QA accuracy drops sharply, and long runs often collapse. Needs clearer absolute metrics.

Statistical rigor missing. No variance, confidence intervals, or multiple seeds reported; unclear if differences are significant. 7B model fine-tuned on 130k tokens with 8×H100s; not practical for most labs. Inference cost not discussed. Planning and manipulation models evaluated separately; no unified control pipeline yet.

### Questions
How is the goal given during interaction? Is the final instruction known from the start and repeated each step, or revealed later? Does the agent get any subgoal hints along the way? Clarifying this would help interpret its planning and autonomy.

Did you try other memory idea like retrieval, recurrent state, or hierarchical summaries? Since all RoPE methods fail past 512k tokens, do you think future work should shift toward explicit or learned memory rather than just longer contexts?

Any comparison to a non-interleaved setup, e.g. separate vision + policy modules like ALFRED or PaLM-E? Even a small-scale test would show if the interleaved model truly helps.

What actually breaks beyond 0.5M tokens—GPU memory, instability, or degraded attention? Any thoughts on mixing retrieval or staged training to go further?

Multi-Evidence Reasoning: which question types fail most—temporal, counting, or spatial? Do models usually pick one wrong clue or combine clues incorrectly? A short error breakdown would be great.

If using a larger model (13B–70B), would memory or training cost be the main blocker? Do you expect bigger models to meaningfully extend reasoning, or still hit the same context wall?

### Soundness
2

### Presentation
3

### Contribution
2
