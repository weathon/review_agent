# LH-DECEPTION: Simulating and Understanding LLM Deceptive Behaviors in Long-Horizon Interactions

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Deception is a pervasive feature of human communication and an emerging concern in large language models (LLMs). While recent studies document instances of LLM deception, most evaluations remain confined to single-turn prompts and fail to capture the long-horizon interactions in which deceptive strategies typically unfold. We introduce a new simulation framework, LH-Deception, for a systematic, empirical quantification of deception in LLMs under extended sequences of interdependent tasks and dynamic contextual pressures. LH-Deception is designed as a multi-agent system: a performer agent tasked with completing tasks and a supervisor agent that evaluates progress, provides feedback, and maintains evolving states of trust. An independent deception auditor then reviews full trajectories to identify when and how deception occurs. We conduct extensive experiments across 11 frontier models, spanning both closed-source and open-source systems, and find that deception is model-dependent, increases with event pressure, and consistently erodes supervisor trust. Qualitative analyses further reveal emergent, long-horizon phenomena, such as ``chains of deception", which are invisible to static, single-turn evaluations. Our findings provide a foundation for evaluating future LLMs in real-world, trust-sensitive contexts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper describes a novel multi-agent architecture where a "performer" agent is given a "task stream" simulating a long-horizon interaction. "Events" are probabilistically inserted of varying intensity, designed to induce deceptive behaviour. A "supervisor" agent monitors the performer throughout, tracking scores such as level of trust in the performer, and a deception auditor retrospectively reviews the history to identify deceptive behaviour.

Results are obtained for 11 frontier models, showing a variety of tendency to engage in deceptive behaviour. The supervisor's trust is seen to anti-correlate with the performer's deception, and increasing pressure-level of events increases deception rate.

### Strengths
Overall well-written paper descibing a solid experiment.

The paper describes an original system which applies the lens of LLM deception to long-horizon tasks.

Using 14 tasks with a probabilistic event stream allows the work to move beyond anecdotal failures to quantify deceptive behaviour systematically.

The 5 categories of deception-inducing events, guided by social science findings, is a delightful and principled original contribution.

Modulating pressure-level allows granular exploration of agent behaviour in different incentive structures.

The structured deception annotations of a natural-language explanation allow quantitative results, differentiating between the frequency of deceptive behaviours across diverse LLM agents, and distinguishing between deception rate and severity to show that e.g. Claude has rare deception but still each with moderate severity. 

The clarity of results and language is excellent.

### Weaknesses
As the authors point out, strategic deceptions in LLMs is already a well-studied phenomenon. I do not feel that the authors sufficiently motivate why moving this lens to long-horizon interactions is particularly novel or needed, beyond that long-horizon interactions are (for reasons unrelated to deception) broadly of interest at this moment in time. It would be interesting to investigate deception split into multiple innocuous steps which are only concerning when combined (and therefore not addressed by previous work) but that does not seem to be the focus of this work. (Presumably, deception happens at the single step just after an Event?)

Why would we expect things like interdependent projects or pressure which unfolds over time (rather than being applied one-shot) to affect deception?

This is suggested by 5.2 (Qualitative Study), but such a crucial point needs to be in the main paper, rather than an Appendix (which I regrettably did not read). I suggest the Definitions could instead be moved to an Appendix to make space.

Fundamentally, it's not clear that this shows more than would be expected by existing short-task deception literature. Explicit and detailed comparisons to short-term deception data would help here. In particular, how does the whole-trajectory judgement differ from per-event judgement?

Alternatively, framing could focus on the /realism/ of the test-cases, arguing that the long-horizon tasks presented here are more realistic than those of existing literature. But the example event in Fig 2 does not seem to me to be realistic in presentation - I'd expect it to be below the bar of situational awareness (see Perez et al) of current frontier models.

Figure 4: these data have no error bars, so it is hard to determine to what extent trust, satisfaction and comfort are predictive of deception rate. Additionally, it's hard to discern in what way these three scores meaningfully differ from each other - what it is they're each picking up on. Presumably there's some anthropomorphic assumption that these words mean the same thing to the LLM judge that they do to us, that low "comfort" perhaps corresponds to the LLM judge getting "bad vibes" from the performer. But the data does not support any such interpretation, leaving one to wonder why there are three scores at all.

Figure 5: again there are no error bars, making it hard to judge the significance of these differences in deception type rates. It's also not clear what the take-away should be regarding such differences, if they were indeed statistically significant.

Use of LLM judge should trigger mention their liability to bias (see e.g. JuStRank, Gera et al), and ideally describe any mitigations taken. Especially in this multi-turn setting where it's established that model accuracy drops for tasks in general. Also cites BondJr & DePaulo r.e. human lie-truth judgements being only marginally above chance when assessed turn-by-turn - I'd want to see actual data for that here, e.g. comparison to human judges for a sample.

Other minor points:

Sycophancy: odd to not cite Sharma et al.

L357: unless you have some reason to specifically mention architectures, seems more appropriate to say "varies across models" since e.g. fine-tuning a model will result in an identical architecture but different behaviour. (I'm also unsure what work "systematically" is doing in that sentence.)

L402: to say that behavioural differences "reflect differences in inductive biases and training regimes" seems a somewhat vacuous and distracting way to end a paragraph.

L481: should this be "increases with event pressure" ?

### Questions
L195: Why is the variability of the event system "essential for studying deception"? Would not any schedule of unexpected events test the same adaptability of the agent? 

Table 1: what do the downward-pointing arrows mean? I'd have thought "ordered sorting", but the right-most column is not sorted.

I can imagine that 'falsification' is hard to distinguish from hallucination/fabrication. Is there a zero-incentive-to-lie baseline?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a novel simulation framework to study and evaluate deceptive behaviors in Large Language models (LLMs) over long-horizon interactions. The proposed framework is a multi-agent system consisting of a "performer" agent (the LLM being evaluated) and a "supervisor" agent that assigns tasks, provides feedback, and maintains an evolving state of trust. The system introduces "events" – probabilistic, contextual pressures grounded in social science theories – to create situations where deception might be incentivized. An independent "deception auditor" then retrospectively analyzes the entire interaction to identify and categorize deceptive acts. The authors conduct experiments on 11 frontier models, revealing that deception varies significantly across models, increases with pressure, and is strongly anti-correlated with the supervisor's trust.

### Strengths
- The paper presents an extensive evaluation across a diverse set of 11 recent and powerful closed and open-source models (Lines 315-321). 
- The primary strength of this paper is the introduction of the first, to my knowledge, comprehensive framework for simulating and evaluating LLM deception in long-horizon interactions. The multi-agent setup, with its performer-supervisor dynamic, is a realistic and effective way to model the relational and trust-based aspects of deception (Lines 16-19, 77-82).
- The paper well-written and organized. The methodology is explained clearly, with formal definitions provided for key concepts like the "structured event set" and the "supervisor agent's state" (Definitions 1 and 2).

### Weaknesses
- It seems that GPT-4o's deception rate drops under "critical" pressure, with the explanation being that the model becomes more "safety-aware" (Lines 460-464).  However, the qualitative analysis mentioned in Appendix C.2 is quite brief. This finding warrants a more in-depth investigation within the main body of the paper.

- The qualitative analysis introduces the concept of a "chain of deception," where a model gradually deviates from its constraints (Lines 467-471, 1324-1327). However, it seems to me that the provided example for Qwen3-235B (Appendix C.1.1) seems to conflate a "traceability lapse" (citing a document from a previous round without specifying the round) with outright fabrication. While the latter is clearly deception, the former is more ambiguous. The paper should more clearly define what constitutes a "link" in this chain and provide a more compelling example where an initial, subtle deceptive act causally leads to a more severe one later on.

### Questions
- Could you elaborate on the decision to use GPT-5-mini as the supervisor agent? How sensitive are the results (e.g., the final trust scores and deception rates of performer agents) to the choice of the supervisor model? Would a less capablesupervisor fundamentally change the result?

- Can the authors confirm if the to-be-released repository includes the full task stream definitions, event instantiations, and the specific prompts used for the supervisor and auditor agents?

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
The paper introduces a long-horizon, multi-agent eval to study LLM deception: a performer solves a 14-task project under dynamic “event” pressures, a supervisor tracks trust/satisfaction/comfort over multiple rounds. An external monitor labels deception (type and severity) based on the full trajectories. Across 11 frontier models, deception frequency and severity vary substantially by model family, increase with event pressure and interaction length, and are anti-correlated with supervisor trust and satisfaction. Qualitatively, deceptive behaviors manifest as falsification, concealment, and equivocation strategies.

### Strengths
* One of the first long-term deception focused eval frameworks for models.
* Seems like the authors put a lot of thought into justifying the choices guiding the design of their eval, and connecting to relevant literature from other disciplines
* Interesting analysis of the qualitative deceptive behaviors displayed by different models, and how it affects supervisor trust, satisfaction, and comfort

### Weaknesses
* **Somewhat overstated contributions.** While I do think that establishing this new benchmark for deception is valuable, two of three of the "key insights" discussed in the intro are already relatively established: (i) model behaviors on deception benchmarks vary quite a lot among model providers (e.g. [this](https://snitchbench.t3.gg/) among other deception evals), (ii) deception rates rising under pressure were studied [here](https://arxiv.org/abs/2311.07590) (which you cite). Moreover, deception as a risk in long-horizon interactions – discussed as a contribution – is already a well-known risk (e.g. discussed [here](https://arxiv.org/abs/2405.17713) among other works)
* Unclear to me whether the eval is reflective of real situations in which models would be deployed. Maybe models are simply role-playing as in agentic misalignment. While it's still significant if models were to deceive in those settings, it's less significant than if this were reflective of realistic deployment scenarios. This also threatens the significance of the results on anticorrelation between deceptiveness and supervisor trust – is this specific to these fictitious domains, or would that also occur in more realistic domains?
* There is a strong risk of this paper being "just another eval" which doesn't end up being picked up on. I would find it helpful to see a contrast between this eval and other evals which measure deception (e.g. [this](https://arxiv.org/abs/2304.03279),  [this](https://snitchbench.t3.gg/), and others) – which explicitly makes the case of what this eval adds that the current state of evals, how this eval could detect certain kinds of harm which other evals would currently miss, etc. While some of this could be argued on a high level, I expect that a strong argument here would require quantitative evidence, e.g. showing that a certain model that is deemed non-deceptive by most current other evals is considered very deceptive by your evals.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a benchmark for evaluating LM deception over long-horizon interactions between a performer and supervisor agent. Deception is evaluated by an LM auditor and the paper evaluates the behaviour of 11 frontier models.

### Strengths
Addresses an important problem of deception in long horizon contexts. 

Well written and presented, the figures are especially useful. 

Clear and reproducible methodology (e.g., full auditor prompts in the appendix). Interesting task design with probabilistic events. 

Mostly good awareness of relevant literature. 

Broad empirical analysis with many LMs.

### Weaknesses
The paper claims to introduce the first long horizon eval for deception but e.g., Meinke 2025 (cited already in the paper) is also a long horizon LM agent eval for deception. The claims to novelty should be reduced. 

Since the deception metrics are based on the Auditor, it would be good to show that the Auditor gives sensible measurements, e.g, by comparing it to human labels, or by presenting AUROCs for the auditor when evaluating benign vs deceptive performers. 

Overall I found the results reasonable but a bit limited and hard to interpret in places (see questions). 

minor

cite https://arxiv.org/abs/2312.01350 for the definition of deception

### Questions
What kind of actions and tools are available to the performer agent? 

Does the Auditor get to see the reasoning of the models?

Can you include some evalutaion of the Auditor like the AUROCs mentioned above?

For the results out deception type --- is falsification more natural in the tasks? Does the performer have the option to falsify information or conceal it? You interpret the results as saying that model'e prefer to falsify but maybe they can only achieve the task by doing that and concealing wouldn't cut it, but they would prefer to conceal information off either strategy would work.

### Soundness
3

### Presentation
3

### Contribution
2
