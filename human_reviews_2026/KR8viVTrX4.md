# Formalising Human-in-the-Loop: Computational Reductions, Failure Modes, and Legal-Moral Responsibility

- Decision: Accept (Poster)
- Scores: 6, 4, 10

## Abstract
We  use the notion of oracle machines and reductions from computability theory to formalise different Human-in-the-loop (HITL) setups for AI systems, distinguishing between trivial human monitoring (i.e., total functions), single endpoint human action (i.e., many-one reductions), and highly involved human-AI interaction (i.e., Turing reductions). We then proceed to show that the legal status and safety of different setups vary greatly. We present a taxonomy to categorise HITL failure modes, highlighting the practical limitations of  HITL setups. We then identify omissions in UK and EU legal frameworks, which focus on HITL setups that may not always achieve the desired ethical, legal, and sociotechnical outcomes. We suggest areas where the law should recognise the effectiveness of different HITL setups and assign responsibility in these contexts, avoiding human `scapegoating'. Our work shows an unavoidable trade-off between attribution of legal responsibility, and  technical explainability.  Overall, we show how HITL setups involve many technical design decisions, and can be prone to failures out of the humans' control. Our formalisation and taxonomy opens up a new analytic perspective on the challenges in creating HITL setups, helping inform AI developers and lawmakers on designing HITL setups to better achieve their desired outcomes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper formalizes human-in-the-loop (HITL) systems by mapping three configurations to concepts from computability theory: trivial monitoring (the human can only stop the process; the system is a total function), endpoint action (a single real human query; a many-one reduction), and involved interaction (potentially unbounded real human queries; a Turing reduction). It unifies related terms such as HIC (human-in-command)—a governance stance where humans retain ultimate authority over system goals and deployment—and MHC (meaningful human control)—designing systems so humans have timely, informed, and effective influence over critical outcomes—into one coherent framework, showing how this helps regulators identify tokenistic oversight and connect failure modes to specific designs. The legal discussion, focusing on EU and UK law, links these configurations to the GDPR and the EU AI Act, warning against “moral crumple zones,” where humans are held responsible for system failures beyond their control. Overall, the paper argues that meaningful oversight requires moving beyond trivial monitoring to designs that incorporate genuine, well-timed human queries, balancing explainability with responsibility.

### Strengths
Interesting case studies that trace multi-category failure cascades—bridges theory to incidents.

Novel and Rigorous Formalization: The application of oracle machines and computational reductions to formalize HITL systems is genuinely creative and mathematically sound. This provides a precise vocabulary for what has been a vague concept, distinguishing trivial monitoring, endpoint action, and involved interaction in a principled way.

Regulatory relevance: concrete reading of GDPR/EU AI Act and the SCHUFA precedent; shows how designs can slip back into “solely automated.”

Honest trade-off analysis (explainability vs. responsibility) with policy-minded remedies to avoid human scapegoating

Overall it was an engaging read even if not a standard paper and worth thinking more about.

### Weaknesses
Compelling case studies that trace multi-category failure cascades—bridges theory to incidents.

Legal scope: law discussion centres on EU/UK; portability to other regimes is not fully analysed.

Hard to translate this into real life: proving a system truly asks real queries (vs. veneer) is hard; guidance for auditors could be deeper. 

Human-as-Oracle Limitations: While B.1 addresses this, the fundamental modeling of humans as fixed oracle functions is problematic. Humans learn, adapt, get fatigued, have strategic behavior, and exercise agency in ways that don't map cleanly to mathematical oracles. The formalization may obscure important sociotechnical dynamics.

Limited Empirical Validation: The taxonomy is based on consultations from 2020-2022 with primarily startups. There's no systematic evaluation of whether: (a) developers find this framework useful, (b) regulators would adopt it, (c) it actually prevents failures when applied. The framework is proposed but not validated as effective.

Limited validation that the framework actually helps prevent failures or improve systems

For an ML conference, limited technical ML content or empirical experimentation
No learned models, training procedures, or optimization
No experiments, datasets, or empirical ML evaluation
"AI systems" discussed are abstract ADMSs, not modern ML systems (in fact ended up googling if Notre Dame was AI based system)

### Questions
How can regulators or auditors verify whether a system truly involves “real human queries” versus superficial or scripted checks?

How might involved interaction be made practical at scale, especially in high-volume or time-sensitive contexts (e.g., content moderation, autonomous driving)?

Do you envision technical audit tools that could automatically classify systems by your HITL taxonomy (e.g., detecting trivial monitoring vs. involved interaction)?

How does your framework prevent systems from adding meaningless queries to appear as involved interactions while remaining functionally automated (like the SCHUFA concern)?

How should designers determine the minimum necessary level of human involvement that remains legally and ethically “meaningful” without creating inefficiency?

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
2

### Summary
The work studies how to formalise, from a computability-theory point of view, various forms of human-in-the-loop (HITL) approaches. Then, the authors propose a taxonomy to categorize how HITL can fail, highlighting a trade-off in terms of attribution of legal responsibility and technical explainability.

### Strengths
The paper's main strengths are:

1. the paper presents a legal perspective for the study of Human-In-The Loop (HiTL) approaches. The perspective is relevant, as Machine Learning (ML) models are widespread to support decisions and correctly framing the responsibility for wrong predictions is very important.
2. The paper is thought-provoking, offering an interesting approach for addressing the responsibility of the actual decision. I appreciated (even if I do not completely agree) the problem of scapegoating single humans for machine failures.
3. I appreciated the focus on explainability not only on ML models, but on the entire decision-making pipeline to ensure that every decision step is transparent. I agree that this is very relevant and important to properly address the responsibilities of various agents.

### Weaknesses
I leave here a few open points I think the authors should discuss:


1. While I appreciate the theoretical formalization of HiTL, I would have liked the authors to offer some operational criteria to tackle the highlighted shortcomings of HiTL approaches. In the current status, it doesn't seem easy to translate the theoretical contribution into a real-life workflow.
2. I think the authors have not properly placed their work in the current literature. Several paradigms in ML aim to involve humans in the loop, such as learning to defer (see e.g., [1]), machine learning models that abstain (see e.g., [2]) and decision support systems through conformal prediction (see e.g., [3]). Discussing these more recent lines of work is, in my view, mandatory.
3. I think the authors should explicitly state why they are focusing only on the EU/UK environment. Enlarging to other countries/law frameworks could be impactful and provide a valuable picture of HiTL's existing legal frameworks worldwide.

[1] Madras, D., Pitassi, T., & Zemel, R. (2018). Predict responsibly: improving fairness and accuracy by learning to defer. Advances in neural information processing systems, 31.

[2] Ruggieri, S., & Pugnana, A. (2025). Things machine learning models know that they don’t know. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 39, No. 27, pp. 28684-28693).

[3] Straitouri, E., Wang, L., Okati, N., & Rodriguez, M. G. (2023, July). Improving expert predictions with conformal prediction. In International Conference on Machine Learning (pp. 32633-32653). PMLR.

### Questions
Please discuss the highlighted weaknesses.
Regarding W1, I think adding some examples of countermeasures that could have been implemented in the Uber case would help.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
This paper is an attempt at formalising several aspects of the human-in-the-loop (HITL) paradigm for automated decision making systems (ADMS).

The contributions of the paper can be summarized into three main points:
- A formalisation of different HITL modes in terms of oracle machines.
- A taxonomy of failure modes in HITL systems.
- A discussion about responsibility and accountability of HITL systems.

The paper defines human-in-the-loop systems as oracle machines, where the human is the oracle, and proposes three main categories depending on the level of involvement of the human: monitoring, endpoint, and involved interaction. In monitoring (called in the paper *human trivially monitoring the loop* or *trivial monitoring*), the only intervention that the human can have is to halt the execution of the ADMS. This models, for example, systems in which a human observer can stop the ADMS and take control as an emergency response. In endpoint (in the paper *human at the end of the loop* or *endpoint action*), the human receives a single query, and the program halts after the human's response. This models systems in which a program presents a set of solutions or courses of actions to a human, and the human makes the final decision. In involved interaction (also called in the paper *human involved in the loop*), the machine can ask an unbounded amount of queries to the human, and these must have a signficant effect in the resulting computation.

The paper classifies HITL failures into five categories (failure of the machine, failure of the human, failure of the human-machine interface, failure of the human-machine workflow, and failure due to exogenous circumstances), and discusses the connection of the failure modes with the HITL categories. The different failure modes are also illustrated through a real-world failure example, in which a HITL autonomous car was involved in a fatal traffic accident.

Finally, the paper discusses the legal frameworks, mainly in the EU and UK, as they pertain to HITL systems, and the tradeoffs in terms of assigning responsibility between the human and the machine.

### Strengths
S1. Relevant and timely topic.

S2. The paper is clearly structured and very well written.

S3. The formalisation in terms of oracle machines is well inspired. The illustration of different failure modes in the different examples (one in the main text, two in the supplementary material) is convincing. 

S4. The paper engages with relevant legal frameworks (in the EU and the UK), analyses the aspects relevant to HITL and proposes improvement points to avoid real problems that have arisen from the use of HITL in safety-critical applications, like the use of the human in the loop as a scapegoat to avoid responsibility by the ADMS deployer.

### Weaknesses
I don't see major weaknesses. As minor points:

**W1.** The paper leaves a lot of the content in the appendix, and the many pointers to the appendix in the main text are distracting. As a reader, it is not clear when I will be able to follow the paper just from the main text, and when I need to go and read the supplementary content.

**W2.** The choice of limiting HITL paradigms to three concrete ones, leaving many intermediate steps unformalised, is a weak point of the paper. I know this is a conscious choice motivated in the text. However, it seems that the discussion, particularly in what regards the EU and UK legal frameworks, would need to include these paradigms with intermediate involvement of the human.

**W3.** The halting abilities of the human are not completely clear. 

- In trivial monitoring, it is not clear whether the human has the ability to halt the ADMS at any time, or only when it asks a query. The former would not be realistic in certain situations, and the latter would then require the nuance of when does the machine query the human as part of the analysis.
- In endpoint interaction, it is not clear whether the human has the ability to halt the machine before it reaches the endpoint query.

**W4.** In lines 349 to 362, you argue that trivial monitoring cannot be seen as the *meaningful oversight* required by the EU GDPR and AI Act. I think the deliberate choice of words (like *trivial*) and the lack of specificity in the halting capabilities of the human (see W3) are playing a hidden but important role in this argument. To take an example from this paper, in the Uber case, the car has a trivial monitoring HITL setup, in which the human can resume control of the vehicle at any time with very low friction (just touching the steering wheel or the pedals). I doubt that the EU law requires more involvement of the human to be considered *meaningful oversight*. While I'm not aware of levels of autonomy as in the Uber case being currently deployed in the EU, classical cruise control systems in cars are very common, definitely within legal margins, and offer no more than trivial monitoring by the human in the loop. Consider maybe not using the word *trivial* in your taxonomy.

### Questions
**Q1.** The difference of failure mode 2 (process and workflow) and failure mode 4 (failure of the human) is not clear to me. Can you provide examples where they are independent failure modes? That is, and example in which there is a failure of the process without a failure of the human, and an example in which there is a failure of the human without a failure of the process.

Q2. In lines 1151 to 1158, you argue that deciding whether an HITL system is actually of the involved interaction type from a moral perspective is particularly difficult because of two reasons: the machine may ignore the human's input, and the machine may act contrary to the human's input. 
- **Q2.1** For the first case, how is this different from the machine not issuing a real query? Shouldn't this case already be covered by the definition of involved interaction as requiring real queries?
- **Q2.2** A machine could be designed to act against the human inputs and still be moral. For example, a human-like robot used to train humans in conflict resolution. Would this be accounted by your framework?

### Soundness
3

### Presentation
4

### Contribution
4
