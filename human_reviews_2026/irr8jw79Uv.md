# Verbalized Bayesian Persuasion

- Decision: Withdrawn (Treated as Reject)
- Scores: 0, 0, 4, 4

## Abstract
Information design (ID) explores how a sender influence the optimal behavior of receivers to achieve specific objectives.
While ID originates from everyday human communication, existing game-theoretic and learning methods often model information structures as numbers, which limits many applications to toy games.
This work leverages LLMs and proposes a verbalized framework in Bayesian persuasion (BP), which extends classic BP to real-world games involving human dialogues for the first time.
Specifically, we map the BP to a verbalized mediator-augmented extensive-form game, where LLMs instantiate the sender and receiver.
To efficiently solve the verbalized game, we propose a generalized equilibrium-finding algorithm combining LLM and game solver. 
The algorithm is reinforced with techniques including verbalized commitment assumptions, verbalized obedience constraints, and information obfuscation.
Experiments in dialogue scenarios, such as recommendation letters, law enforcement, diplomacy with press, validate that our framework can reproduce theoretical results in classic BP and discover effective persuasion strategies in more complex natural language and multi-stage scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper proposes a framework that extends the classical Bayesian Persuasion framework to settings where communication between sender and receiver takes place through natural language, mediated by LLMs. The authors map Bayesian persuasion into a verbalized mediator-augmented extensive-form game and adapt the Prompt-PSRO to optimize strategies via prompt manipulation rather than direct policy optimization.  The framework is evaluated on classical examples from Bayesian persuasion literature such as recommendation letters, courtroom persuasion, law enforcement deployment, and on the complex setting of full-press Diplomacy.

### Strengths
The paper tries to bridge theory and real-world settings in Bayesian persuasion by enabling settings in which sender-receiver communications happen through natural language. This direction is interesting and potentially impactful.

### Weaknesses
While I find the general research direction to be quite interesting, its realisation requires substantial improvement. 

In general, one problem is that the paper is not clear about what exactly is being shown and how. Many important steps are only explained by words, without any formal definition, thereby hiding many important details. This contributes to creating confusion when reading the paper. 

I don’t understand what is the goal of Section 3 and the set-up it introduces. Definition 3.1 seems just to add a big layer of notation from extensive-form games, which does not seem to be particularly useful for what comes next. Also from a conceptual perspective, this appears to be just a complex way of writing the same problem. The fact that the following explanations are very vague does not help clarifying this.  It would really help if some parts of the paper were more formal, so that one can understand what the authors are really talking about. One example: section 4 paragraph 250 (e.g., “the infoset is available only […]”). 

Along this direction, many statements need to be more formal and better justified, otherwise they seem unsubstantiated high level claims. Examples: Line 148 to 150, 266—268. Given the confused and high level description above, I don’t understand how proposition 4.1 fits in the bigger picture. How can you prove this when interactions are mediated by LLMs? The proof does not seem to take any of the complexities described above into account so I’m not even sure what the purpose of the proposition is. 

The experiments section provides zero details on how baselines are defined or how rewards are computed for natural language outcomes. In Section 5.1, the only interpretable comparison is Figure 4. Subsequent results (especially in multistage and Diplomacy tasks) lack enough methodological detail to assess their validity.

### Questions
None.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper proposes a new framework called verbalized bayesian persuasion, which aims to bring BP closer to real-world settings by using natural language. In classic BP, information is modeled with discrete signals and probability distributions over state of the world, here, both sender and receivers are LLMs. The authors formulate this as a verbalized mediator-augmented extensive-form game and propose an equilibrium-finding approach inspired by Prompt-Space Policy Response Oracles (Prompt-PSRO).

### Strengths
The paper tackles a conceptually interesting and timely question: how to operationalize information design and persuasion in the natural language domain.

### Weaknesses
Unfortunately, the current presentation makes it very hard to assess the actual contribution or scientific soundness of the work. The paper is poorly written and often reads as a mix of buzzwords rather than a clear, well-defined technical contribution.

* It is unclear what exact problem the paper is solving. Is the goal to compute optimal signaling schemes in language? To translate classical BP instances into a verbalized domain? Or to show that LLMs can approximate equilibria? None of these are made precise.
* The formalism in Section 3 (“Mediator-Augmented Game Formulation for BP”) feels disconnected from the natural language motivation. It mostly re-states standard EFG definitions without making clear how the “verbalized” aspect modifies or extends them.
* Section 4: Figure 3 is presented as central to the framework, but the diagram and the surrounding text are almost entirely uninformative — it’s not clear what the arrows mean, what information flows where, or how verbalization interacts with the equilibrium computation.
* The theoretical contributions appear minimal or, at a minimum, unclear.
* The paper is extremely difficult to read. Sentences are often grammatically incorrect, vague, or overloaded with jargon (“verbalized mediator-augmented, extensive-form game”, “verbalized commitment assumptions”, etc.).

### Questions
What is the precise step-by-step definition of a VBP problem?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a verbalized framework that uses large language models (LLMs) to tackle Bayesian persuasion problems. The approach relies on LLMs to simulate game participants and employs both LLMs and game solvers to find equilibria. The experiments aim to reproduce well-known Bayesian persuasion results and to handle more complex games.

### Strengths
The topic is interesting, and using LLMs for game-theoretic problems is also an interesting direction. The paper is clearly written and well-structured.

### Weaknesses
I have some concerns about the strength of the contribution. The proposed framework converts a Bayesian persuasion problem into a verbalized version and then solves it with LLMs. I understand that in some real applications, beliefs cannot be explicitly expressed as distributions and that the persuasion process is done through natural language. But what is the advantage of the proposed method compared methods like directly predicting the agents' behaviors using a ML model and then solving it using the standard approach?

### Questions
Please see my comments above.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Verbalized Bayesian persuasion (VBP) takes Bayesian persuasion (BP) to natural-language dialogue between a sender and receiver implemented by a large language model (LLM). Strategy search becomes prompt optimization via a policy-space response oracle (PSRO) variant (Prompt-PSRO) with verbalized commitment and obedience, obfuscation feedback, and conditional prompts learned with FunSearch. A proposition claims convergence to epsilon-approximate Bayes-correlated equilibrium (BCE) in static settings and epsilon-approximate Bayes-Nash equilibrium (BNE) in multi-stage settings.

The method maps states, signals, and actions to text and optimizes prompts rather than model parameters. For multi-stage problems, it learns history-conditioned prompt functions. Best responses use OPRO (Optimization by Prompting). Obedience is enforced with sampling-based penalties, and a self-reflection loop mitigates alignment bias.

Experiments span Recommendation Letter, Courtroom, Law Enforcement, and Diplomacy with press. Metrics include sender/receiver reward, honesty rates, exploitability vs BCE, expected calibration error, and an obedience penalty. A small human study (N=15) compares VBP to a simple DeepSeek-R1 prompt.

Results: With polarized signals, VBP matches or beats BCE and strong multi-agent baselines on payoffs without ignoring honesty. Without polarization it stays close. In multi-stage settings, conditional prompting reduces deception over time. In Diplomacy, score gains are modest but reliability improves clearly. Human judgments favor VBP.

### Strengths
The paper tries to reconcile information design (ID) with BP beyond toy numeric signals to realistic dialogue settings. The MAG formulation and Prompt-PSRO reduction are coherent and leverage established game-theoretic tools while keeping LLMs in-the-loop via prompt search. The conditional prompt function idea for multi-stage persuasion is natural, effective, and well motivated. The empirical suite spans REL/COR/LAE and a demanding Diplomacy benchmark. Reported results indicate that VBP reproduces BP intuitions in S1/S2, exhibits history-aware bargaining dynamics in S3, and increases reliability and calibration in S4; the human-agreement study, albeit small, favors VBP over a strong simple-prompting baseline. The paper also articulates an ethics statement acknowledging manipulation risks, which is nice.

### Weaknesses
The theoretical claim (epsilon-approx BCE/Bayes-Nash) rests on the MAG reduction and Prompt-PSRO approximations, but some core assumptions and proofs are missing as far as I can see; clearer main-text conditions and limitations would strengthen soundness. Measurement of lying/honesty and several scores relies on LLM-based evaluators--this is problematic, as it risks circularity and model-judge bias; more non-LLM or human-verified assessments would increase credibility and, in my view, these are needed.
The human study is small (N=15) and limited to agreement, not real outcomes; statistical treatment is reported, but broader human evaluation and preregistration would help.
Also, baseline comparisons are small: more comparisons to recent LLM game-solvers and agent workflow methods would paint a clearer picture. The Diplomacy improvements on final score are modest. Notably, the framework operationalizes deception -- while ethics are discussed, concrete safeguards and misuse mitigations in deployment are limited.

### Questions
Theory: Please state in the main paper the precise conditions under which Proposition 4.1 holds. What are the exact approximation sources from Prompt-PSRO, categorical OPRO sampling, and FunSearch selection, and how do they compose to yield ε bounds? This not too clear to me.

Could you report exploitability vs iterations with confidence intervals for all tasks in main text?
  
Several key metrics (honesty/lying, signal polarization labels) depend on LLM judges. Can you provide sensitivity analyses with non-LLM annotators or rule-based ground truth, plus cross-model judges to test robustness?
   
Do you have any plan to scale beyond N=15?

Please compare against at least one agent-workflow or equilibrium-search LLM approach and a policy-space response oracle (PSRO) variant without verbalization in order to isolate the contribution of verbalization.

What guardrails prevent misuse of VBP to optimize harmful persuasion in real deployments? Any red-teaming results or policy filters beyond the alignment-induced honesty oscillations discussion?

### Soundness
1

### Presentation
3

### Contribution
3
