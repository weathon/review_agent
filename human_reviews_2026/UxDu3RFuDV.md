# Language and Experience: A Computational Model of Social Learning in Complex Tasks

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8, 6

## Abstract
The ability to combine linguistic guidance from others with direct experience is central to human development, enabling safe and rapid learning in new environments. How do people integrate these two sources of knowledge, and how might AI systems? We present a computational framework that models human social learning as joint probabilistic inference over structured, executable world models given sensorimotor and linguistic data. We make this possible by turning a pretrained language model into a probabilistic model of how humans share advice conditioned on their beliefs, allowing our agents both to generate advice for others and to interpret linguistic input as evidence during Bayesian inference. Using behavioral experiments and simulations across 10 video games, we show how linguistic guidance can shape exploration and accelerate learning by reducing risky interactions and speeding up key discoveries in both humans and models. We further explore how knowledge can accumulate across generations through iterated learning experiments and demonstrate successful knowledge transfer between humans and models—revealing how structured, language-compatible representations might facilitate human-machine collaborative learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a Bayesian model by which agents use both linguistic guidance and direct experience to develop a world model of their environment. Their proposed agent updates their distribution of world models based on compatibility with game observations and linguistic guidance from a model, weighted by probability assigned by an LLM-based speaker model. Experiments on the VGDL suite of games shows that their agents learn significantly more quickly than Deep RL and LLM agent-based baselines, and that linguistic guidance greatly accelerates learning when added to direct experience. They also find that model-written guidance can also improve learning for human participants, although both models and humans learn better from their own guidance.

### Strengths
- Thorough theoretical framing of the important problem of learning jointly from language and experience, from which the proposed model clearly follows.
- Results across a variety of game settings show that the model learns much more efficiently than Deep RL and LLM agent-based approaches, suggesting that the model does, in fact, approximate human learning mechanisms better than existing methods.
- Rigorous study of the impact of human vs model guidance, as well as differences in communication styles, motivates some pretty interesting future directions related to collaboration in human+AI communities.

### Weaknesses
- The baselines used are both highly limited within the time horizon given, which limits interpretation of results. It would be useful to run the pure LLM and DQN baselines over a longer horizon until some learning happens, to get a better sense of how much more efficient the proposed model is. This could also help us better understand the relative gains from incorporating linguistic guidance in the model vs pure LLM settings, which would strengthen the argument that the new agent better models human learning improvements from linguistic guidance.
- As the authors note, they evaluate agents over a very limited set of game environments. Fully open-ended environments would be out of scope of the paper, but it’s not clear to me that the games used are significantly more complex than the environments used in past work (e.g. the 2024 Zhi-Xuan work cited) that are dismissed as overly simplistic.

### Questions
- It’s not clear to me from the citation given in 3.4 why language-conditioned RL methods require thousands of episodes to learn language-policy mappings, especially since the citation is from 2019 and more recent methods use LLMs to become more sample efficient (e.g. "Language instructed reinforcement learning for human-ai coordination”, Hengyuan Hu et al, ICML 2023). Since this seems to be more relevant to the proposed method than the Deep RL or LLM Agent baselines, it would be useful to spend more time justifying the decision to not benchmark against them.
- I’d recommend including median hourly pay for the crowdworkers, either in Appendix G or Section 4.

### Soundness
3

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
This paper develops a method to integrate natural language advice into planning as well as transform experience into natural language advice. The authors demonstrate that both human-generated and model-generated advice improve sample efficiency in learning a suite of tasks. 

Overall, I recommend acceptance because the method facilitates effective knowledge transfer between humans and models, providing a valuable avenue for future work on human-AI co-learning. However, because the method is restricted to the VGDL game, it could be difficult to scale to real-world tasks.

### Strengths
The results convincingly show that advice, whether human-generated or model-generated, improves learning performance. The evaluation is conducted across multiple tasks, demonstrating the robustness of the approach. The method is well-motivated and the results are intuitive (advice improves learning, humans can learn better from human advice than model advice). The generational learning experiments provide additional evidence that the model-generated advice is effective, strengthening the paper's claims.

### Weaknesses
As the authors acknowledge, the method is currently limited to the VGDL game space. Scalability to more complex, diverse, or realistic environments could be difficult.

The prompts for integrating natural language advice could potentially have information leak (see questions below), which may bias the results.

### Questions
**Questions for clarification:**

You mention "prompting the LM with the received message L and instructing it to answer multiple-choice questions about specific VGDL rules." Wouldn't this give the model an advantage over the human or pure LM baseline because the multiple choice answers already narrow what the feasible rules are? Is it possible to evaluate the method without this step by sampling candidate rules directly from the LM?

---

**Minor comments:**

- It would be helpful to add effect sizes to the reported results.
- The authors state that "We compare our approach to three baselines: 1) Oracle: a model that plans to solve the game using ground-truth game rules, (Line 467) " but the oracle baseline is not shown in the main results.
- In the paragraph "Variability across games," the authors conclude that "These patterns suggest that social learning strategies like selecting high-performing teachers" (line 400), but I don't think this conclusion follows from the results. The results don't imply that higher-performing participants gave better advice.
- The authors state that "We show more examples of human- and model-generated messages in Appendix Section I and on our website (Line 385)." The website only has some game demos. I couldn't find the examples of human and model-generated messages.

---

**Suggestions for improvement (not factored into score):**

The authors state "Human-generated messages often include game abstractions, high-level strategies, and planning heuristics that our model currently cannot leverage (Line 467)." While the authors do provide examples of human-generated and model-generated messages in the appendix, it would be helpful to more systematically categorize the differences between human and model-generated messages. This would be beneficial for further work on human-AI collaboration.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores how language and experience jointly support social learning in humans and computational agents. The work takes the view of human social learning as probabilistic inference over world models. Empirically, they study human participants and Q-learning agents performing ten games, examining how linguistic guidance shapes exploration and accelerates learning. A pretrained large language model (LLM) is used both to generate advice for other agents and to interpret linguistic input as evidence.
The results show that linguistic advice improves learning efficiency for both humans and models. Human participants learn equally well from model- and human-generated advice, whereas models benefit more from model-generated advice. The authors attribute this to differences in communication styles — human messages often include metacognitive reflections and analogies that LLMs find difficult to parse.
The paper also investigates iterated (generational) learning, where agents interact with the environment for limited “lifetimes” and pass advice to subsequent agents. Performance generally increases across generations, though with some regressions depending on the game.

### Strengths
* The writing is clear and engaging.
* The combination of behavioral experiments with humans and artificial agents is very creative.
* The results on cross-generational knowledge transfer are intriguing and connect well to theories of cumulative culture and iterated learning.
* The work highlights a bidirectional exchange between humans and models — both learn from each other, which is a timely topic to consider.

### Weaknesses
* _Framing_ The title and framing promise a computational model of social learning, but the core of the paper is some empirical observations of social learning among and between agents and humans. The authors should consider revising the title or clarifying that the contribution is primarily empirical and conceptual, not a new formal model. 
* It is unclear what are the structured world models under the claimed “joint probabilistic inference over structured world models.”
* The claim that the LLM performs Bayesian inference is not supported by direct evidence. The LLM generates or interprets advice, but it is not shown to approximate a Bayesian posterior or perform probabilistic updates.
* The results for iterated (generational) learning vary substantially across games (e.g., improvement for Bess and Birds, but not Plaque Attack or Missile Command). It remains unclear why certain tasks show cumulative improvement while others regress, and what properties of the games modulate this effect.
* Figure 6 (the Aliens game) may not be a good test of generational learning, as performance saturates quickly, leaving no room for measurable improvement.

### Questions
1. What exactly constitutes the “computational model” here — is it the Q-learning setup augmented with linguistic input, or the LLM itself viewed as a Bayesian speaker–listener model? 
2. Is there any quantitative or behavioral evidence that the LLM performs probabilistic inference rather than pattern-based generation? 
3. Why do some forms of advice promote generational improvement while others do not? Are there identifiable linguistic or structural features  that predict transfer success?
4. Given that human messages contain analogies and metacognitive strategies that LLMs fail to parse, could fine-tuning on human teaching data improve model interpretability and alignment?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a framework that models human social learning as joint probabilistic inference over structured world models, integrating both linguistic and sensorimotor information. The core method used is to transform a pretrained language model into a probabilistic model of human advice, enabling agents to generate and interpret guidance. Experiments across ten video games show that linguistic input enhances exploration efficiency, safety, and learning speed. Iterated learning experiments further demonstrate knowledge accumulation and transfer between humans and models, supporting more effective human–AI collaboration.

### Strengths
(1) The paper is well-written, with clear motivation and an accessible presentation. The formulation of joint inference over experiential and linguistic guidance through Bayesian updating is both insightful and valuable.

(2) The human evaluation experiments on building better learning partners are interesting and effectively demonstrate the role of linguistic guidance in improving video game performance. Moreover, the study’s extension to bidirectional communication between humans and models provides an elegant and symmetric perspective on social learning.

### Weaknesses
(1) Limited generalization: The current modeling approach is evaluated across different video games. However, in real-world agentic environments, the belief space can be far more complex, posing additional challenges for both modeling and linguistic guidance. As a result, the proposed method may have limited applicability beyond specific domains.

(2) Model selection discussion: It would be valuable to include an analysis of different LLM choices. Stronger models may offer more effective guidance for both LLMs and human experts—or potentially the opposite. An additional study could examine whether the same phenomenon shown in Figure 5 persists when the guidance-providing model and the student model belong to different model families.

### Questions
The difficulty of different games can vary significantly depending on the complexity of their underlying theories and gameplay mechanics. It would be helpful to clarify whether the size of the candidate theory set varies across games and how much this variation affects performance. I am also interested in more details about the selection of M (candidate theory), particularly the reasoning behind choosing 20.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel computational framework for social learning in AI, which rigorously integrates linguistic guidance and direct sensorimotor experience through joint probabilistic inference. The core innovation is leveraging a pre-trained Large Language Model (LLM) as a probabilistic model of human advice ($P(\text{Language} | \text{Model})$). By treating language as evidence to infer underlying structured, executable world models, the agent achieves superior sample efficiency and rapid, safe knowledge acquisition in complex tasks, significantly outperforming conventional learning methods.

### Strengths
1. Novelty in Theoretical Framework: The paper introduces an innovative computational model that treats language not merely as an input feature, but as probabilistic evidence for Bayesian inference over potential world models. This is a theoretically rigorous approach to language grounding.

2. High Sample Efficiency: By leveraging linguistic guidance as a strong prior, the method significantly reduces the hypothesis space, leading to substantially faster learning and higher sample efficiency compared to purely experiential methods.

3. Emphasis on Structured Knowledge: The model focuses on inferring structured, executable world models (rules and mechanisms), which inherently promotes generalization across tasks and provides better interpretability than learning simple policies or value functions.

4. Robust Integration: The probabilistic nature of the framework allows for the principled integration of two heterogeneous data sources (noisy language and imperfect experience), making the system robust to ambiguity and uncertainty in human advice.

5. Applicability to Theory of Mind: By converting the LLM into $P(\text{Language} | \text{Model})$, the agent performs a form of reverse inference, effectively modeling the human advisor's underlying knowledge or "mind," which is a powerful concept in social learning.

### Weaknesses
1. Scalability Challenge in Model Space: While the framework operates over "structured world models," the computational feasibility of performing inference (or even representation) over the vast space of possible models ($\text{Model}$) in real-world, high-dimensional, or continuous state environments remains a major concern. The paper needs to better address how efficient search or approximation is maintained.
2. Experimental Complexity Limitation: Although the title claims to address "Complex Tasks," the typical experimental setups (often discrete environments like Block-World or Grid-World, based on the context of similar research) may not fully demonstrate the method's effectiveness and scalability in truly unstructured, high-fidelity, or continuous environments, limiting the generalizability claim.
3. Insufficient Comparison to Language Baselines: The comparison with existing methods may be incomplete. Specifically, the paper often omits a direct comparison with state-of-the-art deep reinforcement learning (DRL) techniques that use language as an auxiliary input feature or recent LLM-based planning methods (e.g., Chain-of-Thought planning), which could be a strong alternative baseline for incorporating guidance.
4. Technical Validity of the LLM Probability Approximation: The paper's core mechanism, the Language Likelihood $P(\text{Language} | \text{Model})$, is approximated using the uncalibrated generation probability of a large proprietary model (LLAMA-3.1-70B) conditioned on a constructed prompt: $P(L|T) \approx P_{LM}(L|\text{prompt}(T))$. This approach raises concerns about its technical validity. LLM generation probabilities are generally not guaranteed to be accurately calibrated for strict Bayesian inference, which is central to the framework. The reliance on prompt engineering for a fundamental probabilistic term introduces a fragility that needs more rigorous theoretical justification or empirical validation of the approximation accuracy.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3
