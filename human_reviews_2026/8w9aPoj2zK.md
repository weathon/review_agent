# CEDAR: A Counter-Example Driven Agent with Regular Restriction

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
We introduce CEDAR, a Counter-Example Driven Agent with Regular Restrictions
in Minecraft, which learns and encodes informal specifications and skills as regular
languages. Our formalizer constructs deterministic finite automata (DFAs) to repre-
sent informal specifications by utilizing membership query responses from a Large
Language Model (LLM) and counterexamples provided by a human. The DFA
alphabet is derived from a global set of environmental events, with words in the
language representing expected event sequences. These learned DFAs are then used
by CEDAR’s skill learner to acquire the necessary skills to fulfill the specifications.
CEDAR supports both goal completion and lifelong learning by leveraging passive
and active DFA learning algorithms, respectively. The DFAs for skills are refined
through counterexamples generated from DFA simulations in the real environment.
Skills acquired by CEDAR can be adapted to new scenarios by modifying the
alphabet and can be logically verified to ensure they meet expected properties.
Empirical evaluations demonstrate that CEDAR surpasses state-of-the-art methods
in controllability, robustness, and extensibility.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the CEDAR framework, which integrates active automata learning with reinforcement learning to enforce formal specifications during policy learning. When agent behavior violates a predefined specification, CEDAR utilizes the resulting trajectory as a counter example. Employing an L* like query based algorithm, it learns a finite automaton, termed a regular restriction, that formally captures the violating behavior pattern. This learned automaton subsequently constrains the agent's policy space, precluding the recurrence of behaviors recognized by the restriction. Through iterative refinement involving counter example generation, restriction learning, and policy constraint, CEDAR facilitates the learning of policies that are demonstrably compliant with the specification while optimizing for the task objective.

### Strengths
1. Integrating active automate learning with RL for handling safely specifications is a promising direction. The idea is intuitive.
2. Unlike many LLM agents focused solely on goal completion, CEDAR treats adherence to human specifications as a core objective, providing mechanisms (Verifier, DFA intersection) for enforcement.
3. Experiments show this method is effective in following human specifications.

### Weaknesses
1. This is minor, line 112, NL2LTL, NL->LTL, please be consistent.
2. Figure 1 is good, but what are CE, EQ, MQ? Please exmaplain these concepts breifly in the catption. Readers may not have clear understanding when they read Figure 1 at the first time.
3. Scability of CEDAR is concerning but not analyzed in the paper. Please demonstrate how efficient are the DFA learner and Verifier, and what are the limitations and potential failing cases.
4. I understand this paper contributes to various domains, but I would like to see a clear description of the problem formulation (including RL, safety RL, Finite Automaton, etc). I believe this helps readers understand how your different framework components and algorithmatic advancements contribute to the whole problem. The writing is not concentrated currently.
5. Many fileds in related work are missing. Given the topic of this paper, it would be better to include related work from safty RL, counter-example learning, etc. 
6. This is minor, please cite the models and embeddings you used in the paper.

### Questions
1. The effectiveness of the framework is heavily rely on DFA learner. Do you have analysis on the correctness of LLM that answers membership queries (MQ)? What are the failing mode?
2. The limitation of this work is discussed in Appendix, but I would like to see some *examples* of these limiations. This could help the entire community.

### Soundness
3

### Presentation
2

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
This paper proposes a novel method for sequential decision making in an open world environment, when given textual description of the objective to achieve and constraints and guidance on how to achieve it. The key novelty of the proposed method, called CEDAR, is that it attempts to learn skills that are represented as DFAs. This works by first using a RAG to select the alphabet needed for achieving the given objective, then using a DFA learning algorithm using an LLM to perform membership queries (i.e., is the action consistent with the guidance given by the human) and interactions with the environment to perform equivalence queries (i.e., can I perform the planned action in the environment and a counter example if you cannot). 
The authors compared CEDAR with VOYAGER on Minecraft task and (minimally) on iThor tasks, showing improved performance and better adherence to the human’s instructions.

### Strengths
The idea is novel and exciting. I think this has a lot of promise, and the evaluation is also reasonable. 
The concept of learning DFAs as an anchor for alignment and using DFA learning algorithm is very interesting. Similar to LTL-based solutions, but the authors clarify well the differences. 
The comparison to VOYAGER is also adequate and shows good results.

### Weaknesses
While I'm excited about the general research direction, I believe the paper requires significant re-writing to be clear and self-contained. At this point, it feels to me more like a technical report or Arxiv paper than a well-written and of the academic writing level expected from a high-quality conference paper.

I list below examples:

Line 78: “… using a human-in-the-loop learning paradigm ….” – I am confused. Is there a human in the loop or is it that you use the LLM to estimate what the human would say based on the instructions given by the human?

Line 80: “The output formal specification are given to the DFA learner.” – which output forma specification? How is it given to the DFA learning? This is vague and not clear (even after reading the paper).

Line 87: “(i) a minimal canonical structure with closure under intersection …” – what do you mean by “canonical”, “minimal”, and “closure under intersection” in this context? The authors show explain more 
clearly what they mean. 

Lines 91-96:  These lines are not coherent to me, and look more like a set of keywords then actual sentences. What do you mean by “regular-language scaffold?” what is an “embodied skill” here? Or that the LLMs “ground intent”? 

Line 101: What is an “minimally adequate” teacher?

Line 120: what do you mean by “canonical temporal controllers”?

Line 147: What is are “event-monitor symbols”?

Line 154: “DFAs provide minimal, ….” – minimal in what sense?

Line 160: What is a “specification-decomposition step”?

Line 201: What is a “unified active” loop? 

See below my questions, which are also pointing to where the paper is not clear enough. 

As a minor issue, the citation format is incorrect in most of the paper, e.g., “Wu et al. (2022);” instead of (Wu et al., 2022). This is easily fixed by using the correct citation command.

### Questions
1.	In line 149 you mention a RAG being used to return a finite subset of the global alphabet. How has this RAG been implemented? Was it tuned to the task at hand in some way? How does it know which alphabet elements to return to a given task?

2.	In line 186 you write that “We analyze the erroneous DFA and reuse EQ counterexamples”. How is this done? 

3.	In line 212: how can a counterexample indicate that a symbol is missing? Is it when a state includes a symbol that is not in the set the RAG returned? 

4.	Do you assume that the environment is deterministic? That is, if a skill has successfully worked, does it mean it must also work in the next time it is applied?

5.	Can you formally define what you mean by a “verb” here? Is this an action you can tell the agent to perform?

6.	In line 248: how can we detect a “missing action”? which logs are you referring to? Aren’t you executing the “policy” directly on the environment?

7.	In the Verifier section, the authors write that when a violation between the skill and the specification is detected, then the alphabets are merged and the intersection of the skill and the specification are taken. How is this done exactly? You later explain formally how to merge skills but not how to “merge” skills and specifications. 

8.	I like a lot the fact that you can chain skills but I am not clear on who initiates this. Is this done by the LLM? Or is this done by CEDAR when something happens? How do you decompose the specification and goal given by the user to the list of skills you want to chain?

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel framework that combines LLM with formal methods, specifically using DFA to represent both skills and human constraints. This paper proposes to encode skills as regular languages using DFA, which is rigorous for agent behavior that enables verification and composition. The proposed framework uses RAG to create task-specific sub-alphabets from large action spaces, which makes learning tractable in complex environment.Through experiments, the proposed framework outperforms Voyager in terms of constraint adherence, task completion, and efficiency.

### Strengths
1. The DFA based approach provides formal guarantees about agent behaviors, which also support constraint enforcement and skill chaining. 
2. The counterexample loop allows continuous refinement and alignment with human specifications.

### Weaknesses
1. CEDAR requires more upfront LLM interactions to learn new skills, though this is amortized through reuse.
2.  Experiments used only five trials per baseline due to API costs, introducing some result variability.

### Questions
The evaluation is limited to small scale. I would like to see more analysis and experiment result.

### Soundness
3

### Presentation
3

### Contribution
2
