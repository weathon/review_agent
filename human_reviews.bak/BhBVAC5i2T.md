# Meta-Referential Games to Learn Compositional Learning Behaviours

- Decision: Reject
- Scores: 6, 6, 5, 5

## Abstract
Human beings use compositionality to generalise from past to novel experiences, assuming that past experiences can be decomposed into fundamental atomic components that can be recombined in novel ways. 
We frame this as the ability to learn to generalise compositionally, and refer to behaviours making use of this ability as compositional learning behaviours (CLBs). 
Learning CLBs requires the resolution of a binding problem (BP). 
While it is another feat of intelligence that human beings perform with ease, it is not the case for artificial agents. 
Thus, in order to build artificial agents able to collaborate with human beings, we develop a novel benchmark to investigate agents’ abilities to exhibit CLBs by solving a domain-agnostic version of the BP. 
Taking inspiration from the Emergent Communication, we propose a meta-learning extension of referential games, entitled Meta-Referential Games, to support our benchmark, the Symbolic Behaviour Benchmark (S2B). 
Baseline results and error analysis show that the S2B is a compelling challenge that we hope will spur the research community to develop more capable artificial agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the Symbolic Behaviour Benchmark (S2B), a meta-learning benchmark designed to test agents’ abilities to generalize compositionally within single episodes through Compositional Learning Behaviours (CLBs). S2B utilizes Meta-Referential Games, an extension of referential games embedding a Binding Problem (BP), challenging agents to dynamically bind and recombine information from limited examples. Baseline results on multi-agent reinforcement learning (MARL) and large language models (LLMs) suggest S2B presents a challenging benchmark for current AI capabilities.

### Strengths
- Originality: The benchmark provides a unique approach to evaluating compositionality in a few-shot learning context, moving beyond static generalization and introducing a dynamic, episode-based test.
- Quality: The benchmark and supporting experiments are rigorously defined, leveraging established methods like referential games and meta-learning frameworks to create a novel testing ground for CLBs.
- Significance: S2B addresses an important challenge in AI—evaluating compositional learning in dynamic environments—which could spur new research into agent architectures and learning strategies.

### Weaknesses
- Presentation: The structure of the meta-RG could be clarified by using a single detailed example to illustrate an episode from beginning to end, making the compositional requirements more apparent.
- Experiments: Evaluating LLMs within this benchmark may not be entirely fair, as the task setup deviates from natural language processing, which LLMs are primarily designed for. The benchmark’s symbolic structure lacks the natural language context that LLMs are optimized to process, raising concerns about using LLMs without modifications to better align with symbolic input.
- Human Baseline: The lack of a human experiment or baseline raises questions about the difficulty of the benchmark for agents versus human performance. Introducing human testing on S2B could provide additional insights, highlighting any inherent complexity in CLB learning and enabling better evaluation of AI performance relative to human compositional understanding.

### Questions
- Could the authors clarify how the vocabulary permutation scheme is implemented and whether it impacts the agents’ learning or communication strategies in any unintended ways?
- How sensitive are the experimental results to variations in the number of symbolic dimensions (Ndim) or other hyperparameters of the SCS representation? Additional experiments on this could offer more insight into the scalability of the benchmark.
- Were any human experiments conducted to provide a baseline for performance on S2B? Including human data could offer a valuable perspective on the complexity of the benchmark.

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
2

### Summary
This paper proposes a benchmark called S2B that can evaluate the ability of artificial intelligence agents in combinatorial learning behaviors (CLBs). S2B uses Meta-Referential Games as the basic framework and uses the SCS method to represent stimuli. This paper uses S2B to test the CLBs of multi-agent reinforcement learning models and LLM.

### Strengths
- The paper introduces the S2B benchmark, designed to evaluate the combinatorial learning behaviors (CLBs) of AI models.
- It proposes the SCS method for representing stimuli in a domain-independent manner, avoiding reliance on specific modalities like visual, verbal, or auditory information.
- Meta-Referential Games are presented as the primary framework within the S2B benchmark, aiming to assess agents' capabilities in symbolic learning and combinatorial learning behaviors (CLBs).

### Weaknesses
- Insufficient validation of domain-agnostic BP. While the S2B benchmark and meta-referential game frameworks intend to construct domain-agnostic BP, there is a lack of sufficient experimental data to validate their applicability in various domains or applications. Whether this benchmark and framework can be extended to different fields such as vision and language still needs to be further verified.
- Terminology and lack of concrete examples: The paper contains a large number of terms (such as CB, CLB, BP, support stage, query stage, etc.), although their concepts are mentioned in the article, the concepts are relatively vague and lack simple examples to help readers understand. It may not be intuitive for readers who are new to these concepts.

### Questions
- In Figure 2, it is not clear what Latent Stimuli is.
- In line 267, the standard deviation sampling interval of the Gaussian distribution is not explained.
- In 4.1 and 4.2 section, there are lacks of details and examples: the representation and meaning of the symbol combination, the specific process of the experiment are not explained in detail, which confuses some readers when understanding the experimental design. While the interaction between multiple agents is mentioned, no specific examples are provided to illustrate how the messaging and identification tasks are performed, which can lead to barriers to understanding.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents a novel benchmark, the "Symbolic Behaviour Benchmark", for evaluating compositional learning behaviors. The study introduces Meta-Referential Games (Meta-RGs), a meta-learning extension of referential games, to test agents' ability to solve a binding problem that is crucial for learning CLBs. This benchmark emphasizes symbolic receptivity and constructivity, encouraging agents to develop compositional generalization skills while interacting with each other.

### Strengths
Relevance: I think this paper tries to address an important problem in that it proposes a benchmark in which succesful behavior means that agents learned to generalize compositionally, instead of only having learned to generalize compositionally. The problem of compositionality and compositional generalization is of general interest to the community. Thus, this benchmark might be generally useful.

Novelty: The introduction of S2B and Meta-RGs adds depth to the compositionality field by pushing beyond mere combinatorial generalization (CG) to a meta-learning context where agents adapt to unseen symbolic structures. The Symbolic Continuous Stimulus (SCS) representation is an innovative method to instantiate a BP, ensuring agents must infer structures over multiple observations, aligning with the real-world learning constraints in open-ended contexts.

Analyses: The experiments establish state-of-the-art limitations effectively, showing that both MARL agents and LLMs struggle with this benchmark, thus illustrating its difficulty and relevance. The experiments are clear and I like that they always come with a hypothesis followed by the results.

Introduction: The first few sections, i.e. overview of the problems of systematicity/compositionality, lingustic compositionality, and compositionality are helpful.

### Weaknesses
Accessibility: The meta-learning setup, combined with the specialized SCS representation, might limit accessibility and reproducibility. The SCS's construction, particularly the Gaussian kernel setup, could be further detailed, I didn't quite get what was going on there. The writing is generally quite verbose and I had really some difficulties in following along. That there are many abbreviations throughout doesn't really help here either. Some of the figures are very small an complicated to read. This makes it again hard to follow.

Scope: The focus on RL and MARL agents is suitable, but extending evaluations to to other models could have been fun. I would have really liked to see something on further multi-modal models here. Many of the evaluations are currently in the simplest form, i.e. the basic form of the proposed game, the most common agents playing them, as well as LLMs without any modifications to the standard prompts. This makes it a little unclear what exactly makes the difference in agents' inabilities to do these games well.

CLB definition: While the paper defines CLBs distinct from CBs, it would benefit from clearer operationalization criteria to guide comparisons. Are there any performance metrics beyond linguistic compositionality and RG accuracy?

LLM behavior: The below-chance LLM performance is interesting but could be further analyzed. 

Fit: It felt a little like this paper would fit better to a more targeted conference but I can be convinced.

### Questions
What do we learn from this? Essentially most of the results just point to an inability of different agents to learn to generalize compositionally. Is the idea that the community should now focus on getting these agents to be better on the benchmark? How would the authors imagine such progress?
I'm not quite sure but what is meant by domain-agnostic BP?
Perhaps adding short summaries to every section about what the main message is could help?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper introduce a novel benchmark aiming to assess the ability of artificial agents to meta-learn behaviors that leverage the compositional nature of their sensory inputs. In this benchmark, two collaborative agent strive to meta-learn to solve referential games. In each episode, the two agents first execute a series of referential games that take in samples endowed with a specific compositional distribution. Then, they test their ability to generalize to novel samples from the same distribution. The global objective is to meta-learn this task for any compositional distribution of inputs. Experiments support that current methods largely fail at this task.

### Strengths
This benchmark introduces a challenging and important problem for current methods. Experiments support the claim of the paper.

### Weaknesses
It is hard to assess the novelty as it is stated in the paper that there is a previous version of the benchmark (published ?). The paper is hard-to-follow and often confusing.

### Questions
The paper could likely be simplified:

The paper introduces a lot of concepts, the abstract refers to compositionality, binding problem, emergent communication, human-agent collaboration and meta-referential game. Also few-shot learning in the introduction.
- Human-agent collaboration is only used as a high-level motivation of the work, so its reference in the middle of the abstract and line 52 mostly confuse the reader with works including humans in the loop.
- Emergent Communication: In the abstract, it is unclear what "Emergent Communication" refers to. Is it a framework ?
- The binding problem (BP): The relation between CLB and binding problem is unclear in the abstract, not very clear in the introduction, and  get clearer in Greff et al. (2020). Most of the statements related to BP are unclear: that an "inherent BP" must be solved be agents to exhibit CLB. "Solving the BP instantiated in such a context, i.e. re-using previously-acquired information in ways that serve the current situation" is done by all learning artificial agents. What does it mean to "instantiate a BP" ? Do any representations of any latent factors instantiate a BP ?
I overall don't understand why this paper needs to talk about BP. This paper is about meta-learning behaviors that leverage the compositional nature of their inputs. 

Some critical concepts are unclearly defined: systematicity, ZSCT (l 174), symbolic space (l195), EoA (l379), posdis (l163) and bosdis (l163). It would help if compositionality measures were briefly explained (posdis, bosdis). 

I do not understand how the object-centric variant of the representation (for the listener) is built. That should be clarified in Section 3.

Minor points:
- generalise -> generalize
- l391: axises -> axis
- l87: What is "a semantic domain that can be probed and queried"
- Figure 2 and Figure 4 are hard to read: the fontsize is too small and the bold text is hard to read.
- l256: partitionaing
- l334: what does it mean do bridge the gap between two conditions "Hill-RSC and Chaa-RSC".
- l349: What is the core memory module ?
- The beginning of 4.2 about meta-RG is very clear and could probably go to Section 3.
- l285-286 : It is confusing as the described set of stimuli used is misaligned with what is described in Section 2. The speaker does not receive the same set of stimuli in both.
- Section 4.2.1: Is it using the rule-based speaker agent ? Is it normal there is no 4.2.2 ? Then 4.3. is using the Posdis-speaker agent ?

### Soundness
3

### Presentation
1

### Contribution
2
