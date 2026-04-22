# The Collaboration Gap

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2

## Abstract
The trajectory of AI development suggests that we will increasingly rely on agent-based systems composed of independently developed agents with different information, privileges, and tools. The success of these systems will critically depend on effective collaboration among these heterogeneous agents, even under partial observability. Despite intense interest, few empirical studies have evaluated such agent–agent collaboration at scale. We propose a collaborative maze-solving benchmark that (i) isolates collaborative capabilities, (ii) modulates problem complexity, (iii) enables scalable automated grading, and (iv) imposes no output-format constraints, preserving ecological plausibility. Using this framework, we evaluate 32 leading open- and closed-source models in solo, homogeneous, and heterogeneous pairings. Our results reveal a “collaboration gap”: models that perform well solo often degrade substantially when required to collaborate. Collaboration can break down dramatically; for instance, small distilled models that solve mazes well alone may fail almost completely in certain pairings. We find that starting with the stronger agent often improves outcomes, motivating a “relay inference” approach where the stronger agent leads before handing off to the weaker one, closing much of the gap. Our findings argue for (1) collaboration-aware evaluation, (2) training strategies developed to enhance collaborative capabilities, and (3) interaction design that reliably elicits agents’ latent skills, guidance that applies to AI–AI and human–AI collaboration.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the collaboration ability of multiple LLMs on tasks with communication challenges. The authors develop a maze environment. Two LLM agents involve in the system and each of them can only observe part of the maze information. They conduct experiments across multiple choices of open/closed-source LLM models, and suggests that a collaboration gap exists. The authors further provides more additional experiments on cases when two LLMs come from different model family and the relay inference setup.

### Strengths
Multi-agent collaboration is indeed a more and more important topics. This paper contributes to identifying key potential issues occurred in LLM cooperation tasks by investigating the LLM performance on a maza task. The paper writing is easy to follow, and the results are explained clearly. The results indeed demonstrates the existence of collaboration gap. I found some experiment results are interesting (e.g. the relay inference part).

### Weaknesses
1. My main concern is that the contribution in this paper, although insightful, may not reach the threshold of the acceptable of this top-tier conference. The main contribution is limited in identifying the collaboration gap, but the authors did not make progress beyond that. I believe some contribution on algorithm design to close the collaboration gap would be helpful and makes the paper stronger.

2. The experiment in this paper is limited in the maza task. It is enough to suggest "the collaboration gap" indeed exists but it may not be sufficient to suggest that such an issue appears widely in more practical domains where cooperation between LLMs are required. For example, a more practical task is applying LLM for software engineering, and it is also a valid scenario to evaluate LLM cooperations. It would be better to provide results in a more diverse range of tasks in the paper.

### Questions
I think the magnitude of the gap depends on how well the prompt engineering on the system/user message is. The authors suggest that one challenge in collaboration (which leads to the gap) is that the LLMs need to exchange their messages and ground their understanding "on the same page". 
At least in this maze task, I think this issue can be partially solved by providing more detailed instructions in system/user prompts, such as some unified symbols/rules for communication or a clear step-by-step guidance on how to exchange information. 

I'm curious about the potential of closing the gap through careful prompt engineering. Concretely, with appropriate system/user prompts, would it be possible to eliminate this collaboration gap? If the gap can be closed by prompt engineering, would it still be reasonable to claim the gap exists?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a collaborative maze-solving benchmark to evaluate AI-AI collaboration among heterogeneous agents with partial observability. The authors evaluated 32 open- and closed-source models in solo, homogeneous, and heterogeneous pairings. Their primary finding is the "collaboration gap": models that perform well individually often experience a substantial performance drop when required to collaborate with an identical copy of themselves. They also observed that collaborative performance is heavily influenced by which agent starts the task. To address this, the paper introduces "relay inference," a strategy where a stronger agent initiates the interaction to "prime/ground" the rollout before a weaker agent takes over, which was shown to significantly boost collaborative performance and close much of the gap. The key contributions are formally defining the collaboration gap, analyzing heterogeneous collaboration dynamics, and proposing the effective relay inference strategy.

### Strengths
- The analysis of homogeneous, same-family (different strengths), and cross-model heterogeneous collaboration provides cool and valuable insights into agent interaction dynamics.

- The overall scope of the evaluations conducted appears quite comprehensive, covering many models and various collaboration settings.

- The proposed collaborative maze-solving benchmark is novel, isolates collaborative capabilities, and imposes minimal output constraints, which is a strong methodological contribution.

### Weaknesses
It is unclear whether LLM collaboration failures in this specific maze task translate to an inability to collaborate effectively in more naturalistic use cases, such as coding tasks.

Human performance on these specific maze tasks is missing, which makes it difficult to fully substantiate claims about the LLM "collaboration gap."

The reliance on the autograder is questionable, as it may introduce systemic biases or errors compared to enforcing a deterministic output format for all models.

The complexity of the ASCII map visualization raises concerns about whether perceived "collaboration failures" are actually failures of perception that could be mitigated with better data representation (e.g., natural language or tool use).

The authors should clarify the methodology of "at least 100 rollouts" by standardizing the exact number of runs across all different model types and evaluation conditions.

The discussion section does not fully justify why mazes are a good lower bound for complex collaboration, especially since the gap might be closed with different prompting or tool-use strategies. In the discussion, the authors talk about how the gap might be wider in more complex cases, but those are cases we have data for, whereas mazes seem like an esoteric example that likely doesn’t appear in the training set that much. 

The section detailing the "relay inference" strategy is conceptually confusing and requires additional clarification on the mechanism and implementation.

The authors should quantify some of the interesting qualitative representational details observed during the model dialogues.

### Questions
- How do the authors think the collaboration failures observed in this esoteric maze might reflect failures in more representative, naturalistic language environments, such as collaborative coding or long-horizon planning tasks?
- Can the authors conduct an ablation study where the ASCII map input is replaced with a more structured, natural language (or JSON) representation to determine if performance gains are due to improved collaboration or simply better perception/parsing of the environment?
- Why not impose a strictly defined, parseable output format for the agents' moves (e.g., a specific JSON/YAML structure) to enable deterministic grading, thereby eliminating the reliance on an LLM autograder and its inherent biases?
- Can the authors include a baseline measurement of human performance on this identical distributed maze-solving task to properly contextualize the LLM failure rates?
- How many rollouts (e.g., exactly 100 or 150) were done for all model types and evaluation conditions, rather than the ambiguous "at least 100 rollouts"?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors introduce a new maze based collaborative test for RL algorithms and test it on LLMs. They show that their test correlates with model "power" and provide suggestions on how to make LLMs more collaborative.

### Strengths
# originality

Evaluating collaboration between AI agents has been an ongoing area of research for decades. This appears to be a variant of the overcooked test [1], but with lower complexity and partial information. As they do not discuss how this fit into the literature it is hard to evaluate the originality. 


# quality

The experiments appear to have been done well, and they test a large number of LLM variants. The use of an automated LLM as grader is concerning, but they discuss the issues with it and seem to have done things correctly. The lack of code release though means I cannot verify these statments.

# clarity

The paper is clear, giving good examples and explains what is happening well. I think putting a full dialogue somewhere prominent (maybe a section in the appendix) would help as I had to dig around in the appendix to understand exactly what transpires in a run.

# significance

As mentioned above I have concerns about how this paper engages with the literature. Taking it on face value there are better and much more established tests for AI agent collaboration (Hanabi, Diplomacy, Overcooked, ...), so I'm not sure if this adds much to the disucssion

[1] Carroll, M., Shah, R., Ho, M.K., Griffiths, T., Seshia, S., Abbeel, P. and Dragan, A., 2019. On the utility of learning about humans for human-ai coordination. Advances in neural information processing systems, 32.

### Weaknesses
The 6x6 maze seems very small, A* can solve that trivially.

The use of a grader AI adds an additional level of complexity to the experiment.

The authors don't appear to engage with the SOTA in collaborative AI, instead focusing on LLMs only.

Most of my other concerns are in the other sections, I think if the paper significantly toned down it's claims it would be publishable.

### Questions
I'm concerned about the reproducibility of this experiment. Why didn't the authors include the code with the submission? This is a complex simulation, code is need for other groups to reproduce the results. Will the the authors do a full code release if the paper is published? The reproducibility statement only talks about the license not access.

I find the use of anthropomorphic language (e.g. on line 302 "The stronger o3 immediately seeks to") concerning. Are the authors arguing that LLMs have goals? I think discussing what is happening with more theoretically grounded language would improve the paper.

Hanabi[1] is the standard for collaborative text based AI research. Why is this maze approach better and how do results compare to Hanabi? More generally was AI collaboration in non-LLM settings discussed at all in the paper?

How do these models perform when paired with non-LLM partners? Training an RL agent to solve these mazes seems trivial.

Can the models share their maps to each other? Was their any filtering of messages?

[1]Bard, N., Foerster, J.N., Chandar, S., Burch, N., Lanctot, M., Song, H.F., Parisotto, E., Dumoulin, V., Moitra, S., Hughes, E. and Dunning, I., 2020. The hanabi challenge: A new frontier for ai research. Artificial Intelligence, 280, p.103216.

### Soundness
2

### Presentation
3

### Contribution
2
