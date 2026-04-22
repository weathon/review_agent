# A Benchmark for Self-Evolving Agents via Experience-Driven Lifelong Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
As AI advances toward general intelligence, the focus is shifting from systems optimized for static tasks to creating open-ended agents that learn continuously and adapt autonomously from experiences. This vision emphasizes long-term memory, self-driven exploration, persistent experience retention, and the internalization of knowledge into intuitive behavior as key to enabling self-evolving agents through experience-driven lifelong learning (ELL). In this paper, we introduce StuLife, a novel benchmark designed to evaluate whether current models can exhibit these foundational capabilities of ELL. Particularly, StuLife simulates a student's holistic college journey, from enrollment to academic and personal development, across three core phases and ten detailed sub-scenarios. StuLife is designed around four key paradigm shifts: From Simulation to Reality, From Passive to Proactive, From Context to Memory, and From Imitation to Learning. In this dynamic environment, agents must acquire and distill practical skills and maintain persistent memory to make decisions based on evolving state variables (e.g., resource availability and time). Critically, these agents are also expected to demonstrate intrinsic motivation by setting their own goals and initiating actions without external prompting. To this end, StuLife provides a comprehensive evaluation platform featuring our novel metrics (e.g., StuGPA) to specifically assess these critical capabilities. Our evaluation reveals that even the best model, GPT-5, scores only 17.9/100, revealing a vast gap toward AGI, demonstrating fundamental deficiencies in retaining long-term memory and acting with self-motivated initiative. Beyond evaluating state-of-the-art LLMs on the StuLife, we also explore the role of context engineering in advancing AGI. Our results suggest that optimizing how we guide models may be as crucial as improving the models themselves, positioning context engineering as a key enabler of progress toward AGI.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces StuLife, a benchmark for evaluating self-evolving agents through assessing the capabilities of experience-driven lifelong learning methods.

### Strengths
StuLife offers a comprehensive and realistic evaluation framework that includes four key paradigm shifts: From Simulation to Reality, From Passive to Proactive, From Context to Memory, and From Imitation to Learning.

### Weaknesses
1. Why are there only large models as agents? Many existing agents in the field of embodied AI can also achieve the ability to self-evolve, but this paper does not consider work related to self-evolving agents. These self-evolving agents also belong to experience-driven lifelong learning methods.
2. There are doubts about the fairness of the comparison or the conclusions of this benchmark.

### Questions
1. Why are there only large models as agents? Many existing agents in the field of embodied AI can also achieve the ability to self-evolve, but this paper does not consider work related to self-evolving agents. These self-evolving agents also belong to experience-driven lifelong learning methods.
2. How does the benchmark ensure that StuGPA and other metrics isolate lifelong learning from task-specific memorization?
3. Can the authors provide a deeper analysis of why models fail in proactive tasks, such as the underlying causes for low PIS scores?
4. How was bias controlled in the data generation process, particularly when using LLMs for creating tasks and solutions?
5. What measures were taken to ensure fair comparison between models with varying architectures and training data?
6. How does the benchmark account for scalability and adaptability to unseen tasks, ensuring it evaluates general self-evolution rather than overfitting to the scenario?
7. Why are the evaluation metrics fair? Why can they be applied to all tasks?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a benchmark for evaluating "self-evolving agents" that learn from experience. The benchmark -- StuLife -- simulates a college education, and involves several tasks presented to a language model in sequence. Experiments with a variety of language models show that such models can systematically struggle in certain tasks.

### Strengths
- The experiments provide results with various models, and provides a thorough breakdown of different measurements of model quality with respect to the benchmark.

### Weaknesses
- The writing in this paper is oftentimes unclear and not well supported
- Unclear value in the benchmark. While it is motivated from a good place, it is almost entirely heuristic and does not clearly demonstrate that it is a useful benchmark to drive real-world progress.
- Several of the properties and motivating principles for the benchmark are contradictory.

### Questions
- Table 2: What does it mean for a task type to be self evolving? The authors should define self-evolving, as it is used throughout the paper and the title but never defined. What does it mean for a benchmark to evaluate self-evolving?
- Section 3.2: In what way do the agents in this task learn from experience? It seems like all the models are frozen, and only used for inference?
- "From simulation to reality": This is not an accurate statement as the proposed benchmark is still a simulation, not reality.
- "From imitation to learning": As stated, this is a false dichotomy. Imitation is a form of learning. While the authors go on to clarify that the agent learn generalizable skills, the nuance of this is not at all reflected, nor does it disqualify the learning of such skills from imitation.
- "From context to memory": Again this feels like an orthogonal dichotomy. Given that you are evaluating language models, their context is their memory and can, in principle, be made arbitrarily large.
- "From passive to proactive": Several reinforcement learning environments already involve such distinctions.
- Table 1: What does it mean for a task to require "long term memory" or "self-motivation"? 
- Table 4: Why is it that several modifications (such as vanilla RAG) actually worsen the model according to these benchmarks?
- StuLife is often referred to as a dataset, but you discuss it like an environment. It is never exactly specified what it is, or how the agent is interfaced with it. For example, how is time simulated?
- "but still fall short in supporting lifelong learning, skill abstraction, and longitudinal growth." 
  While statements like this are found in the results, they are not fully justified, explained or investigated in detail How are these different axes actually compared?
- Other statements are made without explanation: "rather than tracking continuous growth or self-evolution"
  What does continuous growth mean in this context? How about "self-evolution"?
- There is little intuition for a competitive reference level of performance. Is it possible to evaluate a human on this benchmark as a hypothetical ideal for each metric, like "GPA"?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a new, quite sophisticated and complex, benchmark for evaluating whether LLMs can do long-horizon, continual learning tasks, by producing a sort of text-adventure simulation of a college environment. Current LLMs are evaluated on the environment and found to be lacking some aspects that it examines.

### Strengths
This is a well-thought-out, complex, and impressively complete environment. (As far as I can tell.) The paper does a good job of interrogating some of the requirements of a more general intelligence and the environment is a novel and interesting instantiation of such an environment. The evaluation of existing LLMs on the benchmark leads to some interesting results. I think the community needs more tasks like this, so despite my reservations (of which there are many, see below) I am currently mildly in favor of acceptance.

The paper is also generally well-written and clear.

### Weaknesses
The primary flaw with this benchmark, as with all public benchmarks, is Goodhart's law. Benchmarks are great when thoughtfully constructed, as this one seems to be. But it is not hard to imagine that, the minute this one hits the street, some team at OpenAI will busy themselves with producing a million example interactions hand-engineered to cover every conceivable StuLife case, train on them, and then declare victory when their model does well. This is what they have done with, e.g., the math olympiad. Such "advances" are no doubt useful, but they're also very expensive and creative ways to miss the point. Anyway, I hope this benchmark lasts longer than usual, but my guess is it will be eventually engineered into oblivion.

The authors should be aware that not everybody thinks an LLM could *ever* be an AGI, no matter how well it does at any language tasks, because language is not the whole of intelligence. It's a thing an intelligence does, but not what an intelligence is. Some people think it does the whole thing, and some people don't. 

Similarly, it's really not clear that ideas like "self motivation" are meaningful outside of having a body that expends energy when you do things. A textual facsimile of motivation is just that. So I think here one ought to be a little humble with terminology, especially in a scientific paper.

Similarly, the buzz-wordy summaries in the little gray text boxes are suitable for LinkedIn posts and pitch decks, but not for a serious scientific paper.

### Questions
None

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces StuLife, a comprehensive benchmark for evaluating Experience-driven Lifelong Learning (ELL) in AI agents. The benchmark simulates a complete college student journey across three phases (In-Class, Daily Campus, Examination) and ten interconnected scenarios. The key innovation lies in four paradigm shifts: (1) From Simulation to Reality—realistic university experiences, (2) From Imitation to Learning—requiring skill abstraction rather than recall, (3) From Context to Memory—demanding persistent long-term memory, and (4) From Passive to Proactive—requiring intrinsic motivation and self-initiated actions.
The authors evaluate 14 state-of-the-art LLMs and find strikingly poor performance: even GPT-5 achieves only 17.9/100 on the StuGPA metric. They identify critical failures in long-term memory retention (LTRR scores mostly <7%) and proactive initiative (PIS scores <5% for most models). Context engineering approaches (proactive prompting, skill augmentation, memory systems) provide modest improvements but cannot overcome fundamental architectural limitations of stateless LLMs.

### Strengths
- The benchmark is interesting and timely, moving away from static task evaluation and providing a life-long learning benchmark that is close to real world scenarios.
- The evaluation is thorough: there are 14 SOTA LLMs as baselines, and context engineering analysis provide insights for future improvement.
- The variety of tasks within the benchmark is diverse, but also well structured and interconnected.
- The evaluation metrics used are novel and important, capturing nuance aspects of autonomous learning like proactiveness and long term memory retention, beyond simply accuracy.
- Experiments are very well documented, contributing to exceptional transparency and reproducibility.

### Weaknesses
- The setting can be too domain specific, only simulating a university student's life. Although claimed to be autonomous learning, 

- There is no human performance as baseline, raising concerns in metric calibration, or what score might be considered "good".

- The way agents "learn" during their life time remains in context and through prompt, which raises concern if this can be considered as life-long learning or has fundamental skill internalization ability.

- All tasks in the benchmark are generated by LLMs, which may create bias in tasks, and advantage for LLMs used for task generation.

### Questions
- Is there evidence that simulating a university student's life provides domain-general insights on continual learning capabilities?

- Can you provide further information and justification on how should researchers evaluate the scores achieved by agents? For example, GPT-5 achieves 17.9 out of 100, what would that number lie in the score distribution from real students?

- Have you considered weight updates methods (EWC, LoRA etc.) to consolidate skills for your experiment? Without this, isn't your benchmark still measuring stateless agent's ability for in context learning and tool use for complex projects?

- Have you compared agents' performance and failure patterns on tasks designed by different LLMs or human? Do you think potential circular evaluation or answer leakage is possible?

### Soundness
3

### Presentation
3

### Contribution
3
