# Lifelong control through Neuro-Evolution

- Avg Score: 3.60
- Decision: Reject
- Scores: 4, 4, 6, 2, 2

## Abstract
Reinforcement learning (RL) under continual environmental changes has remained a central challenge for decades.
Novel designs of loss functions, training procedures and neural network architectures have not yet managed to alleviate the main mode of failure in lifelong learning: loss of plasticity.  
Here, we turn to a very different family of optimisers: neuro-evolution (NE).
Through an extensive evaluation on diverse lifelong control tasks, we see that both population-based and distribution-based approaches exhibit a remarkable ability to adapt where RL fails catastrophically.
We observe that, in the present of environmental shifts, NE naturally increases its diversity of solutions, evolving the ability to rapidly discover well-performing specialist individuals.
We propose that NE can be a promising approach towards tackling the need for lifelong adaptation and that future work should focus on the benefit of diversity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors show that neuroevolution methods perform better than deep RL methods in the context of environment shifts as they seem to suffer less from a loss of plasticity. They compare a GA and OpenAI ES against PPO and TRAC-PPO on a variety of environments: 3 classic control environments, 3 minatar environments and medium difficulty Kinetix environment.

### Strengths
The question addressed by the authors is interesting, the methodology looks sound and the results are of interest.

### Weaknesses
The paper suffers from severe editing issues: 
- it refers to appendices, but appendices are absent. In particular, the paper refers a lot to Fig. 6, but we cannot see it. It is hard to provide a thorough review in this context...
- Figure 1 is supposed to show the performance of GA, OpenAI ES, PPO and TRAC-PPO, but TRAC-PPO is not shown
- the paper mentions an anonymous github repo but the url is only available in the "pdf-with-hyperlink" format. It happens that this github repository violates anonymity rules.
- the paper is full of typos.

Beyond that, there are scientific issues:

- the authors claim that failure of RL in the context of environment shifts is due to a loss of plasticity (or they consider that this failure is by definition a loss of plasticity), but the mechanisms behind these failures are not investigated in the light of the plasticity loss literature.

### Questions
Footnote 1 is unclear: what vector do the authors add? Why?

In Kinetix, the authors "use the manually designed tasks of medium size(.)". Can they be more specific about these tasks (what do they consist of, how many are them, etc.)?

Could you expand the caption of Fig. 3, to better explain what should be seen?

Typos:
- the authors are often using the future when they should use the present. Track "will" are remove the ones that do not seem necessary.
- abstract: "in the present of" -> presence
- the Koza citation is inadequate. Please provide all the necessary fields (year, etc.)
- there are two references for the same Michael Matthews' paper. Please merge them.
- Risi et al. (2025) -> use \citep{}
- p1: avoidng
- p2: distnct
- solution This -> missing dot
- it deal(s) with
- p3, line 112: missing ref (?)
- algorithms(Chalumeau -> missing space
- p4: pose pose
- p5: 10 time -> times
- We, first, turn -> rephrase and give more context
- argueably -> arguably 
- p6: th performance -> the
- shifts In  -> missing dot
- medium size  -> missing dot
- optimiszation -> choose s or z :)
- p7: stationartiy ... the ablity ... exhibitnt ... thebottom
- p8: environements -> environments

### Soundness
2

### Presentation
1

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
Summary
This paper investigates the potential of neuroevolution (NE) as an alternative optimization paradigm for lifelong control, emphasizing its robustness to non-stationary environments where reinforcement learning (RL) often suffers from loss of plasticity. The authors benchmark several NE algorithms—Genetic Algorithm (GA), Evolution Strategies (ES)—against PPO and TRAC-PPO across multiple control domains, including classic control, Minatar, and Kinetix. The study reports that NE methods, especially GA, maintain adaptability under environmental shifts, suggesting that population diversity provides a natural mechanism for continual learning.

### Strengths
- **Timely and Conceptually Interesting Topic.**

 The paper tackles a central challenge in continual reinforcement learning—loss of plasticity—and revisits neuroevolution as a biologically inspired, diversity-preserving approach. This perspective is refreshing and well-motivated within the current debate on lifelong learning.

- **Comprehensive Empirical Evaluation.**

 The authors perform a systematic empirical comparison between NE and RL methods across three task families and various network architectures (including a Transformer-based controller), offering an informative dataset on NE’s robustness to environmental shifts.

- **Insightful Qualitative Analysis.**

 The population diversity analysis (Fig. 2–3) provides valuable intuition on how environmental variability induces diversity and triggers phase transitions in evolutionary dynamics, highlighting an underexplored mechanism behind continual adaptation.

### Weaknesses
- **Lack of Engineering and Efficiency Analysis.**

 The study does not quantify practical aspects such as wall-clock training time, computational cost, or memory footprint. Matching algorithms only by environment steps ignores the substantial difference in hardware utilization between population-based NE and gradient-based RL. Without these measurements, it is unclear whether NE offers any realistic advantage in deployment scenarios.

- **Limited Task Coverage and Insufficient Hyperparameter Study.**

 The benchmark suite is narrow and mostly low-dimensional. Complex control domains (e.g., DM Control, MetaWorld) and visual tasks are missing. In addition, NE methods use mostly default hyperparameters with minimal tuning; no ablation or sensitivity analysis is provided. As a result, claims of generality and robustness remain preliminary.

- **Insufficient Methodological Formalization.**

 The paper provides only high-level textual descriptions of GA and ES without formal equations, pseudocode, or notation. Readers unfamiliar with NE cannot reconstruct the optimization objectives, mutation/selection mechanisms, or implementation details. The lack of mathematical rigor and algorithmic clarity weakens reproducibility and undermines the technical contribution.

### Questions
See the weaknesses above.

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
3

### Summary
This paper presents a compelling comparative study between Neuroevolution (NE) and Reinforcement Learning (RL) in the context of lifelong learning, where agents must continuously adapt to environmental shifts. The central thesis is that population-based NE methods inherently possess a superior ability to maintain plasticity compared to gradient-based RL, which often suffers from a "loss of plasticity" in non-stationary environments. The authors benchmark two NE approaches (a Genetic Algorithm and Evolution Strategies) against PPO and a lifelong variant (TRAC-PPO) across three diverse task families: classic control, simplified Atari (Minatar), and a complex physics-based control suite (Kinetix) featuring a Transformer-based policy.  The authors support their claims with an analysis of population diversity, suggesting that environmental shifts naturally promote diversity, which acts as a buffer and facilitates adaptation, sometimes through an abrupt "phase transition" to a better solution.

### Strengths
1. The paper focuses on lifelong plasticity, which is a critical issue for embodied AI and robotics. 

2. The empirical study is thorough. Using three distinct task families with different network architectures (feedforward, CNN, Transformer) and challenges (sparse rewards, complex dynamics, perceptual shifts) provides a comprehensive evaluation that is rare in the literature. The inclusion of Kinetix, a high-dimensional physics-based environment, is particularly valuable for the robotics community.

3. The consistent outperformance of NE, especially the GA, in the face of environmental shifts is a strong and well-supported result. It effectively challenges the prevailing RL-centric paradigm for continual learning and successfully argues for NE as a powerful, hyperparameter-robust alternative.

### Weaknesses
1. The identified weakness of the GA in sparse reward environments (MountainCar, some Minatar games) is a major practical limitation. For robotics, where informative rewards are often hard to engineer, this is a significant drawback. The paper would be stronger if it proposed or discussed potential solutions (e.g., hybridizing NE with novelty search or quality-diversity methods such as MAP-ELITES) to mitigate this.

2. The complete failure of ES on the Transformer-based Kinetix tasks is noted but not sufficiently investigated.

### Questions
1. Your results suggest a "best of both worlds" approach. Have you considered or do you plan to investigate hybrid algorithms? such as evolutionary reinforcement learning?

2. Your diversity analysis focuses on genotypic (parameter) diversity. In embodied intelligence, behavioral diversity is often more meaningful. Did you observe a correlation between parameter diversity and behavioral diversity in your populations? 

3. In the Minatar and Kinetix experiments, you use a fixed task sequence. Did you observe any evidence of catastrophic forgetting in NE, where performance on a previous task drops after a shift? How does the implicit curriculum induced by your task sequence interact with the evolutionary process?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes that neuroevolution outperforms traditional reinforcement learning at lifelong learning tasks and provides empirical evidence to back up this claim.
The study compares the genetic algorithm used in Such et al. 2018, the evolution strategy used in Salimans et al. 2017, and PPO, at how well they respond to environmental changes when optimising neural network policies for a suite of control tasks.
The results show that the evolutionary algorithms (EA) do outperform PPO in all of the experiments.
Additionally, an analysis into how the diversity and size of the populations produced by the EAs correlates with performance is carried out.

### Strengths
1. The current failure of foundational models to perform continual learning is one of the most pressing current issues in the field of AI, thus the line of inquiry explored in this paper is very important.
2. The findings are interesting and do show that the neuroevolution algorithms outperform PPO at a substantial number of lifelong learning tasks.
3. Multiple experiments were ran for each algorithm and task suggesting robust results.
4. As far as I am aware this is the first study of its kind directly comparing RL and NE algorithms in lifelong learning tasks (although I do not know all the literature in depth).
5. I think it is wise to compare using the number of environment steps as opposed to other metrics, such as wall clock time.

### Weaknesses
1. I think more algorithms should have been evaluated in this study in order to provide a more comprehensive comparison. It is unclear whether the relatively poor performance exhibited by PPO is reflective of RL algorithms as a whole or just with this particular algorithm. Also, it seems unusual that a paper studying the plasticity of neuroevolution algorithms does not also include plastic and hebbian neural network algorithms, such as those surveyed in Soltoggio et al 2018. It would have also been interesting to evaluate Novelty Search in order to compare diversity across the behavioural space rather than just the parameter space. Without including these other algorithms the comparison feels incomplete.
2. The paper is written quite carelessly and with many mistakes. There are too many typos to note and many grammatical errors; there are missing references in the bibliography, such as Such et al. 2018; the authors state that both CMAES and TRAC PPO will also be evaluated but these results are not reported; there are missing axes on some of the plots making them unclear; the appendix is referred to but has not been provided; and certain points are repeated multiple times.
3. The abstract claims: 'We observe that, in the presence of environmental shifts, NE naturally increases its diversity of solutions...'; however, the results don't necessarily backup this claim. In fact, the diversity plots for Cartpole in Figure 2 illustrate an acute _decrease_ in diversity at the point that the environment shifts occur, and the other two diversity plots show no increase in diversity with environmental shifts. It is true that Figure 3 shows a marked increase in diversity at generation 9000 correlating with an increase in fitness for the larger population; however, this does seem rather arbitrary considering there have been multiple environmental shifts up until this point and we do not see this jump in those cases. Also, it seems suspicious that all of a sudden the fitness completely stabilises at this arbitrary point in the run despite being highly variable up until this point.
4. It is true that the plots in Figure 3 show a large difference in fitness after the 10000 generation mark between the two population sizes, and that there is also a higher diversity in the larger population; however, this is not necessarily proof that the diversity itself is solely (or at all) responsible for the fitness increase.

### Questions
1. Why do you believe a phase transition occurred at generation 9000? Is there anything special about this point? Why do you think the fitness stabilised at this point in particular?
2. Why were more algorithms not compared?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes NeuroEvolution (NE) as a promising approach for lifelong adaptation due to its ability to increase the diversity of solutions compared to reinforcement learning. Using empirical results, the authors propose NE as an alternative to RL for lifelong learning. The main arguments being that NE is better in terms of plasticity, and that it is better method to maintain diversity.

### Strengths
The overall idea is clear, and the authors consider the very important problem of lifelong learning from a non-RL perspective.

### Weaknesses
The main issues with the paper is a lack of clarity when it comes to describing the various quantities like diversity, plasticity etc. Due to this, the results are not easy to interpret and the question of what exactly the authors are investigating remains unclear. For instance, it is already known that NE maintains better diversity of solutions/policies (although I am unsure whether this is what the authors refer to when they mention diversity). Apart from this, there is also some ambiguity when it comes to what the main claim is – for instance, it is mentioned that NE is proposed as an alternative to RL. However, it is also mentioned in the discussion that NE is not meant as a replacement to RL, and is meant to be complementary. There are also several typos, grammatical and punctuation errors throughout the manuscript. Some citations are not listed properly (eg: Koza on pg 1). It would be good to clearly summarise all claims at the end of the Introduction. Also, I notice some sections as well as figure references (like fig 6) are present in the appendix. Ideally these should be self contained. With reference to Fig 3, please name the sub figures as (a), (b) etc.,

### Questions
What is formally meant by plasticity and diversity?

In line 68, what is meant by “abilities”? What is the context?

NE is designed to be more robust to environmental shifts. So isn’t it expected to perform better?

### Soundness
2

### Presentation
1

### Contribution
1
