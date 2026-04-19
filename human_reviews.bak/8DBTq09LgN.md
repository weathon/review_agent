# Synthesizing Programmatic Reinforcement Learning Policies with Large Language Model Guided Search

- Decision: Accept (Poster)
- Scores: 6, 8, 6

## Abstract
Programmatic reinforcement learning (PRL) has been explored for representing policies through programs as a means to achieve interpretability and generalization. Despite promising outcomes, current state-of-the-art PRL methods are hindered by sample inefficiency, necessitating tens of millions of program-environment interactions. To tackle this challenge, we introduce a novel LLM-guided search framework (LLM-GS). Our key insight is to leverage the programming expertise and common sense reasoning of LLMs to enhance the efficiency of assumption-free, random-guessing search methods. We address the challenge of LLMs' inability to generate precise and grammatically correct programs in domain-specific languages (DSLs) by proposing a Pythonic-DSL strategy — an LLM is instructed to initially generate Python codes and then convert them into DSL programs. To further optimize the LLM-generated programs, we develop a search algorithm named Scheduled Hill Climbing, designed to efficiently explore the programmatic search space to improve the programs consistently. Experimental results in the Karel domain demonstrate our LLM-GS framework's superior effectiveness and efficiency. Extensive ablation studies further verify the critical role of our Pythonic-DSL strategy and Scheduled Hill Climbing algorithm. Moreover, we conduct experiments with two novel tasks, showing that LLM-GS enables users without programming skills and knowledge of the domain or DSL to describe the tasks in natural language to obtain performant programs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Programmatic policies for reinforcement learning have gained popularity in recent years, with state-of-the-art methods often employing search techniques that utilize various heuristics and initialization strategies. Given that these search methods are typically localized and proximity-based, the choice of initialization point becomes crucial. The authors present a novel LLM-based search algorithm, LLM-GS, which uses large language models (LLMs) for initializing the search, capitalizing on their coding capabilities. They acknowledge that LLMs often struggle to generate effective initializations within the domain-specific language (DSL) of the problem and introduce a Pythonic-DSL approach to address this challenge. Additionally, they propose a new search algorithm called scheduled hill climbing, which optimizes the search budget by starting with a smaller allocation and gradually increasing it as the search progresses. Through experiments in the Karel the Robot domain, they demonstrate the superiority of their algorithm compared to several search baselines and conduct thorough ablation studies to assess the impact of each contribution on performance.

### Strengths
* The authors have identified a significant limitation in existing search algorithms for programmatic reinforcement learning: their localized nature and the crucial role of initialization. They effectively combine this insight with the proven capabilities of LLMs in coding tasks to develop a straightforward yet powerful algorithm, LLM-GS. Furthermore, they propose a search strategy that aligns with their initialization approach, recognizing that they are likely starting from a strong position and can save their search budget for later stages.
* They demonstrate the superior performance of LLM-GS compared to several search baselines, instilling confidence in its effectiveness, particularly in the Karel domain.
* The ablation studies are well-structured and thorough, effectively investigating each component to clarify the sources of performance improvements.
* The authors address potential concerns regarding LLM memorization and data leakage, which adds to the credibility of their results.
* I found the LLM Revision experiment in Section 5.5 particularly fascinating, as it explores whether LLMs can be leveraged more effectively in the search process to refine the initially proposed program. This experiment highlights that existing simple techniques may not perform as well as anticipated.

### Weaknesses
I have two main concerns with the paper that prevent me from giving it a higher score:
* The authors have conducted experiments solely in the Karel the Robot domain. Testing the LLM-GS algorithm in another domain, such as MicroRTS, would enhance confidence in its effectiveness across different problem domains and DSLs. **Addressing this issue could lead me to reconsider my score.**
* The ablation study comparing search algorithms is quite limited, especially given the close results shown in Figure 6. It appears that the LLM provides such a strong starting point that the specific search algorithm has minimal impact. All search algorithms perform reasonably well on CleanHouse, and the results are very similar among the top 2-3 algorithms on DoorKey. Including results from additional domains would help establish the significance of the search algorithm. Furthermore, the scheduled hill climbing algorithm involves several design choices, such as the sinusoidal schedule and logarithmic interpolation; conducting separate ablation studies on the importance of each of these choices could be beneficial.

### Questions
* Do the equations for the scheduled hill climbing algorithm derive from prior work, or is there additional intuition apart from the decision to increase the search budget as the algorithm progresses? There are several key choices here—such as the logarithmic interpolation and sinusoidal schedule—that don’t seem entirely intuitive and are introduced without sufficient justification. Could you provide more details on these decisions?
* Are the program embeddings used for latent space search generated by the LLM?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
In this paper, the authors propose a framework for PRL (Programmatic Reinforcement Learning) based on a new search algorithm in program space (SHC), whose initial population of programs  is generated from a capable LLM rather than obtained via random generation.
The authors' main contributions are a prompting strategy which correctly outlines each PRL task to the LLM, coupled with the idea of having the LLM generate both Python code and the DSL in which the solution to a given PRL task is supposed to be expressed; this allows the authors to get the LLM to generate programs with the correct DSL syntax, despite the DSL not having presumably been experienced by the LLM  during training.

### Strengths
The paper's idea is quite simple but effective, and one can see that it significantly improves over SOTA methods when it comes to sample efficiency/number of program evaluations during search, and for some tasks, also in terms of final average return. The paper is written clearly and features comprehensive ablation studies.

### Weaknesses
It is not quite clear which one of the two contributions (SHC and its initialisation with LLM-generated programs) is the one which contributes the most to the final result. From figure 4, it does not appear that the SHC algorithm is really that much more efficient than HC. Plus, the role of the Python -> DSL parser is not quite clear.

### Questions
I have two main points relating to the weaknesses mentioned above, which I would like to see addressed before possibly raising my score:
- The authors should run two more ablation studies of their LLM-GS method; one in in which they remove the LLM generated initialisation for SHC, and one in in which they run SHC which the initialisation strategy used by the SOTA HC. This would help address the weakness mentioned above.
- What is exactly the role of the parser? Does the LLM generate both Python and DSL code natively, or does it only generate Python code to be processed by the parser? Did the authors implemenent the parser themselves, of did they use an off-the-shelf parser? 

More but less pressing questions:
- The advantage of SHC in terms of # of program evaluations is manifest. However, how large is its advantage in terms of simple walltime/wait time?
- HC (and therefore SHC as well) is basically a greedy search algorithm. Isn't there a strong possibility of it getting stuck in local maxima? To me this seems to be a bigger problem of HC compared to its sample inefficiency.
- Most of the discussion in section 3 (aside from the outline of the Karel domain) feels more like a Related Work section. I suggest that the authors move it to Related Work or to the Appendix.
- Which initialisation do the authors use for the ablation study in figure 6? To they use LLM-generated programs or randomly generated ones?
 - How is the experiment in section 5.5 conducted? Is the LLM asked to revise programs, and these are then used as initialisation for SHC? Or is SHC completely absent from the experiment?

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
4

### Summary
This paper studies learning programmatic-actions for RL, through heuristic search methods such as hill climbing. The paper focuses on the initialization problem. Rather than randomly initializing programs, the authors propose using LLM guided programs where the environment is described in natural language and a GPT-based model is prompted to bootstrap a set of programs. These programs are further improved using heuristic methods based on environment feedback. The authors also propose a scheduler for the hill climbing that allocates budget based on a sinusoidal scheduling. Finally, they find that generating DSLs directly is challenging due to domain gap; instead, they propose python-dsl where a python program is generated by the LLM and converted into DSL using rules. The resulting initialization and heuristic lead to improvement in sample complexity of the heuristics in a toy RL navigation benchmark.

### Strengths
Update: Authors addressed my concerns with many additional experiments. I increased my score accordingly.

1. The paper proposes LLM-based initialization for heuristic search methods. This is applicable to broader range of domains.

2. Experimental results indicate sample efficiency.

### Weaknesses
My main concerns are lack of realistic domains, novelty of the proposed idea, and significance of the experimental comparison.

1. The proposed idea is only tested on a toy RL benchmark which only add marginal practical value. Are there no other environments where programs can be actions and your idea can be applied?

2. In Figure-4, HC and your method compare very similarly. In fact, confidence intervals overlap in many of the tasks. It is not clear if the your method is significantly better. Can you conduct a statistical test to better compare these two methods?

3. Related to (2), HC learning curve is steeper than yours. Can you explain why?

4. While the complexity focus is mainly on number of programs, the cost of running a LLM is ignored. Can you add the cost of initialization to your plots to understand if spending more “flops” is in general better?

5. What is task variance? Is it random initialization in environment state or policy pool?

6. What happens if you initialize other heuristics with LLM-based programs? Do they compare favorably compared to your Scheduling-HC?

7. In line 425, it should be 500k.

### Questions
Please see above for specific questions.

### Soundness
2

### Presentation
3

### Contribution
2
