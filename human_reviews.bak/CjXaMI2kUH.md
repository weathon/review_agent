# MrSteve: Instruction-Following Agents in Minecraft with What-Where-When Memory

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Significant advances have been made in developing general-purpose embodied AI in environments like Minecraft through the adoption of LLM-augmented hierarchical approaches. While these approaches, which combine high-level planners with low-level controllers, show promise, low-level controllers frequently become performance bottlenecks due to repeated failures. In this paper, we argue that the primary cause of failure in many low-level controllers is the absence of an episodic memory system. To address this, we introduce MrSteve (Memory Recall Steve), a novel low-level controller equipped with Place Event Memory (PEM), a form of episodic memory that captures what, where, and when information from episodes. This directly addresses the main limitation of the popular low-level controller, Steve-1. Unlike previous models that rely on short-term memory, PEM organizes spatial and event-based data, enabling efficient recall and navigation in long-horizon tasks. Additionally, we propose an Exploration Strategy and a Memory-Augmented Task Solving Framework, allowing agents to alternate between exploration and task-solving based on recalled events. Our approach significantly improves task-solving and exploration efficiency compared to existing methods. We will release our code and demos on the project page: https://sites.google.com/view/mr-steve.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work approaches the failures of low-level controllers for navigation in embodied AI environments. It argues that one of the crucial reasons for these failures is the lack of memory for recalling objects and events, which could lead to redundant exploration. For this purpose, it proposes Place Event Memory (PEM), an episodic memory mechanism that tracks entities alongside with their locations and associated events. By using PEM, they employ a count-based exploration strategy, which alternates exploration/exploitation behaviors based on the recalled events. The work also proposes a new low-level navigator built on top of the TrXL architecture. Experiments in simple minecrafts navigation tasks demonstrate gains over prior methods and baselines with either no or simpler mechanisms
of memory.

### Strengths
- The work is well written and structured, easy to follow, with good illustrations; the proposed method and motivation are clear.

- The particular choice of tasks is very smart since it shows different scenarios where only place memory or only event memory are not enough, justifying a combined mechanism as proposed by the work. This provides good positive evidence for the scenarios where the method is effective.

- The evaluation setup also comprises different setups and evaluation criteria to evaluate the method in terms of exploration and under circumstances of sparsity, constrained memory, and long-horizon navigation.

### Weaknesses
- The major criticism comes from the fact that the paper is motivated by advancing methods for general-purpose embodied AI, but the evaluated scenarios are simplified versions of the SLAM [1] problem. The only considered task is navigation, and the work assumes access to a grid map with the perfect localization of entities. Entities are also assumed to be static (see the point below). In this setup, the only challenge seems to be identifying the different objects, which is actually easily solvable by visual foundational models (MineCLIP). In this setup, one could easily build and maintain a topological map of the environment as in prior work [2, 3] and perform exploration and navigation on top of it. Therefore, it is unclear whether (1) the tasks are challenging enough to reflect general-purpose embodied AI problems and (2) what are the advantages of the proposed method over classic SLAM methods. These points should be clarified in the paper.

- As mentioned in the previous point, entities are assumed to be static, which assumes the entities in the map have fixed location and state. Nevertheless, in open-world settings this does not hold true, i.e., a “zombie” may move around or “be burnt” while the agent is not observing. These changes in the entities would add an uncertainty component in the memory that is disregarded by the proposed method - in other words, memory may become stale over time and the proposed mechanism does not have a way to update/discard these memories. Again, the work should clarify this potential limitation.

- The Success Rate bar plots (in Figures 5, 6, 12, 13, 14) should bring the error bars (confidence intervals) so that it is possible to analyze statistical significance of the reported results.

### Questions
As mentioned in my points above:

- What are the advantages of the proposed method in comparison with other methods for the SLAM/robot navigation problem?

- How does the proposed method function in the case of non-static entities, as pointed out in my second concern?


**Summary of the Review**:

The work is well written, the method is very clearly presented, and the evaluation setup is well diversified, covering important aspects of what is proposed. Nonetheless, it is unclear if the considered navigation tasks are challenging enough to reflect the adopted motivation. It is also unclear if the method is advantageous over the classic robot navigation method and how the work is placed in comparison with this literature. I believe these are crucial questions to be answered by the work in order to understand what we can take from it.

**References**

[1] Simultaneous localization and mapping (SLAM). Wikipedia, available in: https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping

[2] Street et. al. Multi-Robot Planning Under Uncertainty with Congestion-Aware Models. AAMAS, 2020.

[3] Garg et. al. RoboHop: Segment-based Topological Map Representation for Open-World Visual Navigation. ICRA, 2024.

**POST-REBUTTAL**

I appreciate the authors' efforts on the rebuttal phase to clarify reviewers' concerns.
After reading the comments, I believe my concerns were only partially addressed, particularly the first concern related to the comparison with other approaches for SLAM. The rebuttal limited to contrast with other recent work, but without any empirical comparison nor considering classic work on the Robotics literature.  My second concern was acknowledged by the authors during the rebuttal, and I understand it is a natural limitation of the work to be addressed in future iterations. My third concern was not addressed and was unclear why it was not possible to use the 100 seeds already executed to compute the error bars.

I am slightly increasing my score to borderline acceptance to reflect the additional evidence and clarifications raised by the authors in response to all reviewers. Still, I recommend authors to consider classic SLAM approaches as future baselines for this line of work. This weakness is a consensus among the reviewers.

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
4

### Summary
The paper tries to address the forgetting problem in long-horizon embodied control tasks, such as Minecraft, by introducing a new episodic-memory approach called ‘Place Event Memory’ (PEM). This method organizes past observations based on their locations within the environment (Place Memory). To minimize memory redundancy, each group within the Place Memory is further refined into an event-based memory by clustering observations according to their latent embeddings. The memory is utilized to more efficiently locate previously visited objects, particularly in long-horizon tasks. The paper also proposes an exploration-exploitation strategy allowing agents to alternate between exploration and task-solving based on recalled events. Experiment results showed that the proposed method outperforms traditional method (short-term memory and traditional FIFO memory) in long-horizon tasks.

### Strengths
1. The paper is technically sound and clearly describes the algorithm. The proposed approach (Place Event Memory, PEM) efficiently manages memory by integrating place-based and event-based memory, effectively reducing redundancy while preventing confusion from similar observations at different locations. This mechanism enables agents to recall previous memories during long sequences of tasks, thereby enhancing task efficiency.
2. The results on long-horizon tasks are good.
3. The presentation of the paper is well-executed.

### Weaknesses
1. The work requires further improvement: 
    - The authors refer to their proposed approach as “Place Event Memory” (PEM), where event memory is implemented by clustering according to the latent embeddings encoded by MineCLIP. However, many frames during the tasks do not constitute meaningful events. For instance, when searching for an object, it may take several steps to navigate in the environment. These observations may not be considered as events (and these observations are also not trained in MineCLIP, I’m uncertain about how they will be clustered). Therefore, it may not be appropriate to call it “event memory”, and there's still some unnessary memory storage.
    - For exploration, the authors use a visitation map and always select positions with the lowest visit frequency. This rule-based method is not efficient and does not leverage the knowledge saved in PEM. Why did the authors not consider using the same mechanisms for both exploration and exploitation?
    - The experimental results require further explanation. In Section 4.4, place-based memory achieves better performance on long-instruction tasks, while event-based memory performs better on long-navigation tasks. This result seems counterintuitive. Can the authors further explain these experimental outcomes?
   - The key idea of this work is to record the positions of visited objects, framing the task as a lifelong navigation problem. There has been significant research on navigation, particularly regarding navigating in unseen environments and exploring to locate target objects while recording previously seen ones (e.g., building a semantic map). Why did the authors not include a comparison with these works?
2. symbol expression: On page 4, the notation $O_t=\{o_t, l_t, t\}$ is used. In Markov Decision Processes (MDPs), uppercase $O$ typically represents the observation space, while lowercase $o_t$ represents an element within that space. This notation should be clarified.
3. spelling errors: There are some spelling errors in the appendix. For example, on page 22, “Evene Cluster Details” should be corrected to “Event Cluster Details.”

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Authors present Mr.Steve, an extension of the Steve-1 instruction-following agent, improving exploration and memory abilities in Minecraft settings. The core contribution of authors is the design of a Place Event Memory system, which allows creating a hierarchical memory: a set of map checkpoints is constructed from the agent trajectory through clustering, and in each checkpoint multiple events are memorized. The memory is built and queried based on CLIP embeddings from a previous work, allowing to compute similarities between a language or visual instruction and image-based memories. Additionally, authors finetune an existing transformer-based trajectory embedding network to learn a goal-conditioned policy and implement a count-based exploration system. Authors conduct multiple experiments to showcase the performance of the overall system, which significantly outperforms Steve-1

### Strengths
This work tackles an important subject of research: how to design efficient language-conditioned agents in complex environments.

The paper is very well written. Experiments are in-depth and clearly described. Results are impressive.

I enjoyed reading it.

### Weaknesses
The current version of the paper could be more pedagogical in some parts.
For instance: How is are the text and video encoder aligned ? More explanation about this should be featured in the paper (i.e. more explanation about MineClip I guess).

Likewise, some details regarding the DP Means algorithm, used to figure out checkpoint locations, could be useful. E.g. how does it compare to the well known K-Means ? Does DP Means selects autonomously the number of clusters ? If no, how is the system detecting how many clusters/checkpoints to create for a given agent trajectory ?

l.243 "This structure improves the efficiency of the read operation by extracting top-k place clusters with their center embeddings first, then fetching relevant frames from these clusters"
--> This hierarchical decomposition assumes the “center embedding” is sufficient to figure out which FIFO memory to read. This looks like a strong assumption, e.g. the center embedding could correspond to the agent looking away from the object of interest. But maybe it is enough if clustering is well done ?

### Novelty/Significance

While very well executed, this work only tackles a Minecraft scenario, on a few individual contributions are proposed and efficiently combined: a hierarchical memory system, a count based exploration mechanism, and goal-conditioned navigation agent. Count based exploration and xy_goal-directed navigation are well known areas. I am not an expert regarding memory systems for decision-making agents, but similar systems might have been proposed in the past. I am also not expert enough to assess whether the considered baselines are sufficient.

I am looking forward to the discussion period to update my score, but from this first review I recommend acceptance, despite my aforementioned concerns. This work efficiently showcases how to combine and scale known components in a complex and relevant setting.


minor:

l.100 "(Hafner et al., 2023; Guss et al., 2019; Cai et al., 2023a; Mao et al., 2022; Lin et al., 2021; Zhou et al., 2024a)" 
--> make sure for any multi-citation to order by year from old to new.

### Questions
see above

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In the presented work, the authors propose an improvement over the widely used STEVE-1 low-level controller used in Minecraft challenges, addressing its limited episodic memory and inefficiency in handling long-horizon tasks. In particular, the authors propose Place Event Memory, a novel approach to storing "what-where-when" information. Utilizing this memory, the authors propose a new navigation policy, VPT-Nav, capable of balancing when exploration is needed and when direct task-solving can be done due to recalling prior information from the hierarchical PEM memory. Minecraft is a great testbed for dynamic environments and provides a challenging task for policies targeted at solving such settings where long-horizon task planning is needed.

### Strengths
- The introduction is well-written and intuitively understandable.
- The event-based clustering in the PEM memory seems novel an intuitive 
- The results of Mr.Steve are strong and supportive of the claims made by the authors.

### Weaknesses
- In the PEM memory, clusters are updated based on some threshold c concerning the similarity of events/locations. However, it is unclear how this is handling events that happen dynamically. Even in the "burning zombies" example given in the PM section, this event would only happen when it's early morning in the game (thus night-zombies are burning), yet the PEM memory doesn't include the game time, meaning that such an event would be unreliable and there doesn't seem to be a way to capture this. 
- The goal of Hierarchical Episodic Exploration seems to be to also prevent re-visiting places that have previously been seen. Given that the location of the agent is part of the place embeddings e_t, that seems feasible, however, if two environments are visually similar, that would mean that the agent would still explore as its location is a strong bias. However, this is a little in contrast to the goal of, for example, finding wood where searching yet another desert (just because the global position is different) would be suboptimal and should be avoided. So, in these settings, the global location would, on the one hand, need to be a strong separator to prevent searching the same environments, yet, at the same time, a weak separator because we wouldn't want to search the same biome for too long. How is this balance handled? I think a discussion on which factors exactly contribute to the creation of a new cluster would be beneficial for the paper.
- While the results in section 4.3 are supportive of the author's claims, it would be great if the same settings as in section 4.2 could be tested to demonstrate the impact of varying global memory limitations. E.g. which of the methods can solve the milk-sand-milk task with the least amount of global memory? Such a comparison would make the contribution much stronger.

### Questions
- In the example where meat needs to be found, what happens if navigating back to the location where cows were previously seen at does not have cows anymore? Is there a way to update/forget information in the PEM? 
- In Figure 5, are the results, particularly comparing Mr.Steve, Steve-EM, Steve-PM, and Steve-FM, statistically significant (particularly for Wool-Dirt-Wool and Milk-Sand-Milk)?

### Soundness
3

### Presentation
4

### Contribution
3
