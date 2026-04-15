# Task Planning for Visual Room Rearrangement under Partial Observability

- Decision: Accept (poster)
- Scores: 6, 6, 8

## Abstract
This paper presents a novel hierarchical task planner under partial observability
that empowers an embodied agent to use visual input to efficiently plan a sequence
of actions for simultaneous object search and rearrangement in an untidy room,
to achieve a desired tidy state. The paper introduces (i) a novel Search Network
that utilizes commonsense knowledge from large language models to find unseen
objects, (ii) a Deep RL network trained with proxy reward, along with (iii) a novel
graph-based state representation to produce a scalable and effective planner that
interleaves object search and rearrangement to minimize the number of steps taken
and overall traversal of the agent, as well as to resolve blocked goal and swap
cases, and (iv) a sample-efficient cluster-biased sampling for simultaneous training
of the proxy reward network along with the Deep RL network. Furthermore,
the paper presents new metrics and a benchmark dataset - RoPOR, to measure
the effectiveness of rearrangement planning. Experimental results show that our
method significantly outperforms the state-of-the-art rearrangement methods Weihs
et al. (2021a); Gadre et al. (2022); Sarch et al. (2022); Ghosh et al. (2022).

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a hierarchical task planner equipped with several proposed components for room rearrangement for user-defined goal states.
Search Network exploits LLMs to query possible receptacles where unseen objects may be present.
The graph-based state representation encodes the objects' spatial relationships and their distances for the current and goal states in the form of graphs.
This is later used for Deep RL based Planner trained by the proposed cluster-biased return reward decomposition.
For the evaluation, the paper introduces a new benchmark, RoPOR, for room rearrangement that addresses blocked goals and swap cases, with newly introduced metrics that mainly measure agents' efficiency.
The proposed method outperforms the baselines in their empirical validations by noticeable margins.

### Strengths
- Tackling the blocked goals and swap cases is well-motivated and sounds sensible. Addressing them seems to be an important problem.
- Exploiting prior knowledge encoded in LLMs for possible target receptacles looks reasonable.
- Exploring an end-to-end framework for room rearrangement for user-defined goal states is intriguing.

### Weaknesses
- In Search Network, it is unclear why we need the "two-staged" approach: 1) filter out some implausible receptacles and 2) obtain the most plausible one. Why not just use SCN alone to get the most plausible receptacle, as the implausible receptacles should result in low scores and thereby be not chosen, consequently?
- The graph-based state representation requires shortest-path computation for all fully connected edges, but this seems quite computationally heavy, especially when we have a large number of nodes, leading to a drastically increasing number of edges.
- The comparison with some baselines seems unfair. For example, Weihs et al. and Gadre et al. do not use depth maps as input while the proposed method does, but they are compared in a single table. In addition, the authors utilize additional training datasets (Sec. 3.1).
- Some new metrics are introduced to measure agents' efficiency but they look a bit similar to SPL in navigation literature, which basically penalizes an agent's success rate by the length of trajectories it took so far. Similarly to the introduced metrics, as agents take more steps for rearrangement or search, SPL penalizes the success rates more, accordingly.
- The environments used in the proposed benchmark look quite "clean." It seems that we have objects only related to rearrangement tasks, as illustrated in Figure 16 and the supplementary video. This looks quite far from practical scenarios as we usually have many objects inside rooms.

\* Minor
 - It might be better to divide a result table (e.g., Table 1) into one for the main results and the other for the ablation study for better readability.

### Questions
See weaknesses above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors tackle the problem of object re-arrangement. To define the goal state, the agent is able to explore the room in its goal state and build up its own (graph-based) internal representation of the room. Then, at test time, the objects in the room are shuffled around such that they can be occluded or hidden in receptacles. The job of the agent is to put the room in the desired goal state as efficiently as possible. To do this, the proposed method keeps track of which objects have been seen and which haven't. A search network predicts probable locations of unseen objects. Both the current scene and goal scene are encoded as graphs, with objects as nodes and geodesic distances between objects as edges, and this scene is encoded with graph networks. Lastly, they use Q-learning with a proxy reward network to train the planner. To this end, the authors also build a dataset to train the search network and the graph network, in addition to contributing a new benchmark in Ai2Thor (RoPOR). The proposed approach significantly outperforms the baselines.

### Strengths
* The authors provide a new benchmark for object re-arrangement in AI2Thor. The new dataset supports swap cases (objects' positions that are swapped compared to the goal scenario) and blocked goals (ie. there is an object occupying the goal position of another object).
* Extensive supplementary video and supplementary material
* Modified RL training approach seems to outperform other approaches in that it converges substantially faster.
* The approach significantly outperforms the baselines.

### Weaknesses
* Convoluted approach that is a bit hard to follow, especially because many details are omitted in the main paper. It would also be helpful to more clearly define the task setup (inputs and outputs and their explicit representations).
* Not obvious why the SRTN needs to be learned instead of manually defined.
* Manual heuristics (where objects are unlikely to be found) and GT information (e.g. odometry, collision vector, etc.) is required.
* More details on the reward design in the main paper would be appreciated
* Critical ablations are not in the main paper.
* There is already an existing dataset (and associated metrics) for the same setup in Sarch et al. (Sec. 4.5 of their paper). While there are some deviations between the proposed dataset and the existing dataset, it still makes sense to benchmark the proposed approach against the already existing dataset that has baselines already benchmarked against it. Is there a reason this wasn't done? The setup of the existing seems like a subset of the proposed dataset's setup, so it seems doable.
* Somewhat similar to Sarch et al.
    * Both use 2D and 3D representations of the environment
    * Both use object detector + a search network to guide exploration
    * Both represent the scene as a graph on top of which they perform inference

Minor comments:
* Some strange wording such as contributing to the "research fraternity", misspelling such as "detetector, "Paramater", etc.
* Assumes perfect motion and manipulation
* Goal definition of having the goal scene already set up and utilizing an exploration stage is a bit impractical, as it requires setting up the goal scene every time it is changed. Also the robot needing to explore the goal state takes time compared to, say, language defined goals.

### Questions
* What are the failure cases? Does the agent ever get stuck in a loop?
* Are duplicate objects handled (e.g. multiple sponges in a scene)? Are these included in the test data, and how does the agent perform under these conditions?
* SRTN requires manually defined rules and heuristics about where objects are unlikely to be found (e.g. cup in bathtub). Is the network unable to learn these probabilities automatically?
* Does the SRTN need to be learned via the MLP? It seems like the probabilities can be pulled directly from the data without learning. We could simply build a hardcoded table of probabilities. For example, the probablity of finding a sponge in the sink can be hardcoded to 0.80. Is there a reason why that number would change? What is the benefit of learning here?
    * It is odd that we are trying to learn where objects are likely to be, but the training dataset is designed such that the authors "ensure a random distribution of object placements" (Appendix C.1).
* How does the agent go from the output of the planner to discrete motions (rotate, move forward, pick up, place)?
    * How is navigation performed when going from point A to point B? Is it assumed that this problem is solved?
* There is already an existing dataset (and associated metrics) for the same setup in Sarch et al. (Sec. 4.5 of their paper). While there are some deviations between the proposed dataset and the existing dataset, it still makes sense to benchmark the proposed approach against the already existing dataset that has baselines already benchmarked against it. Is there a reason this wasn't done? The setup of the existing seems like a subset of the proposed dataset's setup, so it seems doable.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper tackles the task of visual room rearrangement where an agent must first explore a 3D scene autonomously to map its content before objects in the scene are moved around. The agent must then, in a second phase, re-organize objects whose position has been changed compared with the initial scene. The authors propose a series of contributions to both evaluate methods, with a new benchmark and metrics, and improve autonomous agents’ overall performance and efficiency, with different architectural and training contributions.

First, the paper introduces a new Search Network whose goal is to predict the position of unseen objects after the room has been untidied. The claim is that such a search process could leverage prior common sense to find candidate receptacles more efficiently. Thus, the Search Network is composed of a large language model (LLM) that incorporates prior knowledge about the relationships between objects and receptacles.

Another contribution is a hybrid action space Deep RL agent that tackles both object search and rearrangement. The state space of this Deep RL agent is a graph representation of both the initial scene (known as goal state) and the current scene (known as current state) to provide information about the position of objects. Finally, the paper proposes a proxy reward network predicting a dense reward signal to facilitate the RL training of the policy.

The method is compared with different baselines on a new benchmark, RoPOR, and additional metrics, evaluating the efficiency of taken paths, are considered.

### Strengths
* **S1**: The paper tackles an important and challenging task, i.e. visual room rearrangement, and motivates its different contributions based on important considerations about the task.

* **S2**: This work proposes many different contributions, both from an evaluation point of view (benchmark, metrics) and from an architectural and training points of view.

* **S3**: The method is compared against different relevant baselines and shows a promising gain in performance.

* **S4**: Many qualitative videos are presented in the Supplementary Material, helping to better compare methods by visualizing their behavior.

### Weaknesses
* **W1**: **[Major]** How can the Search Network leverage prior knowledge as the room is untied? More specifically, if object shuffling/placement is done randomly, shouldn’t there be no prior remaining? Indeed, in phase 2, objects are placed in locations were there shouldn’t be (so that the agent can re-organize them). It is hard for me to understand how, in this case, the Search Network can learn anything meaningful. A result in the ablation study (Table 2) seems to confirm this intuition: the performance of Ours-RS, where the Search Network is replaced with a uniform random sampling of the next receptacle to visit, is very close (1p Success Rate) to the performance of Ours. The difference does not seem to be significant enough to claim the Search Network does more than random search. In order to claim a gain, authors should report the mean and standard deviation over a few training runs (random seeds). The following could also be done by authors:
    * **W1.1**: Reporting the performance of Ours-RS on the introduced RoPOR benchmark.
    * **W1.2**: Comparison of the Search Network with another simple baseline: selection of the closest receptacle (with reported performance on both RoPOR and RoomR).
    * **W1.3**: Provide more details about how objects are moved in phase 2: there should not be any prior remaining, and if there is, it might mean that the comparison is unfair with other methods because the authors’ search model might have been trained to learn those “shuffling priors” while it is probably not the case of previous work.

* **W2**: **[Major]** This comment is quite related to the previous one: as mentioned in the paper, the Search Network is finetuned to incorporate prior knowledge about object-receptacle relationships (see W1 about why I am not convinced any such relationship can be learned in the untidy scenario). Authors should still show the pre-training of the LLM brings a performance gain. What about the same LLM architecture initialized with random weights and trained as done in the paper?

* **W3**: **[Major]** It is not clear to me how the ground-truth data to train the Sorting Network (SRTN) is generated. Could authors elaborate on this?

* **W4**: **[Major]** One might argue that the Sorting Network only could be enough to predict the most likely object-receptacle pairs. Authors should provide an ablation study showing the impact of the additional Scoring Network.

* **W5**: **[Major]** I would like authors to clarify the following points regarding the Proxy Reward network:
    * **W5.1**: What is the interest of this Proxy Reward network? The paper mentions it is a way to predict a dense reward to train the RL agent. However, given a simulator, couldn’t we simply compute a dense reward from privileged simulator information at training time?
    * **W5.2**: What is the training ground truth for the Proxy Reward network?
    * **W5.3**: What is the average return on the y-axis of Figure 3? Such return is indeed associated with a specific reward function: what are the terms of this reward function? Moreover, I would like the authors to provide more details about the comparison done in Figure 3 and thus the conclusions we can draw from it.

* **W6**: **[Major]** The paper mentions the introduced method “assumes the availability of perfect motion planning and manipulation capabilities”. While this is a strong assumption, it does not outweigh the contributions in this work. However, an important question is: Are all the baselines this method is compared against also benefiting from the same assumption? Otherwise, this could be considered as an unfair comparison.

* **W7**: **[Minor]** When introducing their SNS metrics, authors should cite *Anderson et al., On Evaluation of Embodied Navigation Agents* that introduced quite similar metrics such as SPL.

* **W8**: **[Minor]** Paper citations are not properly inserted in the text. Authors should use parentheses (\citep{} in Latex) when needed, and remove double citations (e.g. “Sarch et al. Sarch et al. (2022)”).

### Questions
All questions and suggestions are already mentioned in the “Weaknesses” section as a list of numbered points.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
