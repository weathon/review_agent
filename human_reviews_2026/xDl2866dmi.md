# Beyond the Proxy: Trajectory-Distilled Guidance for Offline GFlowNet Training

- Decision: Reject
- Scores: 6, 4, 2, 8

## Abstract
Generative Flow Networks (GFlowNets) are effective at sampling diverse, high-reward objects, but in many real-world settings where new reward queries are infeasible, they must be trained from offline datasets. The prevailing training methods rely on a proxy model to provide reward feedback for online sampled trajectories. However, in scenarios where constructing a reliable proxy is challenging due to data scarcity or cost, one must turn to static offline trajectories for training. Nevertheless, current proxy-free approaches often rely on coarse constraints that may limit the model's ability to explore. To overcome these challenges, we propose **Trajectory-Distilled GFlowNet (TD-GFN)**, a novel proxy-free training framework. TD-GFN learns dense, transition-level edge rewards from offline trajectories via inverse reinforcement learning to provide rich structural guidance for efficient exploration. Crucially, to ensure robustness, these rewards are used indirectly to guide the policy through DAG pruning and prioritized backward sampling of training trajectories. This ensures that final gradient updates depend only on ground-truth terminal rewards from the dataset, thereby preventing the error propagation. Experiments show that TD-GFN significantly outperforms a broad range of existing baselines in both convergence speed and final sample quality, establishing a more robust and efficient paradigm for offline GFlowNet training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Trajectory-Distilled GFlowNet (TD-GFN), a new proxy-free GFlowNet training that eliminates the need for learned reward proxies. TD-GFN learns dense transition-level edge rewards from offline trajectories using reward-weighted inverse reinforcement learning, capturing structural importance and correlation with the rewards in the DAG. These edge rewards are used indirectly through DAG pruning and prioritized backward sampling, allowing exploration efficiency without error propagation from proxy models. Experimental results show that TD-GFN beats existing offline GFlowNet-based methods.

### Strengths
- The paper is well-written and easy to understand, with a clear structure and intuitive exposition.
- The motivation and proposed method are intuitive and straightforward, and the adaptation of edge-reward learning through an IRL perspective (bridging GFlowNet and RL formulations) is particularly interesting.
- The proposed method shows strong empirical results across diverse benchmarks, including Hypergrid, Biosequence, and Molecule design tasks.

### Weaknesses
- It is unclear why a discriminator-based formulation is necessary. Could a reward-weighted likelihood maximization over observed trajectories achieve a similar effect?
- It would be informative to analyze how well the model generates novel states unobserved during training.
- Although the method is developed for the offline setting, a discussion on how the proposed backward policy mechanism could extend to online learning would make the work more compelling.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies the setting of learning GFlowNet from trajectory-level datasets without relying on proxy reward models. TD-GFN algorithm is proposed, which employs IRL to learn edge-level rewards from the trajectory data, after that using them to prune the environment graph and define a backward policy used for prioritized sampling to collect the data that will be utilized for training of the final GFlowNet policy. The approach is evaluated and compared to a number of baselines on hypergrid, biosequence design and molecule design environments that are standard for GFlowNet literature.

### Strengths
The authors study an interesting setting of learning GFlowNets from online data without relying on proxy reward models, which has high practical potential in my opinion. The paper is generally well-written and has a good structure. The presented experimental evaluation is thorough, presents a large number of baselines for comparison, studies the performance of the proposed method on data of different quality, provides various ablation studies, and uses a broad range of metrics for comparison of the algorithms, including training convergence speed, distribution approximation quality, diversity and rewards of the generated samples. The results of the experimental evaluation are promising, showing strong performance of TD-GFN on various tasks in comparison to the baselines.

### Weaknesses
I believe that this is a solid paper, however, there is one crucial weakness that prevents me from recommending acceptance in its current form. The setting studied in the paper is learning GFlowNets from trajectory-level data, i.e. data consisting of trajectories $s_0 \to s_1 \to \dots \to s_n$ and rewards $R(s_n)$ given for terminal states. However, the paper has no experiments on non-synthetic trajectory-level data, thus I believe it is hard to make conclusions about the potential utility of the proposed algorithm in real-world problems. Experiments on hypergrids and molecules use trajectory level datasets collected by another GFlowNet pre-trained on proxy rewards, and experiments on biosequence design use a dataset containing only terminal states $s_n$ and rewards $R(s_n)$. The latter type of data is generally more widely available and easier to collect than whole trajectories, and is the type of data used to train proxy-reward models for GFlowNets, which the paper aims at bypassing. Thus, I believe that the presented experiments do not fully fairly follow the setting that the authors claim to be studying. If the authors demonstrate the utility of the proposed approach and compare to baselines on some real-world trajectory-level dataset during the rebuttal (containing the full set of intermediate states and transitions from the initial state to the terminal state $s_0 \to s_1 \to \dots \to s_n$), I will be glad to raise my score.

In addition, even though the authors present some intuition for the proposed pruning algorithm, the paper lacks any real theoretical explanation or motivation on why it might be beneficial.

### Questions
1) Can you please elaborate on the proposed reward-guided pruning procedure. It may be possible that a certain edge $s \to s'$ has a low learned reward, but further edges that can be reached from it may have high rewards (since IRL works here with the setting where the policy has to maximize the sum of rewards along the trajectory plus its entropy). Still, $s \to s'$ will be pruned, potentially blocking acess to a region where the GFlowNet reward $R$ has high values. Wouldn't this be a problem? In addition, if I understand correctly, the pruning may lead to completely removing access to some parts of the environment if all edges crossing some cut in the graph are removed. This may also lead to some modes of the reward distribution remaining hidden from the GFlowNet policy.

### Soundness
2

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
5

### Summary
The paper proposes Trajectory-Distilled GFlowNet (TD-GFN), a proxy-free method for training GFlowNet based on rewards from pre-collected offline trajectories rather than a learned reward function. It (1) uses GAIL on a pre-collected dataset to learn edge-level rewards over the environment DAG; (2) prunes low-utility edges using a threshold; and (3) performs prioritized backward sampling on the pruned DAG graph while updating the forward policy only with ground-truth terminal rewards from the dataset. Experiments on Hypergrid, antimicrobial peptide (AMP) design, and molecular design show faster convergence and stronger top-k rewards vs. proxy-based, proxy-free, imitation, and offline-RL baselines.

### Strengths
* Turning trajectory data into dense, structural guidance via IRL-derived edge rewards is new in the GFlowNet offline setting.

* The pruning criterion and prioritized backward sampler are simple and interpretable.

* Pipeline and training objectives are well explained.

### Weaknesses
1. The claimed advantages of TD-GFN rest primarily on HyperGrid benchmarks, but the chosen grid size ($8^4$) is too small to offer convincing evidence. Moreover, HyperGrids are homogeneous across dimensions, so grid height (also equal to the trajectory length) matters far more than dimensionality. With such a short trajectory length ($=8$), environment-DAG pruning becomes less meaningful.   Compared to girds with large trajectory length (e.g.  $256$ and $512$),  just making random transitions can easily traverse the reward landscapes and reach all reward modes. 

2. It is misleading to state that proxy-free methods based on pre-collected rewards and trajectories is more efficient than learning from the reward oracle.  Since your trajectories are simulated (we can only observe the terminal states in reality) and your rewards are computed from the oracle, this can not happens.  The point is achieving higher top-k rewards does not establish a better GFlowNets as the goal is capturing distributions, which demands both optimality and diversity.

3. The claim that proxy-free training is superior to proxy-based approaches is overstated. TD-GFN is “proxy-free” in that it trains on simulated trajectories terminating in a small subset of $\mathcal{X}$ with provided rewards. Conventional offline GFlowNet methods (e.g., TB, DB, Sub-TB) could adopt the same setup by restricting training to those trajectories; in other words, these methods can be run in proxy-based, proxy-free, or hybrid modes.  

4. The necessity of proxy-free training on purely offline trajectories is **not** justified for the tasks considered, for two reasons:
    * First, in domains like self-driving, proxy-free training is needed because we must not only achieve high terminal rewards but also mimic the actions/transitions along pre-collected trajectories. By contrast, in typical GFlowNet applications, only terminal-state signals are observed in the real world and the intermediate trajectories are fully simulated—so the “offline-only trajectories” argument does not apply.

    * Seconds, the meaning of learning proxies from provided data is to obtain reward values over the entire terminal space; while imperfect, such proxies can facilitate exploration beyond $\mathcal{X}$ and discover new modes. This further underscores why capturing the overall terminal distributions matters[1].

5. If the stated advantage of TD-GFN is better performance (in terms of either capturing distribution or discovering modes) than proxy-based GFNs, the evaluation does not establish this fairly. Most results compare against proxy-free GFN/RL baselines; rigorous baselines using conventional proxy-based GFlowNets (e.g., TB, Sub-TB with learned proxies) are absent or underreported. Moreover, because conventional proxy-based approaches are offline, there are substantial prior works on offline data samplers with replay buffers (e.g. [2]). As they does not touch the conventional training objectives, these GFN methods are clearly more straightforward than the proposed tree-phase pipeline ( IRL > DAG-pruning > policy updating) and should be included for a fair comparison.

6. While it is claimed by the authors,  the paper does not demonstrate a true exploration–exploitation balance. Exploration (novel terminal-state visitation) and exploitation (concentration on known high-reward terminals) are not controlled. Because TD-GFN is proxy-free, it primarily optimizes a policy around the given limited set of rewarded terminals, i.e., exploitation.

[1] Jain, Moksh, Tristan Deleu, Jason Hartford, Cheng-Hao Liu, Alex Hernandez-Garcia, and Yoshua Bengio. "Gflownets for ai-driven scientific discovery." Digital Discovery 2, no. 3 (2023): 557-577.

[2] Kim, Minsu, et al. "Local Search GFlowNets." The Twelfth International Conference on Learning Representations.

### Questions
* Can you provide experiment results on grids with long trajectory length? (e.g.  $128^3$,   $256^2$, and $512^2$).

* Why the training curves of AMP tasks are not provided? 

* Where is the diversity reporting for sEH task?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Trajectory-Distilled GFlowNet (TD-GFN), a proxy-free framework for training Generative Flow Networks on offline datasets. To overcome limitations of existing methods—such as error propagation in proxy-based approaches and restricted exploration in proxy-free methods—TD-GFN employs inverse reinforcement learning to derive fine-grained edge rewards from trajectories. These rewards indirectly guide policy learning via directed acyclic graph pruning and prioritized backward sampling, ensuring updates rely only on ground-truth terminal rewards. Experiments demonstrate that TD-GFN achieves superior convergence speed and sample quality, offering a more robust and efficient paradigm for offline GFlowNet training.

### Strengths
**Novel Proxy-Free Paradigm**: Departing from existing paradigms, this work pioneers a proxy-free approach that leverages estimated edge rewards. This novel framework effectively circumvents the key limitations inherent in both proxy-based methods, such as error propagation, and prior proxy-free methods, which often rely on coarse-grained constraints.

**High Efficiency**: By integrating DAG pruning and prioritized backward sampling, TD-GFN sets a new state of the art for offline GFlowNet training. It demonstrates robust superiority over a wide range of baselines, achieving faster convergence, higher sample quality, and a closer fit to the target distribution.

**Clear Presentation and Comprehensive Results**: The paper is well-written with clear motivation and methodology, and the appendix is very comprehensive. The contribution of the paper looks solid.

### Weaknesses
**Insufficient Analysis for Design Choices** : Although this paper does a pretty good ablation study in the appendix, it lacks analysis on the detailed design choices and what problems this design might lead to. For instance, pruning seems like a more 'extreme' version of weighted sampling, so would it also work to remove this part while somewhat adjusting the weighted sampling method to make it 'harsher' for those low-reward edges? In practice, purely clean datasets are often difficult to obtain in real-world scenarios, and pruning seems to make it prone to mistakes that removes some actually important edges. It is just an example, but I believe if the authors have tried more options and finally chosen this design, there should have been more details and insights here.

**Limited Experiment Settings**: The experiment results are mostly based on synthetic datasets, while only the biosequences are real-world data. Moreover, 1500 trajectories are relatively small, which does not prove if the proposed method has similar scalabilities as baselines.

**Insufficient Baselines**: It would also be beneficial if they can also compare the results with some non-GFlownet methods, i.e. some recent offline-RL methods.

### Questions
Just what I've mentioned in the 'weakness' part.

### Soundness
3

### Presentation
3

### Contribution
3
