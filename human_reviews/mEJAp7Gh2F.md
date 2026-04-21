# Language-Conditioned Imitation Learning With Base Skill Priors Under Unstructured Data

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 3

## Abstract
The growing interest in language-conditioned robot manipulation aims to develop robots capable of understanding and executing complex tasks, with the objective of enabling robots to interpret language commands and manipulate objects accordingly. While language-conditioned approaches demonstrate impressive capabilities for addressing tasks in familiar environments, they encounter limitations in adapting to unfamiliar environment settings. In this study, we propose a general-purpose, language-conditioned approach that combines base skill priors and imitation learning under unstructured data to enhance the algorithm's generalization in adapting to unfamiliar environments. We assess our model's performance in both simulated and real-world environments using a zero-shot setting. In the simulated environment, the proposed approach surpasses previously reported scores for CALVIN benchmark, especially in the challenging Zero-Shot Multi-Environment setting. The average completed task length, indicating the average number of tasks the agent can continuously complete, improves more than 2.5 times compared to the state-of-the-art method HULC. In addition, we conduct a zero-shot evaluation of our policy in a real-world setting, following training exclusively in simulated environments without additional specific adaptations. In this evaluation, we set up ten tasks and achieved an average 30% improvement in our approach compared to the current state-of-the-art approach, demonstrating a high generalization capability in both simulated environments and the real world. For further details, including access to our code and videos, please refer to our supplementary materials.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces SPIL (base skill prior based imitation learning), which improves generalization of robotic manipulation tasks to new environments by biasing skill embeddings toward one of three predefined base skills for robot manipulation: translation, rotation, and grasping. The encoder learns a continuous skill embedding based on a discrete skill prediction of which of the three discrete skills should be executed, the current image observation (from two angles), and language instruction. A VAE, trained to maximize ELBO, maps H-length action sequences into a continuous skill embedding space and decodes them back into action sequences.

Authors evaluate their method on CALVIN and demonstrate similar to HULC performance on in-domain tasks but stronger performance on out-of-domain tasks. They also evaluate their method zero-shot in the real world where results again outperform HULC.

### Strengths
(1) The results look promising on generalization to new envs, outperforming HULC.

(2) Authors set up a real world environment and got good results there over HULC.

(3) Appendix has a derivation for the loss, though I did not look into this carefully.

### Weaknesses
Method Weaknesses

(M1) Requires training a classifier on a sum of action magnitudes per dimension over the H-length horizon, which presumably requires ground truth labels on what skill an input action sequence corresponds to. Needing labels (or even needing to tune such a classifier if it were trained in an unsupervised fashion) is a significant limitation and not possible with most robotics datasets.

(M2) Unable to scale to more skills: If a new skill is needed in a new domain, everything would need to be trained from scratch.

(M3) Hard-to-classify action sequences: With explicit skill labeler supervision, it seems hard for SPIL to provide a skill embedding for H-length subtrajectories that have a mix of multiple base skills (such as one that encompasses the transition between translation and grasping an object).

(M4) Skills are blind to the state, the meaningfulness of action sequences (ie: the skill embedding space has no understanding of which skills are good vs just encoding random behavior), and are only as informative as what is expressed in the action magnitudes of each dim. A skill not conditioned on the current state is hard to efficiently adapt to a new domain with a sufficiently different state distribution, as the learning algorithm must determine which skills are appropriate for the current state. For instance, grasping-related skills are not good to execute when an object is already in the robot’s gripper.

(M5) Skill priors have been studied a lot. Behavior Priors (Parrot [1], SKiP [2]--modulo the human feedback, OPAL [3]). How do the authors orient this work to those previous skill-learning frameworks? Appendix A4 compares to previous skill-based methods SpiRL and SkiMO, but these do not look like domain-generalization results. Based on Tables 1 and 2, Table 7 looks like in-domain results with train == test env. Is this correct? If so, these results do not seem particularly relevant to the paper’s argument for better domain generalization.



Experimental Results Weaknesses

(E1) Lacking architectural ablations in general. There are a lot of modules in the architecture, and it is not clear to me why each of them is necessary. Some of them, including the discrete skill selector I mentioned earlier, seem to restrict the expressivity of this skill embedding space. What is the performance of the method without predefined base skills? This seems to be the crux of the paper’s contribution, so an ablation is well-advised.

(E2) All 10 Real robot rollouts on each task have “identical starting positions.” What is the value of doing 10 rollouts with a (presumably deterministic) policy? Are object positions randomized too or made to match, as closely as possible, to the sim?



Presentation Weaknesses

(P1) Writing in several places needs work, including fixing grammar issues. Citations are not formatted properly (entirely separated by parentheses), hurting readability.

(P2) Method section in general was quite hard to understand. Notation is confusing. For instance, $x$ is not defined in equation 2. I’m assuming it is an action sequence from an expert demo. Later, in section 3.3, $\tau_t$ is suddenly introduced, and it seems to represent the same thing as $x$, except that it is a predicted action sequence. If I’m interpreting these variables correctly, perhaps a better naming would be $\tau$ for expert demo action sequence, and ${\hat{\tau}}$ for the predicted action sequence. Naming of modules needs to be made less confusing. There are 4 modules that start with “Skill,” out of 5 modules total (Figure 2).

(P3) Figure 2 references huber loss, on the reconstructed actions, but there seems to be no mention of huber loss in the paper (until the Appendix, where I see a $\| x - \hat{x} \|_2$ term). It is also unclear what the cat loss in Figure 2 refers to.

(P4) Listing equation 2 and then 4 seems a bit redundant. They look really similar besides renaming some variables. It would probably be less confusing to introduce the elbo loss in one equation instead of similar-looking equations 3 pages apart. Perhaps the loss equation 19 can replace equation 4. 



References:

[1] “Parrot: Data-driven Behavioral Priors for Reinforcement Learning.” Singh et al. https://arxiv.org/pdf/2011.10024.pdf

[2] “Skill Preferences: Learning to Extract and Execute Robotic Skills from Human Feedback.” Wang et al.  https://arxiv.org/pdf/2108.05382.pdf

[3] “OPAL: Offline Primitive Discovery for Accelerating Offline Reinforcement Learning.” Ajay et al. https://arxiv.org/pdf/2010.13611.pdf

### Questions
(1) It seems like $w_k$ in equation 1 are learned (in training a Bi-LSTM). Where did the ground-truth action labels for Equation 1 come from? Does the learned skill embedding z-space not cluster grasping, rotation, and translation separately?

(2) It would strengthen the paper to visualize the z-space (t-SNE) and color z-space points based on which of the 3 skills they are labeled as.

(3) Is $p(z|y)$ just fitted to the action sequences in the data based on their class labels? Is it trained before the phase depicted in Figure 2? If there are two different phases as suggested by Figure 1 and 2, they should be clearly labeled as such.

(4) Is the encoder from figure 1 $q_{\phi}(z|x)$ finetuned in figure 2 as $q(z|x,y,c)$? Or are these different “encoders”?

(5) Section 3.2.2: How are two embeddings (language goal embedding and language embedding) extracted from the language instruction alone? I understand that there is a shared task embedding space in joint language + goal image embedding space, but this part of the paper is not explained well.

(6) Section 3.2.2: The authors write: “The policy $\pi(\cdot)$ should also identify the optimal base skill $y$ under the current observation.” If $y$ is already an input to the decoder, why should the decoder predict $y$ again?

(7) Looking only at the figures, how can the decoder (Skill generator) be frozen in Figure 2 if it only takes $z$ as input in Figure 1, and in Figure 2, is additionally conditioned on $y, c$?

(8) Remove the list of percentages in 4.2.1, as they are not important to the argument of the paper on generalization to new envs.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors propose a novel skill-prior based imitation learning algorithm. The proposed algorithm is able to learn skill priors from unstructured data, and use those skill priors in a language conditioned imitation learning setup. The structure of the paper is the following: first, the two stage algorithm is introduced, which first learns the skill prior distribution from the play dataset, and then learns a language conditioned imitation learning policy off of the demonstrations with labels. Next, the authors present some experiments, first in a sim environment, Calvin, and then in a real robot benchmark that they created. Unfortunately, the paper ends there without much more details, such as ablation experiments.

### Strengths
The paper is comprehensive, showing the formulation of the skill prior informed imitation learning formulation, and learning the priors from the play data. The primary strengths of this paper are:
1. Simplifying the skill prior space. Generally, discrete skill prior based works struggle from the chicken and egg problem of classifying skill priors from data and learning them properly. By constraining the skill priors to three semantic kind of actions (translation, rotation, grasp) the algorithm makes the problem tractable.
2. Showing the algorithm scales to a real robot: a lot of time the results in simulation based papers can overfit to certain kinds of environments or quirks in the simulation, but the robot experiment show that the risk of such is not high.

### Weaknesses
However, there are certain major shortcomings in the evaluation in the paper and the algorithm, which are detailed below:
1. The algorithm seems very much "overfit" to the Calvin benchmark, while not being very generalizable beyond the setup. As a primary example, even the three basic "skills" seem to be overfitting to the Calvin demo behaviors, since it ignores possible robot behaviors that mix two of these skills. One easy example is opening a hinged door requires rotation and translation at the same time, which isn't covered by the algorithm's use case. 
2. Another example could be the fact that the paper only focuses on skill-based manipulation algorithms, which is again a quirk of the Calvin benchmark's high-frequency control setup. However, recently there has been improvements in high-frequency controls that does not use a notion of skills, such as [1] or [2], which can be combined with learning-from-play-data algorithms such as [3] for a skill-free formulation. To show that skills are necessary for language conditioned imitation, either a comparison with such an algorithm, or a comparison on a different benchmark such as Language Table [4] would be quite useful.
3. Similarly, the real world performance is quite poor from the algorithm, which could be a case of the preset skills not really capturing the diversity of human behavior, but this question is left unanswered in the paper. The authors seemed to be content by beating out the single real baseline, HULC, which also seem to be a poor fit for the problem in hand.
4. While the language conditioning is presented as an important part of the algorithm, the "grounding" abilities are not convincing enough to show that it is a major part of the presented algorithm. Without a proper ablation experiment, this is hard to reliably conclude, which is also not presented in the paper. Similarly, ablation over the horizon may be quite important here, which is also not present in the main paper.
5. Finally, how important is extra play data if there is already sufficient language conditioned, labelled data available to learn a policy? If that is the case under which we are operating, can this algorithm still be called "learning from unstructured data"? Such questions can be answered by varying the dataset size, but because of an overdependence on Calvin as a benchmark, the authors are unable to present a real answer/experiment for this.

[1] Zhao, Tony Z., et al. "Learning fine-grained bimanual manipulation with low-cost hardware." arXiv preprint arXiv:2304.13705 (2023).    
[2] Chi, Cheng, et al. "Diffusion policy: Visuomotor policy learning via action diffusion." arXiv preprint arXiv:2303.04137 (2023).    
[3] Cui, Zichen Jeff, et al. "From play to policy: Conditional behavior generation from uncurated robot data." arXiv preprint arXiv:2210.10047 (2022).    
[4] Lynch, Corey, et al. "Interactive language: Talking to robots in real time." IEEE Robotics and Automation Letters (2023).

### Questions
1. How was the horizon length of 5 decided upon?
2. As I understand, the three base skills are interchangable, so how are they labelled as "translation", "rotation", and "grasp"?
3. How large were the real world datasets?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work proposes a skill-based language-conditioned policy. The objective is for the robot to understand human language commands, breaking down into skills to be executed consecutively. The architecture composes of a skill selector, labeler, base skill locator and generator.

### Strengths
1. The work considered language-conditioned skill-based policy, which is a good problem to study because language contains high level information that can be naturally broken down into skills.

2. Experiment setting: The tasks considered are unseen tasks that are not trained on during training, which is a good setting to evaluate skill learning.

### Weaknesses
1. Implementation of basic skills: The basic skills translation, rotation, and grasping are quite limited, as they only cover certain basic motion; they do not reflect the true distribution of real-world manipulation tasks. 
- There are also quite a few existing works on using predefined skill primitives like MAPLE (https://arxiv.org/abs/2110.03655), Dalal et al. (https://proceedings.neurips.cc/paper/2021/file/b6846b0186a035fcc76b1b1d26fd42fa-Paper.pdf). How do the authors compare this work to prior works that also uses skill primitives? 

2. Tasks are short horizon and limited: The tasks used in this work are very short horizon, e.g. "lift blue block". In other skill learning / skill primitive works, this could be already considered as a unit of a basic skill like lifting; there is no need to break it down into smaller units. Also, the point of using skills is to tackle those long-horizon tasks like "first lift blue block, then toggle switch". Therefore, I would consider tasks like this unable to evaluate the effectiveness of skill learning.

3. Missing baselines: the work compared with several skill-based RL works; but it fails to compare with MAPLE and Dalal et al. mentioned above.

### Questions
1. Implementation of basic skills: How are the three basic skills translation, rotation, and grasping implemented? Could you provide more details on how does these three skills decide their hyperparameters, e.g. how to know the translation distance or rotation angle?

2. Suppose the robot needs to learn a new skill (e.g. pouring), does the skill classifier needs to be retrained?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors present Skill Prior based Imitation Learning (SPIL), a framework for robotic imitation learning that breaks down a task into 3 base skills: translation, rotation, and grasping. The framework includes a low-level policy for generating action sequences from skills and a high-level policy that generates sequences of skills. The authors show strong performance on the CALVIN benchmark as well as a real robot using sim2real transfer.

### Strengths
The idea of decomposing a robotic manipulation task hierarchically using base skills is interesting and seems sound to me. Using translation, rotation, and grasping is widely applicable to many robot embodiments. The results on CALVIN are strong, demonstrating state-of-the-art performance. Any nonzero success on sim2real transfer is impressive.

### Weaknesses
While the basic idea of the paper seems sound, as far as I can tell, I believe it suffers from significant clarity issues. The method is complicated and has a lot of moving parts that are not fully explained. I find it difficult to evaluate the soundness and contribution of the paper due to these issues.

- In Section 3.2.1, it is not clear at all that trans, rot, and grasp correspond to groups of dimensions of the action space. The variables $x$ and $y$ are also not defined. The base skill classifier switches from $p(y = k \mid x)$ to $q(y = k \mid x)$ in the next section.
- While I generally figured out what was going on from Figure 1, I found the explanations in Section 3.2.2 fairly unclear. I also found Section 3.3 quite difficult to follow: for example, the "plan embedding" is never defined. I think the methods section could be improved by spending more time concretely explaining the authors' instantiation of skill learning rather than speaking so much in the generic terminology of variational inference.
- It is never explained how action sequences are sampled from the dataset. Wouldn't most action sequences include multiple base skills?
- If I understand correctly, the base skill locator is just an embedding lookup table for each of the 3 base skills. This could be clarified.
- Please fix the missing parentheses around references; it makes the paper more difficult to read.

### Questions
- How are the action sequences sampled during training? It seems to me that most action sequences would be likely to include multiple base skills, e.g., translation and grasping. How does the method deal with this?
- How were the magic scales $w_k$ chosen?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor
