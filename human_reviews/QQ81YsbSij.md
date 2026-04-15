# Learning Conditional Policy for Crystal Design using Offline Reinforcement Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 3, 3

## Abstract
Navigating through the exponentially large chemical space to search for desirable materials is an extremely challenging task in material discovery. Recent developments in generative and geometric deep learning have shown promising results in molecule and material discovery but often lack evaluation with high-accuracy computational methods. This work aims to design novel and stable crystalline materials conditioned on a desired band gap. To achieve conditional generation, we:  1. Formulate crystal design as a sequential decision-making problem, create relevant trajectories based on high-quality materials data and use conservative Q-learning to learn a conditional policy from these trajectories. To do so, we formulate a reward function that incorporates constraints for energetic and electronic properties obtained directly from density functional theory (DFT) calculations;  2. Evaluate the generated materials from the policy using DFT calculations for both energy and band gap; 3. Compare our results to relevant baselines, including a random policy, behavioral cloning, and unconditioned policy learning. Our experiments show that our conditioned policies achieve more targeted crystal structure designs and demonstrate the capability to perform crystal structure design evaluated with accurate and computationally expensive DFT calculations.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper tries to design novel and stable crystalline materials conditioned on a desired band gap. The authors turn the problem into a sequential decision-making problem and use offline RL algorithms to learn an effective policy to design materials. They design the reward function for material design. For experiments, they compare their methods against behavior cloning, random policy, etc, and demonstrate the effectiveness.

### Strengths
+ They formulate the crystal design problem into an offline RL problem which utilizes total energy and band gap values to get reward functions. By using DFT calculations, they can optimize the rewards thus solving the crystal structure problems;
+ They built an offline dataset from the Materials Project so that offline RL algorithms could learn from the dataset;
+ The illustration and explanation of the material designs are easy to understand for ML people, the authors abstract away the complex details and make an effort in the presentation to make the paper accessible.

### Weaknesses
- I am in no way a crystal design expert, but from the experiments, it feels like lacking strong baselines.  The authors compare their method to some natural baselines deviated from their proposed method (CQL). It would be great if the authors could compare/mention the state-of-the-art methods used in the field of crystal design;

### Questions
The expert reviewer on DFT/crystal designs should do their gatekeeping job on the paper. From my educated guess from an RL researcher's point of view, the paper is ok.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work focuses on inverse design for crystal structures which has a desired band gap. They model the problem as a MDP, and apply off-the-shelf algorithms from offline RL. 

The results are evaluated by DFT, therefore, might be promising. But I'm not an expert on band gap, so I cannot judge whether the results are strong.

### Strengths
RL-based inverse design for 3D structures is relatively under-explored. This work proves that RL methods have the potential to resolve some challenges of inverse design in science.

### Weaknesses
1. No novel algorithm is proposed. But for application-oriented research, novel algorithm is not always necessary. The focus here is on implementing and adapting existing methods in a novel context.
2. The analyses of the results are limited.  It would be beneficial to quantify how many new structures were generated and clarify whether the algorithm simply reproduces known structures from the training set. Given that conservative RL is used, I suspect the output is close to the training set.
3. Since DFT is actually included in the loop, how about using online RL? Is there any chance to find some novel materials through automated exploration.

### Questions
See above.

### Soundness
3 good

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
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors formalize the problem of crystal design as a MDP, produce an offline dataset for this MDP, and train CQL policies conditioned on the band gap desired in order to attempt to design novel semiconductors. They evaluate a conditional CQL policy along side a random policy, behavior cloning, and an unconditional CQL policy. 

They evaluate whether the crystal is valid, whether the total energy is low enough, and whether the generated and true band-gap distributions are accurate. 

They find that conditional policies generate materials closer to the desired band gap range, that greater conditioning seems to lead to better outputs, that unconditional policies recreate the original distirbution of data better, and that random policies do not generate good output.

### Strengths
Crystal design is an important problem for society and it is important that we bring our best tools to bear. Determining whether ML can be helpful and how best to apply it is a useful scientific enterprise and it seems this paper is a step in that direction. It seems that the authors have created 1) a useful new method, 2) a dataset that might be of independent value to the matsci community. These are both valuable.

### Weaknesses
This paper is not suitable for the program of ICLR, as it doesn’t advance the state of the art in machine learning. CQL is at this point a widely known algorithm and the application to crystal design is not one which pushes ML forward.

It is also not clear to me that the MDP formalism is the correct one for this problem setting. It seems to me that using a conditional generative model over the entire data would be a more natural choice (think diffusion model or autoregressive transformer). The authors don’t evaluate against those baselines which I think might be better suited.

### Questions
Have people tried generative modeling approaches to this problem? How well did they work? Can we quantitatively compare?

Is there a venue in materials science that would be more appropriate for this work?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper tackles the problem of inversely designing crystals by using offline reinforcement learning on a materials dataset with recomputed DFT property labels. The goal of this work is to learn a conditional policy that can efficiently perform atom substitution of given crystal skeletons, such that the resulting material satisfies a given target condition. It is shown that the learned conditional policy outperforms several baselines on a crystal design task where the generated materials have to satisfy particular band gap constraints.

### Strengths
The paper is clearly written and the use of offline RL is well-motivated by the prohibitively expense DFT calculations that would be required in an online setting. To my knowledge, this is also the first paper that uses offline RL for materials design. Limitations of the approach are clearly communicated, too. 

Overall, the chosen approach makes sense and the definition of the reward function and the Q-function are reasonable. Having DFT energies and band gap values consistently calculated with QE could be useful for researchers that cannot afford commercial DFT software.

### Weaknesses
As the authors point out, the paper addresses a vastly simplified problem. The policy has a very limited action space, and is basically just learning a more efficient way to substitute atoms. I do not see any evidence that the approach could be easily extended/scaled to a more complex action space as described in the future work section, so I am not sure how useful this approach is in practice. The RL approach itself is not very original since it is mostly a straightforward adaptation of CQL. And while I appreciate the effort to generate the dataset with QE, I would have liked to see more details about how the dataset was generated (e.g., which settings have been used and how were failures handled) if this is one of the main contributions of the paper.

The paper also has substantial flaws in its methodology. First of all, total energy says nothing about thermodynamic stability; instead, energy above convex hull is a much better measure. Secondly, I found the presented baselines pretty weak since they are either random or completely oblivious of the target condition (if I understand correctly). For example, a stronger BC baseline might be to train the policy on only those materials whose band gap values are close to the target condition. Finally, I would have liked to see how the generated crystals look like.

In terms of presentation, I found the quality of the tables and figures to be quite low. Given that this is an RL for materials design paper, I also would have liked to see a few more references at this intersection. See below for a few examples (however, note that I have not checked all of them thoroughly):
Sui et al. (2021). Deep Reinforcement Learning for Digital Materials Design.
Law et al. (2022). Upper-Bound Energy Minimization to Search for Stable Functional Materials with Graph Neural Networks.
Pan et al. (2022). Deep Reinforcement Learning for Inverse Inorganic Materials Design.
Zheng et al. (2022). Designing mechanically tough graphene oxide materials using deep reinforcement learning.

Minor comments:
- p. 1: "large and discrete space" - While the search space in the paper is discrete, the space of all possible materials is not discrete.
- p. 2: "in the Materials Project database" - citation missing
- p. 3: "is the is the" -> "is the"
- p. 3: "230 having the" -> "230 has the"
- Table 2: Given that the table does not add much additional information, I would suggest to move this to the appendix

### Questions
1. Could you please provide more information regarding how the dataset was generated?
2. How is the initial crystal skeleton $G_0$ for each episode chosen/sampled?
3. How do the crystals generated by the policy look like? Are they sensible?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
