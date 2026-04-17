# MorphoGen: Evolving Robot Morphologies with Large Language Models

- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Designing high-performing robot morphologies is a grand challenge for developing specialized autonomous agents. However, the vast, combinatorial, and non-differentiable nature of the morphological design space has been a primary obstacle. Existing methods tackle this problem indirectly, relying on either semantically-blind genetic operators or reinforcement learning with predefined modification actions, both of which constrain exploration. In this work, we introduce MorphoGen, a novel framework that reframes morphological design as a code generation problem. MorphoGen leverages large language models (LLMs) to directly iterate the XML files as codes that define an agent’s morphology, solving the original open problem without being limited by any prior constraints or fixed action spaces. Gradient-like textual guidance is provided to steer the evolution of robot morphologies through prompted mutations and crossovers. Our approach allows the LLMs to apply its understanding of structure and syntax to generate complex and semantically coherent design variations, enabling an unconstrained and efficient exploration of the design space. On a suite of challenging locomotion benchmarks, MorphoGen discovers novel and high-performing morphologies, significantly outperforming strong baselines by over 52.9% in downstream motoring evaluation. Our work unlocks a new paradigm for automated robotic design, demonstrating the effectiveness of LLMs in navigating complex, structured engineering search spaces. Codes for our work are released anonymously at https://anonymous.4open.science/r/MorphoGen-ACC/

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses a fundamental challenge in evolutionary Robotics, i.e., robot design automation. An LLM-based evolutionary strategy is proposed, which employs a critic LLM for textual guidance that facilitates more informative and comprehensive evolutionary search. A couple of other techniques, including structure pretraining, two-stage proxy fitness evaluation, and a hybrid sampling strategy, are also proposed that yield additional performance gains. Extensive experiments on simulated locomotion tasks validate the superiority of the proposed method over traditional evolutionary algorithms and RL-based design algorithms.

### Strengths
1.The paper is well written and easy to follow. The abundant illustrations and qualitative results greatly help with the reader’s understanding. The analysis in Section 5.3 is particularly interesting, revealing the correspondence between evolved robots and natural priors. 

2.The paper tackles a long-lasting challenge in Robotics, and proposes an effective approach that leverages the semantic priors of large language models to aid more efficient search. The algorithmic designs are intuitive and straightforward, yet proved to be competitive experimentally. 

3.The authors explicitly consider low-compute scenarios where only a minor fine-tuning budget is allowed, and confirm that the proposed approach is still competitive, adding to its practicality.

### Weaknesses
1.The baseline algorithms seem dated. As numerous LLM-based evolutionary strategies have been proposed in recent years (with some of them listed below), the authors are suggested to compare against one or two of them, or at least clarify the relationships with them. The absence of such analysis leaves the paper’s novelty largely unclear. 

- Qiu K, Pałucki W, Ciebiera K, et al. Robomorph: Evolving robot morphology using large language models[J]. arXiv preprint arXiv:2407.08626, 2024.

- Song J, Yang Y, Xiao H, et al. Laser: Towards diversified and generalizable robot design with large language models[C]//The Thirteenth International Conference on Learning Representations. 2025.

- Yang C, Wang X, Lu Y, et al. Large language models as optimizers[C]//The Twelfth International Conference on Learning Representations. 2023.

2.The comprehensive sampling strategy introduced in Section 4.1 seem to be a key component, and the increase of morphological diversity is claimed throughout the paper. However, the paper lacks any quantitative measurement and comparison of diversity, with only a couple of examples given in Figure 6. 

3.As structure pretraining turns out critical in ablation studies, it becomes questionable regarding how much the performance gains over baseline algorithms are merely due to the additional expert knowledge rather than the algorithm itself.

### Questions
1.Beside the fitness of evolved robots, could the authors describe the influence of structure pretraining on the evolutionary dynamics? More specifically, to what extent does the algorithm stick to the structures given in the initial population, and how much does it discover new, beneficial structures? 

2.Does the critic LLM also receive fitness scores as input? If not, could the inclusion of such information help the critic LLM better identify beneficial substructures against harmful or neutral ones?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper deals with the problem of optimizing robot morphology to maximize the performance on locomotion tasks. In the past there were approaches which performed local changes on the morphology without full semantic understanding of the full robot. The authors compare their new approach against those methods.

Main method:
The main method is based on using LLM to guide an evolutionary process by suggesting changes to robot morphology. After a change is performed the pool of robots is evaluated using RL-based policy training. As the framework of evolutionary process the AlphaEvolve approach is used - with its islands model to maximize fitness while maintaining diversity by keeping explicit robot sets (islands).

The approach is evaluated on locomotion tasks in the Mujoco simulator.

### Strengths
1. The main idea of using LLM as a mutation operator is definitely appealing (but unfortunately not new, see weaknesses).
2. The paper is well presented.
3. The problem itself is important and given the recent progress in LLMs one can expect significant results in the niche of robot morphology optimization.

### Weaknesses
My main problem with the submission is that the main result follows the approach of Qiu et al.:
RoboMorph: Evolving Robot Morphology using Large Language Models,
https://arxiv.org/abs/2407.08626
which was released on arxiv over a year ago. 

There were even follow-up papers to RoboMorth, such as RoboMore, released in May this year:
RoboMoRe: LLM-based Robot Co-design via Joint Optimization of Morphology and Reward
https://arxiv.org/abs/2506.00276

If we assume that the usage of LLM as a modifier operator on robot morphology was already known, then as far as I understand the main novelty of the paper in question is the specific evolutionary strategy based on AlphaEvolve.

### Questions
1. How your approach differs from RoboMorph?
https://arxiv.org/abs/2407.08626
2. Do you have ablation studies showing the effects of the proxy fitness idea (page 5)?

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
5

### Summary
The paper presents an approach to use LLMs for generating and evolving XML files representing simulated agents (similar to HalfCheetah, Walker, Swimmer and other Mujoco tasks). The LLM is mainly used to evolve the XML file structures, while for control standard reinforcement learning in the form of PPO is used.
The paper compares the proposed method against a limited set of prior work using evolutionary-based or policy-based methods for evolving agent structures.

Overall, I aprpeciate the paper for its interesting idea but it could be improved in terms of technical accuracy, theoretical insight and adding neccessary details ( see also below).

### Strengths
- I like the general idea of using LLMs for advancing the xml structure of agents
- The proposed approach seems to outperform other graph-based methods using either evolutionary algorithms or a policy to manipulate agent graph structures
- Experiments are conducted across 4 environments.
- Visualizations are nice
- The paper provides clear evidence that a critic LLM helps to improve performance

### Weaknesses
- The paper lacks clarity, formalism and technical accuracy at times. For example, it is not clear which LLM was used without looking at the provided code. However, the type and model of LLM has a clear impact on the overall performance of the system. No ablations on different LLM architectures are provided as well. This is not only an issue ofr reproducability, but also neglects the effect different LLMs could have on the final result.
- The environments used were introduced in prior work. It stands to reason that these environments have been part of the general training data of the coding LLMs, as they can be found on Github. Hence, the generalizability of the paper is in question without further investigation (eg good results could be mere lookups from the training data).
- I am not favourable of the formalism in section 3. Why do you differentiate between body parts B, joints J and actuators A? Why are the joints encoding DoF and not the combination of links and actuators? Generally, it reads like a mix between XML file structures and the graph definition from prior work like NGE. It is not clear how M_0 \in M defined a robot - eg if you have three body parts, three joints and one actuator, how are the relationships between these objects modelled? I strongly recommend clarifying this section and looking at the problem definitions of prior work.
- Using  (Yuan et al., 2022) as a reference for the PPO learning algorithm used on individual agents is a bit misleading. Please, reference the original PPO paper.
- Appendix B introduces the standard framework of RL - however you have a contextual MDP process at hand. Furthermore, the transition function should be a mapping from S x A to S and not S-delta.

### Questions
Please see the review and points raised above.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents an approach that frames the problem of morphological design as that of code generation by using LLMs to directly iterate XML files that specify an agent’s morphology. The claim is that the method enables the exploration of complex designs. The solutions are validated empirically and shows superior performance to several competing approaches for designing morphologies.

### Strengths
The approach seems quite novel, and aims to apply LLMs for designing morphology. The general approach seems valid and plausible. Overall, the method is also presented well.

### Weaknesses
From the paper, I see that the inner loop is run for a fixed number of episodes. However, I suspect it would be fairer to consider a fixed number of steps instead, as episode lengths could possibly vary drastically for different designs.
Given the specification of morphology only via code, I wonder whether operations such as mutations and crossovers are necessary. For example, I wonder whether other gradient or non-gradient based methods could be used to modify the XML code. In general, there may be different ways of achieving elitism, diversity and randomness, and it is not clear why the choices made in this paper are necessarily superior to other alternatives.

### Questions
1.	How exactly is the inner loop handled and what are the associated assumptions relating to the same?
2.	What motivates the use of evolution-inspired mechanisms such as mutation and crossover? In general, could other approaches such as Generative flow networks have been used?
3.	Are the LLMs specifically pretrained on XML-structure relationships?
4.	What guided the design choices pertaining to the criteria of Elitism, Diversity and Randomness as described in lines 203-212? Have other approaches for these been explored or considered?
5.	Why is a fixed number of episodes (and not steps) considered for the fitness calculation? As episode lengths could vary drastically depending on the design and other factors, wouldn’t a fixed interaction budget be fairer?
6.	How sensitive is the approach to the quality of initial genotypes? How would the performance vary if for instance, the initial designs were randomly generated instead of expert-influenced?
7.	Does the fact that a diverse set of high quality expert design morphologies is needed skew the designs towards those that are very similar to expert-designed ones?
8.	In addition to the designs, it would be interesting to see the gaits followed for each design either through videos/gifs or a sequence of images
9.	Are there any constraints applied to limb dimensions and other elements?

### Soundness
3

### Presentation
3

### Contribution
2
