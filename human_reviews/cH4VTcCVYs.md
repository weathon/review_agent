# Policy optimization emerges from noisy representation learning

- Decision: Reject
- Scores: 5, 5, 5

## Abstract
Nervous systems learn representations of the world and policies to act within it. We present a framework that uses reward-dependent noise to facilitate policy optimization in representation learning networks. These networks balance extracting normative features and task-relevant information to solve tasks. Moreover, their representation changes reproduce several experimentally observed shifts in the neural code during task learning. Our framework presents a biologically plausible mechanism for emergent policy optimization amid evidence that representation learning plays a vital role in governing neural dynamics.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors propose a new algorithm, NETO, for representation learning within the context of Contextual Bandits and Markov Decision Processes.
Their algorithm is posed as a theory of representation learning within the central nervous systems, and as a direct competitor to neuromodulation-based learning rules in e.g. spiking neural networks. 

At the core of NETO, both the network activities and the network weights perform noisy gradient descent on a representation learning loss, which they task to be the Similarity Matching objective. Descent on this objective yields Hebbian learning with linear weight decay for both the feedforward and recurrent weights. The noise in the weight update rules is modulated by reward, with higher reward yielding less noise; this allows the network to explore possible representations until one is found for which the resultant policy performs well. 

NETO is tested on both a contextual bandit task - corresponding to discrimination of a pair of linearly separable clusters in input space - and the control task Cart Pole. The authors demonstrate by a mathematical analysis that the stationary distribution of solutions learned by NETO has high mass on high-reward solutions, and that as modulation is sharped ($\beta \to \infty$) the stationary distribution sharpens to an exact solution. 

The authors then demonstrate that NETO can learn in an MDP setting by testing its performance on Cart Pole. They show that, under a variety of input normalisations, NETO still successfully solves the Cart Pole task. 

Lastly, the authors discuss experimental observations that connect with the NETO framework.

### Strengths
Quality:
- The theoretical analysis of NETO's behavior for the contextual bandit task is the strongest part of the paper, and is very solid. The mathematical details provide useful and valuable insight into how NETO behaves, and why it is able to successfully solve the task. 

Originality:
- The combination of similarity matching objectives with reward modulated noise is novel, and presents a specific, distinct learning rule that a neural network can implement. 

Clarity:
- The paper was in general very clearly written, with most arguments easy to follow and engage with. The clarity of the paper was high. 

Significance:
- The paper proposes a novel learning rule which can be implemented locally and therefore could be of interest to the neuroscience community.

### Weaknesses
The primary weakness of the paper is that it is unclear what problem it is addressing, and whether it wishes to be evaluated as a contribution to computational neuroscience or a contribution to machine learning.

If the paper is intended as a work of computational neuroscience, it has significant shortcomings on multiple accounts:
1. (038) "[NETO] hypothesizes that the noise in the nervous system is reward-dependent, decreasing in magnitude when the nervous system's weight configuration learns a rewarding policy." No direct experimental evidence is given for this claim within the paper, at least not when it is made. The value of this hypothesis would then only be in indirectly explaining other experimental results; but if so, which ones? The hypothesis is presented with minimal indication of why a neuroscientist would accept it as true. 
2. (157) In (4), the activity noise is set to zero. Earlier in (037) you write that "NETO treats the brain's intrinsic noise as a feature rather than a bug". Do you mean only noise in weights? If so, you should say this. But then how does NETO relate to noise in activities?
3. (082) If spiking neural networks can learn Atari games (Capone and Paolucci, 2024), can NETO also learn to solve those games? If not, this seems like a major strike against NETO as a plausible theory of representation learning. If so, you should provide evidence. Additionally, you write that previous work with SNNs "was based on rules with little experimental validation". What is the experimental validation that NETO has that these rules do not? For example, what experiment should I look at to see why the form of the weight noise (9) is plausible? 
4. (060) A minor detail, but if this is a neuroscience paper then the coloured areas you have in Fig. 1 really ought to correspond to correct brain areas. Modulation should be in the basal ganglia - you have it in occipital/temporal cortex; representation learning should be in occipital cortex/PPC/PFC - you have it in sensory cortex and motor cortex; Motor should be in motor cortex - you have it in the frontal lobe. If your theory is entirely abstract, and you do not intend for modulation to be interpreted as corresponding to the dopaminergic system, you ought not to say in (087) "we return to the idea that a neuromodulatory system acts on a network operating under experimentally validated plasticity rules", as it gives the misleading impression that you intend for your theory to correspond to biology. 
5. (480) What falsifiable claims does NETO make about representational drift which could be experimentally tested, and distinguish it from other theories of representation learning? Additionally in (506), you speculate that the behaviour of NETO with the NNSM objective may recapitulate experimental results; but then in (508) seem to imply that this speculation should be taken as experimental support for NETO. If your claim regarding NNSM is correct, please demonstrate so empirically. Otherwise, this speculation does not provide even circumstantial evidence for NETO's validity.   


Alternativley, if the paper is intended as a work of machine learning, its empirical results are not strong enough for it to be a contribution of general interest.
1. The only network architecture considered in this work is a 1 layer inhibitory recurrent network without non-linearities, and perceptron-style readout. Does NETO scale to more complicated architectures? In (121), you make the observation that NETO can be *applied to* arbitrary architectures - but this is very different for it *working* for more complicated architectures. 
2. The performance of NETO, treated as an ML algorithm, is not benchmarked on either of the tasks it performs. While you show that it can learn to perform a simple contextual bandit task, you do not compare its empirical regret against other learning rules. Similarly, no benchmark is given to understand how NETO's performance on Cart Pole compares to that of, e.g., a TD-learning-style algorithm. These failings seem significant if NETO is to be understood as a competitor to classical TD-learning rules.  
3. The tasks that the network performs are simply not complicated enough to be of interest from a machine learning perspective. How does NETO perform on harder tasks, like the continuous control MuJoCo suite or the discrete Atari suite?  


Additional weaknesses:
- (309) "Previous work used linear stability analysis to show that the diffusion coefficient characterizing this process is linear in $\eta_\theta$". Please cite the work to which you are referring. Be more explicit in the argument being made here. 
- (420) I find the argument made for why projection into the principle subspace is insufficient to solve Cart Pole to be unconvincing. While you are correct that the variance of the pole angle is low, a successful policy will additionally have low variance for the other variables - momentum, angular momentum, and position. It is unclear to me that the pole angle will definitely fall outside of the leading PCs. If your argument is correct, I would like to see a numerical experiment which demonstrates the claim being made, namely that if you train a linear readout on the top 2 PCs of the observations for the Cart Pole optimal policy, you cannot recover a good policy for the task.

### Questions
Is the SM objective an intrinstic part of NETO, or do you view it as just one possible choice for the representational objective L? If it is only one choice, can you provide evidence that NETO still functions when another representational objective is used?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces a novel framework, termed NETO, designed to facilitate policy optimization through a noisy representation learning objective. The efficacy of the framework is demonstrated using two simple RL tasks. Additionally, the paper discusses the biological plausibility of the proposed framework.

### Strengths
- The NETO framework seems like a novel contribution that hypotheizes that noisy representation learning can also facilitate policy optimization with reward-modulated noise.
- The paper presents a clear argument for why biological learning could be using a NETO-like framework. 
- The paper is well presented and easy to understand.

### Weaknesses
- The tasks considered in the paper are very basic to gain any insights into whether this framework will work for more complex settings. While the section on the contextual-bandits task is clear and legible, it feels over-explained as of now. Although the section is helpful in visualizing the diffusive search that the reward-dependent noise induces, much of the explanations feel redundant and can be moved to the appendix.
- Even though the framework is about online agents, the tasks used to evaluate the framework are simple and don't require online-learning. The authors should include tasks that are more suitable for online learning (which can also be a toy task). A simple experiment showing that the framework enables efficient transfer across tasks will also be interesting. 
- There is no comparative baseline in the paper to highlight the strengths of the framework.
- If the aim of the paper is to ***only*** serve as an introduction to a biologically plausible framework for policy learning, then a heavy neuroscience focused conference or journal might be a better avenue.

### Questions
Suggestions:
- Figure 3 & 5 (left) might look better with log-log scale or log x-scale.
- The x-axis on figure 4 (left) could be more legible with pi-based labels. Eg., labels of $0, \pi/2, \pi, 3\pi/2, 2\pi$.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper proposes NEural Thermal Optimization (NETO)—a representation learning method that implicitly learns rewarding policies from reward-dependent noise. The theoretical analysis demonstrates that network activities converge to the solution space of the representation objective, subsequently drifting within this space due to reward-dependent noise, effectively searching for reward-optimizing policies. Intuitively, this is achieved by reducing the noise-scale, and hence the exploration, when the reward increases. Experimental validation is provided through Contextual Bandit and CartPole problems.

### Strengths
The overall idea of the paper is interesting and the theoretical analysis is comprehensive and thorough.

### Weaknesses
I believe the paper would greatly benefit from using lemmas and theorems to highlight its key contributions. As it stands, it is challenging to parse and distinguish between the rigorous claims the authors make and their intuitive insights.

Additionally, the paper would be strengthened by offering more high-level intuitions before delving into the maths. For example, the idea of "converging to the desired solution space and then exploring it using reward-dependent noise" is intuitively accessible and could be presented earlier in the paper. Currently, for readers who are not very familiar with the related work, this idea only becomes clear after some effort in digesting the math.

Besides that, I have some reservations about the theoretical contributions. 

1. What are the guarantees for learning optimal policies? Authors claim "With the appropriate choice of $\eta_{\theta}(R),$ the agent is guaranteed to learn a policy that maximizes reward." (L. 375). What is appropriate choice?
2. What is the applicability of the results? Do they apply only to contextual bandits? Only to contextual bandits with the specific set of parameters?

### Questions
1. Why do the authors use MDPs, that are discrete-time with PDEs that imply continuous time?
2. While Similarity Matching is never explicitly defined. I think it would be appropriate to include the definition into the paper.
3. How would your method perform if there are isolated locally optimal policies? Would the diffusive search be able to escape such optima?
4. Figure 5 is missing a legend.

Potentially, I could have missunderstood some claims of the paper. I am willing to increase my score if the authors clarify the confusions and provide explanations to the points above.

### Soundness
3

### Presentation
2

### Contribution
2
