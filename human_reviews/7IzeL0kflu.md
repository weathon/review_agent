# Simplifying Deep Temporal Difference Learning

- Decision: Accept (Spotlight)
- Scores: 8, 8, 8, 6

## Abstract
$Q$-learning played a foundational role in the field reinforcement learning (RL).
However, TD algorithms with off-policy data, such as $Q$-learning, or nonlinear function approximation like deep neural networks require several additional tricks to stabilise training, primarily a large replay buffer and target networks. Unfortunately, the delayed updating of frozen network parameters in the target network harms the sample efficiency and, similarly, the large replay buffer introduces memory and implementation overheads. In this paper, we investigate whether it is possible to accelerate and simplify off-policy TD training while maintaining its stability. Our key theoretical result demonstrates for the first time that regularisation techniques such as LayerNorm can yield provably convergent TD algorithms without the need for a target network or replay buffer, even with off-policy data. Empirically, we find that online, parallelised sampling enabled by vectorised environments stabilises training without the need for a large replay buffer. Motivated by these findings, we propose PQN, our simplified deep online $Q$-Learning algorithm. Surprisingly, this simple algorithm is competitive with more complex methods like: Rainbow in Atari, PPO-RNN in Craftax, QMix in Smax, and can be up to 50x faster than traditional DQN without sacrificing sample efficiency. In an era where PPO has become the go-to RL algorithm, PQN reestablishes off-policy $Q$-learning as a viable alternative.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Parallelised Q-Network (PQN), a streamlined deep online Q-Learning algorithm. PQN comprises two key components: a TD Learning objective without a target network, which instead applies layer normalization and L2 regularization, and a parallelized sampling approach that avoids the use of a replay buffer by leveraging vectorized environments. The authors provide theoretical analysis to support the claim that regularization can stabilize TD Learning in the absence of target networks. PQN demonstrates competitive performance across a wide range of environments, achieving results in significantly less wall-clock time.

### Strengths
1. PQN is straightforward and easy to implement. It removes the need for a target network and simplifies TD Learning.
2. The paper includes theoretical analysis showing that regularization can help keep TD Learning stable without using target networks.
3. PQN achieves higher computational efficiency compared with baseline methods, with minimal impact in sample efficiency.

### Weaknesses
1. It’s somewhat counterintuitive that PQN maintains sample efficiency while training only on online samples without a replay buffer. Additional explanation would help readers understand this aspect better.
2. The removal of the target network in TD Learning and the parallelized sampling are independent components of the algorithm, yet their individual contributions to overall performance are unclear. More controlled experiments, like the one in Figure 6.d, would clarify the impact of each component.
3. The parallelized sampling approach depends on vectorized environments, which feels more like an engineering choice than a novel contribution and is not feasible in many real-world applications. A significant portion of the wall-clock savings seems to come from this aspect. If my understanding is correct, DQN could also potentially eliminate the replay buffer and use a similar parallelized sampling approach. It would be informative to see how DQN performs under this setup.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper analyses stability in temporal difference (TD) methods. The main contributions are theoretical proofs that (i) TD instability can be established using a Jacobian evaluated on the unit circle; and (ii) using the Layernorm regularisation technique can ensure stability. This then leads the authors to propose a deep Q-learning algorithm called PQN which is comptetive with the PPO approach to reinforcement learning.

### Strengths
The paper is well written, well formatted and quite readable. The authors present the essence of their results very well and show that stability of TD algorithms reduces to checking that a Jacobian is negative definite on the unit circle. This is a nice succinct and somehow intuitive result. Given the complexity of the proof, it was good to see the summary presented so concisely. 

Other results are then presented after this, including some insight into the causes of instability and then the approach using the layernorm to obtain stabilisation. The presentation here was not as clear as the above, but still acceptable and still concise. 

The authors claim that their parallelised version of Q learning is motivated well and this seems to be backed up by experiments.

The overall implications of the authors' results are very significant: they have discovered and captured the root cause for TD instability, they have proposed an improvement which guarantees instability and they have showed that their new PQN performs exceptionally well on some examples.

### Weaknesses
A criticism is that the proof of the main theoretical results is long (there are 20 pages of additional material) and I would say not particularly well organised. Before the authors give the proofs, in my view, it would be good for them to outline the main steps. I found the proofs hard to follow and as one goes through the proofs, there is a feeling of being somewhat adrift. In other words, the summary in the main paper is good; the actual proofs in the appendix are less well clear.

### Questions
One question I have is why does the Jacobian have to be negative definite on the unit Circle? Why is simple negative definiteness not enough? Since any vector can be written as || u || = c || v || where c is a positive constant, it seems that simply negative definiteness of the Jacobian is required? It would be helpful to the reader if the authors could give more justification and/or insight into the reason negative definiteness on unit circle is required, or the authors may wish to reconsider their results and see whether the restriction can be removed.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes simplifications to multiple components of the Deep Q-Network (DQN)/TD learning method to enable more efficient training on a single GPU, offering a potential DQN baseline for future research. The modifications include:
- Eliminating target network tricks: The authors theoretically demonstrate that combining layer normalization with L2 regularization leads to convergent temporal difference (TD) learning. They then conducted experiments to validate this empirically by removing the target network update tricks.
- Removing replay buffer for experience replay: The paper identifies the replay buffer as a memory bottleneck that limits single-GPU training. While directly removing the replay buffer impacts sample efficiency, the paper demonstrates that when combined with vectorized environments, the GPU-based training method achieves better wall-clock time efficiency.
- Batch-wise rollout using vectorized environments: The paper implements DQN training in a batch-wise manner, leveraging GPU parallelization (after removing the replay buffer). Instead of single actor rollout, the paper uses vectorized environments to generate rollouts in batch.

To validate their approach, the authors conduct comprehensive experiments on both theoretical/proof-of-concept environments (Baird's counterexample) and standard benchmarks (Atari and Crafter). Their results show that the proposed Parallelized Q-Network (PQN) achieves comparable performance to well-known baselines like PPO and Rainbow. Through ablation studies, they further demonstrate the importance of network normalization and justify the removal of the replay buffer.

### Strengths
- This paper is well-written and easy to follow, with clear presentation of both theoretical derivations and experimental results.
- The paper's motivation is clear and compelling enough to me: it mainly provides a simplified Q-learning baseline that effectively leverages GPU parallelization and vectorized environments.
- The proposed experimental evaluation is relatively comprehensive: it covers multiple domains including proof-of-concept environments, standard single-agent benchmarks such as Atari and Crafter, and multi-agent scenarios. They also covers variants of the Q learning methods to support the claim better in general.

### Weaknesses
There is no major weakness of this paper, but feel free to check the question section for minor questions.

### Questions
- Including Baird's counterexample results in the main text would strengthen the paper in my opinion, by providing a clearer connection between the theoretical analysis and experimental validation.
- (Minor) The PPO baseline comparison in Figure 3 could be more consistent, though I understand the thinking to save compute. The paper could either compare both methods using 4e8 training frames in Figure 3(a), or include PPO results directly in Figure 3(b) across all Atari environments. Either approach would more effectively demonstrate PQN's competitiveness against established policy gradient methods, in terms of sample efficiency.

### Soundness
4

### Presentation
3

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
Modern deep-reinforcement learning resorts to techniques such as replay buffers and target networks to provide stability with nonlinear off-policy learning. However, learning becomes unstable without a replay buffer or target networks and can diverge. Recently, several works suggested using layer normalization or layer normalization in addition to l2 regularization to remedy this learning instability issue. This paper theoretically studies layer normalization’s role and identifies how layer normalization helps with stability and convergence. The paper also proposed a new method called PQN that uses layer normalization and parallelized environments to stabilize learning. The authors show the effectiveness of their method through a series of experiments on different domains of environments.

### Strengths
The work on simplifying deep reinforcement learning and removing techniques that might not be necessary, like replay buffer and target networks, is undoubtedly fundamental to deep reinforcement research. This has vast implications for rethinking the widely existing deep RL approaches and can help in other important directions, like scaling RL with the number of parameters/samples. This paper provides a unique view that challenges existing beliefs on the importance of replay buffers and target networks. The approach is effective and efficient since it can be implemented parallelized on GPU, which outperforms other baselines with respect to wall clock time.

### Weaknesses
- The theoretical part of the manuscript is largely incoherent. 
  - The current manuscript scatters many things in the theory parts. It lacks a proper flow of ideas when describing the theoretical results and their implications, which makes it difficult to follow. Currently, it reads as bullet points, listing findings quickly without proper linking between subsequent findings or results.
  - For example, the current theorems and lemmas are not well integrated with the text before and after them. They read as detached components, making reading unnecessarily harder.
  - Notations can be improved. I suggest following Sutton & Barto (2018).
  - Two things that are instrumental for the results based on Jacobian analysis: inequality 3 and inequality 4. More discussion is needed to understand these two conditions and their implications.
  - Off-policy instability and nonlinear instability require their own theorem statements and separate proofs (even if previous works have shown them). In addition, the TD stability criterion needs a theorem statement about contraction mapping. 
  - The connection for why l2 is needed was not clear. It was directly introduced after layer norm without proper linking.
- PQN solves a problem different from the baseline DQN, but this was never discussed
  - The authors emphasize that PQN does not use a replay buffer or target network as an advantage over other methods, which is great. However, a similar emphasis is needed for the fact that PQN requires parallel environments and probably may fail if a single environment were used (the setting of the other baselines). Additionally, PQN solves another orthogonal problem (parallelized worlds) to the original RL problem (single world).
  - Figure 1 is inaccurate. The replay buffer is part of the agent, not an external component. This needs to be fixed.
  - The difference between Distributed DQN and PQN is unclear from Figure 1, although both solve the same problem (parallelized worlds), especially the point on synchronism and GPU is not clear.
- The paper has several inaccuracies.
  - The authors claim that their PQN is based on Peng’s Q($\lambda$), but the actual algorithm does not use eligibility traces. Instead, the authors use Q-learning with $\lambda$-return. This needs to be corrected. Additionally, the equation (line 326) needs to be derived from the first principle to make the paper accessible to the unfamiliar reader.
  - Algorithm 1 is Q-learning with one-step targets. The authors mention that they use $\lambda$-return target, so Algorithm 1 needs to be replaced with the algorithm actually used (Algorithm 2 in Appendix C).
  - The authors claim they stabilize learning Baird’s counterexample and use Figure 7a to demonstrate that. However, in Figure 7a, the error increases from the starting point until it plateaus. I don’t think meaningful learning has happened since the error has increased instead of decreased. I see that with layer norm or layer norm + L2 regularization, the error doesn’t increase without bound but at the same time, the problem is not solved either.
- Unfair or unclear empirical evaluation
  - The authors compare algorithms that work with parallel environments against ones that do not, which requires careful experiments to compare them. 
  - In Figure 3, the authors need to write rainbow (200M) or DQN (200M) to understand what those horizontal lines represent clearly.
  - When the authors say that PQN was trained for 400M frames, does that include the parallel environments? For example, if 128 parallel environments are used (according to Table 5), does this mean 3.125M frames were collected from each environment, resulting in a total of 400M frames, or does it mean that 400M frames were collected from each environment, resulting in 128x400M=51200M overall frames? The first option gives a fair comparison, but the second option is biased towards PQN since significantly more experience is used. I would like the authors to clarify this point.
  - The authors used only 3 independent runs for Atari, relying on precedence.  This is a too low number to provide any statistical significance. Even if something was accepted before, it does not mean it is correct. I highly suggest the authors increase the number of independent runs to at least 10. This should be possible since both PQN and PPO are efficient (small clock time) and easy to run, according to the paper's claims.
  - The authors mentioned that DQN/Rainbow uses 50M updates compared to 700k updates for PQN. I think it is unclear how these numbers are obtained, especially for PQN.
  - In Figure 6a, PQN still learns well without divergence when no layer normalization is used. What is the reason? Why has no divergence happened?
  - Since the authors compare against DQN and Rainbow (methods that use a single environment), an ablation where a different number of environments are considered (e.g., n=1 and n=10) to understand the provided stability coming from parallelized environments. I think parallel environments make the gradient signal more reliable compared to the single environment case, which is more prone to noisy gradients. This may be instrumental for PQN; thus, an ablation is needed.

&nbsp;

**Minor issues:**

- line 9 in Algorithm 1 and line 10 in Algorithm 2 are incorrect. The negative sign should be a plus. Additionally, I think the use of $\texttt{StopGrad}$ can be eliminated with the TD error vector you gave in Eq. 2

&nbsp;
&nbsp;

Overall, I believe the ideas from this paper could serve as a good contribution to the community, but the current manuscript is not ready for publication and needs significant improvement. I'm willing to increase my score, given that the authors improved their manuscript's quality by 1) rewriting the theoretical part to make it coherent and more rigorous, 2) Fixing the inaccuracies, and 3) improving the empirical evaluation quality.

### Questions
- Is PPO using the same number of parallel environments as PQN in all experiments? I couldn’t find this information in the paper. Could you share this information and add it to the paper’s revision?
- In line 428, the authors refer to a histogram in Appendix E, but there is no histogram. Do they mean the bar plot in Figure 12?
- Typically, layer norm is not applied to the post-activation but instead to the pre-activation. This is an important distinction. The theory still works with the preactivation layer norm as long as you don’t use activation functions that scale up the inputs.

### Soundness
3

### Presentation
3

### Contribution
3
