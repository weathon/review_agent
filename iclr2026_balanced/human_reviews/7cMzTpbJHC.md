## Human Reviewer 1

### Summary
The manuscript addresses the question of how artificial recurrent networks can store information, specifically in settings when the dimensionality of the state space is smaller than the total number of features that needs to be represented. Specifically, the authors investigate a phenomenon similar to that of spatial superposition, but in the temporal domain. They find that, when inputs are temporally sparse, even small recurrent neural networks can represent features over a comparatively long time by packing them into a subspace of the dynamics that is orthogonal to the network output, and by forgetting inputs once they become irrelevant. The manuscript focuses on a rather simple task in very simple RNN (linear and non-linear, low-d) to achieve a thorough understanding of the underlying mechanisms, combining analytical insights and simulations.

### Strengths
The paper is well motivated and well written. Relevant literature and the implications of the work are discussed appropriately. The paper is technically strong.

The focus on simple, small RNN, and on a single simple task, allows the authors to achieve a deep understanding of the underlying mechanisms. In particular, the authors combine simulations with insights from analytical derivations, which together make a very convincing case for their findings and conclusions.

The paper makes an interesting conceptual advance towards understanding the basic principles by which recurrent neural networks operate, by identifying a phenomenon akin to spatial summation in the temporal domain.

### Weaknesses
The focus of the paper on very simple, small RNNs and a single, comparatively simple task allows the authors to achieve a deep understanding of the underlying mechanisms, but also raises some questions about the broader relevance of the resulting insights. The main weakness may be the exclusive focus on the k-delay task. Many, if not all, findings about temporal superposition presented in the paper seem to be a direct consequence of using this task, which requires RNNs to remember an input feature for a fixed number of time-steps, then output it, and finally forget it. The authors make a strong case that that the type of dynamics they observe, which involves their mechanisms of temporal superposition, is “optimal” for this task. In fact, the mechanism they describe “makes a lot of sense” given what we know about RNN trained on simple tasks and is arguably not entirely surprising. It is less clear what are the implications of their findings to other tasks, which may require RNN to maintain information for a variable amount of time (variable k, whereby e.g. the RNN output is interrogated by providing a dedicated “go” signal) or to do more than just remember an input feature (e.g. by producing outputs that could control simple movements). 

In particular, I would expect that for a variable k (i.e. a randomized time between input and output) the type of solutions implemented by the RNNs are very different. In simple neuroscience decision-making tasks, RNN trained with fixed delays are known to produce rotational dynamics that is timed just right to put their activity into the right location of state space when say an output is needed, which is exactly what the authors find. On the same tasks, training with random delays instead leads to stable representations during the delay period (based e.g. on stable fixed points). In such a setting, temporal summation does not seem to be relevant, and instead of spatial summation is at play.

### Questions
The authors should discuss or explore the implication of their findings for other tasks, specifically also tasks that involve variable, randomized delays, whereby RNN outputs are triggered by the onset or offset of a dedicated go cue. 

The authors should relate their work to past studies with RNN that have found dynamics that seem closely related to what is shown in this manuscript.

I found the description of the phase transition hard to follow, only few results are shown to validate the point. The corresponding section is less clear and weaker than the remainder of the paper.

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
Authors consider a set of linear and nonlinear RNNs trained to perform a K-back task and study the properties of learned solutions, primarily for RNNs with 2 units. For the linear case, authors are able to write down the loss function in terms of four distinct components and name the loss contributions in terms of what they call projection and composition interference. Essentially, this paper has the flavor of applying interpretability methods used in LLM world to the studies of computation performed by RNNs.

### Strengths
- The analytical derivation of the loss components is novel, and insightful for training and reverse-engineering RNNs to perform the K-back task. 
- The paper is packed with a lot of fun to read results/observations, e.g., the phase transition, the generalization to multi-input case and how all but one inputs get dropped out due to constrained representational capacity. 
- To me, this paper was instrumental in realizing how far ahead we need to go to in our LLM interpretability studies compared to what the field has achieved for RNNs. So, it is also a nice toy model and has pedagogical implications.

### Weaknesses
- As authors admit themselves, the work focuses primarily on a single task and a very small RNN. There is not necessarily a generalizable insight/evidence that I was able to take away, and that does diminish the contribution.
- I find that the authors are less direct with how far RNN studies have come (specifically, the low-rank RNN studies). In many cases, we can now study the low-dimensional subspaces learned by RNNs and draw the exact flow maps they learn (not just the feature dimensions). In that sense, more modern discussion of low-rank RNNs is desirable.
- It is unclear how many RNNs have developed these representations (especially for nonlinear ones) and how many failed to learn or learned completely different solutions. More rigor in reporting is desirable.

### Questions
I only have few questions/comments:

- Eq. (1) is actually the more common RNN architecture in computational neuroscience. Hence, Appendix F.1. is making incorrect claims. Also, the connection of Eq. (1) to other form of vanilla RNN is well known. See [1]. 

- Please emphasize in the abstract that your main contributions hold exactly for linear RNNs, and then you show that some of the insights do generalize to nonlinear counterparts. However, generalization beyond the particular task is not shown, please state this explicitly as well.

- What makes feature A different than others in Figure 4? 

- Figure 5 is very interesting, why is it in Appendix? I would argue it may be the most interesting figure in this paper.

- Are you aware of [2]? If so, how does your work, especially Fig. 5, compare to their findings? In my reading, it seems you also find that geometric restructuring (as Haputhanthri et al. 2024 has defined it) can happen even without emergence of attractors, as your case simply uses a spiral. That seems like a noteworthy connection/extension of prior work as well.

My overall assessment is as follows: This work is limited in ambition and has some issues with rigor, both of which limit its contributions. However, it does introduce an interesting toy model and compared to how far we have come with RNN analyses, it does show how rudimentary the superposition/interpretability ideas in LLMs are. We need to do a lot better. With correct placement into the RNN literature, in which we can now study high-dimensional RNNs and their learned algorithms exactly (see recent low-rank RNN studies), I think this manuscript can be a fun read for the ICLR audience. 

[1] https://pubmed.ncbi.nlm.nih.gov/22023194/
[2] https://openreview.net/forum?id=njmXdqzHJq

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper investigates the representation geometry of RNNs, focusing on how time affects the so-called superposition hypothesis. 

The authors introduce the concept of temporal superposition, analyze the resulting interferences and their interaction with spatial superposition, and derive analytical results on a simple task.

### Strengths
Very well written, organized and easy to follow. The figures helps.

The initial analysis of a simpler model help intuition.

Great related works, very comprehensive.

Important generalization of an hypothesis that has garnered a lot of attention in the community. Significant.

### Weaknesses
Technically, the contribution could feel somewhat incremental compared to prior work on spatial superposition. However, the significance is still there.

Many key results are presented only in the Appendix, which reduces the overall impact of the paper. This makes me wonder whether the work might be better suited for a journal that allows a longer format. For instance, Figure 9 is particularly helpful for understanding the core concept but is unfortunately buried in the Appendix. 

The analysis with multiple dimensions should also be moved to the main text and discussed in more depth.

### Questions
The qualitative results shown in Figures 2 and 3 are a bit unclear: are these plots averaged over 100 runs? or are they representative examples? Could you include error bars to assess variability.

Does the analysis of spatial or temporal superposition provide any insight into how we should train or test our networks? Beyond being conceptually interesting, are there practical consequences of this phenomenon of temporal superposition?

Have there been any empirical observations of temporal superposition in animal data?

Minor: Figure 3 is not cited in the main text.

I’m a bit unclear on why feature A competes with itself through time, since, in effect, the latent space at time t is different from the latent space at time t+1. The network doesn’t hold At and At+1 simultaneously, only ht worries about At and ht+1 worries about At+1. I’m viewing the RNN as unrolled.

Put another way, could temporal superposition also occur in a feedforward neural network, with the layer index playing the role of time? If not, why not?

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 4

### Summary
In this paper, the authors introduce a novel concept of temporal superposition in RNNs, the idea that beyond spatial superposition of features, memory demand also poses a pressure for the network to pack features presented at different time steps into limited hidden dimensions. 

The authors analyze RNNs on a k-delay task and identify two interference modes: projection interference and composition interference. They analytically decomposed the loss into four interpretable terms,  including the task benefits and the two interference modes, which can be used to explain the learned geometry of feature directions in the hidden state. The authors also identified a phase transition in RNN's representational geometry as a function of the sparsity of features. In their analysis of nonlinear RNNs, the authors provide a geometric perspective on how nonlinear RNNs can implement sharp forgetting while linear RNNs can only implement gradual forgetting. 

Finally, the authors also study the interaction between spatial and temporal superposition as a function of memory demand, and discovered that as memory demand increases, the network goes from representing all features equally to representing only the most important feature for all time steps.

### Strengths
- This paper presents very interesting results regarding temporal superposition of features in RNNs under memory demand, which I believe can be of broad relevance for the mechanistic interpretability and computational neuroscience community. 
- The concept of temporal superposition is novel, and the authors use well-designed experiments to illustrate this effect on three types of RNNs while contrasting the role of nonlinearity in feature representation. 
- The authors provide clear, interpretable theory on RNN geometric strategy by decomposing the loss into four terms that correspond to the task performance, correction effect, and two types of interference. 
- The paper is exceptionally clear: key concepts are precisely defined, notation is consistent, and figures concretely illustrate the geometric claims, making it easy to follow.

### Weaknesses
- The core analysis revolves around the simple k-delay task. While this isolates temporal superposition effectively, it may not capture real-world memory demands involving manipulation, variable delays, or context-dependent processing.
- Most results are in low-dimensional hidden states (Nh=2), with higher dimensions (up to Nh=10) only briefly explored in the appendix. Scaling to realistic model sizes could reveal different behaviors but those were not considered or discussed.

### Questions
- The k-delay task effectively isolates memory demands, but as noted in Limitations, it focuses on reproduction rather than manipulation. What preliminary insights do you have on how temporal superposition manifests in tasks with variable delays or manipulation of information?

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
4