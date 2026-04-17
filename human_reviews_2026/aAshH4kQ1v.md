# Path Channels and Plan Extension Kernels: a Mechanistic Description of Planning in a Sokoban RNN

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
We partially reverse-engineer a convolutional recurrent neural network (RNN) trained with model-free reinforcement learning to play the box-pushing game Sokoban. We find that the RNN stores future moves (plans) as activations in particular channels of the hidden state, which we call *path channels*. A high activation in a particular location means that, when a box is in that location, it will get pushed in the channel's assigned direction.
We examine the convolutional kernels between path channels and find that they encode the change in position resulting from each possible action, thus representing part of a learned *transition model*.
The RNN constructs plans by starting at the boxes and goals.
These kernels, *extend* activations in path channels forwards from boxes and backwards from the goal.
Negative values are placed in channels at obstacles. This causes the extension kernels to propagate the negative value in reverse, thus pruning the last few steps and letting an alternative plan emerge; a form of backtracking. 
Our work shows that, a precise understanding of the plan representation allows us to directly understand the bidirectional planning-like algorithm learned by model-free training in more familiar terms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Sokoban is a 2d grid based game where the aim is to push blocks onto targets.  However, not all moves are reversible, meaning that solutions that work need a understand the longer time consequences of their actions to ensure they don't get stuck.  Deep Repeated ConvLSTM (DRC) is a Recurrent Neural Networks (RNN) model that can learn to play Sokoban. (Bush et al. 2004) demonstrated that that DRC is performing planning by using linear probes on the activations to predict the long term effects of its actions.  They also showed that the plans proceed both forward from the agent position and backward from the targets, and that it was possible via interventions to affect the plan that the agent took.

The current paper goes into considerably more depth about how the network is actually representing future plans, including assigning interpretable meanings to many of the different channels of the DRC network as opposed to only using linear probes.  This includes finding that each channel encodes a propensity to move in a specific direction.  They also separate out box movement and agent movement channels. Note that not all channels are interpretable, and the full meaning of the redundant layers of each type is not fully discernible either.  Furthermore, they show how the network can deal with overlapping channels, where the same grid square is used at different time points by highlighting the activations at different times.

They also dig into the specific ways the convolutional kernels are used to build up the plan over inference time (not agent time), including being blocked by walls and what they term Turn Plan Extensions.

They use this knowledge in a variety of ways, like scaling the magnitude of certain weights to scale to larger maps, and like Bush et al, intervening to make the agent take sub-optimal plans. 

Furthermore they demonstrate the winner takes all mechanism where short term plans suppress each other.

### Strengths
## Originality

The work is incremental on top of Bush et al, 2004, but it does go deeper into explaining how an RNN can solve the planning problem.  This level of detailed analysis of a planning RNN is to my knowledge an original contribution 

## Quality

It's not a fully generative description of the RNN - the knowledge they find is not sufficient to build one from first principles, so it is only a partial understanding.  This in turn leads to the paper begin a bit partial, like the knowledge they find.  However, the hypothesis they generate are generally well tested empirically and illustrated well in the paper.  

## Clarity
The paper is generally well written, and it would be straightforward to replicate their results and experiments.

## Significance
Planning is a vital step in understanding and controlling the world.  The current paper, while it is not a fully integrated solution may well lead to further insights into how planning is both learnt and performed in other systems like LLMs as well, which would be a very significant potential future impact.

### Weaknesses
As said before, it's not a full explanation of how the DRC networks works, it still provides various valuable insights into how different layers work.  Obviously being able to construct a simplified planning network given the knowledge would be a very powerful contribution, but that is outside of scope of this paper, and this paper is a step towards that goal.

### Questions
Did you examine any intermediate checkpoints to see when the different behaviors emerged?

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
3

### Summary
This paper analyzes a Sokoban-trained DRC(3,3) agent and shows that it performs internal planning through path channels and plan extension kernels. Using ablation and causal tests, the authors demonstrate that these components are responsible for planning behavior, providing a clear mechanistic view of how model-free networks can develop search-like computation.

### Strengths
This paper offers a detailed mechanistic analysis of the DRC(3,3) reinforcement learning agent trained on Sokoban. Its originality lies in the depth of interpretability achieved—the authors go beyond linear probes to reveal path channels and plan extension kernels that implement a neural form of bidirectional planning.

The study combines qualitative visualization, quantitative ablation, and causal intervention experiments in a careful way. The causal and ablation results convincingly show that these channels are genuinely responsible for planning.

Overall, it provides strong evidence that a model-free ConvLSTM can internally develop search-like planning dynamics, offering a valuable model system for studying emergent planning and mesa-optimization.

### Weaknesses
Despite its depth, the paper can be very difficult to read.  Readers unfamiliar with DRC(3,3) may struggle to follow the layer/tick conventions and channel indexing. A schematic overview early in the paper would improve clarity.

Another issue is limited generality: all findings are restricted to Sokoban. It remains unclear whether the same mechanisms emerge in other planning-heavy environments or different architectures. Some comparative evidence (e.g., other environments) would help validate the universality of these “path channels.”

The causal analysis is strong but statistical validation could be deeper: no confidence intervals or significance testing accompany success-rate reductions. Moreover, it is unclear whether the WTA mechanism and negative activation propagation are emergent or enforced by architectural biases. Explicitly disentangling inductive structure from emergent behavior would strengthen the argument.


Some other isseus:
1. Many references in the paper are broken. Line 834, Line 1050, Line 1215
2. I don't understand the "?" in Table 8.
3. Figure 15 is broken.
4. Line 041: agent's goal [rectange], should be "agent's [rectange] goal"?
5. L131: locationns. 
6. Line 20: What do you mean by "parst"?

### Questions
When multiple boxes are present, how exactly does the network choose which one to move first? The WTA mechanism seems to operate locally across directions, but does it also act across different entities (boxes)?

Could similar path channel dynamics be found in agents trained on non-grid planning tasks (e.g., MiniGrid, navigation, or graph problems)?

How stable are the discovered channels across random seeds or retraining runs—are their spatial roles (e.g., “box-right”) consistent?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper provides an important case study that
manually inspects the weights and the activations of a successful RNN-based Sokoban solver
that was trained by reinforcement learning.
The authors showed that, through manual inspection of activations and weights,

-   some layers model possible paths of the boxes (path channels)
-   some convolutional kernels will extend the paths from the init and the goal, as if running a bidirectional search (plan extension kernel)
-   resulting paths are rejected by winner-takes-all mechanism

These claims are checked qualitatively and quantitatively using various tools, e.g., by causal analysis, or by measuring the overlap between
the actual movable locations of the boxes and the predicted locations in the path channels.

This is a unique paper in the modern era; I have never read a paper like this,
to be honest, although I know the early NN literature has full of papers like this that
perform a similar manual probing on a CNN for handwritten digit recognition.

This paper may hype. However I don't know if Neurips is appropriate for this kind of paper,
because, to me, conference papers are usually about technical contributions.
I really don't know how to review or score this paper unlike other papers.
I admit I might be narrow-minded, and I prefer to defer the decision to SPCs/ACs.
I apologize that my review is unusually short despite being from the planning background.

### Strengths
I really do not have much to say.

It may have some implications on the RL-trained the language models,
especially those that reuse the same layer several times (e.g. diffusion language models, Universal Transformer, Sparse Universal Transformer)
and use think tags.

### Weaknesses
The analysis is performed on a particular set of weights of a particular RNN architecture using a particular dataset, random seed, etc.,
leaving it unclear whether the findings generalize to wider applications,
or even to a different set of weights from a different random seed.
(the authors acknowledged this in the appendix)

The paper does not have a technical contribution.
Rather, the authors simply manually found the existence of path channels,
which renders the use of linear probes / logistic regression probes unnecessary.

I still think it is a bit of a stretch to call the bidirectional path extension / retraction behavior as "running a search algorithm".
It is clearly not running a systematic tree search;
It may not backtrack when 3 directions all failed,
and it may incorrectly backtrack early before trying all 3 directions.

### Questions
A natural next step of this paper might be this:
What would happen if you manually bake in a similar kernel into the RNN during the weight initialization
and optionally fixing those weights?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper reverse-engineers a Sokoban DRC(3,3) ConvLSTM agent trained with model-free RL and shows that its hidden state contains path channels, which directly encode future moves for boxes/agent by direction. Planning emerges via plan-extension kernels that (i) initialize short path fragments near boxes/targets from encoder features, (ii) extend them forward from boxes and backward from targets, (iii) stop at obstacles via negative activations, and (iv) use a winner-takes-all inhibition to select between competing short-term moves. Quantitatively: ablating 59 labeled path channels drops solve rate by 57.6% vs 10.5% for 37 non-path channels; random 37-path-channel ablation: 41.3%. Path channels are predictive of future moves (AUC curves separate short- vs long-term). Causal interventions on PNA/GNA channels achieve ~99% action flips; movement channels ~88% (agent-movement lower due to condition mismatch). Weight-level analysis also explains backtracking (negative activation propagation) and shows “weight steering” (scaling extension kernels) stabilizes longer paths on larger boards. The authors argue this constitutes a concrete, mechanistic description of bidirectional planning in a model-free agent and discuss links to mesa-optimization.

### Strengths
1. Plans live in identifiable channels; no probes needed to find them.
2. Concrete kernels for initialize/extend/stop/compete; elegant WTA tying to action selection.
3. Targeted edits flip actions at ~99% via GNA/PNA; large drops from path-channel ablations. 
4. Long/short-term split with j-gate-mediated transfer explains overlapping plans. 
5. The “weight steering” generalization and a clean recipe others can reuse.

### Weaknesses
1. Risk of confirmation bias; limited inter-rater reliability reporting. 
2. Define solve-rate protocol, seeds, and statistical tests more precisely; report CIs on all numbers. 
3. One architecture (DRC(3,3)), one domain (Sokoban). 
4. No check on other DRC sizes, Atari-DRC, or different training runs. 
5. Interesting, but current evidence is circumstantial; could be toned down or supported with extra tests (e.g., explicit internal objective readouts). 
6. Success metric is “any alternate action”; consider targeted action flips and off-policy rollouts to quantify downstream task impact.

### Questions
1. How reproducible are channel group assignments? Provide criteria + agreement stats; release per-channel labels.
2. Does the same path-kernel story hold for different DRC depths/widths, training seeds, and other grid sizes without weight steering? 
3. If you zero WTA connections, what is the task-level effect on solve rate and branching? Similarly, for turning vs linear extension kernels. 
4. Report intervention success vs magnitude; do small signed nudges suffice to steer plans reliably? 
5. Can you predict the critic head from path-channel activations (counts/energies) across states? Provide R^2/ablation on the value head. 
6. How often do negative “stop” signals incorrectly prune valid paths? Any systematic failure modes (e.g., tunnels, multi-box couplings)? 
7. Please release weights, code for kernel visualizations, and the exact Boxoban splits used.

### Soundness
3

### Presentation
3

### Contribution
4
