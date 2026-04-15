# Planning in a recurrent neural network that plays Sokoban

- Decision: Reject
- Scores: 3, 3, 3, 5

## Abstract
How a neural network (NN) generalizes to novel situations depends on whether it has learned to select actions heuristically or via a planning process. Guez et al., (2019, "An investigation of model-free planning") found that recurrent NN (RNN) trained to play Sokoban appears to plan, with extra computation steps improving the RNN's success rate. We replicate and expand on their behavioral analysis, finding the RNN learns to give itself extra computation steps in complex situations by "pacing" in cycles. Moreover, we train linear probes that predict the future actions taken by the network and find that intervening on the hidden state using these probes controls the agent’s subsequent actions. Leveraging these insights, we perform model surgery, enabling the convolutional NN to generalize beyond its $10 \times 10$ architectural limit to arbitrarily sized levels. The resulting model solves challenging, highly off-distribution levels. We open-source our model and code, and believe its small size (1.29M parameters) makes it an excellent model organism to deepen our understanding of learned planning.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper investigates an RL agent that can pace itself by taking extra computational steps in complex scenarios. The paper claims that by adjusting the network's hidden states, they extended its capabilities, allowing it to solve much larger and more challenging levels, and have open-sourced the compact, 1.29M-parameter model for further study in learned planning.

### Strengths
The claims of the paper are ambitious.

### Weaknesses
Some claims seem unrealistic, for instance in Figure 2, it is written "Left: XSokoban-31, which the DRC(3, 3) solves with 0 thinking steps", I'm not convinced that this claim is correct. Are the trained models already shared such that this could be checked? (beside the promise "We will open-source our code and trained model(s)").

Some words do not seem to have a clear meaning, e.g. in the abstract "perform model surgery" and "makes it an excellent model organism".

Some other parts are unclear: 
- For instance, at the end of the introduction, it is written "The evidence highly suggests that the RNN implements an online search algorithm". What evidence exactly is there for such a "high suggestion"?
- Beginning of Section 3.1 "the extra thinking time disproportionately helps with more difficult levels": what does disproportionately mean in this case?

### Questions
See questions in the weaknesses about the unclear points of the paper.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper studies the behaviour of a Deep Repeated ConvLSTM (DRC) network architecture that learns to play Sokoban. The paper specifically documents the phenomenon of “pacing” whereby the network moves in cycles before performing goal-directed actions. The authors interpret these behaviours as an indication that the model gives itself additional computational steps before the execution of a strategy. The authors also engage in an interpretability analysis of the models via the training of linear probes on LSTM states. The authors find that probes (logistic regressions) trained on these states are able to decode a variety of task relevant variables such as the future movements of boxes and agent movements for locations in the grid world. The performance of these probes varies over the different probe targets but are generally predictive of future actions. One specific type of probe, encoding the direction of boxes, is also successfully leveraged to intervene in the model activations to change (steer) model behaviour. Finally, the authors modify the DRC decoder to allow the model to somewhat successfully generalise to new puzzles of differing sizes and with variable number of boxes.

### Recommendation

I think the paper should be rejected because (1) the claims are not well supported, (2) it is unclear and unpolished in many places, and (3) I think quite disconnected from the literature it claims to be connected to. See more detailed criticisms above.

### Strengths
- **Nice task.** I think the use of Sokoban is interesting as it is a hard planning problem that has many failure modes for greedy models which perform actions without careful regard and deliberation of future actions. I think research that interprets how models perform planning is especially relevant now with current developments happening in the reasoning abilities of large (language) models.  
- **A variety of interpretability techniques.** The authors make use of a variety of techniques of mechanistic interpretability.  
- **Planning in a model free setting.** I think it is interesting that the authors attempt to explicitly study the ability to plan in a model-free agent, where historically planning has been in the camp of model-based RL. DRC represents a very flexible model class with minimal inductive biases. This flexibility leaves open if the model learns to plan in the traditional sense. It is an interesting question if model representations and computation might represent plans explicitly.

### Weaknesses
Despite these strengths I think the paper has many weaknesses and does not do a great job of elucidating if the model is planning. In addition, I think the paper is very unclear in many places. 

- **Framing of results.** The paper is abundant with sections in which the authors make quite substantial and large claims. I.e. in the introduction claims that “linear probes show that activations represent a plan that causes actions” and that “the behaviour of the RNN suggests that it performs search”. However, Section 4 openly admits that they find little evidence to suggest that the model is performing search. I think the claims that the paper makes and the presented evidence are quite disconnected and need substantial revamping. I also think that a contribution statement would make clear what really is being contributed.  
- **Strength of evidence.** I do not think some of the author's evidence is strong enough to support their claims. In Section 4 on linear probing the predictive power of the next box and next target probes are much lower than direction probes. I think intuitively should these not be a much better measure of planning depth as they are more relevant to long term plans? The results on generalization also appear somewhat limited as the performance is not as good as the intro and abstract might make you believe.   
- **Novelty.** Section 5 shows some generalisation ability of DRCs but this has already been shown in a variety of settings by Guez et al. (2019) and I think it is not super clear what the current paper adds.   
- **Clarity.** I think there are also places in which the elaboration of the methods used are not clear. For example, I found myself quite confused about the methods used in Section 5.1 *Spatial aggregation*. I was looking in the appendix but could not find an exact explanation of what was done here. What exactly is being aggregated? Also, what is the number column in Table 3?  
- **Interpretability results left uninterpreted.** There are several places where the authors fail to give sufficient explanations, interpretation, or context for their findings. For example, the section on linear probes finds good predictiveness but *much* worse causal probe results. Why is this the case? I felt like I was just left wondering as the reader.  I am also wondering what is special about pacing as opposed to other forms of taking inconsequential actions that are not cyclical, both should yield the same result, right? I think the authors would have to give an explanation about what is special about pacing.  
- **Unpolishedness.** There are several places where I felt that the paper was not well polished and presented. For example the caption in Figure 5 does not have labels for the subplots (i.e. left, right, centre) even though these are referred to in the main text. Similarly, in Figure 3 the x-axis labels are below the subplot caption, which makes the figure unclear. Further, often the terms NN, RNN, and DRC are used interchangeably. I think it would be good to just stick to one term referring to the model (DRC is used the majority of times). There is also the sentence “we did not bother collecting an i.i.d. test set” in Section 5, I personally think such language is not perfectly appropriate for a paper.   
- **Placing in the literature.** I think the paper does quite a poor job of placing the paper in the literature. Especially the related work section (Section 6\) I find quite strange. For example it is not clear to me how the presented work connects to the “Ethical treatment of AIs”  or to “Goal misgeneralization and mesa-optimizers for alignment”. 

### Potential Improvements
1. The claims should be appropriately fitted to the findings.  
2. The clarity of the paper should be improved with respect to the methods used in Section 5.1.  
3. The related works section should be more carefully written.   
4. Perhaps reconsider having the section on sparse autoencoders in the main text. If they do not perform well then why have them with your main results as opposed to the appendix?

### Questions
1) Why does the causal probe perform so poorly compared to their predictive power?  
2) What is special about pacing as opposed to other forms of moving around?  
3) What is being aggregated over in Section 5.1.?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper makes an study on how a neural network within an RL agent learns to play Sokoban. Specifically, this work studies how and where the network abstracts the problem at hand, plans and where decision making takes place.

### Strengths
The paper explores an interesting research direction on how a neural network within an emobied agent abstract their problem at hand, projects a plan and makes the decision making. It is an interesting problem that trhough a sound study can be very useful and benefitial for the community

### Weaknesses
While the line of research is of interest, the current paper has important shortcomings that require significant changes. 

Specifically, there are three major weaknesses in this paper:

* Clarity: As I will detail below the paper is very difficult to understand and it would greatly benefit from a rewritting. Specifically, I would suggest authors to do a simple exercise at the beggining what are the hypothesis that are being tested, how this thesis is going to be tested and what results would be interpreted as a validation or violation of the hypothesis. Additionally, there are several concepts that are never introduced in the paper or that are introduced first and explained better. Last, a diagram about the architecture studied and where the probes (i.e., the system to track the hypothesis) are placed within the architecture would greatly help readers

* Generalization of the results: the whole study focuses exlusively on the environment of Sokoban and studies only two neural networks, casting doubts about how general the conclussions extracted are.

* Related literature. There are several works that study which drivers help best to RL agents to generalize in similar problems, both within the environment, the nn architecture or the perception field. The present paper doesn't follow any of the recomendations of those works,and they can be influential in the conclussions extracted, e.g. [1] demonstrated that for an RL agent to generalise and not memorize puzzle solving, should have an egocentric perspective, how does using a egocentric perspectice  or not effect planning?. [3,4,5] highlighted changes in the ability of rl agents to generalize their plans when using different architectures, do we observe changing on abstractions when using any of those configurations?


[1] Hill, Felix, et al. "Environmental drivers of systematicity and generalization in a situated agent." International Conference on Learning Representations.
[2] Hill, F., Tieleman, O., von Glehn, T., Wong, N., Merzic, H., & Clark, S. Grounded Language Learning Fast and Slow. In ICLR 2021.
[3] León, Borja G., Murray Shanahan, and Francesco Belardinelli. "Agent, do you see it now? systematic generalisation in deep reinforcement learning." ICLR Workshop on Agent Learning in Open-Endedness.
[4] León, Borja G., Murray Shanahan, and Francesco Belardinelli. "In a Nutshell, the Human Asked for This: Latent Goals for Following Temporal Specifications." ICLR 2022.
[5] Lake, Brenden M., and Marco Baroni. "Human-like systematic generalization through a meta-learning neural network." Nature 623.7985 (2023): 115-121.

### Questions
Besides my comments above some specific points on clarity:
* You never explain what probes are and how their work, if their reader is not already very familiar iwth them they won't understand the work.
* Lines 153-156 is difficult to follow the thinking process here, for instance you mention there that adding extra thinking steps solve 4.7% extra levels, but in line 157 you mention that the extra thinking  helps greatly in dificult problems. Are only a very small percentage of the results these har problems?
* It is not clear how the ResNet would perform cycles, the few it does
* Line 200-202: here is a good example of a point that a research question is made but the paper only offers evidence that "suggests" not that confirm. In that and the lines below i would be good to put down the reasoning that would give us certainty that the agent is pacying, and giving an intuition of what is pacing here.
*  The later two paragrapghs in section 4.0 would be better placed as a summary at the end of the section pointing out to where the claims are supported.
* Line 278 explain what is the standard of evidence in Li et al.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper dives into understanding the inner mechanisms of planning by deep reinforcement learning (RL) agents in the Sokoban game, especially using a deep repeated convLSTM (DRC) (Guez et al. 2019) network structure. This work proposes several interesting findings including the effect of thinking steps, "pacing", probing the interpretable information, and generalizability to different sizes of Sokoban. All these results unveil a path to open the black box of a (recurrent) neural networks for model-free planning. I believe the major contribution of the paper lies on, like the authors stated, understanding the inner objective of the planning network.

### Strengths
1. The current manuscript is highly informative a number of interesting analysis, many of which are novel to my knowledge (beyond those discussed in Guez et al. 2019).
2. Some of the results are thought-involking, which I have not thought about, such as the "pacing" mechanism.
3. The being touched task (planning in Sokoban) is quite interesting (while simple enough for humans to interpret) and potentially important for understanding planning.

### Weaknesses
While the current study presents a number of interesting results. I believe the manuscript could be improved in several aspects.

1. The paper could benefit from a better struture and more scientific style of writting. The current version, while informative, is not self-contained and often makes me feel difficult to understand the implementation details of each analyze method. The authors should assume minimal pre-knowledge of the methods used in this paper, for a more general audience.
2. The paper does spend enough efforts to convince the audience that the presented results are not cherry-picking. Although I tend to believe the presented findings are mostly general, the paper could be improved by e.g., discussing the variety among random seeds in training, and providing more error bars in the plots.

See the following questions for detailed comments.

### Questions
1. Could the authors clarify the relation to Guez et al. 2019 by explaining the overlapping parts between this paper and Guez et al. 2019, and new results.
2. For self-containness, could the authors provides a preliminary about DRC since it is the backbone of the current work.
3. Could the authors provides some illutrative examples  (rendering) of "pacing"?
4. How does the network deal with different size of input (other than 10x10) in the CNN part (Sec. 5.1)?  I am confused about how the convLSTM was adjust in Sec 5.1. Maybe a visualized diagram could help. 
5. What was the learning objective in spatial aggregation ("We learn their relative weights ....... for 10000 steps.")?
6. DRC was compared only with ResNet, how about plain LSTM (as in Guez et al. 2019)? And I am curious about how a Transformer network performs in this case. 
7. Many of the plots seems to miss error-bars, is it because the error bars are too small? Could the authors clarify the error bars (and number of repeats) not only in Tables but also in plotted curves?
8. In the Related Work section, could the authors make a more concrete explanation about "reasoning NN architectures"?
9. Is it necessary to include "ethical treatment of AIs" in the main texts? 
10. Could the authors clarify "perhaps due to errors" ? (line 790)

### Soundness
3

### Presentation
2

### Contribution
3
