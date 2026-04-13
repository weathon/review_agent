## Human Reviewer 1

### Summary
This work presents an automated and scalable method for locating features associated with semantic concepts in language models. Given a natural language description of a concept, as well as base and counterfactual prompts, a transformer encodes the description and uses the prompts to locate token positions relevant to the concept, and the internal feature responsible for representing the concept. They evaluate their method on the RAVEL dataset, achieving SOTA performance, and they verify that their method reveals true causal structure via steering experiments.

### Strengths
Interesting and creative application of transformers towards automated, scalable interpretability methods

Locating internal semantic concepts has important implications for steering/controlling model behaviour to better align with intentions

### Weaknesses
The RAVEL dataset and the notion of localizing features could be explained more clearly

An explanation of the role of the Householder transformation would be useful

### Questions
Is there some way of utilizing a pre-trained language model to assist in interpreting the natural language instruction, as opposed to relying solely on a model trained from scratch on RAVEL?

What was the motivation for using the Householder transformation?

How does this approach to locating concepts relate to the approach of representation reading (as in Zou et al. (2023) "Representation Engineering")?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
The authors present a method for identifying directions in the activation space (hidden state) of language models corresponding to chosen concepts of interest. In particular they present a technique which takes as input a base prompt, a counterfactual prompt and a concept label. It then attempts to isolate and transfer the representation of the concept from the hidden state of the counterfactual prompt to the hidden state of base prompt (by intervening on the the forward pass of the base prompt).

Their technique is a new variation of the DAS method. DAS learns a rotation and projection of the the activation space that isolates a causally mediating direction for a particular concept.
Their new method, HyperDAS, uses an additional transformer hypernetwork to choose which token positions in the base prompt to intervene on and which token positions in the counterfactual prompt to read activation values from. The hypernetwork uses attention blocks which read keys and values from the base and counterfactual hidden states.

### Strengths
Their method seems like a necessary and logical extension to address the problem of token matching when performing interventions with the DAS method. They report state of the art results on the concept disentanglement benchmark. This line of research seems like a promising approach to better understand the internal states of neural networks.

The authors conduct detailed ablation experiments that help to isolate which aspects of their methodology are most important to achieve strong performance.

They also are careful to consider whether their intervention is philosophically justified, or whether they are adding too much complexity to faithfully interpret the model. I feel mostly persuaded that their method does in fact uncover properties of the underlying model, rather than learning new representations (or at very least, it does not seem to be worse in this regard than DAS).

### Weaknesses
Figure 3(b) makes the claim that HyperDAS beats MDAS appear less impressive. At many intervention layers, MDAS appears to be superior, so this result feels somewhat cherry-picked, although it is true that layer 15 is by a small margin the best layer for both methods. (However, the claim on line 377 seems to contradict my reading of the graph, so I may be misunderstanding something). 

Although it is briefly explored in one of the ablations, the authors do not adequately explain why they train the hypernetwork to take natural language input. They train a separate hypernetwork for each domain, so it seems that there would be a limited number of possible queries, and they would be no need to train a general language model such that the intervention can be specified in natural language.

Nitpicks:
- Abstract: “identifying neural networks features” -> “identifying neural network features”
- Line 142: “targe concept”
- Line 159: the index j is not defined (and clashes with the j-th token index used later)
- Line 481: “For examples”

### Questions
- Is the hypernetwork highly specialized to the distribution of the RAVEL benchmark, or can it be used to isolate concepts using real texts from more natural sources?
- Can a single network be trained that performs all of the tasks in RAVEL? It seems it should be possible to train a hypernetwork to make very general interventions, given the natural language input.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 3

### Summary
This work proposes a hyper-network based approach to enforcing causal interventions  in foundation models. Given an instruction and two text inputs, one base input and a counterfactual input, the model aims to localize the token in the base input which answers the instruction and instead return the counterfactual input answer. All other aspects of the output should remain unchanged. To achieve this the model has two output heads: one for finding the corresponding answers in the two inputs, and one for transforming the token embedding of the base entity to the counterfactual embedding for the corresponding entity. Some ablations are conducted on the localization head which requires sparsity to work effectively. It is shown that for most entities in the RAVEL dataset the HyperDAS approach outperforms the MDAS baseline, especially at the Iso score which depicts that interventions are more controlled than for MDAS. This increase in Iso score never corresponds to a drop in Causal score and so the overall disentanglement score of the model is improved over MDAS.

### Strengths
# Originality
The use of a hyper-network to learn to automate causal interventions is novel to my knowledge. The exact implementation itself also appears novel, with the location and intervention heads being a modular and measurable approach to training the interventions. This also means that the separation into Causal and Iso scores is possible and this is useful for evaluating the model.

# Quality
Overall the model work is of a high quality with clear hypotheses and appropriate experimental design. I find the separate consideration of Iso and Causal scores to be useful and lend insight into the behaviour of the model. The more detailed consideration of the localization head is also useful and are important ablations in the study. In most cases limitations are clearly acknowledges - for example on lines 421 to 429 where it is mentioned that the model is sensitive to the sparsity hyper-parameter.

# Clarity
Overall the  paper is well written and language appropriate. The structure of the work is also intuitive and aids with the understandability of the work. Figures are clear and legible with helpful captions.

# Significance
The improvement at disentanglement of the model over the baseline presents a clear step forward. In addition the detailed ablations and insight into the localization head could lead to future work. Overall I think this work is of clear interest to the field and makes a clear contribution towards causal interventions in transformer models and working with foundation models.

### Weaknesses
# Clarity
Some portions of the model are not fully explain. For example in Equation 1 the inverse of the low-rank orthogonal matrix R is used ($R^{-1}$) but if this matrix is low rank how is it inverted? Is this a pseudo-inverse? Another example is how the localized token positions are used to apply an intervention. In Figure 1 (left) it is shown that the positions are fed into the intervention module but Figure 1 (right) does not show this. I think exactly what the input and output of each head is could be described in far more detail. For an intricate model this is very important and limits clarity. Another part of the model which is not explained is the learning of the rotation matrix. Firstly, why is it necessary to learn $R'$ first and then apply the Householder transformation? Why not directly learn $R$? What does the Householder represent in an embedding space such that it performs a useful computation here? I do think on the whole the model and its workings become evident but this could still be far clearer and more explicitly explained.

# Quality
On the side of quality I think a couple limitations are still not given due consideration. Particularly, the fact that a hyper-network is trained for every entity type. This seems to add a huge amount of computational overhead and limits the scalability of this approach. However, beyond line 92 I do not see this being discussed anywhere. Similarly, there are a couple statements such as the claim that the method helps ``crack open black box models'' on line 415 which do not quite seem true. I am not certain how this  approach does this and it is not clearly explained. Similarly I think there is a lingering assumption that a single token in the base input maps to a single counterfactual token. This seems to be a property of the dataset (which is totally fine) but then I am not certain the black box has been cracked open when the model then identifies this property of the dataset. Lines 514 and 515 have a similar issue for me. Perhaps I am not appreciating some nuance to this, but I think the model makes a clear contribution without needing to be too far reaching in its claims. For example, the ability to easily and clearly manipulate the black box seems equally impressive to me - especially when considering the precision of the approach demonstrated by the Iso score. Similarly, for Figure 6 - it is stated that HyperDAS learns different feature subspaces for different attributes but in general the clusters are very tightly packed. I don't think that this statement is clear from the figure but also doesn't seem to be the main point of the work anyway. Lastly, I think more consideration should be given to the fact that when using too much sparsity loss the model does not behave appropriately but still obtains a near perfect disentanglement score. This demonstrates a limitation of the experimental design. I note that this phenomenon is  noted in the work, but it stops short of actually acknowledging it as limitation of the method which is necessary. I recommend the work is revised such that the claims are made more exact.

If I am not misunderstanding something, then I would increase my score one level for each of the two main points above: 1) more clarity needs to be given on the exact details of the model, 2) the claims needs to be made more precise.

### Questions
I have left some questions in the weaknesses section above which I would like answered. However two clear question are:

- How much extra computational overhead is added when using HyperDAS over MDAS?
- A small one: is $y$ missing from the end of Equation 13?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
5

---

## Human Reviewer 4

### Summary
This work proposes a method for automating the selection of particular linear directions in feature space that represent interpretable concepts or features. In prior work, model steering or activation patching has been performed utilizing optimization or datasets of prompts, but each method requires some manual effort or search in order to pair a particular neuron to a corresponding concept. This work proposes to use a hypernetwork that is conditioned on a counterfactual prompt, as well as intermediate features of an LLM with respect to a base prompt. The hypernetwork predicts for each token of the counterfactual prompt, its corresponding position in the base prompt in addition to an aligned counterfactual representation that is able to "override" the base prompt features. 

The hypernetwork is trained on RAVEL, with cross entropy to enforce correct token pairings, as well as a sparsity loss to encourage a one to one mapping between tokens. When evaluating on RAVEL, generations are scored according to whether or not the target attribute was successfully changed, and also on whether or not untargeted attributes were left alone. Evaluations compared to MDAS are favorable, and ablations show

----------------------------------------------------------------------
I believe this work will be of interest to the community. I hope that the discussed changes and additions make it into the final camera-ready. I will keep my score.

### Strengths
Originality / Significance: 
An automated method for finding / aligning concept directions at the token level is significant. I also think its important to note that it even performs better in evals than non automated methods. 

Clarity/Writing: I found the paper to be well written. 

Results:
One thing I found particularly compelling was the ISO score of HyperDAS on RAVEL. When performing interventions for model steering (or even SAEs), there is often a failure to isolate/disentangle particular directions, yielding steering in a spurious direction. A very high ISO here is a good sign that the discriminative power of the hypernetwork is high. 

I also appreciated the discussion of what we're really investigating or uncovering when we train supervised interpretability tools.

### Weaknesses
The largest weakness of this work is lack of baselines or evaluations, there is the one dataset and one baseline method. Not that interpretability methods have to be quantitatively better than others, but contextualization helps. I would encourage showing additional baselines. 

This does not influence my score, but I am broadly concerned that any method trained to perform these interventions is adding information that is not in the model under investigation.

### Questions
1. What makes this a hypernetwork? Hypernetworks should predict some weights of a target network. 

2. Since you don't actually clamp features to 0, sparsity is only softly encouraged with the sparsity loss, meaning there is always some feature entanglement. I'm confused about why adding too much sparsity loss results in poor performance. Doesn't better disentanglement imply that we're targeting only the target attribute? 
More specifically, figure 8 states "[...] it demolishes the model’s ability to form interpretable intervention patterns and adhere to specified constraints". What does this mean?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3