# Percept Activation Graph (PAG): Decomposing LLM Computation into Perceptual Entities and Their Interactions

- Decision: Reject
- Scores: 0, 2, 6, 2

## Abstract
Understanding the computations performed by large-scale neural network models remains an important challenge.
Recent work has motivated holistic approaches that focus on population-level dynamics of neurons in these networks, suggesting that these dynamics reflect statistical regularities in data and that the human perceptual tendency for chunking can be leveraged to identify recurring cognitive entities.
We extend this line of work by introducing new techniques, inspired by cognitive science and neuroscience, to analyze LLM computations. We formalize chunking in neural data through the perceiving function, which maps recurring high-dimensional activities into a dictionary of recognizable entities. Building on this definition, we decompose neural activations in large-scale networks into a finite set of chunks and find that model activations exhibit compressible regularities across both tokens and layers.
Based on these chunks, we define the Percept Activation Graph (PAG) which captures the causal structure of chunks across layers.
We apply this analysis to LLMs to examine how they represent compositionality in context, analyzing layer-wise activations during in-context learning on the SCAN meta-learning dataset. Within the PAG, we identify distinct components that encode primitives, and demonstrate that perturbing these components predictably alters the model’s compositional generalization behavior. Our method provides a pathway to automatically extract structured relations between chunks that causally and controllably influence the computation of large-scale neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The authors study "chunking" as an approach for interpreting LLMs. They apply their approach to an LLM processing the novel *Emma* and a subset of the SCAN task. This study appears to overlap closely with Wu et al (https://arxiv.org/abs/2505.11576), with the primary novelty being the two new datasets and an attempt to measure transition probabilities between chunks.

### Strengths
The authors study an interesting and timely topic.

### Weaknesses
This work overlaps closely with Wu et al. (https://arxiv.org/abs/2505.11576), with differences seeming to be incremental and vague. Indeed, the concepts of chunking and grafting to measure causal impact all seem to derive from this prior work. It appears that the idea to measure transition probabilities between chunks may be new, but it's unclear how important this measure is, and what we gain by having it. Can we claim to understand an LLM better by having access to its transition probabilities between (mostly uninterpretable) chunks? Does this information grant meaningful predictive value? Can you predict whether an LLM will hallucinate, answer a question correctly, generalize to a particular OOD example, and so on, given this information?

This paper also introduces two new tasks (*Emma* and SCAN), but the depth and rigor of evaluation all appear to be far less than presented in the original Wu et al. work. Please see Questions below for additional specific points.

Additional minor comments:
- In-text citations are missing parantheses
- Inconsistent use of "perceptual function" vs. "perceiving function" vs. "Perceiving function" vs. "Perceiving Function" and so on. Please consider adopting a single nomenclature, and consistently using it throughout the manuscript.
- $\mathcal{P}$ = $\mathcal{D}$? These two terms seem to be used interchangeably. Sometimes, $\mathcal{D}$ seems to mean chunks, though chunks were labeled $c_e$ elsewhere. 
- Notation in general seemed to be inconsistent. Please avoid introducing variable names if they are not used again. Doing so tends to clutter and confuse the text, rather than clarify.
- Figure text is generally small, blurry, and unreadable. Please consider enlarging the text.

### Questions
- Figure 2: What's the takeaway from this figure? What have we learned from these chunks? Are they interpretable? What does it mean for a word to belong in a particular chunk? Are these good compression ratios? What would be the baseline for a good compression ratio?
- Figure 3: How should we read this graph? What is the advantage of having these transition probabilities? Do they tell us something more than we could have learned by performing simple model surgery on the weights and activations directly? What does panel 3b mean? Does perturbation/grafting work in this setting?
- Figure 6: the text is too small for me to read in this Figure, but the general sense seems to be that only a small portion of the chunks are interpretable? What have we learned about the SCAN task from this experiment? What have we learned about how an LLM can solve it, beyond the observation that grafting a hidden representation can sometimes change the downstream prediction in a consistent way?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a new method for analyzing the internal activations of language models, which unifies ideas from chunking in cognitive psychology and neuroscience with ideas from dictionary learning and neural network interpretability. This method results in a dictionary of “chunks” of activations that represent motifs in the neural manifold. These chunks are then organized into a percept activation graph, which tracks the co-occurence of chunks across layers. This method reveals structure in the neural activation space, resulting in a compressed chunk representation of language model activations when processing narrative text. This method also reveals causal sets of activations, responsible for carrying out simple algorithmic tasks in the SCAN dataset.

### Strengths
The connection between analyzing activations in neural networks and chunking in cognitive psychology/neuroscience is novel and interesting, and the formalization thereof is potentially useful for future studies. It is also useful to quantify the regularities within the neural activation patterns, as the authors do using their chunk compression ratio in Figure 2d.

### Weaknesses
This paper suffers from a serious presentation problem. The work seems to rely extensively on a dictionary learning algorithm that was introduced in a previous paper, but does not even partially explain how this algorithm works. Including this algorithm in an appendix is crucial for this paper to stand on its own. Furthermore, the applications of this algorithm are described at a high level in text, but many of the details are unclear. For example, I am not exactly sure what the authors mean by “We can then extrapolate the neural population activity of all LLaMA-3 layers as the network processes prompts from the corpus, and use the chunk indices closest to the embedding activity in each layer, to denote the neural level activity as the network’s computes the prompt.” And it is unclear exactly what the authors mean by “population-level” with respect to the hidden states of an LM. Does this always correspond to the residual stream activations in a single token at a particular layer?

The presentation of the experiments is also in need of serious revision. The description of the experiment that demonstrates that the chunks are causal is quite unclear, as is the description of how the compression ratio is computed. In general, all experimental sections should include more precise descriptions of the design and hypotheses.

However, the experiments themselves also seem potentially problematic. If I understand correctly, the causal experiments in Section 5 are demonstrating that representations at one layer cause representations at a future layer. This might well be true, but it does not mean that this is a nontrivial finding. For example, imagine if two chunks denoted the exact same activations within a subspace of the hidden state, just present in different layer dictionaries. Then one would get this causal result as a direct consequence of the residual connection between layers. While this is true, it is not a powerful demonstration of the merits of this method. To remedy this, one should 1) do some analysis of the chunks that are causally connected and 2) introduce a proper baseline, like replacing the chunk dictionary with random subspaces of the hidden states, and see if one gets the same results. 

Additionally, the use of SCAN with a pretrained model is a bit odd, as the point of SCAN is to assess whether from-scratch or meta-trained models can learn compositional behavior. When one applies it to a pretrained model, then the pretrained model already has access to the semantics of the tokens that comprise scan, giving it quite a helping hand relative to the models it was intended to be used with. Even so, this paper only seems to identify token-identity chunks, which are exactly what is given by the embedding of e.g., the token “run”. Again, a baseline would be helpful for contextualizing these results. Perhaps simply swapping out the whole hidden state would give you the same counterfactual performance!

### Questions
Why choose SCAN for this demonstration with a pretrained model?

Why choose the Emma dataset? Are there any specific insights to be gained from this dataset?

See “weaknesses” for other technical questions.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This study introduces a novel framework, Percept Activation Graph (PAG), for interpreting large language models (LLMs) by decomposing their internal neural activation into discrete, recurring cognitive entities called chunks, inspired by human perceptual chunking in cognitive science and neuroscience. The authors formalize this process via a perceiving function that maps high-dimensional activation patterns to recognizable entities. Using this, they extract structured interactions across layers in the form of a directed probabilistic graph (PAG), which captures how these chunks evolve and interact during computation. The work bridges cognitive principles with mechanistic analysis of deep networks, offering both conceptual insight and practical tools for understanding LLMs.

### Strengths
The integration of cognitive science concepts, particularly chunking and perceptual organization, into neural network interpretability is innovative and well-motivated.

### Weaknesses
1. Although the perceiving function is formally defined, the actual algorithmic implementation for identifying chunks lacks sufficient detail.
2. Limited Evaluation Scope: The experiments focus on a single model (LLaMA-3-8B) and a task (SCAN), which limits external validity.
3. The paper lacks a flowchart to clearly illustrate the algorithmic details.

### Questions
What is the computational overhead of constructing the PAG? Is it feasible to apply this method dynamically during inference, or is it primarily an offline analysis tool?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes to analyse the internal activations in large-scale networks by decomposing them into discrete chunks using an unsupervised learning method. The resulting transition matrix (Percept Activation Graph) between chunks as a network processes a prompt can yield insights into its processing strategy, including the ability to (imperfectly) predict effects of causal interventions. The paper uses this methodology to understand aspects of how a LLM can learn a compositional task in context.

### Strengths
The paper addresses a very important topic: how to understand the computations undertaken by a trained language model. 

The method to decompose activity into discrete chunks (from another paper, not this paper) is able to extract chunks with clear correlations to relevant inputs, and most impressive, clear usefulness in predicting causal interventions.

The analysis of in-context learning on the SCAN dataset is interesting, most notably in the demonstration that swapping chunks corresponding to primitives can cause a network to apply the right 'compositional' rule to the swapped primitive.

The paper is clear and the figures are easy to follow.

### Weaknesses
The extracted graph between chunks would seem to be similar to influence graphs extracted using SAEs, which also support (imperfect) causal interventions. The paper could be strengthened by more directly contrasting these approaches--what phenomena are accessible via discrete chunks that are not accessible via SAEs and circuit tracing?

Since the chunk extraction method was proposed in prior work, the contributions of this paper are primarily the extraction of the graph between chunks, and the application to a simple component of the SCAN dataset. The most concrete insight is that primitives can be swapped by swapping chunks. Further insights like this would help highlight the importance of the method.

Some of the insights obtained via the approach seem generic to the point that they may be inevitable--for instance, that chunks may be found that correspond to memory and prediction of words, or that chunks exhibit structure and regularity. It would be useful to find ways of demonstrating that this could be otherwise, to illustrate the insights the method provides.

A main contribution claimed by the paper is to break activity patterns into interpretable units using an entity-extraction method. The paper could be strengthened by explaining why this is a conceptual shift relative to an SAE, which arguably does a similar operation.

I am not sure that 'cognition chunks perceptual input into discrete recurring entitities.' Although we may know that we are looking at a cat (a relatively discrete entity), we also know many continuous things about that cat--the way its tail is draped, the patterning on its fur, etc. Perhaps it would be more accurate to say that part of cognition does this, i.e. it is one aspect of our perceptual representations.

### Questions
What phenomena are accessible via discrete chunks that are not accessible via SAEs and circuit tracing?

Is discreteness vs sparse latent codes the key distinction between the chunk method and SAEs?

### Soundness
3

### Presentation
3

### Contribution
2
