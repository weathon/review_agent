# On the Dynamics of Coherent Memory Structures in Neural Fields

- Decision: Reject
- Scores: 4, 8, 2, 4

## Abstract
Memory in biological neural networks is often supported by coherent spatiotemporal patterns, such as traveling waves and neural activity confined to low dimensional manifolds, which are captured at mesoscopic scales by continuum neural field models. Substantial progress has been made in mechanistically analyzing both biological and artificial neural network architectures. Recent works obtain interpretable latent states by imposing traveling waves or low-dimensional invariant manifolds, but typically do not provide data-driven explanations for when and why such structures emerge during task training. We develop a theoretical framework for studying latent dynamics based on the Mori-Zwanzig projection-operator formalism. Our approach casts memory as a family of time-dependent projections that reveal how coupled dynamics support memory encoding and decoding. We instantiate a neural-field-inspired architecture, and evaluate it on both long-range benchmarks and neuroscience applications involving EEG and ECoG prediction. Across these tasks, we observe robust long-range accuracy and interpretable memory modes in the learned latent dynamics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
This paper proposes a theory that can account for coherent “memory structures’, i.e., patterns of dynamical systems such as biological networks and RNNs: stable attractors and traveling waves.

In particular, it learns a time-dependent projection operators that allow to capture coherent patterns. 

Inspired by the theory, they propose a novel neural architecture, termed “neural wave field” that can automatically discover memory patterns and enhance long-range memory.

### Strengths
S1. The related works on how attractors and waves have appeared in neuroscience and AI is very comprehensive.

S2. The authors introduce a solid mathematical formalism.

S3. The experiments are extensive, across synthetic and real datasets. This is impressive since the paper has both a strong theoretical and a strong experimental component.

S4. The paper tackles the important problem of understanding memory in neuroscience and AI, and enhancing it for AI architectures.

### Weaknesses
I am torn about this paper because it seems very comprehensive and thorough, mixing strong mathematics and extensive experiments, and addressing an important problem of understanding and improving long-term memory. However, despite several readings, I have trouble understanding the framework.

W1. The theoretical sections are hard to grasp. There is quite a step between the introduction and description of memory/patterns in biological and artificial neural networks, and the intense mathematics of the background. It would have been helpful, during the background to explain which math concept correspond to what in the neural network. And to use figures to introduce the concepts. See my questions to the authors.

For example, it would have been helpful to define “resolved” and “coherence” and give their intuition. The authors seem to assume that the readers are familiar with these terms.

Likewise, GFDR is introduced very rapidly, and while it seems crucial to the development of the theory, it is hard to understand what it is.

W2. There are several typos/ presentation problems:

“by evaluate it” in 1.1

“Langevine”

“with Proposition ??."

“In Figure 2, ,”

“as the size of the latent stat is reduce.” 

“using and oscillatory model.”

Vaswani et al is missing the year.

Muller et al  is missing the year.

Figures on model comparison have fonts that are way too small to read. Eg fig 4.

### Questions
I'm writing questions that emerge while I was reading the paper, and prevented me from understanding the framework.

What is the manifold M in the neuroscience or AI context? is it the "neural manifold" corresponding to the attractor of the RNN seen as a dynamical system? 

What is g? it's defined as a function over the manifold: is it going to be the wave over the neural manifold?

How does the Liouville operator depend on phi_t? *Why* can the evolution of g be described linearly?

Why is the center term in Eq (2) corresponding to memory? Because it is "integrating" information from 0 to t? If that is the case, I could think of other ways of integrating information from 0 to t: why is this version interesting? What is the point of the projection operators P and Q there, what do they bring?

Where does the decomposition into resolved and unresolved come from? If g will correspond to the wave, then what does the two components correspond to? Is there a link between identifiability of the dynamics?

What is the “latent space” of a neural network: is it what is sometimes called the “neural state space”, ie R^n for n neurons in a RNN? Or is it a lower-dimensional latent space within the neural state space?

What does it mean, intuitively and for neuro/AI applications, that a process is in equilibrium with its reverse process? What ANNs or biological neural network verify or don't verify this?

Why is F(s) the noise? Where does it come from? Does noise only appear in Eq (4) and if so: why wasn't it present in the previous equations?

What is the significance of the theoretical results in Section 3? Eg, in layman terms, what do they mean for neuro and AI models?

What is a "lifting" operator. What is lifted and to where? Does "lift" mean "transport" as defined by the map in Assumption 3.1?

Are the assumptions made to derive the theoretical framework restrictive, or not? What do they mean for the neuro / AI models?

What is the measure mut associated with the projection operator Pt ("their" measure)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors present an extensive theoretical analysis of 'Coherent Structures' (such as traveling waves) as memory mechanisms in neural network architectures using the Mori-Zwanzig formalism. This approach makes precise the memory benefits of prior wave-based recurrent neural network models, while simultaneously allowing for generalization of such models to enable more flexible memory and dynamics. The key contribution lies in the use a time-dependent projection operator which can be seen as a moving reference frame in the neural state, and critically tying this to the latent drift. The authors validate this generalized understanding through the development of the Neural Wave Field model, demonstrating improved expressivity and sequence modeling performance.

### Strengths
- The work provides a strong theoretical backing to one of the most well-reported computational roles of traveling waves in neural systems, namely their role in memory.  In the words of the authors, this is a potential 'unifying theoretical framework to explain the flow of information in neural systems'. 
- The introduction and related work are very thorough. 
- The presentation of the theory and background are additionally complete.
- The empirical results are strong and support the theory, with the Neural Wave Field architecture outperforming relevant baselines particularly at the small hidden state regime where the theory is applicable. The model also is able to perform selective-copy tasks, relevant to much of the recent State Space Modeling literature. 
- The model can be seen as a formalized generalization of many existing 'wave based' memory models which exist to date. 
- Figure 1 is a much appreciated visualization for interpreting the theory.

### Weaknesses
- The Neural Wave field architecture in Section 3.3 is difficult to understand due to the reliance on notion from the prior sections. If it could be described in a self-contained manner and related to the experiments in which it is used, that would improve readability. 
- A number of citations are given only as (xxx et al.) and not the full citation (e.g. line 061, line 073, line 371).

*Typos*
- Line 152: Equation 5 (should be equation 1)
- Line 329: "We find that the derived GFDR provides enhances the robustness of the coherence across"
- Line 337: "Proposition ??"
- Line 375: "size of the latent stat is reduce"

### Questions
None.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
Authors derive and study a generalized Langevin equation, using which they introduce an oscillatory neural network model. They motivate this model through traveling waves and oscillatory dynamics present in biological neural networks. Authors conclude by showing that the model performs well on a several benchmarks.

### Strengths
- The theory, once presented well, looks like a fresh look into an important problem (traveling waves/spatiotemporal dynamics). 

- Authors honestly report when their algorithm falls behind others, e.g., Table 1. This is certainly a commendable approach, and in my opinion, a positive for the current manuscript.

### Weaknesses
In my opinion, the major weakness of this work is its presentation. I was not able to follow most of the arguments, and many central topics remained undefined. In many cases, heavy assumptions were made on what the reader knows about this topic and in that sense, the paper feels like it was written for a specialized audience. In contrast, ICLR is an ML conference that caters to a broad audience with general interests. This main weakness constitutes my current recommendation.

### Questions
As I was reading this work, despite being a computational neuroscientist with a theoretical physics background, I was not able to follow the line of arguments. Please find below some suggestions/comments/questions:

- What do "trivial" vs "emergent" dynamics mean? Similarly, what is a neural wave field and what is a coherent memory structure? I kindly recommend minimizing the use of jargon in abstract.

- In my opinion, words are sometimes used without precision and outside of their semantic meanings, which makes reading the manuscript even harder. For instance, "manifold trajectories" likely refers to "neural activity traces confined within low-dimensional manifolds," as manifold is a K-dimensional object, not a 1D trajectory. In that sense, several rounds of proof-reading focusing on this particular aspect may be beneficial.

- There are several cursory/blanket statements such as "many artificial neural networks–from gated recurrent units to recent state-space-models remain black-box mechanisms." Most computational neuroscientists, including myself, would strongly disagree with this statement. We understand a great deal about RNNs, and nowadays can exactly extract the flow maps they learn to implement neural algorithms, or even predict their learning trajectories analytically (see recent works on low-rank RNNs). Such blanket statements, apart from being wrong, do not contribute much to the manuscript. I recommend removing these statements or reworking them into a more accurate format.

- Statement "Attractor dynamics and traveling waves are typically modeled in isolation;" is incorrect. Several work in biophysics focusing on reaction-diffusion systems studied attractor dynamics in the absence of diffusion to study the properties of traveling waves. This goes all the way back to Turing's seminal paper on Morphogenesis.

- "RNNs suffer from inherent information bottlenecks" Sussillo et al. 2013 has nothing to do with this statement. Neither does Rajan et al 2016. This does create the feeling, even if not true, that other references may also simply be not true/relevant either.

- A comment on section 3 as a whole: It is too dense. I am not able to understand, appreciate, or even follow the reasons for the statements. No prior motivation is given in many cases. For instance, why does the reader care about Corollary 3.1? It seems to me that the amount of theoretical information that wants to be conveyed can better be achieved in a specialized journal. 

- And a final comment on section 4: Most experiments are anecdotal, and figures too small to see. Error bars missing, or not reportable (since only one network is shown). More rigor is desirable here in terms of both reporting and experimentation. 

My overall assessment is as follows: While I was able to intuit a potential broader appeal for the work, it seems underdeveloped/unfinished in terms of presentation and experimental rigor. As it stands, imho, the current manuscript is not a good fit for consumption by general audience and may be better submitted to a specialized journal. A lot more care needs to go into writing before it can be reviewed in the context of a top ML conference. Hence, I reluctantly recommend rejection.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors presents a theoretical framework for modeling the latent dynamics of a neural network during sequence learning. A generalized Langevin equation is derived via time-dependent projections, and based on this, the neural wave architecture is introduced, having Mori-Zwanzig dynamics. This model demonstrates robust long-range recall, minimal memory dimension, and interpretable latent modes across synthetic and neuroscience datasets.  In particular, the work addresses a timely and important direction, given the growing interest in dynamical representation and the need for more interpretable machine learning models.

### Strengths
The derivations are clear, and the problem is well motivated.

Mathematically, the difference between the derived GLE with the FFE-MZ is well explained

The long-range copy and selective copy tasks show improved results, especially when compared to other SSM architectures (such as Mamba) and transformers.  

The paper is well written and discusses limitations and potential future extensions of the work.

### Weaknesses
When compared to other frameworks implementing MZ dynamics, it is not shown whether there is a practical performance benefit or how this architecture exhibits better interpretability.  

While theoretically this framework has a lot of potential, the authors have not demonstrated in a clear way how this model contrasts to architectures having a state-dependent kernel other than requiring a marginally smaller latent state. 

Chaotic time-series forecasting is mentioned in the abstract, but it is not shown in the main paper.

### Questions
Is there a clear case where the MZ dynamics provide a practical performance benefit?

Similarly, is there a clear case where the MZ dynamics provides better interpretability?

### Soundness
3

### Presentation
3

### Contribution
3
