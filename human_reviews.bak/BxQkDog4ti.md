# Range, not Independence, Drives Modularity in Biologically Inspired Representations

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Why do biological and artificial neurons sometimes modularise, each encoding a single meaningful variable, and sometimes entangle their representation of many variables? In this work, we develop a theory of when biologically inspired networks---those that are nonnegative and energy efficient---modularise their representation of source variables (sources). We derive necessary and sufficient conditions on a sample of sources that determine whether the neurons in an optimal biologically-inspired linear autoencoder modularise. Our theory applies to any dataset, extending far beyond the case of statistical independence studied in previous work. Rather we show that sources modularise if their support is ``sufficiently spread''. From this theory, we extract and validate predictions in a variety of empirical studies on how data distribution affects modularisation in nonlinear feedforward and recurrent neural networks trained on supervised and unsupervised tasks. Furthermore, we apply these ideas to neuroscience data, showing that range independence can be used to understand the mixing or modularising of spatial and reward information in entorhinal recordings in seemingly conflicting experiments. Further, we use these results to suggest alternate origins of mixed-selectivity, beyond the predominant theory of flexible nonlinear classification. In sum, our theory prescribes precise conditions on when neural activities modularise, providing tools for inducing and elucidating modular representations in brains and machines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper seeks to explain why and when a population of biological or artificial neurons sometimes modularise and sometimes entangle the representation of source variables. This is a fundamental question that is highly relevant to both neuroscience and AI. The authors propose and prove a new theory emphasising the importance of the shape of empirical data distribution in extreme regions in dictating whether neurons are mixed selective or modular. Specifically if the sources to be represented are supported in all extreme regions, the neurons modularise. The application of this theory outside of linear autoencoders is tested in feed forward and recurrent neural networks, including experiments that provide explanations for discrepancies in previous neuroscience literature.

### Strengths
This work is original, of high quality and undoubtedly contributes to the community’s understanding of neural modularisation. The nonlinear verification of theory and additional application to neuroscience results are significant for the field and a strength of the paper. The submission is well written and clear throughout, although its clarity suffers somewhat due to the amount this submission seeks to cover.

### Weaknesses
- In my opinion this submission contains too much, and would benefit from more focus and time spent on fewer experiments. The appendix is already large but some experiments could be moved there.  
- Figure text and panels are too small throughout.
- It is not clear how relevant encoding of the extreme points of source distributions are for computation / cognition. I.e. neurons do not just autoencode.
- The bio description of energy minimisation assumes l2 penalty is an appropriate penalisation function for modelling biology. This is a fair starting assumption, but no argument is presented about how this maps to the costs biological neurons will be seeking to minimise.

### Questions
- Biologically inspired linear rnns are repeatedly mentioned. Linear rnns are less like biological circuits (which are nonlinear).  Is the biologically inspired term referring to the energy costs?
150: satisfied for all w is a little unclear. The proof could  
- Autoencoding seems limited as an objective for theory, in that brains perform computations over inputs and states and produce behaviour. Is it not the relevance of representing the extreme points for behaviour that is important? 
- Section 2.2 assumes positive weights?
- 120 "be better" is imprecise 
- Fig 1d, neuron’s angle isn’t described? 
- I don’t understand the what where regression justification. 
- How relevant are these results for more typical ANN experiments?  E.g classic image benchmarks or language modelling tasks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the conditions under which modularity should arise in optimal representations. The authors developed a mathematical theory of when a modular representation should be favored in a linear autoencoder. They found that modularity should appear when the support of the sources is “sufficiently spread”. The paper also presented simulation results to show that some of theoretical results may be generalizable to non-linear problems. The later sections of the paper applied these theoretical ideas to explain some experimental observations from neurophysiological experiments in cognitive tasks. 
The paper makes several interesting points about when modularity should be favored, and applies the theory to several examples.

### Strengths
Originality: The theoretical part of the work builds up prior work by Whittington et al, 2023 and other studies. Previous work by Whittington et al, 2023 assumed mutual independence of the sources. In the current work, the authors show that, with several additional assumptions, “sufficient spread” of the factors of variation can also lead to modular representation. The theory has some new elements, although it is a bit incremental. The application to the several neuroscience problems seems to be new.

Quality: The paper considered both linear and nonlinear cases. This is a strength. For the former, analytical results were provided. For the latter, some preliminary numerical results were given. 
The paper also considered several neuroscience applications. This may also be seen as a strength.


Clarity: The overall structure of the paper is clear. Some intuitions behind the theory were provided.


Significance: The question of when modularity arises in optimal representation is an interesting one and we still lack a clear understanding. This work made a few interesting points on this problem.

### Weaknesses
The writing needs improvements throughout the paper. In particular, the description of the theory can be substantially improved. For example, Theorem 2.1 should be made more accessible. 


While several applications are attempted, each application appears to be preliminary. If the model predictions and experimental tests can be made more rigorous, that would strengthen the paper.

In Section 5, there are some qualitative differences between the model predictions and the data. As the paper pointed out, Panichello and Buschman (2021) showed that a substantial fraction of the neurons were tuned to both colors, contradicting a key prediction of the model. This seems to be a more important feature of the data compared to the issue of orthogonality v.s. non-orthogonality.

Looking at the math, the theory appears to only work for scalar variables. Can it be applied to circular variables? If the answer is no, the applications to real data would be questionable. In section 5, color is sampled from a color-wheel. 

Based on the way the theory is written, the results seem to rely the assumption that the energy (or cost) is a quadratic function of neural activity. If the cost scales linearly with neural activity, would the theoretical results change fundamentally? Assuming a linear scaling could make sense biologically, as the metabolic cost may scale linearly with the number of spikes. 

Relevant earlier theoretical literature on grid cell modularity was not cited/discussed (e.g., by Fiete/Burak et al, and Wei/Prentice/Balasubramanian).


It is difficult to understand what is really going on in Fig. 1. Fig.1 may come out too early and the results need to be better unpacked. 


The theory seems to ignore biologically-relevant noise.

### Questions
What are the assumptions about the noise in the system being studied? How does noise affect the theoretical results?

In section 5, the authors seems to be equating orthogonality with modularity. Am I understanding this correctly? If this is the case, can the authors unpack the idea?

Can they authors unpack the results in Fig. 5c? Was that an actual simulation or just a schematic?

The definition of the matrix in Eq. 10 is unclear. Please clarify. 


Can the authors explain what the first part of the title mean?

In the abstract, it is stated that “From this theory, we extract and validate predictions…” What does “extract” predictions mean? [minor point]

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
This paper investigates why neural representations in biologically inspired networks sometimes form modular structures, where each neuron encodes a single variable, and other times create mixed-selective representations. The authors develop a theory that predicts when modularization will occur in networks optimized for energy efficiency with nonnegative firing rates. They derive necessary and sufficient conditions for modularity based on the spread of source variables. The theory is validated in both linear and nonlinear networks. The theory provides a cohesive explanation for the conflicting findings in the prefrontal cortex and entorhinal cortex data from neuroscience studies.

### Strengths
1. The paper introduces a novel theory that precisely predicts necessary and sufficient conditions for modular representations in biologically inspired networks, extending previous work beyond statistical dependencies.

2. The mathematical formulation is rigorously derived, and validated across various neural network architectures and experiments.

3. The theory provides explanations for conflicting neuroscience findings and has close links to biologically plausible architectures and brain representations.

4. The paper provides a cohesive theory for understanding modularity in neural representations, with implications for both interpreting biological neural data and guiding the design of artificial neural networks for better interpretability and efficiency.

5. The paper is well-written, presenting complex theoretical concepts with clarity and intuition.

### Weaknesses
1. The experiments use nonnegative activities in neural networks, which aligns with biological plausibility, but it would be valuable to discuss inhibitory neurons in the brain and how inhibition might relate to the theory and findings.

2. While the L2 norm of firing rates and weights is a reasonable approximation for biological energy, other biological constraints (e.g., sparse connectivity, synaptic range, anatomical structure, and decoding flexibility) may also play a role.

3. It’s unclear how the theory would extend to more complex datasets. For example, what would the different conditions/variations in source variables mean in naturalistic data (e.g., natural images, audio, text)? How might we approximate "spread", and quantify modularity conditions in such stimuli?

4. The discrepancies in prefrontal working memory modeling and the brain data could have further explanations, particularly why some neurons tune to both colors despite orthogonal encoding and why exact subspace angles were not obtained.

### Questions
1. Beyond prefrontal working memory and the entorhinal cortex, does the theory generalize to other modular representations in the brain?

2. If decoding flexibility were considered as a biological constraint, how might it impact the theory and its predictions?

### Soundness
3

### Presentation
4

### Contribution
3
