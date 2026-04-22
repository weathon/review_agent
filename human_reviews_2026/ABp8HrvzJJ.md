# BioRNN: Bio-Inspired Synergistic Integration of Neuromodulation and Wave Propagation in Recurrent Networks

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Training recurrent networks that directly implement physical wave equations has been hindered by numerical instability and incompatibility with gradient-based optimization. We introduce BioRNN, a recurrent architecture that embeds two-dimensional wave propagation dynamics on a neural grid and achieves stable training via a mixed finite-difference scheme with learnable damping. Inspired by neuromodulation in biological systems, BioRNN incorporates a lightweight frequency-modulation stage that transforms inputs into oscillatory patterns, enabling the recurrent layer to exploit resonance and frequency selectivity. This combination allows BioRNN to model spatiotemporal dependencies through constructive interference while retaining theoretical guarantees of stability during backpropagation. On sequential visual (sMNIST, noisy CIFAR-10) and auditory (ESC-50) benchmarks, BioRNN achieves competitive performance across domains, with pronounced gains on frequency-rich auditory tasks and comparable accuracy on vision. This work demonstrates that integrating biologically inspired neuromodulation with physically grounded wave dynamics yields recurrent models that are both biologically grounded and reliably trainable within modern deep learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces an input-modulated wave-based RNN architecture that leverages oscillatory resonance to process sequences. The authors provide theory demonstrating that this model has robust spectral properties, and further demonstrate that it has decently strong performance on long sequence modeling tasks.

### Strengths
- The paper makes an interesting connection between input modulation and wave dynamics, relating this to neuromodulation and spatiotemporal dynamics in biological neural networks. 
- The topic is very relevant for the NeuroAI community where modulation effects have yet to be abstracted to a computationally useful level, especially in combination with wave dynamics. 
- The plots studying performance vs. model parameters are welcome and interesting. 
- The ablation study in Table 3 is helpful to understand the impact of the different model components, and highlights the difference between visual and audio processing.

### Weaknesses
-  The discussion of the input modulation in Section 2.1 is too short and may lead to confusion. It would be helpful if some examples of modulators could be included earlier. 
- The main results in Table 2 have no error bars or standard deviation. While I know this is standard for the field, this makes the minor differences between models mean relatively little in practice. 
- The authors claim "In summary, the modulator acts as a task-adaptive transformation that aligns well with BioRNN’s resonant dynamics, enabling competitive vision performance and clear advantages on spectrally rich audio tasks." However, the high performance of the BioRNN on the ESC50 task (where it performs the best) appears to not be due to the modulation (modulation only helps slightly). 
- Similarly to the above, the main contribution of this paper appears to be the input modulation, however, this appears to have only a minor impact on model performance. If the authors could clearly enumerate their contributions that would help greatly. 
- The paper accentuates the importance of input modulation and resonant dynamics, but the theory section does not focus on this and instead focuses on stability. The empirical results are then only minimally supportive of the benefits of these dynamics.

### Questions
- In equation three you say: for efficient computation we re-parameterize o'=co. How does this enable efficient computation? And efficiency in what sense?
- Why does the modulation help so significantly on sequential MNIST, but appears to only have a minor effect on other tasks? 
- Is there any more concrete or analytic way that one can interpret the interaction of the input modulation and the recurrent dynamics? 
- In the ablations, how do you disable wave propagation but retain spatial coupling?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces BioRNN, an RNN that first modulates input sequences with time-dependent M(t), before passing the input to a 2D neural sheet that acts as the dynamical transition function of the RNN.

The authors go on to prove the stability properties of BioRNN - a property missing in existing approaches that incorporated physical waves into their dynamics.

### Strengths
Good empirical benchmarking against relevant baselines and similar approaches, with solid ablations to ensure value of modulation module (and other design choices). The paper provides rigourous stability guarantees, which are again missing in previous works (according to the authors - this is not my area of expertise)

### Weaknesses
Minor: citations should be in brackets, i.e. using \citep

Notation is confusing, given that IIUC x and y are used as both the input and the coordinates of a 2D input

It's also not immediately obvious how the 2D neural sheet relates to the transition function S - making this link more clear in the main text would help motivate the in-depth analysis below

There is not much done in the way of interpretability of the wave properties. Given the biological motivation of these design features, some comparison to the role of waves in biological neural networks would have been warranted.

There's no information about training time, memory requirements, or inference speed, which makes it difficult to assess practical trade-offs for the more sophisticated architecture.

### Questions
Why were these specific forms of modulation used for each dataset? Did you try alternative forms/using modulation from different datasets?

Furthermore, the modulation functions are manually designed for each dataset. This limits practical applicability - how would one design modulation for new tasks?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces BioRNN, a recurrent model that combines a lightweight input “modulator” with a 2D grid of units designed to mimic wave-like propagation and damping. The modulator transforms inputs into oscillatory patterns that the grid can process, aiming for an interpretable, physically inspired alternative to standard RNNs. Experiments on sequential MNIST, noisy CIFAR-10, and ESC-50 report competitive results. A stability-aware update scheme is proposed to improve reliability at larger time steps, and diagnostics relate performance to key hyperparameters. While promising—especially for audio—the approach is not consistently stronger than GRU baselines, and more controlled comparisons are needed to separate the impact of the modulator from the recurrent dynamics.

### Strengths
- The idea to explicitly separate neuromodulatory preprocessing from physically inspired wave dynamics is well-argued and visually communicated. 
- The p/o split and use of divergence/gradient operators make the mechanism interpretable as storage vs. transport.
- The mixed forward/backward finite differences plus implicit damping are a effective contribution
- Further investigations give transparency about which hyperparameters actually matter task-specifically.
- Nice ablations show that wave propagation, spatial coupling, and the auxiliary field each add measurable value.

Personal note: I really like this approach as an unusual way to process time series data, using a modulator plus wave-like recurrent dynamics to spark rich transients, and I see real potential here for reservoir computing, even if the core is learned rather than fixed.

### Weaknesses
- The paper does not convincingly demonstrate competitiveness. On ESC-50, BioRNN (mod.) is noticeably below a simple GRU. On nsCIFAR-10, BioRNN (54.2%) is far below coRNN. On sMNIST (not a good selection as a core benchmark anymore - why not at least permuted?) 98.1% cannot really be seen as competitive, because this is an accuracy that can be achieved easily with much simpler methods. In a nutshell, the approach’s absolute effectiveness even over standard gated RNNs is not yet demonstrated. A comparison against actual state-of-the-art recurrent sequence learning models (such as several state space model-like networks) is entirely missing.

- The modulator is crucial for BioRNN (e.g. for sMNIST is causes a huge jumps), but LSTM/GRU are not evaluated with the same modulated inputs. Without that control, it’s hard to ascribe gains to the recurrent dynamics vs. the input transformation.

- There is no information (plots) about the convergence behavior during training. Is there any advantage? Resonator-based neurons (damped harmonic oscillators), for instance, are known to converge much faster than usual RNN structures.

### Questions
- What did the modulator learn? Can you visualize the learned f, ϕ, α (for audio) and their distribution over Mel bins/time? Do they concentrate on critical bands, or track class-discriminative frequencies? 

- Which boundary conditions are used on the grid and how sensitive are results to that choice? Just 0?

- How well does BioRNN perform in comparison with strong recent sequence learning models (cf. weaknesses)?

As mentioned above, I appreciate the idea and find it genuinely interesting, but I cannot recommend acceptance in its current form, as the paper lacks sufficient evidence to support its potential. Instead, I have two suggestions:
- Explore other problem domains where the system’s inductive bias may better match the task.
- Consider developing this approach further in the direction of reservoir computing, where it could be a natural fit.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work presents a recurrent neural network model that implements a physical wave equation in its recurrent dynamics, with two innovative features. First, it has an time-dependent input modulation scheme that adds oscillatory patterns to the inputs. Second, it uses a discretization scheme accompanied by theoretical stability guarantees. It finds that these features enhance model performance by experimentation and, in the case of the discretization scheme, mathematical analysis.

### Strengths
The text was well-written, and the presentation was elegant and easy to follow.

The input modulation and discretization schemes appear innovative.

The motivation for combining short-term (waves) and long-term (neuromodulator) dynamics in RNNs is interesting.

### Weaknesses
1. The biological motivation for BioRNN is weak. It is not at all clear why a physical wave equation should make for a more suitable recurrence in a biologically inspired model, in comparison to the oscillator networks mentioned in the paper. As for the time-dependent input modulation scheme, the introduction makes a vague link to neurotransmitters, but no compelling link. Appendix A.2 alludes to “neuroscience findings that emphasize the role of input-driven oscillations in shaping cortical activity,” but gives no citation. Therefore, the biological motivation for this network is unpersuasive.

2. A major claim of the paper is that BioRNN resolves a widespread difficulty of training RNNs based on physical wave equations: “By embedding a mixed finite-difference scheme with learnable damping, BioRNN resolves the long-standing instability and incompatibility of physical wave equations with gradient-based training.” However, there is no experimental comparison between the chosen mixed finite-difference discretization scheme and other discretization schemes in BioRNN, nor is there any citation to support the claim of such a longstanding difficulty in the literature. Consequently, we do not know whether this really is a longstanding problem, or whether this choice of discretization scheme really eases training and performance in practice for BioRNN in practice. The results of applying this discretization scheme to other wave-equation RNNs could also be reported.

4. The paper claims the time-dependent input modulation scheme as a major innovation. The comparisons to oscillator network RNN models might be an informative supplemental analysis, but why not compare BioRNN to other wave-equation RNNs, implemented with and without the input modulation? Ideally, one can show that this input modulation scheme improves performance for those models as well. If the paper claims the time-dependent input modulation operator as a key innovation, why not report the results of selectively ablating it, instead of jointly ablating the auxiliary field and the input modulation? 

5. The task performance results are not very strong.

### Questions
What are the citations to support the claim of the "long-standing instability and incompatibility of physical wave equations with gradient-based training"?

It would be helpful to clarify in more detail the link between the input modulation and neurotransmitters.

### Soundness
2

### Presentation
3

### Contribution
2
