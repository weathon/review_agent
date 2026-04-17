# ProTDyn: A Foundation Protein Language Model for Thermodynamics and Dynamics Generation

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
Molecular dynamics (MD) simulation has long been the principal computational tool for exploring protein conformational landscapes, but its application is limited by high computational cost. We present ProTDyn, a foundation protein language model that unifies conformational ensemble generation and multi-timescale dynamics modeling within a single framework. Unlike prior approaches that treat these tasks separately, ProTDyn allows flexible i.i.d ensemble sampling and dynamic trajectory simulation. Across diverse protein systems, ProTDyn yields thermodynamically consistent ensembles, faithfully reproduces dynamical properties over multiple timescales, and generalizes to proteins beyond its training data—offering a scalable and efficient alternative to conventional MD simulations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The work proposes ProTDyn, a model that is able to do conformation ensemble generation and multi-timescale dynamics modeling within a single model.

### Strengths
The paper is well-written and easy to follow. The fact that the authors present a model that is able to perform simultaneously equilibrium conformation ensemble generation and multi-timescale dynamic trajectory generation is interesting and as far as I am aware novel.

### Weaknesses
As the paper is not near my main area of research, it is hard for me to make questions or comment on the weaknesses although I found the paper easy to follow and appreciated especially the explanation on page 4. 

The paper is a rather standard application of ML modelling techniques but unfortunately I cannot comment on the strength of the results. For this reason I will leave a weaker confidence score.

### Questions
I'm slightly confused by Figure 1 as the authors mention it is a language model, but there does not seem to be language/text as input in the diagram. Could the authors please clarify? Are the authors using "language model" to highlight the fact that it is an autoregressive (causal/decoder) Transformer architecture and not that the input is language.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors present ProTDyn which is a transformer operating on ESM3 structure tokens trained for conformation and dynamics sampling. The model is trained on a mixture of the alphafold synthetic database of structures and molecular dynamic datasets. It is trained for both equilibrium distribution sampling as well as jumping ahead in MD simulations or inpainting in between coarse time steps in MD simulations. The model improves over BioEmu, a recent deep learning model for conformer generation, in terms of distributional match to MD.

### Strengths
The training of the model is simple in that all 3 tasks of thermodynamics, dynamics and dynamics inpainting are trained jointly and simultaneously in a single model. This avoids the need for any task specific fine-tuning which is a downside of prior approaches. All tasks are unified in terms of autoregressive modelling with different factorizations.

The experimental results look strong in terms of improvement over the BioEmu baseline model. Distibutional fit on MD test data looks quite convincingly better in Table 1 and in Figure 2.

The authors make some interesting observations in terms of the time resolution with which to sample their dynamics model. They find that sampling with a finer grid can lead to worse overall performance potentially due to error accumulation. This finding will be useful for further research into deep learning based dynamics models.

### Weaknesses
A main weakness of the paper is the lack of ablations with respect to the stated contribution of ProTDyn being a unified model for Thermodynamics and Dynamics generation. On L56-L58 in the introduction the authors describe how joint training can be mutually beneficial for both tasks. I would therefore expect experiments and ablations showing that this is indeed the case. You could train a version of your model on the tasks individually and compare performance with the multi-task trained model. I don't think comparing to BioEmu is enough to make this claim since BioEmu is pretrained on Thermodynamics style tasks and fine-tuned on dynamics tasks which can be seen as a form of multitask training.

With regards to Table 2, where these proteins are in your test set but BioEmu's training set, I am unsure how ProTDyn ended up with a different test set to BioEmu that made this comparison difficult. If you are using the same datasets as BioEmu, why not also use the same splits? In the end, it makes it very hard to draw any conclusions from Table 2 since the models are not comparable.

The paper is quite unclear with the usage of the term 'module'. The authors state that they have a 'thermodynamics module' and a 'dynamics module'. This would imply separate parts of the network that are being trained for these two tasks which would be in contradiction with the introduction describing a single unified model which is a key stated contribution of the work. I am unsure what the authors are referring to with regards to these modules and this should be made clearer. 

Further, in terms of clarity, on L149 the authors describe their model as a 'multimodal protein language model'. In what way is this a multimodal model because as far as I am aware the model just generates structural tokens (and no other modalities).


I am tending to accept this paper due to the strong experimental results with regards to BioEmu and simple training scheme. This model seems to make a significant improvement on the state of the art for conformational modelling.

### Questions
Why did you decide to use a causal autoregressive distribution for the generation of structural tokens? In ESM3, an any-order autoregressive model was used to generate tokens in parallel which seems to make sense for protein data. Did you try any other parameterizations?

How were the proteins in Figure 2 selected? If these were cherry-picked as to best show the benefits of your method this should be stated clearly. Or if these are exhaustive of the CATH2 dataset, this should also be made clear because this is important for judging the significance of your findings.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces ProTDyn, a foundation PLM to unify equilibrium distribution sampling, time-dependent dynamics sampling as well as inpainting for fine-grained timesteps based on coarse grained timesteps. It has been trained on a large scale molecular dynamics dataset, and demonstrated promising performance in the generated sample distribution in comparison with baseline models. I find this methodology very interesting and novel, and has shown scaling up in the model and data.

### Strengths
**Novel methodology**
- The method is novel to incorporate three tasks in one model:
  1. i.i.d. equilibrium distribution sampling
  2. time lagged predictions with multiple time scales
  3. inpainting: given two coarse grained time steps, predict fine grained steps in between
- This methodology overcomes the lack of capability in the description of dynamics/kinetics in the equilibrium sampling models

**Transferability**
- The model is trained on multiple time scales, showing transferability in the time dependence
- The model is trained on a large scale of MD data, showing transferability in the chemical space

**Performance**
- The performance has been compared on MD sample distributions with proper baselines and benchmarks, and has demonstrated the advantage of this model

### Weaknesses
- **Generalization**: For the dynamics tasks, the test proteins were seen by the model during thermodynamics training, which may not represent true generalization. A stronger evaluation would involve holding out proteins based on sequence similarity
- **Baseline**: This model is only compared against BioEmu as the only baseline. It can consider adding more baseline models, such as Alphaflow which is also trained on mdCATH, as well as a few other works that baked in the time dependence for dynamics generation of protein models.
- **Inpainting task**: Dynamics inpainting is highlighted as a key capability, but its performance is only measured indirectly through the quality of the final, end-to-end generated trajectories. There are no specific benchmarks that isolate and quantify the accuracy of the inpainting process itself

### Questions
- Could you clarify the train/test splitting procedure? Was any filtering based on sequence identity or structural similarity
- Have the authors considered other architectures such as those AF-based models? What was the specific rationale for choosing the ESM3 architecture?
- BioEmu has been finetuned on experimental data after training on MD, so the comparison on MD distribution might not be the most direct comparison, which can be noted.
- The method aims to generate coarse-grained dynamics. It'll be helpful to demonstrate the computational cost in comparison with MD simulation
- Have the authors considered the asymptotic behavior of the time dependence? e.g. when the $\delta t \to \inf$, it should approach an i.i.d. distribution

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a protein foundation model based on ESM3 that unifies equilibrium distribution modelling and dynamics generation for protein structures. Training of the model optimizes the loss for distribution modeling, dynamic trajectory prediction, and dynamic trajectory inpainting at the same time. To enable the modeling of dynamics, ProTDyn proposes two-layer rotary embedding scheme to combine time rotary embedding. Experiment results show that ProTDyn performs than baselines in metrics including distributional similarity. ProTDyn also shows reasonable performance in capturing state transitions consistent with molecular dynamics.

### Strengths
1.The paper proposes an interesting view of enabling one model for both equilibrium distribution sampling and temporal molecular dynamic modeling.

2.The experiment results prove the effectiveness of the method in distribution modeling compared with strong baselines.

3.The paper is clearly written with the mothod being simple yet elegant.

### Weaknesses
1.The benefit of unifying thermodynamics and dynamics generation is not explicitly discussed in this paper. It would be interesting to see how the two tasks interfered with each other in terms of performance. This would also provide stronger support for the motivation of the unification.

2.Some important details of the method are missing. For example, how the “two-layer rotary embedding scheme” is designed to combine temporal and residue positions. Given the model architecture leverages the ESM3 backbone, such details are important for the readers to understand the key adaptation to enabling the dynamics generation.

3.The evaluation metric for the distributional similarity are along low dimensional collective variables. Specifically, distribution along the Rg and RMSD (which are single values not reflecting details of the protein structures) w.r.t. the native structure seems to be less significant in reflecting the true distribution in 3D Euclidean space with all the residues.

### Questions
1.Before ProTDyn, the paper “Two for One: Diffusion Models and Force Fields for Coarse-Grained Molecular Dynamics” also discusses how the connections between score-based generative models and force fields can be leverage to train a diffusion model on MD simulations, and the trained score function can be used to simulate MD trajectories. It would be valuable to clarify the difference and advantage of ProTDyn when compared with these methods.

2.It would be interesting to see how ProTDyn works on larger proteins. Particularly, it would both enhance the presentation of the paper and provide more evidence of the generalizability of ProTDyn if the predicted structure/distribution can be demonstrated with figures.

### Soundness
3

### Presentation
2

### Contribution
2
