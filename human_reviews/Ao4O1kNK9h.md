# Scaling Properties For Artificial Neural Network Models of the $\textit{C. elegans}$ Nervous System

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5

## Abstract
The nematode worm $\textit{C. elegans}$ enables straightforward optical measurement of neural activity, presenting a unique platform for exploring intrinsic neural dynamics. This paper investigates the scaling properties essential for self-supervised neural activity prediction based on past neural data, omitting behavioral aspects. Specifically, we investigate how predictive accuracy, quantified by the mean squared error (MSE), scales with the amount of training data, considering variables such as the number of neurons recorded, recording duration, and diversity of datasets. We also examine the relationship between these scaling properties and various parameters of artificial neural network models (ANNs), including size, architecture, and hyperparameters. Employing the nervous system of $\textit{C. elegans}$ as an experimental platform, we elucidate the critical influence of data volume and model complexity in self-supervised neural prediction, demonstrating a logarithmic decrease in the MSE with an increase in the amount of training data, consistent across diverse datasets. Additionally, we observe nonlinear changes in MSE as the size of the ANN model varies. These findings emphasize the need for enhanced high-throughput tools for extended imaging of entire mesoscale nervous systems to acquire sufficient data for developing highly accurate ANN models of neural dynamics, with significant implications for systems neuroscience and biologically-inspired AI.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors collect and preprocess various C. elegans datasets, then test three DL models on predicting "next neural state". The goal is to assess the relationship between volume of training data, size of model, and accuracy of predictions.

### Strengths
The program is very methodical (eg how datasets are processed, models are tested, etc).

The write-up is clear and well-done.

### Weaknesses
The motivation for using C. elegans datasets, and for the overall program, was not sufficiently convincing to me. The findings seem to repeat what has been found before, so I don't know what new finding to take away.

Some of the preprocessing is counter-intuitive: Ignoring behavior (when this is known to heavily drive worm neural dynamics) and considering only the slowest-scale dynamics (through LP filtering).

Reviewer limitation: Due to item 1 above, I am perhaps not an optimal person to assess this paper. I will certainly defer to other reviewers.

Notes to ICLR: 

1. Please include line numbers in the template. They make reviewing much easier! 

2. Please reformat the default bibliography style to make searching the bib easier! eg numbered, last name first, initials only except for last name.

### Questions
General review context: Please note that I am simply another researcher, with some overlap of expertise with the content of the paper. In some cases my comments may reflect points that I believe are incorrect or incomplete. In most cases, my comments reflect spots where an average well-intentioned reader might stumble for various reasons, reducing the potential impact of the paper. The issue may be the text, or my finite understanding, or a combination. These comments point to opportunities to clarify and smooth the text to better convey the intended story. I urge the authors to decide how or whether to address these comments. I regret that the tone can come out negative even for a paper I admire; it's a time-saving mechanism for which I apologize.

1.1, "causal manipulability, increased analytical accessibility": These are not qualities I associate with ANNs. By "increased" do you mean "increasing", as in methods are being developed to probe ANNs so that they are not such black-boxes as they have been in the past?

1.1, "system that aligns closely with ANNs on these aspects": This does not sound right to me. C. elegans has a very small network,with inhibition, recurrent connections, non-spicking dynamics, specialized connectivity, and specialized neuron functions. These properties are all distinct from the ANNs discussed (and most ANNs).

General: Perhaps reformat the bibliography for easier searching, eg numbered, last name first, initials only except for last name.

1.3 paragraph 2 "internal dynamics": Would "emergent" be a more accurate term?

1.3, paragraph 3: I did not find this motivation for the program sufficiently convincing. Also, I expect that ignoring behavior when examining elegans' neural dynamics guarantees huge residuals, since these so heavily affect dynamics (cf eigenworm).

2.1 "these differing conditions were not considered": This seems risky - see comment above.

Preprocessing: "lowest 10%": some questions here 1. what is the rationale for this? It seems to risk throwing out a lot of important faster dynamics. The later choice of baseline as "no change" emphasizes how salient dynamics are removed. Was this done to accommodate the sleeping worm dataset? 2. Are "frequencies" actual worm dynamics, or experiment noise (at frequencies higher than fastest worm dynamics), and do they include high frequency randomness in neural responses that are part of the biological system? 

Preprocessing: "we selected delta(t) = 1 second". Doesn't this lead to upsampling, not downsampling. Eg {0, 1.7, 3.4, 5.1} -> {0, 1, 2, 3, 4, 5}.

Train test split: A k-fold split would give an option of providing +/- std dev in results.

Amount of data: why is a 50:50 train:test split desirable? I'm not familiar with this approach.

Baseline Loss, "MSE": Is there a reason for using MSE for neural data? ie what is an appropriate way to measure accuracy of predictions of neural dynamics? If behavior were in the mix, one could predict the behavior. What is a neurally-salient approach here? (this is admittedly an open-ended question).

3.1.1 Result Fig 3: "as a function of the amount of training data.": I do not see this. The x-axis looks like a measure of training epochs. How does data amount affect this?

3.2 Result: My sense from ML literature is that this is a well-established fact. A question that might extend the finding: how do models trained on 1-behavior sets generalize to n-behavior sets, and vice versa? 

Fig 6: Is a quadratic the correct parametrization? spline, quartic? eg, linear (green) in B is not quadratic.

Discussion: "our models, while ... neural dynamics.": I regret that I am not convinced of this (see above comments re preprocessing).

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper is an empirical study of multi-variate time series prediction of the neuron activity in calcium imaging measurements from *C. elegans* across multiple datasets, different sizes of training, and different models.  Three classes of models are compared fully connected, LSTM, and transformers with sequence embedding. The results indicate that all models benefit from additional data, but the LSTM improves at the fastest rate. Inter-dataset validation performance reveals that using the largest datasets in training improve performance, that all models are able to specialize to one dataset, but smaller dataset do not enable generalization to the large dataset. Given the full data, the dimensionality of the model's hidden layer is assessed with models having an optimal within a range. Finally, examples of the dynamic predictions for the same neuron in different datasets are made where the input to the model is the true data (teacher forcing) or the previous output.

### Strengths
The motivation and opportunities for exploring models of neuronal population dynamics on a model organism is very clearly articulated. The idea is original.

Bringing together the datasets is a useful contribution. 

Some of the ideas tested (scaling and dataset transfer) are thoughtful and would be helpful for the neuroscience community.

### Weaknesses
Self-supervised is a little misleading as it is essential limited to a single-time-step prediction (also called sequence-to-sequence in the paper). Self-supervised learning typically entails using the learned representation for separate downstream tasks too. The statement in Section 1.3 is confusing "not to dismiss the relationship between neural activity and behavior". I don't think the relationship is between the neural activity and behavior is ever tested. While the paper doesn't dismiss it, it doesn't address it.

**Major concern (lack of clarity in formulations regarding causality)**
It is not clear if all the models are trained to make causal predictions (or if they have access to future time points).  The formulation in step 1 on page 4 is not clear if the non-linear projection operates at each time point independently or with a causal memory. Same for the subsequent core. If the linear layer is done for each time point or an embedding. We can assume that the LSTM is going to have memory and can be causal due to its definition, but is the transformer causal with its positional encoding?  Finally, the formulation is not clear if the output layer is done for each time point. Figure 2 doesn't help.  The models are obviously learning something, but without details the reader is left to guess what type of relationships is being learned.

The one-step prediction task should be stated up front in 2.2 "Shared model structure". Only during Section 2.4 "Training" does it describe that it is a one-timestep forward shifted. It doesn't make sense to describe the loss before the task. The notation for the prediction being $\hat{X}(t)$ as a function of the input at previous time $t-1$ should be made explicit in the formulation. 

It is not clear that without specific training for autoregressive prediction the models would perform well at this task. I.e. training a model to do well for autoregressive prediction is above and beyond what it was trained to do. The context for teacher forcing is also misleading in Figure 7 since the context window changes. 

**Sampling and filtering**
The difficulty of this task depends on the sampling rate as it may be the case that nearby time points are very correlated in the calcium imaging and none of the models are actually predicting novel calcium spikes. Especially since the data was filtered with a low-pass filter in the frequency domain.  Also this ideal filtering is not causal if it is done across the full time-series. 

It is not clear what "only the lowest 10% of frequencies" means in practice. Are the time steps not uniform? This is not standard way of describing a range of frequencies. Especially if the sampling rate is not uniform.  The discussion at the bottom of page 3 is not clear. 

**Relation to neural population dynamics**
The data is essentially neural population level dynamics but measured through calcium imaging. In principle deconvolution of calcium imaging can be used to get approximate spike times. Then, like in  electrophysiology a generalized linear model (GLM) of the spike trains could be performed. To my knowledge the bumps in calcium imaging are smoothed spikes.  Related work for latent factor models for spike trains should be mentioned. 


**Minor points**
The reasoning about the amount of data is not logical. "Given a dataset D with a fixed number of worms N " it is not possible to increase the number of neurons since this is a dataset not experimental collection. Also "increasing N by incorporating more worm" is not consistent with the statement that the dataset is given... The subsequent discussion of nested datasets is perfectly fine, but the precursor discussion is confusing. 

Figure 6's MSE scaling across hidden dimension could also be evaluated as training set changes sizes. 

Figure 7 should be made more clear to document that it is a particular neuron "AVER".

### Questions
Are the models making causal predictions? If not can they be modified.

At what maximal time delay can predictive models be trained?

Can the models be trained to perform auto-regressive prediction? 

Why are only the lowest frequencies kept? Is this done per window or across the full time series.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper explores the scaling behavior of simple models of neural activity of C. elegans neurons, as a function of the amount and diversity of training data and hyperparameters and capacity of the neural network predictor. The authors find regular scaling behaviors for the tested dependencies for three basic model architectures, with the functional relationships consistent across architectures.

### Strengths
- First systematic exploration of modeling of brain activity in C. elegans and its scaling properties.
- Diverse open source datasets utilized in the study, recording up to 30% of the worm's neurons simultaneously.
- Experiments cover relevant basic architecture classes (MLP, RNN, transformer)
- The paper is well written and easy to read.

### Weaknesses
- Code for the study does not appear to be available.
- No evaluation of the impact of the size of the embedding space H and the size of the context window L.
- No exploration of models with more than a single layer in the 'core'.
- Fig. 6 is visibly poorly fitted by the quadratic function, but this is not commented upon in the text.
- No comments about why increasing model size beyond some limit decreases model accuracy (which seems not in line with other domains), and no evaluation on how that critical size threshold might depend on the amount of training data.
- Teacher forcing in the context of the present study is not described in sufficient detail. TF is an algorithm for training, whereas the text makes it sound like an inference technique.

### Questions
- What is the coverage of the neurons across the different datasets and worms? Are some neurons recorded more frequently than others? Is that accounted for in any way in the evaluations? Would it be possible to include a plot of num_recordings(neuron ID) in the appendices?
- How did you decide to retain only the lowest 10% of frequencies? Are the results expected to be significantly sensitive to that choice? What was the cutoff frequency in Hz?
- Have you considered to what degree the scaling properties you found are a function of the underlying system (C elegans) vs of the networks you're training? I understand that brain activity data is hard to find, but perhaps it would be worthwhile to have a comparison baseline generated by a known dynamical system (e.g. coupled oscillators, etc)?
- Is the x axis in Fig. 3 linear? With a single tick label, it is impossible to tell by just looking at the figure.
- Have you tried running (perhaps a subset of) the experiments without the learning rate scheduler to verify that the reported relations still hold?
- Are all the metrics reported in Sec. 3 obtained by computing the MSE as defined in Sec. 2.4 on the test set, i.e. do they all measure the difference between the predicted values and L-1 input (conditioning) values and 1 predicted value that was not part of the input? Also, should the one time-step offset not be accounted for in the formula?
- Since the test/train split is arbitrary, have you tried swapping the test and training sets to see if the observed relations still hold?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
