# From movement to cognitive maps: recurrent neural networks reveal how locomotor development shapes hippocampal spatial coding

- Avg Score: 6.50
- Decision: Accept (Oral)
- Scores: 10, 8, 2, 6

## Abstract
The hippocampus contains neurons whose firing correlates with an animal's location and orientation in space. Collectively, these neurons are held to support a cognitive map of the environment, enabling the recall of and navigation to specific locations. Although recent studies have characterised the timelines of spatial neuron development, no unifying mechanistic model has yet been proposed. Moreover, the processes driving the emergence of spatial representations in the hippocampus remain unclear (Tan et al., 2017). Here, we combine computational analysis of postnatal locomotor development with a recurrent neural network (RNN) model of hippocampal function to demonstrate how changes in movement statistics -- and the resulting sensory experiences -- shape the formation of spatial tuning. First, we identify distinct developmental stages in rat locomotion during open-field exploration using published experimental data. Then, we train shallow RNNs to predict upcoming visual stimuli from concurrent visual and vestibular inputs, exposing them to trajectories that reflect progressively maturing locomotor patterns. Our findings reveal that these changing movement statistics drive the sequential emergence of spatially tuned units, mirroring the developmental timeline observed in rats. The models generate testable predictions about how spatial tuning properties mature -- predictions we confirm through analysis of hippocampal recordings. Critically, we demonstrate that replicating the specific statistics of developmental locomotion -- rather than merely accelerating sensory change -- is essential for the emergence of an allocentric spatial representation. These results establish a mechanistic link between embodied sensorimotor experience and the ontogeny of hippocampal spatial neurons, with significant implications for neurodevelopmental research and predictive models of navigational brain circuits.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This paper presents a compelling computational account of how locomotor
development shapes hippocampal spatial representations. The authors identify
distinct developmental locomotion stages through clustering analysis, then show
that RNNs trained sequentially on these movement patterns recapitulate the
biological timeline of spatial cell emergence. Notably, the model predicts
developmental increases in conjunctive place-direction cells, which the authors
confirm in experimental data.

### Strengths
* Rigorous methodology with excellent reproducibility
* Novel mechanistic insight linking embodied experience to neural development
* Strong experimental validation across multiple datasets
* Comprehensive controls ruling out alternative explanations
* Makes and validates testable predictions

### Weaknesses
## Minor Issues
* Missing ICLR requirement: The paper lacks the required statement about LLM usage in manuscript preparation. Please add this declaration.
* Grid cell specification: The grid cell input for the adult model (Eq. 2, Table 3) could be better justified. Please clarify: (a) what the scale values (0.2, 0.4, 0.6) represent in physical units, (b) how these relate to biological grid scales (given that they are usually increased from one scale to the next by a factor of something close to √{2}, and (c) rationale for the specific orientation value of 0.1 (although I guess this is due to the grid cell orientation relative to walls).
* Statistical reporting: Consider reporting exact p-values rather than just significance markers, particularly for borderline cases (e.g., Figure 4c,g "ns").

## Minor clarifications:
* Explain the min-max normalization approach more explicitly when first introduced
* In Figure 1f, "rat's eye view" should clarify this is simulated panoramic camera perspective (or if this is not a panoramic camera model, which camera model was used)

### Questions
1) Have you considered testing the model's predictions about conjunctive coding emergence through targeted experimental manipulations of locomotor development?
2) Could the framework extend to other sensory modalities or species with different developmental timelines?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper examines how motor development of rodents may shape the spatial representations found in the brain. By analyzing real rodent behavior during development, the authors show that their trajectories can be broadly classified into 3 stages. The authors then train RNNs on predicting the next sensory and motor state, given the current sensory and motor state. They use sensory inputs that are inspired by real experiments, as well as real rodent trajectories. Analyzing the learned representations from the RNN, the authors find that place and head direction cells emerge with training, mirroring what is found in experiments. The authors also show that the model predicts a greater increase in direction selective place cells, which they find evidence for when re-analyzing existing neurophysiology data.

### Strengths
1. This paper is - to my knowledge - the first computational work to show how motor development impacts spatial representations. This is an important point, and one that will (and should) change how the field thinks about place and head direction cells. 

2. The paper is well written and easy to follow. 

3. The experiments are well done, clearly explained, and aligned with experimental evidence. 

4. The use of the RNN model to make a prediction about spatial development that is then found in re-analyzed neurophysiological data is great and further strengthens the case for using RNNs to study this.

### Weaknesses
I identified no major weaknesses of this submission. However, there are a few things that I think should be addressed to make the paper stronger:

1. The authors train the RNN to do one time-step ahead prediction of both motor and sensory state. For path integration, this kind of one step prediction makes sense to me. E.g. when it's dark and I walk from my bed to my bathroom, I'm constantly updating where I think I am. But it was unclear to me: a) why this kind of one time-step prediction would be done in the hippocampus with sensory information b) why this prediction should be one time-step ahead (as opposed to other time-scales, or multiple time-scales). Adding more discussion on this to provide better motivation would be helpful I think. 

2. The authors correctly say that their approach for identifying place and head direction cells - based on a spatial information metric - is standard. However, there is a growing understanding in the systems neuroscience field that a fixed threshold may not be the best way to classify. Many labs now use measures of cross-validation, robustness, consistency, etc., as well as compare to shuffled controls, to identify place cells. I understand the authors are primarily comparing to older experiments where this fixed thresholding approach was used, but it would be helpful I think to include some quantification of robustness of the place and head direction cells. If you split the trial in half and compute the SI for each half, is it similar? 

3. Related to the point above, it would be helpful to have population summary level plots of the distribution of spatial information and RVL, in addition to having individual units plotted (Fig. 3). This would help the readers understand the extent to which the RNN model develops these kinds of properties. 

4. Border cells seem to emerge in both the rodent recordings and the RNN model. Does the border score (and percent border cells) also increase in a similar way between RNN and neurophsyiological data? If so, this could be a further strengthening of the results (and could potentially be a supplemental figure). If no, then this suggests something different between the RNN and rodent development that would be interesting to comment on. 

5. Finally, Cueva and Wei (2019) looked at how an RNN trained to perform path integration develops spatial representations. They find that HD units emerge first, then border units, then grid units. This is in rough agreement with actual experiments. While Cueva and Wei (2019) did not consider how properties of trajectories the RNN follows impacts the development (as the authors of this submission do), I do think it is reasonable for the authors to acknowledge this prior work and its attempt to study development of spatial representation through RNNs.

### Questions
1. Why do the authors choose the loss function they do (one time-step ahead prediction)? 

2. How robust are the place and HD cells the model produces? 

3. What are the population level properties of the RNN units (i.e., distribution of SI, distribution of RVL)? 

4. How does border score emerge with training?

### Soundness
4

### Presentation
4

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
This paper investigates the relationship between locomotor development and the emergence of spatial representations in the hippocampus. The authors employ a computational approach, analyzing rat locomotion data to define developmental stages (crawl, walk, run, adult) and training a recurrent neural network (RNN) model to predict visual input from previous movement states. The core hypothesis is that changes in locomotor development drive the sequential emergence of spatial tuning properties in the hippocampus.

### Strengths
The most notable strength of this work lies in its data-driven finding regarding the emergence of directional tuning. The analysis suggests that hippocampal directional tuning during development tends to arise from cells that initially encode location. This specific observation, derived from their data analysis, offers a potentially valuable insight into the developmental trajectory of hippocampal function.

### Weaknesses
My assessment of this paper reveals fundamental weaknesses that severely undermine its conclusions and overall scientific contribution.

1. Overly Strong and Unjustified Core Hypothesis: The central assumption that developmental changes in brain spatial representations are solely determined by locomotor patterns is a drastic oversimplification. This hypothesis completely neglects the crucial role of genetically programmed neural system development, which is an undeniable and fundamental aspect of brain maturation. To validate such a strong, almost certainly unreasonable assumption, the authors would need exceptionally compelling evidence, which is conspicuously absent.

2. Insufficient Evidence to Support the Core Hypothesis: The model, at best, can only capture some features of CA1 representational development. Crucially, the authors themselves acknowledge that their model fails to spontaneously generate grid cells, and struggles to adequately characterize the developmental features of head-direction (HD) cells in the medial entorhinal cortex (MEC) and other relevant brain regions. This demonstrates that their core hypothesis lacks the explanatory power to account for the full complexity of the navigation system's development. If locomotor patterns were the primary driver, the model should reproduce these key elements more robustly.

3. Biological Implausibility of Model Choice for CA1: The authors' argument that their RNN model is specifically modeling CA1 is highly problematic and fundamentally flawed.
(1) **Lack of Biological Constraints**: The RNN model is trained without any specific biological constraints pertinent to CA1. Without articulating why brain regions other than CA1 cannot be similarly modeled as RNNs, this claim is entirely unsubstantiated.
(2) **Contradiction with CA1 Connectivity**: CA1 is widely recognized for having very sparse recurrent connections; indeed, it is primarily considered a feedforward processing stage from CA3 and entorhinal cortex. Therefore, using a recurrent neural network (RNN), whose defining characteristic is its rich internal recurrence, to model a region known for its lack of such connections is biologically inconsistent and fundamentally unsound. This choice of model severely undermines any mechanistic insights the paper claims to offer regarding CA1 function.

### Questions
Could the authors provide insight into why their current framework might be insufficient for this emergence, especially when compared to other models, such as the Tolman-Eichenbaum Machine (TEM), which have successfully produced grid-like representations? Specifically, what fundamental architectural or mechanistic differences in TEM, if incorporated into the authors' model, might enable the emergence of grid cells?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates how developmental changes in locomotion shape the emergence of spatial representations in the hippocampus. The authors analyzed experimental data on rat locomotor behavior, identified three developmental stages (crawl, walk, and run) via clustering. Then, the paper trained RNNs to predict visual input from visual and vestibular inputs using simulated trajectories that matched those observed in experimental data. They showed that the model exhibited the emergence of spatial tuned units in orders resembling the biological development timelines. The paper also provided a novel prediction where directional selectivity emerges through conjunctive place-direction coding and confirmed this in experimental data.

### Strengths
1. Originality: provided a novel mechanistic model connecting locomotion experience to hippocampal spatial neuron development, and provided novel predictions that are confirmed with experimental data. 
2. Quality: well-controlled study with good ablation and control experiments (e.g., reversed developmental order). 
3. Clarity: very clearly written. The figures are also informative and clear. 
4. Significance: makes a substantive contribution to understanding how sensory-motor inputs shape spatial neuron formations in the hippocampus. It also opens an interesting direction for embodied AI, raising the question of how incorporating embodied inputs and developmental principles could improve the way AI systems learn useful internal representations for spatial tasks.

### Weaknesses
1. Biological motivation for the model architecture: The paper could provide a clearer biological rationale for the modeling choice. Why should we expect a single-layer RNN trained to reconstruct high-dimensional visual input to correspond to the hippocampal circuitry? Is there recurrent interactions between the spatially tuned neurons modeled in the paper? The author should better justify this correspondence. 
2. Task: It would be useful to discuss why the paper chooses to predict the full high-dimensional visual input. Right now, it reads a bit arbitrary as an objective for studying spatially tuned neurons. Is the task choice critical for the results in the paper? Could an alternative task that is lower-dimensional or spatially-relevant (e.g., next location prediction?) yield similar results? Clarifying this could strengthen the argument that the finding reflects some general principle rather than task-specific results. 
3. The context of prior computational work could be presented more clearly. The authors could provide a systematic comparison explaining what phenomena previous models have captured, what gaps remain, and how this work extends or differentiates itself.

### Questions
1. In the model, grid cell input is introduced as an external input provided only in the adult stage. Is it known that grid cells develop independently from the spatially tuned cells studied here? Could their development process be somewhat interactive? 
2. How does the model and its learned representation generalize to novel maps or out-of-distribution visual conditions? Could the model predict developmental abnormalities if exposed to impoverished or atypical sensory inputs?

### Soundness
4

### Presentation
4

### Contribution
1
