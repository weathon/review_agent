# Self-Supervised Grid Cells Without Path Integration

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 3, 6, 5

## Abstract
Grid cells, found in the medial Entorhinal Cortex, are known for their regular spatial firing patterns. These cells have been proposed as the neural solution to a range of computational tasks, from performing path integration to serving as a metric for space. Their exact function, however, remains fiercely debated. In this work, we explore the consequences of demanding local distance preservation in networks subject to a capacity constraint. We consider two distinct self-supervised models, a feedforward network that learns to solve a purely spatial, local distance-based encoding task, and a recurrent network that solves the same problem during path integration. We find that this task leads to the emergence of highly grid cell-like representations in both networks. However, the recurrent network also features units with band-like representations. We subsequently prune velocity inputs to subsets of recurrent units, and find that their grid score is negatively correlated with path integration contribution. Thus, grid cells emerge without path integration in the feedforward network, and they appear significantly less important than band cells for path integration in the recurrent network. Our work provides a minimal model for learning grid-like spatial representations, and questions the role of grid cells as neural path integrators. Instead, it seems that local distance preservation and high population capacity is a more likely candidate task for learning grid cells in artificial neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a self-supervised objective function demanding distance preservation subject to a capacity constraint, as a model of grid cells in the MEC. The authors used the objective to train two different neural network architectures and found that grid cells emerged in both models. The authors have also performed a pruning experiment to understand the role of grid cells and band cells in path integration and reached the conclusion that band cells instead of grid cells are responsible for successful path integration. They also performed a thorough investigation into the pattern formation and generalization capabilities of the models.

### Strengths
I enjoyed reading the experiments of this paper, which are very comprehensive. The motivation and details of each experiment are also clearly presented. I particularly like the authors’ discussions on the role of band cells as this cell type has been understudied in many of the previous works on modeling grid cells. The Methods section is also very clear and well-presented.

### Weaknesses
- Writing: Despite the clear presentation of each experiment, I believe the paper can be improved by adding subsection titles to each of its 3 experiments. From row 107 to 154 the paper is clearly presenting the result that grid cells do emerge in models trained with the proposed objective, which can be put under one subtitle. Similarly, from row 155 to row 199 the paper is talking about pruning and from 199 onwards it is presenting results on pattern formation, connectivity and generalization of the models. It would be nice to have subtitles for readers to navigate through the experiments.
- Comparison to Schaeffer et al. 2023: A critical component that is missing from this paper is a formal comparison to other self-supervised learning models for grid cells, in particular Schaeffer et al. 2023, as it bears a lot of similarities to the self-supervised loss proposed in this paper. From what I understand (also what the authors claimed), the loss in this paper is simplified compared to Schaeffer et al. 2023 as it only keeps the distance preservation and capacity terms, but discarded the path invariance and conformal isometry terms. I have the following questions (the authors don’t have to run extra experiments to answer these questions; some conceptual clarification in the introduction should suffice. Although experiments are welcomed if they can provide further clarification):
    - What is the motivation for removing these two terms except for simplicity? 
    - Why does it still work without these terms (I remember Schaeffer et al. performed an ablation study and found that path invariance is critical to the emergence of grid cells)? 
    - Is it related to the L1 capacity constraint? Does it introduce path invariance implicitly?
- Novelty (related to previous point): If I understand correctly, the novelty of this paper is 1) a new self-supervised model for grid cells and 2) using the model to understand the necessity and sufficiency between grid cells and path integration. The second novelty is a valid and interesting one and I like it a lot; however, without a formal comparison to Schaeffer et al. 2023, I found it hard to understand why the authors used their new model instead of the other models (Shaeffer et al. 2023, or even path-integrating RNN models). This is particularly important given that the authors claimed in the Summary “Our approach, featuring a minimal model, has allowed us to isolate key factors contributing to the emergence of grid-like representations” - how is the isolation of key factors unique to the proposed model?

### Questions
- I wonder if the significance of band cells to path integration has something to do with the speed of the simulated agent? In my own experiments with path integrating RNNs, I found that band cells are more likely to emerge with high-speed agents. Intuitively this makes sense because if an agent moves too fast then two firing peaks of a grid cell might be linked together. Feel free to do some experiments with speed if you have time/space in the paper!
- Fig2d: This figure is quite hard to understand. I understand what each column means (time step), but what does each row mean? Do they correspond to c, e, and f? Maybe I missed it but I didn’t find any information about the meaning of each row in the main text.
- Fig3c: the caption for this panel has some typo.

Overall, I'm giving a score 5 to this paper. However, if the authors can clarify the difference to Schaeffer et al., 2023 and highlight the key factors in their loss that 1) enabled the emergence of grid cells without path invariance and conformal isometry and 2) enabled them to perform the pruning experiments that explored the role of grid/band cells in path integration, I would be happy to raise the score to 6.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper explores the functionality of grid cells and investigates why they emerge in biological systems using an RNN-based deep learning model with a specially designed loss function. The loss function consists of two components: the first enforces the Euclidean distance between RNN representations to reflect corresponding spatial distances within a defined range, effectively aiming for a locally isometric mapping between physical locations and neural representations. The second component is an L1 norm on all neural activations, intended to reduce overall neural energy consumption. By training the RNN with this loss, the authors observe the emergence of grid cells and band cells similar to those observed in experiments. Two training paradigms are used: one with velocity input and one without. The findings suggest that the emergence of grid cells is independent of path integration. Additionally, selective removal of velocity input from neurons with higher grid scores showed that path integration errors remained unaffected, suggesting that grid cells might not be directly involved in path integration tasks.

### Strengths
The loss function design is novel in its approach to generating grid cells, explicitly incorporating spatial distance representations within the loss. This addition reinforces existing theories positing grid cells as spatial metric providers.

### Weaknesses
1. **Novelty:** While the specific mathematical form of this loss function is novel, the core idea has been implemented in slightly different forms in previous studies. For example, in Dorrell (2022), the loss function encourages distant physical points to have more different neural representations. Although the formulation differs, the underlying concept is similar. Additionally, the observed phenomena regarding grid cells and path integration, such as grid cell emergence independent of path integration tasks and band cells being more likely to perform path integration than grid cells in RNNs, have been previously reported (see Sorscher et al. 2022, Schøyen et al., 2023).  
   
2. **Grid Cell Characteristics:** The grid cells emerging in the RNN exhibit a single spacing and varying orientations, which contradicts experimental observations. In biological systems, grid cells are organized into modules with similar orientations and distinct spacings.

3. **Loss Function Design:** The design of the loss function itself raises concerns. If grid cells are intended to maintain an isometric mapping, why is this restricted only to the local environment? This constraint appears to have been added to explain the hexagonal firing patterns of grid cells, yet this assumption lacks biological justification and complicates the functional interpretation of grid cells. This restriction might also explain why the model produces grid cells with a single spacing, further detracting from its biological plausibility.

### Questions
In Figure 2b, it seems that only two neuron types emerge in the model: grid cells and band cells, with classification based on grid score. However, in other studies, such as Sorscher et al. (2022), many other neuron types have been observed, including neurons without clear spatial periodicity. I am curious whether such neuron types are indeed absent from the model proposed in this paper. If so, could the authors discuss why this might be the case? A comparison between model designs to highlight factors driving these differences could be interesting. Alternatively, if these neurons are present but not mentioned, I recommend that the authors clarify this in the manuscript.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents both feedforward (FF) and recurrent neural network (RNN) models that are trained with the same self-supervised loss function but different inputs such that the RNN but not the FF model learns to path integrate. With an appropriate choice of loss function hyperparameters, both the RNN and FF models develop units with grid-like spatial representations, supporting a recently proposed hypothesis that the function of grid cells is not to support path integration. However, only the RNN develops band-like units. The authors present evidence that it is the band-like units, not grid units, that primarily support the RNN’s ability to path integrate.

### Strengths
The authors adapt and simplify a self-supervised loss function that leads RNNs to develop grid-like representations. The authors also show evidence in support of a novel hypothesis that it is band-like units and not grid-like units that are primarily driving the RNN’s ability to path integrate. This suggests a new interpretation for neurons with these response profiles in the brain.

### Weaknesses
There are many design choices that influence the representations that RNNs develop. My primary concern is that the conclusions from this RNN model are idiosyncratic to it and do not actually teach us anything about the brain, or perhaps even about RNN models more generally. For example, the authors mention other models that accomplish path integration using RNNs with purely grid-like representations. So we are left with the conclusion that in some models grid-like units are responsible for path integration and in some models, like this one, they are not. This being said, I appreciate the novel modeling work in this study and the new evidence in support of the interesting hypothesis raised by Schoyen et al. iScience (2023) that band-like units, and not grid-like ones, serve a primary role in path integration performance.

### Questions
* The RNN in this model path integrates for 10 timesteps. This seems small relative to some other models. For example, I believe Schaeffer, Khona, et al. 2023 used 60 timesteps, Banino et al. 2018 used 100 timesteps, and Cueva & Wei 2018 used 500 timesteps. This gets back to the relevance of the model for biology. The ratemaps produced for a neuron in an animal are taken from long trajectories where the animal traverses large portions of the environment. What do the ratemaps look like if you let your model run for more timesteps and gather unit activity from a single long trajectory? Do you still see grid-like and band-like patterns?

* The finding that a feedforward network can develop grid-like responses without path integration was also observed by Dordek et al. eLife 2016 so this should be cited.

* Do any of the feedforward models, that do not path integrate, have band-like units? For example, the Grid Score of the FF model in Figure 1d also has a peak near 0 (although less pronounced than the RNN), and some of the FF models in Figure A1 have Grid Scores that peak near 0, potentially indicating the presence of band-like units.

* How well can the RNN be used to path integrate? For example, what is the difference between the ground truth (x,y) trajectory specified by the velocity inputs, in comparison to the (x,y) estimate if you linearly combine the activities of the recurrent units (g) by training a matrix of size 2 x 256 to weight the 256 RNN units?

* Presumably the RNN does not have band-like representations before the RNN is trained. Does the RNN develop these band-like representations at the same time it starts to perform well on the path integration task? Or does the RNN perform well on path integration before these band-like representations have developed?

* Do the other models you trained with different hyperparameters also rely on band-like units for path integration? In other words, if a model doesn’t have units with Grid Scores near 0 does this mean it cannot path integrate?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This model studies computational models of grid cells. The authors constructed and investigated two classes of models: Feedforward models (FF) and Recurrent Neural networks (RNNs). They found that under certain training objectives, both classes of models led to hexagonal grid-like firing patterns. Their training objective encourages the network representation to preserve the local distance structure. The paper also performed several additional analyses to understand the trained networks. Some of the results are nice and interesting, although overall the framework is a bit incremental.

### Strengths
Originality: This study follows a line of work using machine learning techniques to understand the grid cells (e.g., Cueva&Wei, 2018, Banino et al, 2018, Sorscher et al, 2022, Whittington et al, 2020, Xu et al, 2022. ) The exact learning objective (Eq. 1) seems to be new, but similar objective functions have been proposed in the literature.

Quality: The authors studied and compared two classes of models. They also performed several additional analyses to understand the trained networks, for example,  the connectivity analysis, the analysis on band cells,  and the generalization of the firing pattern outside the training box. These are strengths of the paper.

Clarity: The presentation is generally clear, except for a few places (see below).

Significance: The paper tries to determine the exact ingredients that would lead to grid-firing patterns in the trained neural networks. This seems to be a step toward the right direction.

### Weaknesses
I feel that some of the results in the paper are quite interesting, yet the work is a bit incremental. The constraint to preserve local physical distances in the representational space appears to be similar to the conformal isometry hypothesis proposed by Xu et al, 2022, and further investigated in Xu et al, 2024. In the current study, a Gaussian envelope is used to specify the spatial scale that the distance-preserving constraints should be enforces. A discussion on the similarity and difference of the model here and those proposed by Xu et al would be useful for the readers. 

The author emphasized the L_1 activity-based regularization in Section 2. However, the consequence of it was not clearly described in the paper. It would be helpful if the authors could show some results to compare the learned firing patterns based on L_1 v.s. L_2 regularizations. 


The band cell were also reported in some prior modeling work, e.g., Cueva and Wei 2018. There is some experimental evidence (although slightly controversial) on band cells in medial entorhinal cortex. It would be nice to refer to these results. 

The model was not as clearly described. For example, it is not exactly clear what were the inputs to the hidden layers in the FF model. In the figure, it seems that only (x,y) was the input, but I didn’t find a description in the text or Method section. 

The paper would be strengthened if the effects of the key hyper-parameters were reported. These include the spatial envelope scale parameter and the weight for the first term in the loss function.

### Questions
Does the L_1 regularization (as opposed to L_2 regularization) in the loss function really matter in terms of the grid patterns?


I don’t fully understand the title “grid cells without path integration”. For the RNNs considered in the paper, there was a velocity input. In what sense this is without path integration? I saw the statement that the grid-like cells in the models “appear substantially less important than band cells for path integration in the recurrent network”, but I don’t think that’s equivalent to say that grid cells show up without path integration in the RNNs. If removing the velocity inputs still result in grid firing patterns in RNNs, that would be justified. I’d appreciate if the authors could further clarify this point. 


The authors report binomial grid cell orientations in some model variants. Were these results consistent with the neural data? 

The units in Fig. 1d is not specified. In particular, what are the units for orientation and spacing?

### Soundness
3

### Presentation
3

### Contribution
2
