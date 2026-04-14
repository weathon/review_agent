# A Computational Framework for Modeling Emergence of Color Vision in the Human Brain

- Decision: Accept (Oral)
- Scores: 8, 10, 8, 6, 8

## Abstract
It is a mystery how the brain decodes color vision purely from the optic nerve signals it receives, with a core inferential challenge being how it disentangles internal perception with the correct color dimensionality from the unknown encoding properties of the eye. 
In this paper, we introduce a computational framework for modeling this emergence of human color vision by simulating both the eye and the cortex. Existing research often overlooks how the cortex develops color vision or represents color space internally, assuming that the color dimensionality is known a priori; however, we argue that the visual cortex has the capability and the challenge of inferring the color dimensionality purely from fluctuations in the optic nerve signals. To validate our theory, we introduce a simulation engine for biological eyes based on established vision science and generate optic nerve signals resulting from looking at natural images. Further, we propose a bio-plausible model of cortical learning based on self-supervised prediction of optic nerve signal fluctuations under natural eye motions. We show that this model naturally learns to generate color vision by disentangling retinal invariants from the sensory signals. When the retina contains $N$ types of color photoreceptors, our simulation shows that $N$-dimensional color vision naturally emerges, verified through formal colorimetry. Using this framework, we also present the first simulation work that successfully boosts the color dimensionality, as observed in gene therapy on squirrel monkeys, and demonstrates the possibility of enhancing human color vision from 3D to 4D.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a computational framework to explore how color perception might emerge in the cortex through a self-supervised learning model inspired by biological vision. The authors aim to simulate the early visual processing stages by building a model that integrates a retinal layer with different types of cone cells (sensitive to various wavelengths) and a cortical layer using predictive coding. The model learns to extract color and shape information from visual inputs without explicit color labels, reflecting a form of emergent perception.

A significant contribution is the model's ability to simulate the transition from dichromatic to trichromatic vision, akin to genetic interventions shown in experimental biology. The framework demonstrates that under different cone configurations, the model can adapt to various perceptual dimensions, which provides insights into color processing mechanisms in the brain. This interdisciplinary approach contributes to our understanding of self-learning in color perception and offers a basis for future studies on adaptive vision systems in neuroscience and AI.

### Strengths
1. The originality stems from the integration of self-supervised learning with a computational framework inspired by biological color perception. By combining a retinal model with a cortical learning mechanism, the authors attempt to create a biologically plausible framework for emergent color and shape perception. This interdisciplinary approach bridges machine learning with neurobiological modeling, which is a valuable exploration in color perception studies.
2 The paper’s methodology reflects a reasonable approach to simulating early visual processing stages. The authors carefully design the retinal model to simulate color cone responses and introduce predictive coding for the cortical learning mechanism. Though some limitations exist, the study provides valuable insights into the use of predictive coding in visual processing models, highlighting how cortical networks could feasibly learn color information.
3 The paper is generally well-written, providing a clear overview of the methods, experiment design, and results. Figures and visualizations are well-crafted, aiding in understanding the experimental findings. The authors successfully explain complex processes, such as predictive coding and color perception, in an accessible way.
4 This study is significant within the niche area of computational neuroscience focusing on color perception. It addresses the unsolved question of how color perception might emerge in a self-learning cortical network, which is a meaningful contribution to understanding visual processing mechanisms. The attempt to simulate gene therapy’s effect on color perception (such as enhancing dichromatic to trichromatic vision) offers a unique perspective that could inspire further studies in perceptual enhancement.

### Weaknesses
Biological Plausibility of Input Data:
The model uses natural RGB images as input, which already contain color information. This approach diverges from biological realism, as the retina does not process pre-encoded color data but rather interprets light of various wavelengths. To improve biological plausibility, the model could use spectrally-encoded images or raw spectral data as inputs to better reflect how retinal cones process different wavelengths. This adjustment would align the model closer to real visual processing and allow more meaningful exploration of emergent color dimensions.

Validation against Biological Data:
While the paper presents a computational framework, it lacks direct validation with biological data, which is essential to support its claims of modeling emergent color perception. Adding quantitative comparisons with known biological data on cortical color responses, such as V1 or V4 neuron responses in primates, would strengthen the study’s credibility. Incorporating benchmarks like psychophysical data on color matching could also make the findings more robust and biologically grounded.

Ambiguity in Learning Mechanisms:
The model’s reliance on predictive coding for learning color dimensions in the cortex is interesting but raises questions about biological accuracy, as it assumes self-supervised learning can fully capture color processing. However, in biological systems, factors like feedback from other cortical regions and adaptive changes play significant roles. Expanding the model to incorporate feedback loops or other cortical interactions could offer a more comprehensive view and make the learning process more aligned with biological color perception theories.

Overstated Novelty in Emergent Color Dimensions:
While the paper claims novelty in emergent color dimensions, related work on unsupervised learning of color and perceptual features is well-established. For example, studies on hierarchical models of color vision (e.g., Komatsu, 2006; Conway et al., 2023) and other unsupervised learning models in color processing (Zhou & Mel, 2008) address similar concepts. The authors could enhance clarity by explicitly discussing these foundational works, explaining how their approach extends or diverges from these models, and adjusting novelty claims accordingly.

Komatsu H. The neural mechanisms of perceptual filling-in. Nature reviews neuroscience. 2006 Mar 1;7(3):220-31.
Zhou C, Mel BW. Cue combination and color edge detection in natural scenes. Journal of vision. 2008 Apr 1;8(4):4-.
Conway BR, Malik-Moraleda S, Gibson E. Color appearance and the end of Hering’s Opponent-Colors Theory. Trends in Cognitive Sciences. 2023 Sep 1.

### Questions
1 There are many parameters in the model, especially in the cortical learning layers. How were these parameters selected, and are they based on biological references or empirical tuning?
2 The experiments are mostly theoretical, focusing on emergent color dimensions in simplified scenarios. Have the authors considered testing their model on more complex datasets or tasks, such as object recognition with varying lighting conditions?
3  The model relies on self-supervised learning for color perception emergence. Biological systems often use feedback and interaction between cortical areas, which may influence color perception development. How do the authors justify the exclusion of such feedback mechanisms?
4 The paper mentions using divisive normalization for edge detection in the model but does not detail why this specific normalization was chosen over others. Could the authors elaborate on why divisive normalization aligns best with their framework, or if they explored alternatives?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
The manuscript introduces a new approach to computational modelling of colour vision, combining established knowledge of human visual systems with modern artificial neural networks. The proposed framework suggests that the cortex (the agent’s processing unit) lacks direct information about the retina (the agent’s sensory hardware) and investigates the types of colour perception that could arise in such a setup. The model is trained using a biologically realistic principle: predicting future optic nerve signals, aligning with predictive coding theory, a successful approach to unsupervised learning in vision research. To examine the internal representations created by this model, the study employs psychophysical tests commonly used in human colour-matching experiments. The results provide interesting insights into the dimensionality of colour perception at the single-pixel level, offering promising avenues for future exploration into the possibility of tetrachromacy in humans.

### Strengths
Overall, I found the article very interesting with several fascinating ideas. Its greatest strength lies in its innovative thinking, presenting a fresh computational approach—sensory-agnostic perception—that stands apart from prior models.

### Weaknesses
I would summarise the current weaknesses of the manuscript into three areas:

1. **Presentation**: While generally well-written and easy to follow, some parts of the manuscript require considerable effort from readers to fully understand. For example, abbreviations like "CMF-SIM" and "NS" are not immediately clear. Later, NS is explained as "Neural Scope," but CMF-SIM remains unclear. In the supplementary materials, the "colour matching function" is mentioned, but what about SIM? This ambiguity demands extra effort from readers. Also, there are multiple unresolved cross-references ("??") in the supplementary materials.

2. **Results**: The results section could benefit from being more thorough. Specifically, a broader set of analyses across multiple training instances would strengthen the rigour of the findings and clarify them for readers.

3. **Related Work**: Although the manuscript introduces a unique approach that differs from previous models, a brief section in Related Work could discuss other methods using artificial neural networks to model aspects of human colour perception, such as colour categories, contrast sensitivity in chromatic channels, and colour opponency.

### Questions
Below is a list of not only questions but also general comments:

### Optic Nerve Signals
- I think the depiction of optic nerve signals in Figure 1 and the supplementary video should include the idea of distinct photoreceptor cells, which create separate pathways, like parvocellular and magnocellular pathways.
- Following the previous point, in the simulation, if I understand correctly, the optic nerve signal (ONS) is modelled as a 2D matrix, which closely resembles raw images from cameras without further processing. However, since photoreceptor positions are known, a more accurate simulation might use a 3D matrix with the third dimension representing photoreceptor channels. This would mean at each spatial location only one channel contains information (light absorption values). I’d be interested in your thoughts on this.
- In the statement, "[...] colors of the scene are spectrally sampled by different types of color sensitive cells (cone cells) in the retina, appearing as a layer of spatial noise in the ONS [...]", I would argue that raw photoreceptor signals, while noisy, are not spatial noise but spatially meaningful. This is reflected in Figure 3, "True ONS at time t". I think Figure 1 should display a similar visualisation rather than the highly noisy image currently shown.

### Colour Decoding or Encoding?
Throughout the manuscript and in the supplementary video, the authors describe human colour vision as decoding colour. For instance, the abstract begins: "It is a mystery how the brain decodes color vision purely from the optic nerve signals it receives, with a core inferential challenge being how it disentangles internal perception with the correct colour dimensionality from the unknown encoding properties of the eye."

I’d suggest that we do not decode colour vision but rather construct it (in a sense, we encode colour to interpret the world more easily). It is not only from optic nerve signals; our colour perception is heavily influenced by semantic and cognitive processes. I don't mean to play with linguistic terms but raising a conceptual difference. For instance, in my view, the function Phi in Figure 3 "encodes" the internal percept and the CMF-SIM block shows different potential decoding of the internal percept. Currently, Phi is based solely on ONS, but a more complete model would incorporate input from cognitive processes that impact how we encode colour information of a scene.

### Measurement of Colour Dimensionality
I agree that colour is represented in the cortex as a high-dimensional space, $R^N$, but I think $N$ includes more than mono-, di-, tri-, and tetrachromacy. It also encompasses other aspects of colour appearance that are independent of photoreceptor dimensionality, such as transparency, glossiness, etc. In essence, properties of colour perception often relate to objects rather than individual pixels (e.g., see "Colour appearance and the end of Hering’s Opponent-Colors Theory", *Trends in Cognitive Sciences*, 2023, and "Colour perception: objects, constancy, and categories", *Annual Review of Vision Science*, 2018). I’m curious if you’ve analysed different dimensions of the internal percept in the cortical model, as hypothesised, and whether similar concepts emerged there.

### Why Did Human Colour Vision Emerge?
One of the manuscript’s interesting findings is that monochromacy, dichromacy, trichromacy, and tetrachromacy emerge in sequence during the learning steps. This suggests that colour space dimensionality increases evolutionarily without altering the ecological conditions of the system (i.e., the system’s interaction with its input remains the same through the learning process). In contrast, the colour vision dimensionality in biological organisms varies based on their environmental needs. For example, human trichromacy is thought to have evolved for tasks like distinguishing ripe fruit and foliage or recognising emotional states in faces. In the manuscript’s unsupervised learning model, the system’s needs remain constant, yet the colour vision dimensionality expands as learning progresses. I have a few questions about this:
   - What drives this change in dimensionality within the network’s colour vision?
   - How consistent is this emergence? If you repeated the process 100 times, how much variation would appear in the resulting colour vision systems?
   - Why does this evolution stop at four dimensions rather than expanding to five or more?

### Minor Points
- What does a "private-line" visual pathway refer to?
- In Figure 2, for consistency, I suggest using the same balloon image as Figure 1 to clarify the model’s detailed processes.
- In the context of this statement, "[...] random distribution of L, M and S cone cells on the retina with a relative probability of 0.63, 0.32 and 0.05 [...]", it might be helpful to mention the large individual variation in these probabilities, which significantly impacts individual colour perception. Have you experimented with different distributions, and could this framework contribute to understanding this variation?
- In the context of this statement "Specifically our DoG kernel has standard deviations of 0.15 and 0.9 cone diameters, respectively, for the positive center and negative surround Gaussians [...]", the authors might consider modelling dynamic, contrast-dependent centre-surround modulation of lateral inhibition (e.g., see physiological studies by Alessandra Angelucci). This could enhance the efficiency of DoG models (e.g., "Colour constancy beyond the classical receptive field," *PAMI*, 2017, and "Feedback and surround modulated boundary detection," *IJCV*, 2018).

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents a novel computational framework for understanding the emergence of color vision in the brain, centered on a self-supervised learning model that interprets optic nerve signals without predetermined color dimensions. The framework includes a simulation engine that processes spectral images through realistic eye mechanisms, including eye motion and cone cell sampling, to generate optic nerve signals. The model demonstrates how different types of color vision (monochromatic, dichromatic, trichromatic, or tetrachromatic) can naturally emerge based on the number of cone types in the retina without requiring predefined color encoding. Notably, the framework successfully models enhancement scenarios, such as the conversion from dichromatic to trichromatic vision demonstrated in squirrel monkey gene therapy experiments. It suggests the possibility of enhancing human color vision from trichromatic to tetrachromatic.

### Strengths
1. The paper proposes a new model that can simulate the emergence of color vision through a cortical learning mechanism. This model uses self-supervised learning to develop color representation from optic nerve input, demonstrating how color perception emerges naturally from visual data without labeled learning.
2. By simulating color vision without explicit biological pathways, the authors propose a new theory of how color vision could emerge in the brain, suggesting that specific neural pathways may not be strictly necessary for color representation.
3. The model’s insights could inform research into treating or understanding color blindness, possibly providing theoretical guidance for therapeutic approaches.

### Weaknesses
1. The paper proposes a cortical learning mechanism for color representation without explicitly mimicking the V1-V2-V4 pathway commonly associated with color processing in biological systems. Does this imply that, in your view, the V1-V2-V4 pathway may not be strictly necessary for achieving color representation? I’m curious if your findings suggest that color perception could emerge through other mechanisms or pathways outside of this traditional hierarchy.
2. Does the model in your paper learn directly from RGB format images? If so, how does it expand from the RGB color space to a 4D color space within the learning process? I’m curious about the mechanisms that allow the model to infer or represent a 4D color space based on RGB input.
3. While your model provides an explanation for the emergence of color representation, I find it challenging to accept that the V1-V2-V4 pathway might not play an essential role in color perception. For instance, the study by Li, Liu, Juusola, and Tang (2014) (see below) on perceptual color mapping in macaque visual area V4 underscores the significance of this pathway in color processing. Could you clarify how your model accounts for or relates to this traditional neural pathway?

> Li, M., Liu, F., Juusola, M., & Tang, S. (2014). Perceptual color map in macaque visual area V4. _Journal of Neuroscience, 34_(1), 202–217. https://doi.org/10.1523/JNEUROSCI.4549-12.2014

### Questions
1. In part 1 of Figure 2, fixational eye movements are modeled as small drifts controlled by a uniform probability distribution (*dX*, *dY* ) ∼ *U*{(−15, 15) × (−15, 15)}. I watched the supplementary vedio about the model, it appears (if I'm correct) that the drift pattern displays frequent small adjustments with occasional larger movements. Could you clarify how the uniform probability distribution achieves this effect, given that it should theoretically produce equally likely shifts across the specified range? And why do you choose 15-pixel as the range of movements?
2. I am also curious about the drift pattern moving from the eyes to the mouth. How do the fixational eye movements capture both the eyes and the mouth?
3. I suggest authors review the use of LaTeX \ref{} commands, as there are missing figure numbers in several instances (e.g., lines 71, 141, and 180) in the Supplementary Materials.
4. In your model, could you explain the process used to convert the "scene stimulus" into "incoming spectra"? Specifically, I'm interested in whether you apply any spectral sampling or decomposition steps and if there are transformations based on the eye’s optical properties to simulate the spectra reaching the retina.
5. Could you clarify how cone cell activation is calculated based on the received light? I am wondering whether you apply spectral sensitivity functions for each cone type (L, M, S) or any other specific computational approach to simulate how cones respond to incoming light.
6. ON/OFF pathways in bipolar cells and the center-surround organization in RGCs work together to enhance contrast and edge/contour detection, but in your Figure 2 part 3, I don't see the edge or contour section of the scene has been enhanced?
7. Could you clarify how lateral inhibition works among cone cells in your model? If I understand correctly, this mechanism balances the activation of different cone cells and adjusts their response to light, and then enhances edge and contour. Additionally, could you provide more details on how cone cells, bipolar cells, and RGCs are connected, especially in terms of how these connections contribute to spike generation? I believe these details would be helpful to discuss.
8. In Part 1 of the simulation engine of biological eyes, are there any specific learning rules or algorithms used to adjust parameters or process visual information? 
9. Why do you choose the standard deviations of 0.15 and 0.9 in DoG kernel?
10. Could you elaborate on the mechanisms that contribute to the emergence of color representation in your model? From an intuitive perspective, I’m curious how the self-supervised learning and prediction error minimization guide the model toward developing a stable and structured color space, especially as it expands beyond initial color dimensions.
11. Concerning Questions point 10, the model predicts a 4D color space (LMQS) from 3D RGB color images. Since we don’t have four types of cones, we can’t truly envision or confirm what 4D images would look like. Have you tried using 2D colorful images to predict a 3D color space? This approach could help test and validate the model’s predictions.
12. According to Weaknesses point 3. I'm interested in understanding whether the mechanisms in your model might have implications for curing color blindness, particularly in primates such as squirrel monkeys. If the color representation depends on ventral pathway of V1-V2-V4, do you think the your model could potentially inform approaches to treat or "cure" color blindness?  I’d appreciate any insights you could share more information with me.
13. A typo "investigated" on line 156 should be corrected.
14. I don't see the code releases.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose that realistic (because of being 3-dimensional) encoding of color percepts emerge assuming a certain number of spectral sensors and temporal predictive coding. They explore this idea with different number of spectral sensors (or kinds of cones).
In their proposal, first, they learn the different elements of the encoding function by maximizing the consistency (minimizing the error) between the signal captured at time t+dt and the prediction computed from the signal at time t and the displacement information between saccades. Then, the dimensionality of the color percept is defined as in classical colorimetry by looking for the minimal number of independent sources that need to be changed to make a color match (to make the encoded responses the same).

### Strengths
The paper is original in proposing predictive coding as a way to fit the elements of color coding. 
Another originality is bringing to the machine learning community a concept from vision science: using a response matching experiment to define the dimensionality of the encoding space for certain concept (in this case, the color percept).
However some of the results obtained from this original view seem obvious (see below).

### Weaknesses
* Authors claim the emergence of color vision (e.g. titles of the section 4 and 6), however what they reproduce is just the dimensionality of the color space, or the number of illuminants required for response matching. This dimensionality (even if correct) is not a full description of color vision, and should not be reported as that. For example, the dimension does not explain the positive and negative lobes of spectral sensitivity of spectral sensors [Wiesel&Hubel J.Neurophysiol.66], it does not explain their adaptivity under change of illuminant [Von Kries 1902], nor the nonlinearity of their response [Krauskopf & Gegenfurtner JOSA 92], nor the non-Euclidean nature of subjective distances between colors [MacAdam JOSA 42], to cite a few facts. 

* Moreover authors refer that the dimension of color space is 3 when assuming 3 different spectral sensors. That is consistent with the result of classical *symmetric* color matching experiments with colors seen in *isolation*, the paradigm of classical tristimulus colorimetry [Wyszeki & Stiles Wiley 82]. However, when seing colors *in context*, the independent dimensions are not 3 but at least 5: not only brightness, hue and colorfulness, but also lightness (not the same as brightness), and chroma (not the same as colorfulness) [see Fairchild, Color Appearanace Models, Wiley 2013]. Therefore, the chromatic content of the spatial context (not only the number of sensors) is critical for the emergence of this 5-dimensional nature, and this is not mentioned in the work. This 3d limitation may come from training limited to regular (white / daylight) illumination, and testing using symmetric color matching with simple (isolated flat) samples, as in the CIE1931 colorimetry. This limitation should be acknowledged and discussed in the work.

* The authors say that previous literature always assumed the dimensionality of the color space. However, there is literature, based on different principles, that can be used to derive the minimal number of sensors and their spectral sensitivity. For instance, [Singh et al. Proc. 3rd Int’l Workshop on Statis. Comp. Theories of Vision 03, Jimenez & Malo IEEE Trans.Geosci.Rem.Sens.14] develop expressions for the accuracy of the recovery of the reflectance of the surfaces and spectrum of the illuminant depending on the use of spatial information and on the number and spectral sensitivity of the sensors. These represent a different principle (inference of object reflectance with color constancy) that can be used to set the dimensionality of the color encoding in a different way as that proposed by the authors.

* The proposed method gives obvious results at some points and these are contradicted at some other points. An example of obvious result is what is said at the abstract (line 23-25): "When the retina contains N types of color photoreceptors, our simulation shows that N-dimensional color vision naturally emerges". However, a later statement seems to contradict the above: in line 352: "CMF-SIM reveals the color dimensionality is 1-D for observers with 3 cone types." 

* Methods as the Color Matching Function Simulation or the Neural Scope are not clearly explained in the paper.

### Questions
* Authors claim the emergence of color vision. Please tone down / reformulate this contribution. What you get is the dimensionality of the color space. 

* The limitation of the 3-d result should be acknowledged and discussed in the work. (My guess) is that richer training and more general testing (e.g. asymmetric matching) would lead to larger dimensionality.

* Cite alternative principles that can be used to derive the number of necessary sensors to encode spectral (or chromatic) information.

* How do you reconcile these two apparently contradictory sentences? (line 23-25): "When the retina contains N types of color photoreceptors, our simulation shows that N-dimensional color vision naturally emerges". (line 352): "CMF-SIM reveals the color dimensionality is 1-D for observers with 3 cone types." 

* The Color Matching Function Simulation in the main paper is only explained in FIg.4 and this may not be enough for the average ICLR reader.

* I missed the explanation of the Neural Scope. Note that the generation of images that simulate the perception of dichromats is not trivial (Brettel et al. JOSA 97, Capilla et al. JOSA 04).

* Please do not refer to supplementary videos to explain concepts (as in line 64 of fig.1 or as in line 201 of the supplementary material). If extra explanations are required, please do it through additional sections in the supplementary material. 

* Please define the acronyms the first time they appear. Pay special attention to CMF-SIM and NS. I think CVD is not even defined.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors claimed that they reverse engineered how color vision may emerge in a biologically inspired cortical learning model. They use simulated optic nerve signals (ONS) as input and trained neural networks to individually approximate three invariant properties of the retina: cell positions, cone spectral type, and magnitudes of lateral inhibition. Their model optimizes the error between the predicted ONS signal at the next sampling step and the real simulated ONS signal. They showed that during the minimization of error between inferred ONS and real ONS, the internal perception of their model gradually adds individual primary colors. To show the aggregation of primary colors, they probe the learned embedding with another separately trained model; they call this separate color matching model CMK-SIM. This CMK-SIM is based on text book colorimetry. They use the observation that CMK-SIM found three primary colors in their trained model (instead of the raw optic nerve signal as the model input) as the success of recovering color vision.

### Strengths
The authors claim that the main novelty of the paper is the framing of the emergence of color vision as a learning problem. I agree the machine learning framework includes a couple biologically inspired constraints. I would appreciate some clarifications to better understand the significance of the cortical learning model.

### Weaknesses
The technical details of this paper are a bit hard to follow (those missing ?? reference in supplementary do not help). For example, I would appreciate further clarification on how the authors choose their baseline models. The authors merely took the entire cortical learning model out (baseline I) or simplified their ONS signal (Baseline II). I guess color/cone spectral type information is present in the ONS signal? Can the author verify this (although CMK-SIM choose to only use 1 color)? 
 
Now another possibility is that the U-net architecture the authors use for the C and D gets way overfit (for example, [1] showed that U-net can generate natural image from white noise inputs and its overfitting occurs between 2400 and 50K steps. The authors here use 100K steps) to recover 3 colors. Considering how much inductive bias is there in the deep architecture, this large number of iterations makes it unclear *where* exactly the color dimensionality emerges. Although the authors postulated the decoder and encoder into factorizable components. There are no ablation studies showing how different learnable components contribute to the emergence of color vision (or if this is not possible given how intricate these components are, would the authors please clarify?)

[1] Ulyanov et al., Deep Image Prior https://arxiv.org/abs/1711.10925

In fact, recent applications of U-net [2] [3] rarely exceeds 5000 steps and they take advantage of its inductive bias on natural images. My main hesitation to fully agree with the results is that the model seems to operate in the "over-fitting" regime to recover the 3rd color. 

[2] Falck et al., A Multi-Resolution Framework for U-Nets with Applications to Hierarchical VAEs NeurIPS 2022

[3] Wang, et al., Learning low-dimensional generalizable natural features from retina using a U-net NeurIPS 2022

### Questions
A few additional questions: 
 
1. What is $N$ in the cortical learning model? What would happen if the model directly set $N=3$? 
 
2. Does the $NX3$ matrix (neural scope) hold constant in all the shown examples (like Figure 3.3) or is this matrix refitted every single time given the different internal percepts? 
 
3. Would the authors comment on why it takes additional 99K steps to learn the 3rd color? Would this (#steps) be different if there were only 2 colors? 
 
4. If I read the manuscript correctly, the visual percepts are the output of U-net? 
 
5. U-net (the backbone for demosaicking here) contains substantial inductive bias in its architecture. Is the separate translation operator a model merely a biologically inspired model choice? It seems that the model is trained on pairs of data (the input ONS is from time t and the output ONS is from $t+\Delta t$) instead of sequences. If U-net absorbs the learning of $\Omega$, would it change the performance? I understand eye movements shift ONS, but is it necessary to map it into a separate subfunction?

### Soundness
3

### Presentation
3

### Contribution
2
