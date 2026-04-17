# FieryGS: In-the-Wild Fire Synthesis with Physics-Integrated Gaussian Splatting

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6

## Abstract
We consider the problem of synthesizing photorealistic, physically plausible combustion effects in in-the-wild 3D scenes. Traditional CFD and graphics pipelines can produce realistic fire effects but rely on handcrafted geometry, expert-tuned parameters, and labor-intensive workflows, limiting their scalability to the real world. Recent scene modeling advances like 3D Gaussian Splatting (3DGS) enable high-fidelity real-world scene reconstruction, yet lack physical grounding for combustion. To bridge this gap, we propose FieryGS, a physically-based framework that integrates physically-accurate and user-controllable combustion simulation and rendering within the 3DGS pipeline, enabling realistic fire synthesis for real scenes. Our approach tightly couples three key modules: (1) multimodal large-language-model-based physical material reasoning, (2) efficient volumetric combustion simulation, and (3) a unified renderer for fire and 3DGS. By unifying reconstruction, physical reasoning, simulation, and rendering, FieryGS removes manual tuning and automatically generates realistic, controllable fire dynamics consistent with scene geometry and materials. Our framework supports complex combustion phenomena—including flame propagation, smoke dispersion, and surface carbonization—with precise user control over fire intensity, airflow, ignition location and other combustion parameters. Evaluated on diverse indoor and outdoor scenes, FieryGS outperforms all comparative baselines in visual realism, physical fidelity, and controllability.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
FieryGS synthesizes realistic, physically grounded fire effects in real 3D scenes by integrating combustion simulation and rendering into the 3D Gaussian Splatting pipeline. It combines scene reconstruction, material reasoning via multimodal LLMs, efficient volumetric combustion simulation,, and unified rendering. The effectiveness of FieryGS is demonstrated by detailed experiments, comprehensive user studies, and visualizations.

### Strengths
The proposed method offers a novel, user-friendly, and highly efficient pipeline for realistic fire rendering in complex scenes. By integrating MLLM–based material reasoning with 3DGS–based segmentation, the approach effectively reduces the need for manual parameter tuning and eliminates dependence on traditional expert-level simulation tools. This design democratizes physically-plausible combustion synthesis, allowing users to achieve compelling visual results with minimal technical overhead. The paper is clearly written and well-structured. Extensive quantitative experiments and qualitative visual analyses robustly support the claims.

### Weaknesses
I do not have substantial complaints about this paper. I move my concerns to "Questions".

### Questions
### Main Question:
- [Comparison to expert-level tools] Since the physical properties can be inferred by MLLMs and the geometry can be modeled by 3DGS, I wonder if all the input required by VFX/CFD (e.g. meshes, material) can also be provided by MLLMs (Line 237). If so, how is the quality and cost comparison between the proposed combustion simulations and expert-level tools? This comparison is more apples-to-apples and can help the readers understand how much the simplified combustion simulation affects the performance.

### Minor Question
- [MLLM comparisons] Is GPT-4o necessary for high-performance? Comparisons with other frontier MLLMs in Table 2/5 and Figure 6 might help readers understand more about how useful the language instructions are.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper propose a method to add fire simulation in 3DGS scenes. It first use MLLM to segment flamable objects and then add fire and charring simulation, with good user controllability, including fire intensity, air flow, and color of the fire based on the material.. The final rendering is a combination of flame simulation with phone shading + 3DGS. Experiments show that the proposed method has better visual quality comparing to exisitng baselines. It also proposes an extension with existing generative models to improve photorealism.

### Strengths
1. The paper is well-written and easy to understand.
2. The method of using a combination of MLLM and fire/charring simulation is reasonable and easy to implement.
3. The method provides good controllability of the scene setup, esp in controlling the fire color based on the material.

### Weaknesses
1. The rendering is one-way. It does not consider the shading of fire on surrounding. Thus causing visual artifacts. Though the generative model may solves the problem in some cases, it may alters the background.
2. The fire simulation does not consider geometry constraint. For example, fire in a building seeing through windows. The simulation of fire / smoke should consider geometry constraints induced by voxels that are "solid".

### Questions
1. Is it possible to improve the quality of rendering and fire simulation for the weakness points?
2. What are the aesthetic qualtiy metric? How it is computed?
3. Could u show the visualization of the fire in different control parameters?

### Soundness
2

### Presentation
3

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
The paper present FireyGS, which is a framework that physically simulates fires and charring of objects in real-world images. The process is fully automated with some user control. 

This is achieved through:
1) making volumetric representation of the real-world scene using 3D Gaussian Splatting, 
2) letting a multi-modal LLM reason about the material characteristics of objects, such as whether they are consumable by fire, 
3) physically simulate fire and smoke using the Navier-Stokes equations while also modelling charring, and 
4) rendering the fire using Phong Illumination and an optional generative model. 

This allows to easily integrate fires into real-world scenes with some user control, such as adjusting fire intensity and airflow, which is quite impressive. The rendering time is very short. The evaluation shows that state-of-the-art methods are far from achieving this.

### Strengths
* The paper is well written and structured well with many images illustrating the capabilities of the method. 
* The results are quite impressive. The paper illustrates how fire can be put into realistic images with little effort. 

The fact that one “easily” can add simulated fires to real-world images is quite astonishing to me. I am very impressed by the results.

### Weaknesses
* The novelty is limited to using an LLM for reasoning about material characteristics and putting well known methods together. If I get this right, there is nothing new about the 3D Gaussian Splatting method, the fire and charring simulation, nor the illumination method. 
* While the results look good, they are far from realistic, especially for larger, more turbulent fires. This can easily be seen by comparing Figure 2 (real-world, live fire) and the images generated by the framework. The smoke and turbulence in the real-world images makes for deeper and visually more “exciting” flames. This is probably because the low resolution of the fire, also the turbulence is not modelled well by the simplified Navier-Stokes. This method works fairly well for small fires with little turbulence – in my opinion. The graphics community, however, have many methods and tricks that would help here, such as dynamic meshes that allows detail to be increased in the areas where there is turbulence. Other more complex and realistic physical simulations have been proposed as well. This would if course hamper computational performance, which is really good for the method presented here. The good thing though is that there are many ways to improve it based on related work in physics-based animation of fire. 
* Some parts could be better explained. It is not clear to me what exactly is fed to GPT4o to reason about material characteristics. I would assume the real-world images, but it is not clear to me from the text.  

On one hand, while the results are impressive, I wonder if the novelty falls short. This paper represents a fairly small step in an evolution, not a revolution. On the other hand, one could easily see that the significance of this work could be high, as it shows how fire could be simulated in real-world scenes. 

While the rendering of the fire leaves much to be wanted (think about the vivid fires and explosions in special effects such the opening scene of Star Wars III: Revenge of the Sith – from 2005 where explosions also are simulated using physics), this does not affect my recommendation much, as I expect this to be improved quickly by future work. 

Because of this I lean towards acceptance.

### Questions
* On line 126, it is implied that the method presented in the paper “maintains physical accuracy”. Is maintaining physical accuracy a good term? The reason I ask is that I would assume that this would be evaluated in some way. It is not, and it is not easy to do either as far as I am concerned. 
* The charring effect is very nice. Why would you not let the fire consume the fuel (material) in each voxel? Have you tried? In theory, this should be straight forward. Are there issues with the visualization?
* It is stated on line 215: “Focusing on efficiency, our method simplifies processes …”. Could you please be explicit about which processes? I would like to understand exactly what you mean. 
* Figure 5f: While the generative refinement improves illumination shadows, it seems to me that the flames get bleaker with less details. Is this true and a problem in general, or is it just this image? What is going on here - if it is the case? Please help me understand.
* Line 409: “A detailed timing breakdown and comparisons with baselines are provided …”. It would be nice to see the running times for the baselines as well.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work presents a system named “Physics-Integrated Gaussian Splatting,” aiming to synthesize 
controllable and physically inspired fire and combustion effects in real-world 3D scenes.  The method integrates several coordinated modules to form a complete pipeline from real-scene  reconstruction to fire simulation:
 1) a multimodal large model predicts material types and combustion-related parameters in 2D projection 
space and back-project them to 3D Gaussian
 2) the 3D scene is voxelized, where solid and air regions are distinguished by Gaussian density—simplified
 fire simulations are applied to air voxels, while thermal diffusion and charring are modeled for solids, with 
intuitive user controls such as ignition position and airflow
 3) a unified rendering framework generates multiple visual effects, including flames, smoke, charring, and 
indirect illumination.
 
The work does not introduce a new rendering theory or physical model but integrates existing techniques
 3D Gaussian Splatting, simplified fluid simulation, and multimodal reasoning—into a complete interactive
 pipeline, producing visually realistic fire results.

 Experiments on multiple real and synthetic scenes demonstrate visually realistic and controllable fire 
generation, and the results appear to outperform existing generation-based approaches

### Strengths
This paper presents an integrated system combining 3D Gaussian Splatting, simplified physical simulation, and multimodal reasoning to construct a physically-informed and controllable fire generation pipeline. 

This integration and simplification provide a certain degree of novelty in application, making complex fire simulation more practical and easier to operate. The system is well-designed, with modules for material prediction, fire simulation, and rendering fire results. Experiments on multiple real and synthetic scenes demonstrate that the generated fire is visually plausible and controllable, outperforming existing generation-based methods. The user interaction design further enhances the system’s operability. 

The paper provides clear and understandable descriptions of the system modules  and workflow, and the authors indicate that the source code will be released in the future, which facilitates follow-up 
research. Overall, this work offers a complete and practical solution for interactive fire generation, with utility for computer graphics, visual effects, and virtual environment fire simulation

### Weaknesses
While the integration offers practical value, the method has limited theoretical and technical novelty.  Moreover, the approach heavily depends on the MLLM’s 2D material inference capability, and its performance on uncommon materials, composite materials, or extreme fire conditions remains unexplored. 
Additionally, the paper does not provide comparisons between the simplified physical simulation and a full 
physics-based simulation, making it difficult to justify the acceptability of the simplifications in practice. Finally
 While FieryGS accounts for multiple effects of fire combustion, it does not explicitly capture or evaluate the 
dynamic lighting effects generated by fire, which contribute to the perceived motion and liveliness of the 
scen

### Questions
- How does the system perform on uncommon materials, composite materials, or objects with unusual textures?

- Could the authors clarify or quantify how much these physical simplifications affect the realism of the generated fire?

### Soundness
3

### Presentation
3

### Contribution
2
