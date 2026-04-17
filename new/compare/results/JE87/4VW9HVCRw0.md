# Review

## Summary
This paper presents a novel task of generating controllable, diverse, and physically plausible hand-object interactions (HOIs) beyond simple grasping, including non-grasping actions like pushing, poking, and rotating. To support this task, the authors introduce WildO2, the first large-scale, in-the-wild 3D HOI dataset, containing 4.4k unique interactions across 92 intents and 610 object categories with detailed semantic annotations. They also propose TOUCH, a three-stage framework centered on a multi-level diffusion model that facilitates fine-grained semantic control to generate versatile hand poses beyond grasping priors. This process leverages explicit contact modeling for conditioning and is subsequently refined with contact consistency and physical constraints to ensure realism. Extensive experiments demonstrate the method's ability to generate controllable, diverse, and physically plausible hand interactions representative of daily activities.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper introduces a new task of generating controllable, diverse, and physically plausible hand-object interactions (HOIs) beyond simple grasping, which is a significant contribution to the field.
2. The authors present a novel dataset, WildO2, which is the first large-scale, in-the-wild 3D HOI dataset, containing a rich set of interactions with detailed semantic annotations. This dataset is a valuable resource for further research in this area.
3. The proposed framework, TOUCH, utilizes a multi-level diffusion model that facilitates fine-grained semantic control, enabling the generation of versatile hand poses beyond grasping priors. This approach is innovative and demonstrates the potential for generating realistic and diverse HOIs.

## Weaknesses
1. The dataset is collected from the Something-Something V2 dataset, which primarily features single-object manipulation. It would be beneficial to include interactions with multiple objects to enhance the dataset's diversity and complexity.
2. The dataset lacks video data, which limits the ability to capture temporal information and dynamics of interactions. This could be important for certain types of analysis or model training.
3. The dataset does not include 6-DoF object pose estimation, which is necessary for fully capturing the 3D position and orientation of objects during interactions. This could limit the accuracy and realism of generated interactions.
4. The dataset could benefit from more diverse objects, including those with complex shapes and textures. This would improve the generalizability and robustness of models trained on this dataset.
5. The dataset lacks annotations for contact forces, which are crucial for understanding the dynamics of interactions and could improve the accuracy of models in predicting interaction stability.

## Questions
1. Can the authors provide more details on the diversity of the objects used in the dataset? Are there a variety of object types, shapes, and textures represented?
2. How does the proposed framework, TOUCH, handle the generation of interactions involving multiple objects? Is this a limitation of the current approach?
3. Can the authors provide more information on the contact force annotations in the dataset? Are there plans to include these annotations in future versions of the dataset?
4. How well does the proposed framework generalize to interactions involving objects with complex shapes and textures? Can the authors provide examples or evaluations on this aspect?
5. How does the proposed framework perform in generating interactions for objects with different levels of physical properties, such as weight, size, and material? Can the authors provide evaluations or examples on this aspect?
6. How does the proposed framework handle the generation of interactions that require fine-grained control, such as precise placement or alignment of objects? Can the authors provide examples or evaluations on this aspect?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
8

## Confidence
4