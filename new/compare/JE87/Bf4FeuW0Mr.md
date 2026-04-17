# Review

## Summary
This paper proposes a method for learning a dexterous grasping policy that can generalize to unseen objects. The method first collects a single successful demonstration of grasping a specific object. Then, the robot action in this demonstration is edited to adapt to novel objects and poses. Finally, reinforcement learning is used to optimize a universal policy across hundreds of objects in parallel in simulation. The method is evaluated in simulation and on real-world testbeds.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
- The method is simple and intuitive. 
- The method is evaluated on a large number of objects in simulation and on real-world testbeds, and shows good results.

## Weaknesses
- The grasping policy is only trained on successful demonstrations. If the demonstrations are not good, the learned policy may not be able to correct the mistakes made in the demonstrations, e.g., if the demonstrations grasp the object too far or too close, the learned policy may not be able to correct this. 
- The success metric used in this paper is to lift the object 10 cm above its original position. This is a relatively low bar and it does not require grasping the object firmly. As a result, the grasping policy may be able to just lift the object 10 cm and then stop, without fully grasping the object. 
- It seems that the grasping policy does not have the ability to re-grasp the object if the initial grasp attempt fails.

## Questions
- How does the method handle objects that are too small or too thin? Does the method require the object to be within a certain size range for the grasping attempt to be successful? 
- How does the method handle obstacles on the table, e.g., if there is a small block in front of the object to be grasped, can the grasping policy re-grasp the object from a different direction?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4