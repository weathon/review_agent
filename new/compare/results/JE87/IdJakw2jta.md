# Review

## Summary
This paper addresses the limitations of current spatio-temporal video grounding (STVG) models that can only handle short video clips. The authors propose a novel framework called ART-STVG (AutoRegressive Transformer for STVG) designed to handle long-form videos. The key innovations include a memory-augmented autoregressive transformer that processes video frames sequentially, memory selection strategies to focus on relevant spatio-temporal context, and a cascaded spatio-temporal decoder design that leverages fine-grained spatial information to assist with temporal localization. The paper also introduces new datasets for long-form STVG. Experimental results show that ART-STVG significantly outperforms existing methods on long videos while maintaining competitive performance on short videos.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
1. The paper introduces a novel approach to handling long-form videos, which is a significant advancement over existing models that are limited to short clips.
2. The proposed ART-STVG framework is well-designed, with innovative components such as memory banks and the cascaded spatio-temporal decoder.
3. The paper provides extensive experimental results demonstrating the effectiveness of the proposed method on both long-form and short-form datasets.

## Weaknesses
1. The paper could provide more detailed analysis of the computational efficiency and scalability of the proposed method, especially when dealing with very long videos.
2. The paper could benefit from a more thorough comparison with existing methods, particularly in terms of model size and inference time.
3. The paper could provide more insights into the limitations of the proposed method and potential directions for future research.

## Questions
1. How does the computational complexity of ART-STVG scale with increasing video length compared to existing methods?
2. Can the proposed method handle videos with multiple simultaneous events or activities?
3. How robust is the method to variations in video quality, such as different resolutions or frame rates?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4