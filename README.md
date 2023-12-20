This repo is a robust implementation of PaDiM with extra registration.

# About the registration
1. There is an registration model created base on DMAD training skill and architecture.
2. Images will be aligned before perform PaDiM inference.
3. But the training data is limited to my own field. For more generalization capacity, you may want to re-train this model on your own data.

# About the inference
1. **Ref** here means a reference that all test images will be registrated to it, and then perform PaDiM inference.'
2. **Anchor** here means *absolute* normal and *diverse* images to generate better static $\mu$ and $\Sigma$.
3. Once these static $\mu$ and $\Sigma$ are generated, they will be saved as a *npz* file.
4. If you have no ideas about what these means, just run *"anormal_det_video.py"* to check how these auto-saved images are look like.
5. The difference between *"anormal_det_video.py"* and *"anormal_det_no_hist.py"* is mainly about the update of *running_mean* and *running_conv*. 
The *"video"* version fuses static variable and the on-line-update variable, and the *"no_hist"* version is do not fuse anything.
