# DILabHelmholtz
In this project we build a topology regularized foundation model for medical image segmentation.
For this we fine-tune the segment anything model (SAM) ADD REFERENCE on a private OCT dataset.
This dataset has 552 images. Each image comes with a corresponding ground truth segmentation which consists of 14 segmentation classes. A raw image with corresponding ground truth segmentation can be seen in figure UPLOAD IMAGES.

For fine-tuning SAM we largely follow the idea of MedSAM ADD REFERENCE. We first preprocess the dataset via octsam/data/preprocessing. Then we retrain SAM's mask decoder in models/training. Here we give multiple options to configure training paramters via command line arguments. The most important options are:
DO ITEMIZE
- image encoder size
- pseudocoloring
- topological loss
- prompt choice
WRITE MORE
After training the final model is saved to the specified model directory and the evaluation results on the test set are printed in the terminal.


