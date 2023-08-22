# stanCode_AI_Poster
Hi there~\
This repository holds the AI project that our team done on the Poster after complete stanCode SC201 course.

# stanCode 201: AI Poster Intro
In stanCode 201 course, every students will be divided in groups to work on an AI project.\
Our team chose "Adversarial Attack and Defense" as our topic.

# Adversarial Attack
* [FGSM](https://towardsdatascience.com/adversarial-examples-to-break-deep-learning-models-e7f543833eae)(Fast Gradient Sign Method)
  * Easy and fast to do adversarial attack. --> *[Demo Code](https://github.com/Helenhjlai/stanCode_AI_Poster/blob/main/FGSM.ipynb)*
* [DeepFool](https://arxiv.org/abs/1511.04599)
  * More accurate and efficient to find perturbation. --> *[Demo Code](https://github.com/Helenhjlai/stanCode_AI_Poster/tree/main/stanCode_AI_Poster/DeepFool)*
* [JSMA](https://arxiv.org/abs/1511.07528)
  * Attack on specific pixels which are the most important feature through Jacobian Matrix. --> *[Demo Code](https://github.com/Helenhjlai/stanCode_AI_Poster/tree/main/stanCode_AI_Poster/JSMA)*

# Defense
* [Adversarial Training](https://github.com/mahyarnajibi/FreeAdversarialTraining):
  * Our insights:
    * More feasible on FGSM-like series attacks.
    * if we try on JSMA, it'll consume lots of compuation and resourses.\
      However, we try to add some adversarial examples into training data and re-train our model, it increases the model accuracy after being attacked --> *[Demo Code](https://github.com/Helenhjlai/stanCode_AI_Poster/blob/main/JSMA_adversarial_training.ipynb)*
* [Padding](https://arxiv.org/abs/1711.01991?fbclid=IwAR3iZWvLmVpR1SmsxxMHr_h0H_TcBK1P7Dfkfh45BruFndLDMViWREAu2dY)
  Based on our experiment:
  * padding one pixel: the confidence of adversarial example drops 74%
  * padding 20 pixel: the adversarial example is correctly classified with the cofidence of 38.5%
  * padding is more feasible for FGSM attack, but does not significantly defense for DeepFool or JSMA
* [GAN](https://arxiv.org/abs/1707.05474): --> this is our future work!
