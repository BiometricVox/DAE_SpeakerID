# A DENOISING AUTOENCODER FOR SPEAKER IDENTIFICATION

This is a Python implementation of the Denoising Autoencoder approach that we proposed for the first Multi-target speaker detection and identification Challenge Evaluation (MCE 2018, [http://www.mce2018.org](http://www.mce2018.org) ).

The basic idea is to train a Denoising Autoencoder to map each individual input ivector to the mean of all ivectors from that speaker. The aim of this DAE is to compensate for inter-session variability and increase the discriminative power of the ivectors.

You can find our system description for the MCE 2018 challenge [here](http://mce.csail.mit.edu/pdfs/BiometricVox_description.pdf).

## ABOUT THE MCE 2018 CHALLENGE

The task for the MCE 2018 Evaluation was to detect if a given speech segment belongs to any of the speakers in a blacklist. The challenge is divided into two related subtasks: Top-S detection, i.e. detecting if the segment belongs to any of the blacklist speakers; and Top-1 detection, i.e. detecting which specific blacklist speaker (if any) is speaking in the segment. The data was generated from real call center user-agent telephone conversations. Instead of raw audio data, organizers processed the original data and provided 600-dimensional ivectors. This way, no special signal processing knowledge was needed to enter the evaluation. More details can be found on the [evaluation plan](https://arxiv.org/abs/1807.06663).

## DATASET

The dataset can be found at:

[https://www.kaggle.com/kagglesre/blacklist-speakers-dataset](https://www.kaggle.com/kagglesre/blacklist-speakers-dataset)

After download, extract the files to _data_ folder.

## SYSTEM TRAINING

Our training script shows how a very simple DAE can bring a very nice improvement over the [baseline](https://github.com/swshon/multi-speakerID). If you run

```
python mce2018_dae_tst.py
```

you should get results like these:

```
Dev set score using train set :
Top S detector EER is 2.40%
Top 1 detector EER is 9.50% (Total confusion error is 343)

Test set score using train set:
Top S detector EER is 6.83%
Top 1 detector EER is 12.42% (Total confusion error is 411)

Test set score using train + dev set:
Top S detector EER is 5.69%
Top 1 detector EER is 8.90% (Total confusion error is 257)
```

Note that these results do not match our official submission to the challenge, were we obtained Top-S EER: 4.33%, Top-1 EER: 6.11%, since our final system was a bit more complex including Probabilistic Linear Discriminant Analysis (PLDA) scoring and Symmetric Normalization (S-Norm).


