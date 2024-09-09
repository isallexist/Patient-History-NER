Transformer-based clinical language models for recognizing patient historical entities

- INPUT folder:
  + xmi_data: original clinical text file with annotation, in .xmi format
  + bio_data: xmi data converted to BIO-tags-format data for Named Entity Recognition
  + txt_data: original clinical text file in text format
  + merge_clamp: BIO-tags-format data which included CLAMP tags (1st CLAMP tag group and 2nd CLAMP tag group)
    
- OUTPUT folder contains the combined input and output of 4 best models (GatorTron-Base, GatorTron-Base+CLAMP, GatorTronS, GatorTronS+CLAMP) based on each sample.
  + Each file has four columns: "Token / word", "Prediction", "Label", "Section tag"
  + "Section tag" indicates the start and the end of the section, for example 'hpi-s' indicates that the HPI section start from this word, and 'hpi-end' indicates that the HPI section end at this word.

- The source code is in src folder:
  + nonCLAMP contains CLMs without CLAMP output
  + CLAMP contains CLMs with CLAMP output support
  + train.py and cross_evaluation.py to train and evaluate the model(s)
  + The model .bin file are not uploaded because of storage consumption

Medical transcription samples and reports used as our training and benchmark dataset are available on [https://www.mtsamples.com](https://www.mtsamples.com )
