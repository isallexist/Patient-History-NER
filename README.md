# Patient-History-NER
Transformer-based clinical language models for recognizing patient historical entities

- INPUT folder: 
  + xmi_data: original clinical text file with annotation, in .xmi format
  + bio_data: xmi data converted to BIO tags for Named Entity Recognition 

- OUTPUT folder contains the combined input and output of 4 best models (GatorTron-Base, GatorTron-Base+CLAMP, GatorTronS, GatorTronS+CLAMP) based on each sample. 
  + Each file has four columns: "Token / word", "Prediction", "Label", "Section tag"
  + "Section tag" indicates the start and the end of the section, for example 'hpi-s' indicates that the HPI section start from this word, and 'hpi-end' indicates that the HPI section end at this word.

- The source code is in src folder: 
  + nonCLAMP contains CLMs without CLAMP output 
  + CLAMP contains CLMs with CLAMP output support
- train.py and cross_evaluation.py to train and evaluate the model(s)
- The model .bin file are not uploaded because of storage consumption
