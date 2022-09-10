#create Model

import torch
import torch.nn as nn
from transformers import BertModel

# Create the BertClassfier class
class BertEmotionMultiLabelClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False,base_model=None,num_classes=2):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        @param    base_model: a pretrained bert model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, num_classes

        # Instantiate BERT model
        if base_model!=None:
          print("using base model")
          self.bert=base_model
        else:
          print("Using pretrained BERT model")
          self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = torch.nn.Linear(768,num_classes)

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False



    def forward(self, input_ids, attention_mask,token_type_ids):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        _,outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,token_type_ids = token_type_ids,return_dict=False)
        
        # # Extract the last hidden state of the token `[CLS]` for classification task
        # last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(outputs)

        return logits
