from torch.utils.data import Dataset
import torch


class EmotionDataset(Dataset):
  def __init__(self,sentences,labels,tokenizer,max_lenght=128):
    super().__init__()
    self.sentences =sentences
    self.labels = labels
    self.tokenizer =tokenizer
    self.max_lenght=max_lenght
  def __len__(self):
    return len(self.sentences)
  def __getitem__(self, index):
      sentence =self.sentences[index]
      label =self.labels[index]
      return prepare_text(sentence,label,self.max_lenght,self.tokenizer)






def prepare_text(sentence,label,max_lenght,tokenizer):
  encoded_sent = tokenizer.encode_plus(
          text=sentence,  
          add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
          max_length=max_lenght,          # Max length to truncate/pad
          pad_to_max_length=True,         # Pad sentence to max length
          return_attention_mask=True,      # Return attention mask
          truncation=True,                  # explicitly truncate examples to max length
          return_token_type_ids=True
          )
      
  # Add the outputs to the lists
  input_ids=encoded_sent.get('input_ids')
  attention_masks=encoded_sent.get('attention_mask')
  token_type_ids = encoded_sent.get("token_type_ids")

  # Convert lists to tensors
  input_ids = torch.tensor(input_ids)
  attention_masks = torch.tensor(attention_masks)
  label =torch.tensor(label,dtype=torch.float)
  token_type_ids=torch.tensor(token_type_ids, dtype=torch.long)

  return {'input':input_ids, "attention":attention_masks,'label':label,'token_type_ids':token_type_ids}