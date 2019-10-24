import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer)
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features


processor = processors['SST-2'.lower()]()

label_list = processor.get_labels()
num_labels = len(label_list)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}

config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']

model = model_class.from_pretrained('/home/raghu/sentiment_bert/')
tokenizer = tokenizer_class.from_pretrained('/home/raghu/sentiment_bert/', do_lower_case=True)

model.eval()
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
def get_features(text,tokenizer,label_list=['0','1']):
    
    '''
    parms: label_list : list of labels in strings ['0','1'], default set for sentiment
           tokenzier
           examples: list of single example for inference
    
    '''
    ex = AttrDict()
    ex.update({
          "guid": "train-1",
          "label": "0",
          "text_a": text,
          "text_b": None
    })
    examples=[ex]
    features = convert_examples_to_features(examples,
                                            tokenizer,
                                            label_list=label_list,
                                            max_length=128,
                                            output_mode="classification",
                                            pad_on_left=False,                 # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=0,
            )
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    eval_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)
    return eval_dataloader
def get_predictions(batch):
    with torch.no_grad():
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[3]}
        outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]
    preds = logits.detach().cpu().numpy()
    return np.argmax(preds, axis=1)




if __name__ == "__main__":
    codes_defined = {0:'Negative',1:'Positive'}
    text = 'its very gud experience to have a ride with vogo'
    eval_dataloader=get_features(text,tokenizer)
    
    print('Prediction: ',codes_defined[[get_predictions(tuple(t for t in x)) for x in eval_dataloader][0][0]])