"""TRUE per-example gradient masking with full-batch CE normalization.

Core formatter preserves only `soft_labels`; its collator drops other custom
fields. Encode a mask as a THIRD, nonpositive component in each training row's
`soft_label`: [weak_prob_0, weak_prob_1, -mask], mask in {0,1}. The formatter
still uses the first two valid probabilities to choose the hard class token
because max(weak_prob_0,weak_prob_1)>=.5 > -mask. The custom trainer ignores
that third component as a class target and uses it ONLY as a mask.

Every example remains in the dataloader once per epoch, with equal optimizer
steps and no selected-example oversampling. The objective is
sum_i mask_i * CE(hard_teacher_i, student_i) / ORIGINAL batch size;
no EOS or off-class vocabulary CE is trained. An all-zero minibatch yields
exact zero class-loss gradient, not a forced nonempty substitute.
"""
import torch
from transformers import Trainer


class MaskedClassOnlyTrainer(Trainer):
    def __init__(self,*args,label_token_ids=None,**kwargs):
        super().__init__(*args,**kwargs)
        if label_token_ids is None or len(label_token_ids)!=2 or len(set(label_token_ids))!=2:
            raise ValueError('Two distinct single-token class IDs required')
        self.label_token_ids=list(label_token_ids)

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        targets=inputs['soft_labels'].float()
        labels=inputs['labels']
        prompt=inputs['prompt_length'].long()
        if targets.ndim!=2 or targets.shape!=(len(prompt),3):
            raise ValueError('soft_labels must be [N,3]: p0,p1,-binary_mask')
        p=targets[:,:2]
        mask=-targets[:,2]
        if ((p<0).any() or (p>1).any() or
                not torch.allclose(p.sum(1),torch.ones_like(mask),atol=1e-4)
                or not torch.isfinite(targets).all() or
                not ((mask==0)|(mask==1)).all()):
            raise ValueError('Invalid weak probability pair or binary mask')
        row=torch.arange(len(prompt),device=labels.device)
        if ((prompt<1)|(prompt>=labels.shape[1])).any():
            raise ValueError('Invalid first class-token position')
        target_tokens=labels[row,prompt]
        valid=torch.isin(target_tokens,
                         torch.tensor(self.label_token_ids,device=labels.device))
        if not valid.all():
            raise ValueError('Class target must be one of the two label tokens')
        hard=(target_tokens==self.label_token_ids[1]).long()
        outputs=model(input_ids=inputs['input_ids'],
                      attention_mask=inputs['attention_mask'])
        class_logits=outputs.logits[row,prompt-1][:,self.label_token_ids].float()
        per_row=torch.nn.functional.cross_entropy(class_logits,hard,reduction='none')
        loss=(mask*per_row).sum()/len(mask)
        return (loss,outputs) if return_outputs else loss


if __name__=='__main__':
    # This file contains no dataset/model execution by default.
    print('Use with train_model trainer_class=MaskedClassOnlyTrainer; no run performed.')
