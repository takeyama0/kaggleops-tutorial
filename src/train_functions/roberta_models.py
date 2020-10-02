import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig, XLMRobertaModel, XLMRobertaConfig
from transformers.modeling_auto import AutoConfig, AutoModelForQuestionAnswering



def build_model(config):
    if   config['MODEL_TYPE']=='RoBERTaBaseSquad2':
        model = RoBERTaBaseSquad2(config)

    elif config['MODEL_TYPE']=='RoBERTaLargeSquad2':
        model = RoBERTaLargeSquad2(config)

    elif config['MODEL_TYPE']=='RoBERTaBaseSquad2MDO':
        model = RoBERTaBaseSquad2MDO(config)

    elif config['MODEL_TYPE']=='RoBERTaLargeSquad2MDO':
        model = RoBERTaLargeSquad2MDO(config)


    return model


class RoBERTaBaseSquad2(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.CONFIG = CONFIG
        self.modelpath = "deepset/roberta-base-squad2"
        
        self.robertaconfig = AutoConfig.from_pretrained(self.modelpath)
        self.robertaconfig.output_hidden_states = True

        self.roberta = AutoModelForQuestionAnswering.from_pretrained(
            self.modelpath, 
            config=self.robertaconfig,
            )._modules['roberta']
            
        self.dropout = nn.Dropout(p=self.CONFIG['DROPOUT'])
        self.l1 = AutoModelForQuestionAnswering.from_pretrained(
            self.modelpath, 
            config=self.robertaconfig,
            )._modules['qa_outputs']
    
    def forward(self, input_ids, attention_mask, token_type_ids, mask_out):
        ## start/end-spanがターゲットの時はマスクを2つに複製する
        if self.CONFIG['QA_TASK']:
            mask_out = mask_out.unsqueeze(-1).repeat(1,1,2)
            
        ## BERTの出力を取得
        output = self.roberta(
            input_ids      = input_ids, 
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )

        ## 13+14レイヤーをconcatとして利用する場合
        if self.CONFIG['CAT_HIDDEN']: # Ex.) self.CONFIG['CAT_HIDDEN'] = [-1, -2, -3]
            output = torch.cat(tuple(output[2][i] for i in self.CONFIG['CAT_HIDDEN']), dim=-1)
        else:
            output = output[0]

        ## Dropout - Linear - Mask
        output = self.dropout(output)
        output = self.l1(output).squeeze(-1)
        # output = output * mask_out
        output[mask_out==0] = -1e10

        return output


class RoBERTaLargeSquad2(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.CONFIG = CONFIG
        self.modelpath = "ahotrod/roberta_large_squad2"
        
        self.robertaconfig = AutoConfig.from_pretrained(self.modelpath)
        self.robertaconfig.output_hidden_states = True

        self.roberta = AutoModelForQuestionAnswering.from_pretrained(
            self.modelpath, 
            config=self.robertaconfig,
            )._modules['roberta']
            
        self.dropout = nn.Dropout(p=self.CONFIG['DROPOUT'])
        self.l1 = AutoModelForQuestionAnswering.from_pretrained(
            self.modelpath, 
            config=self.robertaconfig,
            )._modules['qa_outputs']
    
    def forward(self, input_ids, attention_mask, token_type_ids, mask_out):
        ## start/end-spanがターゲットの時はマスクを2つに複製する
        if self.CONFIG['QA_TASK']:
            mask_out = mask_out.unsqueeze(-1).repeat(1,1,2)
            
        ## BERTの出力を取得
        output = self.roberta(
            input_ids      = input_ids, 
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )

        ## 13+14レイヤーをconcatとして利用する場合
        if self.CONFIG['CAT_HIDDEN']: # Ex.) self.CONFIG['CAT_HIDDEN'] = [-1, -2, -3]
            output = torch.cat(tuple(output[2][i] for i in self.CONFIG['CAT_HIDDEN']), dim=-1)
        else:
            output = output[0]

        ## Dropout - Linear - Mask
        output = self.dropout(output)
        output = self.l1(output).squeeze(-1)
        # output = output * mask_out
        output[mask_out==0] = -1e10

        return output


class RoBERTaBaseSquad2MDO(nn.Module):
    """
    QA_TASK and ADD_TOKEN_LOSS only
    """
    def __init__(self, CONFIG):
        super().__init__()
        self.CONFIG = CONFIG
        self.modelpath = "deepset/roberta-base-squad2"
        
        self.robertaconfig = AutoConfig.from_pretrained(self.modelpath)
        self.robertaconfig.output_hidden_states = True

        self.roberta = AutoModelForQuestionAnswering.from_pretrained(
            self.modelpath, 
            config=self.robertaconfig,
            )._modules['roberta']
        
        self.drop_out = nn.Dropout(p=self.CONFIG['DROPOUT'])
        self.high_dropout = nn.Dropout(p=self.CONFIG['HIGH_DROPOUT'])
        
        n_weights = self.robertaconfig.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:self.CONFIG['MAIN_LAYERS']] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)
        
        if self.CONFIG['QA_TASK']:
            if not self.CONFIG['ADD_TOKEN_LOSS']:
                self.classifier = nn.Linear(self.robertaconfig.hidden_size, 2)
            else:
                self.classifier = nn.Linear(self.robertaconfig.hidden_size, 3)
    
    def forward(self, input_ids, attention_mask, token_type_ids, mask_out):
        if self.CONFIG['QA_TASK']:
            if not self.CONFIG['ADD_TOKEN_LOSS']:
                mask_out = mask_out.unsqueeze(-1).repeat(1,1,2)
            else:
                mask_out = mask_out.unsqueeze(-1).repeat(1,1,3)
        
        ## BERTの出力を取得
        output = self.roberta(
            input_ids      = input_ids, 
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        
        hidden_layers = output[2]

        cls_outputs = torch.stack(
            [self.drop_out(layer[:, :, :]) for layer in hidden_layers], dim=3
        )

        cls_output = (torch.softmax(self.layer_weights, dim=0) * cls_outputs).sum(-1)

        
        logits = torch.mean(
            torch.stack(
                [self.classifier(self.high_dropout(cls_output)) for _ in range(self.CONFIG['N_MDO_SAMPLE'])],
                dim=0,
            ),
            dim=0,
        )

        logits = logits.squeeze(-1)
        # output = output * mask_out
        logits[mask_out==0] = -1e10
        
        return logits


class RoBERTaLargeSquad2MDO(nn.Module):
    """
    QA_TASK and ADD_TOKEN_LOSS only
    """
    def __init__(self, CONFIG):
        super().__init__()
        self.CONFIG = CONFIG
        self.modelpath = "ahotrod/roberta_large_squad2"
        
        self.robertaconfig = AutoConfig.from_pretrained(self.modelpath)
        self.robertaconfig.output_hidden_states = True

        self.roberta = AutoModelForQuestionAnswering.from_pretrained(
            self.modelpath, 
            config=self.robertaconfig,
            )._modules['roberta']
        
        self.drop_out = nn.Dropout(p=self.CONFIG['DROPOUT'])
        self.high_dropout = nn.Dropout(p=self.CONFIG['HIGH_DROPOUT'])
        
        n_weights = self.robertaconfig.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:self.CONFIG['MAIN_LAYERS']] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)
        
        if self.CONFIG['QA_TASK']:
            if not self.CONFIG['ADD_TOKEN_LOSS']:
                self.classifier = nn.Linear(self.robertaconfig.hidden_size, 2)
            else:
                self.classifier = nn.Linear(self.robertaconfig.hidden_size, 3)
    
    def forward(self, input_ids, attention_mask, token_type_ids, mask_out):
        if self.CONFIG['QA_TASK']:
            if not self.CONFIG['ADD_TOKEN_LOSS']:
                mask_out = mask_out.unsqueeze(-1).repeat(1,1,2)
            else:
                mask_out = mask_out.unsqueeze(-1).repeat(1,1,3)
        
        ## BERTの出力を取得
        output = self.roberta(
            input_ids      = input_ids, 
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        
        hidden_layers = output[2]

        cls_outputs = torch.stack(
            [self.drop_out(layer[:, :, :]) for layer in hidden_layers], dim=3
        )

        cls_output = (torch.softmax(self.layer_weights, dim=0) * cls_outputs).sum(-1)

        
        logits = torch.mean(
            torch.stack(
                [self.classifier(self.high_dropout(cls_output)) for _ in range(self.CONFIG['N_MDO_SAMPLE'])],
                dim=0,
            ),
            dim=0,
        )

        logits = logits.squeeze(-1)
        # output = output * mask_out
        logits[mask_out==0] = -1e10
        
        return logits