from asyncio.log import logger
import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder,TransformerEncoderLayer
from TransformerEncoderFinalLayer import TransformerEncoderFinalLayer
class GraphEncoder(torch.nn.Module):
    def __init__(self, args):
        super(GraphEncoder,self).__init__()
        encoder_layer = TransformerEncoderLayer(
                    args.d_model, args.nhead, args.d_model*4,dropout=0.1,activation='gelu',
                    norm_first=True)
        self.final_layer = TransformerEncoderFinalLayer(
                    args.d_model, args.nhead, args.d_model*4,dropout=0.1,activation='gelu',
                    norm_first=True)
        self.encoder = TransformerEncoder(encoder_layer,args.graph_layer-1)
    def forward(self,x, need_weight = False):
        x = self.encoder(x)
        output,weight = self.final_layer(x)
        if need_weight:
            return output,weight
        else:
            return output

class SeqEncoder(torch.nn.Module):
    def __init__(self, args):
        super(SeqEncoder,self).__init__()
        encoder_layer = TransformerEncoderLayer(
                    args.d_model, args.nhead, args.d_model*4,dropout=0.1,activation='gelu',
                    norm_first=True)
        self.encoder = TransformerEncoder(encoder_layer,args.seq_layer)
    def forward(self,src, src_key_padding_mask):
        x = self.encoder(src=src,src_key_padding_mask=src_key_padding_mask)
        return x