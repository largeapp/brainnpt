import argparse

class Flags:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.use_data_args()
        self.adjust_base_args()
        self.adjust_hyper_args()

    def get_parser(self):
        return self.parser.parse_args()

    def use_data_args(self):
        self.parser.add_argument("--dataset_name", type=str, default="all", help="the name of dataset")
        self.parser.add_argument("--output", type=str, default="./model/pretrain", help="save the checkpoints and logs")
        self.parser.add_argument("--resume_file", type=str, default="", help="resume the checkpoint")
        self.parser.add_argument("--num_classes", type=int, default=2, help="Number for classification")
        self.parser.add_argument("--d_model", type=int, default=200, help="Dimension of node feature")
        self.parser.add_argument("--nhead", type=int, default=5, help="Number for self-attention heads")
        self.parser.add_argument("--graph_layer", type=int, default=6, help="Number for graph transformer layers")
        self.parser.add_argument("--seq_layer", type=int, default=6, help="Number for sequence transformer layers")
    def adjust_base_args(self):
        self.parser.add_argument("--task", type=str, default="") # gen、cls
        self.parser.add_argument("--seed", type=int, default=42)
        self.parser.add_argument("--num_workers", type=int, default=2)
        self.parser.add_argument("--running_type", type=list, default=["train", "dev", "test"])
        self.parser.add_argument("--finetune", type=bool, default=True)
        self.parser.add_argument("--exp_name", type=str, default='pretrain_mlm')

    def adjust_hyper_args(self):
        # model hyper parameters
        self.parser.add_argument("--vocab_size", type=int, default=1)
        self.parser.add_argument("--special_tokens", type=list, default=["测试"])
        self.parser.add_argument("--max_input_size", type=int, default=192)
        self.parser.add_argument("--max_output_size", type=int, default=175)
        self.parser.add_argument("--num_hidden_layers", type=int, default=4)
        # training hyper parameters
        self.parser.add_argument("--lr", type=float, default=1)
        self.parser.add_argument("--lr_scale", type=float, default=0.01)
        self.parser.add_argument("--dropout_prob", type=float, default=0.1)
        self.parser.add_argument("--weight_decay", type=float, default=2e-4)
        self.parser.add_argument("--accum_iter", type=int, default=2)
        self.parser.add_argument("--epochs", type=int, default=100)
        self.parser.add_argument("--train_bs", type=int, default=64)
        self.parser.add_argument("--dev_bs", type=int, default=64)
        self.parser.add_argument("--copy_source", type=str, default="content")  # title、passage       
        self.parser.add_argument("--share_tokenizer", type=bool, default=True)
        self.parser.add_argument("--share_word_embedding", type=bool, default=True)
        self.parser.add_argument("--using_RL", type=bool, default=False)    
        self.parser.add_argument("--mlm_probability", type=float, default=0.5)    
        self.parser.add_argument("--shuffle_prob", type=float, default=0.5)   
flags = Flags()
args_pretrain = flags.get_parser()
