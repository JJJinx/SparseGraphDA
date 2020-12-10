### CONFIGS ###
device = 'cuda'
distribution_type = 'Both' #'Norm' or 'Gumbel' or 'Both' or 'Rel'

#dataset = 'cora'
#model = 'VGAE'
input_dim = 1433 
categorical_dim = 2
n_hidden = [64      ,32      ,16        ,8              ,32] #notice that the input dimention of decoder_mlp1 =vi_mlp_out/cat=8
########### gcn1_out,gcn2_out,vi_mlp_out,decoder_mlp1_in,decoder_mlp1_out
num_epoch = 400
batch_size = 512
learning_rate = 0.01