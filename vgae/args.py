### CONFIGS ###
device = 'cpu'
distribution_type = 'Both' #'Norm' or 'Gumbel' or 'Both' or 'Rel'

#dataset = 'cora'
#model = 'VGAE'
input_dim = 1433 
categorical_dim = 2
n_hidden = [256      ,128      ,64        ,32             ,128] #notice that the input dimention of decoder_mlp1 =vi_mlp_out/cat=8
########### gcn1_out,gcn2_out,vi_mlp_out,decoder_mlp1_in,decoder_mlp1_out
num_epoch = 200
batch_size = 1024
learning_rate = 0.001