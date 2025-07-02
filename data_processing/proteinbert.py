from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

pretrained_model_generator, input_encoder = load_pretrained_model()
model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(1024))
#### example usage:
encoded_x = input_encoder.encode_X(seqs, seq_len)
local_representations, global_representations = model.predict(encoded_x, batch_size=batch_size)
# ... use these as features for other tasks, based on local_representations, global_representations