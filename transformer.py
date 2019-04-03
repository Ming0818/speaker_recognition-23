from keras_transformer.position import TransformerCoordinateEmbedding
from keras_transformer.transformer import TransformerBlock


def get_transformer(transformer_input, transformer_depth=2):
    transformer_block = TransformerBlock(
        name='transformer',
        num_heads=8,
        residual_dropout=0.1,
        attention_dropout=0.1,
        use_masking=True)
    add_coordinate_embedding = TransformerCoordinateEmbedding(
        transformer_depth,
        name='coordinate_embedding')

    output = transformer_input  # shape: (<batch size>, <sequence length>, <input size>)
    for step in range(transformer_depth):
        output = transformer_block(
            add_coordinate_embedding(output, step=step))
    return output
