from dataset import DataSet
from model import load_model
from feature_transform import get_vector, distance, mean_vectors


def model_simple_test(model_path, file_path, output_shape, sample_rate, process_class):
    num_of_voice_to_be_anchor = 3
    model = load_model(model_path)
    dataset = DataSet(file_dir=file_path, output_shape=output_shape, sample_rate=sample_rate)
    x, y = dataset.get_train_data(process_class=process_class)

    anchor = []
    done = 0
    index = 0
    while done < num_of_voice_to_be_anchor:
        if y[index] == 0:
            anchor.append(x.pop(index))
            y.pop(index)
            done += 1
        index+=1
    anchor_vectors = get_vector(anchor, model)




