import keras


def l2_softmax(alpha):
    def internal(y_true, y_pred):
        y_normal = alpha * keras.backend.l2_normalize(y_pred)

        return keras.losses.categorical_crossentropy(y_true, y_normal)

    return internal
