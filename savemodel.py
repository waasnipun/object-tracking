import tensorflow as tf
import tensorflow_hub as hub

# module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
# model = hub.load(module_handle).signatures['default']

# model.save(model, "/mobilenet_model/")

model = tf.saved_model.load(
    "/Users/nipunwaas/Documents/FYP/object-tracking/mobilenet_model"
)
print(list(model.signatures.keys()))  # ["serving_default"]

infer = model.signatures["default"]
print(infer.structured_outputs)

# labeling = infer(tf.constant(x))[model.output_names[0]]

# decoded = imagenet_labels[np.argsort(labeling)[0,::-1][:5]+1]


# print("Result after saving and loading:\n", model)