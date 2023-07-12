
import numpy as np
import tensorflow as tf
from utils import eval_accuracy, load_office_home_sources, load_office_home_targets
from models import classificationNN
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.applications.resnet50 import ResNet50
import argparse

parser = argparse.ArgumentParser(description='Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--METHOD', type=str, default="ERM", help='ERM')
parser.add_argument('--TARGET', type=str, default="0", help='0')
parser.add_argument('--TRIAL', type=str, default="0", help='0')
args = parser.parse_args()

METHOD = args.METHOD
TRIAL = args.TRIAL

SRCS = [0, 1, 2, 3]
TRGS = [int(args.TARGET)]
SRCS.remove(int(args.TARGET))

CHECKPOINT_PATH = "./checkpoints/vanilla_dg_resnet50_DG_" + METHOD + "_target_" + str(TRGS[0])+ "_trial_" + str(TRIAL)

EPOCHS = 2001
BATCH_SIZE = 64
NUM_CLASSES = 65
NUM_DOMAINS = len(SRCS)
REP_DIM = 2048

HEIGHT = 224
WIDTH = 224
NCH = 3

src_data_loaders, val_data, val_labels = load_office_home_sources(SRCS, BATCH_SIZE)
target_data, target_labels = load_office_home_targets(TRGS)

X_val = [item for sublist in val_data for item in sublist]
Y_val = [item for sublist in val_labels for item in sublist]

X_test = [item for sublist in target_data for item in sublist]
Y_test = [item for sublist in target_labels for item in sublist]

val_X = preprocess_input_resnet50(255 * np.array(X_val, dtype=np.float32).reshape([-1, HEIGHT, WIDTH, NCH]))
val_Y = tf.keras.utils.to_categorical(Y_val, NUM_CLASSES)

trg_X = preprocess_input_resnet50(255 * np.array(X_test, dtype=np.float32).reshape([-1, HEIGHT, WIDTH, NCH]))
trg_Y = tf.keras.utils.to_categorical(Y_test, NUM_CLASSES)

base_model = ResNet50(weights='imagenet', include_top=False, pooling="avg")
classifier = classificationNN(REP_DIM, NUM_CLASSES)

for layer in base_model.layers:
    if "_bn" in layer.name:
        layer.trainable = False
        layer._per_input_updates = {}
        
optimizer_base_model = tf.keras.optimizers.SGD(learning_rate=5*1E-3)
optimizer_logits = tf.keras.optimizers.SGD(learning_rate=5*1E-3)

ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

ckpt = tf.train.Checkpoint(base_model = base_model, classifier = classifier)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=1) 


@tf.function
def train_step_min_erm(data, class_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        rep = base_model(data, training=False)
        outputs = classifier(rep, training=True)
        loss = tf.reduce_mean(ce_loss_none(class_labels, outputs))

    gradients_basemodel = tape.gradient(loss, base_model.trainable_variables)    
    gradients_logits = tape.gradient(loss, classifier.trainable_variables)
    
    optimizer_base_model.apply_gradients(zip(gradients_basemodel, base_model.trainable_variables)) 
    optimizer_logits.apply_gradients(zip(gradients_logits, classifier.trainable_variables))
    return loss
    
best_val_accuracy = 0
best_test_accuracy = 0
for epoch in range(EPOCHS):
    
    for j in range(len(SRCS)):
        x, y = next(iter(src_data_loaders[j]))
        
        x_content = np.array(x.permute(0, 2, 3, 1).numpy())
        x_content_preprocessed = preprocess_input_resnet50(np.array(x.permute(0, 2, 3, 1).numpy()) * 255)
        y_content = tf.keras.utils.to_categorical(y.numpy(), NUM_CLASSES)
        
        for _ in range(1):
            l = train_step_min_erm(x_content_preprocessed, y_content)
                
    if epoch % 100 == 0:
        print("\nTest Domains:", TRGS, SRCS, METHOD, CHECKPOINT_PATH)
        val_accuracy, _ = eval_accuracy(val_X, val_Y, base_model, classifier)
        target_test_accuracy, _ = eval_accuracy(trg_X, trg_Y, base_model, classifier)
        print("ERM Epoch:", epoch)
        print("Targets:", target_test_accuracy)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_target_accuracy = target_test_accuracy
            ckpt_model_save_path = ckpt_manager.save()
        print("best_val_accuracy:", best_val_accuracy, "best_target_accuracy", best_target_accuracy)
        print("\n")
