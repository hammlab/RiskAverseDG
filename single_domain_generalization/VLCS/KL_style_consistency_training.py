
import os
import numpy as np
import tensorflow as tf
from utils import load_VLCS_sources, load_VLCS_targets, eval_accuracy, restore_original_image_from_array_vgg, plot_images
import argparse
from models import classificationNN
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.applications.resnet50 import ResNet50
from utils_adaIN import get_decoder, get_encoder, get_loss_net, ada_in
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
parser = argparse.ArgumentParser(description='Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--METHOD', type=str, default="ERM", help='ERM')
parser.add_argument('--SOURCE', type=str, default="0", help='0')
parser.add_argument('--TRIAL', type=str, default="0", help='0')
args = parser.parse_args()

METHOD = args.METHOD
TRIAL = args.TRIAL

TRGS = [0, 1, 2, 3]
SRCS = [int(args.SOURCE)]
TRGS.remove(int(args.SOURCE))

CHECKPOINT_PATH = "./checkpoints/style_smoothed_KL_pretrained_decoder_resnet50_single_source_" + METHOD + "_source_" + str(SRCS[0]) + "_trial_" + str(TRIAL)
CHECKPOINT_PATH_adaIN_decoder = "./checkpoints/adaIN_wikiart_mscoco_vgg19_decoder"

EPOCHS = 3001
NUM_STYLES = 4
BATCH_SIZE = int(64/NUM_STYLES)
NUM_CLASSES = 5
NUM_DOMAINS = len(SRCS)
REP_DIM = 2048

HEIGHT = 224
WIDTH = 224
NCH = 3

src_data_loaders_1, val_data, val_labels = load_VLCS_sources(SRCS, BATCH_SIZE)
src_data_loaders_2, _, _ = load_VLCS_sources(SRCS, NUM_STYLES)
target_data, target_labels = load_VLCS_targets(TRGS)

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

optimizer_base_model = tf.keras.optimizers.SGD(learning_rate=1E-2)
optimizer_logits = tf.keras.optimizers.SGD(learning_rate=1E-2)

ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
kl_loss_none = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)

ckpt = tf.train.Checkpoint(base_model = base_model, classifier = classifier)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=1) 

encoder_vgg19 = get_encoder()
encoder_vgg19_wo_preprocessing = get_encoder(preprocessing=False)
decoder = get_decoder()
loss_net = get_loss_net()
loss_net_wo_preprocessing = get_loss_net(preprocessing=False)
loss_fn = tf.keras.losses.MeanSquaredError()

ckpt_decoder_pretrained = tf.train.Checkpoint(decoder = decoder)
ckpt_manager_decoder_pretrained = tf.train.CheckpointManager(ckpt_decoder_pretrained, CHECKPOINT_PATH_adaIN_decoder, max_to_keep=1) 
ckpt_decoder_pretrained.restore(ckpt_manager_decoder_pretrained.latest_checkpoint).expect_partial()

@tf.function
def train_step_min_erm(processed_content_data, content_labels, content_data, style_data):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        #Base Classifier Loss
        rep = base_model(processed_content_data, training=True)
        outputs = classifier(rep, training=True)
        loss_1 = tf.reduce_mean(ce_loss_none(content_labels, outputs))
        
        #Multiple Style loss
        content_data_reshaped = tf.reshape(content_data, [-1, HEIGHT*WIDTH*NCH])
        repeated_content_data = tf.tile(content_data_reshaped, [1, NUM_STYLES])
        repeated_content_data = tf.reshape(repeated_content_data, [-1, HEIGHT*WIDTH*NCH])
        repeated_content_data = tf.reshape(repeated_content_data, [-1, HEIGHT, WIDTH, NCH])
        
        style_data_reshaped = tf.reshape(style_data, [-1, HEIGHT*WIDTH*NCH])
        repeated_style_data = tf.tile(style_data_reshaped, [BATCH_SIZE, 1])
        repeated_style_data = tf.reshape(repeated_style_data, [-1, HEIGHT*WIDTH*NCH])
        repeated_style_data = tf.reshape(repeated_style_data, [-1, HEIGHT, WIDTH, NCH])
        
        style_encoded = encoder_vgg19(repeated_style_data)
        content_encoded = encoder_vgg19(repeated_content_data)
        
        t = ada_in(style_encoded, content_encoded)
        
        stylized_images = decoder(t)
        
        stylized_rep = base_model(stylized_images, training=True)
        stylized_outputs = classifier(stylized_rep, training=True)
        stylized_outputs_softmax = tf.nn.softmax(stylized_outputs)
        
        stylized_outputs_reshaped = tf.reshape(stylized_outputs, [-1, NUM_STYLES, NUM_CLASSES])
        stylized_avg_softmax = tf.reduce_mean(tf.nn.softmax(stylized_outputs_reshaped, axis = 2), axis = 1)
        repeated_stylized_avg_softmax = tf.tile(stylized_avg_softmax, [1, NUM_STYLES])
        repeated_stylized_avg_softmax = tf.reshape(repeated_stylized_avg_softmax, [-1, NUM_CLASSES])
        
        loss_2 = tf.reduce_mean(kl_loss_none(repeated_stylized_avg_softmax, stylized_outputs_softmax))
        
        repeated_content_labels = tf.tile(content_labels, [1, NUM_STYLES])
        repeated_content_labels = tf.reshape(repeated_content_labels, [-1, NUM_CLASSES])
        
        loss_3 = tf.reduce_mean(ce_loss_none(repeated_content_labels, stylized_outputs))
        
        stylized_avg_softmax_log = tf.math.log(stylized_avg_softmax + 1E-10)
        loss_4 = tf.reduce_mean(tf.reduce_sum(-stylized_avg_softmax * stylized_avg_softmax_log, 1))
        
        smoothed_loss = 2*loss_2 + 0.5 * loss_4 + loss_3 + loss_1
        
    gradients_basemodel = tape.gradient(smoothed_loss, base_model.trainable_variables)    
    gradients_logits = tape.gradient(smoothed_loss, classifier.trainable_variables)
    
    optimizer_base_model.apply_gradients(zip(gradients_basemodel, base_model.trainable_variables)) 
    optimizer_logits.apply_gradients(zip(gradients_logits, classifier.trainable_variables))
    return smoothed_loss
    
###############################
best_val_accuracy = 0
best_test_accuracy = 0
for epoch in range(EPOCHS):
    
    x, y = next(iter(src_data_loaders_1[0]))
    
    x_content = np.array(x.permute(0, 2, 3, 1).numpy())
    x_content_preprocessed = preprocess_input_resnet50(np.array(x.permute(0, 2, 3, 1).numpy()) * 255)
    y_content = tf.keras.utils.to_categorical(y.numpy(), NUM_CLASSES)
    
    x_style = np.array(next(iter(src_data_loaders_2[0]))[0].permute(0, 2, 3, 1).numpy())
    
    for _ in range(1):
        train_step_min_erm(x_content_preprocessed, y_content, x_content, x_style)
                
    if epoch % 200 == 0:
        print("\nTest Domains:", TRGS, SRCS, METHOD, CHECKPOINT_PATH)
        target_test_accuracy, _ = eval_accuracy(trg_X, trg_Y, base_model, classifier)
        val_accuracy, _ = eval_accuracy(val_X, val_Y, base_model, classifier)
        print("ERM Epoch:", epoch)
        print("Target:", target_test_accuracy)
        ckpt_model_save_path = ckpt_manager.save()
        print("\n")