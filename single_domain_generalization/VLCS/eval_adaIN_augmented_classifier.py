import os
import sys
import numpy as np
import tensorflow as tf
from utils import load_VLCS_sources, load_VLCS_targets, eval_accuracy, restore_original_image_from_array_vgg, plot_images, mini_batch_class_balanced
from models import classificationNN
import argparse
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.applications.resnet50 import ResNet50
from utils_adaIN import get_decoder, get_encoder, get_loss_net, ada_in
from math import ceil
from scipy.stats import binom_test

def _count_arr(arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts   

def _sample_styles(content_image, num):

    counts = np.zeros(NUM_CLASSES, dtype=int)
    sum_softmax = np.zeros(NUM_CLASSES, dtype=int)
    for _ in range(ceil(num / BATCH_SIZE)):
        this_batch_size = min(BATCH_SIZE, num)
        num -= this_batch_size

        content_reshaped = tf.reshape(content_image, [-1, HEIGHT*WIDTH*NCH])
        repeated_content_image = tf.tile(content_reshaped, [1, this_batch_size])
        repeated_content_image = tf.reshape(repeated_content_image, [-1, HEIGHT*WIDTH*NCH])
        repeated_content_image = tf.reshape(repeated_content_image, [-1, HEIGHT, WIDTH, NCH])
        repeated_content_image = repeated_content_image.numpy()
        
        style_images = np.array(next(iter(src_data_loaders[0]))[0].permute(0, 2, 3, 1).numpy())
        
        style_encoded_batch = encoder_vgg19(style_images)
        content_encoded_batch = encoder_vgg19(repeated_content_image)
        t = ada_in(style_encoded_batch, content_encoded_batch)
        
        stylized_content = decoder(t)
        stylized_outputs = classifier(base_model(stylized_content, training=False), training=False)
        
        predictions = stylized_outputs.numpy().argmax(1)
        counts += _count_arr(predictions, NUM_CLASSES)
        
        sum_softmax += tf.reduce_sum(stylized_outputs, 0)
            
    x_content_plot = np.array(repeated_content_image[:10])
    x_style_plot = np.array(style_images[:10])
    x_content_stylized_plot = np.array(stylized_content[:10])
    
    plot_images(np.concatenate([x_content_plot, x_style_plot, restore_original_image_from_array_vgg(x_content_stylized_plot)]),
                "smoothed_reconstructed_examples_"+str(SRCS[0]), 3, 10)
    return counts, np.argmax(sum_softmax.numpy()/N)

parser = argparse.ArgumentParser(description='Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--METHOD', type=str, default="ERM", help='ERM')
parser.add_argument('--SOURCE', type=str, default="3", help='Source domain')
parser.add_argument('--MODE', type=str, default="0", help='Mode')
parser.add_argument('--TRIAL', type=str, default="0", help='Trial')
args = parser.parse_args()

METHOD = args.METHOD
TRIAL = args.TRIAL
MODE = int(args.MODE)
TRGS = [0, 1, 2, 3]
SRCS = [int(args.SOURCE)]
TRGS.remove(int(args.SOURCE))
print("SRCS:", SRCS, "TRG:", TRGS, "MODE:", MODE)

if MODE==0:    
    CHECKPOINT_PATH_model = "./checkpoints/vanilla_dg_resnet50_S_DG_" + METHOD + "_source_" + str(SRCS[0])+ "_trial_" + str(TRIAL)
    CHECKPOINT_PATH_adaIN_decoder = "./checkpoints/adaIN_wikiart_mscoco_vgg19_decoder"
elif MODE==1:    
    CHECKPOINT_PATH_model = "./checkpoints/style_smoothed_KL_pretrained_decoder_resnet50_single_source_" + METHOD + "_source_" + str(SRCS[0]) + "_trial_" + str(TRIAL)
    CHECKPOINT_PATH_adaIN_decoder = "./checkpoints/adaIN_wikiart_mscoco_vgg19_decoder"

if not os.path.exists(CHECKPOINT_PATH_model):
    sys.exit("No Model:"+CHECKPOINT_PATH_model)
if not os.path.exists(CHECKPOINT_PATH_adaIN_decoder):
    sys.exit("No Model:"+CHECKPOINT_PATH_adaIN_decoder)

NUM_CLASSES = 5
NUM_DOMAINS = len(SRCS)
REP_DIM = 2048

HEIGHT = 224
WIDTH = 224
NCH = 3

ALPHA = 0.001
BATCH_SIZE = 100
N = 10

src_data_loaders, _, _ = load_VLCS_sources(SRCS, min(N, BATCH_SIZE))
target_data, target_labels = load_VLCS_targets(TRGS)

X_test = [item for sublist in target_data for item in sublist]
Y_test = [item for sublist in target_labels for item in sublist]

for i in range(len(target_data)):
    x_test_list = np.array(target_data[i], dtype=np.float32).reshape([-1, HEIGHT, WIDTH, NCH])
    y_test_list = tf.keras.utils.to_categorical(target_labels[i], NUM_CLASSES)
    
    selected_indices = np.arange(len(y_test_list))
    
    if i == 0:
        trg_X_selected = np.array(x_test_list[selected_indices])
        trg_Y_selected = np.array(y_test_list[selected_indices])
    else:
        trg_X_selected = np.array(np.concatenate([trg_X_selected, x_test_list[selected_indices]]))
        trg_Y_selected = np.array(np.concatenate([trg_Y_selected, y_test_list[selected_indices]]))
    
trg_X = np.array(X_test, dtype=np.float32).reshape([-1, HEIGHT, WIDTH, NCH])
trg_Y = tf.keras.utils.to_categorical(Y_test, NUM_CLASSES)

base_model = ResNet50(weights='imagenet', include_top=False, pooling="avg")
classifier = classificationNN(REP_DIM, NUM_CLASSES)

ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

encoder_vgg19 = get_encoder()
encoder_vgg19_wo_preprocessing = get_encoder(preprocessing=False)
decoder = get_decoder()
loss_net = get_loss_net()
loss_net_wo_preprocessing = get_loss_net(preprocessing=False)
loss_fn = tf.keras.losses.MeanSquaredError()

ckpt = tf.train.Checkpoint(base_model = base_model, classifier = classifier)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH_model, max_to_keep=1) 
ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

ckpt_decoder = tf.train.Checkpoint(decoder = decoder)
ckpt_manager_decoder = tf.train.CheckpointManager(ckpt_decoder, CHECKPOINT_PATH_adaIN_decoder, max_to_keep=1) 
ckpt_decoder.restore(ckpt_manager_decoder.latest_checkpoint).expect_partial()

target_test_accuracy, _ = eval_accuracy(preprocess_input_resnet50(np.array(255*trg_X)), trg_Y, base_model, classifier)
print("Target:", target_test_accuracy)

conf_values = [0.2, 0.4, 0.6, 0.8, 0.9999]
conf_abstained = np.zeros(len(conf_values))
conf_accuracy_on_non_abstained = np.zeros(len(conf_values))
for t_i in range(len(trg_X_selected)):
    
    x_content = np.array(trg_X_selected[t_i]).reshape([1, HEIGHT, WIDTH, NCH])
    
    counts, soft_voting_prediction = _sample_styles(x_content, N)
    
    top2 = counts.argsort()[::-1][:2]
    count1 = counts[top2[0]]
    pred_class_smoothed_non_abstained = top2[0]
    for i in range(len(conf_values)):
        threshold = conf_values[i]
        if (count1 / N) > threshold:
            if pred_class_smoothed_non_abstained == np.argmax(trg_Y_selected[t_i]):
                conf_accuracy_on_non_abstained[i] += 1
        else:
            conf_abstained[i] += 1
    
    if t_i%500 == 0 or t_i == len(trg_X_selected) - 1:
        print(args)
        print(t_i+1, "N =", N)
        print("".join(str(conf_values)))
        print("".join(str([100*conf_accuracy_on_non_abstained[j]/(t_i + 1 - conf_abstained[j]) for j in range(len(conf_values))])))
        print("".join(str((100*conf_abstained/(t_i+1)).tolist())), "\n")