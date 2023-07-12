
import numpy as np
import tensorflow as tf
from utils import eval_accuracy, load_PACS_sources, load_PACS_targets, restore_original_image_from_array_vgg, plot_images
from models import classificationNN
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.applications.resnet50 import ResNet50
import argparse
from utils_adaIN import get_decoder, get_encoder, get_loss_net, ada_in

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
print("SRCS:", SRCS, "TRG:", TRGS)

CHECKPOINT_PATH = "./checkpoints/style_smoothed_KL_pretrained_decoder_resnet50_DG_" + METHOD + "_target_" + str(TRGS[0])+ "_trial_" + str(TRIAL)
CHECKPOINT_PATH_adaIN_decoder = "./checkpoints/adaIN_wikiart_mscoco_vgg19_decoder"

EPOCHS = 2001
NUM_STYLES = 4
BATCH_SIZE = int(64/NUM_STYLES)
NUM_CLASSES = 7
NUM_DOMAINS = len(SRCS)
REP_DIM = 2048

HEIGHT = 224
WIDTH = 224
NCH = 3

LR = 1E-3

src_data_loaders_content, val_data, val_labels = load_PACS_sources(SRCS, BATCH_SIZE)
src_data_loaders_style, _, _ = load_PACS_sources(SRCS, NUM_STYLES)
target_data, target_labels = load_PACS_targets(TRGS)

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
        
optimizer_base_model = tf.keras.optimizers.SGD(learning_rate=LR)
optimizer_logits = tf.keras.optimizers.SGD(learning_rate=LR)

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

ckpt_decoder = tf.train.Checkpoint(decoder = decoder)
ckpt_manager_decoder = tf.train.CheckpointManager(ckpt_decoder, CHECKPOINT_PATH_adaIN_decoder, max_to_keep=1) 
ckpt_decoder.restore(ckpt_manager_decoder.latest_checkpoint)

@tf.function
def train_step_style_consistency(processed_content_data, content_labels_1, content_data_1, style_data_1):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        rep = base_model(processed_content_data, training=False)
        outputs = classifier(rep, training=True)
        loss_1 = tf.reduce_mean(ce_loss_none(content_labels_1, outputs))
        
        #Multiple Style loss
        content_data_reshaped = tf.reshape(content_data_1, [-1, HEIGHT*WIDTH*NCH])
        repeated_content_data = tf.tile(content_data_reshaped, [1, NUM_STYLES])
        repeated_content_data = tf.reshape(repeated_content_data, [-1, HEIGHT*WIDTH*NCH])
        repeated_content_data = tf.reshape(repeated_content_data, [-1, HEIGHT, WIDTH, NCH])
        
        style_data_reshaped = tf.reshape(style_data_1, [-1, HEIGHT*WIDTH*NCH])
        repeated_style_data = tf.tile(style_data_reshaped, [BATCH_SIZE, 1])
        repeated_style_data = tf.reshape(repeated_style_data, [-1, HEIGHT*WIDTH*NCH])
        repeated_style_data = tf.reshape(repeated_style_data, [-1, HEIGHT, WIDTH, NCH])
        
        style_encoded = encoder_vgg19(repeated_style_data)
        content_encoded = encoder_vgg19(repeated_content_data)
        
        t = ada_in(style_encoded, content_encoded)
        
        stylized_images = decoder(t)
        
        stylized_rep = base_model(stylized_images, training=False)
        stylized_outputs = classifier(stylized_rep, training=True)
        stylized_outputs_softmax = tf.nn.softmax(stylized_outputs)
        
        stylized_outputs_reshaped = tf.reshape(stylized_outputs, [-1, NUM_STYLES, NUM_CLASSES])
        stylized_avg_softmax = tf.reduce_mean(tf.nn.softmax(stylized_outputs_reshaped, axis = 2), axis = 1)
        repeated_stylized_avg_softmax = tf.tile(stylized_avg_softmax, [1, NUM_STYLES])
        repeated_stylized_avg_softmax = tf.reshape(repeated_stylized_avg_softmax, [-1, NUM_CLASSES])
        
        loss_2 = tf.reduce_mean(kl_loss_none(repeated_stylized_avg_softmax, stylized_outputs_softmax))
        
        repeated_content_labels = tf.tile(content_labels_1, [1, NUM_STYLES])
        repeated_content_labels = tf.reshape(repeated_content_labels, [-1, NUM_CLASSES])
        
        loss_3 = tf.reduce_mean(ce_loss_none(repeated_content_labels, stylized_outputs))
        
        stylized_avg_softmax_log = tf.math.log(stylized_avg_softmax + 1E-10)
        loss_4 = tf.reduce_mean(tf.reduce_sum(-stylized_avg_softmax * stylized_avg_softmax_log, 1))
        
        smoothed_loss = loss_3 + 5*loss_2 + 0.5 * loss_4 + loss_1

    gradients_basemodel = tape.gradient(smoothed_loss, base_model.trainable_variables)    
    gradients_logits = tape.gradient(smoothed_loss, classifier.trainable_variables)
    
    optimizer_base_model.apply_gradients(zip(gradients_basemodel, base_model.trainable_variables)) 
    optimizer_logits.apply_gradients(zip(gradients_logits, classifier.trainable_variables))

   
best_val_accuracy = 0
best_test_accuracy = 0
for epoch in range(EPOCHS):
    
    for k in range(len(SRCS)):
        if k == 0:
            x_styles = np.array(next(iter(src_data_loaders_style[k]))[0].permute(0, 2, 3, 1).numpy())
        else:
            x_styles = np.concatenate([x_styles, np.array(next(iter(src_data_loaders_style[k]))[0].permute(0, 2, 3, 1).numpy())])
    
    np.random.shuffle(x_styles)
    
    for j in range(len(SRCS)):
        
        x, y = next(iter(src_data_loaders_content[j]))
        
        x_content = np.array(x.permute(0, 2, 3, 1).numpy())
        x_content_preprocessed = preprocess_input_resnet50(np.array(x.permute(0, 2, 3, 1).numpy()) * 255)
        y_content = tf.keras.utils.to_categorical(y.numpy(), NUM_CLASSES)
        
        x_style = x_styles[j*NUM_STYLES:(j+1)*NUM_STYLES]
        for _ in range(1):
            train_step_style_consistency(x_content_preprocessed, y_content, x_content, x_style)
                        
                
    if epoch % 100 == 0:
        print("\nTest Domains:", TRGS, SRCS, METHOD, CHECKPOINT_PATH)
        target_test_accuracy, _ = eval_accuracy(trg_X, trg_Y, base_model, classifier)
        print("ERM Epoch:", epoch)
        print("Targets:", target_test_accuracy)
        
        for j in range(64):
            x_style_test_1, _ = next(iter(src_data_loaders_style[0]))
            x_style_test_1 = np.array(x_style_test_1.permute(0, 2, 3, 1).numpy())
            if j == 0:
                x_style_test = np.array(x_style_test_1)
            else:
                x_style_test = np.concatenate([x_style_test, x_style_test_1])
                
        x_style_test = x_style_test[:128]    
        ind_val = np.random.randint(0, len(val_X), 128)
        x_content_val = val_X[ind_val]
        y_content_val = val_Y[ind_val]
        
        style_encoded_val = encoder_vgg19(x_style_test)
        content_encoded_val = encoder_vgg19_wo_preprocessing(x_content_val)
        reconstructed_image_val_a = decoder(ada_in(style=style_encoded_val, content=content_encoded_val)).numpy()
        reconstructed_image_val_b = restore_original_image_from_array_vgg(np.array(reconstructed_image_val_a))
        
        stylized_val_accuracy, _ = eval_accuracy(reconstructed_image_val_a, y_content_val, base_model, classifier)
        print("Val Stylized:", stylized_val_accuracy)
        
        x_style_test = x_style_test[:128]    
        ind_test = np.random.randint(0, len(trg_X), 128)
        x_content_test = trg_X[ind_test]
        y_content_test = trg_Y[ind_test]
        
        style_encoded_test = encoder_vgg19(x_style_test)
        content_encoded_test = encoder_vgg19_wo_preprocessing(x_content_test)
        reconstructed_image_test_a = decoder(ada_in(style=style_encoded_test, content=content_encoded_test)).numpy()
        reconstructed_image_test_b = restore_original_image_from_array_vgg(np.array(reconstructed_image_test_a))
        
        stylized_test_accuracy, _ = eval_accuracy(reconstructed_image_test_a, y_content_test, base_model, classifier)
        print("Target Stylized:", stylized_test_accuracy)
        
        x_content_plot = np.array(restore_original_image_from_array_vgg(np.array(x_content_test[:10])))
        x_style_plot = np.array(x_style_test[:10])
        x_generated_plot = np.array(reconstructed_image_test_b[:10])
        
        if stylized_val_accuracy > best_val_accuracy:
            best_val_accuracy = stylized_val_accuracy
            best_target_accuracy = target_test_accuracy
            best_target_stylized_accuracy = stylized_test_accuracy
            ckpt_model_save_path = ckpt_manager.save()
        print("best_val_accuracy:", best_val_accuracy, 
              "best_target_accuracy", best_target_accuracy, 
              "best_target_stylized_accuracy", best_target_stylized_accuracy)
        
        print("\n")
