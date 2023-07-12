
import numpy as np
import tensorflow as tf
from utils import restore_original_image_from_array_vgg
from utils_adaIN import get_decoder, get_encoder, get_loss_net, ada_in, get_mean_std, deprocess, get_wikiart_set, get_coco_training_set

CHECKPOINT_PATH = "./checkpoints/adaIN_wikiart_mscoco_vgg19_decoder"

EPOCHS = 5001
BATCH_SIZE = 100
LR = 1E-4

mscoco = get_coco_training_set().repeat().shuffle(30).batch(BATCH_SIZE)
mscoco_iter = iter(mscoco)

wikiart = get_wikiart_set().repeat().shuffle(30).batch(BATCH_SIZE)
wikiart_iter = iter(wikiart)

encoder = get_encoder()
decoder = get_decoder()
loss_net = get_loss_net()
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

ckpt_decoder = tf.train.Checkpoint(decoder = decoder)
ckpt_manager_decoder = tf.train.CheckpointManager(ckpt_decoder, CHECKPOINT_PATH, max_to_keep=1) 

@tf.function
def compute_adv_cw(content_images, style_images):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    loss_content = 0.0
    loss_style = 0.0
    with tf.GradientTape(persistent=True) as tape:
        
        style_encoded = encoder(style_images)
        content_encoded = encoder(content_images)
        
        t = ada_in(style=style_encoded, content=content_encoded)
        
        reconstructed_image_1 = decoder(t)
        reconstructed_image_2 = deprocess(reconstructed_image_1)
        reconstructed_image_3 = tf.reverse(reconstructed_image_2, axis=[-1])
        reconstructed_image_4 = tf.clip_by_value(reconstructed_image_3/255., 0.0, 1)
        
        # Compute the losses.
        reconstructed_vgg_features = loss_net(reconstructed_image_4)
        style_vgg_features = loss_net(style_images)
        loss_content = loss_fn(t, reconstructed_vgg_features[-1])
        for inp, out in zip(style_vgg_features[:-1], reconstructed_vgg_features[:-1]):
            mean_inp, std_inp = get_mean_std(inp)
            mean_out, std_out = get_mean_std(out)
            loss_style += loss_fn(mean_inp, mean_out) + loss_fn(std_inp, std_out)
            
        TV_loss = tf.image.total_variation(reconstructed_image_4)
        total_loss = 1E1 * loss_style + 1E0 * loss_content + 1E-3 * TV_loss
        
    trainable_vars = decoder.trainable_variables
    gradients = tape.gradient(total_loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))
    return loss_content, loss_style, TV_loss, total_loss

for epoch in range(EPOCHS):    
    
    if epoch%500 == 0:
        new_lr = LR * 0.9
        LR = new_lr
        optimizer.lr.assign(LR)
    
    x_content = next(mscoco_iter)
    x_style = next(wikiart_iter)
    
    l1, l2, l3, l4 = compute_adv_cw(x_content, x_style)
    
    if epoch % 50 == 0:
        
        style_encoded_test = encoder(x_style[:10])
        content_encoded_test = encoder(x_content[:10])
        reconstructed_image_test_a = decoder(ada_in(style=style_encoded_test, content=content_encoded_test))
        reconstructed_image_test_b = restore_original_image_from_array_vgg(np.array(reconstructed_image_test_a[:10]))
        
        reconstructed_image_test_aa = decoder(ada_in(style=content_encoded_test, content=content_encoded_test))
        reconstructed_image_test_bb = restore_original_image_from_array_vgg(np.array(reconstructed_image_test_aa[:10]))
        
        x_content_plot = np.array(x_content[:10])
        x_style_plot = np.array(x_style[:10])
        x_generated_plot = np.array(reconstructed_image_test_b[:10])
        x_generated_plot_a = np.array(reconstructed_image_test_bb[:10])
        
        print("Epoch:", epoch)
        print("Style_loss:", tf.reduce_mean(l2).numpy(), 
              "Content loss:", tf.reduce_mean(l1).numpy(),
              "TV loss:", tf.reduce_mean(l3).numpy(),
              "Total loss:", tf.reduce_mean(l4).numpy(), "\n")
        
        ckpt_model_save_path = ckpt_manager_decoder.save()