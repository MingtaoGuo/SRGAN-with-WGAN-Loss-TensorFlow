from network import generator, discriminator, vggnet
import tensorflow as tf
from utils import read_crop_data
import numpy as np
from PIL import Image
import scipy.misc as misc
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
import time

def up_scale(downsampled_img):
    downsampled_img = downsampled_img[np.newaxis, :, :, :]
    downsampled = tf.placeholder(tf.float32, [None, None, None, 3])
    train_phase = tf.placeholder(tf.bool)
    G = generator("generator")
    SR = G(downsampled, train_phase)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "./save_para/.\\model.ckpt")
    SR_img = sess.run(SR, feed_dict={downsampled: downsampled_img/127.5 - 1, train_phase: False})
    Image.fromarray(np.uint8((SR_img[0, :, :, :] + 1)*127.5)).show()
    Image.fromarray(np.uint8((downsampled_img[0, :, :, :]))).show()
    sess.close()

def test(downsampled_img, img, B):
    downsampled_img = downsampled_img[np.newaxis, :, :, :]
    downsampled = tf.placeholder(tf.float32, [None, None, None, 3])
    train_phase = tf.placeholder(tf.bool)
    G = generator("generator", B)
    SR = G(downsampled, train_phase)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "./save_para/.\\model.ckpt")
    SR_img = sess.run(SR, feed_dict={downsampled: downsampled_img/127.5 - 1, train_phase: False})
    Image.fromarray(np.uint8((SR_img[0, :, :, :] + 1)*127.5)).show()
    Image.fromarray(np.uint8((downsampled_img[0, :, :, :]))).show()
    h = img.shape[0]
    w = img.shape[1]
    bic_img = misc.imresize(downsampled_img[0, :, :, :], [h, w])
    Image.fromarray(np.uint8((bic_img))).show()
    SR_img = misc.imresize(SR_img[0, :, :, :], [h, w])
    p = psnr(img, SR_img)
    s = ssim(img, SR_img, multichannel=True)
    p1 = psnr(img, bic_img)
    s1 = ssim(img, bic_img, multichannel=True)
    print("SR PSNR: %f, SR SSIM:%f, BIC PSNR: %f, BIC SSIM: %f"%(p, s, p1, s1))
    sess.close()

def train(batch_size=4, lambd=1e-3, init_lr=1e-4, clip_v=0.05, B=16, max_itr=100000, path_trainset="./ImageNet/", path_vgg="./vgg_para/", path_save_model="./save_para/"):
    inputs = tf.placeholder(tf.float32, [None, 96, 96, 3])
    downsampled = tf.placeholder(tf.float32, [None, 24, 24, 3])
    train_phase = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32)
    G = generator("generator", B)
    D = discriminator("discriminator")
    SR = G(downsampled, train_phase)
    phi = vggnet(tf.concat([inputs, SR], axis=0), path_vgg)
    phi = tf.split(phi, num_or_size_splits=2, axis=0)
    phi_gt = phi[0]
    phi_sr = phi[1]
    real_logits = D(inputs, train_phase)
    fake_logits = D(SR, train_phase)
    D_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
    G_loss = -tf.reduce_mean(fake_logits) * lambd + tf.nn.l2_loss(phi_sr - phi_gt) / batch_size
    clip_D = [var.assign(tf.clip_by_value(var, -clip_v, clip_v)) for var in D.var_list()]
    D_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(D_loss, var_list=D.var_list())
    G_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss, var_list=G.var_list())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # saver.restore(sess, "./save_para/.\\model.ckpt")
    lr0 = init_lr
    for itr in range(max_itr):
        if itr == max_itr // 2 or itr == max_itr * 3 // 4:
            lr0 = lr0 / 10
        s0 = time.time()
        batch, down_batch = read_crop_data(path_trainset, batch_size, [96, 96, 3], 4)
        e0 = time.time()
        batch = batch/127.5 - 1
        down_batch = down_batch/127.5 - 1
        s1 = time.time()
        sess.run(D_opt, feed_dict={inputs: batch, downsampled: down_batch, train_phase: True, learning_rate: lr0})
        sess.run(clip_D)
        sess.run(G_opt, feed_dict={inputs: batch, downsampled: down_batch, train_phase: True, learning_rate: lr0})
        e1 = time.time()
        if itr % 200 == 0:
            [d_loss, g_loss, sr] = sess.run([D_loss, G_loss, SR], feed_dict={downsampled: down_batch, inputs: batch, train_phase: False})
            raw = np.uint8((batch[0] + 1) * 127.5)
            bicub = misc.imresize(np.uint8((down_batch[0] + 1) * 127.5), [96, 96])
            gen = np.uint8((sr[0, :, :, :] + 1) * 127.5)
            print("Iteration: %d, D_loss: %f, G_loss: %e, PSNR: %f, SSIM: %f, Read_time: %f, Update_time: %f" % (itr, d_loss, g_loss, psnr(raw, gen), ssim(raw, gen, multichannel=True), e0 - s0, e1 - s1))
            Image.fromarray(np.concatenate((raw, bicub, gen), axis=1)).save("./results/" + str(itr) + ".jpg")
        if itr % 5000 == 0:
            saver.save(sess, path_save_model+"model.ckpt")

