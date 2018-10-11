from config import Config as conf
from utils import conv2d, batch_norm, deconv2d,Identity_block_for_G,Identity_block_for_D
import tensorflow as tf

class CGAN(object):

    def __init__(self):
        self.image = tf.placeholder(tf.float32, shape=(1,conf.img_size, conf.img_size, 1))
        self.cond = tf.placeholder(tf.float32, shape=(1,conf.img_size, conf.img_size, 1))

        self.gen_img = self.generator(self.cond)

        pos = self.discriminator(self.image, self.cond, False)
        neg = self.discriminator(self.gen_img, self.cond, True)
        pos_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pos, labels=tf.ones_like(pos)))
        neg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg, labels=tf.zeros_like(neg)))

        self.d_loss = tf.multiply(pos_loss + neg_loss,0.5)
        L1_loss=tf.multiply(conf.L1_lambda,tf.reduce_mean(tf.abs(self.image - self.gen_img)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg, labels=tf.ones_like(neg)))+L1_loss
        
                    

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'disc' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]


    def discriminator(self, img, cond, reuse): 
        with tf.variable_scope("disc", reuse=reuse):
            image = tf.concat([img, cond], 3)
            h0 = tf.nn.elu(batch_norm(conv2d(image, 64, name="h0"),'h0')) #128
            h0a = Identity_block_for_D(h0, [16,16,64] ,stage='Dstge1a')
            h0b = Identity_block_for_D(h0a, [16,16,64] ,stage='Dstge1b')
            h1 = tf.nn.elu(batch_norm(conv2d(h0b, 128, name="h1"), "h1")) #64
            h1a = Identity_block_for_D(h1, [32,32,128] ,stage='Dstge2a')
            h1b = Identity_block_for_D(h1a, [32,32,128] ,stage='Dstge2b')
            h2 = tf.nn.elu(batch_norm(conv2d(h1b, 256, name="h2"), "h2")) #32
            h2a = Identity_block_for_D(h2, [64,64,256] ,stage='Dstge3a')
            h2b = Identity_block_for_D(h2a, [64,64,256] ,stage='Dstge3b')
            h2c = Identity_block_for_D(h2b, [64,64,256] ,stage='Dstge3c')
            h3 = tf.nn.elu(batch_norm(conv2d(h2c, 512, stride=1, name="h3",pad='SAME'), "h3")) #32
            h3a = Identity_block_for_D(h3, [128,128,512] ,stage='Dstge4a')
            h3b = Identity_block_for_D(h3a, [128,128,512] ,stage='Dstge4b')
            h4 = tf.nn.elu(batch_norm(conv2d(h3b, 512,stride=1,name="h4",pad='VALID'),'h4'))#29
            h5 = conv2d(h4, 1,f=4,stride=1,name="h5",pad='VALID')##26
            return h5

    def generator(self, cond):
        with tf.variable_scope("gen"):
            e1 = batch_norm(conv2d(cond , 64, f=4, name="e1"),'e1') ##128x128x64
            e10 = tf.nn.elu(e1)
            e1a = Identity_block_for_G(e10, [16,16,64] ,stage='Gstge1a')
            e1b = Identity_block_for_G(e1a, [16,16,64] ,stage='Gstge1b')
            e1c = Identity_block_for_G(e1b, [16,16,64] ,stage='Gstge1c')
            e2 = batch_norm(conv2d(e1c, 128, f=4, name="e2"),"e2") #64x64x128
            e20 = tf.nn.elu(e2)
            e2a = Identity_block_for_G(e20, [32,32,128] ,stage='Gstge2a')
            e2b = Identity_block_for_G(e2a, [32,32,128] ,stage='Gstge2b')
            e2c = Identity_block_for_G(e2b, [32,32,128] ,stage='Gstge2c')
            e3 = batch_norm(conv2d(e2c, 256, f=4, name="e3"),"e3") #32x32x256
            e30 = tf.nn.elu(e3)
            e3a = Identity_block_for_G(e30, [64,64,256] ,stage='Gstge3a')
            e3b = Identity_block_for_G(e3a, [64,64,256] ,stage='Gstge3b')
            e3c = Identity_block_for_G(e3b, [64,64,256] ,stage='Gstge3c')
            e4 = batch_norm(conv2d(e3c, 512, f=4, name="e4"),"e4") #16x16x512
            e40 = tf.nn.elu(e4)
            e4a = Identity_block_for_G(e40, [128,128,512] ,stage='Gstge4a')
            e4b = Identity_block_for_G(e4a, [128,128,512] ,stage='Gstge4b')
            e4c = Identity_block_for_G(e4b, [128,128,512] ,stage='Gstge4c')

            e5 = batch_norm(conv2d(e4c, 512, f=4, name="e5"),"e5") #8x8x512
            e50 = tf.nn.elu(e5)
            e5a = Identity_block_for_G(e50, [128,128,512] ,stage='Gstge5a')
            e5b = Identity_block_for_G(e5a, [128,128,512] ,stage='Gstge5b')
            e6 = batch_norm(conv2d(e5b, 512, f=4, name="e6"),"e6") #4x4x512
            e60 = tf.nn.elu(e6)




            d1 = batch_norm(deconv2d(e60, [1,8,8,512], name="d1"),'d1')
            d10 = tf.nn.elu(tf.add(d1,e5))
            d1a = Identity_block_for_G(d10, [128,128,512] ,stage='Gstge6a')
            d1b = Identity_block_for_G(d1a, [128,128,512] ,stage='Gstge6b')


            d2 = batch_norm(deconv2d(d1b, [1,16,16,512], name="d2"),'d2')
            d20 = tf.nn.elu(tf.add(e4,d2))
            d2a = Identity_block_for_G(d20, [128,128,512] ,stage='Gstge7a')
            d2b = Identity_block_for_G(d2a, [128,128,512] ,stage='Gstge7b')
            d2c = Identity_block_for_G(d2b, [128,128,512] ,stage='Gstge7c')
            d3 = batch_norm(deconv2d(d2c, [1,32,32,256], name="d3"),'d3')
            d30 = tf.nn.elu(tf.add(e3,d3))
            d3a = Identity_block_for_G(d30, [64,64,256] ,stage='Gstge8a')
            d3b = Identity_block_for_G(d3a, [64,64,256] ,stage='Gstge8b')
            d3c = Identity_block_for_G(d3b, [64,64,256] ,stage='Gstge8c')
            d4 = batch_norm(deconv2d(d3c, [1,64,64,128], name="d4"),'d4')
            d40 = tf.nn.elu(tf.add(e2,d4))
            d4a = Identity_block_for_G(d40, [32,32,128] ,stage='Gstge9a')
            d4b = Identity_block_for_G(d4a, [32,32,128] ,stage='Gstge9b')
            d4c = Identity_block_for_G(d4b, [32,32,128] ,stage='Gstge9c')
            d5 = batch_norm(deconv2d(d4c, [1,128,128,64], name="d5"),'d5')
            d50 = tf.nn.elu(tf.add(e1,d5))
            d5a = Identity_block_for_G(d50, [16,16,64] ,stage='Gstge10a')
            d5b = Identity_block_for_G(d5a, [16,16,64] ,stage='Gstge10b')
            d5c = Identity_block_for_G(d5b, [16,16,64] ,stage='Gstge10c')
            d6 = deconv2d(d5c, [1,256,256,1], name="d6")


            return tf.tanh(d6)

    