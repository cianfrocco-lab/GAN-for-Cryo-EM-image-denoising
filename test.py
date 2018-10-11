from config import Config as conf
from data import *
import scipy.misc
from model import CGAN

import tensorflow as tf
import numpy as np
import time
import sys



def prepocess_test(img, cond):

    # img = scipy.misc.imresize(img, [conf.train_size, conf.train_size])
    # cond = scipy.misc.imresize(cond, [conf.train_size, conf.train_size])
    img = img.reshape(1, conf.img_size, conf.img_size, conf.img_channel)
    cond = cond.reshape(1, conf.img_size, conf.img_size, conf.img_channel)
    img = img/127.5 - 1.
    cond = cond/127.5 - 1.
    return img,cond

def test():

    if not os.path.exists(conf.output_path):
        os.makedirs(conf.output_path)
    data = load_data()
    model = CGAN()

    saver = tf.train.Saver()

    counter = 0
    start_time = time.time()

    with tf.Session() as sess:
        saver.restore(sess, conf.model_path_test)
        test_data = data["test"]()
        for img, cond, name in test_data:
            pimg, pcond = prepocess_test(img, cond)
            
            gen_img = sess.run(model.gen_img, feed_dict={model.image:pimg, model.cond:pcond})

            gen_img = gen_img.reshape(gen_img.shape[1:-1])
                        
            gen_img1 = (gen_img + 1.) * 127.5
           
            print(gen_img1)
            path_save = conf.output_path +"/" + "%s"  %(name)
            print(path_save)
            scipy.misc.imsave( path_save,gen_img1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'gpu=':
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[1][4:])
    else:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
    test()
