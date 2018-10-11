class Config:
    data_path_train = "/Users/zhenytan/GAN_trp_on_particles_and_real_images"
    data_path_test = "/Users/zhenytan/GAN_trp_on_particles_and_real_images"
    data_path_checkpoint = "/Users/zhenytan/control_ex/test2" ##is the path for saving  model 

    model_path_train = ""
    model_path_test = "/Users/zhenytan/control_ex/test2/checkpoint/model_21.ckpt" 
    output_path = "/Users/zhenytan/control_ex/test2/results" ## the results output path

    img_size = 256
    adjust_size = 256
    train_size = 256
    img_channel = 1
    conv_channel_base = 64

    learning_rateD= 0.00015
    learning_rateG= 0.0004
    beta1 = 0.5
    max_epoch = 21
    L1_lambda = 100.0
    save_per_epoch=7
