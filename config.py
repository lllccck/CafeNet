CURRENT_DIR="."
CONFIG={
    "dataset_dir":CURRENT_DIR+"/datasets",
    "output_dir":CURRENT_DIR+"/output",

    # "model_dir":CURRENT_DIR+"/output/Kvasir-SEG/models",
    # "segm_dir":CURRENT_DIR+"/output/Kvasir-SEG/segments",
    # "visualize_dir":CURRENT_DIR+"/output/Kvasir-SEG/visualize",

    # "model_dir":CURRENT_DIR+"/output/CVC-ClinicDB/models",
    # "segm_dir":CURRENT_DIR+"/output/CVC-ClinicDB/segments",
    # "visualize_dir":CURRENT_DIR+"/output/CVC-ClinicDB/visualize",

    # "model_dir":CURRENT_DIR+"/output/bkai-igh-neopolyp/models",
    # "segm_dir":CURRENT_DIR+"/output/bkai-igh-neopolyp/segments",
    # "visualize_dir":CURRENT_DIR+"/output/bkai-igh-neopolyp/visualize",

    # "model_dir": CURRENT_DIR + "/output/CVC-300/models",
    # "segm_dir": CURRENT_DIR + "/output/CVC-300/segments",
    # "visualize_dir": CURRENT_DIR + "/output/CVC-300/visualize",

    "model_dir": CURRENT_DIR + "/output/ETIS-LaribPolypDB/models",
    "segm_dir": CURRENT_DIR + "/output/ETIS-LaribPolypDB/segments",
    "visualize_dir": CURRENT_DIR + "/output/ETIS-LaribPolypDB/visualize",

    "dataset":{
        # "name":"Kvasir-SEG",                                      # "Kvasir-SEG", "CVC-ClinicDB" ,"bkai-igh-neopolyp"
        # "name":"CVC-ClinicDB",
        # "name":"bkai-igh-neopolyp",
        # "name":"CVC-300",
        "name":"ETIS-LaribPolypDB",
        "random_state":2021,
        "split_ratio_train_testval":0.2,
        "split_ratio_test_val":0.5,
    },
    "device":"cuda",                                             #"cuda", "cpu"
    "mode":"train",                                              #"train", "test", "val"
    "typeloss":"BceDiceLoss",                                    #"StructureLoss", "BceDiceLoss", "BceIoULoss"
    "imagenet_mean":[0.485, 0.456, 0.406],
    "imagenet_std":[0.229, 0.224, 0.225],
    "models":{
        "num_classes":1,
    },
    "resume":{
        "is_resume":False,
        "epoch_start":0,
        "pretrain_model":"",
    },
    "train":{
        "image_size":(256,256),                                 #h w 
        "image_data_augmentation":True,
        "batch_size":8,                                         # 8 (256x256) for "Kvasir-SEG", "CVC-ClinicDB";
        "optimizer":"Adam",                                      #"SGD", "Adam"
        "learningschedule":"LambdaLR",                          #"LambdaLR", "MultiStepLR"
        "optimizer_sgd":{
            "momentum":0.9,
            "weight_decay":1e-5,
        },
        "optimizer_adam":{
            "betas":(0.9, 0.999),
            "eps":1e-8,
            "weight_decay":1e-5,
        },
        "schedule_lambdalr":{
            "power":0.9
        },
        "schedule_multisteplr":{
            "learing_rate_schedule_factor":0.1,
            "learing_epoch_steps":[80,150,180],
        },
        "learing_rate":0.0001,
        "max_epochs":200,
        "training_time_out":3600*24*3,
        "early_stop":200                         #50
    },
    "val":{
        "image_size":(320,320),     # h w  (320x320) for "Kvasir-SEG",“ETIS-LaribPolypDB”,“CVC-300”
        # "image_size":(288,384),   #      (288x384) for "CVC-ClinicDB"
        # "image_size":(480,480),   #      (480,480) for BKAI
        "batch_size":1,

        # "pretrain_model": "./output/Kvasir-SEG/models/model_CafeNetModel.pth",
        # "pretrain_model": "./output/CVC-ClinicDB/models/model_CafeNetModel.pth",
        # "pretrain_model": "./output/bkai-igh-neopolyp/models/model_CafeNetModel.pth",
    },
    "test":{
        "image_size":(320,320),             # h w  (320x320) for "Kvasir-SEG",“ETIS-LaribPolypDB”,“CVC-300”
        # "image_size": (288, 384),         #      (288x384) for "CVC-ClinicDB"
        # "image_size":(480,480),           #      (480,480) for BKAI
        "batch_size":1,

        # "pretrain_model": "./output/Kvasir-SEG/models/model_CafeNetModel.pth",
        # "pretrain_model": "./output/CVC-ClinicDB/models/model_CafeNetModel.pth",
        # "pretrain_model": "./output/bkai-igh-neopolyp/models/model_CafeNetModel.pth",

        "mask_generating":True,
        "mask_visualize":True
    },
}