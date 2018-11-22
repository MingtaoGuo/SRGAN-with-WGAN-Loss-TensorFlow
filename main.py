import argparse
from train import train, test, up_scale
from PIL import Image
import numpy as np
import scipy.misc as misc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=4) #The paper: 16
    parser.add_argument("--lambd", type=float, default=1e-3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--clip_v", type=float, default=0.05)
    parser.add_argument("--B", type=int, default=5) #The paper: 16
    parser.add_argument("--max_itr", type=int, default=100000) #The paper: 600000
    parser.add_argument("--path_trainset", type=str, default="./ImageNet/")
    parser.add_argument("--path_vgg", type=str, default="./vgg_para/")
    parser.add_argument("--path_save_model", type=str, default="./save_para/")

    parser.add_argument("--is_trained", type=bool, default=False)

    args = parser.parse_args()

    if args.is_trained:
        parser.add_argument("--path_test_img", type=str, default="./test/0.jpg")
        args = parser.parse_args()
        img = np.array(Image.open(args.path_test_img))
        h, w = img.shape[0] // 4, img.shape[1] // 4 #down sample factor: 4
        downsampled_img = misc.imresize(img, [h, w])
        test(downsampled_img, img, args.B)
    else:
        train(batch_size=args.batch_size, lambd=args.lambd, init_lr=args.learning_rate, clip_v=args.clip_v, B=args.B,
              max_itr=args.max_itr, path_trainset=args.path_trainset, path_vgg=args.path_vgg,
              path_save_model=args.path_save_model)


