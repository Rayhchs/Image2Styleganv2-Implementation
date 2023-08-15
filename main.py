import os
import warnings
import argparse
import sys, time
from StyleGAN import StyleGAN
import multiprocessing as mp
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Ignore tf warning, info
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of StyleGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--progressive', type=str2bool, default=True, help='use progressive training')
    parser.add_argument('--sn', type=str2bool, default=False, help='use spectral normalization')
    parser.add_argument('--start_res', type=int, default=8, help='The number of starting resolution')
    parser.add_argument('--iteration', type=int, default=120, help='The number of images used in the train phase')
    parser.add_argument('--max_iteration', type=int, default=2500, help='The total number of images')

    parser.add_argument('--phase', type=str, default='reconstruction', help='[train, augmentation]')
    parser.add_argument('--dataset', type=str, default='FFHQ', help='The dataset name what you want to generate')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch in the test phase')
    parser.add_argument('--gpu_id', type=str, default='all', help='The number of gpu')
    parser.add_argument('--img_size', type=int, default=1024, help='The target size of image')
    parser.add_argument('--test_folder', type=str, default='dataset/',
                        help='Directory name to load attribute weight')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/StyleGAN_FFHQ_8to1024',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results/',
                        help='Directory name to save the generated images')

    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

args = parse_args()


def process(args, img_paths):

    args = parse_args()
    print('[*] Data Loaded')

    for img_path in img_paths:
        t0 = time.time()
        with tf.Session() as sess:

            # Build stylegan model
            gan = StyleGAN(sess, args)
            gan.build_model()

            # Load image
            img = load_test_data(args.test_folder+img_path, args.img_size)

            # Find latent code of input image
            print('[*] Optimizing W+ Space And Noise Space')
            M = np.ones((img.shape[0], img.shape[1], 1), np.float32)
            latent_code, opt_noise, opt_noise_ = gan.find_z(img, img, M)

            # Img and flipped image attribute boosting
            print('[*] Start Image Generation')

            # Pose transition
            recon_img = gan.reconstruction(latent_code, opt_noise, opt_noise_)
            save_images(recon_img, img.shape, args.result_dir+img_path)

        # Re-define session graph
        tf.reset_default_graph()


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # Data augmentation for each image in dataset (pose transition, attribute boosting)
    if args.phase.lower() == 'reconstruction':

        img_paths = os.listdir(args.test_folder)
        if not img_paths == []:
            process(args, img_paths)


if __name__ == '__main__':
    main()