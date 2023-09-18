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
    parser.add_argument('--start_res', type=int, default=4, help='The number of starting resolution')
    parser.add_argument('--iteration', type=int, default=120, help='The number of images used in the train phase')
    parser.add_argument('--max_iteration', type=int, default=2500, help='The total number of images')
    parser.add_argument('--dataset', type=str, default='FFHQ', help='The dataset name what you want to generate')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch in the test phase')

    parser.add_argument('--phase', type=str, default='style_transfer', help='[reconstruction, crossover, inpainting, style_transfer]')
    parser.add_argument('--img_size', type=int, default=1024, help='The target size of image')
    parser.add_argument('--img_a', type=str, default='Ryan.jpg', help='Filename of image a')
    parser.add_argument('--img_b', type=str, default='cartoon.jpg', help='Filename of image b')
    parser.add_argument('--test_folder', type=str, default='dataset/', help='Directory name to load attribute weight')
    parser.add_argument('--mask', type=list, default=[350, 650, 650, 850], help='mask (x0, y0, x1, y1)')
    parser.add_argument('--epoch_w', type=int, default=5000, help='Interation for optimizing w+ space')
    parser.add_argument('--epoch_n', type=int, default=1000, help='Interation for optimizing noise space')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/StyleGAN_FFHQ_8to1024',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results/',
                        help='Directory name to save the generated images')

    return parser.parse_args()

def process(args, M=None):

    print('[*] Data Loaded')
    with tf.Session() as sess:

        # Build stylegan model
        gan = StyleGAN(sess, args)
        gan.build_model()

        # Load image
        imgA = load_test_data(args.test_folder+args.img_a, args.img_size)
        imgB = load_test_data(args.test_folder+args.img_b, args.img_size)

        if args.phase.lower() == 'inpainting':
            imgA = imgA.astype(np.float32) * M
            tmp = cv2.cvtColor(imgA.astype(np.uint8), cv2.COLOR_RGB2BGR)* M
            cv2.imwrite(args.result_dir+'cropped_'+args.img_a, tmp)

        # Find latent code of input image
        print('[*] Optimizing W+ Space And Noise Space')
        if M is None:
            M = np.ones((args.img_size, args.img_size, 1), np.float32)

        if args.phase.lower() != 'style_transfer':
            wx, x_noise, x_noise_ = gan.find_z(imgA, imgB, M, args)
        else:
            wy, y_noise, y_noise_ = gan.find_z(imgB, imgB, M, args)
            wx, x_noise, x_noise_ = gan.find_z(imgA, imgA, M, args)
            wx[:,9:,:] = wy[:,9:,:]

        # Reconstruction
        print('[*] Start Image Generation')
        recon_img = gan.reconstruction(wx, x_noise, x_noise_)
        stylename = args.img_b.split('.jpg')[0]
        savename = args.result_dir+args.phase+"_"+args.img_a if args.phase.lower() != 'style_transfer' else args.result_dir+stylename+"_"+args.img_a
        save_images(recon_img, imgA.shape, savename)

    # Re-define session graph
    tf.reset_default_graph()


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit('1')

    x0, y0, x1, y1 = args.mask

    # Data augmentation for each image in dataset (pose transition, attribute boosting)
    if args.phase.lower() == 'reconstruction':
        args.lambda_mse, args.lambda_p, args.lambda_mse1, args.lambda_mse2 = (1, 1e-5, 1, 1)
        process(args)

    elif args.phase.lower() == 'crossover':
        args.lambda_mse, args.lambda_p, args.lambda_mse1, args.lambda_mse2 = (1, 1e-5, 1, 1)
        M = np.ones((args.img_size, args.img_size, 1), np.float32)
        M[y0:y1, x0:x1, :] = 0
        M = cv2.GaussianBlur(M, (401, 401), 0)
        M = np.expand_dims(M, -1)
        process(args, M)

    elif args.phase.lower() == 'inpainting':
        args.lambda_mse, args.lambda_p, args.lambda_mse1, args.lambda_mse2 = (1, 1e-5, 1, 100)
        M = np.ones((args.img_size, args.img_size, 1), np.float32)
        M[y0:y1, x0:x1, :] = 0
        process(args, M)

    elif args.phase.lower() == 'style_transfer':
        args.lambda_mse, args.lambda_p, args.lambda_mse1, args.lambda_mse2 = (1000, 1, 1, 1)
        process(args)

if __name__ == '__main__':
    main()