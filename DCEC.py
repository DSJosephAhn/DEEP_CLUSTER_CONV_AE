import os
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import src.util.metrics as metrics
from src.util.AE_model import DCEC, args

def gpu_boost():
    ## if your PC is equipped with GPU device,tensorflow gpu activation
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    gpus= tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    return gpus

if __name__ == "__main__":
    ## if you have gpu device on your computer,
    gpus= gpu_boost()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    from src.util.datasets import load_mnist, load_usps
    if args.dataset == 'mnist':
        x, y = load_mnist()
    elif args.dataset == 'usps':
        x, y = load_usps('data/usps')
    elif args.dataset == 'mnist-test':
        x, y = load_mnist()
        x, y = x[60000:], y[60000:]

    # prepare the DCEC model
    dcec= DCEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10], n_clusters=10)
    # dcec = DCEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10], n_clusters=args.n_clusters)
    plot_model(dcec.model, to_file=args.save_dir + '/dcec_model.png', show_shapes=True)
    dcec.model.summary()

    # begin clustering.
    optimizer = 'adam'
    dcec.compile(loss=['kld', 'mse'], loss_weights=[args.gamma, 1], optimizer=optimizer)
    dcec.fit(x, y=y, tol=args.tol, maxiter=args.maxiter,
             update_interval=args.update_interval,
             save_dir=args.save_dir,
             cae_weights=args.cae_weights)
    y_pred = dcec.y_pred
    print('acc = %.4f, nmi = %.4f, ari = %.4f' % (metrics.acc(y, y_pred), metrics.nmi(y, y_pred), metrics.ari(y, y_pred)))
