import argparse
import warnings
warnings.filterwarnings("ignore")



def main():
    #wandb.init(project="Attentive Collaborative Filtering")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                type=str,
                default="pinterest",
                help='dataset')
    parser.add_argument('--optim',
                type=str,
                default='adam',
                help='optimizer')
    parser.add_argument('--epochs',
                type=int,
                default=400,
                help='epoch')
    parser.add_argument('--batch_size',
                type=int,
                default=256,
                help='batch size')
    parser.add_argument('--dim',
                type=int,
                default=128,
                help='dimension')    
    parser.add_argument('--lr',
                type=float,
                default=0.01,
                help='learning rate')    
    parser.add_argument('--reg',
                type=float,
                default=0.01,
                help='l2_regularization') 
    parser.add_argument('--hold_out',
                type=int,
                default=1,
                help='cv')


    args = parser.parse_args()
    #wandb.config.update(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    



if __name__ == '__main__':
    main()
        