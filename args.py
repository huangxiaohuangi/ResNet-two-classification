import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='test', type=str)

# datasets 
# parser.add_argument('-dataset_path', type=str, default='G:/water/code/light/', help='the path to save imgs')
# parser.add_argument('-dataset_txt_path', type=str, default='./dataset/pink.txt')
# parser.add_argument('-train_txt_path', type=str, default='./dataset/pink_train.txt')
# parser.add_argument('-test_txt_path', type=str, default='./dataset/pink_test.txt')
# parser.add_argument('-val_txt_path', type=str, default='./dataset/pink_val.txt')

# optimizer
parser.add_argument('--optimizer', default='adam', choices=['sgd', 'rmsprop', 'adam', 'radam'])
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument('--lr-fc-times', '--lft', default=5, type=int,
                    metavar='LR', help='initial model last layer rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--no_nesterov', dest='nesterov',
                    action='store_false',
                    help='do not use Nesterov momentum')
parser.add_argument('--alpha', default=0.99, type=float, metavar='M',
                    help='alpha for ')
parser.add_argument('--beta1', default=0.9, type=float, metavar='M',
                    help='beta1 for Adam (default: 0.9)')
parser.add_argument('--beta2', default=0.999, type=float, metavar='M',
                    help='beta2 for Adam (default: 0.999)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

# training
parser.add_argument("--checkpoint", type=str, default='./checkpoints')
parser.add_argument("--resume", default='', type=str, metavar='PATH', help='path to save the latest checkpoint')
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--start_epoch", default=0, type=int, metavar='N')
parser.add_argument('--epochs', default=60, type=int, metavar='N')

parser.add_argument('--image-size', type=int, default=80)
parser.add_argument('--arch', default='resnet18', choices=['resnet34', 'resnet18', 'resnet50'])
parser.add_argument('--num_classes', default=2, type=int)

# model path
parser.add_argument('--model_path', default='./checkpoints/model_6_9802_9825.pth', type=str)
parser.add_argument('--result_csv', default='./result.csv')

args = parser.parse_args()
