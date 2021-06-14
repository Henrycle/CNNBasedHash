import torch
import torch.nn
import argparse
import time

import utils
from model.hidden import *

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--options-file', '-o', required=True, type=str,
                        help='The file where the simulation options are stored.')
    parser.add_argument('--checkpoint-file', '-c', required=True, type=str, help='Model checkpoint file')
    parser.add_argument('--batch-size', '-b', required=True, type=int, help='The batch size.')
    parser.add_argument('--source-image', '-s', required=True, type=str,
                        help='The test image')

    args = parser.parse_args()


    train_options, hidden_config = utils.load_options(args.options_file)

    checkpoint = torch.load(args.checkpoint_file,map_location='cpu')
    #checkpoint = torch.load(args.checkpoint_file)

    hidden_net = Hidden(hidden_config, device)
    utils.model_from_checkpoint(hidden_net, checkpoint)

    test_data = utils.get_testdata_loaders(args.source_image, args.batch_size)

    Hash_result = 'HashResult-'+f'{time.strftime("%Y.%m.%d--%H-%M-%S")}.csv'

    i=1
    for image,_ in test_data:
        image = image.to(device)
        _,hash = hidden_net.validate_on_batch(image)
        utils.write_test_hash('Hash_result.csv', hash, args.batch_size)
        print('The '+str(i)+' batch image hash is generated.')
        i=i+1

if __name__ == '__main__':
    main()