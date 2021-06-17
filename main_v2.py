import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence
import torch.backends.cudnn as cudnn

from str_dataset import STRDataset
from model import Encoder, DecoderWithAttention, Classifier
from utils import clip_gradient, accuracy
import resnet_1d

def parse():
    parser = argparse.ArgumentParser(description="Predict repeat number for STR in pytorch.")
    parser.add_argument('--seed', type=int, default=9487,
                        help='random seed for torch and numpy')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet152',
                        choices=['resnet152'],
                        help='model architecture: ' +
                        ' | '.join(['resnet152']) +
                        ' (default: resnet152)')
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N', 
                        help='number of data loading workers (default: 6)')
    parser.add_argument('--display_freq', '-p', default=100, type=int,
                        metavar='N', help='display frequency (default: 100)')
    parser.add_argument('--ckpt_dir', default='./checkpoints', type=str, metavar='PATH',
                        help='path to checkpoints folder')
    parser.add_argument('--resume', default='./checkpoints/best_classifier_loss.ckpt', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: ./checkpoints/best_classifier_loss.ckpt)')
    parser.add_argument('--ckpt_last', default='last.ckpt', type=str, metavar='PATH',
                        help='path to the latest checkpoint (default: last.ckpt)')
    parser.add_argument('--ckpt_best_loss', default='best_loss.ckpt', type=str, metavar='PATH',
                        help='path to the best loss checkpoint (default: best_loss.ckpt)')
    parser.add_argument('--ckpt_best_acc', default='best_acc.ckpt', type=str, metavar='PATH',
                        help='path to the best accuracy checkpoint (default: best_acc.ckpt)')
    parser.add_argument('--ckpt_best_classifier_loss', default='best_classifier_loss.ckpt', type=str, metavar='PATH',
                        help='path to the best classifier loss checkpoint (default: best_classifier_loss.ckpt)')
    parser.add_argument('--save_freq', type=int, default=int(1e3), help='saving last model frequency')
    parser.add_argument('--gpu', type=int, default=[0], nargs='+', help='used gpu')
    parser.add_argument('--mode', default='train', choices=['train', 'valid', 'test', 'test_NT'],
                        help='mode: ' + ' | '.join(['train', 'valid', 'test', 'test_NT']) + ' (default: train)')
    parser.add_argument('--test_file', type=str, default='./testset/test_predict_20190220.csv', help='test file path')

    # Model parameters
    parser.add_argument('--encoded_image_size', type=int , default=14, help='dimension of encoded image')
    parser.add_argument('--emb_dim', type=int , default=512, help='dimension of word embedding vectors')
    parser.add_argument('--attention_dim', type=int , default=128, help='dimension of attention linear layers')
    parser.add_argument('--decoder_dim', type=int , default=128, help='dimension of decoder lstm hidden states')
    parser.add_argument('--dropout', type=float , default=0.5, help='dropout ratio')
    parser.add_argument('--classifier_avg_pool_dim', type=int , default=7, help='dimension of classifier avg. pool')
    parser.add_argument('--classifier_hidden_dim', type=int , default=4096, help='dimension of classifier hidden layer')

    # Training parameters
    parser.add_argument('--total_step', default=int(1e5), type=int, metavar='N',
                        help='number of total steps to run')
    parser.add_argument('--start-step', default=0, type=int, metavar='N',
                        help='manual step number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=512, type=int,
                        metavar='N', help='mini-batch size per process (default: 512)')
    parser.add_argument('--encoder_lr', default=1e-4, type=float,
                        metavar='LR', help='Initial learning rate for encoder.')
    parser.add_argument('--decoder_lr', default=1e-4, type=float,
                        metavar='LR', help='Initial learning rate for decoder.')
    parser.add_argument('--classifier_lr', default=1e-4, type=float,
                        metavar='LR', help='Initial learning rate for classifier.')
    parser.add_argument('--grad_clip', default=5., type=float,
                        metavar='CLIP', help='clip gradients at an absolute value of.')
    parser.add_argument('--alpha_c', default=1., type=float,
                        help='regularization parameter for `doubly stochastic attention`, as in the paper.')
    
    # apex parameters
    parser.add_argument('--apex', action='store_true', help="enable apex")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--sync_bn', action='store_true',
                    help='enabling apex sync BN.')
    parser.add_argument('--opt-level', default='O1', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    best_loss = 1e6
    best_acc = 0
    best_classifier_loss = 1e6

    args = parse()

    if args.apex:
        try:
            from apex.parallel import DistributedDataParallel as DDP
            from apex.fp16_utils import *
            from apex import amp, optimizers
            from apex.multi_tensor_apply import multi_tensor_applier
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
        print("opt_level = {}".format(args.opt_level))
    
    # Set device for model and PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    # Specify the used gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)

    # Set to true only if inputs to model are fixed size; otherwise lot of computational overhead
    cudnn.benchmark = True

    # Fix the random seeds
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Create folder if it does not exist
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    # Generators
    str_dataset = STRDataset(
        alphabet=['A', 'T', 'C', 'G', '<', '>', '.'],
        min_repeat_number=2, max_repeat_number=50,
        min_pattern_length=2, max_pattern_length=6,
        max_prefix_length=0, max_postfix_length=0,
        insertion_prob=0.000044, deletion_prob=0.00097, mutation_prob=0.00072)
    # Maximum length of input.
    print('str_dataset.max_seq_length:', str_dataset.max_seq_length)
    alphabet_size = len(str_dataset.alphabet)
    print('alphabet_size:', alphabet_size)

    # Parameters for the data loader.
    params = {'batch_size': args.batch_size,
            'num_workers': args.workers}
    str_generator = data.DataLoader(str_dataset, **params)

    # Build the models
    print("=> creating encoder '{}'".format(args.arch))
    encoder = Encoder(args.arch, args.encoded_image_size)
    print("=> creating decoder")
    decoder = DecoderWithAttention(
        attention_dim=args.attention_dim,
        embed_dim=args.emb_dim,
        decoder_dim=args.decoder_dim,
        vocab_size=alphabet_size,
        dropout=args.dropout)
    print("=> creating classifier")
    classifier = Classifier(
        avg_pool_dim=args.classifier_avg_pool_dim,
        channels=2048,
        hidden_dim=args.classifier_hidden_dim,
        pred_pattern_dim=(str_dataset.max_pattern_length + 1) * alphabet_size,
        dropout=args.dropout,
        num_classes=1)
    
    if args.apex and args.sync_bn:
        import apex
        print("using apex synced BN")
        encoder = apex.parallel.convert_syncbn_model(encoder)
        decoder = apex.parallel.convert_syncbn_model(decoder)
        classifier = apex.parallel.convert_syncbn_model(classifier)

    # Move to GPU, if available
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    classifier = classifier.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    encoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, encoder.parameters()),
        lr=args.encoder_lr)
    decoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, decoder.parameters()),
        lr=args.decoder_lr)
    classifier_criterion = nn.MSELoss().to(device)
    classifier_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=args.classifier_lr)

    if args.mode == 'train':
        encoder.train()
        decoder.train()
        classifier.train()
    # Optionally resume from a checkpoint
    elif args.mode == 'valid' or args.mode == 'test' or args.mode == 'test_NT':
        # Use a local scope to avoid dangling references
        def resume():
            encoder_path = args.resume.replace('.ckpt', '_encoder.ckpt')
            decoder_path = args.resume.replace('.ckpt', '_decoder.ckpt')
            classifier_path = args.resume.replace('.ckpt', '_classifier.ckpt')
            print(encoder_path)
            print(decoder_path)
            print(classifier_path)
            if os.path.isfile(encoder_path) and os.path.isfile(decoder_path) and os.path.isfile(classifier_path):
                print("=> loading checkpoint '{}'".format(encoder_path))
                encoder.load_state_dict(torch.load(encoder_path))
                print("=> loading checkpoint '{}'".format(decoder_path))
                decoder.load_state_dict(torch.load(decoder_path))
                print("=> loading checkpoint '{}'".format(classifier_path))
                classifier.load_state_dict(torch.load(classifier_path))
                print("=> loaded checkpoints")
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        resume()
        encoder.eval()
        decoder.eval()
        classifier.eval()

    if args.apex:
        # Allow Amp to perform casts as required by the opt_level
        encoder, encoder_optimizer = amp.initialize(encoder, encoder_optimizer, opt_level="O1")
        decoder, decoder_optimizer = amp.initialize(decoder, decoder_optimizer, opt_level="O1")
        classifier, classifier_optimizer = amp.initialize(classifier, classifier_optimizer, opt_level="O1")

    # Training, validation or testing
    print('=> starting mode: %s' % (args.mode))
    if args.mode == 'train' or args.mode == 'valid':
        step = 0
        start_time = time.time()
        batch_xs = []
        batch_ys = []
        batch_ps = []
        batch_len_ps = []
        batch_pred_ps = []
        batch_pred_ys = []
        loss_queue = []
        avg_losses = []
        accuracy_queue = []
        avg_accuracies = []
        classifier_loss_queue = []
        avg_classifier_losses = []
        losses = []
        accuracies = []
        classifier_losses = []
        for x, len_x, y, p, len_p in str_generator:
            if args.mode == 'train':
                step += 1

                # Transfer to GPU, if available
                x = x.type(torch.FloatTensor).to(device)
                y = y.type(torch.FloatTensor).to(device).unsqueeze(-1)
                p = p.type(torch.LongTensor).to(device)
                len_p = len_p.type(torch.LongTensor).to(device)

                # =====================
                # =Encoder and Decoder=
                # =====================
                # Forward
                features = encoder(x.transpose(1, 2))
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(features, p, len_p)

                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = caps_sorted[:, 1:]

                # Keep the unsorted scores
                _, unsort_ind = sort_ind.sort()
                unsort_scores = scores[unsort_ind]

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
                
                # Calculate loss
                loss = criterion(scores, targets)
                # Add doubly stochastic attention regularization
                loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

                # Backward propagation
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                if args.apex:
                    # loss.backward() becomes:
                    with amp.scale_loss(loss, encoder_optimizer) as scaled_loss:
                        scaled_loss.backward()
                    with amp.scale_loss(loss, decoder_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # Clip gradients
                if args.grad_clip is not None:
                    clip_gradient(encoder_optimizer, args.grad_clip)
                    clip_gradient(decoder_optimizer, args.grad_clip)

                # Update weights
                encoder_optimizer.step()
                decoder_optimizer.step()
                
                # Collect loss and accuracy
                if len(loss_queue) >= args.display_freq:
                    loss_queue.pop(0)
                loss_queue.append(loss.item())

                if len(accuracy_queue) >= args.display_freq:
                    accuracy_queue.pop(0)
                accuracy_queue.append(accuracy(scores, targets, 1))
                
                # ====================
                # ===  Classifier  ===
                # ====================
                # Forward pass and calculate loss
                pred_pattern = unsort_scores.view(unsort_scores.size(0), -1)
                pred_y = classifier(features.detach(), pred_pattern.detach())
                classifier_loss = classifier_criterion(pred_y, y)

                # Backward propagation
                classifier_optimizer.zero_grad()
                if args.apex:
                    # loss.backward() becomes:
                    with amp.scale_loss(loss, classifier_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    classifier_loss.backward()
                
                # Clip gradients and update weights
                if args.grad_clip is not None:
                    clip_gradient(classifier_optimizer, args.grad_clip)    
                classifier_optimizer.step()

                # Collect loss
                if len(classifier_loss_queue) >= args.display_freq:
                    classifier_loss_queue.pop(0)
                classifier_loss_queue.append(classifier_loss.item())

                if step % args.display_freq == 0:
                    torch.cuda.synchronize()
                    avg_loss = sum(loss_queue) / len(loss_queue)
                    avg_losses.append(avg_loss)
                    avg_accuracy = sum(accuracy_queue) / len(accuracy_queue)
                    avg_accuracies.append(avg_accuracy)
                    avg_classifier_loss = sum(classifier_loss_queue) / len(classifier_loss_queue)
                    avg_classifier_losses.append(avg_classifier_loss)
                    np.save(os.path.join(args.ckpt_dir, args.arch + '_avg_losses.npy'), np.array(avg_losses))
                    np.save(os.path.join(args.ckpt_dir, args.arch + '_avg_accuracy.npy'), np.array(avg_accuracies))
                    np.save(os.path.join(args.ckpt_dir, args.arch + '_avg_classifier_losses.npy'), np.array(avg_classifier_losses))
                    print('Step [{}/{}], Loss: {:.4f}, Accuracy: {:.6f}, Classifier Loss: {:.4f}, Elapsed time: {:.2f}s'.format(
                        step, 
                        args.total_step, 
                        avg_loss, 
                        avg_accuracy, 
                        avg_classifier_loss, 
                        (time.time() - start_time)))
                    start_time = time.time()
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        print('Saving model with the best loss: {:.4f}'.format(best_loss))
                        torch.save(encoder.state_dict(), os.path.join(args.ckpt_dir, args.ckpt_best_loss.replace('.ckpt', '_encoder.ckpt')))
                        torch.save(decoder.state_dict(), os.path.join(args.ckpt_dir, args.ckpt_best_loss.replace('.ckpt', '_decoder.ckpt')))
                        torch.save(classifier.state_dict(), os.path.join(args.ckpt_dir, args.ckpt_best_loss.replace('.ckpt', '_classifier.ckpt')))
                    if avg_accuracy > best_acc:
                        best_acc = avg_accuracy
                        print('Saving model with the best acc: {:.6f}'.format(best_acc))
                        torch.save(encoder.state_dict(), os.path.join(args.ckpt_dir, args.ckpt_best_acc.replace('.ckpt', '_encoder.ckpt')))
                        torch.save(decoder.state_dict(), os.path.join(args.ckpt_dir, args.ckpt_best_acc.replace('.ckpt', '_decoder.ckpt')))
                        torch.save(classifier.state_dict(), os.path.join(args.ckpt_dir, args.ckpt_best_acc.replace('.ckpt', '_classifier.ckpt')))
                    if avg_classifier_loss < best_classifier_loss:
                        best_classifier_loss = avg_classifier_loss
                        print('Saving model with the best classifier loss: {:.4f}'.format(best_classifier_loss))
                        torch.save(encoder.state_dict(), os.path.join(args.ckpt_dir, args.ckpt_best_classifier_loss.replace('.ckpt', '_encoder.ckpt')))
                        torch.save(decoder.state_dict(), os.path.join(args.ckpt_dir, args.ckpt_best_classifier_loss.replace('.ckpt', '_decoder.ckpt')))
                        torch.save(classifier.state_dict(), os.path.join(args.ckpt_dir, args.ckpt_best_classifier_loss.replace('.ckpt', '_classifier.ckpt')))
                if step % args.save_freq == 0:
                    torch.cuda.synchronize()
                    print('Saving the last model checkpoint.')
                    torch.save(encoder.state_dict(), os.path.join(args.ckpt_dir, args.ckpt_last.replace('.ckpt', '_encoder.ckpt')))
                    torch.save(decoder.state_dict(), os.path.join(args.ckpt_dir, args.ckpt_last.replace('.ckpt', '_decoder.ckpt')))
                    torch.save(classifier.state_dict(), os.path.join(args.ckpt_dir, args.ckpt_last.replace('.ckpt', '_classifier.ckpt')))
            
            elif args.mode == 'valid':
                step += 1
                batch_xs.append(x.cpu().numpy())
                batch_ys.append(y.cpu().numpy())
                batch_ps.append(p.cpu().numpy())
                batch_len_ps.append(len_p.cpu().numpy())

                with torch.no_grad():
                    # Transfer to GPU, if available
                    x = x.type(torch.FloatTensor).to(device)
                    y = y.type(torch.FloatTensor).to(device).unsqueeze(-1)
                    p = p.type(torch.LongTensor).to(device)
                    len_p = len_p.type(torch.LongTensor).to(device)

                    # =====================
                    # =Encoder and Decoder=
                    # =====================
                    # Forward
                    features = encoder(x.transpose(1, 2))
                    scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(features, p, len_p)

                    # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                    targets = caps_sorted[:, 1:]

                    # Keep the unsorted scores
                    _, unsort_ind = sort_ind.sort()
                    unsort_scores = scores[unsort_ind]

                    # Remove timesteps that we didn't decode at, or are pads
                    # pack_padded_sequence is an easy trick to do this
                    scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
                    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
                    
                    # Calculate loss
                    loss = criterion(scores, targets)
                    # Add doubly stochastic attention regularization
                    loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

                    batch_pred_ps.append(unsort_scores.argmax(dim=-1).cpu().numpy())

                    # ====================
                    # ===  Classifier  ===
                    # ====================
                    # Forward pass and calculate loss
                    pred_pattern = unsort_scores.view(unsort_scores.size(0), -1)
                    pred_y = classifier(features.detach(), pred_pattern.detach())
                    classifier_loss = classifier_criterion(pred_y, y)
                    batch_pred_ys.append(pred_y.cpu().numpy()[:, 0])

                losses.append(loss.item())
                avg_loss = sum(losses) / len(losses)
                accuracies.append(accuracy(scores, targets, 1))
                avg_accuracy = sum(accuracies) / len(accuracies)
                classifier_losses.append(classifier_loss.item())
                avg_classifier_loss = sum(classifier_losses) / len(classifier_losses)
                print('Step [{}/{}], Loss: {:.4f}, Accuracy: {:.6f}, Classifier Loss: {:.4f}, Elapsed time: {:.2f}s'.format(
                    step, 
                    args.total_step, 
                    avg_loss, 
                    avg_accuracy, 
                    avg_classifier_loss, 
                    (time.time() - start_time)))
                start_time = time.time()

            if step >= args.total_step:
                break
    elif args.mode == 'test' or args.mode == 'test_NT':
        start_time = time.time()
        print('Reading test file:', args.test_file)
        test_csv = pd.read_csv(args.test_file)

        xs = []
        ys = []
        ps = []
        test_x = []
        test_p = []
        if args.mode == 'test':
            filters = ['a', 'b', 'c', 'd', 'e', 'f']
            filters_x = [' ']
            for x in test_csv['Repeat.Sequence']:

                for f in filters_x:
                	if (f in x):
                		x = x.replace(f, '')

                xs.append(x)
                print(x)
                test_x.append(str_dataset.vectorize_X(x))
            test_x = np.array(test_x, dtype=np.uint8)
            print ('test_x.shape:', test_x.shape)
        else:
            for y, p, x in zip(test_csv['Repeat Number'], test_csv['Repeat Pattern'], test_csv['Repeat Sequence.1']):
                xs.append(x)
                test_x.append(str_dataset.vectorize_X(x))
                ys.append(y)
                ps.append(p)
                test_p.append(str_dataset.vectorize_P(p))
            test_x = np.array(test_x, dtype=np.uint8)
            test_y = np.array(ys, dtype=np.float)
            test_p = np.array(test_p, dtype=np.uint8)
            print ('test_x.shape:', test_x.shape, ', test_y.shape', test_y.shape, ', test_p.shape', test_p.shape)

        with torch.no_grad():
            # Transfer to GPU, if available
            x = torch.from_numpy(test_x).type(torch.FloatTensor).to(device)
            if len(ps) > 0: p = torch.from_numpy(test_p).type(torch.LongTensor).to(device)

            batch_pred_ps = []
            batch_pred_ys = []
            losses = []
            accuracies = []
            classifier_losses = []
            for batch_start in range(0, x.size(0), args.batch_size):
                batch_end = batch_start + args.batch_size
                batch_end = x.size(0) if batch_end > x.size(0) else batch_end
                
                # =====================
                # =Encoder and Decoder=
                # =====================
                # Forward
                features = encoder(x[batch_start:batch_end].transpose(1, 2))
                scores, alphas, sample_ids = decoder.sample(features, str_dataset.max_pattern_length + 1, str_dataset.ctable.char_indices['<'])

                # Keep the unpacked scores
                unpacked_scores = scores

                if len(ps) > 0: 
                    # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                    targets = p[batch_start:batch_end, 1:]

                    # Remove timesteps that we didn't decode at, or are pads
                    # pack_padded_sequence is an easy trick to do this
                    scores = pack_padded_sequence(scores, [str_dataset.max_pattern_length + 1] * (batch_end - batch_start), batch_first=True).data
                    targets = pack_padded_sequence(targets, [str_dataset.max_pattern_length + 1] * (batch_end - batch_start), batch_first=True).data
                
                    # Calculate loss
                    loss = criterion(scores, targets)
                    # Add doubly stochastic attention regularization
                    loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

                    losses.append(loss.item())
                    accuracies.append(accuracy(scores, targets, 1))
                
                batch_pred_ps.append(sample_ids.cpu().numpy())

                mask = torch.ones(batch_end - batch_start, str_dataset.max_pattern_length + 1, alphabet_size).to(device)
                for b in range(batch_end - batch_start):
                    find = False
                    for t in range(str_dataset.max_pattern_length + 1):
                        if sample_ids[b, t] == str_dataset.ctable.char_indices['>'] and not find:
                            find = True
                        elif find:
                            mask[b, t:, :] = 0
                            break
                masked_unpacked_scores = unpacked_scores * mask

                # ====================
                # ===  Classifier  ===
                # ====================
                # Forward pass and calculate loss
                pred_pattern = masked_unpacked_scores.view(masked_unpacked_scores.size(0), -1)
                pred_y = classifier(features.detach(), pred_pattern.detach())
                # classifier_loss = classifier_criterion(pred_y, y[batch_start:batch_end])
                # classifier_losses.append(classifier_loss)
                batch_pred_ys.append(pred_y.cpu().numpy()[:, 0])

            # if len(ps) > 0:
            #     avg_loss = sum(losses) / len(losses)
            #     avg_accuracy = sum(accuracies) / len(accuracies)
            #     avg_classifier_loss = sum(classifier_losses) / len(classifier_losses)
            #     print('Loss: {:.4f}, Accuracy: {:.6f}, Classifier Loss: {:.4f}, Elapsed time: {:.2f}s'.format(
            #             avg_loss, 
            #             avg_accuracy, 
            #             avg_classifier_loss, 
            #             (time.time() - start_time)))
            #     start_time = time.time()
            # else:
            #     avg_classifier_loss = sum(classifier_losses) / len(classifier_losses)
            #     print('Classifier Loss: {:.4f}, Elapsed time: {:.2f}s'.format(
            #             avg_classifier_loss, 
            #             (time.time() - start_time)))
            #     start_time = time.time()
        
        test_pred_p = []
        for batch_pred_p in batch_pred_ps:
            for pred_p in batch_pred_p:
                test_pred_p.append(str_dataset.ctable.indices_seq(pred_p))
        test_pred_p = [x.split('>')[0] for x in test_pred_p]
        test_pred_p = [x.replace('<', '') for x in test_pred_p]
        test_pred_p = [x.replace('>', '') for x in test_pred_p]
        test_pred_p = [x.replace('.', '') for x in test_pred_p]

        test_pred_y = []
        for batch_pred_y in batch_pred_ys:
            for pred_y in batch_pred_y:
                test_pred_y.append(pred_y)

        # Calculate the correct rate for pattern and repeat number
        if len(ps) > 0:
            correct_p = []
            for test, pred in zip(ps, test_pred_p):
                if test == pred:
                    correct_p.append(1)
                else:
                    correct_p.append(0)
            print (len(ps), sum(correct_p), sum(correct_p) / len(test_pred_p))

        # correct_y = []
        # for test, pred in zip(test_y, test_pred_y):
        #     if abs(test - pred) < 0.5:
        #         correct_y.append(1)
        #     else:
        #         correct_y.append(0)
        # print (len(test_y), sum(correct_y), sum(correct_y) / len(test_pred_y))

        # Save the testing data and the predictions to .csv file
        if args.mode == 'test':
            df = pd.DataFrame()
            df['Locus'] = test_csv['Locus']
            df['test x'] = xs
            # df['test y'] = test_y
            df['pred y'] = test_pred_y
            # df['correct y'] = correct_y
            df['pred p'] = test_pred_p
            df['Start'] = test_csv['Start']
            df['End'] = test_csv['End']
            df['reads_count'] = test_csv['Reads_count']
            print('save predict file:', args.test_file.replace('.csv', '_predict.csv'))
            df.to_csv(args.test_file.replace('.csv', '_predict.csv'), index=0)
        else:
            df = pd.DataFrame()
            df['Repeat Number'] = test_csv['Repeat Number']
            df['Predict Number'] = test_pred_y
            df['correct y'] = correct_y
            df['Repeat Pattern'] = test_csv['Repeat Pattern']
            df['Predict Pattern'] = test_pred_p
            df['correct p'] = correct_p
            df['Repeat Sequence'] = test_csv['Repeat Sequence']
            df['Repeat Sequence.1'] = test_csv['Repeat Sequence.1']
            df.to_csv(args.test_file.replace('.csv', '_predict.csv'), index=0)

    # The end of training and validation
    if args.mode == 'train':
        print('Saving the last model checkpoint.')
        torch.save(encoder.state_dict(), os.path.join(args.ckpt_dir, args.ckpt_last.replace('.ckpt', '_encoder.ckpt')))
        torch.save(decoder.state_dict(), os.path.join(args.ckpt_dir, args.ckpt_last.replace('.ckpt', '_decoder.ckpt')))
    elif args.mode == 'valid':
        # Collect the validation data and the predictions
        valid_x = []
        valid_y = []
        valid_p = []
        valid_pred_p = []
        valid_pred_y = []
        for batch_x, batch_y, batch_p, batch_pred_p, batch_pred_y in zip(batch_xs, batch_ys, batch_ps, batch_pred_ps, batch_pred_ys):
            for x, y, p, pred_p, pred_y in zip(batch_x, batch_y, batch_p, batch_pred_p, batch_pred_y):
                valid_x.append(str_dataset.ctable.decode(x))
                valid_y.append(y)
                valid_p.append(str_dataset.ctable.indices_seq(p))
                valid_pred_p.append(str_dataset.ctable.indices_seq(pred_p))
                valid_pred_y.append(pred_y)
        valid_x = [x.split('>')[0] for x in valid_x]
        valid_x = [x.replace('<', '') for x in valid_x]
        valid_x = [x.replace('>', '') for x in valid_x]
        valid_x = [x.replace('.', '') for x in valid_x]
        valid_p = [x.split('>')[0] for x in valid_p]
        valid_p = [x.replace('<', '') for x in valid_p]
        valid_p = [x.replace('>', '') for x in valid_p]
        valid_p = [x.replace('.', '') for x in valid_p]
        valid_pred_p = [x.split('>')[0] for x in valid_pred_p]
        valid_pred_p = [x.replace('<', '') for x in valid_pred_p]
        valid_pred_p = [x.replace('>', '') for x in valid_pred_p]
        valid_pred_p = [x.replace('.', '') for x in valid_pred_p]

        # Calculate the correct rate
        correct_p = []
        for valid, pred in zip(valid_p, valid_pred_p):
            if valid == pred:
                correct_p.append(1)
            else:
                correct_p.append(0)
        print (len(valid_p), sum(correct_p), sum(correct_p) / len(valid_pred_p))

        correct_y = []
        for valid, pred in zip(valid_y, valid_pred_y):
            if abs(valid - pred) < 0.5:
                correct_y.append(1)
            else:
                correct_y.append(0)
        print (len(valid_p), sum(correct_y), sum(correct_y) / len(valid_pred_y))

        # Save the validation data and the predictions to .csv file
        df = pd.DataFrame()
        df['valid x'] = valid_x
        df['valid p'] = valid_p
        df['pred p'] = valid_pred_p
        df['correct p'] = correct_p
        df['valid y'] = valid_y
        df['pred y'] = valid_pred_y
        df['correct y'] = correct_y
        df.to_csv(args.ckpt_dir + '/valid.csv', index=0)
