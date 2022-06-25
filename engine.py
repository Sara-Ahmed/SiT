import os

import warnings
warnings.filterwarnings("ignore")

import sys
import math
from pathlib import Path

import torch
import torchvision

from datasets import datasets_utils


import utils


def train_one_epoch(SiT_model, data_loader, optimizer, lr_schedule, wd_schedule, epoch, fp16_scaler, args):
    
    save_recon = os.path.join(args.output_dir, 'reconstruction_samples')
    Path(save_recon).mkdir(parents=True, exist_ok=True)
    bz = args.batch_size
    plot_ = True if args.rec_head==1 else False
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, ((im, rot, im_corr, im_mask), _) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        if args.drop_replace > 0:
            im_corr, im_mask = datasets_utils.GMML_replace_list(im, im_corr, im_mask, drop_type=args.drop_type,
                                                                            max_replace=args.drop_replace, align=args.drop_align)
            
        # move to gpu
        im = [im.cuda(non_blocking=True) for im in im]
        rot = [r.type(torch.LongTensor).cuda(non_blocking=True) for r in rot]
        im_corr = [c.cuda(non_blocking=True) for c in im_corr]
        im_mask = [m.cuda(non_blocking=True) for m in im_mask]
        
        
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            rot_l, contr_l, recons_l, rot_l_w, contr_l_w, recons_l_w, rec_imgs = SiT_model(im, im_corr, im_mask, rot)
            
            #-------------------------------------------------
            if plot_==True and utils.is_main_process():# and args.saveckp_freq and epoch % args.saveckp_freq == 0:
                plot_ = False
                #validating: check the reconstructed images
                print_out = save_recon + '/epoch_' + str(epoch).zfill(5)  + '.jpg' 
                imagesToPrint = torch.cat([im[0][0: min(15, bz)].cpu(),  im_corr[0][0: min(15, bz)].cpu(),
                                       rec_imgs[0: min(15, bz)].cpu()], dim=0)
                torchvision.utils.save_image(imagesToPrint, print_out, nrow=min(15, bz), normalize=True, range=(-1, 1))
                
                
            
            loss = rot_l_w + contr_l_w + recons_l_w
            
                        
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # model update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(SiT_model, args.clip_grad)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer) 
                param_norms = utils.clip_gradients(SiT_model, args.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()


        # logging
        torch.cuda.synchronize()
  
        metric_logger.update(rot_l=rot_l.item() if hasattr(rot_l, 'item') else 0.)
        metric_logger.update(rot_l_w=rot_l_w.item() if hasattr(rot_l, 'item') else 0.)
        
        metric_logger.update(contr_l=contr_l.item() if hasattr(contr_l, 'item') else 0.)
        metric_logger.update(contr_l_w=contr_l_w.item() if hasattr(contr_l, 'item') else 0.)
        
        metric_logger.update(recons_l=recons_l.item() if hasattr(recons_l, 'item') else 0.)
        metric_logger.update(recons_l_w=recons_l_w.item() if hasattr(recons_l, 'item') else 0.)
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
